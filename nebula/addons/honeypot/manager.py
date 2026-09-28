import logging
import random
import json

try:
    from .mutation import DefenseStrategyGenerator
    from .detector import HoneyDetector
    from .dataset import HoneyDataset
except ImportError:
    logging.warning("⚠️ Could not import local HoneyPot modules. Dummy mode active.")
    DefenseStrategyGenerator = None
    HoneyDetector = None
    HoneyDataset = None

class HoneyPotManager:
    HOLD_POSITION = "__HOLD__"
    BACKTRACK_REQUIRED = "__BACKTRACK__"
    # How many ACTUAL hops (real pivots, not re-validation calls) a locked navigation target gets
    # before we give up on it. Measured in the same units as the journey (visited_history growth)
    # so it can't expire before the honeypot even attempts to move — see
    # reputation_navigation_suspect for why call-cadence-based grace failed.
    NAV_HOP_BUDGET = 5

    @staticmethod
    def _detect_num_classes(engine, default=10):
        """
        Robustly determine the dataset's real class count for building the honey_map.

        The naive `engine.trainer.datamodule.num_classes` returned 10 for CIFAR100 (should be
        100) because that attribute was not the loaded value at honeypot-construction time.
        The authoritative source is the partition handler, which loads num_classes from the
        HDF5 attrs. We probe every plausible source and take the LARGEST valid value so a
        stale/default 10 can never shadow the real 100.
        """
        candidates = []
        try:
            dm = getattr(getattr(engine, 'trainer', None), 'datamodule', None)
            if dm is not None:
                # Direct attribute on the datamodule.
                v = getattr(dm, 'num_classes', None)
                if isinstance(v, int) and v > 0:
                    candidates.append(v)
                # Partition handlers hold the value loaded from the HDF5 attrs (authoritative).
                for attr in ('train_set', 'test_set', 'partition_handler', 'train_data', 'test_data'):
                    ph = getattr(dm, attr, None)
                    v = getattr(ph, 'num_classes', None)
                    if isinstance(v, int) and v > 0:
                        candidates.append(v)
        except Exception:
            pass
        # Fall back to the scenario config if the datamodule was unhelpful.
        try:
            cfg = getattr(engine, 'config', None)
            data_args = {}
            if cfg is not None and hasattr(cfg, 'participant'):
                data_args = cfg.participant.get('data_args', {}) or {}
            v = data_args.get('num_classes')
            if isinstance(v, int) and v > 0:
                candidates.append(v)
            # Last resort: map the known dataset name to its class count (timing-independent).
            dataset_name = str(data_args.get('dataset', '')).upper()
            known = {
                "MNIST": 10, "FASHIONMNIST": 10, "FMNIST": 10,
                "CIFAR10": 10, "SVHN": 10,
                "EMNIST": 47, "CIFAR100": 100,
            }
            if dataset_name in known:
                candidates.append(known[dataset_name])
        except Exception:
            pass
        if candidates:
            return max(candidates)
        logging.warning(f"[HoneyManager] Could not detect num_classes; falling back to default={default}.")
        return default

    def __init__(self, config=None, engine=None, seed: float = 0.5, role_behavior=None):
        self.config = config
        self.engine = engine
        self.role_behavior = role_behavior  # Reference to the HoneypotRoleBehavior

        real_seed = seed
        if isinstance(config, (float, int)):
            real_seed = config

        if DefenseStrategyGenerator:
            # Detect the real number of classes. The honeypot's honey_map MUST cover every
            # class or it is malformed (the 2026-07-09 CIFAR100 bug: read 10 instead of 100
            # because engine.trainer.datamodule.num_classes was not yet the loaded value).
            # We probe several sources and take the largest plausible value, because the
            # partition handler (which loads num_classes from the HDF5 attrs) is authoritative.
            num_classes = self._detect_num_classes(engine, default=10)
            logging.info(f"🔍 [HoneyManager] Detected num_classes: {num_classes}")

            self.strategy = DefenseStrategyGenerator(real_seed, num_classes=num_classes)
            self.detector = HoneyDetector()
        else:
            self.strategy = None
            self.detector = None

        self.current_map = {}
        self.previous_map = {}  # Store previous map to reduce false positives
        self.visited_history = []
        self.path_history = []
        self.reputation_history = {}  # Historial global acumulado
        self.locked_target = None     # Fixed Target
        self._navigation_target = None  # Sticky destination for reputation_navigation_suspect
        self._navigation_lock_hop_count = 0  # len(visited_history) at the moment we locked on

        # PERMANENT MAP: The honey_map remains constant throughout the entire execution
        # to ensure benign nodes have enough time to converge on the bait.
        self.map_round_counter = 0

        # PER-NODE Grace Period: Track rounds spent at current node
        self.rounds_at_current_node = 0  # Reset when pivoting
        # Must be >= LOCAL_STREAK_REQUIRED (4, see below): the grace period is the only
        # window in which analyze_neighbor's sustained-streak conviction path can fire
        # (pivot is held while grace is active). With grace=3 < streak_required=4, sparse
        # topologies where the honeynode has no other reason to linger past grace pivoted
        # away one round before the 4th consecutive hit, so local_streak never reached the
        # threshold (observed 2026-07-22: MNIST/Fashion/CIFAR10 edge=0.2 Continue re-runs,
        # max streak seen capped at 0-4, never converting). Set to 4 to close that gap.
        self.grace_rounds_per_node = 4  # Must be >= LOCAL_STREAK_REQUIRED
        self.current_node_id = None      # Track which node we're at

        # ============================================================================
        # SELF-MEASURED SOURCE LOCALIZATION (reputation-free)
        # ============================================================================
        # Physics: the attacker poisons 70% of ITS OWN data, so its model is the poison at
        # MAXIMUM concentration; neighbors only receive FedAvg-diluted poison. Poison
        # intensity therefore DECAYS with distance from the source. We localize the source by
        # gradient ascent on a self-measured poison-intensity scalar and convict the node that
        # is the local INTENSITY PEAK (dominates its neighborhood) AND rejects our directly-fed
        # bait. Both are relative/differential signals (spatial peak, temporal bait-rejection)
        # that a diluted victim cannot exhibit — no reputation module involved.
        self.SOURCE_INTENSITY = 0.55       # absolute floor for a node to be considered the source
        self.DOMINANCE_FACTOR = 1.5        # source must be >=1.5x more intense than 2nd-most-intense neighbor
        self.SOURCE_MIN_ROUNDS = 4         # rounds of direct bait-feeding before a resistance verdict
        self.best_intensity_seen = 0.0     # highest neighbor intensity seen across all hops (gradient memory)
        self.best_intensity_node = None

        # SUSPECT CONFIRMATION: Track suspects before declaring as attackers
        # Format: {node_id: confirmation_rounds}
        # If a suspect gets backdoor during confirmation → False positive, continue search
        # If after 3 rounds still no backdoor → Confirmed attacker
        self.suspect_confirmation = {}    # Track rounds monitoring each suspect
        self.confirmation_rounds_required = 3  # Phase 6.5: Reduced to 3 for faster conviction

        # ============================================================================
        # OPTIMIZED NEIGHBOR TRACKING SYSTEM
        # Fast benign verification (1 round) + Conservative malicious confirmation (5 rounds)
        # ============================================================================
        self.neighbor_tracking = {}  # {node_id: {status, negative_count, start_round, verified_round}}
        self.NEGATIVE_THRESHOLD = 5  # INCREASED FASE 3: 5 consecutive negative rounds to allow for bait learning in Ring
        self.recent_detections = {}  # {node_id: detection_count} - Fast safety buffer
        self.missing_rounds = {}     # {node_id: missing_count} - Patience limit for synchronization

        # ============================================================================
        # ADAPTIVE BACKDOOR STRENGTHENING SYSTEM
        # Progressively strengthen backdoor for nodes showing 0% to distinguish
        # weak backdoor (benign) from active filtering (malicious)
        # ============================================================================
        self.weak_backdoor_nodes = {}  # {node_id: {round_started, attempts, original_params}}
        self.strengthening_enabled = True
        self.strengthening_max_attempts = 10  # INCREASED: 10 attempts to overcome strong attackers
        self.strengthening_injection_step = 2.0   # INCREASED: +200% factor per attempt
        self.strengthening_weight_step = 10.0     # INCREASED: +10x weight per attempt to compete with poison
        self.strengthening_lr_step = 5.0          # INCREASED: +5x LR per attempt
        self.base_injection_ratio = 0.3           # Start at 30% (Aggressive start)
        self.base_weight_boost = 1.5              # Base weight boost
        self.base_lr_boost = 1.0                  # Base LR boost

        # Generate initial map
        if self.strategy:
            self.current_map = self.strategy.get_honey_map()
            logging.info(f"🔄 [HoneyManager] INITIAL Honey Map Generated: {self.current_map}")


    def reset_tracking(self):
        """Reset all neighbor tracking memory and status (useful after global reset)."""
        self.neighbor_tracking = {}
        self.suspect_confirmation = {}
        self.weak_backdoor_nodes = {}
        self.recent_detections = {}
        self.missing_rounds = {}
        logging.warning("[HoneyManager] 🧠 Neighbor tracking memory RESET.")

    def new_round(self):
        # Honey_map PERMANENTEMENTE ESTABLE - nunca cambia
        # Esto permite convergencia total del backdoor y facilita la detección
        if self.strategy:
            logging.info(f"🔒 [HoneyManager] GLOBAL PERMANENT Honey Map active: {self.current_map}")

        # ROUND-BASED HARDENING: Increment attempts for nodes already in strengthening pipeline
        # This ensures power increases every round even if analyze_neighbor hasn't run yet
        if self.strengthening_enabled:
            for neighbor_id in list(self.weak_backdoor_nodes.keys()):
                info = self.weak_backdoor_nodes[neighbor_id]
                if info["attempts"] < self.strengthening_max_attempts:
                    info["attempts"] += 1
                    logging.critical(
                        f"[Manager] ⚡ ROUND-BASED HARDENING: Boosting power for {neighbor_id} "
                        f"(attempt {info['attempts']}/{self.strengthening_max_attempts})"
                    )

        return self.current_map

    def get_dataset(self, original_dataset, injection_ratio=None):
        """
        Get honeypot dataset with backdoor.

        Args:
            original_dataset: Original training dataset
            injection_ratio: Optional custom injection ratio for adaptive strengthening
                           If None, uses default base_injection_ratio (0.8)
        """
        if self.strategy:
            # Use custom injection ratio if provided (for adaptive strengthening)
            # Otherwise use base ratio (0.8 = ~25% poisoned samples)
            ratio = injection_ratio if injection_ratio is not None else self.base_injection_ratio
            return HoneyDataset(original_dataset, self.current_map, injection_ratio=ratio)
        return original_dataset

    def get_strengthened_params(self, neighbor_id=None):
        """
        Returns strengthened parameters for nodes showing weak backdoor.

        Progressive strengthening over 3 attempts (Aggressive):
        - Attempt 1: LR x2, Injection x1.25
        - Attempt 2: LR x3, Injection x1.50
        - Attempt 3: LR x4, Injection x1.75

        Args:
            neighbor_id: Specific neighbor to get params for, or None for max params

        Returns:
            dict with strengthened parameters, or None if no strengthening needed
        """
        if not self.strengthening_enabled or not self.weak_backdoor_nodes:
            return None

        # If specific neighbor requested
        if neighbor_id:
            if neighbor_id not in self.weak_backdoor_nodes:
                return None
            info = self.weak_backdoor_nodes[neighbor_id]
        else:
            # Get max strengthening across all weak nodes
            if not self.weak_backdoor_nodes:
                return None
            info = max(self.weak_backdoor_nodes.values(), key=lambda x: x["attempts"])

        attempts = info["attempts"]

        # Progressive strengthening
        injection_multiplier = 1.0 + (self.strengthening_injection_step * attempts)
        weight_multiplier = 1.0 + (self.strengthening_weight_step * attempts)
        lr_multiplier = 1.0 + (self.strengthening_lr_step * attempts)

        strengthened_injection = min(0.95, info["base_injection"] * injection_multiplier)
        strengthened_weight = info["base_weight"] * weight_multiplier
        strengthened_lr = self.base_lr_boost * lr_multiplier

        return {
            "injection_ratio": strengthened_injection,
            "weight_boost": strengthened_weight,
            "lr_boost": strengthened_lr,
            "attempt": attempts + 1,
            "target_node":neighbor_id if neighbor_id else "multiple"
        }

    def _prune_for_logging(self, data_dict: dict) -> dict:
        """
        Prune potentially large or non-serializable structures for safe logging.
        """
        pruned = {}
        try:
            for k, v in (data_dict or {}).items():
                try:
                    pruned[k] = {
                        "status": v.get("status"),
                        "negative_count": v.get("negative_count"),
                        "suspicious_count": v.get("suspicious_count"),
                        "rounds_tested": v.get("rounds_tested"),
                        "max_compliant_seen": float(v.get("max_compliant_seen", 0.0)),
                        "first_backdoor_round": v.get("first_backdoor_round"),
                        "carrier_suspect": v.get("carrier_suspect", False),
                        "clash_count": v.get("clash_count", 0),
                        "compliant_history_tail": v.get("compliant_history", [])[-5:]
                    }
                except Exception:
                    pruned[k] = str(v)
        except Exception:
            return {"error": "prune_failed"}
        return pruned

    def _log_neighbor_analysis_details(self, neighbor_id, current_round, is_valid, has_backdoor, compliant_rate, is_suspicious, state):
        """
        Log comprehensive neighbor analysis with all internal calculations.
        """
        BENIGN_COMPLIANT_THRESHOLD = 0.02
        BENIGN_WINDOW = 3
        BENIGN_CONSISTENT_ROUNDS = 1
        NEGATIVE_THRESHOLD = self.NEGATIVE_THRESHOLD
        EXTENDED_MALICIOUS_THRESHOLD = 10

        recent_history = state.get("compliant_history", [])[-BENIGN_WINDOW:]
        consistent_rounds = sum(1 for _, r in recent_history if float(r) >= BENIGN_COMPLIANT_THRESHOLD)

        analysis = (
            f"\n{'='*70}\n"
            f"[NEIGHBOR ANALYSIS] {neighbor_id} @ Round {current_round}\n"
            f"{'='*70}\n"
            f"DETECTOR OUTPUT:\n"
            f"  ├─ is_valid: {is_valid}\n"
            f"  ├─ compliant_rate: {compliant_rate:.4f} (threshold for bait: 0.02)\n"
            f"  ├─ has_backdoor: {has_backdoor}\n"
            f"  └─ is_suspicious: {is_suspicious}\n"
            f"\nSTATE HISTORY:\n"
            f"  ├─ max_compliant_seen: {state.get('max_compliant_seen', 0):.4f}\n"
            f"  ├─ suspicious_count: {state.get('suspicious_count', 0)}\n"
            f"  ├─ negative_count: {state.get('negative_count', 0)}\n"
            f"  ├─ rounds_tested: {state.get('rounds_tested', 0)}\n"
            f"  ├─ first_backdoor_round: {state.get('first_backdoor_round')}\n"
            f"  ├─ carrier_suspect: {state.get('carrier_suspect', False)}\n"
            f"  ├─ clash_count: {state.get('clash_count', 0)}\n"
            f"  └─ current_status: {state.get('status')}\n"
            f"\nCONSISTENCY CHECKS:\n"
            f"  ├─ recent compliance history: {recent_history[-3:]}\n"
            f"  ├─ consistent_rounds: {consistent_rounds}/{BENIGN_WINDOW}\n"
            f"  ├─ has_genuine_bait: {consistent_rounds >= BENIGN_CONSISTENT_ROUNDS}\n"
            f"  └─ threshold (BENIGN): {BENIGN_CONSISTENT_ROUNDS}/{BENIGN_WINDOW} rounds >= {BENIGN_COMPLIANT_THRESHOLD:.0%}\n"
            f"\nTHRESHOLD STATUS:\n"
            f"  ├─ NEGATIVE_THRESHOLD: {NEGATIVE_THRESHOLD}\n"
            f"  ├─ EXTENDED_MALICIOUS_THRESHOLD: {EXTENDED_MALICIOUS_THRESHOLD}\n"
            f"  ├─ CONFIRMATION_ROUNDS_REQUIRED: {self.confirmation_rounds_required}\n"
            f"  └─ STRENGTHENING_ENABLED: {self.strengthening_enabled}\n"
            f"\nWEAK BACKDOOR PIPELINE:\n"
            f"  ├─ in_strengthening: {neighbor_id in self.weak_backdoor_nodes}\n"
        )
        if neighbor_id in self.weak_backdoor_nodes:
            info = self.weak_backdoor_nodes[neighbor_id]
            analysis += (
                f"  ├─ attempts: {info.get('attempts')}/{self.strengthening_max_attempts}\n"
                f"  ├─ started_round: {info.get('round_started')}\n"
                f"  └─ base_injection: {info.get('base_injection'):.1%}\n"
            )
        else:
            analysis += f"  └─ (not in strengthening)\n"

        analysis += f"{'='*70}\n"

        logging.error(analysis)



    def verify_model(self, model, clean_samples) -> tuple:
        """
        Verifica si el modelo es sospechoso.
        Retorna (is_suspicious: bool, severity: float)
        """
        if not self.detector:
            return False, 0.0

        # 1. Check Current Map
        is_suspicious_current = False
        rate_current = 0.0

        result = self.detector.check(model, clean_samples, self.current_map)
        if isinstance(result, tuple):
            if len(result) >= 2:
                is_suspicious_current = bool(result[0])
                rate_current = float(result[1])
                logging.info(
                    f"[HoneyManager] verify_model current_map -> raw_result_len={len(result)}, "
                    f"is_suspicious={is_suspicious_current}, rate_current={rate_current:.4f}, raw={result}"
                )
            else:
                is_suspicious_current = bool(result[0])
                rate_current = 1.0 if is_suspicious_current else 0.0
        else:
            is_suspicious_current = bool(result)
            rate_current = 1.0 if is_suspicious_current else 0.0

        if not is_suspicious_current:
            return False, 0.0

        # 2. Check Previous Map (If exists)
        if self.previous_map:
            result_prev = self.detector.check(model, clean_samples, self.previous_map)
            is_suspicious_prev = False
            if isinstance(result_prev, tuple):
                is_suspicious_prev = bool(result_prev[0]) if len(result_prev) >= 1 else False
            else:
                is_suspicious_prev = result_prev

            if not is_suspicious_prev:
                logging.info("[Safe] Node failed current map but passed previous map. marked as SAFE.")
                return False, 0.0

        # Failed checks. Return True and the Severity Rate of the current violation
        return True, rate_current

    # ============================================================================
    # OPTIMIZED NEIGHBOR VERIFICATION SYSTEM
    # ============================================================================

    def analyze_neighbor(self, neighbor_id, neighbor_model, current_round):
        """
        Analyzes neighbor model and updates verification status.

        MEMORY-BASED VERIFICATION LOGIC:
        - If node EVER showed backdoor (max_compliant >= 5%) → BENIGN
        - If node NEVER showed backdoor for 3 rounds → MALICIOUS
        - Prevents false positives from FedAvg dilution

        Returns: Current status ("TESTING", "BENIGN", "MALICIOUS")
        """
        # Use helper to ensure consistent initialization and prevent KeyErrors
        state = self._get_or_create_neighbor_state(neighbor_id, current_round)

        # Hoisted here so all code paths can access it without UnboundLocalError
        reputation_module = getattr(self.engine, "_reputation", None)

        # Skip if already verified as MALICIOUS (no need to re-verify)
        if state["status"] == "MALICIOUS":
            return "MALICIOUS"

        # Analyze model for backdoor presence (returns is_valid + has_backdoor + compliant_rate + is_suspicious + dominant_target + suspicious_rate + honest_rate)
        (
            is_valid,
            has_backdoor,
            compliant_rate,
            is_suspicious,
            dominant_target,
            suspicious_rate,
            honest_rate,
        ) = self._check_neighbor_has_backdoor(neighbor_model)

        logging.info(
            f"[Manager] 🔍 analyze_neighbor: {neighbor_id} @round {current_round} -> "
            f"CR={compliant_rate:.4%}, SR={suspicious_rate:.4%}, HR={honest_rate:.4%}, suspicious={is_suspicious}, "
            f"state(max_seen={state.get('max_compliant_seen'):.4f}, susp_cnt={state.get('suspicious_count')}, "
            f"neg_cnt={state.get('negative_count')}, status={state.get('status')})"
        )

        # Log comprehensive analysis details with all internal calculations
        self._log_neighbor_analysis_details(neighbor_id, current_round, is_valid, has_backdoor, compliant_rate, is_suspicious, state)

        if not is_valid:
            # Phase 6.3: Skip this round for this neighbor to avoid False Positives from lag artifacts
            logging.debug(
                f"[Manager] analyze_neighbor skip: invalid model for {neighbor_id} @round {current_round}. "
                f"Likely lag or missing validation batch."
            )
            return "TESTING"

        # Update history
        state["rounds_tested"] += 1
        state["compliant_history"].append((current_round, compliant_rate))
        state["max_compliant_seen"] = max(state["max_compliant_seen"], compliant_rate)

        # Persist last semantic metrics (CR/SR/HR)
        state["last_cr"] = float(compliant_rate)
        state["last_sr"] = float(suspicious_rate)
        state["last_hr"] = float(honest_rate)
        if dominant_target is not None:
            state.setdefault("dominant_target_history", []).append((current_round, dominant_target))
            if state.get("last_dominant_target") == dominant_target:
                state["dominant_target_streak"] = int(state.get("dominant_target_streak", 0)) + 1
            else:
                state["dominant_target_streak"] = 1
            state["last_dominant_target"] = dominant_target

        recent_targets = [target for _, target in state.get("dominant_target_history", [])[-3:]]
        stable_attack_target = (
            dominant_target is not None
            and (
                state.get("dominant_target_streak", 0) >= 2
                or recent_targets.count(dominant_target) >= 2
            )
        )

        # === SEMANTIC VERIFICATION (MadHoney Spec) ===
        TAU_SUSP = getattr(self.detector, "threshold", 0.40)
        EPS = 0.02
        TAU_HONEST = 0.80
        LOW_HONESTY_THRESHOLD = 0.35

        semantic_malicious_hit = (
            suspicious_rate >= TAU_SUSP
            and compliant_rate < EPS
            and honest_rate < LOW_HONESTY_THRESHOLD
        )
        if semantic_malicious_hit:
            semantic_rounds = state.setdefault("semantic_malicious_rounds", [])
            semantic_rounds.append(current_round)
            last_round = state.get("last_semantic_malicious_round")
            if last_round is not None and current_round - last_round <= 1:
                state["semantic_malicious_streak"] = int(state.get("semantic_malicious_streak", 0)) + 1
            else:
                state["semantic_malicious_streak"] = 1
            state["last_semantic_malicious_round"] = current_round
            state["status"] = "TESTING"
            logging.warning(
                f"[Manager] ⚠️ SEMANTIC MALICIOUS pattern observed for {neighbor_id}: "
                f"SR={suspicious_rate:.2%} >= {TAU_SUSP:.2%} & CR={compliant_rate:.2%} < {EPS:.2%}. "
                f"HR={honest_rate:.2%} < {LOW_HONESTY_THRESHOLD:.2%}. "
                f"Holding for confirmation (streak={state['semantic_malicious_streak']}, "
                f"target={dominant_target}, stable_target={stable_attack_target})."
            )
        else:
            state["semantic_malicious_streak"] = 0

        # ============================================================================
        # EXTREME-SIGNATURE STREAK (self-measured source discriminator, no reputation)
        # ============================================================================
        # Forensic result: in a fully-meshed topology the poison fingerprint reaches every
        # node in one hop, so CR / intensity / bait-response cannot separate source from
        # victim. The ONE axis that DID separate in the real logs is the EXTREME, SUSTAINED
        # SR/HR signature: the true attacker sits pinned at SR≈100% / HR≈0% every single round
        # (its 70%-poisoned local training dominates), whereas honest neighbors that merely
        # aggregate the diluted poison fluctuate (SR ~45-93%, with meaningful HR) and never
        # hold the absolute extreme continuously. We therefore track a SEPARATE streak with a
        # much stricter bar than the ordinary semantic streak, and require it to be SUSTAINED.
        EXTREME_SR = 0.98          # attacker pins ~100%; victims peaked ~93% but not sustained at ~100%
        EXTREME_HR = 0.02          # attacker ~0% honest; victims retain real honesty
        extreme_hit = (
            suspicious_rate >= EXTREME_SR
            and honest_rate <= EXTREME_HR
            and compliant_rate < EPS
        )
        if extreme_hit:
            last_ext = state.get("last_extreme_round")
            if last_ext is not None and current_round - last_ext <= 1:
                state["extreme_streak"] = int(state.get("extreme_streak", 0)) + 1
            else:
                state["extreme_streak"] = 1
            state["last_extreme_round"] = current_round
            logging.warning(
                f"[Manager] 🔴 EXTREME signature for {neighbor_id}: SR={suspicious_rate:.2%} (>= {EXTREME_SR:.0%}) "
                f"& HR={honest_rate:.2%} (<= {EXTREME_HR:.0%}) & CR={compliant_rate:.2%}. "
                f"extreme_streak={state['extreme_streak']} (sustained pinned-extreme = attacker signature)."
            )
        else:
            state["extreme_streak"] = 0

        # ============================================================================
        # PRIORITY CONVICTION: hybrid gate (local signature + reputation attribution)
        # ============================================================================
        # Evaluated BEFORE the legacy VICTIM-CARRIER / consensus gates so the source is not
        # intercepted and held in TESTING forever (the 07:56:10 / 15:24 "VICTIM CARRIER 90/3"
        # and manager.py:696 "qualified_reporters=0" stalls). Local signature detects the
        # attack presence fast; reputation is consulted only to attribute the source.
        if state["status"] != "MALICIOUS" and self._confirm_malicious(
            neighbor_id, state, current_round, reputation_module=reputation_module, reason="priority"
        ):
            return "MALICIOUS"

        # Case 2: Carrier → SR > 0 and CR > 0
        if suspicious_rate > 0.0 and compliant_rate >= EPS:
            state["carrier_suspect"] = True
            if state.get("status") != "MALICIOUS":
                state["status"] = "TESTING"
            logging.warning(
                f"[Manager] 🧩 SEMANTIC CARRIER: SR={suspicious_rate:.2%} & CR={compliant_rate:.2%}. "
                f"Pivot-through candidate: {neighbor_id}."
            )

        # Case 3: Benign → HR high (only after minimum evidence)
        if state["rounds_tested"] >= 3 and honest_rate >= TAU_HONEST and suspicious_rate < TAU_SUSP:
            state["status"] = "BENIGN"
            state["verified_round"] = current_round
            logging.info(
                f"[Manager] ✅ SEMANTIC BENIGN: HR={honest_rate:.2%} >= {TAU_HONEST:.2%}. "
                f"Verified {neighbor_id}."
            )
            return "BENIGN"

        # CONSISTENCY CHECK: A node must show compliance consistently to be considered "carrying" our bait.
        # This prevents FedAvg artifacts from being mistaken for real bait absorption.
        # RCA 22:45 (Fase 6.6): Reduced back to 0.02 (2%) because 15% was causing
        # False Positivos in Ring DFL due to signal dilution.
        BENIGN_COMPLIANT_THRESHOLD = 0.02
        BENIGN_CONSISTENT_ROUNDS   = 2
        BENIGN_WINDOW              = 3
        recent_history    = state["compliant_history"][-BENIGN_WINDOW:]
        consistent_rounds = sum(1 for _, r in recent_history if float(r) >= BENIGN_COMPLIANT_THRESHOLD)
        has_genuine_bait  = consistent_rounds >= BENIGN_CONSISTENT_ROUNDS

        # ============================================================================
        # DETECTION LOGIC (from first principles):
        #
        # COMPLIANT (model shows our backdoor) = BENIGN  → definitive, 100% certain
        #   A node that aggregated our bait MUST be honest. This overrides everything.
        #
        # NOT COMPLIANT = UNKNOWN → needs multiple rounds of testing
        #   Round 1: node hasn't received/aggregated bait yet → can't judge
        #   Round 2+: if STILL 0% compliance AND suspicious → accumulate evidence
        #   Round 3+: if consistently suspicious with 0% compliance → MALICIOUS
        #
        # SUSPICIOUS flag (from detector) is treated as evidence, NOT a verdict.
        # ============================================================================

        # Track suspicious rounds
        if is_suspicious:

            # CONTRADICTION CHECK: Does the attacker's target clash with our HoneyMap?
            # If our map expected Y' for samples of class Y, and the attacker predicts T,
            # and T is NOT Y', then it's a direct override of our bait.
            if dominant_target is not None and self.current_map:
                # Check if ANY rule in the honey_map is being overridden by the dominant_target
                clash_detected = False
                for y_origin, y_expected in self.current_map.items():
                    if dominant_target == y_origin and dominant_target != y_expected:
                         # The attacker is forcing the origin label (ignoring bait)
                         clash_detected = True
                         break

                if clash_detected:
                    # RCA 23:10:38 - IMPROVEMENT: If we detect a DIRECT CONTRADICTION CLASH,
                    # it means the node is intentionally ignoring our bait to favor its own target.
                    state["clash_count"] += 1
                    state.setdefault("clash_rounds", []).append(current_round)
                    logging.warning(
                        f"[Manager] ⚠️ CONTRADICTION CLASH on {neighbor_id} (targets {dominant_target}). "
                        f"Bait presence: {max(compliant_rate, state['max_compliant_seen']):.2%}. "
                        f"Verdict: CARRIER SUSPECT (Clash count: {state['clash_count']})."
                    )
                    # We let it fall through to accumulate suspicion_count

            # Update suspect history
            # NEW: Don't flag as suspicious if we didn't send bait (Bait Safety for trusted nodes)
            is_extra_trusted = reputation_module and reputation_module.get_score(neighbor_id) > 1.5

            if is_extra_trusted:
                logging.debug(f"[Manager] 🛡️ Node {neighbor_id} lacks bait but is Extra Trusted. Skipping suspicion increment.")
            else:
                # IMPORTANT: increment once per analyzed round to avoid artificial escalation
                state["suspicious_count"] += 1
                state.setdefault("suspicious_rounds", []).append(current_round)
                self.recent_detections[neighbor_id] = self.recent_detections.get(neighbor_id, 0) + 1
                logging.warning(
                    f"[Manager] ⚠️ Neighbor {neighbor_id} flagged SUSPICIOUS (count: {state['suspicious_count']}) "
                    f"- recent detections: {self.recent_detections[neighbor_id]}"
                )
        else:
            # DECAY suspicion (don't reset to 0 immediately) to allow for some noise but punish persistence
            if state["suspicious_count"] > 0:
                state["suspicious_count"] -= 1

        # ============================================================================
        # LAYERED DEFENSE DECISION TREE
        # ============================================================================

        # 1. ESCALATION: Prevent false positives on victims (Carrier Nodes)
        # In DFL, victims are "Smart Attackers" (Suspicious + Compliant).
        # We ONLY escalate to MALICIOUS if they are suspicious AND have NO bait,
        # OR if they triggered a direct CONTRADICTION CLASH with our HoneyMap.
        # FASE 3: Aumentado a 5 para evitar desconexiones prematuras en el anillo.

        # Absolute Suspicion Threshold (e.g. 80%) bypasses VICTIM CARRIER protection
        is_overwhelmingly_suspicious = (
            state["rounds_tested"] >= 3
            and state["suspicious_count"] > 0
            and (is_suspicious and getattr(self.detector, 'suspicious_rate', 0) > 0.80)
        )

        if state["suspicious_count"] >= 5 or is_overwhelmingly_suspicious:
             # If they have bait, we treat them as victims (CARRIERS).
             # We stay in TESTING to allow pivoting THROUGH them.
             # EXCEPT if they are overwhelmingly suspicious (>80% error rate directly against our bait)
             #
             # CRITICAL: gate on has_genuine_bait (consistent absorption across recent rounds),
             # NOT on max_compliant_seen. A single isolated FedAvg dilution spike (~3.12% in the
             # ring) would otherwise pin max_compliant_seen above the 2% threshold permanently and
             # trap a genuine attacker in VICTIM CARRIER forever, blocking conviction. A real carrier
             # absorbs bait repeatedly, so has_genuine_bait stays true across rounds.
             if has_genuine_bait and not is_overwhelmingly_suspicious:
                 logging.warning(
                     f"[Manager] ⚠️ Neighbor {neighbor_id} is persistently suspicious ({state['suspicious_count']}) "
                     f"BUT shows CONSISTENT bait ({consistent_rounds}/{BENIGN_WINDOW} recent rounds, "
                     f"max_seen={state['max_compliant_seen']:.2%}). Treating as VICTIM CARRIER."
                 )
                 # Note: We return "TESTING" but analyze_neighbors_at_current_node will see it's a good pivot target
                 return "TESTING"
             else:
                  if is_overwhelmingly_suspicious and has_genuine_bait:
                       logging.error(f"[Manager] ‼️ Overwhelmingly Suspicious (>80%) despite historical bait! Bypassing VICTIM CARRIER protection for {neighbor_id}.")
                       if self._confirm_malicious(neighbor_id, state, current_round, reputation_module, reason="overwhelming-suspicion"):
                           return "MALICIOUS"

                  # RCA 16:17 (Fase 5): If we are already planning to investigate (pivot through)
                  # because it's the only path, we give it 2 more rounds of grace even with 0% bait.
                  # This prevents P0 from flagging P9 as MALICIOUS before pivoting.
                  # RCA 23:40 (Fase 6.13): REINFORCED INQUIRY.
                  # We NEVER pivot to a node with 0% bait. We wait for differentiation.
                  # If we see 0% bait + poison, we trigger STRENGTHENING and wait 10 rounds.
                  # RCA 10:22 (Fase 6.14): CLASH SAFETY. If the node has shown a CLASH,
                  # it's a confirmed victim carrier. We allow pivoting and skip conviction.
                  # RCA 11:45 (Fase 8): ANALYTICAL IDENTITY (No protocol cheating).
                  # If we see CLASHES but NO bait after a short reinforcement (3 rounds),
                  # we know it's the attacker origin, not a victim.
                  # SAFETY: require multi-round evidence before conviction to reduce false positives.
                  CLASH_GRACE_ROUNDS = 3
                  MIN_CLASHES_FOR_CONVICTION = 2
                  MIN_ROUNDS_FOR_CONVICTION = 3
                  MIN_CONSECUTIVE_SUSP_ROUNDS = 3
                  external_reporters = 0
                  reporter_list = []
                  qualified_reporters = []
                  if reputation_module and hasattr(reputation_module, "get_reporters"):
                       try:
                           reporter_list = [
                               r for r in reputation_module.get_reporters(neighbor_id)
                               if r and r != getattr(self.engine, "addr", None)
                           ]
                           for reporter_id in sorted(set(reporter_list)):
                               reporter_score = 1.0
                               if hasattr(reputation_module, "get_score"):
                                   try:
                                       reporter_score = float(reputation_module.get_score(reporter_id))
                                   except Exception:
                                       reporter_score = 1.0

                               reporter_state = self.neighbor_tracking.get(reporter_id, {})
                               reporter_suspicious = reporter_state.get("suspicious_count", 0)

                               # Reporter quality gate: use reputation score only.
                               # reporter_benign/reporter_consistent require local neighbor_tracking
                               # state that is unavailable for remote nodes in a multi-hop topology,
                               # so those checks would always fail and leave qualified_reporters empty.
                               if reporter_suspicious <= 1 and reporter_score >= 0.75:
                                   qualified_reporters.append(reporter_id)

                           external_reporters = len(set(qualified_reporters))
                       except Exception:
                           external_reporters = 0

                  has_cross_confirmation = external_reporters >= 2
                  top_suspect, consensus_ambiguous, consensus_candidates = self._get_global_suspect_consensus(reputation_module)
                  suspicious_rounds = state.get("suspicious_rounds", [])
                  clash_rounds = state.get("clash_rounds", [])
                  # Compute consecutive suspicion streak length from the end
                  streak_len = 1
                  for i in range(len(suspicious_rounds) - 1, 0, -1):
                      if suspicious_rounds[i] - suspicious_rounds[i - 1] <= 1:
                          streak_len += 1
                      else:
                          break
                  has_recent_susp_streak = streak_len >= MIN_CONSECUTIVE_SUSP_ROUNDS
                  has_recent_clash = any((current_round - r) <= 2 for r in clash_rounds)
                  local_extreme_evidence = (
                      state["clash_count"] >= 3
                      and state["suspicious_count"] >= 10
                      and state["rounds_tested"] >= 8
                      and has_recent_susp_streak
                      and has_recent_clash
                  )

                  if state["clash_count"] > 0:
                       if (
                           state["clash_count"] >= MIN_CLASHES_FOR_CONVICTION
                           and state["rounds_tested"] >= MIN_ROUNDS_FOR_CONVICTION
                           and state["suspicious_count"] >= CLASH_GRACE_ROUNDS
                           and state["max_compliant_seen"] < BENIGN_COMPLIANT_THRESHOLD
                           and has_recent_susp_streak
                           and has_recent_clash
                           and stable_attack_target
                           and (has_cross_confirmation or local_extreme_evidence)
                       ):
                           if top_suspect and top_suspect != neighbor_id:
                               logging.warning(
                                   f"[Manager] 🌍 Holding conviction for {neighbor_id}: "
                                   f"global suspect leader is {top_suspect}, candidates={consensus_candidates}."
                               )
                               return "TESTING"
                           if consensus_ambiguous:
                               logging.warning(
                                   f"[Manager] 🌫️ Holding conviction for {neighbor_id}: "
                                   f"global suspect map is ambiguous {consensus_candidates}."
                               )
                               return "TESTING"
                           if not state.get("conviction_pending_round"):
                               state["conviction_pending_round"] = current_round
                               logging.warning(
                                   f"[Manager] ⏳ Conviction pending for {neighbor_id}. "
                                   f"Will re-check next round before blocking."
                               )
                               return "TESTING"

                           if (current_round - state.get("conviction_pending_round", current_round)) < 1:
                               return "TESTING"

                           logging.error(
                               f"[Manager] ‼️ ANALYTICAL IDENTITY signature for {neighbor_id}. "
                               f"Persistent Clashes + 0% Bait + "
                               f"{'cross-confirmation' if has_cross_confirmation else 'extreme local evidence'}."
                           )
                           if self._confirm_malicious(neighbor_id, state, current_round, reputation_module, reason="analytical-identity"):
                               return "MALICIOUS"
                           return "TESTING"

                       if (
                           state["clash_count"] >= MIN_CLASHES_FOR_CONVICTION
                           and state["rounds_tested"] >= MIN_ROUNDS_FOR_CONVICTION
                           and state["suspicious_count"] >= CLASH_GRACE_ROUNDS
                           and state["max_compliant_seen"] < BENIGN_COMPLIANT_THRESHOLD
                       ):
                           logging.warning(
                               f"[Manager] 🔍 Holding conviction for {neighbor_id}: "
                               f"missing robust evidence (qualified_reporters={external_reporters}, "
                               f"raw_reporters={sorted(set(reporter_list)) if reporter_list else []}, "
                               f"qualified={sorted(set(qualified_reporters)) if qualified_reporters else []}, "
                               f"recent_susp_streak={has_recent_susp_streak}, recent_clash={has_recent_clash}, "
                               f"stable_target={stable_attack_target}, consensus={consensus_candidates})."
                           )

                       # A node with persistent CLASHES + 0% bait is EITHER the source OR a
                       # relay victim. "VICTIM CARRIER → keep pivoting through" is only correct
                       # for the latter; applied to the source it makes the DFS try to pivot
                       # past a terminal attacker forever (the 07:56:10 .4↔.11 oscillation, where
                       # the gate was never reached). Consult the reputation-asymmetry gate here
                       # to disambiguate: a silent sink is the source → CONVICT; a participant is
                       # a victim → keep treating as carrier and pivot through.
                       if (
                           state["clash_count"] >= MIN_CLASHES_FOR_CONVICTION
                           and state["rounds_tested"] >= MIN_ROUNDS_FOR_CONVICTION
                           and state["max_compliant_seen"] < BENIGN_COMPLIANT_THRESHOLD
                           and self._confirm_malicious(neighbor_id, state, current_round, reputation_module, reason="clash-carrier-disambiguation")
                       ):
                           return "MALICIOUS"

                       logging.warning(f"[Manager] 🛡️ {neighbor_id} is suspicious but has CLASHES ({state['clash_count']}). Treating as VICTIM CARRIER (Grace: {state['suspicious_count']}/{CLASH_GRACE_ROUNDS}).")
                       return "TESTING"

                  # FIX: Reduced to 5 rounds (Phase 22) to avoid excessive delay in containment.
                  REINFORCEMENT_WAIT_ROUNDS = 5
                  if state["suspicious_count"] < REINFORCEMENT_WAIT_ROUNDS:
                       logging.warning(
                           f"[Manager] 🕵️ Neighbor {neighbor_id} is highly suspicious ({state['suspicious_count']}) "
                           f"with NO bait seen. REINFORCED INQUIRY: Holding pos and strengthening bait."
                       )
                       return "TESTING"

        # Count how many recent rounds exceeded the threshold (uses global BENIGN_COMPLIANT_THRESHOLD defined above)
        recent_history = state["compliant_history"][-BENIGN_WINDOW:]
        consistent_rounds = sum(1 for _, r in recent_history if float(r) >= BENIGN_COMPLIANT_THRESHOLD)
        has_consistent_bait = consistent_rounds >= BENIGN_CONSISTENT_ROUNDS

        # Still track first time ANY bait was seen (for logging / DFS carrier detection)
        if state["max_compliant_seen"] >= 0.02:
            if state["first_backdoor_round"] is None:
                state["first_backdoor_round"] = current_round

        if has_genuine_bait:
            is_active_reporter = bool(reputation_module and hasattr(reputation_module, "is_active_reporter") and reputation_module.is_active_reporter(neighbor_id))
            strong_consensus, consensus_candidates = self._has_strong_global_suspect_consensus(neighbor_id, reputation_module)
            semantic_sr_now = 0.0
            semantic_hr_now = 1.0
            semantic_cr_now = compliant_rate
            try:
                semantic_sr_now = float(suspicious_rate)
            except Exception:
                semantic_sr_now = 0.0
            try:
                semantic_hr_now = float(honest_rate)
            except Exception:
                semantic_hr_now = 1.0
            try:
                semantic_cr_now = float(compliant_rate)
            except Exception:
                semantic_cr_now = state["max_compliant_seen"]

            # Silent + suspicious nodes must not be upgraded to BENIGN with weak bait leakage.
            if not is_active_reporter and state["suspicious_count"] > 0:
                attack_target_persistent = (
                    stable_attack_target
                    or state.get("dominant_target_streak", 0) >= 2
                    or state.get("semantic_malicious_streak", 0) >= 1
                )
                first_contact_direct_attacker_signature = (
                    state["rounds_tested"] >= 2
                    and state["suspicious_count"] >= 1
                    and attack_target_persistent
                    and semantic_sr_now >= 0.75
                    and semantic_hr_now <= 0.16
                    and semantic_cr_now <= 0.20
                    and state["max_compliant_seen"] <= 0.25
                    and (
                        strong_consensus
                        or state["suspicious_count"] >= 2
                        or state["semantic_malicious_streak"] >= 1
                    )
                )
                if first_contact_direct_attacker_signature:
                    if self._confirm_malicious(neighbor_id, state, current_round, reputation_module, reason="first-contact-signature"):
                        return "MALICIOUS"
                severe_semantic_attacker_signature = (
                    state["rounds_tested"] >= 3
                    and state["suspicious_count"] >= 2
                    and stable_attack_target
                    and semantic_sr_now >= 0.70
                    and semantic_hr_now <= 0.16
                    and semantic_cr_now <= 0.20
                    and state["max_compliant_seen"] <= 0.40
                    and (
                        strong_consensus
                        or state["semantic_malicious_streak"] >= 2
                    )
                )
                if severe_semantic_attacker_signature:
                    if self._confirm_malicious(neighbor_id, state, current_round, reputation_module, reason="severe-semantic"):
                        return "MALICIOUS"
                if (
                    strong_consensus
                    and stable_attack_target
                    and state["rounds_tested"] >= 3
                    and state["suspicious_count"] >= 3
                    and state["semantic_malicious_streak"] >= 1
                    and state["max_compliant_seen"] <= 0.40
                ):
                    if self._confirm_malicious(neighbor_id, state, current_round, reputation_module, reason="diluted-bait-consensus"):
                        return "MALICIOUS"
                state["status"] = "TESTING"
                logging.warning(
                    f"[Manager] ⚠️ Holding {neighbor_id} in TESTING: silent node with suspicious history "
                    f"(susp={state['suspicious_count']}, max_bait={state['max_compliant_seen']:.2%})."
                )
                return "TESTING"

            # FEATURE: If node is CONSISTENTLY compliant but currently suspicious,
            # hold in TESTING — could be a smart adaptive attacker.
            if is_suspicious:
                if state["status"] == "BENIGN":
                    logging.warning(f"[Manager] ⚠️ Demoting {neighbor_id} from BENIGN to TESTING due to new suspicion.")
                state["status"] = "TESTING"
                logging.info(
                    f"[Manager] ⚠️ Neighbor {neighbor_id} is CONSISTENTLY COMPLIANT but SUSPICIOUS. "
                    f"Holding status as TESTING (Suspicion Count: {state['suspicious_count']})."
                )
                return "TESTING"

            # SAFETY RE-CHECK: High raw suspicion overrides compliance.
            if state["suspicious_count"] > 0 and state["rounds_tested"] > 0:
                raw_suspicion = state["suspicious_count"] / state["rounds_tested"]
                if raw_suspicion > 0.50:
                    if state["status"] == "BENIGN":
                        logging.warning(f"[Manager] ⚠️ Demoting {neighbor_id} to TESTING (High raw suspicion: {raw_suspicion:.1%})")
                    state["status"] = "TESTING"
                    return "TESTING"

            # Consistent compliance + no suspicion → truly BENIGN
            state["status"] = "BENIGN"
            state["verified_round"] = state["first_backdoor_round"]
            logging.info(
                f"[Manager] ✅ Neighbor {neighbor_id} VERIFIED as BENIGN in round {current_round} "
                f"(consistent={consistent_rounds}/{BENIGN_WINDOW} rounds ≥ {BENIGN_COMPLIANT_THRESHOLD:.0%}, "
                f"max_seen={state['max_compliant_seen']:.2%})"
            )
            return "BENIGN"



        else:
            # No backdoor ever seen at >= 2% → check strengthening / negative counters
            state["negative_count"] += 1

            # ============================================================================
            # TRIGGER STRENGTHENING: Not just for 0%, but also for suspicious nodes with low compliance
            # ============================================================================
            is_low_compliance_suspect = (is_suspicious and compliant_rate < 0.20)
            is_totally_clean = (compliant_rate == 0.0)

            if self.strengthening_enabled and (is_totally_clean or is_low_compliance_suspect) and neighbor_id not in self.weak_backdoor_nodes:
                # Initiate strengthening pipeline
                self.weak_backdoor_nodes[neighbor_id] = {
                    "round_started": current_round,
                    "attempts": 1,
                    "base_injection": self.base_injection_ratio,
                    "base_weight": self.base_weight_boost
                }
                logging.warning(
                    f"[Manager] 🔬 Triggering STRENGTHENING for {neighbor_id}. "
                    f"Reason: {'Suspicious with Low Compliance' if is_low_compliance_suspect else 'Zero Compliance'}"
                )
                try:
                    init_params = self.get_strengthened_params(neighbor_id)
                    logging.info(f"[Manager] 🔬 Initial strengthened params for {neighbor_id}: {init_params}")
                except Exception:
                    logging.debug(f"[Manager] 🔬 Failed to compute initial strengthened params for {neighbor_id}")

            # ============================================================================
            # ADAPTIVE STRENGTHENING PROGRESSION
            # ============================================================================
            if self.strengthening_enabled and neighbor_id in self.weak_backdoor_nodes:
                info = self.weak_backdoor_nodes[neighbor_id]

                if compliant_rate > 0.0 and not is_suspicious:
                    # Node showed compliance in this round. Update tracking but
                    # require consistency before declaring BENIGN (same rule as main path).
                    state["max_compliant_seen"] = max(state.get("max_compliant_seen", 0.0), compliant_rate)
                    if state["first_backdoor_round"] is None:
                        state["first_backdoor_round"] = current_round
                    # CONSISTENCY CHECK: need >= BENIGN_COMPLIANT_THRESHOLD in >= BENIGN_CONSISTENT_ROUNDS of BENIGN_WINDOW
                    # (Uses threadholds defined at the beginning of this method)
                    recent_history = state["compliant_history"][-BENIGN_WINDOW:]
                    consistent_rounds = sum(1 for _, r in recent_history if float(r) >= BENIGN_COMPLIANT_THRESHOLD)
                    if consistent_rounds >= BENIGN_CONSISTENT_ROUNDS:
                        logging.info(
                            f"[Manager] ✅ Node {neighbor_id} consistently BENIGN after strengthening "
                            f"({consistent_rounds}/{BENIGN_WINDOW} rounds ≥ {BENIGN_COMPLIANT_THRESHOLD:.0%})."
                        )
                        state["status"] = "BENIGN"
                        state["verified_round"] = current_round
                        self.weak_backdoor_nodes.pop(neighbor_id, None)
                        return "BENIGN"
                    else:
                        logging.info(
                            f"[Manager] 🔬 Node {neighbor_id} showed {compliant_rate:.2%} compliance but not yet consistent "
                            f"({consistent_rounds}/{BENIGN_WINDOW} rounds ≥ {BENIGN_COMPLIANT_THRESHOLD:.0%}). Continuing strengthening."
                        )
                        return "TESTING"

                elif info["attempts"] >= self.strengthening_max_attempts:
                    # EXHAUSTED - Still unverified after max attempts
                    # We simply pop them from the pipeline. Execution will fall through
                    # to the symmetric MALICIOUS vs CARRIER_SUSPECT determination below.
                    self.weak_backdoor_nodes.pop(neighbor_id, None)
                    logging.info(f"[Manager] ⚠️ Node {neighbor_id} exhausted strengthening pipeline.")
                else:
                    logging.info(f"[Manager] 🔬 Node {neighbor_id} in strengthening pipeline (attempt {info['attempts']}/3)")
                    return "TESTING"


            # MALICIOUS vs CARRIER_SUSPECT discrimination (fully symmetric with BENIGN rule)
            # ============================================================================
            # OLD: has_ever_shown_bait = max_compliant_seen > 1%
            #      → One accidental 6.25% round (FedAvg artifact) → CARRIER_SUSPECT (wrong)
            #
            # NEW: has_genuine_bait uses the SAME consistency bar as BENIGN:
            #      ≥5% compliance in ≥2 of the last 3 rounds.
            #      An attacker that occasionally mixes bait via FedAvg produces a one-off
            #      spike that won't meet this threshold.
            #      A genuine carrier consistently absorbs bait (it always aggregates with
            #      multiple honest neighbors, so the signal is stable).
            # ============================================================================
            MALICIOUS_ZERO_ROUNDS_REQUIRED = max(self.NEGATIVE_THRESHOLD, 3)
            if neighbor_id not in self.weak_backdoor_nodes and state["negative_count"] >= MALICIOUS_ZERO_ROUNDS_REQUIRED:
                # FASE 4: Priorizar evidencia histórica (max_compliant_seen) sobre consistencia inmediata.
                # En el anillo (3.12% dilución), el cebo puede saltar rondas.
                strong_consensus, consensus_candidates = self._has_strong_global_suspect_consensus(neighbor_id, reputation_module)
                diluted_bait_but_attacker = (
                    strong_consensus
                    and stable_attack_target
                    and state["rounds_tested"] >= 3
                    and state["suspicious_count"] >= 3
                    and state["semantic_malicious_streak"] >= 1
                    and state["max_compliant_seen"] <= 0.40
                )
                if diluted_bait_but_attacker:
                    if self._confirm_malicious(neighbor_id, state, current_round, reputation_module, reason="diluted-bait-strong-consensus"):
                        return "MALICIOUS"
                if has_genuine_bait:
                    # Genuine carrier: consistently absorbs bait (>=2 of last 3 rounds) → pivot through to find real source.
                    # NOTE: a single isolated max_compliant_seen spike (e.g. a one-round FedAvg
                    # dilution artifact in ring topologies, ~3.12%) is NOT sufficient on its own —
                    # requiring consistency here keeps genuine attackers from getting stuck
                    # permanently as CARRIER_SUSPECT, which would block fast conviction.
                    state["carrier_suspect"] = True
                    state["status"] = "TESTING"
                    logging.warning(
                        f"[Manager] 🚚 {neighbor_id} CARRIER_SUSPECT — consistent bait "
                        f"despite suspicion ({state['negative_count']} rounds). DFS will pivot through to find real source."
                    )
                    return "TESTING"
                else:
                    top_suspect, consensus_ambiguous, consensus_candidates = self._get_global_suspect_consensus(reputation_module)
                    if not stable_attack_target:
                        logging.warning(
                            f"[Manager] 🧭 Holding {neighbor_id} in TESTING: unstable dominant target "
                            f"(target={dominant_target}, history={recent_targets})."
                        )
                        return "TESTING"
                    if top_suspect and top_suspect != neighbor_id:
                        logging.warning(
                            f"[Manager] 🌍 Holding {neighbor_id} in TESTING: "
                            f"global suspect leader is {top_suspect}, candidates={consensus_candidates}."
                        )
                        return "TESTING"
                    if consensus_ambiguous:
                        logging.warning(
                            f"[Manager] 🌫️ Holding {neighbor_id} in TESTING: "
                            f"global suspect map is ambiguous {consensus_candidates}."
                        )
                        return "TESTING"
                    # RCA 22:11 (Fase 6): Increased to 10 to give honest nodes time
                    # to show bait even under heavy local poisoning.
                    EXTENDED_MALICIOUS_THRESHOLD = 10 # Increased from 5
                    if state["negative_count"] < EXTENDED_MALICIOUS_THRESHOLD:
                         logging.warning(
                             f"[Manager] 🕵️ Neighbor {neighbor_id} suspiciously lacks bait ({state['negative_count']} rounds) "
                             f"but hasn't reached EXTENDED_MALICIOUS_THRESHOLD ({EXTENDED_MALICIOUS_THRESHOLD}). Holding in TESTING."
                         )
                         return "TESTING"

                    # No genuine consistent bait + persistent suspicion → attacker signature.
                    # Still requires reputation-asymmetry corroboration to avoid convicting a
                    # relay victim that happens to sit at 0% bait.
                    if self._confirm_malicious(neighbor_id, state, current_round, reputation_module, reason="persistent-zero-bait"):
                        return "MALICIOUS"
                    return "TESTING"

            return "TESTING"




    def _check_neighbor_has_backdoor(self, neighbor_model):
        """
        Check if neighbor model contains the honeypot backdoor.

        Returns:
            tuple: (is_valid: bool, has_backdoor: bool, compliant_rate: float, is_suspicious: bool, dominant_target: int)
                - is_valid: True if model is structurally sound (not an empty lag artifact)
                - has_backdoor: True if compliant rate >= 2%
                - compliant_rate: Measure of backdoor presence (0.0 to 1.0)
                - is_suspicious: True if detector flagged SUSPICIOUS patterns
                - dominant_target: The label most targeted by suspicious predictions
        """
        if not self.detector or not neighbor_model:
            logging.debug(
                f"[Manager] _check_neighbor_has_backdoor early-exit: detector={bool(self.detector)}, "
                f"neighbor_model={type(neighbor_model)}"
            )
            return False, False, 0.0, False, None, 0.0, 0.0

        # Phase 6.3 (STABILITY): Model Sanity Check (Lag Protection)
        # Empty or malformed models (e.g., 3.4 KB artifacts) from system lag are ignored
        # to prevent False Positives during investigation.
        if len(neighbor_model) < 5:
            logging.warning(
                f"[Manager] Skipping model from neighbor (Sanity check failed: only {len(neighbor_model)} params). Likely lag."
            )
            return False, False, 0.0, False, None, 0.0, 0.0

        try:
            # Get validation data
            clean_batch = None
            if self.role_behavior and hasattr(self.role_behavior, '_engine'):
                engine = self.role_behavior._engine
                trainer = engine.trainer

                if hasattr(trainer, 'datamodule'):
                    trainer.datamodule.setup("fit")
                    val_loader = trainer.datamodule.val_dataloader()
                    clean_batch = next(iter(val_loader))

                if hasattr(trainer, 'model') and trainer.model and clean_batch:
                    # Save current model and always restore it, even if detector/logging fails.
                    current_params = {k: v.clone() for k, v in trainer.model.state_dict().items()}
                    try:
                        # Load neighbor's model
                        trainer.set_model_parameters(neighbor_model)

                        # Run detector check (receiving dominant_target)
                        result = self.detector.check(trainer.model, clean_batch, self.current_map)
                        if isinstance(result, tuple):
                            is_suspicious = bool(result[0])
                            severity = float(result[1]) if len(result) > 1 else 0.0
                            det_compliant_rate = float(result[2]) if len(result) > 2 else 0.0
                            dominant_target = result[3] if len(result) > 3 else None
                            suspicious_rate = float(result[4]) if len(result) > 4 else (severity if is_suspicious else 0.0)
                            honest_rate = float(result[5]) if len(result) > 5 else 0.0
                        else:
                            is_suspicious = bool(result)
                            severity = 1.0 if is_suspicious else 0.0
                            det_compliant_rate = 0.0
                            dominant_target = None
                            suspicious_rate = severity
                            honest_rate = 0.0

                        # Calculate compliant rate (DIRECTLY from detector)
                        compliant_rate = det_compliant_rate
                        has_backdoor = compliant_rate >= 0.02  # 2% threshold
                        model_size = len(neighbor_model) if hasattr(neighbor_model, "__len__") else "na"
                        model_type = type(neighbor_model).__name__
                        logging.info(
                            f"[Manager] 📊 _check_neighbor_has_backdoor(type={model_type}, size={model_size}) -> "
                            f"CR={compliant_rate:.4f}, SR={suspicious_rate:.4f}, HR={honest_rate:.4f}, "
                            f"has_backdoor={has_backdoor}, is_suspicious={is_suspicious}, dominant_target={dominant_target}"
                        )

                        return True, has_backdoor, compliant_rate, is_suspicious, dominant_target, suspicious_rate, honest_rate
                    finally:
                        trainer.model.load_state_dict(current_params)

                logging.debug(
                    f"[Manager] _check_neighbor_has_backdoor missing prerequisites: "
                    f"has_model={bool(getattr(trainer, 'model', None))}, clean_batch={bool(clean_batch)}"
                )
            else:
                logging.debug(
                    f"[Manager] _check_neighbor_has_backdoor missing role_behavior/engine for neighbor model check."
                )

        except Exception:
            logging.exception("[Manager] Error checking backdoor presence")

        return False, False, 0.0, False, None, 0.0, 0.0

    def should_send_backdoor(self, neighbor_id):
        """
        Decide if honeypot should send backdoored model to this neighbor.
        Only send to neighbors in TESTING status.
        """
        state = self.neighbor_tracking.get(neighbor_id, {"status": "TESTING"})
        decision = state["status"] == "TESTING"
        logging.info(f"[Manager] 💌 should_send_backdoor({neighbor_id}) -> status={state.get('status')} decision={decision}")
        return decision

    def get_neighbor_status(self, neighbor_id):
        """
        Get current verification status of a neighbor.
        Returns: "TESTING", "BENIGN", or "MALICIOUS"
        """
        state = self.neighbor_tracking.get(neighbor_id, {"status": "TESTING"})
        return state["status"]

    def register_visit(self, node_id: str):
        if node_id not in self.visited_history:
            self.visited_history.append(node_id)
        self.path_history.append(node_id)

    def is_visited(self, node_id: str) -> bool:
        return node_id in self.visited_history

    def get_confirmed_malicious_nodes(self) -> set:
        """Set of neighbor IDs confirmed MALICIOUS (and not benign carrier_suspects)."""
        return {
            nid for nid, st in self.neighbor_tracking.items()
            if st.get("status") == "MALICIOUS" and not st.get("carrier_suspect", False)
        }

    def next_hop_towards(self, topology: dict, my_id: str, target_id: str, avoid: set = None) -> str:
        """
        Shortest-path (BFS) first hop from my_id toward target_id over the topology.

        Proximity guidance for the DFS: when the honeypot has identified an attacker
        (locked_target) but its normal depth-first exploration hits a dead end — because
        every unvisited branch leads away from the attacker — we must be able to re-approach
        rather than getting stranded (the 16:52:44 failure: honeypot pivoted to a carrier
        topologically distant from the attacker and never moved again for 97 rounds).

        This deliberately IGNORES visited_history for pathing so the honeypot can re-tread
        nodes to close in on a known attacker. `avoid` (e.g. confirmed-malicious nodes we
        must never route THROUGH as an intermediate hop) is still respected.
        Returns the immediate next neighbor to pivot to, or None if unreachable.
        """
        if not topology or my_id not in topology or target_id is None:
            return None
        if my_id == target_id:
            return None
        avoid = avoid or set()
        # BFS from my_id; record the first hop taken on each shortest path.
        from collections import deque
        queue = deque()
        for nbr in topology.get(my_id, []):
            if nbr in avoid and nbr != target_id:
                continue
            queue.append((nbr, nbr))  # (current_node, first_hop)
        seen = {my_id}
        while queue:
            node, first_hop = queue.popleft()
            if node in seen:
                continue
            seen.add(node)
            if node == target_id:
                return first_hop
            for nbr in topology.get(node, []):
                if nbr in seen:
                    continue
                if nbr in avoid and nbr != target_id:
                    continue
                queue.append((nbr, first_hop))
        return None

    def update_current_node(self, node_id: str):
        """
        Update tracking when honeypot arrives at a new node.
        Resets the grace period counter for the new node.
        """
        if self.current_node_id != node_id:
            # New node: reset counter
            self.current_node_id = node_id
            self.rounds_at_current_node = 0
            self.missing_rounds = {}  # Reset patience tracking when pivoting
            logging.info(f"[Manager] 🎯 Arrived at new node: {node_id} (grace period reset)")
        else:
            # Same node: increment counter
            self.rounds_at_current_node += 1
            logging.info(f"[Manager] ⏱️ Round {self.rounds_at_current_node} at node {node_id}")

    def is_grace_period_active(self, neighbors_models=None) -> bool:
        """
        Check if we're still in the grace period at current node.
        Grace period = 2 rounds to allow backdoor propagation.

        EARLY EXIT OPTIMIZATION:
        If we already know all neighbors have the backdoor (from previous rounds/tracking),
        we can skip the rest of the grace period immediately.
        """
        # Basic check: have we spent enough rounds?
        basic_grace_active = self.rounds_at_current_node < self.grace_rounds_per_node

        if not basic_grace_active:
            return False

        return True

    def export_state(self):
        # CRITICAL: Include _last_pivot_source to prevent ping-pong
        last_pivot_source = None
        if self.role_behavior and hasattr(self.role_behavior, '_last_pivot_source'):
            last_pivot_source = self.role_behavior._last_pivot_source

        state = {
            "history": self.visited_history,
            "path_history": self.path_history,
            "reputation_history": self.reputation_history,
            "locked_target": self.locked_target,
            "navigation_target": self._navigation_target,
            "navigation_lock_hop_count": self._navigation_lock_hop_count,
            "transfer_source": getattr(self.engine, 'addr', None) if self.engine else None,
            "last_pivot_source": last_pivot_source,  # Prevent backtracking
            "rounds_at_current_node": self.rounds_at_current_node,  # Transfer grace counter
            "current_node_id": self.current_node_id,  # Transfer node position
            "suspect_confirmation": self.suspect_confirmation,  # Transfer suspect tracking
            "neighbor_tracking": self.neighbor_tracking,  # CRITICAL: Preserve neighbor memory across pivots
            "best_intensity_seen": self.best_intensity_seen,  # Gradient-ascent memory across hops
            "best_intensity_node": self.best_intensity_node,
        }
        if self.role_behavior and hasattr(self.role_behavior, "_dfs_path_stack"):
            state["dfs_path_stack"] = list(self.role_behavior._dfs_path_stack)
        if self.role_behavior and hasattr(self.role_behavior, "_honeypot_epoch"):
            state["honeypot_epoch"] = int(getattr(self.role_behavior, "_honeypot_epoch", 0))
        if self.role_behavior and hasattr(self.role_behavior, "_handover_mode"):
            state["handover_mode"] = getattr(self.role_behavior, "_handover_mode", "forward")
        if self.role_behavior and hasattr(self.role_behavior, "_temporarily_rejected_forward_targets"):
            state["temporarily_rejected_forward_targets"] = dict(
                getattr(self.role_behavior, "_temporarily_rejected_forward_targets", {})
            )
        if self.strategy:
            state["seed_state"] = self.strategy.get_state()
        return state

    def import_state(self, state):
        if not state:
            logging.warning("[Manager] Import State called with EMPTY state.")
            return None

        logging.info(f"[Manager] Importing State: keys={list(state.keys())}")

        if "history" in state: self.visited_history = state["history"]
        if "path_history" in state: self.path_history = state.get("path_history", [])
        if "reputation_history" in state: self.reputation_history = state.get("reputation_history", {})
        if "best_intensity_seen" in state: self.best_intensity_seen = float(state.get("best_intensity_seen", 0.0) or 0.0)
        if "best_intensity_node" in state: self.best_intensity_node = state.get("best_intensity_node")

        if "locked_target" in state:
            self.locked_target = state.get("locked_target")
            logging.info(f"[Manager] 🔓=>🔒 LOCKED TARGET imported: {self.locked_target}")
        else:
            logging.info("[Manager] No 'locked_target' in state.")

        if "navigation_target" in state:
            self._navigation_target = state.get("navigation_target")
            if self._navigation_target:
                logging.info(f"[Manager] 🧭 Navigation target imported: {self._navigation_target}")
        self._navigation_lock_hop_count = int(state.get("navigation_lock_hop_count", 0) or 0)

        # CRITICAL: Restore _last_pivot_source to prevent ping-pong
        if "last_pivot_source" in state and state["last_pivot_source"]:
            last_pivot_source = state["last_pivot_source"]
            if self.role_behavior:
                self.role_behavior._last_pivot_source = last_pivot_source
                logging.info(f"[Manager] 🔙 PIVOT SOURCE restored: {last_pivot_source} (prevents backtrack)")
        if self.role_behavior and "dfs_path_stack" in state:
            self.role_behavior._dfs_path_stack = list(state.get("dfs_path_stack") or [])
            logging.info(f"[Manager] 🪜 DFS path restored: {self.role_behavior._dfs_path_stack}")
        if self.role_behavior and "honeypot_epoch" in state:
            self.role_behavior._honeypot_epoch = int(state.get("honeypot_epoch", 0))
            logging.info(f"[Manager] 🕒 Honeypot epoch restored: {self.role_behavior._honeypot_epoch}")
        if self.role_behavior and "handover_mode" in state:
            self.role_behavior._handover_mode = state.get("handover_mode", "forward")
        if self.role_behavior and "temporarily_rejected_forward_targets" in state:
            self.role_behavior._temporarily_rejected_forward_targets = dict(
                state.get("temporarily_rejected_forward_targets", {}) or {}
            )
            logging.info(
                f"[Manager] 🚧 Temporarily rejected forward targets restored: "
                f"{self.role_behavior._temporarily_rejected_forward_targets}"
            )

        # Restore per-node grace period counter
        if "rounds_at_current_node" in state:
            self.rounds_at_current_node = state["rounds_at_current_node"]
            logging.info(f"[Manager] ⏱️ Rounds at current node: {self.rounds_at_current_node}")

        if "current_node_id" in state:
            self.current_node_id = state["current_node_id"]
            logging.info(f"[Manager] 📍 Current node position: {self.current_node_id}")

        # Restore suspect confirmation tracking
        if "suspect_confirmation" in state:
            self.suspect_confirmation = state["suspect_confirmation"]
            logging.info(f"[Manager] 🔍 Suspect confirmation tracking restored: {self.suspect_confirmation}")

        # Extract and RETURN transfer source so the engine can assign it to role_behavior
        transfer_source = state.get("transfer_source", None)
        if transfer_source:
            logging.info(f"[Manager] 🔍 Transfer source extracted: {transfer_source}")
        else:
            logging.info(f"[Manager] No transfer_source in state")

        if "seed_state" in state and self.strategy:
            # RESTORE SEED: Ensure the chaotic map continues the sequence from the previous Honeypot
            self.strategy.state = state["seed_state"]
            logging.info(f"🧬 [Manager] Defense Strategy Seed Restored: {state['seed_state']:.6f}")

        # Restore neighbor verification memory
        if "neighbor_tracking" in state:
            self.neighbor_tracking = state["neighbor_tracking"]
            logging.info(f"[Manager] 🧠 Neighbor tracking memory restored ({len(self.neighbor_tracking)} nodes)")

        return transfer_source

    def decide_pivot_target(self, reputation_module, my_id, threshold_trust=0.4):
        """
        LÓGICA DE CONSENSO:
        1. Ignora Ronda 0 (Datos inestables).
        2. Agrupa reportes de Ronda 1 en adelante.
        3. Devuelve el ID del ATACANTE (Threat) para que el Role calcule la ruta hacia él.
        """
        # PRIORIDAD: Objetivo Persistente (Evitar Ping-Pong)
        if self.locked_target:
             if str(self.locked_target) != str(my_id):
                 logging.info(f"🔒 [Manager] Persisting LOCKED TARGET: {self.locked_target}")
                 return self.locked_target
             else:
                 logging.info(f"🔓 [Manager] Locked target IS ME ({my_id}). Releasing lock for recalculation.")
                 self.locked_target = None

        if not reputation_module:
            return None

        # Datos crudos: (reporter, suspect, round) -> [scores]
        all_reports = reputation_module.reputation_with_all_feedback
        try:
            logging.info(f"[Manager] 📡 decide_pivot_target: raw_reports_count={len(all_reports)} (from reputation module)")
        except Exception:
            logging.info("[Manager] 📡 decide_pivot_target: cannot introspect raw reports")

        suspect_aggregation = {}
        reporters_activity = set()

        # Primera pasada: Recolectar actividad de reportes
        for (reporter, _, _), _ in all_reports.items():
            reporters_activity.add(reporter)

        for (reporter, suspect, rnd), scores in all_reports.items():
            if suspect == my_id: continue

            # --- FILTRO: IGNORAR R0 ---
            if rnd == 0: continue
            # --------------------------

            if suspect not in suspect_aggregation:
                suspect_aggregation[suspect] = []

            # Promediamos los scores de este reporte
            report_avg = sum(scores) / len(scores)
            suspect_aggregation[suspect].append(report_avg)

        # --- GENERACIÓN DE INFORME ---
        logging.info(f"📊 --- HONEYPOT INTELLIGENCE REPORT ---")

        worst_suspect = None
        worst_score = 100.0 # Score de riesgo (mientras mas bajo peor para confianza, pero aqui usaremos logica inversa o mantenemos 'worst_score' como reputacion baja)

        # Mantenemos 'worst_score' como reputacion: Queremos encontrar el MINIMO
        worst_score = 1.1

        if not suspect_aggregation:
            logging.warning("   (No intelligence gathered yet)")
            return None

        # Estrategia: "Silent Attacker Priority"
        # Si un nodo tiene reputación baja Y NO reporta a nadie, es mas sospechoso que uno que tiene rep baja pero participa.

        final_candidates = []

        for suspect in suspect_aggregation.keys():
            # CHANGE: Usar la reputación ACTUAL del modulo (que incluye Manual Updates) si existe
            consensus_score = 0.5
            if suspect in reputation_module.reputation and "reputation" in reputation_module.reputation[suspect]:
                 consensus_score = float(reputation_module.reputation[suspect]["reputation"])
            else:
                 # Fallback a calculo crudo si no está sincronizado
                 score_list = suspect_aggregation[suspect]
                 consensus_score = sum(score_list) / len(score_list)

            is_silent = suspect not in reporters_activity

            # Penalizacion por silencio: Si es silencioso, su reputacion efectiva percibida BAJA aun mas para darle prioridad
            effective_score = consensus_score

            # Usamos raw list solo para contar reports
            score_list = suspect_aggregation[suspect]
            num_reporters = len(score_list)

            if is_silent:
                status = "SILENT (Potential Threat)"
                # Penalizacion: El usuario indica que el atacante NUNCA reporta.
                # PERO: Cuidado con Falsos Positivos por lejanía en la red.

                # Solo marcamos como OBJETIVO ABSOLUTO si la reputación es YA muy mala.
                # Bajamos umbral de 0.8 a 0.5 para perdonar a nodos con problemas de red (victimas).
                if consensus_score < 0.5:
                     status += " [CONFIRMED]"
                     effective_score = -999.0 # Prioridad Absoluta
                else:
                     effective_score -= 10.0 # Castigo estándar para silenciosos dudosos

            else:
                status = "ACTIVE (Echo/Victim)"
                # BOOST: Si es activo (reporta), es muy probable que sea un eco.
                # Le SUMAMOS puntos para que NO sea elegido como target.
                effective_score += 5.0

            logging.info(f"   🎯 Suspect: {suspect} | Avg: {consensus_score:.2f} | Status: {status} | Eff Score: {effective_score:.2f}")

            final_candidates.append((suspect, effective_score))

        # --- MERGE INTO GLOBAL HISTORY (Shared Knowledge) ---
        # We merge local findings into the traveling 'brain' of the Honeypot across nodes.
        for suspect, score in final_candidates:
            if str(suspect) not in self.reputation_history:
                self.reputation_history[str(suspect)] = score
            else:
                # Keep the WORST (lowest) score seen globally. If it was flagged as threat once, we remember it.
                current_val = self.reputation_history[str(suspect)]
                if score < current_val:
                     self.reputation_history[str(suspect)] = score

        # --- DECIDE FROM GLOBAL HISTORY ---
        # Convert map to list for sorting
        global_rank = []
        for susp, score in self.reputation_history.items():
            global_rank.append((susp, score))

        # Ordenar por score efectivo (menor es mas peligroso)
        # Deterministic Sort: Score ascending, then ID ascending (to avoid ping-pong if ties)
        global_rank.sort(key=lambda x: (x[1], x[0]))

        worst_suspect = None
        worst_score = 100.0

        if global_rank:
            worst_suspect, worst_score = global_rank[0]
            logging.info(f"   🌍 GLOBAL CONSENSUS THREAT: {worst_suspect} (Eff: {worst_score:.2f})")
        else:
             logging.info("   (No global threats found yet)")

        logging.info("---------------------------------------------")

        # Devolvemos el ID de la AMENAZA si supera el umbral
        if worst_suspect and worst_score < threshold_trust:
            # FIX: Lock this target to ensure persistence during pivots
            self.locked_target = worst_suspect
            return worst_suspect

        return None

    def _get_or_create_neighbor_state(self, neighbor_id, current_round=0):
        """Helper to ensure neighbor_tracking dictionary is consistently populated."""
        if neighbor_id not in self.neighbor_tracking:
            self.neighbor_tracking[neighbor_id] = {
                "status": "TESTING",
                "negative_count": 0,
                "suspicious_count": 0,
                "rounds_tested": 0,
                "start_round": current_round,
                "verified_round": None,
                "max_compliant_seen": 0.0,
                "compliant_history": [],
                "first_backdoor_round": None,
                "suspicious_rounds": [],
                "clash_rounds": [],
                "conviction_pending_round": None
            }
            logging.info(f"[Manager] 🆕 Initializing tracking for neighbor: {neighbor_id}")

        # Repair missing keys (for backwards compatibility/partial updates)
        state = self.neighbor_tracking[neighbor_id]
        defaults = {
            "status": "TESTING",
            "negative_count": 0,
            "suspicious_count": 0,
            "rounds_tested": 0,
            "start_round": current_round,
            "verified_round": None,
            "max_compliant_seen": 0.0,
            "compliant_history": [],
            "first_backdoor_round": None,
            "suspicious_rounds": [],
            "clash_rounds": [],
            "conviction_pending_round": None,
            "carrier_suspect": False,  # True when suspicious but upstream attacker suspected
            "clash_count": 0,           # Counter for Honeymap contradictions (Honey target predictions)
        }
        for key, val in defaults.items():
            if key not in state:
                state[key] = val

        extra_defaults = {
            "semantic_malicious_streak": 0,
            "semantic_malicious_rounds": [],
            "last_semantic_malicious_round": None,
            "dominant_target_history": [],
            "last_dominant_target": None,
            "dominant_target_streak": 0,
            "extreme_streak": 0,
            "last_extreme_round": None,
        }
        for key, val in extra_defaults.items():
            if key not in state:
                state[key] = val

        return state

    def _reputation_asymmetry(self, neighbor_id, reputation_module):
        """
        The ONE discriminator that separates the attack SOURCE from a victim/relay.

        Forensic finding (runs 2026-07-03 10:43/14:00/17:56): local honeypot metrics
        (SR, CR, HR, clash_count, dominant_target) are IDENTICAL between the true poisoner
        and the honest neighbors that merely aggregate its poisoned gradients — victims are
        often even more extreme. Convicting on those metrics blocked innocent aggregators
        while the real attacker escaped. Worse, "sustained 0% bait" is INVERTED: the attacker
        aggregates its honest neighbors' bait-laden updates and leaks a little bait, whereas a
        clean relay victim can sit at exactly 0%.

        The robust asymmetry is behavioural in the reputation graph:
          - The attacker is a reputation SINK: accused by several honest peers, but it never
            accuses anyone (it won't denounce the poison it is producing) and does not actively
            participate in the reputation protocol.
          - A victim/relay PARTICIPATES: it reports on its own neighbors, so it both accuses
            others and is a recent active reporter.

        Returns dict with:
          accused_by      : number of distinct peers accusing neighbor_id
          accuses_others  : number of distinct nodes neighbor_id accuses
          is_active_reporter : whether neighbor_id recently participated in reputation
          is_silent_sink  : True iff strongly-accused AND accuses no one AND not an active reporter
        """
        result = {
            "accused_by": 0,
            "accuses_others": 0,
            "is_active_reporter": False,
            "is_silent_sink": False,
        }
        if not reputation_module:
            return result

        self_addr = getattr(self.engine, "addr", None)

        # How many distinct peers accuse this node (exclude ourself: our own bait-based
        # accusation must not be counted as independent corroboration).
        try:
            accusers = set(reputation_module.get_reporters(neighbor_id))
            accusers.discard(self_addr)
            result["accused_by"] = len(accusers)
        except Exception:
            pass

        # How many distinct nodes does THIS node accuse? (participation signal)
        try:
            latest = getattr(reputation_module, "latest_accusations", {}) or {}
            accuses = {
                suspect for suspect, reporters in latest.items()
                if neighbor_id in reporters and suspect != neighbor_id
            }
            result["accuses_others"] = len(accuses)
        except Exception:
            pass

        # Did it recently participate as a reporter at all?
        try:
            if hasattr(reputation_module, "is_active_reporter"):
                result["is_active_reporter"] = bool(
                    reputation_module.is_active_reporter(neighbor_id, freshness_window=5)
                )
        except Exception:
            pass

        # Silent sink = the attacker profile: corroborated by >=2 independent peers,
        # while itself denouncing no one and not participating in reputation.
        # Silent-sink = the attacker profile. The INFALSIFIABLE-by-a-victim part is the
        # silence: a real victim/relay participates in reputation (accuses its own neighbors
        # and/or is a recent active reporter), so `accuses_others == 0 AND not active_reporter`
        # is something only the true poisoner exhibits. We deliberately do NOT require a high
        # accused_by count: in sparse/edge topologies the attacker may have only one honest
        # neighbor able to accuse it (observed: only .2 accuses .4), and requiring >=2 would
        # let the real attacker escape. Corroboration is provided by the honeypot's own strong
        # local signature at the call site plus at least one independent accuser.
        result["is_silent_sink"] = (
            result["accuses_others"] == 0
            and not result["is_active_reporter"]
            and result["accused_by"] >= 1
        )
        return result

    def _bfs_distance(self, topology, src, dst):
        """Hop count from src to dst over topology, or None if unreachable/unknown."""
        if not topology or src not in topology or src == dst:
            return 0 if src == dst else None
        seen = {src}
        frontier = [src]
        dist = 0
        while frontier:
            dist += 1
            nxt = []
            for u in frontier:
                for v in topology.get(u, []):
                    if v in seen:
                        continue
                    if v == dst:
                        return dist
                    seen.add(v)
                    nxt.append(v)
            frontier = nxt
        return None

    def reputation_navigation_suspect(self, reputation_module, my_id=None, topology=None):
        """
        NAVIGATION-ONLY: which non-visited node should the honeypot travel toward?

        In sparse topologies the honeypot may not be a direct neighbor of the attacker, so it
        never measures it and never convicts (the 22:13:35 failure). Reputation is used here
        ONLY to pick a travel destination — NOT to convict and NOT to fetch models. The global
        accusation graph points at the attacker even from afar; we steer the honeypot toward the
        most-accused silent-sink candidate so it can become a direct neighbor and then run the
        normal local+reputation conviction there.

        STICKY TARGET (fix 2026-07-15a): this used to recompute the "best" candidate from
        scratch on every call, with no memory. Early in a run, many honest nodes simply haven't
        accused anyone yet (they look "silent" only for lack of data), so the ranking flip-flops
        round to round. We now lock onto a candidate and keep steering toward it as long as it
        remains a valid silent sink; only re-evaluate when it stops qualifying.

        PLAUSIBILITY BY DISTANCE (fix 2026-07-15b): the sticky lock alone just made the honeypot
        commit faithfully to whichever candidate happened to satisfy the silent-sink test FIRST
        in dict-iteration order — no notion of whether that candidate was anywhere near the
        honeypot's own reachable frontier. Observed on EMNIST edge0.2: it locked onto a dead-end
        node 4 hops away in the wrong branch while a stronger, 1-hop-closer candidate (on the
        actual shortest path to the attacker) sat unused. When `topology` is available we now
        rank candidates by BFS distance from `my_id` (closer = more plausible to be the reason
        we're seeing gossiped accusations at all) and only fall back to accused_by as a tie-break,
        instead of the reverse. Unreachable candidates are skipped entirely.

        Returns the candidate node id (most plausible silent sink) or None.
        """
        if not reputation_module:
            return None

        # If we're already committed to a target, keep going until we either reach it (it becomes
        # visited/analyzed) or a bounded number of ACTUAL HOPS have passed without success. This
        # is what turns "chase the loudest signal this instant" into "commit to a hypothesis and
        # walk the path to verify it" — the latter is what actually converges.
        #
        # NO RE-VALIDATION BY DESIGN (fix 2026-07-15d, replacing the 15c grace-counter attempt):
        # reputation visibility is NOT globally consistent — a candidate that looks silent from
        # one node's gossip view can look like an active reporter from a neighbor a single hop
        # away with fresher/more-complete accusation data (observed: .3 looked silent from .5,
        # but actively-reporting from .6). Re-validating "is this still a silent sink?" on every
        # call — even with a strike counter — ties abandonment to CALL CADENCE (~1/round while
        # stationary), which is a different, faster clock than the PIVOT cadence (gated by
        # grace_rounds_per_node): all 3 strikes could burn while the honeypot is still waiting out
        # its own per-node grace period, before it ever attempts the hop toward the target
        # (confirmed on EMNIST edge0.2, 4th attempt — strikes 1-3 all fired within 2 rounds at the
        # same host, zero hops attempted in between). So we no longer re-validate silent-sink
        # status once locked at all. The only ways to release a target now are: it becomes our
        # direct neighbor (handled by the caller, which stops navigating and analyzes locally),
        # it gets visited, or NAV_HOP_BUDGET actual hops elapse without reaching it — hop count is
        # tracked via len(visited_history) growth, which only advances on real pivots, so this
        # grace is measured in the same units as the journey itself.
        if self._navigation_target and self._navigation_target != my_id:
            if not self.is_visited(self._navigation_target):
                hops_since_lock = len(self.visited_history) - self._navigation_lock_hop_count
                if hops_since_lock < self.NAV_HOP_BUDGET:
                    return self._navigation_target
                logging.info(
                    f"[Manager] 🧭 Navigation target {self._navigation_target} not reached after "
                    f"{hops_since_lock} hops (budget={self.NAV_HOP_BUDGET}). Re-evaluating."
                )
            self._navigation_target = None

        try:
            latest = getattr(reputation_module, "latest_accusations", {}) or {}
        except Exception:
            return None

        # Rank candidates by (distance ascending, accused_by descending). Distance is the
        # primary key: a node we can actually reach soon is a more useful hypothesis to commit
        # to than a maximally-accused node stuck behind an unrelated branch of the graph.
        best, best_key = None, None
        for suspect, reporters in latest.items():
            if suspect == my_id or self.is_visited(suspect):
                continue
            asym = self._reputation_asymmetry(suspect, reputation_module)
            # Same bar as conviction's is_silent_sink: accuses no one, not an active reporter,
            # AND actually accused by someone. Without the accused_by>=1 requirement, a node
            # that simply hasn't reported yet (common early in a run) looks identical to a
            # silent attacker, which is what caused the mis-navigation.
            if not asym["is_silent_sink"]:
                continue
            if topology is not None and my_id is not None:
                dist = self._bfs_distance(topology, my_id, suspect)
                if dist is None:
                    continue  # unreachable in the known topology view — skip
            else:
                dist = 0  # no topology info available: fall back to pure accused_by ranking
            key = (dist, -asym["accused_by"])
            if best_key is None or key < best_key:
                best, best_key = suspect, key
        if best is not None:
            self._navigation_target = best
            self._navigation_lock_hop_count = len(self.visited_history)
            logging.info(
                f"[Manager] 🧭 Reputation navigation LOCKED onto {best} "
                f"(distance={best_key[0]}, accused_by={-best_key[1]}, silent). Hop budget={self.NAV_HOP_BUDGET}. "
                f"Honeypot will steer toward it until it becomes a direct neighbor, is reached, or budget expires."
            )
            return best
        return None

    def _poison_intensity(self, state) -> float:
        """
        Self-measured poison-intensity scalar for a neighbor (NO reputation).

        The attacker poisons 70% of its OWN data, so its model carries the poison at maximum
        concentration; neighbors only see FedAvg-diluted poison. Intensity therefore peaks at
        the source and decays with hop-distance. This scalar combines only quantities the
        honeypot measures directly:
          - suspicious_rate (SR): fraction of bait probes the node answers with the attack label
          - clash density: how often the node's dominant prediction directly overrides our
            honey_map remap (clashes per analyzed round)
          - target stability: a real source pushes ONE consistent target; relays wobble
          - sustained bait rejection: source never absorbs our directly-fed bait
        Range ~[0,1]; higher = closer to / is the source.
        """
        rounds = max(1, int(state.get("rounds_tested", 0)))
        sr = max(0.0, min(1.0, float(state.get("last_sr", 0.0))))
        clash_density = min(1.0, int(state.get("clash_count", 0)) / rounds)
        susp_density = min(1.0, int(state.get("suspicious_count", 0)) / rounds)
        target_stability = min(1.0, int(state.get("dominant_target_streak", 0)) / 4.0)
        bait_rejection = 1.0 if float(state.get("max_compliant_seen", 0.0)) < 0.02 else 0.0
        return (
            0.35 * sr
            + 0.25 * clash_density
            + 0.15 * susp_density
            + 0.15 * target_stability
            + 0.10 * bait_rejection
        )

    # Sustained-extreme conviction parameters (self-measured, reputation-free).
    # Local attack-signature streak required to OPEN an investigation (fast, low bar).
    LOCAL_STREAK_REQUIRED = 4     # consecutive rounds of strong local poison signature

    def _confirm_malicious(self, neighbor_id, state, current_round, reputation_module=None,
                           reason="", peer_intensities=None):
        """
        HYBRID conviction gate: LOCAL honeypot detection + REPUTATION for attribution only.

        Empirically established across all runs: the honeypot's LOCAL poison fingerprint
        (SR/CR/HR/clash/intensity) detects the PRESENCE of the attack but CANNOT attribute the
        SOURCE — it propagates via FedAvg, so victims score as high as the attacker (honest .2
        reached SR=93.75%, same as the attacker's peak; no SR threshold separates them). What
        DOES separate source from victim is a behavioural asymmetry in the reputation graph:
        the attacker is a SILENT SINK (it accuses no one and does not participate in reputation,
        while honest peers accuse it), whereas a victim/relay actively reports its neighbors.

        Division of labour (by design):
          * LOCAL (necessary trigger): a sustained strong attack signature — this is what the
            honeypot measures itself by feeding bait and probing the neighbor's model. It flags
            a node as attack-INVOLVED and opens the investigation fast (LOCAL_STREAK_REQUIRED).
          * REPUTATION (attribution ONLY): among nodes with the local signature, convict the one
            that is the silent sink. Reputation is used ONLY to disambiguate WHO the source is —
            never to fetch neighbor models and never to mitigate (containment stays local).

        Both are required: local signature (presence) AND reputation asymmetry (attribution).
        A victim has the local signature but participates in reputation → spared.
        """
        sr = float(state.get("last_sr", 0.0))
        hr = float(state.get("last_hr", 1.0))
        cr = float(state.get("last_cr", 0.0))
        clash = int(state.get("clash_count", 0))
        rounds_tested = int(state.get("rounds_tested", 0))
        # Local streak: reuse the semantic streak (SR>=TAU & CR<EPS & HR low) — the ordinary,
        # attainable attack signature (NOT the unreachable 98% extreme). This is the fast trigger.
        local_streak = max(
            int(state.get("semantic_malicious_streak", 0)),
            int(state.get("extreme_streak", 0)),
        )
        stable_target = (
            int(state.get("dominant_target_streak", 0)) >= 2
            or int(state.get("semantic_malicious_streak", 0)) >= 2
        )

        # 1) LOCAL trigger — necessary. Strong, sustained, attack-shaped signature + clashes.
        local_signature = (
            local_streak >= self.LOCAL_STREAK_REQUIRED
            and clash >= 3
            and stable_target
            and cr < 0.02
            and rounds_tested >= self.LOCAL_STREAK_REQUIRED
        )
        if not local_signature:
            logging.warning(
                f"[Manager] 🛡️ Holding {neighbor_id} in TESTING [{reason}]: local attack signature not yet "
                f"sustained (local_streak={local_streak}/{self.LOCAL_STREAK_REQUIRED}, clash={clash}, "
                f"stable_target={stable_target}, SR={sr:.2%}, CR={cr:.2%})."
            )
            return False

        # 2) REPUTATION attribution — the disambiguator. Only the SOURCE is a silent sink.
        asym = self._reputation_asymmetry(neighbor_id, reputation_module)
        is_silent_sink = (
            asym["accuses_others"] == 0
            and not asym["is_active_reporter"]
            and asym["accused_by"] >= 1
        )

        if is_silent_sink:
            state["status"] = "MALICIOUS"
            state["verified_round"] = current_round
            logging.critical(
                f"[Manager] 🚨 {neighbor_id} CONFIRMED MALICIOUS [{reason}/local+reputation] — sustained LOCAL "
                f"attack signature (local_streak={local_streak}, clash={clash}, SR={sr:.2%}, HR={hr:.2%}, "
                f"CR={cr:.2%}) ATTRIBUTED to source by reputation asymmetry "
                f"(accused_by={asym['accused_by']}, accuses_others={asym['accuses_others']}, "
                f"active_reporter={asym['is_active_reporter']})."
            )
            return True

        logging.warning(
            f"[Manager] 🛡️ Holding {neighbor_id} in TESTING [{reason}]: strong LOCAL signature "
            f"(local_streak={local_streak}) but reputation attribution NOT met "
            f"(silent_sink={is_silent_sink}: accused_by={asym['accused_by']}, "
            f"accuses_others={asym['accuses_others']}, active_reporter={asym['is_active_reporter']}). "
            f"Likely a victim/relay that participates in reputation — spared; DFS keeps investigating."
        )
        return False

    def _get_global_suspect_consensus(self, reputation_module, max_score_gap=0.08, min_reporters=1):
        if not reputation_module or not hasattr(reputation_module, "get_global_trust_map"):
            return None, False, []

        try:
            suspects_scores, _ = reputation_module.get_global_trust_map()
        except Exception:
            return None, False, []

        ranked = []
        for suspect, avg_score in suspects_scores.items():
            reporters = 0
            if hasattr(reputation_module, "get_reporters"):
                try:
                    reporters = len(set(reputation_module.get_reporters(suspect)))
                except Exception:
                    reporters = 0
            if reporters >= min_reporters:
                ranked.append((suspect, float(avg_score), reporters))

        ranked.sort(key=lambda item: (item[1], -item[2], item[0]))
        if not ranked:
            return None, False, []

        ambiguous = False
        if len(ranked) > 1:
            leader = ranked[0]
            runner_up = ranked[1]
            ambiguous = (
                abs(runner_up[1] - leader[1]) <= max_score_gap
                and runner_up[2] >= max(leader[2] - 1, 1)
            )

        return ranked[0][0], ambiguous, ranked[:3]

    def _has_strong_global_suspect_consensus(self, neighbor_id, reputation_module):
        top_suspect, ambiguous, candidates = self._get_global_suspect_consensus(reputation_module)
        if ambiguous or top_suspect != neighbor_id:
            return False, candidates

        leader_score = None
        leader_reporters = 0
        runner_up_score = None
        if candidates:
            leader_score = float(candidates[0][1])
            leader_reporters = int(candidates[0][2])
        if len(candidates) > 1:
            runner_up_score = float(candidates[1][1])

        strong_margin = runner_up_score is None or (runner_up_score - leader_score) >= 0.12
        strong_reporters = leader_reporters >= 2
        strong_score = leader_score is not None and leader_score <= 0.72
        return strong_margin and strong_reporters and strong_score, candidates

    def analyze_neighbors_at_current_node(self, neighbors_models: dict, reputation_module, came_from: str = None, my_neighbors: set = None) -> tuple:
        """
        Analiza los vecinos del nodo actual para detectar al atacante usando DFS + HoneyDoor.

        NUEVO SISTEMA DE GRACE PERIOD POR NODO:
        1. Llegar al nodo → Resetear contador
        2. Inyectar HoneyDoor durante 2 rondas (grace period)
        3. En la ronda 3: Analizar vecinos
           - Si vecino NO tiene backdoor → ATACANTE (detenerse)
           - Si todos tienen backdoor → PIVOTAR al siguiente nodo
        4. Repetir hasta encontrar al malicioso

        Args:
            neighbors_models: dict con {node_id: model_obj} para cada vecino
            reputation_module: módulo de reputación para checar silencio
            came_from: de dónde vinimos (para no retroceder)
            my_neighbors: set con los vecinos directos del nodo actual

        Returns:
            (found_attacker, attacker_id) o (False, next_pivot)
        """
        logging.info(f"[DFS] Analyzing {len(neighbors_models)} neighbors at current node...")
        try:
            # Log basic info about neighbor models to help debug missing bait propagation
            neighbor_summaries = {}
            for nid, m in (neighbors_models or {}).items():
                try:
                    size = len(m) if m is not None else 0
                except Exception:
                    size = -1
                neighbor_summaries[nid] = {"model_size": size, "visited": nid in self.visited_history}
            logging.debug(f"[DFS] Neighbor models summary: {json.dumps(neighbor_summaries)}")
        except Exception:
            logging.debug("[DFS] Failed to summarise neighbor models for logging")

        if my_neighbors is None:
            my_neighbors = set(neighbors_models.keys())

        # NUEVO: Usar sistema de grace period POR NODO con EARLY EXIT optimization
        grace_period_active = self.is_grace_period_active(neighbors_models)

        if grace_period_active:
            # During grace we STILL analyze every neighbor each round (so the sustained-extreme
            # streak accumulates and the attacker can be convicted immediately once it has held
            # SR≈100%/HR≈0% long enough — critical in fully-meshed topologies where the honeypot
            # is already adjacent to the attacker and needs no travel). Grace only suppresses the
            # PIVOT, not the analysis. If a neighbor reaches the sustained-extreme verdict, we
            # convict right now.
            current_round = getattr(self.engine, 'round', 0) if self.engine else 0
            for nid, model in neighbors_models.items():
                if nid == came_from:
                    continue
                status = self.analyze_neighbor(nid, model, current_round)
                if status == "MALICIOUS":
                    logging.critical(f"[DFS] 🎯 ATTACKER CONFIRMED during grace: {nid} (sustained-extreme).")
                    return (True, nid)

            logging.info(f"[DFS] ⏳ GRACE PERIOD at current node (round {self.rounds_at_current_node}/{self.grace_rounds_per_node}) — analyzing but holding pivot")
            return (False, self.HOLD_POSITION)

        # DESPUÉS DEL GRACE PERIOD: Analizar vecinos para detectar atacante
        logging.info(f"[DFS] ✅ Grace period COMPLETE - Starting neighbor analysis")

        # SAFETY GUARD: Ensure all unvisited neighbors have provided models for the current round.
        # This avoids premature "dead end" conclusions if a neighbor is just lagging (e.g. Experiment 18:29:21).
        if my_neighbors:
            waiting_for = []
            for neighbor_id in my_neighbors:
                if neighbor_id != came_from and not self.is_visited(neighbor_id):
                    # NEW: Skip node if it is permanently blocked in reputation
                    # This prevents synchronization deadlocks when a node is blocked but still in topology.
                    if reputation_module and hasattr(reputation_module, "permanently_blocked"):
                        if neighbor_id in reputation_module.permanently_blocked:
                            logging.info(f"[DFS] Skipping blocked neighbor {neighbor_id} in waiting guard.")
                            continue

                    if neighbor_id not in neighbors_models:
                        # TRACKING: Increment missing rounds
                        missing_count = self.missing_rounds.get(neighbor_id, 0) + 1
                        self.missing_rounds[neighbor_id] = missing_count

                        # Only wait if under patience threshold (3 rounds)
                        # We use 3 consistent rounds to be sure they are offline/blocked/lagging
                        # Increased from 1 round to allow for minor network drift
                        if missing_count <= 3:
                            waiting_for.append(neighbor_id)
                        else:
                            logging.warning(
                                f"[DFS] ⚠️ PATIENCE EXCEEDED for neighbor {neighbor_id} "
                                f"({missing_count} rounds missing). Proceeding without it."
                            )
                    else:
                        # Clean up missing rounds if we finally got it
                        self.missing_rounds.pop(neighbor_id, None)

            if waiting_for:
                logging.warning(f"[DFS] ⏳ HOLDING: Waiting for models from lagging neighbors: {waiting_for}")
                return (False, self.HOLD_POSITION)

        compliant_neighbors = []     # Neighbors verified BENIGN (bait + clean round)
        investigation_neighbors = [] # Neighbors suspicious BUT COMPLIANT (victims/carriers)
        suspicious_neighbors = []    # Neighbors suspicious AND NO BAIT (threats)
        current_round = getattr(self.engine, 'round', 0) if self.engine else 0
        peer_intensities = {}        # Self-measured poison intensity per analyzed neighbor

        for node_id, model_obj in neighbors_models.items():
            if node_id == came_from:
                logging.info(f"[DFS] Skipping {node_id} - came from there (no backtrack)")
                continue

            # ============================================================================
            # UNIFIED ANALYSIS: Use memory-based analyze_neighbor
            # ============================================================================
            status = self.analyze_neighbor(node_id, model_obj, current_round)

            # Self-measured poison intensity (for gradient-ascent pivoting + peak conviction)
            peer_intensities[node_id] = self._poison_intensity(self.neighbor_tracking.get(node_id, {}))

            # Use tracking state instead of non-existent detector methods
            state = self.neighbor_tracking.get(node_id, {})
            has_bait = state.get("max_compliant_seen", 0) >= 0.02
            is_currently_suspicious = state.get("suspicious_count", 0) > 0 # Simple heuristic for DFS branching

            if status == "MALICIOUS":
                # FP GUARD: Check if this node is actually a CARRIER_SUSPECT.
                # A carrier_suspect was flagged because it fails bait absorption, but the
                # reason is an upstream attacker (not its own poison). We must NOT convict
                # it immediately — instead pivot THROUGH it to expose the real attacker.
                node_state = self.neighbor_tracking.get(node_id, {})
                if node_state.get("carrier_suspect", False):
                    if self.is_visited(node_id):
                        logging.info(f"[DFS] 🔄 {node_id} is CARRIER_SUSPECT but already visited. Skipping re-pivot.")
                    else:
                        logging.warning(
                            f"[DFS] 🚚 {node_id} has MALICIOUS status but is a CARRIER_SUSPECT. "
                            f"Prioritizing as high-priority investigation target (pivot through)."
                        )
                        # High priority (score=2.0) — pivot through ASAP to find the real attacker
                        investigation_neighbors.append((node_id, 2.0))
                else:
                    # analyze_neighbor already applied the full conviction gate
                    # (_confirm_malicious: strong local signature AND reputation asymmetry —
                    # silent sink + accuser). A MALICIOUS status here is therefore already
                    # corroborated; no second, divergent silence check is needed.
                    logging.critical(f"[DFS] 🎯 ATTACKER CONFIRMED: {node_id} (Verdict from Manager)")
                    return (True, node_id)

            elif status == "BENIGN":
                compliant_neighbors.append((node_id, 0.0))
                logging.info(f"[DFS] ✅ {node_id} prioritized as COMPLIANT/BENIGN")
            elif has_bait:
                # Suspicious but has bait -> Target for investigation (Victim Carrier)
                investigation_neighbors.append((node_id, 1.0))
                logging.warning(f"[DFS] 🧩 {node_id} identified as CARRIER (Suspicious + Compliant). Prioritizing for investigation.")
            else:
                # Suspicious and NO bait seen yet
                # ─────────────────────────────────────────────
                # SINGLE-NEIGHBOR BLIND SPOT FIX:
                # If this node has been analyzed for at least 1 round but shows 0% bait,
                # it could still be a carrier (bait signal drowned by upstream attacker).
                # Put it in investigation_neighbors so DFS pivots THROUGH it rather than
                # holding indefinitely waiting for a confirmation that will never come.
                # We only do this if suspicious_count >= 1 (we have real evidence it's involved).
                # ─────────────────────────────────────────────
                node_state = self.neighbor_tracking.get(node_id, {})
                rounds_with_suspicion = node_state.get("suspicious_count", 0)
                clash_count = node_state.get("clash_count", 0)

                # RCA 23:40 (Fase 6.13): SAFETY FIRST - ZERO BAIT NO PIVOT.
                # RCA 10:22 (Fase 6.14): BRAVE PIVOT - If we have a CLASH, we pivot immediately.
                if clash_count > 0:
                    # ANTI-OSCILLATION / TERMINAL-SOURCE GUARD: "0% bait + clashes → pivot through"
                    # assumes the node is a RELAY victim. If we have ALREADY visited this node
                    # (the honeypot has been here / pivoted here before) and it STILL shows the
                    # attack signature, it is not a pass-through relay — it is the terminal source.
                    # Consult the reputation-asymmetry gate: a silent sink → CONVICT (stops the
                    # 07:56:10 attacker↔neighbor oscillation where the gate was never reached).
                    if self.is_visited(node_id) and self._confirm_malicious(
                        node_id, node_state, current_round, reputation_module, reason="dfs-terminal-clash-carrier"
                    ):
                        logging.critical(f"[DFS] 🎯 ATTACKER CONFIRMED at terminal carrier {node_id} (visited + silent sink).")
                        return (True, node_id)
                    logging.info(f"[DFS] 🧩 {node_id} has 0% bait but {clash_count} CLASHES. Victim carrier — pivoting through.")
                    investigation_neighbors.append((node_id, 2.0))
                elif rounds_with_suspicion >= 1:
                    logging.warning(f"[DFS] 🛡️ {node_id} has 0% bait/clashes. HOLDING position to differentiate via reinforcement.")
                    suspicious_neighbors.append((node_id, 2.0))
                else:
                    # Not enough evidence yet → pure monitoring
                    suspicious_neighbors.append((node_id, 1.0))
                    logging.warning(f"[DFS] ⏳ {node_id} is under TESTING/MONITORING (Current Status: {status})")

            # Prefer more convincing targets first to avoid drifting into safe but irrelevant branches.
            compliant_neighbors.sort(key=lambda item: item[0])
            investigation_neighbors.sort(key=lambda item: (-item[1], item[0]))
            suspicious_neighbors.sort(key=lambda item: (-item[1], item[0]))

        # ============================================================================
        # SELF-MEASURED SOURCE LOCALIZATION (gradient ascent on poison intensity)
        # ============================================================================
        if peer_intensities:
            peak_node = max(peer_intensities, key=peer_intensities.get)
            peak_intensity = peer_intensities[peak_node]
            logging.info(
                f"[DFS] 🌡️ Poison-intensity map: "
                f"{ {k: round(v,2) for k,v in sorted(peer_intensities.items(), key=lambda x:-x[1])} } "
                f"(peak={peak_node}:{peak_intensity:.2f}, best_seen={self.best_intensity_seen:.2f})"
            )

            # CONVICTION: is the peak neighbor the SOURCE? (local intensity peak + bait rejection)
            peak_state = self.neighbor_tracking.get(peak_node, {})
            if self._confirm_malicious(
                peak_node, peak_state, current_round, reputation_module=reputation_module,
                reason="gradient-peak", peer_intensities=peer_intensities
            ):
                logging.critical(f"[DFS] 🎯 SOURCE LOCALIZED: {peak_node} is the poison-intensity peak. CONVICTING.")
                return (True, peak_node)

            # GRADIENT ASCENT: if the peak is uphill (more intense than anything seen so far)
            # and unvisited, climb toward it — this drives the honeypot to the source fast.
            if peak_intensity > self.best_intensity_seen:
                self.best_intensity_seen = peak_intensity
                self.best_intensity_node = peak_node
            if peak_intensity >= self.SOURCE_INTENSITY * 0.5 and not self.is_visited(peak_node):
                logging.critical(
                    f"[DFS] ⛰️ GRADIENT ASCENT: climbing toward higher poison intensity via {peak_node} "
                    f"(intensity={peak_intensity:.2f}). Prioritizing this hop."
                )
                # Put the peak at the very front of the investigation queue.
                investigation_neighbors = [(peak_node, 3.0)] + [
                    it for it in investigation_neighbors if it[0] != peak_node
                ]

        # ============================================================================
        # PIVOT OR HOLD DECISION
        # ============================================================================

        # 1. Priority: Pivot to INVESTIGATION targets first (follow strongest evidence)
        if investigation_neighbors:
            for neighbor_id, priority in investigation_neighbors:
                if not self.is_visited(neighbor_id):
                    state = self.neighbor_tracking.get(neighbor_id, {})
                    raw_suspicion = state.get("suspicious_count", 0) / max(1, state.get("rounds_tested", 1))
                    has_bait = state.get("max_compliant_seen", 0) >= 0.02

                    logging.critical(
                        f"\n[DFS] 🧩 PIVOT DECISION: Follow poison trail through CARRIER\n"
                        f"   Target: {neighbor_id}\n"
                        f"   Priority: {priority}\n"
                        f"   Has bait: {has_bait} (max_seen={state.get('max_compliant_seen'):.2%})\n"
                        f"   Suspicion rate: {raw_suspicion:.1%} ({state.get('suspicious_count')}/{state.get('rounds_tested')} rounds)\n"
                        f"   Logic: Node carries our bait + is suspicious = compass to real attacker upstream\n"
                        f"   Action: PIVOT THROUGH to expose poison source\n"
                    )
                    return (False, neighbor_id)

        # 2. Then explore verified BENIGN branches.
        if compliant_neighbors:
            for neighbor_id, _ in compliant_neighbors:
                if not self.is_visited(neighbor_id):
                    logging.critical(
                        f"\n[DFS] ✅ PIVOT DECISION: Explore BENIGN branch\n"
                        f"   Target: {neighbor_id}\n"
                        f"   Reason: Verified as BENIGN (shows consistent bait, clean behavior)\n"
                        f"   Logic: Safe branch exploration (Standard DFS)\n"
                    )
                    return (False, neighbor_id)

        # 3. If everything unvisited is a pure THREAT (no bait), we HOLD to confirm
        if suspicious_neighbors:
            suspect_id = suspicious_neighbors[0][0]
            state = self.neighbor_tracking.get(suspect_id, {})
            rounds = state.get("suspicious_count", 0)
            neg_rounds = state.get("negative_count", 0)

            logging.critical(
                f"\n[DFS] ⏳ PIVOT DECISION: NO PIVOT - HOLD TO CONFIRM\n"
                f"   Suspect: {suspect_id}\n"
                f"   Zero bait evidence: {state.get('max_compliant_seen'):.4f} (threshold: 0.02)\n"
                f"   Suspicious rounds: {rounds}\n"
                f"   Negative rounds: {neg_rounds}/{self.NEGATIVE_THRESHOLD}\n"
                f"   Policy: Will wait {self.confirmation_rounds_required} rounds OR reach EXTENDED_MALICIOUS_THRESHOLD (10 neg rounds)\n"
                f"   Action: HOLD & STRENGTHEN to differentiate weak backdoor (benign) from strong filtering (malicious)\n"
            )
            return (False, self.HOLD_POSITION)

        # 4. If everything visited/analyzed, we stay put (Active Monitoring)
        logging.error(
            f"\n[DFS] 🏁 PIVOT DECISION: DEAD END\n"
            f"   All neighboring branches already VISITED: {list(self.visited_history)}\n"
            f"   Action: HOLD POSITION for local monitoring\n"
            f"   Stronghold details:\n"
            f"     - Locked target: {self.locked_target}\n"
            f"     - Weak backdoor nodes (strengthening): {list(self.weak_backdoor_nodes.keys())}\n"
            f"     - Visited: {list(self.visited_history)}\n"
        )
        return (False, self.BACKTRACK_REQUIRED)

    def _is_node_silent_to_neighbors(self, suspect_node: str, my_neighbors: set, reputation_module) -> bool:
        """
        Verifica si un nodo es SILENCIOSO a sus vecinos.
        Un nodo es silencioso si sus vecinos directos NO reciben reportes de reputación de él.

        Es decir: "suspect_node NO ha reportado reputación sobre sus vecinos (el honeypot y otros)"

        Args:
            suspect_node: el nodo a verificar
            my_neighbors: conjunto de vecinos del nodo actual (incluye al suspect_node)
            reputation_module: módulo de reputación

        Returns:
            True si el nodo es silencioso, False en caso contrario
        """
        if not reputation_module:
            return False

        if not hasattr(reputation_module, 'reputation_with_all_feedback'):
            return False

        all_reports = reputation_module.reputation_with_all_feedback

        # Buscar si "suspect_node" ha reportado sobre alguno de sus vecinos (my_neighbors)
        # Si el atacante fuera honesto, estaría reportando sobre otros nodos
        # Pero si es silencioso, NO reporta nada

        has_reported_anything = False

        for (reporter, target, round_num), scores in all_reports.items():
            # ¿Este reporte viene de "suspect_node"?
            if reporter == suspect_node:
                # Sí, el suspect_node ha reportado sobre "target"
                # Verifica si "target" es alguno de nuestros vecinos
                if target in my_neighbors or target in [str(n) for n in my_neighbors]:
                    has_reported_anything = True
                    logging.info(f"[DFS] {suspect_node} HAS reported on {target} (its neighbors)")
                    break

        is_silent = not has_reported_anything

        if is_silent:
            logging.warning(f"[DFS] {suspect_node} is SILENT - hasn't reported on any of its neighbors {my_neighbors}")

        return is_silent

    def get_dfs_pivot_direction(self, topology: dict, my_id: str, came_from: str = None, exclude_nodes: set = None) -> str:
        """
        Selecciona la siguiente dirección de pivotaje usando DFS.

        Dado la topología, elige un vecino del nodo actual hacia el cual pivotar.
        Evita retroceder (came_from) y nodos excluidos.

        Args:
            topology: dict con {node_id: [neighbors]}
            my_id: mi ID actual
            came_from: de dónde vinimos (para evitar retroceso)
            exclude_nodes: conjunto de nodos a evitar (ej. sospechosos)

        Returns:
            next_node_id o None
        """
        if my_id not in topology:
            logging.warning(f"[DFS] Node {my_id} not in topology")
            return None

        neighbors = list(topology[my_id])

        # Filtrar: no retroceder
        if came_from:
            neighbors = [n for n in neighbors if n != came_from]

        # Filtrar: excluidos
        if exclude_nodes:
            neighbors = [n for n in neighbors if n not in exclude_nodes]

        # Prioridad: seleccionar el que no hemos visitado aún
        for neighbor in neighbors:
            if neighbor not in self.visited_history:
                nt = self.neighbor_tracking.get(neighbor, {})
                if nt.get("status") == "MALICIOUS" and not nt.get("carrier_suspect", False):
                    logging.warning(f"[DFS] Skipping confirmed MALICIOUS node {neighbor} as pivot target.")
                    continue
                logging.info(f"[DFS] Next pivot direction: {neighbor} (unvisited)")
                return neighbor

        if neighbors:
            logging.info(f"[DFS] All forward neighbors already visited from {my_id}. Need backtrack.")
            return None

        logging.warning(f"[DFS] No available next hop from {my_id}")
        return None
