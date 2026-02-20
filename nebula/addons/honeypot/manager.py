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
    def __init__(self, config=None, engine=None, seed: float = 0.5, role_behavior=None):
        self.config = config
        self.engine = engine
        self.role_behavior = role_behavior  # Reference to the HoneypotRoleBehavior

        real_seed = seed
        if isinstance(config, (float, int)):
            real_seed = config

        if DefenseStrategyGenerator:
            self.strategy = DefenseStrategyGenerator(real_seed)
            self.detector = HoneyDetector()
        else:
            self.strategy = None
            self.detector = None

        self.current_map = {}
        self.previous_map = {}  # Store previous map to reduce false positives
        self.visited_history = []
        self.reputation_history = {}  # Historial global acumulado
        self.locked_target = None     # Fixed Target

        # PERMANENT MAP: The honey_map remains constant throughout the entire execution
        # to ensure benign nodes have enough time to converge on the bait.
        self.map_round_counter = 0

        # PER-NODE Grace Period: Track rounds spent at current node
        self.rounds_at_current_node = 0  # Reset when pivoting
        self.grace_rounds_per_node = 1   # Speed up search (Reduced from 2)
        self.current_node_id = None      # Track which node we're at

        # SUSPECT CONFIRMATION: Track suspects before declaring as attackers
        # Format: {node_id: confirmation_rounds}
        # If a suspect gets backdoor during confirmation → False positive, continue search
        # If after 3 rounds still no backdoor → Confirmed attacker
        self.suspect_confirmation = {}    # Track rounds monitoring each suspect
        self.confirmation_rounds_required = 3  # INCREASED: Wait 3 rounds before confirming attacker (reduced false positives)

        # ============================================================================
        # OPTIMIZED NEIGHBOR TRACKING SYSTEM
        # Fast benign verification (1 round) + Conservative malicious confirmation (3 rounds)
        # ============================================================================
        self.neighbor_tracking = {}  # {node_id: {status, negative_count, start_round, verified_round}}
        self.NEGATIVE_THRESHOLD = 3  # INCREASED: 3 consecutive negative rounds to allow for bait learning
        self.recent_detections = {}  # {node_id: detection_count} - Fast safety buffer
        self.missing_rounds = {}     # {node_id: missing_count} - Patience limit for synchronization

        # ============================================================================
        # ADAPTIVE BACKDOOR STRENGTHENING SYSTEM
        # Progressively strengthen backdoor for nodes showing 0% to distinguish
        # weak backdoor (benign) from active filtering (malicious)
        # ============================================================================
        self.weak_backdoor_nodes = {}  # {node_id: {round_started, attempts, original_params}}
        self.strengthening_enabled = True
        self.strengthening_max_attempts = 3   # OPTIMIZED: 3 attempts (User Request: Fast & Strong)
        self.strengthening_injection_step = 1.0   # +100% factor (doubles ratio) per attempt
        self.strengthening_weight_step = 2.0      # +2.0x weight per attempt -> x3.0, x5.0, x7.0
        self.strengthening_lr_step = 2.0          # +2.0x LR per attempt -> x3.0, x5.0, x7.0
        self.base_injection_ratio = 0.3           # Start at 30% (Aggressive start)
        self.base_weight_boost = 1.5              # Base weight boost
        self.base_lr_boost = 1.0                  # Base LR boost

        # Generate initial map
        if self.strategy:
            self.current_map = self.strategy.get_honey_map()
            logging.info(f"🔄 [HoneyManager] INITIAL Honey Map Generated: {self.current_map}")


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
                    logging.info(
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
             is_suspicious_current, rate_current = result
        else:
             is_suspicious_current = result
             rate_current = 1.0 if result else 0.0

        if not is_suspicious_current:
            return False, 0.0

        # 2. Check Previous Map (If exists)
        if self.previous_map:
            result_prev = self.detector.check(model, clean_samples, self.previous_map)
            is_suspicious_prev = False
            if isinstance(result_prev, tuple):
                 is_suspicious_prev, _ = result_prev
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

        # Skip if already verified as MALICIOUS (no need to re-verify)
        if state["status"] == "MALICIOUS":
            return "MALICIOUS"

        # Analyze model for backdoor presence (returns bool + compliant_rate + is_suspicious + dominant_target)
        has_backdoor, compliant_rate, is_suspicious, dominant_target = self._check_neighbor_has_backdoor(neighbor_model)

        # Update history
        state["rounds_tested"] += 1
        state["compliant_history"].append((current_round, compliant_rate))
        state["max_compliant_seen"] = max(state["max_compliant_seen"], compliant_rate)

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
            state["suspicious_count"] += 1

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
                    # RCA 21:37:11 - Don't block aggregators that are honest but aggregate poison.
                    # If the node has bait (current or historical), we treat the clash as a
                    # "Smart Attacker" signal rather than an immediate block.
                    if has_backdoor or state["max_compliant_seen"] >= 0.02:
                         logging.info(
                             f"[Manager] ⚠️ CONTRADICTION CLASH on {neighbor_id} (targets {dominant_target}) "
                             f"BUT has bait ({max(compliant_rate, state['max_compliant_seen']):.2%}). "
                             "Marking as SUSPICIOUS but not blocking yet (Aggregator Resilience)."
                         )
                    else:
                        # FIX: In global Honey Maps (all labels mapped), EVERY dominant target clashes with some rule.
                        # We must NOT instantly ban here, otherwise innocent non-IID nodes or aggregators get falsely banned
                        # before they even have a chance to learn the bait in round 1.
                        logging.warning(f"[Manager] ⚠️ CONTRADICTION CLASH on {neighbor_id} (targets {dominant_target}). Waiting for patience limit.")

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
        if state["suspicious_count"] >= 3:
             # If they have bait, we treat them as victims (CARRIERS).
             # We stay in TESTING to allow pivoting THROUGH them.
             if state["max_compliant_seen"] >= 0.02:
                 logging.warning(
                     f"[Manager] ⚠️ Neighbor {neighbor_id} is persistently suspicious ({state['suspicious_count']}) "
                     f"BUT has bait ({state['max_compliant_seen']:.2%}). Treating as VICTIM CARRIER."
                 )
                 # Note: We return "TESTING" but analyze_neighbors_at_current_node will see it's a good pivot target
                 return "TESTING"
             else:
                 logging.critical(
                     f"[Manager] 🚨 Neighbor {neighbor_id} CONTRADICTION: Persistent suspicion ({state['suspicious_count']}) "
                     f"with NO bait seen. Verdict: MALICIOUS"
                 )
                 state["status"] = "MALICIOUS"
                 state["verified_round"] = current_round
                 return "MALICIOUS"

        # 2. VERIFICATION: Backdoor presence — CONSISTENCY REQUIRED
        # ============================================================================
        # FALSE NEGATIVE FIX:
        # The old rule (max_compliant_seen >= 2%) declared BENIGN on a single round.
        # An attacker that aggregates honest neighbors can accidentally exceed 2% in
        # one round (aggregation artifact) and get permanently marked BENIGN.
        #
        # NEW RULE: A node must show compliance >= BENIGN_COMPLIANT_THRESHOLD in at
        # least BENIGN_CONSISTENT_ROUNDS of the last BENIGN_WINDOW rounds.
        # This rejects one-off flukes from FedAvg dilution.
        # ============================================================================
        BENIGN_COMPLIANT_THRESHOLD = 0.05   # Minimum compliance per round (5%)
        BENIGN_CONSISTENT_ROUNDS   = 2      # Must hit threshold in N of last W rounds
        BENIGN_WINDOW              = 3      # Look back window

        # Count how many recent rounds exceeded the threshold
        recent_history = state["compliant_history"][-BENIGN_WINDOW:]
        consistent_rounds = sum(1 for _, r in recent_history if float(r) >= BENIGN_COMPLIANT_THRESHOLD)
        has_consistent_bait = consistent_rounds >= BENIGN_CONSISTENT_ROUNDS

        # Still track first time ANY bait was seen (for logging / DFS carrier detection)
        if state["max_compliant_seen"] >= 0.02:
            if state["first_backdoor_round"] is None:
                state["first_backdoor_round"] = current_round

        if has_consistent_bait:
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
                    BENIGN_COMPLIANT_THRESHOLD = 0.05
                    BENIGN_CONSISTENT_ROUNDS   = 2
                    BENIGN_WINDOW              = 3
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
                    self.weak_backdoor_nodes.pop(neighbor_id, None)

                    if state["suspicious_count"] >= 3:
                        # ====================================================================
                        # CARRIER-VS-ATTACKER DISCRIMINATION (False Positive Prevention)
                        # -------------------------------------------------------------------
                        # A node that fails bait absorption consistently could be EITHER:
                        #   (A) A genuine attacker (generates poison locally)
                        #   (B) A carrier (honest node whose FedAvg is dominated by an upstream attacker)
                        #
                        # A carrier CANNOT absorb our bait because its stronger upstream neighbor
                        # (e.g. P8 sending with weight 1M) overrides our bait signal every round.
                        #
                        # STRATEGY: Instead of immediately convicting the node as MALICIOUS,
                        # mark it as CARRIER_SUSPECT and return TESTING. The DFS will then
                        # pivot THROUGH this node to inspect its neighbors — exposing the
                        # real attacker (e.g. P8) one hop further.
                        # ====================================================================
                        state["carrier_suspect"] = True
                        # Keep TESTING status so DFS can pivot through
                        state["status"] = "TESTING"
                        state["verified_round"] = current_round
                        logging.warning(
                            f"[Manager] 🚚 Neighbor {neighbor_id} CARRIER_SUSPECT — suspicious but "
                            f"bait never absorbed (upstream attacker likely). DFS will pivot through "
                            f"to find real source. (suspicious_count={state['suspicious_count']})"
                        )
                        return "TESTING"
                    else:
                        # Not suspicious but unverified after strengthening → TESTING, not BENIGN.
                        # Could be catastrophic forgetting or just a slow learner.
                        # DFS will continue to monitor without convicting.
                        logging.warning(f"[Manager] ⚠️ Node {neighbor_id} unverified after strengthening. Keeping TESTING (not enough evidence for BENIGN).")
                        return "TESTING"
                else:
                    logging.info(f"[Manager] 🔬 Node {neighbor_id} in strengthening pipeline (attempt {info['attempts']}/3)")
                    return "TESTING"

            # MALICIOUS consistency check (symmetric with BENIGN consistency fix)
            # ============================================================================
            # Old rule: negative_count >= threshold → immediately MALICIOUS
            # Problem: a carrier or a node under only 1 analysis round gets convicted too fast.
            #
            # NEW RULE: Declare MALICIOUS only if the node has NEVER shown any bait (max_compliant=0)
            # AND has been suspicious for >= NEGATIVE_THRESHOLD rounds consistently.
            # This mirrors the BENIGN rule: if we need N consistent rounds for BENIGN,
            # we should require the same for MALICIOUS.
            # ============================================================================
            MALICIOUS_ZERO_ROUNDS_REQUIRED = max(self.NEGATIVE_THRESHOLD, 3)  # At least 3 rounds of 0% bait + suspicion
            has_ever_shown_bait = state.get("max_compliant_seen", 0.0) > 0.01  # Even 1% at any point → not a clean attacker signature

            if neighbor_id not in self.weak_backdoor_nodes and state["negative_count"] >= MALICIOUS_ZERO_ROUNDS_REQUIRED:
                if has_ever_shown_bait:
                    # Has shown SOME bait historically → mark as CARRIER_SUSPECT, not MALICIOUS
                    state["carrier_suspect"] = True
                    state["status"] = "TESTING"
                    logging.warning(
                        f"[Manager] 🚚 {neighbor_id} reached negative threshold but has historical bait "
                        f"({state['max_compliant_seen']:.2%}). Marking CARRIER_SUSPECT — DFS will pivot through."
                    )
                    return "TESTING"
                else:
                    # Truly never showed bait + consistently suspicious → genuine attacker
                    state["status"] = "MALICIOUS"
                    state["verified_round"] = current_round
                    logging.critical(
                        f"[Manager] 🚨 Neighbor {neighbor_id} CONFIRMED as MALICIOUS "
                        f"(Never showed backdoor in {state['negative_count']} rounds, suspicious={state['suspicious_count']})"
                    )
                    return "MALICIOUS"

            return "TESTING"


    def _check_neighbor_has_backdoor(self, neighbor_model):
        """
        Check if neighbor model contains the honeypot backdoor.

        Returns:
            tuple: (has_backdoor: bool, compliant_rate: float, is_suspicious: bool, dominant_target: int)
                - has_backdoor: True if compliant rate >= 2%
                - compliant_rate: Measure of backdoor presence (0.0 to 1.0)
                - is_suspicious: True if detector flagged SUSPICIOUS patterns
                - dominant_target: The label most targeted by suspicious predictions
        """
        if not self.detector or not neighbor_model:
            return False, 0.0, False, None

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
                    # Save current model
                    current_params = {k: v.clone() for k, v in trainer.model.state_dict().items()}

                    # Load neighbor's model
                    trainer.set_model_parameters(neighbor_model)

                    # Run detector check (receiving dominant_target)
                    is_suspicious, severity, det_compliant_rate, dominant_target = self.detector.check(trainer.model, clean_batch, self.current_map)

                    # Restore original model
                    trainer.model.load_state_dict(current_params)

                    # Calculate compliant rate (DIRECTLY from detector)
                    compliant_rate = det_compliant_rate
                    has_backdoor = compliant_rate >= 0.02  # 2% threshold

                    return has_backdoor, compliant_rate, is_suspicious, dominant_target

        except Exception as e:
            logging.debug(f"[Manager] Error checking backdoor presence: {e}")

        return False, 0.0, False, None

    def should_send_backdoor(self, neighbor_id):
        """
        Decide if honeypot should send backdoored model to this neighbor.
        Only send to neighbors in TESTING status.
        """
        state = self.neighbor_tracking.get(neighbor_id, {"status": "TESTING"})
        return state["status"] == "TESTING"

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

    def is_visited(self, node_id: str) -> bool:
        return node_id in self.visited_history

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
            "reputation_history": self.reputation_history,
            "locked_target": self.locked_target,
            "transfer_source": getattr(self.engine, 'addr', None) if self.engine else None,
            "last_pivot_source": last_pivot_source,  # Prevent backtracking
            "rounds_at_current_node": self.rounds_at_current_node,  # Transfer grace counter
            "current_node_id": self.current_node_id,  # Transfer node position
            "suspect_confirmation": self.suspect_confirmation,  # Transfer suspect tracking
            "neighbor_tracking": self.neighbor_tracking  # CRITICAL: Preserve neighbor memory across pivots
        }
        if self.strategy:
            state["seed_state"] = self.strategy.get_state()
        return state

    def import_state(self, state):
        if not state:
            logging.warning("[Manager] Import State called with EMPTY state.")
            return None

        logging.info(f"[Manager] Importing State: keys={list(state.keys())}")

        if "history" in state: self.visited_history = state["history"]
        if "reputation_history" in state: self.reputation_history = state.get("reputation_history", {})

        if "locked_target" in state:
            self.locked_target = state.get("locked_target")
            logging.info(f"[Manager] 🔓=>🔒 LOCKED TARGET imported: {self.locked_target}")
        else:
            logging.info("[Manager] No 'locked_target' in state.")

        # CRITICAL: Restore _last_pivot_source to prevent ping-pong
        if "last_pivot_source" in state and state["last_pivot_source"]:
            last_pivot_source = state["last_pivot_source"]
            if self.role_behavior:
                self.role_behavior._last_pivot_source = last_pivot_source
                logging.info(f"[Manager] 🔙 PIVOT SOURCE restored: {last_pivot_source} (prevents backtrack)")

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
                "first_backdoor_round": None
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
            "carrier_suspect": False,  # True when suspicious but upstream attacker suspected
        }
        for key, val in defaults.items():
            if key not in state:
                state[key] = val

        return state

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

        if my_neighbors is None:
            my_neighbors = set(neighbors_models.keys())

        # NUEVO: Usar sistema de grace period POR NODO con EARLY EXIT optimization
        grace_period_active = self.is_grace_period_active(neighbors_models)

        if grace_period_active:
            logging.info(f"[DFS] ⏳ GRACE PERIOD at current node (round {self.rounds_at_current_node}/{self.grace_rounds_per_node})")
            logging.info(f"[DFS] 🎣 Injecting HoneyDoor backdoor - waiting for propagation before analysis")
            # Durante grace: NO analizar, solo inyectar backdoor
            # Retornar None para indicar que aún no hay decisión
            return (False, None)

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
                return (False, None)

        compliant_neighbors = []     # Neighbors verified BENIGN (bait + clean round)
        investigation_neighbors = [] # Neighbors suspicious BUT COMPLIANT (victims/carriers)
        suspicious_neighbors = []    # Neighbors suspicious AND NO BAIT (threats)
        current_round = getattr(self.engine, 'round', 0) if self.engine else 0

        for node_id, model_obj in neighbors_models.items():
            if node_id == came_from:
                logging.info(f"[DFS] Skipping {node_id} - came from there (no backtrack)")
                continue

            # ============================================================================
            # UNIFIED ANALYSIS: Use memory-based analyze_neighbor
            # ============================================================================
            status = self.analyze_neighbor(node_id, model_obj, current_round)

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
                    logging.warning(
                        f"[DFS] 🚚 {node_id} has MALICIOUS status but is a CARRIER_SUSPECT. "
                        f"Prioritizing as high-priority investigation target (pivot through)."
                    )
                    # High priority (score=2.0) — pivot through ASAP to find the real attacker
                    investigation_neighbors.append((node_id, 2.0))
                else:
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
                rounds_tested = node_state.get("rounds_tested", 0)
                if rounds_with_suspicion >= 1 and rounds_tested >= 1:
                    # Suspicious but unverified → treat as high-priority carrier candidate
                    investigation_neighbors.append((node_id, 1.5))  # Higher than normal carrier (1.0)
                    logging.warning(
                        f"[DFS] 🔎 {node_id} suspicious with 0% bait — potential carrier (upstream attacker). "
                        f"Prioritizing for investigation pivot (suspicious={rounds_with_suspicion}/{rounds_tested} rounds)."
                    )
                else:
                    # Not enough evidence yet → pure monitoring
                    suspicious_neighbors.append((node_id, 1.0))
                    logging.warning(f"[DFS] ⏳ {node_id} is under TESTING/MONITORING (Current Status: {status})")

        # ============================================================================
        # PIVOT OR HOLD DECISION
        # ============================================================================

        # 1. Priority: Explore verified BENIGN branches (Standard DFS)
        if compliant_neighbors:
            for neighbor_id, _ in compliant_neighbors:
                if not self.is_visited(neighbor_id):
                    logging.info(f"[DFS] Pivoting to BENIGN neighbor {neighbor_id} to explore branch.")
                    return (False, neighbor_id)

        # 2. Sequential Priority: Pivot to INVESTIGATION targets (Follow the poison trail)
        if investigation_neighbors:
            for neighbor_id, _ in investigation_neighbors:
                if not self.is_visited(neighbor_id):
                    state = self.neighbor_tracking.get(neighbor_id, {})
                    raw_suspicion = state.get("suspicious_count", 0) / max(1, state.get("rounds_tested", 1))

                    # BRAVE HONEYPOT: We pivot even if suspicion is high to follow the poison trail.
                    # High suspicion in a carrier is a "compass" toward the origin.
                    logging.info(f"[DFS] 🕵️ Pivoting to CARRIER neighbor {neighbor_id} to follow poison trail (Suspicion: {raw_suspicion:.1%}).")
                    return (False, neighbor_id)

        # 3. If everything unvisited is a pure THREAT (no bait), we HOLD to confirm
        if suspicious_neighbors:
            suspect_id = suspicious_neighbors[0][0]
            state = self.neighbor_tracking.get(suspect_id, {})
            rounds = state.get("suspicious_count", 0)
            logging.info(f"[DFS] ⏳ HOLDING POSITION: Confirming pure suspect {suspect_id} ({rounds}/{self.confirmation_rounds_required} rounds).")
            return (False, None)

        # 4. If everything visited/analyzed, we stay put (Active Monitoring)
        logging.info("[DFS] 🏁 End of trail reached. All branches from this node already visited. Holding position for local monitoring.")
        return (False, None)

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
                logging.info(f"[DFS] Next pivot direction: {neighbor} (unvisited)")
                return neighbor

        # Si todos los vecinos han sido visitados, elegir aleatorio (exploración ciclada)
        if neighbors:
            next_hop = random.choice(neighbors)
            logging.info(f"[DFS] All neighbors visited, cycling: {next_hop}")
            return next_hop

        logging.warning(f"[DFS] No available next hop from {my_id}")
        return None
