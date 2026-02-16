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

        # Stability mechanism: Keep same honey_map for multiple rounds
        self.map_stable_rounds = 5  # Number of rounds to keep same map
        self.map_round_counter = 0  # Counter for current map usage

        # PER-NODE Grace Period: Track rounds spent at current node
        self.rounds_at_current_node = 0  # Reset when pivoting
        self.grace_rounds_per_node = 2   # OPTIMIZED: 2 rounds enough for backdoor propagation
        self.current_node_id = None      # Track which node we're at

        # SUSPECT CONFIRMATION: Track suspects before declaring as attackers
        # Format: {node_id: confirmation_rounds}
        # If a suspect gets backdoor during confirmation → False positive, continue search
        # If after 3 rounds still no backdoor → Confirmed attacker
        self.suspect_confirmation = {}    # Track rounds monitoring each suspect
        self.confirmation_rounds_required = 2  # OPTIMIZED: Wait 2 rounds before confirming attacker

        # ============================================================================
        # OPTIMIZED NEIGHBOR TRACKING SYSTEM
        # Fast benign verification (1 round) + Conservative malicious confirmation (3 rounds)
        # ============================================================================
        self.neighbor_tracking = {}  # {node_id: {status, negative_count, start_round, verified_round}}
        self.NEGATIVE_THRESHOLD = 2  # OPTIMIZED: 2 consecutive negative rounds

        # ============================================================================
        # ADAPTIVE BACKDOOR STRENGTHENING SYSTEM
        # Progressively strengthen backdoor for nodes showing 0% to distinguish
        # weak backdoor (benign) from active filtering (malicious)
        # ============================================================================
        self.weak_backdoor_nodes = {}  # {node_id: {round_started, attempts, original_params}}
        self.strengthening_enabled = True
        self.strengthening_max_attempts = 5   # OPTIMIZED: 5 attempts (Extended verification)
        self.strengthening_injection_step = 0.15  # 15% step increase
        self.strengthening_weight_step = 0.5      # 0.5x weight increase
        self.base_injection_ratio = 0.2           # Start low (20%), ramp up to ~95%
        self.base_weight_boost = 1.5              # Base weight boost

        # Generate initial map
        if self.strategy:
            self.current_map = self.strategy.get_honey_map()
            logging.info(f"🔄 [HoneyManager] INITIAL Honey Map Generated: {self.current_map}")


    def new_round(self):
        # Honey_map PERMANENTEMENTE ESTABLE - nunca cambia
        # Esto permite convergencia total del backdoor
        if self.strategy:
            logging.info(f"🔒 [HoneyManager] STABLE Honey Map (permanent): {self.current_map}")
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

        Progressive strengthening over 3 attempts:
        - Attempt 1: injection * 1.15, weight * 1.5
        - Attempt 2: injection * 1.30, weight * 2.0
        - Attempt 3: injection * 1.45, weight * 2.5

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
        injection_multiplier = 1.0 + (self.strengthening_injection_step * (attempts + 1))  # 1.15, 1.30, 1.45
        weight_multiplier = 1.0 + (self.strengthening_weight_step * (attempts + 1))       # 1.5, 2.0, 2.5

        strengthened_injection = min(0.95, info["base_injection"] * injection_multiplier)
        strengthened_weight = info["base_weight"] * weight_multiplier

        return {
            "injection_ratio": strengthened_injection,
            "weight_boost": strengthened_weight,
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
        # Initialize if new neighbor
        if neighbor_id not in self.neighbor_tracking:
            self.neighbor_tracking[neighbor_id] = {
                "status": "TESTING",
                "negative_count": 0,
                "start_round": current_round,
                "verified_round": None,
                "max_compliant_seen": 0.0,          # Track highest compliant rate
                "compliant_history": [],            # List of (round, rate) tuples
                "first_backdoor_round": None        # When backdoor first detected
            }
            logging.info(f"[Manager] 🆕 New neighbor detected: {neighbor_id} - Status: TESTING")

        state = self.neighbor_tracking[neighbor_id]

        # Skip if already verified
        if state["status"] in ["BENIGN", "MALICIOUS"]:
            return state["status"]

        # Analyze model for backdoor presence (returns bool + compliant_rate + is_suspicious)
        has_backdoor, compliant_rate, is_suspicious = self._check_neighbor_has_backdoor(neighbor_model)

        # Update history
        state["compliant_history"].append((current_round, compliant_rate))
        state["max_compliant_seen"] = max(state["max_compliant_seen"], compliant_rate)

        # 🔑 PRIORITY CHECK: If detector flags SUSPICIOUS → MALICIOUS immediately
        # This catches Sample Poisoners whose models predict a THIRD target (neither real label nor honeypot bait)
        if is_suspicious:
            state["status"] = "MALICIOUS"
            state["verified_round"] = current_round
            logging.critical(
                f"[Manager] 🚨 Neighbor {neighbor_id} CONFIRMED as MALICIOUS in round {current_round} "
                f"(Sample Poisoning detected: model predicts suspicious third-target labels)"
            )
            # Remove from strengthening if it was being tracked
            if neighbor_id in self.weak_backdoor_nodes:
                del self.weak_backdoor_nodes[neighbor_id]
            return "MALICIOUS"

        # 🔑 KEY LOGIC: Check historical maximum (MEMORY-BASED)
        if state["max_compliant_seen"] >= 0.02:  # Lowered to 2% to catch weaker signals
            # Node showed backdoor at some point → BENIGN (even if diluted now)
            if state["first_backdoor_round"] is None:
                state["first_backdoor_round"] = current_round

            state["status"] = "BENIGN"
            state["verified_round"] = state["first_backdoor_round"]
            logging.info(
                f"[Manager] ✅ Neighbor {neighbor_id} VERIFIED as BENIGN in round {current_round} "
                f"(max_compliant={state['max_compliant_seen']:.2%}, current={compliant_rate:.2%})"
            )
            return "BENIGN"

        else:
            # No backdoor ever seen at >= 2% → check strengthening / negative counters
            state["negative_count"] += 1

            # ============================================================================
            # ADAPTIVE STRENGTHENING
            # ============================================================================
            if self.strengthening_enabled and neighbor_id in self.weak_backdoor_nodes:
                info = self.weak_backdoor_nodes[neighbor_id]

                if compliant_rate > 0.0:
                    # SUCCESS! Node showed SOME compliance → it's honest, not filtering
                    logging.info(
                        f"[Manager] ✅ Node {neighbor_id} shows {compliant_rate:.2%} after "
                        f"strengthening attempt {info['attempts']}. BENIGN confirmed (honest values detected)."
                    )
                    state["status"] = "BENIGN"
                    state["verified_round"] = current_round
                    state["max_compliant_seen"] = compliant_rate
                    state["first_backdoor_round"] = current_round

                    del self.weak_backdoor_nodes[neighbor_id]
                    return "BENIGN"

                elif info["attempts"] >= self.strengthening_max_attempts:
                    # EXHAUSTED - Still 0% after max strengthening attempts.
                    # VERDICT: BENIGN (Catastrophic Forgetting).
                    # Experiments prove benign nodes consistently erase the backdoor.
                    # We cannot distinguish this from active filtering by compliance alone.
                    # Marking BENIGN to prevent false positives that collapse the network.
                    logging.warning(
                        f"[Manager] ⚠️ Node {neighbor_id} still 0% after {info['attempts']} "
                        f"strengthening attempts. Assuming BENIGN (Catastrophic Forgetting). "
                        f"Cannot distinguish from filtering by compliance alone."
                    )
                    state["status"] = "BENIGN"
                    state["verified_round"] = current_round
                    state["max_compliant_seen"] = 0.0

                    del self.weak_backdoor_nodes[neighbor_id]
                    return "BENIGN"

                else:
                    # Continue strengthening
                    info["attempts"] += 1
                    logging.info(
                        f"[Manager] 🔬 Node {neighbor_id} still 0% - strengthening attempt "
                        f"{info['attempts']}/{self.strengthening_max_attempts}"
                    )

            elif self.strengthening_enabled and compliant_rate == 0.0 and neighbor_id not in self.weak_backdoor_nodes:
                # First time seeing 0% - mark for strengthening
                self.weak_backdoor_nodes[neighbor_id] = {
                    "round_started": current_round,
                    "attempts": 1,
                    "base_injection": self.base_injection_ratio,
                    "base_weight": self.base_weight_boost
                }
                logging.info(
                    f"[Manager] 🔬 Node {neighbor_id} shows 0% compliant (attempt 1/{self.strengthening_max_attempts}) - "
                    f"will strengthen backdoor progressively"
                )

            # Regular negative threshold check (only if not in strengthening mode)
            if neighbor_id not in self.weak_backdoor_nodes and state["negative_count"] >= self.NEGATIVE_THRESHOLD:
                state["status"] = "MALICIOUS"
                state["verified_round"] = current_round
                logging.critical(
                    f"[Manager] 🚨 Neighbor {neighbor_id} CONFIRMED as MALICIOUS in round {current_round} "
                    f"(never showed backdoor in {state['negative_count']} rounds, max_compliant={state['max_compliant_seen']:.2%})"
                )
                return "MALICIOUS"

            else:
                # Still in testing period
                logging.warning(
                    f"[Manager] ⏳ Neighbor {neighbor_id} negative {state['negative_count']}/{self.NEGATIVE_THRESHOLD} "
                    f"(current={compliant_rate:.2%}, max_seen={state['max_compliant_seen']:.2%})"
                )
                return "TESTING"


    def _check_neighbor_has_backdoor(self, neighbor_model):
        """
        Check if neighbor model contains the honeypot backdoor.

        Returns:
            tuple: (has_backdoor: bool, compliant_rate: float, is_suspicious: bool)
                - has_backdoor: True if compliant rate >= 2%
                - compliant_rate: Measure of backdoor presence (0.0 to 1.0)
                - is_suspicious: True if detector flagged SUSPICIOUS patterns
                  (e.g. Sample Poisoning: model predicts third-target labels)
        """
        if not self.detector or not neighbor_model:
            return False, 0.0, False

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

                    # Run detector check
                    is_suspicious, severity = self.detector.check(trainer.model, clean_batch, self.current_map)

                    # Restore original model
                    trainer.model.load_state_dict(current_params)

                    # Calculate compliant rate (inverse of severity if not suspicious)
                    # Higher compliant_rate = more backdoor presence
                    compliant_rate = (1.0 - severity) if not is_suspicious else 0.0
                    has_backdoor = compliant_rate >= 0.02  # 2% threshold

                    return has_backdoor, compliant_rate, is_suspicious

        except Exception as e:
            logging.debug(f"[Manager] Error checking backdoor presence: {e}")

        return False, 0.0, False

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

        # EARLY EXIT: If we have neighbor info, check if we can skip
        if neighbors_models:
            all_verified = True
            for node_id in neighbors_models:
                # Check tracking memory
                # If node has EVER shown backdoor (max_compliant_seen > 0), it's safe to proceed
                node_info = self.neighbor_tracking.get(node_id, {})
                max_seen = node_info.get("max_compliant_seen", 0.0)

                if max_seen < 0.01: # Less than 1% compliant seen
                    all_verified = False
                    break

            if all_verified and len(neighbors_models) > 0:
                logging.info(f"[Manager] 🚀 EARLY EXIT from Grace Period: All {len(neighbors_models)} neighbors already verified (max_compliant > 0)")
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
            "suspect_confirmation": self.suspect_confirmation  # Transfer suspect tracking
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

        compliant_neighbors = []  # Vecinos con backdoor (honestos)
        suspicious_neighbors = []  # Vecinos sin backdoor (sospechosos)

        for node_id, model_obj in neighbors_models.items():
            if node_id == came_from:
                logging.info(f"[DFS] Skipping {node_id} - came from there (no backtrack)")
                continue

            # VERIFICAR HONEYDOOR: ¿El vecino tiene el backdoor en su modelo?
            has_backdoor = False
            severity = 0.0

            if model_obj and self.detector and self.current_map:
                try:
                    # Obtener datos de validación y la instancia del modelo
                    clean_batch = None
                    model_instance = None

                    if self.role_behavior and hasattr(self.role_behavior, '_engine'):
                        engine = self.role_behavior._engine
                        trainer = engine.trainer

                        try:
                            # Obtener batch de validación
                            if hasattr(trainer, 'datamodule'):
                                trainer.datamodule.setup("fit")
                                val_loader = trainer.datamodule.val_dataloader()
                                clean_batch = next(iter(val_loader))

                            # Cargar el state_dict del vecino en el modelo temporal
                            if hasattr(trainer, 'model') and trainer.model:
                                # Guardar el modelo actual temporalmente
                                current_params = {k: v.clone() for k, v in trainer.model.state_dict().items()}

                                # Cargar parámetros del vecino
                                trainer.set_model_parameters(model_obj)
                                model_instance = trainer.model

                                # Ejecutar el HoneyDetector
                                if clean_batch and model_instance:
                                    is_suspicious, severity = self.detector.check(model_instance, clean_batch, self.current_map)
                                    # INVERTIR: detector retorna is_suspicious (True=atacante, False=honesto)
                                    # Nosotros necesitamos has_backdoor (True=honesto con backdoor, False=sospechoso)
                                    has_backdoor = not is_suspicious
                                    logging.info(f"[DFS] 🔍 {node_id}: HasBackdoor={has_backdoor}, Suspicious={is_suspicious}, Severity={severity:.2f}")

                                # Restaurar el modelo original
                                trainer.model.load_state_dict(current_params)

                        except Exception as e:
                            logging.debug(f"[DFS] Error during HoneyDoor check for {node_id}: {e}")
                            # Intentar restaurar el modelo en caso de error
                            if 'current_params' in locals() and hasattr(trainer, 'model'):
                                try:
                                    trainer.model.load_state_dict(current_params)
                                except:
                                    pass

                    if not clean_batch or not model_instance:
                        logging.debug(f"[DFS] Could not perform HoneyDoor check for {node_id} (missing data or model)")

                except Exception as e:
                    logging.warning(f"[DFS] HoneyDoor check failed for {node_id}: {e}")

            # CLASIFICAR VECINO:
            # - has_backdoor=True → COMPLIANT (honesto, agregó nuestro modelo)
            # - has_backdoor=False → SUSPICIOUS (posible atacante, pero necesita confirmación)

            # ============================================================================
            # FIX: Check neighbor_tracking first to prevent conflicts with memory-based system
            # If a node was already verified as BENIGN by analyze_neighbor(), respect that decision
            # ============================================================================
            if node_id in self.neighbor_tracking:
                tracked_status = self.neighbor_tracking[node_id].get("status", "TESTING")
                if tracked_status == "BENIGN":
                    # Node already verified by memory-based system → Skip DFS analysis
                    compliant_neighbors.append((node_id, severity))
                    logging.info(f"[DFS] ✅ {node_id} is BENIGN (verified by memory-based tracking, max_compliant={self.neighbor_tracking[node_id].get('max_compliant_seen', 0):.2%})")

                    # Clear from suspect confirmation if it was there
                    if node_id in self.suspect_confirmation:
                        del self.suspect_confirmation[node_id]
                    continue

            if has_backdoor:
                compliant_neighbors.append((node_id, severity))
                logging.info(f"[DFS] ✅ {node_id} is COMPLIANT (has honeypot backdoor)")

                # Si este nodo estaba en confirmación, fue un FALSO POSITIVO (ya recibió el backdoor)
                if node_id in self.suspect_confirmation:
                    logging.info(f"[DFS] ✅ FALSE POSITIVE: {node_id} now has backdoor (was suspect for {self.suspect_confirmation[node_id]} rounds). Cleared.")
                    del self.suspect_confirmation[node_id]
            else:
                suspicious_neighbors.append((node_id, severity))
                logging.warning(f"[DFS] 🚨 {node_id} is SUSPICIOUS (NO honeypot backdoor)")

        # ============================================================================
        # CONFIRMACIÓN DE SOSPECHOSOS: Esperar 3 rondas antes de declarar atacante
        # ============================================================================

        # Actualizar contadores de confirmación para sospechosos actuales
        for node_id, severity in suspicious_neighbors:
            if node_id not in self.suspect_confirmation:
                # Primer detección como sospechoso
                self.suspect_confirmation[node_id] = 1
                logging.warning(f"[DFS] ⚠️ NEW SUSPECT: {node_id} (round 1/{self.confirmation_rounds_required}). Monitoring for backdoor propagation...")
            else:
                # Incrementar contador de confirmación
                self.suspect_confirmation[node_id] += 1
                rounds = self.suspect_confirmation[node_id]
                logging.warning(f"[DFS] ⚠️ SUSPECT MONITORING: {node_id} still without backdoor (round {rounds}/{self.confirmation_rounds_required})")

        # Verificar si algún sospechoso alcanzó el umbral de confirmación
        confirmed_attacker = None
        for node_id, rounds in self.suspect_confirmation.items():
            if rounds >= self.confirmation_rounds_required:
                confirmed_attacker = node_id
                break

        if confirmed_attacker:
            # ATACANTE CONFIRMADO después de 3 rondas sin backdoor
            logging.critical(f"[DFS] 🎯 ATTACKER CONFIRMED: {confirmed_attacker} (NO backdoor after {self.suspect_confirmation[confirmed_attacker]} rounds)")
            logging.info(f"[DFS] 🛑 STOPPING - Staying at current node to execute containment")

            # Limpiar el tracking de este sospechoso (ya fue confirmado)
            del self.suspect_confirmation[confirmed_attacker]

            return (True, confirmed_attacker)

        # Si hay sospechosos pero ninguno confirmado aún, QUEDARSE para seguir monitoreando
        if suspicious_neighbors:
            suspect_id = suspicious_neighbors[0][0]
            rounds = self.suspect_confirmation.get(suspect_id, 0)
            logging.info(f"[DFS] ⏳ HOLDING POSITION: Monitoring suspect {suspect_id} ({rounds}/{self.confirmation_rounds_required} rounds). Waiting for backdoor propagation...")
            return (False, None)  # No pivotar, quedarse monitoreando

        # ============================================================================
        # FINAL SAFEGUARD: Check memory-based suspects before pivoting
        # If we have tracked suspects (even if silent this round), we HOLD.
        # ============================================================================
        if self.suspect_confirmation:
            logging.info(f"[DFS] ⏳ HOLDING POSITION: Suspects in memory {list(self.suspect_confirmation.keys())} (silent this round). Waiting...")
            return (False, None)

        # ============================================================================
        # FINAL SAFEGUARD: Check memory-based suspects before pivoting
        # If we have tracked suspects (even if silent this round), we HOLD.
        # ============================================================================
        if self.suspect_confirmation:
            logging.info(f"[DFS] ⏳ HOLDING POSITION: Suspects in memory {list(self.suspect_confirmation.keys())} (silent this round). Waiting...")
            return (False, None)

        # TOTALMENTE SEGUROS: No hay sospechosos activos ni en memoria.
        # Si hay vecinos compliant, pivotamos.
        if compliant_neighbors:
            # CRITICAL FIX: Elegir vecino que NO haya sido visitado (DFS correcto)
            next_pivot = None
            for neighbor_id, severity in compliant_neighbors:
                if not self.is_visited(neighbor_id):
                    next_pivot = neighbor_id
                    logging.info(f"[DFS] All neighbors are compliant. Pivoting to {next_pivot} to continue search (unvisited node)")
                    return (False, next_pivot)

            # Si TODOS los vecinos ya fueron visitados, elegir el de menor severidad (exploración completa)
            if next_pivot is None:
                next_pivot = min(compliant_neighbors, key=lambda x: x[1])[0]
                logging.warning(f"[DFS] All compliant neighbors were visited. Pivoting to {next_pivot} (lowest severity, re-exploration)")
                return (False, next_pivot)

        # No hay vecinos disponibles para analizar
        logging.warning("[DFS] No neighbors available for analysis.")
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
