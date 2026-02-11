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

    def new_round(self):
        if self.strategy:
            self.strategy.next_epoch()
            # Rotate maps
            self.previous_map = self.current_map
            self.current_map = self.strategy.get_honey_map()
        return self.current_map

    def get_dataset(self, original_dataset):
        if self.strategy:
            return HoneyDataset(original_dataset, self.current_map)
        return original_dataset

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

    def register_visit(self, node_id: str):
        if node_id not in self.visited_history:
            self.visited_history.append(node_id)

    def is_visited(self, node_id: str) -> bool:
        return node_id in self.visited_history

    def export_state(self):
        state = {
            "history": self.visited_history,
            "reputation_history": self.reputation_history,
            "locked_target": self.locked_target,
            "transfer_source": getattr(self.engine, 'addr', None) if self.engine else None
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

        En cada nodo en el que pivota el honeypot:
        1. Verifica si algún vecino tiene el BACKDOOR (honeydoor) en su modelo
        2. Si el backdoor está presente (Model Replacement o Direct Backdoor) → ES EL ATACANTE
        3. Si el modelo es limpio (sin backdoor) → ES SEGURO PIVOTAR hacia él
        4. Combina con análisis de Silent como criterio secundario

        PERIODO DE GRACIA: Solo ejecuta el HoneyDoor check después de Ronda 5 para permitir
        que el backdoor se propague por la red y evitar falsos positivos prematuros.

        Args:
            neighbors_models: dict con {node_id: model_obj} para cada vecino
            reputation_module: módulo de reputación para checar silencio
            came_from: de dónde vinimos (para no retroceder)
            my_neighbors: set con los vecinos directos del nodo actual

        Returns:
            (found_attacker, attacker_id) o (False, None)
        """
        logging.info(f"[DFS] Analyzing {len(neighbors_models)} neighbors at current node...")

        if my_neighbors is None:
            my_neighbors = set(neighbors_models.keys())

        # Obtener ronda actual para el periodo de gracia
        current_round = 0
        if self.role_behavior and hasattr(self.role_behavior, '_engine'):
            current_round = getattr(self.role_behavior._engine, 'round', 0)

        # PERIODO DE GRACIA: No ejecutar HoneyDoor check en rondas tempranas (<= 10)
        # Esto permite que el backdoor se propague por la red antes de detectar
        # Aumentado a 10 porque el backdoor puede tardar ~7-8 rondas en propagarse
        grace_period_active = current_round <= 10
        if grace_period_active:
            logging.info(f"[DFS] ⏳ GRACE PERIOD active (Round {current_round} <= 10). Skipping HoneyDoor checks to allow backdoor propagation.")

        safe_pivot_candidates = []  # Nodos sin backdoor, seguros para pivotar
        attacker_candidates = []    # Nodos con backdoor o comportamiento sospechoso

        for node_id, model_obj in neighbors_models.items():
            if node_id == came_from:
                logging.info(f"[DFS] Skipping {node_id} - came from there (no backtrack)")
                continue

            # 1. VERIFICAR HONEYDOOR: ¿El vecino tiene el backdoor en su modelo?
            is_malicious = False
            severity = 0.0

            # Solo ejecutar HoneyDoor check después del periodo de gracia
            if not grace_period_active and model_obj and self.detector and self.current_map:
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
                                    is_malicious, severity = self.detector.check(model_instance, clean_batch, self.current_map)
                                    logging.info(f"[DFS] HoneyDoor check on {node_id}: Malicious={is_malicious}, Severity={severity:.2f}")

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

            # 2. Check if silent (criterio secundario)
            is_silent = self._is_node_silent_to_neighbors(node_id, my_neighbors, reputation_module)

            logging.info(f"[DFS]   {node_id}: Malicious={is_malicious}, Severity={severity:.2f}, Silent={is_silent}")

            # DECISIÓN: ¿Es atacante confirmado?
            if is_malicious:
                # ENCONTRADO: El vecino tiene el backdoor → ES EL ATACANTE
                logging.critical(f"[DFS] 🎯 FOUND ATTACKER at neighbor: {node_id} (HoneyDoor detected backdoor, Severity={severity:.2f})")
                return (True, node_id)

            # Si no es malicioso pero es silent, marcarlo como sospechoso
            if is_silent:
                attacker_candidates.append((node_id, is_silent, severity))
            else:
                # Nodo seguro para pivotar (sin backdoor, no silent)
                safe_pivot_candidates.append(node_id)

        # Si llegamos aquí, no encontramos atacante confirmado
        # Priorizar pivotaje hacia nodos SEGUROS (sin backdoor, no silent)
        if safe_pivot_candidates:
            next_pivot = safe_pivot_candidates[0]
            logging.info(f"[DFS] No attacker found. Pivoting to safe node {next_pivot} (no backdoor detected)")
            return (False, next_pivot)

        # Si NO hay nodos seguros pero hay sospechosos, quedarse monitoreando
        if attacker_candidates:
            logging.warning(f"[DFS] Found {len(attacker_candidates)} suspicious neighbors (Silent) but no confirmed attacker via HoneyDoor.")
            logging.info(f"[DFS] Staying in current position to continue monitoring and accumulate detections.")
            return (False, None)

        # No hay vecinos disponibles para analizar
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
