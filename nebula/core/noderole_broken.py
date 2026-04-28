from __future__ import annotations
import logging
import asyncio
from nebula.addons.attacks.attacks import create_attack
from nebula.addons.functions import print_msg_box
from nebula.config.config import Config
from nebula.core.utils.locker import Locker
from nebula.core.eventmanager import EventManager
from nebula.core.nebulaevents import UpdateReceivedEvent, ModelPropagationEvent, HoneypotDetectionEvent, RoleTransferEvent
from nebula.addons.honeypot.manager import HoneyPotManager
import random
import copy
from enum import Enum
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from nebula.core.network.actions import ControlAction
import json
if TYPE_CHECKING:
    from nebula.core.engine import Engine

#TODO ensure attacks works properly

"""                                                         ##############################
                                                            #        ROLE BEHAVIORS      #
                                                            ##############################
"""

class Role(Enum):
    """
    This class defines the participant roles of the platform.
    """
    TRAINER = "trainer"
    AGGREGATOR = "aggregator"
    PROXY = "proxy"
    IDLE = "idle"
    SERVER = "server"
    MALICIOUS = "malicious"
    HONEYPOT = "honeypot"

def factory_node_role(role: str) -> Role:
    if role == "trainer":
        return Role.TRAINER
    elif role == "aggregator":
        return Role.AGGREGATOR
    elif role == "proxy":
        return Role.PROXY
    elif role == "idle":
        return Role.IDLE
    elif role == "server":
        return Role.SERVER
    elif role == "malicious":
        return Role.MALICIOUS
    elif role == "honeypot":
        return Role.HONEYPOT
    else:
        return ""

class RoleBehavior(ABC):
    """
    Abstract base class for defining the role-specific behavior of a node in CFL, DFL, or SDFL systems.

    Each subclass encapsulates the logic needed for a particular node role (e.g., trainer, aggregator),
    providing custom implementations for role-related operations such as training cycles,
    update aggregation, and recovery strategies.

    Attributes:
        _next_role (Role): The role to which the node is expected to transition.
        _next_role_locker (Locker): An asynchronous lock to protect access to _next_role.
        _source_to_notificate (Optional[Any]): The source node to notify once a role change is applied.
    """
    def __init__(self):
        self._next_role: Role = None
        self._next_role_locker = Locker("next_role_locker", async_lock=True)
        self._source_to_notificate = None

    @abstractmethod
    def get_role(self):
        """
        Returns the Role enum value representing the current role of the node.
        """
        raise NotImplementedError

    @abstractmethod
    def get_role_name(self, effective=False):
        """
        Returns a string representation of the current role.

        Args:
            effective (bool): Whether to return the name of the current effective role when going as malicious.

        Returns:
            str: Name of the role.
        """
        raise NotImplementedError

    @abstractmethod
    async def extended_learning_cycle(self):
        """
        Performs the main learning or aggregation cycle associated with the current role.

        This method encapsulates all the logic tied to the behavior of the node in its current role,
        including training, aggregating updates, and coordinating with neighbors.
        """
        raise NotImplementedError

    @abstractmethod
    async def select_nodes_to_wait(self):
        """
        Determines which neighbors the node should wait for during the current cycle.

        This logic varies depending on whether the node is an aggregator, trainer, or other role.

        Returns:
            Set[Any]: A set of neighbor node identifiers to wait for.
        """
        raise NotImplementedError

    @abstractmethod
    async def resolve_missing_updates(self):
        """
        Defines the fallback strategy when expected model updates are not received.

        For example, an aggregator might default to a fresh model, while a trainer might proceed
        with its own local model.

        Returns:
            Any: The resolution outcome depending on the role's specific logic.
        """
        raise NotImplementedError

    async def set_next_role(self, role: Role, source_to_notificate = None):
        """
        Schedules a role change and optionally stores the source to notify upon completion.

        Args:
            role (Role): The new role to transition to.
            source_to_notificate (Optional[Any]): Identifier of the node that triggered the change.
        """
        async with self._next_role_locker:
            self._next_role = role
            self._source_to_notificate = source_to_notificate

    async def get_next_role(self) -> Role:
        """
        Retrieves and clears the next role value.

        Returns:
            Role: The next role to transition into.
        """
        async with self._next_role_locker:
            next_role = self._next_role
            self._next_role = None
        return next_role

    async def get_source_to_notificate(self):
        """
        Retrieves and clears the stored source to notify after a role change.

        Returns:
            Any: The source node identifier, or None if not set.
        """
        async with self._next_role_locker:
            source_to_notificate = self._source_to_notificate
            self._source_to_notificate = None
        return source_to_notificate

    async def update_role_needed(self):
        """
        Checks whether a role update is scheduled.

        Returns:
            bool: True if a role update is pending, False otherwise.
        """
        async with self._next_role_locker:
            updt_needed = self._next_role != None
        return updt_needed

"""                                                         ##############################
                                                            #     MALICIOUS BEHAVIOR     #
                                                            ##############################
"""

class MaliciousRoleBehavior(RoleBehavior):
    def __init__(self, engine: Engine, config: Config):
        super().__init__()
        self._engine = engine
        self._config = config
        self.attack = create_attack(self._engine)
        self.aggregator_bening = self._engine._aggregator

        # Fallback: If fake_behavior not in config, use "aggregator" as default
        benign_role = "aggregator"
        if "adversarial_args" in self._config.participant and "fake_behavior" in self._config.participant["adversarial_args"]:
             benign_role = self._config.participant["adversarial_args"]["fake_behavior"]

        self._fake_role_behavior = factory_role_behavior(benign_role, self._engine, self._config)
        self._role = factory_node_role("malicious")

        # Configuración de Pivote - Leer de adversarial_args del nodo malicioso, no del honeypot
        self._attacker_pivoting = False
        self._pivot_round = 10
        if "adversarial_args" in self._config.participant:
             adv_config = self._config.participant["adversarial_args"]
             self._attacker_pivoting = adv_config.get("attacker_pivoting", False)
             self._pivot_round = adv_config.get("pivot_round", 10)
        self._pivoted = False

        # Atributos dummy para evitar errores si algo intenta leerlos
        self.threat_confirmed_locally = False
        self.blacklist = set()

    def get_role(self): return self._role
    def get_role_name(self, effective=False):
        return self._fake_role_behavior.get_role_name() if effective else self._role.value

    async def extended_learning_cycle(self):
        # 1. Ataque
        if self.attack:
            try: await self.attack.attack()
            except Exception: logging.exception("Attack failed")

        # 2. Comportamiento base (Sincronización)
        await self._fake_role_behavior.extended_learning_cycle()

        # 3. Pivote (Infección)
        if self._attacker_pivoting and not self._pivoted:
            if self._engine.round == self._pivot_round:
                 nodes = await self._engine.cm.get_addrs_current_connections(only_direct=False, myself=False)
                 if nodes:
                     target = random.choice(list(nodes))
                     logging.info(f"[Malicious] 🏃 Pivoting to {target} at Round {self._engine.round}")
                     if target not in self._engine.cm.connections:
                          await self._engine.cm.establish_connection(target)

                     # ENVÍO CORRECTO (log como argumento)
                     msg = self._engine.cm.create_message("control", "leadership_transfer", log="MALICIOUS_PIVOT_TRANSFER")
                     asyncio.create_task(self._engine.cm.send_message(target, msg))

                     self._pivoted = True
                     if hasattr(self._engine, "rb"):
                         await self._engine.rb.set_next_role(Role.AGGREGATOR)

    async def select_nodes_to_wait(self): return await self._fake_role_behavior.select_nodes_to_wait()
    async def resolve_missing_updates(self): return await self._fake_role_behavior.resolve_missing_updates()

class TrainerAggregatorRoleBehavior(RoleBehavior):
    def __init__(self, engine: Engine, config: Config):
        super().__init__()
        self._engine = engine
        self._config = config
        self._role = factory_node_role("aggregator")

    def get_role(self):
        return self._role

    def get_role_name(self, effective=False):
        return self._role.value

    async def extended_learning_cycle(self):
        """
        Ciclo completo: Test -> Train -> Auto-Reporte -> Propagación -> Agregación.
        """

        # 1. Testear modelo actual (opcional, pero recomendado)
        await self._engine.trainer.test()

        # 2. Entrenar (Protegido por Lock para evitar condiciones de carrera)
        await self._engine.trainning_in_progress_lock.acquire_async()
        try:
            await self._engine.trainer.train()
        except Exception as e:
            logging.error(f"[TrainerAggregator] Error during training: {e}")
        finally:
            try:
                await self._engine.trainning_in_progress_lock.release_async()
            except RuntimeError:
                # Lock was not acquired, that's fine
                pass

        # 3. AUTO-REPORTE: Inyectar nuestro propio modelo entrenado en el Agregador local
        # Esto es vital: si no hacemos esto, el agregador esperará eternamente nuestra propia parte.
        try:
            self_update_event = UpdateReceivedEvent(
                self._engine.trainer.get_model_parameters(),
                self._engine.trainer.get_model_weight(),
                self._engine.addr,
                self._engine.round
            )
            await EventManager.get_instance().publish_node_event(self_update_event)
        except Exception as e:
            logging.error(f"[TrainerAggregator] Error publishing self-update: {e}")

        # 4. Propagar a vecinos (Enviar lo que hemos entrenado)
        try:
            neighbors = await self._engine.cm.get_addrs_current_connections(only_direct=True, myself=False)
            if neighbors:
                mpe = ModelPropagationEvent(neighbors, "stable")
                await EventManager.get_instance().publish_node_event(mpe)
        except Exception as e:
            logging.error(f"[TrainerAggregator] Error propagating model: {e}")

        # 5. Esperar actualizaciones de vecinos y Agregar (Sync)
        # La función _waiting_model_updates del engine se encarga de esperar
        # y llamar al agregador.
        await self._engine._waiting_model_updates()

class AggregatorRoleBehavior(RoleBehavior):
    def __init__(self, engine: Engine, config: Config):
        super().__init__()
        self._engine = engine
        self._config = config
        self._role = factory_node_role("aggregator")
        self._transfer_send = False

    def get_role(self):
        return self._role

    def get_role_name(self, effective=False):
        return self._role.value

    async def extended_learning_cycle(self):
        await self._engine.trainer.test()

        # 2. Train (Protegido por Lock, copiado de TrainerAggregator)
        await self._engine.trainning_in_progress_lock.acquire_async()
        try:
            await self._engine.trainer.train()
        except Exception as e:
            logging.error(f"[Aggregator] Error during training: {e}")
        finally:
            try:
                await self._engine.trainning_in_progress_lock.release_async()
            except RuntimeError:
                # Lock was not acquired, that's fine
                pass

        # 3. AUTO-REPORTE: Inyectar nuestro propio modelo entrenado en el Agregador local
        try:
            self_update_event = UpdateReceivedEvent(
                self._engine.trainer.get_model_parameters(),
                self._engine.trainer.get_model_weight(),
                self._engine.addr,
                self._engine.round
            )
            await EventManager.get_instance().publish_node_event(self_update_event)
        except Exception as e:
            logging.error(f"[Aggregator] Error publishing self-update: {e}")

        # 4. Propagar a vecinos (Enviar lo que hemos entrenado)
        mpe = ModelPropagationEvent(await self._engine.cm.get_addrs_current_connections(only_direct=True, myself=False), "stable")
        await EventManager.get_instance().publish_node_event(mpe)

        # 5. Esperar actualizaciones de vecinos y Agregar (Sync)
        await self._engine._waiting_model_updates()

        # Transfer leadership
        neighbors = await self._engine.cm.get_addrs_current_connections(myself=False)
        if len(neighbors) and not self._transfer_send:
            random_neighbor = random.choice(list(neighbors))
            lt_message = self._engine.cm.create_message("control", "leadership_transfer")
            logging.info(f"Sending transfer leadership to: {random_neighbor}")
            asyncio.create_task(self._engine.cm.send_message(random_neighbor, lt_message))
            self._transfer_send = True

    async def select_nodes_to_wait(self):
        nodes = await self._engine.cm.get_addrs_current_connections(only_direct=True, myself=False)
        return nodes

    async def resolve_missing_updates(self):
        return (self._engine.trainer.get_model_parameters(), self._engine.trainer.BYPASS_MODEL_WEIGHT)

"""                                                         ##############################
                                                            #       SERVER BEHAVIOR      #
                                                            ##############################
"""

class ServerRoleBehavior(RoleBehavior):
    from datetime import datetime

    def __init__(self, engine: Engine, config: Config):
        super().__init__()
        self._engine = engine
        self._config = config
        self._start_time = ServerRoleBehavior.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        self._role = factory_node_role("server")

    def get_role(self):
        return self._role

    def get_role_name(self, effective=False):
        return self._role.value

    async def extended_learning_cycle(self):
        await self._engine.trainer.test()

        await self._engine._waiting_model_updates()

        mpe = ModelPropagationEvent(await self._engine.cm.get_addrs_current_connections(only_direct=True, myself=False), "stable")
        await EventManager.get_instance().publish_node_event(mpe)

    async def select_nodes_to_wait(self):
        nodes = await self._engine.cm.get_addrs_current_connections(only_direct=True, myself=False)
        return nodes

    async def resolve_missing_updates(self):
        return (self._engine.trainer.get_model_parameters(), self._engine.trainer.BYPASS_MODEL_WEIGHT)

"""                                                         ##############################
                                                            #      TRAINER BEHAVIOR      #
                                                            ##############################
"""

class TrainerRoleBehavior(RoleBehavior):
    def __init__(self, engine: Engine, config: Config):
        super().__init__()
        self._engine = engine
        self._config = config
        self._role = factory_node_role("trainer")

    def get_role(self):
        return self._role

    def get_role_name(self, effective=False):
        return self._role.value

    async def extended_learning_cycle(self):
        logging.info("Waiting global update | Assign _waiting_global_update = True")

        await self._engine.trainer.test()
        await self._engine.trainer.train()

        mpe = ModelPropagationEvent(await self._engine.cm.get_addrs_current_connections(only_direct=True, myself=False), "stable")
        await EventManager.get_instance().publish_node_event(mpe)

        await self._engine._waiting_model_updates()

    async def select_nodes_to_wait(self):
        nodes = await self._engine.cm.get_addrs_current_connections(only_direct=True, myself=False)
        return nodes

    async def resolve_missing_updates(self):
        return (self._engine.trainer.get_model_parameters(), self._engine.trainer.get_model_weight())

"""                                                         ##############################
                                                            #       IDLE BEHAVIOR        #
                                                            ##############################
"""

class IdleRoleBehavior(RoleBehavior):
    def __init__(self, engine: Engine, config: Config):
        super().__init__()
        self._engine = engine
        self._config = config
        self._role = factory_node_role("idle")

    def get_role(self):
        return self._role

    def get_role_name(self, effective=False):
        return self._role.value

    async def extended_learning_cycle(self):
        logging.info("Waiting global update | Assign _waiting_global_update = True")
        await self._engine._waiting_model_updates()

    async def select_nodes_to_wait(self):
        nodes = await self._engine.cm.get_addrs_current_connections(only_direct=True, myself=False)
        return nodes

    async def resolve_missing_updates(self):
        raise NotImplementedError

"""                                                         ##############################
                                                            #       PROXY BEHAVIOR       #
                                                            ##############################
"""

class ProxyRoleBehavior(RoleBehavior):
    def __init__(self, engine: Engine, config: Config):
        super().__init__()
        self._engine = engine
        self._config = config
        self._role = factory_node_role("proxy")

    def get_role(self):
        return self._role

    def get_role_name(self, effective=False):
        return self._role.value

    async def extended_learning_cycle(self):
        logging.info("Waiting global update | Assign _waiting_global_update = True")
        await self._engine._waiting_model_updates()

    async def select_nodes_to_wait(self):
        nodes = await self._engine.cm.get_addrs_current_connections(only_direct=True, myself=False)
        return nodes

    async def resolve_missing_updates(self):
        raise NotImplementedError

"""                                                         ##############################
                                                            #    UTILS ROLE BEHAVIORS    #
                                                            ##############################
"""

class roleBehaviorException(Exception):
    pass

def factory_role_behavior(role: str, engine: Engine, config: Config) -> RoleBehavior | None:

    # --- AUTO-DETECT HONEYPOT CONFIGURATION ---
    # Logic: Deterministic Random Selection based on Scenario Seed.
    # This ensures consistent selection across distributed nodes without communication.

    actual_role = config.participant.get("device_args", {}).get("role", "")

    if role == "aggregator" and actual_role != "malicious":
        try:
             defense_args = config.participant.get("defense_args", {})
             honeypot_args = defense_args.get("honeypot", {})

             # Check if we have already served/retired to prevent infinite loops
             is_already_retired = getattr(engine, "has_served_as_honeypot", False)

             if honeypot_args.get("enabled", False) and not is_already_retired:
                 scenario_args = config.participant.get("scenario_args", {})
                 device_args = config.participant.get("device_args", {})

                 seed_val = scenario_args.get("random_seed", 42)
                 n_nodes = scenario_args.get("n_nodes", 10)
                 target_count = honeypot_args.get("count", 1)
                 my_idx = device_args.get("idx") # Integer assumption

                 # Deterministic Selection
                 rng = random.Random(seed_val)
                 all_indices = list(range(n_nodes))

                 # IMPORTANT: Ensure stability. Sort before sample? range is sorted.
                 # Sample behavior is standard.
                 selected_indices = set(rng.sample(all_indices, target_count))

                 if my_idx in selected_indices:
                     logging.info(f"🕵️ [Honeypot Selection] I am the CHOSEN ONE (ID: {my_idx}). Seed: {seed_val}. Upgrading role.")
                     role = "honeypot"
                 else:
                     # logging.info(f"🛡️ [Honeypot Selection] I am NOT selected (ID: {my_idx}). Selected: {selected_indices}. Staying Aggregator.")
                     pass

        except Exception as e:
            logging.warning(f"Error checking honeypot config: {e}")
    # ------------------------------------------

    role_behaviors = {
        "malicious": MaliciousRoleBehavior,
        "trainer": TrainerRoleBehavior,
        "aggregator": AggregatorRoleBehavior,
        "server": ServerRoleBehavior,
        "proxy": ProxyRoleBehavior,
        "idle": IdleRoleBehavior,
        "honeypot": HoneypotRoleBehavior,
    }

    node_role = role_behaviors.get(role, None)

    if node_role:
        return node_role(engine, config)
    else:
        raise roleBehaviorException(f"Node Role Behavior {role} not found")

def change_role_behavior(old_role: RoleBehavior, new_role: Role, *parameters) -> RoleBehavior:
    engine, config = parameters
    if not isinstance(old_role, MaliciousRoleBehavior):
        new_behavior = factory_role_behavior(new_role.value, engine, config)

        # If switching TO Honeypot, save the previous role
        if new_role == Role.HONEYPOT:
             # Assuming old_role is the one we are leaving
             prev_role_name = old_role.get_role_name()
             if hasattr(new_behavior, "set_previous_role"):
                 new_behavior.set_previous_role(prev_role_name)
                 logging.info(f"[Honeypot] Previous role saved: {prev_role_name}")

        return new_behavior
    else:
        fake_behavior = factory_role_behavior(new_role.value, engine, config)
        old_role._fake_role_behavior = fake_behavior
        return old_role

"""                                                         ##############################
                                                            #       HONEYPOT BEHAVIOR      #
                                                            ##############################
"""

class HoneypotRoleBehavior(AggregatorRoleBehavior):
    def __init__(self, engine: Engine, config: Config):
        super().__init__(engine, config)
        self._role = factory_node_role("honeypot")

        seed = 0.5
        if hasattr(config, "participant") and "device_args" in config.participant:
             seed = config.participant["device_args"].get("honeypot_seed", 0.5)

        self.manager = HoneyPotManager(engine=engine, seed=seed, role_behavior=self)
        self._defense_active = True

        # Bandera clave: Si es True, hemos encontrado al malo y no nos movemos.
        self.threat_confirmed_locally = False

        # Contador de rondas limpias para desactivar el Honeypot
        self.consecutive_clean_rounds = 0

        # Track previous pivots to avoid cycling
        self._last_pivot_target = None
        self._pivot_history = set()
        self._last_pivot_round = -1

        # NEW DFS: Track pivot path to avoid backtracking
        self._last_pivot_source = None  # De dónde venimos (para no retroceder)

        # NEW: Control independent pivot allowance for indirect threats
        # When True, honeypot can pivot to find threat source even if threat_confirmed
        # This allows searching for root cause while maintaining threat awareness
        self._allow_pivot_for_indirect_threats = False

        # Counter for rounds spent searching for indirect threat source
        # Prevents infinite pivoting if source can't be found
        self._indirect_threat_pivot_rounds = 0
        self._max_indirect_threat_pivot_rounds = 3  # Max 3 rounds of indirect threat pivoting

        # --- ATTACKER PIVOTING DETECTION ---
        # Detecta cuando el atacante pivota a otro nodo
        self._attacker_pivoting_enabled = False
        self._pivot_round = 10
        if hasattr(config, "participant") and "device_args" in config.participant:
             hp_config = config.participant["device_args"].get("honeypot", {})
             if isinstance(hp_config, dict):
                 self._attacker_pivoting_enabled = hp_config.get("attacker_pivoting", False)
                 self._pivot_round = hp_config.get("pivot_round", 10)

        # Track the reputation of the known threat before and after to detect pivoting
        self._known_threat_node = None
        self._known_threat_reputation_history = {}  # {round: reputation_score}
        self._reputation_history = {}  # {round: {node_id: rep}} - Historial completo para detectar caídas
        self._pivot_detection_threshold = 0.3  # Si la reputación sube más de esto en una ronda, pivotó

        # Para un honeypot secundario que busca el nuevo atacante
        self._secondary_honeypot_candidates = set()  # Nodos vecinos que podrían ser convertidos a honeypot
        self._recovery_rounds_counter = 0  # Contador de rondas donde la reputación se recupera
        self._max_recovery_rounds = 3  # Si se recupera 3 rondas seguidas, desaparecer

    async def extended_learning_cycle(self):
        # --- FIX: Mimic Benign Aggregator Behavior FIRST ---
        # The Honeypot must train (or fake it) and PROPAGATE its model so neighbors don't deadlock.

        # 0. Update Map / Verify Threat State
        if not self.threat_confirmed_locally:
             self.manager.new_round()
        else:
             logging.info("[Honeypot] ⚠️ Threat confirmed. Continuing BAIT injection.")

        # 1. Train with BAIT (Dataset Injection)
        await self._engine.trainning_in_progress_lock.acquire_async()
        _original_loader_method = None
        trainer_wrapper = self._engine.trainer

        try:
            # --- INJECTION ---
            from nebula.addons.honeypot.dataset import HoneyDataset
            from torch.utils.data import DataLoader

            if trainer_wrapper and trainer_wrapper.datamodule:
                original_dm = trainer_wrapper.datamodule

                # Check for existing loader method
                if hasattr(original_dm, 'train_dataloader'):
                    _original_loader_method = original_dm.train_dataloader

                    # Create the hook
                    def baited_loader_factory():
                        base_loader = _original_loader_method()
                        honey_ds = HoneyDataset(base_loader.dataset, self.manager.current_map)
                        return DataLoader(
                            honey_ds,
                            batch_size=base_loader.batch_size,
                            shuffle=True,
                            num_workers=getattr(base_loader, 'num_workers', 0)
                        )

                    # Apply hook
                    original_dm.train_dataloader = baited_loader_factory
                    logging.info("[Honeypot] 🎣 BAIT INJECTED into training data.")

            await self._engine.trainer.train()

        except Exception as e:
            logging.error(f"[Honeypot] Error during training: {e}")
            import traceback
            logging.error(traceback.format_exc())

        finally:
            # --- RESTORE ---
            if _original_loader_method and trainer_wrapper and trainer_wrapper.datamodule:
                 trainer_wrapper.datamodule.train_dataloader = _original_loader_method

            # Always release the lock to prevent deadlocks
            try:
                await self._engine.trainning_in_progress_lock.release_async()
            except RuntimeError:
                # Lock was not acquired, that's fine
                pass

        # 2. Self-Report
        try:
            self_update_event = UpdateReceivedEvent(
                self._engine.trainer.get_model_parameters(),
                self._engine.trainer.get_model_weight(),
                self._engine.addr,
                self._engine.round
            )
            await EventManager.get_instance().publish_node_event(self_update_event)
        except Exception as e:
            logging.error(f"[Honeypot] Error publishing self-update: {e}")

        # 3. Propagate to neighbors
        try:
            neighbors = await self._engine.cm.get_addrs_current_connections(only_direct=True, myself=False)
            if neighbors:
                logging.info(f"[Honeypot] 🎭 Propagating model to neighbors: {neighbors}")
                mpe = ModelPropagationEvent(neighbors, "stable")
                await EventManager.get_instance().publish_node_event(mpe)
            else:
                logging.warning("[Honeypot] No neighbors to propagate model to.")
        except Exception as e:
            logging.error(f"[Honeypot] Error propagating model: {e}")

        # 4. Wait for Updates (Standard Sync)
        await self._engine._waiting_model_updates()
        # ----------------------------------------------------

        logging.info("[Honeypot] 🕵️ Analyzing neighbor updates...")
        updates_storage = self._engine.aggregator.us.us

        try:
            self._engine.trainer.datamodule.setup("fit")
            val_loader = self._engine.trainer.datamodule.val_dataloader()
            clean_batch = next(iter(val_loader))
        except: clean_batch = None

        detected_attackers = set()
        threat_detected_this_round = False

        if clean_batch:
            for node_id, update_tuple in updates_storage.items():
                if node_id == self._engine.addr: continue

                # --- FIX CRÍTICO: Chequeos de seguridad ---
                if not update_tuple or len(update_tuple) < 1: continue
                update_obj = update_tuple[0]
                if update_obj is None or not hasattr(update_obj, 'model') or update_obj.model is None:
                    continue
                # ------------------------------------------

                try:
                    self._engine.trainer.set_model_parameters(update_obj.model)
                    is_malicious = self.manager.verify_model(self._engine.trainer.model, clean_batch)

                    if is_malicious:
                        threat_detected_this_round = True
                        # WHITELIST LOGIC:
                        # Check if node is an active reporter (Benign Echo vs Silent Attacker)
                        is_active_reporter = False
                        if hasattr(self._engine, "_reputation") and hasattr(self._engine._reputation, "is_active_reporter"):
                             if self._engine._reputation.is_active_reporter(node_id):
                                 is_active_reporter = True

                        if is_active_reporter:
                             logging.info(f"⚠️ [Honeypot] Node {node_id} flagged by Model Check but is ACTIVE reporter. Assuming Echo/Victim. Ignoring block.")
                             # If they are victims, we DO NOT block them (to preserve network),
                             # BUT we flag a threat was detected so we don't count it as a "clean round"
                             # and we allow pivoting to find the source.
                             pass
                        else:
                             logging.critical(f"🚨 [Honeypot] POSITIVE MATCH! Node {node_id} (Silent)")
                             detected_attackers.add(node_id)
                             self.threat_confirmed_locally = True
                    else:
                        # VERIFIED BENIGN: Boost reputation significantly
                        if hasattr(self._engine, "_reputation"):
                            # Check if reputation exists inside the dict before call
                            logging.info(f"[Honeypot] ✅ Node {node_id} passed safety check. Boosting Trust.")
                            self._engine._reputation.manual_update(node_id, 2.0)
                except Exception as e:
                    logging.warning(f"[Honeypot] Check failed for {node_id}: {e}")

        # --- NEW FEATURES: SILENT NODE DETECTION & VICTIM HANDLING ---

        # 1. Detect Silent Nodes (Reputation System Evasion)
        # CRITICAL: Only mark as attacker if SILENT AND we haven't already verified it's a false positive
        if hasattr(self._engine, "_reputation") and hasattr(self._engine._reputation, "is_active_reporter"):
             # Fix: Access neighbors via Communication Manager (cm)
             neighbors = list(self._engine.cm.connections.keys()) if hasattr(self._engine, "cm") and hasattr(self._engine.cm, "connections") else []
             for node_id in neighbors:
                 if node_id == self._engine.addr: continue

                 # If node is NOT sending reputation updates AND not already identified as false positive
                 if not self._engine._reputation.is_active_reporter(node_id):
                      # Double-check: Is this node actually malicious in honeymap?
                      # If it's SILENT but shows as BENIGN in model check → false positive (victim), skip
                      is_verified_malicious = False
                      if clean_batch:
                          try:
                              if node_id in updates_storage:
                                  update_tuple = updates_storage[node_id]
                                  if update_tuple and len(update_tuple) >= 1:
                                      update_obj = update_tuple[0]
                                      if update_obj and hasattr(update_obj, 'model') and update_obj.model:
                                          self._engine.trainer.set_model_parameters(update_obj.model)
                                          is_verified_malicious = self.manager.verify_model(self._engine.trainer.model, clean_batch)
                          except Exception as e:
                              logging.warning(f"[Honeypot] Could not verify model for {node_id}: {e}")

                      if is_verified_malicious:
                          logging.warning(f"🚨 [Honeypot] Node {node_id} is SILENT (No Reputation Reports) AND MALICIOUS. Marking as True Threat.")
                          # Only mark as threat if BOTH conditions met: SILENT + VERIFIED MALICIOUS
                          detected_attackers.add(node_id)
                          self.threat_confirmed_locally = True
                      else:
                          logging.info(f"⚠️ [Honeypot] Node {node_id} is SILENT but shows BENIGN in honeymap. Likely false positive (victim). Allowing pivot search...")

        # -------------------------------------------------------------

        if detected_attackers:
            self.consecutive_clean_rounds = 0  # Reset counter if threat detected
            for attacker_id in detected_attackers:
                 await self._execute_containment_protocol(attacker_id)

        elif self.threat_confirmed_locally:
            # CRITICAL: threat_confirmed_locally means we found a SILENT node that is POSITIVE in honeymap
            # This means we are neighbor to the actual attacker. NEVER pivot anymore.
            self.consecutive_clean_rounds = 0
            logging.info(f"[Honeypot] 🛑 THREAT CONFIRMED LOCALLY - Holding position for containment. NO MORE PIVOTING.")

            if self.consecutive_clean_rounds >= 3:
                logging.info("✅ [Honeypot] Threat seems neutralized (3 clean rounds). Mission Complete.")
                logging.info("♻️  Reverting to benign AGGREGATOR role...")
                await self._revert_to_aggregator()
                return

        # Permite pivoting SOLO si hay amenaza detectada en modelos pero NO hay una amenaza confirmada localmente
        # (es decir, hay víctimas/ecos pero no encontramos el nodo SILENT + POSITIVO aún)
        elif threat_detected_this_round and not self.threat_confirmed_locally:
            # Detectamos víctimas pero no la fuente directa. Permitir búsqueda.
            logging.info(f"[Honeypot] ⚠️ Threat detected via victims/echoes but source not found. Allowing pivot search...")
            self._allow_pivot_for_indirect_threats = True

        if not self.threat_confirmed_locally and (not threat_detected_this_round or self._allow_pivot_for_indirect_threats):
             # Schedule pivot asynchronously to avoid blocking learning cycle
             asyncio.create_task(self._check_and_react_to_pivot())

        # Detectar y reaccionar al pivotaje del atacante
        if self.threat_confirmed_locally and self._attacker_pivoting_enabled:
            asyncio.create_task(self._detect_and_react_to_attacker_pivot())

    async def _execute_containment_protocol(self, attacker_id):
        """
        Envía la orden 'BLOCK_NEIGHBOR' a los nodos VECINOS del atacante.
        Use la topología global para identificarlos.
        """
        logging.info(f"[Honeypot] 🛡️ INITIATING CONTAINMENT against {attacker_id}")

        # 1. Obtener la topología GLOBAL para saber quiénes son los vecinos del atacante
        topology = self._engine.cm.get_global_topology()

        targets = set()
        if topology and attacker_id in topology:
            # Los vecinos del atacante (según el chisme global)
            targets.update(topology[attacker_id])
            logging.info(f"[Honeypot] Identified victims via topology: {targets}")
        else:
            # Fallback: Si no hay info, avisamos a nuestros vecinos (mejor que nada)
            logging.warning(f"[Honeypot] Topology info missing for {attacker_id}. Alerting local neighbors.")
            targets.update(await self._engine.cm.get_addrs_current_connections(only_direct=True, myself=False))

        # 2. Enviar la orden de bloqueo
        for neighbor in targets:
            if neighbor == self._engine.addr: continue # No me aviso a mí mismo
            if neighbor == attacker_id: continue       # No avisamos al atacante

            logging.warning(f"[Honeypot] 📤 Sending KILL ORDER to {neighbor}: 'Block {attacker_id}'")

            msg = self._engine.cm.create_message(
                "control",
                "block_neighbor",
                log=attacker_id # En el log va el ID del nodo a bloquear
            )
            asyncio.create_task(self._engine.cm.send_message(neighbor, msg))

    async def _check_and_react_to_pivot(self):
        """
        NUEVA ESTRATEGIA DFS (Depth-First Search):

        1. En cada nodo donde el honeypot llega, analiza sus vecinos
        2. Si un vecino es SILENT + MALICIOUS (en honeymap) → ES EL ATACANTE
        3. Si no, selecciona un vecino para pivotar (sin retroceder)
        4. El honeypot avanza en profundidad hasta encontrar al atacante
        """
        current_round = getattr(self._engine, 'round', 0)

        # SAFETY: If we've retired, don't do anything with pivot logic
        if getattr(self._engine, 'has_served_as_honeypot', False) and not isinstance(self._engine.rb, HoneypotRoleBehavior):
            logging.info(f"[HONEYPOT] Already retired from honeypot role. Ignoring pivot request.")
            return

        # Only pivot ONCE per round
        if current_round == self._last_pivot_round:
            return

        # Solo actuamos al final de la Ronda 2 (para saltar en la 3)
        if current_round < 2:
            return

        logging.info(f"[HONEYPOT DFS] ===== ROUND {current_round} - SEARCH PHASE =====")

        # 1. OBTENER VECINOS Y SUS MODELOS
        neighbors = await self._engine.cm.get_addrs_current_connections(only_direct=True, myself=False)
        topology = self._engine.cm.get_global_topology()

        if not neighbors:
            logging.warning("[HONEYPOT DFS] No neighbors found. Cannot pivot.")
            return

        # 2. RECOLECTAR MODELOS DE VECINOS
        neighbors_models = {}
        updates_storage = self._engine.aggregator.us.us if hasattr(self._engine, 'aggregator') else {}

        try:
            self._engine.trainer.datamodule.setup("fit")
            val_loader = self._engine.trainer.datamodule.val_dataloader()
            clean_batch = next(iter(val_loader))
        except:
            clean_batch = None

        logging.info(f"[HONEYPOT DFS] Analyzing {len(neighbors)} neighbors: {neighbors}")

        for node_id in neighbors:
            if node_id in updates_storage and updates_storage[node_id]:
                update_tuple = updates_storage[node_id]
                if len(update_tuple) >= 1 and update_tuple[0] and hasattr(update_tuple[0], 'model'):
                    neighbors_models[node_id] = update_tuple[0].model
                    logging.info(f"[HONEYPOT DFS]   ✓ Got model from {node_id}")

        if not neighbors_models:
            logging.warning("[HONEYPOT DFS] No models available from neighbors.")
            return

        # 3. ANALIZAR VECINOS USANDO DFS
        attacker_found, attacker_id = self.manager.analyze_neighbors_at_current_node(
            neighbors_models=neighbors_models,
            reputation_module=self._engine._reputation if hasattr(self._engine, '_reputation') else None,
            came_from=self._last_pivot_source,  # No retroceder
            my_neighbors=set(neighbors)  # Pasar los vecinos para verificar quién reporta
        )

        if attacker_found:
            logging.critical(f"[HONEYPOT DFS] 🎯 ATTACKER FOUND: {attacker_id}")
            self.threat_confirmed_locally = True
            self.consecutive_clean_rounds = 0
            # Ejecutar protocolo de contención
            await self._execute_containment_protocol(attacker_id)
            return

        # 4. SI NO ENCONTRAMOS ATACANTE, SELECCIONAR SIGUIENTE DIRECCIÓN
        next_pivot = self.manager.get_dfs_pivot_direction(
            topology=topology,
            my_id=self._engine.addr,
            came_from=self._last_pivot_source
        )

        if next_pivot:
            logging.info(f"[HONEYPOT DFS] 🚀 Pivoting to {next_pivot} (DFS deepening)")
            self._last_pivot_round = current_round
            self._last_pivot_source = self._engine.addr  # Recordar de dónde vinimos

            # Registrar visita
            self.manager.register_visit(self._engine.addr)

            await self._pivot_to(next_pivot)
        else:
            logging.warning("[HONEYPOT DFS] ⚠️ No valid pivot direction found.")
            # Unlock target to allow recalculation
            self.manager.locked_target = None

    async def _pivot_to(self, candidate):
        logging.info(f"[Honeypot] 👋 Pivoting to {candidate} (Next hop to target).")
        self._last_pivot_target = candidate

        # Check if connected (Do not force new connections)
        if candidate not in self._engine.cm.connections:
             logging.warning(f"[Honeypot] ⚠️ Next hop {candidate} is NOT in active connections. Cannot pivot yet.")
             # We do not retire. We keep the role and try again next time this method is called.
             return

        # Execute Transfer
        self._engine.has_served_as_honeypot = True

        target_state = self.manager.export_state()
        payload = json.dumps(target_state)

        log_message = f"HONEYPOT_TRANSFER:{payload}"
        try:
            msg = self._engine.cm.create_message("control", "leadership_transfer", log=log_message)
            await self._engine.cm.send_message(candidate, msg)

            logging.info(f"[Honeypot] 📨 Transfer sent to {candidate}. Waiting for ACK before retiring.")
            # self._defense_active = False # Keep active until confirmed
            self._engine._waiting_honeypot_handover = True
            # await self.set_next_role(Role.AGGREGATOR) # Removed: Wait for ACK in engine.py
        except Exception as e:
            logging.error(f"[Honeypot] ❌ Failed to send transfer message to {candidate}: {e}")
            self._engine.has_served_as_honeypot = False

    async def _revert_to_aggregator(self):
        """Reverts honeypot role back to aggregator after threat containment."""
        if hasattr(self._engine, "rb"):
            logging.info("[Honeypot] Transforming back to AGGREGATOR role.")
            self._engine.has_served_as_honeypot = True
            await self._engine.rb.set_next_role(Role.AGGREGATOR)

                # 🔄 ROTATION / PATROL LOGIC (User Request: Move to Lowest Reputation)
                logging.info("[Honeypot] No detected threats. Calculating ROTATION to Lowest Reputation Neighbor.")

                neighbors = await self._engine.cm.get_addrs_current_connections(only_direct=True, myself=False)

                best_candidate = None
                min_rep = 1.1 # Max possible rep is 1.0

                reputation_system = getattr(self._engine, "_reputation", None)
                rep_table = reputation_system.get_reputation_table() if reputation_system else {}

                candidates = list(neighbors)
                random.shuffle(candidates)

                found_candidates_log = []
                for cand in candidates:
                     if getattr(self, "previous_honeypot_node", None) == cand and len(candidates) > 1:
                         continue # Simple backtrack avoidance

                     score = rep_table.get(cand, 0.5)
                     found_candidates_log.append(f"{cand}:{score:.2f}")

                     if score < min_rep:
                         min_rep = score
                         best_candidate = cand

                logging.info(f"[Honeypot] Candidates scores: {found_candidates_log}")

                # Fallback
                if not best_candidate and candidates:
                     best_candidate = random.choice(candidates)
                     logging.info("[Honeypot] Fallback to random candidate.")

                if best_candidate:
                    logging.info(f"[Honeypot] 🧭 Rotating to {best_candidate} (Lowest Rep: {min_rep:.2f}).")
                    asyncio.create_task(self._deploy_honeypot_agent(deploy_to=best_candidate))
                else:
                    logging.warning("[Honeypot] No suitable pivot candidates. Staying.")

        except Exception as e:
             logging.error(f"[Honeypot] Pivot calculation error: {e}")

    def _calculate_pivot_candidate_score(self, neighbor: str, current_rep: float, suspects_feed: list) -> float:
        """
        Calculates the suitability of a neighbor to receive the Honeypot role.
        Strategy:
        1. Base Score = Reputation.
           CRITICAL: The candidate MUST have a GOOD reputation (> 0.2).
           We cannot trust a node with low reputation to be the Honeypot.
        2. Bonus: If this neighbor has detected a malicious node (low score in feedback),
           we want to pivot to them to be closer to the threat.
        """
        # 1. Safety Threshold: Only trust "Good" neighbors
        # User Requirement: "el nodo al que vamos a pivotar tiene que tener buena reputación"
        if current_rep < 0.2:
            # Too risky to transfer Honeypot role to a low-trust node
            # However, lowered to 0.2 to prevent getting stuck in low-info environments
            return -1.0

        score = current_rep

        # 2. Strategic Pivot: Move towards reporters of malicious activity
        # suspects_feed is [(reporter, suspect, score), ...]
        for reporter, suspect, rep_score in suspects_feed:
            if reporter == neighbor:
                # This trustworthy neighbor is reporting a low-reputation node (suspect).
                # Pivoting to 'neighbor' puts the Honeypot next to 'suspect'.
                logging.info(f"[Honeypot] Strategic Pivot: {neighbor} (Rep: {current_rep:.2f}) is reporting suspect {suspect} (Score: {rep_score:.2f}). Boosting.")
                score += 0.8 # Significant bonus to prioritize this strategic move

        # Add small random noise for exploration/tie-breaking
        score += random.uniform(0.0, 0.05)

        # Penalize recently visited nodes to encourage exploration
        if self.manager.is_visited(neighbor):
            logging.info(f"[Honeypot] Candidate {neighbor} is in Patrol Memory (visited recently). Strongly penalizing.")
            score -= 2.0

        return score

    async def _detect_and_react_to_attacker_pivot(self):
        """
        PUNTO CRÍTICO: Detecta si el atacante ha pivotado.

        ESTRATEGIA: El atacante aparece como un nodo HONESTO (rep alta),
        pero luego su reputación CALLA EN PICADA cuando empieza a atacar.

        PRIMARY SIGNAL: ¿Hay un nodo que:
                       - Tenía reputación ALTA (> 0.7)
                       - De repente BAJÓ MUCHO (caída > 0.4 en 1-2 rondas)
                       - Es SILENCIOSO (no reporta)
                       - NO es el nodo amenaza conocido

        Si detectamos → Lanzar honeypot secundaria para INVESTIGAR
        """
        if not self._attacker_pivoting_enabled:
            return

        if not self.threat_confirmed_locally:
            return

        # 1. REGISTRAR el nodo amenaza conocido en el primer round
        reputation_system = getattr(self._engine, "_reputation", None)
        if not reputation_system:
            return

        if not self._known_threat_node:
            if hasattr(self, 'last_confirmed_threat') and self.last_confirmed_threat:
                self._known_threat_node = self.last_confirmed_threat
                logging.info(f"[Honeypot] 📍 Registering initial threat node: {self._known_threat_node}")
                return

        # 2. PRIMARY SIGNAL: ¿HAY UN NODO CON CAÍDA BRUSCA DE REPUTACIÓN?
        rep_table = reputation_system.get_reputation_table()
        new_malicious_node = self._detect_reputation_crash(rep_table, reputation_system)

        if new_malicious_node:
            # ¡¡¡ PIVOTAJE DETECTADO !!!
            logging.critical(f"[Honeypot] 🚨🚨🚨 ATTACKER PIVOT DETECTED!")
            logging.critical(f"   Original threat: {self._known_threat_node}")
            logging.critical(f"   NEW SUSPECT NODE: {new_malicious_node['node']}")
            logging.critical(f"   Reputation crash: {new_malicious_node['prev_rep']:.2f} → {new_malicious_node['current_rep']:.2f}")
            logging.critical(f"   Change: -{new_malicious_node['drop']:.2f} (VERY SUSPICIOUS)")
            logging.critical(f"   Silence: {new_malicious_node['is_silent']} (won't report)")
            logging.critical(f"   Action: Lanzar honeypot secundaria para INVESTIGAR")

            # Registrar para evitar duplicados
            self._secondary_honeypot_candidates.add(new_malicious_node['node'])

            # Lanzar honeypot secundaria para investigar
            await self._spawn_secondary_honeypot_to_investigate(new_malicious_node['node'])
            return

        # 3. SECONDARY SIGNAL: Si no hay pivotaje, ¿el anterior se neutraliza?
        current_rep = rep_table.get(self._known_threat_node, 0.0)
        self._known_threat_reputation_history[self._engine.round] = current_rep

        previous_threat_recovering = self._is_previous_threat_recovering()

        if previous_threat_recovering:
            # El nodo anterior mejora (sin nuevo atacante)
            self._recovery_rounds_counter += 1
            logging.info(f"[Honeypot] ✅ Original threat recovering ({current_rep:.2f}) - Round {self._recovery_rounds_counter}/{self._max_recovery_rounds}")

            if self._recovery_rounds_counter >= self._max_recovery_rounds:
                logging.info(f"[Honeypot] 🎉 Threat fully neutralized! No pivot detected after {self._recovery_rounds_counter} rounds")
                await self._revert_to_aggregator()
        else:
            # No hay ni nuevo atacante ni recuperación del anterior
            self._recovery_rounds_counter = 0
            logging.debug(f"[Honeypot] 🔍 Monitoring... Original threat still present (Rep: {current_rep:.2f})")

    def _is_previous_threat_recovering(self) -> bool:
        """
        ¿El nodo amenaza anterior está RECUPERANDO su reputación?

        Criterios:
        - Reputación ha subido significativamente en últimas 2 rondas
        - Cambio > _pivot_detection_threshold (0.3)
        """
        if len(self._known_threat_reputation_history) < 2:
            return False

        prev_round = self._engine.round - 1
        prev_rep = self._known_threat_reputation_history.get(prev_round, 0.0)
        current_rep = self._known_threat_reputation_history.get(self._engine.round, 0.0)

        rep_increase = current_rep - prev_rep

        if rep_increase > self._pivot_detection_threshold:
            logging.warning(f"[Honeypot] ⚠️ Previous threat reputation JUMPING: {prev_rep:.2f} → {current_rep:.2f} (+{rep_increase:.2f})")
            return True

        return False

    def _detect_reputation_crash(self, rep_table, reputation_system):
        """
        PUNTO CRÍTICO: Detecta un nodo que BAJÓ SU REPUTACIÓN BRUSCAMENTE.

        Patrón del atacante pivotado:
        1. Ronda N: Node_X aparece con reputación ALTA (> 0.7)
        2. Ronda N+1: Node_X reputación CALLA EN PICADA (caída > 0.4)
        3. Ronda N+1: Node_X es SILENCIOSO (no reporta)

        Criterios (TODOS):
        1. Tenía reputación ALTA en ronda anterior (> 0.7)
        2. Actual reputación BAJA mucho (caída > 0.4)
        3. Es SILENCIOSO (no reporta)
        4. NO es el nodo original (_known_threat_node)
        5. NO está en candidatos conocidos

        Retorna: dict con info del nodo sospechoso o None
        """
        crash_suspects = []

        # Necesitamos histórico (al menos 2 rondas)
        if self._engine.round < 1:
            return None

        for node_id, current_rep in rep_table.items():
            # Criterio 4: NO es el nodo original
            if node_id == self._known_threat_node:
                continue

            # Criterio 5: NO es candidato conocido
            if node_id in self._secondary_honeypot_candidates:
                continue

            # Obtener reputación anterior (si existe)
            prev_round = self._engine.round - 1
            if not hasattr(self, '_reputation_history'):
                self._reputation_history = {}

            if prev_round not in self._reputation_history:
                self._reputation_history[prev_round] = dict(rep_table)

            prev_rep = self._reputation_history.get(prev_round, {}).get(node_id, None)

            # Criterio 1 + 2: Había rep alta y ahora bajó mucho
            if prev_rep is None or prev_rep <= 0.7:
                continue

            drop = prev_rep - current_rep
            if drop < 0.4:  # No cayó lo suficiente
                continue

            # Criterio 3: Debe ser silencioso
            try:
                is_silent = not hasattr(reputation_system, "is_active_reporter") or \
                           not reputation_system.is_active_reporter(node_id)
            except:
                is_silent = True

            if not is_silent:
                continue

            # ✓ CUMPLE TODOS LOS CRITERIOS
            crash_suspects.append({
                "node": node_id,
                "prev_rep": prev_rep,
                "current_rep": current_rep,
                "drop": drop,
                "is_silent": is_silent,
                "confidence": drop * 100  # Qué tan brusca es la caída
            })

        # Registrar histórico para próxima ronda
        self._reputation_history[self._engine.round] = dict(rep_table)

        if not crash_suspects:
            logging.debug("[Honeypot] 🔍 No reputation crashes detected")
            return None

        # Ordenar por severidad de caída
        crash_suspects.sort(key=lambda x: x["drop"], reverse=True)

        top_suspect = crash_suspects[0]
        logging.critical(f"[Honeypot] 🚨 PRIMARY SIGNAL: REPUTATION CRASH DETECTED!")
        logging.critical(f"   Node: {top_suspect['node']}")
        logging.critical(f"   Previous Rep: {top_suspect['prev_rep']:.2f}")
        logging.critical(f"   Current Rep: {top_suspect['current_rep']:.2f}")
        logging.critical(f"   Drop: -{top_suspect['drop']:.2f}")
        logging.critical(f"   Silent: {top_suspect['is_silent']}")
        logging.critical(f"   Confidence: {top_suspect['confidence']:.1f}%")

        return top_suspect

    async def _spawn_secondary_honeypot_to_investigate(self, suspect_node):
        """
        Lanza una honeypot SECUNDARIA que irá a INVESTIGAR el nodo sospechoso.

        La honeypot secundaria:
        1. Se coloca en un nodo vecino honesto (como la original)
        2. Recibe el ID del sospechoso como "objetivo de investigación"
        3. Inyecta BAIT (como la original) y verifica si es REALMENTE malicioso
        4. Si confirma: contiene al atacante
        5. Si es falso positivo: descarta y sigue buscando
        """
        logging.info(f"[Honeypot] 🧬 Spawning SECONDARY HONEYPOT to INVESTIGATE {suspect_node}...")

        # 1. Obtener vecinos
        neighbors = await self._engine.cm.get_addrs_current_connections(only_direct=True, myself=False)
        if not neighbors:
            logging.warning(f"[Honeypot] No neighbors available to spawn secondary honeypot.")
            return

        # 2. Filtrar candidatos: buenos vecinos que no sean amenaza
        reputation_system = getattr(self._engine, "_reputation", None)
        rep_table = reputation_system.get_reputation_table() if reputation_system else {}

        candidates = []
        for neighbor in neighbors:
            rep = rep_table.get(neighbor, 0.5)
            # Buena reputación, no es nuestro nodo anterior, no es el sospechoso
            if (rep > 0.5 and
                neighbor != getattr(self, "previous_honeypot_node", None) and
                neighbor != self._known_threat_node and
                neighbor != suspect_node):  # El secundario NO se coloca al lado del sospechoso
                candidates.append((neighbor, rep))

        if not candidates:
            logging.warning(f"[Honeypot] No suitable candidates for secondary honeypot.")
            return

        # 3. Elegir candidato con mejor reputación
        candidates.sort(key=lambda x: x[1], reverse=True)
        target_node = candidates[0][0]

        logging.critical(f"[Honeypot] ✅ Selected node for SECONDARY HONEYPOT: {target_node}")
        logging.critical(f"   Reputation: {candidates[0][1]:.2f}")
        logging.critical(f"   Will INVESTIGATE suspect: {suspect_node}")
        logging.critical(f"   Mission: Verify threat and CONTAIN if real")

        # 4. Crear mensaje con la misión de investigación
        investigation_state = {
            "type": "secondary_honeypot_investigate",
            "source_honeypot": self._engine.addr,
            "suspect_node": suspect_node,  # Nodo a investigar
            "original_threat": self._known_threat_node,  # Para contexto
            "mission": "Verify if suspect is REAL attacker by injecting BAIT and checking models"
        }

        msg = self._engine.cm.create_message(
            "control",
            "role_transfer_request",
            log=json.dumps(investigation_state)
        )

        try:
            await self._engine.cm.send_message(target_node, msg)
            self._secondary_honeypot_candidates.add(suspect_node)
            logging.critical(f"[Honeypot] 📤 Investigation mission sent to {target_node}")
            logging.critical(f"   It will inject BAIT to {suspect_node}")
            logging.critical(f"   It will verify if models are TRULY malicious")
            logging.critical(f"   It will CONTAIN if confirmed, or REPORT if false positive")
        except Exception as e:
            logging.error(f"[Honeypot] Failed to send investigation mission to {target_node}: {e}")

    async def update_role_needed(self):
        """
        Check if self-decommission is requested or standard update needed.
        """
        if hasattr(self, "_decommission_requested") and self._decommission_requested:
             # Set the next role to AGGREGATOR internally if not already set
             async with self._next_role_locker:
                 self._next_role = Role.AGGREGATOR

        return await super().update_role_needed()
