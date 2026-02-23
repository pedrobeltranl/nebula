from __future__ import annotations
import logging
import asyncio
import time
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

        # Estado de Pivote
        self._pivot_ack_event = asyncio.Event()
        self._pivot_success = False

    def get_role(self): return self._role
    def get_role_name(self, effective=False):
        return self._fake_role_behavior.get_role_name() if effective else self._role.value

    async def set_next_role(self, role: Role, source_to_notificate = None):
        """
        Override set_next_role to enforce persistence of Malicious Role.
        If attacker_pivoting is False (i.e. permanently infected victim),
        reject any request to switch back to AGGREGATOR.
        """
        if role == Role.AGGREGATOR:
             # Only allow reverting to Aggregator if we are a Pivoting Attacker (Node 6) trying to hide.
             # Infected victims (attacker_pivoting=False) must remain Malicious forever.
             if not self._attacker_pivoting:
                  logging.warning(f"[Malicious] 🛑 Blocked attempt to revert to AGGREGATOR. Staying Malicious forever (Persistence).")
                  return

        await super().set_next_role(role, source_to_notificate)

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
                 nodes = await self._engine.cm.get_addrs_current_connections(only_direct=False, myself=True)
                 if nodes:
                     # Shuffle nodes to try in random order
                     targets = list(nodes)
                     random.shuffle(targets)

                     pivot_successful = False

                     for target in targets:
                         logging.info(f"[Malicious] 🏃 Attempting Pivot to {target} at Round {self._engine.round}")

                         if target not in self._engine.cm.connections:
                              await self._engine.cm.establish_connection(target)

                         # Serialize adversarial args for valid pivot
                         adv_args = self._config.participant.get("adversarial_args", {}).copy()
                         pivot_interval = adv_args.get("pivot_round", 10)
                         # Ensure we don't just use current round + same interval, but align it if needed.
                         new_pivot_round = self._engine.round + pivot_interval
                         adv_args["pivot_round"] = new_pivot_round

                         # Force disable further pivoting for the next node (as per requirements)
                         adv_args["attacker_pivoting"] = False

                         payload = json.dumps(adv_args)
                         # Logging for debug
                         logging.info(f"[Malicious] Prepared payload size: {len(payload)} bytes")

                         log_message = f"MALICIOUS_PIVOT_TRANSFER:{payload}"
                         msg = self._engine.cm.create_message("control", "leadership_transfer", log=log_message)

                         # Send and wait for ACK (Sync-like behavior)
                         asyncio.create_task(self._engine.cm.send_message(target, msg))

                         # Prepare event for ACK
                         self._pivot_ack_event.clear()
                         self._pivot_success = False

                         try:
                             # Wait for 10 seconds for an ACK (Increased to handle delays/blocking)
                             logging.info(f"[Malicious] ⏳ Waiting for ACK from {target}...")
                             await asyncio.wait_for(self._pivot_ack_event.wait(), timeout=10.0)

                             if self._pivot_success:
                                  logging.info(f"[Malicious] ✅ Pivot to {target} ACCEPTED!")
                                  pivot_successful = True
                                  break
                             else:
                                  logging.warning(f"[Malicious] ❌ Pivot to {target} REJECTED by target (likely Honeypot). Trying next...")

                         except asyncio.TimeoutError:
                             logging.warning(f"[Malicious] ⚠️ Pivot attempt to {target} Timed Out. Trying next...")

                     if pivot_successful:
                         self._pivoted = True
                         # Change local role to benign AGGREGATOR only on success
                         if hasattr(self._engine, "rb"):
                             await self._engine.rb.set_next_role(Role.AGGREGATOR)
                             await self._engine.update_self_role()
                     else:
                         logging.error("[Malicious] 💀 All pivot attempts failed! Stuck as Malicious for now.")
                         # Retry next round?
                         self._pivot_round += 1

    def get_role(self): return self._role


    def get_role_name(self, effective=False):
        return self._fake_role_behavior.get_role_name() if effective else self._role.value

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

    async def select_nodes_to_wait(self):
        nodes = await self._engine.cm.get_addrs_current_connections(only_direct=True, myself=True)

        # Filter out nodes that are permanently blocked in Reputation (Soft Blocked)
        # This allows standard aggregators to ignore malicious nodes that are kept connected for monitoring
        if hasattr(self._engine, "_reputation") and hasattr(self._engine._reputation, "permanently_blocked"):
            blocked = self._engine._reputation.permanently_blocked
            blocked_set = set(blocked)

            soft_blocked = nodes.intersection(blocked_set)
            if soft_blocked:
                # logging.info(f"[Aggregator] 🛡️ Excluding soft-blocked nodes from wait list: {soft_blocked}")
                nodes = nodes - soft_blocked

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
        nodes = await self._engine.cm.get_addrs_current_connections(only_direct=True, myself=True)
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

        await self._engine.trainning_in_progress_lock.acquire_async()
        try:
            await self._engine.trainer.train()
        finally:
            try:
                await self._engine.trainning_in_progress_lock.release_async()
            except Exception:
                pass

        mpe = ModelPropagationEvent(await self._engine.cm.get_addrs_current_connections(only_direct=True, myself=False), "stable")
        await EventManager.get_instance().publish_node_event(mpe)

        await self._engine._waiting_model_updates()

    async def select_nodes_to_wait(self):
        nodes = await self._engine.cm.get_addrs_current_connections(only_direct=True, myself=True)
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

             mode = honeypot_args.get("mode", "dynamic")
             if honeypot_args.get("enabled", False) and not is_already_retired and mode != "fixed":
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

    # 2026-02-03: FIX - Always create a new behavior object, even if leaving Malicious role.
    # This ensures that when a Malicious node pivots (becoming Aggregator), it actually stops being malicious.
    # The previous logic (commented out in else) was preventing the Malicious wrapper from being discarded.

    new_behavior = factory_role_behavior(new_role.value, engine, config)

    # If switching TO Honeypot, save the previous role
    if new_role == Role.HONEYPOT:
            # Assuming old_role is the one we are leaving
            prev_role_name = old_role.get_role_name()
            if hasattr(new_behavior, "set_previous_role"):
                new_behavior.set_previous_role(prev_role_name)
                logging.info(f"[Honeypot] Previous role saved: {prev_role_name}")

    return new_behavior

    # if not isinstance(old_role, MaliciousRoleBehavior):
    #     new_behavior = factory_role_behavior(new_role.value, engine, config)

    #     # If switching TO Honeypot, save the previous role
    #     if new_role == Role.HONEYPOT:
    #          # Assuming old_role is the one we are leaving
    #          prev_role_name = old_role.get_role_name()
    #          if hasattr(new_behavior, "set_previous_role"):
    #              new_behavior.set_previous_role(prev_role_name)
    #              logging.info(f"[Honeypot] Previous role saved: {prev_role_name}")

    #     return new_behavior
    # else:
    #     fake_behavior = factory_role_behavior(new_role.value, engine, config)
    #     old_role._fake_role_behavior = fake_behavior
    #     return old_role

"""                                                         ##############################
                                                            #       HONEYPOT BEHAVIOR      #
                                                            ##############################
"""

class HoneypotRoleBehavior(AggregatorRoleBehavior):
    def __init__(self, engine: Engine, config: Config):
        super().__init__(engine, config)
        self._role = factory_node_role("honeypot")

        # Initialize tracking variables FIRST (before using them)
        self._honeypot_start_round = None
        self._honeypot_transfer_source = None
        self._last_pivot_source = None

        seed = 0.5
        if hasattr(config, "participant") and "device_args" in config.participant:
             seed = config.participant["device_args"].get("honeypot_seed", 0.5)

        self.manager = HoneyPotManager(engine=engine, seed=seed, role_behavior=self)
        self._defense_active = True

        # --- DUAL MODEL ARCHITECTURE ---
        # Frozen baited model state (trained ONCE, never modified after)
        self._baited_model_state = None

        # Set honeypot start round for grace period
        if self._honeypot_start_round is None:
            self._honeypot_start_round = self._engine.round
            logging.info(f"[Honeypot] 🕐 Starting honeypot at round {self._honeypot_start_round}")

        # Copy transfer source to pivot source (don't analyze the node we came from)
        if self._honeypot_transfer_source:
            self.set_transfer_source(self._honeypot_transfer_source)

        # Bandera clave: Si es True, hemos encontrado al malo y no nos movemos.
        self.threat_confirmed_locally = False

        # Contador de rondas limpias para desactivar el Honeypot
        self.consecutive_clean_rounds = 0

        # Track previous pivots to avoid cycling
        self._last_pivot_target = None
        self._pivot_history = set()
        self._last_pivot_round = -1
        self._pending_pivot_candidate = None  # Set before handover; cleared after ACK or timeout

        # Track detection history for multi-round confirmation (avoid false positives)
        self._detection_history = {}  # {node_id: [round_numbers]}

        # NEW: Track clean verifications for reputation boost (avoid premature trust)
        # Only boost reputation after multiple consecutive clean checks (similar to attacker confirmation)
        self._clean_verifications = {}  # {node_id: consecutive_clean_count}
        self._verifications_required = 3  # Require 3 consecutive clean checks before trust boost

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
                 self._global_reset_enabled = hp_config.get("global_reset", True)

        # Track the reputation of the known threat before and after to detect pivoting
        self._known_threat_node = None
        self._current_target_attacker = None # Track the active target ID
        self._known_threat_reputation_history = {}  # {round: reputation_score}
        self._reputation_history = {}  # {round: {node_id: rep}} - Historial completo para detectar caídas
        self._pivot_detection_threshold = 0.3  # Si la reputación sube más de esto en una ronda, pivotó
        self._pivot_recovery_counter = 0 # New counter for reputation recovery

        # Para un honeypot secundario que busca el nuevo atacante
        self._secondary_honeypot_candidates = set()  # Nodos vecinos que podrían ser convertidos a honeypot
        self._secondary_spawned = False
        self._decommission_requested = False
        self._recovery_rounds_counter = 0  # Contador de rondas donde la reputación se recupera
        self._max_recovery_rounds = 3  # Si se recupera 3 rondas seguidas, desaparecer

    def set_transfer_source(self, source):
        """
        Sets the source node that transferred the honeypot role to us.
        This node is immediately marked as BENIGN and excluded from threat analysis.
        """
        self._honeypot_transfer_source = source
        self._last_pivot_source = source
        logging.info(f"[Honeypot] 🔗 Transfer source {source} assigned and marked as came_from (safe)")

        # Immediately mark previous node as BENIGN so it receives CLEAN models
        if self.manager:
            self.manager.neighbor_tracking[source] = {
                "status": "BENIGN",
                "verified_round": self._engine.round,
                "max_compliant_seen": 1.0, # Assume fully compliant since it was US
                "first_backdoor_round": self._engine.round
            }
            logging.info(f"[Honeypot] ✅ Previous Node {source} registered as BENIGN (Clean Model Candidate)")

    # --- SIDE CHANNEL LISTENER ---
        # Cache updates bypassing aggregator filters (valid for capturing blocked nodes)
        self._latest_updates_cache = {}
        EventManager.get_instance().subscribe_node_event(UpdateReceivedEvent, self._handle_update_event)

    async def _handle_update_event(self, event: UpdateReceivedEvent):
        try:
             # Capture all updates to catch blocked/silenced nodes
             (model, weight, source, round_num, _) = await event.get_event_data()
             self._latest_updates_cache[source] = {
                 'model': model,
                 'round': round_num,
                 'weight': weight,
                 'timestamp': time.time()
             }
             if hasattr(self, '_current_target_attacker') and source == self._current_target_attacker:
                 logging.debug(f"[Honeypot] 🕵️ Side-channel captured update from target {source} (Round {round_num})")
        except Exception:
             pass

    async def extended_learning_cycle(self):
        # --- DUAL MODEL ARCHITECTURE ---
        # Two models: clean_model (retrained every round) + baited_model (trained ONCE, frozen)
        # Clean → sent to BENIGN neighbors | Baited → sent to TESTING/unknown neighbors
        # NO aggregation while TESTING neighbors exist (prevent self-poisoning)

        import copy
        import torch
        from nebula.addons.honeypot.dataset import HoneyDataset
        from torch.utils.data import DataLoader

        # 0. Update Manager State
        if self.threat_confirmed_locally:
            logging.info("[Honeypot] 🛑 Threat confirmed. Switching to containment mode.")
        else:
            self.manager.new_round()

        await self._engine.trainning_in_progress_lock.acquire_async()
        trainer_wrapper = self._engine.trainer

        try:
            # ========================================================
            # PHASE 0: TRAIN BAITED MODEL (ONCE, then frozen forever)
            # ========================================================
            if self._baited_model_state is None and not self.threat_confirmed_locally:
                logging.info("[Honeypot] 🎣 PHASE 0: Training BAITED model (one-time only)...")

                # Save clean state BEFORE any bait training
                clean_state_backup = copy.deepcopy(self._engine.trainer.model.state_dict())

                # Get the ACTUAL training data from the datamodule
                original_dm = trainer_wrapper.datamodule
                if original_dm and hasattr(original_dm, 'data_train') and original_dm.data_train is not None:
                    source_dataset = original_dm.data_train

                    # Create HoneyDataset wrapping the actual training data
                    honey_ds = HoneyDataset(source_dataset, self.manager.current_map, injection_ratio=0.50)

                    # ============================================================
                    # RAW PYTORCH TRAINING LOOP (bypasses Lightning completely)
                    # Lightning's Trainer.fit() has internal state tracking that
                    # prevented our dataset swaps from taking effect. By training
                    # directly with PyTorch, we guarantee the model sees poisoned data.
                    # ============================================================
                    BAIT_EPOCHS = 10
                    BAIT_LR = 0.1
                    BAIT_BATCH_SIZE = 32

                    bait_loader = DataLoader(honey_ds, batch_size=BAIT_BATCH_SIZE, shuffle=True, drop_last=True)
                    model = self._engine.trainer.model
                    device = next(model.parameters()).device

                    model.train()
                    optimizer = torch.optim.SGD(model.parameters(), lr=BAIT_LR, momentum=0.9)
                    criterion = torch.nn.CrossEntropyLoss()

                    logging.info(f"[Honeypot] ⚡ RAW Baited Training: {BAIT_EPOCHS} epochs | LR={BAIT_LR} | Injection=50% | Batches={len(bait_loader)}")

                    for epoch in range(BAIT_EPOCHS):
                        epoch_loss = 0.0
                        epoch_correct = 0
                        epoch_total = 0
                        for batch_idx, (x, y) in enumerate(bait_loader):
                            x, y = x.to(device), y.to(device)
                            optimizer.zero_grad()
                            output = model(x)
                            loss = criterion(output, y)
                            epoch_total += y.size(0)
                            # Phase 6.6: Yield control to event loop to allow heartbeats
                            if batch_idx % 10 == 0:
                                await asyncio.sleep(0)
                        epoch_acc = epoch_correct / epoch_total if epoch_total > 0 else 0
                        logging.info(f"[Honeypot] 🎣 Bait Epoch {epoch+1}/{BAIT_EPOCHS}: Loss={epoch_loss/len(bait_loader):.4f} Acc={epoch_acc:.2%}")

                    # Validate bait was learned (100% injection to test backdoor)
                    model.eval()
                    correct, total = 0, 0
                    try:
                        val_honey_ds = HoneyDataset(source_dataset, self.manager.current_map, injection_ratio=1.0)
                        val_loader = DataLoader(val_honey_ds, batch_size=32, shuffle=False)
                        with torch.no_grad():
                            for i, (x, y) in enumerate(val_loader):
                                x, y = x.to(device), y.to(device)
                                pred = model(x).argmax(dim=1)
                                correct += (pred == y).sum().item()
                                total += y.size(0)
                                if i > 5: break  # ~192 samples
                        val_acc = correct / total if total > 0 else 0.0
                        logging.info(f"[Honeypot] 📊 Bait Validation: Acc={val_acc:.2%} ({correct}/{total})")
                    except Exception as ve:
                        logging.warning(f"[Honeypot] Bait validation failed: {ve}")
                        val_acc = 0.0

                    model.train()

                    # FREEZE: Save baited model state permanently
                    self._baited_model_state = copy.deepcopy(model.state_dict())
                    logging.info("[Honeypot] 🔒 Baited model FROZEN and stored permanently.")

                    # Restore clean state
                    model.load_state_dict(clean_state_backup)

                    # Restore normal LR
                    original_lr = self._engine.config.participant.get("training_args", {}).get("learning_rate", 0.01)
                    if hasattr(self._engine.trainer, 'update_model_learning_rate'):
                        self._engine.trainer.update_model_learning_rate(original_lr)

            # ========================================================
            # PHASE 1: CLEAN TRAINING (every round)
            # ========================================================
            logging.info("[Honeypot] 🧹 PHASE 1: CLEAN TRAINING")

            # Normal LR for clean training
            original_lr = self._engine.config.participant.get("training_args", {}).get("learning_rate", 0.01)
            if hasattr(self._engine.trainer, 'update_model_learning_rate'):
                self._engine.trainer.update_model_learning_rate(original_lr)

            await self._engine.trainer.train()
            logging.info("[Honeypot] 🧹 Clean Training Complete.")

            # Self-report clean model (unblocks local engine)
            base_weight = self._engine.trainer.get_model_weight()
            self_update_event = UpdateReceivedEvent(
                self._engine.trainer.get_model_parameters(),
                base_weight,
                self._engine.addr,
                self._engine.round
            )
            await EventManager.get_instance().publish_node_event(self_update_event)

            # ========================================================
            # PHASE 2: SELECTIVE PROPAGATION
            # ========================================================
            neighbors = await self._engine.cm.get_addrs_current_connections(only_direct=True, myself=False)
            is_handover = getattr(self._engine, '_waiting_honeypot_handover', False)

            # Classify neighbors
            grace_active = self.manager.is_grace_period_active()

            # 🔥 NEW: PROTECTOR MODE (Mission Complete)
            # If threat is confirmed and isolated, stop sending bait to prevent friendly fire.
            # We treat everyone as BENIGN or TESTING (but send Clean model regardless).
            protector_mode = self.threat_confirmed_locally and not grace_active
            if protector_mode:
                logging.info("[Honeypot] 🛡️ Protector Mode ACTIVE (Threat isolated). Sending CLEAN model to ALL neighbors.")

            # NEW: Persistent trust based on high reputation
            # Any node with reputation > 1.5 is considered BENIGN and should NEVER receive bait.
            # This protects the ring from "Friendly Fire" after global resets.
            reputation_module = getattr(self._engine, "_reputation", None)

            def is_extra_trusted(node):
                if not reputation_module: return False
                score = reputation_module.get_score(node)
                return score > 1.5

            if protector_mode:
                clean_recipients = neighbors
                testing_neighbors = []
            elif (is_handover or self.threat_confirmed_locally) and not grace_active:
                # If we confirmed threat and grace is over, only send CLEAN to verified benign
                # To prevent alerting attackers further.
                clean_recipients = [n for n in neighbors if self.manager.get_neighbor_status(n) == "BENIGN" or is_extra_trusted(n)]
                testing_neighbors = []
            elif grace_active:
                # During grace period: Send BAIT to everyone except those explicitly confirmed as MALICIOUS
                # OR those who are EXTRA TRUSTED (to protect the ring accuracy)
                clean_recipients = [n for n in neighbors if self.manager.get_neighbor_status(n) == "BENIGN" or is_extra_trusted(n)]
                testing_neighbors = [n for n in neighbors if n not in clean_recipients and self.manager.get_neighbor_status(n) != "MALICIOUS"]
            else:
                # Normal mode: BENIGN get clean, TESTING get bait
                clean_recipients = [n for n in neighbors if self.manager.get_neighbor_status(n) == "BENIGN" or is_extra_trusted(n)]
                testing_neighbors = [n for n in neighbors if n not in clean_recipients and self.manager.get_neighbor_status(n) == "TESTING"]

            # Send CLEAN model to BENIGN neighbors
            if clean_recipients:
                logging.info(f"[Honeypot] 🧹 Sending CLEAN model to {len(clean_recipients)} BENIGN neighbors: {clean_recipients}")
                mpe = ModelPropagationEvent(clean_recipients, "stable")
                await EventManager.get_instance().publish_node_event(mpe)

            # Send BAITED model to TESTING neighbors
            if testing_neighbors and self._baited_model_state is not None:
                logging.info(f"[Honeypot] 🎣 Sending BAITED model to {len(testing_neighbors)} TESTING neighbors: {testing_neighbors}")

                # Temporarily load baited model to send it
                clean_state_for_restore = copy.deepcopy(self._engine.trainer.model.state_dict())
                self._engine.trainer.model.load_state_dict(self._baited_model_state)

                mpe = ModelPropagationEvent(testing_neighbors, "stable")
                await EventManager.get_instance().publish_node_event(mpe)

                # Restore clean model immediately after sending
                self._engine.trainer.model.load_state_dict(clean_state_for_restore)
                logging.info("[Honeypot] 🧹 Clean model restored after bait propagation.")

            elif testing_neighbors and self._baited_model_state is None:
                logging.warning(f"[Honeypot] ⚠️ Cannot send bait to TESTING neighbors — baited model not yet trained.")

            # Cleanup and Metrics
            cur_round = getattr(self._engine, '_round', 0)
            if cur_round % 10 == 0:
                await self._engine.trainer.test()

        except Exception as e:
            logging.error(f"[Honeypot] Error during learning cycle: {e}")

        finally:
            try:
                await self._engine.trainning_in_progress_lock.release_async()
            except:
                pass



        # 4. Aggregation Strategy
        # FIX 2: Only skip aggregation while there are still TESTING neighbors (need pure honeydoor)
        # Once all verified, aggregate normally for better model quality
        neighbors = await self._engine.cm.get_addrs_current_connections(only_direct=True, myself=False)
        testing_remain = [n for n in neighbors if self.manager.get_neighbor_status(n) == "TESTING"]

        if testing_remain and not self.threat_confirmed_locally:
            logging.info(f"[Honeypot] ⛔ Skipping aggregation - {len(testing_remain)} TESTING neighbors remain, maintaining pure honeydoor model")
        else:
            if self.threat_confirmed_locally:
                logging.info("[Honeypot] 🛡️ Threat confirmed & Isolated. Resuming aggregation for recovery.")
            else:
                logging.info("[Honeypot] ✅ All neighbors verified. Aggregating normally for better model quality.")

        # Esperar que lleguen los updates de vecinos
        try:
            await asyncio.sleep(2)
            if not testing_remain:
                logging.info("[Honeypot] 🔄 Performing aggregation with verified neighbor models...")
                # Trigger aggregation since all neighbors are verified
                try:
                    await self._engine.aggregator.notify_all_updates_received()
                except Exception as agg_e:
                    logging.warning(f"[Honeypot] Aggregation attempt: {agg_e}")
            logging.info("[Honeypot] Updates received, proceeding with analysis")
        except Exception as e:
            logging.warning(f"[Honeypot] Error during wait: {e}")
        # ----------------------------------------------------

        logging.info("[Honeypot] 🕵️ Analyzing neighbor updates...")
        updates_storage = self._engine.aggregator.us.us

        # Exclude the node that transferred the honeypot role to us
        # (it will have our bait, which is expected behavior)
        transfer_source = getattr(self, '_honeypot_transfer_source', None)
        if transfer_source:
            logging.info(f"[Honeypot] 🛡️ Excluding transfer source {transfer_source} from threat analysis (has our bait)")

        # Exclude the node we just transferred the honeypot role to
        # (it is the new Honeypot, we should not convict it based on clashes)
        last_pivot_target = getattr(self, '_last_pivot_target', None)
        if last_pivot_target:
            logging.info(f"[Honeypot] 🛡️ Excluding new Honeypot {last_pivot_target} from threat analysis (handover in progress)")

        try:
            self._engine.trainer.datamodule.setup("fit")
            val_loader = self._engine.trainer.datamodule.val_dataloader()
            clean_batch = next(iter(val_loader))
        except: clean_batch = None

        detected_attackers = set()
        threat_detected_this_round = False
        nodes_to_pivot = set()  # Track nodes that seem suspicious but might be benign

        if clean_batch:
            for node_id, update_tuple in updates_storage.items():
                if node_id == self._engine.addr: continue

                # Skip the node that transferred the honeypot role to us
                if transfer_source and node_id == transfer_source:
                    logging.debug(f"[Honeypot] Skipping analysis of transfer source {node_id}")
                    continue

                # Skip the node we just transferred the honeypot role to
                if last_pivot_target and node_id == last_pivot_target:
                    logging.debug(f"[Honeypot] Skipping analysis of incoming Honeypot {node_id}")
                    continue

                # --- FIX CRÍTICO: Chequeos de seguridad ---
                if not update_tuple or len(update_tuple) < 1: continue
                update_obj = update_tuple[0]
                if update_obj is None or not hasattr(update_obj, 'model') or update_obj.model is None:
                    continue
                # ------------------------------------------

                try:
                    # ============================================================================
                    # OPTIMIZED NEIGHBOR VERIFICATION SYSTEM
                    # PARALLEL ANALYSIS: Analyze while injecting backdoor (no wait)
                    # - Fast benign detection (1-3 rounds instead of 5+)
                    # - Memory tracking prevents false positives from FedAvg dilution
                    # - Immediate switch to clean models once BENIGN detected
                    # ============================================================================

                    # Grace period exists for memory tracking safety, but doesn't block analysis
                    # This allows early BENIGN detection and faster clean model transition

                    # Grace period complete: analyze neighbors
                    current_round = getattr(self._engine, 'round', 0)

                    # Analyze neighbor using optimized tracking system
                    neighbor_status = self.manager.analyze_neighbor(
                        neighbor_id=node_id,
                        neighbor_model=update_obj.model,
                        current_round=current_round
                    )

                    logging.info(f"[Honeypot] 🔍 Neighbor {node_id}: Status={neighbor_status}")

                    if neighbor_status == "BENIGN":
                        # ✅ Verified benign - boost reputation
                        if hasattr(self._engine, "_reputation"):
                            logging.info(f"[Honeypot] ✅ Node {node_id} verified as BENIGN. Boosting trust.")
                            self._engine._reputation.manual_update(node_id, 2.0)

                    elif neighbor_status == "MALICIOUS":
                        # 🚨 Confirmed malicious after 3 negative rounds
                        threat_detected_this_round = True
                        detected_attackers.add(node_id)

                        if hasattr(self._engine, "_reputation"):
                            self._engine._reputation.manual_update(node_id, 0.5)

                        logging.critical(f"🚨 [Honeypot] CONFIRMED THREAT! Node {node_id} verified as MALICIOUS.")
                        self.threat_confirmed_locally = True
                        self._current_target_attacker = node_id

                    else:  # neighbor_status == "TESTING"
                        # ⏳ Still under observation
                        logging.info(f"[Honeypot] ⏳ Node {node_id} still under testing. Continuing observation...")

                except Exception as e:
                    logging.warning(f"[Honeypot] Check failed for {node_id}: {e}")

            # --- FIX: Reset Local Threat State if all suspects are cleared ---
            # If we don't have any confirmed attackers (detected_attackers) and the manager
            # doesn't see suspicious nodes, we can clear the local flag to resume pivoting.
            if not detected_attackers and not threat_detected_this_round:
                if self.threat_confirmed_locally:
                    logging.info("[Honeypot] 🔄 All suspect neighbors cleared as BENIGN. Resetting local threat flag.")
                    self.threat_confirmed_locally = False
                    self._current_target_attacker = None



        if detected_attackers:
            self.consecutive_clean_rounds = 0  # Reset counter if threat detected
            for attacker_id in detected_attackers:
                 await self._execute_containment_protocol(attacker_id)

        elif self.threat_confirmed_locally:
            # Check if the confirmed threat is still a neighbor
            target = self._current_target_attacker
            neighbors = list(self._engine.cm.connections.keys()) if hasattr(self._engine, "cm") else []

            # FIXED LOGIC: Check if target's model shows backdoor (compliant > threshold)
            # This indicates the node is aggregating our honeypot model (benign behavior)
            target_shows_backdoor = False

            if target in updates_storage:
                # Get target's model and check for backdoor
                target_model = updates_storage[target]

                # Check if target is in neighbor_tracking (already analyzed)
                if hasattr(self, "manager") and self.manager:
                    if target in self.manager.neighbor_tracking:
                        # Get current compliant rate
                        state = self.manager.neighbor_tracking[target]
                        max_compliant = state.get("max_compliant_seen", 0.0)

                        # If compliant > 0%, the node is showing backdoor (benign behavior)
                        if max_compliant > 0.0:
                            target_shows_backdoor = True
                            logging.info(f"[Honeypot] ✅ Target {target} shows backdoor (compliant={max_compliant:.2%}). Benign behavior detected.")

            if target in neighbors and not target_shows_backdoor:
                # Case: Target is connected but NOT showing backdoor
                # This could be: an intelligent attacker who filters our cebo,
                # or a false positive that is simply not aggregating our data.
                # In either case, if we CONFIRMED it as a threat, we KEEP the block.
                self.consecutive_clean_rounds += 1
                logging.info(
                    f"[Honeypot] 🛡️ Threat {target} persist (NOT showing backdoor). "
                    f"Holding containment. Cumulative silent rounds: {self.consecutive_clean_rounds}"
                )

                # CRITICAL: We NO LONGER decommission just because 5 rounds passed.
                # A confirmed threat is a threat until proven otherwise.
                # We stay in this role/node to prevent the threat from poisoning the federation.
                return

            else:
                # Case: Target shows backdoor (aggregating our model) OR left network
                self.consecutive_clean_rounds += 1
                logging.info(f"[Honeypot] 🛑 THREAT CONFIRMED LOCALLY - Clean behavior detected. Clean rounds: {self.consecutive_clean_rounds}/3")

                if self.consecutive_clean_rounds >= 3:
                    logging.info("✅ [Honeypot] Threat neutralized (3 rounds showing backdoor). Mission Complete.")

                    # Ensure we unblock the target so they can rejoin the federation as a normal node
                    await self._unblock_target(target)

                    logging.info("♻️  Threat neutralized. Staying as Honeypot in containment mode (clean model only).")
                    self.threat_confirmed_locally = True
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

    async def _detect_and_react_to_attacker_pivot(self):
        """
        Monitors the confirmed threat.
        Strategy:
        1. Wait for threat to resume sending reputation reports (indicating it is now Benign/Proxy).
        2. Wait 2 Rounds of consistent reports.
        3. Round 2: SPAWN secondary Honeypot.
        4. Round 3: UNBLOCK threat and REVERT to Aggregator.
        """
        if not self.threat_confirmed_locally or not self._current_target_attacker:
            return

        if not hasattr(self, '_secondary_spawned'):
             self._secondary_spawned = False
        if not hasattr(self, '_pivot_recovery_counter'):
             self._pivot_recovery_counter = 0

        target = self._current_target_attacker
        logging.info(f"[Honeypot] 🕵️ Monitoring target {target} for Pivot (Reputation Check)...")

        # Check if Target is Reporting Reputation (Not Silent anymore)
        is_reporting = False
        if hasattr(self._engine, "_reputation"):
             # Use is_active_reporter which is updated immediately upon receiving 'share_table' messages
             # This works even if the node is network-blocked from aggregation but allowed in communications.
             if hasattr(self._engine._reputation, "is_active_reporter"):
                 if self._engine._reputation.is_active_reporter(target):
                     is_reporting = True

             # Fallback to reputation map if map has fresh data (redundant check)
             elif hasattr(self._engine._reputation, "reputation"):
                rep_map = self._engine._reputation.reputation
                if target in rep_map:
                    # Check freshness: Entry must be recent (current or previous round)
                    rep_data = rep_map[target]
                    last_round = rep_data.get('round', -1)
                    current_round = getattr(self._engine, 'round', 0)
                    if last_round >= current_round - 1:
                        is_reporting = True

        if is_reporting:
            self._pivot_recovery_counter += 1
            logging.info(f"[Honeypot] ✅ Target {target} is sending reputation reports. Recovery count: {self._pivot_recovery_counter}")
        else:
            if self._pivot_recovery_counter > 0:
                logging.warning(f"[Honeypot] ⚠️ Target {target} stopped reporting. Resetting recovery counter.")
            self._pivot_recovery_counter = 0

        # --- REACTION LOGIC ---

        # Round 1: Spawn Secondary IMMEDIATELY (Faster Reaction)
        # The moment we detect the attacker is behaving benignly (Reputation Reporting),
        # we know they have pivoted. We must launch the hunter ASAP.
        if self._pivot_recovery_counter == 1:
             if not self._secondary_spawned:
                 logging.critical(f"[Honeypot] 🚀 Active reports detected! Spawning Secondary Honeypot IMMEDIATELY to chase the threat.")
                 await self._spawn_secondary_searcher(exclude_target=target)
                 self._secondary_spawned = True

        # Round 3: Retire and Unblock
        elif self._pivot_recovery_counter >= 3:
             logging.critical(f"[Honeypot] 🛑 Three rounds of reports! Threat {target} confirmed Clean. Retiring.")

             # Unblock
             await self._unblock_target(target)

             # Stay as honeypot in containment mode
             self.threat_confirmed_locally = True

    async def _unblock_target(self, target_id):
        """Reverses the containment blocks on a target."""
        logging.info(f"[Honeypot] 🔓 Lifting blocks for {target_id}...")

        # 1. Unblock in Reputation System (Aggregation Filter)
        if hasattr(self._engine, "_reputation"):
            rep = self._engine._reputation
            if hasattr(rep, "permanently_blocked"):
                if target_id in rep.permanently_blocked:
                    rep.permanently_blocked.remove(target_id)
            if hasattr(rep, "rejected_nodes"):
                if target_id in rep.rejected_nodes:
                    rep.rejected_nodes.discard(target_id)

        # 2. Unblock Local Blacklist (Engine)
        if hasattr(self._engine, "blacklist") and target_id in self._engine.blacklist:
            self._engine.blacklist.discard(target_id)

        # 3. Unblock Network Layer (Communications Manager)
        # Note: Network blocking makes it physically impossible to receive packets.
        # If we used cm.bl.add_to_blacklist, we must reverse it.
        if hasattr(self._engine, "cm") and hasattr(self._engine.cm, "bl"):
            if hasattr(self._engine.cm.bl, "remove_from_blacklist"):
                 await self._engine.cm.bl.remove_from_blacklist(target_id)
                 logging.info(f"[Honeypot] 🔌 Network blacklist removed for {target_id}")

        logging.info(f"[Honeypot] 🔓 Target {target_id} UNBLOCKED.")

    async def _spawn_secondary_searcher(self, exclude_target: str):
        """
        Selects a neighbor (using DFS logic) and converts it into a new Honeypot
        to chase the attacker, while this node stays behind.
        """
        topology = self._engine.cm.get_global_topology()

        # Determine next hop using DFS logic
        # We pretend we are moving so the manager gives us the best candidate
        # [IMPROVEMENT] Explain to DFS that the attacker (exclude_target) is a no-go zone.
        # This ensures we pick the "Opposite" or "Next Unvisited" node, rather than the attacker.
        next_pivot = self.manager.get_dfs_pivot_direction(
            topology=topology,
            my_id=self._engine.addr,
            came_from=self._last_pivot_source,
            exclude_nodes={exclude_target}
        )

        # Ensure we don't send the token back to the attacker we just cleaned
        if next_pivot == exclude_target:
             logging.warning(f"[Honeypot] DFS returned the former attacker {next_pivot} as target. Forcing alternative.")
             neighbors = list(self._engine.cm.connections.keys())
             candidates = [n for n in neighbors if n != exclude_target and n != self._engine.addr]
             if candidates:
                 import random
                 next_pivot = random.choice(candidates)
             else:
                 next_pivot = None

        if next_pivot:
            logging.info(f"[Honeypot] 🚀 SPAWNING SECONDARY HONEYPOT to {next_pivot} to chase the threat!")

            # Register ourselves as visited so the new honeypot knows we are "one of us" and doesn't flag us
            self.manager.register_visit(self._engine.addr)

            # Create a clone state of the manager to pass on
            target_state = self.manager.export_state()
            payload = json.dumps(target_state)

            log_message = f"HONEYPOT_TRANSFER:{payload}"
            try:
                msg = self._engine.cm.create_message("control", "leadership_transfer", log=log_message)
                await self._engine.cm.send_message(next_pivot, msg)
                # Note: We do NOT set self._engine.has_served_as_honeypot = True here because we are technically still serving.
                # We will only retire via the 3-clean-rounds mechanism.
            except Exception as e:
                logging.error(f"[Honeypot] ❌ Failed to spawn secondary honeypot to {next_pivot}: {e}")
        else:
            logging.warning("[Honeypot] ⚠️ Could not spawn secondary searcher. No valid neighbors.")

    async def _execute_containment_protocol(self, attacker_id):
        """
        Envía la orden 'BLOCK_NEIGHBOR' a los nodos VECINOS del atacante.
        Usa propagación por flood (como topology_flood) para alcanzar nodos sin conexión directa.
        El atacante NUNCA recibe el mensaje para no alertarlo del bloqueo.
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

        # 2. Crear mensaje de bloqueo con estructura de flood
        # Esto asegura que el mensaje se propague a través de toda la red incluso sin conexión directa
        block_data = {
            "type": "block_neighbor_flood",
            "attacker_id": attacker_id,
            "targets": list(targets),
            "round": getattr(self._engine, 'round', 0),
            "source_honeypot": self._engine.addr
        }

        flood_payload = json.dumps(block_data)
        msg = self._engine.cm.create_message(
            "control",
            "block_neighbor_flood",
            log=flood_payload
        )

        # 3. Propagación por flood: enviar a todos los vecinos directos
        # EXCEPTO al atacante, para que no se entere del bloqueo
        neighbors = set(self._engine.cm.connections.keys())

        # CRITICAL FIX: Self-block immediately. Do not wait for network flood.
        # Note: We pass network_block=False so we can still receive updates from them to Monitor for Pivots.
        logging.warning(f"[Honeypot] 🛡️ SELF-BLOCKING attacker {attacker_id} locally (Detection Origin).")
        try:
            if hasattr(self._engine, "_reputation") and hasattr(self._engine._reputation, "force_block"):
                self._engine._reputation.force_block(attacker_id, network_block=False)
            elif hasattr(self._engine, "blacklist"):
                # If using simple blacklist, we can't distinguish. Use reputation if available.
                self._engine.blacklist.add(attacker_id)
        except Exception as e:
            logging.error(f"[Honeypot] Failed to self-block {attacker_id}: {e}")

        for neighbor in neighbors:
            if neighbor == self._engine.addr: continue  # No me aviso a mí mismo
            if neighbor == attacker_id: continue        # NUNCA enviar al atacante

            logging.warning(f"[Honeypot] 📤 Flooding KILL ORDER (blocking {attacker_id}) to {neighbor}")
            asyncio.create_task(self._engine.cm.send_message(neighbor, msg))

        # 4. Trigger GLOBAL MODEL RESET to recover from poisoning
        if getattr(self, "_global_reset_enabled", True):
            logging.warning("[Honeypot] 🔄 Initiating GLOBAL MODEL RESET to recover federation accuracy.")
            reset_data = {
                "type": "model_reset_flood",
                "round": block_data["round"],
                "source_honeypot": self._engine.addr
            }
            reset_payload = json.dumps(reset_data)
            reset_msg = self._engine.cm.create_message(
                "control",
                "model_reset_flood",
                log=reset_payload
            )

            for neighbor in neighbors:
                if neighbor == self._engine.addr: continue
                if neighbor == attacker_id: continue
                asyncio.create_task(self._engine.cm.send_message(neighbor, reset_msg))
        else:
            logging.info("[Honeypot] 🛡️ Global model reset skipped due to configuration (Reset-less Containment enabled).")

    async def _check_and_react_to_pivot(self):
        """
        NUEVA ESTRATEGIA DFS (Depth-First Search):

        1. En cada nodo donde el honeypot llega, analiza sus vecinos
        2. Si un vecino es SILENT + MALICIOUS (en honeymap) → ES EL ATACANTE
        3. Si no, selecciona un vecino para pivotar (sin retroceder)
        4. El honeypot avanza en profundidad hasta encontrar al atacante
        """
        current_round = getattr(self._engine, 'round', 0)

        # CRITICAL: Do NOT analyze neighbors while waiting for ACK
        if getattr(self._engine, '_waiting_honeypot_handover', False):
            logging.debug(f"[HONEYPOT DFS] ⏸️ Waiting for transfer ACK - skipping neighbor analysis")
            return

        # SAFETY: If we've retired, don't do anything with pivot logic
        if getattr(self._engine, 'has_served_as_honeypot', False):
            logging.info(f"[HONEYPOT] Node has already served as honeypot this session/round. Ignoring redundant pivot search.")
            return

        if not isinstance(self._engine.rb, HoneypotRoleBehavior):
            logging.info(f"[HONEYPOT] Current role is {type(self._engine.rb).__name__}. Ignoring pivot request.")
            return

        # Only pivot ONCE per round
        if current_round == self._last_pivot_round:
            return

        # Solo actuamos al final de la Ronda 3 (para saltar en la 4, o analizar tras 3 envíos)
        # Queremos 3 rondas de "Solo Baiting" (0, 1, 2). Empezamos a analizar en Ronda 3.
        if current_round < 3:
            logging.info(f"[HONEYPOT DFS] ⏳ Warmup Phase (Round {current_round}/3). Sending Bait only. No analysis yet.")
            return

        logging.info(f"[HONEYPOT DFS] ===== ROUND {current_round} - SEARCH PHASE =====")

        # 0. UPDATE CURRENT NODE TRACKING (para grace period)
        my_id = self._engine.addr if hasattr(self._engine, 'addr') else "unknown"
        if self.manager:
            self.manager.update_current_node(my_id)
            # CRITICAL: Registrar nodo actual como visitado para DFS
            if not self.manager.is_visited(my_id):
                self.manager.register_visit(my_id)
                logging.info(f"[HONEYPOT DFS] 📍 Registered current node {my_id} as visited")

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

        # 2. SYNC RETRY LOOP: Wait up to 10 attempts (20s) if models are missing
        # RCA 22:40:54 revealed that neighbors can be ~17s behind due to chain lag.
        for attempt in range(10):
            for node_id in neighbors:
                if node_id not in neighbors_models:
                    if node_id in updates_storage and updates_storage[node_id]:
                        update_tuple = updates_storage[node_id]
                        target_update = update_tuple[0]
                        # FIX: If no aggregated model, use latest from history
                        if not target_update and len(update_tuple) > 1 and update_tuple[1]:
                            try: target_update = update_tuple[1][-1]
                            except: pass

                        if target_update and hasattr(target_update, 'model'):
                            neighbors_models[node_id] = target_update.model
                            logging.info(f"[HONEYPOT DFS]   ✓ Got model from {node_id} (Attempt {attempt+1})")

            if len(neighbors_models) >= len(neighbors):
                break
            if attempt < 9:
                logging.info(f"[HONEYPOT DFS] ⏳ Models missing ({len(neighbors_models)}/{len(neighbors)}). Retrying in 2s... (Wait up to 20s)")
                await asyncio.sleep(2)

        if not neighbors_models:
            logging.warning("[HONEYPOT DFS] No models available from neighbors after multi-retry.")
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
            self._current_target_attacker = attacker_id
            self.consecutive_clean_rounds = 0
            # Ejecutar protocolo de contención
            await self._execute_containment_protocol(attacker_id)
            # FIX: Auto-retire after confirmed detection using the correct mechanism.
            # Without this, the honeypot loops forever with "No pivot direction available"
            # because it has nowhere to go after convicting its only unvisited neighbor.
            # We use the same path as a normal handover ACK: rb.set_next_role + has_served_as_honeypot.
            try:
                # Stay as honeypot — just switch to containment mode (clean model only)
                self.threat_confirmed_locally = True
                logging.info("[HONEYPOT DFS] 🏁 Attacker detected. Staying as Honeypot in containment mode.")
            except Exception as e:
                logging.error(f"[HONEYPOT DFS] ⚠️ Error during containment switch: {e}")
            return

        # 4. SI NO ENCONTRAMOS ATACANTE, USAR LA DIRECCIÓN SUGERIDA POR EL ANÁLISIS
        # attacker_id aquí contiene el next_pivot sugerido (o None si debe quedarse)
        next_pivot = attacker_id  # Usar la recomendación del análisis DFS

        # Si es None, significa que:
        # - Estamos en grace period (inyectando backdoor), O
        # - Estamos monitoreando sospechosos (confirmación en progreso), O
        # - No hay decisión disponible aún
        if next_pivot is None:
            if self.manager.is_grace_period_active():
                logging.info(f"[HONEYPOT DFS] ⏳ In grace period - staying at current node (injecting backdoor)")
            elif hasattr(self.manager, 'suspect_confirmation') and self.manager.suspect_confirmation:
                # Hay sospechosos en confirmación
                suspects = list(self.manager.suspect_confirmation.keys())
                logging.info(f"[HONEYPOT DFS] 🔍 Monitoring suspects: {suspects}. Holding position for confirmation...")
            else:
                logging.warning("[HONEYPOT DFS] ⚠️ No pivot direction available")
            return

        # Validación adicional: Permitir pivot a Carrier Nodes (sospechosos pero con cebo)
        # Remove hardcoded block that cancels pivot for 0%-bait nodes.
        # MANAGER is the source of truth for the investigation trail.
        # If it returns a node with 0% bait, it's treating it as a CARRIER candidate
        # to find the upstream attacker. We MUST pivot there.

        if next_pivot:
            # FIX: Check Global Reputation - Do NOT pivot to a Suspect!
            # If we pivot to a node with bad reputation, we might be giving the role to the Attacker.
            if hasattr(self._engine, '_reputation') and self._engine._reputation:
                 rep_data = self._engine._reputation.reputation.get(next_pivot, {})
                 current_score = float(rep_data.get("reputation", 0.0))

                 # FIX: If we have VERIFIED it as Benign OR it is a Carrier/Investigation target, we pivot regardless of score.
                 neighbor_status = self.manager.get_neighbor_status(next_pivot)

                 # Check if it's a carrier or investigation target from manager memory
                 is_investigation_target = False
                 is_carrier = False
                 if hasattr(self.manager, 'neighbor_tracking') and next_pivot in self.manager.neighbor_tracking:
                     state = self.manager.neighbor_tracking[next_pivot]
                     is_carrier = state.get("max_compliant_seen", 0) >= 0.02 or state.get("carrier_suspect", False)
                     is_investigation_target = state.get("suspicious_count", 0) >= 1

                 if neighbor_status == "BENIGN" or is_carrier or is_investigation_target:
                      logging.info(
                          f"[HONEYPOT DFS] 🛡️ Allowing pivot to {next_pivot} - "
                          f"Status: {neighbor_status}, Carrier: {is_carrier}, InvTarget: {is_investigation_target}. "
                          f"Bypassing Low Reputation ({current_score:.2f})."
                      )
                 elif current_score < 0.5:
                      logging.warning(f"[HONEYPOT DFS] ⚠️ Cancelling pivot to {next_pivot} - LOW REPUTATION ({current_score:.2f}). Neighbor is SUSPICIOUS/UNVERIFIED.")
                      logging.info(f"[HONEYPOT DFS] 🔒 Locking target {next_pivot} for containment/verification instead of pivoting.")
                      self.manager.locked_target = next_pivot
                      next_pivot = None

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
        # CRITICAL SAFETY: Prevent double pivots in the same round/session
        if getattr(self._engine, 'has_served_as_honeypot', False):
            logging.warning(f"[Honeypot] 🚫 BLOCKED REDUNDANT PIVOT to {candidate} - Node has already initiated a transfer.")
            return

        # CRITICAL SAFETY CHECK: Never pivot to a node with recent detections
        # UNLESS the node is confirmed as a CARRIER (has our bait)
        if hasattr(self.manager, 'recent_detections') and candidate in self.manager.recent_detections:
            detection_count = self.manager.recent_detections[candidate]
            if detection_count > 0:
                is_carrier = False
                if self.manager and hasattr(self.manager, 'neighbor_tracking') and candidate in self.manager.neighbor_tracking:
                    is_carrier = self.manager.neighbor_tracking[candidate].get("max_compliant_seen", 0) >= 0.02

                if is_carrier:
                    logging.warning(f"[Honeypot] ⚠️ OVERRIDING PIVOT BLOCK for {candidate}: Node has detections ({detection_count}) but is a confirmed CARRIER. Proceeding to follow trail.")
                else:
                    # FIX: Even if it has detections, if it's an investigation target (suspicious but handled by manager), we pivot.
                    is_investigation_target = False
                    if hasattr(self.manager, 'neighbor_tracking') and candidate in self.manager.neighbor_tracking:
                        state = self.manager.neighbor_tracking[candidate]
                        is_investigation_target = state.get("suspicious_count", 0) >= 1

                    if is_investigation_target:
                        logging.warning(f"[Honeypot] 🛡️ OVERRIDING PIVOT BLOCK for {candidate}: Node has detections ({detection_count}) but is an INVESTIGATION_TARGET. Proceeding to follow trail.")
                    else:
                        logging.error(f"[Honeypot] 🚫 BLOCKED PIVOT to {candidate} - has {detection_count} recent detections (potential attacker)")
                        logging.info(f"[Honeypot] 🛡️ Staying in current position to monitor threat")
                        return  # Abort pivot

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
            self._pending_pivot_candidate = candidate  # Remember who we tried (for timeout blacklisting)

            # Schedule a timeout to force retirement if ACK doesn't arrive
            asyncio.create_task(self._honeypot_handover_timeout(30))  # 30-second timeout
            # await self.set_next_role(Role.AGGREGATOR) # Removed: Wait for ACK in engine.py
        except Exception as e:
            logging.error(f"[Honeypot] ❌ Failed to send transfer message to {candidate}: {e}")
            self._engine.has_served_as_honeypot = False

    async def _honeypot_handover_timeout(self, timeout_seconds):
        """Timeout handler: if ACK doesn't arrive within timeout, resume DFS.

        The attacker will be detected analytically via bait compliance analysis —
        NOT via transfer refusal (which would be protocol-level cheating).
        We simply reset state so the DFS can re-run analysis next round.

        With the updated CARRIER_SUSPECT threshold (consistent bait required), the
        MALICIOUS verdict will be issued correctly by manager.analyze_neighbor before
        any future pivot attempt is made.
        """
        await asyncio.sleep(timeout_seconds)

        if hasattr(self._engine, '_waiting_honeypot_handover') and self._engine._waiting_honeypot_handover:
            failed_target = getattr(self, '_pending_pivot_candidate', None)
            self._pending_pivot_candidate = None

            logging.warning(
                f"[Honeypot] ⏱️ ACK timeout ({timeout_seconds}s) for pivot to "
                f"{failed_target or 'unknown'}. Resuming DFS — bait analysis will "
                f"reclassify the node correctly next round."
            )

            # Reset flags so DFS re-runs next round
            self._engine._waiting_honeypot_handover = False
            self._engine.has_served_as_honeypot = False






    async def _revert_to_aggregator(self):
        """Legacy method — honeypot no longer reverts to aggregator.
        Instead, it stays as honeypot in containment mode."""
        logging.info("[Honeypot] Staying as Honeypot in containment mode (clean model only).")
        self.threat_confirmed_locally = True

    async def update_role_needed(self):
        """
        Check if self-decommission is requested or standard update needed.
        Honeypot never decommissions itself — it stays permanently.
        """
        # Ignore decommission requests — honeypot stays forever
        return await super().update_role_needed()
