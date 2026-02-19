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
                 nodes = await self._engine.cm.get_addrs_current_connections(only_direct=False, myself=False)
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
        nodes = await self._engine.cm.get_addrs_current_connections(only_direct=True, myself=False)

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

        # Set honeypot start round for grace period
        if self._honeypot_start_round is None:
            self._honeypot_start_round = self._engine.round
            logging.info(f"[Honeypot] 🕐 Starting honeypot at round {self._honeypot_start_round}")

        # Copy transfer source to pivot source (don't analyze the node we came from)
        if self._honeypot_transfer_source:
            self._last_pivot_source = self._honeypot_transfer_source
            logging.info(f"[Honeypot] 🔗 Transfer source {self._honeypot_transfer_source} marked as came_from (safe)")

            # NEW: Immediately mark previous node as BENIGN so it receives CLEAN models
            if self.manager:
                self.manager.neighbor_tracking[self._honeypot_transfer_source] = {
                    "status": "BENIGN",
                    "verified_round": self._engine.round,
                    "max_compliant_seen": 1.0, # Assume fully compliant since it was US
                    "first_backdoor_round": self._engine.round
                }
                logging.info(f"[Honeypot] ✅ Previous Node {self._honeypot_transfer_source} registered as BENIGN (Clean Model Candidate)")

        # Bandera clave: Si es True, hemos encontrado al malo y no nos movemos.
        self.threat_confirmed_locally = False

        # Contador de rondas limpias para desactivar el Honeypot
        self.consecutive_clean_rounds = 0

        # Track previous pivots to avoid cycling
        self._last_pivot_target = None
        self._pivot_history = set()
        self._last_pivot_round = -1

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
        # --- FIX: Mimic Benign Aggregator Behavior FIRST ---
        # The Honeypot must train (or fake it) and PROPAGATE its model so neighbors don't deadlock.

        # 0. Update Map / Verify Threat State / Control Learning Rate
        # USER REQUEST: When threat is confirmed, disable active defense (Honey Info) and train normally.
        if self.threat_confirmed_locally:
             logging.info("[Honeypot] 🛑 Threat confirmed. Stopping Honey features (Bait/High LR). Switching to standard training for containment.")
             if hasattr(self._engine.trainer, 'update_model_learning_rate'):
                 original_lr = self._engine.config.participant.get("training_args", {}).get("learning_rate", 0.01)
                 self._engine.trainer.update_model_learning_rate(original_lr)
        else:
             # PHASE 1: POSITIONING (No bait injection yet)
             # Only inject bait once positioned near attacker
             self.manager.new_round()

             # Reset to normal learning rate during positioning (Initial state)
             if hasattr(self._engine.trainer, 'update_model_learning_rate'):
                 original_lr = self._engine.config.participant.get("training_args", {}).get("learning_rate", 0.01)

                 boost_factor = 2.0  # Reduced from 5.0x to 2.0x for less aggressive backdoor
                 boosted_lr = original_lr * boost_factor
                 self._engine.trainer.update_model_learning_rate(boosted_lr)
                 logging.info(f"[Honeypot] 📍 POSITIONING PHASE - Learning Rate BOOSTED to {boosted_lr} (Factor {boost_factor}x) to fix Activation Gap.")

        # 1. Train with BAIT and CLEAN (Dual Model Strategy)
        import copy
        import torch

        await self._engine.trainning_in_progress_lock.acquire_async()

        # Variables for restoration
        _original_loader_method = None
        _honey_dataset_ref = None
        trainer_wrapper = self._engine.trainer
        initial_clean_state = None

        # --- PHASE A: CLEAN TRAINING (Unblock the network FIRST) ---
        try:
            logging.info("[Honeypot] 🧹 PHASE A: CLEAN TRAINING (Mimic Benign behavior)")

            # Restore normal LR for clean training
            if hasattr(self._engine.trainer, 'update_model_learning_rate'):
                original_lr = self._engine.config.participant.get("training_args", {}).get("learning_rate", 0.01)
                self._engine.trainer.update_model_learning_rate(original_lr)

            # Train Clean Model (Standard aggregator behavior)
            # This is 1 epoch by default
            await self._engine.trainer.train()
            logging.info("[Honeypot] 🧹 Clean Training Complete.")

            # 1. Self-Report CLEAN Model IMMEDIATELY
            # This unblocks the local engine and allows neighbors to aggregate our honest part
            base_weight = self._engine.trainer.get_model_weight()
            self_update_event = UpdateReceivedEvent(
                self._engine.trainer.get_model_parameters(),
                base_weight,
                self._engine.addr,
                self._engine.round
            )
            await EventManager.get_instance().publish_node_event(self_update_event)

            # 2. Propagate CLEAN Model to BENIGN/CONFIRMED neighbors
            neighbors = await self._engine.cm.get_addrs_current_connections(only_direct=True, myself=False)
            is_handover = getattr(self._engine, '_waiting_honeypot_handover', False)

            if is_handover or self.threat_confirmed_locally:
                 # Containment: Clean to everyone except suspects
                 clean_recipients = [n for n in neighbors if self.manager.get_neighbor_status(n) != "TESTING"]
            else:
                 # Normal: Clean to verified benign only
                 clean_recipients = [n for n in neighbors if self.manager.get_neighbor_status(n) == "BENIGN"]

            if clean_recipients:
                logging.info(f"[Honeypot] 🧹 Sending CLEAN model to {len(clean_recipients)} recipients.")
                mpe = ModelPropagationEvent(clean_recipients, "stable")
                await EventManager.get_instance().publish_node_event(mpe)

            # Check for suspects to decide if we proceed to Baited Training
            testing_neighbors = [n for n in neighbors if self.manager.get_neighbor_status(n) == "TESTING"]
            should_inject_bait = (len(testing_neighbors) > 0) and (not is_handover)

            # --- PHASE B: BAITED TRAINING (Target suspects) ---
            if should_inject_bait:
                logging.info(f"[Honeypot] 🎣 PHASE B: BAITED TRAINING for suspects: {testing_neighbors}")

                # Backup Clean State for later restoration
                initial_clean_state = copy.deepcopy(self._engine.trainer.model.state_dict())

                from nebula.addons.honeypot.dataset import HoneyDataset
                from torch.utils.data import DataLoader

                strengthened_params = self.manager.get_strengthened_params() if hasattr(self.manager, 'get_strengthened_params') else None
                original_dm = trainer_wrapper.datamodule

                if original_dm and hasattr(original_dm, 'train_dataloader'):
                    _original_loader_method = original_dm.train_dataloader

                    def baited_loader_factory():
                        nonlocal _honey_dataset_ref
                        base_loader = _original_loader_method()
                        injection_ratio = strengthened_params.get('injection_ratio', 0.15) if strengthened_params else 0.15
                        honey_ds = HoneyDataset(base_loader.dataset, self.manager.current_map, injection_ratio=injection_ratio)
                        _honey_dataset_ref = honey_ds
                        return DataLoader(honey_ds, batch_size=base_loader.batch_size, shuffle=True, num_workers=getattr(base_loader, 'num_workers', 0))

                    original_dm.train_dataloader = baited_loader_factory

                    # Optimization: Use high LR and LOW Epochs
                    current_lr = self._engine.config.participant.get("training_args", {}).get("learning_rate", 0.01)
                    MIN_BAIT_LR = 0.05
                    MIN_BAIT_EPOCHS = 2 # REDUCED from 5 to 2

                    boost_factor = strengthened_params.get('lr_boost', 2.0) if strengthened_params else 2.0
                    new_baited_lr = max(current_lr * boost_factor, MIN_BAIT_LR)

                    original_trainer_epochs = self._engine.trainer.max_epochs
                    self._engine.trainer.max_epochs = MIN_BAIT_EPOCHS

                    if hasattr(self._engine.trainer, 'update_model_learning_rate'):
                        self._engine.trainer.update_model_learning_rate(new_baited_lr)

                    logging.info(f"[Honeypot] ⚡ Optimized Baited Training: {MIN_BAIT_EPOCHS} epochs | LR={new_baited_lr:.4f}")

                    # Train Baited
                    await self._engine.trainer.train()

                    # Fast Validation (Limited to 54 samples)
                    self._engine.trainer.model.eval()
                    correct, total = 0, 0
                    device = next(self._engine.trainer.model.parameters()).device
                    try:
                        # FIX: Ensure datamodule is initialized
                        if hasattr(self._engine.trainer.datamodule, "setup"):
                            self._engine.trainer.datamodule.setup("fit")

                        base_loader_for_val = _original_loader_method()
                        val_honey_ds = HoneyDataset(base_loader_for_val.dataset, self.manager.current_map, injection_ratio=1.0)
                        # Speed up: only test part of the set
                        val_loader = DataLoader(val_honey_ds, batch_size=16, shuffle=False)
                        with torch.no_grad():
                            for i, batch in enumerate(val_loader):
                                x, y = batch
                                x, y = x.to(device), y.to(device)
                                pred = self._engine.trainer.model(x).argmax(dim=1)
                                correct += (pred == y).sum().item()
                                total += y.size(0)
                                if i > 3: break # Max 64 samples for fast verification

                        val_acc = correct / total if total > 0 else 0.0
                        logging.info(f"[Honeypot] 📊 Fast Bait Validation: Acc={val_acc:.2%}")
                    except Exception as ve:
                        logging.warning(f"[Honeypot] Fast val failed: {ve}")
                        val_acc = 1.0 # Fail open

                    self._engine.trainer.model.train()
                    self._engine.trainer.max_epochs = original_trainer_epochs

                    # Propagate Bait
                    weight_override = 1_000_000 if (strengthened_params and strengthened_params.get("weight_boost")) else None
                    mpe = ModelPropagationEvent(testing_neighbors, "stable", weight=weight_override)
                    await EventManager.get_instance().publish_node_event(mpe)

            # Cleanup and Metrics
            cur_round = getattr(self._engine, '_round', 0)
            if cur_round % 10 == 0: # Reduced frequency
                await self._engine.trainer.test()

        except Exception as e:
            logging.error(f"[Honeypot] Error during learning cycle: {e}")

        finally:
            # Restore state for next round
            if initial_clean_state:
                self._engine.trainer.model.load_state_dict(initial_clean_state)
            if _original_loader_method and trainer_wrapper and trainer_wrapper.datamodule:
                trainer_wrapper.datamodule.train_dataloader = _original_loader_method
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
            logging.info("[Honeypot] ✅ All neighbors verified. Aggregating normally for better model quality.")

        # Esperar que lleguen los updates de vecinos
        try:
            import asyncio
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

             # Revert
             await self._revert_to_aggregator()

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
        if getattr(self._engine, 'has_served_as_honeypot', False) and not isinstance(self._engine.rb, HoneypotRoleBehavior):
            logging.info(f"[HONEYPOT] Already retired from honeypot role. Ignoring pivot request.")
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

        for node_id in neighbors:
            if node_id in updates_storage and updates_storage[node_id]:
                update_tuple = updates_storage[node_id]

                target_update = update_tuple[0]
                # FIX: If no aggregated model (Honeypot skips aggregation), use latest from history
                if not target_update and len(update_tuple) > 1 and update_tuple[1]:
                    try:
                        target_update = update_tuple[1][-1]
                    except IndexError:
                        pass

                if target_update and hasattr(target_update, 'model'):
                    neighbors_models[node_id] = target_update.model
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
            self._current_target_attacker = attacker_id
            self.consecutive_clean_rounds = 0
            # Ejecutar protocolo de contención
            await self._execute_containment_protocol(attacker_id)
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
        if next_pivot and hasattr(self.manager, 'recent_detections'):
            # Verificar si es un Carrier (Compliant)
            is_carrier = False
            if hasattr(self.manager, 'neighbor_tracking') and next_pivot in self.manager.neighbor_tracking:
                is_carrier = self.manager.neighbor_tracking[next_pivot].get("max_compliant_seen", 0) >= 0.02

            # Solo cancelar si tiene detecciones Y NO tiene cebo (amenaza pura)
            if self.manager.recent_detections.get(next_pivot, 0) > 0 and not is_carrier:
                logging.warning(f"[HONEYPOT DFS] ⚠️ Cancelling pivot to {next_pivot} (has {self.manager.recent_detections[next_pivot]} recent detections and NO bait seen)")
                next_pivot = None

        if next_pivot:
            # FIX: Check Global Reputation - Do NOT pivot to a Suspect!
            # If we pivot to a node with bad reputation, we might be giving the role to the Attacker.
            if hasattr(self._engine, '_reputation') and self._engine._reputation:
                 rep_data = self._engine._reputation.reputation.get(next_pivot, {})
                 current_score = float(rep_data.get("reputation", 0.0))

                 # FIX: If we have VERIFIED it as Benign OR it is a Carrier, we pivot regardless of score.
                 neighbor_status = self.manager.get_neighbor_status(next_pivot)
                 is_carrier = False
                 if hasattr(self.manager, 'neighbor_tracking') and next_pivot in self.manager.neighbor_tracking:
                     is_carrier = self.manager.neighbor_tracking[next_pivot].get("max_compliant_seen", 0) >= 0.02

                 if neighbor_status == "BENIGN" or is_carrier:
                      logging.info(f"[HONEYPOT DFS] 🛡️ Allowing pivot to {next_pivot} - Status is {neighbor_status} (Carrier/Bait detected: {is_carrier}) even if Reputation ({current_score:.2f}) is low.")
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

            # Schedule a timeout to force retirement if ACK doesn't arrive
            asyncio.create_task(self._honeypot_handover_timeout(30))  # 30-second timeout
            # await self.set_next_role(Role.AGGREGATOR) # Removed: Wait for ACK in engine.py
        except Exception as e:
            logging.error(f"[Honeypot] ❌ Failed to send transfer message to {candidate}: {e}")
            self._engine.has_served_as_honeypot = False

    async def _honeypot_handover_timeout(self, timeout_seconds):
        """Timeout handler: if ACK doesn't arrive within timeout, cancel transfer and stay."""
        await asyncio.sleep(timeout_seconds)

        # Check if we're still waiting
        if hasattr(self._engine, '_waiting_honeypot_handover') and self._engine._waiting_honeypot_handover:
            logging.warning(f"[Honeypot] ⏱️ ACK timeout after {timeout_seconds}s. Transfer failed.")
            logging.info("[Honeypot] 🔄 Cancelling handover and REMAINING Honeypot (Target unresponsive).")

            # Reset flag so we resume normal Honeypot duties
            self._engine._waiting_honeypot_handover = False

            # Do NOT retire. Stay as Honeypot.
            # We might want to blacklist the target we tried to pivot to, to avoid loop?
            # For now, just staying alive satisfies "no deberiamos de desaparecer".

    async def _revert_to_aggregator(self):
        """Reverts honeypot role back to aggregator after threat containment."""
        if hasattr(self._engine, "rb"):
            logging.info("[Honeypot] Transforming back to AGGREGATOR role.")
            self._engine.has_served_as_honeypot = True
            await self._engine.rb.set_next_role(Role.AGGREGATOR)

    async def update_role_needed(self):
        """
        Check if self-decommission is requested or standard update needed.
        """
        if hasattr(self, "_decommission_requested") and self._decommission_requested:
             # Set the next role to AGGREGATOR internally if not already set
             async with self._next_role_locker:
                 self._next_role = Role.AGGREGATOR

        return await super().update_role_needed()
