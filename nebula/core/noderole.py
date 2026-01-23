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
        print_msg_box(
            msg=f"Role Behavior Malicious initialization",
            indent=2,
            title="Role initialization",
        )
        self._engine = engine
        self._config = config
        logging.info("Creating attack behavior...")
        self.attack = create_attack(self._engine)
        logging.info("Attack behavior created")
        self.aggregator_bening = self._engine._aggregator
        benign_role = self._config.participant["adversarial_args"]["fake_behavior"]
        self._fake_role_behavior = factory_role_behavior(benign_role, self._engine, self._config)
        self._role = factory_node_role("malicious")

        # Pivoting Configuration
        self._attacker_pivoting = False
        self._pivot_round = 10
        if "defense_args" in self._config.participant and "honeypot" in self._config.participant["defense_args"]:
             hp_config = self._config.participant["defense_args"]["honeypot"]
             self._attacker_pivoting = hp_config.get("attacker_pivoting", False)
             self._pivot_round = hp_config.get("pivot_round", 10)
        self._pivoted = False
    
    def get_role(self):
        return self._role
        
    def get_role_name(self, effective=False):
        if effective:
            return self._fake_role_behavior.get_role_name()
        # User Change: Return just "malicious" instead of "malicious as trainer"
        return self._role.value
    
    async def extended_learning_cycle(self):     
        try:
            await self.attack.attack()
        except Exception:
            attack_name = self._config.participant["adversarial_args"]["attacks"]
            logging.exception(f"Attack {attack_name} failed")
            
        await self._fake_role_behavior.extended_learning_cycle()

        # Pivoting Logic: Pivot once if enabled
        if self._attacker_pivoting and not self._pivoted:
            # Pivot at specific round
            if self._engine.round == self._pivot_round:
                 # User Request: Jump to ANY node in the network, not just neighbors.
                 # We assume get_addrs_current_connections(only_direct=False) provides the "Known World".
                 network_nodes = await self._engine.cm.get_addrs_current_connections(only_direct=False, myself=False)
                 
                 if network_nodes:
                     target = random.choice(list(network_nodes))
                     logging.info(f"[Malicious] 🏃 Pivoting initiated at Round {self._engine.round}! Moving malicious role to {target}")
                     
                     # Ensure we are connected before sending
                     if target not in self._engine.cm.connections:
                          logging.info(f"[Malicious] Establishing connection to target {target} for pivot...")
                          await self._engine.cm.establish_connection(target)
                     
                     msg = self._engine.cm.create_message("control", "leadership_transfer")
                     msg.log = "MALICIOUS_PIVOT_TRANSFER"
                     asyncio.create_task(self._engine.cm.send_message(target, msg))
                     
                     self._pivoted = True
                     # Revert self to Benignbehavior for next round
                     if hasattr(self._engine, "rb"):
                         await self._engine.rb.set_next_role(Role.AGGREGATOR)
                         logging.info("[Malicious] Reverting to honest behavior (AGGREGATOR) after pivot.")
        
    async def select_nodes_to_wait(self):
        nodes = await self._fake_role_behavior.select_nodes_to_wait()
        return nodes
    
    async def resolve_missing_updates(self):
        return await self._fake_role_behavior.resolve_missing_updates()

"""                                                         ###############################
                                                            # TRAINER AGGREGATOR BEHAVIOR #
                                                            ###############################
"""
        
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
        await self._engine.trainer.test()
        await self._engine.trainning_in_progress_lock.acquire_async()
        await self._engine.trainer.train()
        await self._engine.trainning_in_progress_lock.release_async()

        self_update_event = UpdateReceivedEvent(
            self._engine.trainer.get_model_parameters(), self._engine.trainer.get_model_weight(), self._engine.addr, self._engine.round
        )
        await EventManager.get_instance().publish_node_event(self_update_event)

        mpe = ModelPropagationEvent(await self._engine.cm.get_addrs_current_connections(only_direct=True, myself=False), "stable")
        await EventManager.get_instance().publish_node_event(mpe)
        
        await self._engine._waiting_model_updates()
        
    async def select_nodes_to_wait(self):
        nodes = await self._engine.cm.get_addrs_current_connections(only_direct=True, myself=True)
        return nodes
    
    async def resolve_missing_updates(self):
        return {}

"""                                                         ##############################
                                                            #    AGGREGATOR BEHAVIOR     #
                                                            ##############################
"""
        
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
            
        await self._engine._waiting_model_updates()
        
        mpe = ModelPropagationEvent(await self._engine.cm.get_addrs_current_connections(only_direct=True, myself=False), "stable")
        await EventManager.get_instance().publish_node_event(mpe)
        
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
             
        self.manager = HoneyPotManager(seed=seed)
        self._defense_active = True
        self.transfer_initiated_round = -1 
        self.previous_role = None
        self.previous_honeypot_node = None # The node that gave us this role
        self.threat_persistence_counter = 0 # Track how long we have been stuck detecting a threat
        
        # Pivoting / Reactive Defense Configuration
        self._attacker_pivoting = False
        if "defense_args" in self._config.participant and "honeypot" in self._config.participant["defense_args"]:
             self._attacker_pivoting = self._config.participant["defense_args"]["honeypot"].get("attacker_pivoting", False)
        self.detected_threats = set() # Track nodes we have already flagged/deployed against
        self.honeymap_confirmed_threats = set() # Track nodes confirmed by HoneyMap

        print("""
\033[93m
      _  _
     ( \/ )
      \  /
      /  \\
     /_/\_\\
    |      |
    |______|
\033[0m
""")
        logging.info("[Honeypot] ROLE ACTIVE - Monitoring Federation...")

    def set_previous_role(self, role_name: str):
        self.previous_role = role_name

    def set_previous_honeypot_node(self, node_id: str):
        self.previous_honeypot_node = node_id
        logging.info(f"[Honeypot] Previous incumbent set to: {node_id} (Will avoid pivoting back)")

    def get_role(self):
        return self._role
    
    def get_role_name(self, effective=False):
        return self._role.value

    async def extended_learning_cycle(self):
        # 0. Check Defense Status (For Pivot Safety)
        if not self._defense_active:
             # Timeout Logic: Allow 2 full rounds grace period for ACK to arrive and be processed due to potential sync delays.
             # Only timeout if we are 3 rounds past initialization.
             # Example: Init R3. Current R4 (Passive). Current R5 (Passive, wait for late ACK). Current R6 (Timeout!).
             if self.transfer_initiated_round != -1 and self.transfer_initiated_round < (self._engine.round - 2):
                 logging.warning(f"[Honeypot] ⚠️ Transfer Timeout! No ACK received from candidate since round {self.transfer_initiated_round}. Aborting transfer and re-enabling defense.")
                 self._defense_active = True
                 self.transfer_initiated_round = -1
                 # Fail safe: continue execution as Honeypot this round to retry pivot later
             else:
                 logging.info(f"[Honeypot] 🛡️ Defense Inactive (Pivot in progress, initiated Round {self.transfer_initiated_round}). Behaving as Passive Trainer.")
                 await super().extended_learning_cycle() # Use standard Trainer behavior (inherited)
                 return

        # Reactive Defense: Check for Pivoting Attackers and spawn counters
        if self._attacker_pivoting:
            self._check_and_react_to_pivot()

        # 1. Update Defense Strategy (HoneyMap)
        self.manager.new_round()
        logging.info(f"[Honeypot] Round {self._engine.round} | HoneyMap Active.")

        # 2. Poison Local Data for Training (HoneyDoor)
        original_train_set = self._engine.trainer.datamodule.train_set
        self._engine.trainer.datamodule.train_set = self.manager.get_dataset(original_train_set)
        
        # 3. Standard Training Cycle (Test -> Train -> Publish)
        await self._engine.trainer.test()
        
        # [MODIFIED] Wait for updates (Aggregator Behavior) to sync with others
        # This makes the Honeypot wait for the rest of the network, slowing it down to match the round pace.
        await self._engine._waiting_model_updates()
        
        # BACKUP CLEAN MODEL (Weights before poisoning)
        # Note: We now have the AGGREGATED model from _waiting_model_updates in self._engine.trainer.model
        clean_model_state = copy.deepcopy(self._engine.trainer.get_model_parameters())

        await self._engine.trainning_in_progress_lock.acquire_async()
        
        # Aggressive Injection
        # Increase LR to ensure the HoneyDoor is learned strongly
        original_lr = self._config.participant["training_args"].get("lr", 0.01)
        boosted_lr = original_lr * 5.0
        logging.info(f"[Honeypot] 🚀 Boosting Learning Rate to {boosted_lr:.4f} (x5) for HoneyDoor injection.")
        self._engine.trainer.update_model_learning_rate(boosted_lr)
        
        try:
            await self._engine.trainer.train()
        finally:
            # Restore original LR
            self._engine.trainer.update_model_learning_rate(original_lr)
            logging.info(f"[Honeypot] Restored Learning Rate to {original_lr:.4f}.")
            
        # Restore Clean Data
        self._engine.trainer.datamodule.train_set = original_train_set
        await self._engine.trainning_in_progress_lock.release_async()

        # Publish Update (Poisoned)
        self_update_event = UpdateReceivedEvent(
            self._engine.trainer.get_model_parameters(), self._engine.trainer.get_model_weight(), self._engine.addr, self._engine.round
        )
        await EventManager.get_instance().publish_node_event(self_update_event)

        # DECEPTION & ISOLATION STRATEGY
        # Instead of broadcasting blindly, we curate the recipients.
        # 1. Honest Neighbors: Receive the HONEYDOOR (Poisoned/Marked) model to verify them.
        # 2. Malicious Node (If identified): Receives a DECEPTIVE/PLACEBO model.
        #    This keeps the attacker happy (connection open) but feeds them junk or reflects their own poison.
        
        all_neighbors = await self._engine.cm.get_addrs_current_connections(only_direct=True, myself=False)
        honest_neighbors = set(all_neighbors)
        
        # Identify known malicious node (from previous round logic or if we track it)
        # We need to access the state found in "Analyzing neighbor responses" but that happens AFTER update.
        # However, we can use 'self.threat_persistence_counter' and 'previous_honeypot_node' 
        # or store the identified threat in 'self' during step 5 for use in step 3 of next round.
    
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
        return score

    def _check_and_react_to_pivot(self):
        """Monitors reputation to detect pivots and spawn new honeypots."""
        if hasattr(self._engine, "_reputation"):
             scores = self._engine._reputation.get_reputation_table()
             logging.info(f"[Honeypot] 🔍 Threat Scan | Current Reputation Scores: {scores}")
             
            # ---------------------------------------------------------
             # PHASE 2 Check: Are we already ENGAGED with a local threat?
             # ---------------------------------------------------------
             neighbors = []
             try:
                 # Get direct connections to check proximity
                 conn_objs = self._engine.cm.connections
                 neighbors = list(conn_objs.keys()) # Format: 'IP:Port' strings
                 if not neighbors:
                     # Fallback if dictionary keys are not strings (should not happen in this version)
                     pass
             except:
                 pass
             
             # Identify threats that are practically touching us (Neighbors)
             local_confirmed_threats = [
                 t for t in self.honeymap_confirmed_threats 
                 # Simple substring match or exact match depending on format
                 if any(t in n or n in t for n in neighbors)
             ]
             
             is_locally_engaged = len(local_confirmed_threats) > 0

             if is_locally_engaged:
                 logging.info(f"[Honeypot] 🛡️ ENGAGED MODE: Holding position against local threats {local_confirmed_threats}.")
                 
                 # -----------------------------------------------------
                 # LOGIC: CLONING (Deploying Scouts against Remote Threats)
                 # -----------------------------------------------------
                 # Look for threats that are CONFIRMED but NOT local
                 remote_threats = [
                     t for t in self.honeymap_confirmed_threats 
                     if t not in local_confirmed_threats
                 ]
                 
                 for remote_suspect in remote_threats:
                     # Check if we already have a clone covering this
                     if not hasattr(self, "deployed_clones_targets"):
                         self.deployed_clones_targets = set()
                         
                     if remote_suspect not in self.deployed_clones_targets:
                         logging.warning(f"[Honeypot] 🧬 MULTI-THREAT DETECTED! Engaged locally, but {remote_suspect} is active elsewhere. SPAWNING CLONE.")
                         self.deployed_clones_targets.add(remote_suspect)
                         asyncio.create_task(self._deploy_honeypot_agent(remote_suspect))
                         # Limit to one clone per cycle to avoid flooding
                         break 
                 
                 return # Stop here, do not attempt to Move/Pivot self if engaged

             # ---------------------------------------------------------
             # PHASE 1: PIVOTING (Searching/Hunting)
             # If we are NOT engaged, we should move to a better location.
             # ---------------------------------------------------------
             
             # 1. Check for New Threats (Low Reputation AND HoneyMap Confirmed)
             # A node is a threat ONLY if it has Low Reputation AND triggers the HoneyMap.
             suspects = [
                 node for node, score in scores.items() 
                 if score < 0.4 and node in self.honeymap_confirmed_threats
             ]
             
             # If we see a confirmed threat and we are NOT engaged (meaning it's not a neighbor yet),
             # we likely need to move closer to it (Strategic Pivot)
             
             # Get Feed for Strategic Pivot
             suspects_feed = []
             if hasattr(self._engine._reputation, "get_suspects_from_feedback"):
                 suspects_feed = self._engine._reputation.get_suspects_from_feedback(threshold=0.5)

             # Calculate Best Move
             try:
                 # Note: neighbors was retrieved above as strings. We might need wait/async for fresh list.
                 # But we can use the existing 'neighbors' list for calculation
                 if not neighbors:
                     return

                 best_candidate = None
                 max_score = -1.0
                 
                 for neighbor in neighbors:
                     # Skip previous incumbent to avoid loops
                     if self.previous_honeypot_node and neighbor == self.previous_honeypot_node:
                         continue
 
                     # Get basic reputation
                     rep = scores.get(neighbor, 0.5)
                     
                     # Filter out blacklisted/zero rep nodes for safety
                     if rep > 0.0:
                         score = self._calculate_pivot_candidate_score(neighbor, rep, suspects_feed)
                         
                         if score > max_score:
                             max_score = score
                             best_candidate = neighbor
                 
                 # TRIGGER PIVOT
                 if best_candidate and max_score > 0.0:
                     # ONLY Pivot if the score indicates a valid move (e.g. not just random low rep)
                     # Or if we have a distant suspect we are trying to reach
                     should_move = False
                     
                     # Condition A: Strategic Pivot (We are chasing someone)
                     if max_score > 1.0: # Bonus was applied
                         should_move = True
                         logging.info(f"[Honeypot] 🧭 STRATEGIC PIVOT initiated towards {best_candidate} (Score {max_score:.2f})")
                     
                     # Condition B: Random Patrol (If enabled or if stuck)
                     # (Optional, can be added later. For now, we stick to reactive)
                     
                     if should_move:
                         logging.info(f"[Honeypot] 👋 Transferring Role to {best_candidate} to hunt threats.")
                         
                         target_state = self.manager.export_state()
                         payload = json.dumps(target_state)
                         log_message = f"HONEYPOT_TRANSFER:{payload}"
                         
                         msg = self._engine.cm.create_message("control", "leadership_transfer", log=log_message)
                         asyncio.create_task(self._engine.cm.send_message(best_candidate, msg))
                         
                         # Deactivate Defense locally (Move)
                         self._defense_active = False 
                         # We do not revert immediately here, we wait for ACK or timeout in the main loop
                         # But effectively we stop acting as Honeypot
                         self.transfer_initiated_round = self._engine.round

             except Exception as e:
                 logging.error(f"[Honeypot] Pivot Analysis Error: {e}")

             # 2. Check for Recovery/Revert (Reputation AND HoneyMap check)
             # User Requirement: "si el honeypot ... detecta que ya no tiene amenaza ... entronces si se debe de revertir"
             
             neighbors = list(scores.keys())
             if neighbors:
                 avg_rep = sum(scores.values()) / len(scores)
                 
                 # Condition 1: Reputation is recovered (No active threats in view)
                 # Using 0.8 as safe threshold
                 rep_safe = all(s > 0.8 for s in scores.values())
                 
                 # Condition 2: HoneyMap allows it (Implicitly managed via state, but we ensure no local alerts)
                 # We also check if we have recently deployed a scout (don't revert instantly if we just detected something)
                 
                 if rep_safe and self._engine.round > 5:
                      logging.info(f"[Honeypot] 🕊️ Mission Accomplished? All neighbors have high reputation (Avg: {avg_rep:.2f}). Retiring/Disappearing...")
                      self._defense_active = False # Disable defense mechanisms
                      asyncio.create_task(self._revert_to_aggregator())

    async def _revert_to_aggregator(self):
        if hasattr(self._engine, "rb"):
             logging.info("[Honeypot] Transforming back to AGGREGATOR role.")
             await self._engine.rb.set_next_role(Role.AGGREGATOR)

    async def _deploy_honeypot_agent(self, target_suspect, deploy_to=None, exclude_nodes=None):
        """Pivots the Honeypot Agent towards a detected threat."""
        try:
            logging.info(f"[Honeypot] _deploy executing for suspect {target_suspect}...")
            
            # Use timeout to detect deadlocks
            try:
                neighbors = await asyncio.wait_for(
                    self._engine.cm.get_addrs_current_connections(only_direct=False, myself=False),
                    timeout=2.0
                )
            except asyncio.TimeoutError:
                logging.error("[Honeypot] TIMEOUT getting connections! Deadlock detected in CM?")
                return

            candidates = list(neighbors)
            logging.info(f"[Honeypot] Candidates: {candidates}")
        
            # Apply exclusions (e.g., old threat locations)
            if exclude_nodes:
                candidates = [c for c in candidates if c not in exclude_nodes]

            # 1. OPTIMAL STATE: If the threat is already a neighbor, we are in position.
            # We STOP pivoting and maintain the position to engage the threat.
            if target_suspect in candidates:
                logging.info(f"[Honeypot] 🎯 Threat {target_suspect} is a neighbor! Holding position to engage.")
                return

            # 2. PIVOT: If threat is not reachable, move through the network.
            target = None
            if deploy_to and deploy_to in candidates:
                target = deploy_to
                logging.info(f"[Honeypot] 🎯 Pivoting to specific target {target} (Chasing suspected pivot).")
            else:
                # INTELLIGENT PIVOT: Move towards the neighbor that reported the threat
                if hasattr(self._engine, "_reputation") and hasattr(self._engine._reputation, "get_reporters"):
                     reporters = self._engine._reputation.get_reporters(target_suspect)
                     # Filter reporters that are current neighbors
                     valid_reporters = [r for r in reporters if r in candidates]
                     
                     if valid_reporters:
                         target = random.choice(valid_reporters)
                         logging.info(f"[Honeypot] 🧭 INTELLIGENT PIVOT: Moving towards informant {target} who reported threat {target_suspect}.")
                     else:
                         logging.info(f"[Honeypot] Informants {reporters} not in neighbors {candidates}.")
                         # logging.warning(f"[Honeypot] Could not find specific informant neighbor for {target_suspect}. Fallback to Random.")
                         
                if not target and candidates:
                     # Fallback: Randomly select a node in the network to move to
                     target = random.choice(candidates)
                     logging.info(f"[Honeypot] 🔄 Pivoting to RANDOM neighbor {target} to search for threat {target_suspect}.")
            
            if not target:
                logging.warning("[Honeypot] Could not pivot - no connections available.")
                return

            # Prepare state first to include in the message creation
            state = self.manager.export_state()
            
            # Flag this transfer as a SCOUT/PIVOT deployment
            state["scout_mission"] = True
            state["target_suspect"] = target_suspect
            
            # 3. ACTION DECISION: CLONE vs MOVE
            # If we are currently "busy" engaging a local threat (we have suspects nearby), we should STAY here and SPAWN a clone.
            # If we are "idle" (patrolling), we should MOVE (pivot) ourselves.
            
            is_engaged_locally = False
            # Check if we have any active local suspects in our reputation table
            if hasattr(self._engine, "_reputation"):
                 scores = self._engine._reputation.get_reputation_table()
                 local_suspects = [node for node, score in scores.items() if score < 0.4 and node in candidates]
                 if local_suspects:
                     is_engaged_locally = True
                     logging.info(f"[Honeypot] 🛡️ Currently engaging local threats {local_suspects}. Will SPAWN a clone instead of moving.")

            encoded_log = f"HONEYPOT_TRANSFER:{json.dumps(state)}"
            msg = self._engine.cm.create_message("control", "leadership_transfer", log=encoded_log)
            
            await self._engine.cm.send_message(target, msg)

            # 4. POST-ACTION: Revert self ONLY if we moved (didn't clone)
            if not is_engaged_locally:
                logging.info("[Honeypot] 👋 Pivot/Move initiated. Reverting self to AGGREGATOR.")
                await self._revert_to_aggregator()
            else:
                logging.info("[Honeypot] 🧬 Clone spawned to chase remote threat. Main node holding position.")
        except Exception as e:
            logging.error(f"[Honeypot] CRASH DETECTED in _deploy_honeypot_agent: {e}")
            import traceback
            logging.error(traceback.format_exc())


        known_threat = getattr(self, "detected_threat_node_persistent", None)
        
        if known_threat and known_threat in honest_neighbors:
             logging.info(f"[Honeypot] 🎭 DECEPTION ACTIVE: Sending PLACEBO model to Attacker {known_threat}")
             honest_neighbors.remove(known_threat)
             
             # Send Deceptive Model (Placebo)
             # What is the placebo?
             # Option A: The clean model (without HoneyDoor) -> They don't learn our trap, but learn the task.
             # Option B: Their own previous model (Status Quo) -> They stagnate.
             # Option C: Random/Noisy model -> They diverge.
             # Selected IMPROVED: Send a "Reflective" model (The Attacker's own poison).
             # We send back the model they sent us in the previous round (if captured).
             # This convinces them that their poisoning was successful and aggregated into the global model.
             
             last_threat_model = getattr(self, "last_threat_model", None)
             import torch
             
             if last_threat_model:
                 logging.info(f"[Honeypot] 🪞 MIRROR DECEPTION: Sending Attacker's OWN poison back to them.")
                 deceptive_params = last_threat_model 
             else:
                 # Fallback if we haven't captured it yet: Clean + Noise
                 logging.info(f"[Honeypot] 🎭 NOISE DECEPTION: Sending Noisy model (fallback).")
                 deceptive_params = copy.deepcopy(clean_model_state)
                 for key in deceptive_params:
                     if isinstance(deceptive_params[key], torch.Tensor):
                         noise = torch.randn_like(deceptive_params[key]) * 0.5 
                         deceptive_params[key] += noise

             deceptive_payload = self.trainer.serialize_model(deceptive_params)
             deceptive_msg = self._engine.cm.create_message(
                 "model", "", self._engine.round, deceptive_payload, self._engine.trainer.get_model_weight()
             )
             await self._engine.cm.send_message(known_threat, deceptive_msg)

        mpe = ModelPropagationEvent(list(honest_neighbors), "stable")
        await EventManager.get_instance().publish_node_event(mpe)
        
        # RESTORE CLEAN MODEL (Wipe local poisoning imprint)
        self._engine.trainer.set_model_parameters(clean_model_state)
        logging.info("[Honeypot] Restored clean model weights to avoid self-contamination for next rounds.")
        
        # 4. Wait for Updates (Standard DFL behavior) - SYNCHRONIZED
        # Ensure we are waiting for current neighbors (sync check)
        try:
            nodes_to_wait = await self._engine.cm.get_addrs_current_connections(only_direct=True, myself=True)
            if len(nodes_to_wait) > 1:
                # Update aggregator expectation 
                await self._engine.aggregator.update_federation_nodes(nodes_to_wait)
                
                # SELF-SYNCHRONIZATION LOOP
                # We pollingly wait until we see updates from neighbors in the storage.
                # This prevents the Honeypot (fast training) from timing out while neighbors are still training.
                waiting_start = time.time()
                min_neighbors_wait = max(1, len(nodes_to_wait) - 1) # Wait for all neighbors (excluding self)
                max_sync_wait = 300 # 5 minutes max safety break
                
                logging.info(f"[Honeypot] ⏳ Syncing... Waiting for {min_neighbors_wait} neighbor udpates before aggregation.")
                
                while time.time() - waiting_start < max_sync_wait:
                    updates_storage = self._engine.aggregator.us.us
                    # Count updates from this round from neighbors
                    neighbor_updates_count = 0
                    current_round = self._engine.round
                    
                    for node_id, update_tuple in updates_storage.items():
                        if node_id != self._engine.addr:
                            # Verify round coherence if wrapper/tuple is used
                            # DFLUpdateHandler stores tuple(Update, deque)
                            update_obj = update_tuple[0]
                            if update_obj.round == current_round:
                                neighbor_updates_count += 1
                                
                    if neighbor_updates_count >= min_neighbors_wait:
                        logging.info(f"[Honeypot] ⚡ Sync Complete. Received {neighbor_updates_count}/{min_neighbors_wait} updates.")
                        break
                    
                    await asyncio.sleep(2) # Poll every 2s
                    
        except Exception as e:
            logging.warning(f"[Honeypot] Error during sync wait: {e}")

        await self._engine._waiting_model_updates()
        
        # 5. HONEYPOT ANALYSIS: Inspect Neighbors' Updates
        logging.info("[Honeypot] Analyzing neighbor responses for malicious patterns...")
        threat_detected = False
        detected_threat_node = None
        detected_echo_node = None
        detected_threat_rep = 0.0
        

        try:
            # 5.1 Prepare Validation Data (Clean sample from val set)
            # Fix: Ensure DataModule is initialized for validation to prevent "Validation dataset not initialized" error
            try:
                self._engine.trainer.datamodule.setup("fit")
            except Exception as setup_err:
                logging.warning(f"[Honeypot] Warning during datamodule setup: {setup_err}")

            val_loader = self._engine.trainer.datamodule.val_dataloader()
            clean_batch = None
            # Get one batch carefully
            if hasattr(val_loader, '__iter__'):
                clean_batch = next(iter(val_loader))
            
            if clean_batch:
                # 5.2 Access Updates from Aggregator Storage
                updates_storage = self._engine.aggregator.us.us
                
                # Backup current model state
                current_params = self._engine.trainer.get_model_parameters()

                for node_id, update_tuple in updates_storage.items():
                    update_obj = update_tuple[0] # The Update object
                    
                    # Skip self
                    if node_id == self._engine.addr:
                        continue
                        
                    # Load neighbor parameters into model
                    if update_obj.model:
                        self._engine.trainer.set_model_parameters(update_obj.model)
                        
                        # Verify using HoneyManager
                        is_malicious = self.manager.verify_model(self._engine.trainer.model, clean_batch)
                        
                        if not is_malicious:
                            if node_id in self.honeymap_confirmed_threats:
                                logging.info(f"[Honeypot] Node {node_id} tested CLEAN on HoneyMap. Removing from confirmed threats.")
                                self.honeymap_confirmed_threats.remove(node_id)
                        
                        if is_malicious:
                            # DISCRIMINATE: Attacker vs Echo
                            # Heuristic: Echo nodes (Victims) usually have High Reputation.
                            # Attackers usually have Low Reputation or are new.
                            reputation_system = getattr(self._engine, "_reputation", None)
                            current_rep = 0.5
                            if reputation_system:
                                rep_table = reputation_system.get_reputation_table()
                                current_rep = rep_table.get(node_id, 0.5)

                            # Threshold lowered to 0.4 to capture "New/Neutral" nodes as potential Echos 
                            # instead of branding them Malicious immediately.
                            if current_rep > 0.4 or self._engine.round < 2:
                                logging.warning(f"[Honeypot] ⚠️ DETECTED ECHO/SUSPECT NODE: {node_id} (Matched HoneyMap, Rep {current_rep:.2f}). Identified as Potential Victim/Echo.")
                                detected_echo_node = node_id
                                # Do NOT nuke reputation yet. Allow Pivot to confirm.
                            else:
                                logging.critical(f"\033[91m[Honeypot] 🚨 MALICIOUS NODE DETECTED: {node_id} (Matched HoneyMap Pattern) 🚨\033[0m")
                                threat_detected = True
                                detected_threat_node = node_id
                                self.honeymap_confirmed_threats.add(node_id)
                                
                                # Store confirmed threat for future pardon/recovery if it becomes clean
                                self.last_confirmed_threat = node_id
                                
                                # CAPTURE THREAT MODEL FOR MIRROR DECEPTION
                                # We treat their successful attack model as the "Ideal Placebo" to reflect back to them.
                                self.last_threat_model = copy.deepcopy(update_obj.model)
                                
                                # Penalize Reputation to 0 only for confirmed threats
                                detected_threat_rep = current_rep 
                                if reputation_system:
                                    logging.info(f"[Honeypot] Penalizing Node {node_id}. Old Rep: {current_rep}")
                                    reputation_system.manual_update(node_id, 0.0)
                                
                                # Broadcast Warning to Neighbors (Gossip) - EXCLUDING THE THREAT
                                try:
                                    # Get all current neighbors
                                    all_current_neighbors = await self._engine.cm.get_addrs_current_connections(only_direct=True, myself=False)
                                    # Filter out the threat
                                    honest_targets = [n for n in all_current_neighbors if n != node_id]
                                    
                                    logging.warning(f"[Honeypot] 📢 BROADCASTING WARNING to {len(honest_targets)} honest neighbors. Excluding {node_id}.")
                                    
                                    for neighbor in honest_targets:
                                        # Send explicit Reputation Update / Warning
                                        # Using 'reputation' type message with 'share' action as per reputation module protocol
                                        msg = self._engine.cm.create_message(
                                            "reputation",
                                            "share",
                                            node_id=node_id, # The node being reported
                                            score=0.0,       # The score (malicious)
                                            round=self._engine.round
                                        )
                                        await self._engine.cm.send_message(neighbor, msg)
                                    
                                except Exception as warn_err:
                                    logging.error(f"[Honeypot] Warning broadcast failed: {warn_err}")
                                    
                        else:
                            logging.info(f"[Honeypot] Node {node_id} response appears legitimate.")
                            # Boost Reputation
                            reputation_system = getattr(self._engine, "_reputation", None)
                            if reputation_system:
                                reputation_system.manual_update(node_id, 1.05) # Small boost

                # Restore Model
                self._engine.trainer.set_model_parameters(current_params)
            else:
                logging.warning("[Honeypot] Could not load validation batch for analysis.")

        except Exception as e:
            logging.error(f"[Honeypot] Analysis failed: {e}")
            
        # 🕸️ GLOBAL THREAT INTELLIGENCE (Reputation Scan)
        # Check global reputation table for sudden drops elsewhere in the topology
        # This signals a new threat emerging in a different neighborhood.
        detected_pivoting_target_area = None
        if self._attacker_pivoting: # Only strict check if defense is active
             reputation_system = getattr(self._engine, "_reputation", None)
             if reputation_system:
                 current_scores = reputation_system.get_reputation_table()
                 # Look for nodes that are NOT my neighbors, NOT the old threat, but have LOW reputation (< 0.4)
                 # This implies someone else is reporting them as malicious.
                 my_neighbors = await self._engine.cm.get_addrs_current_connections(only_direct=True, myself=False)
                 
                 for node, score in current_scores.items():
                     if node not in my_neighbors and node != self.previous_honeypot_node and score < 0.4:
                         logging.warning(f"[Honeypot] 🛰️ GLOBAL INTEL: Detected Low Reputation Cluster at {node} (Score {score:.2f}). Possible Pivot Destination.")
                         detected_pivoting_target_area = node
                         break


        # 6. PIVOT STRATEGY (Role Transfer)
        # Condition: STOP pivoting if a threat is detected to maintain surveillance
        # Logic: If we detect a threat, we usually want to stay.
        # BUT, if we stay too long (Stagnation), it might be an Echo (False Positive).
        # We must allow pivoting after N rounds of persistent detection to explore/triangulate.
        target_threat_node = None
        
        # Initialize counter if not exists
        if not hasattr(self, "clean_rounds_counter"):
            self.clean_rounds_counter = 0

        if threat_detected:
            self.threat_persistence_counter += 1
            self.clean_rounds_counter = 0 # Reset clean counter
            
            # FAST PIVOT UPDATE: Removed 2-round hold to speed up tracking.
            logging.warning(f"[Honeypot] ⚠️ Threat Detected ({self.threat_persistence_counter} rounds). Triggering immediate pivot for faster response.")
            target_threat_node = detected_threat_node
            
            # Persist the threat ID for next round's DECEPTION logic (Placebo sending)
            self.detected_threat_node_persistent = detected_threat_node
            
        else:
            self.threat_persistence_counter = 0
            
            # NEW LOGIC HERE: Check for Pivot via Reputation Drop
            # If we see a global reputation drop in a non-neighbor, it means someone else is under attack.
            if self._attacker_pivoting and detected_pivoting_target_area:
                 logging.info(f"[Honeypot] 🕵️‍♂️ Global Reputation Alert! Node {detected_pivoting_target_area} has low reputation. Suspecting Pivot Attack there.")
                 
                 # Deploy scout in that direction (if possible) or randomly to find path
                 # Since we cannot 'teleport', we deploy to a random neighbor to continue the search/spread defense.
                 to_exclude = []
                 if getattr(self, "detected_threat_node_persistent", None):
                      to_exclude.append(self.detected_threat_node_persistent)
                 
                 asyncio.create_task(self._deploy_honeypot_agent(target_suspect=None, exclude_nodes=to_exclude))
                 
                 # Clear persistence
                 self.detected_threat_node_persistent = None
                 
                 # Ensure we don't retire immediately
                 self.clean_rounds_counter = 0 
            
            # CHECK FOR MISSION COMPLETION (The "Finished" state)
            # If we don't detect threats for N rounds, and we are not tracking an Echo,
            # we assume the network is clean or the attacker has stopped.
            if not detected_echo_node and not detected_pivoting_target_area:
                if threat_detected:
                     self.clean_rounds_counter = 0 # Reset if local threat found
                else: 
                     self.clean_rounds_counter += 1
                     logging.info(f"[Honeypot] No threats detected. Clean streak: {self.clean_rounds_counter} rounds.")

                if self.clean_rounds_counter >= 3:
                     logging.info(f"[Honeypot] ✅ MISSION ACCOMPLISHED: No threats detected for {self.clean_rounds_counter} rounds.")
                     
                     # RECOVERY PROTOCOL: Pardon old threats
                     if hasattr(self, "last_confirmed_threat") and self.last_confirmed_threat:
                         try:
                             old_threat = self.last_confirmed_threat
                             all_neighbors = await self._engine.cm.get_addrs_current_connections(only_direct=True, myself=False)
                             # Filter neighbors (excluding the threat itself, although notifying itself doesn't hurt)
                             targets = [n for n in all_neighbors if n != old_threat]
                             
                             logging.info(f"[Honeypot] 🕊️ PARDONING: Broadcasting CLEAN status for old threat {old_threat} to neighbors.")
                             
                             for neighbor in targets:
                                 # Send High Reputation score (1.0) to lift Shadow Ban
                                 msg = self._engine.cm.create_message(
                                     "reputation",
                                     "share",
                                     node_id=old_threat,
                                     score=1.0, 
                                     round=self._engine.round
                                 )
                                 await self._engine.cm.send_message(neighbor, msg)
                             
                             # Clear local tracking
                             self.last_confirmed_threat = None
                         except Exception as recover_err:
                             logging.error(f"[Honeypot] Recovery broadcast failed: {recover_err}")
                     
                     # Check if we should really retire or if we still have suspicious activity nearby
                     # SAFETY: Only retire if we are absolutely sure.
                     logging.info("[Honeypot] Decommissioning Honeypot Role -> Transforming to AGGREGATOR.")
                     
                     # Self-demotion to AGGREGATOR
                     # Trigger Role Update in Engine
                     # 1. Set flag for Engine to pick up
                     self._decommission_requested = True
                     return # End cycle immediately

            # If we found an Echo but no Attacker, use the Echo as the pivot target
            if detected_echo_node:
                self.clean_rounds_counter = 0 # Reset clean counter if we find Echos (still work to do)
                logging.info(f"[Honeypot] No active Attacker found, but detected Echo Node {detected_echo_node}. Setting as pivot target to clean/investigate.")
                target_threat_node = detected_echo_node

        try:
            logging.info("[Honeypot] Calculating Pivot Strategy for next round...")
            neighbors = await self._engine.cm.get_addrs_current_connections(only_direct=True, myself=False)
            
            # Register current location to prevent immediate return
            self.manager.register_visit(self._engine.addr)
            
            best_candidate = None
            max_score = -1.0
            
            reputation_system = getattr(self._engine, "_reputation", None)
            
            suspects_feed = []
            rep_table = {}
            
            if reputation_system:
                rep_table = reputation_system.get_reputation_table()
                if hasattr(reputation_system, "get_suspects_from_feedback"):
                    suspects_feed = reputation_system.get_suspects_from_feedback(threshold=0.5)

            # AGGRESSIVE HUNTING: Target the suspected node directly to investigate/traverse
            # SAFETY CHECK: We MUST NOT jump to the actual Attacker.
            # We discriminante between "Attacker" and "Echo" (Victim) by checking if the suspect 
            # is itself reporting someone else (indicating it is analyzing neighbors and finding threats).
            if target_threat_node and target_threat_node in neighbors:
                 # ANTI-PING-PONG CHECK
                 if self.previous_honeypot_node and target_threat_node == self.previous_honeypot_node:
                     logging.warning(f"[Honeypot] ⚠️ Detected Pivot Ping-Pong! Suspect {target_threat_node} is the previous Honeypot. Ignoring aggressive hunt to avoid infinite loop.")
                     target_threat_node = None # Disable aggressive target, fall back to reputation
                 else:
                     is_echo_victim = False
                     for reporter, suspect, score in suspects_feed:
                         if reporter == target_threat_node:
                             logging.info(f"[Honeypot] Analysis: Suspect {target_threat_node} is reporting {suspect} as malicious. It behaves like an Honest Victim (Echo).")
                             is_echo_victim = True
                             break
                     
                     if is_echo_victim:
                         logging.info(f"[Honeypot] 🎯 Aggressive Hunting: Identifying {target_threat_node} as ECHO (Active Reporter). Pivoting TO it to reach real source.")
                         best_candidate = target_threat_node
                         max_score = 999.0
                     elif detected_threat_rep > 0.6:
                         logging.info(f"[Honeypot] 🎯 Aggressive Hunting: Identifying {target_threat_node} as ECHO (High Reputation History {detected_threat_rep:.2f}). Pivoting TO it.")
                         best_candidate = target_threat_node
                         max_score = 999.0
                     elif not threat_detected:
                         # CASE: detected_echo_node (Green Light)
                         # It was classified as an Echo earlier (due to Rep > 0.4 or Early Round), so we allow pivot even if behavior is suspiciously silent.
                         logging.warning(f"[Honeypot] ⚠️ Suspect {target_threat_node} classified as POTENTIAL ECHO (Rep {detected_threat_rep:.2f}). Pivoting to investigate.")
                         best_candidate = target_threat_node
                         max_score = 999.0
                     else:
                         # CASE: detected_threat_node (Red Light)
                         # It is a CONFIRMED THREAT (Low Rep, Late Round) and is not reporting anyone.
                         # This implies it is likely the SOURCE (Node 0). PIVOTING WOULD BE SUICIDE.
                         logging.critical(f"[Honeypot] ⛔ SAFETY STOP: Suspect {target_threat_node} is NOT reporting others, has LOW history ({detected_threat_rep:.2f}) and is a CONFIRMED THREAT. Likely the COMPROMISED SOURCE. Cannot Pivot.")
                         self._defense_active = True
                         return 
            
            # Only run standard election if we haven't already forced a target
            if not best_candidate:
                
                for neighbor in neighbors:
                    # Skip previous incumbent to avoid loops
                    if self.previous_honeypot_node and neighbor == self.previous_honeypot_node:
                        continue

                    # Get basic reputation
                    rep = rep_table.get(neighbor, 0.5)
                    
                    # Filter out blacklisted/zero rep nodes for safety
                    if rep > 0.0:
                        score = self._calculate_pivot_candidate_score(neighbor, rep, suspects_feed)
                        
                        if score > max_score:
                            max_score = score
                            best_candidate = neighbor
            
            # Fallback: If no candidate selected (e.g. all 0.0 or empty table), pick random neighbor
            if not best_candidate and neighbors:
                best_candidate = random.choice(list(neighbors))
                logging.info(f"[Honeypot] No high-rep candidate found. Fallback to random neighbor: {best_candidate}")

            if best_candidate:
                logging.info(f"[Honeypot] 🛡️ Selected Pivot Candidate: {best_candidate} (Score: {max_score:.2f})")
                
                # Create Remote Control Message for Role Transfer
                target_state = self.manager.export_state()
                payload = json.dumps(target_state)
                log_message = f"HONEYPOT_TRANSFER:{payload}"
                
                msg = self._engine.cm.create_message(
                    "control", 
                    "LEADERSHIP_TRANSFER",
                    log=log_message
                )
                await self._engine.cm.send_message(best_candidate, msg)
                logging.info(f"[Honeypot] Role Transfer Message Sent -> {best_candidate}")
                
                # Deactivate local defense after transfer
                self._defense_active = False 
                self.transfer_initiated_round = self._engine.round
                logging.info(f"[Honeypot] Waiting for transfer acceptance from {best_candidate}. Timeout check set for Round {self._engine.round + 2}.")

                # Wait for ACK to change role. 
                # The engine handles LEADERSHIP_TRANSFER_ACK and sets the next role to TRAINER.
                logging.info(f"[Honeypot] Waiting for transfer acceptance from {best_candidate}...")

            else:
                 logging.info("[Honeypot] No suitable pivot candidate found (No High Rep neighbors). Maintaining position.")
                 self._defense_active = True

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

    async def update_role_needed(self):
        """
        Check if self-decommission is requested or standard update needed.
        """
        if hasattr(self, "_decommission_requested") and self._decommission_requested:
             # Set the next role to AGGREGATOR internally if not already set
             async with self._next_role_locker:
                 self._next_role = Role.AGGREGATOR
             
        return await super().update_role_needed()






