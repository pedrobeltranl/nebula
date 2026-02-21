import logging
import random
import time
import numpy as np
import torch
import asyncio

from datetime import datetime
from typing import TYPE_CHECKING
from nebula.addons.functions import print_msg_box
from nebula.core.eventmanager import EventManager
from nebula.core.nebulaevents import AggregationEvent, RoundStartEvent, UpdateReceivedEvent, DuplicatedMessageEvent, UpdateNeighborEvent
from nebula.core.utils.helper import (
    cosine_metric,
    euclidean_metric,
    jaccard_metric,
    manhattan_metric,
    minkowski_metric,
    pearson_correlation_metric,
)

if TYPE_CHECKING:
    from nebula.config.config import Config
    from nebula.core.engine import Engine

class Metrics:
    def __init__(
        self,
        num_round=None,
        current_round=None,
        fraction_changed=None,
        threshold=None,
        latency=None,
    ):
        self.fraction_of_params_changed = {
            "fraction_changed": fraction_changed,
            "threshold": threshold,
            "round": num_round,
        }

        self.model_arrival_latency = {"latency": latency, "round": num_round, "round_received": current_round}

        self.messages = []

        self.similarity = []


class Reputation:
    """
    Class to define and manage the reputation of a participant in the network.
    """

    REPUTATION_THRESHOLD = 0.6
    SIMILARITY_THRESHOLD = 0.6
    INITIAL_ROUND_FOR_REPUTATION = 1
    INITIAL_ROUND_FOR_FRACTION = 1
    HISTORY_ROUNDS_LOOKBACK = 4
    WEIGHTED_HISTORY_ROUNDS = 3
    FRACTION_ANOMALY_MULTIPLIER = 1.20
    THRESHOLD_ANOMALY_MULTIPLIER = 1.15

    # Augmentation factors
    LATENCY_AUGMENT_FACTOR = 1.4
    MESSAGE_AUGMENT_FACTOR_EARLY = 2.0
    MESSAGE_AUGMENT_FACTOR_NORMAL = 1.1

    # Penalty and decay factors
    HISTORICAL_PENALTY_THRESHOLD = 0.9
    NEGATIVE_LATENCY_PENALTY = 0.3
    CURRENT_VALUE_WEIGHT_HIGH = 0.9
    CURRENT_VALUE_WEIGHT_LOW = 0.2
    PAST_VALUE_WEIGHT_HIGH = 0.8
    PAST_VALUE_WEIGHT_LOW = 0.1
    ZERO_VALUE_DECAY_FACTOR = 0.1
    REPUTATION_CURRENT_WEIGHT = 0.9
    REPUTATION_FEEDBACK_WEIGHT = 0.1
    THRESHOLD_VARIANCE_MULTIPLIER = 0.1
    DYNAMIC_MIN_WEIGHT_THRESHOLD = 0.1
    REPUTATION_SCALING_THRESHOLD = 0.7
    REPUTATION_SCALING_RANGE = 0.3

    def __init__(self, engine: "Engine", config: "Config"):
        self._engine = engine
        self._config = config
        self._addr = engine.addr
        self._log_dir = engine.log_dir
        self._idx = engine.idx

        self._initialize_data_structures()
        self._configure_constants()
        self._load_configuration()
        self._setup_connection_metrics()
        self._configure_metric_weights()
        self._log_initialization_info()

    def _configure_constants(self):
        reputation_config = self._config.participant.get("defense_args", {}).get("reputation", {})
        constants_config = reputation_config.get("constants", {})

        self.REPUTATION_THRESHOLD = constants_config.get("reputation_threshold", self.REPUTATION_THRESHOLD)
        self.SIMILARITY_THRESHOLD = constants_config.get("similarity_threshold", self.SIMILARITY_THRESHOLD)
        self.INITIAL_ROUND_FOR_REPUTATION = constants_config.get("initial_round_for_reputation", self.INITIAL_ROUND_FOR_REPUTATION)
        self.INITIAL_ROUND_FOR_FRACTION = constants_config.get("initial_round_for_fraction", self.INITIAL_ROUND_FOR_FRACTION)
        self.HISTORY_ROUNDS_LOOKBACK = constants_config.get("history_rounds_lookback", self.HISTORY_ROUNDS_LOOKBACK)
        self.WEIGHTED_HISTORY_ROUNDS = constants_config.get("weighted_history_rounds", self.WEIGHTED_HISTORY_ROUNDS)
        self.FRACTION_ANOMALY_MULTIPLIER = constants_config.get("fraction_anomaly_multiplier", self.FRACTION_ANOMALY_MULTIPLIER)
        self.THRESHOLD_ANOMALY_MULTIPLIER = constants_config.get("threshold_anomaly_multiplier", self.THRESHOLD_ANOMALY_MULTIPLIER)
        self.LATENCY_AUGMENT_FACTOR = constants_config.get("latency_augment_factor", self.LATENCY_AUGMENT_FACTOR)
        self.MESSAGE_AUGMENT_FACTOR_EARLY = constants_config.get("message_augment_factor_early", self.MESSAGE_AUGMENT_FACTOR_EARLY)
        self.MESSAGE_AUGMENT_FACTOR_NORMAL = constants_config.get("message_augment_factor_normal", self.MESSAGE_AUGMENT_FACTOR_NORMAL)
        self.HISTORICAL_PENALTY_THRESHOLD = constants_config.get("historical_penalty_threshold", self.HISTORICAL_PENALTY_THRESHOLD)
        self.NEGATIVE_LATENCY_PENALTY = constants_config.get("negative_latency_penalty", self.NEGATIVE_LATENCY_PENALTY)
        self.CURRENT_VALUE_WEIGHT_HIGH = constants_config.get("current_value_weight_high", self.CURRENT_VALUE_WEIGHT_HIGH)
        self.CURRENT_VALUE_WEIGHT_LOW = constants_config.get("current_value_weight_low", self.CURRENT_VALUE_WEIGHT_LOW)
        self.PAST_VALUE_WEIGHT_HIGH = constants_config.get("past_value_weight_high", self.PAST_VALUE_WEIGHT_HIGH)
        self.PAST_VALUE_WEIGHT_LOW = constants_config.get("past_value_weight_low", self.PAST_VALUE_WEIGHT_LOW)
        self.ZERO_VALUE_DECAY_FACTOR = constants_config.get("zero_value_decay_factor", self.ZERO_VALUE_DECAY_FACTOR)
        self.REPUTATION_CURRENT_WEIGHT = constants_config.get("reputation_current_weight", self.REPUTATION_CURRENT_WEIGHT)
        self.REPUTATION_FEEDBACK_WEIGHT = constants_config.get("reputation_feedback_weight", self.REPUTATION_FEEDBACK_WEIGHT)
        self.THRESHOLD_VARIANCE_MULTIPLIER = constants_config.get("threshold_variance_multiplier", self.THRESHOLD_VARIANCE_MULTIPLIER)
        self.DYNAMIC_MIN_WEIGHT_THRESHOLD = constants_config.get("dynamic_min_weight_threshold", self.DYNAMIC_MIN_WEIGHT_THRESHOLD)
        self.REPUTATION_SCALING_THRESHOLD = constants_config.get("reputation_scaling_threshold", self.REPUTATION_SCALING_THRESHOLD)
        self.REPUTATION_SCALING_RANGE = constants_config.get("reputation_scaling_range", self.REPUTATION_SCALING_RANGE)

    def _initialize_data_structures(self):
        self.reputation = {}
        self.reputation_with_feedback = {}
        self.reputation_with_all_feedback = {}
        self.reputation_history = {}
        self.latest_accusations = {}
        self.rejected_nodes = set()
        self.fraction_of_params_changed = {}
        self.history_data = {}
        self.metric_weights = {}
        self.connection_metrics = {}
        self.messages_number_message = []
        self.number_message_history = {}
        self._messages_received_from_sources = {}
        self.round_timing_info = {}
        self.neighbor_reputation_history = {}
        self.fraction_changed_history = {}
        self.messages_model_arrival_latency = {}
        self.model_arrival_latency_history = {}
        self.previous_threshold_number_message = {}
        self.previous_std_dev_number_message = {}
        self.previous_percentile_25_number_message = {}
        self.previous_percentile_85_number_message = {}
        # New: Track nodes that actively participate in reputation. Maps node_id -> last_round_seen
        self.active_reporters = {}
        # New: Nodes permanently blocked by Honeypot orders
        self.permanently_blocked = set()

    def mark_reporter(self, node_id: str):
        """Register a node as an active participant in the reputation system."""
        val = getattr(self._engine, 'round', 0)
        current_round = val if val is not None else 0
        self.active_reporters[node_id] = current_round

    def is_active_reporter(self, node_id: str, freshness_window: int = 1) -> bool:
        """Check if a node has reported reputation data recently."""
        if node_id not in self.active_reporters:
            return False

        last_seen = self.active_reporters[node_id]
        val = getattr(self._engine, 'round', 0)
        current_round = val if val is not None else 0

        # If it was seen in current round or previous X rounds
        return (current_round - last_seen) <= freshness_window

    def force_block(self, node_id: str, network_block: bool = True):
        """Permanently block a node from aggregation (Honeypot Order)."""
        self.permanently_blocked.add(node_id)
        logging.warning(f"[Reputation] 🚫 Node {node_id} permanently BLOCKED by containment order.")

        # CRITICAL FIX: Emit an event to softly remove the node from aggregation
        # without cutting the physical TCP connection
        # This prevents the Round Synchronization Deadlock (60s timeouts)
        if hasattr(self, "_engine") and hasattr(self._engine, "cm"):
            logging.warning(f"[Reputation] ⚡ IGNORING AGGREGATION from blocked neighbor {node_id} (soft-disconnect) (network_block={network_block}).")

            async def publish_removal():
                event = UpdateNeighborEvent(node_id, removed=True)
                await EventManager.get_instance().publish_node_event(event)

            try:
                loop = asyncio.get_running_loop()
                loop.create_task(publish_removal())
            except RuntimeError:
                pass # Event loop not running edge case

        # Also enforce network-level blacklist to stop receiving messages
        # Only if requested (Honeypots might want to keep listening to monitor)
        if network_block and hasattr(self, "_engine") and hasattr(self._engine, "cm") and hasattr(self._engine.cm, "bl"):
            try:
                # 1. Enforce network-level blacklist (stop receiving)
                asyncio.create_task(self._engine.cm.bl.add_to_blacklist(node_id))
                logging.info(f"[Reputation] 🔌 Network blacklist scheduled for {node_id}")

            except Exception as e:
                logging.error(f"[Reputation] Failed to schedule network blacklist: {e}")

    def _load_configuration(self):
        defense_args = self._config.participant.get("defense_args", {})
        reputation_config = defense_args.get("reputation", {})
        honeypot_defense = defense_args.get("honeypot", {})

        is_honeypot_defense_active = honeypot_defense.get("enabled", False)
        node_role = self._config.participant.get("device_args", {}).get("role", "").lower()

        # Only force enable reputation defaults if this node is part of a Honeypot-enabled scenario
        if is_honeypot_defense_active:
            self._enabled = True
            use_provided_metrics = False
            if "metrics" in reputation_config:
                 if any(m.get("enabled", False) for m in reputation_config["metrics"].values()):
                     use_provided_metrics = True
                     self._metrics = reputation_config["metrics"]

            if not use_provided_metrics and reputation_config.get("enabled", False) is not False:
                 logging.info("[Reputation] ⚠️ Honeypot active but all reputation metrics disabled in config. Enforcing defaults to enable threat detection.")
                 self._metrics = {
                     "model_similarity": {"enabled": True, "weight": 0.25},
                     "num_messages": {"enabled": True, "weight": 0.25},
                     "model_arrival_latency": {"enabled": True, "weight": 0.25},
                     "fraction_parameters_changed": {"enabled": True, "weight": 0.25}
                 }
            elif not use_provided_metrics:
                 logging.info("[Reputation] ℹ️ Honeypot active but Reputation explicitly disabled. Running in passive mode (No metrics).")
                 self._metrics = {}

            self._initial_reputation = float(reputation_config.get("initial_reputation", 0.5))
            self._weighting_factor = reputation_config.get("weighting_factor", "dynamic")
        else:
            self._enabled = reputation_config.get("enabled", False)
            self._metrics = reputation_config.get("metrics", {})
            self._initial_reputation = float(reputation_config.get("initial_reputation", 0.5))
            self._weighting_factor = reputation_config.get("weighting_factor", "dynamic")

        if not isinstance(self._metrics, dict):
            logging.error(f"Invalid metrics configuration: expected dict, got {type(self._metrics)}")
            self._metrics = {}

    def _setup_connection_metrics(self):
        neighbors_str = self._config.participant["network_args"]["neighbors"]
        for neighbor in neighbors_str.split():
            self.connection_metrics[neighbor] = Metrics()

    def _configure_metric_weights(self):
        default_weight = 0.25
        metric_names = ["model_arrival_latency", "model_similarity", "num_messages", "fraction_parameters_changed"]

        if self._weighting_factor == "static":
            self._weight_model_arrival_latency = float(
                self._metrics.get("model_arrival_latency", {}).get("weight", default_weight)
            )
            self._weight_model_similarity = float(
                self._metrics.get("model_similarity", {}).get("weight", default_weight)
            )
            self._weight_num_messages = float(
                self._metrics.get("num_messages", {}).get("weight", default_weight)
            )
            self._weight_fraction_params_changed = float(
                self._metrics.get("fraction_parameters_changed", {}).get("weight", default_weight)
            )
        else:
            for metric_name in metric_names:
                if metric_name not in self._metrics:
                    self._metrics[metric_name] = {}
                elif not isinstance(self._metrics[metric_name], dict):
                    self._metrics[metric_name] = {"enabled": bool(self._metrics[metric_name])}
                self._metrics[metric_name]["weight"] = default_weight

            self._weight_model_arrival_latency = default_weight
            self._weight_model_similarity = default_weight
            self._weight_num_messages = default_weight
            self._weight_fraction_params_changed = default_weight

    def _log_initialization_info(self):
        msg = f"Reputation system: {self._enabled}"
        msg += f"\nReputation metrics: {self._metrics}"
        msg += f"\nInitial reputation: {self._initial_reputation}"
        print_msg_box(msg=msg, indent=2, title="Defense information")

    @property
    def engine(self):
        return self._engine

    def _is_metric_enabled(self, metric_name: str, metrics_config: dict = None) -> bool:
        config_to_use = metrics_config if metrics_config is not None else getattr(self, '_metrics', None)

        if not isinstance(config_to_use, dict):
            if metrics_config is not None:
                logging.warning(f"metrics_config is not a dictionary: {type(metrics_config)}")
            else:
                logging.warning("_metrics is not properly initialized")
            return False

        metric_config = config_to_use.get(metric_name)
        if metric_config is None:
            return False

        if isinstance(metric_config, dict):
            return metric_config.get('enabled', True)
        return bool(metric_config)

    def save_data(
        self,
        type_data: str,
        nei: str,
        addr: str,
        num_round: int = None,
        time: float = None,
        current_round: int = None,
        fraction_changed: float = None,
        threshold: float = None,
        latency: float = None,
    ):
        if addr == nei:
            return

        if nei not in self.connection_metrics:
            from nebula.addons.reputation.reputation import Metrics
            self.connection_metrics[nei] = Metrics()

        try:
            metrics_instance = self.connection_metrics[nei]
            # ... resto del código ...

            if type_data == "number_message":
                message_data = {"time": time, "current_round": current_round}
                if not isinstance(metrics_instance.messages, list):
                    metrics_instance.messages = []
                metrics_instance.messages.append(message_data)
            elif type_data == "fraction_of_params_changed":
                fraction_data = {
                    "fraction_changed": fraction_changed,
                    "threshold": threshold,
                    "current_round": current_round,
                }
                metrics_instance.fraction_of_params_changed.update(fraction_data)
            elif type_data == "model_arrival_latency":
                latency_data = {
                    "latency": latency,
                    "round": num_round,
                    "round_received": current_round,
                }
                metrics_instance.model_arrival_latency.update(latency_data)
            else:
                logging.warning(f"Unknown data type: {type_data}")

        except Exception:
            logging.exception(f"Error saving data for type {type_data} and neighbor {nei}")

    async def setup(self):
        if self._enabled:
            await EventManager.get_instance().subscribe_node_event(RoundStartEvent, self.on_round_start)
            await EventManager.get_instance().subscribe_node_event(AggregationEvent, self.calculate_reputation)
            if self._is_metric_enabled("model_similarity"):
                await EventManager.get_instance().subscribe_node_event(UpdateReceivedEvent, self.recollect_similarity)
            if self._is_metric_enabled("fraction_parameters_changed"):
                await EventManager.get_instance().subscribe_node_event(
                    UpdateReceivedEvent, self.recollect_fraction_of_parameters_changed
                )
            if self._is_metric_enabled("model_arrival_latency"):
                await EventManager.get_instance().subscribe_node_event(
                    UpdateReceivedEvent, self.recollect_model_arrival_latency
                )
            if self._is_metric_enabled("num_messages"):
                await EventManager.get_instance().subscribe(("model", "update"), self.recollect_number_message)
                await EventManager.get_instance().subscribe(("model", "initialization"), self.recollect_number_message)
                await EventManager.get_instance().subscribe(("control", "alive"), self.recollect_number_message)
                await EventManager.get_instance().subscribe(
                    ("federation", "federation_models_included"), self.recollect_number_message
                )
                await EventManager.get_instance().subscribe_node_event(DuplicatedMessageEvent, self.recollect_duplicated_number_message)

    async def init_reputation(
        self, federation_nodes=None, round_num=None, last_feedback_round=None, init_reputation=None
    ):
        if not self._enabled:
            return

        if not self._validate_init_parameters(federation_nodes, round_num, init_reputation):
            return

        neighbors = self._validate_federation_nodes(federation_nodes)
        if not neighbors:
            logging.error("init_reputation | No valid neighbors found")
            return

        await self._initialize_neighbor_reputations(neighbors, round_num, last_feedback_round, init_reputation)

    def _validate_init_parameters(self, federation_nodes, round_num, init_reputation) -> bool:
        if not federation_nodes:
            logging.error("init_reputation | No federation nodes provided")
            return False

        if round_num is None:
            logging.warning("init_reputation | Round number not provided")

        if init_reputation is None:
            logging.warning("init_reputation | Initial reputation value not provided")

        return True

    async def _initialize_neighbor_reputations(self, neighbors: list, round_num: int, last_feedback_round: int, init_reputation: float):
        for nei in neighbors:
            self._create_or_update_reputation_entry(nei, round_num, last_feedback_round, init_reputation)
            await self.save_reputation_history_in_memory(self._addr, nei, init_reputation)

    def _create_or_update_reputation_entry(self, nei: str, round_num: int, last_feedback_round: int, init_reputation: float):
        reputation_data = {
            "reputation": init_reputation,
            "round": round_num,
            "last_feedback_round": last_feedback_round,
        }

        if nei not in self.reputation:
            self.reputation[nei] = reputation_data
        elif self.reputation[nei].get("reputation") is None:
            self.reputation[nei].update(reputation_data)

    def _validate_federation_nodes(self, federation_nodes) -> list:
        if not federation_nodes:
            return []

        valid_nodes = [node for node in federation_nodes if node and str(node).strip()]

        if not valid_nodes:
            logging.warning("No valid federation nodes found after filtering")

        return valid_nodes

    async def _calculate_static_reputation(
        self,
        addr: str,
        nei: str,
        metric_values: dict,
    ):
        static_weights = {
            "num_messages": self._weight_num_messages,
            "model_similarity": self._weight_model_similarity,
            "fraction_parameters_changed": self._weight_fraction_params_changed,
            "model_arrival_latency": self._weight_model_arrival_latency,
        }

        reputation_static = sum(
            metric_values.get(metric_name, 0) * static_weights[metric_name]
            for metric_name in static_weights
        )

        logging.info(f"Static reputation for node {nei} at round {await self.engine.get_round()}: {reputation_static}")

        avg_reputation = await self.save_reputation_history_in_memory(self.engine.addr, nei, reputation_static)

        metrics_data = {
            "addr": addr,
            "nei": nei,
            "round": await self.engine.get_round(),
            "reputation_without_feedback": avg_reputation,
            **{f"average_{name}": weight for name, weight in static_weights.items()}
        }

        await self._update_reputation_record(nei, avg_reputation, metrics_data)

    async def _calculate_dynamic_reputation(self, addr, neighbors):
        if not hasattr(self, '_metrics') or self._metrics is None:
            logging.warning("_metrics is not properly initialized")
            return

        average_weights = await self._calculate_average_weights()
        await self._process_neighbors_reputation(addr, neighbors, average_weights)

    async def _calculate_average_weights(self):
        average_weights = {}

        for metric_name in self.history_data.keys():
            if self._is_metric_enabled(metric_name):
                average_weights[metric_name] = await self._get_metric_average_weight(metric_name)

        return average_weights

    async def _get_metric_average_weight(self, metric_name):
        if metric_name not in self.history_data or not self.history_data[metric_name]:
            logging.debug(f"No history data available for metric: {metric_name}")
            return 0

        valid_entries = [
            entry for entry in self.history_data[metric_name]
            if (entry.get("round") is not None and
                entry["round"] >= await self._engine.get_round() and
                entry.get("weight") not in [None, -1])
        ]

        if not valid_entries:
            return 0

        try:
            weights = [entry["weight"] for entry in valid_entries if entry.get("weight") is not None]
            return sum(weights) / len(weights) if weights else 0
        except (TypeError, ZeroDivisionError) as e:
            logging.warning(f"Error calculating average weight for {metric_name}: {e}")
            return 0

    async def _process_neighbors_reputation(self, addr, neighbors, average_weights):
        for nei in neighbors:
            metric_values = await self._get_neighbor_metric_values(nei)

            if all(metric_name in metric_values for metric_name in average_weights):
                await self._update_neighbor_reputation(addr, nei, metric_values, average_weights)

    async def _get_neighbor_metric_values(self, nei):
        metric_values = {}

        for metric_name in self.history_data:
            if self._is_metric_enabled(metric_name):
                for entry in self.history_data.get(metric_name, []):
                    if (entry.get("round") == await self._engine.get_round() and
                        entry.get("metric_name") == metric_name and
                        entry.get("nei") == nei):
                        metric_values[metric_name] = entry.get("metric_value", 0)
                        break

        return metric_values

    async def _update_neighbor_reputation(self, addr, nei, metric_values, average_weights):
        reputation_with_weights = sum(
            metric_values.get(metric_name, 0) * average_weights[metric_name]
            for metric_name in average_weights
        )

        logging.info(
            f"Dynamic reputation with weights for {nei} at round {await self._engine.get_round()}: {reputation_with_weights}"
        )

        avg_reputation = await self.save_reputation_history_in_memory(self._engine.addr, nei, reputation_with_weights)

        metrics_data = {
            "addr": addr,
            "nei": nei,
            "round": await self._engine.get_round(),
            "reputation_without_feedback": avg_reputation,
        }

        for metric_name in metric_values:
            metrics_data[f"average_{metric_name}"] = average_weights[metric_name]

        await self._update_reputation_record(nei, avg_reputation, metrics_data)

    async def _update_reputation_record(self, nei: str, reputation: float, data: dict):
        current_round = await self._engine.get_round()

        if nei not in self.reputation:
            self.reputation[nei] = {
                "reputation": reputation,
                "round": current_round,
                "last_feedback_round": -1,
            }
        else:
            self.reputation[nei]["reputation"] = reputation
            self.reputation[nei]["round"] = current_round

        logging.info(f"Reputation of node {nei}: {self.reputation[nei]['reputation']}")

        # Check if we are ACTIVELY acting as a Honeypot (Role = HONEYPOT)
        # OR if we are in a Honeypot-enabled scenario (Passive Monitoring mode)
        is_honeypot_scenario = self._config.participant.get("defense_args", {}).get("honeypot", {}).get("enabled", False)
        is_active_honeypot = False
        if hasattr(self._engine, "rb"):
             is_active_honeypot = (self._engine.rb.get_role_name() == "honeypot")

        # Mitigation (filtering) is ONLY active if NOT in a Honeypot scenario
        # This fulfills the goal of: Calculate all metrics, but disable mitigation.
        should_filter = not (is_active_honeypot or is_honeypot_scenario)

        if should_filter and self.reputation[nei]["reputation"] < self.REPUTATION_THRESHOLD and current_round > 0:
            self.rejected_nodes.add(nei)
            logging.info(f"Rejected node {nei} at round {current_round}")

    def calculate_weighted_values(
        self,
        avg_messages_number_message_normalized,
        similarity_reputation,
        fraction_score_asign,
        avg_model_arrival_latency,
        history_data,
        current_round,
        addr,
        nei,
        reputation_metrics,
    ):
        if current_round is None:
            return

        self._ensure_history_data_structure(history_data)
        active_metrics = self._get_active_metrics(
            avg_messages_number_message_normalized,
            similarity_reputation,
            fraction_score_asign,
            avg_model_arrival_latency,
            reputation_metrics
        )
        self._add_current_metrics_to_history(active_metrics, history_data, current_round, addr, nei)

        if current_round >= self.INITIAL_ROUND_FOR_REPUTATION and len(active_metrics) > 0:
            adjusted_weights = self._calculate_dynamic_weights(active_metrics, history_data)
        else:
            adjusted_weights = self._calculate_uniform_weights(active_metrics)

        self._update_history_with_weights(active_metrics, history_data, adjusted_weights, current_round, nei)

    def _ensure_history_data_structure(self, history_data: dict):
        required_keys = [
            "num_messages",
            "model_similarity",
            "fraction_parameters_changed",
            "model_arrival_latency",
        ]

        for key in required_keys:
            if key not in history_data:
                history_data[key] = []

    def _get_active_metrics(
        self,
        avg_messages_number_message_normalized,
        similarity_reputation,
        fraction_score_asign,
        avg_model_arrival_latency,
        reputation_metrics
    ) -> dict:
        all_metrics = {
            "num_messages": avg_messages_number_message_normalized,
            "model_similarity": similarity_reputation,
            "fraction_parameters_changed": fraction_score_asign,
            "model_arrival_latency": avg_model_arrival_latency,
        }

        return {k: v for k, v in all_metrics.items() if self._is_metric_enabled(k, reputation_metrics)}

    def _add_current_metrics_to_history(self, active_metrics: dict, history_data: dict, current_round: int, addr: str, nei: str):
        for metric_name, current_value in active_metrics.items():
            history_data[metric_name].append({
                "round": current_round,
                "addr": addr,
                "nei": nei,
                "metric_name": metric_name,
                "metric_value": current_value,
                "weight": None,
            })

    def _calculate_dynamic_weights(self, active_metrics: dict, history_data: dict) -> dict:
        deviations = self._calculate_metric_deviations(active_metrics, history_data)

        if all(deviation == 0.0 for deviation in deviations.values()):
            return self._generate_random_weights(active_metrics)
        else:
            normalized_weights = self._normalize_deviation_weights(deviations)
            return self._adjust_weights_with_minimum(normalized_weights, deviations)

    def _calculate_metric_deviations(self, active_metrics: dict, history_data: dict) -> dict:
        deviations = {}

        for metric_name, current_value in active_metrics.items():
            historical_values = history_data[metric_name]
            metric_values = [
                entry["metric_value"]
                for entry in historical_values
                if "metric_value" in entry and entry["metric_value"] != 0
            ]

            mean_value = np.mean(metric_values) if metric_values else 0
            deviation = abs(current_value - mean_value)
            deviations[metric_name] = deviation

        return deviations

    def _generate_random_weights(self, active_metrics: dict) -> dict:
        num_metrics = len(active_metrics)
        random_weights = [random.random() for _ in range(num_metrics)]
        total_random_weight = sum(random_weights)

        return {
            metric_name: weight / total_random_weight
            for metric_name, weight in zip(active_metrics, random_weights, strict=False)
        }

    def _normalize_deviation_weights(self, deviations: dict) -> dict:
        max_deviation = max(deviations.values()) if deviations else 1
        normalized_weights = {
            metric_name: (deviation / max_deviation)
            for metric_name, deviation in deviations.items()
        }

        total_weight = sum(normalized_weights.values())
        if total_weight > 0:
            return {
                metric_name: weight / total_weight
                for metric_name, weight in normalized_weights.items()
            }
        else:
            num_metrics = len(deviations)
            return dict.fromkeys(deviations.keys(), 1 / num_metrics)

    def _adjust_weights_with_minimum(self, normalized_weights: dict, deviations: dict) -> dict:
        mean_deviation = np.mean(list(deviations.values()))
        dynamic_min_weight = max(self.DYNAMIC_MIN_WEIGHT_THRESHOLD, mean_deviation / (mean_deviation + 1))

        adjusted_weights = {}
        total_adjusted_weight = 0

        for metric_name, weight in normalized_weights.items():
            adjusted_weight = max(weight, dynamic_min_weight)
            adjusted_weights[metric_name] = adjusted_weight
            total_adjusted_weight += adjusted_weight

        if total_adjusted_weight > 1:
            for metric_name in adjusted_weights:
                adjusted_weights[metric_name] /= total_adjusted_weight

        return adjusted_weights

    def _calculate_uniform_weights(self, active_metrics: dict) -> dict:
        num_metrics = len(active_metrics)
        if num_metrics == 0:
            return {}
        return dict.fromkeys(active_metrics, 1 / num_metrics)

    def _update_history_with_weights(self, active_metrics: dict, history_data: dict, weights: dict, current_round: int, nei: str):
        for metric_name in active_metrics:
            weight = weights.get(metric_name, -1)
            for entry in history_data[metric_name]:
                if (entry["metric_name"] == metric_name and
                    entry["round"] == current_round and
                    entry["nei"] == nei):
                    entry["weight"] = weight

    async def calculate_value_metrics(self, addr, nei, metrics_active=None):
        try:
            current_round = await self._engine.get_round()
            metrics_instance = self.connection_metrics.get(nei)

            if not metrics_instance:
                logging.warning(f"No metrics found for neighbor {nei}")
                return self._get_default_metric_values()

            metric_results = {
                "messages": self._process_num_messages_metric(metrics_instance, addr, nei, current_round, metrics_active),
                "fraction": self._process_fraction_parameters_metric(metrics_instance, addr, nei, current_round, metrics_active),
                "latency": self._process_model_arrival_latency_metric(metrics_instance, addr, nei, current_round, metrics_active),
                "similarity": self._process_model_similarity_metric(nei, current_round, metrics_active)
            }

            self._log_metrics_graphics(metric_results, addr, nei, current_round)

            return (
                metric_results["messages"]["avg"],
                metric_results["similarity"],
                metric_results["fraction"],
                metric_results["latency"]
            )

        except Exception as e:
            logging.exception(f"Error calculating reputation. Type: {type(e).__name__}")
            return 0, 0, 0, 0

    def _get_default_metric_values(self) -> tuple:
        return (0, 0, 0, 0)

    def _process_num_messages_metric(self, metrics_instance, addr: str, nei: str, current_round: int, metrics_active) -> dict:
        if not self._is_metric_enabled("num_messages", metrics_active):
            return {"normalized": 0, "count": 0, "avg": 0}

        filtered_messages = [
            msg for msg in metrics_instance.messages if msg.get("current_round") == current_round
        ]

        for msg in filtered_messages:
            self.messages_number_message.append({
                "number_message": msg.get("time"),
                "current_round": msg.get("current_round"),
                "key": (addr, nei),
            })

        normalized, count = self.manage_metric_number_message(
            self.messages_number_message, addr, nei, current_round, True
        )

        avg = self.save_number_message_history(addr, nei, normalized, current_round)

        if avg is None and current_round > self.HISTORY_ROUNDS_LOOKBACK:
            avg = self.number_message_history[(addr, nei)][current_round - 1]["avg_number_message"]

        return {"normalized": normalized, "count": count, "avg": avg or 0}

    def _process_fraction_parameters_metric(self, metrics_instance, addr: str, nei: str, current_round: int, metrics_active) -> float:
        if not self._is_metric_enabled("fraction_parameters_changed", metrics_active):
            return 0

        score_fraction = 0
        if metrics_instance.fraction_of_params_changed.get("current_round") == current_round:
            fraction_changed = metrics_instance.fraction_of_params_changed.get("fraction_changed")
            threshold = metrics_instance.fraction_of_params_changed.get("threshold")
            score_fraction = self.analyze_anomalies(addr, nei, current_round, fraction_changed, threshold)

        if current_round >= self.INITIAL_ROUND_FOR_FRACTION:
            return self._calculate_fraction_score_assignment(addr, nei, current_round, score_fraction)
        else:
            return 0

    def _calculate_fraction_score_assignment(self, addr: str, nei: str, current_round: int, score_fraction: float) -> float:
        key_current = (addr, nei, current_round)

        if score_fraction > 0:
            return self._calculate_positive_fraction_score(addr, nei, current_round, score_fraction, key_current)
        else:
            return self._calculate_zero_fraction_score(addr, nei, current_round, key_current)

    def _calculate_positive_fraction_score(self, addr: str, nei: str, current_round: int, score_fraction: float, key_current: tuple) -> float:
        past_scores = []
        for i in range(1, 5):
            key_prev = (addr, nei, current_round - i)
            score_prev = self.fraction_changed_history.get(key_prev, {}).get("finally_fraction_score")
            if score_prev is not None and score_prev > 0:
                past_scores.append(score_prev)

        if past_scores:
            avg_past = sum(past_scores) / len(past_scores)
            fraction_score_asign = score_fraction * 0.2 + avg_past * 0.8
        else:
            fraction_score_asign = score_fraction

        self.fraction_changed_history[key_current]["finally_fraction_score"] = fraction_score_asign
        return fraction_score_asign

    def _calculate_zero_fraction_score(self, addr: str, nei: str, current_round: int, key_current: tuple) -> float:
        key_prev = (addr, nei, current_round - 1)
        prev_score = self.fraction_changed_history.get(key_prev, {}).get("finally_fraction_score")

        if prev_score is not None:
            fraction_score_asign = prev_score * self.ZERO_VALUE_DECAY_FACTOR
        else:
            fraction_neighbors_scores = {
                key: value.get("finally_fraction_score")
                for key, value in self.fraction_changed_history.items()
                if value.get("finally_fraction_score") is not None
            }
            fraction_score_asign = np.mean(list(fraction_neighbors_scores.values())) if fraction_neighbors_scores else 0

        if key_current not in self.fraction_changed_history:
            self.fraction_changed_history[key_current] = {}

        self.fraction_changed_history[key_current]["finally_fraction_score"] = fraction_score_asign
        return fraction_score_asign

    def _process_model_arrival_latency_metric(self, metrics_instance, addr: str, nei: str, current_round: int, metrics_active) -> float:
        if not self._is_metric_enabled("model_arrival_latency", metrics_active):
            return 0

        latency_normalized = 0
        if metrics_instance.model_arrival_latency.get("round_received") == current_round:
            round_num = metrics_instance.model_arrival_latency.get("round")
            latency = metrics_instance.model_arrival_latency.get("latency")
            latency_normalized = self.manage_model_arrival_latency(addr, nei, latency, current_round, round_num)

        if latency_normalized >= 0:
            avg_latency = self.save_model_arrival_latency_history(nei, latency_normalized, current_round)
            if avg_latency is None and current_round > 1:
                avg_latency = self.model_arrival_latency_history[(addr, nei)][current_round - 1]["score"]
            return avg_latency or 0

        return 0

    def _process_model_similarity_metric(self, nei: str, current_round: int, metrics_active) -> float:
        if current_round >= 1 and self._is_metric_enabled("model_similarity", metrics_active):
            return self.calculate_similarity_from_metrics(nei, current_round)
        return 0

    def _log_metrics_graphics(self, metric_results: dict, addr: str, nei: str, current_round: int):
        self.create_graphics_to_metrics(
            metric_results["messages"]["count"],
            metric_results["messages"]["avg"],
            metric_results["similarity"],
            metric_results["fraction"],
            metric_results["latency"],
            addr,
            nei,
            current_round,
            self.engine.total_rounds,
        )

    def create_graphics_to_metrics(
        self,
        number_message_count: float,
        number_message_norm: float,
        similarity: float,
        fraction: float,
        model_arrival_latency: float,
        addr: str,
        nei: str,
        current_round: int,
        total_rounds: int,
    ):
        if current_round is None or current_round >= total_rounds:
            return

        self.engine.trainer._logger.log_data(
            {f"R-Model_arrival_latency_reputation/{addr}": {nei: model_arrival_latency}},
            step=current_round
        )
        self.engine.trainer._logger.log_data(
            {f"R-Count_messages_number_message_reputation/{addr}": {nei: number_message_count}},
            step=current_round
        )
        self.engine.trainer._logger.log_data(
            {f"R-number_message_reputation/{addr}": {nei: number_message_norm}},
            step=current_round
        )
        self.engine.trainer._logger.log_data(
            {f"R-Similarity_reputation/{addr}": {nei: similarity}},
            step=current_round
        )
        self.engine.trainer._logger.log_data(
            {f"R-Fraction_reputation/{addr}": {nei: fraction}},
            step=current_round
        )

    def analyze_anomalies(
        self,
        addr,
        nei,
        current_round,
        fraction_changed,
        threshold,
    ):
        try:
            key = (addr, nei, current_round)
            self._initialize_fraction_history_entry(key, fraction_changed, threshold)

            if current_round == 0:
                return self._handle_initial_round_anomalies(key, fraction_changed, threshold)
            else:
                return self._handle_subsequent_round_anomalies(key, addr, nei, current_round, fraction_changed, threshold)

        except Exception:
            logging.exception("Error analyzing anomalies")
            return -1

    def _initialize_fraction_history_entry(self, key: tuple, fraction_changed: float, threshold: float):
        if key not in self.fraction_changed_history:
            self.fraction_changed_history[key] = {
                "fraction_changed": fraction_changed or 0,
                "threshold": threshold or 0,
                "fraction_score": None,
                "fraction_anomaly": False,
                "threshold_anomaly": False,
                "mean_fraction": None,
                "std_dev_fraction": None,
                "mean_threshold": None,
                "std_dev_threshold": None,
            }

    def _handle_initial_round_anomalies(self, key: tuple, fraction_changed: float, threshold: float) -> float:
        self.fraction_changed_history[key].update({
            "mean_fraction": fraction_changed,
            "std_dev_fraction": 0.0,
            "mean_threshold": threshold,
            "std_dev_threshold": 0.0,
            "fraction_score": 1.0,
        })
        return 1.0

    def _handle_subsequent_round_anomalies(
        self, key: tuple, addr: str, nei: str, current_round: int, fraction_changed: float, threshold: float
    ) -> float:
        prev_stats = self._find_previous_valid_stats(addr, nei, current_round)

        if prev_stats is None:
            logging.debug(f"No valid previous stats found for {addr}, {nei}, round {current_round}. Using default score.")
            return 1.0

        anomalies = self._detect_anomalies(fraction_changed, threshold, prev_stats)
        values = self._calculate_anomaly_values(fraction_changed, threshold, prev_stats, anomalies)
        fraction_score = self._calculate_combined_score(values)
        self._update_fraction_statistics(key, fraction_changed, threshold, prev_stats, anomalies, fraction_score)

        return max(fraction_score, 0)

    def _find_previous_valid_stats(self, addr: str, nei: str, current_round: int) -> dict:
        for i in range(1, current_round + 1):
            candidate_key = (addr, nei, current_round - i)
            candidate_data = self.fraction_changed_history.get(candidate_key, {})

            required_keys = ["mean_fraction", "std_dev_fraction", "mean_threshold", "std_dev_threshold"]
            if all(candidate_data.get(k) is not None for k in required_keys):
                return candidate_data

        return None

    def _detect_anomalies(self, current_fraction: float, current_threshold: float, prev_stats: dict) -> dict:
        upper_mean_fraction = (prev_stats["mean_fraction"] + prev_stats["std_dev_fraction"]) * self.FRACTION_ANOMALY_MULTIPLIER
        upper_mean_threshold = (prev_stats["mean_threshold"] + prev_stats["std_dev_threshold"]) * self.THRESHOLD_ANOMALY_MULTIPLIER

        return {
            "fraction_anomaly": current_fraction > upper_mean_fraction,
            "threshold_anomaly": current_threshold > upper_mean_threshold,
            "upper_mean_fraction": upper_mean_fraction,
            "upper_mean_threshold": upper_mean_threshold,
        }

    def _calculate_anomaly_values(
        self, current_fraction: float, current_threshold: float, prev_stats: dict, anomalies: dict
    ) -> dict:
        fraction_value = 1.0
        threshold_value = 1.0

        if anomalies["fraction_anomaly"]:
            mean_fraction_prev = prev_stats["mean_fraction"]
            if mean_fraction_prev > 0:
                penalization_factor = abs(current_fraction - mean_fraction_prev) / mean_fraction_prev
                fraction_value = 1 - (1 / (1 + np.exp(-penalization_factor)))

        if anomalies["threshold_anomaly"]:
            mean_threshold_prev = prev_stats["mean_threshold"]
            if mean_threshold_prev > 0:
                penalization_factor = abs(current_threshold - mean_threshold_prev) / mean_threshold_prev
                threshold_value = 1 - (1 / (1 + np.exp(-penalization_factor)))

        return {
            "fraction_value": fraction_value,
            "threshold_value": threshold_value,
        }

    def _calculate_combined_score(self, values: dict) -> float:
        fraction_weight = 0.5
        threshold_weight = 0.5
        return fraction_weight * values["fraction_value"] + threshold_weight * values["threshold_value"]

    def _update_fraction_statistics(
        self, key: tuple, current_fraction: float, current_threshold: float,
        prev_stats: dict, anomalies: dict, fraction_score: float
    ):
        self.fraction_changed_history[key]["fraction_anomaly"] = anomalies["fraction_anomaly"]
        self.fraction_changed_history[key]["threshold_anomaly"] = anomalies["threshold_anomaly"]

        self.fraction_changed_history[key]["mean_fraction"] = (current_fraction + prev_stats["mean_fraction"]) / 2
        self.fraction_changed_history[key]["mean_threshold"] = (current_threshold + prev_stats["mean_threshold"]) / 2

        fraction_variance = ((current_fraction - prev_stats["mean_fraction"]) ** 2 + prev_stats["std_dev_fraction"] ** 2) / 2
        threshold_variance = ((self.THRESHOLD_VARIANCE_MULTIPLIER * (current_threshold - prev_stats["mean_threshold"]) ** 2) + prev_stats["std_dev_threshold"] ** 2) / 2

        self.fraction_changed_history[key]["std_dev_fraction"] = np.sqrt(fraction_variance)
        self.fraction_changed_history[key]["std_dev_threshold"] = np.sqrt(threshold_variance)
        self.fraction_changed_history[key]["fraction_score"] = fraction_score

    def manage_model_arrival_latency(self, addr, nei, latency, current_round, round_num):
        try:
            current_key = nei

            self._initialize_latency_round_entry(current_round, current_key, latency)

            if current_round >= 1:
                score = self._calculate_latency_score(current_round, current_key, latency)
                self._update_latency_entry_with_score(current_round, current_key, score)
            else:
                score = 0

            return score

        except Exception as e:
            logging.exception(f"Error managing model_arrival_latency: {e}")
            return 0

    def _initialize_latency_round_entry(self, current_round: int, current_key: str, latency: float):
        if current_round not in self.model_arrival_latency_history:
            self.model_arrival_latency_history[current_round] = {}

        self.model_arrival_latency_history[current_round][current_key] = {
            "latency": latency,
            "score": 0.0,
        }

    def _calculate_latency_score(self, current_round: int, current_key: str, latency: float) -> float:
        target_round = self._get_target_round_for_latency(current_round)
        all_latencies = self._get_all_latencies_for_round(target_round)

        if not all_latencies:
            return 0.0

        mean_latency = np.mean(all_latencies)
        augment_mean = mean_latency * self.LATENCY_AUGMENT_FACTOR

        if latency is None:
            logging.info(f"latency is None in round {current_round} for nei {current_key}")
            return -0.5

        if latency <= augment_mean:
            return 1.0
        else:
            return 1 / (1 + np.exp(abs(latency - mean_latency) / mean_latency)) if mean_latency != 0 else 0.0

    def _get_target_round_for_latency(self, current_round: int) -> int:
        target_round = current_round - 1
        return target_round if target_round in self.model_arrival_latency_history else current_round

    def _get_all_latencies_for_round(self, target_round: int) -> list:
        return [
            data["latency"]
            for data in self.model_arrival_latency_history.get(target_round, {}).values()
            if data.get("latency") not in (None, 0.0)
        ]

    def _update_latency_entry_with_score(self, current_round: int, current_key: str, score: float):
        target_round = self._get_target_round_for_latency(current_round)
        all_latencies = self._get_all_latencies_for_round(target_round)
        mean_latency = np.mean(all_latencies) if all_latencies else 0

        self.model_arrival_latency_history[current_round][current_key].update({
            "mean_latency": mean_latency,
            "score": score,
        })

    def save_model_arrival_latency_history(self, nei, model_arrival_latency, round_num):
        try:
            current_key = nei

            self._initialize_latency_history_entry(round_num, current_key, model_arrival_latency)

            if model_arrival_latency > 0 and round_num >= 1:
                avg_model_arrival_latency = self._calculate_latency_weighted_average_positive(
                    round_num, current_key, model_arrival_latency
                )
            elif model_arrival_latency == 0 and round_num >= 1:
                avg_model_arrival_latency = self._calculate_latency_weighted_average_zero(
                    round_num, current_key
                )
            elif model_arrival_latency < 0 and round_num >= 1:
                avg_model_arrival_latency = abs(model_arrival_latency) * self.NEGATIVE_LATENCY_PENALTY
            else:
                avg_model_arrival_latency = 0

            self.model_arrival_latency_history[round_num][current_key]["avg_model_arrival_latency"] = (
                avg_model_arrival_latency
            )

            return avg_model_arrival_latency

        except Exception:
            logging.exception("Error saving model_arrival_latency history")

    def _initialize_latency_history_entry(self, round_num: int, current_key: str, latency_value: float):
        if round_num not in self.model_arrival_latency_history:
            self.model_arrival_latency_history[round_num] = {}

        if current_key not in self.model_arrival_latency_history[round_num]:
            self.model_arrival_latency_history[round_num][current_key] = {}

        self.model_arrival_latency_history[round_num][current_key].update({
            "score": latency_value,
        })

    def _calculate_latency_weighted_average_positive(self, round_num: int, current_key: str, current_value: float) -> float:
        past_values = []
        for r in range(round_num - 3, round_num):
            val = (
                self.model_arrival_latency_history.get(r, {})
                .get(current_key, {})
                .get("avg_model_arrival_latency", None)
            )
            if val is not None and val != 0:
                past_values.append(val)

        if past_values:
            avg_past = sum(past_values) / len(past_values)
            return current_value * self.CURRENT_VALUE_WEIGHT_LOW + avg_past * self.PAST_VALUE_WEIGHT_HIGH
        else:
            return current_value

    def _calculate_latency_weighted_average_zero(self, round_num: int, current_key: str) -> float:
        previous_avg = (
            self.model_arrival_latency_history.get(round_num - 1, {})
            .get(current_key, {})
            .get("avg_model_arrival_latency", None)
        )
        return previous_avg * self.ZERO_VALUE_DECAY_FACTOR if previous_avg is not None else 0

    def manage_metric_number_message(
        self, messages_number_message: list, addr: str, nei: str, current_round: int, metric_active: bool = True
    ) -> tuple[float, int]:
        try:
            if current_round == 0 or not metric_active:
                return 0.0, 0

            messages_count = self._count_relevant_messages(messages_number_message, addr, nei, current_round)
            neighbor_stats = self._calculate_neighbor_statistics(messages_number_message, current_round)

            normalized_messages = self._calculate_normalized_messages(messages_count, neighbor_stats)

            normalized_messages = self._apply_historical_penalty(
                normalized_messages, addr, nei, current_round
            )

            self._store_message_history(addr, nei, current_round, normalized_messages)
            normalized_messages = max(0.001, normalized_messages)

            return normalized_messages, messages_count

        except Exception:
            logging.exception("Error managing metric number_message")
            return 0.0, 0

    def _count_relevant_messages(self, messages: list, addr: str, nei: str, current_round: int) -> int:
        current_addr_nei = (addr, nei)
        relevant_messages = [
            msg for msg in messages
            if msg["key"] == current_addr_nei and msg["current_round"] == current_round
        ]
        return len(relevant_messages)

    def _calculate_neighbor_statistics(self, messages: list, current_round: int) -> dict:
        previous_round = current_round - 1
        all_messages_previous_round = [
            m for m in messages if m.get("current_round") == previous_round
        ]

        neighbor_counts = {}
        for m in all_messages_previous_round:
            key = m.get("key")
            neighbor_counts[key] = neighbor_counts.get(key, 0) + 1

        counts_all_neighbors = list(neighbor_counts.values())

        if not counts_all_neighbors:
            return {
                "percentile_reference": 0,
                "std_dev": 0,
                "mean_messages": 0,
                "augment_mean": 0,
            }

        mean_messages = np.mean(counts_all_neighbors)

        return {
            "percentile_reference": np.percentile(counts_all_neighbors, 25),
            "std_dev": np.std(counts_all_neighbors),
            "mean_messages": mean_messages,
            "augment_mean": mean_messages * self.MESSAGE_AUGMENT_FACTOR_EARLY if current_round <= self.INITIAL_ROUND_FOR_REPUTATION else mean_messages * self.MESSAGE_AUGMENT_FACTOR_NORMAL,
        }

    def _calculate_normalized_messages(self, messages_count: int, neighbor_stats: dict) -> float:
        normalized_messages = 1.0
        penalties_applied = []

        relative_increase = self._calculate_relative_increase(messages_count, neighbor_stats["percentile_reference"])
        dynamic_margin = self._calculate_dynamic_margin(neighbor_stats)

        if relative_increase > dynamic_margin:
            penalty_ratio = self._calculate_penalty_ratio(relative_increase, dynamic_margin)
            normalized_messages *= np.exp(-(penalty_ratio**2))
            penalties_applied.append(f"relative_penalty({penalty_ratio:.3f})")

        if self._should_apply_extra_penalty(messages_count, neighbor_stats):
            extra_penalty_factor = self._calculate_extra_penalty_factor(messages_count, neighbor_stats)
            normalized_messages *= np.exp(-((extra_penalty_factor) ** 2))
            penalties_applied.append(f"extra_penalty({extra_penalty_factor:.3f})")

        if penalties_applied:
            logging.debug(f"Message penalties applied: {', '.join(penalties_applied)} -> score: {normalized_messages:.4f}")

        return normalized_messages

    def _calculate_relative_increase(self, messages_count: int, percentile_reference: float) -> float:
        if percentile_reference > 0:
            raw_relative_increase = (messages_count - percentile_reference) / percentile_reference
            return np.log1p(raw_relative_increase)
        return 0.0

    def _calculate_dynamic_margin(self, neighbor_stats: dict) -> float:
        std_dev = neighbor_stats["std_dev"]
        percentile_reference = neighbor_stats["percentile_reference"]
        return (std_dev + 1) / (np.log1p(percentile_reference) + 1)

    def _calculate_penalty_ratio(self, relative_increase: float, dynamic_margin: float) -> float:
        epsilon = 1e-6
        return np.log1p(relative_increase - dynamic_margin) / (np.log1p(dynamic_margin + epsilon) + epsilon)

    def _should_apply_extra_penalty(self, messages_count: int, neighbor_stats: dict) -> bool:
        return (neighbor_stats["mean_messages"] > 0 and
                messages_count > neighbor_stats["augment_mean"])

    def _calculate_extra_penalty_factor(self, messages_count: int, neighbor_stats: dict) -> float:
        epsilon = 1e-6
        mean_messages = neighbor_stats["mean_messages"]
        augment_mean = neighbor_stats["augment_mean"]

        extra_penalty = (messages_count - mean_messages) / (mean_messages + epsilon)
        amplification = 1 + (augment_mean / (mean_messages + epsilon))
        return extra_penalty * amplification

    def _apply_historical_penalty(self, normalized_messages: float, addr: str, nei: str, current_round: int) -> float:
        if current_round <= 1:
            return normalized_messages

        prev_data = (
            self.number_message_history.get((addr, nei), {})
            .get(current_round - 1, {})
        )

        prev_score = prev_data.get("normalized_messages")
        was_previously_penalized = prev_data.get("was_penalized", False)

        if prev_score is not None and prev_score < self.HISTORICAL_PENALTY_THRESHOLD:
            original_score = normalized_messages

            if was_previously_penalized:
                penalty_factor = self.HISTORICAL_PENALTY_THRESHOLD * 0.8
                logging.debug(f"Repeated penalty applied to {nei}: stricter historical penalty")
            else:
                penalty_factor = self.HISTORICAL_PENALTY_THRESHOLD

            normalized_messages *= penalty_factor
            logging.debug(f"Historical penalty applied to {nei}: {original_score:.4f} -> {normalized_messages:.4f} (prev_score: {prev_score:.4f}, was_penalized: {was_previously_penalized})")

        return normalized_messages

    def _store_message_history(self, addr: str, nei: str, current_round: int, normalized_messages: float):
        key = (addr, nei)
        if key not in self.number_message_history:
            self.number_message_history[key] = {}

        was_penalized = normalized_messages < 1.0

        self.number_message_history[key][current_round] = {
            "normalized_messages": normalized_messages,
            "was_penalized": was_penalized,
            "penalty_severity": 1.0 - normalized_messages if was_penalized else 0.0
        }

    def save_number_message_history(self, addr, nei, messages_number_message_normalized, current_round):
        try:
            key = (addr, nei)

            self._initialize_message_history_entry(key, current_round, messages_number_message_normalized)

            if messages_number_message_normalized > 0 and current_round >= 1:
                avg_number_message = self._calculate_weighted_average_positive(key, current_round, messages_number_message_normalized)
            elif messages_number_message_normalized == 0 and current_round >= 1:
                avg_number_message = self._calculate_weighted_average_zero(key, current_round)
            elif messages_number_message_normalized < 0 and current_round >= 1:
                avg_number_message = abs(messages_number_message_normalized) * self.NEGATIVE_LATENCY_PENALTY
            else:
                avg_number_message = 0

            self.number_message_history[key][current_round]["avg_number_message"] = avg_number_message
            return avg_number_message

        except Exception:
            logging.exception("Error saving number_message history")
            return -1

    def _initialize_message_history_entry(self, key: tuple, current_round: int, messages_normalized: float):
        if key not in self.number_message_history:
            self.number_message_history[key] = {}

        if current_round not in self.number_message_history[key]:
            self.number_message_history[key][current_round] = {}

        self.number_message_history[key][current_round].update({
            "number_message": messages_normalized,
        })

    def _calculate_weighted_average_positive(self, key: tuple, current_round: int, current_value: float) -> float:
        past_values = []
        for r in range(current_round - self.WEIGHTED_HISTORY_ROUNDS, current_round):
            val = self.number_message_history.get(key, {}).get(r, {}).get("avg_number_message", None)
            if val is not None and val != 0:
                past_values.append(val)

        if past_values:
            avg_past = sum(past_values) / len(past_values)
            return current_value * self.CURRENT_VALUE_WEIGHT_HIGH + avg_past * self.PAST_VALUE_WEIGHT_LOW
        else:
            return current_value

    def _calculate_weighted_average_zero(self, key: tuple, current_round: int) -> float:
        previous_avg = (
            self.number_message_history.get(key, {})
            .get(current_round - 1, {})
            .get("avg_number_message", None)
        )
        return previous_avg * self.ZERO_VALUE_DECAY_FACTOR if previous_avg is not None else 0

    async def save_reputation_history_in_memory(self, addr: str, nei: str, reputation: float) -> float:
        try:
            key = (addr, nei)
            current_round = await self._engine.get_round()

            if key not in self.reputation_history:
                self.reputation_history[key] = {}

            self.reputation_history[key][current_round] = reputation

            rounds = sorted(self.reputation_history[key].keys(), reverse=True)[:2]

            if len(rounds) >= 2:
                current_rep = self.reputation_history[key][rounds[0]]
                previous_rep = self.reputation_history[key][rounds[1]]

                current_weight = self.REPUTATION_CURRENT_WEIGHT
                previous_weight = self.REPUTATION_FEEDBACK_WEIGHT
                avg_reputation = (current_rep * current_weight) + (previous_rep * previous_weight)

                logging.info(f"Current reputation: {current_rep}, Previous reputation: {previous_rep}")
                logging.info(f"Reputation ponderated: {avg_reputation}")
            else:
                avg_reputation = reputation

            return avg_reputation

        except Exception:
            logging.exception("Error saving reputation history")
            return -1

    def calculate_similarity_from_metrics(self, nei: str, current_round: int) -> float:
        try:
            metrics_instance = self.connection_metrics.get(nei)
            if not metrics_instance:
                return 0.0

            relevant_metrics = [
                metric for metric in metrics_instance.similarity
                if metric.get("nei") == nei and metric.get("current_round") == current_round
            ]

            if not relevant_metrics:
                relevant_metrics = [
                    metric for metric in metrics_instance.similarity
                    if metric.get("nei") == nei
                ]

            if not relevant_metrics:
                return 0.0
            neighbor_metric = relevant_metrics[-1]

            similarity_weights = {
                "cosine": 0.25,
                "euclidean": 0.25,
                "manhattan": 0.25,
                "pearson_correlation": 0.25,
            }

            similarity_value = sum(
                similarity_weights[metric_name] * float(neighbor_metric.get(metric_name, 0))
                for metric_name in similarity_weights
            )

            return max(0.0, min(1.0, similarity_value))

        except Exception:
            return 0.0

    async def calculate_reputation(self, ae: AggregationEvent):
        if not self._enabled:
            return

        (updates, _, _) = await ae.get_event_data()
        await self._log_reputation_calculation_start()

        neighbors = set(await self._engine._cm.get_addrs_current_connections(only_direct=True))

        await self._process_neighbor_metrics(neighbors)
        await self._calculate_reputation_by_factor(neighbors)
        await self._handle_initial_reputation()
        await self._process_feedback()
        await self._finalize_reputation_calculation(updates, neighbors)

    async def _log_reputation_calculation_start(self):
        current_round = await self._engine.get_round()
        logging.info(f"Calculating reputation at round {current_round}")
        logging.info(f"Active metrics: {self._metrics}")
        logging.info(f"rejected nodes at round {current_round}: {self.rejected_nodes}")
        self.rejected_nodes.clear()
        logging.info(f"Rejected nodes clear: {self.rejected_nodes}")

    async def _process_neighbor_metrics(self, neighbors):
        for nei in neighbors:
            metrics = await self.calculate_value_metrics(
                self._addr, nei, metrics_active=self._metrics
            )

            if self._weighting_factor == "dynamic":
                await self._process_dynamic_metrics(nei, metrics)
            elif self._weighting_factor == "static" and await self._engine.get_round() >= 1:
                await self._process_static_metrics(nei, metrics)

    async def _process_dynamic_metrics(self, nei, metrics):
        (metric_messages_number, metric_similarity, metric_fraction, metric_model_arrival_latency) = metrics

        self.calculate_weighted_values(
            metric_messages_number,
            metric_similarity,
            metric_fraction,
            metric_model_arrival_latency,
            self.history_data,
            await self._engine.get_round(),
            self._addr,
            nei,
            self._metrics,
        )

    async def _process_static_metrics(self, nei, metrics):
        (metric_messages_number, metric_similarity, metric_fraction, metric_model_arrival_latency) = metrics

        metric_values_dict = {
            "num_messages": metric_messages_number,
            "model_similarity": metric_similarity,
            "fraction_parameters_changed": metric_fraction,
            "model_arrival_latency": metric_model_arrival_latency,
        }
        await self._calculate_static_reputation(self._addr, nei, metric_values_dict)

    async def _calculate_reputation_by_factor(self, neighbors):
        if self._weighting_factor == "dynamic" and await self._engine.get_round() >= 1:
            await self._calculate_dynamic_reputation(self._addr, neighbors)

    async def _handle_initial_reputation(self):
        if await self._engine.get_round() < 1 and self._enabled:
            federation = self._engine.config.participant["network_args"]["neighbors"].split()
            await self.init_reputation(
                federation_nodes=federation,
                round_num=await self._engine.get_round(),
                last_feedback_round=-1,
                init_reputation=self._initial_reputation,
            )

    async def _process_feedback(self):
        status = await self.include_feedback_in_reputation()
        current_round = await self._engine.get_round()

        if status:
            logging.info(f"Feedback included in reputation at round {current_round}")
        else:
            logging.info(f"Feedback not included in reputation at round {current_round}")

    async def _finalize_reputation_calculation(self, updates, neighbors):
        if self.reputation is not None:
            self.create_graphic_reputation(self._addr, await self._engine.get_round())
            await self.update_process_aggregation(updates)
            await self.send_reputation_to_neighbors(neighbors)

    async def send_reputation_to_neighbors(self, neighbors):
        current_round = await self._engine.get_round()
        for neighbor in neighbors:
            for node_id, data in self.reputation.items():
                if data["reputation"] is not None:
                    message = self._engine.cm.create_message(
                        "reputation",
                        "share_table",
                        node_id=node_id,
                        score=float(data["reputation"]),
                        round=current_round,
                    )
                    await self._engine.cm.send_message(neighbor, message)
                    logging.info(f"[Reputation] Flooded reputation of {node_id} ({data['reputation']}) to {neighbor}")


    def create_graphic_reputation(self, addr: str, round_num: int):
        try:
            valid_reputations = {
                node_id: float(data["reputation"])
                for node_id, data in self.reputation.items()
                if data.get("reputation") is not None
            }

            if valid_reputations:
                reputation_data = {f"Reputation/{addr}": valid_reputations}
                self._engine.trainer._logger.log_data(reputation_data, step=round_num)

        except Exception:
            logging.exception("Error creating reputation graphic")

    async def update_process_aggregation(self, updates):
        honeypot_enabled = self._config.participant.get("defense_args", {}).get("honeypot", {}).get("enabled", False)

        # Enforce permanent blocks
        if self.permanently_blocked:
             self.rejected_nodes.update(self.permanently_blocked)
             logging.info(f"[Reputation] Enforcing blocks on: {self.permanently_blocked}")

        if not honeypot_enabled:
            for rn in self.rejected_nodes:
                if rn in updates:
                    updates.pop(rn)

        if await self.engine.get_round() >= 1:
            for nei in list(updates.keys()):
                if nei in self.reputation:
                    rep = self.reputation[nei].get("reputation", 0)

                    if honeypot_enabled:
                        continue

                    if rep >= self.REPUTATION_SCALING_THRESHOLD:
                        weight = (rep - self.REPUTATION_SCALING_THRESHOLD) / self.REPUTATION_SCALING_RANGE
                        model_dict = updates[nei][0]
                        extra_data = updates[nei][1]

                        scaled_model = {k: v * weight for k, v in model_dict.items()}
                        updates[nei] = (scaled_model, extra_data)

                        logging.info(f"✅ Nei {nei} with reputation {rep:.4f}, scaled model with weight {weight:.4f}")
                    else:
                        logging.info(f"⛔ Nei {nei} with reputation {rep:.4f}, model rejected")

        logging.info(f"Updates after rejected nodes: {list(updates.keys())}")
        logging.info(f"Nodes rejected: {self.rejected_nodes}")

    async def include_feedback_in_reputation(self):
        weight_current_reputation = self.REPUTATION_CURRENT_WEIGHT
        weight_feedback = self.REPUTATION_FEEDBACK_WEIGHT

        if self.reputation_with_all_feedback is None:
            logging.info("No feedback received.")
            return False

        updated = False

        for (current_node, node_ip, round_num), scores in self.reputation_with_all_feedback.items():
            if not scores:
                logging.info(f"No feedback received for node {node_ip} in round {round_num}")
                continue

            if node_ip not in self.reputation:
                # logging.info(f"No reputation for node {node_ip}") # Too noisy
                continue

            if (
                "last_feedback_round" in self.reputation[node_ip]
                and self.reputation[node_ip]["last_feedback_round"] >= round_num
            ):
                continue

            avg_feedback = sum(scores) / len(scores)
            logging.info(f"Receive feedback to node {node_ip} with average score {avg_feedback}")

            current_reputation = self.reputation[node_ip]["reputation"]
            if current_reputation is None:
                logging.info(f"No reputation calculate for node {node_ip}.")
                continue

            combined_reputation = (current_reputation * weight_current_reputation) + (avg_feedback * weight_feedback)
            logging.info(f"Combined reputation for node {node_ip} in round {round_num}: {combined_reputation}")

            self.reputation[node_ip] = {
                "reputation": combined_reputation,
                "round": await self._engine.get_round(),
                "last_feedback_round": round_num,
            }
            updated = True
            logging.info(f"Updated self.reputation for {node_ip}: {self.reputation[node_ip]}")

        if updated:
            return True
        else:
            return False

    def get_suspects_from_feedback(self, threshold=0.4):
        suspects = []
        if self.reputation_with_all_feedback:
            for (reporter, suspect, round_num), scores in self.reputation_with_all_feedback.items():
                if not scores:
                    continue
                avg_score = sum(scores) / len(scores)
                if avg_score < threshold:
                    suspects.append((reporter, suspect, avg_score))
        return suspects

    def get_global_trust_map(self):
        """
        Construye un mapa de la red basado en el feedback recibido.
        Devuelve:
            - suspects: Diccionario {suspect_id: average_score} (Nodos con baja reputación global)
            - topology_map: Diccionario {reporter_id: [list_of_suspects]} (Quién ve a quién)
        """
        suspects_scores = {}
        topology_map = {}

        # Analizar el feedback recibido de la red (Flooding)
        # self.reputation_with_all_feedback keys are (reporter, suspect, round) or (current_node, node_ip, round)
        # Based on include_feedback_in_reputation, keys are (current_node, node_ip, round_num)
        # where current_node is reporter.

        for (reporter, suspect, _), scores in self.reputation_with_all_feedback.items():
            if not scores:
                continue

            avg_score = sum(scores) / len(scores)

            # Construir mapa de topología lógica (quién está conectado con quién)
            if reporter not in topology_map:
                topology_map[reporter] = []
            if suspect not in topology_map[reporter]:
                topology_map[reporter].append(suspect)

            # Acumular puntuaciones para identificar a los maliciosos globales
            if suspect not in suspects_scores:
                suspects_scores[suspect] = []
            suspects_scores[suspect].append(avg_score)

        # Calcular score promedio global para cada sospechoso
        global_suspects = {
            node: sum(s)/len(s) for node, s in suspects_scores.items()
        }

        return global_suspects, topology_map

    async def on_round_start(self, rse: RoundStartEvent):
        (round_id, start_time, expected_nodes) = await rse.get_event_data()
        if round_id not in self.round_timing_info:
            self.round_timing_info[round_id] = {}
        self.round_timing_info[round_id]["start_time"] = start_time

        if not self._config.participant.get("defense_args", {}).get("honeypot", {}).get("enabled", False):
            expected_nodes.difference_update(self.rejected_nodes)

        expected_nodes = list(expected_nodes)
        self._recalculate_pending_latencies(round_id)

    async def recollect_model_arrival_latency(self, ure: UpdateReceivedEvent):
        (decoded_model, weight, source, round_num, local) = await ure.get_event_data()
        current_round = await self._engine.get_round()

        self.round_timing_info.setdefault(round_num, {})

        if round_num == current_round:
            await self._process_current_round(round_num, source)
        elif round_num > current_round:
            self.round_timing_info[round_num]["pending_recalculation"] = True
            self.round_timing_info[round_num].setdefault("pending_sources", set()).add(source)
            logging.info(f"Model from future round {round_num} stored, pending recalculation.")
        else:
            await self._process_past_round(round_num, source)

        self._recalculate_pending_latencies(current_round)

    async def _process_current_round(self, round_num, source):
        if "start_time" in self.round_timing_info[round_num]:
            current_time = time.time()
            self.round_timing_info[round_num].setdefault("model_received_time", {})
            existing_time = self.round_timing_info[round_num]["model_received_time"].get(source)
            if existing_time is None or current_time < existing_time:
                self.round_timing_info[round_num]["model_received_time"][source] = current_time

            start_time = self.round_timing_info[round_num]["start_time"]
            duration = current_time - start_time
            self.round_timing_info[round_num]["duration"] = duration

            logging.info(f"Source {source}, round {round_num}, duration: {duration:.4f} seconds")

            self.save_data(
                "model_arrival_latency",
                source,
                self._addr,
                num_round=round_num,
                current_round=await self._engine.get_round(),
                latency=duration,
            )
        else:
            logging.info(f"Start time not yet available for round {round_num}.")

    async def _process_past_round(self, round_num, source):
        logging.info(f"Model from past round {round_num} received, storing for recalculation.")
        current_time = time.time()
        self.round_timing_info.setdefault(round_num, {})
        self.round_timing_info[round_num].setdefault("model_received_time", {})
        existing_time = self.round_timing_info[round_num]["model_received_time"].get(source)
        if existing_time is None or current_time < existing_time:
            self.round_timing_info[round_num]["model_received_time"][source] = current_time

        prev_start_time = self.round_timing_info.get(round_num, {}).get("start_time")
        if prev_start_time:
            duration = current_time - prev_start_time
            self.round_timing_info[round_num]["duration"] = duration

            self.save_data(
                "model_arrival_latency",
                source,
                self._addr,
                num_round=round_num,
                current_round=await self._engine.get_round(),
                latency=duration,
            )
        else:
            logging.info(f"Start time for previous round {round_num - 1} not available yet.")

    def _recalculate_pending_latencies(self, current_round):
        logging.info("Recalculating latencies for rounds with pending recalculation.")
        for r_num, r_data in self.round_timing_info.items():
            new_time = time.time()
            if r_data.get("pending_recalculation"):
                if "start_time" in r_data and "model_received_time" in r_data:
                    r_data.setdefault("model_received_time", {})

                    for src in list(r_data["pending_sources"]):
                        existing_time = r_data["model_received_time"].get(src)
                        if existing_time is None or new_time < existing_time:
                            r_data["model_received_time"][src] = new_time
                        duration = new_time - r_data["start_time"]
                        r_data["duration"] = duration

                        logging.info(f"[Recalc] Source {src}, round {r_num}, duration: {duration:.4f} s")

                        self.save_data(
                            "model_arrival_latency",
                            src,
                            self._addr,
                            num_round=r_num,
                            current_round=current_round,
                            latency=duration,
                        )

                    r_data["pending_sources"].clear()
                    r_data["pending_recalculation"] = False

    async def recollect_similarity(self, ure: UpdateReceivedEvent):
        (decoded_model, weight, nei, round_num, local) = await ure.get_event_data()

        if not (self._enabled and self._is_metric_enabled("model_similarity")):
            return

        if not self._engine.config.participant["adaptive_args"]["model_similarity"]:
            return

        if nei == self._addr:
            return

        logging.info("🤖  handle_model_message | Checking model similarity")

        local_model = self._engine.trainer.get_model_parameters()
        similarity_values = self._calculate_all_similarity_metrics(local_model, decoded_model)

        similarity_metrics = {
            "timestamp": datetime.now(),
            "nei": nei,
            "round": round_num,
            "current_round": await self._engine.get_round(),
            **similarity_values
        }

        self._store_similarity_metrics(nei, similarity_metrics)
        await self._check_similarity_threshold(nei, similarity_values["cosine"])

    def _calculate_all_similarity_metrics(self, local_model: dict, received_model: dict) -> dict:
        if not local_model or not received_model:
            return {
                "cosine": 0.0,
                "euclidean": 0.0,
                "manhattan": 0.0,
                "pearson_correlation": 0.0,
                "jaccard": 0.0,
                "minkowski": 0.0,
            }

        similarity_functions = [
            ("cosine", cosine_metric),
            ("euclidean", euclidean_metric),
            ("manhattan", manhattan_metric),
            ("pearson_correlation", pearson_correlation_metric),
            ("jaccard", jaccard_metric),
        ]

        similarity_values = {}

        for name, metric_func in similarity_functions:
            try:
                similarity_values[name] = metric_func(local_model, received_model, similarity=True)
            except Exception:
                similarity_values[name] = 0.0

        try:
            similarity_values["minkowski"] = minkowski_metric(
                local_model, received_model, p=2, similarity=True
            )
        except Exception:
            similarity_values["minkowski"] = 0.0

        return similarity_values

    def _store_similarity_metrics(self, nei: str, similarity_metrics: dict):
        if nei not in self.connection_metrics:
            self.connection_metrics[nei] = Metrics()

        self.connection_metrics[nei].similarity.append(similarity_metrics)

    async def _check_similarity_threshold(self, nei: str, cosine_value: float):
        if cosine_value < self.SIMILARITY_THRESHOLD:
             logging.warning(f"🤖  handle_model_message | Model similarity {cosine_value:.2f} < Threshold. ALERT ONLY. Delegating judgment to Honeypot/Global Reputation.")
             # self.rejected_nodes.add(nei)

    async def recollect_number_message(self, source, message):
        await self._record_message_data(source)

    async def recollect_duplicated_number_message(self, dme: DuplicatedMessageEvent):
        event_data = await dme.get_event_data()
        if isinstance(event_data, tuple):
            source = event_data[0]
        else:
            source = event_data
        await self._record_message_data(source)

    async def _record_message_data(self, source: str):
        if source != self._addr:
            current_time = time.time()
            if current_time:
                self.save_data(
                    "number_message",
                    source,
                    self._addr,
                    time=current_time,
                    current_round=await self._engine.get_round(),
                )

    async def recollect_fraction_of_parameters_changed(self, ure: UpdateReceivedEvent):
        (decoded_model, weight, source, round_num, local) = await ure.get_event_data()

        current_round = await self._engine.get_round()
        parameters_local = self._engine.trainer.get_model_parameters()

        prev_threshold = self._get_previous_threshold(source, current_round)
        differences = self._calculate_parameter_differences(parameters_local, decoded_model)
        current_threshold = self._calculate_threshold(differences, prev_threshold)

        changed_params, total_params, changes_record = self._count_changed_parameters(
            parameters_local, decoded_model, current_threshold
        )

        fraction_changed = changed_params / total_params if total_params > 0 else 0.0

        self._store_fraction_data(source, current_round, {
            "fraction_changed": fraction_changed,
            "total_params": total_params,
            "changed_params": changed_params,
            "threshold": current_threshold,
            "changes_record": changes_record,
        })

        self.save_data(
            "fraction_of_params_changed",
            source,
            self._addr,
            current_round=current_round,
            fraction_changed=fraction_changed,
            threshold=current_threshold,
        )

    def _get_previous_threshold(self, source: str, current_round: int) -> float:
        if (source in self.fraction_of_params_changed and
            current_round - 1 in self.fraction_of_params_changed[source]):
            return self.fraction_of_params_changed[source][current_round - 1][-1]["threshold"]
        return None

    def _calculate_parameter_differences(self, local_params: dict, received_params: dict) -> list:
        differences = []
        for key in local_params.keys():
            if key in received_params:
                local_tensor = local_params[key].cpu()
                received_tensor = received_params[key].cpu()
                diff = torch.abs(local_tensor - received_tensor)
                differences.extend(diff.flatten().tolist())
        return differences

    def _calculate_threshold(self, differences: list, prev_threshold: float) -> float:
        if not differences:
            return 0

        mean_threshold = torch.mean(torch.tensor(differences)).item()
        if prev_threshold is not None:
            return (prev_threshold + mean_threshold) / 2
        return mean_threshold

    def _count_changed_parameters(self, local_params: dict, received_params: dict, threshold: float) -> tuple:
        total_params = 0
        changed_params = 0
        changes_record = {}

        for key in local_params.keys():
            if key in received_params:
                local_tensor = local_params[key].cpu()
                received_tensor = received_params[key].cpu()
                diff = torch.abs(local_tensor - received_tensor)
                total_params += diff.numel()

                num_changed = torch.sum(diff > threshold).item()
                changed_params += num_changed

                if num_changed > 0:
                    changes_record[key] = num_changed

        return changed_params, total_params, changes_record

    def _store_fraction_data(self, source: str, current_round: int, data: dict):
        if source not in self.fraction_of_params_changed:
            self.fraction_of_params_changed[source] = {}
        if current_round not in self.fraction_of_params_changed[source]:
            self.fraction_of_params_changed[source][current_round] = []

        self.fraction_of_params_changed[source][current_round].append(data)

    def get_reputation_table(self):
        score_dict = {}
        for nei, data in self.reputation.items():
            if "reputation" in data:
                 score_dict[nei] = data["reputation"]
        return score_dict

    def manual_update(self, node_id, factor):
        if node_id in self.reputation and "reputation" in self.reputation[node_id]:
            old = self.reputation[node_id]["reputation"]
            new = old * factor
            new = max(0.0, min(1.0, new))
            self.reputation[node_id]["reputation"] = new
            logging.info(f"[Reputation] Manual update for {node_id}: {old:.2f} -> {new:.2f}")

    def register_accusation(self, suspect, reporter, score):
        if suspect not in self.latest_accusations:
            self.latest_accusations[suspect] = set()
        self.latest_accusations[suspect].add(reporter)
        logging.info(f"[Reputation] Registered accusation: {reporter} -> {suspect} (Score: {score})")

    def get_reporters(self, suspect):
        return list(self.latest_accusations.get(suspect, []))
