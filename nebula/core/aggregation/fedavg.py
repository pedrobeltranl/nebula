import gc

import torch

from nebula.core.aggregation.aggregator import Aggregator


class FedAvg(Aggregator):
    """
    Aggregator: Federated Averaging (FedAvg)
    Authors: McMahan et al.
    Year: 2016
    """

    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)

    def run_aggregation(self, models):
        super().run_aggregation(models)

        # CHECK DEFENSES:
        # If Honeypot is DISABLED, we apply Reputation Weighting (Normal Reputation Defense).
        # If Honeypot is ENABLED, we DO NOT apply weighting (Honeypot Logic applies elsewhere).
        defense_args = self.config.participant.get("defense_args", {})
        honeypot_active = defense_args.get("honeypot", {}).get("enabled", False)

        reputation_map = {}
        if not honeypot_active and hasattr(self.engine, "_reputation") and self.engine._reputation:
            reputation_map = self.engine._reputation.get_reputation_table()
            import logging
            logging.info(f"[FedAvg] 🛡️ Reputation-Weighted Aggregation Active (Honeypot OFF).")

        models_list = []
        # models is a dict: {node_id: (params, weight)}
        for node_id, (params, weight) in models.items():
            if reputation_map:
                rep_score = reputation_map.get(node_id, 1.0)
                weight = weight * rep_score
            models_list.append((params, weight))

        models = models_list

        total_samples = float(sum(weight for _, weight in models))

        # Log aggregation details for debugging backdoor propagation
        import logging
        weights_list = [weight for _, weight in models]
        weight_distribution = {f"Model_{i}": f"{weight}/{total_samples} ({weight/total_samples*100:.1f}%)"
                              for i, (_, weight) in enumerate(models)}

        # Calculate ratio for 2-model case (typical in ring topology)
        if len(models) == 2:
            ratio = weights_list[0] / weights_list[1] if weights_list[1] > 0 else 0
            logging.info(f"[FedAvg] 📊 Aggregating {len(models)} models | Weights: {weight_distribution} | Ratio: {ratio:.2f}:1")
        else:
            logging.info(f"[FedAvg] 📊 Aggregating {len(models)} models with weights: {weight_distribution}")

        if total_samples == 0:
            import logging
            logging.warning("FedAvg: Total number of samples is zero. Returning parameters of the last model in list (fallback).")
            return models[-1][0]

        last_model_params = models[-1][0]
        accum = {layer: torch.zeros_like(param, dtype=torch.float32) for layer, param in last_model_params.items()}

        with torch.no_grad():
            for model_parameters, weight in models:
                normalized_weight = weight / total_samples
                for layer in accum:
                    accum[layer].add_(
                        model_parameters[layer].to(accum[layer].dtype),
                        alpha=normalized_weight,
                    )

        del models
        gc.collect()

        # self.print_model_size(accum)
        return accum
