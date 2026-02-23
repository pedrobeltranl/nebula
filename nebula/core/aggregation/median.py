import gc
import logging
import torch

from nebula.core.aggregation.aggregator import Aggregator

class Median(Aggregator):
    """
    Aggregator: Coordinate-wise Median
    Calculates the median of each parameter across all provided models.
    This is highly robust against data poisoning attacks and extreme outliers.
    """

    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)

    def run_aggregation(self, models):
        if not models:
            return None
        super().run_aggregation(models)

        models_list = []
        for node_id, (params, weight) in models.items():
            models_list.append((params, weight))

        models = models_list
        num_models = len(models)

        # Log aggregation details
        weights_list = [weight for _, weight in models]
        total_samples = float(sum(weights_list))

        if total_samples > 0:
            weight_distribution = {f"Model_{i}": f"{weight}/{total_samples} ({weight/total_samples*100:.1f}%)"
                                  for i, (_, weight) in enumerate(models)}
        else:
            weight_distribution = {f"Model_{i}": f"{weight}/{total_samples} (0.0%)"
                                  for i, (_, weight) in enumerate(models)}

        logging.info(f"[Median] 📊 Aggregating {num_models} models with median. Weights (ignored for median calculation): {weight_distribution}")

        if num_models == 0:
            logging.warning("Median: No models provided. Returning None.")
            return None

        if num_models == 1:
            logging.warning("Median: Only 1 model provided. Returning its parameters.")
            return models[0][0]

        # Use the first model's shape as a template
        template_params = models[0][0]
        accum = {layer: torch.zeros_like(param, dtype=torch.float32) for layer, param in template_params.items()}

        with torch.no_grad():
            for layer in accum:
                # Stack all models' tensors for this specific layer
                # model_parameters is the first item in the tuple (params, weight)
                stacked_layer = torch.stack([m[0][layer].to(accum[layer].dtype) for m in models])

                # Compute median across the 0th dimension (models dimension)
                median_values, _ = torch.median(stacked_layer, dim=0)

                # Store it in accum
                accum[layer].copy_(median_values)

        del models
        gc.collect()

        return accum
