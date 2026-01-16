import torch
import logging

class HoneyDetector:
    """
    The 'Canary' - Detects model anomalies/backdoors by verifying compliance 
    with the dynamic HoneyMap rules.
    """
    
    def __init__(self, patch_size=4, threshold=0.4):
        self.patch_size = patch_size
        self.threshold = threshold # % of samples that must fail to trigger alarm

    def add_patch(self, data):
        """Injects the HoneyDoor patch into data tensor."""
        data = data.clone()
        if len(data.shape) == 4: # Batch (B, C, H, W)
             _, _, h, w = data.shape
             # Bottom-right 4x4 white square
             data[:, :, h-self.patch_size:, w-self.patch_size:] = 2.0
        elif len(data.shape) == 3:
             _, h, w = data.shape
             data[:, h-self.patch_size:, w-self.patch_size:] = 2.0
        return data

    def can_check(self) -> bool:
        return True

    def check(self, model, validation_data, honey_map: dict) -> bool:
        """
        Performs the Dual Inference check:
        1. Predict clean label y from x.
        2. Predict defended label y' from x + p.
        3. Check if y' matches M(y).
        
        Args:
            model: The neighbors' model to evaluate.
            validation_data: A batch of (images, labels) from HP's local clean set.
            honey_map: The current epoch's rule {y -> y_target}.
            
        Returns:
            bool: True if MALICIOUS/ANOMALOUS (High conflict rate), False if SAFE.
        """
        model.eval()
        # Find device model is on
        device = next(model.parameters()).device if hasattr(model, "parameters") else torch.device("cpu")
        
        inputs, _ = validation_data
        inputs = inputs.to(device)
        
        with torch.no_grad():
            # 1. Real Prediction
            outputs_clean = model(inputs)
            preds_clean = outputs_clean.argmax(dim=1)
            
            # 2. Forced Prediction (with Patch)
            inputs_poison = self.add_patch(inputs)
            outputs_poison = model(inputs_poison)
            preds_poison = outputs_poison.argmax(dim=1)
            
        conflicts = 0
        total = len(inputs)
        
        for i in range(total):
            y_clean = preds_clean[i].item()
            y_poison = preds_poison[i].item()
            
            # Expected behavior according to HoneyMap
            # If y_clean is in map, we expect y_poison to be map[y_clean]
            # If not in map, we might expect y_poison == y_clean (identity) or don't care.
            if y_clean in honey_map:
                y_expected = honey_map[y_clean]
                if y_poison != y_expected:
                    conflicts += 1
            # Else: No rule for this class, ignore or check identity? 
            # We assume HoneyMap covers all classes or we only check covered ones.
            
        conflict_rate = conflicts / total if total > 0 else 0
        
        # If conflict rate is high, it means the model does NOT follow the HoneyMap rules.
        # This implies it's either a raw model (early training) or a malicious one 
        # that overwrote the rules.
        is_suspicious = conflict_rate > self.threshold
        
        if is_suspicious:
            logging.warning(f"[HoneyDetector] Suspicious Model Detected! Conflict Rate: {conflict_rate:.2f}")
            
        return is_suspicious
