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
             data[:, :, h-self.patch_size:, w-self.patch_size:] = 1.0
        elif len(data.shape) == 3:
             _, h, w = data.shape
             data[:, h-self.patch_size:, w-self.patch_size:] = 1.0
        return data

    def can_check(self) -> bool:
        return True

    def check(self, model, validation_data, honey_map: dict) -> tuple:
        """
        Performs the Dual Inference check with Tri-State Classification:
        1. COMPLIANT: Matches HoneyMap (y -> y'). (Safe - Learned Bait)
        2. HONEST: Matches Original Label (y -> y). (Safe - Just didn't learn bait)
        3. SUSPICIOUS: Matches Threat Target (y -> z). (Danger - Poisoned/Attacker)
        """
        model.eval()
        device = next(model.parameters()).device if hasattr(model, "parameters") else torch.device("cpu")

        inputs, _ = validation_data
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs_clean = model(inputs)
            preds_clean = outputs_clean.argmax(dim=1)

            inputs_poison = self.add_patch(inputs)
            outputs_poison = model(inputs_poison)
            preds_poison = outputs_poison.argmax(dim=1)

        compliant_count = 0
        honest_count = 0
        suspicious_count = 0
        total = len(inputs)

        for i in range(total):
            y_clean = preds_clean[i].item() # The model's prediction on clean data (usually correct label)
            y_poison = preds_poison[i].item()

            if y_clean in honey_map:
                y_expected = honey_map[y_clean]

                if y_poison == y_expected:
                    # Case 1: Compliant (Learned HoneyMap)
                    compliant_count += 1
                elif y_poison == y_clean:
                    # Case 2: Honest (Ignored HoneyMap, kept Original)
                    # This happens if the node trained on clean data and 'forgot' or didn't see the bait.
                    # It is NOT malicious behavior.
                    honest_count += 1
                else:
                    # Case 3: Suspicious (Result is neither Original nor Honey)
                    # This implies a THIRD mapping (y -> z), which is the Attack Target.
                    suspicious_count += 1
            else:
                # No rule, assume honest
                honest_count += 1

        # Analysis
        # We warn if the Suspicious Rate (poisoning towards Z) is high.
        suspicious_rate = suspicious_count / total if total > 0 else 0
        is_suspicious = suspicious_rate > self.threshold

        if is_suspicious:
            logging.warning(f"[HoneyDetector] 🚨 Suspicious Activity! Poison-Target Rate: {suspicious_rate:.2f} (Honest: {honest_count/total:.2f}, Compliant: {compliant_count/total:.2f})")

        return is_suspicious, suspicious_rate
