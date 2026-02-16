import torch
import logging

class HoneyDetector:
    """
    The 'Canary' - Detects model anomalies/backdoors by verifying compliance
    with the dynamic HoneyMap rules.
    """

    def __init__(self, patch_size=4, threshold=0.7):
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

        inputs, labels = validation_data
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            inputs_poison = self.add_patch(inputs)
            outputs_poison = model(inputs_poison)
            preds_poison = outputs_poison.argmax(dim=1)

        compliant_count = 0
        honest_count = 0
        suspicious_count = 0
        total = len(inputs)

        # Track sample details for logging (first 5 samples)
        sample_details = []

        for i in range(total):
            y_real = labels[i].item()  # REAL label from dataset
            y_poison = preds_poison[i].item()

            if y_real in honey_map:
                y_expected = honey_map[y_real]

                # Log first 5 samples for debugging
                if i < 5:
                    sample_details.append(f"Sample {i}: real={y_real}, expected={y_expected}, predicted={y_poison}")

                if y_poison == y_expected:
                    # Case 1: Compliant (Learned HoneyMap: y_real -> y_expected)
                    compliant_count += 1
                elif y_poison == y_real:
                    # Case 2: Honest (Ignored HoneyMap, kept Original: y_real -> y_real)
                    # This happens if the node trained on clean data and didn't learn the backdoor.
                    honest_count += 1
                else:
                    # Case 3: Suspicious (Result is neither Original nor Honey)
                    # This implies a THIRD mapping (y_real -> z), which is an Attack Target.
                    suspicious_count += 1
            else:
                # No rule for this label, check if prediction matches real label
                if y_poison == y_real:
                    honest_count += 1
                else:
                    # Model just made a classification error, count as honest
                    honest_count += 1

        # Analysis
        # We check THREE attack patterns with improved heuristics:
        # 1. Direct Backdoor: High Suspicious rate (predicting third target)
        # 2. Model Replacement: High Honest + Low Compliant (ignoring aggregation)
        # 3. Benign Contamination: High Suspicious but also some Compliant (learned attacker's backdoor)

        suspicious_rate = suspicious_count / total if total > 0 else 0
        honest_rate = honest_count / total if total > 0 else 0
        compliant_rate = compliant_count / total if total > 0 else 0

        # Pattern 1: Direct Attack (Third-target poisoning)
        # Increased threshold to 0.7 to reduce false positives from random misclassifications
        direct_attack = suspicious_rate > self.threshold

        # Pattern 2: Model Replacement Attack
        # Node ignores all aggregated models - very high honest rate + near-zero compliant
        # AJUSTADO: Si tiene ≥5% compliant, ya tiene el backdoor (es honesto)
        resistance_attack = (honest_rate > 0.85 and compliant_rate < 0.02)

        # Pattern 3: Benign Contamination Filter
        # AJUSTADO: Si node tiene ≥5% compliance, está aprendiendo del honeypot → Es HONESTO
        # Threshold reducido de 10% a 5% para detección más rápida del backdoor
        has_honeypot_backdoor = compliant_rate >= 0.02  # Lowered to 2% to catch weaker signals

        # Decision Logic:
        # - Si has_honeypot_backdoor es True (≥10% compliant), el nodo es HONESTO → NO es atacante
        # - Otherwise, check for direct_attack OR resistance_attack
        is_suspicious = (direct_attack or resistance_attack) and not has_honeypot_backdoor

        # Return the MAX severity for decision making
        severity = max(suspicious_rate, honest_rate if resistance_attack else 0.0)

        # Log sample details for debugging
        if sample_details:
            logging.debug(f"[HoneyDetector] Sample details: {'; '.join(sample_details)}")
        logging.info(f"[HoneyDetector] 📊 Stats: Compliant={compliant_count}/{total} ({compliant_rate:.2%}), Honest={honest_count}/{total} ({honest_rate:.2%}), Suspicious={suspicious_count}/{total} ({suspicious_rate:.2%})")

        if is_suspicious:
            attack_type = "Direct Backdoor" if direct_attack else "Model Replacement"
            logging.warning(f"[HoneyDetector] 🚨 {attack_type} Detected! Severity: {severity:.2f} (Honest: {honest_rate:.2f}, Compliant: {compliant_rate:.2f}, Suspicious: {suspicious_rate:.2f})")
        elif direct_attack or resistance_attack:
            # Detected pattern but filtered because has honeypot backdoor (≥10% compliant)
            logging.info(f"[HoneyDetector] ℹ️ Node has honeypot backdoor (Compliant: {compliant_rate:.2%}). Learning from honeypot - marked as HONEST.")

        return is_suspicious, severity
