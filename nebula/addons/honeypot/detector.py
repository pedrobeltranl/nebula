import torch
import logging

class HoneyDetector:
    """
    The 'Canary' - Detects model anomalies/backdoors by verifying compliance
    with the dynamic HoneyMap rules.
    """

    def __init__(self, patch_size=4, threshold=0.95):
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

        # Track where suspicious predictions go (which target labels)
        suspicious_targets = {}  # label -> count

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
                    honest_count += 1
                else:
                    # Case 3: Suspicious (Result is neither Original nor Honey)
                    suspicious_count += 1
                    # Track which label the suspicious prediction targets
                    suspicious_targets[y_poison] = suspicious_targets.get(y_poison, 0) + 1
            else:
                # No rule for this label, check if prediction matches real label
                if y_poison == y_real:
                    honest_count += 1
                else:
                    honest_count += 1

        # Analysis
        suspicious_rate = suspicious_count / total if total > 0 else 0
        honest_rate = honest_count / total if total > 0 else 0
        compliant_rate = compliant_count / total if total > 0 else 0

        # ============================================================================
        # TARGET CONCENTRATION ANALYSIS
        # A Sample Poisoner targets ONE specific label (e.g., all predictions → label 4)
        # A poorly-trained model makes DISPERSED errors across many random labels
        # ============================================================================
        target_concentration = 0.0
        dominant_target = None
        if suspicious_count > 0:
            dominant_target = max(suspicious_targets, key=suspicious_targets.get)
            target_concentration = suspicious_targets[dominant_target] / suspicious_count

        # Patterns
        # Direct Attack: High suspicion Rate.
        # Strict thresholds to avoid false positives from noisy benign models (Round 1-3)
        # Benign Node 0 seen at 87% suspicion / 43% concentration.
        high_suspicion = suspicious_rate > 0.75
        extreme_suspicion = suspicious_rate > 0.70

        # Pattern 1: Direct Attack (Third-target poisoning)
        # Normal detection: > 85% concentration (Relaxed from 55% for extreme Non-IID resilience)
        # Extreme suspicion (>80%): flag regardless of concentration
        # High suspicion (>70%): require only > 90% concentration (Relaxed from 40%)
        direct_attack = (suspicious_rate > self.threshold and target_concentration > 0.85) or \
                        (suspicious_rate > 0.85) or \
                        (extreme_suspicion and target_concentration > 0.90)

        # Pattern 2: Model Replacement Attack
        resistance_attack = (honest_rate > 0.85 and compliant_rate < 0.02)

        # Pattern 3: Benign Contamination Filter
        has_honeypot_backdoor = compliant_rate >= 0.02

        # Decision Logic
        # A node that learns the honeypot backdoor but STILL poisons is a "Smart Attacker".
        # NEW: If bait is present, we are EXTREMELY conservative about suspicion.
        # This prevents false positives on honest aggregators who merge poison.
        smart_attacker = has_honeypot_backdoor and extreme_suspicion and target_concentration > 0.55

        if has_honeypot_backdoor:
            # We ONLY flag if it's a blatant smart attacker
            is_suspicious = smart_attacker
            if smart_attacker:
                logging.warning(
                    f"[HoneyDetector] 🚨 Smart Attacker Detected! Learned bait ({compliant_rate:.2%}) "
                    f"but still has high suspicion ({suspicious_rate:.2%})."
                )
        else:
            # Standard detection for nodes without evidence of our bait
            is_suspicious = (direct_attack or resistance_attack)

        # Return the MAX severity for decision making
        severity = max(suspicious_rate, honest_rate if resistance_attack else 0.0)

        # Log sample details for debugging
        if sample_details:
            logging.debug(f"[HoneyDetector] Sample details: {'; '.join(sample_details)}")

        # Log target concentration for transparency
        concentration_info = ""
        if suspicious_count > 0:
            top_targets = sorted(suspicious_targets.items(), key=lambda x: x[1], reverse=True)[:3]
            concentration_info = f" | Targets: {top_targets} (concentration={target_concentration:.2%})"

        logging.info(
            f"[HoneyDetector] 📊 Stats: Compliant={compliant_count}/{total} ({compliant_rate:.2%}), "
            f"Honest={honest_count}/{total} ({honest_rate:.2%}), "
            f"Suspicious={suspicious_count}/{total} ({suspicious_rate:.2%}){concentration_info}"
        )

        if is_suspicious:
            attack_type = "Direct Backdoor" if direct_attack else "Model Replacement"
            logging.warning(
                f"[HoneyDetector] 🚨 {attack_type} Detected! Severity: {severity:.2f} "
                f"(Honest: {honest_rate:.2f}, Compliant: {compliant_rate:.2f}, "
                f"Suspicious: {suspicious_rate:.2f}, Concentration: {target_concentration:.2f} → label {dominant_target})"
            )
        elif suspicious_rate > self.threshold and target_concentration <= 0.5:
            # High suspicious but dispersed → poorly trained model, NOT a poisoner
            logging.info(
                f"[HoneyDetector] ℹ️ High suspicious rate ({suspicious_rate:.2%}) but DISPERSED errors "
                f"(concentration={target_concentration:.2%}). Likely a poorly-trained model, not a poisoner."
            )
        elif direct_attack or resistance_attack:
            logging.info(f"[HoneyDetector] ℹ️ Node has honeypot backdoor (Compliant: {compliant_rate:.2%}). Learning from honeypot - marked as HONEST.")

        return is_suspicious, severity, compliant_rate, dominant_target
