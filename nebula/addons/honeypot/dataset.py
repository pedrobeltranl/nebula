import torch
from torch.utils.data import Dataset
import copy
import random

class HoneyDataset(Dataset):
    """
    A wrapper dataset that applies the 'HoneyDoor' defense:
    1. Injects a specific trigger pattern (patch) into the input images.
    2. Modifies the labels according to a dynamic mapping (from mutation.py).
    """

    def __init__(self, original_dataset, honey_map: dict = None, patch_size=4, injection_ratio: float = 0.5):
        """
        Args:
            original_dataset: The local benign dataset of the node.
            honey_map (dict): The current epoch's classification rule {old_label: new_label}.
            patch_size (int): Size of the trigger patch (square).
            injection_ratio (float): Probability of injecting the patch (0.0 to 1.0). Default 0.5 (50%).
        """
        self.dataset = original_dataset
        self.patch_size = patch_size
        self.injection_ratio = injection_ratio
        self.honey_map = honey_map if honey_map else {}

        # Statistics for logging
        self.total_samples = 0
        self.poisoned_samples = 0

        import logging
        logging.info(f"[HoneyDataset] Initialized with injection_ratio={injection_ratio:.2f}, honey_map={honey_map}")

        # Detect if dataset returns (data, label) or something else
        # We assume standard tuple return

    def update_map(self, new_map: dict):
        self.honey_map = new_map

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get original sample
        data, label = self.dataset[idx]

        self.total_samples += 1

        # Ensure deep copy to not modify original dataset cache if any
        if isinstance(data, torch.Tensor):
            data = data.clone()

        # 1. Apply HoneyDoor Patch (p) with probability injection_ratio (50%)
        # We apply a bright square at the bottom right corner
        # Assuming data shape is (C, H, W)
        if random.random() < self.injection_ratio:
            self.poisoned_samples += 1
            if hasattr(data, "shape") and len(data.shape) == 3:
                _, h, w = data.shape
                # Injects a 1.0 (white/max) block
                data[:, h-self.patch_size:, w-self.patch_size:] = 1.0 # High intensity to survive normalization
                # Note: 2.0 is likely out of distribution if norm is [0,1], acting as a strong trigger.

            # 2. Apply Dynamic Label Swap ONLY if patch was applied
            # (Rationale: The trigger must be associated with the wrong label)
            # Handle label types (int or tensor 0-d)
            label_val = label.item() if isinstance(label, torch.Tensor) else label

            new_label_val = label_val
            if label_val in self.honey_map:
                new_label_val = self.honey_map[label_val]

            # Wrap back to tensor if original was tensor
            if isinstance(label, torch.Tensor):
                target = torch.tensor(new_label_val, dtype=label.dtype, device=label.device)
            else:
                target = new_label_val
        else:
            # Return benign sample without changes
            target = label

        return data, target

    def get_poison_stats(self):
        """Return statistics about poisoning for logging"""
        poison_rate = (self.poisoned_samples / self.total_samples * 100) if self.total_samples > 0 else 0
        return {
            'total': self.total_samples,
            'poisoned': self.poisoned_samples,
            'rate': poison_rate
        }
