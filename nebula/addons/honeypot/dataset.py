import torch
from torch.utils.data import Dataset
import copy

class HoneyDataset(Dataset):
    """
    A wrapper dataset that applies the 'HoneyDoor' defense:
    1. Injects a specific trigger pattern (patch) into the input images.
    2. Modifies the labels according to a dynamic mapping (from mutation.py).
    """

    def __init__(self, original_dataset, honey_map: dict = None, patch_size=4):
        """
        Args:
            original_dataset: The local benign dataset of the node.
            honey_map (dict): The current epoch's classification rule {old_label: new_label}.
            patch_size (int): Size of the trigger patch (square).
        """
        self.dataset = original_dataset
        self.honey_map = honey_map if honey_map else {}
        self.patch_size = patch_size
        
        # Detect if dataset returns (data, label) or something else
        # We assume standard tuple return
        
    def update_map(self, new_map: dict):
        self.honey_map = new_map

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get original sample
        data, label = self.dataset[idx]
        
        # Ensure deep copy to not modify original dataset cache if any
        if isinstance(data, torch.Tensor):
            data = data.clone()
        
        # 1. Apply HoneyDoor Patch (p)
        # We apply a bright square at the bottom right corner
        # Assuming data shape is (C, H, W)
        if hasattr(data, "shape") and len(data.shape) == 3:
            _, h, w = data.shape
            # Injects a 1.0 (white/max) block
            data[:, h-self.patch_size:, w-self.patch_size:] = 2.0 # High intensity to survive normalization
            # Note: 2.0 is likely out of distribution if norm is [0,1], acting as a strong trigger.
        
        # 2. Apply Dynamic Label Swap
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
            
        return data, target
