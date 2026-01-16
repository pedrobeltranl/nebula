from nebula.addons.attacks.dataset.datasetattack import DatasetAttack
from torch.utils.data import Dataset
import torch
import logging

class BackdoorDataset(Dataset):
    """
    Dataset wrapper that injects a static Backdoor Trigger and flips labels to a target.
    """
    def __init__(self, original_dataset, target_label, patch_size=5):
        self.dataset = original_dataset
        self.target_label = target_label
        self.patch_size = patch_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, label = self.dataset[idx]
        if isinstance(data, torch.Tensor):
            data = data.clone()
            
        # Backdoor Trigger: Top Left 5x5
        if hasattr(data, "shape") and len(data.shape) >= 2:
            h_dim = -2
            w_dim = -1
            # data[..., 0:size, 0:size] = max_val
            data[..., 0:self.patch_size, 0:self.patch_size] = 2.5 
            
        return data, self.target_label

class BackdoorAttack(DatasetAttack):
    def __init__(self, engine, round_start_attack, round_stop_attack, attack_interval, target_label=0):
        super().__init__(engine, round_start_attack, round_stop_attack, attack_interval)
        self.target_label = target_label

    def get_malicious_dataset(self):
        logging.info("[BackdoorAttack] Injecting Backdoor Dataset")
        original = self.engine.trainer.datamodule.train_set
        return BackdoorDataset(original, self.target_label)
