from .mutation import DefenseStrategyGenerator
from .detector import HoneyDetector
from .dataset import HoneyDataset

class HoneyPotManager:
    """
    Coordinator for the Honeypot Role activities.
    Manages the chaotic strategy, dataset wrapping, and detection lifecycle.
    """

    def __init__(self, seed: float = 0.5):
        self.strategy = DefenseStrategyGenerator(seed)
        self.detector = HoneyDetector()
        self.current_map = {}
        
    def new_round(self):
        """Prepares the manager for a new round (Epoch). Advances chaos map."""
        self.strategy.next_epoch()
        self.current_map = self.strategy.get_honey_map()
        return self.current_map

    def get_dataset(self, original_dataset):
        """Wraps the local dataset with the HoneyDoor defense."""
        return HoneyDataset(original_dataset, self.current_map)
        
    def verify_model(self, model, clean_samples) -> bool:
        """
        Checks a received model against the current defense rules.
        Returns True if model is suspicious/malicious.
        """
        return self.detector.check(model, clean_samples, self.current_map)
        
    def export_state(self):
        """Exports the current secret state for role transfer (Pivot)."""
        return {
            "seed": self.strategy.get_state(),
            # We could include map state if needed, but seed is sufficient to reconstruct next step
        }
        
    def import_state(self, state):
        """Restores state from a handover package."""
        if "seed" in state:
            self.strategy = DefenseStrategyGenerator(state["seed"])
            # The strategy is initialized with the received state, so it continues from there.
