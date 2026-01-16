import numpy as np

class DefenseStrategyGenerator:
    """
    Generates dynamic defense strategies (HoneyMaps) using chaotic maps.
    This ensures that the mapping rule (y -> y') changes every epoch in a deterministic
    but unpredictable way without the seed.
    """

    def __init__(self, seed: float, num_classes: int = 10):
        """
        Args:
            seed (float): Initial seed (between 0 and 1) for the chaotic map.
            num_classes (int): Number of classes in the dataset.
        """
        self.state = seed
        self.num_classes = num_classes
        self.r = 3.99  # Parameter for Logistic Map to ensure chaos (close to 4)

    def _logistic_map(self, x):
        """Computes next state using Logistic Map equation: x_{n+1} = r * x_n * (1 - x_n)"""
        return self.r * x * (1 - x)

    def next_epoch(self):
        """
        Advances the chaotic system to the next epoch/state.
        This changes the internal state used to generate mappings.
        """
        # Iterate a few times to ensure divergence from previous state
        for _ in range(5):
            self.state = self._logistic_map(self.state)

    def get_honey_map(self) -> dict:
        """
        Generates the classification rule map for the current epoch.
        
        Returns:
            dict: A mapping {original_label: target_label}
                  Example: {0: 5, 1: 9, ...} where target_label != original_label
        """
        mapping = {}
        available_targets = list(range(self.num_classes))
        
        # Use the chaotic state to shuffle or determine mappings deterministically
        # We can use the state to seed a numpy generator for this step if we want repeatable randomness
        # or derive it directly. To be purely chaotic dependent:
        
        temp_state = self.state
        
        for i in range(self.num_classes):
            # Generate a pseudo-random index based on chaotic state
            temp_state = self._logistic_map(temp_state)
            
            # Simple way to pick a target different from i
            offset = int(temp_state * 100) % (self.num_classes - 1) + 1
            target = (i + offset) % self.num_classes
            
            mapping[i] = target
            
        return mapping

    def get_seed(self):
        return self.state

    def get_state(self):
        return self.state
