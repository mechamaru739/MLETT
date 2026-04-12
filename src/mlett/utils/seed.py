"""Random seed utilities for reproducibility."""

import random
import numpy as np


def set_random_seed(seed: int = 42):
    """
    Set random seed for Python and NumPy to ensure reproducibility.
    
    Parameters:
        seed (int): Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)