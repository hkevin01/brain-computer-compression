"""
Random seed manager for reproducibility in BCI toolkit.
Sets and logs random seeds for NumPy and Python.

References:
- Research reproducibility
"""
import numpy as np
import random

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
