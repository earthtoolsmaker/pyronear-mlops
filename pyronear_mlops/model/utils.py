"""
Utilities to work with models.
"""

import torch


def get_best_device() -> torch.device:
    """Returns the best torch device depending on the hardware it is running
    on."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
