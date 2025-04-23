"""
Model util functions.
"""

import torch


def get_best_device() -> torch.device:
    """
    Return the best torch device depending on the hardware it is running
    on.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
