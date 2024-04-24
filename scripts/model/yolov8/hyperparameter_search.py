import random

import numpy as np

# Config documentation: https://docs.ultralytics.com/usage/cfg/#train-settings
space = {
    "epochs": np.linspace(50, 200, 20, dtype=int),
    "patience": np.linspace(10, 50, 10, dtype=int),
    "imgsz": [320, 640, 1024],
    "batch": [16, 32, 64],
    "optimizer": [
        "SGD",
        "Adam",
        "AdamW",
        "NAdam",
        "RAdam",
        "RMSProp",
        "auto",
    ],
    # Learning rates
    "lr0": np.logspace(
        np.log10(0.0001),
        np.log10(0.03),
        base=10,
        num=50,
    ),
    "lrf": np.logspace(
        np.log10(0.001),
        np.log10(0.01),
        base=10,
        num=50,
    ),
    # Data Augmentation
    "mixup": [0, 0.2],
    "close_mosaic": np.linspace(0, 35, 10, dtype=int),
    "degrees": np.linspace(0, 10, 10),
    "translate": np.linspace(0, 0.4, 10),
}


def draw_configuration(space: dict, random_seed: int = 0) -> dict:
    """Draws a configuration from the space using the provided
    `random_seed`."""
    random.seed(random_seed)
    return {k: random.choice(v) for k, v in space.items()}


# draw_configuration(space, random_seed=42)
