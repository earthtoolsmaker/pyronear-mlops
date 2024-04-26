import random

import numpy as np

# Config documentation: https://docs.ultralytics.com/usage/cfg/#train-settings
space = {
    "model_type": np.array(["yolov8n", "yolov8s", "yolov8m"]),
    "epochs": np.linspace(50, 200, 20, dtype=int),
    "patience": np.linspace(10, 50, 10, dtype=int),
    "imgsz": np.array([320, 640, 1024], dtype=int),
    "batch": np.array([16, 32, 64]),
    "optimizer": np.array(
        [
            "SGD",
            "Adam",
            "AdamW",
            "NAdam",
            "RAdam",
            "RMSProp",
            "auto",
        ]
    ),
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
    "mixup": np.array([0, 0.2]),
    "close_mosaic": np.linspace(0, 35, 10, dtype=int),
    "degrees": np.linspace(0, 10, 10),
    "translate": np.linspace(0, 0.4, 10),
}


def draw_configuration(
    space: dict[str, np.ndarray],
    random_seed: float | int = 0,
) -> dict:
    """Draws a configuration from the space using the provided
    `random_seed`."""
    random.seed(random_seed)
    return {k: random.choice(v).item() for k, v in space.items()}


def draw_n_random_configurations(
    space: dict[str, np.ndarray],
    n: int,
    random_seed: float | int = 0,
) -> list:
    """Draws n configurations from the space using the provided
    `random_seed`."""
    random.seed(random_seed)
    return [draw_configuration(space, random_seed=random.random()) for _ in range(n)]


# draw_configuration(space, random_seed=42)
# draw_n_random_configurations(space, 10)
