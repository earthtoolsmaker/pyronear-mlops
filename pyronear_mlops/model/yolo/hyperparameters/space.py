"""
Module to generate hyperparameter search spaces for training YOLO models.
"""

import random
from dataclasses import dataclass

import numpy as np

from pyronear_mlops.model.yolo.utils import (
    YOLOModelSize,
    YOLOModelVersion,
    model_version_to_model_type,
)

ALLOWED_BATCHS_SIZES = [16, 32, 64, 128]


@dataclass
class HyperparameterSpace:
    """
    Simple DataClass modeling an Hyperparameter Space.
    """

    space: dict[str, np.ndarray]


def make_model_types(
    model_versions: list[YOLOModelVersion],
    model_sizes: list[YOLOModelSize] = [
        YOLOModelSize.nano,
        YOLOModelSize.small,
        YOLOModelSize.medium,
        YOLOModelSize.large,
    ],
) -> list[str]:
    """
    Make possible model types based on model_versions and model_sizes.
    """
    return [
        model_version_to_model_type(model_version, model_size)
        for model_size in model_sizes
        for model_version in model_versions
    ]


def make_space(
    model_versions: list[YOLOModelVersion],
    model_sizes: list[YOLOModelSize],
    batch_sizes: list[int],
) -> HyperparameterSpace:
    """
    Make an HyperparameterSpace.

    Arguments:
        model_version (YOLOModelVersion): the YOLO model version to target.
        model_sizes (list[YOLOModelSize]): the YOLO model sizes to target.

    Returns:
        hyperparameter_space (HyperparameterSpace).

    Throws:
        assertError: when the batch_sizes are not in ALLOWED_BATCHS_SIZES.

    __Note__: Config documentation: https://docs.ultralytics.com/usage/cfg/#train-settings
    """
    model_types = np.array(
        make_model_types(
            model_versions=model_versions,
            model_sizes=model_sizes,
        )
    )
    space = {
        "model_type": model_types,
        # "epochs": np.linspace(50, 200, 20, dtype=int),
        "epochs": np.linspace(50, 70, 3, dtype=int),
        "patience": np.linspace(10, 50, 10, dtype=int),
        "imgsz": np.array([1024], dtype=int),
        "batch": np.array(batch_sizes, dtype=int),
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
        "mixup": np.array([0, 0.2], dtype=float),
        "close_mosaic": np.linspace(0, 35, 10, dtype=int),
        "degrees": np.linspace(0, 10, 10, dtype=int),
        "translate": np.linspace(0, 0.4, 10, dtype=float),
    }
    return HyperparameterSpace(space=space)


def draw_configuration(
    hyperparameter_space: HyperparameterSpace,
    random_seed: float,
) -> dict:
    """
    Draw a configuration from the space using the provided
    `random_seed`.
    """
    rng = random.Random(random_seed)
    return {k: rng.choice(v).item() for k, v in hyperparameter_space.space.items()}


def draw_n_random_configurations(
    hyperparameter_space: HyperparameterSpace,
    n: int,
    random_seed: float | int = 0,
) -> list:
    """
    Draw n configurations from the space using the provided
    `random_seed`.
    """
    rng = random.Random(random_seed)
    return [
        draw_configuration(hyperparameter_space, random_seed=rng.random())
        for _ in range(n)
    ]


# # REPL
# model_version = YOLOModelVersion.version_12
# space = make_space(model_version)
# space
# draw_configuration(space, random_seed=42)
# draw_n_random_configurations(space, 10)
# model_size = YOLOModelSize.small
# model_version_to_model_type(model_version, model_size)
# model_version.value
