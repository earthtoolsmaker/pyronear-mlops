"""
Utilities to work with YOLO models.
"""

from enum import Enum


class YOLOModelVersion(Enum):
    """
    Supported YOLO Model Versions.
    """

    version_8 = 8
    version_9 = 9
    version_10 = 10
    version_11 = 11
    version_12 = 12


class YOLOModelSize(Enum):
    """
    Supported YOLO Model Sizes.
    """

    nano = "n"
    small = "s"
    medium = "m"
    large = "l"
    xlarge = "x"


def model_version_to_model_type(
    model_version: YOLOModelVersion,
    model_size: YOLOModelSize,
) -> str:
    """
    Build the model_type based on model_version and model_size.

    Eg. yolo12s.pt, yolo8l, etc.
    """
    prefix = "yolov" if model_version.value <= 10 else "yolo"
    suffix = ".pt" if model_version.value >= 9 else ""

    return f"{prefix}{model_version.value}{model_size.value}{suffix}"
