from pathlib import Path

from ultralytics import YOLO


def load_pretrained_model(model_str: str) -> YOLO:
    """Loads the pretrained `model`"""
    return YOLO(model_str)


def train(
    model: YOLO,
    data_yaml_path: Path,
    params: dict,
    project: str = "data/04_models/yolov8/",
    experiment_name: str = "train",
):
    """Main function for running a train run."""
    assert data_yaml_path.exists(), f"data_yaml_path does not exist, {data_yaml_path}"
    model.train(
        project=project,
        name=experiment_name,
        data=data_yaml_path.absolute(),
        epochs=params["epochs"],
        imgsz=params["imgsz"],
        close_mosaic=params["close_mosaic"],
        # Data Augmentation parameters
        degrees=params["degrees"],
        flipud=params["flipud"],
        translate=params["translate"],
    )
