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
    default_params = {
        "batch": 16,
        "epochs": 100,
        "patience": 100,
        "imgsz": 640,
        "lr0": 0.01,
        "lrf": 0.01,
        "optimizer": "auto",
        # data augmentation
        "mixup": 0.0,
        "close_mosaic": 10,
        "degrees": 0.0,
        "translate": 0.1,
        "flipud": 0.0,
        "fliplr": 0.5,
    }
    params = {**default_params, **params}
    model.train(
        project=project,
        name=experiment_name,
        data=data_yaml_path.absolute(),
        # data=data_yaml_path,
        epochs=params["epochs"],
        lr0=params["lr0"],
        lrf=params["lrf"],
        optimizer=params["optimizer"],
        imgsz=params["imgsz"],
        close_mosaic=params["close_mosaic"],
        # Data Augmentation parameters
        mixup=params["mixup"],
        degrees=params["degrees"],
        flipud=params["flipud"],
        translate=params["translate"],
    )
