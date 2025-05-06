"""
Module to train the YOLO models.
"""

from pathlib import Path

from ultralytics import YOLO


def load_pretrained_model(model_str: str) -> YOLO:
    """
    Loads the pretrained `model`
    """
    return YOLO(model_str)


def train(
    model: YOLO,
    data_yaml_path: Path,
    params: dict,
    project: str = "data/04_models/yolo/",
    experiment_name: str = "train",
):
    """
    Main function for running a train run.
    """
    assert data_yaml_path.exists(), f"data_yaml_path does not exist, {data_yaml_path}"
    default_params = {
        # train parameters
        "batch": 16,
        "box": 7.5,
        "cls": 0.5,
        "cos_lr": False,
        "dfl": 1.5,
        "epochs": 100,
        "imgsz": 640,
        "lr0": 0.01,
        "lrf": 0.01,
        "optimizer": "auto",
        "patience": 100,
        "single_cls": True,
        # val parameters
        "iou": 0.6,
        # data augmentation parameters
        "close_mosaic": 10,
        "degrees": 0.0,
        "fliplr": 0.5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "mixup": 0.0,
        "shear": 0.0,
        "translate": 0.1,
    }
    params = {**default_params, **params}
    model.train(
        project=project,
        name=experiment_name,
        data=data_yaml_path.absolute(),
        # train Parameters
        batch=params["batch"],
        epochs=params["epochs"],
        imgsz=params["imgsz"],
        lr0=params["lr0"],
        lrf=params["lrf"],
        optimizer=params["optimizer"],
        box=params["box"],
        cls=params["cls"],
        dfl=params["dfl"],
        single_cls=params["single_cls"],
        cos_lr=params["cos_lr"],
        # val parameters
        iou=params["iou"],
        # Data Augmentation parameters
        close_mosaic=params["close_mosaic"],
        degrees=params["degrees"],
        fliplr=params["fliplr"],
        hsv_h=params["hsv_h"],
        hsv_s=params["hsv_s"],
        hsv_v=params["hsv_v"],
        mixup=params["mixup"],
        shear=params["shear"],
        translate=params["translate"],
    )
