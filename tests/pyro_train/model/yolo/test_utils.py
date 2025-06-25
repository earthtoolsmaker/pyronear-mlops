from pyro_train.model.yolo.utils import (
    YOLOModelSize,
    YOLOModelVersion,
    model_version_to_model_type,
)


def test_model_version_to_model_type():
    assert (
        model_version_to_model_type(
            model_version=YOLOModelVersion.version_12,
            model_size=YOLOModelSize.small,
        )
        == "yolo12s.pt"
    )

    assert (
        model_version_to_model_type(
            model_version=YOLOModelVersion.version_10,
            model_size=YOLOModelSize.large,
        )
        == "yolov10l.pt"
    )
