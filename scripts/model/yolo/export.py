"""
CLI script to export a YOLO model to different formats (ONNX, NCNN).
"""

import argparse
import logging
import shutil
from pathlib import Path

from ultralytics import YOLO

from pyro_train.data.utils import yaml_read


def make_cli_parser() -> argparse.ArgumentParser:
    """
    Make the CLI parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        help="directory that contains the result of the ultralytics training of the model",
        default="./data/04_models/yolo/best/",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        help="path to save the model_artifacts",
        default="./data/04_models/yolo-export/best/",
        type=Path,
    )
    parser.add_argument(
        "--format",
        help="export format (onnx, ncnn)",
        choices=["onnx", "ncnn"],
        type=str,
        required=True,
    )
    parser.add_argument(
        "--device",
        help="device to target: cpu, gpu, mps",
        choices=["cpu", "gpu", "mps"],
        type=str,
        required=True,
    )
    parser.add_argument(
        "-log",
        "--loglevel",
        default="info",
        help="Provide logging level. Example --loglevel debug, default=warning",
    )
    return parser


def validate_parsed_args(args: dict) -> bool:
    """
    Return whether the parsed args are valid.
    """
    if not args["model_dir"].exists():
        logging.error("Invalid --model-dir directory does not exist")
        return False
    else:
        return True


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=args["loglevel"].upper())
    if not validate_parsed_args(args):
        logging.error(f"Could not validate the parsed args: {args}")
        exit(1)
    else:
        logger.info(args)
        model_dir = args["model_dir"]
        output_dir = args["output_dir"]
        format = args["format"]
        device = args["device"]
        logger.info(
            f"exporting model from {model_dir} to {format} format targetting {device} devices and saving results to {output_dir}"
        )
        args_train = yaml_read(model_dir / "args.yaml")
        logger.info(f"args.yaml: {args_train}")
        filepath_weights = model_dir / "weights" / "best.pt"
        if device == "gpu":
            device = "0"
        logger.info(f"load model from {filepath_weights}")
        model = YOLO(filepath_weights)
        model.info()
        save_dir = output_dir / format / device
        save_dir.mkdir(parents=True, exist_ok=True)
        if format == "onnx":
            filepath_export = Path(
                model.export(
                    format=format,
                    dynamic=True,
                    imgsz=args_train["imgsz"],
                    device=device,
                )
            )
        else:
            filepath_export = Path(
                model.export(
                    format=format,
                    imgsz=args_train["imgsz"],
                    device=device,
                )
            )

        # Do not include torchscript
        # filepath_torchscript = (
        #     filepath_weights.parent / f"{filepath_weights.stem}.torchscript"
        # )
        # if filepath_torchscript.exists():
        #     shutil.move(
        #         src=filepath_torchscript, dst=save_dir / filepath_torchscript.name
        #     )

        shutil.move(src=filepath_export, dst=save_dir / filepath_export.name)
        logger.info(f"Model successfully exported to {save_dir} ✅")
