"""
CLI script to generate a manifest file for the best model.
"""

import argparse
import logging
from pathlib import Path

from pyro_train.data.utils import yaml_read, yaml_write
from pyro_train.utils import compute_file_content_sha256


def make_cli_parser() -> argparse.ArgumentParser:
    """
    Make the CLI parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-dir",
        help="directory to save the manifest.",
        type=Path,
        default=Path("./data/06_reporting/yolo/best/"),
    )
    parser.add_argument(
        "--dir-model",
        help="directory pointing to the yolo model.",
        type=Path,
        default=Path("./data/04_models/yolo/best/"),
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
    if not args["dir_model"].exists():
        logging.error(f"invalid --dir-model, does not exist {args['dir_model']}")
        return False
    return True


def make_data_manifest(dir_model: Path) -> dict:
    """
    Make the data manifest for the dir_model.
    It is meant to persist as much information as possible about the model and
    training environment.
    """
    filepath_weights = dir_model / "weights" / "best.pt"
    filepath_args = dir_model / "args.yaml"
    sha256_weights = compute_file_content_sha256(filepath_weights)
    args_model_train_run = yaml_read(filepath_args)
    filepath_raw_wildfire_dvc = Path("./data/01_raw/wildfire.dvc")
    raw_wildfire_dvc = yaml_read(filepath_raw_wildfire_dvc)
    data_yaml = Path(args_model_train_run["data"]).relative_to(Path(".").absolute())
    filepath_dvc_lock = Path("./dvc.lock")
    dvc_lock = yaml_read(filepath_dvc_lock)
    return {
        "model": {
            "weights": {
                "filepath": str(filepath_weights),
                "sha256": sha256_weights,
            },
            "model_type": args_model_train_run["model"],
            "dvc": dict(dvc_lock["stages"]["train_yolo_best"]["outs"][0]),
        },
        "data": {
            "data_yaml": str(data_yaml),
            "dvc": raw_wildfire_dvc["outs"][0],
        },
        "train_run_args": args_model_train_run,
        "dvc_lock": dvc_lock,
    }


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=args["loglevel"].upper())
    if not validate_parsed_args(args):
        exit(1)
    else:
        logger.info(args)
        dir_model = args["dir_model"]
        save_dir = args["save_dir"]
        save_dir.mkdir(parents=True, exist_ok=True)
        filepath_manifest = save_dir / "manifest.yaml"
        data_manifest = make_data_manifest(dir_model=dir_model)
        logger.info(f"saving manifest.yaml file in {save_dir}")
        yaml_write(to=filepath_manifest, data=data_manifest)
        exit(0)
