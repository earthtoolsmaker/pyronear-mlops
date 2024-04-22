import argparse
import logging
import os
import random
import shutil
from pathlib import Path

from pyronear_mlops.data.utils import yaml_read, yaml_write


def make_cli_parser() -> argparse.ArgumentParser:
    """Makes the CLI parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        help="path pointing to the raw dataset",
        default="./data/01_raw/DS-71c1fd51-v2",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        help="path to save the model_input",
        default="./data/03_model_input/yolov8/",
        type=Path,
    )
    parser.add_argument(
        "-log",
        "--loglevel",
        default="warning",
        help="Provide logging level. Example --loglevel debug, default=warning",
    )
    return parser


def validate_parsed_args(args: dict) -> bool:
    """Returns whether the parsed args are valid."""
    if not args["input_dir"].exists():
        logging.error(f"Invalid --input-dir directory, it does not exist")
        return False
    else:
        return True


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logging.basicConfig(level=args["loglevel"].upper())
    if not validate_parsed_args(args):
        logging.error(f"Could not validate the parsed args: {args}")
        exit(1)
    else:
        logging.info(args)
        input_dir = args["input_dir"]
        output_dir = args["output_dir"]
        logging.info(f"Creating dir at {output_dir}")
        # Make the output directory idempotent
        shutil.rmtree(output_dir, ignore_errors=True)
        # output_dir.mkdir(exist_ok=True, parents=True)
        shutil.copytree(src=input_dir, dst=output_dir)
        # TODO: rewrite data_yaml_file
        # write_data_yaml(output_dir)
        exit(0)

# TODO: make a small version of each split with random sampling
# TODO: fix data_yaml file paths for training model: use relative path
