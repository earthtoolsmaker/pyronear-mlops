"""
CLI script to run random hyperparameter search for training a YOLO
model the object detection task of fire smokes.
"""

import argparse
import logging
import random
import uuid
from datetime import datetime
from pathlib import Path

from ultralytics import settings

import pyro_train.model.yolo.hyperparameters.space as hyperparameters
from pyro_train.model.yolo.train import load_pretrained_model, train


def make_cli_parser() -> argparse.ArgumentParser:
    """
    Make the CLI parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        help="filepath to the data_yaml config file for the dataset",
        default="./data/03_model_input/yolo/small/datasets/data.yaml",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        help="path to save the model_artifacts",
        default="./data/04_models/yolo/",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--experiment-name",
        help="experiment name",
        default="random_hyperparameter_search",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--filepath-space-yaml",
        help="Filepath to the Hyperparamter Space Definition",
        type=Path,
        default=Path("./scripts/model/yolo/spaces/default.yaml"),
        required=True,
    )
    parser.add_argument(
        "--n",
        help="number of random configurations to run",
        default=10,
        type=int,
        required=True,
    )
    parser.add_argument(
        "-log",
        "--loglevel",
        default="warning",
        help="Provide logging level. Example --loglevel debug, default=warning",
    )
    return parser


def validate_parsed_args(args: dict) -> bool:
    """
    Return whether the parsed args are valid
    """
    if not args["data"].exists():
        logging.error("Invalid --data filepath does not exist")
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
        n = args["n"]
        random_seed = datetime.now().timestamp()
        filepath_space_yaml = args["filepath_space_yaml"]
        logging.info(f"Initializing random seed: {random_seed}")
        random.seed(random_seed)

        hyperparameter_space = hyperparameters.parse_space_yaml(
            filepath_space=filepath_space_yaml
        )

        configurations = hyperparameters.draw_n_random_configurations(
            hyperparameter_space=hyperparameter_space,
            n=n,
            random_seed=random_seed,
        )

        # Update ultralytics settings to log with MLFlow
        settings.update({"mlflow": True})
        logging.info(f"Generated {len(configurations)} configurations")

        for idx, configuration in enumerate(configurations):
            run_id = uuid.uuid4().hex
            logging.info(
                f"Starting train run {idx} with the following configuration: {configuration}"
            )
            logging.info(f"loading pretrained model: {configuration['model_type']}")
            model = load_pretrained_model(configuration["model_type"])
            train(
                model=model,
                data_yaml_path=args["data"],
                params=configuration,
                project=str(args["output_dir"]),
                experiment_name=f"{args['experiment_name']}_{run_id}",
            )
        exit(0)
