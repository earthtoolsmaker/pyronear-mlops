"""
CLI script to aggregate all the results from the YOLO train runs.
"""

import argparse
import logging
import os
from pathlib import Path

import pandas as pd

from pyro_train.data.utils import yaml_read


def make_cli_parser() -> argparse.ArgumentParser:
    """
    Make the CLI parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        help="root directory containing YOLO train runs.",
        default="./data/04_models/yolo/",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        help="path to save the benchmark.",
        default="./data/06_reporting/yolo/",
        type=Path,
    )
    parser.add_argument(
        "-log",
        "--loglevel",
        default="warning",
        help="Provide logging level. Example --loglevel debug, default=warning.",
    )
    return parser


def validate_parsed_args(args: dict) -> bool:
    """
    Return whether the parsed args are valid.
    """
    if not args["input_dir"].exists():
        logging.error("Invalid --input-dir directory does not exist")
        return False
    else:
        return True


def add_args_columns(df_results: pd.DataFrame, args: dict):
    """
    Return a dataframe that has the same columns as df_results and also the
    keys of args with the same value for each row (args[key]).
    """
    df = df_results.copy()
    args_str = {k: str(v) for k, v in args.items()}
    df_args = pd.DataFrame([args_str] * len(df))
    return pd.concat([df, df_args], axis=1)


def make_benchmark(input_dir: Path) -> pd.DataFrame:
    """
    Return the df_benchmark dataframe containing the concatenated results
    of each train runs and their associated model parameters.
    """
    train_dirs = [
        input_dir / f for f in os.listdir(input_dir) if (input_dir / f).is_dir()
    ]
    dfs = []
    for train_dir in train_dirs:
        args_filepath = train_dir / "args.yaml"
        results_filepath = train_dir / "results.csv"
        if not args_filepath.exists() or not results_filepath.exists():
            logging.warning(f"Skipping {train_dir} - missing artifact")
        else:
            logging.info(f"Loading files {args_filepath} and {results_filepath}")
            cli_args = yaml_read(args_filepath)
            df_results = pd.read_csv(results_filepath)
            df_results_with_args_columns = add_args_columns(
                df_results=df_results, args=cli_args
            )
            dfs.append(df_results_with_args_columns)
    df_benchmark = pd.concat(dfs, ignore_index=True)
    return df_benchmark


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    cli_args = vars(cli_parser.parse_args())
    logging.basicConfig(level=cli_args["loglevel"].upper())
    if not validate_parsed_args(cli_args):
        logging.error(f"Could not validate the parsed args: {cli_args}")
        exit(1)
    else:
        logging.info(cli_args)
        input_dir = cli_args["input_dir"]
        output_dir = cli_args["output_dir"]
        df_benchmark = make_benchmark(input_dir)
        output_filepath = output_dir / "benchmark.csv"
        logging.info(f"Saving results in {output_filepath}")
        os.makedirs(output_dir, exist_ok=True)
        df_benchmark.to_csv(output_dir / "benchmark.csv", index=False)
        exit(0)
