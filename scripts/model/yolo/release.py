"""
CLI script to release the pyronear models.
"""

import argparse
import logging
import os
from pathlib import Path

import requests


def make_cli_parser() -> argparse.ArgumentParser:
    """
    Make the CLI parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        help="server version scheme, always incrementing",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--github-owner",
        help="Github Owner",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--github-repo",
        help="Github Owner",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--release-name",
        help="Unique name to reference the models (eg. wise wolf) - the naming convention is an adjective and an animal with the first letter matching and it goes in alphabetical order for each release.",
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


def is_valid_version(version: str) -> bool:
    """
    Check whether the `version` is valid.
    """
    parts_version = version.split(".")
    return len(parts_version) == 3


def is_valid_release_name(release_name: str) -> bool:
    """
    Check whether the `release_name` is valid. It should follow this naming
    convention.

    eg.
    adjective + animal name starting with the same letter
    agile armadillo
    clever cat
    wise wolf
    ...

    """
    parts_release_name = release_name.split(" ")
    if not len(parts_release_name) == 2:
        return False
    elif not parts_release_name[0][0] == parts_release_name[1][0]:
        return False
    else:
        return True


def validate_parsed_args(args: dict) -> bool:
    """
    Return whether the parsed args are valid.
    """
    if not is_valid_version(args["version"]):
        logging.error(f"invalid --version, should follow semver eg. v1.0.2")
        return False
    if not is_valid_release_name(args["release_name"]):
        logging.error(
            f"invalid --release-name, should follow the release naming convention eg. wise wolf,  v1.0.2"
        )
        return False

    return True


def create_release(
    owner: str,
    repo: str,
    version: str,
    release_name: str,
    github_access_token: str,
) -> dict:
    """
    Create a release using the Github API.
    Returns the response as a dict.
    """

    url = f"https://api.github.com/repos/{owner}/{repo}/releases"
    body = "Pyronear ML Model for early forest fire detection 🔥"

    # Set the headers
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {github_access_token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    # Define the data to be sent in the POST request
    data = {
        "tag_name": version,
        "target_commitish": "main",
        "name": f"Release {version} aka {release_name.title()}",
        "body": body,
        "draft": False,
        "prerelease": False,
        "generate_release_notes": False,
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()


def upload_asset(
    owner: str,
    repo: str,
    filepath_asset: Path,
    name: str,
    release_id: int,
    github_access_token: str,
) -> dict:
    """
    Upload asset for a given release identified with `release_id`.
    """
    url = f"https://uploads.github.com/repos/{owner}/{repo}/releases/{release_id}/assets?name={name}"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {github_access_token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/octet-stream",
    }
    # Open the file in binary mode and make the POST request
    with open(filepath_asset, "rb") as file:
        response = requests.post(url, headers=headers, data=file)
        return response.json()


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=args["loglevel"].upper())
    if not validate_parsed_args(args):
        exit(1)
    else:
        logger.info(args)
        version = args["version"]
        owner = args["github_owner"]
        repo = args["github_repo"]
        # owner = "earthtoolsmaker"
        # repo = "pyronear-mlops"
        release_name = args["release_name"]
        GITHUB_ACCESS_TOKEN = os.getenv("GITHUB_ACCESS_TOKEN")
        assert GITHUB_ACCESS_TOKEN, "You must set the env variable GITHUB_ACCESS_TOKEN"

        # Set the URL for the GitHub API request
        url = "https://api.github.com/repos/earthtoolsmaker/pyronear-mlops/releases"

        # Set the headers
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {GITHUB_ACCESS_TOKEN}",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        response = requests.get(url, headers=headers)
        print(response.json())
        response_release = create_release(
            owner=owner,
            repo=repo,
            version=version,
            release_name=release_name,
            github_access_token=GITHUB_ACCESS_TOKEN,
        )
        release_id = response_release["id"]
        logger.info(f"release created: {response_release}")
        logger.info(f"release_id: {release_id}")
        response_upload_manifest_yaml = upload_asset(
            owner=owner,
            repo=repo,
            release_id=release_id,
            filepath_asset=Path("./data/06_reporting/yolo/best/manifest.yaml"),
            name="manifest.yaml",
            github_access_token=GITHUB_ACCESS_TOKEN,
        )
        response_upload_manifest_yaml = upload_asset(
            owner=owner,
            repo=repo,
            release_id=release_id,
            filepath_asset=Path("./data/04_models/yolo/best/weights/best.pt"),
            name="best.pt",
            github_access_token=GITHUB_ACCESS_TOKEN,
        )
        exit(0)
