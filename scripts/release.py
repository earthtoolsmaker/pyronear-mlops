"""
CLI script to release the pyronear models.
"""

import argparse
import logging
import os
import tarfile
from pathlib import Path

import requests
import tqdm

from pyro_train.data.utils import yaml_read


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
    Check whether the `release_name` is valid. It must adhere to the following naming
    convention:

    The name should consist of an adjective and an animal name, both starting with the same
    letter, and separated by a space. The adjective and animal should also follow a specific
    alphabetical order for each release.

    Examples:
    - agile armadillo
    - clever cat
    - wise wolf
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
        logging.error("invalid --version, should follow semver eg. v1.0.2")
        return False
    if not is_valid_release_name(args["release_name"]):
        logging.error(
            "invalid --release-name, should follow the release naming convention eg. wise wolf,  v1.0.2"
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
    Create a release using the GitHub API.

    This function creates a new release in the specified GitHub repository and returns the
    response from the API as a dictionary. The release includes details such as the tag name,
    target branch, release name, and description.

    Args:
        owner (str): The owner of the GitHub repository.
        repo (str): The name of the GitHub repository.
        version (str): The version tag for the release, following semantic versioning.
        release_name (str): The name of the release, typically a descriptive title.
        github_access_token (str): A personal access token for authenticating with the GitHub API.

    Returns:
        dict: The JSON response from the GitHub API containing information about the created release.
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
        "name": f"{release_name.title()}",
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
) -> dict | None:
    """
    Upload asset for a given release identified with `release_id`.

    This function uploads an asset to a specific release on GitHub using the provided
    parameters. It will raise an HTTPError if the upload request fails.

    Args:
        owner (str): The GitHub owner of the repository.
        repo (str): The GitHub repository name.
        filepath_asset (Path): The path to the asset file to be uploaded.
        name (str): The name to assign to the uploaded asset.
        release_id (int): The ID of the release to which the asset will be uploaded.
        github_access_token (str): The GitHub access token for authentication.

    Returns:
        dict | None: The JSON response from the GitHub API if successful,
                      or None if the upload fails.
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
        # Check if the request was successful
        if response.status_code == 201:
            return response.json()  # Return the JSON response if successful
        else:
            response.raise_for_status()  # Raise an error for unsuccessful requests


def get_model_name(version: str, release_name: str, filepath_manifest: Path) -> str:
    """
    Return the model name, which includes the release name, version, and the
    associated training data hash. The model name is formatted as:
    `{model_type}_{formatted_release_name}_{version}_{short_md5_hash}`.

    Args:
        version (str): The version of the release.
        release_name (str): The unique name for the model, following the
                            convention of an adjective and an animal name
                            starting with the same letter.
        filepath_manifest (Path): The path to the manifest file containing
                                  model and data information.

    Returns:
        str: The formatted model name.
    """
    model_name = release_name.replace(" ", "-")
    manifest = yaml_read(filepath_manifest)
    model_type = manifest["model"]["model_type"].split(".")[0]
    md5_data = manifest["data"]["dvc"]["md5"]
    md5_data_short = md5_data[:7]
    return f"{model_type}_{model_name}_{version}_{md5_data_short}"


def create_archive(source_folder: Path, archive_name: str) -> Path:
    """
    Create a tar.gz archive of the specified source folder.

    This function compresses all files and subdirectories within the
    source_folder into a single tar.gz file named `archive_name`.
    It provides progress feedback during the archiving process.

    Args:
        source_folder (Path): The path to the folder to be archived.
        archive_name (str): The name to give to the created archive file.

    Returns:
        Path: The path to the created archive file.
    """
    # Create a tar.gz archive of the source folder
    archive_path = Path("/tmp") / archive_name
    with tarfile.open(archive_path, "w:gz") as tar:
        total_files = sum(len(files) for _, _, files in os.walk(source_folder))
        with tqdm.tqdm(total=total_files, desc="Creating archive") as pbar:
            for root, _, files in os.walk(source_folder):
                for file in files:
                    tar.add(
                        os.path.join(root, file),
                        arcname=os.path.relpath(
                            os.path.join(root, file), source_folder
                        ),
                    )
                    pbar.update(1)
    return archive_path


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
        release_name = args["release_name"]
        GITHUB_ACCESS_TOKEN = os.getenv("GITHUB_ACCESS_TOKEN")
        assert GITHUB_ACCESS_TOKEN, "You must set the env variable GITHUB_ACCESS_TOKEN"
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
        filepath_manifest = Path("./data/06_reporting/yolo/best/manifest.yaml")
        assert filepath_manifest.exists()
        logger.info("Uploading the manifest.yaml file")
        response_upload_manifest_yaml = upload_asset(
            owner=owner,
            repo=repo,
            release_id=release_id,
            filepath_asset=filepath_manifest,
            name="manifest.yaml",
            github_access_token=GITHUB_ACCESS_TOKEN,
        )
        model_name = get_model_name(
            version=version,
            release_name=release_name,
            filepath_manifest=filepath_manifest,
        )
        dir_exports = Path("./data/04_models/yolo-export/best/")
        subdirs = [d for d in dir_exports.iterdir() if d.is_dir()]

        for subdir_format in subdirs:
            for subdir_device in [d for d in subdir_format.iterdir() if d.is_dir()]:
                export_device = subdir_device.stem
                export_format = subdir_format.stem
                logger.info(
                    f"Making an archive for the export in {export_format} format for the device {export_device}"
                )
                archive_name = f"{export_format}_{export_device}_{model_name}.tar.gz"
                filepath_archive = create_archive(
                    source_folder=subdir_device,
                    archive_name=archive_name,
                )
                logger.info(
                    f"Enclosing the archive {filepath_archive} to the release assets"
                )
                upload_asset(
                    owner=owner,
                    repo=repo,
                    release_id=release_id,
                    filepath_asset=filepath_archive,
                    name=archive_name,
                    github_access_token=GITHUB_ACCESS_TOKEN,
                )
                filepath_archive.unlink()

        response_upload_best_model = upload_asset(
            owner=owner,
            repo=repo,
            release_id=release_id,
            filepath_asset=Path("./data/04_models/yolo/best/weights/best.pt"),
            name=f"{model_name}.pt",
            github_access_token=GITHUB_ACCESS_TOKEN,
        )
        exit(0)
