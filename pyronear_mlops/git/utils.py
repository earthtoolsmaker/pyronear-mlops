"""
Git helper functions.
"""

import subprocess


def get_git_revision_hash():
    """
    Get the full git revision hash.
    """
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")


def get_git_revision_short_hash():
    """
    Get the short git revision hash.
    """
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .strip()
        .decode("utf-8")
    )
