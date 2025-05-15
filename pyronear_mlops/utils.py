import hashlib
from pathlib import Path


def compute_file_content_sha256(filepath: Path) -> str:
    """
    Compute the file content hash of the `filepath`.

    Returns:
        hexdigest (str)
    """
    # Create a hash object
    hash_sha256 = hashlib.sha256()

    # Open the file in binary mode
    with open(filepath, "rb") as f:
        # Read the file in chunks to avoid using too much memory
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)

    # Return the hexadecimal digest of the hash
    return hash_sha256.hexdigest()
