"""Shared file hashing and Parquet input discovery for training tools."""

import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file without reading it all into memory."""
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def input_paths(data: list[str]) -> list[Path]:
    """Expand directories and files into sorted, unique Parquet input paths."""
    paths = set()
    for value in data:
        path = Path(value).resolve()
        if path.is_dir():
            matches = list(path.glob("*.parquet"))
            if not matches:
                raise ValueError(f"No Parquet files found in {path}")
            paths.update(matches)
        elif path.is_file() and path.suffix == ".parquet":
            paths.add(path)
        else:
            raise ValueError(f"Expected a Parquet file or directory: {path}")
    if not paths:
        raise ValueError("At least one input Parquet file is required")
    return sorted(paths)
