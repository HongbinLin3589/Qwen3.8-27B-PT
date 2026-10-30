"""Create a reproducible train/validation split from prepared SFT Parquet files."""

import argparse
import json
from pathlib import Path
import random

import pyarrow as pa
import pyarrow.parquet as pq

from qwenpt.train.common.files import input_paths, sha256


def prepare_sft_data(
    data: list[str], output: str, *, seed: int = 20261020, val_ratio: float = 0.1,
    reuse_existing: bool = False,
) -> dict:
    """Split input files into output Parquet files and return their audit manifest.

    ``data`` accepts directories or Parquet files; ``output`` must not already
    contain split artifacts unless ``reuse_existing`` verifies a matching split.
    ``seed`` fixes the shuffle, and ``val_ratio`` sets the validation count as
    ``floor(total_rows * val_ratio)``.
    """
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    paths = input_paths(data)
    output_dir = Path(output).resolve()
    artifact_paths = [output_dir / name for name in (
        "train.parquet", "val.parquet", "split_manifest.json",
    )]
    existing = [path for path in artifact_paths if path.exists()]
    if existing and not reuse_existing:
        raise FileExistsError(f"Refusing to overwrite split artifact: {existing[0]}")
    if existing and len(existing) != len(artifact_paths):
        raise FileExistsError(f"Incomplete split artifacts in {output_dir}")

    parquet_files = [pq.ParquetFile(path) for path in paths]
    inputs = [
        {"path": str(path), "sha256": sha256(path), "rows": part.metadata.num_rows}
        for path, part in zip(paths, parquet_files, strict=True)
    ]
    if existing:
        manifest = json.loads(artifact_paths[2].read_text())
        if (manifest.get("format_version") != 1 or manifest.get("inputs") != inputs
                or manifest.get("seed") != seed or manifest.get("val_ratio") != val_ratio):
            raise ValueError(f"Existing split inputs or parameters differ in {output_dir}")
        hashes = {path.name: sha256(path) for path in artifact_paths[:2]}
        if manifest.get("output_sha256") != hashes:
            raise ValueError(f"Existing split output hashes differ in {output_dir}")
        return manifest

    tables = [part.read(use_threads=False) for part in parquet_files]
    table = pa.concat_tables(tables, promote_options="default")
    val_count = int(len(table) * val_ratio)
    if not 0 < val_count < len(table):
        raise ValueError("The requested split must contain both training and validation rows")
    indices = list(range(len(table)))
    random.Random(seed).shuffle(indices)
    split_indices = {"validation": indices[:val_count], "train": indices[val_count:]}
    manifest = {
        "format_version": 1,
        "inputs": inputs,
        "seed": seed,
        "val_ratio": val_ratio,
        "counts": {"total": len(table), "train": len(table) - val_count, "validation": val_count},
        "split_indices": split_indices,
        "output_files": {"train": "train.parquet", "validation": "val.parquet"},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, row_indices in split_indices.items():
        part = table.take(pa.array(row_indices, type=pa.int64()))
        labels = pa.array([name] * len(part), type=pa.string())
        # Source delivery labels must not leak into the newly randomized split.
        if "split" in part.column_names:
            part = part.set_column(part.schema.get_field_index("split"), "split", labels)
        else:
            part = part.append_column("split", labels)
        pq.write_table(part, output_dir / manifest["output_files"][name])
    manifest["output_sha256"] = {path.name: sha256(path) for path in artifact_paths[:2]}
    artifact_paths[2].write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main() -> None:
    """Parse split input paths, destination, seed, and ratio and print row counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    manifest = prepare_sft_data(
        args.data, args.output, seed=args.seed, val_ratio=args.val_ratio,
        reuse_existing=args.reuse_existing,
    )
    print(json.dumps({"output": str(Path(args.output).resolve()), "counts": manifest["counts"]}))


if __name__ == "__main__":
    main()
