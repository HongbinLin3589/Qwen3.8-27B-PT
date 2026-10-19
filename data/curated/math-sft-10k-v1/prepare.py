"""Verify the distributed shards and materialize the training entry-point files."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def materialize(table, destination, expected_rows):
    import pyarrow.parquet as pq
    if table.num_rows != expected_rows:
        raise ValueError(f'{destination.name}: expected {expected_rows}, got {table.num_rows}')
    if destination.exists():
        if not pq.ParquetFile(destination).read().equals(table):
            raise FileExistsError(f'Refusing to replace different data at {destination}; use another --output-dir')
        return
    fd, name = tempfile.mkstemp(prefix=destination.name, suffix='.tmp', dir=destination.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        pq.write_table(table, temporary, compression='zstd')
        if not pq.ParquetFile(temporary).read().equals(table):
            raise ValueError('Parquet round-trip changed rows')
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=HERE / 'prepared')
    parser.add_argument('--tokenizer', help='Local model/tokenizer directory for full integration validation')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((HERE / 'validation.json').read_text())
    splits = {'train': [], 'validation': []}
    for artifact in manifest['artifacts']:
        path = HERE / artifact['file']
        if path.parent != HERE or sha256(path) != artifact['sha256']:
            raise ValueError(f'Artifact checksum mismatch: {artifact["file"]}')
        if pq.ParquetFile(path).metadata.num_rows != artifact['rows']:
            raise ValueError(f'Artifact row count mismatch: {path.name}')
        table = pq.ParquetFile(path).read()
        if artifact['split'] != 'mixed' or 'split' not in table.column_names:
            raise ValueError(f'Expected explicitly split mixed shard: {path.name}')
        if set(table['split'].to_pylist()) != {'train', 'validation'}:
            raise ValueError(f'Unexpected split values: {path.name}')
        for split in splits:
            part = table.filter(pc.equal(table['split'], split)).drop(['split'])
            if part.num_rows != artifact[f'{split}_rows']:
                raise ValueError(f'Split row count mismatch: {path.name}:{split}')
            splits[split].append(part)
    materialize(pa.concat_tables(splits['train']), output / 'train.parquet', 9500)
    materialize(pa.concat_tables(splits['validation']), output / 'val.parquet', 500)
    if args.tokenizer:
        sys.path.insert(0, str(REPO / 'src'))
        from transformers import AutoTokenizer
        from qwenpt.train.verl.dataset import QwenSFTDataset
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        for name, count in [('train', 9500), ('val', 500)]:
            dataset = QwenSFTDataset(str(output / f'{name}.parquet'), tokenizer,
                                    {'max_length': 8192, 'pad_mode': 'no_padding', 'truncation': 'error'})
            if len(dataset) != count:
                raise ValueError('Unexpected dataset length')
            for index in range(len(dataset)):
                item = dataset[index]
                if not item['loss_mask'].any():
                    raise ValueError(f'Missing assistant supervision at {name}:{index}')
            print(f'{name}: {count} samples passed the local model tokenizer / QwenSFTDataset check', flush=True)
    print(json.dumps({'train_file': str(output / 'train.parquet'), 'val_file': str(output / 'val.parquet'),
                      'train_rows': 9500, 'val_rows': 500, 'max_length': 8192}, indent=2))


if __name__ == '__main__':
    main()
