"""Fetch the immutable raw artifacts listed in source_manifest.json."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, default=HERE / 'raw')
    args = parser.parse_args()
    root = args.raw_dir.resolve()
    manifest = json.loads((HERE / 'source_manifest.json').read_text())

    def fetch(item):
        target = (root / item['raw_path']).resolve()
        if root not in target.parents:
            raise ValueError('Invalid raw artifact path')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == item['sha256']:
            return
        partial = target.with_suffix(target.suffix + '.part')
        digest = hashlib.sha256()
        with urlopen(Request(item['url'], headers={'User-Agent': 'qwenpt-data-curation/1.0'}), timeout=180) as response, partial.open('wb') as out:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                out.write(chunk)
        if partial.stat().st_size != item['bytes'] or digest.hexdigest() != item['sha256']:
            raise ValueError(f'Raw artifact integrity failure: {target.name}')
        partial.replace(target)
        print(item['raw_path'], flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, manifest['artifacts']))


if __name__ == '__main__':
    main()
