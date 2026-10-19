"""Command line.

  datapipe download --sources configs/sources/hf.yaml --out data/
  datapipe run      --data data/ --out runs/r1 [--config configs/minimal.yaml] [--set k=v ...]
  datapipe export   --run runs/r1 --out export/r1 [--config ...]
  datapipe sft      --input samples.jsonl [samples.parquet ...] --out export/sft [--config ...]
  datapipe validate FILE [--kind sft|rl] [--key images=imgs ...]   # check a .parquet/.jsonl/.json against the format
  datapipe list                 # registered stages, backends, verifiers, loaders, mappers
  datapipe show-config [--config ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .config import load_config


def _load_yaml(path):
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="datapipe")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="download datasets and convert them to tasks / eval sets")
    d.add_argument("--sources", required=True, help="sources YAML (see configs/sources/)")
    d.add_argument("--out", required=True)
    d.add_argument("--only", nargs="*", help="only these source names")

    r = sub.add_parser("run", help="run the filtering pipeline")
    r.add_argument("--data", help="a `download` output dir: uses <data>/tasks.jsonl and <data>/eval/*.jsonl")
    r.add_argument("--input", help="tasks JSONL (instead of --data)")
    r.add_argument("--format", default="task", choices=["task", "math2code"])
    r.add_argument("--out", required=True)
    r.add_argument("--config")
    r.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    r.add_argument("--overwrite", action="store_true")

    v = sub.add_parser("validate", help="check SFT / RL rows in a .parquet / .jsonl / .json file")
    v.add_argument("file")
    v.add_argument("--kind", default="auto", choices=["auto", "sft", "rl"])
    v.add_argument("--key", action="append", default=[], metavar="NAME=COLUMN",
                   help="column names if not the defaults, e.g. images=imgs")
    v.add_argument("--media-root", help="base directory for relative media paths (default: current directory)")
    v.add_argument("--no-check-media", action="store_true", help="validate structure only, without local media existence checks")

    e = sub.add_parser("export", help="write training files (format from export.format) from a run directory")
    e.add_argument("--run", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--config")
    e.add_argument("--set", action="append", default=[])

    s = sub.add_parser("sft", help="import and export prepared SFT JSON / JSONL / Parquet samples")
    s.add_argument("--input", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--config")
    s.add_argument("--set", action="append", default=[])

    sub.add_parser("list", help="list registered components")
    c = sub.add_parser("show-config", help="print the merged config")
    c.add_argument("--config")
    c.add_argument("--set", action="append", default=[])
    a = ap.parse_args(argv)

    if a.cmd == "download":
        from .sources import download_sources
        m = download_sources(_load_yaml(a.sources), a.out, a.only)
        for s in m["sources"]:
            extra = f"{s['written']} written / {s['rows']} rows" if s["status"] == "ok" else s["error"]
            print(f"[{s['status']:5}] {s['role']:5} {s['name']}: {extra}")
        return 0 if all(s["status"] == "ok" for s in m["sources"]) else 1

    if a.cmd == "validate":
        from .export import validate_file
        res = validate_file(a.file, a.kind, dict(x.split("=", 1) for x in a.key),
                            check_media_exists=not a.no_check_media, media_root=a.media_root)
        json.dump(res, sys.stdout, ensure_ascii=False, indent=2)
        print()
        return 0 if res["invalid"] == 0 and "media_column_nulls" not in res else 1

    if a.cmd == "list":
        from .backends import REGISTRY
        from .export import EXPORTERS
        from .sources import LOADERS, MAPPERS
        from .stages import STAGES
        from .verifiers import VERIFIERS
        for name, keys in [("stages", STAGES), ("backends", REGISTRY), ("verifiers", VERIFIERS),
                           ("loaders", LOADERS), ("mappers", MAPPERS), ("exporters", EXPORTERS)]:
            print(f"{name:10} {', '.join(keys)}")
        return 0

    cfg = load_config(a.config, a.set)
    if a.cmd == "show-config":
        json.dump(cfg, sys.stdout, ensure_ascii=False, indent=2)
        print()
        return 0

    if a.cmd == "run":
        from .pipeline import load_tasks, render_report, run_pipeline
        inp = a.input
        if a.data:
            inp = inp or os.path.join(a.data, "tasks.jsonl")
            cfg["decontam_split"]["eval_sets"] = list(cfg["decontam_split"].get("eval_sets", [])) + \
                [os.path.join(os.path.abspath(a.data), "eval", "*.jsonl")]
        if not inp:
            ap.error("run needs --data or --input")
        report = run_pipeline(load_tasks(inp, a.format), cfg, a.out, overwrite=a.overwrite)
        print(render_report(report))
        return 0

    if a.cmd == "export":
        from .export import export
        for k, v in export(a.run, a.out, cfg.get("export", {})).items():
            print(f"{k}: {v}")
        return 0

    if a.cmd == "sft":
        from .export import export_sft_files
        for k, v in export_sft_files(a.input, a.out, cfg.get("export", {})).items():
            print(f"{k}: {v}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
