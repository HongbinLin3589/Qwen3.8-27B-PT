"""CLI for prepared-bank API benchmarks (independent of the BOLT GPU runner)."""
import argparse
import json
import sys

from .suite import load_config, load_env, plan, run, score


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "run", "score"):
        sub = commands.add_parser(command)
        sub.add_argument("--config", required=True)
        sub.add_argument("--split", required=True, choices=("subset", "full"))
        sub.add_argument("--output", required=True)
        sub.add_argument("--env-file")
        if command == "run":
            sub.add_argument("--backend", choices=("api", "mock"), default="api")
        if command == "score":
            sub.add_argument("--score-tag", default="default")
    args = parser.parse_args(argv)
    try:
        if args.env_file:
            load_env(args.env_file)
        config = load_config(args.config)
        if args.command == "plan":
            result = plan(config, args.split, args.output)
        elif args.command == "run":
            result = run(config, args.split, args.output, args.backend)
        else:
            result = score(config, args.split, args.output, args.score_tag)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if result.get("complete", True) else 2
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        # Config validation messages are written without credential values; keep
        # unexpected transport/library messages out of user-visible tracebacks.
        message = str(error)
        for key, value in __import__("os").environ.items():
            if value and any(word in key.upper() for word in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
                message = message.replace(value, "[REDACTED]")
        print(json.dumps({"ok": False, "error": type(error).__name__, "message": message}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
