"""General-model benchmark preparation, capability suites and API evaluation."""
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
    preparation = commands.add_parser("prepare", help="Prepare official instruction/science/code/math sources")
    preparation.add_argument("--config", required=True)
    preparation.add_argument("--output", required=True)
    preparation.add_argument("--download", action="store_true")
    preparation.add_argument("--tasks", nargs="+")
    preparation.add_argument("--model-revision", default="FILL_CHECKPOINT_REVISION")
    preparation.add_argument("--env-file")
    project = commands.add_parser("project", help="Run the Feishu capability matrix, preserving missing tasks")
    project.add_argument("action", choices=("list", "plan", "doctor", "run", "score"))
    project.add_argument("--config", required=True)
    project.add_argument("--profile", choices=("core", "scenarios", "all"))
    project.add_argument("--split", choices=("subset", "full"))
    project.add_argument("--output")
    project.add_argument("--tasks", nargs="+")
    project.add_argument("--model-revision")
    project.add_argument("--env-file")
    project.add_argument("--backend", choices=("api", "mock"), default="api")
    project.add_argument("--score-tag", default="default")
    args = parser.parse_args(argv)
    try:
        if args.env_file:
            load_env(args.env_file)
        if args.command == "prepare":
            from .prepare_general import prepare_general
            result = prepare_general(args.config, args.output, download=args.download,
                                     tasks=args.tasks, model_revision=args.model_revision)
        elif args.command == "project":
            from .project import load_project, catalog, execute_project
            config = load_project(args.config)
            if args.action == "list":
                result = catalog(config)
            else:
                if not args.profile or not args.split or not args.output:
                    raise ValueError("Project operations require explicit --profile, --split and --output")
                result = execute_project(config, args.action, args.split, args.output, profile=args.profile,
                                         tasks=args.tasks, model_revision=args.model_revision,
                                         backend=args.backend, score_tag=args.score_tag)
        elif args.command == "plan":
            config = load_config(args.config)
            result = plan(config, args.split, args.output)
        elif args.command == "run":
            config = load_config(args.config)
            result = run(config, args.split, args.output, args.backend)
        else:
            config = load_config(args.config)
            result = score(config, args.split, args.output, args.score_tag)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if result.get("complete", True) else 2
    except (ValueError, RuntimeError, OSError, ImportError, KeyError, TypeError) as error:
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
