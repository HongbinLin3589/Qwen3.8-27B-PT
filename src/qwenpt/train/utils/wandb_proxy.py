"""Keep a Slurm job's W&B-only SSH proxy alive for its training command."""

import ipaddress
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
from urllib.parse import urlsplit, urlunsplit


_STANDARD_PROXIES = ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")


def is_loopback(host: str | None) -> bool:
    """Return whether a hostname or IP address refers to local loopback."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host or "").is_loopback
    except ValueError:
        return False


def proxy_config_path(command: list[str], environment: dict[str, str]) -> Path | None:
    """Return an existing private config only for online Slurm commands using W&B."""
    if not environment.get("SLURM_JOB_ID") or environment.get("WANDB_MODE", "online") != "online":
        return None
    if any(arg in ("--cfg", "--help", "-h", "--hydra-help") or arg.startswith("--cfg=") for arg in command):
        return None
    loggers = [arg.split("=", 1)[1] for arg in command if arg.lstrip("+").startswith("trainer.logger=")]
    if loggers and "wandb" not in re.findall(r"[\w]+", loggers[-1]):
        return None
    path = Path(environment.get("QWENPT_WANDB_PROXY_CONFIG", "~/.config/qwenpt/wandb-proxy.json")).expanduser()
    return path if path.is_file() else None


def start_tunnel(config: dict) -> tuple[subprocess.Popen, str]:
    """Start a loopback SSH forward and return its process and rewritten proxy URL."""
    try:
        proxy = urlsplit(config["proxy_url"])
        remote_host = proxy.hostname
        remote_port = proxy.port or 80
        ssh_host = config["ssh_host"]
        alias = config["ssh_host_key_alias"]
        identity = config.get("ssh_identity_file")
        identity_options = ["-i", str(Path(identity).expanduser()), "-o", "IdentitiesOnly=yes"] if identity else []
        if proxy.scheme != "http" or not is_loopback(remote_host) or not ssh_host or not alias:
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise RuntimeError("Invalid private W&B proxy configuration.") from None
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        local_port = listener.getsockname()[1]
    # SSH checks the existing login-host identity even when routing to its numeric IP.
    tunnel = subprocess.Popen(
        ["ssh", "-N", "-T", *identity_options, "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
         "-o", "ConnectTimeout=10", "-o", "ConnectionAttempts=1",
         "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3",
         "-o", f"HostKeyAlias={alias}", "-o", "StrictHostKeyChecking=yes",
         "-L", f"127.0.0.1:{local_port}:{remote_host}:{remote_port}", "--", ssh_host],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            if tunnel.poll() is not None:
                raise RuntimeError("Could not establish the job-local W&B proxy.")
            try:
                with socket.create_connection(("127.0.0.1", local_port), timeout=0.2):
                    break
            except OSError:
                time.sleep(0.1)
        else:
            raise RuntimeError("Timed out establishing the job-local W&B proxy.")
    except BaseException:
        stop_process(tunnel)
        raise
    userinfo = proxy.netloc.rsplit("@", 1)[0] + "@" if "@" in proxy.netloc else ""
    proxy_url = urlunsplit((proxy.scheme, f"{userinfo}127.0.0.1:{local_port}", proxy.path, proxy.query, proxy.fragment))
    return tunnel, proxy_url


def stop_process(process: subprocess.Popen | None, *, group: bool = False) -> None:
    """Stop and reap the owned process, optionally signaling its command group."""
    if process is None or process.poll() is not None:
        return
    try:
        if group:
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if group:
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait()
    except ProcessLookupError:
        pass


def run(command: list[str]) -> int:
    """Run a command with optional private W&B routing and return its exit code."""
    environment = os.environ.copy()
    path = proxy_config_path(command, environment)
    tunnel = None
    child = None
    previous_handlers = {}

    def forward_signal(signum, _frame):
        if child is not None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass
        else:
            raise SystemExit(128 + signum)

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[signum] = signal.signal(signum, forward_signal)
        if path is not None:
            for name in _STANDARD_PROXIES:
                environment.pop(name, None)
            supplied_proxy = any(
                value and not is_loopback(urlsplit(value).hostname)
                for name in ("WANDB_HTTP_PROXY", "WANDB_HTTPS_PROXY")
                if (value := environment.get(name))
            )
            if not supplied_proxy:
                try:
                    config = json.loads(path.read_text())
                except (OSError, ValueError):
                    raise RuntimeError("Could not read private W&B proxy configuration.") from None
                tunnel, proxy_url = start_tunnel(config)
                environment["WANDB_HTTP_PROXY"] = proxy_url
                environment["WANDB_HTTPS_PROXY"] = proxy_url
                print("W&B: using a job-local proxy tunnel.", flush=True)
        child = subprocess.Popen(command, env=environment, start_new_session=True)
        while True:
            try:
                return child.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                if tunnel is not None and tunnel.poll() is not None:
                    raise RuntimeError("The job-local W&B proxy tunnel stopped unexpectedly.")
    finally:
        stop_process(child, group=True)
        stop_process(tunnel)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def main() -> None:
    """Launch the command following ``--`` without exposing private proxy details."""
    command = sys.argv[1:]
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise SystemExit("Usage: python -m qwenpt.train.utils.wandb_proxy -- <command...>")
    try:
        result = run(command)
    except (OSError, RuntimeError):
        print("W&B proxy setup or command execution failed; check connectivity and private configuration.", file=sys.stderr)
        result = 1
    raise SystemExit(result if result >= 0 else 128 - result)


if __name__ == "__main__":
    main()
