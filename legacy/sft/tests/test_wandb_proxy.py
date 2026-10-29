"""Check private W&B routing and process cleanup without SSH or network access."""

from contextlib import nullcontext
import json
import os
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest

from legacy.sft.utils import wandb_proxy


@pytest.fixture
def proxy_config(tmp_path, monkeypatch):
    """Provide a private dummy config independently of the user's real settings."""
    path = tmp_path / "proxy.json"
    path.write_text(json.dumps({
        "ssh_host": "192.0.2.1", "ssh_host_key_alias": "login.example",
        "proxy_url": "http://dummy-user:dummy-password@127.0.0.1:50170",
        "ssh_identity_file": "~/.ssh/example-key",
    }))
    monkeypatch.setenv("QWENPT_WANDB_PROXY_CONFIG", str(path))
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("WANDB_MODE", "online")
    for name in ("WANDB_HTTP_PROXY", "WANDB_HTTPS_PROXY"):
        monkeypatch.delenv(name, raising=False)
    return path


@pytest.mark.parametrize("arguments,updates", [
    (["torchrun"], {"SLURM_JOB_ID": ""}),
    (["torchrun"], {"WANDB_MODE": "offline"}),
    (["torchrun", "--cfg", "job"], {}),
    (["torchrun", "--help"], {}),
    (["torchrun", "trainer.logger=[wandb]", "trainer.logger=[console,file]"], {}),
])
def test_preview_and_non_online_commands_bypass_tunnel(proxy_config, arguments, updates):
    """Configuration previews and logging overrides must never require SSH."""
    environment = {**os.environ, **updates}
    assert wandb_proxy.proxy_config_path(arguments, environment) is None


def test_last_logger_override_can_enable_wandb(proxy_config):
    """Hydra's final logger override determines whether the proxy is needed."""
    assert wandb_proxy.proxy_config_path(
        ["torchrun", "trainer.logger=[console]", "trainer.logger=[console,wandb]"], dict(os.environ),
    ) == proxy_config


def test_existing_remote_proxy_is_honored_and_parent_environment_is_unchanged(proxy_config, monkeypatch):
    """An explicit remote proxy needs no SSH and must not affect Torch traffic."""
    monkeypatch.setenv("WANDB_HTTPS_PROXY", "http://proxy.example:8080")
    monkeypatch.setenv("http_proxy", "http://unwanted.example:8080")
    captured = {}

    def launch(command, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(wait=lambda **_: 7, poll=lambda: 7)

    monkeypatch.setattr(wandb_proxy.subprocess, "Popen", launch)
    monkeypatch.setattr(wandb_proxy, "start_tunnel", lambda _: pytest.fail("Unexpected SSH tunnel"))
    assert wandb_proxy.run(["train"]) == 7
    assert captured["env"]["WANDB_HTTPS_PROXY"] == "http://proxy.example:8080"
    assert "http_proxy" not in captured["env"]
    assert os.environ["http_proxy"] == "http://unwanted.example:8080"


def test_tunnel_url_identity_and_loopback_binding(proxy_config, monkeypatch):
    """Preserve dummy authentication in the child URL, keeping it out of SSH argv."""
    config = json.loads(proxy_config.read_text())
    captured = {}
    listener = SimpleNamespace(bind=lambda _: None, getsockname=lambda: ("127.0.0.1", 45123))
    monkeypatch.setattr(wandb_proxy.socket, "socket", lambda: nullcontext(listener))
    monkeypatch.setattr(wandb_proxy.socket, "create_connection", lambda *args, **kwargs: nullcontext())
    tunnel = SimpleNamespace(poll=lambda: None)

    def launch(command, **kwargs):
        captured["command"] = command
        return tunnel

    monkeypatch.setattr(wandb_proxy.subprocess, "Popen", launch)
    result, proxy_url = wandb_proxy.start_tunnel(config)
    command = captured["command"]
    assert result is tunnel
    assert proxy_url == "http://dummy-user:dummy-password@127.0.0.1:45123"
    assert command[command.index("-L") + 1] == "127.0.0.1:45123:127.0.0.1:50170"
    assert command[command.index("-i") + 1] == str(Path("~/.ssh/example-key").expanduser())
    assert "IdentitiesOnly=yes" in command and "StrictHostKeyChecking=yes" in command
    assert "dummy-password" not in " ".join(command)


def test_child_exit_and_tunnel_cleanup(proxy_config, monkeypatch, capsys):
    """Flush lifetime extends through child exit; clean up only the owned tunnel."""
    events = []
    tunnel = SimpleNamespace(
        poll=lambda: None, terminate=lambda: events.append("terminate tunnel"),
        wait=lambda **_: events.append("reap tunnel"),
    )
    monkeypatch.setattr(wandb_proxy, "start_tunnel", lambda _: (tunnel, "http://dummy:secret@127.0.0.1:45123"))
    captured = {}

    def launch(command, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(wait=lambda **_: events.append("child exited") or 17, poll=lambda: 17)

    monkeypatch.setattr(wandb_proxy.subprocess, "Popen", launch)
    assert wandb_proxy.run(["train"]) == 17
    assert events == ["child exited", "terminate tunnel", "reap tunnel"]
    assert captured["env"]["WANDB_HTTP_PROXY"] == captured["env"]["WANDB_HTTPS_PROXY"]
    assert "WANDB_HTTP_PROXY" not in os.environ
    assert "secret" not in capsys.readouterr().out


def test_signal_forwarded_to_child_group_and_handlers_restored(proxy_config, monkeypatch):
    """SIGTERM reaches the launched process group and retains its exit status."""
    monkeypatch.setenv("WANDB_MODE", "offline")
    original = signal.getsignal(signal.SIGTERM)
    forwarded = []
    monkeypatch.setattr(wandb_proxy.os, "killpg", lambda pid, signum: forwarded.append((pid, signum)))

    def wait(**kwargs):
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        return -signal.SIGTERM

    monkeypatch.setattr(wandb_proxy.subprocess, "Popen", lambda *args, **kwargs: SimpleNamespace(
        pid=98765, wait=wait, poll=lambda: -signal.SIGTERM,
    ))
    assert wandb_proxy.run(["train"]) == -signal.SIGTERM
    assert forwarded == [(98765, signal.SIGTERM)]
    assert signal.getsignal(signal.SIGTERM) is original


def test_child_spawn_failure_still_cleans_tunnel(proxy_config, monkeypatch):
    """A missing training executable must not leave an SSH process running."""
    stopped = []
    tunnel = SimpleNamespace(poll=lambda: None, terminate=lambda: stopped.append(True), wait=lambda **_: None)
    monkeypatch.setattr(wandb_proxy, "start_tunnel", lambda _: (tunnel, "http://127.0.0.1:45123"))

    def fail(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(wandb_proxy.subprocess, "Popen", fail)
    with pytest.raises(FileNotFoundError):
        wandb_proxy.run(["missing-train"])
    assert stopped == [True]
