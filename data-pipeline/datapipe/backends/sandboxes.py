"""Sandbox implementations.

LocalDirSandbox  copies the task repo into a fresh temp dir and runs commands
                 on the host. No isolation: for development and small samples
                 only. Ignores env.image.
DockerSandbox    one new container per create(), from env.image, network off by
                 default. This is the one to use for real SWE / terminal tasks.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
import uuid
from typing import Optional

from ..schema import Solution, Task
from .base import Container, Sandbox


def _safe_rel(path: str) -> str:
    rel = os.path.normpath(path).lstrip("/")
    if rel.startswith(".."):
        raise ValueError(f"path escapes workdir: {path}")
    return rel


class _LocalContainer(Container):
    def __init__(self, task: Task, root: str):
        self.task, self.root = task, root
        self.workdir = os.path.join(root, "w")
        if task.env.repo_dir:
            shutil.copytree(task.env.repo_dir, self.workdir, symlinks=True)
        else:
            os.makedirs(self.workdir)
        if task.env.setup_cmd:
            code, out, err = self.run(task.env.setup_cmd, task.env.timeout_sec)
            if code != 0:
                raise RuntimeError(f"setup failed ({code}): {err[-500:]}")

    def apply(self, solution: Optional[Solution]) -> None:
        if not solution or not solution.file_edits:
            return
        for path, content in solution.file_edits.items():
            full = os.path.join(self.workdir, _safe_rel(path))
            os.makedirs(os.path.dirname(full), exist_ok=True)
            if content is None:
                if os.path.exists(full):
                    os.remove(full)
                continue
            with open(full, "w") as f:
                f.write(content)

    def run(self, cmd: str, timeout: float) -> tuple[int, str, str]:
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", HOME=self.root)
        p = subprocess.Popen(["bash", "-c", cmd], cwd=self.workdir, env=env, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        try:
            out, err = p.communicate(timeout=timeout)
            return p.returncode, out, err
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            out, err = p.communicate()
            return 124, out, err + "\n[timeout]"

    def read_file(self, path: str) -> Optional[str]:
        try:
            with open(os.path.join(self.workdir, _safe_rel(path)), errors="replace") as f:
                return f.read()
        except (FileNotFoundError, IsADirectoryError):
            return None

    def list_files(self) -> list[str]:
        out = []
        for d, _, files in os.walk(self.workdir):
            for f in files:
                out.append(os.path.relpath(os.path.join(d, f), self.workdir))
        return sorted(out)

    def destroy(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


class LocalDirSandbox(Sandbox):
    name = "local_dir"

    def __init__(self, tmp_root: Optional[str] = None):
        self.tmp_root = tmp_root

    def create(self, task: Task) -> Container:
        return _LocalContainer(task, tempfile.mkdtemp(prefix=f"dp-{task.id}-", dir=self.tmp_root))


class _DockerContainer(Container):
    def __init__(self, sb: "DockerSandbox", task: Task):
        self.sb, self.task = sb, task
        self.workdir = task.env.workdir
        image = task.env.image or sb.default_image
        if not image:
            raise ValueError(f"task {task.id}: no env.image and no default_image")
        name = f"dp-{uuid.uuid4().hex[:12]}"
        args = [sb.docker, "run", "-d", "--name", name, "-w", self.workdir]
        if not sb.network:
            args += ["--network", "none"]
        if sb.cpus:
            args += ["--cpus", str(sb.cpus)]
        if sb.memory:
            args += ["--memory", sb.memory]
        args += [image, "sleep", "infinity"]
        self.cid = self._docker(args, timeout=sb.start_timeout).strip()
        self._docker([sb.docker, "exec", self.cid, "mkdir", "-p", self.workdir])
        if task.env.repo_dir:
            self._docker([sb.docker, "cp", os.path.join(task.env.repo_dir, "."), f"{self.cid}:{self.workdir}"])
        if task.env.setup_cmd:
            code, out, err = self.run(task.env.setup_cmd, task.env.timeout_sec)
            if code != 0:
                self.destroy()
                raise RuntimeError(f"setup failed ({code}): {err[-500:]}")

    def _docker(self, args, timeout=120, stdin: Optional[str] = None) -> str:
        r = subprocess.run(args, input=stdin, text=True, capture_output=True, timeout=timeout)
        if r.returncode != 0:
            raise RuntimeError(f"{' '.join(args[:3])} failed: {r.stderr[-500:]}")
        return r.stdout

    def apply(self, solution: Optional[Solution]) -> None:
        if not solution or not solution.file_edits:
            return
        for path, content in solution.file_edits.items():
            rel = _safe_rel(path)
            if content is None:
                self.run(f"rm -f -- '{rel}'", 30)
                continue
            d = os.path.dirname(rel) or "."
            self._docker([self.sb.docker, "exec", "-i", "-w", self.workdir, self.cid, "sh", "-c",
                          f"mkdir -p '{d}' && cat > '{rel}'"], stdin=content)

    def run(self, cmd: str, timeout: float) -> tuple[int, str, str]:
        args = [self.sb.docker, "exec", "-w", self.workdir, self.cid, "timeout", "-s", "KILL",
                str(int(max(1, timeout))), "bash", "-c", cmd]
        try:
            r = subprocess.run(args, text=True, capture_output=True, timeout=timeout + 30)
        except subprocess.TimeoutExpired:
            return 124, "", "[timeout]"
        code = 124 if r.returncode == 137 else r.returncode
        return code, r.stdout, r.stderr

    def read_file(self, path: str) -> Optional[str]:
        code, out, _ = self.run(f"cat -- '{_safe_rel(path)}'", 30)
        return out if code == 0 else None

    def list_files(self) -> list[str]:
        _, out, _ = self.run("find . -type f | sed 's|^\\./||' | sort", 60)
        return [l for l in out.splitlines() if l]

    def destroy(self) -> None:
        subprocess.run([self.sb.docker, "rm", "-f", self.cid], capture_output=True)


class DockerSandbox(Sandbox):
    name = "docker"

    def __init__(self, default_image: Optional[str] = None, network: bool = False,
                 cpus: Optional[float] = None, memory: Optional[str] = None,
                 docker: str = "docker", start_timeout: float = 300):
        self.default_image, self.network = default_image, network
        self.cpus, self.memory, self.docker, self.start_timeout = cpus, memory, docker, start_timeout

    def create(self, task: Task) -> Container:
        return _DockerContainer(self, task)
