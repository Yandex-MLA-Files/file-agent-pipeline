import logging
import os
import subprocess
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_SANDBOX_IMAGE = os.getenv("SANDBOX_IMAGE", "file-agent-sandbox:latest")
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MEMORY_LIMIT_MB = 512
DEFAULT_CPUS = 1.0
DEFAULT_MAX_OUTPUT_CHARS = 4000


@dataclass(frozen=True)
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    truncated: bool


def run_sandboxed_code(
    source_paths: Mapping[str, Path],
    code: str,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
    cpus: float = DEFAULT_CPUS,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    image: str = DEFAULT_SANDBOX_IMAGE,
) -> SandboxResult:

    resolved_paths: dict[str, Path] = {}
    for file_name, source_path in source_paths.items():
        if ":" in file_name:
            return SandboxResult(
                stdout="",
                stderr=f"Unsupported file name for mounting: {file_name}",
                exit_code=1,
                timed_out=False,
                truncated=False,
            )
        resolved_path = Path(source_path).resolve()
        if not resolved_path.is_file():
            return SandboxResult(
                stdout="",
                stderr=f"Input file not found: {resolved_path}",
                exit_code=1,
                timed_out=False,
                truncated=False,
            )
        resolved_paths[file_name] = resolved_path

    container_name = f"sandbox-{uuid.uuid4().hex[:12]}"
    with tempfile.TemporaryDirectory(prefix="file-agent-sandbox-") as tmp_dir:
        code_path = Path(tmp_dir) / "code.py"
        code_path.write_text(code, encoding="utf-8")

        command = [
            "docker",
            "run",
            "--rm",
            "--name",
            container_name,
            "--network",
            "none",
            f"--memory={memory_limit_mb}m",
            f"--memory-swap={memory_limit_mb}m",
            f"--cpus={cpus}",
            "--pids-limit=64",
            "--read-only",
            "--tmpfs",
            "/tmp:size=64m,mode=1777",
            "--tmpfs",
            "/sandbox/work:size=64m,mode=1777",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            "1000:1000",
            "-v",
            f"{code_path}:/sandbox/code.py:ro",
        ]
        for file_name, resolved_path in resolved_paths.items():
            command += ["-v", f"{resolved_path}:/data/{file_name}:ro"]
        command += ["--workdir", "/sandbox", image, "python", "/sandbox/code.py"]

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            _force_kill(container_name)
            return SandboxResult(stdout="", stderr="", exit_code=1, timed_out=True, truncated=False)
        except FileNotFoundError:
            logger.error("Docker CLI not found - sandbox tool unavailable on this host.")
            return SandboxResult(
                stdout="",
                stderr="Sandbox unavailable: docker CLI not found on this host.",
                exit_code=1,
                timed_out=False,
                truncated=False,
            )

        stdout, stdout_truncated = _truncate(completed.stdout or "", max_output_chars)
        stderr, stderr_truncated = _truncate(completed.stderr or "", max_output_chars)
        return SandboxResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=completed.returncode,
            timed_out=False,
            truncated=stdout_truncated or stderr_truncated,
        )


def _force_kill(container_name: str) -> None:

    subprocess.run(["docker", "kill", container_name], capture_output=True, timeout=5, check=False)


def _truncate(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True
