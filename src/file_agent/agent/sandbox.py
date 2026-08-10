import logging
import os
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_SANDBOX_IMAGE = os.getenv("SANDBOX_IMAGE", "file-agent-sandbox:latest")
DEFAULT_TIMEOUT_SECONDS = 10.0
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
    source_path: Path,
    code: str,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
    cpus: float = DEFAULT_CPUS,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    image: str = DEFAULT_SANDBOX_IMAGE,
) -> SandboxResult:
    """Run untrusted Python against one input file in an isolated, ephemeral container.

    No network, read-only rootfs, dropped capabilities, and hard resource
    limits - the container is the security boundary, not the Python code.
    """
    source_path = Path(source_path).resolve()
    if not source_path.is_file():
        return SandboxResult(
            stdout="",
            stderr=f"Input file not found: {source_path}",
            exit_code=1,
            timed_out=False,
            truncated=False,
        )

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
            "/tmp:size=64m",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            "1000:1000",
            "-v",
            f"{code_path}:/sandbox/code.py:ro",
            "-v",
            f"{source_path}:/data/input.xlsx:ro",
            "--workdir",
            "/sandbox",
            image,
            "python",
            "/sandbox/code.py",
        ]

        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=timeout_seconds
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

        stdout, stdout_truncated = _truncate(completed.stdout, max_output_chars)
        stderr, stderr_truncated = _truncate(completed.stderr, max_output_chars)
        return SandboxResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=completed.returncode,
            timed_out=False,
            truncated=stdout_truncated or stderr_truncated,
        )


def _force_kill(container_name: str) -> None:
    # subprocess's own timeout kills the client process, not the container it
    # started - without this the sandboxed code keeps running past the timeout.
    subprocess.run(["docker", "kill", container_name], capture_output=True, timeout=5, check=False)


def _truncate(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True
