import subprocess
from types import SimpleNamespace

import pytest

from file_agent.agent.sandbox import run_sandboxed_code


def test_run_sandboxed_code_reports_missing_input_file(tmp_path):
    result = run_sandboxed_code(
        source_paths={"missing.xlsx": tmp_path / "missing.xlsx"}, code="print(1)"
    )

    assert result.exit_code == 1
    assert "not found" in result.stderr


def test_run_sandboxed_code_rejects_file_names_with_a_colon(tmp_path):
    source_path = tmp_path / "data.xlsx"
    source_path.write_text("fake xlsx", encoding="utf-8")

    result = run_sandboxed_code(source_paths={"bad:name.xlsx": source_path}, code="print(1)")

    assert result.exit_code == 1
    assert "Unsupported file name" in result.stderr


def test_run_sandboxed_code_returns_captured_output(monkeypatch, tmp_path):
    source_path = tmp_path / "data.xlsx"
    source_path.write_text("fake xlsx", encoding="utf-8")
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="42\n", stderr="", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_sandboxed_code(source_paths={"data.xlsx": source_path}, code="print(42)")

    assert result.stdout == "42\n"
    assert result.exit_code == 0
    assert not result.timed_out
    assert not result.truncated

    command = calls[0]
    assert command[:3] == ["docker", "run", "--rm"]
    assert "--network" in command and "none" in command
    assert "--read-only" in command
    assert any(part.startswith("--memory=") for part in command)
    assert f"{source_path.resolve()}:/data/data.xlsx:ro" in " ".join(command)


def test_run_sandboxed_code_mounts_every_document(monkeypatch, tmp_path):
    first_path = tmp_path / "first.xlsx"
    first_path.write_text("fake xlsx", encoding="utf-8")
    second_path = tmp_path / "notes.txt"
    second_path.write_text("fake text", encoding="utf-8")
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    run_sandboxed_code(
        source_paths={"first.xlsx": first_path, "notes.txt": second_path}, code="pass"
    )

    command_str = " ".join(calls[0])
    assert f"{first_path.resolve()}:/data/first.xlsx:ro" in command_str
    assert f"{second_path.resolve()}:/data/notes.txt:ro" in command_str


def test_run_sandboxed_code_runs_with_no_documents(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="ok\n", stderr="", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_sandboxed_code(source_paths={}, code="print('ok')")

    assert result.stdout == "ok\n"
    assert "/data/" not in " ".join(calls[0])


def test_run_sandboxed_code_kills_container_on_timeout(monkeypatch, tmp_path):
    source_path = tmp_path / "data.xlsx"
    source_path.write_text("fake xlsx", encoding="utf-8")
    kill_calls = []

    def fake_run(command, **kwargs):
        if command[:2] == ["docker", "kill"]:
            kill_calls.append(command)
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        raise subprocess.TimeoutExpired(cmd=command, timeout=kwargs.get("timeout", 0))

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_sandboxed_code(
        source_paths={"data.xlsx": source_path}, code="while True: pass", timeout_seconds=1
    )

    assert result.timed_out
    assert len(kill_calls) == 1


def test_run_sandboxed_code_reports_missing_docker_cli(monkeypatch, tmp_path):
    source_path = tmp_path / "data.xlsx"
    source_path.write_text("fake xlsx", encoding="utf-8")

    def fake_run(command, **kwargs):
        raise FileNotFoundError("docker not found")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_sandboxed_code(source_paths={"data.xlsx": source_path}, code="print(1)")

    assert result.exit_code == 1
    assert "docker CLI not found" in result.stderr


def test_run_sandboxed_code_truncates_long_output(monkeypatch, tmp_path):
    source_path = tmp_path / "data.xlsx"
    source_path.write_text("fake xlsx", encoding="utf-8")

    def fake_run(command, **kwargs):
        return SimpleNamespace(stdout="x" * 100, stderr="", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_sandboxed_code(
        source_paths={"data.xlsx": source_path}, code="print('x' * 100)", max_output_chars=10
    )

    assert len(result.stdout) == 10
    assert result.truncated


@pytest.mark.parametrize("exit_code", [1, 2])
def test_run_sandboxed_code_surfaces_non_zero_exit_codes(monkeypatch, tmp_path, exit_code):
    source_path = tmp_path / "data.xlsx"
    source_path.write_text("fake xlsx", encoding="utf-8")

    def fake_run(command, **kwargs):
        return SimpleNamespace(stdout="", stderr="Traceback...", returncode=exit_code)

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_sandboxed_code(source_paths={"data.xlsx": source_path}, code="raise ValueError()")

    assert result.exit_code == exit_code
    assert result.stderr == "Traceback..."
