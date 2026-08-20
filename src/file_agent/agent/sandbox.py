"""Restricted Python execution for the agent's data tools.

``query_table`` and ``calculate`` let the model compute over parsed tables
(sums, group-bys, joins, ratios) instead of reading hundreds of rows and
guessing. The code is model-written, so it runs under a guard: no imports, no
dunder or private attribute access, no file or network I/O through the pandas
and numpy surfaces, a small builtin whitelist, a wall-clock limit and a cap on
the output size. This is a guard against mistakes and careless code, not a
security boundary against a hostile model; the data it touches is the user's
own upload, already in memory.
"""

import ast
import builtins
import collections
import functools
import io
import math
import re
import statistics
import threading
from typing import Any

DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_MAX_OUTPUT_CHARS = 4000

_SAFE_BUILTINS: dict[str, Any] = {
    name: getattr(builtins, name)
    for name in (
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "divmod",
        "enumerate",
        "filter",
        "float",
        "format",
        "frozenset",
        "int",
        "isinstance",
        "len",
        "list",
        "map",
        "max",
        "min",
        "pow",
        "print",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
    )
}

# pandas/numpy entry points that read or write outside the in-memory frames.
_BLOCKED_ATTRIBUTES = frozenset(
    {
        "io",
        "compat",
        "testing",
        "ctypeslib",
        "f2py",
        "load",
        "loads",
        "save",
        "savetxt",
        "savez",
        "fromfile",
        "memmap",
        "tofile",
        "pipe",
        "apply_async",
        "eval",
    }
)
# ``to_*`` conversions that stay in memory and are genuinely useful; every other
# ``to_*`` / ``read_*`` name is treated as I/O and rejected.
_ALLOWED_CONVERSIONS = frozenset(
    {
        "to_dict",
        "to_list",
        "tolist",
        "to_numpy",
        "to_string",
        "to_markdown",
        "to_frame",
        "to_records",
        "to_datetime",
        "to_numeric",
        "to_timedelta",
        "to_period",
        "to_series",
    }
)
_BLOCKED_NAMES = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "open",
        "input",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "type",
        "super",
        "breakpoint",
        "exit",
        "quit",
        "help",
        "memoryview",
        "__import__",
        "__builtins__",
    }
)


class SandboxError(Exception):
    """The code was rejected by the guard or failed while running."""


def base_namespace() -> dict[str, Any]:
    """Modules and helpers every sandboxed snippet can use."""
    namespace: dict[str, Any] = {
        "math": math,
        "re": re,
        "statistics": statistics,
        "Counter": collections.Counter,
        "defaultdict": collections.defaultdict,
    }
    try:
        import pandas as pd

        namespace["pd"] = pd
    except ImportError:  # pragma: no cover - pandas is a project dependency
        pass
    try:
        import numpy as np

        namespace["np"] = np
    except ImportError:  # pragma: no cover
        pass
    return namespace


def run_code(
    code: str,
    namespace: dict[str, Any] | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
) -> str:
    """Execute ``code`` and return what it printed plus the value of its last expression."""
    code = code.strip()
    if not code:
        raise SandboxError("No code was given.")
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise SandboxError(f"Syntax error: {exc.msg} (line {exc.lineno}).") from exc
    _Guard().visit(tree)

    # Show the value of a trailing expression, like a notebook cell does.
    trailing: ast.Expr | None = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        trailing = tree.body.pop()  # type: ignore[assignment]

    scope: dict[str, Any] = dict(base_namespace())
    if namespace:
        scope.update(namespace)
    scope["__builtins__"] = dict(_SAFE_BUILTINS)

    body_code = compile(tree, "<agent>", "exec")
    value_code = (
        compile(ast.Expression(trailing.value), "<agent>", "eval") if trailing is not None else None
    )

    # print() is rebound to a private buffer instead of redirecting sys.stdout:
    # the redirect would be process-wide, and a snippet that outlives its
    # timeout would keep every other thread's output captured.
    stdout = io.StringIO()
    scope["__builtins__"]["print"] = functools.partial(print, file=stdout)
    outcome: dict[str, Any] = {}

    def target() -> None:
        try:
            exec(body_code, scope)  # noqa: S102 - guarded by _Guard above
            if value_code is not None:
                outcome["value"] = eval(value_code, scope)  # noqa: S307 - guarded
        except BaseException as exc:  # noqa: BLE001 - reported to the model
            outcome["error"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(timeout_seconds)
    if worker.is_alive():
        raise SandboxError(f"Execution exceeded {timeout_seconds:g} s and was abandoned.")
    if "error" in outcome:
        exc = outcome["error"]
        raise SandboxError(f"{type(exc).__name__}: {exc}")

    parts: list[str] = []
    printed = stdout.getvalue().rstrip()
    if printed:
        parts.append(printed)
    if "value" in outcome and outcome["value"] is not None:
        parts.append(_format_value(outcome["value"]))
    output = "\n".join(parts).strip()
    if not output:
        output = "(no output; print() the values you need)"
    if len(output) > max_output_chars:
        output = output[:max_output_chars].rstrip() + "\n[output truncated]"
    return output


def _format_value(value: Any) -> str:
    try:
        import pandas as pd

        if isinstance(value, pd.DataFrame | pd.Series):
            with pd.option_context(
                "display.max_rows", 60, "display.max_columns", 40, "display.width", 200
            ):
                return value.to_string(max_rows=60)
    except ImportError:  # pragma: no cover
        pass
    # numpy scalars print as ``np.float64(4.0)``; the model wants ``4.0``.
    if hasattr(value, "item") and not isinstance(value, str | bytes):
        try:
            value = value.item()
        except (ValueError, TypeError, AttributeError):
            pass
    if isinstance(value, float):
        return repr(round(value, 6))
    return repr(value)


class _Guard(ast.NodeVisitor):
    """Reject syntax that reaches outside the in-memory data."""

    def visit_Import(self, node: ast.Import) -> None:
        raise SandboxError(
            "Imports are not allowed; pandas (pd), numpy (np), math and re are preloaded."
        )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        raise SandboxError(
            "Imports are not allowed; pandas (pd), numpy (np), math and re are preloaded."
        )

    def visit_Attribute(self, node: ast.Attribute) -> None:
        name = node.attr
        if name.startswith("_"):
            raise SandboxError(f"Private attribute '{name}' is not allowed.")
        if name in _BLOCKED_ATTRIBUTES or (
            (name.startswith("to_") or name.startswith("read_"))
            and name not in _ALLOWED_CONVERSIONS
        ):
            raise SandboxError(f"'{name}' is not available here (no file or network I/O).")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in _BLOCKED_NAMES or node.id.startswith("__"):
            raise SandboxError(f"'{node.id}' is not allowed.")
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        raise SandboxError("'global' is not allowed.")

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        raise SandboxError("'nonlocal' is not allowed.")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        raise SandboxError("Class definitions are not allowed.")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        raise SandboxError("Async code is not allowed.")

    def visit_Await(self, node: ast.Await) -> None:
        raise SandboxError("Async code is not allowed.")
