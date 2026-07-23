from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("id", "question", "answer_model", "contexts", "answer")


@dataclass
class RunValidationError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


def load_run(path: str | Path) -> pd.DataFrame:

    path = Path(path)
    if not path.exists():
        raise RunValidationError(f"file not found: {path}")

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
        if "contexts" in df.columns:
            # CSV can't hold native lists — expect a JSON-encoded string
            import json

            df["contexts"] = df["contexts"].apply(json.loads)
    else:
        raise RunValidationError(f"unsupported file format: {path.suffix}")

    _validate_schema(df)
    df["contexts"] = df["contexts"].apply(list)
    return df


def _validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise RunValidationError(f"run file is missing required columns: {missing}")

    if df["id"].isnull().any():
        raise RunValidationError("some rows have an empty id")

    dup_ids = df["id"][df["id"].duplicated()].tolist()
    if dup_ids:
        raise RunValidationError(f"duplicate ids: {dup_ids[:5]}")

    bad_contexts = df[
        ~df["contexts"].apply(lambda x: pd.api.types.is_list_like(x) and not isinstance(x, dict))
    ]
    if len(bad_contexts) > 0:
        raise RunValidationError(
            f"contexts must be list[str], but {len(bad_contexts)} row(s) "
            f"aren't (first offending id: {bad_contexts['id'].iloc[0]}). "
            f"Contexts were likely joined into a single string instead of "
            f"kept as a list."
        )

    empty_answers = (
        df["answer_model"].isnull().sum() + (df["answer_model"].astype(str).str.strip() == "").sum()
    )
    if empty_answers > 0:
        raise RunValidationError(f"{empty_answers} row(s) with an empty answer_model")
