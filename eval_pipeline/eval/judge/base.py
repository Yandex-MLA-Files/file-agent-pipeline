"""Abstract judge interface.

A concrete implementation (RAGAS, DeepEval, a custom prompt, ...) plugs in
here without changes to the runner or the report: they only depend on
Judge.evaluate() taking a run-shaped DataFrame and returning one with
metric columns added.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Judge(ABC):
    """Base interface for any judge (RAGAS, DeepEval, a custom prompt, ...)."""

    #: names of the metric columns this judge adds to the DataFrame
    metric_names: tuple[str, ...] = ()

    @abstractmethod
    def evaluate(self, run_df: pd.DataFrame) -> pd.DataFrame:
        """Score each row.

        Parameters
        ----------
        run_df:
            DataFrame with the run schema (id, question, answer, contexts,
            ground_truth).

        Returns
        -------
        The same DataFrame with `self.metric_names` columns added, one
        float value (usually 0..1) per cell.
        """
        raise NotImplementedError
