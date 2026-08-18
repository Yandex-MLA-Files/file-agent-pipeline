from __future__ import annotations

import json
import logging
import os
import uuid
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

import numpy as np
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings as LangchainHuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas import RunConfig
from ragas import evaluate as ragas_evaluate
from ragas.callbacks import ChainRun
from ragas.cost import BaseCallbackHandler, LLMResult
from ragas.dataset_schema import EvaluationDataset
from ragas.embeddings import BaseRagasEmbeddings, LangchainEmbeddingsWrapper
from ragas.llms.base import BaseRagasLLM, LangchainLLMWrapper
from ragas.metrics import (
    AnswerCorrectness,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)
from ragas.metrics.base import Metric, ModeMetric

DEFAULT_PRICE_PER_1K_INPUT_TOKENS = 0.3
DEFAULT_PRICE_PER_1K_OUTPUT_TOKENS = 0.5
DEFAULT_PRICE_PER_1K_CACHED_TOKENS = 0.075

DEFAULT_USAGE_LOG_PATH = "logs/usage_log.jsonl"
DEFAULT_TRACE_LOG_PATH = "logs/judge_trace_log.jsonl"
DEFAULT_MAX_CONCURRENCY = 8
DEFAULT_TIMEOUT = 300
DEFAULT_MAX_RETRIES = 3
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

logger = logging.getLogger(__name__)

RUN_TO_RAGAS_COLUMNS = {
    "question": "user_input",
    "answer_model": "response",
    "contexts": "retrieved_contexts",
    "answer": "reference",
}


def _result_column(metric: Metric) -> str:

    if isinstance(metric, ModeMetric):
        return f"{metric.name}(mode={metric.mode})"
    return metric.name


_LANGUAGE_MATCH_INSTRUCTION = (
    " Respond in the same language as the input text you are given "
    "(question, answer, or context) -- e.g. answer in Russian for Russian "
    "input, English for English input."
)


def _match_response_language_to_input(
    faithfulness: Faithfulness,
    answer_correctness: AnswerCorrectness,
    answer_relevancy: ResponseRelevancy,
    context_precision: LLMContextPrecisionWithReference,
    context_recall: LLMContextRecall,
) -> None:

    faithfulness.statement_generator_prompt.instruction += _LANGUAGE_MATCH_INSTRUCTION
    faithfulness.nli_statements_prompt.instruction += _LANGUAGE_MATCH_INSTRUCTION
    answer_correctness.statement_generator_prompt.instruction += _LANGUAGE_MATCH_INSTRUCTION
    answer_correctness.correctness_prompt.instruction += _LANGUAGE_MATCH_INSTRUCTION
    answer_relevancy.question_generation.instruction += _LANGUAGE_MATCH_INSTRUCTION
    context_precision.context_precision_prompt.instruction += _LANGUAGE_MATCH_INSTRUCTION
    context_recall.context_recall_prompt.instruction += _LANGUAGE_MATCH_INSTRUCTION


class _TokenUsageCallback(BaseCallbackHandler):
    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.cached_tokens = 0
        self.model = ""

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        llm_output = response.llm_output or {}
        usage = llm_output.get("token_usage") or {}
        self.input_tokens += usage.get("prompt_tokens", 0)
        self.output_tokens += usage.get("completion_tokens", 0)
        cached_details = usage.get("prompt_tokens_details") or {}
        self.cached_tokens += cached_details.get("cached_tokens", 0)
        self.model = llm_output.get("model_name") or self.model


def _usage_cost(input_tokens: int, output_tokens: int, cached_tokens: int) -> float:
    billable_input_tokens = max(input_tokens - cached_tokens, 0)
    price_input = float(
        os.environ.get("JUDGE_PRICE_PER_1K_INPUT_TOKENS", DEFAULT_PRICE_PER_1K_INPUT_TOKENS)
    )
    price_output = float(
        os.environ.get("JUDGE_PRICE_PER_1K_OUTPUT_TOKENS", DEFAULT_PRICE_PER_1K_OUTPUT_TOKENS)
    )
    price_cached = float(
        os.environ.get("JUDGE_PRICE_PER_1K_CACHED_TOKENS", DEFAULT_PRICE_PER_1K_CACHED_TOKENS)
    )
    return (
        billable_input_tokens / 1000 * price_input
        + cached_tokens / 1000 * price_cached
        + output_tokens / 1000 * price_output
    )


def _log_usage(
    path: str | Path,
    usage_cb: _TokenUsageCallback,
    n_rows: int,
    fallback_model: str = "",
    *,
    fallback_metric_attempts: dict[str, int] | None = None,
) -> None:
    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "model": usage_cb.model or fallback_model,
        "n_rows": n_rows,
        "input_tokens": usage_cb.input_tokens,
        "cached_tokens": usage_cb.cached_tokens,
        "output_tokens": usage_cb.output_tokens,
        "cost_rub": round(
            _usage_cost(usage_cb.input_tokens, usage_cb.output_tokens, usage_cb.cached_tokens), 4
        ),
        "fallback_metric_attempts": fallback_metric_attempts or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _json_default_trace(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return str(obj)


def _per_run_trace_path(base_path: str | Path, now: datetime) -> Path:

    base = Path(base_path)
    stamp = now.strftime("%Y%m%dT%H%M%S%f") + "Z"
    return base.with_name(f"{base.stem}_{stamp}{base.suffix}")


def _parse_row_traces(
    ragas_traces: dict[str, ChainRun],
    run_id: str | None,
) -> list[dict[str, Any]]:
    root_traces = [
        chain_trace for chain_trace in ragas_traces.values() if chain_trace.parent_run_id == run_id
    ]
    root_trace = root_traces[0]

    row_traces = []
    for row_uuid in root_trace.children:
        row_trace = ragas_traces[row_uuid]
        scores: dict[str, Any] = {}
        calls: dict[str, list[dict[str, Any]]] = {}
        for metric_uuid in row_trace.children:
            metric_trace = ragas_traces[metric_uuid]
            scores[metric_trace.name] = metric_trace.outputs.get("output", {})
            metric_calls = []
            for prompt_uuid in metric_trace.children:
                prompt_trace = ragas_traces[prompt_uuid]
                output = prompt_trace.outputs.get("output", {})
                output = output[0] if isinstance(output, list) else output
                metric_calls.append(
                    {
                        "prompt": prompt_trace.name,
                        "input": prompt_trace.inputs.get("data", {}),
                        "output": output,
                    }
                )
            calls[metric_trace.name] = metric_calls
        row_traces.append({"scores": scores, "calls": calls})

    return row_traces


def _log_judge_trace(
    path: str | Path,
    run_df: pd.DataFrame,
    row_traces: list[dict[str, Any]],
) -> None:

    run_id = str(uuid.uuid4())
    now = datetime.now(UTC)
    timestamp = now.isoformat()

    run_path = _per_run_trace_path(path, now)
    run_path.parent.mkdir(parents=True, exist_ok=True)
    with open(run_path, "w", encoding="utf-8") as f:
        for (_, row), row_trace in zip(run_df.iterrows(), row_traces, strict=True):
            entry = {
                "run_id": run_id,
                "timestamp": timestamp,
                "id": row["id"],
                "question": row["question"],
                "answer_model": row["answer_model"],
                "reference": row["answer"],
                "contexts": list(row["contexts"]),
                "verdict": row_trace["scores"],
                "reasoning_trace": row_trace["calls"],
                "fallback_metrics": row_trace.get("fallback_metrics", []),
                "fallback_failed_metrics": row_trace.get("fallback_failed_metrics", []),
            }
            f.write(json.dumps(entry, ensure_ascii=False, default=_json_default_trace) + "\n")


def _optional_positive_int(env_name: str) -> int | None:
    raw_value = os.environ.get(env_name, "").strip()
    if not raw_value:
        return None
    value = int(raw_value)
    if value <= 0:
        raise ValueError(f"{env_name} must be greater than zero")
    return value


def _openai_compatible_reasoning_mode(reasoning_mode: str | None) -> str | None:
    """Return a reasoning mode only when the configured gateway accepts it.

    Yandex AI Studio exposes ``reasoning_mode`` through its native SDK, but its
    OpenAI-compatible ``/v1`` endpoint currently rejects that extra request
    field. In that configuration a bounded provider-default retry is safer than
    sending a request that is guaranteed to fail with HTTP 400.
    """

    if not reasoning_mode:
        return None

    base_url = os.environ.get("JUDGE_BASE_URL", "")
    hostname = (urlparse(base_url).hostname or "").lower()
    if hostname == "ai.api.cloud.yandex.net":
        logger.warning(
            "JUDGE reasoning_mode=%s is not supported by the Yandex "
            "OpenAI-compatible endpoint; using provider_default for this attempt",
            reasoning_mode,
        )
        return None
    return reasoning_mode


class _BoundedLangchainLLMWrapper(LangchainLLMWrapper):
    """Keep an HTTP timeout tighter than Ragas' whole-metric timeout."""

    def __init__(self, langchain_llm: ChatOpenAI, request_timeout: int | None) -> None:
        self._request_timeout = request_timeout
        super().__init__(langchain_llm)

    def set_run_config(self, run_config: RunConfig) -> None:
        super().set_run_config(run_config)
        if self._request_timeout is not None:
            self.langchain_llm.request_timeout = min(self._request_timeout, run_config.timeout)


def _build_default_llm(
    model: str,
    *,
    reasoning_mode: str | None = None,
    max_tokens: int | None = None,
    request_timeout: int | None = None,
) -> BaseRagasLLM:
    chat_kwargs: dict[str, Any] = {
        "base_url": os.environ.get("JUDGE_BASE_URL"),
        "api_key": os.environ.get("JUDGE_API_KEY", "not-needed"),
        "model": model,
        "temperature": 0,
    }
    if max_tokens is not None:
        chat_kwargs["max_tokens"] = max_tokens
    if request_timeout is not None:
        chat_kwargs["timeout"] = request_timeout
    compatible_reasoning_mode = _openai_compatible_reasoning_mode(reasoning_mode)
    if compatible_reasoning_mode:
        # Yandex AI Studio exposes its provider-specific reasoning switch as
        # an additional OpenAI-compatible request field. Keep it out of the
        # primary request unless explicitly configured, so existing judge
        # behaviour remains unchanged.
        chat_kwargs["extra_body"] = {"reasoning_mode": compatible_reasoning_mode.upper()}

    chat = ChatOpenAI(
        **chat_kwargs,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return _BoundedLangchainLLMWrapper(chat, request_timeout)


def _build_default_embeddings(model: str) -> BaseRagasEmbeddings:
    embeddings = LangchainHuggingFaceEmbeddings(model_name=model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return LangchainEmbeddingsWrapper(embeddings)


def _build_metrics() -> list[Metric]:
    faithfulness = Faithfulness(name="faithfulness")
    answer_correctness = AnswerCorrectness(name="answer_correctness")
    answer_relevancy = ResponseRelevancy(name="answer_relevancy", strictness=1)
    context_precision = LLMContextPrecisionWithReference(name="context_precision")
    context_recall = LLMContextRecall(name="context_recall")
    _match_response_language_to_input(
        faithfulness, answer_correctness, answer_relevancy, context_precision, context_recall
    )
    return [
        faithfulness,
        answer_correctness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]


def _annotate_trace_attempt(
    row_traces: list[dict[str, Any]], *, attempt: str, reasoning_mode: str
) -> None:
    for row_trace in row_traces:
        for calls in row_trace["calls"].values():
            for call in calls:
                call["attempt"] = attempt
                call["reasoning_mode"] = reasoning_mode


def _numeric_scores(scores: pd.DataFrame, metric: Metric) -> np.ndarray:
    column = _result_column(metric)
    if column not in scores:
        return np.full(len(scores), np.nan, dtype=float)
    values = pd.to_numeric(scores[column], errors="coerce").to_numpy(dtype=float, copy=True)
    values[~np.isfinite(values)] = np.nan
    return values


class RagasJudge:
    #: names of the metric columns evaluate() adds to the DataFrame
    metric_names = (
        "faithfulness",
        "answer_correctness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    )

    def __init__(
        self,
        model: str | None = None,
        llm: BaseRagasLLM | None = None,
        fallback_llm: BaseRagasLLM | None = None,
        embeddings: BaseRagasEmbeddings | None = None,
        usage_log_path: str | Path | None = None,
        trace_log_path: str | Path | None = None,
    ):
        model_name = model or os.environ.get("JUDGE_MODEL", "")
        resolved_model = model_name or os.environ["JUDGE_MODEL"]
        self.reasoning_mode = os.environ.get("JUDGE_REASONING_MODE", "").strip()
        self.fallback_reasoning_mode = os.environ.get("JUDGE_FALLBACK_REASONING_MODE", "").strip()
        primary_request_reasoning_mode = _openai_compatible_reasoning_mode(self.reasoning_mode)
        fallback_request_reasoning_mode = _openai_compatible_reasoning_mode(
            self.fallback_reasoning_mode
        )
        self.effective_reasoning_mode = primary_request_reasoning_mode or "provider_default"
        self.effective_fallback_reasoning_mode = (
            fallback_request_reasoning_mode or "provider_default"
        )
        self.max_tokens = _optional_positive_int("JUDGE_MAX_TOKENS")
        self.request_timeout = _optional_positive_int("JUDGE_REQUEST_TIMEOUT")
        self.llm = llm or _build_default_llm(
            resolved_model,
            reasoning_mode=primary_request_reasoning_mode,
            max_tokens=self.max_tokens,
            request_timeout=self.request_timeout,
        )
        if fallback_llm is not None:
            self.fallback_llm = fallback_llm
        elif self.fallback_reasoning_mode:
            self.fallback_llm = _build_default_llm(
                resolved_model,
                reasoning_mode=fallback_request_reasoning_mode,
                max_tokens=self.max_tokens,
                request_timeout=self.request_timeout,
            )
        else:
            self.fallback_llm = None
        self.embeddings = embeddings or _build_default_embeddings(
            os.environ.get("JUDGE_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        )
        self._model_name = model_name
        self.usage_log_path = usage_log_path or os.environ.get(
            "JUDGE_USAGE_LOG_PATH", DEFAULT_USAGE_LOG_PATH
        )
        self.trace_log_path = trace_log_path or os.environ.get(
            "JUDGE_TRACE_LOG_PATH", DEFAULT_TRACE_LOG_PATH
        )
        self.max_concurrency = int(os.environ.get("JUDGE_MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY))
        self.timeout = int(os.environ.get("JUDGE_TIMEOUT", DEFAULT_TIMEOUT))
        self.max_retries = int(os.environ.get("JUDGE_MAX_RETRIES", DEFAULT_MAX_RETRIES))
        self._metrics = _build_metrics()
        self._fallback_metrics = (
            dict(zip(self.metric_names, _build_metrics(), strict=True))
            if self.fallback_llm is not None
            else {}
        )

    @property
    def checkpoint_config(self) -> dict[str, Any]:
        """Judge policy recorded in new and automatically upgraded manifests."""

        return {
            "reasoning_mode": self.reasoning_mode or "provider_default",
            "fallback_reasoning_mode": self.fallback_reasoning_mode or None,
            "max_tokens": self.max_tokens,
            "request_timeout": self.request_timeout,
            "metric_timeout": self.timeout,
            "max_retries": self.max_retries,
        }

    def _run_ragas(
        self,
        run_df: pd.DataFrame,
        metrics: list[Metric],
        llm: BaseRagasLLM,
        usage_cb: _TokenUsageCallback,
    ):
        ragas_df = run_df.rename(columns=RUN_TO_RAGAS_COLUMNS)[list(RUN_TO_RAGAS_COLUMNS.values())]
        dataset = EvaluationDataset.from_pandas(ragas_df)
        return ragas_evaluate(
            dataset,
            metrics=metrics,
            llm=llm,
            embeddings=self.embeddings,
            callbacks=[usage_cb],
            show_progress=False,
            run_config=RunConfig(
                max_workers=self.max_concurrency,
                timeout=self.timeout,
                max_retries=self.max_retries,
            ),
        )

    def evaluate(self, run_df: pd.DataFrame) -> pd.DataFrame:
        usage_cb = _TokenUsageCallback()
        result = self._run_ragas(run_df, self._metrics, self.llm, usage_cb)
        run_id = str(result.run_id) if result.run_id is not None else None
        row_traces = _parse_row_traces(result.ragas_traces, run_id)
        _annotate_trace_attempt(
            row_traces,
            attempt="primary",
            reasoning_mode=self.effective_reasoning_mode,
        )
        for row_trace in row_traces:
            row_trace["fallback_metrics"] = []
            row_trace["fallback_failed_metrics"] = []

        scores = result.to_pandas()
        final_scores: dict[str, np.ndarray] = {}
        fallback_metric_attempts: dict[str, int] = {}

        for name, metric in zip(self.metric_names, self._metrics, strict=True):
            values = _numeric_scores(scores, metric)
            invalid_positions = np.flatnonzero(~np.isfinite(values)).tolist()

            if invalid_positions and self.fallback_llm is not None:
                fallback_metric_attempts[name] = len(invalid_positions)
                fallback_df = run_df.iloc[invalid_positions].reset_index(drop=True)
                fallback_ids = fallback_df["id"].tolist()
                logger.warning(
                    "Judge fallback: retrying metric %s for id(s) %s with reasoning_mode=%s",
                    name,
                    fallback_ids,
                    self.effective_fallback_reasoning_mode,
                )
                fallback_metric = self._fallback_metrics[name]
                fallback_result = self._run_ragas(
                    fallback_df, [fallback_metric], self.fallback_llm, usage_cb
                )
                fallback_run_id = (
                    str(fallback_result.run_id) if fallback_result.run_id is not None else None
                )
                fallback_traces = _parse_row_traces(fallback_result.ragas_traces, fallback_run_id)
                _annotate_trace_attempt(
                    fallback_traces,
                    attempt="fallback",
                    reasoning_mode=self.effective_fallback_reasoning_mode,
                )
                fallback_values = _numeric_scores(fallback_result.to_pandas(), fallback_metric)
                fallback_succeeded_ids: list[str] = []
                fallback_failed_ids: list[str] = []

                for local_position, global_position in enumerate(invalid_positions):
                    fallback_trace = fallback_traces[local_position]
                    row_trace = row_traces[global_position]
                    row_trace["fallback_metrics"].append(name)
                    row_trace["calls"].setdefault(name, []).extend(
                        fallback_trace["calls"].get(name, [])
                    )
                    fallback_value = fallback_values[local_position]
                    row_trace["scores"][name] = fallback_trace["scores"].get(name, {})
                    if np.isfinite(fallback_value):
                        values[global_position] = fallback_value
                        fallback_succeeded_ids.append(fallback_ids[local_position])
                    else:
                        row_trace["fallback_failed_metrics"].append(name)
                        fallback_failed_ids.append(fallback_ids[local_position])

                if fallback_succeeded_ids:
                    logger.warning(
                        "Judge fallback succeeded for metric %s, id(s) %s",
                        name,
                        fallback_succeeded_ids,
                    )
                if fallback_failed_ids:
                    logger.error(
                        "Judge fallback remained incomplete for metric %s, id(s) %s",
                        name,
                        fallback_failed_ids,
                    )

            final_scores[name] = values

        _log_usage(
            self.usage_log_path,
            usage_cb,
            len(run_df),
            self._model_name,
            fallback_metric_attempts=fallback_metric_attempts,
        )
        _log_judge_trace(self.trace_log_path, run_df, row_traces)

        df = run_df.copy()
        for name in self.metric_names:
            df[name] = final_scores[name]
        return df
