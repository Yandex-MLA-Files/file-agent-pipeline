from __future__ import annotations

import json
import os
import uuid
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings as LangchainHuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas import RunConfig
from ragas import evaluate as ragas_evaluate
from ragas.callbacks import ChainRun
from ragas.cost import BaseCallbackHandler, LLMResult
from ragas.dataset_schema import EvaluationDataset
from ragas.embeddings import BaseRagasEmbeddings, LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.llms.base import BaseRagasLLM
from ragas.metrics import (
    FactualCorrectness,
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
DEFAULT_MAX_RETRIES = 15
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

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
    answer_correctness: FactualCorrectness,
    answer_relevancy: ResponseRelevancy,
    context_precision: LLMContextPrecisionWithReference,
    context_recall: LLMContextRecall,
) -> None:

    faithfulness.statement_generator_prompt.instruction += _LANGUAGE_MATCH_INSTRUCTION
    faithfulness.nli_statements_prompt.instruction += _LANGUAGE_MATCH_INSTRUCTION
    answer_correctness.claim_decomposition_prompt.instruction += _LANGUAGE_MATCH_INSTRUCTION
    answer_correctness.nli_prompt.instruction += _LANGUAGE_MATCH_INSTRUCTION
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
    path: str | Path, usage_cb: _TokenUsageCallback, n_rows: int, fallback_model: str = ""
) -> dict[str, Any]:
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
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def _json_default_trace(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return str(obj)


def _per_run_trace_path(base_path: str | Path, now: datetime) -> Path:

    base = Path(base_path)
    stamp = now.strftime("%Y%m%dT%H%M%S%f") + "Z"
    return base.with_name(f"{base.stem}_{stamp}{base.suffix}")


_CALL_DID_NOT_COMPLETE = (
    "<no output recorded - this call likely raised (e.g. exhausted its JSON-repair "
    "retries, see PydanticPrompt.generate_multiple's retries_left=3) rather than "
    "returning {} for real. ragas.callbacks.RagasTracer only records outputs via "
    "on_chain_end and has no on_chain_error hook, so no exception detail survives "
    "into this trace - check the row's NaN metric score(s) for which one failed.>"
)


def _completed_output(outputs: dict[str, Any]) -> Any:
    # outputs defaults to {} on the ChainRun itself (ragas.callbacks.ChainRun) and
    # is only ever overwritten by on_chain_end - a chain that raised leaves it at
    # that same default, indistinguishable by shape alone from a call that
    # legitimately returned {}. Surfacing that ambiguity explicitly beats letting a
    # real failure quietly look like a valid empty response in the log.
    if not outputs:
        return _CALL_DID_NOT_COMPLETE
    return outputs.get("output", {})


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
            scores[metric_trace.name] = _completed_output(metric_trace.outputs)
            metric_calls = []
            for prompt_uuid in metric_trace.children:
                prompt_trace = ragas_traces[prompt_uuid]
                output = _completed_output(prompt_trace.outputs)
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
) -> Path:

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
            }
            f.write(json.dumps(entry, ensure_ascii=False, default=_json_default_trace) + "\n")
    return run_path


def _build_default_llm(model: str) -> BaseRagasLLM:
    chat = ChatOpenAI(
        base_url=os.environ.get("JUDGE_BASE_URL"),
        api_key=os.environ.get("JUDGE_API_KEY", "not-needed"),
        model=model,
        temperature=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return LangchainLLMWrapper(chat)


def _build_default_embeddings(model: str) -> BaseRagasEmbeddings:
    embeddings = LangchainHuggingFaceEmbeddings(model_name=model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return LangchainEmbeddingsWrapper(embeddings)


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
        embeddings: BaseRagasEmbeddings | None = None,
        usage_log_path: str | Path | None = None,
        trace_log_path: str | Path | None = None,
    ):
        model_name = model or os.environ.get("JUDGE_MODEL", "")
        self.llm = llm or _build_default_llm(model_name or os.environ["JUDGE_MODEL"])
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
        faithfulness = Faithfulness(name="faithfulness")
        answer_correctness = FactualCorrectness(name="answer_correctness", mode="recall")

        answer_relevancy = ResponseRelevancy(name="answer_relevancy", strictness=1)
        context_precision = LLMContextPrecisionWithReference(name="context_precision")
        context_recall = LLMContextRecall(name="context_recall")
        _match_response_language_to_input(
            faithfulness, answer_correctness, answer_relevancy, context_precision, context_recall
        )
        self._metrics = [
            faithfulness,
            answer_correctness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]
        # Populated by evaluate(): cost/token totals and the trace file path
        # for the *last* call, so callers (e.g. run_eval.py) can forward them
        # to MLflow without re-deriving anything evaluate() already computed.
        self.last_usage: dict[str, Any] | None = None
        self.last_trace_path: Path | None = None

    def evaluate(self, run_df: pd.DataFrame) -> pd.DataFrame:
        ragas_df = run_df.rename(columns=RUN_TO_RAGAS_COLUMNS)[list(RUN_TO_RAGAS_COLUMNS.values())]
        dataset = EvaluationDataset.from_pandas(ragas_df)

        usage_cb = _TokenUsageCallback()

        result = ragas_evaluate(
            dataset,
            metrics=self._metrics,
            llm=self.llm,
            embeddings=self.embeddings,
            callbacks=[usage_cb],
            show_progress=False,
            run_config=RunConfig(
                max_workers=self.max_concurrency,
                timeout=self.timeout,
                max_retries=self.max_retries,
            ),
        )
        self.last_usage = _log_usage(self.usage_log_path, usage_cb, len(run_df), self._model_name)
        run_id = str(result.run_id) if result.run_id is not None else None
        row_traces = _parse_row_traces(result.ragas_traces, run_id)
        self.last_trace_path = _log_judge_trace(self.trace_log_path, run_df, row_traces)

        scores = result.to_pandas()
        df = run_df.copy()
        for name, metric in zip(self.metric_names, self._metrics, strict=True):
            df[name] = scores[_result_column(metric)].values
        return df
