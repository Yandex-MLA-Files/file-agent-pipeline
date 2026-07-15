"""LLM-as-judge over any OpenAI-compatible API.

Both open-weight model hosts (Together/Fireworks/Groq/OpenRouter/a local
vLLM or SGLang server) and an internal corporate endpoint almost certainly
speak the same protocol: the OpenAI chat completions API. Nothing here is
tied to a specific provider, only to 3 environment variables:

    JUDGE_BASE_URL   API URL (an open model host today; an internal
                      endpoint later, just change the value)
    JUDGE_API_KEY    key for whichever endpoint is configured above
    JUDGE_MODEL      model name served at that endpoint

Switching providers is a matter of changing these three variables. This
file does not change at all.

Two kinds of metrics:
  - Direct scoring (faithfulness, answer_correctness, answer_relevancy) —
    one model call, it returns a 0..1 score directly.
  - Classify + compute in Python (context_precision, context_recall) — the
    model classifies chunks/statements individually, the metric itself is
    computed here, not by the model. This is more reliable: models are bad
    at computing aggregates but good at classifying short, isolated items.

NOTE: answer_relevancy here is a simplified LLM-graded version (direct
scoring of "does the answer address the question"), not the original RAGAS
algorithm (which generates questions from the answer and compares them to
the original question via embeddings). Numbers from this implementation
are not directly comparable to the `ragas` library.

NOTE: in manual testing against a real model (DeepSeek-V3), an earlier
version of RELEVANCY_PROMPT without the explicit "ignore factual
correctness" instruction caused the judge to conflate answer_relevancy with
answer_correctness — a confidently wrong-but-on-topic answer scored 0 on
both, when it should score high on relevancy and low on correctness (two
different things). The prompt below now explicitly tells the model to
judge relevancy independent of factual correctness. Re-check this if you
swap in a different judge model — different models vary in how well they
keep the two concepts separate.

NOTE: the prompt text below is in Russian, matching the language of the
dataset being judged (question/answer/context/ground_truth are Russian).
This is deliberate, not an oversight: prompt content is data sent to an
API, not documentation, so the "code in English" convention doesn't apply
to it. Matching the prompt's language to the content's language removes a
class of risk (weaker instruction-following on smaller open-weight models
when instruction and content language differ) at no cost. If the judge
model or the dataset's language changes, update these constants — nothing
else in this file assumes a particular language.
"""

from __future__ import annotations

import json
import os
import time

import pandas as pd

from eval.judge.base import Judge

FAITHFULNESS_PROMPT = """Ты — строгий эксперт-проверяющий. Оцени, подтверждается \
ли ответ приведённым контекстом (не выдумал ли ответ факты, которых нет в контексте).

Контекст:
{context}

Ответ:
{answer}

Верни ТОЛЬКО JSON без пояснений вокруг: {{"score": <число от 0 до 1>, \
"reasoning": "<короткое обоснование>"}}
где 1 — всё в ответе подтверждается контекстом, 0 — ответ полностью выдуман."""

CORRECTNESS_PROMPT = """Ты — строгий эксперт-проверяющий. Сравни ответ модели с \
эталонным ответом и оцени, насколько они совпадают по смыслу и фактам.

Эталонный ответ:
{ground_truth}

Ответ модели:
{answer}

Верни ТОЛЬКО JSON без пояснений вокруг: {{"score": <число от 0 до 1>, \
"reasoning": "<короткое обоснование>"}}
где 1 — ответы совпадают по сути, 0 — полностью расходятся."""

RELEVANCY_PROMPT = """Ты — строгий эксперт-проверяющий. Оцени, отвечает ли ответ \
по существу на заданный вопрос (не уходит ли в сторону, не является ли \
уклончивым или слишком общим там, где можно было ответить конкретно).

Важно: оценивай ТОЛЬКО то, обращается ли ответ к сути вопроса — прямой ли \
это, конкретный ответ на заданный вопрос. Фактическая правильность ответа \
здесь не оценивается (для этого есть отдельная метрика) — даже если ответ \
неверен по фактам, но прямо и конкретно отвечает на то, что спрашивалось, \
это высокая релевантность.

Вопрос:
{question}

Ответ:
{answer}

Верни ТОЛЬКО JSON без пояснений вокруг: {{"score": <число от 0 до 1>, \
"reasoning": "<короткое обоснование>"}}
где 1 — ответ прямо и конкретно обращается к сути вопроса (независимо от \
фактической правильности), 0 — ответ уклончив, расплывчат или не по теме \
вопроса."""

CONTEXT_PRECISION_PROMPT = """Ты — строгий эксперт-проверяющий. Ниже пронумерованные \
чанки контекста в том порядке, в котором их вернул retriever (чанк 1 — самый \
первый по рангу). Для каждого чанка определи, помогает ли он ответить на \
вопрос — сверяйся с эталонным ответом, а не с общими знаниями.

Вопрос:
{question}

Эталонный ответ:
{ground_truth}

Чанки:
{numbered_context}

Верни ТОЛЬКО JSON без пояснений вокруг: {{"relevance": [<0 или 1 для чанка 1>, \
<для чанка 2>, ...]}}
Длина массива должна точно совпадать с числом чанков ({n_chunks})."""

CONTEXT_RECALL_PROMPT = """Ты — строгий эксперт-проверяющий. Разбей эталонный \
ответ на отдельные самостоятельные утверждения (claims). Для каждого утверждения \
определи, подтверждается ли оно хотя бы одним из приведённых чанков контекста.

Контекст:
{context}

Эталонный ответ:
{ground_truth}

Верни ТОЛЬКО JSON без пояснений вокруг:
{{"statements": [{{"statement": "<текст утверждения>", "attributed": <0 или 1>}}, ...]}}"""


def _extract_json(text: str) -> dict:
    """Extract the first valid, balanced JSON object from a model response.

    Reasoning models (e.g. DeepSeek-R1) prepend a chain-of-thought trace
    before the actual answer, and that trace often contains stray '{'/'}'
    characters of its own (e.g. discussing JSON format, code, or math) — a
    naive "first {  to last }" regex swallows the whole trace and produces
    invalid JSON ("Extra data" errors). This scans every '{' in the text,
    extracts the brace-balanced span starting there (tracking string
    literals so braces inside quoted text don't throw off the count), and
    returns the first span that actually parses as JSON — skipping spans
    that don't, which is what happens when a '{' isn't really the start of
    the intended JSON object.
    """
    for start, char in enumerate(text):
        if char != "{":
            continue
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            c = text[i]
            if in_string:
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                elif c == '"':
                    in_string = False
                continue
            if c == '"':
                in_string = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break  # this '{' wasn't real JSON — try the next one
    raise ValueError(f"no JSON found in judge response: {text!r}")


def _parse_json_score(text: str) -> dict:
    data = _extract_json(text)
    score = float(data["score"])
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"score out of range 0..1: {score}")
    return data


def _context_precision_from_relevance(relevance: list[int]) -> float:
    """Average precision: relevant chunks ranked earlier count for more.

    context_precision = sum(Precision@k * v_k for k in 1..K) / (number of
    relevant chunks), where v_k = 1 if the chunk at rank k is relevant.
    If there are no relevant chunks at all, returns 0 (not 1, not a
    division by zero).
    """
    total_relevant = sum(relevance)
    if total_relevant == 0:
        return 0.0
    hits = 0
    weighted_sum = 0.0
    for k, v in enumerate(relevance, start=1):
        if v:
            hits += 1
            weighted_sum += hits / k
    return weighted_sum / total_relevant


def _context_recall_from_statements(statements: list[dict]) -> float:
    """Fraction of reference-answer claims supported by the context."""
    if not statements:
        return 0.0
    attributed = sum(int(s["attributed"]) for s in statements)
    return attributed / len(statements)


class LLMJudge(Judge):
    """Real LLM-as-judge over an OpenAI-compatible API.

    `client` can be passed explicitly (for tests — a fake client with no
    network calls); by default it's built from environment variables.
    """

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
        client=None,
        max_retries: int = 5,
        retry_base_delay: float = 2.0,
        max_retry_delay: float = 60.0,
    ):
        self.model = model or os.environ["JUDGE_MODEL"]
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.max_retry_delay = max_retry_delay
        if client is not None:
            self.client = client
        else:
            from openai import OpenAI

            self.client = OpenAI(
                base_url=os.environ.get("JUDGE_BASE_URL"),
                api_key=os.environ.get("JUDGE_API_KEY", "not-needed"),
            )

    def evaluate(self, run_df: pd.DataFrame) -> pd.DataFrame:
        df = run_df.copy()
        n = len(df)

        faithfulness, correctness, relevancy, precision, recall = [], [], [], [], []
        for i, r in enumerate(df.itertuples(), start=1):
            print(f"[{i}/{n}] scoring {r.id}...")
            faithfulness.append(self._score_faithfulness(r.answer, r.contexts))
            correctness.append(self._score_correctness(r.answer, r.ground_truth))
            relevancy.append(self._score_relevancy(r.question, r.answer))
            precision.append(self._score_context_precision(r.question, r.ground_truth, r.contexts))
            recall.append(self._score_context_recall(r.ground_truth, r.contexts))

        df["faithfulness"] = faithfulness
        df["answer_correctness"] = correctness
        df["answer_relevancy"] = relevancy
        df["context_precision"] = precision
        df["context_recall"] = recall
        return df

    def _score_faithfulness(self, answer: str, contexts: list[str]) -> float:
        prompt = FAITHFULNESS_PROMPT.format(context="\n".join(contexts), answer=answer)
        return _parse_json_score(self._call(prompt))["score"]

    def _score_correctness(self, answer: str, ground_truth: str) -> float:
        prompt = CORRECTNESS_PROMPT.format(ground_truth=ground_truth, answer=answer)
        return _parse_json_score(self._call(prompt))["score"]

    def _score_relevancy(self, question: str, answer: str) -> float:
        prompt = RELEVANCY_PROMPT.format(question=question, answer=answer)
        return _parse_json_score(self._call(prompt))["score"]

    def _score_context_precision(
        self, question: str, ground_truth: str, contexts: list[str]
    ) -> float:
        if not contexts:
            return 0.0
        numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(contexts, start=1))
        prompt = CONTEXT_PRECISION_PROMPT.format(
            question=question,
            ground_truth=ground_truth,
            numbered_context=numbered,
            n_chunks=len(contexts),
        )
        data = _extract_json(self._call(prompt))
        relevance = [int(x) for x in data["relevance"]]
        if len(relevance) != len(contexts):
            raise ValueError(
                f"judge returned {len(relevance)} relevance scores for {len(contexts)} chunks"
            )
        return _context_precision_from_relevance(relevance)

    def _score_context_recall(self, ground_truth: str, contexts: list[str]) -> float:
        prompt = CONTEXT_RECALL_PROMPT.format(
            context="\n".join(contexts), ground_truth=ground_truth
        )
        data = _extract_json(self._call(prompt))
        return _context_recall_from_statements(data["statements"])

    def _call(self, prompt: str) -> str:
        from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

        # Retried: rate limits, transient connection drops, and 5xx from the
        # provider. NOT retried: 4xx like bad request/auth errors — those
        # need a human to fix the config, not a delay.
        retryable = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                return response.choices[0].message.content
            except retryable as e:
                last_error = e
                if attempt == self.max_retries:
                    raise
                delay = self._retry_delay_seconds(e, attempt)
                if delay > self.max_retry_delay:
                    # The server (or our own backoff math) wants a wait
                    # longer than we're willing to block for — this usually
                    # means an anti-abuse block (hours, not seconds), not an
                    # ordinary rate limit. Sleeping through that silently
                    # would hang the process for an absurd amount of time;
                    # fail fast instead so a human can decide what to do
                    # (switch provider, wait it out deliberately, etc.).
                    raise RuntimeError(
                        f"{type(e).__name__} asked for a {delay:.0f}s wait, "
                        f"which is over max_retry_delay ({self.max_retry_delay:.0f}s). "
                        f"Not retrying automatically — this usually means an "
                        f"anti-abuse block rather than an ordinary rate limit. "
                        f"Original error: {e}"
                    ) from e
                print(
                    f"  {type(e).__name__}, retry {attempt + 1}/{self.max_retries} in {delay:.0f}s"
                )
                time.sleep(delay)
        raise last_error  # pragma: no cover - unreachable, loop always returns or raises

    def _retry_delay_seconds(self, error: Exception, attempt: int) -> float:
        """Prefer the server's Retry-After hint; fall back to exponential backoff.

        A Retry-After of 0 (or negative) is treated as "not provided" rather
        than "retry immediately" — a rate limit that hasn't reset yet won't
        reset just because the server said 0, and retrying instantly just
        burns through all attempts in a fraction of a second without ever
        giving the limit time to clear.
        """
        response = getattr(error, "response", None)
        if response is not None:
            header = response.headers.get("retry-after")
            if header is not None:
                try:
                    retry_after = float(header)
                    if retry_after > 0:
                        return retry_after
                except ValueError:
                    pass
        return self.retry_base_delay * (2**attempt)
