import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pandas as pd
import pytest
from openai import RateLimitError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.judge.llm_judge import (
    LLMJudge,
    _context_precision_from_relevance,
    _context_recall_from_statements,
    _extract_json,
    _parse_json_score,
)

# --- metric math, tested directly on known numbers, no LLM involved ---


def test_context_precision_all_relevant():
    # all 3 chunks relevant -> perfect ranking -> precision = 1.0
    assert _context_precision_from_relevance([1, 1, 1]) == pytest.approx(1.0)


def test_context_precision_relevant_chunk_first_beats_last():
    # 1 relevant chunk out of 3: ranked first scores higher than ranked last
    precision_first = _context_precision_from_relevance([1, 0, 0])
    precision_last = _context_precision_from_relevance([0, 0, 1])
    assert precision_first == pytest.approx(1.0)
    assert precision_last == pytest.approx(1 / 3)
    assert precision_first > precision_last


def test_context_precision_no_relevant_chunks_is_zero_not_error():
    assert _context_precision_from_relevance([0, 0, 0]) == 0.0


def test_context_precision_known_value():
    # chunks 1 and 3 relevant (1-indexed): precision@1=1/1, precision@3=2/3
    # context_precision = (1/1 + 2/3) / 2 relevant
    result = _context_precision_from_relevance([1, 0, 1])
    assert result == pytest.approx((1 / 1 + 2 / 3) / 2)


def test_context_recall_all_attributed():
    statements = [{"statement": "a", "attributed": 1}, {"statement": "b", "attributed": 1}]
    assert _context_recall_from_statements(statements) == 1.0


def test_context_recall_half_attributed():
    statements = [{"statement": "a", "attributed": 1}, {"statement": "b", "attributed": 0}]
    assert _context_recall_from_statements(statements) == 0.5


def test_context_recall_empty_statements_is_zero_not_error():
    assert _context_recall_from_statements([]) == 0.0


# --- judge response parsing ---


def test_parse_json_score_wrapped_in_markdown():
    text = '```json\n{"score": 0.5, "reasoning": "partial"}\n```'
    assert _parse_json_score(text)["score"] == 0.5


def test_parse_json_score_out_of_range_raises():
    with pytest.raises(ValueError, match="0..1"):
        _parse_json_score('{"score": 1.5, "reasoning": "..."}')


def test_extract_json_skips_stray_brace_in_reasoning_trace():
    # a reasoning model (e.g. DeepSeek-R1) prepending chain-of-thought that
    # itself mentions a brace before the real JSON — must not be swallowed
    # into one invalid blob by a naive "first { to last }" match.
    text = (
        "Let me think about the expected format, something like {a bare "
        'word, not json}. Given that, my answer is: {"score": 0.7, '
        '"reasoning": "looks faithful"}'
    )
    assert _extract_json(text) == {"score": 0.7, "reasoning": "looks faithful"}


def test_extract_json_handles_nested_structure():
    # context_precision / context_recall responses nest arrays of objects
    # inside the outer object — brace-balance counting must handle that.
    text = (
        '{"statements": [{"statement": "a", "attributed": 1}, {"statement": "b", "attributed": 0}]}'
    )
    result = _extract_json(text)
    assert len(result["statements"]) == 2
    assert result["statements"][0]["attributed"] == 1


def test_extract_json_no_json_raises():
    with pytest.raises(ValueError, match="no JSON found"):
        _extract_json("just plain text, no JSON anywhere")


# --- full evaluate() via a fake client, no network ---


def _fake_response(text: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class FakeClient:
    """Stands in for openai.OpenAI(...).chat.completions.create(...), no network.

    canned_responses are returned in order, one per call, so faithfulness/
    correctness/etc. can be told apart in a test.
    """

    def __init__(self, canned_responses: list[str]):
        self._responses = iter(canned_responses)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.calls = []

    def _create(self, model, messages, temperature):
        self.calls.append({"model": model})
        return _fake_response(next(self._responses))


def test_evaluate_all_five_metrics_one_row():
    run_df = pd.DataFrame(
        [
            {
                "id": "ex_001",
                "question": "question?",
                "answer": "answer",
                "contexts": ["chunk 1", "chunk 2"],
                "ground_truth": "reference",
            }
        ]
    )
    # call order in evaluate(): faithfulness, correctness, relevancy,
    # context_precision, context_recall
    fake_client = FakeClient(
        [
            '{"score": 0.9, "reasoning": "faithful"}',
            '{"score": 0.7, "reasoning": "close"}',
            '{"score": 0.8, "reasoning": "on topic"}',
            '{"relevance": [1, 0]}',
            '{"statements": [{"statement": "x", "attributed": 1}]}',
        ]
    )
    judge = LLMJudge(model="test-model", client=fake_client)
    scored = judge.evaluate(run_df)

    assert len(fake_client.calls) == 5
    assert scored.loc[0, "faithfulness"] == 0.9
    assert scored.loc[0, "answer_correctness"] == 0.7
    assert scored.loc[0, "answer_relevancy"] == 0.8
    assert scored.loc[0, "context_precision"] == pytest.approx(1.0)  # [1,0] -> relevant chunk first
    assert scored.loc[0, "context_recall"] == 1.0


def test_context_precision_length_mismatch_raises():
    run_df = pd.DataFrame(
        [
            {
                "id": "ex_001",
                "question": "q",
                "answer": "a",
                "contexts": ["chunk 1", "chunk 2"],
                "ground_truth": "gt",
            }
        ]
    )
    # judge returned relevance for only 1 chunk instead of 2 -> should raise clearly
    fake_client = FakeClient(
        [
            '{"score": 0.9, "reasoning": "x"}',
            '{"score": 0.9, "reasoning": "x"}',
            '{"score": 0.9, "reasoning": "x"}',
            '{"relevance": [1]}',
        ]
    )
    judge = LLMJudge(model="test-model", client=fake_client)
    with pytest.raises(ValueError, match="2 chunks"):
        judge.evaluate(run_df)


# --- retry-on-rate-limit behavior ---


def _rate_limit_error(retry_after: str | None = None) -> RateLimitError:
    headers = {"retry-after": retry_after} if retry_after else {}
    response = httpx.Response(
        status_code=429,
        headers=headers,
        request=httpx.Request("POST", "https://example.com"),
    )
    return RateLimitError("rate limited", response=response, body=None)


class FlakyClient:
    """Raises RateLimitError a fixed number of times, then returns canned responses."""

    def __init__(
        self, n_failures: int, canned_responses: list[str], retry_after: str | None = None
    ):
        self._n_failures = n_failures
        self._responses = iter(canned_responses)
        self._retry_after = retry_after
        self.call_count = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model, messages, temperature):
        self.call_count += 1
        if self.call_count <= self._n_failures:
            raise _rate_limit_error(self._retry_after)
        return _fake_response(next(self._responses))


def test_call_retries_on_rate_limit_then_succeeds(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr("eval.judge.llm_judge.time.sleep", lambda s: sleep_calls.append(s))

    client = FlakyClient(n_failures=2, canned_responses=['{"score": 0.5, "reasoning": "ok"}'])
    judge = LLMJudge(model="test-model", client=client)

    result = judge._call("some prompt")

    assert result == '{"score": 0.5, "reasoning": "ok"}'
    assert client.call_count == 3  # 2 failures + 1 success
    assert len(sleep_calls) == 2  # slept before each retry, not after success


def test_call_honors_retry_after_header(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr("eval.judge.llm_judge.time.sleep", lambda s: sleep_calls.append(s))

    client = FlakyClient(
        n_failures=1, canned_responses=['{"score": 1.0, "reasoning": "ok"}'], retry_after="7"
    )
    judge = LLMJudge(model="test-model", client=client)

    judge._call("some prompt")

    assert sleep_calls == [7.0]


def test_call_treats_zero_retry_after_as_not_provided(monkeypatch):
    # a server saying "retry in 0s" for a limit that hasn't reset yet just
    # burns through all attempts instantly if taken literally — must fall
    # back to exponential backoff instead of a zero-second sleep.
    sleep_calls = []
    monkeypatch.setattr("eval.judge.llm_judge.time.sleep", lambda s: sleep_calls.append(s))

    client = FlakyClient(
        n_failures=1, canned_responses=['{"score": 1.0, "reasoning": "ok"}'], retry_after="0"
    )
    judge = LLMJudge(model="test-model", client=client, retry_base_delay=2.0)

    judge._call("some prompt")

    assert sleep_calls == [2.0]  # exponential fallback, attempt 0 -> 2.0 * 2**0
    assert sleep_calls[0] > 0


def test_call_raises_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr("eval.judge.llm_judge.time.sleep", lambda s: None)

    client = FlakyClient(n_failures=10, canned_responses=[])  # never succeeds
    judge = LLMJudge(model="test-model", client=client, max_retries=2)

    with pytest.raises(RateLimitError):
        judge._call("some prompt")

    assert client.call_count == 3  # initial attempt + 2 retries


def test_call_fails_fast_when_retry_after_exceeds_max_delay(monkeypatch):
    # a 23-hour Retry-After (anti-abuse block, not an ordinary rate limit)
    # must not be slept through silently — fail immediately instead.
    sleep_calls = []
    monkeypatch.setattr("eval.judge.llm_judge.time.sleep", lambda s: sleep_calls.append(s))

    client = FlakyClient(
        n_failures=1, canned_responses=['{"score": 1.0, "reasoning": "ok"}'], retry_after="85290"
    )
    judge = LLMJudge(model="test-model", client=client, max_retry_delay=60.0)

    with pytest.raises(RuntimeError, match="max_retry_delay"):
        judge._call("some prompt")

    assert sleep_calls == []  # never slept — failed before sleeping
    assert client.call_count == 1  # never even got to retry


def test_call_retries_on_connection_error(monkeypatch):
    from openai import APIConnectionError

    sleep_calls = []
    monkeypatch.setattr("eval.judge.llm_judge.time.sleep", lambda s: sleep_calls.append(s))

    class FlakyConnectionClient:
        def __init__(self):
            self.call_count = 0
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, model, messages, temperature):
            self.call_count += 1
            if self.call_count == 1:
                # APIConnectionError has no HTTP response at all (a dropped
                # connection, not a status code) — this is the case our
                # retry-delay fallback (no `.response`) needs to handle.
                raise APIConnectionError(request=httpx.Request("POST", "https://example.com"))
            return _fake_response('{"score": 0.6, "reasoning": "ok"}')

    client = FlakyConnectionClient()
    judge = LLMJudge(model="test-model", client=client, retry_base_delay=1.0)

    result = judge._call("some prompt")

    assert result == '{"score": 0.6, "reasoning": "ok"}'
    assert client.call_count == 2
    assert sleep_calls == [
        1.0
    ]  # no Retry-After available -> exponential fallback, attempt 0 -> 1.0*2**0
