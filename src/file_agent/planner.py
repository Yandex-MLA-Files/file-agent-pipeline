import json
import logging

from file_agent.llm.base import LLMClient
from file_agent.llm.json_utils import extract_json_object
from file_agent.telemetry import tracer

logger = logging.getLogger(__name__)

MAX_SUBQUERIES = 4


def build_planner_prompt(question: str) -> str:
    return (
        "Разбейте сложный вопрос пользователя на самостоятельные подзапросы для "
        "поиска по документам. Каждый подзапрос должен быть отдельным, "
        "самодостаточным поисковым запросом, понятным без остального вопроса. "
        f"Не более {MAX_SUBQUERIES} подзапросов. Если вопрос уже простой и "
        "неделимый, верните его же одним подзапросом.\n\n"
        f"Вопрос:\n{question}\n\n"
        'Ответьте строго JSON без пояснений: {"subqueries": ["...", "..."]}'
    )


def plan_subqueries(question: str, llm_client: LLMClient) -> list[str]:
    with tracer.start_as_current_span("file_agent.plan_subqueries") as span:
        span.set_attribute("file_agent.question", question)

        prompt = build_planner_prompt(question)
        try:
            raw_response = llm_client.generate(prompt)
        except Exception:
            logger.warning(
                "Planner LLM call failed for question %r, falling back to the original question",
                question,
                exc_info=True,
            )
            subqueries = [question]
        else:
            subqueries = _parse_subqueries(raw_response, fallback_question=question)

        span.set_attribute("file_agent.subquery_count", len(subqueries))
        logger.info("Planned %d subquery(ies) for question %r", len(subqueries), question)
        return subqueries


def _parse_subqueries(raw_response: str, fallback_question: str) -> list[str]:
    try:
        payload = json.loads(extract_json_object(raw_response))
        subqueries = [str(item).strip() for item in payload["subqueries"] if str(item).strip()]
        if not subqueries:
            raise ValueError("empty subqueries list")
        return subqueries[:MAX_SUBQUERIES]
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        logger.warning(
            "Could not parse planner response %r, falling back to the original question",
            raw_response,
        )
        return [fallback_question]
