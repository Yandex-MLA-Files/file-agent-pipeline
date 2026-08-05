import json
import logging
from enum import StrEnum

from file_agent.llm.base import LLMClient
from file_agent.llm.json_utils import extract_json_object
from file_agent.telemetry import tracer

logger = logging.getLogger(__name__)


class QueryType(StrEnum):
    SIMPLE = "simple"
    COMPLEX = "complex"
    TOOL = "tool"


def build_router_prompt(question: str) -> str:
    return (
        "Классифицируйте вопрос пользователя ровно в одну категорию:\n"
        '- "simple" — короткий фактический вопрос, на который достаточно найти '
        "один релевантный фрагмент текста;\n"
        '- "complex" — вопрос из нескольких частей или требующий сопоставления '
        "информации из разных мест документа(ов);\n"
        '- "tool" — вопрос про табличные, числовые данные (суммы, подсчёты, '
        "агрегации) или визуальные элементы (диаграммы, изображения).\n\n"
        f"Вопрос:\n{question}\n\n"
        'Ответьте строго JSON без пояснений: {"query_type": "simple" | "complex" | "tool"}'
    )


def classify_query(question: str, llm_client: LLMClient) -> QueryType:
    with tracer.start_as_current_span("file_agent.classify_query") as span:
        span.set_attribute("file_agent.question", question)

        prompt = build_router_prompt(question)

        try:
            raw_response = llm_client.generate(prompt)
        except Exception:
            logger.warning(
                "Router LLM call failed for question %r, defaulting to %s",
                question,
                QueryType.SIMPLE.value,
                exc_info=True,
            )
            query_type = QueryType.SIMPLE
        else:
            query_type = _parse_query_type(raw_response)

        span.set_attribute("file_agent.query_type", query_type.value)
        logger.info("Classified question %r as %s", question, query_type.value)
        return query_type


def _parse_query_type(raw_response: str) -> QueryType:
    try:
        payload = json.loads(extract_json_object(raw_response))
        return QueryType(str(payload["query_type"]).strip().lower())
    except (json.JSONDecodeError, KeyError, ValueError):
        logger.warning(
            "Could not parse router response %r, defaulting to %s",
            raw_response,
            QueryType.SIMPLE.value,
        )
        return QueryType.SIMPLE
