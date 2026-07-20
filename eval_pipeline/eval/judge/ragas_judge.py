from __future__ import annotations

import json
import os
import warnings
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

import pandas as pd
from langchain_openai import ChatOpenAI
from ragas import evaluate as ragas_evaluate
from ragas.cost import BaseCallbackHandler, LLMResult
from ragas.dataset_schema import EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.llms.base import BaseRagasLLM
from ragas.metrics import (
    AspectCritic,
    FactualCorrectness,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)
from ragas.metrics.base import Metric, ModeMetric

from eval.judge.base import Judge

DEFAULT_PRICE_PER_1K_INPUT_TOKENS = 0.3
DEFAULT_PRICE_PER_1K_OUTPUT_TOKENS = 0.5
DEFAULT_PRICE_PER_1K_CACHED_TOKENS = 0.075

DEFAULT_USAGE_LOG_PATH = "usage_log.jsonl"

RELEVANCY_DEFINITION = (
    "Отвечает ли ответ прямо и конкретно на суть заданного вопроса, "
    "независимо от того, является ли он фактически верным? Ответ, который "
    "прямо и конкретно отвечает по существу, оценивается высоко (1), даже "
    "если он фактически неверен; уклончивый, расплывчатый или не по теме "
    "ответ оценивается низко (0), даже если он фактически верен."
)

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


def _localize_prompts_to_russian(
    faithfulness: Faithfulness,
    answer_correctness: FactualCorrectness,
    answer_relevancy: AspectCritic,
    context_precision: LLMContextPrecisionWithReference,
    context_recall: LLMContextRecall,
) -> None:

    # --- faithfulness: statement generation + NLI verification ---
    gen_in, gen_out = faithfulness.statement_generator_prompt.examples[0]
    faithfulness.statement_generator_prompt.instruction = (
        "Дан вопрос и ответ. Проанализируй сложность каждого предложения в "
        "ответе. Разбей каждое предложение на одно или несколько полностью "
        "самостоятельных утверждений. Не используй местоимения ни в одном "
        "утверждении. Выведи результат в формате JSON."
    )
    faithfulness.statement_generator_prompt.examples = [
        (
            gen_in.model_copy(
                update={
                    "question": "Кем был Альберт Эйнштейн и чем он прежде всего известен?",
                    "answer": (
                        "Он был физиком-теоретиком, родившимся в Германии, широко "
                        "признанным одним из величайших и наиболее влиятельных физиков "
                        "всех времён. Он прежде всего известен разработкой теории "
                        "относительности, а также внёс важный вклад в развитие "
                        "квантовой механики."
                    ),
                }
            ),
            gen_out.model_copy(
                update={
                    "statements": [
                        "Альберт Эйнштейн был физиком-теоретиком, родившимся в Германии.",
                        "Альберт Эйнштейн признан одним из величайших и наиболее "
                        "влиятельных физиков всех времён.",
                        "Альберт Эйнштейн прежде всего известен разработкой теории "
                        "относительности.",
                        "Альберт Эйнштейн также внёс важный вклад в развитие квантовой механики.",
                    ]
                }
            ),
        )
    ]

    nli_instruction = (
        "Твоя задача — оценить достоверность набора утверждений на основе "
        "данного контекста. Для каждого утверждения верни вердикт 1, если "
        "утверждение можно напрямую вывести из контекста, и 0, если нельзя."
    )
    nli_in_1, nli_out_1 = faithfulness.nli_statements_prompt.examples[0]
    nli_in_2, nli_out_2 = faithfulness.nli_statements_prompt.examples[1]
    nli_examples = [
        (
            nli_in_1.model_copy(
                update={
                    "context": (
                        "Джон — студент университета XYZ. Он учится по специальности "
                        "«Информатика». В этом семестре он записан на несколько курсов, "
                        "включая «Структуры данных», «Алгоритмы» и «Управление базами "
                        "данных». Джон — прилежный студент и уделяет много времени учёбе "
                        "и выполнению заданий. Он часто задерживается допоздна в "
                        "библиотеке, работая над своими проектами."
                    ),
                    "statements": [
                        "Джон специализируется на биологии.",
                        "Джон проходит курс по искусственному интеллекту.",
                        "Джон — прилежный студент.",
                        "У Джона есть подработка.",
                    ],
                }
            ),
            nli_out_1.model_copy(
                update={
                    "statements": [
                        nli_out_1.statements[0].model_copy(
                            update={
                                "statement": "Джон специализируется на биологии.",
                                "reason": (
                                    "Специальность Джона прямо указана как «Информатика». "
                                    "Нет информации, указывающей на то, что он "
                                    "специализируется на биологии."
                                ),
                                "verdict": 0,
                            }
                        ),
                        nli_out_1.statements[1].model_copy(
                            update={
                                "statement": "Джон проходит курс по искусственному интеллекту.",
                                "reason": (
                                    "В контексте перечислены курсы, на которые записан Джон "
                                    "в этом семестре, и искусственный интеллект среди них не "
                                    "упоминается. Поэтому нельзя сделать вывод, что он изучает ИИ."
                                ),
                                "verdict": 0,
                            }
                        ),
                        nli_out_1.statements[2].model_copy(
                            update={
                                "statement": "Джон — прилежный студент.",
                                "reason": (
                                    "В контексте сказано, что он уделяет много времени учёбе "
                                    "и выполнению заданий. Также упоминается, что он часто "
                                    "задерживается допоздна в библиотеке, работая над "
                                    "проектами, что подразумевает прилежность."
                                ),
                                "verdict": 1,
                            }
                        ),
                        nli_out_1.statements[3].model_copy(
                            update={
                                "statement": "У Джона есть подработка.",
                                "reason": (
                                    "В контексте нет никакой информации о том, что у Джона "
                                    "есть подработка."
                                ),
                                "verdict": 0,
                            }
                        ),
                    ]
                }
            ),
        ),
        (
            nli_in_2.model_copy(
                update={
                    "context": (
                        "Фотосинтез — процесс, используемый растениями, водорослями и "
                        "некоторыми бактериями для преобразования энергии света в "
                        "химическую энергию."
                    ),
                    "statements": ["Альберт Эйнштейн был гением."],
                }
            ),
            nli_out_2.model_copy(
                update={
                    "statements": [
                        nli_out_2.statements[0].model_copy(
                            update={
                                "statement": "Альберт Эйнштейн был гением.",
                                "reason": "Контекст и утверждение никак не связаны между собой.",
                                "verdict": 0,
                            }
                        ),
                    ]
                }
            ),
        ),
    ]
    faithfulness.nli_statements_prompt.instruction = nli_instruction
    faithfulness.nli_statements_prompt.examples = nli_examples

    # answer_correctness reuses the same NLI prompt shape for its own
    # nli_prompt -- same translated instruction/examples apply.
    answer_correctness.nli_prompt.instruction = nli_instruction
    answer_correctness.nli_prompt.examples = nli_examples

    # --- answer_correctness: claim decomposition ---
    cd_in_1, cd_out_1 = answer_correctness.claim_decomposition_prompt.examples[0]
    cd_in_2, cd_out_2 = answer_correctness.claim_decomposition_prompt.examples[1]
    answer_correctness.claim_decomposition_prompt.instruction = (
        "Разбей и раздели каждое из входных предложений на одно или несколько "
        "самостоятельных утверждений. Каждое утверждение должно быть отдельным "
        "тезисом, который можно проверить независимо.\n"
        "Соблюдай уровень атомарности и охвата, как показано в примерах."
    )
    answer_correctness.claim_decomposition_prompt.examples = [
        (
            cd_in_1.model_copy(
                update={
                    "response": (
                        "Чарльз Бэббидж был французским математиком, философом и "
                        "критиком в области кулинарии."
                    )
                }
            ),
            cd_out_1.model_copy(update={"claims": ["Чарльз Бэббидж был математиком и философом."]}),
        ),
        (
            cd_in_2.model_copy(
                update={
                    "response": (
                        "Альберт Эйнштейн был немецким физиком-теоретиком. Он "
                        "разработал теорию относительности, а также внёс вклад в "
                        "развитие квантовой механики."
                    )
                }
            ),
            cd_out_2.model_copy(
                update={
                    "claims": [
                        "Альберт Эйнштейн был немецким физиком.",
                        "Альберт Эйнштейн разработал теорию относительности и внёс "
                        "вклад в квантовую механику.",
                    ]
                }
            ),
        ),
    ]

    answer_relevancy.single_turn_prompt.instruction = (
        "Оцени входные данные на основе указанного критерия. Используй "
        "только «Да» (1) и «Нет» (0) как вердикт.\n"
        f"Критерий: {answer_relevancy.definition}"
    )

    # --- context_precision ---
    qac_1, ver_1 = context_precision.context_precision_prompt.examples[0]
    qac_2, ver_2 = context_precision.context_precision_prompt.examples[1]
    qac_3, ver_3 = context_precision.context_precision_prompt.examples[2]
    context_precision.context_precision_prompt.instruction = (
        "Дан вопрос, ответ и контекст. Проверь, был ли контекст полезен для "
        "получения данного ответа. Верни вердикт «1», если полезен, и «0», "
        "если нет, в формате JSON."
    )
    context_precision.context_precision_prompt.examples = [
        (
            qac_1.model_copy(
                update={
                    "question": "Что ты можешь рассказать об Альберте Эйнштейне?",
                    "context": (
                        "Альберт Эйнштейн (14 марта 1879 — 18 апреля 1955) — "
                        "физик-теоретик, родившийся в Германии, широко считающийся "
                        "одним из величайших и наиболее влиятельных учёных всех времён. "
                        "Прежде всего известен разработкой теории относительности, он "
                        "также внёс важный вклад в квантовую механику и был центральной "
                        "фигурой в революционном переосмыслении научной картины природы, "
                        "произошедшем в первые десятилетия двадцатого века благодаря "
                        "современной физике. Его формула эквивалентности массы и энергии "
                        "E = mc², вытекающая из теории относительности, была названа "
                        "«самым известным уравнением в мире». Он получил Нобелевскую "
                        "премию по физике 1921 года «за заслуги перед теоретической "
                        "физикой и особенно за открытие закона фотоэлектрического "
                        "эффекта» — важнейший шаг в развитии квантовой теории. Его "
                        "работы также известны своим влиянием на философию науки. В "
                        "опросе 1999 года среди 130 ведущих физиков мира, проведённом "
                        "британским журналом Physics World, Эйнштейн был признан "
                        "величайшим физиком всех времён. Его интеллектуальные "
                        "достижения и оригинальность сделали имя Эйнштейна синонимом "
                        "гениальности."
                    ),
                    "answer": (
                        "Альберт Эйнштейн, родившийся 14 марта 1879 года, был "
                        "физиком-теоретиком, родившимся в Германии, широко "
                        "считающимся одним из величайших и наиболее влиятельных "
                        "учёных всех времён. Он получил Нобелевскую премию по физике "
                        "1921 года за заслуги перед теоретической физикой."
                    ),
                }
            ),
            ver_1.model_copy(
                update={
                    "reason": (
                        "Предоставленный контекст действительно был полезен для "
                        "получения данного ответа. Контекст содержит ключевую "
                        "информацию о жизни и вкладе Альберта Эйнштейна, которая "
                        "отражена в ответе."
                    ),
                    "verdict": 1,
                }
            ),
        ),
        (
            qac_2.model_copy(
                update={
                    "question": "кто выиграл чемпионат мира по крикету ICC 2020 года?",
                    "context": (
                        "Чемпионат мира по крикету ICC среди мужчин в формате Т20 2022 "
                        "года, проходивший с 16 октября по 13 ноября 2022 года в "
                        "Австралии, стал восьмым розыгрышем турнира. Изначально "
                        "запланированный на 2020 год, он был перенесён из-за пандемии "
                        "COVID-19. Победу одержала сборная Англии, обыгравшая Пакистан "
                        "в финале с разницей в пять калиток и завоевавшая свой второй "
                        "титул чемпиона мира ICC по крикету Т20."
                    ),
                    "answer": "Англия",
                }
            ),
            ver_2.model_copy(
                update={
                    "reason": (
                        "контекст был полезен для прояснения ситуации с чемпионатом "
                        "мира ICC 2020 года и указания на то, что Англия стала "
                        "победителем турнира, который должен был состояться в 2020 "
                        "году, но фактически прошёл в 2022 году."
                    ),
                    "verdict": 1,
                }
            ),
        ),
        (
            qac_3.model_copy(
                update={
                    "question": "Какая самая высокая гора в мире?",
                    "context": (
                        "Анды — самая длинная континентальная горная цепь в мире, "
                        "расположенная в Южной Америке. Она простирается через семь "
                        "стран и включает многие из самых высоких вершин Западного "
                        "полушария. Хребет известен своими разнообразными "
                        "экосистемами, включая высокогорное Андское плато и часть "
                        "Амазонских тропических лесов."
                    ),
                    "answer": "Эверест.",
                }
            ),
            ver_3.model_copy(
                update={
                    "reason": (
                        "приведённый контекст описывает горную цепь Анд, которая, хотя "
                        "и впечатляюща, не включает Эверест и напрямую не относится к "
                        "вопросу о самой высокой горе в мире."
                    ),
                    "verdict": 0,
                }
            ),
        ),
    ]

    # --- context_recall ---
    qca, classifications = context_recall.context_recall_prompt.examples[0]
    context_recall.context_recall_prompt.instruction = (
        "Даны контекст и ответ. Проанализируй каждое предложение в ответе и "
        "определи, можно ли его отнести к данному контексту. Используй "
        "только «Да» (1) или «Нет» (0) как бинарную классификацию. Выведи "
        "JSON с обоснованием."
    )
    context_recall.context_recall_prompt.examples = [
        (
            qca.model_copy(
                update={
                    "question": "Что ты можешь рассказать об Альберте Эйнштейне?",
                    "context": (
                        "Альберт Эйнштейн (14 марта 1879 — 18 апреля 1955) — "
                        "физик-теоретик, родившийся в Германии, широко считающийся "
                        "одним из величайших и наиболее влиятельных учёных всех времён. "
                        "Прежде всего известен разработкой теории относительности, он "
                        "также внёс важный вклад в квантовую механику и был центральной "
                        "фигурой в революционном переосмыслении научной картины природы, "
                        "произошедшем в первые десятилетия двадцатого века благодаря "
                        "современной физике. Его формула эквивалентности массы и энергии "
                        "E = mc², вытекающая из теории относительности, была названа "
                        "«самым известным уравнением в мире». Он получил Нобелевскую "
                        "премию по физике 1921 года «за заслуги перед теоретической "
                        "физикой и особенно за открытие закона фотоэлектрического "
                        "эффекта» — важнейший шаг в развитии квантовой теории. Его "
                        "работы также известны своим влиянием на философию науки. В "
                        "опросе 1999 года среди 130 ведущих физиков мира, проведённом "
                        "британским журналом Physics World, Эйнштейн был признан "
                        "величайшим физиком всех времён. Его интеллектуальные "
                        "достижения и оригинальность сделали имя Эйнштейна синонимом "
                        "гениальности."
                    ),
                    "answer": (
                        "Альберт Эйнштейн, родившийся 14 марта 1879 года, был "
                        "физиком-теоретиком, родившимся в Германии, широко считающимся "
                        "одним из величайших и наиболее влиятельных учёных всех времён. "
                        "Он получил Нобелевскую премию по физике 1921 года за заслуги "
                        "перед теоретической физикой. В 1905 году он опубликовал 4 "
                        "статьи. Эйнштейн переехал в Швейцарию в 1895 году."
                    ),
                }
            ),
            classifications.model_copy(
                update={
                    "classifications": [
                        classifications.classifications[0].model_copy(
                            update={
                                "statement": (
                                    "Альберт Эйнштейн, родившийся 14 марта 1879 года, был "
                                    "физиком-теоретиком, родившимся в Германии, широко "
                                    "считающимся одним из величайших и наиболее влиятельных "
                                    "учёных всех времён."
                                ),
                                "reason": "Дата рождения Эйнштейна чётко указана в контексте.",
                                "attributed": 1,
                            }
                        ),
                        classifications.classifications[1].model_copy(
                            update={
                                "statement": (
                                    "Он получил Нобелевскую премию по физике 1921 года за "
                                    "заслуги перед теоретической физикой."
                                ),
                                "reason": (
                                    "Точно такое же предложение присутствует в данном контексте."
                                ),
                                "attributed": 1,
                            }
                        ),
                        classifications.classifications[2].model_copy(
                            update={
                                "statement": "В 1905 году он опубликовал 4 статьи.",
                                "reason": (
                                    "В данном контексте нет упоминания о "
                                    "статьях, которые он написал."
                                ),
                                "attributed": 0,
                            }
                        ),
                        classifications.classifications[3].model_copy(
                            update={
                                "statement": "Эйнштейн переехал в Швейцарию в 1895 году.",
                                "reason": "В контексте нет подтверждающих сведений об этом.",
                                "attributed": 0,
                            }
                        ),
                    ]
                }
            ),
        )
    ]


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
    }
    with open(Path(path), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _build_default_llm(model: str) -> BaseRagasLLM:
    chat = ChatOpenAI(
        base_url=os.environ.get("JUDGE_BASE_URL"),
        api_key=os.environ.get("JUDGE_API_KEY", "not-needed"),
        model=model,
        temperature=0,
    )
    with warnings.catch_warnings():
        # LangchainLLMWrapper is ragas' deprecated LLM interface -- see the
        # module docstring for why it's still the one that works here.
        warnings.simplefilter("ignore", DeprecationWarning)
        return LangchainLLMWrapper(chat)


class RagasJudge(Judge):
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
        usage_log_path: str | Path | None = None,
    ):
        model_name = model or os.environ.get("JUDGE_MODEL", "")
        self.llm = llm or _build_default_llm(model_name or os.environ["JUDGE_MODEL"])
        self._model_name = model_name
        self.usage_log_path = usage_log_path or os.environ.get(
            "JUDGE_USAGE_LOG_PATH", DEFAULT_USAGE_LOG_PATH
        )
        faithfulness = Faithfulness(name="faithfulness")
        answer_correctness = FactualCorrectness(name="answer_correctness")
        answer_relevancy = AspectCritic(name="answer_relevancy", definition=RELEVANCY_DEFINITION)
        context_precision = LLMContextPrecisionWithReference(name="context_precision")
        context_recall = LLMContextRecall(name="context_recall")
        _localize_prompts_to_russian(
            faithfulness, answer_correctness, answer_relevancy, context_precision, context_recall
        )
        self._metrics = [
            faithfulness,
            answer_correctness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]

    def evaluate(self, run_df: pd.DataFrame) -> pd.DataFrame:
        ragas_df = run_df.rename(columns=RUN_TO_RAGAS_COLUMNS)[list(RUN_TO_RAGAS_COLUMNS.values())]
        dataset = EvaluationDataset.from_pandas(ragas_df)

        usage_cb = _TokenUsageCallback()
        # raise_exceptions=False (ragas default): a single row/metric that
        # fails to score becomes NaN in the result instead of aborting the
        # whole run.
        result = ragas_evaluate(
            dataset,
            metrics=self._metrics,
            llm=self.llm,
            callbacks=[usage_cb],
            show_progress=False,
        )
        _log_usage(self.usage_log_path, usage_cb, len(run_df), self._model_name)

        scores = result.to_pandas()
        df = run_df.copy()
        for name, metric in zip(self.metric_names, self._metrics, strict=True):
            df[name] = scores[_result_column(metric)].values
        return df
