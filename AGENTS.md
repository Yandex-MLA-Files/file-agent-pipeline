# AGENTS.md

## О проекте

`file-agent-pipeline` — ML Python-проект. Он реализует
простой RAG-пайплайн поверх пользовательских файлов:

```text
файлы
  -> парсинг
  -> Document / Block
  -> chunking
  -> retrieval
  -> QA prompt
  -> LLM
  -> ответ и источники
```

Пользователь может загрузить один или несколько документов, просмотреть
извлечённый текст, найти релевантные chunks и получить ответ LLM с источниками.

## Текущее состояние

Реализовано:

- единое представление `Document` / `Block`;
- парсинг Markdown, PDF, HTML, XLSX и PPTX;
- разбиение документов на chunks;
- BM25 retrieval;
- semantic retrieval через `sentence-transformers`;
- объединение результатов BM25 и semantic search через RRF;
- QA prompt layer и общий RAG-слой;
- OpenAI-compatible LLM-клиент;
- два LLM backend: Yandex AI Studio и локальный OpenAI-compatible endpoint;
- Streamlit-интерфейс с загрузкой нескольких файлов, preview, поиском и генерацией
  ответа;
- pytest-тесты для основных слоёв.

Поддерживаемые расширения: `.md`, `.pdf`, `.html`, `.htm`, `.xlsx`, `.pptx`.

## Структура проекта

```text
app.py                         # Streamlit-интерфейс
src/file_agent/
  document.py                 # Document и Block
  pipeline.py                 # выбор парсера по расширению
  chunking.py                 # разбиение документов на chunks
  retrieval.py                # BM25, semantic search и RRF
  qa.py                       # сбор контекста и QA prompt
  rag.py                      # полный RAG-пайплайн
  parsers/                    # парсеры поддерживаемых форматов
  llm/                        # интерфейс, клиент и фабрика LLM
tests/                        # pytest-тесты
docs/local_inference.md       # запуск локального LLM endpoint
```

## Архитектурные правила

- Держать парсинг, retrieval, QA и интеграцию с LLM отдельными слоями.
- Все парсеры должны возвращать единое представление `Document` с набором
  `Block`.
- Сохранять доступные метаданные:
  - `page` для PDF;
  - `slide` для PPTX;
  - `sheet` для XLSX;
  - `block_type` для типа блока;
  - имя исходного файла и прочие полезные координаты источника.
- Выбор парсера по расширению держать в `src/file_agent/pipeline.py`.
- Работу с LLM выполнять только через интерфейс `LLMClient`.
- Настройки конкретных backend держать в `src/file_agent/llm/factory.py` и
  переменных окружения.
- Не добавлять сложные абстракции без практической необходимости. Предпочитать
  простой читаемый код с type hints.
- Для нового парсера или нового поведения добавлять тесты.
- Тесты не должны выполнять реальные запросы к облачным или локальным LLM.
  Сетевое взаимодействие проверять через mocks/fakes.

## Ограничения текущего этапа

Пока не добавлять без отдельной задачи:

- LangChain и LangGraph;
- OCR и VLM;
- сложную агентную архитектуру;
- отдельную vector database или FAISS;
- анализ изображений из PPTX;
- вычисление Excel-формул.

Для XLSX используется `data_only=True`: парсер читает сохранённые значения формул,
но сам формулы не вычисляет.

## LLM и переменные окружения

Секреты и идентификаторы нельзя хардкодить. Настоящий `.env` нельзя коммитить;
при изменении конфигурации нужно актуализировать `.env.example`.

Общие настройки:

- `LLM_BACKEND` — `yandex` или `local`.

Yandex AI Studio:

- `YANDEX_API_KEY`;
- `YANDEX_FOLDER_ID`;
- `YANDEX_MODEL`;
- `YANDEX_BASE_URL`.

Локальный OpenAI-compatible backend:

- `LOCAL_LLM_BASE_URL`;
- `LOCAL_LLM_API_KEY`;
- `LOCAL_LLM_MODEL`.

Не выполнять реальные API-запросы в тестах и не добавлять рабочие ключи в код,
тестовые данные, логи или документацию.

## Стек

- Python 3.11+;
- Streamlit;
- PyMuPDF;
- BeautifulSoup;
- Markdown;
- openpyxl;
- python-pptx;
- sentence-transformers;
- openai;
- pytest.

## Правила внесения изменений

Перед изменением изучить соответствующий модуль и существующие тесты. Сохранять
обратную совместимость, если задача явно не требует другого поведения.

После изменения:

- добавить или обновить тесты для затронутого поведения;
- запустить как минимум релевантные тесты;
- по возможности запустить весь test suite;
- обновить `README.md`, `.env.example` или `docs/`, если изменились интерфейс,
  конфигурация, поддерживаемые форматы или команды запуска.

Не коммитить временные файлы, кэш, модели, пользовательские документы, `.env` и
другие секреты.

## Команды

Установка зависимостей в Windows:

```powershell
uv sync
```

Запуск всех тестов:

```powershell
uv run pytest
```

Проверка линтером и форматированием:

```powershell
uv run ruff check .
uv run ruff format --check .
```

Запуск Streamlit:

```powershell
uv run streamlit run app.py
```

Подробнее о локальном inference см. в `docs/local_inference.md`.
