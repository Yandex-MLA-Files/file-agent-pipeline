# File Agent Pipeline

ML-стажировочный проект для построения пайплайна работы с файлами.

Проект принимает файл, извлекает текст, приводит документ к единому внутреннему
представлению, разбивает документ на chunks, ищет релевантные chunks по запросу
пользователя и готовит prompt для будущего LLM-ответа.

Сейчас реального доступа к YandexGPT API нет, поэтому настоящие API-запросы не
выполняются. Для проверки полного пайплайна используется `FakeLLM`.

## Текущий Статус MVP

Уже реализовано:

- `Document` / `Block` IR для единого представления документов.
- Парсинг `.md` через `MarkdownParser`.
- Парсинг `.pdf` через `PDFParser` и PyMuPDF.
- Парсинг `.html` и `.htm` через `HTMLParser` и BeautifulSoup.
- Парсинг `.xlsx` через `XLSXParser` и openpyxl.
- Парсинг `.pptx` через `PPTXParser` и python-pptx.
- Chunking документов.
- Простой keyword-based retrieval по chunks.
- QA prompt layer для сборки контекста и prompt.
- `YandexGPTClient` подготовлен заранее, но не подключен к Streamlit и не вызывается в тестах.
- `FakeLLM` mode для проверки полного QA-пайплайна без настоящего API.
- Streamlit-интерфейс для загрузки файла, просмотра текста, поиска chunks и проверки FakeLLM.
- Pytest-тесты для парсеров, chunking, retrieval, QA layer, FakeLLM и YandexGPTClient.

Поддерживаемые форматы:

- `.md`
- `.pdf`
- `.html`
- `.htm`
- `.xlsx`
- `.pptx`

## Как Работает Текущий Пайплайн

```text
file -> parse_file -> Document/Block -> chunk_document -> search_chunks -> QA prompt -> FakeLLM/YandexGPT later -> answer
```

Основные шаги:

- `parse_file(file_path)` выбирает парсер по расширению файла.
- Парсер возвращает `Document` с набором `Block`.
- `chunk_document(document)` разбивает blocks на chunks.
- `search_chunks(query, chunks)` ищет релевантные chunks простым keyword scoring.
- QA layer собирает context из найденных chunks и строит prompt.
- Сейчас prompt можно проверить через `FakeLLM`; позже его можно будет отправлять в YandexGPT.

## Парсеры

`MarkdownParser` читает Markdown-файлы и создаёт один `Block` с исходным текстом документа.

`PDFParser` читает PDF через PyMuPDF и создаёт отдельный `Block` для каждой страницы. В metadata сохраняется номер страницы и имя исходного файла.

`HTMLParser` читает HTML через BeautifulSoup, удаляет `script` и `style`, затем извлекает читаемый текст страницы в один `Block`.

`XLSXParser` читает Excel-файлы через openpyxl в режиме `read_only=True` и `data_only=True`. Он создаёт отдельный `Block` для каждого листа, а строки листа превращает в табличный текст с разделителем `\t`.

`PPTXParser` читает PowerPoint-презентации через python-pptx и создаёт отдельный `Block` для каждого слайда. Он извлекает заголовки, обычные текстовые блоки и текст из таблиц.

## FakeLLM

`FakeLLM` — временная заглушка для проверки пайплайна без доступа к YandexGPT.

Важно:

- `FakeLLM` не генерирует настоящий ответ по документу.
- Он проверяет, что найденные chunks были собраны в context.
- Он проверяет, что prompt был построен.
- Он проверяет, что был вызван `llm_client.generate(prompt)`.
- В ответе выводится тестовое сообщение, длина prompt и preview prompt.

После получения доступа к YandexGPT `FakeLLM` можно будет заменить на `YandexGPTClient`.

## Установка

Проект использует [uv](https://docs.astral.sh/uv/) для управления зависимостями.

```bash
uv sync
```

Чтобы линт и форматирование запускались автоматически перед каждым коммитом,
один раз установи git-хуки:

```bash
uv run pre-commit install
```

## Запуск Тестов

```bash
uv run pytest
```

## Линт и форматирование

```bash
uv run ruff check .
uv run ruff format --check .
```

## Запуск Streamlit

```bash
uv run streamlit run app.py
```

Если Streamlit не подхватывает новые парсеры, нужно полностью остановить старый
процесс сервера и запустить команду выше заново.

## Переменные Окружения

В проекте есть пример файла `.env.example`:

```env
YANDEX_API_KEY=your_api_key_here
YANDEX_FOLDER_ID=your_folder_id_here
YANDEX_MODEL=yandexgpt-lite
```

Для локального запуска с настоящим YandexGPT позже нужно будет создать файл
`.env` и заполнить реальные значения.

Настоящий `.env` нельзя коммитить. Он уже добавлен в `.gitignore`.

## Текущие Ограничения

- Настоящий ответ YandexGPT пока не используется, потому что доступа к API ещё нет.
- `YandexGPTClient` подготовлен, но не подключен к Streamlit.
- Retrieval пока keyword-based и может искать не идеально.
- OCR пока нет.
- VLM пока нет.
- Изображения из PPTX пока не анализируются.
- Excel-формулы не вычисляются; используется `data_only=True`, поэтому берутся сохранённые значения ячеек.
- Embeddings, vector search, FAISS, LangChain и LangGraph пока не добавлены.

## Roadmap

Следующие шаги:

- Подключить настоящий YandexGPT.
- Добавить режим YandexGPT в Streamlit.
- Улучшить retrieval.
- Добавить evaluation dataset.
- Добавить OCR/VLM.
- Исследовать агентный подход и LangGraph.

## Структура MVP

Ключевые файлы:

- `src/file_agent/document.py` — `Document` и `Block`.
- `src/file_agent/pipeline.py` — выбор парсера по расширению файла.
- `src/file_agent/parsers/` — парсеры Markdown, PDF, HTML, XLSX и PPTX.
- `src/file_agent/chunking.py` — разбиение документов на chunks.
- `src/file_agent/retrieval.py` — простой keyword retrieval.
- `src/file_agent/qa.py` — сбор context и QA prompt.
- `src/file_agent/llm/fake.py` — FakeLLM для локальной проверки.
- `src/file_agent/llm/yandexgpt.py` — подготовленный клиент YandexGPT.
- `app.py` — Streamlit MVP.
- `tests/` — pytest-тесты.
