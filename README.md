# File Agent Pipeline

ML проект для построения пайплайна работы с файлами.

Проект принимает файл, извлекает текст, приводит документ к единому внутреннему
представлению, разбивает документ на chunks, ищет релевантные chunks по запросу
пользователя и готовит prompt для будущего LLM-ответа.

Сейчас реального доступа к YandexGPT API нет, поэтому настоящие API-запросы не
выполняются. Для проверки полного пайплайна используется `FakeLLM`.

## Текущий Статус MVP

Уже реализовано:

- `Document` / `Block` IR для единого представления документов.
- Парсинг Markdown-файлов.
- Парсинг PDF через PyMuPDF.
- Парсинг HTML через BeautifulSoup.
- Chunking документов.
- Простой keyword retrieval по chunks.
- QA prompt layer для сборки контекста и prompt.
- `YandexGPTClient` подготовлен заранее, но не подключен к Streamlit и не вызывается в тестах.
- `FakeLLM` mode для проверки полного QA-пайплайна без настоящего API.
- Streamlit-интерфейс для загрузки файла, просмотра текста, поиска chunks и проверки FakeLLM.
- Pytest-тесты для парсеров, chunking, retrieval, QA layer, FakeLLM и YandexGPTClient.

## Поддерживаемые Форматы

- `.md`
- `.pdf`
- `.html`
- `.htm`

## Как Работает Текущий Пайплайн

```text
file
  -> parse_file
  -> Document / Block
  -> chunk_document
  -> search_chunks
  -> build QA prompt
  -> FakeLLM / YandexGPT later
  -> answer
```

Основные шаги:

- `parse_file(file_path)` выбирает парсер по расширению файла.
- Парсер возвращает `Document` с набором `Block`.
- `chunk_document(document)` разбивает blocks на chunks.
- `search_chunks(query, chunks)` ищет релевантные chunks простым keyword scoring.
- QA layer собирает context из найденных chunks и строит prompt.
- Сейчас prompt можно проверить через `FakeLLM`.

## FakeLLM

`FakeLLM` — временная заглушка для проверки пайплайна без доступа к YandexGPT.

Важно:

- `FakeLLM` не генерирует настоящий ответ по документу.
- Он проверяет, что найденные chunks были собраны в context.
- Он проверяет, что prompt был построен.
- Он проверяет, что был вызван `llm_client.generate(prompt)`.
- В ответе выводится тестовое сообщение, длина prompt и preview prompt.

После получения доступа к YandexGPT `FakeLLM` можно будет заменить на
`YandexGPTClient`.

## Установка

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Запуск Тестов

```bash
.\.venv\Scripts\python.exe -m pytest
```

## Запуск Streamlit

```bash
.\.venv\Scripts\python.exe -m streamlit run app.py
```

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

- Настоящий ответ YandexGPT пока не используется, потому что доступа к API еще нет.
- `YandexGPTClient` подготовлен, но не подключен к Streamlit.
- Keyword retrieval очень простой и может искать не идеально.
- OCR/VLM пока не реализованы.
- XLSX/PPTX пока не реализованы.
- Embeddings, vector search, FAISS, LangChain и LangGraph пока не добавлены.

## Roadmap

Следующие шаги:

- Подключить настоящий YandexGPT API.
- Добавить режим YandexGPT в Streamlit.
- Добавить `XLSXParser`.
- Добавить `PPTXParser`.
- Улучшить retrieval.
- Добавить embeddings / vector search.
- Исследовать агентный подход и LangGraph.
- Добавить OCR/VLM для сканов и изображений.
- Собрать тестовый датасет и метрики качества.

## Структура MVP

Ключевые файлы:

- `src/file_agent/document.py` — `Document` и `Block`.
- `src/file_agent/pipeline.py` — выбор парсера по расширению файла.
- `src/file_agent/parsers/` — парсеры Markdown, PDF и HTML.
- `src/file_agent/chunking.py` — разбиение документов на chunks.
- `src/file_agent/retrieval.py` — простой keyword retrieval.
- `src/file_agent/qa.py` — сбор context и QA prompt.
- `src/file_agent/llm/fake.py` — FakeLLM для локальной проверки.
- `src/file_agent/llm/yandexgpt.py` — подготовленный клиент YandexGPT.
- `app.py` — Streamlit MVP.
- `tests/` — pytest-тесты.
