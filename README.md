# File Agent Pipeline

Python-проект для построения RAG-пайплайна поверх пользовательских файлов.

Система принимает один или несколько документов, извлекает текст, приводит данные к
единому представлению `Document` / `Block`, разбивает документ на chunks, ищет
релевантный контекст и отправляет prompt в LLM.

## Текущий статус

Реализовано:

- единое представление `Document` / `Block`;
- парсинг `.md`, `.pdf`, `.html`, `.htm`, `.xlsx`, `.pptx`;
- chunking документов;
- retrieval: BM25 + semantic search через `sentence-transformers` + RRF;
- QA prompt layer;
- общий RAG-слой в `src/file_agent/rag.py`;
- генерация через OpenAI-compatible API;
- два режима LLM backend:
  - Yandex AI Studio / Qwen;
  - локальный OpenAI-compatible endpoint, например vLLM или SGLang;
- Streamlit-интерфейс для загрузки нескольких файлов, поиска chunks и генерации ответа;
- pytest-тесты для парсеров, chunking, retrieval, QA, RAG и LLM-клиентов.

## Как работает пайплайн

```text
files
  -> parse_file
  -> Document / Block
  -> chunk_document
  -> search_chunks
  -> build QA prompt
  -> LLMClient.generate
  -> answer + sources
```

Ключевые слои:

- `pipeline.py` выбирает парсер по расширению файла.
- `chunking.py` разбивает `Document` на chunks.
- `retrieval.py` ищет релевантные chunks через BM25, semantic search и RRF.
- `qa.py` собирает контекст и prompt.
- `rag.py` связывает полный путь от файлов до ответа.
- `llm/` содержит общий интерфейс и OpenAI-compatible клиент.

## Структура проекта

```text
src/file_agent/
  document.py
  pipeline.py
  rag.py
  chunking.py
  retrieval.py
  qa.py

  parsers/
    base.py
    md_parser.py
    pdf_parser.py
    html_parser.py
    xlsx_parser.py
    pptx_parser.py

  llm/
    base.py
    openai_compatible.py
    factory.py
```

## LLM backend

В проекте используется один общий интерфейс:

```python
class LLMClient(Protocol):
    def generate(self, prompt: str) -> str:
        ...
```

`OpenAICompatibleClient` ходит в endpoint вида:

```text
{base_url}/chat/completions
```

Один и тот же клиент используется для облачного и локального режима. Разница только
в переменных окружения.

### Yandex AI Studio

```env
LLM_BACKEND=yandex

YANDEX_API_KEY=your_api_key_here
YANDEX_FOLDER_ID=your_folder_id_here
YANDEX_MODEL=qwen3.6-35b-a3b
YANDEX_BASE_URL=https://ai.api.cloud.yandex.net/v1
```

Если `YANDEX_MODEL` не начинается с `gpt://`, код соберет model URI так:

```text
gpt://<YANDEX_FOLDER_ID>/<YANDEX_MODEL>
```

### Local LLM

```env
LLM_BACKEND=local

LOCAL_LLM_BASE_URL=http://localhost:8000/v1
LOCAL_LLM_API_KEY=
LOCAL_LLM_AUTH_SCHEME=Bearer
LOCAL_LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct
```

Локальная модель должна быть поднята отдельно через vLLM, SGLang или другой
OpenAI-compatible server. Подробнее см. `docs/local_inference.md`.

## Установка

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Запуск тестов

```bash
.\.venv\Scripts\python.exe -m pytest
```

## Запуск Streamlit

```bash
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Переменные окружения

В репозитории есть `.env.example`. Для реального запуска создайте локальный `.env`
и заполните нужный backend.

Настоящий `.env` нельзя коммитить. Он должен оставаться в `.gitignore`.

## Текущие ограничения

- OCR пока нет.
- VLM пока нет.
- Изображения из PPTX не анализируются.
- Excel-формулы не вычисляются, используется `data_only=True`.
- Embeddings используются только для semantic retrieval через `sentence-transformers`;
  FAISS и отдельное vector storage пока не добавлены.
- LangChain и LangGraph не используются.
