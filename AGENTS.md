# AGENTS.md

## Проект

Это Python-проект для ML-стажировки.

Цель проекта — построить пайплайн для работы с файлами:
пользователь загружает файл и задаёт вопрос, а система извлекает содержимое файла
и позже отвечает на вопрос с помощью LLM.

Поддерживаемые форматы в будущем:
- PDF
- PPTX
- XLSX
- Markdown
- HTML

## Текущая цель MVP

Базовый MVP уже реализован:

- парсинг Markdown, PDF, HTML
- Document / Block
- chunking
- простой keyword retrieval
- Streamlit-интерфейс для загрузки файла, просмотра текста и поиска по chunks

Следующая цель — добавить LLM QA:

file + question -> relevant chunks -> LLM answer

Разрешено добавить:
- обёртку над YandexGPT
- prompt builder для ответа по контексту
- функцию answer_question(document, question)

Пока не добавлять:
- LangChain
- LangGraph
- OCR
- VLM
- embeddings / FAISS
- сложную агентную архитектуру

## Стек

- Python 3.11+
- Streamlit для демо-интерфейса
- PyMuPDF для парсинга PDF
- BeautifulSoup для парсинга HTML
- markdown для Markdown-файлов
- pytest для тестов

## Архитектурные правила

- Логику парсинга файлов держать отдельно от логики LLM.
- Все файлы приводить к единому представлению Document.
- Сохранять метаданные, если они доступны:
  - page для PDF
  - slide для PPTX
  - sheet для XLSX
  - block_type для типа блока
- Не усложнять архитектуру раньше времени.
- Писать простой и читаемый код.
- Использовать type hints.
- Для каждого парсера добавлять тесты.

## Команды

Запуск тестов:

```bash
pytest