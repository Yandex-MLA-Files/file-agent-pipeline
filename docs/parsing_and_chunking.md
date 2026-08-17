# Parsing, chunking and OCR/VLM: design and evaluation

This document describes the structured ingestion pipeline (default since the
`feat/parsing-chunking` branch), the fallbacks that keep the previous behaviour
available, and how the changes were measured with the project's LLM-as-judge
(`eval_pipeline`, ragas metrics) on the team dataset
`sandrik1271/RAG-QA-Dataset` (127 questions, 21 documents, 7 formats).

## 1. Why the previous ingestion lost accuracy

| Problem observed on the dataset documents | Effect on answers |
|---|---|
| DOCX went through Docling, which splits a paragraph into formatting runs (`"Целью"`, `"вступительного испытания …"`) — sentence fragments became blocks | Fragmented chunks, wrong sentence boundaries, headings without body |
| PPTX/XLSX/HTML/TXT were flattened into one text blob per slide/sheet/file (`"Slide 3\nText: …"`, TSV rows) | No headings, no tables, slide numbers not citable, header rows lost after splitting |
| Transcript TXT kept a `[mm:ss]` timestamp every 5 words | ~40 % of every chunk was timestamps |
| The embedding model (`paraphrase-multilingual-MiniLM-L12-v2`) has a **128-token** window | Chunks of ~350 characters; a table header alone could exceed the budget |
| One DOCX made the chunker step one character at a time (a table row denser in tokens than the window) | Row could not be processed within hours (recorded as a failure in the baseline) |
| Only the nearest heading was repeated on continuation chunks; PDF headings all had level 1 | Chunks did not know which chapter/section they belong to |
| Scanned/skewed PDF pages went to EasyOCR | Garbled text on skewed scans, formulas lost |
| Figures were described only in PDFs and only when a separate VLM was configured (`VLM_BACKEND=off` by default) | Diagrams in DOCX/PPTX invisible to retrieval |

## 2. What the pipeline does now

### 2.1 Structured parsers (`PARSER_PROFILE=structured`, default)

Every parser produces the same `Document`/`Block` contract: typed blocks
(`heading` with `hierarchy_level`, `text`, `list`, `table` as Markdown,
`figure` with `image_bytes`, `code`, `formula`), page/slide/sheet
coordinates and a document title.

| Format | Implementation | Structure recovered |
|---|---|---|
| PDF | Docling (layout model, reading order, tables) + post-processing in `docling_parser.py` | heading levels inferred from numbering (`1.2.3` → level 3), consecutive list items grouped into one list block (bullet glyphs stripped), captions folded into figure/table blocks, headings split over two lines stitched, running headers/footers dropped |
| DOCX | `python-docx` (`docx_parser.py`), Docling as fallback | whole paragraphs; heading levels from `Heading N`/`Заголовок N` styles, outline levels or bold-and-larger formatting; numbered/bulleted lists; tables with merged cells; embedded pictures with captions; monospace paragraphs as code |
| PPTX | `python-pptx` (`pptx_parser.py`) | slide title → heading (level 1 for section dividers, else 2), body in visual reading order with grouped shapes flattened, bullet lists with indentation, tables and charts as Markdown, pictures with image bytes, speaker notes; slide number stored as `page_number` |
| XLSX | `openpyxl` (`xlsx_parser.py`) | one heading per sheet, one Markdown table per data region (blank rows split regions), one- or two-row header detection, merged cells filled, note cells kept as text, `1100.0 → 1100`, ISO dates; a **profile block** per table (row count, column types, min/max with row label, sums/means, distinct values, sums grouped by every low-cardinality column) so aggregate questions are answerable from retrieval |
| HTML | BeautifulSoup walker (`html_parser.py`) | `h1–h6`, paragraphs, nested lists, tables, `pre` code, `img` alt text; nav/header/footer/script/style removed |
| Markdown | `md_parser.py` | ATX/setext headings, fenced code, pipe tables, lists, images, YAML front matter |
| TXT | `txt_parser.py` | encoding detection (UTF-8/16, cp1251, koi8-r, cp866); prose: paragraph reflow of hard-wrapped lines and title detection (`* CAPS *`, standalone short lines); transcripts: timestamps removed, captions re-flowed into ~140-word paragraphs with `time_start` metadata |

The original parsers are kept unchanged in `parsers/legacy/` and selected
with `PARSER_PROFILE=legacy` (for DOCX that means Docling).

### 2.2 OCR and VLM through the chat model (`VLM_BACKEND=llm`, `OCR_ENGINE=vlm`)

The answering model served for the project (Qwen3.5-27B on vLLM) is
multimodal, so by default the **same endpoint** is used for vision:

- **Scanned pages.** `parsers/routing.py` still decides per page (locally,
  no network) whether a text layer is missing. Those pages are rendered at
  150 dpi and transcribed to Markdown by `parsers/vlm_ocr.py` with a strict
  "transcribe, do not interpret" prompt (headings, lists, tables, LaTeX
  formulas kept; page furniture skipped). The transcript replaces Docling's
  placeholder blocks for those pages, in reading order. Thinking is disabled
  for vision calls (about 1–3 s per page on an A100). If no VLM endpoint is
  configured, or the endpoint fails, the classic path runs (EasyOCR inside
  Docling; `OCR_ENGINE=easyocr|rapidocr` forces it).
- **Figures.** `parsers/enhancer.py` describes the largest figures
  (`VLM_MAX_FIGURES`, default 8) of a document — PDF page crops as before,
  and now the embedded images of DOCX/PPTX — and folds the description into
  the block text so it is indexed. One failed call aborts the remaining
  figures of that document instead of paying a timeout each.

`VLM_BACKEND=openai` (separate vision endpoint), `smolvlm` (local) and `off`
remain available.

### 2.3 Structured chunker (`CHUNKING_STRATEGY=structured`, default)

`chunking.py` keeps the section-packing, token-budgeted, small-to-big design
and adds:

- **Heading path.** Sections are leaf sections with their full ancestor path;
  every chunk is prefixed with `Title > Chapter > Section` (bounded to 20 %
  of the budget, nearest headings kept) and stores `heading_path`,
  `doc_title`, `section` in metadata. Retrieval (dense and BM25) therefore
  sees where a passage lives; queries that name a topic or document match
  even when the passage itself does not repeat it.
- **Type-aware splitting.** Prose is split on sentence boundaries with
  sentence overlap (Russian/English abbreviations protected), lists on
  items, code on lines, tables on rows with caption + header repeated (or the
  header once when it is too wide); pieces never consist of a heading alone.
- **Parent windows.** A piece of a long section carries in
  `metadata["context"]` a window of the section *around the piece* (not its
  first 4000 characters), opened by the heading path — that is what the LLM
  reads.
- **Budget.** With the tokenizer of the embedding model, chunks target
  `CHUNK_TARGET_TOKENS` (384) capped by the encoder window; sizes are cached;
  the character-window fallback always advances by at least half a window
  (this removes the multi-hour pathological case above).

`CHUNKING_STRATEGY=legacy` runs the original chunker (`chunking_legacy.py`).

### 2.4 Retrieval and answering

- Dense embeddings default to `BAAI/bge-m3` (8192-token window, strong on
  Russian) loaded in fp16 on the GPU; hybrid BM25 + vector search with RRF is
  unchanged. `EMBEDDING_MODEL` selects any sentence-transformers model
  (the chunker budgets with the same model's tokenizer).
- Optional cross-encoder reranking (`RERANKER_MODEL`, recommended
  `BAAI/bge-reranker-v2-m3`): the top-20 hybrid hits are re-scored with the
  query and passage side by side, which is what finally ranks a numeric
  financial table above prose that merely repeats the query words.
- Document-diverse top-k (`RETRIEVAL_DIVERSIFY_DOCS`, on): with several
  indexed files, the best hit of every file is kept before the remaining
  slots are filled by score, so "compare A and B" questions see both files.
- The QA prompt (`QA_PROMPT=v2`) shows every passage under a compact header
  (`source_file`, pages, heading path) instead of raw retrieval metadata and
  asks for a complete, grounded answer with sources, and for an explicit
  conclusion on comparison / "does the document mention" questions; `v1` is
  the original prompt.

### 2.5 Evaluation harness changes

- `generate_hf_dataset.py --continue-on-error --record-timeout N` records a
  row that fails or exceeds `N` seconds as `[PIPELINE_ERROR] …` with empty
  contexts (scored 0 by the judge, as agreed in the team) instead of aborting
  the run; `failed_count` is written to the manifest and each checkpoint
  records its wall-clock time.
- `eval_pipeline` accepts `JUDGE_EXTRA_BODY` (JSON forwarded with every
  judge request) so a Qwen3 judge can run with thinking disabled — a 4-row
  smoke test went from ~20 min / 182k output tokens to ~2 min / 14k tokens
  with the same metric ranges.

## 3. Evaluation

Setup: `mla-yac-a100-5`, Qwen3.5-27B (vLLM, `enable_thinking=false`,
temperature 0, top-k 5) both as the answering model and as the judge
(`JUDGE_MODEL=Qwen/Qwen3.5-27B`, thinking disabled), dataset
`sandrik1271/RAG-QA-Dataset` train split (127 rows), ragas metrics from
`eval_pipeline`. Rows the pipeline could not process count as 0.

### 3.1 Runs

| Run | What it is |
|---|---|
| `pc-baseline-rag-127` | Code before this branch (`main` + agent branch): Docling for PDF/DOCX with the pypdfium backend, flat PPTX/XLSX/HTML/TXT parsers, original chunker, `paraphrase-multilingual-MiniLM-L12-v2` (128-token window), original prompt. 5 rows could not be processed (the original chunker never finishes on `Курс лекций Основы философии.docx`; recorded as failures after a 240 s timeout). |
| `pc-structured-rag-127` (v2) | Structured parsers + structured chunker + `bge-m3` + VLM figure descriptions and VLM page OCR through the chat model + QA prompt v2. |
| `pc-structured-v4-127` (v4, **default configuration**) | v2 + Docling `docling-parse` backend with ACCURATE TableFormer (row labels of financial tables recovered) + cross-encoder reranking (`bge-reranker-v2-m3` over the top-20 hybrid hits) + document-diverse top-k + spreadsheet profile blocks + prompt rule for comparison/"does it mention" questions. |
| `pc-structured-v4-norerank-127` | v4 without the reranker (ablation). |
| `pc-structured-v4-agent-127` | v4 ingestion with the multi-step agent (`--answer-mode agent`). |

### 3.2 Results (ragas, judge = Qwen3.5-27B without thinking; pipeline failures scored 0)

| Run | rows | failed | faithfulness | answer_correctness | answer_relevancy | context_precision | context_recall | avg s/question |
|---|---|---|---|---|---|---|---|---|
| baseline (old code) | 127 | 5 | 0.686 | 0.412 | 0.538 | 0.548 | 0.605 | — |
| structured v2 | 127 | 0 | 0.847 | 0.593 | 0.738 | 0.715 | 0.786 | 22.5 |
| structured v4 (default) | 127 | 0 | 0.840 | 0.601 | 0.831 | 0.767 | 0.811 | 26.2 |
ROWS_PLACEHOLDER

Means over successfully processed rows only differ for the baseline (0.715 /
0.429 / 0.560 / 0.570 / 0.630 over 122 rows). Judge noise: three v4 rows
timed out in the judge on `answer_correctness` (counted as 0 above; the
mean over judged rows is 0.616).

Reading the table:

- Every metric improved from the baseline; the largest jumps are answer
  relevancy (+0.29), context recall (+0.21) and context precision (+0.22),
  i.e. retrieval now brings the right passages and the answers stay on the
  question. Answer correctness (recall of the reference facts) went from
  0.41 to 0.60.
- v4 over v2: recovering table row labels fixed the financial-statement
  questions (e.g. "доходы от пассажирских перевозок за 3 месяца 2026" — v2
  answered "нет информации", v4 answers 96 083 млн руб. from page 5), the
  spreadsheet profiles made aggregate questions answerable ("регион с
  наибольшей выручкой" → Utah 9 925.63), and document-diverse retrieval
  helped two-file questions.
- Remaining weak spots: questions that need cross-file joins over raw
  spreadsheet rows (e.g. the intersection of product names of two files),
  reference answers not grounded in the document (the tea question about
  cold-season drinking has no such passage in `Чай.md`), and multi-hop
  questions that combine a fact from one document with a claim about the
  other; the agent mode is the intended tool for the latter.

### 3.3 Per-format observations

- **PDF (Docling)**: `docling-parse` + ACCURATE TableFormer keeps row labels
  that the pypdfium backend with cell matching dropped (`|  |  | 96 083 |`
  became `| Доходы от пассажирских перевозок | 96 083 | 86 983 |`). On
  Windows the parser falls back to pypdfium (with cell matching off, which
  also keeps labels).
- **DOCX**: python-docx keeps whole paragraphs; the philosophy course
  (2465 Docling fragments) becomes 1127 blocks and chunks in under a second.
- **XLSX**: Markdown tables per region plus a profile block; the chunker
  repeats the header on every piece now that the budget is 384 tokens.
- **PPTX**: slide titles become headings with slide numbers; deck-level
  and figure names in any language are filtered.
- **TXT**: the transcript loses ~40 % timestamp noise per chunk; the book
  gets 37 chapter/story headings.
- **Scanned pages**: 29 skewed pages of `AB_test.pdf` are transcribed by the
  VLM (Markdown with headings/tables) instead of EasyOCR text; 1–3 s per
  page on the shared A100.

## 4. Reproducing a run

```bash
source ~/fa_env.sh                     # HF_HOME, PATH, PYTHONPATH (see docs/hf_dataset_generation.md)
cd ~/file-agent-pipeline
export RUN_NAME=structured-rag-127
export RUN_DIR="$FILE_AGENT_STORAGE/runs/$RUN_NAME"
uv run python generate_hf_dataset.py \
  --dataset-id sandrik1271/RAG-QA-Dataset --split train \
  --cache-dir "$HF_HOME" --output-dir "$RUN_DIR" \
  --top-k 5 --resume --continue-on-error --record-timeout 900

# baseline / ablations: PARSER_PROFILE=legacy CHUNKING_STRATEGY=legacy \
#   EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 QA_PROMPT=v1

cd eval_pipeline
JUDGE_EXTRA_BODY='{"chat_template_kwargs": {"enable_thinking": false}}' \
uv run python scripts/run_eval.py --run "$RUN_DIR/answers.parquet" --out "$FILE_AGENT_STORAGE/reports/$RUN_NAME"
```
