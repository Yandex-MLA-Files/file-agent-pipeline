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

### 2.0 The contract everything else is built on

One dataclass carries a document from the parser to the index, and every
decision downstream reads it rather than the file:

```python
Block(
    id, text, type,                # stable interface (legacy consumers use it)
    metadata,                      # source_file, caption, note_type, enrichment, …
    block_type=BlockType.TABLE,    # heading | text | list | table | figure |
                                   # image | formula | code
    page_number=7,                 # page, slide or sheet index — what a citation needs
    bbox=(x0, y0, x1, y1),         # top-left origin, used to crop the region again
    vlm_description="a bar chart", # filled by the figure describer
    image_bytes=b"\x89PNG…",       # DOCX/PPTX media, never serialised
)
```

Three properties of that contract matter more than they look:

- **`block_type` decides how a block is split.** Prose is cut on sentences, a
  list on items, a table on rows, code on lines. A parser that returns
  untyped text still works — it simply gets the character-window behaviour.
- **`page_number` and `bbox` are what let a later pass go back to the pixels.**
  Formula enrichment, table repair and figure description all re-render the
  region from the original PDF instead of guessing from text.
- **Metadata travels.** Anything a parser records (a footnote's id, a table's
  caption, `nested_in_table`, `enrichment`) ends up in the chunk metadata and
  therefore in the prompt header the model sees.

`Document` adds the file name, the file type, the block list, a table of
contents built from the headings, and a metadata dictionary with the parsing
method, the title, per-format counters (figures, tables, footnotes, formulas,
text boxes) and — when they ran — the OCR and enrichment statistics. Those
counters are how the audit tool notices that a document arrived without its
formulas.

#### The order things happen in, for a PDF

The PDF path is the one with the most moving parts; it runs in this order,
and each step is guarded so that its failure cannot take the document down:

1. **Routing** (`parsers/routing.py`, PyMuPDF only, no models). Per page:
   characters in the text layer, raster coverage, vector-drawing count →
   `needs_ocr` and a reason. This is what keeps OCR off born-digital files.
2. **Page transcription** for the pages that need it (`parsers/vlm_ocr.py`) —
   concurrent, validated, cached, with EasyOCR as the per-page fallback.
3. **Layout parsing** (Docling): reading order, headings, tables
   (TableFormer ACCURATE), figures, formula/code *regions*.
4. **Formula and code enrichment** (`parsers/formula_enrichment.py`): the
   regions from step 3 are cropped and transcribed, concurrently and cached.
5. **Post-processing** (`docling_parser.py`): list items merged, run
   fragments rejoined, split headings stitched, hyphenation repaired, running
   headers dropped, two-column order repaired when the evidence is
   unambiguous.
6. **OCR merge**: transcript blocks replace the placeholder blocks of the
   pages that were transcribed, in reading order; the document title is
   recomputed, because a scanned first page usually carries the real one.
7. **Table repair** (`parsers/table_repair.py`): only tables that came out
   degenerate are re-read from the page image, and only a repair with *more*
   structure than the original is accepted.
8. **Figure description** (`parsers/enhancer.py`): the largest figures, with a
   budget that scales with the document.

Non-PDF formats skip 1–7: they carry their own structure, so the parser reads
it directly and only step 8 applies.

### 2.1 Structured parsers (`PARSER_PROFILE=structured`, default)

Every parser produces the same `Document`/`Block` contract: typed blocks
(`heading` with `hierarchy_level`, `text`, `list`, `table` as Markdown,
`figure` with `image_bytes`, `code`, `formula`), page/slide/sheet
coordinates and a document title.

| Format | Implementation | Structure recovered |
|---|---|---|
| PDF | Docling (layout model, reading order, tables) + post-processing in `docling_parser.py` | heading levels inferred from numbering (`1.2.3` → level 3), consecutive list items grouped into one list block (bullet glyphs stripped), captions folded into figure/table blocks, headings split over two lines stitched, running headers/footers dropped, words broken by justification rejoined (`обыкновен- ных` → `обыкновенных`: 26 such breaks in one financial report, each one a term the query could not match), **formulas and code re-read by Docling's enrichment models** (`PDF_ENRICHMENT`) so a formula arrives as LaTeX instead of glyph soup, and a **two-column page whose reading order crosses the gutter** is re-sorted column by column |
| DOCX | `python-docx` (`docx_parser.py`), Docling as fallback | whole paragraphs; **Word equations (`m:oMath`) as LaTeX** (a display equation becomes a formula block, an inline one stays inside its sentence); heading levels from `Heading N`/`Заголовок N` styles, outline levels or bold-and-larger formatting; **list numbers replayed from `numbering.xml`** (decimal/letter/roman, multi-level `%1.%2.` templates, per-instance start overrides), tables with merged cells and **tables nested in cells**, **footnotes and endnotes** (marked in the sentence, emitted next to it), **text frames** (their own blocks instead of being concatenated into the host paragraph), embedded pictures with captions, monospace paragraphs as code |
| PPTX | `python-pptx` (`pptx_parser.py`) | slide title → heading (level 1 for section dividers, else 2), body in visual reading order with grouped shapes flattened, bullet lists with indentation, tables and charts as Markdown, pictures with image bytes, speaker notes; slide number stored as `page_number` |
| XLSX | `openpyxl` (`xlsx_parser.py`) | a **workbook overview** (sheet count, sheet names, rows and columns of each) so questions about the file itself are answerable; one heading per sheet, one Markdown table per data region, one- or two-row header detection, merged cells filled, note cells kept as text, `1100.0 → 1100`, ISO dates; a **profile block** per table (row count, column types, min/max with row label, sums/means, distinct values, sums grouped by every low-cardinality column) so aggregate questions are answerable from retrieval. A blank row inside a table no longer starts a new one: a 92-row sheet used to become ten fragments with ten contradictory automatic summaries, and "the maximum rating in the table" was answered from eleven rows |
| HTML | BeautifulSoup walker (`html_parser.py`) | `h1–h6`, paragraphs, nested lists, tables, `pre` code, `img` alt text; nav/header/footer/script/style removed |
| Markdown | `md_parser.py` | ATX/setext headings, fenced code, pipe tables, lists, images, YAML front matter |
| TXT | `txt_parser.py` | encoding detection (UTF-8/16, cp1251, koi8-r, cp866); prose: paragraph reflow of hard-wrapped lines and title detection (`* CAPS *`, standalone short lines); transcripts: timestamps removed, captions re-flowed into ~140-word paragraphs with `time_start` metadata |

The original parsers are kept unchanged in `parsers/legacy/` and selected
with `PARSER_PROFILE=legacy` (for DOCX that means Docling).

#### Did the per-format parsers really replace Docling?

Docling still does the PDFs; for DOCX, PPTX, XLSX and HTML the project parses
the file itself. That trade was measured in both directions on the corpus
(`parser_vs_docling.py`): sentences Docling finds and we do not, and sentences
we find and Docling does not.

| | ours | Docling | sentences only Docling has | sentences only we have |
|---|---|---|---|---|
| 4 DOCX | 2 403 blocks, 600 078 chars | 5 293 items, 599 722 chars | **0 of 240** | 3 of 240 |
| 1 PPTX | 52 blocks, 2 546 chars | 49 items, 2 512 chars | **0 of 26** | 0 of 26 |
| 3 XLSX | 16 blocks, 63 526 chars | 53 items, **78 chars** | 0 (nothing to compare) | 177 of 180 |

Nothing Docling extracts is lost, and three things are gained. Spreadsheets:
Docling's XLSX backend returns table structure with no text at all, so the
data of `inventory-data.xlsx` and `sales-data.xlsx` would not be indexed.
Tables in Word: Docling found 1 table in the MongoDB manual where the OOXML
has 23. Structure: 27 headings against 1 in the philosophy course, and whole
paragraphs instead of formatting runs (17 blocks against 198 for the exam
programme) — which is what the chunker needs to cut on sentence boundaries.

The one thing Docling did better was **Word equations**, and that gap is now
closed (above): its OMML→LaTeX converter is reused, and the placement — inline
equations inside their sentence, display equations as their own block, and
equations inside table cells, which `cell.text` never returns — is ours.

### 2.2 OCR and VLM through the chat model (`VLM_BACKEND=llm`, `OCR_ENGINE=auto`)

The answering model served for the project (Qwen3.5-27B on vLLM) is
multimodal, so by default the **same endpoint** is used for vision:

- **Scanned pages.** `parsers/routing.py` still decides per page (locally,
  no network) whether a text layer is missing. Those pages are rendered at
  150 dpi and transcribed to Markdown by `parsers/vlm_ocr.py` with a strict
  "transcribe, do not interpret" prompt (headings, lists, tables, LaTeX
  formulas kept; hyphenated words rejoined; page furniture skipped). The
  transcript replaces Docling's placeholder blocks for those pages, in
  reading order. Pages are transcribed **concurrently**
  (`VLM_OCR_CONCURRENCY`, default 4), blank pages never reach the model, and
  a page the model reports as empty is left to Docling so its picture can
  still be described. A validated transcript is **cached by the pixels of the
  render**: a scanned deck costs its 16 minutes once, and — because the model
  is not deterministic under batching — a re-ingestion now produces the *same*
  text instead of a fresh sample, which is what makes two evaluation runs
  comparable at all.
- **Validation and fallback (`OCR_ENGINE=auto`).** A generative transcriber
  fails in ways a classic engine cannot, so every transcript is checked:
  empty output on a page full of ink, a repetition loop, a refusal, foreign
  script, and output far shorter than the page's text lines imply. A page
  that fails is retried with double the token budget (also when the endpoint
  reports `finish_reason=length`) and then handed to EasyOCR. The result is
  therefore never worse than classic OCR. `OCR_ENGINE=vlm` is the same path
  without the local safety net, `easyocr`/`rapidocr` force the classic
  engines inside Docling, `off` disables OCR.
- **Figures.** `parsers/enhancer.py` describes the largest figures of a
  document — PDF page crops, and the embedded images of DOCX/PPTX — and folds
  the description into the block text so it is indexed. The budget scales
  with the document (a quarter of its page count, at least 8 and at most 32;
  `VLM_MAX_FIGURES` pins a fixed number): eight descriptions cover a report
  but only a tenth of an eighty-slide deck whose charts *are* the content.
  One failed call aborts the remaining figures of that document instead of
  paying a timeout each.

`VLM_BACKEND=openai` (separate vision endpoint), `smolvlm` (local) and `off`
remain available.

#### Formulas and code (`PDF_ENRICHMENT`)

A formula has no text layer worth reading: Docling emitted **zero** formula
blocks for the 85-page probability lecture of the corpus, and the glyphs that
did survive landed in the surrounding paragraph as noise. With enrichment on,
the same document yields 378 formula blocks, 374 of them proper LaTeX
(`P ( A ) \colon = \sum _ { \omega \in A } P ( \omega ) .`) and 31 % more
indexed text.

It is a vision model per region, and the cost follows the number of formulas,
not the number of pages:

| Document | enrichment off | enrichment on |
|---|---|---|
| 85-page lecture with formulas | 19 s, 0 formulas | 523 s, 378 formulas (374 LaTeX) |
| 26-page English paper | 21 s, 3 code blocks | 37 s, same content |

On a CPU the same pass takes tens of minutes, which is why the default is
`auto`: enrichment runs when a CUDA device is visible and is skipped
otherwise. `on` / `off` force it, and a conversion that cannot load the models
repeats itself without them rather than failing.

##### Where those 523 seconds went

Docling's own stage timings put **503 s of the 523 in the enrichment model** —
layout, tables and page parsing together are 17 s, and two consecutive runs
differ by 1 s. The document is not slow; transcribing 378 formulas five at a
time at 1.33 s each is. The model already runs in bfloat16 (its model spec
says so), 8-bit quantization is off and there is no flash-attention build
here, so inside Docling only the batch is left to turn:

| Setting | model time | per formula | output |
|---|---|---|---|
| shipped (5 regions per pass) | 503 s | 1331 ms | reference |
| repeat of the same run | 504 s | 1333 ms | identical, 378/378 |
| 16 regions per pass | 358 s | 947 ms | 355/378 identical; of the 23 that differ, 15 are crops the model already failed to read, 6 are cosmetic (`b_n^{\prime}` against `b^{\prime}_n`), one is worse |
| 32 regions per pass | 47 s | — | **all 378 formulas empty**: CUDA OOM inside the stage |

`PDF_ENRICHMENT_BATCH` exposes the batch (unset = Docling's 5). The last row
is the trap worth naming: Docling catches every exception inside the stage,
an out-of-memory included, and returns empty text for the whole batch — so
the document parses *faster* than with enrichment off and arrives without a
single formula. The parser now counts the regions that came back empty and
warns when most of them did.

##### What replaced it: the serving model, concurrently
(`PDF_ENRICHMENT_ENGINE`, default `auto`)

A batch of five is the wrong axis. The multimodal model that already answers
the project's questions runs on vLLM, which is built to serve many requests at
once, so `parsers/formula_enrichment.py` crops each region from the page and
transcribes them **in parallel** (`PDF_ENRICHMENT_CONCURRENCY`, default 8).
Measured on the same 85-page lecture, same 378 regions:

| Path | enrichment | whole document | per formula |
|---|---|---|---|
| Docling CodeFormulaV2, 5 per pass | 503 s | 523 s | 1331 ms |
| **serving Qwen3.5-27B, 8 in flight** | **184 s** | **202 s** | **486 ms** |
| serving Qwen3.5-27B, 16 in flight | 135 s | 154 s | 357 ms |
| the same document again, from the cache | 3 s | 21 s | 7 ms |

**2.6× on the document, and the output is better.** Both transcriptions of all
378 regions were compared by a vision judge that sees the crop and both
candidates, with the sides alternated, on a random sample of 120 regions where
the two disagree: **73 for the serving model, 35 for CodeFormulaV2, 12 ties**.
Two counts do not depend on a judge at all: CodeFormulaV2 emits undefined
control sequences for the Russian words inside formulas (`\i a p { \i }` for
"пары") in 7 transcripts and never produces a Cyrillic letter, while the
serving model produces none of that junk and keeps the Russian words in 11.
The judge is the same model family as one of the candidates, so its preference
is reported next to those counts, not instead of them.

The other properties are the ones a generative transcriber needs:

- **Validation.** Refusals, descriptions of the image ("The image shows…"),
  decoding loops, output cut off mid-formula and answers longer than any
  formula are rejected; a suspect answer is retried once with twice the
  budget, and a region that still fails is left empty rather than filled with
  an invention. On the lecture: 371 clean reads, 5 kept after a retry, 2
  regions too small to be worth a request, 0 refusals or loops.
- **A cache keyed by the pixels of the crop**, so re-ingesting a corpus — the
  normal case while tuning retrieval — costs 3 s instead of 184 s, and
  identical crops inside one document are transcribed once.
- **No GPU needed on the ingesting machine.** This is why `auto` prefers it:
  on a laptop Docling's enrichment is minutes per document, so formulas used
  to be a server-only feature.
- **The endpoint costs one request to fail, not one per region**: the first
  crop is sent alone, and if it fails the rest are skipped.

`PDF_ENRICHMENT_ENGINE=docling` keeps the local model, `off` disables
enrichment, and `PDF_ENRICHMENT=off` still wins over both. When the local
model does run, its batch is sized from the memory actually free on the card
(`PDF_ENRICHMENT_BATCH=auto`): ~8 GB next to vLLM gives 16, a busy card falls
back to Docling's 5. That is not a tuning knob but a safety one — the stage
hides an out-of-memory and answers with empty formulas.

##### Where the formulas are indexed (`FORMULA_INDEXING`, default `inline`)

Adding 378 formulas to the lecture also moves it from 236 chunks to 336, and
a control run with enrichment off scored 0.015 higher answer correctness —
which looked like the formulas diluting the prose. That number sits inside the
±0.016 the judge moves on its own, so the question was settled at the
retrieval level instead, with no model in the loop: two indexes built from the
*same* parse, 200 prose queries (the opening words of a real sentence; a hit
is the passage containing it) and 60 formula queries (the LaTeX of a real
formula).

| Formulas | chunks | prose hit@1 | prose hit@5 | formula hit@1 | formula hit@5 |
|---|---|---|---|---|---|
| **embedded (`inline`)** | 336 | **0.980** | **1.000** | **0.917** | **1.000** |
| in the parent only (`context`) | 236 | 0.965 | 0.995 | 0.300 | 0.567 |

The dilution never existed: taking the formulas out of the embedded text made
prose retrieval slightly *worse* (fewer, larger chunks are less precise), and
it cost formula retrieval almost everything — a passage carrying the formula
a question needs is missing from the top five in four cases out of ten. The
default is therefore to index formulas; `FORMULA_INDEXING=context` remains for
a corpus where nothing is ever asked about a formula and the smaller index is
worth it. Inline equations inside a DOCX sentence are part of the sentence and
are always indexed with it.

Two smaller costs were measured next to it. Reusing one Docling converter
across the documents of a run instead of building one per file saves ~2.4 s
per document (34.6 s against 27.5 s over three PDFs) — that is what the
`+16 s` on the formula-free paper mostly is, model setup rather than work.
And nothing is gained by de-duplicating crops *within* this document: all 378
formulas of the lecture are distinct.

#### Why the model reads scans better than an OCR engine

The question "is a VLM really better than EasyOCR here, in quality *and*
cost?" was measured rather than assumed (`ocr_bench.py`, reproduced in
§3.4): pages that carry a real text layer are rendered to images, degraded
to imitate scanning, transcribed by both engines and compared to the text
layer, which is exact ground truth.

| Condition | CER, VLM | CER, EasyOCR | WER, VLM | WER, EasyOCR | s/page, VLM | s/page, EasyOCR |
|---|---|---|---|---|---|---|
| clean render (150 dpi) | **0.122** | 0.285 | **0.175** | 0.479 | 33.0 | 2.0 |
| scanner-like (0.6° skew, JPEG 55, blur, noise) | **0.107** | 0.305 | **0.178** | 0.520 | 32.7 | 1.9 |
| photo-like (2° skew, ~100 dpi, JPEG 35, uneven light) | **0.113** | 0.562 | **0.181** | 0.829 | 32.9 | 1.9 |

Read per page, the gap is wider than the averages suggest: on Russian prose
0.08 against 0.54, on a financial table 0.06 against 0.32, on medical prose
0.002 against 0.10. Two findings matter more than the averages:

- **The model is stable under degradation and the engine is not.** EasyOCR
  doubles its error rate between a clean render and a photo-like scan
  (0.285 → 0.562); the VLM does not move (0.122 → 0.113). Skew and low
  contrast are exactly what a real scan has.
- **Only the VLM preserves structure.** EasyOCR returns lines of text; the
  transcript keeps headings, lists and Markdown tables, which is what the
  chunker and the answering prompt work with. On the two financial-table
  pages the VLM's word error rate is 0.07 and 0.26 against 0.60 and 0.81.

The one page where EasyOCR scores better (a lecture with probability
formulas: CER 0.21 against 0.35) is a measurement artefact: the model writes
formulas as LaTeX (`$\mathrm{P}(A \mid B)=\mathrm{P}(A)$`) while the text
layer holds them as plain glyphs, so a *better* transcript scores as a worse
one.

The real cost is latency: ~33 s per page against ~2 s. That is why pages are
transcribed concurrently (a 29-page deck takes ~2.5 min, not 16), why only
pages without a text layer are sent at all, and why the classic engine
remains the fallback rather than being removed.

#### What the two riskiest repairs actually do to this corpus

A repair that rewrites reading order, and one that invents numbers that are
nowhere in the file, can both *lose* information if they misfire. Both were
replayed over the corpus rather than argued about (`column_audit.py`,
`numbering_audit.py`).

**Column repair fires on 0 of 393 PDF pages.** 201 pages are rejected because
a full-width block proves the page is not a clean two-column layout, 191
because they hold fewer than six blocks, and one page of `Agentic Memory.pdf`
because it alternates between the halves three times where four are required.
The corpus contains two genuine two-column arXiv papers, and Docling's layout
model orders both correctly — which is the point: the repair is a safety net
for the case where the model fails, and on documents where it does not fail
the guard never triggers. Its behaviour when it *is* needed is pinned by
`tests/test_docling_postprocess.py`.

**Automatic numbering matches what the file declares.** Across the four DOCX
of the corpus 350 paragraphs carry a `numId`; the counters reproduce the
exam programme as 1…N (the dataset asks about "пункт 26"), and the two lists
of the philosophy course that start at 2 and 6 do so because the document's
own `w:start` says 2 and 6 — not because a number was lost. No item came out
with a doubled number.

The audit did find one real defect, since fixed: Word advances a counter for
*every* paragraph carrying the `numId`, including the numbered section
headings a list is nested under, while the parser only advanced it for
paragraphs it emitted as list items. In the lab manual of the corpus that
numbered the second section's sub-items "1.1…1.5" where the document shows
"2.1…2.5". The counter now runs in document order before the paragraph is
classified, numbered headings carry their number ("2. Теоретическое
обоснование", as the document displays it), and a number already typed into
the text is not repeated.

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
  items, code on lines, tables on rows with caption + header repeated;
  pieces never consist of a heading alone.
- **Column names survive a wide table.** A financial statement has fourteen
  columns whose names are sentences: the header cannot be repeated on every
  piece, and without it the continuation pieces are rows of numbers with
  nothing to say which column each belongs to (22 such chunks in the report of
  the corpus). The names are abbreviated instead — 18, 12, 8 or 6 characters
  each, whichever fits the budget, numbered apart when two shorten to the same
  string — and repeated on every piece.
- **A heading with nothing under it is not a chunk.** The same report repeats
  its company name as a section header on every page, which produced identical
  contentless chunks competing with real ones. They are dropped; the heading
  still travels in the breadcrumb of the sections below it. (A document that
  is *only* headings falls back to indexing them, so an outline is not lost.)
- **Parent windows.** A piece of a long section carries in
  `metadata["context"]` a window of the section *around the piece* (not its
  first 4000 characters), opened by the heading path — that is what the LLM
  reads.
- **Budget.** With the tokenizer of the embedding model, chunks target
  `CHUNK_TARGET_TOKENS` (384) capped by the encoder window; sizes are cached;
  the character-window fallback always advances by at least half a window
  (this removes the multi-hour pathological case above).
- **Row records for tables (`TABLE_ROW_RECORDS`, on).** A table is indexed
  twice: as row *windows* (what the table looks like) and as one record per
  row, rendered `Column: value; Column: value`. A window answers "show me
  this part of the table"; it does not answer "which row has FIDE 1260",
  because the query value and the answer sit in different columns of one row
  and fifteen rows of digits dilute both. The record puts them side by side,
  which is what BM25 and the encoder can match, while `metadata["context"]`
  still hands the LLM the surrounding table. Bounded to tables of 4…600 rows,
  ≤40 columns and short cells, so prose tables and huge exports keep the
  window representation only.
- **Topic-boundary splitting (`CHUNK_SEMANTIC_SPLIT`, off).** For sections
  three times over the budget with no internal headings (transcripts,
  lecture notes), sentences are embedded and the cuts are placed where
  neighbouring sentences are least similar instead of at the budget. Opt-in:
  it costs an encoder pass over the section.

`CHUNKING_STRATEGY=legacy` runs the original chunker (`chunking_legacy.py`).

#### How the chunker actually walks a document

The order below is the whole algorithm; the subtleties are in the conditions,
not in the shape.

**1. Sections.** Blocks are grouped into *leaf* sections: a heading opens one,
everything until the next heading of the same or a shallower level belongs to
it, and the headings above it are remembered as its `path`. A document without
headings is one section; a document that is only headings still produces
chunks (there is an explicit fallback for it).

**2. Packing.** Sections are accumulated into a buffer while they fit the
budget. The buffer is flushed when adding the next section would exceed the
limit, when it is already at least a third of the budget (`minimum`), or when
the next section comes from a different branch of the outline and the buffer
is at least a sixth of the budget. The last rule is what stops two unrelated
subsections from sharing a chunk just because both are short.

**3. Oversized sections** are packed block by block instead. A table or any
block larger than the limit is emitted on its own (and split by its own type
rules); everything else accumulates until the limit. Each piece keeps a
*parent window* — the section text around it, grown in both directions up to
4000 characters, opened by the heading path.

**4. Type-specific splitting.**

| Block | Cut on | Kept together |
|---|---|---|
| prose | sentence boundaries (`т.е.`, `рис.`, `e.g.` protected) | sentence overlap between neighbours |
| list | items | the marker with its item |
| code | lines | blank lines |
| table | rows | caption + header repeated on every piece; a row too wide for one piece is cut on cell boundaries |

**5. The breadcrumb.** Every chunk opens with `Doc title > Chapter > Section`,
capped at 20 % of the budget: the deepest crumbs are dropped first, each crumb
is shortened to 80 characters, and the crumb is *not* repeated when the chunk
body already starts with that heading. It is prepended to the text (so both
BM25 and the encoder see it) and stored in `metadata["heading_path"]`.

**6. Emission guarantees.** `_emit` is the only place a chunk is created, and
it enforces four things: a chunk never exceeds the encoder budget (a final
window pass re-splits anything that slipped through, keeping the breadcrumb on
every window); a piece with no letter or digit is never emitted; the parent
passage is attached only when it is genuinely larger than the piece itself;
and the id ties the chunk back to the block it came from
(`block-3cce0d0c-chunk-21`).

**7. Extra representations.** After the windows, tables are indexed a second
and third time — one record per row, and one automatic summary per table
(§2.3 above). They carry `representation` = `row` / `profile` in metadata, so
retrieval evaluation and the UI can tell them apart from window chunks.

#### The budget is counted in the encoder's own tokens

With a tokenizer (the default path) the limit is
`min(model_max_length, CHUNK_TARGET_TOKENS=384)`, the minimum is a third of
it, and the overlap keeps the caller's *ratio* rather than its absolute value
(100 of 1000 characters → 10 % of the token budget). Sizes are memoised, and
the cache is cleared past 4096 entries so a large corpus cannot grow it
without bound.

Without a tokenizer the same code counts characters, and the character-window
fallback always advances by at least half a window. That is not a detail: the
original chunker could step one character at a time on a table row denser in
tokens than the window, which is the failure that made one DOCX unprocessable
in the baseline run.

#### What a chunk carries

```python
Chunk(
    id="block-3cce0d0c-chunk-21",
    text="Doc > Chapter > Section\n\n…the passage…",
    metadata={
        "source_file": "Конспект ОВиТМ.pdf", "file_type": "pdf",
        "block_ids": [...], "block_type": "text", "block_types": ["text"],
        "doc_title": "…", "section": "…", "sections": [...],
        "heading_path": ["…", "…"],
        "page_number": 12, "page_numbers": [12, 13], "bbox": (...),
        "context": "…the parent passage the model reads…",
        "representation": "row",          # row records / profiles only
        "row_index": 7, "table_profile": True,
        "caption": "…", "note_type": "footnote", "enrichment": "vlm",
    },
)
```

Per-block parser internals (`docling_label`, `hierarchy_level`, `style`,
`figure_index`, …) are deliberately dropped at this boundary: they are useful
while parsing and only noise in a prompt.

### 2.3a Caches, determinism and what they cost

Two passes in the pipeline call a generative model, and both are now cached by
**the pixels they read** (SHA-1 of the rendered PNG plus the prompt version):

| Cache | Key | Cold | Warm | Invalidated by |
|---|---|---|---|---|
| page transcripts (`vlm_ocr.py`) | page render at `OCR_DPI` | ~33 s per page | 0 | `OCR_PROMPT_VERSION` |
| formula/code crops (`formula_enrichment.py`) | region crop at `PDF_ENRICHMENT_DPI` | 486 ms per region | 0 | `PROMPT_VERSION` |

Both live under `~/.cache/file_agent/` by default and are disabled with
`PDF_ENRICHMENT_CACHE=off`. Only *validated* output is stored, so a rejected
transcript is never served back.

The point is not only speed. A model on vLLM is not deterministic under
batching, so before the cache the same scanned deck produced slightly
different text on every ingestion — which moved the five questions about it by
±0.4 between runs of *identical code* and made small differences impossible to
measure. With the cache, a re-ingestion produces the same document, and the
only remaining variance in an evaluation is the judge's own.

### 2.3b Failure modes and the guard that catches each

Every one of these was observed on the project's own corpus, not imagined:

| Failure | Where it shows | Guard |
|---|---|---|
| The endpoint is down / the model is text-only | every page or region would time out in turn | the first page and the first crop are sent alone; on failure the rest are skipped and the classic engine takes over |
| The model describes the image instead of transcribing it | "На изображении представлена формула…" indexed as a formula | prose/refusal detection, retry, then the region is left empty |
| A decoding loop | one line repeated to the token limit | repetition check on lines and on short fragments |
| Output cut off mid-formula | unbalanced braces | retry with double the budget; the better of the two answers wins |
| Docling swallows a CUDA OOM in enrichment | the document parses *faster* and arrives with **no** formulas | empty-region counter + warning; the batch is sized from free GPU memory |
| A wide table loses its header | rows of numbers with no column names | abbreviated header repeated on every piece |
| A running header becomes a section | identical contentless chunks | a body-less heading is dropped only when the document repeats it |
| A blank page invites invention | a page of nothing gets "text" | ink ratio below 5·10⁻⁵ → the page never reaches the model |
| python-docx has no numbering part | `NotImplementedError` aborts the parse | numbering is optional; without it items are plain bullets |
| A document with thousands of formula regions | an hour of ingestion | `PDF_ENRICHMENT_MAX_REGIONS` (1500) with a warning naming what was skipped |

### 2.3c Configuration reference (ingestion)

| Variable | Default | Effect |
|---|---|---|
| `PARSER_PROFILE` | `structured` | format-aware parsers, or `legacy` for the original flat ones |
| `CHUNKING_STRATEGY` | `structured` | section packing with breadcrumbs, or `legacy` |
| `CHUNK_TARGET_TOKENS` | `384` | token budget per chunk, capped by the encoder window |
| `TABLE_ROW_RECORDS` | `true` | index every table row as `Column: value` as well |
| `CHUNK_SEMANTIC_SPLIT` | `false` | cut long unstructured prose at topic boundaries |
| `FORMULA_INDEXING` | `inline` | embed standalone formulas; `context` shows them only through the parent |
| `OCR_ENGINE` | `auto` | validated model transcription with an EasyOCR fallback; `vlm`, `easyocr`, `rapidocr`, `off` |
| `OCR_DPI` / `OCR_LANGS` | `150` / `ru,en` | page render resolution, classic-engine languages |
| `VLM_OCR_CONCURRENCY` | `4` | page transcriptions in flight |
| `PDF_ENRICHMENT` | `auto` | read formula/code regions at all |
| `PDF_ENRICHMENT_ENGINE` | `auto` | the serving model when one is configured, else Docling's CodeFormulaV2 |
| `PDF_ENRICHMENT_CONCURRENCY` | `8` | formula requests in flight (16 is ~27 % faster) |
| `PDF_ENRICHMENT_DPI` | `200` | crop resolution for those requests |
| `PDF_ENRICHMENT_BATCH` | `auto` | Docling-engine batch, sized from free GPU memory |
| `PDF_ENRICHMENT_MAX_REGIONS` | `1500` | ingestion budget per document |
| `PDF_ENRICHMENT_CACHE` | `~/.cache/file_agent/formula_enrichment` | transcript cache, `off` disables |
| `PDF_TABLE_VLM` | `auto` | re-read degenerate tables from the page image |
| `DOCLING_PDF_BACKEND` | auto | `docling` (default, better cells) or `pypdfium` (Windows fallback) |
| `VLM_BACKEND` / `VLM_MAX_FIGURES` | `llm` / scaled | figure description backend and budget |

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
- Single-document search (`search(query, top_k, source_file=…)`): the file name
  is indexed as its own LanceDB column — a filter cannot look inside the JSON
  the rest of the metadata is stored as — and the predicate runs as a
  *prefilter*, so the top-k is filled from that document instead of being
  filtered down to whatever survives. Diversification switches itself off for
  such a query: there is only one document to diversify over. This is the entry
  point an agent tool needs ("what does *this* file say about X?"), which over a
  corpus of twenty files is otherwise answered from whichever file scores
  highest.
- The QA prompt (`QA_PROMPT=v3`, default) shows every passage under a compact
  header (`source_file`, pages, heading path) instead of raw retrieval
  metadata and asks for a complete, grounded answer, with an explicit
  conclusion on comparison / "does the document mention" questions.
  `v2` is the same prompt with a trailing `Источники: file, page` footer,
  and `v1` is the original short prompt.

  The footer was the default until it was measured. It is a claim about
  document metadata, and no retrieved passage supports it, so the judge
  counts it as unsupported: in the v4 run 126 of 127 answers carried it and
  averaged 0.845 faithfulness, while the single answer without it scored
  1.0. Worse, `answer_correctness` runs in recall mode over claims — an
  answer that repeated the reference word for word still scored 0 when the
  only claims it added (file name, page number) were absent from the
  reference. All 28 zero scores in the run were answers with a footer.
  Provenance did not need it: the passage headers carry file, page and
  heading path into the prompt, and the UI shows the source chunks.

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
| `pc-baseline-rag-127` | Code before this branch: Docling for PDF/DOCX with the pypdfium backend, flat PPTX/XLSX/HTML/TXT parsers, original chunker, `paraphrase-multilingual-MiniLM-L12-v2` (128-token window), original prompt. 5 rows could not be processed (the original chunker never finishes on `Курс лекций Основы философии.docx`; recorded as failures after a 240 s timeout). |
| `pc-structured-rag-127` (v2) | Structured parsers + structured chunker + `bge-m3` + VLM figure descriptions and VLM page OCR through the chat model + QA prompt v2. |
| `pc-structured-v4-127` (v4) | v2 + Docling `docling-parse` backend with ACCURATE TableFormer (row labels of financial tables recovered) + cross-encoder reranking (`bge-reranker-v2-m3` over the top-20 hybrid hits) + document-diverse top-k + spreadsheet profile blocks + prompt rule for comparison/"does it mention" questions. |
| `pc-structured-v4-norerank-127` | v4 without the reranker (ablation). Note: only one ingestion process fits next to vLLM on the shared A100 (~8 GB free); a second concurrent generation run fails with CUDA OOM. |
| `pc-structured-v4-agent-127` | v4 ingestion answered by the multi-step agent. Measured from the separate `feat/agent` branch: this branch is the shared baseline and carries no agent, so the row is kept as evidence about the ingestion, not as a feature of it. |
| `pc-v4-nocite-127` | The v4 answers with the `Источники: …` footer stripped by a regular expression — identical answers, identical contexts, so the difference isolates what the footer costs in the judge's eyes. |
| `pc-v5-rag-127` (v5, **default configuration**) | v4 + OCR `auto` (validated VLM transcripts, concurrent, EasyOCR per-page fallback) + spreadsheet fragment merging and workbook overview + table row records + hyphenation repair + document-scaled figure budget + QA prompt v3 (no source footer). |
| `pc-v6-concise-127` | v5 + QA prompt v4 (answer the question and nothing beside it) and the late fixes: document title no longer taken from "Оглавление", VLM repair of degenerate PDF tables, automatic profiles for tables outside spreadsheets. |
| `pc-v7-topk8-127` | v5 (default prompt) with `--top-k 8`: eight passages per question instead of five, for the multi-document questions whose context recall is the lowest of the set. |
| `pc-v5-rejudge` | The v5 run judged a second time, unchanged, to measure how much of a difference between runs is the judge's own variance. |
| `pc-v8-parsing3-127` | v5 defaults plus the third round of parsing work: formula/code enrichment for PDF, DOCX footnotes, text frames, nested tables and real list numbers, HTML merged cells, and the two-column reading-order repair. (Hyperlink targets landed after the run started and are the one item it does not cover.) |
| `pc-v9-vlmenrich-127` | v8 with formulas read by the serving model instead of Docling's, plus the chunking fixes the ingestion audit produced (abbreviated headers on wide tables, no contentless chunks, summaries trimmed to one chunk). |
| `pc-v10-enrich-fixes-127` | v9 with the two corrections its own numbers demanded: an empty heading is dropped only when the document *repeats* it, and enriched formulas are stored without their typesetting macros. |
| `pc-v11-noenrich-127` | v10 with `PDF_ENRICHMENT=off` — the control that prices what the formulas cost this dataset. |
| `pc-v12-formula-context-127` (**current default**) | v10 with formulas kept out of the embedded chunk text (`FORMULA_INDEXING=context`), Word equations read as LaTeX, and the enrichment batch sized from free GPU memory. |
| `pc-v13-final-127` | The same code as v12, run a second time (and populating the new page-OCR cache): the pair measures how much this table moves when nothing changes. |
| `pc-v14-inline-127` (**current default**) | v12 with `FORMULA_INDEXING=inline`, the value the retrieval A/B chose. Confirms the final configuration end to end; the page-OCR cache makes its scanned-deck answers identical to v13's. |

### 3.1a Auditing the ingestion itself (`audit_ingestion.py`)

The judged runs measure answers; they say nothing about a chunk that quietly
lost its table header. `audit_ingestion.py` parses and chunks a folder
and checks structural invariants — empty blocks, chunks over the encoder
budget, table pieces without a header, contentless or duplicated chunks, lost
page numbers, mojibake — and with `--coverage N` samples sentences from an
*independent* read of each file (PyMuPDF text layer, python-docx paragraphs,
openpyxl cells) and verifies they are still findable in the chunks.

Run over the project's corpus it found the two defects §2.3 describes, and
after the fixes:

| Rule | before | after |
|---|---|---|
| table chunk without a header | 51 | 14 (regions that have no header row in the source) |
| chunk with no content of its own | 2 | 0 |
| duplicated chunk | 11 | 9 (rows repeated verbatim in the source table) |
| automatic summaries split over several chunks (the report) | 69 pieces for 34 tables | 14 |

Content survival, with OCR and figure description switched off so that only
parsing and chunking are measured
(`VLM_BACKEND=off OCR_ENGINE=off PDF_ENRICHMENT=off python
audit_ingestion.py ~/testdocs --coverage 40`): **640 of 651 sampled
sentences**, and the losses are all of two kinds — a table-of-contents line
with dot leaders (`1.3 Эпидемиология … 8`), and text that lives inside a
figure and therefore reaches the index through the description pass this run
disables. DOCX, Markdown, TXT, PPTX and XLSX are at 100 %; the PDFs lose those
eleven sentences between them, five of them in the scanned deck whose pages
this run does not OCR.

### 3.2 Results (ragas, judge = Qwen3.5-27B without thinking; pipeline failures scored 0)

| Run | rows | failed | faithfulness | answer_correctness | answer_relevancy | context_precision | context_recall | avg s/question |
|---|---|---|---|---|---|---|---|---|
| baseline (old code) | 127 | 5 | 0.686 | 0.412 | 0.538 | 0.548 | 0.605 | — |
| structured v2 | 127 | 0 | 0.847 | 0.593 | 0.738 | 0.715 | 0.786 | 22.5 |
| structured v4 | 127 | 0 | 0.840 | 0.601 | 0.831 | 0.767 | 0.811 | 26.2 |
| v4 + agent mode (from `feat/agent`) | 127 | 0 | 0.751 | 0.537 | 0.861 | 0.768 | 0.824 | 36.4 |
| v4 without reranker | 127 | 0 | 0.837 | 0.604 | 0.855 | 0.705 | 0.827 | 25.0 |
| v4, source footer stripped | 127 | 0 | 0.945 | 0.603 | 0.830 | 0.760 | 0.814 | — |
| **v5 (default)** | 127 | 0 | **0.958** | 0.605 | 0.831 | 0.777 | **0.865** | **18.5** |
| v5, judged a second time | 127 | 0 | 0.956 | 0.616 | 0.823 | 0.763 | 0.861 | — |
| v5 + concise prompt (v4) | 127 | 0 | 0.941 | 0.554 | 0.823 | 0.772 | 0.854 | 10.8 |
| v5 + top-k 8 | 127 | 0 | 0.962 | 0.598 | 0.846 | 0.744 | 0.875 | 14.1 |
| v8 = v5 + parsing round 3 | 127 | 0 | 0.950 | 0.597 | 0.822 | 0.767 | 0.847 | 18.3 |
| v9 = v8 + VLM enrichment + chunking fixes | 127 | 0 | 0.959 | 0.565 | 0.824 | 0.752 | 0.852 | 14.9 |
| v10 = v9 + the two corrections | 127 | 0 | 0.954 | 0.577 | 0.823 | 0.762 | 0.843 | 15.1 |
| v11 = v10 with enrichment off (control) | 127 | 0 | 0.965 | 0.592 | 0.823 | 0.762 | 0.852 | 13.3 |
| v12 = v10 + formulas in context + Word equations | 127 | 0 | 0.958 | 0.584 | 0.832 | 0.755 | 0.844 | 13.3 |
| v13 = v12 + OCR page cache (same code, second sample) | 127 | 0 | 0.952 | 0.568 | 0.824 | 0.751 | 0.843 | 13.4 |
| **v14 = v12 with formulas indexed (current default)** | 127 | 0 | 0.957 | 0.578 | 0.823 | 0.754 | 0.848 | **12.4** |
| v15 = v14 + retrieval branch (lemmatised BM25, distinct passages, HyDE multi-query; see [retrieval.md](retrieval.md)) | 127 | 0 | 0.956 | 0.585 | 0.844 | 0.752 | 0.876 | 13.3 |

Means over successfully processed rows only differ for the baseline (0.715 /
0.429 / 0.560 / 0.570 / 0.630 over 122 rows).

**The measurement floor of this table is ±0.016 on `answer_correctness`,
measured.** v12 and v13 run the *same code* on the *same questions*: 118 of
127 answers come out byte-identical, and the mean still moves 0.584 → 0.568.
On the identical-answer rows alone it moves 0.580 → 0.566, so that part is the
judge re-reading the same text in a new session; the nine answers that differ
are the scanned A/B deck, whose 29 pages are transcribed afresh by a model
that is not deterministic under batching. Any difference in this table smaller
than that is not a result. (Both sources are now closed: page transcripts are
cached by their pixels from v13 on, exactly like formula crops, so a
re-ingestion produces the same text — the remaining variance is the judge's.)

Against that floor, **the current default (v14) is level with v5, the
configuration it replaces**: −0.027 correctness, −0.008 relevancy, −0.017
recall — and on the answers that are byte-identical between runs the new
pipeline scores *higher* (v12 against v5: 0.628 → 0.648). What it adds is
content v5 never indexed — formulas, footnotes, equations from Word, real list
numbers, column names on wide tables — at **12.4 s per question against 18.5**
(−33 %) and 2.6× faster ingestion of a formula-dense document.

The `inline`/`context` pair is the clearest illustration of the floor: v12 and
v13 (`context`) scored 0.584 and 0.568, v14 (`inline`) 0.578 — the judge
cannot separate them, while the retrieval measurement separates them cleanly
(formula hit@5 1.000 against 0.567). When a design question is about
*retrieval*, measure retrieval.

**v10 against v8, read properly.** 108 of the 127 answers are *byte-identical*
to v8's, and on those rows the judge gives 0.607 → 0.605 — that is the
measurement floor. Of the 19 answers that did change, two carry the whole
difference: `q0029` and `q0030` ("does the lecture state a minimum sample
size / give a formula for the confidence interval?") went 1.00 → 0.00 while
saying the same thing, the new answer merely opening with "Нет," before the
sentence v8 scored 1.00 for. Both keep faithfulness 1.00 and context recall
1.00 in both runs. Two rows are 0.016 of a 127-row mean — about four fifths of
the −0.020 gap — and faithfulness on the same 19 changed rows *rose*, 0.908 →
0.950. Those two questions belong to `AB_test.pdf`, whose text comes from 29
model-transcribed pages that differ between runs, so the wording change is not
something the parser chose.

Read together with §2.2: the accuracy of the answers is unchanged, ingestion
of a formula-dense document is 2.6× faster, a question is answered 3.2 s
faster, and the formulas themselves are measurably better transcribed. The
first attempt (v9) is kept in the table because it is where the two
corrections came from: dropping *every* empty heading cost Exam.md four chunks
and the philosophy course two, and each lost a question with them.

**`answer_correctness` is the noisy column of this table and must not be read
alone.** The judge measures it as ragas `FactualCorrectness` in recall mode:
the answer is decomposed into claims and each is checked against the
reference. Two properties of that protocol dominate the number:

- A *more complete* answer scores lower. Of eight v5 rows scoring exactly 0,
  six were factually right and matched the reference — they added a fact that
  the terse reference does not mention (the segment breakdown next to the
  total revenue, childhood incidence next to adult incidence), and every
  added claim counts against the answer.
- A byte-identical answer can score differently in different judge runs.
  `q0064` has exactly the same 846 characters in the footer-stripped v4 run
  and in v5; it scored 1.00 in the first judging and 0.00 in the second.
  Judging the *same* run twice (`pc-v5-rejudge`) shows the effect is small in
  aggregate — mean per-row difference 0.017 for correctness, 0.008–0.015 for
  the other metrics, and no row moved by more than 0.5 — but between the v4
  and the footer-stripped v4 judging four rows of 127 (3 %) flipped between
  0 and 1. Single rows are therefore not evidence; only differences well
  above ±0.02 in the mean are.

Faithfulness, context precision and context recall are stable and are what
the v4 → v5 comparison rests on.

The `pc-v4-nocite-127` row is a controlled experiment that separates the two
kinds of change, because it holds the pipeline fixed and edits only the answer
text:

- **The source footer alone cost 0.105 faithfulness** (0.840 → 0.945 with the
  same answers, same retrieval, same contexts).
- **The judge is stable on the context metrics**: with byte-identical
  contexts, precision and recall moved by 0.007 and 0.003 — that is the noise
  floor for those columns, so v5's +0.054 recall is a real effect of the
  ingestion changes, not measurement drift.
- **The footer was not what produced the zero correctness scores**: they are
  still there without it (0.601 → 0.603).

**The third round of parsing work is metric-neutral on this corpus, and that
is the expected result.** Formula/code enrichment, DOCX footnotes, text
frames, nested tables and real list numbers, HTML merged cells and the
column repair recover content that would otherwise be silently missing — 378
formulas in one lecture, 9 footnotes, 4 text frames, the item numbers of the
exam programme — but these 127 questions do not ask about that content, and
the corpus contains no HTML at all. Per document, 26 of the 29 groups score
*exactly* the same as in v5; the three that moved all involve `AB_test.pdf`,
the only file whose text comes from 29 VLM-transcribed pages, and those
transcripts differ between runs (the model is not deterministic under
batching). The changes are kept because losing a document's formulas or
footnotes is a correctness bug that this dataset simply does not measure.

Two settings were tried on top of v5 and neither is adopted:

- **top-k 8 instead of 5** (`pc-v7-topk8-127`) buys 0.010 context recall for
  0.033 context precision, and answer correctness does not move (0.598 against
  0.605 and 0.616 in the two v5 judgings). The multi-document questions that
  motivated it are not short of passages — they need the *right* passage, which
  is a retrieval-quality problem, not a budget one. The default stays 5.
- **the concise prompt** below.

The zeros looked like a penalty for answering *too fully*, so prompt v4 was
written to answer the question and nothing beside it, and measured
(`pc-v6-concise-127`). It is a clear negative result: answers got 40 %
shorter (667 → 398 characters on average) and twice as fast (18.5 → 10.8 s
per question), and correctness **fell** by 0.05 (0.605 → 0.554) with
faithfulness down 0.017. In recall mode the reference claims the answer
*misses* cost more than the extra claims it volunteers, so completeness
wins. The default stays `QA_PROMPT=v3`; v4 is worth choosing only when the
halved latency matters more than 0.05 of correctness.

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
- The reranker ablation shows its effect is concentrated in context
  precision (+0.06); the other metrics move within judge noise (±0.02), so
  the reranker is a recommended but optional setting (`RERANKER_MODEL`),
  costing ~1 s per question on the GPU.
- Agent mode (run from `feat/agent`) on the same ingestion reaches the
  highest answer relevancy and
  context recall (it can search twice and read whole sections), but the
  judge scores its faithfulness lower: the agent also reads sections through
  `read_section`, and that text is not part of the exported `contexts`
  the judge checks the answer against, so grounded statements look
  unsupported. Single-pass RAG remains the better default for the judge
  protocol; an agent built on this baseline is the tool for multi-hop
  questions.
- Remaining weak spots: questions that need cross-file joins over raw
  spreadsheet rows (e.g. the intersection of product names of two files),
  reference answers not grounded in the document (the tea question about
  cold-season drinking has no such passage in `Чай.md`), and multi-hop
  questions that combine a fact from one document with a claim about the
  other; an agent built on this baseline is the intended tool for the latter.

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
- **Scanned pages**: the 29 image pages of `AB_test.pdf` are transcribed by
  the VLM (Markdown with headings and tables) instead of EasyOCR text. In v5
  the transcript validator accepted all 29 on the first attempt
  (`ocr_stats: {'vlm_ok': 29}`); an earlier calibration that estimated the
  expected length from raw ink flagged 27 of them and paid a second request
  each, which is why the estimate now counts text lines instead.
- **Table questions** (v4 → v5): the four chess-rating lookups that returned
  "no information" are answered correctly ("FIDE 1260 → Chess.com Bullet
  1000"), the maximum-rating question stopped being answered from an
  eleven-row fragment, and the sheet inventory question is answered from the
  workbook overview. Measured per document: `Chess_Rating_Comparison_2016`
  +0.22 answer correctness, `sales-data.xlsx` +0.80, `inventory + sales`
  +0.15, `AB_test.pdf` +0.30, `Сотрясение головного мозга.pdf` +0.30.

### 3.4 OCR engine benchmark (how the numbers in §2.2 were produced)

`ocr_bench.py` (kept with the operator scripts, not in the package) takes six
pages of the corpus that carry a real text layer — Russian prose, a lecture
with formulas, two financial tables, a medical guideline and a two-column
English paper — renders each at 150 dpi, degrades it three ways, and asks both
engines to transcribe the image. The text layer is the ground truth; the
comparison normalises whitespace, quotes, `ё/е` and Markdown syntax, and
rejoins hyphenated line breaks on both sides, so neither engine is penalised
for formatting.

Two caveats belong with the numbers. The degradations are synthetic (a real
scanner adds artefacts this does not model), and character error rate
undervalues the model on pages with formulas, where LaTeX is the better
transcript but the further one from the text layer. Both were checked by
reading the transcripts, not only the metric.

The judge runs are the end-to-end check on the *real* scan of the corpus
(`Документационное обеспечение…pdf`, eight scanned pages, seven questions):
its answer correctness is 0.86 in the v4 run, the second-best document of the
set.

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
