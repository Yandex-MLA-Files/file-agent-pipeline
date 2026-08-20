# Retrieval: how a question becomes five passages

This is the companion to [parsing_and_chunking.md](parsing_and_chunking.md):
that document ends where the chunks are indexed, this one starts there. It
describes what `LanceDBRetriever.search()` does stage by stage, why each stage
exists, how each was measured, and what the judge can and cannot tell about
it.

## 1. The stages

```
question
   │
   ├─ 1. formulations      MULTI_QUERY: the question + N model-written variants
   │                        (paraphrases / a hypothetical answer passage)
   │
   ├─ 2. hybrid search     one per formulation: BM25 (two lexical views) + dense
   │      per formulation   (bge-m3), fused by LanceDB's reciprocal-rank fusion
   │
   ├─ 3. fusion            reciprocal-rank fusion across formulations
   │                        (original weight 1.0, variants MULTI_QUERY_WEIGHT)
   │
   ├─ 4. quoted phrases    hits that carry every "quoted phrase" move first
   │
   ├─ 5. rerank            cross-encoder re-scores the top RERANKER_CANDIDATES
   │                        against the *original* question
   │
   ├─ 6. diversify         the best hit of every document keeps a slot
   │
   └─ 7. unique passages   top-k is filled with distinct parent passages
```

Every stage is optional and independently switchable; §4 lists the switches.
Stages 1, 4, 5 (the wider pool and the knobs) and 7 are new on this branch;
2 gained the second lexical view.

### 1.1 Two lexical views of every chunk (BM25)

The full-text index (LanceDB / tantivy) tokenises, lowercases and applies the
Snowball Russian stemmer. Two things it does not do cost real matches:

* **`ё` and `е` are different characters to it.** `ещё` / `еще`, `учёт` /
  `учет` — a query spelled one way misses a document spelled the other way
  outright. Measured on the toy index in the tests: `елки` does not match
  `ёлки` without folding.
* **Stemming is a suffix heuristic, not morphology.** `люди` and `человек`,
  `шёл` and `идти` never meet; short stems collide across unrelated words.

So every chunk is indexed twice: the raw `text` (stemmed by the index) and a
`fts_text` column produced by `text_normalization.normalize_for_fts()` — NFC,
lowercase, `ё` folded, and every Cyrillic word replaced by its pymorphy3 lemma
(`BM25_LEMMATIZE`, `auto` = when pymorphy3 is installed). The query goes
through the same function, and the lexical query is a boolean OR of the raw
words against `text` and the normalised words against `fts_text`: a chunk
scores through whichever view matches, so a lemma the analyser gets wrong
(`пришли` → `прислать`) still meets its stem. Lemmatisation costs ~160 µs per
distinct word and is cached per token; on the corpus it is invisible.

The `fts_text` index carries positions, so a phrase the user put in quotes
(`"выручка компании"`, `«…»`) becomes a positional `PhraseQuery` on the
lexical side; because the dense side of a hybrid search knows nothing about
quotes, hits that contain every quoted phrase are then moved ahead of the rest
(stage 4) rather than the rest being dropped — recall survives OCR spelling
and case endings, the user's emphasis is honoured.

### 1.2 Multi-query fusion

A question and the passage that answers it rarely share wording: the user
asks *сколько заработала компания*, the report says *выручка составила*. Dense
retrieval bridges some of that, BM25 none of it. `query_expansion.py` asks the
answering model for alternative formulations and the retriever searches with
all of them:

| `MULTI_QUERY_MODE` | what is generated | when it helps |
|---|---|---|
| `paraphrase` | N rewordings: synonyms, the document's terms instead of the asker's, an expanded abbreviation | the general case |
| `hyde` | one short passage written *as if quoting the document that answers* | short factual questions whose answer is a paragraph: the passage's embedding sits where the answer sits, not where the question sits |
| `mixed` | N−1 rewordings + 1 hypothetical passage | both |

What comes back is parsed defensively (numbering, bullets, quotes and chatter
stripped; duplicates and the original dropped; a length cap), and any model
failure degrades to the single original query — the retriever never fails
because the expander did.

Each formulation runs the same hybrid search; the lists are fused by
reciprocal rank (`Σ w / (60 + rank)`), the original question at weight 1.0.
The cross-encoder then sees the fused pool but scores it against the
**original** question only — the variants widen the pool, they do not get a
say in the final order.

Formulations are cached on disk (`MULTI_QUERY_CACHE`, default
`~/.cache/file_agent/multi_query`, keyed by question, mode, count, model and
prompt version): a repeated question costs no model call, a benchmark is
reproducible, and the dataset run pays once per distinct question.

Cost: one model call per new question (~1–3 s on the 27B without thinking),
N extra encodes (milliseconds) and a larger reranker pool. Sample of what the
model writes for the dataset's questions:

```
Q: какие темы по матану надо повторить?
   - какие разделы математического анализа требуют повторения
   - какие темы из курса матанализа нужно освежить
   - какие главы по математическому анализу необходимо повторить
Q: Какие симптомы могут наблюдаться при сотрясении головного мозга?
   - Какие признаки характерны для черепно-мозговой травмы легкой степени?
   - Проявления и клиническая картина при контузии головного мозга.
   - Симптоматика легкого закрытого повреждения головы.
```

### 1.3 Reranker

`RERANKER_MODEL=BAAI/bge-reranker-v2-m3` (fp16 on the GPU, ~1.2 GB next to
the serving model) re-scores query/chunk pairs. What changed:

* **the pool**: `max(6·top_k, 30)` candidates by default (was `4·top_k, 20`),
  `RERANKER_CANDIDATES` overrides — a cross-encoder can only promote what the
  first stage hands it;
* **the cost knobs**: `RERANKER_BATCH_SIZE` (32) and `RERANKER_MAX_LENGTH`
  (1024 tokens per pair) bound one pass; the model is loaded once per process;
* **`RERANKER_BLEND`** (0..1, default 0): mixes the first-stage RRF rank back
  in as a normalised term, a guard against a cross-encoder that is confidently
  wrong on one pair. At 0 the cross-encoder decides alone.

Measured (§3): the pool size beyond 20 does not change what reaches top-5 on
this corpus, and blending does not help — both are kept as knobs, neither is
on. The honest reading of the reranker on this dataset is the one the v4
ablation gave: it buys context precision (+0.06) and pays a little recall.

### 1.4 Distinct passages

The prompt shows the *parent passage* of a chunk (`metadata["context"]`), and
`select_context_passages()` collapses chunks that share one. Before this
branch that collapsing happened *after* the top-k cut: five chunks of two
sections meant the model read two passages and three slots were wasted.
`RETRIEVAL_UNIQUE_PASSAGES` (default on) does the same collapsing *before* the
cut, so top-k means five distinct passages. In the v14 audit sample one
question's five contexts contained the same passage twice.

### 1.5 Diversification and single-document search

Unchanged from the ingestion branch: with several files indexed, the best hit
of every file keeps a slot (`RETRIEVAL_DIVERSIFY_DOCS`), and
`search(..., source_file=…)` restricts a query to one document as a LanceDB
prefilter — the entry point a per-document agent tool needs. Diversification
switches itself off for a single-document query.

## 2. How retrieval is measured

The judge's `context_recall` moves by ±0.004 between two judgings of the same
contexts, but the whole answer pipeline sits between a retrieval change and
the judge's number, and the generation itself is not deterministic. For
retrieval questions there is a direct instrument (`bench_retrieval.py` on the
server, `~/bench_cache/`):

* the 127 dataset questions, each searched over *its own* documents exactly
  as the pipeline does (parse and chunk cached, embeddings cached across
  configurations, so a configuration costs a minute);
* the proxy for "did retrieval deliver the evidence": lemma recall of the
  reference answer's content words (no prepositions, conjunctions,
  particles, pronouns; numbers kept) inside the passages the model would read.
  `recall_best` is the best single passage, `recall_union` the union of the
  five, `hit@.5`/`hit@.7` the share of questions whose best passage covers at
  least half / 70 % of the reference. Questions whose reference is "нет
  информации" (17 of 127) have no evidence to find and are excluded;
* validated against the judge on v14: Spearman 0.47 with `context_recall`;
  every question the proxy scores ≥ 0.7 the judge scores 1.0; questions it
  scores < 0.3 still average 0.73 with the judge, because references paraphrase
  heavily. It is conservative and monotone — the right shape for a *paired*
  A/B on the same questions, where only the direction and size of a change
  matter, and its noise is zero.

## 3. Results

### 3.1 Per question over its own documents (what the evaluation pipeline does)

110 questions with a content reference, top-5, reranker on
(`BAAI/bge-reranker-v2-m3`), 2026-08-19. `s/q` is the search alone; the
multi-query rows include the model call for a question not yet cached.

| config | recall_best | recall_union | hit@.5 | hit@.7 | passages | chars | s/q |
|---|---|---|---|---|---|---|---|
| v14 (candidates 20, no lemmas, chunks not passages, single query) | 0.624 | 0.714 | 0.645 | 0.473 | 4.50 | 9 945 | 0.15 |
| + lemmatised BM25 view | 0.626 | 0.714 | 0.645 | 0.482 | 4.50 | 9 997 | 0.09 |
| + distinct passages | 0.628 | 0.722 | 0.655 | 0.482 | 5.00 | 11 222 | 0.09 |
| + reranker candidates 30 | 0.628 | 0.722 | 0.655 | 0.482 | 5.00 | 11 222 | 0.09 |
| + reranker candidates 50 | 0.628 | 0.722 | 0.655 | 0.482 | 5.00 | 11 239 | 0.12 |
| + multi-query, 3 paraphrases | 0.629 | 0.720 | 0.655 | 0.482 | 5.00 | 11 208 | 2.97 |
| + multi-query, 2 paraphrases | 0.627 | 0.721 | 0.655 | 0.473 | 5.00 | 11 248 | 2.03 |
| + multi-query, mixed (2 + HyDE) | 0.629 | 0.721 | 0.655 | 0.482 | 4.99 | 11 205 | 6.17 |
| + multi-query, HyDE | 0.629 | 0.722 | 0.655 | 0.482 | 5.00 | 11 210 | 4.21 |
| + multi-query 3, variant weight 0.7 | 0.629 | 0.720 | 0.655 | 0.482 | 5.00 | 11 254 | 0.15 |
| + multi-query 3, reranker blend 0.3 | 0.629 | 0.723 | 0.655 | 0.473 | 5.00 | 11 110 | 0.15 |
| lemmas + passages, **no reranker** | 0.628 | 0.723 | 0.645 | 0.491 | 4.98 | 10 976 | 0.02 |
| multi-query 3, no reranker | 0.623 | 0.716 | 0.645 | 0.482 | 4.98 | 11 134 | 0.06 |

Everything sits within ±0.005 of everything else, and that is not a failure
of the changes — it is the ceiling. `bench_headroom.py` compares what was
retrieved with the *oracle*, the best single passage anywhere in the
question's own documents:

| | hit@.5 | mean recall_best |
|---|---|---|
| oracle (best passage that exists) | 0.664 | 0.639 |
| retrieved (lemmas + distinct passages) | 0.655 | 0.628 |

One question in a hundred is lost to ranking; the rest of the distance to
1.0 is content no single passage holds — a reference that paraphrases the
document, an aggregate over a spreadsheet (`q0121`–`q0127`, all at oracle ≈
retrieved < 0.4), a comparison across two files. Each question's index is
23–870 chunks; a hybrid top-30 over that already contains what there is to
find, and the cross-encoder, the pool size, the lemmas and the extra
formulations then re-order a set that already has the answer in it. This is
also why the judged runs (§3.3) cannot separate retrieval configurations on
this dataset.

The one change that does register here is **distinct passages**: 5.00 passages
per question instead of 4.50 (+0.008 union recall, +0.010 hit@.5) — ten
percent more evidence in the prompt for free.

### 3.2 Over the whole corpus (what an agent or the app faces with a folder)

The same questions, one index over all 21 documents (3 847 chunks), so the
right passage competes with twenty other files. `doc_hit`: the right document
got at least one slot; `doc_all`: every document of a two-document question
did.

| config | recall_best | recall_union | hit@.5 | hit@.7 | doc_hit | doc_all | s/q |
|---|---|---|---|---|---|---|---|
| v14 | 0.591 | 0.695 | 0.600 | 0.418 | 0.906 | 0.898 | 0.17 |
| + lemmatised BM25 view | 0.590 | 0.690 | 0.600 | 0.418 | 0.906 | 0.898 | 0.11 |
| + distinct passages | 0.590 | 0.696 | 0.600 | 0.418 | 0.906 | 0.898 | 0.11 |
| + reranker candidates 30 | 0.590 | 0.696 | 0.600 | 0.418 | 0.906 | 0.898 | 0.11 |
| + reranker candidates 50 | 0.591 | 0.701 | 0.600 | 0.418 | 0.906 | 0.906 | 0.14 |
| + multi-query, 3 paraphrases | 0.603 | 0.707 | 0.609 | 0.436 | 0.906 | 0.866 | 0.25* |
| + multi-query, 2 paraphrases | 0.600 | 0.716 | 0.609 | 0.427 | 0.913 | 0.898 | 0.18* |
| + multi-query, mixed (2 + HyDE) | 0.606 | 0.719 | 0.618 | 0.436 | 0.913 | 0.890 | 0.21* |
| **+ multi-query, HyDE** | **0.609** | **0.735** | **0.618** | **0.436** | **0.921** | 0.898 | 0.14* |
| + multi-query 3, variant weight 0.7 | 0.603 | 0.704 | 0.609 | 0.436 | 0.906 | 0.874 | 0.19* |
| + multi-query 3, reranker blend 0.3 | 0.597 | 0.705 | 0.591 | 0.427 | 0.906 | 0.866 | 0.19* |
| lemmas + passages, no reranker | 0.596 | 0.704 | 0.591 | 0.436 | 0.906 | 0.890 | 0.02 |
| multi-query 3, no reranker | 0.595 | 0.703 | 0.600 | 0.445 | 0.906 | 0.866 | 0.09* |

\* formulations served from the cache; a question the model has not seen
costs its call on top (2–4 s on the shared 27B: ~3 s for three paraphrases,
~4 s for a HyDE passage, ~6 s for both).

Here the changes separate. **HyDE** is the best formulation to add: +0.019
recall_best, +0.039 union recall, +0.018 hit@.5 and the highest document hit
rate, at one model call — the hypothetical passage's embedding lands where the
answer sits, which is what a corpus of twenty files needs and a two-file index
does not. Paraphrases help less and, with three of them, cost `doc_all`: a
paraphrase now and then pulls a wrong file into a two-file question. Blending
the first-stage rank back into the reranker (`RERANKER_BLEND=0.3`) hurts on
both benchmarks and stays off. Widening the reranker pool from 20 to 50 adds
0.005 union recall at +30 % reranker time — 30 is the default, 50 is a knob.
The reranker itself is neutral on this proxy in both settings; its measured
value is precision (§3.3), not recall.

The oracle is the same 0.639, so the corpus setting has ~0.03 of headroom and
HyDE takes ~0.02 of it.

**Defaults after measurement**: lemmatised BM25 view (`auto`), distinct
passages (`on`), reranker candidates 30, `MULTI_QUERY=off` in code — it is a
per-deployment choice because it costs a model call per new question — with
`MULTI_QUERY_MODE=hyde` as the value to turn on for a multi-document
deployment; `.env.example` ships it on.

### 3.3 Judged runs (ragas, judge = Qwen3.5-27B without thinking)

Two runs, same 127 questions, same documents, same answering model; the
only difference is the retrieval configuration. Pipeline failures score 0
(there were none). `s/q` is wall-clock per question over the run; v15's
formulations came from the benchmark's cache, a cold question adds ~4 s.

| run | faithfulness | answer_correctness | reference coverage | answer_relevancy | context_precision | context_recall | s/q |
|---|---|---|---|---|---|---|---|
| v14 — ingestion branch as merged (candidates 20, chunks, single query) | 0.957 | 0.578 | 0.716 | 0.823 | 0.754 | 0.848 | 12.4 |
| **v15 — this branch** (lemmatised BM25, distinct passages, candidates 30, multi-query HyDE) | 0.956 | 0.585 | 0.727 | **0.844** | 0.752 | **0.876** | 13.3 |

Read against the measured floors — ±0.004 on the context metrics, ±0.016 on
answer_correctness (§5) — the context_recall gain of **+0.028** and the
answer_relevancy gain of **+0.021** are real; faithfulness, precision and
correctness did not move. Row by row: 76 questions saw a different set of
passages, 58 a different answer, and on those 58 the change is where it should
be —

| on the 58 questions whose answer changed | v14 | v15 |
|---|---|---|
| context_recall | 0.750 | **0.810** |
| answer_relevancy | 0.785 | **0.832** |
| context_precision | 0.693 | 0.709 |
| faithfulness | 0.939 | 0.933 |
| answer_correctness | 0.580 | 0.582 |

— and the model read 5.00 distinct passages per question instead of 4.48.
The recall gain concentrates on the documents where the answer is spread over
several sections (the RZD financial report +0.20, the A/B lecture +0.20, the
chess rating workbook +0.14); the one document group that lost (A/B + Agentic
Memory, −0.125 over four questions) is the two-file comparison type where the
HyDE passage steers toward one of the two files.

This is the highest context_recall of any run so far except `v7` (top-k 8,
0.875), and unlike v7 it does not pay for it in precision (v7: 0.744, v15:
0.752): the extra evidence comes from filling the same five slots better,
not from adding slots.

## 4. Configuration reference (retrieval)

| Variable | Default | Effect |
|---|---|---|
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | dense encoder; the chunker budgets with its tokenizer |
| `BM25_LEMMATIZE` | `auto` | lemmatise the lexical index column with pymorphy3 (`on` requires it, `off` = ё-folding + stemming) |
| `MULTI_QUERY` | `off` (`.env.example`: `on`) | search with generated formulations of the question and fuse |
| `MULTI_QUERY_MODE` | `paraphrase` (`.env.example`: `hyde`) | `paraphrase` / `hyde` / `mixed` |
| `MULTI_QUERY_COUNT` | `3` | formulations to add (1..8) |
| `MULTI_QUERY_WEIGHT` | `1.0` | RRF weight of a formulation against the question's 1.0 |
| `MULTI_QUERY_CACHE` | `~/.cache/file_agent/multi_query` | disk cache of formulations, `off` disables |
| `RERANKER_MODEL` | — (`.env.example`: `BAAI/bge-reranker-v2-m3`) | cross-encoder; empty disables |
| `RERANKER_CANDIDATES` | `max(6·top_k, 30)` | first-stage hits re-scored |
| `RERANKER_BATCH_SIZE` | `32` | pairs per forward pass |
| `RERANKER_MAX_LENGTH` | `1024` | tokens per query+passage pair |
| `RERANKER_BLEND` | `0` | share of the first-stage rank in the final score |
| `RETRIEVAL_UNIQUE_PASSAGES` | `true` | fill top-k with distinct parent passages |
| `RETRIEVAL_DIVERSIFY_DOCS` | `true` | keep the best hit of every document |

Every one of these is recorded in the generation fingerprint
(`retrieval_settings_fingerprint()`), so a checkpoint written under one
configuration is never resumed under another; `RAG_PIPELINE_VERSION` moved to
`structured-parsers-multi-query-v4`.

## 5. Can the judge be trusted? An audit

The question was asked directly — *are the judge's numbers real, or drawn?* —
so it was answered in three ways on the v14 run (`pc-v14-inline-127`, trace
`judge_trace_log_20260818T155354827222Z.jsonl`).

### 5.1 The numbers are derivable from the judge's own recorded reasoning

Every judged row carries the judge's intermediate steps: the statements it
decomposed the answer into and its verdict on each, the claims of the answer
and of the reference with the entailment verdicts both ways, the reference
sentences it attributed to a context, its verdict on every retrieved passage.
Recomputing each metric from those steps and comparing with the reported
score (`judge_recompute.py`, `ref_coverage.py` on the server):

| Metric | reproduced from the trace |
|---|---|
| faithfulness = supported statements / statements | **127 / 127** |
| answer_correctness = tp / (tp + fn) over claims (ragas `FactualCorrectness`, `mode="recall"`) | **127 / 127** |
| context_recall = attributed reference sentences / sentences | **127 / 127** |
| context_precision = average precision over the per-passage verdicts | **127 / 127** |
| answer_relevancy | needs the embedding model; the generated questions it is computed from are logged for 127 / 127 |

Nothing is drawn: every number is a deterministic function of a verdict the
judge wrote down together with its reason, and anyone can rerun the check.

### 5.2 A blind re-score of 30 rows

Thirty rows were drawn stratified by the judge's `answer_correctness` (10 with
≤ 0.2, 10 in between, 10 with ≥ 0.8) and re-scored blind — question,
reference and answer only, the judge's number hidden — on the same 0..1
scale, before the two were compared:

| | rows |
|---|---|
| within 0.2 of the judge | 13 |
| ≥ 0.4 apart | 14 |
| of those, judge **lower** than the reader | 13 |
| judge higher | 1 |

The disagreements are not random: in eight of the thirteen the answer is
*byte-for-byte what the reference says* and the judge scored it 0.00 —
`q0044` (reference "Шардинг.", answer "…является шардинг…"), `q0069`
(reference: five numbers of the TrueNorth chip, answer: the same five
numbers), `q0112` (two sheets, Comparisons and Data), `q0036`, `q0029`,
`q0030`. The trace of `q0069` shows why:

> *Контекст указывает на технологический процесс 28 нм, но не называет
> устройство IBM TrueNorth. Утверждение не может быть напрямую выведено из
> контекста без внешних знаний.* — verdict 0, for all five claims.

The metric verifies each claim of the answer against the **reference text
alone**, without the question. The reference "Технологический процесс 28 нм;
1 млн нейронов; …" never names the chip — the *question* does — so no claim of
the answer is entailed, tp = 0, and the score is 0 although the answer covers
the reference completely (fn = 0). The same happens whenever the reference is
terse: a single word, a list of numbers, "В документе нет информации об этом".
This dataset's references are often exactly that.

On the ten context rows the direction is the same: the judge is stricter than
the reader on faithfulness in 3 of 10 (it split "чаще среди детей, молодых до
30 и пожилых" into three statements and rejected each because the context
"lists three groups"), and on context_recall in 3 of 10 (it concedes in its
own reason that the context *implies* the reference and still writes 0; a
reference that states a *negative* — "the notes do not connect X with Y" —
cannot be "attributed" to a passage). Nowhere in the forty rows did it score
something *higher* than a careful reader would by more than 0.4, except one
row (`q0028`) where it gave 1.0 to an answer that says "no direct
information".

### 5.3 What that means for the tables

* **faithfulness, context_precision, context_recall are trustworthy and
  conservative**: reproducible, and where they disagree with a reader they
  err on the strict side. Differences in them are real.
* **answer_correctness is not a correctness measure on this dataset.** It is
  reproducible, but the protocol (`FactualCorrectness`, `mode="recall"`, claims
  verified against the reference without the question) returns 0 for correct
  answers to terse references. That is why the column is the noisy one, why a
  more complete answer scores lower, and why the earlier documents warned not
  to read it alone.
* The half of that metric that asks the right question is in the traces:
  **reference coverage** = share of the *reference's* claims that the answer
  states (`1 − fn / |reference claims|`), read off the same NLI pass. It is
  more stable between judgings (v5 judged twice: 0.722 / 0.723, where
  answer_correctness gave 0.605 / 0.616) and it does not zero a perfect answer:

  | run | answer_correctness | reference coverage | rows scored 0 with full coverage |
  |---|---|---|---|
  | baseline | 0.412 | 0.471 | 10 |
  | v5 | 0.605 | 0.722 | 13 |
  | v5 judged again | 0.616 | 0.723 | 13 |
  | v10 | 0.577 | 0.724 | 15 |
  | v12 | 0.584 | 0.712 | 13 |
  | v13 | 0.568 | 0.704 | 14 |
  | v14 | 0.578 | 0.716 | 14 |

  Read with coverage instead of answer_correctness, the ingestion branch is
  not "−0.027 against v5" but level (0.716 against 0.722), and every run since
  v5 sits within 0.02 of every other — which is what the byte-identical-answer
  analysis said all along.

The judge itself was left as it is: `eval_pipeline/` is shared and other
branches have changed it heavily; changing the protocol would also break
comparability with every earlier run. The recommendation for the team is (a)
to report reference coverage next to answer_correctness — it costs nothing,
it is in every trace already — and (b) if the metric is to be fixed, to pass
the question together with the reference as the premise of the claim check,
which is what a human reader does.
