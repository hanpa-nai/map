# Evaluation — the quality gate

The question this answers: **does the fully-open MAP stack beat grep?** The
query set was authored before the harness that scores it.

**Scope: source code.** Every query and judgment here is ripgrep, in Rust,
authored by one person. MAP indexes any text and assumes nothing about what a
resource is, but that is a property of the pipeline, not a result — see
*Known limits*.

## What is here

```
eval/
  corpora/ripgrep-8372866/
    queries.jsonl     47 queries, mixed register and facet
    qrels.jsonl      120 graded relevance judgments over spans
```

`ripgrep-8372866` names the corpus **and the exact commit**. Judgments are line
ranges, so `scripts/fetch-corpus` pins.

## How the set was authored

- Authored from ripgrep's source only. `map` was never run while writing them,
  and no judgment was taken from any system's output.
- Written before the harness existed.
- Every span was read. Three cited ranges initially ran past end-of-file and
  were corrected; four were re-read line by line against the `why` field.
- The author of these judgments also wrote MAP's tokenizer, so shared blind
  spots are possible and independent review is the only control for that. Where
  a query was located with a text search, that biases toward grep.

## Format

`queries.jsonl` — one question per line.

```json
{"id":"rg-001","text":"How does ripgrep decide a file is binary and stop searching it?","kind":"mechanism"}
```

`text` is the canonical input. Each system's adapter turns it into whatever that
system consumes — a bag of terms for BM25, a regex for grep, an embedding for
RAG. Every system is handed the same input; a hand-tuned regex for grep is a
baseline definition and belongs in a separate file.

Every query is tagged on two axes.

**`form` — the register the query is written in.**

| form | example |
|---|---|
| `keyword` | `binary detection quit convert byte` |
| `prose` | `Handling a line longer than the read buffer by rolling unconsumed bytes.` |
| `question` | `How does ripgrep decide whether to memory-map a file?` |

**`facet` — the retrieval challenge being stressed.**

| facet | what it stresses | n |
|---|---|---|
| `anchor` | query names an exact identifier; lexical should excel | 11 |
| `concept` | described in words that appear in the code | 19 |
| `vocab-gap` | query **avoids** the code's vocabulary; lexical should fail, and the semantic dimension is what must rescue it | 6 |
| `distributed` | answer legitimately spans several files | 2 |
| `absent` | **no answer exists**; the ideal response is nothing | 6 |

An `absent` query carries no judgments and is scored not on nDCG but on how
confident the system was that it found something. Fused scores are normalized to
`[0, 1]`, so the top score is a real confidence and lower is better.

**Absence is a property of a corpus, not of a query**, so each one records the
corpora its absence was checked against:

```json
{"id":"rg-039","text":"sql database connection pool","facet":"absent",
 "absent_in":["ripgrep"]}
```

The harness scores an absent query only when **every** searched corpus appears
in `absent_in`, and reports how many it skipped. Three of the six are curated
for `ripgrep + flask`; the other three (`rg-039`, `rg-040`, `rg-044`) are
answerable in Flask and are skipped under that federation until recurated.

`qrels.jsonl` — one judgment per line. A query has several.

```json
{"query_id":"rg-001","resource":"crates/searcher/src/searcher/mod.rs",
 "start_line":34,"end_line":118,"grade":2,"why":"BinaryDetection: the ..."}
```

| Grade | Meaning |
|---|---|
| **2** | Answers the question. Reading this span, you know. |
| **1** | Genuinely useful supporting context — the caller, the type produced, the same idea in another backend. Not the answer. |
| absent | Not relevant. Unjudged is treated as 0. |

`why` records what the judgment is claiming, so it can be reviewed and argued
against.

## Scoring, at two granularities

The same hit scores both ways:

- **File level** — a hit counts if it names a judged file. Every system can be
  scored this way, including line-oriented grep.
- **Span level** — a hit counts if its returned range *overlaps* a judged span,
  which distinguishes "found the right file" from "found the right function".

Metrics: **nDCG@10** primary, with Recall@10 and MRR reported alongside, plus
bytes-returned as a token-cost proxy.

## Known limits

- **47 queries is small**, and only 41 are scored — the 6 absent queries carry
  no judgments by design. Per-facet slices are smaller still (`distributed` is
  only 2). Differences under a few points of nDCG are not significant.
- **Facet and form slices share no intents**, so a per-slice difference
  confounds the axis with which queries happened to land in it. The controlled
  version — the same intent phrased in all three registers — would remove that
  at the cost of tripling the set.
- **Judgments are not exhaustive.** A file that answers a query but that no
  reviewer thought of scores 0. Pooling would reduce this and has not been done.
- **`rg-008` is judged over a 375-line span** because `default_types.rs` is one
  table and nothing else, which makes span-level scoring lenient for that query.
- **One corpus, one language.** `flask` joins at federation.
- **One kind of resource.** Every query, judgment and tuning decision here is
  over source code. Nothing establishes how MAP retrieves prose, transcripts,
  tickets or exported records. The two knobs most likely to matter off code are
  the **tokenizer** (no stopwords, no stemming, `[alnum_]` word characters) and
  the **segment size** (40 lines). Both are reachable from config, and neither
  should move until there is a non-code corpus with judgments to move it
  against.

## Results

Headline retrieval numbers — every dimension subset against grep, on this
corpus — live in [`../README.md`](../README.md) under *Retrieval quality*.

Reproduce them with:

```sh
scripts/bench-implementations.sh ../ripgrep
```

`baseline-lexical.json` and `baseline-fused.json` freeze the scores of the two
deterministic configurations; a difference is a regression. Any configuration
involving `descriptive` is authored by a model, moves between indexing runs, and
is deliberately not frozen.
