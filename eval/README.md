# Evaluation

This folder holds the golden set that measures MAP retrieval against grep.

**Scope: source code.** All queries and judgments are for ripgrep, which is
Rust code. One person wrote them. MAP can build an index of text of all types,
but no result here measures text that is not code. See
[Known limits](#known-limits).

## Contents

```
eval/
  baseline-lexical.json     frozen scores for `lexical`
  baseline-fused.json       frozen scores for `lexical` + `semantic`
  corpora/ripgrep-8372866/
    queries.jsonl           47 queries
    qrels.jsonl             120 graded relevance judgments on line ranges
```

The folder name `ripgrep-8372866` gives the corpus and the commit. The
judgments are line ranges. Thus `scripts/fetch-corpus` gets only that commit.

## How the author wrote the set

- The author wrote 44 of the 47 queries from the ripgrep source only. The
  author did not run `map` in that work. No judgment comes from the output of a
  system.
- The author wrote those 44 queries before the author wrote the harness.
- The three `component` queries are different. The author wrote them after the
  author read MAP output for this corpus. To decrease that bias, the files of
  each component come from the module structure of ripgrep.
- The author read each span that has a judgment.
- The author of the judgments also wrote the MAP tokenizer. Thus the two can
  have the same errors. A review by a different person is the only control for
  that.
- For some queries, the author used a text search to find the answer. That
  method helps grep.

## Format

### `queries.jsonl`

Each line is one query:

```json
{"id":"rg-001","text":"binary detection quit convert byte","kind":"mechanism","form":"keyword","facet":"anchor"}
```

`text` is the input for all systems. Each system has an adapter that changes
the text into the input for that system:

- a set of terms for BM25
- a regex for grep
- an embedding for a tensor dimension

Each system gets the same text. A regex that a person tunes for grep is a
different baseline, and it goes in a different file.

Each query has three tags.

**`form`** is the style of the query text.

| form | example | queries |
|---|---|---|
| `keyword` | `binary detection quit convert byte` | 14 |
| `prose` | `Handling a line longer than the read buffer by rolling unconsumed bytes to the front and growing capacity.` | 17 |
| `question` | `How does ripgrep decide whether to memory-map a file instead of reading it incrementally?` | 16 |

**`facet`** is the retrieval problem of the query.

| facet | problem | queries |
|---|---|---|
| `anchor` | The query contains an identifier from the code. `lexical` usually finds the answer. | 11 |
| `concept` | The query uses words that are in the code. | 19 |
| `vocab-gap` | The query does not use the words of the code. `lexical` usually does not find the answer, and an embedding dimension must find it. | 6 |
| `distributed` | The answer is in more than one file. | 2 |
| `component` | The answer is all the files of one component. | 3 |
| `absent` | The corpus has no answer. The best response is no hit. | 6 |

**`kind`** is the type of question: `mechanism`, `locate`, `behavior`,
`interface`, `component`, or `absent`. The harness does not group results by
`kind`.

#### `absent` queries

An `absent` query has no judgments. Thus it has no nDCG. The harness shows the
top score that the system gives for the query. MAP normalizes fused scores to
the range 0 to 1. Thus you can compare the top scores of different queries, and
a lower top score is better. The harness prints this value as
`top confidence`.

A query is `absent` only in relation to a corpus. A different corpus can
contain the answer. Thus each `absent` query gives a list of corpora. The
author made sure that those corpora contain no answer:

```json
{"id":"rg-041","text":"How does it schedule pods onto Kubernetes nodes?","kind":"absent","form":"question","facet":"absent","absent_in":["ripgrep","flask"]}
```

The harness gives a score to an `absent` query only when `absent_in` contains
each corpus in the search. It shows the number of `absent` queries that it
ignores.

For three of the six `absent` queries, `absent_in` contains `ripgrep` and
`flask`. The other three (`rg-039`, `rg-040`, `rg-044`) have an answer in
Flask. The harness ignores those three when `flask` is in the search.

### `qrels.jsonl`

Each line is one judgment. A query can have more than one judgment.

```json
{"query_id":"rg-001","resource":"crates/searcher/src/searcher/mod.rs",
 "start_line":34,"end_line":118,"grade":2,"why":"BinaryDetection: the ..."}
```

| grade | definition |
|---|---|
| **2** | The span answers the question. |
| **1** | The span helps, but it is not the answer. Examples: the caller, the type that the code makes, or the same function in a different backend. |
| no row | The span is not relevant. A span with no judgment scores 0. |

`why` records the cause of the judgment. A reviewer can then examine the
judgment and reject it.

## Scores

The harness gives each hit a score at two levels:

- **File level.** A hit counts if its file has a judgment. All systems can get
  this score, and that includes grep.
- **Span level.** A hit counts if its line range has an overlap with a span
  that has a judgment. This level shows the difference between the correct file
  and the correct function.

The primary metric is nDCG@10. The harness also shows Recall@10 and MRR. It
also shows the number of different files in each 10 results.

## Known limits

- **The set is small.** It has 47 queries, and only 41 have judgments. The 6
  `absent` queries have no judgments, because the corpus has no answer. The
  facet groups are smaller: `distributed` has 2 queries. A difference of a
  small number of points of nDCG is not significant (1 point = 0.01).
- **The queries in one group are about different subjects.** This applies to
  the facet groups and to the form groups. Thus a difference between two groups
  can come from the subjects of the queries, not from the facet or the form. A
  controlled set has each subject in all three forms. That set removes the
  problem, but it is three times as large.
- **The judgments do not include all correct files.** If a file has the answer
  to a query and no reviewer found that file, the file gets a score of 0. A
  pooled judgment method decreases this problem. This set does not use one.
- **`rg-008` has a judgment for a span of 375 lines.** The file
  `default_types.rs` is one table. Thus a hit in that file almost always gets
  the span-level score.
- **One corpus, one language.** The judgments are for ripgrep only. The harness
  can add `flask` as a second root with `--root`, but `flask` has no
  judgments.
- **One type of resource.** All queries, judgments, and tuned values are for
  source code. No result shows how MAP does on prose, transcripts, tickets, or
  records from other systems. Two settings can change the result for such text:
  - The tokenizer has no stopwords, no stemming, and `[alnum_]` word
    characters. A different tokenizer is a different classifier `impl`.
  - The segment size is 40 lines. It is a `[segmenter]` setting.

  Do not change these values until a corpus that is not code has judgments.

## Results

The scores for each dimension set and for grep are in
[`../README.md`](../README.md#retrieval-quality).

`baseline-lexical.json` and `baseline-fused.json` hold the frozen scores of the
two deterministic configurations. A difference from a frozen score is a
regression. A configuration that includes `descriptive` has no frozen score. A
model writes those descriptors, and the scores change between index builds.

### Reproduce the `lexical` row

For this row, no model, no key, and no network are necessary after the corpus
download. Run these commands from the root of the MAP repository:

```sh
scripts/fetch-corpus.sh                 # or scripts/fetch-corpus.ps1
map init ../ripgrep
map index ../ripgrep
cargo run --release -p map-eval -- ../ripgrep --dim lexical --baseline eval/baseline-lexical.json
```

The output shows each score and its difference from the frozen score. Each
value has three digits after the point. Each difference must show `0.000`.

### Reproduce all rows

```sh
scripts/bench-implementations.sh ../ripgrep
```

Two items are necessary for this script:

- a binary with the `distilled` and `llm` features
- an index that has all four dimensions

An LLM endpoint is necessary for the `descriptive` dimension, and its index
build has a cost.

The script does not make the grep row. For that row, run this command:

```sh
cargo run --release -p map-eval -- ../ripgrep --grep
```
