# MAP — Model Awareness Plane

[![ci](https://github.com/hanpa-nai/map/actions/workflows/ci.yml/badge.svg)](https://github.com/hanpa-nai/map/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

MAP is a retrieval index for agents that lives **beside your resources and
travels with them**. `.map/` sits next to `.git/`, is committed like source, and
whoever clones the repo can query it immediately without indexing anything.

A resource is any text file in the root: source code, documentation, notes,
transcripts, exported records. Nothing in the pipeline knows what kind of thing
it is indexing — discovery walks a directory, segmentation cuts line windows,
and what the material *is* is something a corpus states in its own prompts and
dimension descriptions.

> **The measured evidence is on code.** The retrieval numbers below come from
> one corpus, ripgrep, in one language. That MAP *runs* on prose and records is
> a property of the pipeline; that it retrieves them *well* is unmeasured, and
> [`eval/README.md`](eval/README.md) says what would have to exist to know.

> **Alpha** (`v0.0.0`). The on-disk format is a **draft** and can still change —
> see [`spec/format-v1.md`](spec/format-v1.md). Not published to crates.io;
> build from source.

Build the index once:

```console
$ map init && map index
$ map find "binary detection" -n 3
crates/core/search.rs:129  0.936
crates/core/flags/hiargs.rs:1185  0.931
crates/searcher/src/searcher/mod.rs:33  0.920
```

Commit `.map/`. Everyone who clones then queries it **without indexing at all**,
and gets the same hits with the same scores:

```console
$ git clone git@github.com:you/your-repo.git && cd your-repo
$ map find "binary detection" -n 3
crates/core/search.rs:129  0.936
crates/core/flags/hiargs.rs:1185  0.931
crates/searcher/src/searcher/mod.rs:33  0.920
```

No API key, no network, no model download for the default build. On ripgrep
(229 files) the default index is 2.4 MB committed and builds in 4 s; a query
is a 50 ms process, end to end.

## Why

Agents find things by probing: search, list, read, repeat. Every probe is
speculative, every miss costs tokens, and the accumulated output bloats the
context window.

MAP hands the model a committed index instead. A **dimension** is one facet of a
query — one field the model fills in — and a single call can address any
combination of them:

| | |
|---|---|
| **Token efficiency** | An N-dimensional query answered in one call — no probe loop |
| **Less context rot** | Structure (cluster labels) returned before content |
| **Precision** | Each facet is scored by the feature built for it, and the scores fuse |
| **Portability** | The index is committed, so it is built once and read by everyone |

Against ripgrep, `lexical + descriptive` scores **0.659** file-nDCG@10 where
grep scores **0.453** — full table in [Retrieval quality](#retrieval-quality).

## What MAP is not

It is not a search server, a RAG framework, or a hosted service. It is a
directory of index objects that lives in your repository, plus a CLI that reads
them. There is no daemon and nothing listens on a port.

It has no understanding of any format it indexes. There is no parser, no
grammar, and no per-language or per-filetype handling anywhere in the pipeline.

## Install

Requires Rust 1.88 or newer; Linux, macOS, and Windows are all exercised in CI.
The embedding and LLM dimensions are **cargo features**, chosen at build time:

```console
$ cargo install --path crates/map-cli                                   # lexical + declaration
$ cargo install --path crates/map-cli --features distilled              # + semantic
$ cargo install --path crates/map-cli --features auto-distilled         # + `map model fetch`
$ cargo install --path crates/map-cli --features "distilled llm"        # + descriptive
```

The default build compiles about 190 crates: 38 s with a warm cargo cache on
a laptop, a few minutes cold.

`auto-distilled` adds `map model fetch`, which downloads the embedding weights.
It is separate from `distilled` so that a build which can *load* weights links
no HTTP stack at all — verifiable with `cargo tree`.

Release binary, measured:

| build | size |
|---|---|
| default (lexical only) | **4.02 MB** |
| `distilled` | 7.07 MB |
| `auto-distilled` | 9.63 MB |
| all features | **9.89 MB** |

`auto-distilled` costs 2.57 MB over `distilled` — that is the TLS stack, and the
reason downloading is a separate feature from loading.

## Concepts

**A dimension** is one facet of a query — one field the model fills in. A single
index can carry several, and a single query can address any subset. Each carries
a `description`, which is what a model reads when deciding what to put in that
field. Dimensions are named for what the caller puts in them, not for the
technique behind them.

| dimension | what you put in the query |
|---|---|
| `lexical` | keywords |
| `declaration` | the name of the thing whose declaration you want |
| `semantic` | content resembling your target |
| `descriptive` | a description of your target |

**A record** is the only searchable unit: `{descriptor: text?, tensor:
numeric?, metadata}`, at least one non-empty. The format stores both payloads
opaquely and never interprets them, so BM25 needs no special case — token
frequencies are just what the lexical dimension puts in its descriptor text.

**A level** is retrieval altitude. Segments are level `0`; a fabricator's
clusters are `1` and up, each a coarser view of the level below. Clusters are
records too, carrying a child list instead of a span.

**A plugin** is a swappable stage implementation, named in that stage's `impl`
field. `structural`, `declaration`, `content`, and `llm` are classifier plugins; `bm25` and
`cosine` are scorers; `distilled` is an embedder; `agglomerative` is a
fabricator. Adding a retrieval method means writing a plugin, not changing the
format.

**A segment** is a line window — 40 lines with 8 of overlap by default, set by
`[segmenter]`. The overlap keeps anything defined at a window boundary attached
to its body. Segmentation is shared by every dimension: segment identity is the
join key that makes a hit in one dimension comparable to a hit in another.
Changing it is a config edit, and it re-segments the corpus, re-keying
everything derived from it.

## The four dimensions

### 1. `lexical` — keywords. Built in, offline, deterministic

```toml
[dimensions.lexical]
description = "Exact words, names, and literals as they appear in the text. Use this for anything you would otherwise search for verbatim."
enabled = true
classifier = { impl = "structural" }
```

One of the two dimensions `map init` writes. The classifier
tokenizes — splitting `camelCase` and `snake_case`, dropping single characters
— and BM25 scores a memory-mapped inverted index. Scores are normalized to
`[0,1]` against a per-query ceiling, so `1.0` keeps an absolute meaning
("every query term present and saturated") and stays comparable across queries.

> The classifier is a **tokenizer, not a parser**. Segmentation is by line
> window, and nothing here understands syntax, symbols, or imports.

The tokenizer has no stopwords and no stemming, and its word-character class is
`[alnum_]`, which splits hyphenated words and contractions. A different
tokenizer is a new classifier `impl`; the format needs nothing for one.

### 2. `declaration` — the name of the thing whose declaration you want. Built in, offline, deterministic

```toml
[dimensions.declaration]
description = "The exact name of the thing whose declaration you want -- a function, type, class, module, or constant. A name here ranks the place that declares it above the places that use it."
enabled = true
classifier = { impl = "declaration" }
```

The other dimension `map init` writes. The classifier keeps only the names a
segment declares: the word after `fn`, `def`, `class`, `struct`, `function`,
`type`, `impl ... for`, `const`, `mod`, and their relatives, plus the
`name(...) {` method shape. A segment that declares nothing has no record
here. BM25 scores the result, so a name in this field ranks the declaring
segment above every segment that only uses it, which is what BM25 over whole
content gets backwards: callers mention a name more often than its one
declaration does.

Fuse it with `lexical` rather than querying it alone: on the ripgrep golden
set the pair scores above either half, and the keyword anchors in that set
carry context words the name alone would drop. Put "what calls X" questions in
`lexical`, where the callers are the answer.

> This is a **line scan, not a parser**. It recognizes the declaration
> keywords most languages share and nothing else, so on prose it emits a
> little noise and on an unknown language it may miss. A grammar is a different
> `impl` behind the same field.

### 3. `semantic` — content resembling your target. Offline, no key, needs a model on disk

```toml
[dimensions.semantic]
description = "Content resembling what you are looking for -- paste or paraphrase the material itself."
enabled = true
classifier = { impl = "content", persist_output = false }
embedder = { impl = "distilled" }
```

The `content` classifier shapes a segment for the embedder and hands it on.
`persist_output = false` means the driver never stores what it produced. Any
stage can declare it.

Build with `--features distilled`. Embeddings come from
[`minishlab/potion-retrieval-32M`](https://huggingface.co/minishlab/potion-retrieval-32M)
(MIT) — a **static distilled** model: a token-embedding table lookup, mean
pool, L2 normalize. No transformer forward pass, so it is fast and needs no
GPU, but it is weaker than a full encoder.

The weights live at `~/.map/models/potion-retrieval-32M/` — `model.safetensors`
(129.2 MB), `tokenizer.json`, `config.json`. Build with `--features
auto-distilled` and fetch them:

```console
$ map model fetch
potion-retrieval-32M — 130.7 MB from minishlab/potion-retrieval-32M at 6fc8051fab2a
  config.json: downloaded
  tokenizer.json: downloaded
  model.safetensors: downloaded
potion-retrieval-32M ready at ~/.map/models/potion-retrieval-32M

$ map model status
potion-retrieval-32M installed at ~/.map/models/potion-retrieval-32M
```

Files come from one pinned revision and each is checked against a hardcoded
sha256 before it is installed, so a download is written to `<name>.part` and
renamed into place only after it verifies. An artifact already present with the
right digest is not transferred again; one whose digest does not match is
re-fetched, which repairs a corrupted directory. `--force` re-downloads
everything.

Measured on ripgrep: the fetch takes 13 s, embedding the 2,481 segments takes
3.6 s on a laptop CPU, and a query that includes `semantic` takes about 225 ms
end to end, since each process loads the model.

Without `auto-distilled`, place the three files yourself. Either way `map index`
never downloads — if the model is absent it **refuses to build** rather than
quietly omitting the dimension:

> ```
> map: this build cannot run 1 configured stage(s):
>   semantic.embedder = "distilled" — model not installed — run `map model fetch`
>
> Nothing was written. Pass --degraded=allow to build without them.
> ```

### 4. `descriptive` — a description of your target. Needs an endpoint and (usually) a key

```toml
[dimensions.descriptive]
description = "A description of what you are looking for, in your own words -- what it does, what it is about, what happens when it runs."
enabled = true
classifier = { impl = "llm", facets = ["purpose", "behaviour", "names[]", "subsystem"], prompts = { "0" = "...", "1" = "..." } }
embedder = { impl = "distilled" }
fabricator = { impl = "agglomerative", threshold = 0.70, min_cluster = 5, max_cluster = 15, max_levels = 3, min_remaining = 8 }
```

An LLM writes a descriptor for each segment; the distilled embedder embeds it.

**The prompt is where a corpus says what it holds.** MAP wraps it in a framing
that states only how many segments there are and what shape to answer in — it
names no subject matter, so an unedited prompt indexes prose and source code on
the same terms. The facets above are a *code* corpus's choice, not a built-in:
`purpose / behaviour / names[] / subsystem` suits a codebase, where a ticket
archive might want `problem / resolution / entities[] / product`.

Both the prompt and that framing are fingerprinted, so changing either
re-classifies rather than reusing stored answers. That re-bills the dimension.

Build with `--features "distilled llm"`, then `map llm login`. Any
OpenAI-compatible `/v1/chat/completions` endpoint works — vLLM, Ollama,
llama.cpp, LM Studio, or a hosted provider. The connection lives in
`~/.map/llm.toml`, **never** in `.map/config.toml`, which is committed:

```toml
endpoint = "http://localhost:8000/v1"
model    = "Qwen2.5-Coder-32B-Instruct"
# protocol = "openai-chat"   # the default; omit it
```

The endpoint must support **structured output** (`response_format` with a JSON
schema). If it accepts the field and ignores it, `map index` says so rather than
building a dimension full of blank descriptors.

Costs are real: an LLM call per *batch of segments* on first index, and
`map find -u` re-bills for a changed file.

#### `facets` — enforcing the answer shape

Without `facets`, a classifier asks for prose and gets a single string. With it,
each named part becomes a **required field of the JSON schema**, so the decoder
enforces the shape:

| spec | meaning |
|---|---|
| `"purpose"` | free text |
| `"names[]"` | a list of short strings, most important first, uncapped |
| `"names[8]"` | the same, capped at 8 |

A cap reaches the endpoint as `maxItems` **and** truncates when the parts are
composed into the stored descriptor. List parts are composed as bare
space-separated tokens rather than a sentence.

`facets` applies at **every** level, so an enforced answer shape makes cluster
labels multi-part too, not concise noun phrases.

#### `prompts` — one per level

`prompts` is keyed by fabric level. Level 0 describes a segment; each level
above summarizes a group from the level below. A level with no entry of its own
uses the nearest one declared below it, so the last prompt covers every level
above it and the tree can grow taller without a config change.

#### `fabricator` — the cluster tree

Adding a `fabricator` clusters the embeddings, labels each cluster, embeds the
labels, and reclusters — a labeled tree over the corpus, retrievable with
`--level`. Labeling runs through the *same* classifier that described the
segments, so a whole level costs one request rather than one per cluster.

| key | meaning |
|---|---|
| `threshold` | cosine similarity required to merge |
| `min_cluster` | drop a cluster smaller than this |
| `max_cluster` | skip a merge that would exceed this size (`0` = uncapped) |
| `max_levels` | how tall to build |
| `min_remaining` | stop when fewer than this many nodes remain |

Each is checked at load, not clamped: `impl` must be `agglomerative`, the
dimension must carry an embedder, `threshold` must be in `(0, 1]`,
`min_cluster` at least 2, `max_cluster` either `0` or at least `min_cluster`,
`max_levels` at least 1, and `min_remaining` at least 1.

### When a stage cannot run

Every stage a dimension names has to resolve — the plugin compiled in, the
endpoint configured, the model on disk, the implementation still existing under
that name. If any does not, `map index` and `map find -u` **report it and write
nothing**.

An interactive run asks; a non-interactive one refuses. Building without a stage
is legitimate — it is how you configure a semantic dimension before its endpoint
is live — and has to be asked for:

```console
$ map index --degraded allow
map: built without 1 configured stage(s) — this index is incomplete:
  semantic.embedder = "distilled" — model not installed — run `map model fetch`
```

The remedy names `map model fetch` only in a build that has it; a `distilled`
build says where to put the files instead.

An accepted partial build still reports what it left out.

## Querying

The positional argument is shorthand for the lexical dimension. `-d` addresses
any dimension by name, with an optional weight:

```console
$ map find "refresh_token"                                  # lexical shorthand
$ map find -d 'lexical=gitignore glob' -d 'descriptive=matching ignore rules' --level 0
crates/ignore/src/dir.rs:33  0.698  [descriptive 0.55, lexical 0.85]
crates/ignore/src/overrides.rs:97  0.568  [descriptive 0.37, lexical 0.76]

$ map find -d 'lexical:0.3=parse' -d 'descriptive:1.0=validates user input'
```

Dimensions are parameters of one query, not separate tools. The per-dimension
breakdown appears whenever more than one dimension scored. A weight of `:0`
excludes a dimension.

A score is absolute, but it is not a confidence that the answer exists. On
the ripgrep golden set the best hit for a topic the corpus does not contain
scores between 0.24 and 0.53 under `lexical`, and the median best hit for an
answered query scores 0.37. Read a low score as "weak match", not as "no
answer here".

| flag | meaning |
|---|---|
| `-d, --dim DIM[:WEIGHT]=TEXT` | Query one dimension; repeatable |
| `-n, --limit N` | Results to return (default 10) |
| `--level LEVELS` | Fabric levels to search (default: all) |
| `--members` | For a cluster hit, list the spans beneath it |
| `--snippet` | Print the matched text |
| `-u, --update` | Refresh the index before searching |
| `--degraded abort\|allow` | What to do if `-u` finds a stage it cannot run |
| `--root PATH` | Search an additional index; repeatable |
| `--path PATH` | Index root (default `.`) |
| `-q, --quiet` | Suppress the staleness notice |

### Retrieval altitude

**A query searches every level by default**, so one call returns precise spans
and cluster overviews in a single ranking, ranked together on score. Narrow it
by naming levels:

```console
$ map find "…" --level 0        # precise spans only
$ map find "…" --level 1,3      # just those two heights
$ map find "…" --level 0-2      # a span of heights
$ map find "…" --level 0,2-4    # mixed
$ map find "…" --level clusters # every level above 0
$ map find "…" --level segments # the same as 0
```

A cluster's own location is meaningless — it spans no file — so `--members`
resolves what it delivers:

```console
$ map find -d 'descriptive=parsing command line flags' --level clusters -n 1 --members
cluster L2  0.770  To implement and document command-line flags for ripgrep's functionality. …
    87 span(s) across 3 file(s)
    crates/core/flags/defs.rs  (79)
    crates/core/flags/mod.rs  (4)
    crates/core/flags/parse.rs  (4)
```

Resolving membership rebuilds the level-0 id map, which is O(corpus), so it is
opt-in.

### Federating across repositories

`--root` queries additional indexes and merges everything into one ranking,
each hit labelled with the repository it came from:

```console
$ map find -d 'descriptive=retry an HTTP request' --root ../other-service
```

Scores are normalized against the **combined** corpus, not per index: document
frequency, corpus size, and average document length are summed across
participants.

Origins must be unique, and the same path in two repositories is two
candidates. A dimension is fused only where `name + artifact_fingerprint`
match: two `descriptive` dimensions built with different prompts or embedders
are refused rather than averaged. An index that simply lacks a queried dimension
is skipped, not an error.

A user-level `~/.map/config.toml` may list standing roots, and every `map find`
adds them as if passed with `--root`:

```toml
[[roots]]
path = "C:/src/other-service"
```

### Inspecting a classifier

`map classify` runs a dimension's classifier over one file and prints the
model's **raw structured answer** next to the text that would be stored,
writing nothing:

```console
$ map classify crates/core/search.rs --dim descriptive -n 2
```

The stored descriptor is flattened, so an index alone cannot show which part
came back empty or whether the identifier list absorbed the prose.

| flag | meaning |
|---|---|
| `-d, --dim DIM` | Dimension whose classifier to run (default `descriptive`) |
| `-n, --limit N` | Stop after this many segments (default 4). One request either way |

### Staying current

`map find` does **not** update the index on its own. It reports drift, and `-u`
updates first:

```console
$ map find "the_new_symbol" -u
map: updated 2 object(s) before searching
crates/globset/src/glob.rs:1665  0.526
```

Without `-u`, an interactive run prints to stderr:

```
map: index is stale (1 changed) — run with -u to update before searching
```

The notice only prints when stderr is a terminal, so an agent shelling out never
sees it and never pays the `stat` walk that produces it.

The same notice and `-u` also fire when `config.toml` changed or the manifest
itself was replaced under an unchanged tree — a pull, a branch switch — not
only when a resource's own stat changed. A run that built without a configured
stage (`--degraded allow`) records no snapshot at all, so the next `-u` does
the work again rather than reporting a clean tree it did not fully build.

## Configuration

`.map/config.toml` is committed and human-editable. It is the input the index is
built from, so editing a dimension changes its fingerprint and invalidates
exactly the objects that dimension produced.

```toml
version = 1

[storage]
commit = "full"             # only `full` is implemented
location = "working-tree"   # only `working-tree` is implemented

[segmenter]
impl    = "window"          # only `window` is implemented
lines   = 40
overlap = 8                 # must be < lines

[dimensions.<name>]
description = "…"           # what a model reads when filling this field
enabled     = true
classifier  = { impl = "…", persist_output = true, … }
embedder    = { impl = "…", … }
fabricator  = { impl = "…", … }
levels      = [0, 1]        # optional; derived from the stages when omitted
```

Every dimension must produce a descriptor, a tensor, or both; one producing
neither is rejected at load. Keys beyond those listed are rejected rather than
ignored, as are dimension names that are not lowercase ASCII, digits, `_`, or
`-` — a name becomes a filename, `.map/cache/<name>.pack`.

Any key on a stage other than `impl` and `persist_output` is passed through to
the named implementation and folded into that dimension's fingerprint.

There is **no `scorer` key**. Which scorer runs follows from what the dimension
produces — BM25 over a descriptor, cosine over a tensor.

`[segmenter]` applies to every dimension. `overlap >= lines` and `lines = 0` are
refused.

There is no freshness policy to configure: `map index` and `map find -u` are the
only things that write.

`levels` is best omitted, and is then derived from the stages. Declaring *more*
levels than exist is harmless; declaring *fewer* than were built is refused.

**Unimplemented settings are refused, not ignored.** `commit` values other than
`full`, `location` values other than `working-tree`, and the `dims` and `quant`
dials are all rejected at config load.

Two dimensions that both use the `llm` classifier must declare identical
`prompts` and `facets` — one request covers the whole group, so there is no
way to serve two different answer shapes from it. Dimensions that disagree are
refused at index time, not silently reconciled to one of them.

Secrets never belong here. Nothing in the committed config takes a key: the LLM
endpoint, model, and credential live in `~/.map/llm.toml`, written by
`map llm login` and read only at index time.

## Commands

| command | what it does |
|---|---|
| `map init [PATH] [--force]` | Create `.map/`, enabling the lexical and declaration dimensions |
| `map index [PATH] [--degraded abort\|allow]` | Build or refresh the index |
| `map find …` | Search (see [Querying](#querying)) |
| `map classify PATH [-d DIM] [-n N]` | Show one file's classifier output; writes nothing (needs `llm`) |
| `map status [PATH]` | Record count and configured dimensions |
| `map gc [PATH] [--prune]` | Find objects no manifest reaches; delete with `--prune` |
| `map llm login` | Prompt for endpoint, model, and key, and cache them (needs `llm`) |
| `map llm status` | Show the cached connection (never the key) (needs `llm`) |
| `map model fetch [--force]` | Download the embedding weights (needs `auto-distilled`) |
| `map model status` | Report whether the model is installed, and where |
| `map merge BASE OURS THEIRS` | Three-way merge `.map/manifest.json`; run by git, not by hand |

## What gets committed

`.map/` is designed to be checked in. With the default two dimensions on
ripgrep that is 629 small files, 2.4 MB. `map init` writes a `.gitignore` that
excludes `cache/` (derived packs, rebuildable) and a `.gitattributes` marking
the index `linguist-generated`, so it collapses in pull-request diffs while
staying expandable and diffable.

Descriptors and cluster labels stay JSON and remain readable in review. A
segment's tensors are stored as raw little-endian frames, which decode about
47× faster on a cold clone; a cluster record keeps its tensor inside its JSON
record.

The manifest itself is written one entry per line — `dimensions`, `objects`,
`roots`, and `clusters` each get one line per key, sorted and two-space
indented, LF only — so two disjoint edits land on different lines, a textual
merge can resolve them, and a review diff shows exactly which entries changed.

Indexing twice produces **byte-identical** objects. Cloning the repository and
querying without indexing returns identical results and identical scores.

Objects are keyed by `hash(input + stage config)` — **input-addressed, not
content-addressed**. An unchanged resource skips reclassification and two
identical files share one object. Because the bytes cannot be verified by
re-hashing them, the manifest records a content hash per object separately.
Re-indexing checks every reused object against that recorded hash and refuses
the run if the bytes on disk no longer match, rather than re-recording altered
bytes under a freshly computed key — restore the object from version control,
or delete it so the stage re-runs.

### Merging

`map init` writes `.gitattributes` with `manifest.json merge=map`, naming
`map merge` as the manifest's git merge driver. `map init` and `map index`
also register the driver command in the repository's **local** git config
whenever `.git` exists, git runs, and the driver is not already set —
printing `map: registered git merge driver for .map/manifest.json` the first
time. Git will not execute a command a repository distributes, so the
attribute travels with a clone but the driver command has to be registered on
each machine.

The driver three-way merges manifests: object tables are unioned, since two
sides adding different objects is the ordinary case, but the same key recorded
with different bytes on each side is a conflict — union cannot resolve it.
Roots and dimension identities merge three-way per key. If both sides
independently re-fabricated the same dimension, its cluster tree is dropped
rather than picked arbitrarily; the cluster objects stay on disk, so the next
`map index` rebuilds the tree and reuses every unchanged subtree. The later
`generated_at` wins for provenance. On a conflict, the driver prints each one
to stderr, leaves the file untouched, and exits 1 so git marks it conflicted.

A clone that has not registered the driver gets git's ordinary text merge with
conflict markers instead — verified with git 2.55, this merges disjoint
dimension, object, and root edits but always conflicts on the provenance line,
since every `map index` run changes it.

### Collecting what nothing reaches

Editing anything that feeds a fingerprint — a prompt, an implementation, an
embedder — gives that dimension's objects **new keys**, and the old ones stay on
disk. Nothing is overwritten.

`map gc` finds them. It reports by default and deletes only with `--prune`:

```console
$ map gc
Reachability: .map is untracked; working tree is the only state
  reachable objects: 978
  unreachable:       2506 objects, 56.17 MB
    cluster     452 objects      3.54 MB
    desc       1027 objects      4.38 MB
    tensor     1027 objects     48.25 MB

Nothing deleted. Re-run with --prune to remove them.
```

**Reachability is the union of every manifest that could become current**, not
just the working tree's — a committed index commits its manifest, so each branch
carries its own. Git is consulted when it is there and is never required. A
repository whose git cannot be run refuses to prune.

Deleting a blob frees the working tree and future clones. It stays in the
history, and cloners still fetch it.

## Retrieval quality

File-nDCG@10 against ripgrep at pinned commit `8372866`. 47 hand-authored
queries, of which 41 carry judgments — the other 6 name concepts ripgrep does
not contain and are scored on how *little* confidence they draw. Method and
judgments in [`eval/README.md`](eval/README.md).

**This is a code result and does not generalize on its own.** One corpus, one
language, one author. The pipeline indexes any text, but nothing here measures
how it does on prose, transcripts, or records — and the two knobs most likely to
matter for them, the tokenizer and the segment size, are set to values tuned
here.

Every non-empty dimension subset, scored against one index. Reproduce with
`scripts/bench-implementations.sh ../ripgrep`.

| configuration | file nDCG | recall | MRR | ms/query |
|---|---|---|---|---|
| grep (coverage) | 0.453 | 0.643 | 0.440 | 17.28 |
| `lexical` | 0.535 | 0.745 | 0.561 | 2.28 |
| `declaration` | 0.459 | 0.599 | 0.474 | 0.50 |
| `semantic` | 0.593 | 0.720 | 0.569 | 4.24 |
| `descriptive` | 0.554 | 0.690 | 0.535 | 4.33 |
| `lexical` + `declaration` | 0.621 | 0.784 | 0.642 | 2.15 |
| `lexical` + `semantic` | 0.624 | 0.761 | 0.604 | 5.39 |
| `lexical` + `descriptive` | 0.641 | 0.773 | 0.648 | 5.57 |
| **`lexical` + `declaration` + `descriptive`** | **0.688** | 0.773 | **0.705** | 5.63 |
| all four | 0.646 | 0.759 | 0.648 | 8.13 |

- **`lexical` is the anchor.** Every pairing containing it beats both its
  halves.
- **`declaration` is weak alone and strong fused.** By itself it answers only
  queries that name a declared symbol; added to `lexical` it lifts every
  anchor query in the set to rank 1, for no measurable query cost.
- **All four is not the best configuration.** At 0.646 it sits below
  `lexical` + `declaration` + `descriptive` at 0.688, for 1.4× the query cost:
  the two embedding dimensions largely agree, so the third dilutes.
- **Component queries are the weak spot.** Segment retrieval spends several of
  its ten slots on different segments of the same file, so it returns fewer
  distinct files than a plain text search does.
- `descriptive` is **nondeterministic**: it is authored by a model, so any row
  involving it moves between indexing runs (the same configuration has scored
  0.612 and 0.554 on two generations of descriptors) and is not frozen as a
  regression baseline. The `lexical` and `lexical` + `semantic` rows are, and
  reproduce bit-exactly.

## Measured

Windows 11, ripgrep at pinned commit `8372866`, all four dimensions
configured — 2,627 records over 229 resources.

Reachable objects, by determinism tier (spec §6):

| tier | holds | objects | size |
|---|---|---|---|
| A | resources, spans, `lexical` and `declaration` descriptors | 618 | 2.16 MB |
| B | embeddings | 206 | 10.18 MB |
| C | LLM descriptors and cluster records | 352 | 2.20 MB |
| **total** | | **1,176** | **14.54 MB** |

Query latency splits into three parts:

| | lexical | all four |
|---|---|---|
| Process start | ~9 ms | ~9 ms |
| Pack load, once per process | ~140 ms | ~140 ms |
| Per query, after load | **2.28 ms** | **8.13 ms** |
| End to end, cold process | ~163 ms | ~176 ms |

The pack load is paid once per process, so a long-lived caller amortizes it and
a shell loop does not.

A cached pack is trusted only when a per-machine keyed marker over the
manifest fingerprint, the dimension name, and the pack bytes matches what is
on disk; otherwise the pack is rebuilt from the verified objects. Hashing all
three of ripgrep's packs — 12.8 MB — costs 8.2 ms per process, the price of
that check.

Binary: **4.02 MB** default, **9.89 MB** with all features — see
[Install](#install) for the per-feature breakdown.

## Development

Corpus repositories are cloned as **siblings** of this one:

```
software/
├─ map/        this repo
├─ ripgrep/    pinned corpus
└─ flask/      pinned corpus
```

```console
$ scripts/fetch-corpus.ps1      # Windows
$ scripts/fetch-corpus.sh       # POSIX
$ cargo fmt --all -- --check
$ cargo clippy --workspace --all-targets
$ cargo test --workspace
```

Pins are exact commits, not branches — `eval/qrels.jsonl` judges specific line
ranges and `eval/baseline-*.json` freezes the scores they produce.

### The fingerprint ledger

`crates/map-index/ledger.jsonl` records, for each stage, the identity string
that keys its stored output alongside a digest of what it produced over a fixed
inline fixture. A stage whose behaviour changes while its identity does not
fails `cargo test`. Entries are append-only.

Regenerate a genuinely new identity with `MAP_LEDGER_APPEND=1 cargo test -p
map-index`; it appends missing keys and refuses to overwrite an existing one.
`cargo test -p map-index --lib inspect_the_fixture -- --ignored --nocapture`
prints the raw stage output the digests summarize.

CI runs fmt, clippy, and tests on Linux/macOS/Windows with `-D warnings`, an
MSRV `check` at 1.88 with all features, and a `cargo audit` pass. **Tests run
with default features only**; the `distilled` and `llm` code is compile-checked
by a separate all-features clippy step but its tests need the potion weights and
so do not run in CI.

`map-eval` scores an index against the golden set:

```console
$ map-eval ../ripgrep --dim lexical --dim descriptive   # any dimension set
$ map-eval ../ripgrep --grep                            # the grep baseline
$ map-eval ../ripgrep --level clusters                  # score cluster hits
$ map-eval ../ripgrep --root ../other-corpus            # federated
```

## Design

```
discover → preprocess → segment → classify → embed → fabricate → retrieve
                                      ▲         ▲
                        dimensions travel together as a bundle;
                        the orchestrator groups them by implementation
                        and invokes each once
```

[`spec/format-v1.md`](spec/format-v1.md) is the normative description of the
on-disk format.

## License

Apache-2.0. See [LICENSE](LICENSE).
