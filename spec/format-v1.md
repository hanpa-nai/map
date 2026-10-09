# `.map` Format — Version 1

The normative description of what a `.map` directory contains and what a
producer and consumer of one must do. **MUST**, **MUST NOT**, **MAY**, and
**SHOULD** carry their usual meaning.

The format is not yet frozen; a consumer should check the `version` field in
`config.toml` and the format version in `manifest.json` and refuse anything it
does not recognize.

## 1. Scope

A `.map` directory is a portable index of the resources of exactly one
repository or resource root. It sits alongside those resources the way `.git`
does. It is **git-friendly but not git-dependent** — a `.map` moved by tarball,
rsync, or object storage is equally valid.

A `.map` indexes **only the resources stored in its own root.** Spanning
multiple roots is composition (§7), never a wider index.

### 1.1 Resource neutrality

A **resource** is an addressable unit of text. The format defines no resource
kind, and a conforming producer MUST NOT vary the layout, the object key
derivation, or the record schema by one. Source code, documentation, notes,
transcripts and exported records are all indexed on identical terms.

Where a stage must be tuned to a kind of material — the vocabulary a tokenizer
splits on, the unit a segmenter cuts, the subject matter a classifier prompt
names — that tuning MUST live in `config.toml` or in the stage implementation
selected by it, never in the format. Two consequences a producer MUST honour:

1. **Anything that shapes stored bytes is fingerprinted.** This includes text a
   stage supplies itself rather than reading from config, such as a built-in
   prompt or the framing wrapped around a configured one. A stage MUST NOT send
   a model any instruction that no fingerprint covers.
2. **Segmentation is one setting for the whole root**, not per dimension.
   Segment identity is the join key that makes a hit in one dimension
   comparable to a hit in another.

## 2. Directory layout

```
.map/
  config.toml            committed   dimensions, stage selection, policy
  manifest.json          committed   object keys + content hashes, dimension
                                     identity, built levels, provenance
  index/
    shared/objects/ab/cd…            resources and segment spans, shared by
                                     every dimension
    desc/objects/ab/cd…              descriptor groups, one per classifier
                                     invocation
    tensor/objects/ab/cd…            embeddings
    cluster/objects/ab/cd…           cluster records
  cache/                 ignored     packs, mmap views, the stat snapshot
  .gitignore                         written by `map init`
  .gitattributes                     written by `map init`
```

An object root is created on first write, so a lexical-only index has no
`tensor/` or `cluster/` directory at all. Object files are stored loose, fanned
out by the **first two hex characters** of the key. Everything under `cache/` is
derived and disposable: deleting it MUST never lose information, only cost time.

A dimension's name becomes a filename — `cache/<name>.pack` — so names are
restricted to lowercase ASCII letters, digits, `_`, and `-`. A name that would
traverse out of the directory is rejected at config load.

## 3. The record

The record is the unit the retriever searches over. Segments and clusters are
the same type; nothing else is searchable.

```
record = { descriptor: text?, tensor: numeric?, metadata }
```

- `descriptor` is a **freeform text field**.
- `tensor` is a **freeform numeric field**.
- **The format never interprets either.** Both are opaque payloads. Only the
  owning dimension's stage implementations know what they mean: the classifier
  writes them, the retriever's scorer reads them.
- **At least one MUST be non-empty.** A dimension that would produce neither is
  a configuration error and MUST be rejected at load.

A new retrieval method is a new stage implementation, never a format change.
BM25 needs no posting-list type: term frequencies are what the lexical dimension
places in its descriptor text.

### 3.1 The level ladder

Every record carries a **level**. Level 0 is a segment; each level above
summarizes a group from the level below. Records at every level live in the
same pool and are scored on the same path — "zoom" is a filter on level, not a
separate mechanism.

**Levels are a pre-filter within a dimension, not a dimension of their own.** A
dimension MAY declare which levels it searches; the candidate set for that
dimension is the intersection of the query's level scope with the dimension's
declared levels. An empty intersection means the dimension contributes nothing
and stays **out of the fused denominator**. A dimension that holds a record at
that altitude and did not match is a real zero and still counts.

Declaring **more** levels than a dimension builds is harmless: a level holding
no records yields no candidates. Declaring **fewer** than were built is refused
at load.

Not every dimension has levels above 0. A dimension whose records are raw
content has no meta-resources to summarize and is flat by construction.

A record above level 0 is a **cluster**: its metadata carries a child list and
its kind is `Cluster`. A segment record MUST NOT carry children, and a cluster
MUST NOT carry a span. Producing a cluster is the same call as producing a
segment — the classifier is handed the texts the record is built from, which is
a segment's own content at level 0 and a group's descriptors above it — so only
the prompt differs by level.

Cluster identity is the one thing that differs:

```
cluster_id = hash(sorted(child_ids) + dimension_config_fingerprint + dimension_name)
```

Bottom-up, so an unchanged subtree keeps its identity, its labels are reused
verbatim, and its stored bytes are untouched. The **name** is included here even
though it is excluded from the artifact fingerprint elsewhere (§4): without it,
two dimensions with byte-identical stage config derive identical child ids and
collide on one key.

A level-0 record's identity is content-sensitive the same way:

```
segment_id = hash(resource_key + start + end + segments_object_key + dimension_fingerprint)
```

The segments object key derives from the resource's normalized content, so
editing a resource changes every leaf id it contains and re-keys the clusters
above them; an unchanged resource keeps its leaf ids. A span's offsets alone do
not identify a leaf.

Clustering is agglomerative on a cosine threshold, so membership is hard rather
than soft, and every level ranks in one pool.

## 4. Object identity and integrity

Objects are keyed by:

```
object_key = hash(input + stage_config_fingerprint)
```

This is **input-addressed, not content-addressed.** The distinction MUST be
preserved in documentation and code comments. Two consequences follow:

1. An object's bytes **cannot** be verified by hashing them. The manifest
   therefore records a **content hash per object**, which provides integrity and
   collision detection.
2. Two producers may generate the **same key with different bytes** (see §6).
   Union is not a valid resolution. Collision semantics MUST be defined
   explicitly — manifest-side resolution with provenance recorded.

**Object granularity equals invocation granularity.** Descriptors from a single
classifier invocation form one object keyed by
`hash(content + impl_config + its_dimension_set + segmenter_config)`, so editing
one dimension's prompt invalidates exactly the group sharing its implementation.

The segmenter belongs in that key, and a producer that omits it is
non-conforming: the payload holds one record **per segment**, so re-cutting the
same content is a different invocation with a different answer. The same applies
to a tensor object, whose payload is likewise per-segment even when the embedder
reads raw segment text and no descriptor exists.

Writing an object whose key already exists with **different** bytes MUST be
refused rather than silently unioned or overwritten. Rewriting identical bytes
is a no-op.

## 5. Manifest

The manifest MUST carry:

- Object roots and, per object, its input key **and** content hash
- **Dimension identity as `name + artifact_fingerprint`**, never name alone
- **The levels actually built** per dimension, so a consumer can tell what is
  on disk from what was merely configured
- **Provenance** — the producing implementation's version and a timestamp

A producer MUST write each object-map field (`dimensions`, `objects`, `roots`,
`clusters`) with one entry per line, keys in sorted order. Line-level
granularity is what lets two disjoint edits merge textually (§9.5) and keeps a
review diff to the entries that actually changed.

The manifest carries nothing else. An optional field can be added whenever a
producer is ready to write it, without breaking a reader that predates it.

**Credentials never enter the committed config.** `config.toml` is committed, so
no stage setting carries a secret value; the LLM endpoint, model, and key live in
user-global configuration outside the index.

The determinism digest covers the object set and the built levels. It excludes
provenance, which varies per run without changing what the index holds.

**An interrupted `map index` MUST NOT leave a partially-described index.** The
manifest is written once, atomically, after every object has landed, so an
interrupted run leaves the *previous* manifest intact and the objects it wrote
unreferenced — a collection problem (§9.4) rather than a corrupt index.

## 6. Determinism tiers

Byte-identical output **across machines is not achievable** for every stage and
nothing may assume it. Model classifiers are nondeterministic even at
temperature 0; float inference varies with SIMD path and thread count; and
agglomerative clustering amplifies one perturbed similarity into a different
dendrogram.

| Tier | Covers | Guarantee |
|---|---|---|
| **A — required** | discover, preprocess, segment, structural and declaration classify, BM25, all serialization and manifest encoding, and clustering *given identical embedding bytes* | bit-identical, on one machine and across platforms |
| **B — producer-authoritative** | embeddings, clusters | the committed artifact *is* the index; consumers verify, never re-derive |
| **C — authored** | model-api descriptors, cluster labels | no reproducibility expected; provenance required |

Tier A requires: no `HashMap`/`HashSet` iteration order anywhere near serialized
output (use `BTreeMap`/`BTreeSet` or sort explicitly), deterministic
tie-breaking on equal similarities, deterministic reduction order.

**Everything a default build produces is Tier A.**

Cross-machine reproducibility is replaced by **verifiability**: the content
hashes of §4 establish that object bytes are what the manifest says, cluster
children resolve to records that exist, and Tier A stages can be re-derived
from source and compared.

### 6.1 Cross-platform normalization

Tier A claims bit-identical output across operating systems. Four platform
differences would break that for a **default build**. The preprocessor MUST
normalize all four, and the normalization MUST be part of its config
fingerprint.

1. **Line endings.** Text resources MUST be normalized to LF before hashing and
   segmentation. Byte offsets recorded in segment spans are offsets into the
   **normalized** content; any consumer resolving a span back to an on-disk file
   must re-normalize first.
2. **Path separators and case.** Resource keys MUST be repository-relative,
   forward-slash separated, and case-preserved exactly as recorded — never
   case-folded, never `\`-separated.
3. **Unicode filename normalization.** Resource keys MUST be normalized to NFC
   before hashing.
4. **Directory iteration order.** The discoverer MUST sort its output by the
   canonical resource key before anything downstream consumes it.

Trailing-whitespace and final-newline differences are **not** normalized: those
are real content and belong in the index.

Indexing the same corpus on Linux, macOS and Windows MUST produce byte-identical
Tier A objects on all three.

This is a requirement the format places on a producer, not a claim about any
particular implementation's test suite. The reference implementation currently
checks determinism at the unit level on all three platforms — canonical
encoding, descriptor encoding, tensor encoding, clustering, and the manifest
determinism digest — and does **not** yet run an end-to-end cross-platform
comparison of a full index. Treat this section as normative and unmet rather
than normative and verified.

## 7. Composition

- A superproject **automatically discovers and loads each submodule's `.map`**.
  Composition follows repository structure; no configuration required.
- Additional roots MAY be listed explicitly, or passed per query.
- The exposed query surface is the **union of dimensions** across loaded
  indices. Each index scores only on the parameters it has and ignores the
  rest. Compatibility is keyed on `name + artifact_fingerprint` — the artifact
  fingerprint, not the whole-config one, so that editing a dimension's
  model-facing `description` does not make two indices incomparable. Same name,
  different fingerprint is refused rather than fused.
- Origins MUST be unique across a federation, and every hit carries the origin
  it came from, so the same path in two repositories is two candidates.
- Results merge against **combined corpus statistics**. Per-index normalization
  is not sufficient for a corpus-dependent scorer. Document frequency, corpus
  size, and average document length are summed across the participating indices
  and every index scores against the totals.
- An index that lacks a queried dimension is skipped, not an error. Federating
  a single index MUST give the same result as querying it alone.

## 8. Trust model

A committed index is **untrusted data that steers a model's attention.**
Descriptors and cluster labels enter model context as retrieval results, so a
malicious descriptor is a prompt-injection payload with a persistent home in
the repository.

- `.gitattributes` uses `linguist-generated=true` **only**. Not `-diff`.
  Index changes must remain reviewable; collapsed-but-expandable is the goal.
- Reviewability protects **authored** payloads. Descriptors and cluster labels
  are stored as text and MUST stay reviewable. Tensors are stored as raw frames
  (§9.2). Content hashes, not readability, are what detect tampering in either
  case.
- Content hashes (§4) make tampering detectable. A decoder MUST reject a
  truncated, over-long, or otherwise malformed object rather than panicking or
  over-allocating on it.
- Descriptors are Tier C and cannot be cheaply re-derived, so **provenance is
  the control**, not recomputation.

## 9. Storage policy

| Setting | Values | Implemented |
|---|---|---|
| `dims` | matryoshka truncation: 768 / 512 / 256 / 128 | no |
| `quant` | `f16` \| `i8` \| `binary` | no |
| `commit` | `full` \| `binary` \| `descriptors-only` \| `none` | `full` only |
| `location` | `working-tree` \| `ref` | `working-tree` only |

**A producer MUST reject a setting it does not honour**, rather than accepting
and ignoring it. This applies to every row above without exception.

`dims` and `quant` change *artifact* bytes and feed `artifact_fingerprint`, so
an index built at a different setting is a distinct identity. That is not a
reason to accept them while unimplemented.

### 9.1 Committed form

Objects are committed loose, under the two-hex-character fan-out of §2. Packs
and mmap views are derived and live in `cache/`, which is gitignored.

A pack MUST be a deterministic function of its member set, so that disjoint
edits land in different packs and merge cleanly, and a same-pack conflict is
resolved by regenerating the pack from the union of members. A pack written by
a different format version is rebuilt rather than refused.

### 9.2 Tensor encoding

A segment's tensor payload is stored as **raw little-endian frames** — the
`MAPT` frames of §3, encoded by `map_core::encode_tensor_payload`. A cluster
record carries its tensor inside its JSON record, as a byte array.

A decoder MUST also accept the JSON array encoding, so indexes written before
the binary encoding keep loading. An object key derives from the embedder
config and dimension set rather than from the serialization, so re-indexing
reuses those objects rather than rewriting them.

Descriptors and cluster labels remain JSON (§8).

A tensor MUST declare a shape matching its byte length, and a decoder MUST
reject one that does not — including a hostile shape that would wrap around
into a plausible empty tensor.

### 9.3 Unresolvable stages

A dimension names an implementation per stage, and a producer resolves what its
build and environment allow: a plugin may not be compiled in, an endpoint may
not be configured, a model may not be installed, or a named implementation may
no longer exist.

**An unresolvable stage MUST NOT be skipped silently.** A partial index is
indistinguishable from a complete one at query time.

A producer encountering one MUST report every affected dimension and stage, and
MUST NOT write objects unless it has explicit consent to build without them.
Consent may be interactive or a flag; **absence of a terminal is not consent.**
Where consent is given, the incompleteness MUST still be reported once the run
finishes.

### 9.4 Collection

Object keys are input-addressed (§4), so editing anything that feeds a
fingerprint yields new keys and leaves the previous generation on disk. An index
accumulates until something collects it.

**The reachable set is the union over every manifest that could become current,
not the working tree's alone.** A committed index commits its manifest, so each
ref carries its own.

A collector MUST union the working-tree manifest with every manifest reachable
from a ref, where refs exist. Where they cannot be enumerated — a repository
whose VCS cannot be run — reachability is *unknown* rather than empty, and it
MUST refuse to delete. Where there is no repository, or the manifest is
untracked, the working tree's manifest is the only one and tracing it is
complete.

A manifest that cannot be parsed MUST also refuse.

**Collection does not reclaim history.** Deleting a blob removes it from future
checkouts; it remains in every pack and every cloner still fetches it.

### 9.5 Merging

Two manifests produced from a common ancestor MUST be resolved by a three-way
merge, not by textual merge alone: objects are keyed by content (§4), and a
line-based merge cannot distinguish a legitimate union of object tables from a
same-key/different-bytes collision.

A merge MUST union the object tables — two sides adding different objects is
the ordinary case — and MUST refuse, rather than silently pick one or overwrite,
the same key recorded with a different content hash on each side. Roots and
dimension identities merge three-way per key.

A merge MAY drop a dimension's cluster tree when both sides independently
re-fabricated it: neither tree then describes the merged object set, and
choosing one would be arbitrary rather than merely lossy. The cluster objects
themselves are left on disk, so a subsequent build reuses every unchanged
subtree. Dropping a tree this way MUST reduce the levels recorded for that
dimension to the levels still present.

## 10. Freshness

Nothing updates an index implicitly. A producer writes only when asked, and the
format carries no policy field saying otherwise.

Because repairing semantic dimensions requires a classifier endpoint,
**staleness MUST be signaled per-result** in the query protocol.
