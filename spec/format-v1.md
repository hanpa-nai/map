# `.map` Format — Version 1

This document is the normative description of a `.map` directory. It gives the
contents of the directory, and the rules for a producer and a consumer.
**MUST**, **MUST NOT**, **MAY**, and **SHOULD** have their usual definitions.

**Status: draft.** The format can change. Recommendation: a consumer reads the
`version` field in `config.toml` and the format version in `manifest.json`. It
rejects a version that it does not know.

## 1. Scope

A `.map` directory is an index of the resources of one repository or one
resource root. You can move it with the resources. Its location is the root,
next to the resources, the same as `.git`. **A `.map` is compatible with git,
but git is not necessary.** A `.map` that you move with a tarball, rsync, or
object storage is equally correct.

**A `.map` includes only the resources in its root.** To search more than one
root, use composition (§7). Do not make an index that includes more than one
root.

### 1.1 Resource neutrality

A **resource** is a unit of text that has an address. The format has no
resource types. A producer that conforms to this specification MUST NOT change
the layout, the object key derivation, or the record schema for a type of
resource. Source code, documentation, notes, transcripts, and records from
other systems all get the same procedure.

A stage can have an adjustment for a type of material. Examples are:

- the vocabulary that a tokenizer uses to split text
- the unit that a segmenter cuts
- the subject that a classifier prompt gives

That adjustment MUST be in `config.toml`, or in the stage implementation that
`config.toml` selects. It MUST NOT be in the format. As a result, a producer
MUST obey two rules:

1. **A fingerprint includes each input that changes stored bytes.** This
   includes text that a stage supplies and does not read from the
   configuration. Examples are a built-in prompt, and the frame text that a
   stage puts around a configured prompt. A stage MUST NOT send a model an
   instruction that is not in a fingerprint.
2. **One segmenter setting applies to the full root.** A dimension cannot have
   a different setting. The segment identity connects the dimensions. With it,
   a consumer can compare a hit in one dimension with a hit in a different
   dimension.

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

A producer makes an object root at the first write to it. Thus an index that
has only BM25 dimensions has no `tensor/` or `cluster/` directory. A producer
stores each object as one loose file. The **first two hex characters** of the
key give the name of the subdirectory.

All data in `cache/` is derived data. A user can delete `cache/` at all times.
The delete operation MUST NOT remove information that a consumer cannot build
again. It only adds time to the next operation.

The name of a dimension becomes a file name: `cache/<name>.pack`. Thus a name
can contain only lowercase ASCII letters, digits, `_`, and `-`. When an
implementation loads the configuration, it rejects a name that points to a
location that is not in the directory.

## 3. The record

The record is the unit that the retriever searches. A segment and a cluster are
the same type of record. The retriever searches no other unit.

```
record = { descriptor: text?, tensor: numeric?, metadata }
```

- `descriptor` is a **text field with no specified structure**.
- `tensor` is a **numeric field with no specified structure**.
- **The format does not read the contents of these two fields.** Only the stage
  implementations of the dimension know the contents. The classifier and the
  embedder write the fields, and the scorer reads them.
- **The descriptor, the tensor, or the two MUST have content.** A dimension
  that makes no descriptor and no tensor is a configuration error. An
  implementation MUST reject it when it loads the configuration.

A new retrieval method is a new stage implementation. It is not a format
change. BM25 uses no posting-list type, because the `lexical` dimension puts
the term frequencies in its descriptor text.

### 3.1 The level ladder

Each record has a **level**. A level 0 record is a segment. A record at a
higher level is a summary of a group of records from the level below it. The
records of all levels are in one pool, and one code path gives them scores. A
level selection is a filter, not a different search procedure.

**A level is a filter in a dimension. It is not a dimension.** A dimension MAY
declare the levels that it searches. The candidate set for that dimension is
the intersection of two sets: the levels of the query and the declared levels
of the dimension.

- If the intersection is empty, the dimension gives no score, and it is **not
  in the denominator of the fused score**.
- If a dimension has a record at a level of the query and the record does not
  match, that score is a zero that counts. It stays in the denominator.

A dimension can declare **more** levels than the producer built. This is safe,
because a level with no records gives no candidates. If a dimension declares a
**smaller number of** levels than the index contains, the index does not load.

Some dimensions have no levels above 0. If the records of a dimension are the
content of the segments, there are no summaries to put in groups. Such a
dimension has only level 0.

A record above level 0 is a **cluster**. Its metadata has a child list, and its
`kind` is `Cluster`. A segment record MUST NOT have children. A cluster MUST
NOT have a span.

A producer makes a cluster with the same classifier call that makes a segment
record. The classifier gets the source texts of the record. At level 0, the
source text is the content of the segment. At a higher level, the source texts
are the descriptors of the group. Thus only the prompt is different between
levels.

The identity of a cluster has a different derivation:

```
cluster_id = hash(sorted(child_ids) + dimension_config_fingerprint + dimension_name)
```

The derivation goes from the leaves up. Thus a subtree that does not change
keeps its identity, its labels, and its stored bytes. The derivation includes
the dimension **name**. The artifact fingerprint (§5) does not include it.
Without the name, two dimensions with the same stage configuration bytes get
the same child ids, and they use one key for two different objects.

The identity of a level 0 record also changes with the content:

```
segment_id = hash(resource_key + start + end + segments_object_key + dimension_fingerprint)
```

The key of the segments object comes from the normalized content of the
resource. Thus an edit to a resource changes each leaf id in that resource, and
it changes the keys of the clusters above those leaves. A resource that does
not change keeps its leaf ids. The offsets of a span are not sufficient to
identify a leaf.

The fabricator makes clusters with an agglomerative method and a cosine
threshold. Thus a record is a member of a maximum of one cluster at each level,
and the records of all levels rank in one pool.

## 4. Object identity and integrity

The key of an object is:

```
object_key = hash(input + stage_config_fingerprint)
```

This key is **input-addressed, not content-addressed.** Documentation and code
comments MUST keep these two terms apart. The difference has two results:

1. The key is not a hash of the object bytes. Thus a consumer **cannot** verify
   an object against its key. The manifest records a **content hash for each
   object**. The content hash gives integrity, and it lets a consumer detect a
   collision.
2. Two producers can make the **same key with different bytes** (see §6). A
   union does not resolve this collision. The collision rules MUST be explicit:
   the manifest resolves the collision, and it records provenance.

**One classifier call makes one object.** The descriptors from one classifier
call are one object. Its key is
`hash(content + impl_config + its_dimension_set + segmenter_config)`. Thus an
edit to the prompt of one dimension invalidates only the group of dimensions
that use the same implementation.

The segmenter configuration is a necessary part of that key. A producer that
omits it does not conform to this specification. The payload has one record
**for each segment**. Thus a different cut of the same content is a different
call with a different result.

The same rule applies to a tensor object. Its payload also has one entry for
each segment. This is also correct when the embedder reads the segment text
directly and there is no descriptor.

A producer MUST NOT write an object when its key is in the store with
**different** bytes. It MUST NOT merge the two objects, and it MUST NOT replace
the stored object. It MUST give an error. A write of the same bytes changes no
data.

## 5. Manifest

The manifest MUST contain:

- the object roots and, for each object, its input key **and** its content hash
- **the identity of each dimension as `name + artifact_fingerprint`**, not only
  the name
- **the levels that the producer built** for each dimension. With this data, a
  consumer knows which levels the index contains and which levels are only in
  the configuration.
- **provenance**: the version of the producer implementation and a timestamp

A producer MUST write each of these fields with one entry on each line and the
keys in sorted sequence: `dimensions`, `objects`, `roots`, `clusters`. Each
entry is on a different line. Thus a text merge can merge two edits that have
no overlap (§9.5), and a review diff shows only the entries that changed.

The manifest contains no other data. A producer can add an optional field at a
subsequent time, and a reader that does not know the field continues to
operate.

**Credentials are not in the committed configuration.** `config.toml` is a
committed file. Thus no stage setting contains a secret value. The LLM
endpoint, model, and key are in the user configuration, which is not in the
index.

The determinism digest includes the object set and the built levels. It does
not include provenance, because provenance changes on each run and the index
content does not.

**When an index build stops before the end, the index on disk MUST NOT have a
manifest that describes only part of it.** A producer writes the manifest one
time, in one atomic operation, after it writes all objects. Thus a build that
stops before the end does not change the previous manifest. No manifest refers
to the objects from that build. They are a task for collection (§9.4), not a
damaged index.

## 6. Determinism tiers

Some stages **cannot give byte-identical output on different machines**. An
implementation must not use byte-identical output from all stages as a fact. The
causes are:

- A model classifier is nondeterministic, also at temperature 0.
- Float inference changes with the SIMD path and the thread count.
- With an agglomerative cluster method, one small change in a similarity can
  cause a different dendrogram.

| Tier | Includes | Guarantee |
|---|---|---|
| **A — mandatory** | `discover`, `preprocess`, `segment`, the `structural` and `declaration` classifiers, BM25, all serialization, the manifest encoding, and the cluster stage *when the embedding bytes are the same* | bit-identical output on one machine and on all platforms |
| **B — the producer output is the reference** | embeddings, clusters | the committed artifact *is* the index; a consumer verifies it and does not derive it again |
| **C — written by a model** | descriptors from a model API, cluster labels | no reproducibility; provenance is mandatory |

Tier A has these requirements:

- Serialized output does not change with the iteration sequence of a `HashMap`
  or a `HashSet`. Use `BTreeMap` or `BTreeSet`, or sort the entries.
- Equal similarities have a deterministic tie-break rule.
- Reductions have a deterministic sequence.

**All output of a default binary is Tier A.**

Tiers B and C do not have reproducibility between machines. **Verification
replaces it:**

- The content hashes of §4 show that the object bytes agree with the manifest.
- The children of each cluster point to records that are in the index.
- A consumer can derive Tier A output again from the source and compare it.

### 6.1 Cross-platform normalization

Tier A gives a guarantee of bit-identical output on all operating systems. Four
platform differences can break that guarantee for a **default binary**. The
preprocessor MUST normalize all four. The normalization MUST be part of its
configuration fingerprint.

1. **Line ends.** A producer MUST normalize text resources to LF before it
   hashes them or cuts them into segments. The byte offsets in a segment span
   are offsets into the **normalized** content. A consumer that converts a span
   back to a location in a file on disk must normalize the file first.
2. **Path separators and uppercase letters.** A resource key MUST be a path
   relative to the repository root. It MUST use `/` as the separator, and it
   MUST NOT use `\`. It MUST keep each uppercase letter and each lowercase
   letter of the recorded path.
3. **Unicode file names.** A producer MUST normalize a resource key to NFC
   before it hashes the key.
4. **Directory sequence.** The discoverer MUST sort its output by canonical
   resource key before a subsequent stage reads it.

A producer does **not** normalize whitespace at the end of a line, or a
difference in the last newline. These are part of the content, and the index
includes them.

An index build on the same corpus MUST make byte-identical Tier A objects on
Linux, macOS, and Windows.

**Status of the reference implementation.** This section is a requirement of
the format on a producer. It does not describe the tests of an implementation.

The reference implementation has determinism tests at the unit level on all
three platforms. These tests include canonical encoding, descriptor encoding,
tensor encoding, the cluster stage, and the manifest determinism digest. It
does **not** compare a full index between platforms. Thus this section is
normative, and no test shows that the reference implementation obeys it for a
full index.

## 7. Composition

- A superproject **finds and loads the `.map` of each submodule
  automatically**. Composition follows the repository structure, and no
  configuration is necessary.
- A user MAY give a list of more roots in a configuration file, or give them
  for one query.
- The query interface is the **union of the dimensions** of the loaded indexes.
  Each index gives scores only for the dimensions that it has, and it ignores
  the others.
- Two dimensions are compatible when `name + artifact_fingerprint` are equal.
  The rule uses the artifact fingerprint, not the fingerprint of the full
  configuration. Thus an edit to the `description` of a dimension does not
  change the compatibility of two indexes. A consumer does not fuse two
  dimensions that have the same name and different fingerprints. It gives an
  error.
- Each origin in a federation MUST be different from all other origins. Each
  hit contains its origin. Thus the same path in two repositories is two
  candidates.
- A consumer merges results with the **statistics of all indexes together**.
  Normalization for each index independently is not sufficient for a scorer
  that uses corpus statistics. The consumer calculates three values for all
  indexes in the federation together: the document frequency, the corpus size,
  and the average document length. Each index then calculates scores from
  those values.
- A consumer ignores an index that does not have a dimension of the query. This
  is not an error. A federation of one index MUST give the same result as a
  query on that index without a federation.

**Status of the reference implementation.** The reference implementation does
not find submodules. It loads only the roots that the caller gives and the
roots in the user configuration.

## 8. Trust model

A committed index is **untrusted data, and it controls the text that a model
reads.** Descriptors and cluster labels go into the model context as search
results. Thus a dangerous descriptor is a prompt-injection payload that stays
in the repository.

- `.gitattributes` uses **only** `linguist-generated=true`. It does not use
  `-diff`. It must be possible for a reviewer to read index changes. The
  purpose: a diff view shows the index diff closed, and a reviewer can open it.
- Review gives protection to the payloads that a model writes. A producer stores
  descriptors and cluster labels as text, and it MUST be possible to read them
  in a review. A producer stores segment tensors as binary frames (§9.2). For
  the two types of payload, content hashes detect a change to the bytes. Text
  that a person can read does not.
- Content hashes (§4) let a consumer detect a change to the bytes. A decoder
  MUST reject an object that is too short, too long, or malformed. It MUST NOT
  panic or allocate too much memory when it reads such an object.
- Descriptors are Tier C, and a second classifier run has a cost. Thus
  **provenance is the control**, not a second classifier run.

## 9. Storage policy

| Setting | Values | Implemented |
|---|---|---|
| `dims` | matryoshka truncation: 768 / 512 / 256 / 128 | no |
| `quant` | `f16` \| `i8` \| `binary` | no |
| `commit` | `full` \| `binary` \| `descriptors-only` \| `none` | `full` only |
| `location` | `working-tree` \| `ref` | `working-tree` only |

**A producer MUST reject a setting that it does not implement.** It MUST NOT
accept the setting and ignore it. This rule applies to each row of the table.

`dims` and `quant` change the *artifact* bytes, and they are inputs to
`artifact_fingerprint`. Thus an index with a different value has a different
identity. This fact does not change the rule: a producer rejects these two
settings when it does not implement them.

### 9.1 Committed form

A repository commits objects as loose files, in the two-character
subdirectories of §2. Packs and mmap views are derived data. They are in
`cache/`, which git ignores.

A pack MUST be a deterministic function of its member set. Thus edits that have
no overlap go into different packs and merge with no conflict. To resolve a
conflict in one pack, make the pack again from the union of the members. A
consumer rebuilds a pack that a different format version wrote. It does not
reject the pack.

### 9.2 Tensor encoding

A producer stores the tensor payload of a segment as **binary little-endian
frames**. Each tensor is one `MAPT` frame. A `MAPTGRP` container contains the
frames of one object. In the reference implementation,
`map_format::codec::encode_tensor` and `map_core::encode_tensor_payload` give
these two layouts. A cluster record contains its tensor in its JSON record, as
a byte array.

A decoder MUST also accept the JSON array encoding of a tensor payload. Thus an
index that uses the JSON encoding continues to load. An object key comes from
the embedder configuration and the dimension set, not from the serialization.
Thus a subsequent index build uses those objects again and does not write them
again.

Descriptors and cluster labels stay JSON (§8).

A tensor MUST declare a shape that agrees with its byte length. A decoder MUST
reject a tensor with a shape that does not agree. This includes a dangerous
shape for which the product overflows and looks the same as a correct empty
tensor.

### 9.3 Unresolvable stages

A dimension gives an implementation name for each stage. A producer can resolve
only the implementations that its binary and its environment supply. A stage is
unresolvable in these conditions:

- The binary does not contain the plugin.
- The user configuration has no endpoint.
- The model is not installed.
- No implementation has the given name.

**A producer MUST NOT ignore an unresolvable stage without a report.** At query
time, a consumer cannot see the difference between an index that is not full
and a full index.

A producer that finds an unresolvable stage MUST show each dimension and stage
that has the problem. It MUST NOT write objects unless it has explicit consent
to continue without those stages. Consent can come from an interactive answer or
from a flag. **A run with no terminal does not give consent.** When a producer
has consent, it MUST also show the missing stages when the run stops.

### 9.4 Collection

Object keys are input-addressed (§4). Thus an edit to an input of a fingerprint
gives new keys, and the previous objects stay on disk. An index becomes larger
until a collector removes those objects.

**The reachable set is the union of all manifests that can become the active
manifest.** It is not only the manifest of the working tree. A committed index
includes its manifest. Thus each ref has a different manifest.

Where refs are available, a collector MUST use the union of the working-tree
manifest and each manifest that a ref can reach. In some conditions a collector
cannot get the list of refs, for example when it cannot run the VCS of the
repository. Reachability is then *unknown*, not empty, and the collector MUST
NOT delete objects.

Where there is no repository, or the manifest is untracked, the working-tree
manifest is the only manifest. That one manifest gives the full reachable set.

A collector MUST NOT delete objects when it cannot parse a manifest.

**Collection does not remove history.** A deleted blob is not in subsequent
checkouts. It stays in the git history, and each clone continues to fetch it.

### 9.5 Merge

An implementation MUST resolve two manifests that have the same ancestor with a
three-way merge. A text merge is not sufficient. Object keys are
input-addressed (§4). A line merge cannot see the difference between a correct
union of object tables and a collision of one key with different bytes.

A merge MUST use the union of the object tables, because usually each side adds
different objects. A merge MUST reject a key that has a different content hash
on each side. It MUST NOT select one side, and it MUST NOT replace an entry
without an error. Roots and dimension identities merge three-way, one key at a
time.

A merge MAY remove the cluster tree of a dimension when each side made a new
tree independently. Then no tree describes the merged object set, and there is
no rule to select one tree. The cluster objects stay on disk. Thus a subsequent
index build uses each subtree that did not change. A merge that removes a tree
MUST decrease the recorded levels of that dimension to the levels that stay.

## 10. Freshness

No operation updates an index automatically. A producer writes only when a user
or a caller tells it to write. The format has no policy field that changes this
rule.

An update of a dimension that uses a model can use a classifier endpoint that
is not available. Thus a consumer cannot always update a stale index. As a
result, the query protocol MUST **show that a result is stale, for each
result**.

**Status of the reference implementation.** The reference implementation does
not show this for each result. `map find` prints one notice for a stale index,
and only when stderr is a terminal.
