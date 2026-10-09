//! The `llm` classifier: LLM prose descriptors for a dimension, at every level.
//!
//! This stage owns only the *classification* logic — the schema over the
//! dimensions, the numbered-item prompt, and the mapping of the reply onto
//! records. The connection (endpoint, model, key, prompting, caching) is the
//! orthogonal [`map_llm::LlmAdapter`]'s job.
//! `LlmClassifier` is the Rust-idiomatic spelling of "LLM classifier"; the
//! acronym lint requires the mixed case.
//!
//! One call shape serves the whole ladder: a batch item is the texts a record
//! is built from — a segment's own content at level 0, a group's descriptors
//! above it — so summarizing a cluster is the same request as describing a
//! segment, with only the prompt differing. That is what let the separate
//! cluster labeler go away, and it is why a level of the tree now costs one
//! request rather than one per cluster.
//!
//! # What is tested here vs not
//!
//! [`build_messages`], [`build_schema`], and [`parse_response`] are pure and
//! unit-tested — the request shape and the reply-to-records mapping are checked
//! offline. Only the HTTP round trip (inside the adapter) needs a live
//! endpoint.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use map_core::{Classifier, ClassifyBatch, DimensionRecords, Error, Result, Stage};
use map_format::{Record, RecordKind, RecordMeta};
use map_llm::{LlmAdapter, LlmError};
use serde_json::{json, Value};

/// Default instruction for level 0 when a dimension configures no prompt.
///
/// Material-neutral, because a root holds whatever text it holds. A corpus that
/// wants the model to know its resources are code, prose, or transcripts says so
/// in its configured prompt — which is fingerprinted, so saying it re-keys.
pub const DEFAULT_PROMPT: &str =
    "Describe what this is and what it does, concisely and in plain language.";

/// Default instruction above level 0 when none is configured for that level.
pub const DEFAULT_CLUSTER_PROMPT: &str =
    "You are given several descriptions of related material. Write one short label — \
     a noun phrase — naming the theme they share.";

/// The structural half of the level-0 system message.
///
/// Says what the model is given and what shape to return, and deliberately says
/// nothing about what the material *is* — that is the configured prompt's job.
///
/// **This template is used and fingerprinted from the same constant**, so
/// editing it re-keys every descriptor it produced. The previous spelling was
/// inlined in [`build_messages`] and covered by no fingerprint at all, which
/// meant a wording change rewrote stored descriptors while reusing their keys —
/// the failure mode this crate has already been bitten by once.
const LEVEL0_FRAMING: &str =
    "You are given {count} numbered segments. Return a JSON object with a \
                              `segments` array holding exactly {count} objects, one per segment in \
                              order. Each object has these {kind}, one per facet: {facets}.{shape}";

/// The same, above level 0, where an item is a group of descriptions rather
/// than a segment. Fingerprinted through [`Stage::fabric_config`], which keys
/// cluster records; it cannot change a segment descriptor's bytes.
const CLUSTER_FRAMING: &str = "You are given {count} numbered groups; each lists the descriptions \
                               of the items it contains. Return a JSON object with a `segments` \
                               array holding exactly {count} objects, one per group in order. Each \
                               object has these {kind}, one per facet: {facets}.{shape}";

/// The framing template for a level.
///
/// One accessor so the templates have a single reader, and the fingerprints
/// below can name the same source rather than a copy of it.
fn framing_for(level: u16) -> &'static str {
    if level == 0 {
        LEVEL0_FRAMING
    } else {
        CLUSTER_FRAMING
    }
}

/// A dimension's prompts, keyed by the fabric level they produce.
///
/// Config spells this as one table starting at level 0:
///
/// ```toml
/// classifier = { impl = "llm", prompts = { "0" = "describe it", "2" = "name the domain" } }
/// ```
///
/// A level with no entry of its own uses the nearest one below it, so the last
/// prompt declared covers every level above it and a tree can grow taller
/// without the config changing.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PromptSet {
    by_level: BTreeMap<u16, String>,
}

impl PromptSet {
    pub fn new(by_level: BTreeMap<u16, String>) -> Self {
        PromptSet { by_level }
    }

    /// One prompt for every level, the pre-ladder spelling.
    pub fn uniform(prompt: impl Into<String>) -> Self {
        PromptSet::new(BTreeMap::from([(0, prompt.into())]))
    }

    /// The prompt for `level`: its own, else the nearest declared below it,
    /// else the built-in default for that altitude.
    pub fn for_level(&self, level: u16) -> &str {
        self.by_level
            .range(..=level)
            .next_back()
            .map(|(_, prompt)| prompt.as_str())
            .unwrap_or(if level == 0 {
                DEFAULT_PROMPT
            } else {
                DEFAULT_CLUSTER_PROMPT
            })
    }

    /// The level-0 prompt, which is what keys stored segment descriptors.
    fn level0(&self) -> &str {
        self.for_level(0)
    }

    /// Identity of the whole ladder, **with the built-in fallbacks named**.
    ///
    /// The declared entries are already in the dimension's
    /// `artifact_fingerprint`, which serializes the config table. The fallbacks
    /// are not, and cannot be: they are constants in this crate, so a config
    /// declaring no cluster prompt still has one, and editing that constant
    /// would otherwise re-label every tree while re-keying nothing.
    fn identity(&self) -> String {
        let mut out = String::new();
        for (level, prompt) in &self.by_level {
            out.push_str(&format!("{level}={prompt};"));
        }
        out.push_str(&format!(
            "fallback0={DEFAULT_PROMPT};fallbackN={DEFAULT_CLUSTER_PROMPT};"
        ));
        out
    }
}

/// Why a reply could not be used, and whether a smaller batch would help.
///
/// The distinction is load-bearing: a reply truncated at the output-token limit
/// looks like a transport error once it reaches [`map_core::Error`], and
/// failing on it kills a whole indexing run over a batch that was merely too
/// large. Halving the batch halves the reply.
enum ReplyFailure {
    /// The endpoint or network failed. Smaller batches will not help.
    Transport(Error),
    /// The model answered with something unusable — truncated, or not JSON.
    Unusable,
}

/// A classifier that asks an LLM for a prose descriptor per item per
/// dimension, over a shared [`LlmAdapter`].
pub struct LlmClassifier {
    adapter: Arc<LlmAdapter>,
    /// Instructions shown to the model, per level; they shape the output, so
    /// they are committed config and part of the dimension's fingerprint.
    prompts: PromptSet,
    /// Dimensions this classifier owns, sorted — one schema field each.
    dimensions: Vec<String>,
    /// The shape each dimension's answer must take.
    facets: Facets,
    /// Segments the model never returned usable structured output for.
    ///
    /// Interior mutability because `classify` takes `&self` through
    /// [`map_core::Classifier`]. Counted rather than returned as an error: one
    /// bad reply must not fail a corpus, but an endpoint that ignores
    /// `response_format` entirely would otherwise build a dimension with no
    /// descriptors in it and say nothing.
    undescribed: AtomicUsize,
}

impl LlmClassifier {
    /// Build over a shared adapter. `prompts` come from the dimension's
    /// committed config; the connection comes from the adapter.
    pub fn new(adapter: Arc<LlmAdapter>, prompts: PromptSet, dimensions: Vec<String>) -> Self {
        Self::with_facets(adapter, prompts, dimensions, Facets::Prose)
    }

    /// As [`new`](Self::new), with an enforced answer shape.
    pub fn with_facets(
        adapter: Arc<LlmAdapter>,
        prompts: PromptSet,
        dimensions: Vec<String>,
        facets: Facets,
    ) -> Self {
        LlmClassifier {
            adapter,
            prompts,
            dimensions,
            facets,
            undescribed: AtomicUsize::new(0),
        }
    }

    /// How many segments this classifier gave up on.
    ///
    /// Non-zero means the endpoint returned replies that did not satisfy the
    /// schema even one segment at a time — the signature of a server that
    /// accepts `response_format` and ignores it.
    pub fn undescribed(&self) -> usize {
        self.undescribed.load(Ordering::Relaxed)
    }

    /// Classify, keeping the model's structured answer.
    ///
    /// The index path deliberately drops it — `Facets::compose` flattens each
    /// answer into the stored string — which makes prompt iteration blind:
    /// which part came back empty, and which absorbed content meant for
    /// another, cannot be recovered from the flattened text. This returns both,
    /// and writes nothing.
    ///
    /// One request, no chunking and no bisection: this is for reading a handful
    /// of segments, and a batch small enough to inspect is small enough to
    /// send whole.
    pub fn inspect(&self, batch: &ClassifyBatch<'_>) -> Result<Vec<Inspected>> {
        let joined: Vec<String> = batch.items.iter().map(|s| s.join("\n\n")).collect();
        let texts: Vec<&str> = joined
            .iter()
            .map(|t| truncate_on_boundary(t, MAX_SEGMENT_CHARS))
            .collect();

        let schema = build_schema(batch.dimensions, &self.facets);
        let prompt = self.prompts.for_level(batch.level);
        let (system, user) =
            build_messages(prompt, batch.dimensions, &texts, batch.level, &self.facets);
        let content = self
            .adapter
            .chat_json(&system, &user, schema, "segment_descriptors")
            .map_err(llm_err)?;

        let entries = reply_entries(&content, texts.len())?;
        let composed = compose_entries(&entries, batch.dimensions, batch.level, &self.facets);
        Ok(entries
            .into_iter()
            .zip(composed)
            .map(|(raw, records)| Inspected { raw, records })
            .collect())
    }
}

/// One item's answer, before and after flattening.
pub struct Inspected {
    /// Exactly what the model returned for this item, per dimension.
    pub raw: Value,
    /// What would have been stored.
    pub records: DimensionRecords,
}

/// Build the (system, user) messages for a batch of item texts.
///
/// The user message numbers each item so the reply array lines up with the
/// input order — the array index *is* the item id.
///
/// The framing around the configured prompt comes from `LEVEL0_FRAMING` and
/// `CLUSTER_FRAMING`, both fingerprinted. Everything the model reads therefore
/// keys what it produced: editing a template re-classifies, editing the
/// configured prompt re-classifies, and there is no longer a path that changes
/// the request while reusing the answer.
pub fn build_messages(
    prompt: &str,
    dimensions: &[String],
    item_texts: &[&str],
    level: u16,
    facets: &Facets,
) -> (String, String) {
    let noun = if level == 0 { "Segment" } else { "Group" };
    let mut user = String::new();
    for (i, text) in item_texts.iter().enumerate() {
        user.push_str(&format!("--- {noun} {} ---\n{}\n\n", i + 1, text));
    }
    // The schema already forces the parts to exist; this only says what to put
    // in them. Naming them twice is deliberate — the decoder guarantees the
    // shape, the instruction guides the content.
    let kind = match facets {
        Facets::Prose => "string fields",
        Facets::Parts(_) => "fields",
    };
    let shape = match facets {
        Facets::Prose => String::new(),
        Facets::Parts(_) => format!(
            " Each facet is an object with these parts: {}.",
            facets.describe()
        ),
    };
    // `{shape}` is substituted last because it is the only value that carries
    // caller-supplied text; a facet name spelling `{count}` must not then be
    // read as a placeholder.
    let framing = framing_for(level)
        .replace("{count}", &item_texts.len().to_string())
        .replace("{kind}", kind)
        .replace("{facets}", &dimensions.join(", "))
        .replace("{shape}", &shape);
    (format!("{prompt}\n\n{framing}"), user)
}

/// The strict JSON schema the reply must satisfy: `{segments: [{<dim>: string},
/// …]}`, every dimension required so decoding cannot silently drop one.
pub fn build_schema(dimensions: &[String], facets: &Facets) -> Value {
    let mut properties = serde_json::Map::new();
    for dimension in dimensions {
        properties.insert(dimension.clone(), facets.value_schema());
    }
    json!({
        "type": "object",
        "properties": {
            "segments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": properties,
                    "required": dimensions,
                    "additionalProperties": false,
                }
            }
        },
        "required": ["segments"],
        "additionalProperties": false,
    })
}

/// The shape one dimension's answer must take.
///
/// Asking for structure in prose gets structure *some* of the time — a
/// four-part instruction produced the four parts in 43% of replies, and
/// one-liners in the rest. Putting the parts in the schema makes them
/// mandatory rather than requested, because `response_format: json_schema`
/// with every field `required` and `additionalProperties: false` is enforced by
/// the decoder rather than by the model's goodwill.
///
/// `Prose` is the original single-string shape and stays the default, so a
/// dimension that has not opted in is byte-for-byte unaffected.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Facets {
    #[default]
    Prose,
    /// Named parts, in order.
    Parts(Vec<Part>),
}

/// One part of an enforced answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Part {
    pub name: String,
    pub kind: PartKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PartKind {
    /// Free text.
    Sentences,
    /// A list of short strings, most important first, at most `max` of them.
    ///
    /// A list is how exact names survive — identifiers, entity names, product
    /// or person names: bare nouns cannot be paraphrased into flowing text the
    /// way a sentence can. The cap is what keeps that from
    /// swamping the description — an uncapped list ran one descriptor to 1,416
    /// characters, and both BM25 and a mean-pooled embedder dilute as the token
    /// count grows. Asking for importance order makes the cap a *selection*
    /// rather than an arbitrary truncation.
    List { max: Option<usize> },
}

impl Facets {
    /// Parse the `facets` setting.
    ///
    /// `"description"` is free text; `"names[]"` is an uncapped list;
    /// `"names[8]"` is a list of at most eight, most important first. The part
    /// names are the corpus's to choose — they become the schema's fields and
    /// are read only by the model.
    pub fn parse(spec: &[String]) -> Self {
        if spec.is_empty() {
            return Facets::Prose;
        }
        Facets::Parts(spec.iter().map(|s| Part::parse(s)).collect())
    }

    /// The JSON schema for one dimension's value.
    fn value_schema(&self) -> Value {
        match self {
            Facets::Prose => json!({ "type": "string" }),
            Facets::Parts(parts) => {
                let mut properties = serde_json::Map::new();
                let mut required = Vec::new();
                for part in parts {
                    required.push(part.name.clone());
                    properties.insert(part.name.clone(), part.schema());
                }
                json!({
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": false,
                })
            }
        }
    }

    /// Flatten one dimension's answer into the stored descriptor.
    ///
    /// List parts are emitted as bare space-separated tokens rather than a
    /// sentence: the descriptor is embedded by a mean-pooled model and indexed
    /// by BM25, and both weight a bare identifier more than the same identifier
    /// buried in prose.
    fn compose(&self, value: &Value) -> String {
        match self {
            Facets::Prose => value.as_str().unwrap_or("").trim().to_owned(),
            Facets::Parts(parts) => {
                let mut out = Vec::new();
                for part in parts {
                    let Some(field) = value.get(&part.name) else {
                        continue;
                    };
                    match part.kind {
                        PartKind::List { max } => {
                            let mut items: Vec<&str> =
                                field.as_array().map_or_else(Vec::new, |a| {
                                    a.iter().filter_map(|v| v.as_str()).collect()
                                });
                            // Enforced here as well as in the schema. Not every
                            // endpoint honours `maxItems`, and an over-long list
                            // that slipped through would defeat the cap
                            // silently. Importance order makes the truncation a
                            // selection rather than a coin toss.
                            if let Some(max) = max {
                                items.truncate(max);
                            }
                            if !items.is_empty() {
                                out.push(items.join(" "));
                            }
                        }
                        PartKind::Sentences => {
                            if let Some(text) = field.as_str() {
                                let text = text.trim();
                                if !text.is_empty() {
                                    out.push(text.to_owned());
                                }
                            }
                        }
                    }
                }
                out.join(" ")
            }
        }
    }

    /// The part names, for the instruction that accompanies the schema.
    fn describe(&self) -> String {
        match self {
            Facets::Prose => String::new(),
            Facets::Parts(parts) => parts
                .iter()
                .map(Part::describe)
                .collect::<Vec<_>>()
                .join(", "),
        }
    }

    /// Identity for the fingerprint. The shape changes the stored bytes, so it
    /// has to re-key them — the same rule the prompt follows.
    fn identity(&self) -> String {
        match self {
            Facets::Prose => "prose".to_owned(),
            Facets::Parts(parts) => parts
                .iter()
                .map(Part::spelling)
                .collect::<Vec<_>>()
                .join(","),
        }
    }
}

impl Part {
    fn parse(spec: &str) -> Self {
        let Some(open) = spec.find('[') else {
            return Part {
                name: spec.to_owned(),
                kind: PartKind::Sentences,
            };
        };
        let inside = spec[open + 1..].trim_end_matches(']');
        Part {
            name: spec[..open].to_owned(),
            kind: PartKind::List {
                max: inside.parse::<usize>().ok().filter(|n| *n > 0),
            },
        }
    }

    fn schema(&self) -> Value {
        match self.kind {
            PartKind::Sentences => json!({ "type": "string" }),
            PartKind::List { max } => {
                let mut schema = json!({ "type": "array", "items": { "type": "string" } });
                if let Some(max) = max {
                    // Advisory: strict structured-output modes accept a subset
                    // of JSON Schema and may drop this. `compose` truncates
                    // regardless, so the bound holds either way.
                    schema["maxItems"] = json!(max);
                }
                schema
            }
        }
    }

    fn describe(&self) -> String {
        match self.kind {
            PartKind::Sentences => self.name.clone(),
            PartKind::List { max: Some(max) } => format!(
                "{} (at most {max} short strings, most important first)",
                self.name
            ),
            PartKind::List { max: None } => format!("{} (a list of short strings)", self.name),
        }
    }

    /// Round-trips through [`Part::parse`], so the fingerprint is stable.
    fn spelling(&self) -> String {
        match self.kind {
            PartKind::Sentences => self.name.clone(),
            PartKind::List { max: Some(max) } => format!("{}[{max}]", self.name),
            PartKind::List { max: None } => format!("{}[]", self.name),
        }
    }
}

/// Parse the model's reply into one record map per item.
///
/// A short or over-long array is an error, not a silent misalignment. Empty
/// descriptions are skipped so no unsearchable record is stored.
///
/// `level` decides the record kind, so a cluster produced here is
/// indistinguishable from one the fabricator used to build by hand.
pub fn parse_response(
    content: &str,
    dimensions: &[String],
    segment_count: usize,
    level: u16,
    facets: &Facets,
) -> Result<Vec<DimensionRecords>> {
    let segments = reply_entries(content, segment_count)?;
    Ok(compose_entries(&segments, dimensions, level, facets))
}

/// The model's answer, one entry per item, validated for length.
///
/// Split out so inspection can see the structured answer: `Facets::compose`
/// flattens it on the way to storage, and by the time anything is on disk the
/// parts are gone — which field came back empty, and which absorbed content
/// meant for another, are exactly the questions prompt iteration asks.
pub fn reply_entries(content: &str, segment_count: usize) -> Result<Vec<Value>> {
    let parsed: Value = serde_json::from_str(content)
        .map_err(|e| data_err(format!("classifier reply was not valid JSON: {e}")))?;
    let segments = parsed
        .get("segments")
        .and_then(|s| s.as_array())
        .ok_or_else(|| data_err("classifier reply had no `segments` array".to_owned()))?;

    if segments.len() != segment_count {
        return Err(data_err(format!(
            "classifier returned {} segments, expected {segment_count}",
            segments.len()
        )));
    }
    Ok(segments.clone())
}

/// Flatten validated entries into records — the stored form.
pub fn compose_entries(
    segments: &[Value],
    dimensions: &[String],
    level: u16,
    facets: &Facets,
) -> Vec<DimensionRecords> {
    let mut out = Vec::with_capacity(segments.len());
    for entry in segments {
        let mut records = DimensionRecords::new();
        for dimension in dimensions {
            let text = entry
                .get(dimension)
                .map(|value| facets.compose(value))
                .unwrap_or_default();
            // An empty description would be an unsearchable record; skip it,
            // matching Record::is_searchable.
            if text.is_empty() {
                continue;
            }
            records.insert(
                dimension.clone(),
                Record {
                    descriptor: Some(text),
                    tensor: None,
                    meta: RecordMeta {
                        kind: if level == 0 {
                            RecordKind::Segment
                        } else {
                            RecordKind::Cluster
                        },
                        dimension: dimension.clone(),
                        level,
                        // The fabricator owns child identity and attaches it;
                        // a classifier never sees the members' keys.
                        children: Vec::new(),
                    },
                },
            );
        }
        out.push(records);
    }
    out
}

/// Fingerprint over the fields that change the produced descriptors: the model
/// and the level-0 prompt, not the transport-only endpoint or key.
///
/// The `chat-completions` prefix is frozen and deliberately unrelated to the
/// plugin's name. It feeds every descriptor object key, so changing it discards
/// a corpus of stored descriptors and re-bills the LLM to regenerate output
/// identical to what was thrown away. A fingerprint must move only when the
/// bytes it identifies would — renaming the plugin is not that.
///
/// Only the level-0 prompt is in, because this fingerprint keys only the
/// segment descriptor objects, whose bytes a cluster prompt cannot change.
/// Clusters are keyed through [`Stage::fabric_config`], which covers the rest.
/// So tuning a level-2 prompt re-labels the tree without re-billing a full
/// corpus classification.
///
/// `LEVEL0_FRAMING` is in because the model reads it. It is a constant rather
/// than config, which is exactly why it has to be hashed rather than trusted:
/// a constant edited in place leaves no other trace, and the whole point is
/// that the request and the stored answer cannot drift apart.
fn fingerprint(model: &str, prompt: &str, dimensions: &[String], facets: &Facets) -> String {
    format!(
        "chat-completions:model={model}:prompt={prompt}:dims={}:facets={}:framing={}",
        dimensions.join(","),
        facets.identity(),
        LEVEL0_FRAMING
    )
}

fn data_err(message: String) -> Error {
    Error::io(
        std::path::PathBuf::from("<llm>"),
        std::io::Error::new(std::io::ErrorKind::InvalidData, message),
    )
}

fn llm_err(e: LlmError) -> Error {
    Error::io(
        std::path::PathBuf::from("<llm-adapter>"),
        std::io::Error::other(e.to_string()),
    )
}

impl Stage for LlmClassifier {
    fn implementation(&self) -> &str {
        "llm"
    }

    fn config(&self) -> String {
        fingerprint(
            self.adapter.model(),
            self.prompts.level0(),
            &self.dimensions,
            &self.facets,
        )
    }

    /// Adds what only a cluster's bytes depend on: the prompts above level 0,
    /// the fallbacks behind them, and the framing they are wrapped in.
    fn fabric_config(&self) -> String {
        format!(
            "{}:ladder={}:framing={}",
            self.config(),
            self.prompts.identity(),
            CLUSTER_FRAMING
        )
    }
}

/// Longest segment text sent to the model. A descriptor is a summary, so a
/// truncated tail costs little; an untruncated 10k-line generated file would
/// blow the request budget on its own.
const MAX_SEGMENT_CHARS: usize = 5_000;

/// Character budget per request. A file's segments are split into chunks under
/// this so a large file does not exceed the endpoint's request-size limit —
/// some reject an oversized body with HTTP 413 rather than truncating. Sized to
/// stay well under that limit (ripgrep's largest file bodies ran ~400 KB when
/// unchunked and drew a 413) while keeping the call count — hence rate-limit
/// pressure — down.
const BATCH_CHAR_BUDGET: usize = 24_000;

/// Cap on segments per request. The response carries one descriptor per
/// segment, so a chunk of many tiny segments could overflow the output-token
/// limit even while its input stays under the char budget; this bounds both.
const BATCH_SEGMENT_CAP: usize = 60;

/// Truncate on a UTF-8 boundary so a multi-byte char is never split.
fn truncate_on_boundary(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Split segment lengths into contiguous chunks that stay within `budget`
/// characters and at most `max_count` segments. A segment at or over the budget
/// alone forms its own chunk — it is truncated to `MAX_SEGMENT_CHARS` before
/// it gets here, and the budget is larger than that, so this is really the
/// "last small one won't fit" case.
///
/// Pure and covering: chunks are contiguous with no gaps, so concatenating
/// their results reproduces the full per-segment sequence in order.
pub fn chunk_ranges(lengths: &[usize], budget: usize, max_count: usize) -> Vec<(usize, usize)> {
    let cap = max_count.max(1);
    let mut ranges = Vec::new();
    let mut i = 0;
    while i < lengths.len() {
        let mut j = i;
        let mut used = 0usize;
        while j < lengths.len() && j - i < cap && (j == i || used + lengths[j] <= budget) {
            used += lengths[j];
            j += 1;
        }
        ranges.push((i, j));
        i = j;
    }
    ranges
}

impl LlmClassifier {
    /// Classify one chunk, recovering from a model that returns the wrong number
    /// of descriptors.
    ///
    /// The strict count check in [`parse_response`] is what prevents a
    /// descriptor landing on the wrong segment — but a model occasionally
    /// miscounts a large batch, and failing the whole corpus over one bad reply
    /// is not acceptable at scale. So on a mismatch the chunk is split and each
    /// half retried: smaller batches miscount far less, and a size-1 batch
    /// cannot misalign. A single segment that still comes back wrong-length is
    /// left undescribed (it falls back to embedding its raw text) rather than
    /// blocking the index.
    fn classify_chunk(
        &self,
        texts: &[&str],
        schema: &Value,
        dimensions: &[String],
        level: u16,
    ) -> Result<Vec<DimensionRecords>> {
        let prompt = self.prompts.for_level(level);
        self.bisect_on_failure(texts, dimensions, level, &|batch| {
            let (system, user) = build_messages(prompt, dimensions, batch, level, &self.facets);
            self.adapter
                .chat_json(&system, &user, schema.clone(), "segment_descriptors")
                .map_err(|e| match e {
                    // The model answered, but with something unusable — most
                    // often a reply truncated at the output-token limit, which
                    // is what a long prompt over a full batch produces. Halving
                    // the batch halves the reply, so this is recoverable and
                    // must not kill the run.
                    LlmError::Reply(_) => ReplyFailure::Unusable,
                    // The endpoint or the network failed. A smaller batch
                    // cannot help, and retrying it would just multiply the
                    // damage.
                    other => ReplyFailure::Transport(llm_err(other)),
                })
        })
    }

    /// The recovery policy itself, over whatever produces a reply.
    ///
    /// Separated from the HTTP call so the give-up path — the one that decides
    /// an index is built with blank descriptors — is testable without an
    /// endpoint. `reply` is called once per surviving sub-chunk.
    fn bisect_on_failure<F>(
        &self,
        texts: &[&str],
        dimensions: &[String],
        level: u16,
        reply: &F,
    ) -> Result<Vec<DimensionRecords>>
    where
        F: Fn(&[&str]) -> std::result::Result<String, ReplyFailure>,
    {
        let content = match reply(texts) {
            Ok(content) => content,
            Err(ReplyFailure::Transport(e)) => return Err(e),
            Err(ReplyFailure::Unusable) if texts.len() > 1 => {
                let mid = texts.len() / 2;
                let mut out = self.bisect_on_failure(&texts[..mid], dimensions, level, reply)?;
                out.extend(self.bisect_on_failure(&texts[mid..], dimensions, level, reply)?);
                return Ok(out);
            }
            Err(ReplyFailure::Unusable) => {
                self.undescribed.fetch_add(1, Ordering::Relaxed);
                return Ok(vec![DimensionRecords::new()]);
            }
        };

        match parse_response(&content, dimensions, texts.len(), level, &self.facets) {
            Ok(records) => Ok(records),
            Err(_) if texts.len() > 1 => {
                let mid = texts.len() / 2;
                let mut out = self.bisect_on_failure(&texts[..mid], dimensions, level, reply)?;
                out.extend(self.bisect_on_failure(&texts[mid..], dimensions, level, reply)?);
                Ok(out)
            }
            Err(_) => {
                self.undescribed.fetch_add(1, Ordering::Relaxed);
                Ok(vec![DimensionRecords::new()])
            }
        }
    }
}

impl Classifier for LlmClassifier {
    /// Classify a file's segments, chunking the request so a large file stays
    /// under the endpoint's size limit.
    ///
    /// The file still yields one descriptor object: the chunks' results
    /// concatenate in segment order. Chunk boundaries are a deterministic
    /// function of the (truncated) segment lengths, so a re-index reuses the
    /// stored object rather than re-billing.
    fn classify(&self, batch: &ClassifyBatch<'_>) -> Result<Vec<DimensionRecords>> {
        // An item is one segment's text at level 0 and a group's descriptors
        // above; joining is the only difference, and truncation applies to the
        // result either way.
        let joined: Vec<String> = batch.items.iter().map(|s| s.join("\n\n")).collect();
        let texts: Vec<&str> = joined
            .iter()
            .map(|t| truncate_on_boundary(t, MAX_SEGMENT_CHARS))
            .collect();
        let lengths: Vec<usize> = texts.iter().map(|t| t.len()).collect();
        // The batch's dimensions, not the ones this instance was built over:
        // the fabricator drives one dimension at a time, and a schema demanding
        // the whole group's facets would ask the model to describe a cluster
        // along a facet it was not clustered by.
        let schema = build_schema(batch.dimensions, &self.facets);

        let mut out: Vec<DimensionRecords> = Vec::with_capacity(texts.len());
        for (start, end) in chunk_ranges(&lengths, BATCH_CHAR_BUDGET, BATCH_SEGMENT_CAP) {
            out.extend(self.classify_chunk(
                &texts[start..end],
                &schema,
                batch.dimensions,
                batch.level,
            )?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dims() -> Vec<String> {
        vec!["descriptive".to_owned(), "data".to_owned()]
    }

    #[test]
    fn schema_requires_every_dimension_as_a_string() {
        let schema = build_schema(&dims(), &Facets::Prose);
        let item = &schema["properties"]["segments"]["items"];
        assert_eq!(item["properties"]["descriptive"]["type"], "string");
        assert_eq!(item["properties"]["data"]["type"], "string");
        assert_eq!(item["additionalProperties"], false);
        assert_eq!(item["required"], json!(["descriptive", "data"]));
    }

    #[test]
    fn messages_number_the_segments_for_alignment() {
        let (system, user) = build_messages(
            "Describe it.",
            &dims(),
            &["ALPHA", "BETA"],
            0,
            &Facets::Prose,
        );
        assert!(system.contains("2 numbered segments"));
        assert!(user.contains("Segment 1") && user.contains("ALPHA"));
        assert!(user.contains("Segment 2") && user.contains("BETA"));
    }

    #[test]
    fn response_maps_onto_per_segment_records() {
        let content = r#"{"segments":[
            {"descriptive":"refreshes a session token","data":"sessions table"},
            {"descriptive":"parses config","data":"config file"}
        ]}"#;
        let out = parse_response(content, &dims(), 2, 0, &Facets::Prose).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(
            out[0]["descriptive"].descriptor.as_deref(),
            Some("refreshes a session token")
        );
        assert_eq!(out[1]["data"].descriptor.as_deref(), Some("config file"));
        assert!(out[0]["descriptive"].validate().is_ok());
    }

    #[test]
    fn a_miscounted_reply_is_an_error_not_a_misalignment() {
        let content = r#"{"segments":[{"descriptive":"x","data":"y"}]}"#;
        assert!(parse_response(content, &dims(), 2, 0, &Facets::Prose).is_err());
    }

    #[test]
    fn an_empty_description_is_skipped_not_stored() {
        let content = r#"{"segments":[{"descriptive":"does a thing","data":"  "}]}"#;
        let out = parse_response(content, &dims(), 1, 0, &Facets::Prose).unwrap();
        assert!(out[0].contains_key("descriptive"));
        assert!(!out[0].contains_key("data"));
    }

    #[test]
    fn non_json_reply_is_rejected() {
        assert!(parse_response("not json at all", &dims(), 1, 0, &Facets::Prose).is_err());
        assert!(parse_response(r#"{"nope":[]}"#, &dims(), 1, 0, &Facets::Prose).is_err());
    }

    #[test]
    fn chunking_covers_every_segment_contiguously_within_budget() {
        // No gaps and no overlap, so concatenating chunk results rebuilds the
        // full per-segment sequence; each chunk (past the first element) fits.
        let lengths = [10usize, 20, 100, 5, 5];
        let ranges = chunk_ranges(&lengths, 30, 40);
        assert_eq!(ranges.first().unwrap().0, 0);
        assert_eq!(ranges.last().unwrap().1, lengths.len());
        for pair in ranges.windows(2) {
            assert_eq!(pair[0].1, pair[1].0, "chunks must be contiguous");
        }
        // The oversized segment (100 > 30) stands alone rather than blocking.
        assert!(ranges.contains(&(2, 3)), "ranges: {ranges:?}");
        // Every within-budget chunk of length > 1 respects the budget.
        for (s, e) in ranges {
            let sum: usize = lengths[s..e].iter().sum();
            assert!(
                e - s == 1 || sum <= 30,
                "chunk {s}..{e} sum {sum} over budget"
            );
        }
    }

    #[test]
    fn chunking_caps_the_segment_count() {
        // Ten tiny segments, char budget generous, but a count cap of 3 forces
        // four chunks — the response-size guard, independent of input bytes.
        let lengths = [1usize; 10];
        let ranges = chunk_ranges(&lengths, 10_000, 3);
        assert_eq!(ranges, vec![(0, 3), (3, 6), (6, 9), (9, 10)]);
    }

    #[test]
    fn truncation_keeps_a_utf8_boundary() {
        let s = "héllo wörld"; // multi-byte chars
        let t = truncate_on_boundary(s, 3);
        assert!(s.starts_with(t));
        assert!(t.len() <= 3);
        // A budget past the end returns the whole string untouched.
        assert_eq!(truncate_on_boundary("abc", 100), "abc");
    }

    #[test]
    fn a_prompt_above_the_last_declared_level_reuses_it() {
        // The ladder: the last prompt covers every level above it, so a tree
        // that grows taller needs no config change.
        let prompts = PromptSet::new(BTreeMap::from([
            (0, "describe it".to_owned()),
            (2, "name the domain".to_owned()),
        ]));
        assert_eq!(prompts.for_level(0), "describe it");
        assert_eq!(prompts.for_level(1), "describe it", "nearest below");
        assert_eq!(prompts.for_level(2), "name the domain");
        assert_eq!(
            prompts.for_level(9),
            "name the domain",
            "the last one holds"
        );
    }

    #[test]
    fn a_level_with_nothing_declared_below_it_falls_back_by_altitude() {
        let empty = PromptSet::default();
        assert_eq!(empty.for_level(0), DEFAULT_PROMPT);
        assert_eq!(empty.for_level(1), DEFAULT_CLUSTER_PROMPT);

        // Declaring only a cluster prompt must not make level 0 inherit it —
        // nothing is declared *below* level 1.
        let clusters_only = PromptSet::new(BTreeMap::from([(1, "name the theme".to_owned())]));
        assert_eq!(clusters_only.for_level(0), DEFAULT_PROMPT);
        assert_eq!(clusters_only.for_level(1), "name the theme");
    }

    fn facets() -> Facets {
        Facets::parse(&[
            "purpose".to_owned(),
            "behaviour".to_owned(),
            "names[]".to_owned(),
            "subsystem".to_owned(),
        ])
    }

    #[test]
    fn the_schema_makes_every_facet_part_mandatory() {
        // Asking for four parts in prose got four parts 43% of the time. The
        // decoder does not negotiate: `required` plus additionalProperties:false
        // is what turns a request into a guarantee.
        let schema = build_schema(&["descriptive".to_owned()], &facets());
        let value = &schema["properties"]["segments"]["items"]["properties"]["descriptive"];
        assert_eq!(value["type"], "object");
        assert_eq!(
            value["required"],
            json!(["purpose", "behaviour", "names", "subsystem"])
        );
        assert_eq!(value["additionalProperties"], false);
        assert_eq!(value["properties"]["purpose"]["type"], "string");
        // The `[]` suffix is what keeps identifiers as discrete tokens rather
        // than letting them be paraphrased into a sentence.
        assert_eq!(value["properties"]["names"]["type"], "array");
        assert_eq!(value["properties"]["names"]["items"]["type"], "string");
    }

    #[test]
    fn composing_facets_preserves_the_identifiers_verbatim() {
        // The whole point: `DirEntry` and `is_symlink` must survive into the
        // stored descriptor as bare tokens, because that is what BM25 matches
        // and what a mean-pooled embedder weights.
        let content = r#"{"segments":[{"descriptive":{
            "purpose":"Represents one entry produced by the directory walker.",
            "behaviour":"Carries an optional error when an ignore file failed to parse.",
            "names":["DirEntry","is_symlink","file_name","depth"],
            "subsystem":"ignore"
        }}]}"#;
        let out = parse_response(content, &["descriptive".to_owned()], 1, 0, &facets()).unwrap();
        let text = out[0]["descriptive"].descriptor.as_deref().unwrap();

        for identifier in ["DirEntry", "is_symlink", "file_name", "depth"] {
            assert!(
                text.contains(identifier),
                "{identifier} missing from {text}"
            );
        }
        assert!(text.contains("directory walker"));
        assert!(text.contains("ignore"));
        // Bare tokens, not a sentence wrapping them.
        assert!(
            text.contains("DirEntry is_symlink file_name depth"),
            "identifiers must stay adjacent and unadorned: {text}"
        );
    }

    #[test]
    fn inspecting_and_storing_agree_on_the_composed_text() {
        // `parse_response` is now `reply_entries` + `compose_entries`, and the
        // inspection path uses the same two. If they could disagree, the thing
        // being read while iterating on a prompt would not be the thing that
        // gets indexed — which is the whole point of reading it.
        let facets = Facets::parse(&["description".to_owned(), "identifiers[8]".to_owned()]);
        let content = r#"{"segments":[{"descriptive":{
            "description":"Decides whether a path is hidden.",
            "identifiers":["is_hidden_path","file_name"]
        }}]}"#;
        let dims = ["descriptive".to_owned()];

        let stored = parse_response(content, &dims, 1, 0, &facets).unwrap();
        let entries = reply_entries(content, 1).unwrap();
        let inspected = compose_entries(&entries, &dims, 0, &facets);

        assert_eq!(
            stored[0]["descriptive"].descriptor,
            inspected[0]["descriptive"].descriptor
        );
        // And the raw entry keeps the parts the stored form threw away.
        assert_eq!(
            entries[0]["descriptive"]["identifiers"],
            json!(["is_hidden_path", "file_name"])
        );
    }

    #[test]
    fn a_capped_list_bounds_the_schema_and_the_stored_text() {
        let capped = Facets::parse(&["description".to_owned(), "identifiers[8]".to_owned()]);
        let schema = build_schema(&["descriptive".to_owned()], &capped);
        let ids = &schema["properties"]["segments"]["items"]["properties"]["descriptive"]
            ["properties"]["identifiers"];
        assert_eq!(ids["type"], "array");
        assert_eq!(ids["maxItems"], 8, "the cap has to reach the endpoint");

        // And it holds even when the endpoint ignores `maxItems` — strict
        // structured-output modes accept only a subset of JSON Schema, so the
        // bound cannot rest on the server honouring it.
        let over = json!({
            "description": "Walks a directory tree.",
            "identifiers": ["a","b","c","d","e","f","g","h","i","j","k","l"]
        });
        let text = capped.compose(&over);
        assert!(
            text.contains("a b c d e f g h"),
            "keeps the first eight: {text}"
        );
        assert!(!text.contains(" i "), "and drops the rest: {text}");
        assert!(!text.ends_with('l'), "{text}");
    }

    #[test]
    fn an_uncapped_list_still_keeps_everything() {
        let open = Facets::parse(&["identifiers[]".to_owned()]);
        let schema = build_schema(&["descriptive".to_owned()], &open);
        let ids = &schema["properties"]["segments"]["items"]["properties"]["descriptive"]
            ["properties"]["identifiers"];
        assert_eq!(ids["type"], "array");
        assert!(ids.get("maxItems").is_none(), "no cap means no bound");
        assert_eq!(
            open.compose(&json!({"identifiers": ["x", "y", "z"]})),
            "x y z"
        );
    }

    #[test]
    fn a_cap_is_part_of_the_answer_shape_identity() {
        // Changing the cap changes the stored bytes, so it must re-key them.
        assert_ne!(
            fingerprint("m", "p", &dims(), &Facets::parse(&["ids[8]".to_owned()])),
            fingerprint("m", "p", &dims(), &Facets::parse(&["ids[12]".to_owned()])),
        );
        assert_ne!(
            fingerprint("m", "p", &dims(), &Facets::parse(&["ids[8]".to_owned()])),
            fingerprint("m", "p", &dims(), &Facets::parse(&["ids[]".to_owned()])),
        );
    }

    #[test]
    fn the_instruction_states_the_cap_and_the_ordering() {
        // The schema bounds the count; only the instruction can ask for the
        // *right* eight, which is what makes truncation a selection.
        let capped = Facets::parse(&["description".to_owned(), "identifiers[8]".to_owned()]);
        let (system, _) = build_messages("Index it.", &dims(), &["X"], 0, &capped);
        assert!(system.contains("at most 8 short strings"), "{system}");
        assert!(system.contains("most important first"), "{system}");
    }

    #[test]
    fn an_empty_facet_answer_is_skipped_like_an_empty_string() {
        let content = r#"{"segments":[{"descriptive":{
            "purpose":"","behaviour":"","names":[],"subsystem":""
        }}]}"#;
        let out = parse_response(content, &["descriptive".to_owned()], 1, 0, &facets()).unwrap();
        assert!(
            out[0].is_empty(),
            "nothing to search on means no record, as with prose"
        );
    }

    #[test]
    fn the_answer_shape_is_in_the_fingerprint() {
        // It changes the stored bytes, so it must re-key them — the same rule
        // the prompt follows, and the same bug as the tensor key if missed.
        assert_ne!(
            fingerprint("m", "p", &dims(), &Facets::Prose),
            fingerprint("m", "p", &dims(), &facets()),
        );
        assert_ne!(
            fingerprint("m", "p", &dims(), &facets()),
            fingerprint(
                "m",
                "p",
                &dims(),
                &Facets::parse(&["purpose".to_owned(), "names[]".to_owned()])
            ),
        );
    }

    #[test]
    fn no_facets_configured_keeps_the_original_shape_exactly() {
        // Opting in must be opt-in: a dimension that has not asked for parts
        // gets byte-identical requests and descriptors to before.
        assert_eq!(Facets::parse(&[]), Facets::Prose);
        let schema = build_schema(&dims(), &Facets::Prose);
        let item = &schema["properties"]["segments"]["items"];
        assert_eq!(item["properties"]["descriptive"]["type"], "string");
    }

    #[test]
    fn cluster_level_messages_frame_items_as_groups() {
        let (system, user) = build_messages(
            "Name the theme.",
            &dims(),
            &["parses a config file\n\nloads config defaults"],
            1,
            &Facets::Prose,
        );
        assert!(system.contains("numbered groups"));
        assert!(system.contains("descriptions of the items it contains"));
        assert!(user.contains("Group 1"));
        assert!(!user.contains("Segment 1"));
    }

    #[test]
    fn the_level_zero_request_says_nothing_about_what_the_material_is() {
        // The framing is structural: how many items, what shape to return. What
        // the resources *are* belongs to the configured prompt, which a corpus
        // owns. A word like "code" here would tell every prose, transcript and
        // record corpus it was being shown something it was not.
        let (system, user) = build_messages("Describe it.", &dims(), &["ALPHA"], 0, &Facets::Prose);
        assert_eq!(
            system,
            "Describe it.\n\nYou are given 1 numbered segments. Return a JSON \
             object with a `segments` array holding exactly 1 objects, one per \
             segment in order. Each object has these string fields, one per facet: \
             descriptive, data."
        );
        assert_eq!(user, "--- Segment 1 ---\nALPHA\n\n");
    }

    #[test]
    fn editing_the_framing_re_keys_the_descriptors_it_produced() {
        // The hazard this closes: the framing used to be inlined in
        // `build_messages` and covered by no fingerprint, so rewording it
        // changed every request while every stored answer kept its key. The
        // fingerprint reads the same constant `build_messages` does, so the two
        // cannot drift — this asserts the constant is actually in there.
        //
        // Asserted through `framing_for(0)` rather than against the constant
        // directly: `framing_for` is the single accessor `build_messages`
        // reads, so this fails if the fingerprint and the request ever stop
        // agreeing on which constant that is. Naming the constant here would
        // still pass a refactor that left `build_messages` reading a different
        // one.
        let fp = fingerprint("m", "p", &dims(), &Facets::Prose);
        assert!(
            fp.contains(framing_for(0)),
            "the framing the model reads must key what it produced: {fp}"
        );
    }

    #[test]
    fn editing_the_cluster_framing_re_keys_the_labels_it_produced() {
        // The other half of the same hazard, and the half nothing covered:
        // `fabric_config` is what keys cluster records, so a reworded cluster
        // framing that never reaches it relabels every tree while reusing the
        // labels it no longer produces.
        let fabric = classifier().fabric_config();
        assert!(
            fabric.contains(framing_for(1)),
            "the cluster framing must key the labels it produced: {fabric}"
        );
        assert!(
            !classifier().config().contains(framing_for(1)),
            "and must stay out of the segment identity, or rewording a label \
             re-bills a whole corpus classification"
        );
    }

    #[test]
    fn a_cluster_prompt_keys_the_tree_without_re_keying_the_corpus() {
        // Both halves matter. Cluster wording must reach `fabric_config`, or an
        // edited label instruction reuses labels it no longer produces; and it
        // must stay out of `config`, or rewording a label re-bills a full corpus
        // classification.
        let base = PromptSet::uniform("p");
        let edited = PromptSet::new(BTreeMap::from([
            (0, "p".to_owned()),
            (2, "name the domain".to_owned()),
        ]));

        assert_eq!(base.level0(), edited.level0(), "level 0 is untouched");
        assert_ne!(
            base.identity(),
            edited.identity(),
            "a cluster prompt must move the fabric identity"
        );
        assert!(
            base.identity().contains(DEFAULT_CLUSTER_PROMPT),
            "the fallback is a prompt too, and no config declares it"
        );
    }

    #[test]
    fn parse_response_honours_the_batch_level() {
        // A classifier that ignored this would store a cluster claiming to be a
        // segment; `Record::validate` then rejects it once the fabricator
        // attaches children, which is the loud failure we want.
        let content = r#"{"segments":[{"descriptive":"config handling","data":"config"}]}"#;
        let out = parse_response(content, &dims(), 1, 2, &Facets::Prose).unwrap();
        assert_eq!(out[0]["descriptive"].meta.kind, RecordKind::Cluster);
        assert_eq!(out[0]["descriptive"].meta.level, 2);
        assert!(out[0]["descriptive"].meta.children.is_empty());
        assert!(out[0]["descriptive"].validate().is_ok());
    }

    #[test]
    fn a_batch_schema_covers_only_the_batch_dimensions() {
        // The fabricator drives one dimension at a time. A schema built from
        // the instance's whole group would ask the model to describe a cluster
        // along a facet it was never clustered by.
        let c = classifier();
        let one = vec!["descriptive".to_owned()];
        let out = c
            .bisect_on_failure(&["alpha"], &one, 1, &|_| {
                Ok(r#"{"segments":[{"descriptive":"config handling"}]}"#.to_owned())
            })
            .unwrap();
        assert_eq!(
            out[0]["descriptive"].descriptor.as_deref(),
            Some("config handling")
        );
        assert!(
            !out[0].contains_key("data"),
            "a dimension outside the batch must not appear"
        );
    }

    #[test]
    fn fingerprint_ignores_cluster_prompts() {
        // It keys only the level-0 descriptor objects, whose bytes a level-2
        // prompt cannot change. Folding them in would re-bill a full corpus
        // classification every time somebody tuned a cluster label.
        let base = fingerprint(
            "m",
            PromptSet::uniform("p").level0(),
            &dims(),
            &Facets::Prose,
        );
        let with_clusters = fingerprint(
            "m",
            PromptSet::new(BTreeMap::from([
                (0, "p".to_owned()),
                (2, "name the domain".to_owned()),
            ]))
            .level0(),
            &dims(),
            &Facets::Prose,
        );
        assert_eq!(base, with_clusters);
    }

    fn classifier() -> LlmClassifier {
        LlmClassifier::new(
            std::sync::Arc::new(map_llm::LlmAdapter::new(map_llm::Connection {
                endpoint: "http://127.0.0.1:1/v1".into(),
                model: "m".into(),
                api_key: None,
                protocol: map_llm::Protocol::OpenAiChat,
            })),
            PromptSet::uniform("p"),
            dims(),
        )
    }

    #[test]
    fn an_endpoint_ignoring_the_schema_is_counted_not_swallowed() {
        // The vLLM/llama.cpp failure mode: the server accepts response_format,
        // ignores it, and answers in prose. Every segment ends up blank, and
        // without this count the index would look like it succeeded.
        let c = classifier();
        let texts = ["alpha", "beta", "gamma", "delta"];
        let out = c
            .bisect_on_failure(&texts, &dims(), 0, &|_| {
                Ok("I'd be happy to help!".to_owned())
            })
            .unwrap();

        assert_eq!(out.len(), texts.len(), "every segment still gets a slot");
        assert!(out.iter().all(|r| r.is_empty()), "and every slot is blank");
        assert_eq!(c.undescribed(), 4, "all four are reported, not swallowed");
    }

    #[test]
    fn a_reply_that_parses_costs_no_bisection_and_no_report() {
        let c = classifier();
        let good = r#"{"segments":[{"descriptive":"does a thing"}]}"#;
        let out = c
            .bisect_on_failure(&["alpha"], &dims(), 0, &|_| Ok(good.to_owned()))
            .unwrap();

        assert_eq!(out.len(), 1);
        assert_eq!(c.undescribed(), 0);
    }

    #[test]
    fn a_transport_error_fails_rather_than_reporting_blanks() {
        // A dead endpoint must not look like 2481 undescribed segments.
        let c = classifier();
        let err = c.bisect_on_failure(&["alpha", "beta"], &dims(), 0, &|_| {
            Err(ReplyFailure::Transport(map_core::Error::io(
                std::path::PathBuf::from("<test>"),
                std::io::Error::other("connection refused"),
            )))
        });
        assert!(err.is_err());
        assert_eq!(c.undescribed(), 0);
    }

    #[test]
    fn a_truncated_reply_bisects_instead_of_failing_the_run() {
        // What killed a 30-minute indexing pass: a verbose prompt over a full
        // batch overflowed the output-token limit, the reply came back cut off
        // mid-JSON, and the whole run died. The batch was merely too large —
        // halving it halves the reply, so this has to recover.
        let c = classifier();
        let calls = std::cell::Cell::new(0usize);
        let out = c
            .bisect_on_failure(&["a", "b", "c", "d"], &dims(), 0, &|batch| {
                calls.set(calls.get() + 1);
                if batch.len() > 1 {
                    return Err(ReplyFailure::Unusable); // "truncated"
                }
                Ok(r#"{"segments":[{"descriptive":"x","data":"y"}]}"#.to_owned())
            })
            .unwrap();

        assert_eq!(out.len(), 4, "every item still gets a slot");
        assert!(
            out.iter().all(|r| r.contains_key("descriptive")),
            "and each one is described once the batch is small enough"
        );
        assert_eq!(c.undescribed(), 0, "nothing was given up on");
        assert!(calls.get() > 1, "it must actually have retried smaller");
    }

    #[test]
    fn a_single_item_that_stays_unusable_is_given_up_on_not_retried_forever() {
        let c = classifier();
        let out = c
            .bisect_on_failure(&["only"], &dims(), 0, &|_| Err(ReplyFailure::Unusable))
            .unwrap();
        assert_eq!(out.len(), 1);
        assert!(out[0].is_empty());
        assert_eq!(c.undescribed(), 1, "reported, not silently blank");
    }

    #[test]
    fn fingerprint_tracks_model_and_prompt() {
        assert_eq!(
            fingerprint("m", "p", &dims(), &Facets::Prose),
            fingerprint("m", "p", &dims(), &Facets::Prose)
        );
        assert_ne!(
            fingerprint("m", "p", &dims(), &Facets::Prose),
            fingerprint("m", "q", &dims(), &Facets::Prose)
        );
        assert_ne!(
            fingerprint("m", "p", &dims(), &Facets::Prose),
            fingerprint("n", "p", &dims(), &Facets::Prose)
        );
    }
}
