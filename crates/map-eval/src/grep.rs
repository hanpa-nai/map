//! A grep baseline, scored by the same harness as MAP.
//!
//! # What "grep" means here, and why
//!
//! The golden queries are natural-language questions; grep takes a pattern, not
//! a question. To compare the two honestly, grep is handed **the same input
//! MAP gets** — the same question, run through the **same tokenizer**, over the
//! **same file set** MAP indexed. The only thing that differs is the ranking
//! algorithm: BM25's IDF-weighted saturation versus grep's match counting.
//! Everything else is held constant on purpose, so a difference in the score is
//! a difference in the retrieval method and nothing else.
//!
//! Ranking is **term coverage**: a file's score is how many distinct query
//! terms appear anywhere in it, tie-broken by total occurrences. This is what a
//! person scanning `rg` output actually does — prefer the file that hits more
//! of the query's vocabulary — and it is deliberately generous to grep, because
//! coverage neutralizes common words (every file contains "the", so it does not
//! discriminate) without grep having any notion of IDF. If MAP still wins
//! against a grep given this much help, that is the meaningful result.
//!
//! # What this is NOT
//!
//! It is not grep at its ceiling. A knowledgeable user hand-crafting a regex
//! per query — `BinaryDetection`, not the words of the question — beats this
//! wherever the right identifier is guessable, and that cannot be produced
//! mechanically from a natural-language question. Read this number as "naive
//! grep, same input as MAP": a hand-tuned line sits somewhere above it.

use std::collections::BTreeSet;
use std::path::Path;

use map_core::{Discoverer, Preprocessor};
use map_eval::Hit;
use map_stages::discover::FsDiscoverer;
use map_stages::lexical::tokenize;
use map_stages::preprocess::{normalize_line_endings, TextPreprocessor};

/// One corpus file, read and normalized once.
pub(crate) struct Doc {
    resource: String,
    /// Per line: the set of tokens on that line, and the line number (1-based).
    lines: Vec<(usize, BTreeSet<String>)>,
}

/// Read and tokenize the same files MAP indexed.
///
/// Uses MAP's own discoverer and preprocessor, so the file universe is
/// identical — same gitignore filtering, same binary rejection, same LF
/// normalization. Any divergence here would make the comparison meaningless.
pub(crate) fn load_corpus(root: &Path) -> Result<Vec<Doc>, String> {
    let discoverer = FsDiscoverer::new();
    let preprocessor = TextPreprocessor;
    let resources = discoverer.discover(root).map_err(|e| e.to_string())?;

    let mut docs = Vec::new();
    for resource in &resources {
        let path = root.join(&resource.key);
        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };
        let Some(content) = preprocessor
            .preprocess(resource, &bytes)
            .map_err(|e| e.to_string())?
        else {
            continue; // binary or non-UTF-8, exactly as the indexer skips it
        };

        let lines = normalize_line_endings(&content.text)
            .lines()
            .enumerate()
            .map(|(i, line)| {
                let tokens: BTreeSet<String> = tokenize(line).into_iter().collect();
                (i + 1, tokens)
            })
            .collect();
        docs.push(Doc {
            resource: resource.key.clone(),
            lines,
        });
    }
    Ok(docs)
}

/// Rank the corpus for one question; return the top `k` files and a confidence.
///
/// Each returned file is represented by its single densest matching line — the
/// grep-authentic span, a pointer at a line rather than a window. A line inside
/// a judged span overlaps it, so this is scored fairly at span level without
/// pretending grep returns ranges it does not.
///
/// The confidence is the top file's term coverage as a fraction of the query's
/// terms, in `[0, 1]`. It is not BM25-comparable, but it is the natural "how
/// much of what I asked for did the best hit contain" signal, and it is what an
/// absent-answer query is scored against — grep has no other notion of "I found
/// nothing good".
pub(crate) fn grep_query(docs: &[Doc], question: &str, k: usize) -> (Vec<Hit>, f64) {
    let mut query: BTreeSet<String> = tokenize(question).into_iter().collect();
    query.retain(|t| !t.is_empty());
    if query.is_empty() {
        return (Vec::new(), 0.0);
    }
    let term_count = query.len() as f64;

    struct Ranked {
        resource: String,
        coverage: usize,
        total: usize,
        best_line: usize,
    }

    let mut ranked: Vec<Ranked> = Vec::new();
    for doc in docs {
        let mut covered: BTreeSet<&str> = BTreeSet::new();
        let mut total = 0usize;
        let mut best_line = 1usize;
        let mut best_line_hits = 0usize;

        for (line_no, tokens) in &doc.lines {
            let hits = query.iter().filter(|t| tokens.contains(*t)).count();
            if hits == 0 {
                continue;
            }
            total += hits;
            for t in &query {
                if tokens.contains(t) {
                    covered.insert(t.as_str());
                }
            }
            if hits > best_line_hits {
                best_line_hits = hits;
                best_line = *line_no;
            }
        }

        if !covered.is_empty() {
            ranked.push(Ranked {
                resource: doc.resource.clone(),
                coverage: covered.len(),
                total,
                best_line,
            });
        }
    }

    // Coverage first, then total occurrences, then path for a stable order.
    ranked.sort_by(|a, b| {
        b.coverage
            .cmp(&a.coverage)
            .then(b.total.cmp(&a.total))
            .then(a.resource.cmp(&b.resource))
    });

    let confidence = ranked
        .first()
        .map_or(0.0, |r| r.coverage as f64 / term_count);
    let hits = ranked
        .into_iter()
        .take(k)
        .map(|r| Hit {
            resource: r.resource,
            start_line: r.best_line,
            end_line: r.best_line,
        })
        .collect();
    (hits, confidence)
}
