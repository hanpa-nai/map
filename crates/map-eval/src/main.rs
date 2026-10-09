//! Run the golden set against an index and print the scores.
//!
//! Built for a fast edit-and-see-the-number loop: it loads the index once,
//! queries in-process rather than shelling out per query, and prints a compact
//! table. Heavier end-to-end proof with task agents is a separate exercise;
//! this exists so a change to a stage can be priced in seconds.
//!
//! ```text
//! map-eval ../ripgrep --corpus eval/corpora/ripgrep-8372866
//! map-eval ../ripgrep --baseline before.json    # print deltas
//! ```

mod grep;

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use map_eval::{mean, parse_jsonl, score_query, Hit, Judgment, Query, QueryReport};

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("map-eval: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

struct Args {
    index: PathBuf,
    /// Extra roots to federate with `index`.
    ///
    /// Judgments name resources in `index` only, so every hit that comes from
    /// elsewhere is an intrusion — it occupies a top-k slot it cannot earn.
    /// That makes the nDCG drop a direct measure of cross-index calibration.
    roots: Vec<PathBuf>,
    corpus: PathBuf,
    k: usize,
    save: Option<PathBuf>,
    baseline: Option<PathBuf>,
    per_query: bool,
    grep: bool,
    fused: bool,
    max_fusion: bool,
    descriptive: bool,
    /// Fabric levels to search: `segments` (default), `clusters`, or `all`.
    ///
    /// The frozen baselines were all taken at `segments`, and a cluster hit is
    /// expanded to the spans beneath it, so anything else is a different
    /// measurement rather than a tweak to this one.
    levels: map_query::LevelFilter,
    /// Dimensions to query as `NAME[:WEIGHT]`, overriding the mode flags.
    ///
    /// The mode flags each stand for a hardcoded dimension set, so they cannot
    /// express a set nobody anticipated — including a single dense dimension.
    ///
    /// The weight is the one genuinely free knob in the whole system: fusion
    /// happens at query time over already-stored scores, so sweeping it costs
    /// no reindex and no LLM call. It was unmeasurable until this existed —
    /// the fusion call hardcoded 1.0.
    dims: Vec<(String, f32)>,
}

fn parse_args() -> Result<Args, String> {
    let mut index = None;
    let mut roots: Vec<PathBuf> = Vec::new();
    let mut corpus = None;
    let mut k = 10usize;
    let mut save = None;
    let mut baseline = None;
    let mut per_query = false;
    let mut grep = false;
    let mut fused = false;
    let mut max_fusion = false;
    let mut descriptive = false;
    let mut levels = map_query::LevelFilter::Segments;
    let mut dims: Vec<(String, f32)> = Vec::new();

    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = |name: &str| it.next().ok_or_else(|| format!("{name} needs a value"));
        match arg.as_str() {
            "--corpus" => corpus = Some(PathBuf::from(value("--corpus")?)),
            "--root" => roots.push(PathBuf::from(value("--root")?)),
            "-k" => k = value("-k")?.parse().map_err(|_| "-k needs a number")?,
            "--save" => save = Some(PathBuf::from(value("--save")?)),
            "--baseline" => baseline = Some(PathBuf::from(value("--baseline")?)),
            "--per-query" => per_query = true,
            "--grep" => grep = true,
            "--fused" => fused = true,
            "--max-fusion" => max_fusion = true,
            "--descriptive" => descriptive = true,
            "--level" => {
                levels = match value("--level")?.as_str() {
                    "segments" => map_query::LevelFilter::Segments,
                    "clusters" => map_query::LevelFilter::Clusters,
                    "all" => map_query::LevelFilter::All,
                    other => {
                        return Err(format!(
                            "--level takes segments|clusters|all, got {other:?}"
                        ))
                    }
                }
            }
            "--dim" => {
                let spec = value("--dim")?;
                let (name, weight) = match spec.split_once(':') {
                    Some((name, w)) => (
                        name.to_owned(),
                        w.parse::<f32>()
                            .map_err(|_| format!("weight in --dim {spec:?} is not a number"))?,
                    ),
                    None => (spec, 1.0),
                };
                if !weight.is_finite() || weight < 0.0 {
                    return Err(format!("weight for {name:?} must be finite and >= 0"));
                }
                dims.push((name, weight));
            }
            "-h" | "--help" => {
                println!("usage: map-eval <index-dir> [--corpus DIR] [-k N] [--save F] [--baseline F] [--per-query] [--grep] [--fused [--max-fusion]] [--descriptive]");
                println!("  --grep         score a coverage-ranked grep baseline");
                println!("  --fused        query lexical + semantic together and fuse (needs --features distilled)");
                println!("  --max-fusion   with --fused, fuse by max instead of mean");
                println!("  --descriptive  query the LLM-descriptor `descriptive` dimension (needs --features distilled)");
                println!("  --level L      segments (default) | clusters | all; a cluster is scored by the spans beneath it");
                println!("  --dim NAME[:W] query exactly these dimensions, repeatable; W weights it in the fusion (default 1)");
                println!("  --root DIR     federate with another index, repeatable");
                std::process::exit(0);
            }
            other if other.starts_with('-') => return Err(format!("unknown flag {other}")),
            other => index = Some(PathBuf::from(other)),
        }
    }

    let index = index.ok_or("give me the directory holding the .map index")?;
    let corpus = corpus.unwrap_or_else(|| PathBuf::from("eval/corpora/ripgrep-8372866"));
    Ok(Args {
        index,
        roots,
        corpus,
        k,
        save,
        baseline,
        per_query,
        grep,
        fused,
        max_fusion,
        descriptive,
        levels,
        dims,
    })
}

fn read(path: &Path) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|e| format!("reading {}: {e}", path.display()))
}

/// A federation member's label: its directory name, matching what `map find`
/// prints.
fn origin_of(path: &Path) -> String {
    path.canonicalize()
        .unwrap_or_else(|_| path.to_path_buf())
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

/// Byte span to 1-based inclusive line numbers, for an arbitrary resource.
///
/// The cluster path needs this because a cluster's members are resolved spans
/// rather than hits, and they name resources the hit itself does not.
fn span_lines(
    root: &Path,
    resource: &str,
    start: u32,
    end: u32,
    cache: &mut BTreeMap<String, String>,
) -> (usize, usize) {
    let text = cache.entry(resource.to_owned()).or_insert_with(|| {
        // Keys come from an index, so they are checked before they touch the
        // filesystem, exactly as `map find` checks them.
        if map_stages::discover::check_resource_key(resource).is_err() {
            return String::new();
        }
        std::fs::read_to_string(root.join(resource))
            .map(|raw| map_stages::preprocess::normalize_line_endings(&raw))
            .unwrap_or_default()
    });
    let first = map_query::line_of(text, start);
    let last = map_query::line_of(text, end.saturating_sub(1)).max(first);
    (first, last)
}

/// Convert a hit's byte span into 1-based inclusive line numbers.
///
/// Re-normalizes line endings first: spans index normalized content, so on a
/// CRLF checkout raw bytes land in the wrong place.
fn line_range(
    root: &Path,
    hit: &map_query::Hit,
    cache: &mut BTreeMap<String, String>,
) -> (usize, usize) {
    let text = cache.entry(hit.resource.clone()).or_insert_with(|| {
        if map_stages::discover::check_resource_key(&hit.resource).is_err() {
            return String::new();
        }
        std::fs::read_to_string(root.join(&hit.resource))
            .map(|raw| map_stages::preprocess::normalize_line_endings(&raw))
            .unwrap_or_default()
    });
    let start = map_query::line_of(text, hit.start);
    // `end` is exclusive; step back one byte so a span ending exactly at a
    // newline does not claim the following line.
    let end = map_query::line_of(text, hit.end.saturating_sub(1)).max(start);
    (start, end)
}

fn run() -> Result<(), String> {
    let args = parse_args()?;

    let queries: Vec<Query> = parse_jsonl(&read(&args.corpus.join("queries.jsonl"))?)?;
    let judgments: Vec<Judgment> = parse_jsonl(&read(&args.corpus.join("qrels.jsonl"))?)?;

    let mut by_query: BTreeMap<&str, Vec<&Judgment>> = BTreeMap::new();
    for j in &judgments {
        by_query.entry(j.query_id.as_str()).or_default().push(j);
    }
    // Absent queries are supposed to have no answer, so they carry no
    // judgments; every other query must. Catching a missing qrel here is the
    // difference between "this query tests abstention" and "I forgot to judge
    // it", which otherwise look identical.
    for q in &queries {
        if !q.is_absent() && !by_query.contains_key(q.id.as_str()) {
            return Err(format!(
                "{} has no judgments (tag facet:absent if intended)",
                q.id
            ));
        }
        if q.is_absent() && by_query.contains_key(q.id.as_str()) {
            return Err(format!("{} is facet:absent but has judgments", q.id));
        }
    }

    let corpus_backed = args.grep;

    // The corpus root is where resource keys resolve. For the MAP index that
    // is the .map root; for grep it is the searched directory.
    let root = if corpus_backed {
        args.index.clone()
    } else {
        map_query::Index::open(&args.index)
            .map_err(|e| e.to_string())?
            .root()
            .to_path_buf()
    };

    let started = std::time::Instant::now();
    // `index` is always the first member, so a plain run and a federated one
    // take the same code path. Federating one index is score-identical to
    // opening it alone, which is what makes that safe.
    let primary = origin_of(&args.index);
    let origins: Vec<String> = std::iter::once(&args.index)
        .chain(&args.roots)
        .map(|p| origin_of(p))
        .collect();
    let index = if corpus_backed {
        None
    } else {
        let members: Vec<(String, PathBuf)> = std::iter::once(&args.index)
            .chain(&args.roots)
            .map(|p| (origin_of(p), p.clone()))
            .collect();
        Some(map_query::Federation::open(members).map_err(|e| e.to_string())?)
    };
    let grep_docs = if args.grep {
        Some(grep::load_corpus(&root)?)
    } else {
        None
    };
    let load_ms = started.elapsed().as_secs_f64() * 1000.0;

    let system: String = if args.grep {
        "grep (coverage)".to_owned()
    } else if !args.dims.is_empty() {
        format!(
            "map ({}{})",
            args.dims
                .iter()
                .map(|(d, w)| if *w == 1.0 {
                    d.clone()
                } else {
                    format!("{d}:{w}")
                })
                .collect::<Vec<_>>()
                .join(" + "),
            if args.dims.len() > 1 {
                format!(", {} fusion", if args.max_fusion { "max" } else { "mean" })
            } else {
                String::new()
            }
        )
    } else if args.fused {
        format!(
            "map fused (lexical + semantic, {} fusion)",
            if args.max_fusion { "max" } else { "mean" }
        )
    } else if args.descriptive {
        "map descriptive (LLM-descriptor `descriptive` dimension)".to_owned()
    } else {
        "map (lexical)".to_owned()
    };
    let system = if args.roots.is_empty() {
        system
    } else {
        format!("{system} federated with {} more", args.roots.len())
    };

    let mut text_cache: BTreeMap<String, String> = BTreeMap::new();
    // Positives are scored on nDCG/recall/MRR; absents only on top confidence.
    let mut reports: Vec<(String, QueryReport)> = Vec::new();
    let mut absent_confidence: Vec<f64> = Vec::new();
    let mut absent_skipped = 0usize;
    let mut intrusions = 0usize;
    let mut scored_slots = 0usize;

    // Hits for grep or the MAP index.
    let mut lexical_or_grep = |text: &str| -> Result<(Vec<Hit>, f64), String> {
        if let Some(docs) = &grep_docs {
            // Same question, same tokenizer, same files — only ranking differs.
            return Ok(grep::grep_query(docs, text, args.k));
        }

        // Every system gets the same input: the question as written. Fused mode
        // queries lexical + semantic; descriptive mode queries the LLM-descriptor
        // `descriptive` dimension alone; otherwise lexical alone.
        let mut fields = map_query::Query::new();
        if !args.dims.is_empty() {
            for (dim, weight) in &args.dims {
                // Weighted at the *query*, not only in the re-fusion below.
                // With a neutral QueryTerm the retriever ranks and truncates the
                // pool unweighted, so a weight could only reorder within it and
                // could never promote a candidate the unweighted mean had
                // already dropped — which made large weights indistinguishable.
                fields.insert(dim.clone(), map_query::QueryTerm::weighted(text, *weight));
            }
        } else if args.descriptive {
            fields.insert("descriptive".to_owned(), map_query::QueryTerm::new(text));
        } else {
            fields.insert("lexical".to_owned(), map_query::QueryTerm::new(text));
            if args.fused {
                fields.insert("semantic".to_owned(), map_query::QueryTerm::new(text));
            }
        }

        // In fused mode, pull a wide pool so re-ranking by a different fusion
        // policy still sees the candidates it would promote — a strong-in-one-
        // dimension hit can sit well below the mean-ranked top-k.
        // A wide pool whenever more than one dimension is in play, so re-ranking
        // by a different fusion policy still sees candidates it would promote.
        let limit = if args.fused || fields.len() > 1 {
            200
        } else {
            args.k
        };
        let pool = index
            .as_ref()
            .unwrap()
            .find_at(&fields, limit, args.levels.clone())
            .map_err(|e| e.to_string())?;

        let asked: Vec<String> = fields.keys().cloned().collect();
        let weights: BTreeMap<&str, f32> =
            args.dims.iter().map(|(d, w)| (d.as_str(), *w)).collect();
        // Re-fuse each candidate from its per-dimension scores. `max` trusts the
        // most confident dimension (complementary strengths); `mean` is the
        // retriever's own policy — the *same* function it fuses with, so this
        // baseline cannot drift from production behaviour.
        let mut ranked: Vec<(&map_query::Hit, f32)> = pool
            .iter()
            .map(|h| {
                let fused = if args.max_fusion {
                    h.per_dimension.values().cloned().fold(0.0, f32::max)
                } else {
                    map_query::fuse_weighted_mean(&h.per_dimension, &asked, |d| {
                        weights.get(d).copied().unwrap_or(1.0)
                    })
                };
                (h, fused)
            })
            .collect();
        ranked.sort_by(|a, b| {
            b.1.total_cmp(&a.1)
                .then(a.0.resource.cmp(&b.0.resource))
                .then(a.0.start.cmp(&b.0.start))
        });

        let confidence = ranked.first().map_or(0.0, |(_, s)| *s as f64);
        let mut hits: Vec<Hit> = Vec::new();
        for (h, _) in ranked.iter() {
            if hits.len() >= args.k {
                break;
            }
            if h.origin != primary {
                // A foreign hit is scored as a miss on purpose. Prefixing the
                // origin guarantees it cannot collide with a judgment, and its
                // line span is not resolved because it resolves against another
                // root entirely.
                intrusions += 1;
                scored_slots += 1;
                hits.push(Hit {
                    resource: format!("{}/{}", h.origin, h.resource),
                    start_line: 0,
                    end_line: 0,
                });
                continue;
            }

            // A cluster spans no file, so scoring it against file- and
            // span-scoped judgments means asking what it *retrieves*: every
            // level-0 segment beneath it. That is the fabric's actual claim —
            // one result standing for a whole component instead of N probes —
            // and expanding it here is the only way to put a number on it.
            //
            // The members occupy the slots their content would have, so `k`
            // counts delivered spans rather than results. A cluster is
            // therefore not free: it spends the budget it fills.
            if h.level > 0 {
                let members = index
                    .as_ref()
                    .and_then(|i| i.cluster_members(h))
                    .unwrap_or_default();
                for member in members {
                    if hits.len() >= args.k {
                        break;
                    }
                    scored_slots += 1;
                    let (start_line, end_line) = span_lines(
                        &root,
                        &member.resource,
                        member.start,
                        member.end,
                        &mut text_cache,
                    );
                    hits.push(Hit {
                        resource: member.resource,
                        start_line,
                        end_line,
                    });
                }
                continue;
            }

            scored_slots += 1;
            let (start_line, end_line) = line_range(&root, h, &mut text_cache);
            hits.push(Hit {
                resource: h.resource.clone(),
                start_line,
                end_line,
            });
        }
        Ok((hits, confidence))
    };

    let query_start = std::time::Instant::now();
    for q in &queries {
        let (hits, confidence): (Vec<Hit>, f64) = lexical_or_grep(&q.text)?;

        if q.is_absent() {
            // Only count a query whose absence was curated against every corpus
            // being searched. Otherwise a corpus that genuinely answers it is
            // scored as false confidence, which measures curation, not the
            // retriever.
            if q.absent_across(&origins) {
                absent_confidence.push(confidence);
            } else {
                absent_skipped += 1;
            }
        } else {
            reports.push((
                q.id.clone(),
                score_query(&hits, &by_query[q.id.as_str()], args.k),
            ));
        }
    }
    let query_ms = query_start.elapsed().as_secs_f64() * 1000.0;

    let all: Vec<QueryReport> = reports.iter().map(|(_, r)| *r).collect();
    let avg = mean(&all);

    if args.per_query {
        println!(
            "{:<8} {:>9} {:>9} {:>9} {:>9} {:>7}",
            "query", "file nDCG", "file rec", "span nDCG", "span rec", "files"
        );
        for (id, r) in &reports {
            println!(
                "{:<8} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>7}",
                id, r.file.ndcg, r.file.recall, r.span.ndcg, r.span.recall, r.distinct_files
            );
        }
        println!();
    }

    println!("system:  {system}");
    println!("queries: {}   k: {}", queries.len(), args.k);
    // Judgments cover the primary index only, so a hit from anywhere else took
    // a slot it could not earn. This is the calibration measurement: if
    // cross-index scores were miscalibrated, foreign hits would crowd the top-k.
    if !args.roots.is_empty() {
        println!(
            "intrusions: {intrusions} of {scored_slots} slots ({:.1}%)",
            100.0 * intrusions as f64 / scored_slots.max(1) as f64
        );
    }
    println!(
        "load: {load_ms:.1} ms   {} queries: {query_ms:.1} ms   ({:.2} ms/query)",
        queries.len(),
        query_ms / queries.len() as f64
    );
    println!();
    println!("{:<7} {:>8} {:>8} {:>8}", "", "nDCG", "recall", "MRR");
    println!(
        "{:<7} {:>8.3} {:>8.3} {:>8.3}",
        "file", avg.file.ndcg, avg.file.recall, avg.file.mrr
    );
    println!(
        "{:<7} {:>8.3} {:>8.3} {:>8.3}",
        "span", avg.span.ndcg, avg.span.recall, avg.span.mrr
    );
    println!();
    println!(
        "distinct files per {} results: {:.1}   ({:.0}% of slots are repeat files)",
        avg.returned,
        avg.distinct_files as f64,
        100.0 * (1.0 - avg.distinct_files as f64 / avg.returned.max(1) as f64)
    );

    // Slice by register and by facet. Both share no intents across slices, so
    // read the differences as indicative rather than controlled — but a
    // term-matching stage doing best on bare keywords and worst on
    // vocabulary-gap queries is exactly the shape the mix exists to expose.
    let form_of: BTreeMap<&str, &str> = queries
        .iter()
        .map(|q| (q.id.as_str(), q.form.as_str()))
        .collect();
    let facet_of: BTreeMap<&str, &str> = queries
        .iter()
        .map(|q| (q.id.as_str(), q.facet.as_str()))
        .collect();

    let slice_by = |axis: &BTreeMap<&str, &str>, value: &str| -> Vec<QueryReport> {
        reports
            .iter()
            .filter(|(id, _)| axis.get(id.as_str()) == Some(&value))
            .map(|(_, r)| *r)
            .collect()
    };

    // Curated order first — roughly easiest to hardest — then anything else the
    // corpus carries. Derived from the data rather than hardcoded, because a
    // hardcoded list drops a newly added facet from the table with no error:
    // the aggregate moves, the row that would explain why is simply absent.
    let ordered = |known: &[&'static str], axis: &BTreeMap<&str, &str>| -> Vec<String> {
        let mut out: Vec<String> = known.iter().map(|s| (*s).to_owned()).collect();
        let mut extra: BTreeSet<&str> = axis.values().copied().collect();
        for name in known {
            extra.remove(name);
        }
        out.extend(extra.into_iter().map(str::to_owned));
        out
    };

    println!();
    println!("{:<12} {:>8} {:>8} {:>5}", "by form", "file", "span", "n");
    for form in ordered(&["keyword", "prose", "question"], &form_of) {
        let slice = slice_by(&form_of, &form);
        if slice.is_empty() {
            continue;
        }
        let m = mean(&slice);
        println!(
            "{:<12} {:>8.3} {:>8.3} {:>5}",
            form,
            m.file.ndcg,
            m.span.ndcg,
            slice.len()
        );
    }

    println!();
    println!("{:<12} {:>8} {:>8} {:>5}", "by facet", "file", "span", "n");
    for facet in ordered(
        &["anchor", "concept", "vocab-gap", "distributed"],
        &facet_of,
    ) {
        let slice = slice_by(&facet_of, &facet);
        if slice.is_empty() {
            continue;
        }
        let m = mean(&slice);
        println!(
            "{:<12} {:>8.3} {:>8.3} {:>5}",
            facet,
            m.file.ndcg,
            m.span.ndcg,
            slice.len()
        );
    }

    // Absent queries have no relevant answer, so nDCG is undefined; the thing
    // being measured is whether the system knows it found nothing. Lower top
    // confidence is better. Only meaningful because scores are normalized.
    if absent_skipped > 0 {
        println!(
            "absent: {absent_skipped} skipped — not curated as absent for {}",
            origins.join(" + ")
        );
    }
    if !absent_confidence.is_empty() {
        let mean_conf = absent_confidence.iter().sum::<f64>() / absent_confidence.len() as f64;
        let max_conf = absent_confidence.iter().cloned().fold(0.0f64, f64::max);
        println!();
        println!(
            "absent ({} queries, no answer exists): mean top confidence {:.3}, worst {:.3}  (lower is better)",
            absent_confidence.len(),
            mean_conf,
            max_conf
        );
    }

    if let Some(path) = &args.baseline {
        let before: BTreeMap<String, [f64; 4]> = serde_json::from_str(&read(path)?)
            .map_err(|e| format!("reading baseline {}: {e}", path.display()))?;
        if let Some(b) = before.get("__mean__") {
            println!();
            println!("vs baseline {}:", path.display());
            let delta = |name: &str, now: f64, was: f64| {
                let d = now - was;
                let mark = if d > 0.0005 {
                    "+"
                } else if d < -0.0005 {
                    ""
                } else {
                    " "
                };
                println!("  {name:<11} {was:.3} -> {now:.3}  ({mark}{d:.3})");
            };
            delta("file nDCG", avg.file.ndcg, b[0]);
            delta("file recall", avg.file.recall, b[1]);
            delta("span nDCG", avg.span.ndcg, b[2]);
            delta("span recall", avg.span.recall, b[3]);
        }
    }

    if let Some(path) = &args.save {
        let mut out: BTreeMap<String, [f64; 4]> = BTreeMap::new();
        for (id, r) in &reports {
            out.insert(
                id.clone(),
                [r.file.ndcg, r.file.recall, r.span.ndcg, r.span.recall],
            );
        }
        out.insert(
            "__mean__".to_owned(),
            [
                avg.file.ndcg,
                avg.file.recall,
                avg.span.ndcg,
                avg.span.recall,
            ],
        );
        let json = serde_json::to_string_pretty(&out).map_err(|e| e.to_string())?;
        std::fs::write(path, json).map_err(|e| format!("writing {}: {e}", path.display()))?;
        println!();
        println!("saved to {}", path.display());
    }

    Ok(())
}
