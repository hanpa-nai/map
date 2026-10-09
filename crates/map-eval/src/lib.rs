//! Scoring for the quality gate.
//!
//! Deliberately system-agnostic: this crate knows about a *ranked list of
//! located results* and a set of graded judgments, and nothing else. Every
//! baseline — MAP, grep — is scored by the same code from the same file, which
//! is the only way the comparison means anything.
//!
//! # Two granularities, one judgment file
//!
//! Judgments are spans, and every run is scored twice:
//!
//! - **File level** — did the result name a file that answers the question?
//!   Line-oriented tools can be scored this way, so it is the common ground.
//! - **Span level** — did the returned range actually overlap the answering
//!   code? This is the only measurement that separates "found the right file"
//!   from "found the right function".
//!
//! Both dedupe before scoring. A system returning six overlapping windows of
//! one file has found one thing, not six, and a metric that rewards it for
//! filling the top ten with near-duplicates is measuring the wrong quantity.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

/// One question in the golden set.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Query {
    /// Stable identifier, e.g. `rg-001`.
    pub id: String,
    /// The query itself, in whatever register `form` names.
    pub text: String,
    /// Loose semantic category, for slicing results. Not scored.
    #[serde(default)]
    pub kind: String,
    /// Query register: `keyword`, `prose`, or `question`.
    ///
    /// Real callers query in all three, and a term-matching stage behaves
    /// differently across them, so the set is a deliberate mix and this field
    /// lets the score be sliced by register.
    #[serde(default)]
    pub form: String,
    /// Retrieval facet under stress:
    ///
    /// - `anchor` — query names an exact term the resource contains; lexical
    ///   excels
    /// - `concept` — described in words that do appear somewhere in the text
    /// - `vocab-gap` — query avoids the resource's own vocabulary; lexical
    ///   should struggle and the semantic dimension is what must rescue it
    /// - `distributed` — the answer legitimately spans many files
    /// - `absent` — no relevant answer exists; the ideal response is nothing,
    ///   and such a query carries no judgments and is scored on false confidence
    #[serde(default)]
    pub facet: String,
    /// Corpora this query was verified to have no answer in.
    ///
    /// Absence is a property of a *corpus*, not of a query — "no SQL connection
    /// pooling here" is true of ripgrep and false of Flask. Scoring an absent
    /// query against a corpus it was not curated for measures nothing, so the
    /// harness only counts it when every searched origin appears here.
    ///
    /// Empty means the query has been curated against nothing and is scored
    /// only for a single-index run.
    #[serde(default)]
    pub absent_in: Vec<String>,
}

impl Query {
    /// Whether this query is expected to have no relevant answer.
    pub fn is_absent(&self) -> bool {
        self.facet == "absent"
    }

    /// Whether this query's absence has been verified for every origin searched.
    ///
    /// A federation is only as absent as its least-curated member: one corpus
    /// that answers the question makes the whole measurement meaningless.
    pub fn absent_across(&self, origins: &[String]) -> bool {
        self.is_absent() && origins.iter().all(|o| self.absent_in.contains(o))
    }
}

/// One graded relevance judgment.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Judgment {
    /// Which query this judges.
    pub query_id: String,
    /// Canonical resource key.
    pub resource: String,
    /// First line of the answering span, 1-based inclusive.
    pub start_line: usize,
    /// Last line of the answering span, 1-based inclusive.
    pub end_line: usize,
    /// 2 = answers the question, 1 = supporting context.
    pub grade: u32,
    /// Why this span was judged as it was. Not scored; makes review possible.
    #[serde(default)]
    pub why: String,
}

/// One result from a system under test, in rank order.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Hit {
    /// Canonical resource key.
    pub resource: String,
    /// First line of the returned range, 1-based inclusive.
    pub start_line: usize,
    /// Last line of the returned range, 1-based inclusive.
    pub end_line: usize,
}

/// Metrics for one granularity.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Scores {
    /// Normalized discounted cumulative gain at k. Uses the graded judgments.
    pub ndcg: f64,
    /// Fraction of relevant items retrieved within k.
    pub recall: f64,
    /// Reciprocal rank of the first relevant result, 0 if none.
    pub mrr: f64,
}

/// Everything measured for one query.
#[derive(Clone, Copy, Debug, Default)]
pub struct QueryReport {
    /// Scored on whether the right file was named.
    pub file: Scores,
    /// Scored on whether the returned range overlapped the answering span.
    pub span: Scores,
    /// Distinct files among the top k results.
    ///
    /// Not a quality metric. It is the redundancy signal: with 40-line windows
    /// and no per-file rollup, a top-10 can be five files twice over, and an
    /// agent pays tokens for the duplicates.
    pub distinct_files: usize,
    /// Results returned, capped at k.
    pub returned: usize,
}

/// Judgments for a single query, indexed for scoring.
struct Answers<'a> {
    spans: Vec<&'a Judgment>,
}

impl<'a> Answers<'a> {
    /// Best grade among judgments naming this file, or 0.
    fn file_grade(&self, resource: &str) -> u32 {
        self.spans
            .iter()
            .filter(|j| j.resource == resource)
            .map(|j| j.grade)
            .max()
            .unwrap_or(0)
    }

    /// Best grade among judged spans this hit overlaps, with the span's index.
    ///
    /// Overlap rather than containment: a 40-line window that clips the first
    /// ten lines of a judged function has found it, and demanding containment
    /// would score segmentation strategy rather than retrieval quality.
    fn span_grade(&self, hit: &Hit) -> Option<(usize, u32)> {
        self.spans
            .iter()
            .enumerate()
            .filter(|(_, j)| {
                j.resource == hit.resource
                    && hit.start_line <= j.end_line
                    && j.start_line <= hit.end_line
            })
            .map(|(i, j)| (i, j.grade))
            .max_by_key(|(_, g)| *g)
    }

    /// Grades of every judged file, descending — the ideal file-level ranking.
    fn ideal_file_grades(&self) -> Vec<u32> {
        let mut best: BTreeMap<&str, u32> = BTreeMap::new();
        for j in &self.spans {
            let e = best.entry(j.resource.as_str()).or_insert(0);
            *e = (*e).max(j.grade);
        }
        let mut grades: Vec<u32> = best.into_values().collect();
        grades.sort_unstable_by(|a, b| b.cmp(a));
        grades
    }

    /// Grades of every judged span, descending — the ideal span-level ranking.
    fn ideal_span_grades(&self) -> Vec<u32> {
        let mut grades: Vec<u32> = self.spans.iter().map(|j| j.grade).collect();
        grades.sort_unstable_by(|a, b| b.cmp(a));
        grades
    }
}

/// Discounted cumulative gain over graded relevance.
///
/// The exponential gain `2^g - 1` is the standard form: it makes a grade-2 hit
/// worth three times a grade-1 one rather than twice, which matches the
/// intent — an answer is categorically more useful than context, not
/// incrementally.
fn dcg(grades: &[u32]) -> f64 {
    grades
        .iter()
        .enumerate()
        .map(|(i, g)| ((2u32.pow(*g) - 1) as f64) / ((i + 2) as f64).log2())
        .sum()
}

/// nDCG, or 0 when nothing is relevant.
fn ndcg(actual: &[u32], ideal: &[u32], k: usize) -> f64 {
    let take = |v: &[u32]| v.iter().copied().take(k).collect::<Vec<u32>>();
    let ideal_dcg = dcg(&take(ideal));
    if ideal_dcg <= 0.0 {
        return 0.0;
    }
    (dcg(&take(actual)) / ideal_dcg).clamp(0.0, 1.0)
}

/// Reciprocal rank of the first non-zero grade.
fn mrr(grades: &[u32]) -> f64 {
    grades
        .iter()
        .position(|g| *g > 0)
        .map(|i| 1.0 / (i + 1) as f64)
        .unwrap_or(0.0)
}

/// Score one query's ranked results against its judgments.
pub fn score_query(hits: &[Hit], judgments: &[&Judgment], k: usize) -> QueryReport {
    let answers = Answers {
        spans: judgments.to_vec(),
    };

    // File level: collapse to first appearance of each file, then score.
    let mut seen_files: BTreeSet<&str> = BTreeSet::new();
    let mut file_grades: Vec<u32> = Vec::new();
    for hit in hits {
        if seen_files.insert(hit.resource.as_str()) {
            file_grades.push(answers.file_grade(&hit.resource));
        }
    }

    // Span level: a judged span already credited must not be credited again,
    // or overlapping windows would inflate recall against a single answer.
    let mut claimed: BTreeSet<usize> = BTreeSet::new();
    let mut span_grades: Vec<u32> = Vec::new();
    for hit in hits {
        match answers.span_grade(hit) {
            Some((index, grade)) if claimed.insert(index) => span_grades.push(grade),
            Some(_) => {}
            None => span_grades.push(0),
        }
    }

    let ideal_files = answers.ideal_file_grades();
    let ideal_spans = answers.ideal_span_grades();

    let relevant_files = ideal_files.len().max(1);
    let relevant_spans = ideal_spans.len().max(1);

    let found_files = file_grades.iter().take(k).filter(|g| **g > 0).count();
    let found_spans = span_grades.iter().take(k).filter(|g| **g > 0).count();

    QueryReport {
        file: Scores {
            ndcg: ndcg(&file_grades, &ideal_files, k),
            recall: found_files as f64 / relevant_files as f64,
            mrr: mrr(&file_grades[..file_grades.len().min(k)]),
        },
        span: Scores {
            ndcg: ndcg(&span_grades, &ideal_spans, k),
            recall: found_spans as f64 / relevant_spans as f64,
            mrr: mrr(&span_grades[..span_grades.len().min(k)]),
        },
        distinct_files: seen_files.len(),
        returned: hits.len().min(k),
    }
}

/// Mean of per-query reports.
pub fn mean(reports: &[QueryReport]) -> QueryReport {
    if reports.is_empty() {
        return QueryReport::default();
    }
    let n = reports.len() as f64;
    let avg = |f: fn(&QueryReport) -> f64| reports.iter().map(f).sum::<f64>() / n;
    QueryReport {
        file: Scores {
            ndcg: avg(|r| r.file.ndcg),
            recall: avg(|r| r.file.recall),
            mrr: avg(|r| r.file.mrr),
        },
        span: Scores {
            ndcg: avg(|r| r.span.ndcg),
            recall: avg(|r| r.span.recall),
            mrr: avg(|r| r.span.mrr),
        },
        distinct_files: (reports.iter().map(|r| r.distinct_files).sum::<usize>() as f64 / n).round()
            as usize,
        returned: (reports.iter().map(|r| r.returned).sum::<usize>() as f64 / n).round() as usize,
    }
}

/// Parse JSONL, reporting the offending line number on failure.
pub fn parse_jsonl<T: for<'de> Deserialize<'de>>(text: &str) -> Result<Vec<T>, String> {
    text.lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(i, l)| serde_json::from_str(l).map_err(|e| format!("line {}: {e}", i + 1)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn judgment(resource: &str, start: usize, end: usize, grade: u32) -> Judgment {
        Judgment {
            query_id: "q".into(),
            resource: resource.into(),
            start_line: start,
            end_line: end,
            grade,
            why: String::new(),
        }
    }

    fn hit(resource: &str, start: usize, end: usize) -> Hit {
        Hit {
            resource: resource.into(),
            start_line: start,
            end_line: end,
        }
    }

    #[test]
    fn a_perfect_ranking_scores_one() {
        let js = [judgment("a.rs", 10, 20, 2), judgment("b.rs", 1, 5, 1)];
        let refs: Vec<&Judgment> = js.iter().collect();
        let hits = vec![hit("a.rs", 10, 20), hit("b.rs", 1, 5)];

        let r = score_query(&hits, &refs, 10);
        assert!(
            (r.file.ndcg - 1.0).abs() < 1e-9,
            "file ndcg {}",
            r.file.ndcg
        );
        assert!(
            (r.span.ndcg - 1.0).abs() < 1e-9,
            "span ndcg {}",
            r.span.ndcg
        );
        assert_eq!(r.file.recall, 1.0);
        assert_eq!(r.file.mrr, 1.0);
    }

    #[test]
    fn ranking_the_answer_below_the_context_scores_less_than_perfect() {
        // The whole point of grading: order matters, and putting supporting
        // context above the actual answer is a worse result.
        let js = [judgment("a.rs", 10, 20, 2), judgment("b.rs", 1, 5, 1)];
        let refs: Vec<&Judgment> = js.iter().collect();

        let best = score_query(&[hit("a.rs", 10, 20), hit("b.rs", 1, 5)], &refs, 10);
        let worse = score_query(&[hit("b.rs", 1, 5), hit("a.rs", 10, 20)], &refs, 10);
        assert!(
            worse.file.ndcg < best.file.ndcg,
            "{} should be < {}",
            worse.file.ndcg,
            best.file.ndcg
        );
    }

    #[test]
    fn nothing_relevant_scores_zero() {
        let js = [judgment("a.rs", 10, 20, 2)];
        let refs: Vec<&Judgment> = js.iter().collect();
        let r = score_query(&[hit("z.rs", 1, 40)], &refs, 10);
        assert_eq!(r.file.ndcg, 0.0);
        assert_eq!(r.file.recall, 0.0);
        assert_eq!(r.file.mrr, 0.0);
    }

    #[test]
    fn a_window_overlapping_the_answer_counts_as_finding_it() {
        // A 40-line window that clips the start of a judged function has found
        // it. Requiring containment would score the segmenter, not retrieval.
        let js = [judgment("a.rs", 100, 140, 2)];
        let refs: Vec<&Judgment> = js.iter().collect();
        let r = score_query(&[hit("a.rs", 81, 120)], &refs, 10);
        assert_eq!(r.span.recall, 1.0);
    }

    #[test]
    fn a_window_stopping_short_of_the_answer_does_not_count() {
        let js = [judgment("a.rs", 100, 140, 2)];
        let refs: Vec<&Judgment> = js.iter().collect();
        let r = score_query(&[hit("a.rs", 41, 80)], &refs, 10);
        assert_eq!(r.span.recall, 0.0);
    }

    #[test]
    fn repeated_windows_of_one_file_are_credited_once() {
        // Six overlapping windows of the same file have found one thing. A
        // metric that pays for all six measures redundancy as if it were
        // recall, which is exactly the failure mode being watched for.
        let js = [judgment("a.rs", 10, 20, 2), judgment("b.rs", 1, 5, 2)];
        let refs: Vec<&Judgment> = js.iter().collect();

        let hits = vec![
            hit("a.rs", 1, 40),
            hit("a.rs", 9, 48),
            hit("a.rs", 17, 56),
            hit("a.rs", 25, 64),
        ];
        let r = score_query(&hits, &refs, 10);
        assert_eq!(r.distinct_files, 1);
        assert_eq!(r.file.recall, 0.5, "one of two judged files found");
        assert_eq!(r.span.recall, 0.5, "one of two judged spans found");
    }

    #[test]
    fn k_truncates_before_scoring() {
        let js = [judgment("a.rs", 1, 5, 2)];
        let refs: Vec<&Judgment> = js.iter().collect();
        let hits = vec![hit("x.rs", 1, 5), hit("y.rs", 1, 5), hit("a.rs", 1, 5)];

        assert_eq!(score_query(&hits, &refs, 2).file.recall, 0.0);
        assert_eq!(score_query(&hits, &refs, 3).file.recall, 1.0);
    }

    #[test]
    fn mrr_reports_the_rank_of_the_first_hit() {
        let js = [judgment("a.rs", 1, 5, 2)];
        let refs: Vec<&Judgment> = js.iter().collect();
        let hits = vec![hit("x.rs", 1, 5), hit("a.rs", 1, 5)];
        assert_eq!(score_query(&hits, &refs, 10).file.mrr, 0.5);
    }

    #[test]
    fn an_empty_run_scores_zero_rather_than_panicking() {
        let js = [judgment("a.rs", 1, 5, 2)];
        let refs: Vec<&Judgment> = js.iter().collect();
        let r = score_query(&[], &refs, 10);
        assert_eq!(r.file.ndcg, 0.0);
        assert_eq!(r.returned, 0);
    }

    #[test]
    fn jsonl_errors_name_the_line() {
        let bad = "{\"id\":\"a\",\"text\":\"t\"}\nnot json\n";
        let err = parse_jsonl::<Query>(bad).unwrap_err();
        assert!(err.starts_with("line 2:"), "{err}");
    }
}
