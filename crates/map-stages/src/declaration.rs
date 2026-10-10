//! A classifier that emits only the names a segment *declares*.
//!
//! BM25 over whole content ranks the files that *use* a name above the one
//! that defines it, because callers mention a name more often than its one
//! definition does. A dimension built from declared names alone gives a query
//! for `OrderRepository` a place where the definition is the only record that
//! matches at all.
//!
//! This is a line-oriented keyword scan, not a parser: it recognizes the
//! declaration keywords most languages share (`fn`, `def`, `class`, `struct`,
//! `function`, `type`, ...) and the `name(...) {` method shape. A grammar per
//! format would be more precise, and is a different `impl` behind the same
//! stage; this one runs on any text and costs nothing to build.
//!
//! # A name and the parts of a name
//!
//! `config_fingerprint` is found by `config` and by `fingerprint` as well as by
//! its whole name, so that a query does not need the exact spelling. But a
//! query for `fingerprint` names the thing that is called `fingerprint`, and
//! a part must not compete with that on equal terms. So the descriptor of a
//! segment counts a declared name [`WHOLE_NAME`] times, and a part of a longer
//! name one time, however many names of the segment have that part. Without
//! the second rule a segment that declares three names ending in
//! `_fingerprint` holds the word three times, and outranks the one segment
//! that declares `fingerprint` itself.

use std::collections::BTreeSet;

use map_core::{Classifier, ClassifyBatch, DimensionRecords, Result, Stage};
use map_format::{Record, RecordKind, RecordMeta};

use crate::lexical::{encode_frequencies, tokenize};

/// How many times a declared name counts in the descriptor of its segment.
///
/// BM25 saturates with term frequency, so no count makes a name worth more
/// than about twice a part. 3 puts a name above its parts. A larger count
/// changes that order little, and it makes a query of plain words follow the
/// names that happen to be those words.
const WHOLE_NAME: usize = 3;

pub struct DeclarationClassifier;

impl Stage for DeclarationClassifier {
    fn implementation(&self) -> &str {
        "declaration"
    }

    fn config(&self) -> String {
        "declaration:v2".to_owned()
    }
}

impl Classifier for DeclarationClassifier {
    fn classify(&self, batch: &ClassifyBatch<'_>) -> Result<Vec<DimensionRecords>> {
        let mut out = Vec::with_capacity(batch.items.len());
        for sources in batch.items {
            let names = sources
                .iter()
                .flat_map(|text| text.lines())
                .filter_map(declared_name);
            let terms = descriptor_terms(names);
            let mut per_dimension = DimensionRecords::new();
            // A segment that declares nothing gets no record in this dimension:
            // an empty descriptor is unsearchable and the pack builder skips
            // the slot, which is the behaviour wanted.
            if !terms.is_empty() {
                let descriptor = encode_frequencies(&terms);
                for dimension in batch.dimensions {
                    per_dimension.insert(
                        dimension.clone(),
                        Record {
                            descriptor: Some(descriptor.clone()),
                            tensor: None,
                            meta: RecordMeta {
                                kind: if batch.level == 0 {
                                    RecordKind::Segment
                                } else {
                                    RecordKind::Cluster
                                },
                                dimension: dimension.clone(),
                                level: batch.level,
                                children: Vec::new(),
                            },
                        },
                    );
                }
            }
            out.push(per_dimension);
        }
        Ok(out)
    }
}

/// The terms of a segment that declares `names`, each as often as it counts.
///
/// The count of a term does not depend on the order of the names: a set holds
/// the parts, and a part that is also a declared name keeps the two counts.
fn descriptor_terms<'a>(names: impl Iterator<Item = &'a str>) -> Vec<String> {
    let mut terms = Vec::new();
    let mut parts = BTreeSet::new();
    for name in names {
        let whole = name.to_lowercase();
        parts.extend(tokenize(name).into_iter().filter(|term| *term != whole));
        terms.extend(std::iter::repeat_n(whole, WHOLE_NAME));
    }
    terms.extend(parts);
    terms
}

/// Words that may precede a declaration keyword without changing its meaning.
const MODIFIERS: &[&str] = &[
    "pub",
    "export",
    "default",
    "async",
    "unsafe",
    "abstract",
    "readonly",
    "private",
    "public",
    "protected",
    "extern",
    "declare",
    "final",
    "override",
    "inline",
    "virtual",
    "internal",
    "static",
    "const",
    "constexpr",
    "sealed",
    "open",
    "data",
    "partial",
];

/// Keywords whose next word is the declared name.
const KEYWORDS: &[&str] = &[
    "fn",
    "func",
    "function",
    "def",
    "defn",
    "class",
    "struct",
    "enum",
    "union",
    "trait",
    "interface",
    "type",
    "typedef",
    "mod",
    "module",
    "namespace",
    "const",
    "static",
    "object",
    "record",
    "protocol",
    "macro_rules!",
    "impl",
    "proc",
    "sub",
];

/// Words that look like `name(` but open control flow, not a declaration.
const CONTROL: &[&str] = &[
    "if", "for", "while", "switch", "catch", "return", "else", "function", "do", "try", "match",
    "loop", "new", "await", "throw", "typeof", "super", "this", "with", "unless", "until", "elif",
    "except", "foreach", "defer", "go", "select", "sizeof", "yield", "case",
];

/// The name a source line declares, if the line is a declaration.
pub fn declared_name(line: &str) -> Option<&str> {
    let trimmed = line.trim_start();
    if trimmed.starts_with("//") || trimmed.starts_with('#') || trimmed.starts_with('*') {
        return None;
    }
    let words: Vec<&str> = trimmed.split_whitespace().collect();
    let mut i = 0;
    // `const fn` and `static async` are modifiers; `const NAME` and `static
    // NAME` are keywords. Treat the word as a modifier only when another
    // keyword or modifier follows it.
    while i < words.len() && MODIFIERS.contains(&strip_parens(words[i])) {
        let next = words.get(i + 1).map(|w| strip_parens(w));
        let is_keyword_too = KEYWORDS.contains(&strip_parens(words[i]));
        match next {
            Some(n) if KEYWORDS.contains(&n) || MODIFIERS.contains(&n) => i += 1,
            _ if is_keyword_too => break,
            _ => i += 1,
        }
    }
    let &head = words.get(i)?;
    let keyword = head.split(['<', '(']).next().unwrap_or(head);
    if KEYWORDS.contains(&keyword) {
        if keyword == "impl" {
            // `impl<T> Trait for Name<T> {` names `Name`; `impl Name {` names `Name`.
            let rest = &words[i + 1..];
            if let Some(pos) = rest.iter().position(|w| *w == "for") {
                return rest.get(pos + 1).and_then(|w| identifier(w));
            }
            let after = if head.contains('<') && !head.ends_with('>') {
                // `impl<T: Bound>` can span several words; skip to the closing `>`.
                rest.iter()
                    .position(|w| w.ends_with('>'))
                    .map(|p| p + 1)
                    .unwrap_or(0)
            } else {
                0
            };
            return rest.get(after).and_then(|w| identifier(w));
        }
        // `static mut COUNTER` and `const mut` name the word after `mut`.
        let name = words
            .get(i + 1)
            .filter(|w| **w != "mut")
            .or_else(|| words.get(i + 2));
        return name.and_then(|w| identifier(w));
    }
    // `name(args) {` or `int name(args) {`: a method or C-style function body.
    if trimmed.ends_with('{') || trimmed.ends_with("=> {") {
        for w in words.iter().skip(i).take(2) {
            if let Some(open) = w.find('(') {
                let name = &w[..open];
                let name = name.split('<').next().unwrap_or(name);
                if is_identifier(name) && !CONTROL.contains(&name) {
                    return Some(name);
                }
                break;
            }
        }
    }
    None
}

fn strip_parens(word: &str) -> &str {
    word.split('(').next().unwrap_or(word)
}

/// The leading identifier of a word such as `Name<T>(`, `Name:`, or `Name;`.
fn identifier(word: &str) -> Option<&str> {
    let end = word
        .char_indices()
        .find(|(_, c)| !(c.is_alphanumeric() || *c == '_'))
        .map(|(i, _)| i)
        .unwrap_or(word.len());
    let name = &word[..end];
    is_identifier(name).then_some(name)
}

fn is_identifier(name: &str) -> bool {
    name.len() >= 2
        && name
            .chars()
            .next()
            .is_some_and(|c| c.is_alphabetic() || c == '_')
        && name.chars().all(|c| c.is_alphanumeric() || c == '_')
}

#[cfg(test)]
mod tests {
    use super::{declared_name, descriptor_terms, WHOLE_NAME};
    use crate::lexical::{decode_frequencies, encode_frequencies};

    /// The count of `term` in the descriptor of a segment that declares `names`.
    fn count(names: &[&str], term: &str) -> u32 {
        let descriptor = encode_frequencies(&descriptor_terms(names.iter().copied()));
        decode_frequencies(&descriptor)
            .into_iter()
            .find(|(t, _)| *t == term)
            .map_or(0, |(_, n)| n)
    }

    #[test]
    fn a_declared_name_counts_more_than_a_part_of_a_longer_name() {
        let whole = WHOLE_NAME as u32;
        assert_eq!(count(&["fingerprint"], "fingerprint"), whole);
        assert_eq!(count(&["config_fingerprint"], "fingerprint"), 1);
        assert_eq!(count(&["config_fingerprint"], "config_fingerprint"), whole);
        // camelCase splits in the same way, and the name is kept in lower case.
        assert_eq!(count(&["LineStep"], "linestep"), whole);
        assert_eq!(count(&["LineStep"], "step"), 1);
    }

    #[test]
    fn a_part_counts_one_time_however_many_names_of_the_segment_have_it() {
        // Three names that end in the word must not add up to the weight of
        // the one name that is the word.
        let names = [
            "config_fingerprint",
            "artifact_fingerprint",
            "fab_fingerprint",
        ];
        assert_eq!(count(&names, "fingerprint"), 1);
        // Each declaration of the name itself does count.
        assert_eq!(
            count(&["Config", "Config"], "config"),
            2 * WHOLE_NAME as u32
        );
    }

    #[test]
    fn the_count_of_a_term_does_not_depend_on_the_order_of_the_names() {
        let a = count(&["fingerprint", "config_fingerprint"], "fingerprint");
        let b = count(&["config_fingerprint", "fingerprint"], "fingerprint");
        assert_eq!(a, b);
        assert_eq!(a, WHOLE_NAME as u32 + 1, "the name, and the part");
    }

    #[test]
    fn keyword_declarations_name_the_next_word() {
        assert_eq!(
            declared_name("pub fn parse_flags(args: &[String]) -> Flags {"),
            Some("parse_flags")
        );
        assert_eq!(
            declared_name("    pub(crate) struct LineStep {"),
            Some("LineStep")
        );
        assert_eq!(
            declared_name("export class OrderRepository {"),
            Some("OrderRepository")
        );
        assert_eq!(
            declared_name("export const formatMoney = (m: Money): string => {"),
            Some("formatMoney")
        );
        assert_eq!(
            declared_name("def hash_password(pw):"),
            Some("hash_password")
        );
        assert_eq!(declared_name("pub const fn new() -> Self {"), Some("new"));
        assert_eq!(declared_name("const MAX: usize = 4;"), Some("MAX"));
        assert_eq!(
            declared_name("impl<'a> Iterator for LineStep<'a> {"),
            Some("LineStep")
        );
        assert_eq!(declared_name("impl GlobMatcher {"), Some("GlobMatcher"));
        assert_eq!(declared_name("macro_rules! bail {"), Some("bail"));
    }

    #[test]
    fn methods_and_c_functions_are_declarations_but_calls_and_control_flow_are_not() {
        assert_eq!(
            declared_name("  async hashPassword(plain: string): Promise<string> {"),
            Some("hashPassword")
        );
        assert_eq!(
            declared_name("int main(int argc, char **argv) {"),
            Some("main")
        );
        assert_eq!(declared_name("  if (x) {"), None);
        assert_eq!(declared_name("  while (true) {"), None);
        assert_eq!(declared_name("  repo.save(order);"), None);
        assert_eq!(declared_name("  hashPassword(pw)"), None);
        assert_eq!(declared_name("// fn not_a_declaration() {"), None);
        // Local bindings are not declarations this dimension indexes.
        assert_eq!(declared_name("let mut count = 0;"), None);
        assert_eq!(
            declared_name("static mut COUNTER: u32 = 0;"),
            Some("COUNTER")
        );
    }
}
