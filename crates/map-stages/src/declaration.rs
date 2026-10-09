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

use map_core::{Classifier, ClassifyBatch, DimensionRecords, Result, Stage};
use map_format::{Record, RecordKind, RecordMeta};

use crate::lexical::{encode_frequencies, tokenize};

pub struct DeclarationClassifier;

impl Stage for DeclarationClassifier {
    fn implementation(&self) -> &str {
        "declaration"
    }

    fn config(&self) -> String {
        "declaration:v1".to_owned()
    }
}

impl Classifier for DeclarationClassifier {
    fn classify(&self, batch: &ClassifyBatch<'_>) -> Result<Vec<DimensionRecords>> {
        let mut out = Vec::with_capacity(batch.items.len());
        for sources in batch.items {
            let mut terms = Vec::new();
            for text in sources {
                for line in text.lines() {
                    if let Some(name) = declared_name(line) {
                        terms.extend(tokenize(name));
                    }
                }
            }
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
    use super::declared_name;

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
