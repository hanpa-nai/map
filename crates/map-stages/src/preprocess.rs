//! Text preprocessor: decode, normalize, reject binaries.

use map_core::{Content, Preprocessor, Resource, Result, Stage};

/// Decodes UTF-8 text and normalizes line endings to LF.
#[derive(Clone, Debug, Default)]
pub struct TextPreprocessor;

/// Normalize line endings to LF.
///
/// This is the single most important cross-platform normalization (spec §6.1).
/// A repository cloned with `core.autocrlf=true` on Windows has different
/// working-tree bytes than the same commit on Linux, and since segment
/// boundaries and object keys derive from content, skipping this would produce
/// a *different index per platform* — a Windows-built index would fail
/// verification everywhere else.
///
/// Handles CRLF and lone CR. Trailing whitespace and final newlines are left
/// alone: those are real content.
pub fn normalize_line_endings(input: &str) -> String {
    if !input.contains('\r') {
        return input.to_owned();
    }
    let mut out = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\r' {
            // Consume the LF of a CRLF pair; a lone CR still becomes LF.
            if chars.peek() == Some(&'\n') {
                chars.next();
            }
            out.push('\n');
        } else {
            out.push(c);
        }
    }
    out
}

/// Whether a byte slice looks like text we can index.
fn looks_textual(bytes: &[u8]) -> bool {
    // A NUL in the first block is the classic binary tell, and it is what git
    // itself uses.
    let probe = &bytes[..bytes.len().min(8000)];
    !probe.contains(&0)
}

impl Stage for TextPreprocessor {
    fn implementation(&self) -> &str {
        "text"
    }

    fn config(&self) -> String {
        "text:v1".to_owned()
    }
}

impl Preprocessor for TextPreprocessor {
    fn preprocess(&self, resource: &Resource, bytes: &[u8]) -> Result<Option<Content>> {
        // Checked here as well as at the driver: a key that escapes the root
        // must never reach a stage, and there is no cost to asserting it twice.
        crate::discover::check_resource_key(&resource.key)?;

        if !looks_textual(bytes) {
            return Ok(None);
        }
        let Ok(text) = std::str::from_utf8(bytes) else {
            return Ok(None);
        };

        Ok(Some(Content {
            key: resource.key.clone(),
            text: normalize_line_endings(text),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crlf_becomes_lf() {
        assert_eq!(normalize_line_endings("a\r\nb\r\n"), "a\nb\n");
    }

    #[test]
    fn lone_cr_becomes_lf() {
        assert_eq!(normalize_line_endings("a\rb"), "a\nb");
    }

    #[test]
    fn lf_only_input_is_untouched() {
        let s = "a\nb\n";
        assert_eq!(normalize_line_endings(s), s);
    }

    #[test]
    fn windows_and_unix_checkouts_agree() {
        // The whole point: the same commit checked out either way must yield
        // identical bytes downstream, or the index differs per platform.
        assert_eq!(
            normalize_line_endings("fn main() {\r\n    println!();\r\n}\r\n"),
            normalize_line_endings("fn main() {\n    println!();\n}\n"),
        );
    }

    #[test]
    fn trailing_whitespace_is_preserved() {
        assert_eq!(normalize_line_endings("a   \n\n"), "a   \n\n");
    }

    #[test]
    fn binary_is_detected() {
        assert!(!looks_textual(b"\x7fELF\0\0\0"));
        assert!(looks_textual(b"fn main() {}"));
    }
}
