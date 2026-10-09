//! Reference segmenter.

use map_core::{Content, Result, Segment, Segmenter, Stage};

/// Splits content into line-bounded windows with overlap.
///
/// Deliberately structure-blind and deterministic: cutting on syntax would need
/// a grammar per format, and this runs on any text resource without one. The
/// cost is that a boundary falls wherever the line count lands, not where the
/// content has a natural seam — a function, a paragraph, a record.
///
/// Overlap matters more than it looks: anything defined at a window boundary
/// would otherwise land in neither segment's descriptor.
///
/// The defaults are set by `[segmenter]` in config and were measured on source
/// code. A line is a poor unit for prose, where paragraphs vary far more in
/// length, and a purely line-based cut makes a single-line resource one segment
/// however large it is. Both are real limits of this implementation rather than
/// of the pipeline: a structure-aware segmenter is another `impl`.
#[derive(Clone, Debug)]
pub struct WindowSegmenter {
    /// Lines per segment.
    pub lines: usize,
    /// Lines of overlap between consecutive segments.
    pub overlap: usize,
}

impl Default for WindowSegmenter {
    fn default() -> Self {
        WindowSegmenter {
            lines: 40,
            overlap: 8,
        }
    }
}

impl Stage for WindowSegmenter {
    fn implementation(&self) -> &str {
        "window"
    }

    fn config(&self) -> String {
        format!("window:lines={},overlap={}", self.lines, self.overlap)
    }
}

impl Segmenter for WindowSegmenter {
    fn segment(&self, content: &Content) -> Result<Vec<Segment>> {
        // Byte offset of the start of each line, plus a sentinel at the end.
        let mut line_starts: Vec<u32> = vec![0];
        for (i, b) in content.text.bytes().enumerate() {
            if b == b'\n' {
                line_starts.push(i as u32 + 1);
            }
        }
        let total_len = content.text.len() as u32;
        if total_len == 0 {
            return Ok(Vec::new());
        }

        let line_count = line_starts.len();
        let stride = self.lines.saturating_sub(self.overlap).max(1);

        let mut segments = Vec::new();
        let mut first_line = 0usize;
        loop {
            let last_line = (first_line + self.lines).min(line_count);
            let start = line_starts[first_line];
            let end = if last_line >= line_count {
                total_len
            } else {
                line_starts[last_line]
            };

            if end > start {
                segments.push(Segment { start, end });
            }
            if last_line >= line_count {
                break;
            }
            first_line += stride;
        }
        Ok(segments)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn content(text: &str) -> Content {
        Content {
            key: "t.rs".into(),
            text: text.into(),
        }
    }

    #[test]
    fn short_content_is_one_segment() {
        let c = content("a\nb\nc\n");
        let segs = WindowSegmenter::default().segment(&c).unwrap();
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].slice(&c), "a\nb\nc\n");
    }

    #[test]
    fn segments_cover_the_whole_input() {
        let text: String = (0..200).map(|i| format!("line {i}\n")).collect();
        let c = content(&text);
        let segs = WindowSegmenter::default().segment(&c).unwrap();

        assert!(segs.len() > 1);
        assert_eq!(segs[0].start, 0);
        assert_eq!(
            segs.last().unwrap().end,
            c.text.len() as u32,
            "last segment must reach the end of the content"
        );
        // Overlap means consecutive segments must not leave a gap.
        for pair in segs.windows(2) {
            assert!(pair[1].start < pair[0].end, "gap between segments");
        }
    }

    #[test]
    fn segmentation_is_deterministic() {
        let text: String = (0..500).map(|i| format!("line {i}\n")).collect();
        let c = content(&text);
        let s = WindowSegmenter::default();
        assert_eq!(s.segment(&c).unwrap(), s.segment(&c).unwrap());
    }

    #[test]
    fn empty_content_yields_no_segments() {
        assert!(WindowSegmenter::default()
            .segment(&content(""))
            .unwrap()
            .is_empty());
    }

    #[test]
    fn config_string_tracks_settings() {
        let a = WindowSegmenter::default();
        let b = WindowSegmenter {
            lines: 80,
            ..Default::default()
        };
        assert_ne!(a.config(), b.config());
    }
}
