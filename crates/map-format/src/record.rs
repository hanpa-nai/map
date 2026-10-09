//! The record — the only searchable unit in the format (spec §3).
//!
//! ```text
//! record = { descriptor: text?, tensor: numeric?, metadata }
//! ```
//!
//! Both payloads are **opaque**. The descriptor is freeform text and the
//! tensor is freeform numeric; the format stores them and never interprets
//! them. Only the owning dimension's stage implementations know what they
//! mean — the classifier writes them, the retriever's scorer reads them.
//!
//! This is why BM25 needs no special case anywhere in this crate: token
//! frequencies are simply what the lexical dimension chooses to put in its
//! descriptor text. A new retrieval method is a new stage implementation,
//! never a format change.
//!
//! **At least one payload must be non-empty.** A record with neither is
//! unsearchable and is rejected.
//!
//! Segments and clusters are the same type. Clusters differ only in their
//! metadata (children and level) and in how their identity is computed
//! ([`crate::hash::ObjectKey::cluster`]).
//!
//! A stored record carries **no resource identity**. Where a segment lives is
//! reattached at load time from the shared segmentation, which is what lets two
//! byte-identical files derive one object key and share a single stored object.

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::hash::ObjectKey;

/// Element type of a tensor payload.
///
/// Structural, not semantic: needed to size and map the buffer, not to
/// interpret what the numbers mean.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[repr(u8)]
pub enum DType {
    /// IEEE-754 single precision.
    F32 = 1,
    /// IEEE-754 half precision.
    F16 = 2,
    /// Signed 8-bit, for quantized vectors.
    I8 = 3,
    /// One bit per component, packed eight to a byte.
    Binary = 4,
}

impl DType {
    /// Decode from the on-disk discriminant.
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(DType::F32),
            2 => Some(DType::F16),
            3 => Some(DType::I8),
            4 => Some(DType::Binary),
            _ => None,
        }
    }

    /// Human-readable name, used in error messages.
    pub fn name(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::I8 => "i8",
            DType::Binary => "binary",
        }
    }

    /// Bytes required for `count` elements, or `None` on overflow.
    ///
    /// [`DType::Binary`] packs eight components per byte, rounding up.
    ///
    /// Checked rather than wrapping because this runs on untrusted committed
    /// bytes: a wrapped product can yield a required size of zero for a huge
    /// declared shape, which would let an empty buffer validate.
    pub fn bytes_for(self, count: usize) -> Option<usize> {
        match self {
            DType::F32 => count.checked_mul(4),
            DType::F16 => count.checked_mul(2),
            DType::I8 => Some(count),
            DType::Binary => Some(count.div_ceil(8)),
        }
    }
}

/// The numeric payload. Opaque to the format.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tensor {
    pub dtype: DType,
    /// Dimensions, outermost first.
    pub shape: Vec<u32>,
    /// Little-endian element bytes.
    pub data: Vec<u8>,
}

impl Tensor {
    /// Total element count implied by [`Tensor::shape`], or `None` on overflow.
    pub fn element_count(&self) -> Option<usize> {
        self.shape
            .iter()
            .try_fold(1usize, |acc, d| acc.checked_mul(*d as usize))
    }

    /// Check that the buffer length matches the declared shape and dtype.
    pub fn validate(&self) -> Result<()> {
        let overflow = || Error::TensorShapeOverflow {
            shape: self.shape.clone(),
        };
        let expected = self
            .element_count()
            .and_then(|n| self.dtype.bytes_for(n))
            .ok_or_else(overflow)?;

        if expected != self.data.len() {
            return Err(Error::TensorShapeMismatch {
                shape: self.shape.clone(),
                dtype: self.dtype.name(),
                expected,
                actual: self.data.len(),
            });
        }
        Ok(())
    }
}

/// Whether a record describes a segment of a resource or a cluster of records.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RecordKind {
    /// A span of an underlying resource.
    Segment,
    /// A node in the fabric, grouping other records.
    Cluster,
}

/// Everything about a record that is not a searchable payload.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordMeta {
    pub kind: RecordKind,
    pub dimension: String,
    /// Height in the fabric. Segments are level 0.
    ///
    /// Retrieval "zoom" is a filter on this field, not a separate mechanism:
    /// bias toward higher levels for a cheap structural overview, toward
    /// level 0 for precision.
    pub level: u16,
    /// Child records, for clusters. Empty for segments.
    ///
    /// Stored sorted so that an unchanged subtree serializes identically
    /// regardless of the order the fabricator emitted it in.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<ObjectKey>,
}

/// A searchable record. See module docs.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Record {
    /// Freeform text. Opaque to the format.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub descriptor: Option<String>,
    /// Freeform numeric. Opaque to the format.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tensor: Option<Tensor>,
    pub meta: RecordMeta,
}

impl Record {
    /// Whether this record carries anything searchable.
    ///
    /// An empty descriptor string counts as absent: a dimension that emits
    /// `""` has produced nothing to search, and letting it through would put
    /// an unreachable record in the index.
    pub fn is_searchable(&self) -> bool {
        let has_text = self.descriptor.as_ref().is_some_and(|d| !d.is_empty());
        let has_numeric = self.tensor.as_ref().is_some_and(|t| !t.data.is_empty());
        has_text || has_numeric
    }

    /// Enforce the spec §3 invariant, kind consistency, and tensor consistency.
    pub fn validate(&self) -> Result<()> {
        if !self.is_searchable() {
            return Err(Error::EmptyRecord);
        }

        // Segments and clusters share a type, so nothing structural stops a
        // producer from emitting a segment with children, which would silently
        // corrupt fabric traversal.
        if self.meta.kind == RecordKind::Segment && !self.meta.children.is_empty() {
            return Err(Error::InconsistentRecord("segment record has children"));
        }

        // Children must already be sorted. Merkle identity and byte-identity
        // both depend on it, and `canonicalize` is easy to forget.
        if self.meta.children.windows(2).any(|w| w[0] > w[1]) {
            return Err(Error::InconsistentRecord(
                "children are not sorted; call Record::canonicalize before validating",
            ));
        }

        if let Some(t) = &self.tensor {
            t.validate()?;
        }
        Ok(())
    }

    /// Sort children into canonical order. Call before serializing a cluster.
    pub fn canonicalize(&mut self) {
        self.meta.children.sort_unstable();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::{Fingerprint, ObjectKey};

    fn meta(kind: RecordKind) -> RecordMeta {
        RecordMeta {
            kind,
            dimension: "lexical".into(),
            level: 0,
            children: vec![],
        }
    }

    #[test]
    fn descriptor_only_is_valid() {
        // The lexical dimension: BM25 term frequencies in the text field,
        // no tensor at all.
        let r = Record {
            descriptor: Some("refresh:3 token:2 session:1".into()),
            tensor: None,
            meta: meta(RecordKind::Segment),
        };
        assert!(r.validate().is_ok());
    }

    #[test]
    fn tensor_only_is_valid() {
        let r = Record {
            descriptor: None,
            tensor: Some(Tensor {
                dtype: DType::Binary,
                shape: vec![768],
                data: vec![0; 96],
            }),
            meta: meta(RecordKind::Segment),
        };
        assert!(r.validate().is_ok());
    }

    #[test]
    fn neither_payload_is_rejected() {
        let r = Record {
            descriptor: None,
            tensor: None,
            meta: meta(RecordKind::Segment),
        };
        assert!(matches!(r.validate(), Err(Error::EmptyRecord)));
    }

    #[test]
    fn empty_descriptor_counts_as_absent() {
        // Otherwise a dimension emitting "" would put an unreachable record
        // into the index and still pass validation.
        let r = Record {
            descriptor: Some(String::new()),
            tensor: None,
            meta: meta(RecordKind::Segment),
        };
        assert!(matches!(r.validate(), Err(Error::EmptyRecord)));
    }

    #[test]
    fn binary_dtype_packs_eight_per_byte() {
        assert_eq!(DType::Binary.bytes_for(768), Some(96));
        assert_eq!(DType::Binary.bytes_for(1), Some(1));
        assert_eq!(DType::Binary.bytes_for(9), Some(2));
        assert_eq!(DType::F32.bytes_for(768), Some(3072));
        assert_eq!(DType::I8.bytes_for(768), Some(768));
    }

    #[test]
    fn hostile_shape_cannot_wrap_into_a_valid_empty_tensor() {
        // Unchecked, this shape multiplies out to 2^62; times 4 bytes for f32
        // it wraps to exactly 0, so an empty buffer would validate against a
        // declared four-exabyte tensor. Reachable from committed bytes.
        let t = Tensor {
            dtype: DType::F32,
            shape: vec![1 << 31, 1 << 31],
            data: vec![],
        };
        assert!(
            matches!(t.validate(), Err(Error::TensorShapeOverflow { .. })),
            "overflowing shape must be rejected, not wrapped"
        );
    }

    #[test]
    fn element_count_reports_overflow_rather_than_panicking() {
        let t = Tensor {
            dtype: DType::I8,
            shape: vec![u32::MAX, u32::MAX, u32::MAX],
            data: vec![],
        };
        assert_eq!(t.element_count(), None);
    }

    #[test]
    fn segment_with_children_is_rejected() {
        let fp = Fingerprint::of(b"cfg");
        let r = Record {
            descriptor: Some("x".into()),
            tensor: None,
            meta: RecordMeta {
                children: vec![ObjectKey::derive(&[b"a"], fp)],
                ..meta(RecordKind::Segment)
            },
        };
        assert!(matches!(r.validate(), Err(Error::InconsistentRecord(_))));
    }

    #[test]
    fn unsorted_children_are_rejected() {
        // Merkle identity and byte-identity both depend on sorted children,
        // and canonicalize() is easy to forget.
        let fp = Fingerprint::of(b"cfg");
        let a = ObjectKey::derive(&[b"a"], fp);
        let b = ObjectKey::derive(&[b"b"], fp);
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };

        let r = Record {
            descriptor: Some("label".into()),
            tensor: None,
            meta: RecordMeta {
                children: vec![hi, lo],
                ..meta(RecordKind::Cluster)
            },
        };
        assert!(matches!(r.validate(), Err(Error::InconsistentRecord(_))));
    }

    #[test]
    fn shape_mismatch_is_rejected() {
        let t = Tensor {
            dtype: DType::F32,
            shape: vec![4],
            data: vec![0; 8], // needs 16
        };
        assert!(t.validate().is_err());
    }

    #[test]
    fn canonicalize_sorts_children() {
        let fp = Fingerprint::of(b"cfg");
        let a = ObjectKey::derive(&[b"a"], fp);
        let b = ObjectKey::derive(&[b"b"], fp);
        let c = ObjectKey::derive(&[b"c"], fp);

        let mut one = Record {
            descriptor: Some("cluster label".into()),
            tensor: None,
            meta: RecordMeta {
                children: vec![c, a, b],
                ..meta(RecordKind::Cluster)
            },
        };
        let mut two = Record {
            meta: RecordMeta {
                children: vec![b, c, a],
                ..one.meta.clone()
            },
            ..one.clone()
        };
        one.canonicalize();
        two.canonicalize();
        assert_eq!(one, two, "emit order must not affect stored bytes");
    }
}
