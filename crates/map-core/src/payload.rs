//! Stage payloads — what actually gets stored in objects.
//!
//! These live here rather than in `map-format` on purpose. The format stores
//! opaque bytes and never interprets them; these shapes are an agreement
//! between the stages that write them and the stages that read them back.
//! Changing one is a stage-implementation change, not a format change.

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::{DimensionRecords, DimensionTensors, Segment};

/// One resource's segmentation, stored in the `shared` object space.
///
/// Shared across every dimension because segment identity is the join key for
/// cross-dimension correlation.
///
/// # Why there is no resource field
///
/// These payloads deliberately carry **nothing that identifies the resource**.
/// Object keys derive from content plus stage config, so two files with
/// identical bytes derive the same key — and if the payload named its path,
/// they would produce the same key with different bytes, which is a genuine
/// collision rather than dedup. The resource name lives in the manifest root
/// that points at this object, where it belongs. Two identical files then
/// legitimately share one stored object.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentsPayload {
    /// Spans into normalized content, in ascending order.
    pub segments: Vec<Segment>,
}

/// Records from a single classifier invocation.
///
/// One object per resource per implementation group — the storage granularity
/// deliberately equals the invocation granularity, so editing one dimension's
/// prompt invalidates exactly the group that shares its implementation.
///
/// Carries no resource identity, for the reason given on [`SegmentsPayload`].
/// Each record's `meta.segment` is `None` in stored form; the driver reattaches
/// the resource and span at load time from the manifest root. This is enforced
/// behaviourally by the `identical_files_share_one_stored_object` test in
/// `map-index`, because the failure mode is silent — the index still works, it
/// just stops deduplicating.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DescriptorPayload {
    /// Dimensions this group covers, sorted.
    pub dimensions: Vec<String>,
    /// One record map per segment, parallel to [`SegmentsPayload::segments`].
    pub per_segment: Vec<DimensionRecords>,
}

/// Tensors from a single embedder invocation — the dense analog of
/// [`DescriptorPayload`].
///
/// One object per resource per embedder group, so re-embedding is skipped when
/// the object already exists (the embedder is the expensive stage for a dense
/// dimension, exactly as the classifier is for a lexical one). Carries no
/// resource identity, for the reason given on [`SegmentsPayload`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TensorPayload {
    /// Dimensions this group covers, sorted.
    pub dimensions: Vec<String>,
    /// One tensor map per segment, parallel to [`SegmentsPayload::segments`].
    pub per_segment: Vec<DimensionTensors>,
}

/// Container magic for the binary form of [`TensorPayload`].
const TENSOR_GROUP_MAGIC: &[u8; 8] = b"MAPTGRP\0";
const TENSOR_GROUP_VERSION: u16 = 1;

/// Encode a [`TensorPayload`] as raw little-endian tensor frames.
///
/// This payload is the one place the index stores bulk float data, and raw
/// frames are chosen for the *decode*, not the size. Every fresh clone and
/// cleared cache parses every tensor object before it can answer a first
/// query: 3.6 ms here against 159 ms for a JSON array of byte literals, over
/// ripgrep's 4,916 vectors. Size barely moves by comparison — git deflates
/// those digit arrays 2.84x, leaving only 1.34x in the packfile. Measurements
/// in `spec/format-v1.md` §9.2.
///
/// ```text
/// magic       8   b"MAPTGRP\0"
/// version     2   u16 le
/// dim_count   2   u16 le, then each: len u16 le + utf-8 name (ascending)
/// seg_count   4   u32 le
/// per segment:    entry_count u16 le, then each entry (dim_idx ascending):
///     dim_idx 2   u16 le, an index into the name table
///     len     4   u32 le
///     frame   len bytes — `map_format::codec::encode_tensor`
/// ```
///
/// Both orderings are required rather than merely produced, so the encoding is
/// canonical: exactly one byte string represents a given payload, which is what
/// Tier A determinism and git dedup rest on.
pub fn encode_tensor_payload(payload: &TensorPayload) -> Result<Vec<u8>> {
    let too_many = |_| Error::MalformedPayload("tensor group exceeds its count fields");
    validate_tensor_payload(payload)?;

    let mut out = Vec::new();
    out.extend_from_slice(TENSOR_GROUP_MAGIC);
    out.extend_from_slice(&TENSOR_GROUP_VERSION.to_le_bytes());
    out.extend_from_slice(
        &u16::try_from(payload.dimensions.len())
            .map_err(too_many)?
            .to_le_bytes(),
    );
    for name in &payload.dimensions {
        out.extend_from_slice(&u16::try_from(name.len()).map_err(too_many)?.to_le_bytes());
        out.extend_from_slice(name.as_bytes());
    }

    out.extend_from_slice(
        &u32::try_from(payload.per_segment.len())
            .map_err(too_many)?
            .to_le_bytes(),
    );
    for per_segment in &payload.per_segment {
        out.extend_from_slice(
            &u16::try_from(per_segment.len())
                .map_err(too_many)?
                .to_le_bytes(),
        );
        // `DimensionTensors` is a BTreeMap and `dimensions` is sorted, so the
        // indices come out ascending without sorting anything here.
        for (name, tensor) in per_segment {
            let index = payload.dimensions.iter().position(|d| d == name).ok_or(
                Error::MalformedPayload("a tensor names a dimension the group does not list"),
            )?;
            let frame = map_format::codec::encode_tensor(tensor)?;
            out.extend_from_slice(&u16::try_from(index).map_err(too_many)?.to_le_bytes());
            out.extend_from_slice(&u32::try_from(frame.len()).map_err(too_many)?.to_le_bytes());
            out.extend_from_slice(&frame);
        }
    }
    Ok(out)
}

/// Everything a decoded [`TensorPayload`] must satisfy, whichever encoding it
/// arrived in.
///
/// The binary reader enforces these structurally — it builds the name table in
/// ascending order, resolves every entry through that table, and runs
/// `decode_tensor` on every frame — so they were only ever checked on that
/// path. The JSON fallback of §9.2 is read from the same untrusted committed
/// bytes and had none of them, which is the gap this closes: a hand-written
/// object could declare a tensor shape its buffer does not match, or name a
/// dimension the group does not list, and be handed to a scorer.
fn validate_tensor_payload(payload: &TensorPayload) -> Result<()> {
    if payload.dimensions.windows(2).any(|w| w[0] >= w[1]) {
        return Err(Error::MalformedPayload(
            "dimension names must be sorted and unique",
        ));
    }
    for per_segment in &payload.per_segment {
        for (name, tensor) in per_segment {
            if !payload.dimensions.iter().any(|d| d == name) {
                return Err(Error::MalformedPayload(
                    "a tensor names a dimension the group does not list",
                ));
            }
            // Rejects both a shape that disagrees with the buffer and one whose
            // product wraps into a plausible length (spec §9.2).
            tensor.validate()?;
        }
    }
    Ok(())
}

/// Decode a [`TensorPayload`] written by [`encode_tensor_payload`].
///
/// Every check here runs on **committed bytes from a cloned repository**, which
/// are untrusted (spec §8): lengths are bounds-checked before use, counts never
/// pre-allocate what they claim, and any deviation from canonical ordering is
/// refused rather than accepted and re-normalized.
///
/// Bytes that do not carry the magic are read as canonical JSON. An object key
/// derives from the embedder config and dimension set, not from how the result
/// was serialized, so an index holding the JSON form is reused rather than
/// rewritten. **Delete this fallback at the format freeze.**
pub fn decode_tensor_payload(bytes: &[u8]) -> Result<TensorPayload> {
    if !bytes.starts_with(TENSOR_GROUP_MAGIC) {
        let payload: TensorPayload = serde_json::from_slice(bytes)?;
        validate_tensor_payload(&payload)?;
        return Ok(payload);
    }

    let mut reader = Reader {
        bytes,
        at: TENSOR_GROUP_MAGIC.len(),
    };

    if reader.u16()? != TENSOR_GROUP_VERSION {
        return Err(Error::MalformedPayload("unsupported tensor group version"));
    }

    let dimension_count = reader.u16()? as usize;
    let mut dimensions: Vec<String> = Vec::new();
    for _ in 0..dimension_count {
        let len = reader.u16()? as usize;
        let name = std::str::from_utf8(reader.take(len)?)
            .map_err(|_| Error::MalformedPayload("dimension name is not utf-8"))?;
        if dimensions.last().is_some_and(|last| name <= last.as_str()) {
            return Err(Error::MalformedPayload(
                "dimension names are not in ascending order",
            ));
        }
        dimensions.push(name.to_owned());
    }

    let segment_count = reader.u32()? as usize;
    let mut per_segment = Vec::new();
    for _ in 0..segment_count {
        let entries = reader.u16()? as usize;
        let mut tensors = DimensionTensors::new();
        let mut previous: Option<u16> = None;
        for _ in 0..entries {
            let index = reader.u16()?;
            if previous.is_some_and(|p| index <= p) {
                return Err(Error::MalformedPayload(
                    "tensor entries are not in ascending dimension order",
                ));
            }
            previous = Some(index);
            let name = dimensions
                .get(index as usize)
                .ok_or(Error::MalformedPayload("dimension index out of range"))?;
            let len = reader.u32()? as usize;
            let tensor = map_format::codec::decode_tensor(reader.take(len)?)?;
            tensors.insert(name.clone(), tensor);
        }
        per_segment.push(tensors);
    }

    if reader.at != bytes.len() {
        return Err(Error::MalformedPayload(
            "trailing bytes after the last segment",
        ));
    }
    let payload = TensorPayload {
        dimensions,
        per_segment,
    };
    // Structurally guaranteed above; run anyway so the two paths cannot drift
    // into enforcing different things.
    validate_tensor_payload(&payload)?;
    Ok(payload)
}

/// Bounds-checked cursor. Every read can fail; none can panic.
struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self
            .at
            .checked_add(len)
            .ok_or(Error::MalformedPayload("length overflows"))?;
        let slice = self
            .bytes
            .get(self.at..end)
            .ok_or(Error::MalformedPayload("truncated tensor group"))?;
        self.at = end;
        Ok(slice)
    }

    fn u16(&mut self) -> Result<u16> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use map_format::{DType, Tensor};

    fn tensor(seed: u8) -> Tensor {
        Tensor {
            dtype: DType::F32,
            shape: vec![4],
            data: (0..16).map(|i| i as u8 ^ seed).collect(),
        }
    }

    fn payload() -> TensorPayload {
        let mut first = DimensionTensors::new();
        first.insert("descriptive".to_owned(), tensor(1));
        first.insert("semantic".to_owned(), tensor(2));
        let mut second = DimensionTensors::new();
        second.insert("semantic".to_owned(), tensor(3));

        TensorPayload {
            dimensions: vec!["descriptive".to_owned(), "semantic".to_owned()],
            per_segment: vec![first, second],
        }
    }

    #[test]
    fn a_payload_survives_the_binary_round_trip() {
        let original = payload();
        let bytes = encode_tensor_payload(&original).unwrap();
        assert_eq!(decode_tensor_payload(&bytes).unwrap(), original);
    }

    #[test]
    fn a_segment_carrying_no_tensor_survives() {
        // The embedder skips a segment it cannot encode, so an empty map is a
        // real case and must not be confused with the end of the payload.
        let original = TensorPayload {
            dimensions: vec!["semantic".to_owned()],
            per_segment: vec![DimensionTensors::new(), DimensionTensors::new()],
        };
        let bytes = encode_tensor_payload(&original).unwrap();
        assert_eq!(decode_tensor_payload(&bytes).unwrap(), original);
    }

    #[test]
    fn the_same_payload_always_encodes_to_the_same_bytes() {
        // Object identity, integrity, and git dedup all rest on this (spec §6).
        assert_eq!(
            encode_tensor_payload(&payload()).unwrap(),
            encode_tensor_payload(&payload()).unwrap()
        );
    }

    #[test]
    fn json_written_before_the_binary_encoding_still_decodes() {
        // An object key derives from the embedder config and dimension set, not
        // from the serialization, so re-indexing reuses objects written as JSON
        // rather than rewriting them. Without this fallback every index built
        // before the change would fail to load.
        let original = payload();
        let json = serde_json::to_vec(&original).unwrap();
        assert_eq!(decode_tensor_payload(&json).unwrap(), original);
    }

    #[test]
    fn a_json_tensor_whose_shape_contradicts_its_bytes_is_rejected() {
        // The JSON fallback is read from the same committed, untrusted bytes as
        // the binary form. It used to hand serde's output straight back, so a
        // shape that disagreed with the buffer reached a scorer unchecked.
        let mut segment = DimensionTensors::new();
        segment.insert(
            "semantic".to_owned(),
            Tensor {
                dtype: DType::F32,
                shape: vec![768],
                data: vec![0u8; 16],
            },
        );
        let json = serde_json::to_vec(&TensorPayload {
            dimensions: vec!["semantic".to_owned()],
            per_segment: vec![segment],
        })
        .unwrap();

        assert!(matches!(
            decode_tensor_payload(&json),
            Err(Error::Format(map_format::Error::TensorShapeMismatch { .. }))
        ));
    }

    #[test]
    fn a_json_shape_that_wraps_into_a_plausible_length_is_rejected() {
        // 2^31 * 2^31 * 4 bytes wraps to exactly 0, so an empty buffer would
        // validate against a declared four-exabyte tensor.
        let mut segment = DimensionTensors::new();
        segment.insert(
            "semantic".to_owned(),
            Tensor {
                dtype: DType::F32,
                shape: vec![1 << 31, 1 << 31],
                data: Vec::new(),
            },
        );
        let json = serde_json::to_vec(&TensorPayload {
            dimensions: vec!["semantic".to_owned()],
            per_segment: vec![segment],
        })
        .unwrap();

        assert!(matches!(
            decode_tensor_payload(&json),
            Err(Error::Format(map_format::Error::TensorShapeOverflow { .. }))
        ));
    }

    #[test]
    fn a_json_tensor_naming_a_dimension_outside_the_table_is_rejected() {
        let mut segment = DimensionTensors::new();
        segment.insert("not-listed".to_owned(), tensor(1));
        let json = serde_json::to_vec(&TensorPayload {
            dimensions: vec!["semantic".to_owned()],
            per_segment: vec![segment],
        })
        .unwrap();

        assert!(matches!(
            decode_tensor_payload(&json),
            Err(Error::MalformedPayload(
                "a tensor names a dimension the group does not list"
            ))
        ));
    }

    #[test]
    fn a_json_dimension_table_out_of_canonical_order_is_rejected() {
        let json = serde_json::to_vec(&TensorPayload {
            dimensions: vec!["semantic".to_owned(), "descriptive".to_owned()],
            per_segment: vec![DimensionTensors::new()],
        })
        .unwrap();

        assert!(matches!(
            decode_tensor_payload(&json),
            Err(Error::MalformedPayload(
                "dimension names must be sorted and unique"
            ))
        ));
    }

    #[test]
    fn trailing_bytes_are_rejected() {
        // Otherwise two byte strings decode to one payload and the encoding is
        // no longer canonical.
        let mut bytes = encode_tensor_payload(&payload()).unwrap();
        bytes.push(0);
        // Assert the specific branch, not merely "some error": a check that
        // rejects for an unrelated reason would look identical here.
        assert!(matches!(
            decode_tensor_payload(&bytes),
            Err(Error::MalformedPayload(
                "trailing bytes after the last segment"
            ))
        ));
    }

    #[test]
    fn unsorted_dimension_names_are_rejected() {
        let unsorted = TensorPayload {
            dimensions: vec!["semantic".to_owned(), "descriptive".to_owned()],
            per_segment: vec![DimensionTensors::new()],
        };
        assert!(matches!(
            encode_tensor_payload(&unsorted),
            Err(Error::MalformedPayload(_))
        ));
    }

    #[test]
    fn a_dimension_index_past_the_table_is_rejected() {
        // Reachable from a hostile committed index: the index is read straight
        // out of a clone, and an out-of-range entry would otherwise panic.
        let mut bytes = encode_tensor_payload(&payload()).unwrap();
        let table_end = TENSOR_GROUP_MAGIC.len() + 2 + 2 + (2 + 11) + (2 + 8);
        let first_entry = table_end + 4 + 2;
        // Sanity-check the offset arithmetic before trusting what it proves:
        // this byte must currently be dimension 0.
        assert_eq!(bytes[first_entry], 0, "offset does not point at a dim_idx");
        bytes[first_entry] = 9;
        assert!(matches!(
            decode_tensor_payload(&bytes),
            Err(Error::MalformedPayload("dimension index out of range"))
        ));
    }

    #[test]
    fn a_truncated_group_is_rejected_rather_than_panicking() {
        let bytes = encode_tensor_payload(&payload()).unwrap();
        for cut in 0..bytes.len() {
            // Must be an error at every truncation point, never a panic.
            let _ = decode_tensor_payload(&bytes[..cut]);
        }
        assert!(decode_tensor_payload(&bytes[..bytes.len() - 1]).is_err());
    }

    #[test]
    fn a_hostile_frame_length_cannot_read_past_the_buffer() {
        let mut bytes = encode_tensor_payload(&payload()).unwrap();
        let table_end = TENSOR_GROUP_MAGIC.len() + 2 + 2 + (2 + 11) + (2 + 8);
        let length_at = table_end + 4 + 2 + 2;
        // The real frame is 10 bytes of header, 4 of shape, and 16 of data.
        assert_eq!(
            u32::from_le_bytes(bytes[length_at..length_at + 4].try_into().unwrap()),
            30,
            "offset does not point at a frame length"
        );
        bytes[length_at..length_at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(
            decode_tensor_payload(&bytes),
            Err(Error::MalformedPayload("truncated tensor group"))
        ));
    }
}
