//! Canonical encodings.
//!
//! Everything the format writes must be byte-reproducible, because object
//! identity, integrity, and git dedup all rest on it (spec §6, Tier A). That
//! rules out any encoder whose output depends on map iteration order,
//! whitespace preference, or locale.
//!
//! Two encodings exist:
//!
//! - [`canonical_json`] for text-shaped data (manifest, descriptor groups).
//! - [`encode_tensor`] / [`decode_tensor`] for the numeric payload, which is
//!   raw little-endian bytes behind a fixed header so it can be memory-mapped
//!   later without a parse step.
//!
//! Note the format does not interpret either payload's *meaning* (spec §3).
//! `dtype` and `shape` are structural — needed to store and map the bytes —
//! not semantic.

use serde::Serialize;

use crate::error::{Error, Result};
use crate::record::{DType, Tensor};

/// Serialize as canonical JSON: no whitespace, deterministic key order.
///
/// Two different orderings are in play, and conflating them is a trap:
///
/// - **Map keys sort.** `BTreeMap` fields and `serde_json::Value` objects emit
///   in sorted order, because `serde_json`'s map is `BTreeMap`-backed.
/// - **Struct fields do not sort** — serde emits them in *declaration order*.
///   That is deterministic, but it means **field order in the Rust source is
///   part of the on-disk format**. Reordering fields in [`crate::manifest`]
///   changes stored bytes and the determinism digest.
///
/// Enabling `serde_json`'s `preserve_order` feature anywhere in the dependency
/// graph switches maps to insertion order and silently breaks canonicality.
/// The run-twice CI test does **not** catch this — insertion order is perfectly
/// stable run-to-run and cross-platform — so the `preserve_order_canary` unit
/// test in this module checks it directly.
pub fn canonical_json<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(value)?)
}

const TENSOR_MAGIC: &[u8; 4] = b"MAPT";
const TENSOR_VERSION: u16 = 1;
const TENSOR_HEADER_LEN: usize = 4 + 2 + 1 + 1 + 2;

/// Encode a tensor as `MAPT` + header + little-endian payload.
///
/// Layout:
///
/// ```text
/// magic    4  b"MAPT"
/// version  2  u16 le
/// dtype    1  u8
/// _pad     1  0
/// rank     2  u16 le
/// shape    4 * rank   u32 le
/// data     remainder
/// ```
pub fn encode_tensor(t: &Tensor) -> Result<Vec<u8>> {
    t.validate()?;

    let mut out = Vec::with_capacity(TENSOR_HEADER_LEN + t.shape.len() * 4 + t.data.len());
    out.extend_from_slice(TENSOR_MAGIC);
    out.extend_from_slice(&TENSOR_VERSION.to_le_bytes());
    out.push(t.dtype as u8);
    out.push(0);
    out.extend_from_slice(&(t.shape.len() as u16).to_le_bytes());
    for d in &t.shape {
        out.extend_from_slice(&d.to_le_bytes());
    }
    out.extend_from_slice(&t.data);
    Ok(out)
}

/// Decode a tensor written by [`encode_tensor`].
pub fn decode_tensor(bytes: &[u8]) -> Result<Tensor> {
    if bytes.len() < TENSOR_HEADER_LEN {
        return Err(Error::MalformedTensor("truncated header"));
    }
    if &bytes[0..4] != TENSOR_MAGIC {
        return Err(Error::MalformedTensor("bad magic"));
    }
    let version = u16::from_le_bytes([bytes[4], bytes[5]]);
    if version != TENSOR_VERSION {
        return Err(Error::MalformedTensor("unsupported tensor version"));
    }
    let dtype = DType::from_u8(bytes[6]).ok_or(Error::MalformedTensor("unknown dtype"))?;
    // Reject a nonzero pad byte: otherwise two distinct byte strings decode to
    // equal tensors, which breaks canonicality.
    if bytes[7] != 0 {
        return Err(Error::MalformedTensor("nonzero pad byte"));
    }
    let rank = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;

    let shape_end = TENSOR_HEADER_LEN + rank * 4;
    if bytes.len() < shape_end {
        return Err(Error::MalformedTensor("truncated shape"));
    }
    let mut shape = Vec::with_capacity(rank);
    for i in 0..rank {
        let o = TENSOR_HEADER_LEN + i * 4;
        shape.push(u32::from_le_bytes([
            bytes[o],
            bytes[o + 1],
            bytes[o + 2],
            bytes[o + 3],
        ]));
    }

    let t = Tensor {
        dtype,
        shape,
        data: bytes[shape_end..].to_vec(),
    };
    t.validate()?;
    Ok(t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn canonical_json_sorts_keys() {
        let mut m = BTreeMap::new();
        m.insert("zebra", 1);
        m.insert("alpha", 2);
        m.insert("middle", 3);
        let bytes = canonical_json(&m).unwrap();
        assert_eq!(bytes, br#"{"alpha":2,"middle":3,"zebra":1}"#);
    }

    #[test]
    fn preserve_order_canary() {
        // If any dependency enables serde_json/preserve_order, Value maps
        // switch from sorted to insertion order and canonicality breaks —
        // silently, because insertion order is still stable run-to-run, so the
        // run-twice determinism test would keep passing. This checks directly.
        let mut obj = serde_json::Map::new();
        obj.insert("zebra".into(), serde_json::json!(1));
        obj.insert("alpha".into(), serde_json::json!(2));
        let bytes = canonical_json(&serde_json::Value::Object(obj)).unwrap();

        assert_eq!(
            bytes, br#"{"alpha":2,"zebra":1}"#,
            "serde_json map is not emitting sorted keys — the `preserve_order` \
             feature has been enabled somewhere in the dependency graph, which \
             breaks canonical encoding (spec §6)"
        );
    }

    #[test]
    fn struct_fields_emit_in_declaration_order() {
        // Documents the contract rather than asserting a preference: struct
        // field order is part of the on-disk format.
        #[derive(Serialize)]
        struct Declared {
            zebra: u8,
            alpha: u8,
        }
        assert_eq!(
            canonical_json(&Declared { zebra: 1, alpha: 2 }).unwrap(),
            br#"{"zebra":1,"alpha":2}"#,
        );
    }

    #[test]
    fn rejects_nonzero_pad_byte() {
        let t = Tensor {
            dtype: DType::I8,
            shape: vec![2],
            data: vec![7, 8],
        };
        let mut encoded = encode_tensor(&t).unwrap();
        encoded[7] = 1; // pad byte
        assert!(decode_tensor(&encoded).is_err());
    }

    #[test]
    fn canonical_json_is_stable_across_insertion_order() {
        let mut a = BTreeMap::new();
        a.insert("one", 1);
        a.insert("two", 2);
        let mut b = BTreeMap::new();
        b.insert("two", 2);
        b.insert("one", 1);
        assert_eq!(canonical_json(&a).unwrap(), canonical_json(&b).unwrap());
    }

    #[test]
    fn tensor_roundtrips() {
        let t = Tensor {
            dtype: DType::F32,
            shape: vec![2, 3],
            data: (0..24).collect(),
        };
        let encoded = encode_tensor(&t).unwrap();
        let decoded = decode_tensor(&encoded).unwrap();
        assert_eq!(decoded, t);
    }

    #[test]
    fn tensor_encoding_is_deterministic() {
        let t = Tensor {
            dtype: DType::Binary,
            shape: vec![768],
            data: vec![0xAB; 96],
        };
        assert_eq!(encode_tensor(&t).unwrap(), encode_tensor(&t).unwrap());
    }

    #[test]
    fn rejects_corrupt_tensor() {
        assert!(decode_tensor(b"").is_err());
        assert!(decode_tensor(b"NOPE....").is_err());

        let t = Tensor {
            dtype: DType::I8,
            shape: vec![4],
            data: vec![1, 2, 3, 4],
        };
        let mut encoded = encode_tensor(&t).unwrap();
        encoded.pop(); // shape now promises more bytes than exist
        assert!(decode_tensor(&encoded).is_err());
    }
}
