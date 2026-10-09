//! Downloading the distilled model's weights — the `auto-distilled` plugin.
//!
//! Gated behind `auto-distilled` so that neither a default build nor a
//! `distilled` one links an HTTP stack. A `distilled` build loads weights that
//! are already on disk and cannot reach the network at all.
//!
//! Nothing here runs implicitly. `map index` never downloads; it reports the
//! missing model and stops. The only caller is `map model fetch`.
//!
//! # What is pinned
//!
//! Files are fetched from one **revision**, not from `main`, and every one is
//! checked against a hardcoded SHA-256 before it is installed. The digests are
//! plain sha256 so they can be checked against HuggingFace's own metadata or
//! with `sha256sum`, independently of anything in this repository.
//!
//! A download is written to `<name>.part` and renamed into place only after its
//! digest matches, so an interrupted or corrupted fetch never leaves a file
//! that [`from_dir`](crate::DistilledEmbedder::from_dir) would try to load.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// The HuggingFace repository the weights come from.
pub const REPO: &str = "minishlab/potion-retrieval-32M";

/// The exact revision fetched. Never `main`: an index's tensors are keyed by
/// the embedder's identity, so weights changing underneath a pinned model would
/// silently alter what a dimension means without re-keying anything.
pub const REVISION: &str = "6fc8051fab2a1e0ee76689cf08c853792ac285e7";

pub use crate::distilled::{model_dir, MODEL_ID};

/// One file, with the length and digest it must have.
pub struct Artifact {
    pub name: &'static str,
    pub len: u64,
    pub sha256: &'static str,
}

/// Everything [`DistilledEmbedder::from_dir`](crate::DistilledEmbedder::from_dir)
/// reads. Verified 2026-08-20 against a fresh download of [`REVISION`].
pub const ARTIFACTS: &[Artifact] = &[
    Artifact {
        name: "config.json",
        len: 202,
        sha256: "63c00d90824c832c04ec1d02b6a983fb90489bf049f29fbff15ba481b8a432ee",
    },
    Artifact {
        name: "tokenizer.json",
        len: 1_493_150,
        sha256: "7d75cbc54318138807c401b0f0c9721117c628b39de8e8e0edb6cb17e0ee7d18",
    },
    Artifact {
        name: "model.safetensors",
        len: 129_210_456,
        sha256: "07609e5bd33aad37900b3fd62f4ec96f6daec88ca4d46b9d8b928bfababf6ea0",
    },
];

#[derive(Debug, thiserror::Error)]
pub enum FetchError {
    #[error("io error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("{name}: {source}")]
    Http {
        name: &'static str,
        #[source]
        source: Box<ureq::Error>,
    },

    /// A length check before the digest, so an endpoint streaming an unbounded
    /// body is cut off rather than filling the disk.
    #[error("{name}: expected {expected} bytes, received {received}")]
    Length {
        name: &'static str,
        expected: u64,
        received: u64,
    },

    #[error(
        "{name}: sha256 mismatch\n  expected {expected}\n  received {received}\n\
         Nothing was installed."
    )]
    Digest {
        name: &'static str,
        expected: &'static str,
        received: String,
    },

    #[error("downloaded weights did not load: {0}")]
    Unusable(String),
}

/// What happened to one artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Disposition {
    /// Already present with the right digest; nothing was transferred.
    Reused,
    Downloaded,
}

/// Reported as bytes arrive, so a caller can draw progress. Called often.
pub struct Progress<'a> {
    pub name: &'a str,
    pub received: u64,
    pub total: u64,
}

/// Whether every artifact is present at its pinned length.
///
/// Length only — this is the cheap check on the query and index paths. The
/// digest is verified when fetching, and a corrupted file surfaces as a load
/// error rather than being re-hashed on every run.
pub fn installed(dir: &Path) -> bool {
    ARTIFACTS.iter().all(|a| {
        fs::metadata(dir.join(a.name))
            .map(|m| m.is_file() && m.len() == a.len)
            .unwrap_or(false)
    })
}

fn url_for(name: &str) -> String {
    format!("https://huggingface.co/{REPO}/resolve/{REVISION}/{name}")
}

fn io(path: impl Into<PathBuf>, source: std::io::Error) -> FetchError {
    FetchError::Io {
        path: path.into(),
        source,
    }
}

/// Hash a file that is already on disk.
fn digest_of(path: &Path) -> Result<String, FetchError> {
    let mut file = fs::File::open(path).map_err(|e| io(path, e))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf).map_err(|e| io(path, e))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex(&hasher.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write as _;
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// Download every artifact that is not already present and correct.
///
/// `progress` is called as bytes arrive. Returns what happened to each
/// artifact, in [`ARTIFACTS`] order.
pub fn fetch(
    dir: &Path,
    progress: &mut dyn FnMut(Progress<'_>),
) -> Result<Vec<(&'static str, Disposition)>, FetchError> {
    fs::create_dir_all(dir).map_err(|e| io(dir, e))?;

    let mut out = Vec::with_capacity(ARTIFACTS.len());
    for artifact in ARTIFACTS {
        let final_path = dir.join(artifact.name);

        // A previous run may have installed this one already. Re-hashing is
        // cheaper than re-downloading 123 MB, and it also repairs a directory
        // where only some files landed.
        if fs::metadata(&final_path).map(|m| m.len()).ok() == Some(artifact.len)
            && digest_of(&final_path)? == artifact.sha256
        {
            out.push((artifact.name, Disposition::Reused));
            continue;
        }

        download_one(artifact, dir, progress)?;
        out.push((artifact.name, Disposition::Downloaded));
    }

    // The digests prove the bytes are the pinned ones; this proves the pinned
    // ones are usable, which is the thing the caller actually wanted.
    crate::DistilledEmbedder::from_dir(dir, MODEL_ID)
        .map_err(|e| FetchError::Unusable(e.to_string()))?;

    Ok(out)
}

fn download_one(
    artifact: &'static Artifact,
    dir: &Path,
    progress: &mut dyn FnMut(Progress<'_>),
) -> Result<(), FetchError> {
    let part_path = dir.join(format!("{}.part", artifact.name));
    let response = ureq::get(&url_for(artifact.name))
        .set("user-agent", concat!("map/", env!("CARGO_PKG_VERSION")))
        .call()
        .map_err(|e| FetchError::Http {
            name: artifact.name,
            source: Box::new(e),
        })?;

    // Read one byte past the pin so an over-long body is detected rather than
    // silently truncated to something that could still hash to the pin.
    let mut reader = response.into_reader().take(artifact.len + 1);
    let mut file = fs::File::create(&part_path).map_err(|e| io(&part_path, e))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 64 * 1024];
    let mut received: u64 = 0;

    let outcome = (|| -> Result<(), FetchError> {
        loop {
            let n = reader.read(&mut buf).map_err(|e| io(&part_path, e))?;
            if n == 0 {
                break;
            }
            use std::io::Write as _;
            file.write_all(&buf[..n]).map_err(|e| io(&part_path, e))?;
            hasher.update(&buf[..n]);
            received += n as u64;
            progress(Progress {
                name: artifact.name,
                received,
                total: artifact.len,
            });
        }
        file.sync_all().map_err(|e| io(&part_path, e))?;

        if received != artifact.len {
            return Err(FetchError::Length {
                name: artifact.name,
                expected: artifact.len,
                received,
            });
        }
        let got = hex(&hasher.finalize_reset());
        if got != artifact.sha256 {
            return Err(FetchError::Digest {
                name: artifact.name,
                expected: artifact.sha256,
                received: got,
            });
        }
        Ok(())
    })();

    drop(file);
    if outcome.is_err() {
        // Never leave a partial or wrong file where the loader would find it.
        let _ = fs::remove_file(&part_path);
        return outcome;
    }

    let final_path = dir.join(artifact.name);
    fs::rename(&part_path, &final_path).map_err(|e| io(&final_path, e))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_artifact_pins_a_full_length_sha256() {
        for a in ARTIFACTS {
            assert_eq!(
                a.sha256.len(),
                64,
                "{} pins a digest that is not sha256",
                a.name
            );
            assert!(
                a.sha256.chars().all(|c| c.is_ascii_hexdigit()),
                "{} pins a non-hex digest",
                a.name
            );
            assert!(a.len > 0, "{} pins a zero length", a.name);
        }
    }

    /// The loader reads exactly these three names, so a rename on either side
    /// would otherwise fetch a complete set that cannot be loaded.
    #[test]
    fn the_artifact_set_is_what_the_loader_opens() {
        let names: Vec<_> = ARTIFACTS.iter().map(|a| a.name).collect();
        assert!(names.contains(&"config.json"));
        assert!(names.contains(&"tokenizer.json"));
        assert!(names.contains(&"model.safetensors"));
        assert_eq!(names.len(), 3);
    }

    #[test]
    fn the_revision_is_a_commit_rather_than_a_branch() {
        assert_eq!(REVISION.len(), 40, "a pin must be a full commit sha");
        assert!(REVISION.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn urls_address_the_pinned_revision() {
        let url = url_for("model.safetensors");
        assert!(url.contains(REVISION), "url must pin a revision: {url}");
        assert!(
            !url.contains("/main/"),
            "url must not track a branch: {url}"
        );
        assert!(url.starts_with("https://"), "url must be https: {url}");
    }

    #[test]
    fn a_directory_missing_a_file_is_not_installed() {
        let dir = std::env::temp_dir().join(format!("map-fetch-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        assert!(!installed(&dir));

        // Right names, wrong lengths: still not installed.
        for a in ARTIFACTS {
            fs::write(dir.join(a.name), b"x").unwrap();
        }
        assert!(
            !installed(&dir),
            "length must be checked, not just presence"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn hex_is_lowercase_and_zero_padded() {
        assert_eq!(hex(&[0x00, 0x0f, 0xff]), "000fff");
    }
}
