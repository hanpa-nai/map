//! Records which commit this binary was built from.
//!
//! Without it every build answers `map --version` the same way, and an install
//! that kept an old binary cannot be told from one that worked. That is not a
//! hypothetical: `cargo install --git` with a persistent target directory
//! prints "Replaced package" and leaves the previous binary in place when the
//! version number did not change.
//!
//! Emits two variables, each empty when there is nothing trustworthy to say:
//!
//! * `MAP_BUILD_COMMIT` — eight hex characters, with `-dirty` appended when
//!   the sources differ from that commit.
//! * `MAP_BUILD_DATE` — the commit's date, not today's, so the same commit
//!   always builds the same string.

use std::path::Path;
use std::process::Command;

/// Stdout of a git command run in `root`, or `None` if git could not be run or
/// reported failure. An empty answer is `Some("")`: for `status` it means
/// "clean", which is not the same as "could not ask".
fn git(root: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8(output.stdout).ok()?.trim().to_owned())
}

fn identity(root: &Path) -> Option<(String, String)> {
    // Only this workspace's own `.git`. A source tree with no git data that
    // sits inside some other repository — a vendored copy, a home directory
    // under version control — would otherwise report that repository's
    // commit as its own.
    if !root.join(".git").exists() {
        return None;
    }
    let commit = git(root, &["rev-parse", "--short=8", "HEAD"]).filter(|c| !c.is_empty())?;
    let date = git(root, &["log", "-1", "--format=%cs"]).unwrap_or_default();

    // Limited to what is compiled. A cargo git checkout carries an untracked
    // `.cargo-ok` at its root, which a whole-tree status would read as a
    // modified source tree on every install.
    let dirty = git(
        root,
        &[
            "status",
            "--porcelain",
            "--",
            "crates",
            "Cargo.toml",
            "Cargo.lock",
        ],
    )
    .is_some_and(|changes| !changes.is_empty());

    // `HEAD` moves on a checkout and `logs/HEAD` on every commit; a worktree
    // keeps both in its own git directory, which is why this asks rather than
    // assuming `.git/`.
    if let Some(git_dir) = git(root, &["rev-parse", "--absolute-git-dir"]) {
        for name in ["HEAD", "logs/HEAD"] {
            let path = Path::new(&git_dir).join(name);
            if path.exists() {
                println!("cargo:rerun-if-changed={}", path.display());
            }
        }
    }

    let commit = if dirty {
        format!("{commit}-dirty")
    } else {
        commit
    };
    Some((commit, date))
}

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR");
    // crates/map-cli -> the workspace root.
    let root = Path::new(&manifest_dir).join("..").join("..");

    // The `-dirty` mark has to follow edits in any crate, not only this one,
    // so every source in the workspace re-runs the script.
    println!("cargo:rerun-if-changed=build.rs");
    for path in ["crates", "Cargo.toml", "Cargo.lock"] {
        println!("cargo:rerun-if-changed={}", root.join(path).display());
    }

    let (commit, date) = identity(&root).unwrap_or_default();
    println!("cargo:rustc-env=MAP_BUILD_COMMIT={commit}");
    println!("cargo:rustc-env=MAP_BUILD_DATE={date}");
}
