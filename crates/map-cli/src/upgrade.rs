//! `map upgrade` — install the latest version over this binary.
//!
//! The work is one `cargo install`, but the plain command goes wrong in three
//! ways, each watched to happen before this module existed:
//!
//! * **It drops features.** Re-running the README command over a binary built
//!   with `--features llm` replaces it with a default build, and cargo says
//!   nothing. So the command is rebuilt from the features compiled in here.
//! * **It can install the old binary again.** With a persistent target
//!   directory (`CARGO_TARGET_DIR`), cargo's artifact identity for a git
//!   source does not include the commit: it prints "Replaced package" and
//!   leaves the previous build in place, and `--force` alone does not help.
//!   So the build always gets a new, empty `--target-dir`.
//! * **On Windows it cannot replace a running program.** `cargo install` ends
//!   with "Access is denied" when the destination is the executable that
//!   called it. A running executable can be renamed though, so this one moves
//!   itself aside first and moves back if the install fails.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};

use crate::build_info;

/// The cargo install root this binary is in: the parent of its `bin`
/// directory. `None` for a binary anywhere else, such as `target/release`.
fn install_root(exe: &Path) -> Option<PathBuf> {
    let bin = exe.parent()?;
    if bin.file_name()? != "bin" {
        return None;
    }
    bin.parent().map(Path::to_path_buf)
}

/// Whether cargo keeps its record of installed packages in `root`.
///
/// `cargo install --root` writes that record beside `bin`. A `bin` directory
/// without one, such as `/usr/local/bin`, belongs to something else, and an
/// install there would start to write cargo's files into it.
fn is_cargo_root(root: &Path) -> bool {
    root.join(".crates.toml").is_file()
}

/// Where the running binary waits while cargo writes its replacement.
fn aside_path(exe: &Path) -> PathBuf {
    let mut name = exe.file_name().unwrap_or_default().to_os_string();
    name.push(".old");
    exe.with_file_name(name)
}

/// The binary that was moved aside, until a new one is at `exe`.
///
/// A guard and not a call at each exit, so that a panic between the rename
/// and the end of the build also leaves a `map` at `exe`. It cannot act for a
/// process that is killed: the message printed before the build is for that.
struct SetAside<'a> {
    exe: &'a Path,
    aside: &'a Path,
}

impl SetAside<'_> {
    /// `why`, completed with what became of the installed binary.
    fn failed(&self, why: String) -> String {
        if self.exe.exists() {
            // cargo put a binary in place and failed afterwards. The previous
            // one has no further use.
            let _ = std::fs::remove_file(self.aside);
            return why;
        }
        match std::fs::rename(self.aside, self.exe) {
            Ok(()) => format!("{why}\nThe installed binary did not change."),
            Err(e) => format!(
                "{why}\nThe previous binary is at {} and could not be moved back: {e}",
                self.aside.display()
            ),
        }
    }
}

impl Drop for SetAside<'_> {
    fn drop(&mut self) {
        if !self.exe.exists() {
            let _ = std::fs::rename(self.aside, self.exe);
        }
    }
}

/// Whether `repository` can be handed to another program as a plain value.
fn is_repository_argument(repository: &str) -> bool {
    !repository.is_empty() && !repository.starts_with('-')
}

/// The commit `HEAD` names in the first line of `git ls-remote` output.
fn parse_remote_head(output: &str) -> Option<String> {
    let commit = output.lines().next()?.split_whitespace().next()?;
    (commit.len() >= 8 && commit.chars().all(|c| c.is_ascii_hexdigit()))
        .then(|| commit.to_ascii_lowercase())
}

/// The newest commit of `repository`, or `None` when it cannot be asked.
///
/// Not being able to ask is not an error: the install below still works
/// without the answer, it just cannot be skipped.
fn remote_head(repository: &str) -> Option<String> {
    let output = Command::new("git")
        // `--` so that the URL can never be read as an option of git.
        .args(["ls-remote", "--", repository, "HEAD"])
        // A repository that wants a password would otherwise stop here and
        // wait for one on a terminal nobody is watching.
        .env("GIT_TERMINAL_PROMPT", "0")
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    parse_remote_head(&String::from_utf8_lossy(&output.stdout))
}

/// Whether this binary is already the newest commit.
///
/// Both sides have to be known. A build with no commit, or a `-dirty` one,
/// is never "the latest": it cannot be shown to be the same code.
fn is_latest(built: Option<&str>, remote: Option<&str>) -> bool {
    match (built, remote) {
        (Some(built), Some(remote)) => remote.starts_with(built),
        _ => false,
    }
}

fn cargo_args(
    repository: &str,
    root: &Path,
    target_dir: &Path,
    features: &[&str],
) -> Vec<OsString> {
    let mut args: Vec<OsString> = ["install", "--git", repository, "map-cli", "--locked"]
        .iter()
        .map(OsString::from)
        .collect();
    // The commit comparison already decided an install is wanted, and cargo's
    // own record of what is installed is the thing that goes stale.
    args.push("--force".into());
    args.push("--root".into());
    args.push(root.into());
    args.push("--target-dir".into());
    args.push(target_dir.into());
    if !features.is_empty() {
        args.push("--features".into());
        args.push(features.join(" ").into());
    }
    args
}

/// A command line a person can read, and paste: arguments with a space are
/// quoted.
fn shown(args: &[OsString]) -> String {
    args.iter()
        .map(|arg| {
            let text = arg.to_string_lossy();
            if text.contains(' ') {
                format!("\"{text}\"")
            } else {
                text.into_owned()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Delete what an earlier upgrade set aside.
///
/// Windows only, and on every start: there the set-aside file is the image of
/// the process that upgraded, so that process cannot delete it and the next
/// one has to. One failed `unlink` when there is nothing to remove.
#[cfg(windows)]
pub(crate) fn remove_previous_binary() {
    if let Ok(exe) = std::env::current_exe() {
        let _ = std::fs::remove_file(aside_path(&exe));
    }
}

pub(crate) fn run(repository: Option<String>) -> Result<ExitCode, String> {
    let repository = repository.unwrap_or_else(|| build_info::REPOSITORY.to_owned());
    // The value goes to git and to cargo as an argument. One that starts with
    // `-` would be an option of theirs, not a place to fetch from, and git has
    // options that run a program.
    if !is_repository_argument(&repository) {
        return Err(format!("{repository:?} is not a repository URL"));
    }
    let features = build_info::features();
    let manual = build_info::install_command(&repository, &features);

    let exe = std::env::current_exe().map_err(|e| format!("cannot find this binary: {e}"))?;
    // Resolved so that a symlink on PATH upgrades the file it points to. The
    // extended-length prefix Windows adds is taken off again: cargo is handed
    // this path, and prints it.
    let exe = exe.canonicalize().unwrap_or(exe);
    let exe = PathBuf::from(crate::display_path(&exe));
    let Some(root) = install_root(&exe).filter(|root| is_cargo_root(root)) else {
        return Err(format!(
            "{} is not in the `bin` directory of a cargo install root, so `map upgrade` \
             cannot replace it. To install the newest version, run:\n  {manual}",
            exe.display()
        ));
    };

    let installed = build_info::version_line();
    let remote = remote_head(&repository);
    if is_latest(build_info::clean_commit(), remote.as_deref()) {
        println!("map {installed} is the newest version.");
        return Ok(ExitCode::SUCCESS);
    }
    eprintln!("map: this binary is map {installed}");
    match &remote {
        Some(commit) => eprintln!("map: the newest commit of {repository} is {}", &commit[..8]),
        None => eprintln!("map: cannot read the newest commit of {repository}; installing"),
    }

    let target_dir = std::env::temp_dir().join(format!("map-upgrade-{}", std::process::id()));
    let args = cargo_args(&repository, &root, &target_dir, &features);

    let aside = aside_path(&exe);
    // The rename replaces a file that an earlier upgrade left at `aside`.
    // That file is not deleted first: a second `map upgrade` that runs at the
    // same time would delete the only copy of the binary the first one holds.
    std::fs::rename(&exe, &aside).map_err(|e| {
        format!(
            "cannot move {} out of the way: {e}. Nothing changed. To install the newest \
             version, run:\n  {manual}",
            exe.display()
        )
    })?;
    // From here every exit has to leave a `map` at `exe`.
    let set_aside = SetAside {
        exe: &exe,
        aside: &aside,
    };

    eprintln!(
        "map: the previous binary is at {} until the build is complete. If this command \
         stops before the end, move that file back to {}",
        aside.display(),
        exe.display()
    );
    eprintln!("map: running: cargo {}", shown(&args));
    let status = Command::new("cargo").args(&args).status();
    let _ = std::fs::remove_dir_all(&target_dir);
    let status = status.map_err(|e| {
        set_aside.failed(format!(
            "cannot run `cargo`: {e}. An upgrade builds MAP from source, so Rust is \
             necessary: https://rustup.rs"
        ))
    })?;
    if !status.success() || !exe.exists() {
        return Err(set_aside.failed("the upgrade did not complete.".to_owned()));
    }

    // Deletable everywhere except Windows, where it is this process's own
    // image; `remove_previous_binary` gets it on the next start there.
    let _ = std::fs::remove_file(&aside);

    // Ask the new binary rather than assume: this is the only check that what
    // cargo installed actually runs.
    let new = Command::new(&exe).arg("-V").output();
    match new {
        Ok(out) if out.status.success() => println!(
            "The upgrade is complete: {}",
            String::from_utf8_lossy(&out.stdout).trim()
        ),
        _ => println!("The upgrade is complete. Run `map --version` to see the new version."),
    }
    Ok(ExitCode::SUCCESS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_binary_in_a_bin_directory_has_an_install_root() {
        // `--root` is what makes cargo write back to where this binary is. A
        // development build in `target/release` has no such root, and
        // guessing one would install somewhere the user is not running from.
        assert_eq!(
            install_root(Path::new("/home/user/.cargo/bin/map")),
            Some(PathBuf::from("/home/user/.cargo"))
        );
        assert_eq!(
            install_root(Path::new("/opt/tools/bin/map")),
            Some(PathBuf::from("/opt/tools"))
        );
        assert_eq!(install_root(Path::new("/repo/target/release/map")), None);
        assert_eq!(install_root(Path::new("map")), None);
    }

    /// A new, empty directory for one test.
    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("map-upgrade-test-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("bin")).unwrap();
        dir
    }

    #[test]
    fn a_bin_directory_that_cargo_does_not_own_is_not_an_install_root() {
        // `/usr/local/bin/map` has a parent named `bin` too. An install with
        // `--root /usr/local` would write cargo's record files there.
        let root = scratch("root");
        assert!(!is_cargo_root(&root));
        std::fs::write(root.join(".crates.toml"), "[v1]\n").unwrap();
        assert!(is_cargo_root(&root));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_panic_during_the_build_leaves_the_previous_binary_in_place() {
        let root = scratch("panic");
        let exe = root.join("bin").join("map");
        let aside = aside_path(&exe);
        std::fs::write(&aside, "previous").unwrap();

        let result = std::panic::catch_unwind(|| {
            let _set_aside = SetAside {
                exe: &exe,
                aside: &aside,
            };
            panic!("the build stopped");
        });

        assert!(result.is_err());
        assert_eq!(std::fs::read_to_string(&exe).unwrap(), "previous");
        assert!(!aside.exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_failed_upgrade_never_replaces_a_binary_that_cargo_installed() {
        // cargo can put the new binary in place and fail afterwards. Moving
        // the previous one back then would undo a completed install.
        let root = scratch("installed");
        let exe = root.join("bin").join("map");
        let aside = aside_path(&exe);
        std::fs::write(&aside, "previous").unwrap();
        std::fs::write(&exe, "new").unwrap();

        let set_aside = SetAside {
            exe: &exe,
            aside: &aside,
        };
        assert_eq!(set_aside.failed("why".to_owned()), "why");
        drop(set_aside);

        assert_eq!(std::fs::read_to_string(&exe).unwrap(), "new");
        assert!(!aside.exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_failed_upgrade_with_no_new_binary_restores_the_previous_one() {
        let root = scratch("failed");
        let exe = root.join("bin").join("map");
        let aside = aside_path(&exe);
        std::fs::write(&aside, "previous").unwrap();

        let set_aside = SetAside {
            exe: &exe,
            aside: &aside,
        };
        assert_eq!(
            set_aside.failed("why".to_owned()),
            "why\nThe installed binary did not change."
        );
        drop(set_aside);

        assert_eq!(std::fs::read_to_string(&exe).unwrap(), "previous");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_build_never_reuses_a_target_directory_and_never_trusts_cargos_record() {
        // Both halves of the stale-install failure: a reused target directory
        // supplies the old artifact, and cargo's record says it is current.
        let args = cargo_args(
            "https://example.invalid/map",
            Path::new("/home/user/.cargo"),
            Path::new("/tmp/map-upgrade-1"),
            &[],
        );
        let args: Vec<String> = args
            .iter()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            args,
            [
                "install",
                "--git",
                "https://example.invalid/map",
                "map-cli",
                "--locked",
                "--force",
                "--root",
                "/home/user/.cargo",
                "--target-dir",
                "/tmp/map-upgrade-1",
            ]
        );
    }

    #[test]
    fn the_features_of_this_binary_go_to_cargo_as_one_argument() {
        let args = cargo_args(
            "https://example.invalid/map",
            Path::new("/r"),
            Path::new("/t"),
            &["distilled", "llm"],
        );
        let tail: Vec<String> = args[args.len() - 2..]
            .iter()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        assert_eq!(tail, ["--features", "distilled llm"]);
        assert!(shown(&args).ends_with("--features \"distilled llm\""));
    }

    #[test]
    fn an_unknown_or_modified_build_is_never_called_the_latest() {
        // Skipping the install is only right when both commits are known.
        let remote = "6b428d8d5f0c2a1e9d3b7c4a8e6f1b2c3d4e5f60";
        assert!(is_latest(Some("6b428d8d"), Some(remote)));
        assert!(!is_latest(Some("897be86b"), Some(remote)));
        assert!(!is_latest(None, Some(remote)));
        assert!(!is_latest(Some("6b428d8d"), None));
    }

    #[test]
    fn the_remote_head_is_the_first_field_of_the_first_line() {
        let output = "6B428D8D5f0c2a1e9d3b7c4a8e6f1b2c3d4e5f60\tHEAD\n";
        assert_eq!(
            parse_remote_head(output).as_deref(),
            Some("6b428d8d5f0c2a1e9d3b7c4a8e6f1b2c3d4e5f60")
        );
        // An error page or a warning is not a commit.
        assert_eq!(parse_remote_head("warning: redirecting\tHEAD\n"), None);
        assert_eq!(parse_remote_head(""), None);
    }

    #[test]
    fn a_repository_that_reads_as_an_option_is_refused() {
        // `git ls-remote --upload-pack=<program>` runs that program. The URL
        // comes from the command line, and it must stay a URL.
        assert!(is_repository_argument("https://example.invalid/map"));
        assert!(is_repository_argument("file:///srv/git/map"));
        assert!(!is_repository_argument("--upload-pack=touch /tmp/x"));
        assert!(!is_repository_argument("-o"));
        assert!(!is_repository_argument(""));
    }

    #[test]
    fn the_set_aside_name_keeps_the_binary_name_as_its_start() {
        // `map.exe.old`, not `map.old`: the cleanup on the next start derives
        // the same name from `current_exe`, and the two have to agree.
        assert_eq!(
            aside_path(Path::new("/r/bin/map.exe")),
            PathBuf::from("/r/bin/map.exe.old")
        );
        assert_eq!(
            aside_path(Path::new("/r/bin/map")),
            PathBuf::from("/r/bin/map.old")
        );
    }
}
