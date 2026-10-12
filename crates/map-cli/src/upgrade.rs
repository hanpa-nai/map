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
//!   called it. So cargo installs into a new directory, and this module puts
//!   the result in place itself: a running executable can be renamed, and two
//!   renames inside `bin` are the whole time with no `map` there.
//!
//! The build comes first and the replace last, so that a build that fails or
//! is interrupted — it takes about a minute — leaves the installed binary
//! exactly as it was.
//!
//! A binary that a release made has a second path, because its machine can
//! have no Rust. It downloads the archive of the newest release with the
//! `curl` and the `tar` of the system. Then the same two renames put the new
//! binary in place.
//!
//! That path has no checksum. A check needs a SHA-256 in each binary, and the
//! default binary links no code that calculates one. The control is the
//! transport: HTTPS, from the address that gave the first binary. The install
//! scripts are the path that examines the SHA-256.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};

use crate::build_info;

/// The last line of each error that stops an upgrade before the replace.
const UNCHANGED: &str = "The installed binary did not change.";

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

/// `exe` with `suffix` after its whole name: `map.exe.old`, not `map.old`.
fn beside(exe: &Path, suffix: &str) -> PathBuf {
    let mut name = exe.file_name().unwrap_or_default().to_os_string();
    name.push(suffix);
    exe.with_file_name(name)
}

/// Where the running binary goes when the new one takes its place.
fn aside_path(exe: &Path) -> PathBuf {
    beside(exe, ".old")
}

/// Where the new binary waits, in the directory of `exe`, for the replace.
fn incoming_path(exe: &Path) -> PathBuf {
    beside(exe, ".new")
}

/// The binary that was moved aside, until a new one is at `exe`.
///
/// A guard and not a call at each exit, so that a panic between the two
/// renames also leaves a `map` at `exe`.
struct SetAside<'a> {
    exe: &'a Path,
    aside: &'a Path,
}

impl SetAside<'_> {
    /// `why`, completed with what became of the installed binary.
    fn failed(&self, why: String) -> String {
        if self.exe.exists() {
            // Something else put a binary there in the meantime: a second
            // `map upgrade`, most likely. It stays, and the previous one has
            // no further use.
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

/// `PATH` for the cargo process, with `dir` in front.
///
/// cargo warns when the `bin` directory it installs into is not on `PATH`,
/// and tells the user to add it. Here that directory is a temporary one, and
/// the advice would be wrong.
fn path_with(dir: &Path) -> Option<OsString> {
    let current = std::env::var_os("PATH").unwrap_or_default();
    std::env::join_paths(std::iter::once(dir.to_path_buf()).chain(std::env::split_paths(&current)))
        .ok()
}

/// Whether a file can be made at `path`.
///
/// Asked before the build. The replace at the end writes beside the binary,
/// and a `bin` directory this user cannot write to should cost a second, not
/// a build of a minute.
fn check_writable(path: &Path) -> std::io::Result<()> {
    std::fs::File::create(path)?;
    std::fs::remove_file(path)
}

/// Put the binary at `built` in the place of `exe`.
///
/// The new binary is copied into the directory of `exe` first. The two
/// renames that follow are then inside one directory, and the time between
/// them is the only time with no `map` at `exe`. The running binary is renamed
/// and not overwritten, because Windows refuses to replace a running program.
fn replace(built: &Path, exe: &Path) -> Result<(), String> {
    let incoming = incoming_path(exe);
    let aside = aside_path(exe);

    std::fs::copy(built, &incoming).map_err(|e| {
        let _ = std::fs::remove_file(&incoming);
        format!("cannot write {}: {e}\n{UNCHANGED}", incoming.display())
    })?;
    // The rename replaces a file that an earlier upgrade left at `aside`.
    if let Err(e) = std::fs::rename(exe, &aside) {
        let _ = std::fs::remove_file(&incoming);
        return Err(format!(
            "cannot move {} out of the way: {e}\n{UNCHANGED}",
            exe.display()
        ));
    }
    // From here every exit has to leave a `map` at `exe`.
    let set_aside = SetAside { exe, aside: &aside };
    if let Err(e) = std::fs::rename(&incoming, exe) {
        let _ = std::fs::remove_file(&incoming);
        return Err(set_aside.failed(format!(
            "cannot put the new binary at {}: {e}",
            exe.display()
        )));
    }
    drop(set_aside);

    // Deletable everywhere except Windows, where it is this process's own
    // image; `remove_previous_binary` gets it on the next start there.
    let _ = std::fs::remove_file(&aside);
    Ok(())
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

/// The path of this binary, as a person writes it.
fn this_binary() -> Result<PathBuf, String> {
    let exe = std::env::current_exe().map_err(|e| format!("cannot find this binary: {e}"))?;
    // Resolved so that a symlink on PATH upgrades the file it points to. The
    // extended-length prefix Windows adds is taken off again: the path is
    // printed in messages.
    let exe = exe.canonicalize().unwrap_or(exe);
    Ok(PathBuf::from(crate::display_path(&exe)))
}

pub(crate) fn run(repository: Option<String>) -> Result<ExitCode, String> {
    // This question comes first, because the code below is not correct for
    // a release binary. Its user can have no cargo, and its directory is not
    // a cargo install root.
    if let Some(asset) = build_info::release_asset() {
        return run_release(asset, repository.as_deref());
    }
    let repository = repository.unwrap_or_else(|| build_info::REPOSITORY.to_owned());
    // The value goes to git and to cargo as an argument. One that starts with
    // `-` would be an option of theirs, not a place to fetch from, and git has
    // options that run a program.
    if !is_repository_argument(&repository) {
        return Err(format!("{repository:?} is not a repository URL"));
    }
    let features = build_info::features();
    let manual = build_info::install_command(&repository, &features);

    let exe = this_binary()?;
    // Only a binary that cargo installed is replaced: anything else has an
    // owner — a package manager, a build directory — that would not know.
    if !install_root(&exe).is_some_and(|root| is_cargo_root(&root)) {
        return Err(format!(
            "{} is not in the `bin` directory of a cargo install root, so `map upgrade` \
             cannot replace it. To install the newest version, run:\n  {manual}",
            exe.display()
        ));
    }

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

    check_writable(&incoming_path(&exe)).map_err(|e| {
        format!(
            "cannot write beside {}: {e}. Nothing changed. To install the newest version, \
             run:\n  {manual}",
            exe.display()
        )
    })?;

    // cargo installs into a root of its own, in a directory nothing else
    // uses. Its record in the real root is therefore not updated, and goes on
    // naming the version cargo installed there last.
    let work = std::env::temp_dir().join(format!("map-upgrade-{}", std::process::id()));
    let staging = work.join("root");
    let args = cargo_args(&repository, &staging, &work.join("target"), &features);
    let built = staging
        .join("bin")
        .join(format!("map{}", std::env::consts::EXE_SUFFIX));

    eprintln!("map: running: cargo {}", shown(&args));
    let mut cargo = Command::new("cargo");
    cargo.args(&args);
    if let Some(path) = path_with(&staging.join("bin")) {
        cargo.env("PATH", path);
    }
    let result = install(cargo.status(), &built, &exe);
    let _ = std::fs::remove_dir_all(&work);
    println!("The upgrade is complete: {}", result?);
    Ok(ExitCode::SUCCESS)
}

/// What follows the build: check its result, then replace `exe`. Returns the
/// version line of the new binary.
///
/// One function so that the build directory is removed in one place, after
/// every exit of this one.
fn install(
    status: std::io::Result<std::process::ExitStatus>,
    built: &Path,
    exe: &Path,
) -> Result<String, String> {
    let status = status.map_err(|e| {
        format!(
            "cannot run `cargo`: {e}. An upgrade builds MAP from source, so Rust is \
             necessary: https://rustup.rs\n{UNCHANGED}"
        )
    })?;
    if !status.success() || !built.is_file() {
        return Err(format!("the upgrade did not complete.\n{UNCHANGED}"));
    }
    start_and_replace(built, exe)
}

/// Start the binary at `built`, then put it in the place of `exe`. Returns
/// the version line of the new binary.
fn start_and_replace(built: &Path, exe: &Path) -> Result<String, String> {
    // Ask the new binary rather than assume: this is the only check that what
    // cargo built, or what a release gave, actually runs, and it comes before
    // anything is replaced.
    let version = match Command::new(built).arg("-V").output() {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout).trim().to_owned(),
        _ => return Err(format!("the new binary does not run.\n{UNCHANGED}")),
    };
    replace(built, exe)?;
    Ok(version)
}

/// What the `VERSION` file of a release says: the version number, and the
/// commit that the release workflow built.
#[derive(Debug, PartialEq)]
struct Release {
    version: String,
    commit: String,
}

/// The release that the first line of a `VERSION` file names.
///
/// `None` for all other text. A proxy can answer a download with a page of
/// its own and a status of success, and that page must not read as a version.
fn parse_release(text: &str) -> Option<Release> {
    let mut fields = text.lines().next()?.split_whitespace();
    let (version, commit) = (fields.next()?, fields.next()?);
    let is_version = version.starts_with(|c: char| c.is_ascii_digit())
        && version
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '+'));
    let is_commit = commit.len() == 40 && commit.chars().all(|c| c.is_ascii_hexdigit());
    (is_version && is_commit && fields.next().is_none()).then(|| Release {
        version: version.to_owned(),
        commit: commit.to_ascii_lowercase(),
    })
}

/// Whether this binary is the binary of `release`.
///
/// The version and the commit, not the version alone: a release that was
/// made again from a different commit keeps its number, and its binary is
/// not this one.
fn is_newest_release(version: &str, built: Option<&str>, release: &Release) -> bool {
    release.version == version && is_latest(built, Some(&release.commit))
}

/// The file name of the archive `asset` in a release.
///
/// Windows gets a zip, because PowerShell can open one with no other program.
/// Each other system gets a `.tar.gz`, which keeps the executable bit.
fn archive_name(asset: &str, os: &str) -> String {
    let extension = if os == "windows" { "zip" } else { "tar.gz" };
    format!("{asset}.{extension}")
}

fn release_url(base: &str, file: &str) -> String {
    format!("{base}/{file}")
}

/// The arguments that make `curl` get `url`. With no `to`, the content goes
/// to stdout.
fn curl_args(url: &str, to: Option<&Path>) -> Vec<OsString> {
    // `--fail` makes a 404 an error: without it, curl saves the error page of
    // the server as the file and reports success.
    let mut args: Vec<OsString> = ["--fail", "--silent", "--show-error", "--location"]
        .iter()
        .map(OsString::from)
        .collect();
    // A release download is a redirect to a second host. This path has no
    // checksum. Thus the redirect must not leave HTTPS.
    args.push("--proto-redir".into());
    args.push("=https".into());
    if let Some(to) = to {
        args.push("--output".into());
        args.push(to.into());
    }
    args.push(url.into());
    args
}

/// The arguments that make `tar` extract `archive` into the directory `into`.
fn tar_args(archive: &Path, into: &Path) -> Vec<OsString> {
    // `-z` is for the `.tar.gz`. The zip is for Windows only, where `tar.exe`
    // is bsdtar: it finds the format of an archive itself, and it reads a zip.
    let extract = if archive.extension().is_some_and(|e| e == "zip") {
        "-xf"
    } else {
        "-xzf"
    };
    vec![extract.into(), archive.into(), "-C".into(), into.into()]
}

/// The error for a system program that did not start.
///
/// The install script is the alternative. On Windows the script does not use
/// `curl` or `tar`, and on Unix it can use `wget`.
fn cannot_run(program: &str, error: &std::io::Error, asset: &str) -> String {
    let fetch = if program == "curl" {
        build_info::WGET
    } else {
        build_info::CURL
    };
    format!(
        "cannot run `{program}`: {error}. `map upgrade` downloads a release with `curl` and \
         extracts it with `tar`.\n{UNCHANGED}\nThe install script also installs the newest \
         version:\n  {}",
        build_info::script_command(asset, std::env::consts::OS, fetch)
    )
}

/// Get `url` with `curl`: into the file `to`, or as the returned bytes.
fn download(url: &str, to: Option<&Path>, asset: &str) -> Result<Vec<u8>, String> {
    let output = Command::new("curl")
        .args(curl_args(url, to))
        .stdin(Stdio::null())
        .output()
        .map_err(|e| cannot_run("curl", &e, asset))?;
    if !output.status.success() {
        // curl gives the cause, such as a host that it cannot find or a 404.
        let cause = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "cannot download {url}\n{}\n{UNCHANGED}",
            cause.trim()
        ));
    }
    Ok(output.stdout)
}

/// The newest release, as the `VERSION` file at `base` gives it.
fn read_release(base: &str, asset: &str) -> Result<Release, String> {
    let url = release_url(base, "VERSION");
    let text = download(&url, None, asset)?;
    parse_release(&String::from_utf8_lossy(&text))
        .ok_or_else(|| format!("{url} does not give a version and a commit.\n{UNCHANGED}"))
}

/// Download the archive of `asset` from `base` and extract it in `work`.
/// Returns the path of the new binary.
fn fetch_binary(asset: &str, base: &str, work: &Path) -> Result<PathBuf, String> {
    let cannot_make =
        |dir: &Path, e: std::io::Error| format!("cannot make {}: {e}\n{UNCHANGED}", dir.display());
    // `create_dir`, not `create_dir_all`: it stops when the directory exists.
    // Each user of a machine can write in its temporary directory, and a
    // directory that this process did not make can hold the file of another
    // user.
    let _ = std::fs::remove_dir_all(work);
    std::fs::create_dir(work).map_err(|e| cannot_make(work, e))?;
    let extracted = work.join("extracted");
    std::fs::create_dir(&extracted).map_err(|e| cannot_make(&extracted, e))?;

    let file = archive_name(asset, std::env::consts::OS);
    let archive = work.join(&file);
    let url = release_url(base, &file);
    eprintln!("map: downloading {url}");
    download(&url, Some(&archive), asset)?;

    let status = Command::new("tar")
        .args(tar_args(&archive, &extracted))
        .stdin(Stdio::null())
        .status()
        .map_err(|e| cannot_run("tar", &e, asset))?;
    let built = extracted.join(format!("map{}", std::env::consts::EXE_SUFFIX));
    if !status.success() || !built.is_file() {
        return Err(format!(
            "cannot get the `map` binary out of {file}.\n{UNCHANGED}"
        ));
    }
    Ok(built)
}

/// The upgrade of a binary that a release made: no cargo and no build.
fn run_release(asset: &str, repository: Option<&str>) -> Result<ExitCode, String> {
    if repository.is_some() {
        return Err(
            "`--git` gives the repository that cargo builds from. A release made this binary, \
             and its upgrade downloads the newest release. Run `map upgrade` without `--git`."
                .to_owned(),
        );
    }
    let exe = this_binary()?;
    let installed = build_info::version_line();
    let base = build_info::RELEASE_BASE;

    let release = read_release(base, asset)?;
    let version = env!("CARGO_PKG_VERSION");
    if is_newest_release(version, build_info::clean_commit(), &release) {
        println!("map {installed} is the newest version.");
        return Ok(ExitCode::SUCCESS);
    }
    eprintln!("map: this binary is map {installed}");
    eprintln!(
        "map: the newest release is {} ({})",
        release.version,
        &release.commit[..8]
    );

    check_writable(&incoming_path(&exe)).map_err(|e| {
        format!(
            "cannot write beside {}: {e}. Nothing changed. To install the newest version, \
             run:\n  {}",
            exe.display(),
            build_info::script_command(asset, std::env::consts::OS, build_info::CURL)
        )
    })?;

    let work = std::env::temp_dir().join(format!("map-upgrade-{}", std::process::id()));
    let result = fetch_binary(asset, base, &work).and_then(|built| start_and_replace(&built, &exe));
    let _ = std::fs::remove_dir_all(&work);
    println!("The upgrade is complete: {}", result?);
    Ok(ExitCode::SUCCESS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_binary_in_a_bin_directory_has_an_install_root() {
        // Only a binary in an install root is replaced. A development build
        // in `target/release` has no such root.
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
    fn a_panic_between_the_two_renames_leaves_the_previous_binary_in_place() {
        let root = scratch("panic");
        let exe = root.join("bin").join("map");
        let aside = aside_path(&exe);
        std::fs::write(&aside, "previous").unwrap();

        let result = std::panic::catch_unwind(|| {
            let _set_aside = SetAside {
                exe: &exe,
                aside: &aside,
            };
            panic!("stopped between the renames");
        });

        assert!(result.is_err());
        assert_eq!(std::fs::read_to_string(&exe).unwrap(), "previous");
        assert!(!aside.exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_new_binary_takes_the_place_and_no_other_file_stays() {
        let root = scratch("replace");
        let exe = root.join("bin").join("map");
        let built = root.join("built");
        std::fs::write(&exe, "previous").unwrap();
        std::fs::write(&built, "new").unwrap();

        replace(&built, &exe).unwrap();

        assert_eq!(std::fs::read_to_string(&exe).unwrap(), "new");
        // In this test `exe` is not a running program, so the set-aside file
        // can be deleted on every platform.
        let left: Vec<_> = std::fs::read_dir(root.join("bin"))
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        assert_eq!(left, ["map"]);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_build_with_no_binary_leaves_the_installed_one_as_it_was() {
        let root = scratch("nobuild");
        let exe = root.join("bin").join("map");
        std::fs::write(&exe, "previous").unwrap();

        let error = replace(&root.join("no-such-file"), &exe).unwrap_err();

        assert!(
            error.ends_with("The installed binary did not change."),
            "{error}"
        );
        assert_eq!(std::fs::read_to_string(&exe).unwrap(), "previous");
        assert!(!incoming_path(&exe).exists());
        assert!(!aside_path(&exe).exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_directory_that_cannot_be_written_is_found_before_the_build() {
        let root = scratch("writable");
        let exe = root.join("bin").join("map");
        assert!(check_writable(&incoming_path(&exe)).is_ok());
        assert!(
            !incoming_path(&exe).exists(),
            "the check must not leave a file"
        );
        assert!(check_writable(&incoming_path(&root.join("no-such-dir").join("map"))).is_err());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_failed_replace_never_overwrites_a_binary_that_appeared_meanwhile() {
        // A second `map upgrade` can put its binary at `exe` between the two
        // renames. Moving the previous one back would undo that install.
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
    fn only_a_version_and_a_full_commit_read_as_a_release() {
        let commit = "6b428d8d5f0c2a1e9d3b7c4a8e6f1b2c3d4e5f60";
        assert_eq!(
            parse_release(&format!("0.0.6 {}\n", commit.to_ascii_uppercase())),
            Some(Release {
                version: "0.0.6".to_owned(),
                commit: commit.to_owned(),
            })
        );
        assert!(parse_release(&format!("1.2.0-rc.1 {commit}\r\n")).is_some());
        // An error page, a short commit, and a line with more fields are not
        // the file that the release workflow writes.
        assert_eq!(parse_release("<!DOCTYPE html>\n<html>\n"), None);
        assert_eq!(parse_release("Not Found"), None);
        assert_eq!(parse_release("0.0.6 6b428d8d\n"), None);
        assert_eq!(parse_release(&format!("0.0.6 {commit} extra\n")), None);
        assert_eq!(parse_release(&format!("v0.0.6 {commit}\n")), None);
        assert_eq!(parse_release(""), None);
    }

    #[test]
    fn the_newest_release_is_the_same_version_and_the_same_commit() {
        // "Is the newest" means that the download gives this binary again.
        let release = Release {
            version: "0.0.6".to_owned(),
            commit: "6b428d8d5f0c2a1e9d3b7c4a8e6f1b2c3d4e5f60".to_owned(),
        };
        assert!(is_newest_release("0.0.6", Some("6b428d8d"), &release));
        assert!(!is_newest_release("0.0.5", Some("6b428d8d"), &release));
        // The same number from a different commit is a different binary.
        assert!(!is_newest_release("0.0.6", Some("897be86b"), &release));
        assert!(!is_newest_release("0.0.6", None, &release));
    }

    #[test]
    fn the_archive_of_a_release_has_no_version_in_its_name() {
        // The name is the same for each release. Thus the address of the
        // newest archive is a constant, and the binary of a previous release
        // knows it.
        assert_eq!(
            archive_name("map-x86_64-pc-windows-msvc", "windows"),
            "map-x86_64-pc-windows-msvc.zip"
        );
        assert_eq!(
            archive_name("map-full-aarch64-unknown-linux-musl", "linux"),
            "map-full-aarch64-unknown-linux-musl.tar.gz"
        );
        assert_eq!(
            archive_name("map-aarch64-apple-darwin", "macos"),
            "map-aarch64-apple-darwin.tar.gz"
        );
        assert_eq!(
            release_url(build_info::RELEASE_BASE, "VERSION"),
            "https://github.com/hanpa-nai/map/releases/latest/download/VERSION"
        );
    }

    #[test]
    fn a_download_never_saves_an_error_page_and_never_leaves_https() {
        // The upgrade from a release has no checksum. Thus `--fail` and the
        // limit on a redirect are the only controls on the file that replaces
        // the binary.
        let shown = |args: Vec<OsString>| -> Vec<String> {
            args.iter()
                .map(|a| a.to_string_lossy().into_owned())
                .collect()
        };
        assert_eq!(
            shown(curl_args("https://example.invalid/VERSION", None)),
            [
                "--fail",
                "--silent",
                "--show-error",
                "--location",
                "--proto-redir",
                "=https",
                "https://example.invalid/VERSION",
            ]
        );
        let to_file = shown(curl_args(
            "https://example.invalid/map.zip",
            Some(Path::new("/tmp/w/map.zip")),
        ));
        assert_eq!(
            to_file[6..],
            [
                "--output",
                "/tmp/w/map.zip",
                "https://example.invalid/map.zip"
            ]
        );
    }

    #[test]
    fn a_missing_program_is_an_error_that_gives_the_install_script() {
        let missing = std::io::Error::from(std::io::ErrorKind::NotFound);
        let error = cannot_run("tar", &missing, "map-full-x86_64-unknown-linux-musl");
        assert!(error.starts_with("cannot run `tar`: "), "{error}");
        assert!(error.contains(UNCHANGED), "{error}");
        assert!(
            error.contains("/releases/latest/download/install."),
            "{error}"
        );
        // The script of a machine with no `curl` cannot come through `curl`.
        if !cfg!(windows) {
            let error = cannot_run("curl", &missing, "map-x86_64-unknown-linux-musl");
            assert!(error.contains("\n  wget -qO- https://"), "{error}");
            assert!(!error.contains("curl -fsSL"), "{error}");
        }
    }

    /// `dir` as a `file://` address. `curl` reads one with no network, so
    /// the tests below run the real download code.
    fn file_url(dir: &Path) -> String {
        let path = dir.to_string_lossy().replace('\\', "/");
        if path.starts_with('/') {
            format!("file://{path}")
        } else {
            format!("file:///{path}")
        }
    }

    /// Make the archive of `asset` for this platform in `dir`. `member` is
    /// its one file.
    fn make_archive(dir: &Path, asset: &str, member: &str) {
        let source = dir.join("source");
        std::fs::create_dir_all(&source).unwrap();
        std::fs::write(source.join(member), "new").unwrap();
        let archive = dir.join(archive_name(asset, std::env::consts::OS));
        // `-a` makes bsdtar select the zip format from the name of the file.
        let create = if cfg!(windows) { "-acf" } else { "-czf" };
        let status = Command::new("tar")
            .arg(create)
            .arg(&archive)
            .arg("-C")
            .arg(&source)
            .arg(member)
            .status()
            .expect("the tests of the release path use the `tar` of the system");
        assert!(status.success());
    }

    #[test]
    fn the_version_file_of_a_release_is_read_with_no_change_to_a_file() {
        let root = scratch("version");
        let commit = "6b428d8d5f0c2a1e9d3b7c4a8e6f1b2c3d4e5f60";
        std::fs::write(root.join("VERSION"), format!("0.0.6 {commit}\n")).unwrap();

        let release = read_release(&file_url(&root), "map-test").unwrap();

        assert_eq!(release.version, "0.0.6");
        assert_eq!(release.commit, commit);
        assert!(is_newest_release("0.0.6", Some("6b428d8d"), &release));
        let left: Vec<_> = std::fs::read_dir(&root)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        assert_eq!(left.len(), 2, "{left:?}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_release_that_cannot_be_read_is_an_error_and_never_a_version() {
        let root = scratch("noversion");

        let missing = read_release(&file_url(&root), "map-test").unwrap_err();
        assert!(missing.starts_with("cannot download file:///"), "{missing}");
        assert!(missing.ends_with(UNCHANGED), "{missing}");

        std::fs::write(root.join("VERSION"), "<html>Sign in</html>\n").unwrap();
        let page = read_release(&file_url(&root), "map-test").unwrap_err();
        assert!(
            page.contains("does not give a version and a commit"),
            "{page}"
        );
        assert!(page.ends_with(UNCHANGED), "{page}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_new_binary_comes_out_of_the_archive_of_this_platform() {
        let root = scratch("archive");
        let member = format!("map{}", std::env::consts::EXE_SUFFIX);
        make_archive(&root, "map-test", &member);
        let work = root.join("work");
        // The function does not use a directory from a previous process. It
        // makes the directory again.
        std::fs::create_dir_all(work.join("extracted")).unwrap();
        std::fs::write(work.join("extracted").join(&member), "stale").unwrap();

        let built = fetch_binary("map-test", &file_url(&root), &work).unwrap();

        assert_eq!(built, work.join("extracted").join(&member));
        assert_eq!(std::fs::read_to_string(&built).unwrap(), "new");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_release_with_no_archive_for_this_binary_is_an_error() {
        let root = scratch("noarchive");

        let error = fetch_binary("map-test", &file_url(&root), &root.join("work")).unwrap_err();

        assert!(error.starts_with("cannot download file:///"), "{error}");
        assert!(error.ends_with(UNCHANGED), "{error}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn an_archive_with_no_binary_is_an_error() {
        let root = scratch("nobinary");
        make_archive(&root, "map-test", "LICENSE");

        let error = fetch_binary("map-test", &file_url(&root), &root.join("work")).unwrap_err();

        assert!(
            error.starts_with("cannot get the `map` binary out of map-test."),
            "{error}"
        );
        assert!(error.ends_with(UNCHANGED), "{error}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_new_binary_that_does_not_start_never_replaces_the_installed_one() {
        let root = scratch("nostart");
        let exe = root.join("bin").join("map");
        let built = root.join("built");
        std::fs::write(&exe, "previous").unwrap();
        // The file is text, and it does not start as a program.
        std::fs::write(&built, "new").unwrap();

        let error = start_and_replace(&built, &exe).unwrap_err();

        assert!(error.ends_with(UNCHANGED), "{error}");
        assert_eq!(std::fs::read_to_string(&exe).unwrap(), "previous");
        let _ = std::fs::remove_dir_all(&root);
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
        assert_eq!(
            incoming_path(Path::new("/r/bin/map.exe")),
            PathBuf::from("/r/bin/map.exe.new")
        );
    }
}
