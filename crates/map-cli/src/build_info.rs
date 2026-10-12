//! What this binary is: its version, its commit, its cargo features, and the
//! release archive it came in, if a release made it.
//!
//! One place, because three callers have to agree: `map --version` shows it,
//! `map upgrade` installs the same features again, and the hint printed for an
//! index this binary is too old to read names the same command.

/// `0.0.1 (6b428d8d 2026-10-09)`, or the number alone when the build had no
/// git data to read.
pub(crate) fn version_line() -> String {
    describe(
        env!("CARGO_PKG_VERSION"),
        env!("MAP_BUILD_COMMIT"),
        env!("MAP_BUILD_DATE"),
    )
}

fn describe(version: &str, commit: &str, date: &str) -> String {
    match (commit.is_empty(), date.is_empty()) {
        (true, _) => version.to_owned(),
        (false, true) => format!("{version} ({commit})"),
        (false, false) => format!("{version} ({commit} {date})"),
    }
}

/// The commit this binary was built from, when that is known and exact.
///
/// `None` for a build with no git data and for a `-dirty` one: neither can be
/// compared with a remote commit and called the same code.
pub(crate) fn clean_commit() -> Option<&'static str> {
    let commit = env!("MAP_BUILD_COMMIT");
    (!commit.is_empty() && !commit.ends_with("-dirty")).then_some(commit)
}

/// The cargo features this binary was compiled with, as `cargo install
/// --features` wants them.
pub(crate) fn features() -> Vec<&'static str> {
    let mut out = Vec::new();
    // `auto-distilled` implies `distilled`, so naming both would be noise.
    if cfg!(feature = "auto-distilled") {
        out.push("auto-distilled");
    } else if cfg!(feature = "distilled") {
        out.push("distilled");
    }
    if cfg!(feature = "llm") {
        out.push("llm");
    }
    out
}

/// The repository `map upgrade` installs from unless told otherwise.
pub(crate) const REPOSITORY: &str = env!("CARGO_PKG_REPOSITORY");

/// The command that installs the latest version with `features`.
///
/// Features are quoted as one argument: the form is the same in `sh` and in
/// PowerShell, and a user can paste the line into either.
pub(crate) fn install_command(repository: &str, features: &[&str]) -> String {
    let mut command = format!("cargo install --git {repository} map-cli --locked");
    if !features.is_empty() {
        command.push_str(&format!(" --features \"{}\"", features.join(" ")));
    }
    command
}

/// The name of the release archive that this binary came in, such as
/// `map-full-x86_64-unknown-linux-musl`.
///
/// `None` for a binary that cargo built on the machine of its user. That is
/// the question that `map upgrade` asks. A release binary can be on a machine
/// with no Rust, and it must not run cargo.
pub(crate) fn release_asset() -> Option<&'static str> {
    let asset = env!("MAP_RELEASE_ASSET");
    (!asset.is_empty()).then_some(asset)
}

/// The location of the files of the newest release. The address has no
/// version in it, and a file name there has none. Thus the address of a file
/// is the same for each release.
pub(crate) const RELEASE_BASE: &str =
    concat!(env!("CARGO_PKG_REPOSITORY"), "/releases/latest/download");

/// How a Unix shell gets the install script.
pub(crate) const CURL: &str = "curl -fsSL";
/// The same with `wget`, for a machine that has no `curl`.
pub(crate) const WGET: &str = "wget -qO-";

/// The command that installs the newest release with the install script.
///
/// `asset` gives the variant. With no variant, the script installs the
/// default binary. A `map-full-` binary then loses features, and the script
/// shows no message. `fetch` is `CURL` or `WGET`, and Windows does not use it.
pub(crate) fn script_command(asset: &str, os: &str, fetch: &str) -> String {
    let full = asset.starts_with("map-full-");
    if os == "windows" {
        let script = format!("irm {RELEASE_BASE}/install.ps1 | iex");
        if full {
            format!("$env:MAP_INSTALL_VARIANT = 'full'; {script}")
        } else {
            script
        }
    } else {
        let variant = if full {
            "MAP_INSTALL_VARIANT=full "
        } else {
            ""
        };
        format!("{fetch} {RELEASE_BASE}/install.sh | {variant}sh")
    }
}

/// The text of `map --version`: the version line, then what an upgrade needs
/// to know.
pub(crate) fn long_version() -> String {
    long_version_of(release_asset())
}

fn long_version_of(asset: Option<&str>) -> String {
    let features = features();
    let listed = if features.is_empty() {
        "none".to_owned()
    } else {
        features.join(" ")
    };
    // The command that installs without `map upgrade`. For a release binary
    // it is not the cargo command: its user can have no Rust.
    let manual = match asset {
        Some(asset) => script_command(asset, std::env::consts::OS, CURL),
        None => install_command(REPOSITORY, &features),
    };
    format!(
        "{}\nfeatures: {listed}\nupgrade:  map upgrade\n          or: {manual}",
        version_line()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_build_with_no_git_data_reports_the_version_alone() {
        // An empty commit must not print as `0.0.1 ()`.
        assert_eq!(describe("0.0.1", "", ""), "0.0.1");
        assert_eq!(describe("0.0.1", "", "2026-10-09"), "0.0.1");
        assert_eq!(describe("0.0.1", "6b428d8d", ""), "0.0.1 (6b428d8d)");
        assert_eq!(
            describe("0.0.1", "6b428d8d", "2026-10-09"),
            "0.0.1 (6b428d8d 2026-10-09)"
        );
    }

    #[test]
    fn the_install_command_keeps_every_feature_in_one_quoted_argument() {
        // Re-running the plain command over a `distilled` binary silently
        // installs one that cannot search `semantic` any more.
        assert_eq!(
            install_command("https://example.invalid/map", &[]),
            "cargo install --git https://example.invalid/map map-cli --locked"
        );
        assert_eq!(
            install_command("https://example.invalid/map", &["distilled", "llm"]),
            "cargo install --git https://example.invalid/map map-cli --locked \
             --features \"distilled llm\""
        );
    }

    #[test]
    fn the_listed_features_are_the_ones_compiled_in() {
        let features = features();
        assert_eq!(features.contains(&"llm"), cfg!(feature = "llm"));
        assert_eq!(
            features.contains(&"auto-distilled"),
            cfg!(feature = "auto-distilled")
        );
        // Never both: `auto-distilled` already turns `distilled` on.
        assert_eq!(
            features.contains(&"distilled"),
            cfg!(feature = "distilled") && !cfg!(feature = "auto-distilled")
        );
    }

    #[test]
    fn the_long_version_starts_with_the_short_one() {
        // `map -V` and the first line of `map --version` are the same string,
        // so a script that reads either sees one answer.
        let long = long_version();
        assert_eq!(long.lines().next(), Some(version_line().as_str()));
        assert!(long.contains("\nupgrade:  map upgrade\n"));
    }

    #[test]
    fn a_release_binary_never_names_cargo_as_the_way_to_install() {
        // The user of a release binary can have no Rust. The cargo command
        // sends that user to the install of a toolchain.
        let long = long_version_of(Some("map-x86_64-unknown-linux-musl"));
        assert!(!long.contains("cargo"), "{long}");
        assert!(
            long.contains("/releases/latest/download/install."),
            "{long}"
        );

        let built = long_version_of(None);
        assert!(built.contains("or: cargo install --git "), "{built}");
    }

    #[test]
    fn the_script_command_keeps_the_variant_of_the_binary() {
        // With no variant, the script installs the default binary. That
        // binary has two dimensions less than the binary with all features.
        let base = "https://github.com/hanpa-nai/map/releases/latest/download";
        assert_eq!(RELEASE_BASE, base);
        assert_eq!(
            script_command("map-aarch64-apple-darwin", "macos", CURL),
            format!("curl -fsSL {base}/install.sh | sh")
        );
        assert_eq!(
            script_command("map-full-x86_64-unknown-linux-musl", "linux", WGET),
            format!("wget -qO- {base}/install.sh | MAP_INSTALL_VARIANT=full sh")
        );
        assert_eq!(
            script_command("map-x86_64-pc-windows-msvc", "windows", CURL),
            format!("irm {base}/install.ps1 | iex")
        );
        assert_eq!(
            script_command("map-full-x86_64-pc-windows-msvc", "windows", CURL),
            format!("$env:MAP_INSTALL_VARIANT = 'full'; irm {base}/install.ps1 | iex")
        );
    }
}
