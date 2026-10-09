//! What this binary is: its version, its commit, and its cargo features.
//!
//! One place, because three callers have to agree: `map --version` shows it,
//! `map upgrade` rebuilds with the same features, and the hint printed for an
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

/// The text of `map --version`: the version line, then what an upgrade needs
/// to know.
pub(crate) fn long_version() -> String {
    let features = features();
    let listed = if features.is_empty() {
        "none".to_owned()
    } else {
        features.join(" ")
    };
    format!(
        "{}\nfeatures: {listed}\nupgrade:  map upgrade\n          or: {}",
        version_line(),
        install_command(REPOSITORY, &features)
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
}
