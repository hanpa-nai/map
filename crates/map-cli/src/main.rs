//! The `map` command-line interface.
//!
//! The CLI is a first-class frontend, not a wrapper around something else.
//! An agent that has only shell access gets every capability, which is what
//! keeps any one protocol from becoming load-bearing.
//!
//! The `///` comments on the argument types below are the `--help` text. A
//! model reads that text as readily as a person does, so it is written in
//! Simplified Technical English: short active sentences and one word for one
//! meaning.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Args, CommandFactory, FromArgMatches, Parser, Subcommand};

mod brief;
mod build_info;
mod init;
mod upgrade;

#[derive(Parser)]
#[command(
    name = "map",
    version,
    about = "Model Awareness Plane — a search index that you commit with your resources",
    long_about = None,
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Make a `.map` directory.
    ///
    /// The new index has the `lexical` and `declaration` dimensions. These two
    /// dimensions are offline. No key, no network, and no model download are
    /// necessary.
    Init {
        /// The directory in which to make `.map`. Default: the current directory.
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Overwrite the configuration of a `.map` that is there.
        #[arg(long)]
        force: bool,
    },

    /// Build or update the index.
    Index {
        /// The root directory of the index. Default: the current directory.
        #[arg(default_value = ".")]
        path: PathBuf,

        /// The action when the binary cannot run a configured stage.
        ///
        /// In an interactive run without this flag, you select the action. A
        /// run with no terminal stops and writes no data.
        #[arg(long, value_name = "abort|allow")]
        degraded: Option<DegradedArg>,
    },

    /// Show the LLM classifier output for one file. This command writes no files.
    ///
    /// MAP puts the parts of the model answer together into one stored
    /// descriptor. Thus an index does not show which part was empty. This
    /// command prints the structured answer of the model next to the text that
    /// MAP stores. Use it to examine a prompt before you build a full index.
    #[cfg(feature = "llm")]
    Classify {
        /// The file to classify.
        path: PathBuf,

        /// The dimension that gives the classifier.
        #[arg(short = 'd', long, default_value = "descriptive")]
        dim: String,

        /// Stop after this number of segments. The command makes one request
        /// for all of them.
        #[arg(short = 'n', long, default_value_t = 4)]
        limit: usize,
    },

    /// Search the index.
    ///
    /// A query has one field for each dimension. Use one `-d DIM[:WEIGHT]=TEXT`
    /// for each dimension that you have text for. All fields are optional. An
    /// optional `:WEIGHT` changes the effect of that dimension on the fused
    /// score.
    Find(FindArgs),

    /// Print a short description of the index for an AI model.
    ///
    /// The output gives each dimension with its description and the query
    /// syntax. It also tells the model if the index is stale, and when an
    /// update has a cost. An agent integration runs this command when a
    /// session starts. When the directory has no index, the command prints no
    /// text and stops with status 0.
    Brief {
        /// The directory to examine. Default: the current directory.
        #[arg(default_value = ".")]
        path: PathBuf,
    },

    /// Three-way merge of `.map/manifest.json`. Git runs this command.
    ///
    /// `map init` and `map index` set this command as the `map` merge driver.
    /// Git runs it as `map merge %O %A %B`. When two branches each change a
    /// different resource and build the index again, the two manifests are in
    /// conflict. A text merge cannot see the difference between a correct
    /// union of object tables and a collision of one key with different bytes.
    ///
    /// The command writes the merged manifest to `ours` and stops with status
    /// 0. On a conflict, it does not change `ours`, and it stops with status
    /// 1. Git then marks the file as a file with a conflict.
    Merge {
        /// The file of the ancestor (`%O` in git). Empty when there is no
        /// ancestor.
        base: PathBuf,
        /// The `ours` file (`%A` in git). The command writes the result to this
        /// file.
        ours: PathBuf,
        /// The `theirs` file (`%B` in git).
        theirs: PathBuf,
    },

    /// Find objects that no manifest reaches. Delete them with `--prune`.
    ///
    /// An edit to an input of a fingerprint gives the objects of that
    /// dimension new keys. Examples of such inputs are a prompt, an
    /// implementation, and an embedder. The previous objects stay on disk, and
    /// this command finds them.
    ///
    /// Without `--prune`, the command only shows the objects and changes no
    /// files. A descriptor from an LLM has a cost, and you can use it again if
    /// you change the configuration back.
    Gc {
        /// The root directory of the index. Default: the current directory.
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Delete the unreachable objects.
        #[arg(long)]
        prune: bool,
    },

    /// Show the record count and the dimensions of the index.
    Status {
        /// The root directory of the index. Default: the current directory.
        #[arg(default_value = ".")]
        path: PathBuf,
    },

    /// Install the newest version of MAP and replace this binary.
    ///
    /// The command runs `cargo install` with the cargo features of this
    /// binary. It builds MAP from source. Thus Rust and a network connection
    /// are necessary. This binary stays in its location until the build is
    /// complete. If this binary is the newest version, the command changes no
    /// files.
    Upgrade {
        /// Get the source from this git repository. Default: the MAP
        /// repository.
        #[arg(long, value_name = "URL")]
        git: Option<String>,
    },

    /// Set or show the LLM connection for dimensions that use the `llm` classifier.
    ///
    /// MAP stores the endpoint and the key in `~/.map/llm.toml`. This file is
    /// in your home directory. It is not in the committed configuration of a
    /// repository.
    #[cfg(feature = "llm")]
    Llm {
        #[command(subcommand)]
        action: LlmAction,
    },

    /// Download or examine the embedding model for the `distilled` embedder.
    #[cfg(feature = "auto-distilled")]
    Model {
        #[command(subcommand)]
        action: ModelAction,
    },
}

#[cfg(feature = "auto-distilled")]
#[derive(clap::Subcommand)]
enum ModelAction {
    /// Download the model weights into `~/.map/models`.
    ///
    /// The files come from one pinned revision. MAP compares each file with a
    /// sha256 digest in the source code before it installs the file. An index
    /// build does not download. Only this command uses the network to get
    /// weights.
    Fetch {
        /// Download and verify all files again, also when they are on disk.
        #[arg(long)]
        force: bool,
    },

    /// Show if the model is installed, and the directory where MAP looks for it.
    Status,
}

/// What to do when `config.toml` names a stage this build cannot run — an
/// endpoint that is not configured, a plugin that is not compiled in, a model
/// that is not installed, or an implementation that no longer exists.
///
/// There is a default answer for neither case, which is why this exists.
/// Building anyway is legitimate — it is how you configure a semantic dimension
/// before its endpoint is live — but the result is an index missing dimensions
/// its own config describes, and nothing about querying it looks wrong.
#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum DegradedArg {
    /// Do not build. Write no data.
    Abort,
    /// Build the dimensions that the binary can run, and omit the others.
    Allow,
}

/// Subcommands under `map llm`.
#[cfg(feature = "llm")]
#[derive(Subcommand)]
enum LlmAction {
    /// Get the endpoint, the model, and the key from you, and save them.
    Login,
    /// Show the saved connection: the endpoint and the model, without the key.
    Status,
}

#[derive(Args)]
struct FindArgs {
    /// The query text for one dimension, as `DIM[:WEIGHT]=TEXT`.
    ///
    /// Use this flag one time for each dimension. An optional `:WEIGHT`
    /// changes the effect of that dimension on the fused score. The default
    /// weight is 1.0, and `:0` removes the dimension. Example:
    /// `-d lexical:2=refresh_token -d descriptive="expires a session"`.
    #[arg(short = 'd', long = "dim", value_name = "DIM[:WEIGHT]=TEXT")]
    dims: Vec<String>,

    /// Short form for `--dim lexical=<text>`. It searches the `lexical`
    /// dimension only.
    #[arg(value_name = "QUERY")]
    lexical: Option<String>,

    /// The maximum number of hits.
    #[arg(short = 'n', long, default_value_t = 10)]
    limit: usize,

    /// The maximum number of hits from one file.
    ///
    /// Segments that are adjacent become one hit. The first hit of a file
    /// shows the lines of the other matches in that file. `0` removes the
    /// limit.
    #[arg(long, default_value_t = 2, value_name = "N")]
    per_file: usize,

    /// Print the first lines of each segment hit, with their line numbers.
    ///
    /// `--snippet` prints 12 lines, and `--snippet=5` prints 5 lines. If the
    /// hit has more lines, a `[TRUNCATED ...]` line shows the lines that the
    /// command does not print. The last one gives the lines of the full hit.
    // `require_equals`: without it, `--snippet "the query"` reads the query
    // as the number of lines.
    #[arg(
        long,
        value_name = "LINES",
        num_args = 0..=1,
        require_equals = true,
        default_missing_value = "12"
    )]
    snippet: Option<usize>,

    /// For a cluster hit, show the spans below it.
    ///
    /// A cluster has no file location, because it is not part of one file.
    /// This flag shows the files of the cluster and the number of spans in
    /// each file. The command builds the level 0 id table for this, and that
    /// cost increases with the corpus size. Thus the flag is optional.
    #[arg(long)]
    members: bool,

    /// Update the index before the search.
    ///
    /// The default is no update. An update changes only the necessary objects.
    /// But an embedding dimension or an LLM dimension can make it slow, and an
    /// LLM dimension has a cost. Without this flag, MAP tells you when the
    /// index is stale, and it does not update the index.
    #[arg(short = 'u', long)]
    update: bool,

    /// Do not print the stale-index notice.
    #[arg(short = 'q', long)]
    quiet: bool,

    /// The levels to search. Default: all levels.
    ///
    /// A segment is level `0`. The clusters of a fabricator are level `1` and
    /// higher, and each level is a summary of the level below it. Give one
    /// level, a range, or a list: `--level 0` for segments only, `--level 1,3`
    /// for two levels, `--level 0-2` for a range, `--level 0,2-4` for a list
    /// and a range.
    ///
    /// You can also use these names: `segments` (level 0), `clusters` (all
    /// levels above 0), and `all`. `clusters` has no number form, because the
    /// height of the tree is different for each corpus.
    #[arg(long, value_name = "LEVELS", default_value = "all")]
    level: String,

    /// The root directory of the index. Default: the current directory.
    #[arg(long, default_value = ".")]
    path: PathBuf,

    /// The action when `-u` finds a configured stage that the binary cannot run.
    #[arg(long, value_name = "abort|allow")]
    degraded: Option<DegradedArg>,

    /// Search one more index. You can use this flag more than one time.
    ///
    /// MAP searches each root and merges the hits into one list. Each hit
    /// shows its repository. MAP normalizes scores with the statistics of all
    /// indexes together. Thus you can compare the scores of different
    /// repositories.
    #[arg(long = "root", value_name = "PATH")]
    roots: Vec<PathBuf>,
}

fn main() -> ExitCode {
    #[cfg(windows)]
    upgrade::remove_previous_binary();

    // The version text carries the commit and the features, which a derive
    // attribute cannot compute. Leaked rather than owned: clap wants
    // `'static` here without an extra feature, and this runs once.
    let short: &'static str = Box::leak(build_info::version_line().into_boxed_str());
    let long: &'static str = Box::leak(build_info::long_version().into_boxed_str());
    let matches = Cli::command()
        .version(short)
        .long_version(long)
        .get_matches();
    let cli = Cli::from_arg_matches(&matches).unwrap_or_else(|e| e.exit());

    match run(cli) {
        Ok(code) => code,
        Err(message) => {
            eprintln!("map: {message}");
            if let Some(hint) = newer_index_hint(&message) {
                eprintln!("map: {hint}");
            }
            ExitCode::FAILURE
        }
    }
}

/// What to say when an error could mean "this binary is older than the index".
///
/// A teammate upgrades, commits an index that uses something new, and every
/// older binary then fails on it with a message that never mentions versions.
/// Matched on the message text because every error reaches `main` as a
/// string; the tests build the real errors, so a reworded message fails there
/// rather than silently losing its hint.
fn newer_index_hint(message: &str) -> Option<&'static str> {
    const CERTAIN: &str =
        "a newer version of MAP wrote this index. Run `map upgrade` to install the newest version.";
    const POSSIBLE: &str =
        "if a newer version of MAP wrote this index, run `map upgrade` to install the newest version.";

    // "index uses config version 2, this build supports 1". Anchored on
    // "index uses " first: text ahead of it can hold the word "version" too,
    // and the numbers would then be read from the wrong place.
    if let Some((_, rest)) = message.split_once("index uses ") {
        if let Some((_, rest)) = rest.split_once(" version ") {
            if let Some((found, expected)) = rest.split_once(", this build supports ") {
                let number = |text: &str| -> Option<u32> {
                    text.split(|c: char| !c.is_ascii_digit())
                        .next()?
                        .parse()
                        .ok()
                };
                return match (number(found), number(expected)) {
                    // An *older* index is a different problem with a different
                    // fix, and upgrading would not help.
                    (Some(found), Some(expected)) if found > expected => Some(CERTAIN),
                    _ => None,
                };
            }
        }
    }
    // A key or a value this binary has never heard of: a newer MAP, or a typo.
    let unknown = message.contains("unknown field") || message.contains("unknown variant");
    if (message.contains("config parse error") && unknown)
        || message.contains("index uses hash algorithm")
    {
        return Some(POSSIBLE);
    }
    None
}

fn run(cli: Cli) -> Result<ExitCode, String> {
    match cli.command {
        Command::Init { path, force } => {
            let created = init::run(&path, force).map_err(|e| e.to_string())?;
            if let Some(root) = created.root.parent() {
                register_merge_driver(root);
            }
            println!("Initialized MAP index in {}", display_path(&created.root));
            println!();
            for f in &created.files {
                println!("  {f}");
            }
            println!();
            println!("Enabled: lexical and declaration dimensions (offline, BM25)");
            println!("Next:    map index      — offline, no key, no model download");
            Ok(ExitCode::SUCCESS)
        }

        #[cfg(feature = "llm")]
        Command::Classify { path, dim, limit } => {
            // `Path::parent` on a bare filename is `Some("")`, not `None`, so
            // the obvious `unwrap_or(".")` never fires and the root search gets
            // an empty path. Treat empty as the current directory.
            let root = match path.parent() {
                Some(parent) if !parent.as_os_str().is_empty() => parent,
                _ => std::path::Path::new("."),
            };
            let out = map_index::classify_resource(root, &path, &dim, limit)
                .map_err(|e| e.to_string())?;

            let mut total = 0usize;
            for (i, one) in out.iter().enumerate() {
                println!(
                    "--- Segment {}  (lines {}-{})",
                    i + 1,
                    one.lines.0,
                    one.lines.1
                );
                println!("raw:");
                match one.raw.get(&dim) {
                    // An enforced answer shape: print each part on its own line
                    // so an empty one is visible rather than inferred.
                    Some(map_index::JsonValue::Object(parts)) => {
                        for (name, value) in parts {
                            println!("  {name:<12} {value}");
                        }
                    }
                    Some(other) => println!("  {other}"),
                    None => println!("  <nothing for dimension {dim:?}>"),
                }
                match &one.composed {
                    Some(text) => {
                        total += text.len();
                        println!("composed ({} chars):", text.len());
                        println!("  {}", terminal_safe(text));
                    }
                    None => println!("composed: <empty — nothing would be stored>"),
                }
                println!();
            }
            if !out.is_empty() {
                // Stability of the stored size is the thing to watch: it is
                // what dilutes a mean-pooled embedding and what the output
                // token budget is spent on.
                println!("{} segments, mean {} chars", out.len(), total / out.len());
            }
            Ok(ExitCode::SUCCESS)
        }

        Command::Index { path, degraded } => {
            let started = std::time::Instant::now();
            let mut ask = prompt_degraded;
            let policy = match degraded {
                Some(DegradedArg::Allow) => map_index::Degraded::Allow,
                Some(DegradedArg::Abort) => map_index::Degraded::Abort,
                None if can_prompt() => map_index::Degraded::Ask(&mut ask),
                None => map_index::Degraded::Abort,
            };
            let stats =
                map_index::run_with_progress(&path, policy, |_| {}).map_err(|e| e.to_string())?;
            let elapsed = started.elapsed();

            // Re-run on every index, not once at init: a clone carries
            // `.gitattributes` but not the config that makes the attribute
            // mean anything, so the person who ran `map init` is the only one
            // `map init` alone could ever reach.
            if let Some(root) = map_core::find_map_root(&path) {
                register_merge_driver(&root);
            }

            println!(
                "Indexed {} resources ({} skipped) into {} segments in {:.2}s",
                stats.indexed,
                stats.skipped,
                stats.segments,
                elapsed.as_secs_f64()
            );
            println!(
                "  objects: {} written, {} reused",
                stats.objects_written, stats.objects_reused
            );
            // Surfaced because it is the number the cost model rests on: one
            // call per implementation group, never one per dimension.
            println!("  classifier calls: {}", stats.classifier_calls);
            // Only meaningful once a dimension configures a fabricator; the
            // label calls are the fabricator's analog of the classifier bill.
            if stats.clusters_written > 0 || stats.label_calls > 0 {
                println!(
                    "  clusters: {} written, {} label calls",
                    stats.clusters_written, stats.label_calls
                );
            }
            // An index that built "successfully" with a blank semantic
            // dimension is worse than one that failed, because nothing about
            // querying it looks wrong. Loud, and on stderr so a pipeline sees
            // it.
            if stats.undescribed_segments > 0 {
                eprintln!(
                    "map: {} segment(s) got no usable descriptor — the endpoint returned \
                     replies that did not match the requested schema. Check that it supports \
                     structured output (response_format json_schema).",
                    stats.undescribed_segments
                );
            }
            // The fabricate-side analogue. Without it a shorter tree looks like
            // the corpus simply had less structure, when the real cause was a
            // classifier declining to label — which is what a prompt edit
            // provokes, and exactly when you need to be told.
            if stats.unlabeled_clusters > 0 {
                eprintln!(
                    "map: {} cluster(s) got no label and were not written — their members \
                     carried forward ungrouped. The tree is shorter than the corpus supports.",
                    stats.unlabeled_clusters
                );
            }
            report_degraded(&stats.degraded);
            Ok(ExitCode::SUCCESS)
        }

        Command::Merge { base, ours, theirs } => {
            // A non-zero status is the only way git learns the file is still
            // conflicted and must be left for the operator.
            let resolved = merge_manifests(&base, &ours, &theirs)?;
            Ok(match resolved {
                true => ExitCode::SUCCESS,
                false => ExitCode::FAILURE,
            })
        }

        Command::Gc { path, prune } => {
            let report = map_index::gc::collect(&path, prune).map_err(|e| e.to_string())?;

            // Say what was consulted before saying what was found: whether the
            // other refs were readable is what decides if the answer is
            // trustworthy at all.
            println!("Reachability: {}", report.scope);
            println!("  reachable objects: {}", report.reachable);

            if report.unreachable.is_empty() {
                println!("  unreachable:       none — nothing to collect");
                return Ok(ExitCode::SUCCESS);
            }

            println!(
                "  unreachable:       {} objects, {:.2} MB",
                report.unreachable.len(),
                report.bytes as f64 / 1_000_000.0
            );
            for (store, (count, bytes)) in report.by_store() {
                println!(
                    "    {store:<9} {count:>5} objects  {:>8.2} MB",
                    bytes as f64 / 1_000_000.0
                );
            }

            if report.pruned {
                println!();
                println!("Deleted. Note this frees the working tree and future clones only —");
                println!("anything already committed stays in the repository's history.");
            } else {
                println!();
                println!("Nothing deleted. Re-run with --prune to remove them.");
            }
            Ok(ExitCode::SUCCESS)
        }

        Command::Status { path } => {
            let index = map_query::Index::open(&path).map_err(|e| e.to_string())?;
            println!("Index at {}", display_path(index.root()));
            println!("  records:    {}", index.len());
            println!("  dimensions: {}", index.dimensions().join(", "));
            Ok(ExitCode::SUCCESS)
        }

        Command::Find(args) => find(args),

        Command::Upgrade { git } => upgrade::run(git),

        Command::Brief { path } => {
            // Nothing at all without an index: the caller is a session-start
            // hook that runs in every repository, and most have no `.map`.
            if let Some(text) = brief::run(&path)? {
                print!("{text}");
            }
            Ok(ExitCode::SUCCESS)
        }

        #[cfg(feature = "llm")]
        Command::Llm { action } => llm(action),

        #[cfg(feature = "auto-distilled")]
        Command::Model { action } => model(action),
    }
}

/// Fetch or inspect the embedding weights.
#[cfg(feature = "auto-distilled")]
fn model(action: ModelAction) -> Result<ExitCode, String> {
    use map_embed::fetch;

    let dir = fetch::model_dir();
    match action {
        ModelAction::Status => {
            if fetch::installed(&dir) {
                println!("{} installed at {}", fetch::MODEL_ID, dir.display());
            } else {
                println!("{} not installed", fetch::MODEL_ID);
                println!("  expected at {}", dir.display());
                println!("  run `map model fetch` to download it");
            }
            Ok(ExitCode::SUCCESS)
        }

        ModelAction::Fetch { force } => {
            if force {
                for artifact in fetch::ARTIFACTS {
                    let _ = std::fs::remove_file(dir.join(artifact.name));
                }
            }

            // Not "fetching": an artifact already present with the right
            // digest is not re-downloaded, so this states the target rather
            // than claiming a transfer that may not happen.
            let total: u64 = fetch::ARTIFACTS.iter().map(|a| a.len).sum();
            eprintln!(
                "{} — {:.1} MB from {} at {}",
                fetch::MODEL_ID,
                total as f64 / 1_000_000.0,
                fetch::REPO,
                &fetch::REVISION[..12],
            );

            // Progress is drawn only for a human. An agent shelling out gets
            // the summary lines and none of the carriage returns.
            let draw = {
                use std::io::IsTerminal;
                std::io::stderr().is_terminal()
            };
            let mut last_percent = u64::MAX;
            let mut report = |p: fetch::Progress<'_>| {
                if !draw {
                    return;
                }
                let percent = p.received * 100 / p.total.max(1);
                if percent != last_percent {
                    last_percent = percent;
                    eprint!("\r  {} {percent:>3}%", p.name);
                }
            };

            let outcome = fetch::fetch(&dir, &mut report).map_err(|e| e.to_string())?;
            if draw {
                eprintln!("\r{:40}\r", "");
            }

            for (name, disposition) in outcome {
                match disposition {
                    fetch::Disposition::Reused => println!("  {name}: already present"),
                    fetch::Disposition::Downloaded => println!("  {name}: downloaded"),
                }
            }
            println!("{} ready at {}", fetch::MODEL_ID, dir.display());
            Ok(ExitCode::SUCCESS)
        }
    }
}

/// Configure or inspect the LLM connection.
#[cfg(feature = "llm")]
fn llm(action: LlmAction) -> Result<ExitCode, String> {
    use map_llm::Connection;
    match action {
        LlmAction::Login => {
            // Always prompts and overwrites, so `login` re-configures rather
            // than silently keeping a stale connection.
            let connection = Connection::prompt_and_save().map_err(|e| e.to_string())?;
            let path = Connection::cache_path()
                .map(|p| p.display().to_string())
                .unwrap_or_default();
            println!();
            println!("Saved connection to {path}");
            println!("  endpoint: {}", connection.endpoint);
            println!("  model:    {}", connection.model);
            println!(
                "  api key:  {}",
                if connection.api_key.is_some() {
                    "set"
                } else {
                    "(none — local endpoint)"
                }
            );
            Ok(ExitCode::SUCCESS)
        }
        LlmAction::Status => match Connection::load() {
            Some(c) => {
                println!(
                    "LLM connection ({}):",
                    Connection::cache_path()
                        .map(|p| p.display().to_string())
                        .unwrap_or_default()
                );
                println!("  endpoint: {}", c.endpoint);
                println!("  protocol: {}", c.protocol);
                println!("  model:    {}", c.model);
                println!(
                    "  api key:  {}",
                    if c.api_key.is_some() { "set" } else { "(none)" }
                );
                Ok(ExitCode::SUCCESS)
            }
            None => {
                println!("No LLM connection configured. Run `map llm login` to set one up.");
                Ok(ExitCode::SUCCESS)
            }
        },
    }
}

/// Git's merge driver for `.map/manifest.json`.
///
/// Git hands three temp files and reads the result back out of `ours`, so the
/// merged manifest is written there and stdout stays empty. `true` means
/// resolved; `false` means leave the file conflicted.
///
/// A missing or empty base file is "no common ancestor" rather than an error —
/// that is what git passes when the file was added on both sides.
fn merge_manifests(base: &Path, ours: &Path, theirs: &Path) -> Result<bool, String> {
    let parse = |bytes: &[u8], side: &str, path: &Path| {
        map_format::Manifest::from_bytes(bytes).map_err(|e| {
            format!(
                "{side} ({}) is not a readable manifest: {e}",
                path.display()
            )
        })
    };
    let read = |path: &Path, side: &str| -> Result<map_format::Manifest, String> {
        let bytes = std::fs::read(path)
            .map_err(|e| format!("cannot read {side} ({}): {e}", path.display()))?;
        parse(&bytes, side, path)
    };

    let base_bytes = std::fs::read(base).unwrap_or_default();
    let base_manifest = match base_bytes.is_empty() {
        true => None,
        false => Some(parse(&base_bytes, "the common ancestor", base)?),
    };
    let ours_manifest = read(ours, "our side")?;
    let theirs_manifest = read(theirs, "their side")?;

    match map_format::Manifest::merge(base_manifest.as_ref(), &ours_manifest, &theirs_manifest) {
        Ok(merged) => {
            let bytes = merged.manifest.to_bytes().map_err(|e| e.to_string())?;
            std::fs::write(ours, &bytes)
                .map_err(|e| format!("cannot write {}: {e}", ours.display()))?;
            // Notes are not decoration: a dropped cluster tree leaves the index
            // queryable but flat, and nothing else would say so.
            for note in &merged.notes {
                eprintln!("map: {note}");
            }
            Ok(true)
        }
        Err(conflicts) => {
            for conflict in &conflicts {
                eprintln!("map: manifest merge conflict: {conflict}");
            }
            // `ours` is left exactly as git wrote it, so the working-tree file
            // still holds our version for the operator to resolve against.
            Ok(false)
        }
    }
}

/// Render a path for a person to read.
///
/// `canonicalize` on Windows returns the extended-length form — `\\?\C:\repo`
/// — which is correct, accepted almost nowhere a user would paste it, and not
/// what anyone typed. A UNC share canonicalizes to `\\?\UNC\server\share`,
/// whose ordinary spelling is `\\server\share`. Anything else is left alone.
fn display_path(path: &Path) -> String {
    let text = path.display().to_string();
    let Some(rest) = text.strip_prefix(r"\\?\") else {
        return text;
    };
    match rest.strip_prefix(r"UNC\") {
        Some(share) => format!(r"\\{share}"),
        None => rest.to_owned(),
    }
}

/// The repository directory at or above `start`, if there is one.
///
/// `.git` is a directory in an ordinary clone and a file in a worktree or a
/// submodule, so existence is the test rather than `is_dir`.
fn repository_at_or_above(start: &Path) -> Option<PathBuf> {
    let mut dir = start;
    loop {
        if dir.join(".git").exists() {
            return Some(dir.to_path_buf());
        }
        dir = dir.parent()?;
    }
}

/// Whether to write the merge driver into local git config.
///
/// `git_runnable` is false when the `git` binary could not be launched at all;
/// `configured` is whatever `git config --get merge.map.driver` printed. Split
/// out from the doing so the policy is testable without a git binary.
///
/// `program_gone` is true when the configured command names a binary that is
/// no longer on disk. The command embeds an absolute path, so a `map` that was
/// installed somewhere else since — or a development build that has been
/// cleaned away — leaves a driver git cannot run, and every manifest merge
/// then fails. That is the one case an already-set driver is replaced.
fn should_register(git_runnable: bool, configured: Option<&str>, program_gone: bool) -> bool {
    let already_set = matches!(configured, Some(value) if !value.trim().is_empty());
    git_runnable && (!already_set || program_gone)
}

/// The program a driver command runs: its first word, without the quotes
/// `register_merge_driver` writes around it.
fn driver_program(command: &str) -> Option<PathBuf> {
    let command = command.trim();
    let program = match command.strip_prefix('"') {
        Some(rest) => rest.split('"').next()?,
        None => command.split_whitespace().next()?,
    };
    (!program.is_empty()).then(|| PathBuf::from(program))
}

fn git_config(repo: &Path, args: &[&str]) -> Option<std::process::Output> {
    std::process::Command::new("git")
        .arg("-C")
        .arg(repo)
        .arg("config")
        .args(args)
        .output()
        .ok()
}

/// Register `map merge` as the driver `.gitattributes` names.
///
/// **Local** config, deliberately. The command embeds this binary's absolute
/// path, which means nothing on another machine, and git will not execute a
/// command a repository distributes — which is why the attribute can travel
/// with the clone and the command cannot.
///
/// Never fails the caller. A checkout where this does not land gets git's
/// ordinary text merge on the manifest, which is where it started.
fn register_merge_driver(start: &Path) {
    let canonical = start.canonicalize().unwrap_or_else(|_| start.to_path_buf());
    let Some(repo) = repository_at_or_above(&canonical) else {
        return;
    };

    let probe = git_config(&repo, &["--get", "merge.map.driver"]);
    let configured = probe.as_ref().and_then(|output| {
        output
            .status
            .success()
            .then(|| String::from_utf8_lossy(&output.stdout).into_owned())
    });
    // Only an absolute path can be shown to be gone. A bare `map` is resolved
    // by git's shell at merge time, and is somebody's deliberate choice.
    let program_gone = configured
        .as_deref()
        .and_then(driver_program)
        .is_some_and(|program| program.is_absolute() && !program.exists());
    if !should_register(probe.is_some(), configured.as_deref(), program_gone) {
        return;
    }

    let Ok(exe) = std::env::current_exe() else {
        return;
    };
    // Forward slashes and quotes: git hands the driver command to a shell,
    // where a backslash escapes rather than separates and a space splits.
    let driver = format!(
        "\"{}\" merge %O %A %B",
        exe.display().to_string().replace('\\', "/")
    );

    let set = |key: &str, value: &str| {
        git_config(&repo, &[key, value]).is_some_and(|output| output.status.success())
    };
    if set("merge.map.name", "MAP manifest merge") && set("merge.map.driver", &driver) {
        eprintln!("map: registered git merge driver for .map/manifest.json");
    }
}

/// Parse `--level` into the set of fabric levels to search.
///
/// Accepts a name (`all`, `segments`, `clusters`), or a comma-separated list of
/// heights and inclusive ranges: `0`, `1,3,4`, `0-5`, `0,2-4`. The names are not
/// sugar for the numbers — `clusters` means every level above zero, which has no
/// fixed-height spelling because how tall the fabric grows depends on the
/// corpus.
fn parse_levels(spec: &str) -> Result<map_query::LevelFilter, String> {
    match spec.trim() {
        "all" => return Ok(map_query::LevelFilter::All),
        "segments" | "segment" => return Ok(map_query::LevelFilter::Segments),
        "clusters" | "cluster" => return Ok(map_query::LevelFilter::Clusters),
        _ => {}
    }

    let mut levels = std::collections::BTreeSet::new();
    for part in spec.split(',') {
        let part = part.trim();
        // An empty entry is a typo — `1,,3` or a trailing comma — and silently
        // dropping it would answer a different question than the one asked.
        if part.is_empty() {
            return Err(format!(
                "empty level in --level {spec:?}; expected a number, a range like 1-3, \
                 or a comma-separated list of either"
            ));
        }

        // A hyphen now means a range, so it can no longer begin a number. That
        // is what turns `-1` into a clear error rather than a silent negative.
        match part.split_once('-') {
            Some((lo, hi)) => {
                let lo = parse_level(lo, part, spec)?;
                let hi = parse_level(hi, part, spec)?;
                if lo > hi {
                    return Err(format!(
                        "range {part:?} in --level {spec:?} counts downward; write it as {hi}-{lo}"
                    ));
                }
                levels.extend(lo..=hi);
            }
            None => {
                levels.insert(parse_level(part, part, spec)?);
            }
        }
    }
    Ok(map_query::LevelFilter::Only(levels))
}

/// One level number, with an error naming the element it came from.
fn parse_level(text: &str, part: &str, spec: &str) -> Result<u16, String> {
    text.trim().parse().map_err(|_| {
        format!(
            "invalid level {part:?} in --level {spec:?}; expected a number, a range \
             like 1-3, or one of: all, segments, clusters"
        )
    })
}

/// Whether there is a person here to answer a question.
///
/// Both streams must be a terminal: stdin because the answer is read from it,
/// stderr because an unseen prompt would look like a hang.
fn can_prompt() -> bool {
    use std::io::IsTerminal;
    std::io::stdin().is_terminal() && std::io::stderr().is_terminal()
}

/// Ask whether to build an index missing some of its configured stages.
///
/// Defaults to no on anything that is not a clear yes, including a read error:
/// the whole point is that this cannot be waved through by accident.
fn prompt_degraded(stages: &[map_index::UnresolvedStage]) -> bool {
    use std::io::Write;

    eprintln!();
    eprintln!(
        "map: this build cannot run {} configured stage(s):",
        stages.len()
    );
    eprint!("{}", map_index::degraded::describe(stages));
    eprintln!();
    eprintln!("Continuing omits those dimensions from the index. A query against the");
    eprintln!("result will not look wrong — it will just be missing what they cover.");
    eprint!("Continue with a partial index? [y/N] ");
    let _ = std::io::stderr().flush();

    let mut answer = String::new();
    if std::io::stdin().read_line(&mut answer).is_err() {
        return false;
    }
    matches!(answer.trim(), "y" | "Y" | "yes" | "Yes")
}

/// Report what an accepted degraded build left out.
///
/// Said again after the fact deliberately: consenting to build without a stage
/// is not a reason to stop mentioning that the index is incomplete.
fn report_degraded(stages: &[map_index::UnresolvedStage]) {
    if stages.is_empty() {
        return;
    }
    eprintln!(
        "map: built without {} configured stage(s) — this index is incomplete:",
        stages.len()
    );
    eprint!("{}", map_index::degraded::describe(stages));
}

/// Parse a `DIM[:WEIGHT]=TEXT` argument into a dimension name and query term.
///
/// Split on the first `=` so the query text may itself contain `=` or `:`. The
/// weight, when present, is the `:`-suffix of the name half and defaults to 1.0.
fn parse_dim(spec: &str) -> Result<(String, map_query::QueryTerm), String> {
    let (head, text) = spec
        .split_once('=')
        .ok_or_else(|| format!("expected DIM[:WEIGHT]=TEXT, got {spec:?}"))?;

    let (name, weight) = match head.split_once(':') {
        None => (head, 1.0_f32),
        Some((name, raw)) => {
            let weight: f32 = raw
                .parse()
                .map_err(|_| format!("invalid weight {raw:?} in {spec:?}; expected a number"))?;
            // Upper bound so a few large weights cannot sum past f32's range and
            // collapse fusion to a meaningless all-zero ranking. 1e6 is far above
            // any sane relative weighting and keeps sums finite across dimensions.
            const MAX_WEIGHT: f32 = 1e6;
            if !weight.is_finite() || !(0.0..=MAX_WEIGHT).contains(&weight) {
                return Err(format!(
                    "weight must be a finite number in [0, {MAX_WEIGHT:.0}], got {raw:?} in {spec:?}"
                ));
            }
            (name, weight)
        }
    };

    if name.is_empty() {
        return Err(format!("missing dimension name in {spec:?}"));
    }
    Ok((
        name.to_owned(),
        map_query::QueryTerm::weighted(text, weight),
    ))
}

/// The roots to search, `--path` first, each labelled by its directory name.
///
/// A directory name is short, stable, and what a person would call the
/// repository. Duplicates are disambiguated by appending the parent, because an
/// origin is half the identity of every hit and two members cannot share one.
fn federated_roots(args: &FindArgs) -> Vec<(String, PathBuf)> {
    let configured = map_query::configured_roots();
    let mut out: Vec<(String, PathBuf)> = Vec::new();
    let mut seen_paths: Vec<PathBuf> = Vec::new();
    for path in std::iter::once(&args.path)
        .chain(&args.roots)
        .chain(&configured)
    {
        // `--path` defaults to `.`, so a configured root for the repository you
        // are standing in would otherwise be searched twice under two labels.
        let key = path.canonicalize().unwrap_or_else(|_| path.clone());
        if seen_paths.contains(&key) {
            continue;
        }
        seen_paths.push(key);
        let canonical = path.canonicalize().unwrap_or_else(|_| path.clone());
        let mut label = canonical
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| display_path(path));
        if out.iter().any(|(existing, _)| *existing == label) {
            let parent = canonical
                .parent()
                .and_then(|p| p.file_name())
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            label = format!("{parent}/{label}");
        }
        out.push((label, path.clone()));
    }
    out
}

/// Escape control characters before printing index-carried text to a terminal.
///
/// A cluster label is fabricator/LLM prose stored in a committed index, so a
/// hostile clone could embed ESC/CR sequences that rewrite the terminal or spoof
/// output. Resource keys are already rejected at index time when they carry
/// control characters, but escaping on the way out too costs nothing and closes
/// the channel however the text arrived. Borrows unchanged — no allocation — in
/// the overwhelmingly common case of clean text.
fn terminal_safe(s: &str) -> std::borrow::Cow<'_, str> {
    if !s.chars().any(char::is_control) {
        return std::borrow::Cow::Borrowed(s);
    }
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if c.is_control() {
            out.extend(c.escape_default());
        } else {
            out.push(c);
        }
    }
    std::borrow::Cow::Owned(out)
}

/// Further matches of a file named by line on its first entry; the rest are counted.
const MORE_SHOWN: usize = 3;

/// What else an entry stands for: the extent of a run of merged segments, and
/// the lines of further matches in the same file that `--per-file` left out.
///
/// `lines` is the entry's own line, the first and last line of the run, then
/// one line for each named further match.
fn spread_note(segments: usize, more: usize, lines: &[usize]) -> Option<String> {
    let mut parts = Vec::new();
    if segments > 1 {
        if let (Some(first), Some(last)) = (lines.get(1), lines.get(2)) {
            parts.push(format!("lines {first}-{last}"));
        }
    }
    if more > 0 {
        let mut also: Vec<String> = lines.iter().skip(3).map(usize::to_string).collect();
        let counted = more.saturating_sub(also.len());
        if counted > 0 {
            also.push(format!("+{counted}"));
        }
        parts.push(format!("also {}", also.join(", ")));
    }
    (!parts.is_empty()).then(|| parts.join("; "))
}

/// What `--snippet` prints for a hit: at most `limit` lines from the start of
/// its best segment, and a notice for each part of the hit that is cut.
///
/// `text` is the best segment, and its first line is `first_line`. `extent` is
/// the first and the last line of the hit. A hit is longer than its best
/// segment when the neighbours of that segment were merged into it, so lines
/// of the hit can be cut before the snippet as well as after it.
///
/// The snippet is only enough of a hit to choose between hits, so each cut is
/// said out loud. A cut that is not marked reads as the whole hit. The last
/// notice also tells the reader what to do: a reader that does not find the
/// match in the snippet otherwise searches again, for lines that this hit
/// already holds.
fn snippet_lines(
    text: &str,
    first_line: usize,
    limit: usize,
    extent: (usize, usize),
) -> Vec<String> {
    let (hit_first, hit_last) = extent;
    let shown = text.lines().count().min(limit);
    let next = first_line + shown;
    let (cut_before, cut_after) = (hit_first < first_line, next <= hit_last);
    let advice = format!(
        "The match can be in the cut lines: read lines {hit_first}-{hit_last} before you search again."
    );

    let width = (first_line + shown.saturating_sub(1)).to_string().len();
    let mut out = Vec::new();
    if cut_before {
        let before = format!(
            "lines {hit_first}-{} are before this snippet",
            first_line - 1
        );
        out.push(if cut_after {
            format!("    [TRUNCATED: {before}]")
        } else {
            format!("    [TRUNCATED: {before}. {advice}]")
        });
    }
    out.extend(
        text.lines()
            .take(shown)
            .enumerate()
            .map(|(offset, line)| format!("    {:>width$}  {line}", first_line + offset)),
    );
    if cut_after {
        out.push(format!(
            "    [TRUNCATED: lines {next}-{hit_last} are after this snippet. {advice}]"
        ));
    }
    out
}

fn find(mut args: FindArgs) -> Result<ExitCode, String> {
    let mut query = map_query::Query::new();

    if let Some(text) = args.lexical.take() {
        query.insert("lexical".to_owned(), map_query::QueryTerm::new(text));
    }
    for spec in &args.dims {
        let (dimension, term) = parse_dim(spec)?;
        query.insert(dimension, term);
    }
    if query.is_empty() {
        return Err("nothing to search for — pass a query or --dim DIM[:WEIGHT]=TEXT".to_owned());
    }

    if args.update {
        // Same policy as `map index`, deliberately: `-u` writes the same
        // objects, so it can leave the same incomplete index behind. One rule
        // is also the only way the two cannot drift apart.
        let mut ask = prompt_degraded;
        let policy = match args.degraded {
            Some(DegradedArg::Allow) => map_index::Degraded::Allow,
            Some(DegradedArg::Abort) => map_index::Degraded::Abort,
            None if can_prompt() => map_index::Degraded::Ask(&mut ask),
            None => map_index::Degraded::Abort,
        };

        let mut reporter = ProgressLine::new(args.quiet);
        let updated = map_index::refresh_with_progress(&args.path, policy, |p| reporter.tick(&p))
            .map_err(|e| e.to_string())?;
        reporter.finish();

        if let Some(stats) = updated {
            if stats.objects_written > 0 && !args.quiet {
                eprintln!(
                    "map: updated {} object(s) before searching",
                    stats.objects_written
                );
            }
            if !args.quiet {
                report_degraded(&stats.degraded);
            }
        }
    }

    let levels = parse_levels(&args.level)?;
    // A request for cluster levels on an index that never built any is the one
    // "no matches" that is a property of the index, not the query.
    let asks_above_zero = match &levels {
        map_query::LevelFilter::Segments | map_query::LevelFilter::All => false,
        map_query::LevelFilter::Clusters => true,
        map_query::LevelFilter::Only(set) => !set.contains(&0),
    };

    // `--path` is always a member, so a bare `map find` and a federated one go
    // through the same code and cannot drift.
    let roots = federated_roots(&args);
    let federated = roots.len() > 1;
    let index = map_query::Federation::open(roots).map_err(|e| e.to_string())?;
    // Every ranked hit, not the first `--limit`: merging neighbours and
    // limiting one file both free places, and `regions` reads as deep as it
    // must to fill them.
    let hits = index
        .find_at(&query, usize::MAX, levels)
        .map_err(|e| e.to_string())?;

    if hits.is_empty() {
        println!("no matches");
        if asks_above_zero && index.cluster_count() == 0 {
            eprintln!("map: this index has no clusters, so no level above 0 can match");
        }
        // Worth saying precisely here: "no matches" on a stale index is the
        // case most likely to be a wrong answer rather than a true negative.
        report_staleness(&args);
        return Ok(ExitCode::SUCCESS);
    }

    let regions = map_query::regions(&hits, args.per_file, args.limit);
    for region in &regions {
        let hit = &region.best;
        let mut spread = None;
        let mut extent = None;
        if hit.level > 0 {
            // A cluster spans no file — show the fabricator's label, not a
            // line. This is the zoomed-out overview the level filter selects.
            let label = index.cluster_label(hit).unwrap_or_default();
            print!(
                "cluster L{}  {:.3}  {}",
                hit.level,
                hit.score,
                terminal_safe(&label)
            );
        } else {
            // One read of the file gives every line number this entry shows.
            let mut offsets = vec![hit.start, region.start, region.end.saturating_sub(1)];
            offsets.extend(region.more.iter().take(MORE_SHOWN));
            let lines = index.lines_of(hit, &offsets).unwrap_or_default();
            let line = lines.first().copied().unwrap_or(1);
            spread = spread_note(region.segments, region.more.len(), &lines);
            extent = lines.get(1).copied().zip(lines.get(2).copied());
            // Prefix the origin only when there is more than one, so a
            // single-index query's output is unchanged and stays greppable.
            let where_ = if federated {
                format!(
                    "{}/{}",
                    terminal_safe(&hit.origin),
                    terminal_safe(&hit.resource)
                )
            } else {
                terminal_safe(&hit.resource).into_owned()
            };
            // file:line is clickable in most terminals, which matters because
            // the consumer here is often an agent pasting it straight back.
            print!("{}:{}  {:.3}", where_, line, hit.score);
        }
        if hit.per_dimension.len() > 1 {
            let breakdown: Vec<String> = hit
                .per_dimension
                .iter()
                .map(|(d, s)| format!("{d} {s:.2}"))
                .collect();
            print!("  [{}]", breakdown.join(", "));
        }
        if let Some(note) = spread {
            print!("  ({note})");
        }
        println!();

        if args.members && hit.level > 0 {
            let members = index.cluster_members(hit).unwrap_or_default();
            // Group by file: a caller wants to know which parts of the corpus
            // this one result stands for, not how the segmenter cut them up.
            let mut by_file: std::collections::BTreeMap<&str, usize> =
                std::collections::BTreeMap::new();
            for member in &members {
                *by_file.entry(member.resource.as_str()).or_default() += 1;
            }
            println!(
                "    {} span(s) across {} file(s)",
                members.len(),
                by_file.len()
            );
            for (resource, spans) in &by_file {
                println!("    {}  ({spans})", terminal_safe(resource));
            }
            println!();
        }

        if let (Some(limit), 0) = (args.snippet, hit.level) {
            if let Ok((text, first_line)) = index.snippet(hit) {
                // Without the lines of the hit, its best segment is all that
                // is known of it.
                let segment = (
                    first_line,
                    first_line + text.lines().count().saturating_sub(1),
                );
                for line in snippet_lines(&text, first_line, limit, extent.unwrap_or(segment)) {
                    println!("{line}");
                }
                println!();
            }
        }
    }

    report_staleness(&args);
    Ok(ExitCode::SUCCESS)
}

/// A single self-overwriting progress line on stderr.
///
/// `-u` is an explicit request, so the cost is already accepted — but accepted
/// is not the same as visible. With a dense or LLM-backed dimension an update
/// can run for a while and spend money, and the person who asked for it should
/// be able to see what it is doing and press Ctrl-C if they change their mind.
///
/// Silent unless stderr is a terminal: an agent gets no carriage-return
/// animation in its captured output, and a redirected stream gets no control
/// characters.
struct ProgressLine {
    enabled: bool,
    last_drawn: std::time::Instant,
    width: usize,
}

impl ProgressLine {
    fn new(quiet: bool) -> Self {
        use std::io::IsTerminal;
        ProgressLine {
            enabled: !quiet && std::io::stderr().is_terminal(),
            // Draw the first tick immediately rather than after the interval.
            last_drawn: std::time::Instant::now() - std::time::Duration::from_secs(1),
            width: 0,
        }
    }

    fn tick(&mut self, progress: &map_index::Progress<'_>) {
        if !self.enabled {
            return;
        }
        // Throttle: redrawing per resource would make the terminal, not the
        // indexer, the bottleneck on a large corpus.
        if self.last_drawn.elapsed() < std::time::Duration::from_millis(80) {
            return;
        }
        self.last_drawn = std::time::Instant::now();

        // Keep the tail of the path — the distinguishing part is the end.
        let resource = progress.resource;
        let shown = if resource.len() > 40 {
            format!("…{}", &resource[resource.len() - 39..])
        } else {
            resource.to_owned()
        };

        let line = format!(
            "  updating {:>3}%  {}/{}  {} classified  {}",
            (progress.fraction() * 100.0) as u32,
            progress.done,
            progress.total,
            progress.classifier_calls,
            shown,
        );
        eprint!("\r{line:<width$}", width = self.width.max(line.len()));
        self.width = line.len();
    }

    /// Erase the line so it does not collide with results.
    fn finish(&mut self) {
        if self.enabled && self.width > 0 {
            eprint!("\r{:width$}\r", "", width = self.width);
        }
    }
}

/// Tell a human their index has drifted — and only a human.
///
/// The notice is an affordance for someone at a terminal who might not realize
/// they need `-u`. It is noise in a model's context window: it does not help
/// answer the question and costs tokens on every call. So it is gated on
/// stderr being a terminal, which an agent invoking `map` through a shell
/// never is.
///
/// The gate also buys the agent path its latency back. Detecting drift costs a
/// directory walk and a `stat` per resource (~22 ms on ripgrep); when nobody
/// will read the answer, the question is not asked at all.
fn report_staleness(args: &FindArgs) {
    use std::io::IsTerminal;

    if args.update || args.quiet || !std::io::stderr().is_terminal() {
        return;
    }
    // Best-effort: failing to compute drift must never fail a query.
    if let Ok(Some(report)) = map_index::stale(&args.path) {
        if report.is_stale() {
            eprintln!("map: index is stale ({report}) — run with -u to update before searching");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        display_path, driver_program, merge_manifests, newer_index_hint, parse_dim, parse_levels,
        repository_at_or_above, should_register, snippet_lines, spread_note,
    };
    use map_format::{Manifest, ObjectEntry, ObjectKey, Tier};
    use map_query::LevelFilter;
    use std::path::{Path, PathBuf};

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            use std::sync::atomic::{AtomicU32, Ordering};
            static N: AtomicU32 = AtomicU32::new(0);
            let n = N.fetch_add(1, Ordering::Relaxed);
            let mut p = std::env::temp_dir();
            p.push(format!("map-cli-test-{tag}-{}-{n}", std::process::id()));
            let _ = std::fs::remove_dir_all(&p);
            std::fs::create_dir_all(&p).unwrap();
            TempDir(p)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn a_canonicalized_root_prints_without_the_extended_length_prefix() {
        // What `canonicalize` returns on Windows. Correct, and not a path
        // anyone can paste back into a shell.
        assert_eq!(
            display_path(Path::new(r"\\?\C:\repo\thing")),
            r"C:\repo\thing"
        );
        assert_eq!(
            display_path(Path::new(r"\\?\UNC\server\share\repo")),
            r"\\server\share\repo"
        );
        assert_eq!(
            display_path(Path::new("/home/user/repo")),
            "/home/user/repo"
        );
    }

    #[test]
    fn an_already_registered_driver_is_not_rewritten() {
        // Re-registering on every `map index` is only acceptable because the
        // already-set case is a config read and a silent return.
        assert!(!should_register(
            true,
            Some("C:/bin/map.exe merge %O %A %B"),
            false
        ));
        // Git absent is not a reason to fail anything, only a reason to stop.
        assert!(!should_register(false, None, false));
        // Unset, and a key present but empty, both need writing.
        assert!(should_register(true, None, false));
        assert!(should_register(true, Some("  \n"), false));
    }

    #[test]
    fn a_driver_whose_binary_is_gone_is_registered_again() {
        // The registered command holds an absolute path. After `map` moves —
        // a new install root, a cleaned build directory — git cannot run it,
        // and without this every later manifest merge fails.
        let stale = Some("\"C:/old/place/map.exe\" merge %O %A %B");
        assert!(should_register(true, stale, true));
        assert!(!should_register(true, stale, false));
        // Still never without git.
        assert!(!should_register(false, stale, true));
    }

    #[test]
    fn the_program_of_a_driver_command_is_its_first_word_without_quotes() {
        assert_eq!(
            driver_program("\"C:/Program Files/map/map.exe\" merge %O %A %B"),
            Some(PathBuf::from("C:/Program Files/map/map.exe"))
        );
        assert_eq!(
            driver_program("/usr/local/bin/map merge %O %A %B\n"),
            Some(PathBuf::from("/usr/local/bin/map"))
        );
        assert_eq!(driver_program("   "), None);
    }

    /// The smallest config that parses, at a chosen format version.
    fn minimal_config(version: u32) -> String {
        format!(
            "version = {version}\n[dimensions.lexical]\ndescription = \"x\"\n\
             classifier = {{ impl = \"structural\" }}\n"
        )
    }

    #[test]
    fn the_minimal_test_config_is_valid_at_the_supported_version() {
        // Otherwise the two hint tests could pass on an unrelated parse error.
        map_format::Config::parse(&minimal_config(1)).unwrap();
    }

    #[test]
    fn an_entry_says_what_else_it_stands_for() {
        // One segment and no other match in the file: the line says it all.
        assert_eq!(spread_note(1, 0, &[10, 10, 49]), None);
        // A run of merged segments gives its extent.
        assert_eq!(
            spread_note(3, 0, &[833, 801, 936]).as_deref(),
            Some("lines 801-936")
        );
        // Matches that `--per-file` left out are named, and the rest counted,
        // so that the limit hides nothing.
        assert_eq!(
            spread_note(1, 5, &[833, 833, 872, 161, 65, 737]).as_deref(),
            Some("also 161, 65, 737, +2")
        );
        assert_eq!(
            spread_note(2, 1, &[5, 1, 72, 400]).as_deref(),
            Some("lines 1-72; also 400")
        );
    }

    #[test]
    fn a_cut_snippet_says_so_and_tells_the_reader_to_read_the_cut_lines() {
        // A segment of 25 lines that starts at line 321. The reader gets 12 of
        // them, and must be told that 13 more are there.
        let text: String = (0..25).map(|i| format!("line {i}\n")).collect();
        let shown = snippet_lines(&text, 321, 12, (321, 345));

        assert_eq!(shown.len(), 13, "12 lines and the notice");
        assert_eq!(shown[0], "    321  line 0");
        assert_eq!(shown[11], "    332  line 11");
        assert_eq!(
            shown[12],
            "    [TRUNCATED: lines 333-345 are after this snippet. The match can be in the cut lines: \
             read lines 321-345 before you search again.]"
        );
    }

    #[test]
    fn a_merged_hit_says_what_is_cut_before_its_snippet_and_after_it() {
        // Three segments are one hit, lines 385 to 488, and the best of them
        // starts at line 449. Lines of the hit are cut on the two sides.
        let text: String = (0..40).map(|i| format!("line {i}\n")).collect();
        let shown = snippet_lines(&text, 449, 12, (385, 488));

        assert_eq!(shown.len(), 14, "a notice, 12 lines, and a notice");
        assert_eq!(
            shown[0],
            "    [TRUNCATED: lines 385-448 are before this snippet]"
        );
        assert_eq!(shown[1], "    449  line 0");
        assert_eq!(shown[12], "    460  line 11");
        assert_eq!(
            shown[13],
            "    [TRUNCATED: lines 461-488 are after this snippet. The match can be in the cut lines: \
             read lines 385-488 before you search again.]"
        );

        // The best segment is the last one and all of it is shown. The one
        // notice is then before the snippet, and it carries the advice.
        let last = snippet_lines(&text, 449, 40, (385, 488));
        assert_eq!(last.len(), 41);
        assert_eq!(
            last[0],
            "    [TRUNCATED: lines 385-448 are before this snippet. The match can be in the cut lines: \
             read lines 385-488 before you search again.]"
        );
        assert!(!last[40].contains("TRUNCATED"));
    }

    #[test]
    fn a_hit_that_fits_in_the_snippet_has_no_truncation_notice() {
        // A notice on a whole hit would send the reader to lines that it
        // already has.
        let text: String = (0..12).map(|i| format!("line {i}\n")).collect();
        let shown = snippet_lines(&text, 97, 12, (97, 108));

        assert_eq!(shown.len(), 12);
        assert_eq!(shown[0], "     97  line 0");
        assert_eq!(shown[11], "    108  line 11");
        assert!(!shown.iter().any(|line| line.contains("TRUNCATED")));
        assert!(snippet_lines("", 1, 12, (1, 0)).is_empty());
    }

    #[test]
    fn the_number_after_the_snippet_flag_sets_the_lines_of_each_hit() {
        let text: String = (0..25).map(|i| format!("line {i}\n")).collect();
        let shown = snippet_lines(&text, 321, 5, (321, 345));

        assert_eq!(shown.len(), 6, "5 lines and the notice");
        assert_eq!(shown[4], "    325  line 4");
        assert!(shown[5].starts_with("    [TRUNCATED: lines 326-345 are after this snippet."));
        // A number as large as the segment gives all of it, with no notice.
        assert_eq!(snippet_lines(&text, 321, 40, (321, 345)).len(), 25);
    }

    #[test]
    fn a_snippet_flag_with_no_number_gives_12_lines_and_does_not_take_the_query() {
        use clap::Parser;
        let find = |args: &[&str]| match super::Cli::try_parse_from(args) {
            Ok(super::Cli {
                command: super::Command::Find(find),
            }) => find,
            _ => panic!("{args:?} is not a `find` command"),
        };

        let bare = find(&["map", "find", "--snippet", "the query"]);
        assert_eq!(bare.snippet, Some(12));
        assert_eq!(bare.lexical.as_deref(), Some("the query"));

        assert_eq!(find(&["map", "find", "--snippet=5", "q"]).snippet, Some(5));
        assert_eq!(find(&["map", "find", "q"]).snippet, None);
    }

    #[test]
    fn an_index_from_a_newer_map_gets_the_upgrade_hint() {
        // Built from the real errors, so rewording one of them fails here
        // instead of quietly dropping the hint.
        let newer = map_format::Config::parse(&minimal_config(2))
            .unwrap_err()
            .to_string();
        assert!(
            newer_index_hint(&newer).is_some_and(|h| h.starts_with("a newer version")),
            "{newer}"
        );

        let unknown_key =
            map_format::Config::parse(&format!("{}weight = 2.0\n", minimal_config(1)))
                .unwrap_err()
                .to_string();
        assert!(
            newer_index_hint(&unknown_key).is_some_and(|h| h.starts_with("if a newer version")),
            "{unknown_key}"
        );
    }

    #[test]
    fn the_upgrade_hint_reads_the_versions_that_follow_index_uses() {
        // A caller can put text in front of the error, and that text can hold
        // the word "version". The two numbers still come from the error.
        let newer = map_format::Config::parse(&minimal_config(2))
            .unwrap_err()
            .to_string();
        let wrapped = format!("map version 0.0.1 cannot open the index: {newer}");
        assert!(
            newer_index_hint(&wrapped).is_some_and(|h| h.starts_with("a newer version")),
            "{wrapped}"
        );
    }

    #[test]
    fn an_older_index_and_an_ordinary_error_get_no_upgrade_hint() {
        // Upgrading cannot fix an index that is older than the binary, and
        // most errors have nothing to do with versions at all.
        let older = map_format::Config::parse(&minimal_config(0))
            .unwrap_err()
            .to_string();
        assert_eq!(newer_index_hint(&older), None, "{older}");
        assert_eq!(
            newer_index_hint("no .map directory found at or above . — run `map init` first"),
            None
        );
        assert_eq!(
            newer_index_hint("unknown dimension \"nosuch\"; this index has: lexical"),
            None
        );
    }

    #[test]
    fn the_repository_is_found_at_or_above_the_index_root() {
        let dir = TempDir::new("repo");
        let nested = dir.0.join("a/b");
        std::fs::create_dir_all(&nested).unwrap();
        assert_eq!(repository_at_or_above(&nested), None, "no .git anywhere");

        // A worktree or submodule has `.git` as a file, not a directory.
        std::fs::write(dir.0.join(".git"), "gitdir: elsewhere\n").unwrap();
        assert_eq!(repository_at_or_above(&nested), Some(dir.0.clone()));
    }

    /// A manifest holding one resource root and the object it points at, so
    /// two of them built with different names are disjoint.
    fn manifest_with(resource: &str, seed: &[u8]) -> Manifest {
        let mut manifest = Manifest::new(&map_format::Config::zero_config(), "test", 0).unwrap();
        let key = ObjectKey::derive(&[seed], manifest.config_fingerprint);
        manifest
            .insert_object(
                key,
                ObjectEntry {
                    content_hash: map_format::ContentHash::of(seed),
                    len: seed.len() as u64,
                    tier: Tier::A,
                },
            )
            .unwrap();
        manifest.roots.insert(
            resource.to_owned(),
            map_format::manifest::ResourceRoot {
                segments: key,
                descriptors: Default::default(),
                tensors: Default::default(),
            },
        );
        manifest
    }

    /// Write base/ours/theirs into a scratch directory the way git does.
    fn merge_fixture(
        tag: &str,
        base: Option<&Manifest>,
        ours: &Manifest,
        theirs: &Manifest,
    ) -> (TempDir, PathBuf, PathBuf, PathBuf) {
        let dir = TempDir::new(tag);
        let write = |name: &str, bytes: &[u8]| {
            let path = dir.0.join(name);
            std::fs::write(&path, bytes).unwrap();
            path
        };
        // Git passes an empty %O when the file was added on both sides.
        let base_path = write(
            "base",
            &base.map(|m| m.to_bytes().unwrap()).unwrap_or_default(),
        );
        let ours_path = write("ours", &ours.to_bytes().unwrap());
        let theirs_path = write("theirs", &theirs.to_bytes().unwrap());
        (dir, base_path, ours_path, theirs_path)
    }

    #[test]
    fn a_disjoint_merge_keeps_both_sides_work() {
        // The measured case: two branches each edit a different file and
        // re-index, so each manifest gains a root and an object the other
        // never saw. Losing either half would silently unindex a resource.
        let base = manifest_with("shared.rs", b"shared");
        let mut ours = base.clone();
        let mut theirs = base.clone();
        for (side, resource, seed) in [
            (&mut ours, "ours.rs", b"ours".as_slice()),
            (&mut theirs, "theirs.rs", b"theirs".as_slice()),
        ] {
            let other = manifest_with(resource, seed);
            side.roots.extend(other.roots);
            side.objects.extend(other.objects);
        }

        let (_dir, base_path, ours_path, theirs_path) =
            merge_fixture("disjoint", Some(&base), &ours, &theirs);
        assert!(merge_manifests(&base_path, &ours_path, &theirs_path).unwrap());

        let merged = Manifest::from_bytes(&std::fs::read(&ours_path).unwrap()).unwrap();
        for resource in ["shared.rs", "ours.rs", "theirs.rs"] {
            assert!(merged.roots.contains_key(resource), "lost {resource}");
        }
        assert_eq!(merged.objects.len(), 3, "every side's objects survive");
    }

    #[test]
    fn a_conflicting_merge_leaves_our_file_untouched() {
        // Git reads the result out of `ours`, so writing a partial resolution
        // there would hand the operator a manifest neither side ever built.
        let base = manifest_with("a.rs", b"base");
        let ours = manifest_with("a.rs", b"ours");
        let theirs = manifest_with("a.rs", b"theirs");

        let (_dir, base_path, ours_path, theirs_path) =
            merge_fixture("conflict", Some(&base), &ours, &theirs);
        let before = std::fs::read(&ours_path).unwrap();

        assert!(!merge_manifests(&base_path, &ours_path, &theirs_path).unwrap());
        assert_eq!(
            std::fs::read(&ours_path).unwrap(),
            before,
            "a conflicted merge must not rewrite our side"
        );
    }

    #[test]
    fn a_side_that_is_not_a_manifest_is_reported_not_merged() {
        let manifest = manifest_with("a.rs", b"a");
        let (dir, base_path, ours_path, theirs_path) =
            merge_fixture("garbage", None, &manifest, &manifest);
        std::fs::write(&theirs_path, b"not json at all").unwrap();
        let before = std::fs::read(&ours_path).unwrap();

        let err = merge_manifests(&base_path, &ours_path, &theirs_path).unwrap_err();
        assert!(err.contains("their side"), "must name the side: {err}");
        assert_eq!(std::fs::read(&ours_path).unwrap(), before);
        drop(dir);
    }

    fn only(levels: &[u16]) -> LevelFilter {
        LevelFilter::Only(levels.iter().copied().collect())
    }

    #[test]
    fn a_bare_query_searches_every_level() {
        // Segments and cluster overviews in one ranking is the default the CLI
        // promises; anything narrower has to be asked for.
        assert_eq!(parse_levels("all").unwrap(), LevelFilter::All);
    }

    #[test]
    fn a_single_height_is_just_that_height() {
        assert_eq!(parse_levels("0").unwrap(), only(&[0]));
        assert_eq!(parse_levels("2").unwrap(), only(&[2]));
    }

    #[test]
    fn a_comma_separated_list_selects_exactly_those_levels() {
        assert_eq!(parse_levels("1,3,4").unwrap(), only(&[1, 3, 4]));
        // Order and repetition in the request must not change the scope.
        assert_eq!(parse_levels("4, 1 ,3,1").unwrap(), only(&[1, 3, 4]));
    }

    #[test]
    fn the_names_still_work_and_are_not_number_sugar() {
        // `clusters` cannot be written as a list: how tall the fabric grows is
        // a property of the corpus, not of the request.
        assert_eq!(parse_levels("segments").unwrap(), LevelFilter::Segments);
        assert_eq!(parse_levels("clusters").unwrap(), LevelFilter::Clusters);
        assert_ne!(parse_levels("clusters").unwrap(), only(&[1]));
    }

    #[test]
    fn malformed_level_specs_are_rejected() {
        assert!(parse_levels("1,,3").is_err(), "empty entry");
        assert!(parse_levels("1,").is_err(), "trailing comma");
        assert!(parse_levels("").is_err(), "empty spec");
        assert!(
            parse_levels("segments,1").is_err(),
            "name mixed into a list"
        );
        assert!(parse_levels("99999999").is_err(), "past u16");
        // A hyphen opens a range, so these are malformed ranges rather than
        // negative numbers — but they must still be refused.
        assert!(parse_levels("-1").is_err(), "no lower bound");
        assert!(parse_levels("3-").is_err(), "no upper bound");
        assert!(parse_levels("1-2-3").is_err(), "two hyphens");
        assert!(parse_levels("4-3").is_err(), "counts downward");
    }

    #[test]
    fn a_range_is_inclusive_at_both_ends() {
        assert_eq!(parse_levels("0-2").unwrap(), only(&[0, 1, 2]));
        assert_eq!(parse_levels("3-4").unwrap(), only(&[3, 4]));
        // A one-element range is the same scope as naming the height.
        assert_eq!(parse_levels("2-2").unwrap(), parse_levels("2").unwrap());
    }

    #[test]
    fn ranges_and_single_heights_compose() {
        assert_eq!(parse_levels("0,2-4").unwrap(), only(&[0, 2, 3, 4]));
        // Overlap is a set union, not a duplicate.
        assert_eq!(parse_levels("1-3,2-4").unwrap(), only(&[1, 2, 3, 4]));
    }

    #[test]
    fn a_reversed_range_says_how_to_fix_it() {
        let err = parse_levels("4-3").unwrap_err();
        assert!(
            err.contains("3-4"),
            "should suggest the corrected range: {err}"
        );
    }

    #[test]
    fn a_dim_without_a_weight_defaults_to_one() {
        let (name, term) = parse_dim("lexical=refresh_token").unwrap();
        assert_eq!(name, "lexical");
        assert_eq!(term.text, "refresh_token");
        assert_eq!(term.weight, 1.0);
    }

    #[test]
    fn a_weight_prefix_is_parsed_off_the_name() {
        let (name, term) = parse_dim("lexical:2=refresh_token").unwrap();
        assert_eq!(name, "lexical");
        assert_eq!(term.weight, 2.0);
        assert_eq!(term.text, "refresh_token");
    }

    #[test]
    fn query_text_keeps_its_own_equals_and_colons() {
        // Only the first '=' splits the arg, and a ':' past it is plain text —
        // the weight is a prefix of the *name*, never of the query text.
        let (name, term) = parse_dim("descriptive=a=b: c").unwrap();
        assert_eq!(name, "descriptive");
        assert_eq!(term.text, "a=b: c");
        assert_eq!(term.weight, 1.0);
    }

    #[test]
    fn zero_is_an_allowed_weight() {
        let (_, term) = parse_dim("lexical:0=alpha").unwrap();
        assert_eq!(term.weight, 0.0);
    }

    #[test]
    fn malformed_specs_are_rejected() {
        assert!(parse_dim("lexical:heavy=x").is_err(), "non-numeric weight");
        assert!(parse_dim("lexical:-1=x").is_err(), "negative weight");
        assert!(parse_dim("lexical:nan=x").is_err(), "NaN weight");
        assert!(parse_dim("lexical:inf=x").is_err(), "infinite weight");
        assert!(parse_dim("lexical:1e30=x").is_err(), "weight past the cap");
        assert!(parse_dim("=x").is_err(), "empty dimension name");
        assert!(parse_dim("lexical").is_err(), "no '=' at all");
    }
}
