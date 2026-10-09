//! Refusing to build a quietly-wrong index.
//!
//! A dimension names an implementation per stage, and a build resolves what it
//! can: the `llm` classifier needs its feature *and* a configured endpoint, the
//! `distilled` embedder needs its feature *and* the model on disk.
//!
//! Skipping an unresolved stage silently produces an index that builds,
//! queries, and looks right while missing a dimension its own config
//! describes. It is worse for a semantic dimension: the embedder falls back to
//! the raw segment text, so the dimension keeps producing tensors that no
//! longer mean what its description claims. Nothing downstream can tell — the
//! pack loads, the scores are plausible, and the only symptom is worse answers.
//!
//! So an unresolved stage is reported and the operator decides. Consent has to
//! be explicit, and on a pipe there is nobody to give it — see [`Degraded`].

use map_format::Config;

/// A stage some dimension names that this build cannot run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnresolvedStage {
    pub dimension: String,
    /// `classifier`, `embedder`, or `fabricator`.
    pub stage: &'static str,
    /// The implementation named in `config.toml`.
    pub implementation: String,
    /// What the operator can do about it.
    pub remedy: String,
}

impl std::fmt::Display for UnresolvedStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}.{} = {:?} — {}",
            self.dimension, self.stage, self.implementation, self.remedy
        )
    }
}

/// What to do when a configured stage cannot be resolved.
///
/// [`Abort`](Degraded::Abort) is the default for every entry point, including
/// the library ones. A partial index is a legitimate thing to want — it is how
/// you configure a semantic dimension before its endpoint is live — but it has
/// to be asked for, because the failure is invisible after the fact.
pub enum Degraded<'a> {
    /// Refuse to build. What a non-interactive run gets, since a pipe cannot
    /// consent.
    Abort,
    /// Build the dimensions that do resolve and omit the rest.
    Allow,
    /// Ask, and build only if the answer is yes. Called at most once, after the
    /// stages resolve and before any resource is read.
    Ask(&'a mut dyn FnMut(&[UnresolvedStage]) -> bool),
}

/// Every configured stage this build cannot run, in dimension order.
///
/// Takes what actually resolved rather than re-deriving it, so this cannot
/// drift from the resolution the run is about to use.
pub(crate) fn unresolved(
    config: &Config,
    classifiers: &[&str],
    embedder: Option<&str>,
    can_fabricate: bool,
) -> Vec<UnresolvedStage> {
    let mut out = Vec::new();
    for (name, dimension) in config.active() {
        if let Some(stage) = &dimension.classifier {
            if !classifiers.contains(&stage.implementation.as_str()) {
                out.push(UnresolvedStage {
                    dimension: name.clone(),
                    stage: "classifier",
                    implementation: stage.implementation.clone(),
                    remedy: classifier_remedy(&stage.implementation),
                });
            }
        }
        if let Some(stage) = &dimension.embedder {
            if embedder != Some(stage.implementation.as_str()) {
                out.push(UnresolvedStage {
                    dimension: name.clone(),
                    stage: "embedder",
                    implementation: stage.implementation.clone(),
                    remedy: embedder_remedy(&stage.implementation),
                });
            }
        }
        if let Some(stage) = &dimension.fabricator {
            if !can_fabricate {
                out.push(UnresolvedStage {
                    dimension: name.clone(),
                    stage: "fabricator",
                    implementation: stage.implementation.clone(),
                    remedy: fabricator_remedy(),
                });
            }
        }
    }
    out
}

/// Separate "this build lacks the plugin" from "the plugin is not set up",
/// because the fix is completely different and the operator cannot tell from
/// the outside which one they hit.
fn classifier_remedy(implementation: &str) -> String {
    match implementation {
        "structural" | "content" | "declaration" => {
            format!("the {implementation} classifier is always available; this should not happen")
        }
        "llm" if cfg!(feature = "llm") => "no endpoint configured — run `map llm login`".to_owned(),
        "llm" => "not compiled in — build with --features llm".to_owned(),
        other => format!(
            "no classifier named {other:?} in this build \
             (known: structural, content, declaration, llm)"
        ),
    }
}

/// Fabrication needs an embedder to cluster and a labeler to summarize, so it
/// fails whenever either does. Telling someone to rebuild with features they
/// already have would send them the wrong way — when the plugins are compiled,
/// the cause is always another entry in this same list.
fn fabricator_remedy() -> String {
    if cfg!(feature = "distilled") {
        "needs a working embedder — fix the other stages listed here".to_owned()
    } else {
        "not compiled in — build with --features distilled".to_owned()
    }
}

fn embedder_remedy(implementation: &str) -> String {
    match implementation {
        // The remedy names `map model fetch` only where that command exists.
        // A `distilled` build without `auto-distilled` cannot download, so
        // pointing it at a command it does not have would send someone in a
        // circle.
        "distilled" if cfg!(feature = "auto-distilled") => {
            "model not installed — run `map model fetch`".to_owned()
        }
        "distilled" if cfg!(feature = "distilled") => {
            "model not installed — place it at ~/.map/models/potion-retrieval-32M".to_owned()
        }
        "distilled" => "not compiled in — build with --features distilled".to_owned(),
        other => format!("no embedder named {other:?} in this build (known: distilled)"),
    }
}

/// The error text, and the prompt body, share one wording.
pub fn describe(stages: &[UnresolvedStage]) -> String {
    let mut out = String::new();
    for stage in stages {
        out.push_str("  ");
        out.push_str(&stage.to_string());
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(toml: &str) -> Config {
        Config::parse(toml).unwrap()
    }

    const SEMANTIC: &str = r#"
version = 1
[dimensions.lexical]
description = "x"
classifier = { impl = "structural" }
[dimensions.descriptive]
description = "y"
classifier = { impl = "llm", prompts = { "0" = "p" } }
embedder = { impl = "distilled" }
"#;

    #[test]
    fn a_fully_resolved_config_reports_nothing() {
        let found = unresolved(
            &config(SEMANTIC),
            &["structural", "llm"],
            Some("distilled"),
            true,
        );
        assert!(found.is_empty(), "{found:?}");
    }

    #[test]
    fn a_classifier_naming_a_missing_implementation_is_reported() {
        // The exact shape of the chat-completions rename: config names a stage
        // this build does not have, and the dimension would otherwise be built
        // from raw text under a description promising LLM prose.
        let renamed = SEMANTIC.replace(r#"impl = "llm""#, r#"impl = "chat-completions""#);
        let found = unresolved(
            &config(&renamed),
            &["structural", "llm"],
            Some("distilled"),
            true,
        );
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].dimension, "descriptive");
        assert_eq!(found[0].stage, "classifier");
        assert!(found[0].remedy.contains("no classifier named"));
    }

    #[test]
    fn an_absent_embedder_is_reported_even_when_the_classifier_resolves() {
        // The missing-model case: the feature may be compiled and the endpoint
        // fine, and the dense dimension still silently produces nothing.
        let found = unresolved(&config(SEMANTIC), &["structural", "llm"], None, false);
        let stages: Vec<&str> = found.iter().map(|s| s.stage).collect();
        assert_eq!(stages, vec!["embedder"]);
    }

    #[test]
    fn a_zero_config_index_is_never_degraded() {
        // The whole first-run promise: no key, no network, no model, no prompt.
        let found = unresolved(
            &Config::zero_config(),
            &["structural", "declaration"],
            None,
            false,
        );
        assert!(found.is_empty(), "{found:?}");
    }

    #[test]
    fn a_disabled_dimension_is_not_reported() {
        let disabled = SEMANTIC.replace(
            r#"[dimensions.descriptive]
description = "y""#,
            r#"[dimensions.descriptive]
description = "y"
enabled = false"#,
        );
        let found = unresolved(&config(&disabled), &["structural"], None, false);
        assert!(found.is_empty(), "{found:?}");
    }

    #[test]
    fn a_fabricator_without_its_stages_is_reported() {
        let with_fabricator = format!("{SEMANTIC}fabricator = {{ impl = \"agglomerative\" }}\n");
        let found = unresolved(
            &config(&with_fabricator),
            &["structural", "llm"],
            Some("distilled"),
            false,
        );
        let stages: Vec<&str> = found.iter().map(|s| s.stage).collect();
        assert_eq!(stages, vec!["fabricator"]);
    }
}
