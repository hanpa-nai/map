//! An orthogonal LLM adapter: connection management plus structured chat calls.
//!
//! This crate knows nothing about MAP's stages, records, or index. It owns the
//! *connection* to an OpenAI-compatible endpoint — endpoint, model, key — and
//! the one operation everything needs: a chat call pinned to a JSON schema.
//! The segment classifier and the cluster labeler share one adapter.
//!
//! # The connection is per-user, cached apart from any repo
//!
//! An endpoint and key are a property of *you*, not of a repository — the same
//! connection serves every index you build. So the adapter prompts for it once
//! and caches it **user-globally** (`~/.map/llm.toml`), deliberately separate
//! from any repository's `.map/`. Two consequences follow, and both are the
//! point:
//!
//! - a key never lives near a repo, so it cannot be committed by accident;
//! - configuring it once works across every index on the machine.
//!
//! Prompting needs a terminal. A non-interactive caller (an agent shelling out,
//! CI) with no cached connection gets [`LlmError::NotConfigured`] rather than a
//! hang — set it up once interactively and every later run just loads it.

use std::io::{self, IsTerminal, Write};
use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// Anything the adapter can fail with.
#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("io error: {0}")]
    Io(#[from] io::Error),
    #[error(
        "no LLM connection configured, and no terminal to prompt on — \
         run `map` interactively once to set the endpoint and key"
    )]
    NotConfigured,
    #[error("LLM request failed: {0}")]
    Request(String),
    #[error("unexpected LLM reply: {0}")]
    Reply(String),
}

/// The wire protocol an endpoint speaks.
///
/// A property of the connection, not of the index: two providers reached over
/// different protocols produce the same kind of descriptor, so this is
/// deliberately absent from the committed config and from the artifact
/// fingerprint. Moving a model between providers never re-keys an object.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Protocol {
    /// `POST {endpoint}/chat/completions` with a strict JSON schema. The
    /// default because vLLM, Ollama, llama.cpp, LM Studio, and most hosted
    /// providers all serve it, so a local endpoint needs no configuration.
    ///
    /// Spelled out rather than left to `rename_all`, which derives
    /// `open-ai-chat` from the variant name. This value is written to a user's
    /// `llm.toml`, so it must not move when a derive convention does.
    #[serde(rename = "openai-chat")]
    #[default]
    OpenAiChat,
}

impl std::fmt::Display for Protocol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Protocol::OpenAiChat => f.write_str("openai-chat"),
        }
    }
}

/// A cached connection to an LLM endpoint.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Connection {
    /// Base URL, e.g. `https://host/v1`.
    pub endpoint: String,
    /// Model id, e.g. `gpt-4o-mini`.
    pub model: String,
    /// Bearer token. `None` for a local endpoint that needs no auth.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// Omitted in a config written before this field existed, and omitted by
    /// anyone pointing at an OpenAI-compatible server — which is every local
    /// one.
    #[serde(default)]
    pub protocol: Protocol,
}

impl Connection {
    /// The user-global cache path, `~/.map/llm.toml`.
    pub fn cache_path() -> Option<PathBuf> {
        let home = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME"))?;
        Some(PathBuf::from(home).join(".map").join("llm.toml"))
    }

    /// Load the cached connection, or `None` if none is stored.
    pub fn load() -> Option<Self> {
        let text = std::fs::read_to_string(Self::cache_path()?).ok()?;
        toml::from_str(&text).ok()
    }

    /// Persist to the user-global cache, trimming a trailing slash so the
    /// endpoint composes cleanly with `/chat/completions`.
    pub fn save(&self) -> Result<(), LlmError> {
        let path = Self::cache_path().ok_or(LlmError::NotConfigured)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let normalized = Connection {
            endpoint: self.endpoint.trim_end_matches('/').to_owned(),
            ..self.clone()
        };
        let text = toml::to_string(&normalized).map_err(|e| LlmError::Io(io::Error::other(e)))?;
        std::fs::write(path, text)?;
        Ok(())
    }

    /// Prompt on the terminal for a connection and save it.
    ///
    /// Errors with [`LlmError::NotConfigured`] when there is no terminal, so a
    /// non-interactive caller fails fast instead of blocking on stdin.
    pub fn prompt_and_save() -> Result<Self, LlmError> {
        if !io::stdin().is_terminal() {
            return Err(LlmError::NotConfigured);
        }
        eprintln!("No LLM connection is configured. Setting one up (saved to ~/.map/llm.toml, never committed).");
        let endpoint = ask("Endpoint (OpenAI-compatible base URL, e.g. https://host/v1): ")?;
        let model = ask("Model (e.g. gpt-4o-mini): ")?;
        let key = ask("API key (leave blank for a local endpoint): ")?;
        let connection = Connection {
            endpoint,
            model,
            api_key: (!key.is_empty()).then_some(key),
            protocol: Protocol::default(),
        };
        connection.save()?;
        Ok(connection)
    }
}

fn ask(prompt: &str) -> Result<String, LlmError> {
    eprint!("{prompt}");
    io::stderr().flush()?;
    let mut line = String::new();
    io::stdin().read_line(&mut line)?;
    Ok(line.trim().to_owned())
}

/// The adapter: a connection plus the operations over it.
pub struct LlmAdapter {
    connection: Connection,
}

impl LlmAdapter {
    /// Wrap an explicit connection (used in tests and by callers that resolve
    /// the connection themselves).
    pub fn new(connection: Connection) -> Self {
        LlmAdapter { connection }
    }

    /// Load the cached connection, or prompt for it once and cache it.
    pub fn load_or_prompt() -> Result<Self, LlmError> {
        match Connection::load() {
            Some(connection) => Ok(LlmAdapter::new(connection)),
            None => Ok(LlmAdapter::new(Connection::prompt_and_save()?)),
        }
    }

    pub fn model(&self) -> &str {
        &self.connection.model
    }

    /// A chat call pinned to a JSON schema; returns the assistant's content.
    ///
    /// `schema` is the JSON Schema the reply must satisfy (strict), and
    /// `schema_name` names it for the endpoint. Structured output means the
    /// caller parses a guaranteed shape rather than repairing prose.
    ///
    /// Retries on a rate-limit (429), a transient 5xx, a transport failure, or
    /// a reply whose body could not be read, honoring a `Retry-After`
    /// header where the server sends one. A hosted endpoint under a per-minute
    /// quota returns 429 in bursts, and a full-corpus index makes hundreds of
    /// calls — without bounded retry it would die partway through. Retries are
    /// capped, so a persistent failure still surfaces rather than hanging.
    pub fn chat_json(
        &self,
        system: &str,
        user: &str,
        schema: Value,
        schema_name: &str,
    ) -> Result<String, LlmError> {
        const MAX_RETRIES: u32 = 6;

        let base = self.connection.endpoint.trim_end_matches('/');
        let (url, body) = match self.connection.protocol {
            Protocol::OpenAiChat => (
                format!("{base}/chat/completions"),
                build_body(&self.connection.model, system, user, schema, schema_name),
            ),
        };

        let mut attempt = 0u32;
        let value: Value = loop {
            // Capped exponential backoff for anything that gives no Retry-After.
            let backoff = Duration::from_secs(2u64.pow(attempt).min(60));

            // ureq consumes the request on send, so rebuild it each attempt.
            let mut request = ureq::post(&url).set("Content-Type", "application/json");
            if let Some(key) = &self.connection.api_key {
                request = request.set("Authorization", &format!("Bearer {key}"));
            }
            match request.send_json(&body) {
                // The body is read inside the loop because a connection can drop
                // after the status line as easily as before it. One reset at
                // minute thirty of a full-corpus index cost the whole run once.
                Ok(r) => match r.into_json() {
                    Ok(v) => break v,
                    Err(_) if attempt < MAX_RETRIES => {
                        std::thread::sleep(backoff);
                        attempt += 1;
                    }
                    Err(e) => return Err(LlmError::Reply(e.to_string())),
                },
                Err(ureq::Error::Status(code, resp))
                    if (code == 429 || code >= 500) && attempt < MAX_RETRIES =>
                {
                    std::thread::sleep(retry_after(&resp).unwrap_or(backoff));
                    attempt += 1;
                }
                // A transport failure — reset, timeout, DNS blip — is as
                // transient as a 5xx and bounded the same way.
                Err(ureq::Error::Transport(_)) if attempt < MAX_RETRIES => {
                    std::thread::sleep(backoff);
                    attempt += 1;
                }
                Err(e) => return Err(LlmError::Request(format!("{url}: {e}"))),
            }
        };

        extract_content(&value).ok_or_else(|| LlmError::Reply("no message content".into()))
    }
}

/// Parse a `Retry-After` header as delta-seconds, capped so a hostile or absurd
/// value cannot stall the process. Absent or unparseable yields `None`, and the
/// caller falls back to exponential backoff.
fn retry_after(response: &ureq::Response) -> Option<Duration> {
    let secs: u64 = response.header("retry-after")?.trim().parse().ok()?;
    Some(Duration::from_secs(secs.min(120)))
}

/// Assemble the chat-completions request body. Pure, so it is testable without
/// a live endpoint.
pub fn build_body(
    model: &str,
    system: &str,
    user: &str,
    schema: Value,
    schema_name: &str,
) -> Value {
    json!({
        "model": model,
        "temperature": 0,
        "messages": [
            { "role": "system", "content": system },
            { "role": "user", "content": user },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": { "name": schema_name, "strict": true, "schema": schema }
        }
    })
}

/// Pull the assistant message content out of a chat-completions reply.
pub fn extract_content(reply: &Value) -> Option<String> {
    reply
        .get("choices")?
        .get(0)?
        .get("message")?
        .get("content")?
        .as_str()
        .map(|s| s.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_pins_the_model_schema_and_messages() {
        let body = build_body(
            "gpt-4o-mini",
            "sys",
            "usr",
            json!({"type": "object"}),
            "descriptors",
        );
        assert_eq!(body["model"], "gpt-4o-mini");
        assert_eq!(body["temperature"], 0);
        assert_eq!(body["messages"][0]["content"], "sys");
        assert_eq!(body["messages"][1]["content"], "usr");
        assert_eq!(body["response_format"]["json_schema"]["strict"], true);
        assert_eq!(
            body["response_format"]["json_schema"]["name"],
            "descriptors"
        );
    }

    #[test]
    fn content_is_extracted_from_a_well_formed_reply() {
        let reply = json!({
            "choices": [ { "message": { "content": "{\"segments\":[]}" } } ]
        });
        assert_eq!(
            extract_content(&reply).as_deref(),
            Some("{\"segments\":[]}")
        );
        assert!(extract_content(&json!({"choices": []})).is_none());
        assert!(extract_content(&json!({})).is_none());
    }

    #[test]
    fn connection_round_trips_through_toml() {
        let conn = Connection {
            endpoint: "https://host/v1".into(),
            model: "gpt-4o-mini".into(),
            api_key: Some("k".into()),
            protocol: Protocol::OpenAiChat,
        };
        let text = toml::to_string(&conn).unwrap();
        let back: Connection = toml::from_str(&text).unwrap();
        assert_eq!(conn, back);
    }

    #[test]
    fn a_connection_without_a_protocol_speaks_openai_chat() {
        // What a local vLLM user writes, and what every llm.toml written before
        // the field existed contains. Omission must not be a parse error.
        let back: Connection = toml::from_str(
            "endpoint = \"http://localhost:8000/v1\"\nmodel = \"Qwen2.5-Coder-32B\"\n",
        )
        .unwrap();
        assert_eq!(back.protocol, Protocol::OpenAiChat);
        assert!(back.api_key.is_none());
    }

    #[test]
    fn protocol_writes_the_pinned_wire_value() {
        let text = toml::to_string(&Connection {
            endpoint: "http://localhost:8000/v1".into(),
            model: "m".into(),
            api_key: None,
            protocol: Protocol::OpenAiChat,
        })
        .unwrap();
        assert!(text.contains("openai-chat"), "got: {text}");
    }
}
