//! `sys1d`: a local HTTP server for the sys1rust runtime that speaks upstream Laya's
//! `/v1/systemone` protocol (see `laya_serve.py` and `docs/http-api.md` upstream).
//!
//! - [`config`]: CLI flags with env-var fallbacks, and resolving the served checkpoint.
//! - [`validate`]: upstream's request checks (400/413/422 and the `model` field rule).

pub mod config;
pub mod validate;

pub use config::{Config, ServedModel};

/// Write one human-readable log line to stderr (stdout is reserved for the ready line).
pub fn log(msg: impl AsRef<str>) {
    eprintln!("sys1d: {}", msg.as_ref());
}

/// Fold line breaks in client-controlled text so it cannot forge log entries.
pub fn sanitize(s: &str) -> String {
    s.replace('\r', "\\r").replace('\n', "\\n")
}
