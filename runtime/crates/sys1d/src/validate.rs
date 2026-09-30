//! Upstream's request checks, in upstream's order and with its `detail` strings
//! (`laya_serve.py`: `_systemone_inner`, `_check_request_limits`, `_resolve_model`), plus the
//! two rules this spec adds: budget overrides are refused, and a `model` naming another
//! checkpoint is an error rather than a routing hint.

mod py_repr;

use crate::config::checkpoint_name;
use axum::http::StatusCode;
use serde_json::Value;

pub use py_repr::py_repr;

pub const MAX_QUESTIONS: usize = 64;
pub const MAX_STATE_CHARS: usize = 50_000;
pub const MAX_BODY_BYTES: usize = 2 * 1024 * 1024;
pub const MAX_CHOICE_OPTIONS: usize = 100;
pub const MAX_SCORE_LEVELS: usize = 32;
pub const MAX_TOTAL_OPTIONS: usize = 512;

/// A refused request: the status and the `detail` string of the error body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rejection {
    pub status: StatusCode,
    pub detail: String,
    /// `Retry-After` seconds, set only on the admission 503.
    pub retry_after: Option<u32>,
}

impl Rejection {
    pub fn new(status: StatusCode, detail: impl Into<String>) -> Self {
        Rejection {
            status,
            detail: detail.into(),
            retry_after: None,
        }
    }

    pub fn with_retry_after(mut self, seconds: u32) -> Self {
        self.retry_after = Some(seconds);
        self
    }
}

fn bad_request(detail: &str) -> Rejection {
    Rejection::new(StatusCode::BAD_REQUEST, detail)
}

fn too_large(detail: String) -> Rejection {
    Rejection::new(StatusCode::PAYLOAD_TOO_LARGE, detail)
}

/// The parts of a valid request handed to the inference thread.
#[derive(Debug)]
pub struct Validated {
    pub state: Value,
    pub questions: Value,
}

/// Parse and check a complete request body against `served_name`.
pub fn validate_body(raw: &[u8], served_name: &str) -> Result<Validated, Rejection> {
    let body: Value =
        serde_json::from_slice(raw).map_err(|_| bad_request("request body must be valid JSON"))?;
    let Value::Object(mut obj) = body else {
        return Err(bad_request(
            "request body must be an object with a 'questions' field",
        ));
    };
    if !obj.contains_key("questions") {
        return Err(bad_request(
            "request body must be an object with a 'questions' field",
        ));
    }
    let state = match obj.remove("state") {
        None | Some(Value::Null) => return Err(bad_request("'state' is required")),
        Some(s) => s,
    };
    let questions = obj.remove("questions").expect("checked above");
    check_limits(&state, &questions)?;
    for key in ["max_len", "head_max_len"] {
        if matches!(obj.get(key), Some(v) if !v.is_null()) {
            return Err(Rejection::new(
                StatusCode::UNPROCESSABLE_ENTITY,
                format!("{key} overrides are not supported by sys1d"),
            ));
        }
    }
    check_model(obj.get("model"), served_name)?;
    Ok(Validated { state, questions })
}

/// `_check_request_limits`: reject absent or oversized requests before tokenization.
pub fn check_limits(state: &Value, questions: &Value) -> Result<(), Rejection> {
    let Value::Object(qs) = questions else {
        return Err(bad_request("'questions' must be an object"));
    };
    if qs.len() > MAX_QUESTIONS {
        return Err(too_large(format!(
            "too many questions ({} > {MAX_QUESTIONS})",
            qs.len()
        )));
    }
    let mut total = 0usize;
    for (qid, q) in qs {
        let Value::Object(q) = q else { continue };
        let crit = q.get("criteria");
        match q.get("type").and_then(Value::as_str) {
            Some("choice") => {
                let count = match crit {
                    Some(Value::Object(m)) => m.len(),
                    Some(Value::Array(a)) => a.len(),
                    _ => continue,
                };
                total += count;
                if count > MAX_CHOICE_OPTIONS {
                    return Err(too_large(format!(
                        "too many choice options for {} ({count} > {MAX_CHOICE_OPTIONS})",
                        py_repr(qid)
                    )));
                }
            }
            Some("score") => {
                let Some(Value::Array(a)) = crit else {
                    continue;
                };
                total += a.len();
                if a.len() > MAX_SCORE_LEVELS {
                    return Err(too_large(format!(
                        "too many score levels for {} ({} > {MAX_SCORE_LEVELS})",
                        py_repr(qid),
                        a.len()
                    )));
                }
            }
            _ => {}
        }
    }
    if total > MAX_TOTAL_OPTIONS {
        return Err(too_large(format!(
            "too many answer options across questions ({total} > {MAX_TOTAL_OPTIONS})"
        )));
    }
    let n = state_chars(state);
    if n > MAX_STATE_CHARS {
        return Err(too_large(format!(
            "state too large ({n} > {MAX_STATE_CHARS} chars)"
        )));
    }
    Ok(())
}

/// Characters in the state as upstream counts them for a string (`len(str)`, i.e. Unicode
/// scalar values). Upstream measures a non-string state as `len(str(state))`, the Python
/// repr of the decoded object (`{'a': 1}`, `True`, `None`); sys1d measures its compact JSON
/// text (`{"a":1}`, `true`, `null`) instead, which differs by the spaces after `,` and `:`
/// and the literal spellings. Both are guards on the same order of magnitude, not a contract.
pub fn state_chars(state: &Value) -> usize {
    match state {
        Value::String(s) => s.chars().count(),
        other => serde_json::to_string(other)
            .map(|s| s.chars().count())
            .unwrap_or(MAX_STATE_CHARS + 1),
    }
}

/// The `model` field: absent, null or anything that is not a known checkpoint name or
/// published id means "the served checkpoint"; a known name for another checkpoint is 400.
pub fn check_model(model: Option<&Value>, served_name: &str) -> Result<(), Rejection> {
    let Some(Value::String(s)) = model else {
        return Ok(());
    };
    match checkpoint_name(s) {
        Some(name) if name != served_name => Err(bad_request(&format!(
            "model {} names checkpoint '{name}', but this server serves only '{served_name}'",
            py_repr(s.trim())
        ))),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The question id in a 413 detail is Python's `repr`, escapes included.
    #[test]
    fn option_limit_detail_uses_python_repr() {
        let opts: serde_json::Map<String, Value> =
            (0..101).map(|i| (format!("o{i}"), json!("x"))).collect();
        let body = json!({"state": "s", "questions": {"zero\u{200b}width": {"type": "choice", "criteria": opts}}});
        let e = validate_body(body.to_string().as_bytes(), "typed-decisions").unwrap_err();
        assert_eq!(e.status, StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(
            e.detail,
            "too many choice options for 'zero\\u200bwidth' (101 > 100)"
        );
    }

    #[test]
    fn state_chars_counts_scalars_or_compact_json() {
        assert_eq!(state_chars(&json!("héllo")), 5);
        assert_eq!(state_chars(&json!({"a": 1})), 7);
        assert_eq!(state_chars(&json!(true)), 4);
    }

    #[test]
    fn model_rule() {
        assert!(check_model(None, "typed-decisions").is_ok());
        assert!(check_model(Some(&Value::Null), "typed-decisions").is_ok());
        assert!(check_model(Some(&json!("jev-1")), "typed-decisions").is_ok());
        assert!(check_model(Some(&json!(7)), "typed-decisions").is_ok());
        assert!(check_model(Some(&json!(" Typed-Decisions ")), "typed-decisions").is_ok());
        let e = check_model(Some(&json!("english")), "typed-decisions").unwrap_err();
        assert_eq!(e.status, StatusCode::BAD_REQUEST);
        assert!(
            e.detail.contains("'english'") && e.detail.contains("'typed-decisions'"),
            "{}",
            e.detail
        );
    }

    #[test]
    fn validate_body_order() {
        let served = "typed-decisions";
        let ok = validate_body(
            br#"{"state":"s","questions":{"q":{"type":"noul","instructions":"?"}}}"#,
            served,
        )
        .unwrap();
        assert_eq!(ok.state, json!("s"));
        assert!(ok.questions.is_object());
        // Overrides are checked after the limits and before the model field.
        let e = validate_body(
            br#"{"state":"s","questions":{},"max_len":1,"model":"english"}"#,
            served,
        )
        .unwrap_err();
        assert_eq!(e.status, StatusCode::UNPROCESSABLE_ENTITY);
        let e = validate_body(
            br#"{"state":"s","questions":{},"max_len":null,"model":"english"}"#,
            served,
        )
        .unwrap_err();
        assert_eq!(e.status, StatusCode::BAD_REQUEST);
        let e = validate_body(br#"{"state":"s","questions":[],"max_len":1}"#, served).unwrap_err();
        assert_eq!(e.detail, "'questions' must be an object");
    }
}
