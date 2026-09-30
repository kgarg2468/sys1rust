//! Question schema validation, normalization and option rendering (`Agent._check_question`,
//! `Agent._to_internal`, `common.render_options`).

use crate::pyjson;
use crate::{Error, Result};
use serde_json::{Map, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QType {
    Choice,
    Score,
    Noul,
}

impl QType {
    pub fn index(self) -> usize {
        match self {
            QType::Choice => 0,
            QType::Score => 1,
            QType::Noul => 2,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            QType::Choice => "choice",
            QType::Score => "score",
            QType::Noul => "noul",
        }
    }
    pub fn from_index(i: usize) -> Option<Self> {
        match i {
            0 => Some(QType::Choice),
            1 => Some(QType::Score),
            2 => Some(QType::Noul),
            _ => None,
        }
    }
}

/// A validated, normalized question with its rendered option texts.
#[derive(Debug, Clone)]
pub struct Question {
    pub id: String,
    pub qtype: QType,
    /// Instruction text as the model sees it (non-string instructions are `json.dumps`ed).
    pub instructions: String,
    /// Option texts in label-index order; for noul always `[false, true]`.
    pub options: Vec<String>,
    /// Choice label keys in option order (empty for other types).
    pub choice_keys: Vec<String>,
    /// Score level descriptions as given (raw JSON values), for the `legend` output.
    pub score_legend: Vec<Value>,
}

fn is_blank(v: Option<&Value>) -> bool {
    matches!(v, None | Some(Value::Null)) || matches!(v, Some(Value::String(s)) if s.is_empty())
}

/// `render_criterion`: strings pass through, anything else is compact-ish JSON.
pub fn render_criterion(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => pyjson::dumps(other),
    }
}

fn resolve_noul_labels(labels: Option<&Value>) -> Result<(String, String)> {
    let bad = || {
        Error::Question(
            "noul labels must map exactly 'false' and 'true' to distinct non-empty strings".into(),
        )
    };
    let Some(labels) = labels else {
        return Ok(("false".into(), "true".into()));
    };
    let Value::Object(o) = labels else {
        return Err(bad());
    };
    if o.len() != 2 || !o.contains_key("false") || !o.contains_key("true") {
        return Err(bad());
    }
    let (Some(Value::String(f)), Some(Value::String(t))) = (o.get("false"), o.get("true")) else {
        return Err(bad());
    };
    let (f, t) = (f.trim().to_string(), t.trim().to_string());
    if f.is_empty() || t.is_empty() || f == t {
        return Err(bad());
    }
    Ok((f, t))
}

fn check(qid: &str, qdef: &Value) -> Result<()> {
    let q = |msg: String| Error::Question(format!("question '{qid}': {msg}"));
    let Value::Object(o) = qdef else {
        return Err(q(format!(
            "definition must be a dict, got {}",
            json_type_name(qdef)
        )));
    };
    let t = o.get("type").and_then(Value::as_str);
    let t = match t {
        Some("choice") => QType::Choice,
        Some("score") => QType::Score,
        Some("noul") => QType::Noul,
        _ => {
            return Err(q(format!(
                "unknown type {}; use one of ['choice', 'noul', 'score']",
                pyrepr(o.get("type"))
            )))
        }
    };
    if !o.contains_key("instructions") {
        return Err(q(
            "no 'instructions'; add the text the model should answer".into()
        ));
    }
    let crit = o.get("criteria");
    match t {
        QType::Choice => match crit {
            Some(Value::Object(m)) if !m.is_empty() => {}
            Some(Value::Array(a)) if !a.is_empty() => {}
            Some(Value::Object(_)) | Some(Value::Array(_)) => {
                return Err(q("a choice question needs at least one criterion".into()))
            }
            _ => return Err(q("a choice question takes 'criteria' as a dict of label -> description, or a list of labels".into())),
        },
        QType::Score => match crit {
            Some(Value::Array(a)) if !a.is_empty() => {}
            Some(Value::Array(_)) => return Err(q("a score question needs at least one level".into())),
            _ => return Err(q("a score question takes 'criteria' as a list of level descriptions, index 0 first".into())),
        },
        QType::Noul => {
            if !matches!(crit, None | Some(Value::Null) | Some(Value::Object(_))) {
                return Err(q("a noul question takes 'criteria' as a dict with optional 'true'/'false' descriptions, or omits it".into()));
            }
        }
    }
    if let Some(labels) = o.get("labels") {
        if t != QType::Noul {
            return Err(q("'labels' is only supported for noul questions".into()));
        }
        resolve_noul_labels(Some(labels)).map_err(|e| q(e.to_string()))?;
    }
    Ok(())
}

fn json_type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(_) => "int",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

fn pyrepr(v: Option<&Value>) -> String {
    match v {
        None | Some(Value::Null) => "None".into(),
        Some(Value::String(s)) => format!("'{s}'"),
        Some(other) => pyjson::py_str(other),
    }
}

fn to_internal(qid: &str, qdef: &Map<String, Value>) -> Result<Question> {
    let qtype = match qdef.get("type").and_then(Value::as_str) {
        Some("choice") => QType::Choice,
        Some("score") => QType::Score,
        _ => QType::Noul,
    };
    let instructions = match &qdef["instructions"] {
        Value::String(s) => s.clone(),
        other => pyjson::dumps(other),
    };
    let crit = qdef.get("criteria");
    let mut options = Vec::new();
    let mut choice_keys = Vec::new();
    let mut score_legend = Vec::new();
    match qtype {
        QType::Choice => {
            // dict: label -> description; list: labels only (dedup, first position wins, like a dict comprehension)
            let mut entries: Vec<(String, Value)> = Vec::new();
            match crit {
                Some(Value::Object(m)) => {
                    entries.extend(m.iter().map(|(k, v)| (k.clone(), v.clone())))
                }
                Some(Value::Array(a)) => {
                    for item in a {
                        let k = pyjson::py_str(item);
                        if !entries.iter().any(|(e, _)| *e == k) {
                            entries.push((k, Value::Null));
                        }
                    }
                }
                _ => unreachable!("validated"),
            }
            for (k, v) in entries {
                options.push(if is_blank(Some(&v)) {
                    k.clone()
                } else {
                    format!("{}: {}", k, render_criterion(&v))
                });
                choice_keys.push(k);
            }
        }
        QType::Score => {
            let Some(Value::Array(a)) = crit else {
                unreachable!("validated")
            };
            for (i, c) in a.iter().enumerate() {
                options.push(format!("level {}: {}", i, render_criterion(c)));
                score_legend.push(c.clone());
            }
        }
        QType::Noul => {
            let (false_label, true_label) = resolve_noul_labels(qdef.get("labels"))
                .map_err(|e| Error::Question(format!("question '{qid}': {e}")))?;
            // keys are normalized with str(k).lower()
            let mut norm: Map<String, Value> = Map::new();
            if let Some(Value::Object(m)) = crit {
                for (k, v) in m {
                    norm.insert(k.to_lowercase(), v.clone());
                }
            }
            let fc = norm.get("false");
            let tc = norm.get("true");
            options.push(format!(
                "{}: {}",
                false_label,
                if is_blank(fc) {
                    "no, the statement does not hold".to_string()
                } else {
                    render_criterion(fc.unwrap())
                }
            ));
            options.push(format!(
                "{}: {}",
                true_label,
                if is_blank(tc) {
                    "yes, the statement holds".to_string()
                } else {
                    render_criterion(tc.unwrap())
                }
            ));
        }
    }
    Ok(Question {
        id: qid.to_string(),
        qtype,
        instructions,
        options,
        choice_keys,
        score_legend,
    })
}

/// Validate and normalize a `{qid: definition}` object, preserving order.
pub fn parse_questions(questions: &Value) -> Result<Vec<Question>> {
    let Value::Object(map) = questions else {
        return Err(Error::Question(
            "questions must be a JSON object mapping question id -> definition".into(),
        ));
    };
    let mut out = Vec::with_capacity(map.len());
    for (qid, qdef) in map {
        check(qid, qdef)?;
        let Value::Object(o) = qdef else {
            unreachable!("validated")
        };
        out.push(to_internal(qid, o)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn renders_like_python() {
        let qs = parse_questions(&json!({
            "d": {"type": "choice", "instructions": "Which?", "criteria": {"billing": "invoices", "other": null, "zero": 0, "e": ""}},
            "u": {"type": "score", "instructions": "How?", "criteria": ["low", {"desc": "x"}]},
            "n": {"type": "noul", "instructions": {"a": 1}},
            "l": {"type": "noul", "instructions": "Money?", "criteria": {"True": "yes money", "FALSE": ""}, "labels": {"false": " B ", "true": "A"}},
            "c": {"type": "choice", "instructions": "x", "criteria": ["a", "b", "a"]},
        })).unwrap();
        assert_eq!(
            qs[0].options,
            ["billing: invoices", "other", "zero: 0", "e"]
        );
        assert_eq!(
            qs[1].options,
            ["level 0: low", "level 1: {\"desc\": \"x\"}"]
        );
        assert_eq!(qs[2].instructions, "{\"a\": 1}");
        assert_eq!(
            qs[2].options,
            [
                "false: no, the statement does not hold",
                "true: yes, the statement holds"
            ]
        );
        assert_eq!(
            qs[3].options,
            ["B: no, the statement does not hold", "A: yes money"]
        );
        assert_eq!(qs[4].choice_keys, ["a", "b"]);
    }

    #[test]
    fn rejects_malformed() {
        assert!(parse_questions(&json!({"q": {"type": "bool", "instructions": "x"}})).is_err());
        assert!(parse_questions(
            &json!({"q": {"type": "choice", "instructions": "x", "criteria": {}}})
        )
        .is_err());
        assert!(parse_questions(
            &json!({"q": {"type": "score", "instructions": "x", "criteria": {"a": 1}}})
        )
        .is_err());
        assert!(parse_questions(&json!({"q": {"type": "choice", "instructions": "x", "criteria": ["a"], "labels": {"false": "a", "true": "b"}}})).is_err());
        assert!(parse_questions(&json!({"q": {"type": "noul", "instructions": "x", "labels": {"false": "a", "true": "a"}}})).is_err());
        assert!(parse_questions(&json!({"q": {"type": "noul"}})).is_err());
        assert!(parse_questions(&json!({"q": "nope"})).is_err());
    }
}
