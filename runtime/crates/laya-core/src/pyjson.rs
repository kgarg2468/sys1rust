//! `json.dumps(obj, ensure_ascii=False)` with Python's default separators `(", ", ": ")`.
//!
//! The model was trained on text produced by Python's `json.dumps`, and the reference
//! implementation feeds it verbatim (`serialize_state`, `render_criterion`, non-string
//! instructions). Byte-identical output is therefore part of the model contract, including
//! Python's float `repr` and its string escaping rules.

use serde_json::Value;
use std::fmt::Write as _;

/// Serialize like `json.dumps(v, ensure_ascii=False)`.
pub fn dumps(v: &Value) -> String {
    let mut out = String::new();
    write_value(v, &mut out);
    out
}

fn write_value(v: &Value, out: &mut String) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                let _ = write!(out, "{i}");
            } else if let Some(u) = n.as_u64() {
                let _ = write!(out, "{u}");
            } else {
                out.push_str(&float_repr(n.as_f64().unwrap_or(0.0)));
            }
        }
        Value::String(s) => write_str(s, out),
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_value(x, out);
            }
            out.push(']');
        }
        Value::Object(o) => {
            out.push('{');
            for (i, (k, x)) in o.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_str(k, out);
                out.push_str(": ");
                write_value(x, out);
            }
            out.push('}');
        }
    }
}

/// Python `json` string escaping with `ensure_ascii=False`.
pub fn write_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Python `repr(float)`: shortest round-trip digits, fixed notation for exponents in
/// `[-4, 16)`, otherwise scientific with a signed two-digit exponent.
pub fn float_repr(x: f64) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 {
            "Infinity".into()
        } else {
            "-Infinity".into()
        };
    }
    if x == 0.0 {
        return if x.is_sign_negative() {
            "-0.0".into()
        } else {
            "0.0".into()
        };
    }
    let s = format!("{x:e}");
    let (mant, exp) = s.split_once('e').expect("Rust {:e} always contains 'e'");
    let exp: i32 = exp.parse().expect("exponent is an integer");
    let neg = mant.starts_with('-');
    let digits: String = mant.chars().filter(|c| c.is_ascii_digit()).collect();
    let mut out = String::new();
    if neg {
        out.push('-');
    }
    if (-4..16).contains(&exp) {
        if exp >= 0 {
            let int_len = exp as usize + 1;
            if digits.len() <= int_len {
                out.push_str(&digits);
                out.push_str(&"0".repeat(int_len - digits.len()));
                out.push_str(".0");
            } else {
                out.push_str(&digits[..int_len]);
                out.push('.');
                out.push_str(&digits[int_len..]);
            }
        } else {
            out.push_str("0.");
            out.push_str(&"0".repeat((-exp - 1) as usize));
            out.push_str(&digits);
        }
    } else {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push('e');
        out.push(if exp < 0 { '-' } else { '+' });
        let ae = exp.abs();
        if ae < 10 {
            out.push('0');
        }
        let _ = write!(out, "{ae}");
    }
    out
}

/// Python `str(x)` for the scalar JSON values that can appear as list-form choice labels.
pub fn py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Null => "None".into(),
        other => dumps(other),
    }
}

/// Python `round(x, dp)`: correctly rounded, ties to even on the exact binary value.
pub fn round_dp(x: f64, dp: u32) -> f64 {
    if !x.is_finite() {
        return x;
    }
    let neg = x.is_sign_negative();
    let ax = x.abs();
    // Exact tie detection: ax * 10^(dp+1) is an integer ending in 5.
    let bits = ax.to_bits();
    let exp_bits = ((bits >> 52) & 0x7ff) as i32;
    let frac = bits & ((1u64 << 52) - 1);
    let (m, e) = if exp_bits == 0 {
        (frac, -1074)
    } else {
        (frac | (1u64 << 52), exp_bits - 1075)
    };
    let scaled = (m as u128) * 5u128.pow(dp + 1);
    let sh = e + dp as i32 + 1;
    let n = if sh >= 0 {
        if sh < 60 {
            Some(scaled << sh)
        } else {
            None
        }
    } else {
        let s = (-sh) as u32;
        if s < 128 && scaled & ((1u128 << s) - 1) == 0 {
            Some(scaled >> s)
        } else {
            None
        }
    };
    let rounded = match n {
        Some(n) if n % 10 == 5 => {
            let q = n / 10;
            let q = if q % 2 == 0 { q } else { q + 1 };
            let digits = format!("{q:0>width$}", width = dp as usize + 1);
            let (ip, fp) = digits.split_at(digits.len() - dp as usize);
            format!("{ip}.{fp}").parse::<f64>().unwrap_or(ax)
        }
        _ => format!("{ax:.*}", dp as usize).parse::<f64>().unwrap_or(ax),
    };
    if neg {
        -rounded
    } else {
        rounded
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn dumps_matches_python_defaults() {
        let v = json!({"from": "user@acme.com", "n": 3, "f": 1.5, "l": [1, "a", null, true], "u": "मुझसे \"q\" \n"});
        assert_eq!(
            dumps(&v),
            "{\"from\": \"user@acme.com\", \"n\": 3, \"f\": 1.5, \"l\": [1, \"a\", null, true], \"u\": \"मुझसे \\\"q\\\" \\n\"}"
        );
        assert_eq!(dumps(&json!({})), "{}");
        assert_eq!(dumps(&json!([])), "[]");
    }

    #[test]
    fn float_repr_matches_python() {
        for (x, s) in [
            (1.0, "1.0"),
            (0.5, "0.5"),
            (100.0, "100.0"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (1.5e-7, "1.5e-07"),
            (123456.789, "123456.789"),
            (-2.5, "-2.5"),
            (1e22, "1e+22"),
            (0.1 + 0.2, "0.30000000000000004"),
            (2.5e-5, "2.5e-05"),
        ] {
            assert_eq!(float_repr(x), s, "repr({x})");
        }
    }

    #[test]
    fn round_matches_python() {
        assert_eq!(round_dp(0.96534, 4), 0.9653);
        assert_eq!(round_dp(0.03125, 4), 0.0312); // exact tie -> even
        assert_eq!(round_dp(0.09375, 4), 0.0938); // exact tie -> even (7 -> 8)
        assert_eq!(round_dp(1.0, 4), 1.0);
        assert_eq!(round_dp(0.86405, 4), 0.864);
        assert_eq!(round_dp(0.12345, 4), 0.1235);
        assert_eq!(round_dp(1.00005, 4), 1.0001);
        assert_eq!(round_dp(0.00015, 4), 0.0001);
        assert_eq!(round_dp(-0.5, 0), -0.0);
        assert_eq!(round_dp(2.5, 0), 2.0);
    }
}
