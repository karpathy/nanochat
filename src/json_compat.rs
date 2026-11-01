//! Parse Python's JSON output, including its legacy non-finite float spellings.
//!
//! Python's `json.dump` emits bare `NaN` and `Infinity` values by default,
//! while those spellings are rejected by strict JSON parsers such as
//! `serde_json`. Checkpoint metadata can contain infinity before the first
//! validation pass, so normalize those values to JSON null while preserving
//! strings and all other input bytes.

use anyhow::Result;
use serde_json::Value;

/// Format a finite float like Python's `str()`/`json.dumps()` short decimal
/// representation, including its fixed/scientific notation thresholds and
/// two-digit exponent padding.
pub(crate) fn python_float_str(value: f64) -> String {
    let rendered = value.to_string();
    let (mantissa, explicit_exponent) = rendered
        .split_once(['e', 'E'])
        .map(|(mantissa, exponent)| (mantissa, exponent.parse::<i32>().unwrap_or_default()))
        .unwrap_or((&rendered, 0));
    let (sign, mantissa) = mantissa
        .strip_prefix('-')
        .map_or(("", mantissa), |mantissa| ("-", mantissa));
    let decimal = mantissa.find('.').unwrap_or(mantissa.len());
    let mut decimal_position = decimal as i32 + explicit_exponent;
    let mut digits = mantissa
        .bytes()
        .filter(|byte| *byte != b'.')
        .map(char::from)
        .collect::<String>();
    while digits.len() > 1 && digits.starts_with('0') {
        digits.remove(0);
        decimal_position -= 1;
    }
    while digits.len() > 1 && digits.ends_with('0') {
        digits.pop();
    }

    let exponent = decimal_position - 1;
    if exponent < -4 || exponent >= 16 {
        let coefficient = if digits.len() == 1 {
            digits
        } else {
            format!("{}.{}", &digits[..1], &digits[1..])
        };
        format!(
            "{sign}{coefficient}e{}{exponent:02}",
            if exponent < 0 { "-" } else { "+" },
            exponent = exponent.unsigned_abs()
        )
    } else if decimal_position <= 0 {
        format!(
            "{sign}0.{}{}",
            "0".repeat(decimal_position.unsigned_abs() as usize),
            digits
        )
    } else if decimal_position as usize >= digits.len() {
        format!(
            "{sign}{}{}.0",
            digits,
            "0".repeat(decimal_position as usize - digits.len())
        )
    } else {
        let decimal_position = decimal_position as usize;
        format!(
            "{sign}{}.{}",
            &digits[..decimal_position],
            &digits[decimal_position..]
        )
    }
}

pub fn parse_value(bytes: &[u8]) -> Result<Value> {
    let mut normalized = Vec::with_capacity(bytes.len());
    let mut index = 0;
    let mut in_string = false;
    let mut escaped = false;

    while index < bytes.len() {
        let byte = bytes[index];
        if in_string {
            normalized.push(byte);
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            index += 1;
            continue;
        }
        if byte == b'"' {
            in_string = true;
            normalized.push(byte);
            index += 1;
            continue;
        }

        let candidate = if bytes[index..].starts_with(b"-Infinity") {
            Some((b"-Infinity".as_slice(), 9))
        } else if bytes[index..].starts_with(b"Infinity") {
            Some((b"Infinity".as_slice(), 8))
        } else if bytes[index..].starts_with(b"NaN") {
            Some((b"NaN".as_slice(), 3))
        } else {
            None
        };
        if let Some((literal, width)) = candidate {
            let left_is_boundary = index == 0
                || matches!(
                    bytes[index - 1],
                    b' ' | b'\t' | b'\r' | b'\n' | b':' | b',' | b'[' | b'{'
                );
            let end = index + width;
            let right_is_boundary = end == bytes.len()
                || matches!(
                    bytes[end],
                    b' ' | b'\t' | b'\r' | b'\n' | b',' | b']' | b'}'
                );
            if left_is_boundary && right_is_boundary {
                debug_assert_eq!(&bytes[index..end], literal);
                normalized.extend_from_slice(b"null");
                index = end;
                continue;
            }
        }
        normalized.push(byte);
        index += 1;
    }

    Ok(serde_json::from_slice(&normalized)?)
}
