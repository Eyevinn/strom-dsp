//! The static listening page.
//!
//! `index.html` holds the page (inline CSS and JS, no external requests)
//! and every number from `listen-data.json`. The audio of each set is in
//! `sets/<group>/<clip>/audio.js`: its MP4 files as base64, loaded with a
//! `<script>` tag when the set is chosen. Script tags load from `file://`
//! where `fetch` does not, so the page works opened straight from disk as
//! well as from any static web server.

use std::io;
use std::path::Path;

use serde_json::Value;

const TEMPLATE: &str = include_str!("page.html");

/// Standard base64 (RFC 4648, with padding).
pub fn base64(data: &[u8]) -> String {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::with_capacity(data.len().div_ceil(3) * 4);
    for c in data.chunks(3) {
        let n = (c[0] as u32) << 16
            | (*c.get(1).unwrap_or(&0) as u32) << 8
            | *c.get(2).unwrap_or(&0) as u32;
        for i in 0..4 {
            if i <= c.len() {
                s.push(A[(n >> (18 - 6 * i) & 63) as usize] as char);
            } else {
                s.push('=');
            }
        }
    }
    s
}

/// JSON safe to embed in a `<script>` element.
fn script_json(v: &Value) -> io::Result<String> {
    Ok(serde_json::to_string(v)
        .map_err(io::Error::other)?
        .replace("</", "<\\/")
        .replace('\u{2028}', "\\u2028")
        .replace('\u{2029}', "\\u2029"))
}

/// Build `index.html` and every set's `audio.js` in `out` from
/// `out/listen-data.json` (written by [`super::render`]).
pub fn build(out: &Path) -> io::Result<()> {
    let data: Value = serde_json::from_str(&std::fs::read_to_string(out.join("listen-data.json"))?)
        .map_err(io::Error::other)?;
    let sets = data["sets"]
        .as_array()
        .ok_or_else(|| io::Error::other("listen-data.json has no sets"))?;
    for set in sets {
        let key = format!(
            "{}/{}",
            set["group"].as_str().unwrap_or_default(),
            set["clip"].as_str().unwrap_or_default()
        );
        let mut audio = serde_json::Map::new();
        for v in set["versions"].as_array().into_iter().flatten() {
            let id = v["id"].as_str().unwrap_or_default();
            let mp4 = v["files"]["mp4"].as_str().unwrap_or_default();
            let bytes = std::fs::read(out.join(mp4))?;
            audio.insert(id.to_string(), Value::String(base64(&bytes)));
        }
        let js = format!(
            "window.__stromListenLoaded({}, {});\n",
            script_json(&Value::String(key))?,
            script_json(&Value::Object(audio))?
        );
        let dir = set["dir"].as_str().unwrap_or_default();
        std::fs::write(out.join(dir).join("audio.js"), js)?;
    }
    let html = TEMPLATE.replacen("/*DATA*/null", &script_json(&data)?, 1);
    std::fs::write(out.join("index.html"), html)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_rfc4648_vectors() {
        for (i, o) in [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("fooba", "Zm9vYmE="),
            ("foobar", "Zm9vYmFy"),
        ] {
            assert_eq!(base64(i.as_bytes()), o);
        }
    }

    #[test]
    fn embedded_json_cannot_close_the_script() {
        let s = script_json(&Value::String("</script><b>".into())).unwrap();
        assert!(!s.contains("</"));
    }

    #[test]
    fn template_has_the_data_slot_and_no_external_requests() {
        assert_eq!(TEMPLATE.matches("/*DATA*/null").count(), 1);
        for bad in [
            "src=\"http",
            "href=\"http",
            "@import",
            "fetch(",
            "XMLHttpRequest",
        ] {
            assert!(!TEMPLATE.contains(bad), "template contains {bad}");
        }
    }
}
