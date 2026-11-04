//! Calls that are never real-time safe stay out of the library.
//!
//! The counting-allocator tests only see an allocation on the platform
//! where it happens: `std::env::var` allocates on Windows but not on Linux
//! or macOS, and takes the process-wide environment lock everywhere. This
//! test scans the library source (outside its `#[cfg(test)]` modules) for
//! such calls, so one fails on every platform. strom-dsp has no use for
//! them anywhere: it reads no environment, prints nothing, keeps no clock
//! and shares no state between threads.

use std::path::Path;

/// Forbidden in library code, with the reason.
const FORBIDDEN: &[(&str, &str)] = &[
    (
        "std::env",
        "reads the environment under a lock (allocates on Windows)",
    ),
    (
        "env::var",
        "reads the environment under a lock (allocates on Windows)",
    ),
    ("println!", "I/O"),
    ("eprintln!", "I/O"),
    ("print!(", "I/O"),
    ("eprint!(", "I/O"),
    ("dbg!(", "I/O"),
    ("format!(", "allocates"),
    ("std::io", "I/O"),
    ("std::fs", "I/O"),
    ("std::thread", "threads"),
    ("Instant::", "clock (system call)"),
    ("SystemTime", "clock (system call)"),
    ("Mutex", "lock"),
    ("RwLock", "lock"),
];

fn scan(dir: &Path, hits: &mut Vec<String>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            scan(&path, hits);
            continue;
        }
        if path.extension().is_none_or(|e| e != "rs") {
            continue;
        }
        let text = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            // The unit-test module ends a source file.
            if line.trim() == "#[cfg(test)]"
                && lines[i + 1..]
                    .iter()
                    .find(|l| !l.trim().is_empty())
                    .is_some_and(|l| l.starts_with("mod "))
            {
                break;
            }
            let code = line.split("//").next().unwrap_or("");
            for (pat, why) in FORBIDDEN {
                if code.contains(pat) {
                    hits.push(format!("{}:{}: {pat} ({why})", path.display(), i + 1));
                }
            }
        }
    }
}

#[test]
fn library_has_no_non_realtime_calls() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut hits = Vec::new();
    scan(&src, &mut hits);
    assert!(
        hits.is_empty(),
        "non-real-time calls in library code:\n{}",
        hits.join("\n")
    );
}
