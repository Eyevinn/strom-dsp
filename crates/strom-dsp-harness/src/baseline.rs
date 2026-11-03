//! The checked-in baseline (`baseline.json` next to this crate's
//! `Cargo.toml`) and the regression check against it.
//!
//! The file has three parts:
//!
//! - `run`: the run length the numbers were made with.
//! - `metrics`: per key, which direction is better and the tolerance.
//!   `better` is `lower`, `higher`, `zero` (smaller magnitude) or `none`
//!   (a characterisation: a change beyond the tolerance either way fails).
//!   A fresh value fails when it is worse than the baseline by more than the
//!   tolerance.
//! - `cases`: per `"<preset>/<case>"`, the numbers (`null` = not
//!   applicable).
//!
//! A change that improves a metric on purpose updates the file in the same
//! commit (`cargo test -p strom-dsp-harness --test baseline -- --ignored
//! write_baseline`), with the reason in the commit message. The `metrics`
//! table is kept when the file is rewritten; a new key needs an entry there
//! first.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde_json::{json, Map, Value};

use crate::chain::{Numbers, RunLength};

/// Path of the baseline file.
pub fn path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("baseline.json")
}

/// Which direction of a metric is better.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Better {
    Lower,
    Higher,
    /// Smaller magnitude.
    Zero,
    /// No direction: any change beyond the tolerance fails.
    None,
}

/// Direction and tolerance of one metric.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rule {
    pub better: Better,
    pub tolerance: f64,
}

/// The parsed baseline.
#[derive(Debug, Clone)]
pub struct Baseline {
    pub run: RunLength,
    pub rules: BTreeMap<String, Rule>,
    pub cases: BTreeMap<String, Numbers>,
    /// The `about` text, kept on rewrite.
    about: Value,
}

/// Result of a comparison.
#[derive(Debug, Default)]
pub struct Report {
    /// Regressions and structural mismatches (fail the test).
    pub failures: Vec<String>,
    /// Improvements beyond the tolerance (the baseline should be updated).
    pub improvements: Vec<String>,
}

impl Baseline {
    /// Load and parse the baseline file. Panics with a message on a missing
    /// or malformed file.
    pub fn load() -> Self {
        let text =
            std::fs::read_to_string(path()).unwrap_or_else(|e| panic!("{}: {e}", path().display()));
        let v: Value = serde_json::from_str(&text).expect("baseline.json is not valid JSON");
        let run = RunLength {
            seconds: v["run"]["seconds"].as_f64().expect("run.seconds"),
            skip_seconds: v["run"]["skip_seconds"].as_f64().expect("run.skip_seconds"),
        };
        let rules = v["metrics"]
            .as_object()
            .expect("metrics")
            .iter()
            .map(|(k, r)| {
                let better = match r["better"].as_str().expect("metrics.*.better") {
                    "lower" => Better::Lower,
                    "higher" => Better::Higher,
                    "zero" => Better::Zero,
                    "none" => Better::None,
                    other => panic!("{k}: unknown direction {other}"),
                };
                let tolerance = r["tolerance"].as_f64().expect("metrics.*.tolerance");
                (k.clone(), Rule { better, tolerance })
            })
            .collect();
        let cases = v["cases"]
            .as_object()
            .map(|m| {
                m.iter()
                    .map(|(name, nums)| {
                        let nums = nums
                            .as_object()
                            .expect("case numbers")
                            .iter()
                            .map(|(k, x)| (k.clone(), x.as_f64()))
                            .collect();
                        (name.clone(), nums)
                    })
                    .collect()
            })
            .unwrap_or_default();
        Self {
            run,
            rules,
            cases,
            about: v["about"].clone(),
        }
    }

    /// Compare fresh numbers (a subset of the cases, e.g. one preset) with
    /// the baseline.
    pub fn compare(&self, fresh: &BTreeMap<String, Numbers>) -> Report {
        let mut r = Report::default();
        for (case, nums) in fresh {
            let Some(base) = self.cases.get(case) else {
                r.failures.push(format!("{case}: not in the baseline"));
                continue;
            };
            for key in base.keys() {
                if !nums.contains_key(key) {
                    r.failures
                        .push(format!("{case}: {key} is in the baseline but not measured"));
                }
            }
            for (key, &now) in nums {
                let Some(rule) = self.rules.get(key) else {
                    r.failures
                        .push(format!("{key}: no direction/tolerance in baseline.json"));
                    continue;
                };
                let Some(&was) = base.get(key) else {
                    r.failures
                        .push(format!("{case}: {key} not in the baseline"));
                    continue;
                };
                let (was, now) = match (was, now) {
                    (None, None) => continue,
                    (Some(w), Some(n)) => (w, n),
                    _ => {
                        r.failures.push(format!(
                            "{case}: {key} applicability changed ({was:?} -> {now:?})"
                        ));
                        continue;
                    }
                };
                let tol = rule.tolerance;
                let (worse, better) = match rule.better {
                    Better::Lower => (now > was + tol, now < was - tol),
                    Better::Higher => (now < was - tol, now > was + tol),
                    Better::Zero => (now.abs() > was.abs() + tol, now.abs() < was.abs() - tol),
                    Better::None => ((now - was).abs() > tol, false),
                };
                let line = format!("{case}: {key} {now:.2} (baseline {was:.2}, tolerance {tol})");
                if worse {
                    r.failures.push(line);
                } else if better {
                    r.improvements.push(line);
                }
            }
        }
        r
    }

    /// Replace the numbers of the given cases (keeping the others, the
    /// rules and the run length) and write the file.
    pub fn write_cases(&self, fresh: &BTreeMap<String, Numbers>, run: RunLength) {
        let mut cases = self.cases.clone();
        for (k, v) in fresh {
            cases.insert(k.clone(), v.clone());
        }
        let round = |x: f64| (x * 100.0).round() / 100.0;
        let cases: Map<String, Value> = cases
            .iter()
            .map(|(name, nums)| {
                let m: Map<String, Value> = nums
                    .iter()
                    .map(|(k, v)| (k.clone(), v.map_or(Value::Null, |x| json!(round(x)))))
                    .collect();
                (name.clone(), Value::Object(m))
            })
            .collect();
        let metrics: Map<String, Value> = self
            .rules
            .iter()
            .map(|(k, r)| {
                let better = match r.better {
                    Better::Lower => "lower",
                    Better::Higher => "higher",
                    Better::Zero => "zero",
                    Better::None => "none",
                };
                (
                    k.clone(),
                    json!({ "better": better, "tolerance": r.tolerance }),
                )
            })
            .collect();
        let v = json!({
            "about": self.about,
            "run": { "seconds": run.seconds, "skip_seconds": run.skip_seconds },
            "metrics": metrics,
            "cases": cases,
        });
        let mut text = serde_json::to_string_pretty(&v).expect("serialise");
        text.push('\n');
        std::fs::write(path(), text).unwrap_or_else(|e| panic!("{}: {e}", path().display()));
    }
}
