//! The harness baseline check: every case of every preset, compared with
//! `baseline.json` (see `strom_dsp_harness::baseline`). One test per preset
//! so they run in parallel.
//!
//! - `write_baseline` (ignored) regenerates the numbers of every case.
//! - `long_run` (ignored) prints the numbers of a 30 s run per case.

use std::collections::BTreeMap;

use strom_dsp::presets::BroadcastPreset;
use strom_dsp_harness::baseline::Baseline;
use strom_dsp_harness::chain::{run_preset, Numbers, RunLength, DEFAULT_RUN};

fn print(fresh: &BTreeMap<String, Numbers>) {
    for (case, nums) in fresh {
        let line: Vec<String> = nums
            .iter()
            .filter_map(|(k, v)| v.map(|v| format!("{k}={v:.2}")))
            .collect();
        println!("{case}: {}", line.join(" "));
    }
}

fn check(preset: BroadcastPreset) {
    let base = Baseline::load();
    assert_eq!(
        base.run, DEFAULT_RUN,
        "baseline made with another run length"
    );
    let fresh = run_preset(preset, DEFAULT_RUN);
    print(&fresh);
    let report = base.compare(&fresh);
    for l in &report.improvements {
        println!("improved beyond tolerance (update the baseline): {l}");
    }
    assert!(
        report.failures.is_empty(),
        "harness regressions:\n{}",
        report.failures.join("\n")
    );
}

#[test]
fn ebu_r128() {
    check(BroadcastPreset::EbuR128);
}

#[test]
fn atsc_a85() {
    check(BroadcastPreset::AtscA85);
}

#[test]
fn streaming() {
    check(BroadcastPreset::Streaming);
}

#[test]
fn speech() {
    check(BroadcastPreset::Speech);
}

#[test]
#[ignore = "rewrites baseline.json"]
fn write_baseline() {
    let base = Baseline::load();
    let fresh: BTreeMap<String, Numbers> = std::thread::scope(|s| {
        let handles: Vec<_> = BroadcastPreset::ALL
            .into_iter()
            .map(|p| s.spawn(move || run_preset(p, DEFAULT_RUN)))
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("preset run"))
            .collect()
    });
    print(&fresh);
    base.write_cases(&fresh, DEFAULT_RUN);
}

#[test]
#[ignore = "long run, prints only"]
fn long_run() {
    let len = RunLength {
        seconds: 30.0,
        skip_seconds: 10.0,
    };
    for p in BroadcastPreset::ALL {
        print(&run_preset(p, len));
    }
}
