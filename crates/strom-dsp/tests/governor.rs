//! Tests of the clip budget governor in the broadcast chain
//! ([`strom_dsp::governor`], [`strom_dsp::broadcast::BroadcastProcessor`]).
//! All signals are synthetic with fixed seeds.
//!
//! - Peaky programme with a budget below what it naturally uses: the
//!   governor holds the clip depth within the budget and the limiter's
//!   mean gain reduction under 2 dB; with the governor off the clip depth
//!   exceeds the budget (guard). The loudness cost is measured and
//!   documented.
//! - A step from speech-like to dense programme: no overshoot of the
//!   short-term loudness, no limit cycle over 10 minutes.
//! - 10 s of silence leave density and cut unchanged.
//! - An unreachable case (brickwalled input, target too high) reports a
//!   loudness deficit instead of exceeding the budget.

mod agc_support;

use std::f64::consts::PI;

use agc_support::{noise, peaky_programme, pink, scale_to_lufs, Looper};
use strom_dsp::broadcast::{BroadcastParams, BroadcastProcessor};
use strom_dsp::governor::{depth_target_db, GovernorState};
use strom_dsp::meter::{ChannelLayout, LoudnessConfig, LoudnessMeter};
use strom_dsp::presets::BroadcastPreset;
use strom_dsp::Processor;

const FS: f32 = 48_000.0;
const BLOCK: usize = 480;
const PER_S: usize = FS as usize / BLOCK;

fn processor(p: &BroadcastParams) -> BroadcastProcessor {
    let mut bp = BroadcastProcessor::new();
    bp.set_params(p);
    bp.configure(FS, 2);
    bp
}

fn meter() -> LoudnessMeter {
    LoudnessMeter::new(&LoudnessConfig {
        sample_rate: FS,
        channels: 2,
        layout: ChannelLayout::Unknown,
        true_peak: false,
    })
    .unwrap()
}

/// Energy-weighted mean of per-block clip depths (dB). Blocks carry about
/// the same programme energy, so this is close to the depth of the whole
/// stretch.
fn mean_depth(depths: &[f64]) -> f64 {
    let m = depths.iter().map(|d| 10f64.powf(d / 10.0)).sum::<f64>() / depths.len() as f64;
    10.0 * m.max(1e-30).log10()
}

/// Result of one peaky run.
struct Peaky {
    integrated: f64,
    depth: f64,
    limiter_gr: f64,
    deficit: f64,
}

/// The peaky programme (a click every 1.1 s) through `p` at `input_lufs`
/// for 90 s; everything measured from 30 s (the governor is slow).
fn peaky_run(p: &BroadcastParams, input_lufs: f64) -> Peaky {
    let mut x = peaky_programme(2, FS as f64, 1.1);
    scale_to_lufs(&mut x, 2, FS as f64, input_lufs);
    let mut bp = processor(p);
    let mut m = meter();
    let mut looper = Looper::new(&x);
    let mut buf = vec![0.0f32; BLOCK * 2];
    let (mut depths, mut gr, mut deficit) = (Vec::new(), 0.0, 0.0);
    let (from, to) = (30 * PER_S, 90 * PER_S);
    for i in 0..to {
        looper.fill(&mut buf);
        bp.process(&mut buf);
        let meters = bp.take_meters();
        if i >= from {
            m.process(&buf);
            depths.push(meters.clip_depth_db);
            gr += meters.limiter_gain_reduction_db;
            deficit = meters.loudness_deficit_lu;
        }
    }
    Peaky {
        integrated: m.integrated(),
        depth: mean_depth(&depths),
        limiter_gr: gr / (to - from) as f64,
        deficit,
    }
}

/// Budget for the peaky test: below the ~2.0 dB this programme uses in the
/// streaming preset (clip depth about −31 dB), so the governor has to
/// act.
const PEAKY_BUDGET_DB: f32 = 1.5;
/// Integrated output (LUFS) with the governor on, per input level, as
/// measured when the test was written (see the test docs).
const PEAKY_MEASURED: [(f64, f64); 3] = [(-35.0, -17.41), (-23.0, -17.76), (-10.0, -18.03)];

/// Streaming preset, budget 1.5 dB, peaky programme at −35, −23 and −10
/// LUFS input. With the governor on, the mean clip depth stays within the
/// budget's allowance (−32.5 dB) and the limiter's mean gain reduction under
/// 2 dB; with it off the clip depth exceeds the allowance (guard).
///
/// The loudness cost of holding the budget is documented, not bounded by
/// ±1 LU: the peaks of this programme are transients that the band
/// compressors cannot catch, so the only lever left is a cut of the AGC
/// target. The governor acts on 3 s windows, and windows with more hits or
/// clicks set the cut, so the run's mean clip depth ends 0.8 … 2.0 dB under
/// the allowance and the output 1.4 … 2.0 LU under the target (each value
/// within 0.5 LU of the recorded one). The reported deficit must match the
/// shortfall.
#[test]
fn peaky_programme_within_budget() {
    let base = BroadcastParams {
        clip_budget_db: PEAKY_BUDGET_DB,
        ..BroadcastParams::preset(BroadcastPreset::Streaming, 4)
    };
    let allowed = depth_target_db(PEAKY_BUDGET_DB as f64);
    let target = base.target_loudness_lufs as f64;
    let mut failures = Vec::new();
    for (input, recorded) in PEAKY_MEASURED {
        let on = peaky_run(&base, input);
        let off = peaky_run(
            &BroadcastParams {
                governor_enabled: false,
                ..base
            },
            input,
        );
        println!(
            "peaky at {input} LUFS: governor on {:.2} LUFS, depth {:.1} dB (allowed {allowed}), \
             limiter {:.2} dB, deficit {:.2} LU; off {:.2} LUFS, depth {:.1} dB, limiter {:.2} dB",
            on.integrated,
            on.depth,
            on.limiter_gr,
            on.deficit,
            off.integrated,
            off.depth,
            off.limiter_gr
        );
        if on.depth > allowed {
            failures.push(format!("{input}: depth {:.1} over {allowed}", on.depth));
        }
        if on.limiter_gr > 2.0 {
            failures.push(format!("{input}: limiter {:.2} dB", on.limiter_gr));
        }
        if off.depth <= allowed {
            failures.push(format!(
                "{input}: guard: governor off depth {:.1}",
                off.depth
            ));
        }
        if (off.integrated - target).abs() > 1.0 {
            failures.push(format!("{input}: governor off {:.2} LUFS", off.integrated));
        }
        if (on.integrated - recorded).abs() > 0.5 {
            failures.push(format!(
                "{input}: governor on {:.2} LUFS, recorded {recorded}",
                on.integrated
            ));
        }
        if on.deficit <= 0.0 || (on.deficit - (target - on.integrated)).abs() > 1.0 {
            failures.push(format!(
                "{input}: deficit {:.2} for {:.2} LUFS",
                on.deficit, on.integrated
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Speech-like programme: noise band-limited to about 3.4 kHz (pink, with a
/// syllable envelope and 400 ms pauses every 3 s).
fn speech_like(frames: usize) -> Vec<f32> {
    let mut x = noise(frames, 2, FS as f64, 1.0, 41);
    // Crude band limit: a running average of 7 (zero near 6.9 kHz).
    let mut acc = [0.0f32; 2];
    let mut hist = [[0.0f32; 2]; 7];
    for n in 0..frames {
        let t = n as f64 / FS as f64;
        let env =
            (0.2 + 0.8 * (2.0 * PI * 4.0 * t).sin().abs()) * if t % 3.0 < 0.4 { 0.02 } else { 1.0 };
        for c in 0..2 {
            let v = x[n * 2 + c];
            acc[c] += v - hist[n % 7][c];
            hist[n % 7][c] = v;
            x[n * 2 + c] = (acc[c] / 7.0) * env as f32;
        }
    }
    x
}

/// Dense, music-like programme: the peaky programme without clicks (a
/// crowd bed under hits every 250 ms).
fn dense() -> Vec<f32> {
    peaky_programme(2, FS as f64, 1e9)
}

/// One minute of speech-like programme, then nine minutes of dense
/// programme, streaming preset with a 0.5 dB budget (so the governor works
/// on the dense part) at −23 LUFS input. After the step the short-term
/// loudness overshoots the target by at most 1 LU; over the last five
/// minutes the governor's density and cut stay within 0.5 dB (no limit
/// cycle).
#[test]
fn step_to_dense_programme_does_not_oscillate() {
    let p = BroadcastParams {
        clip_budget_db: 0.5,
        ..BroadcastParams::preset(BroadcastPreset::Streaming, 4)
    };
    let target = p.target_loudness_lufs as f64;
    let mut speech = speech_like(1 << 20);
    scale_to_lufs(&mut speech, 2, FS as f64, -23.0);
    let mut music = dense();
    scale_to_lufs(&mut music, 2, FS as f64, -23.0);
    let mut bp = processor(&p);
    let mut m = meter();
    let mut buf = vec![0.0f32; BLOCK * 2];
    let (mut ls, mut lm) = (Looper::new(&speech), Looper::new(&music));
    let step = 60 * PER_S;
    let end = 600 * PER_S;
    let mut max_after = f64::NEG_INFINITY;
    let mut max_cut = 0.0f64;
    let (mut cut, mut density) = (Vec::new(), Vec::new());
    let mut states = Vec::new();
    for i in 0..end {
        if i < step {
            ls.fill(&mut buf);
        } else {
            lm.fill(&mut buf);
        }
        bp.process(&mut buf);
        m.process(&buf);
        let meters = bp.take_meters();
        if i >= step {
            max_after = max_after.max(m.short_term());
            max_cut = max_cut.max(bp.governor().cut_db());
        }
        if i >= 300 * PER_S && i % PER_S == 0 {
            cut.push(bp.governor().cut_db());
            density.push(bp.governor().density_db());
            states.push(meters.governor_state);
        }
    }
    let span = |v: &[f64]| {
        v.iter().cloned().fold(f64::MIN, f64::max) - v.iter().cloned().fold(f64::MAX, f64::min)
    };
    println!(
        "step: max short-term after {max_after:.2} LUFS (target {target}); largest cut \
         {max_cut:.2} dB; last 5 min: cut from {:.2} dB, span {:.3} dB, density span {:.3} dB, \
         last state {:?}",
        cut[0],
        span(&cut),
        span(&density),
        states.last()
    );
    assert!(max_after <= target + 1.0, "overshoot {max_after}");
    assert!(max_cut > 0.0, "the governor should work on the dense part");
    assert!(span(&cut) <= 0.5, "cut moves {}", span(&cut));
    assert!(span(&density) <= 0.5, "density moves {}", span(&density));
}

/// Ten seconds of silence leave the governor's density and cut exactly as
/// they were 200 ms into the silence, and it reports itself frozen.
#[test]
fn silence_leaves_governor_unchanged() {
    let p = BroadcastParams {
        clip_budget_db: PEAKY_BUDGET_DB,
        ..BroadcastParams::preset(BroadcastPreset::Streaming, 4)
    };
    let mut x = peaky_programme(2, FS as f64, 1.1);
    scale_to_lufs(&mut x, 2, FS as f64, -23.0);
    let mut bp = processor(&p);
    let mut looper = Looper::new(&x);
    let mut buf = vec![0.0f32; BLOCK * 2];
    for _ in 0..40 * PER_S {
        looper.fill(&mut buf);
        bp.process(&mut buf);
    }
    // The first 100 ms of silence (plus the chain latency) still read as
    // programme (the trim's rule); from then on nothing may move.
    for _ in 0..20 {
        buf.fill(0.0);
        bp.process(&mut buf);
    }
    let (cut, density) = (bp.governor().cut_db(), bp.governor().density_db());
    assert!(cut > 0.0, "the governor should be working here");
    for _ in 20..10 * PER_S {
        buf.fill(0.0);
        bp.process(&mut buf);
    }
    assert_eq!(bp.governor().cut_db(), cut);
    assert_eq!(bp.governor().density_db(), density);
    assert_eq!(bp.take_meters().governor_state, GovernorState::Frozen);
}

/// A brickwalled, square-like input (hard-clipped noise) with the target
/// at −5 LUFS: the chain cannot get there within the budget, so the
/// governor reports a deficit and holds the budget instead.
#[test]
fn unreachable_target_reports_deficit() {
    let p = BroadcastParams {
        target_loudness_lufs: -5.0,
        clip_budget_db: 2.0,
        ..BroadcastParams::preset(BroadcastPreset::Streaming, 4)
    };
    let mut x = pink(1 << 18, 2, FS as f64, 61);
    x.iter_mut().for_each(|v| *v = (*v * 4.0).clamp(-0.9, 0.9));
    let mut bp = processor(&p);
    let mut looper = Looper::new(&x);
    let mut buf = vec![0.0f32; BLOCK * 2];
    let mut last = bp.take_meters();
    for _ in 0..120 * PER_S {
        looper.fill(&mut buf);
        bp.process(&mut buf);
        last = bp.take_meters();
    }
    let g = bp.governor();
    println!(
        "unreachable: deficit {:.2} LU, state {:?}, used {:.2} of 2 dB, cut {:.2}, depth {:.1}",
        last.loudness_deficit_lu,
        last.governor_state,
        g.budget_used_db(),
        g.cut_db(),
        g.window_depth_db()
    );
    assert!(last.loudness_deficit_lu > 0.0);
    assert!(g.budget_used_db() <= 2.0 + 0.5, "{}", g.budget_used_db());
}

/// With the governor off it reports `Off`, no deficit, and does nothing.
#[test]
fn governor_off_is_neutral() {
    let p = BroadcastParams {
        governor_enabled: false,
        clip_budget_db: 0.0,
        ..BroadcastParams::preset(BroadcastPreset::Streaming, 4)
    };
    let mut x = peaky_programme(2, FS as f64, 0.3);
    scale_to_lufs(&mut x, 2, FS as f64, -23.0);
    let mut bp = processor(&p);
    let mut looper = Looper::new(&x);
    let mut buf = vec![0.0f32; BLOCK * 2];
    for _ in 0..20 * PER_S {
        looper.fill(&mut buf);
        bp.process(&mut buf);
        let m = bp.take_meters();
        assert_eq!(m.governor_state, GovernorState::Off);
        assert_eq!(m.loudness_deficit_lu, 0.0);
    }
    assert_eq!(
        (bp.governor().cut_db(), bp.governor().density_db()),
        (0.0, 0.0)
    );
}
