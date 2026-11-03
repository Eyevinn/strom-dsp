//! Tests of the clip budget governor in the broadcast chain
//! ([`strom_dsp::governor`], [`strom_dsp::broadcast::BroadcastProcessor`]).
//! All signals are synthetic with fixed seeds.
//!
//! Loudness compliance comes first: the governor never takes the output
//! below the target to stay within the clip budget.
//!
//! - Peaky programme with a budget below what it naturally uses: the
//!   output lands within ±1 LU of the target, the governor reports the
//!   budget excess (no loudness deficit), and the limiter's mean gain
//!   reduction stays under 2 dB.
//! - A step from speech-like to dense programme: no overshoot of the
//!   short-term loudness, on target, no limit cycle over 10 minutes.
//! - 10 s of silence leave density and excess unchanged.
//! - An unreachable budget (brickwalled input, target −5 LUFS): the
//!   output reaches the target and the governor reports the budget
//!   excess, not a deficit.

mod agc_support;
mod programme_support;

use std::f64::consts::PI;

use agc_support::{noise, pink, scale_to_lufs, Looper};
use programme_support::peaky_programme;
use strom_dsp::broadcast::{BroadcastParams, BroadcastProcessor};
use strom_dsp::governor::{depth_target_db, GovernorState};
use strom_dsp::meter::{ChannelLayout, LoudnessConfig, LoudnessMeter};
use strom_dsp::presets::BroadcastPreset;
use strom_dsp::Processor;

const FS: f32 = 48_000.0;
const BLOCK: usize = 480;
const PER_S: usize = FS as usize / BLOCK;

/// The chain with `p`, spike guard off: these tests load the clipper and
/// the limiter with the programme's clicks and transients, which is what
/// the governor works on. With the spike guard on, the clicks never reach
/// the clipper and the peaky programme stays far under every budget.
fn processor(p: &BroadcastParams) -> BroadcastProcessor {
    let p = BroadcastParams {
        spike_guard_enabled: false,
        ..*p
    };
    let mut bp = BroadcastProcessor::new();
    bp.set_params(&p);
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
    /// Mean budget excess over the measured part (dB).
    excess: f64,
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
    let (mut depths, mut gr, mut deficit, mut excess) = (Vec::new(), 0.0, 0.0f64, 0.0);
    let (from, to) = (30 * PER_S, 90 * PER_S);
    for i in 0..to {
        looper.fill(&mut buf);
        bp.process(&mut buf);
        let meters = bp.take_meters();
        if i >= from {
            m.process(&buf);
            depths.push(meters.clip_depth_db);
            gr += meters.limiter_gain_reduction_db;
            deficit = deficit.max(meters.loudness_deficit_lu);
            excess += meters.budget_excess_db;
        }
    }
    Peaky {
        integrated: m.integrated(),
        depth: mean_depth(&depths),
        limiter_gr: gr / (to - from) as f64,
        deficit,
        excess: excess / (to - from) as f64,
    }
}

/// Budget for the peaky test: below the ~2.0 dB this programme uses in the
/// streaming preset with the spike guard off (clip depth about −31 dB).
const PEAKY_BUDGET_DB: f32 = 1.5;

/// Streaming preset, budget 1.5 dB, peaky programme at −35, −23 and −10
/// LUFS input (spike guard off, so the clicks load the clipper). The
/// programme needs more than the budget at the target: its peaks are
/// transients the band compressors cannot catch. With the governor on, the
/// output lands within ±1 LU of the target, the governor reports the
/// budget excess (mean over the run > 0) and no loudness deficit, and the
/// limiter's mean gain reduction stays under 2 dB. With the governor off
/// the clip depth is over the allowance too (the case is over budget).
///
/// Guard: the earlier law cut the AGC target to hold the budget and missed
/// the target by 1.4 … 2.0 LU here (−17.41, −17.76, −18.03 LUFS against
/// −16).
#[test]
fn peaky_programme_holds_target_and_reports_excess() {
    let base = BroadcastParams {
        clip_budget_db: PEAKY_BUDGET_DB,
        ..BroadcastParams::preset(BroadcastPreset::Streaming, 4)
    };
    let allowed = depth_target_db(PEAKY_BUDGET_DB as f64);
    let target = base.target_loudness_lufs as f64;
    let mut failures = Vec::new();
    for input in [-35.0, -23.0, -10.0] {
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
             limiter {:.2} dB, excess {:.2} dB, deficit {:.2} LU; off {:.2} LUFS, depth {:.1} dB, \
             limiter {:.2} dB",
            on.integrated,
            on.depth,
            on.limiter_gr,
            on.excess,
            on.deficit,
            off.integrated,
            off.depth,
            off.limiter_gr
        );
        if (on.integrated - target).abs() > 1.0 {
            failures.push(format!("{input}: governor on {:.2} LUFS", on.integrated));
        }
        if on.excess <= 0.0 {
            failures.push(format!("{input}: no budget excess reported"));
        }
        if on.deficit > 0.0 {
            failures.push(format!("{input}: deficit {:.2} LU", on.deficit));
        }
        if on.limiter_gr > 2.0 {
            failures.push(format!("{input}: limiter {:.2} dB", on.limiter_gr));
        }
        if off.depth <= allowed {
            failures.push(format!(
                "{input}: the case is within budget: governor off depth {:.1}",
                off.depth
            ));
        }
        if (off.integrated - target).abs() > 1.0 {
            failures.push(format!("{input}: governor off {:.2} LUFS", off.integrated));
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
/// programme, streaming preset with a 0.5 dB budget (so the dense part is
/// over budget) at −23 LUFS input. After the step the short-term loudness
/// overshoots the target by at most 1 LU; over the last five minutes the
/// short-term loudness stays within ±1 LU of the target, the governor
/// reports an excess at times (the dense part is over budget), and its
/// density stays within 0.5 dB (no limit cycle; density is the only value
/// the governor moves, the excess is a reading of the programme).
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
    let (mut excess, mut density, mut short) = (Vec::new(), Vec::new(), Vec::new());
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
        }
        if i >= 300 * PER_S && i % PER_S == 0 {
            excess.push(meters.budget_excess_db);
            density.push(bp.governor().density_db());
            short.push(m.short_term());
            states.push(meters.governor_state);
        }
    }
    let span = |v: &[f64]| {
        v.iter().cloned().fold(f64::MIN, f64::max) - v.iter().cloned().fold(f64::MAX, f64::min)
    };
    let lo = short.iter().cloned().fold(f64::MAX, f64::min);
    let hi = short.iter().cloned().fold(f64::MIN, f64::max);
    println!(
        "step: max short-term after {max_after:.2} LUFS (target {target}); last 5 min: \
         short-term {lo:.2} … {hi:.2} LUFS, excess from {:.2} dB, span {:.3} dB, density span \
         {:.3} dB, last state {:?}",
        excess[0],
        span(&excess),
        span(&density),
        states.last()
    );
    assert!(max_after <= target + 1.0, "overshoot {max_after}");
    assert!(
        lo >= target - 1.0 && hi <= target + 1.0,
        "short-term {lo} … {hi}"
    );
    assert!(
        excess.iter().any(|&e| e > 0.0),
        "the dense part should be over budget"
    );
    assert!(span(&density) <= 0.5, "density moves {}", span(&density));
}

/// Ten seconds of silence leave the governor's density and budget excess
/// exactly as they were 200 ms into the silence, and it reports itself
/// frozen.
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
    let (excess, density) = (bp.governor().budget_excess_db(), bp.governor().density_db());
    assert!(excess > 0.0, "the governor should be working here");
    for _ in 20..10 * PER_S {
        buf.fill(0.0);
        bp.process(&mut buf);
    }
    assert_eq!(bp.governor().budget_excess_db(), excess);
    assert_eq!(bp.governor().density_db(), density);
    assert_eq!(bp.take_meters().governor_state, GovernorState::Frozen);
}

/// Brickwalled, square-like input (hard-clipped noise) through the
/// streaming preset with a target the clip budget cannot hold.
///
/// - Target −7 LUFS: reachable, but only well over the 2 dB budget.
///   Loudness comes first: the output reaches the target (within 1 LU,
///   from 60 s), the governor reports the budget excess and the state
///   `over-budget`, and no loudness deficit. The earlier law cut the AGC
///   target here and reported a deficit instead.
/// - Target −5 LUFS: out of reach under the −1 dBTP ceiling even with all
///   density. The governor reports the excess and, with its density used
///   up, the remaining deficit (a shortfall it does not cause).
fn brickwalled(target: f32) -> (f64, strom_dsp::broadcast::BroadcastMeters, f64) {
    let p = BroadcastParams {
        target_loudness_lufs: target,
        clip_budget_db: 2.0,
        ..BroadcastParams::preset(BroadcastPreset::Streaming, 4)
    };
    let mut x = pink(1 << 18, 2, FS as f64, 61);
    x.iter_mut().for_each(|v| *v = (*v * 4.0).clamp(-0.9, 0.9));
    let mut bp = processor(&p);
    let mut m = meter();
    let mut looper = Looper::new(&x);
    let mut buf = vec![0.0f32; BLOCK * 2];
    let mut last = bp.take_meters();
    for i in 0..120 * PER_S {
        looper.fill(&mut buf);
        bp.process(&mut buf);
        if i >= 60 * PER_S {
            m.process(&buf);
        }
        last = bp.take_meters();
    }
    let g = bp.governor();
    println!(
        "brickwalled, target {target}: output {:.2} LUFS, excess {:.2} dB, deficit {:.2} LU, \
         state {:?}, used {:.2} of 2 dB, density {:.2} dB, depth {:.1} dB",
        m.integrated(),
        last.budget_excess_db,
        last.loudness_deficit_lu,
        last.governor_state,
        g.budget_used_db(),
        g.density_db(),
        g.window_depth_db()
    );
    (m.integrated(), last, g.density_db())
}

#[test]
fn unreachable_budget_reports_excess() {
    let (out, last, _) = brickwalled(-7.0);
    assert!((out + 7.0).abs() <= 1.0, "{out}");
    assert!(last.budget_excess_db > 0.0);
    assert_eq!(last.loudness_deficit_lu, 0.0);
    assert_eq!(last.governor_state, GovernorState::OverBudget);

    let (out, last, density) = brickwalled(-5.0);
    assert!(out < -6.0, "{out}: expected out of reach");
    assert!(last.budget_excess_db > 0.0);
    assert_eq!(density, strom_dsp::governor::DENSITY_MAX_DB);
    assert!(last.loudness_deficit_lu > 0.5);
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
        assert_eq!(m.budget_excess_db, 0.0);
    }
    assert_eq!(
        (bp.governor().budget_excess_db(), bp.governor().density_db()),
        (0.0, 0.0)
    );
}
