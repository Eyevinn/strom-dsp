//! Tests of the clip budget governor in the broadcast chain
//! ([`strom_dsp::governor`], [`strom_dsp::broadcast::BroadcastProcessor`]).
//! All signals are synthetic with fixed seeds.
//!
//! - 10 s of silence leave density and excess unchanged.
//! - With the governor off it reports `Off` and does nothing.
//!
//! The long programme runs (target and excess on peaky programme at three
//! input levels, the 10-minute step test, the unreachable budget) are in
//! the harness (`strom-dsp-harness/tests/long_runs/governor.rs`), run by
//! the extended workflow. Per change, the harness baseline guards the same
//! readings: `governor_excess_db`, `governor_deficit_lu`,
//! `governor_density_db` and `loudness_error_lu` on the over-budget cases
//! (`streaming/speech_clicks_100ms`, `streaming/speech_clicks_300ms_low_latency`).

use crate::agc_support::{scale_to_lufs, Looper};
use crate::programme_support::peaky_programme;
use strom_dsp::broadcast::{BroadcastParams, BroadcastProcessor};
use strom_dsp::governor::GovernorState;
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

/// Budget below the ~2.0 dB the peaky programme uses in the streaming
/// preset with the spike guard off (clip depth about −31 dB), so the
/// governor is working when the silence starts.
const PEAKY_BUDGET_DB: f32 = 1.5;

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
