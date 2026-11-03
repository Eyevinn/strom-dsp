//! Broadcast presets: the defaults the GStreamer elements start with.
//!
//! The library's `*Params::default()` values are kept as they are (many
//! theory tests are written against them). The GStreamer elements instead
//! start from the presets in this module, chosen so that each element is
//! useful on its own for **live broadcast** (speech and programme audio)
//! without any property set.
//!
//! These are strom-dsp's own choices, derived from textbook and common
//! broadcast practice, not from any product's presets:
//!
//! - **Levels.** EBU R68 aligns digital audio at −18 dBFS (sine) for the
//!   alignment level; EBU R128 / ITU-R BS.1770 normalise programme loudness
//!   to −23 LUFS with a maximum true peak of −1 dBTP. Speech at that
//!   loudness has its average (RMS) level around −20 dBFS and its loud
//!   syllables around −15 … −10 dBFS (sine-calibrated detector reading).
//!   Studio noise floors (room tone, microphone self-noise, HVAC) of a
//!   reasonable speech studio are well below −50 dBFS at that alignment.
//! - **Time constants.** Speech syllables come at about 4 … 5 per second
//!   (200 … 250 ms apart); plosive and consonant onsets rise within a few
//!   milliseconds; gaps inside a word are shorter than ~100 ms. Dynamic
//!   time constants must be long compared with the period of the lowest
//!   frequency of interest to avoid waveform distortion (Giannoulis,
//!   Massberg & Reiss, JAES 60(6), 2012; Zölzer, *DAFX*, dynamics chapter).
//!
//! Every value below states its reason. The README (per element, "Why these
//! defaults") and `PROVENANCE.md` repeat them.

use crate::compressor::CompressorDetector;
use crate::gate::GateMode;
use crate::{CompressorParams, GateParams, LimiterParams};

impl GateParams {
    /// Broadcast default of the gate: closes cleanly in speech pauses.
    ///
    /// | Field | Value | Why |
    /// |---|---|---|
    /// | `open_threshold_db` | −45 dB | well below speech (≈ −20 dBFS average at R68/R128 alignment, so even quiet word endings at −35 … −40 keep it open) and above a typical studio noise floor (< −50 dBFS) |
    /// | `close_threshold_db` | −50 dB | 5 dB of hysteresis: a level hovering around the threshold (breath, room tone) cannot make the gate chatter |
    /// | `attack_ms` | 3 ms | faster than a consonant/plosive onset (≈ 5 … 10 ms) so word starts are not clipped; slow enough that the gain rise from −35 dB does not click |
    /// | `hold_ms` | 120 ms | bridges gaps inside and between words (< ~100 ms) so the gate does not chatter mid-sentence; shorter than a speech pause |
    /// | `release_ms` | 200 ms | time to close the full range (−35 dB): a 0.175 dB/ms fade that follows a natural room decay instead of cutting it, and is done well inside a typical speech pause (≥ 300 ms) |
    /// | `reduction_db` | −35 dB | pushes a −50 dBFS noise floor to −85 dBFS, inaudible under programme, without the "dead air" contrast and pumping of a full −60 … −80 dB mute |
    pub fn broadcast() -> Self {
        Self {
            open_threshold_db: -45.0,
            close_threshold_db: -50.0,
            attack_ms: 3.0,
            hold_ms: 120.0,
            release_ms: 200.0,
            reduction_db: -35.0,
        }
    }
}

/// Broadcast default gate mode: a classic gate. An expander is a choice for
/// an operator who has set a threshold for a specific source; out of the box
/// a gate with a moderate `reduction` is the predictable behaviour.
pub const GATE_BROADCAST_MODE: GateMode = GateMode::Gate;

/// Broadcast default expander ratio (1:2), used only in expander mode: the
/// gentlest textbook downward-expansion slope, so switching to expander mode
/// without setting a ratio gives a soft, natural noise reduction.
pub const GATE_BROADCAST_RATIO: f32 = 2.0;

impl CompressorParams {
    /// Broadcast default of the compressor: gentle speech/programme
    /// levelling that compresses out of the box.
    ///
    /// | Field | Value | Why |
    /// |---|---|---|
    /// | `threshold` | 0.125 (−18.06 dBFS) | the EBU R68 alignment level (−18 dBFS sine); speech at −23 LUFS sits around it, so loud syllables are compressed and quiet speech is not. 0.125 = 2⁻³ is exact in `f32` and reads back exactly |
    /// | `ratio` | 3 | moderate textbook ratio for speech and programme: audible levelling without the flattened sound of ≥ 4:1 |
    /// | `attack_ms` | 10 ms | lets the first ~10 ms of a transient through (keeps articulation and punch) while catching each syllable |
    /// | `release_ms` | 150 ms | shorter than the syllable spacing (200 … 250 ms) so the gain recovers between syllables; long compared with the period of the lowest bass (20 ms at 50 Hz) so it neither distorts nor pumps |
    /// | `makeup_gain` | 1 | no level jump when the element is inserted; automatic makeup is a separate property |
    /// | `knee` | 0.5 (≈ 12 dB wide) | a soft knee from −24 to −12 dBFS, so compression sets in gradually (Giannoulis et al. 2012); 0.5 is exact in `f32` |
    pub fn broadcast() -> Self {
        Self {
            threshold: 0.125,
            ratio: 3.0,
            attack_ms: 10.0,
            release_ms: 150.0,
            makeup_gain: 1.0,
            knee: 0.5,
        }
    }
}

/// Broadcast default compressor detector: RMS. An average level follows
/// perceived loudness better than the sample peak, the classic choice for
/// speech and programme levelling (Zölzer, *DAFX*; Giannoulis et al. 2012).
/// Both detectors read a steady sine at its peak, so the threshold means the
/// same in both.
pub const COMPRESSOR_BROADCAST_DETECTOR: CompressorDetector = CompressorDetector::Rms;

/// Broadcast default RMS averaging time (ms): 10 ms, the same as the gate's
/// detector. At least one period of 100 Hz, so the level of voiced speech
/// does not ripple, and short enough that the attack time (10 ms) still
/// sets the reaction to a syllable.
pub const COMPRESSOR_BROADCAST_RMS_WINDOW_MS: f32 = 10.0;

/// Broadcast default of automatic makeup: off, so inserting the element
/// never raises the level by itself. The automatic makeup is being
/// redesigned; this default may be revisited once that has landed.
pub const COMPRESSOR_BROADCAST_AUTO_MAKEUP: bool = false;

impl LimiterParams {
    /// Broadcast default of the limiter: a programme-safe peak ceiling.
    ///
    /// | Field | Value | Why |
    /// |---|---|---|
    /// | `enabled` | true | a safety limiter is meant to be on |
    /// | `threshold_db` | −1 dBFS | the EBU R128 maximum true-peak level (−1 dBTP). This is a sample-peak ceiling, so inter-sample peaks can still exceed it slightly |
    /// | `attack_ms` | 1 ms | a smooth gain ramp (no step on a peak) that fits well inside the lookahead, so the ceiling is still guaranteed |
    /// | `release_ms` | 200 ms | programme-suited: long compared with bass periods (no low-frequency distortion), short enough to recover between phrases without audible pumping |
    /// | `lookahead_ms` | 5 ms | 5 ms of latency, which Strom relies on; covers the attack with margin |
    pub fn broadcast() -> Self {
        Self {
            enabled: true,
            threshold_db: -1.0,
            attack_ms: 1.0,
            release_ms: 200.0,
            lookahead_ms: 5.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compressor::RMS_WINDOW_RANGE;
    use crate::gate::EXPANDER_RATIO_RANGE;
    use crate::{Compressor, Processor};

    #[test]
    fn presets_are_in_range() {
        let g = GateParams::broadcast();
        assert_eq!(g.clamped(), g);
        assert!(g.close_threshold_db < g.open_threshold_db);
        let c = CompressorParams::broadcast();
        assert_eq!(c.clamped(), c);
        let l = LimiterParams::broadcast();
        assert!((-60.0..=0.0).contains(&l.threshold_db));
        assert!((0.01..=50.0).contains(&l.attack_ms));
        assert!((1.0..=2000.0).contains(&l.release_ms));
        assert!((0.0..=crate::MAX_LOOKAHEAD_MS).contains(&l.lookahead_ms));
        assert!(l.attack_ms <= l.lookahead_ms);
        assert!((EXPANDER_RATIO_RANGE.0..=EXPANDER_RATIO_RANGE.1).contains(&GATE_BROADCAST_RATIO));
        assert!(
            (RMS_WINDOW_RANGE.0..=RMS_WINDOW_RANGE.1).contains(&COMPRESSOR_BROADCAST_RMS_WINDOW_MS)
        );
    }

    /// The compressor preset compresses out of the box: a 1 kHz sine at
    /// −6 dBFS (above the knee, which ends at T + 6 dB) comes out at
    /// T + (x − T)/3 ≈ −14.04 dBFS (Giannoulis et al. 2012, static curve).
    #[test]
    fn compressor_preset_compresses() {
        let fs = 48_000.0f32;
        let mut c = Compressor::new();
        c.configure(fs, 1);
        c.set_detector(COMPRESSOR_BROADCAST_DETECTOR);
        c.set_rms_window(COMPRESSOR_BROADCAST_RMS_WINDOW_MS);
        c.set_auto_makeup(COMPRESSOR_BROADCAST_AUTO_MAKEUP);
        c.set_params(&CompressorParams::broadcast());
        let amp = 10f32.powf(-6.0 / 20.0);
        let mut buf: Vec<f32> = (0..fs as usize)
            .map(|n| amp * (2.0 * std::f32::consts::PI * 1000.0 * n as f32 / fs).sin())
            .collect();
        c.process(&mut buf);
        let tail = &buf[buf.len() - 4800..];
        let peak = tail.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let out_db = 20.0 * peak.log10();
        let t = 20.0 * 0.125f32.log10();
        let expected = t + (-6.0 - t) / 3.0;
        assert!(
            (out_db - expected).abs() < 0.5,
            "out {out_db} dB, expected {expected} dB"
        );
    }
}
