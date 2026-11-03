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
//! Every value below states its reason. Each element's page,
//! `docs/elements/<element>.md` ("Why these defaults"), and `PROVENANCE.md`
//! repeat them.

use crate::broadcast::{BandExpanderParams, BroadcastParams, MAX_CROSSOVERS};
use crate::compressor::CompressorDetector;
use crate::gate::GateMode;
use crate::multiband::{MultibandBandParams, MAX_BANDS};
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

// ---------------------------------------------------------------------------
// Broadcast final processor presets
// ---------------------------------------------------------------------------

/// A preset of the broadcast final processor (`stromdsp-broadcastprocessor`,
/// its `preset` property). Every value is strom-dsp's own, derived from the
/// loudness standards (EBU R 128, ATSC A/85, AES TD1004/TD1008) and textbook
/// dynamics practice (Giannoulis, Massberg & Reiss, JAES 2012; Zölzer,
/// *DAFX*). The presets are deliberately **gentle** (low ratios, no drive):
/// over-processing is the usual failure, and an operator can always push
/// harder.
///
/// # Values common to every preset
///
/// | Field | Value | Why |
/// |---|---|---|
/// | `input_gain_db` | 0 dB | the AGC does the levelling; the input is not touched |
/// | `spike_guard_enabled` | true | clicks and other short, isolated peaks are removed before they pump the AGC or count toward the loudness |
/// | `spike_threshold_db` | 8 dB | gentle: only peaks more than 8 dB over the programme's own peak envelope, and at most 1 ms long, count as spikes; they are brought down to that level, so a real transient that happens to qualify loses at most what is above it |
/// | `agc_enabled` | true | the AGC is what makes the output land on the target |
/// | `multiband_enabled` | true | |
/// | `multiband_drive_db` | 0 dB | modest: no extra drive into the bands |
/// | `limiter_enabled` | true | the true-peak ceiling is a hard delivery requirement |
/// | band `makeup_db` | 0 dB | the AGC sets the level; makeup would only move the loudness away from the target |
/// | band `knee_db` | 6 dB | soft knee, compression sets in gradually (Giannoulis et al. 2012). Not an element property: fixed for every preset |
/// | `expander_hold_ms` | 100 ms | bridges the gaps inside words (< ~100 ms), so the expander does not chop word endings |
/// | `bass_clipper_enabled` | true | catches the bass peaks the 30 ms band-0 attack lets through, without harmonics |
/// | `bass_clip_threshold_db` | target + 9 dB | 4 dB above the band threshold: a steady bass at target + 13 dB comes out of the 2:1 band compressor at target + 9, so only louder peaks are clipped |
///
/// # Values per preset
///
/// | Preset | target | ceiling | AGC max gain / cut | AGC rate up / down | `agc_gate_lufs` | band threshold | band ratio | `limiter_release_ms` |
/// |---|---|---|---|---|---|---|---|---|
/// | `EbuR128` | −23 LUFS | −1 dBTP | 24 / 24 dB | 1 / 3 dB/s | −50 | −18 dB | 2 | 200 |
/// | `AtscA85` | −24 LKFS | −2 dBTP | 24 / 24 dB | 3 / 6 dB/s | −50 | −19 dB | 2 | 200 |
/// | `Streaming` | −16 LUFS | −1 dBTP | 24 / 24 dB | 3 / 6 dB/s | −50 | −11 dB | 2 | 200 |
/// | `Speech` | −18 LUFS | −1 dBTP | 24 / 24 dB | 3 / 6 dB/s | −40 | −11 dB | 1.5 | 150 |
///
/// | Preset | expander threshold (every band) | expander ratio | expander floor | bass clip threshold |
/// |---|---|---|---|---|
/// | `EbuR128` | −47 dB | 1.5 | 4 dB | −14 dB |
/// | `AtscA85` | −48 dB | 1.5 | 4 dB | −15 dB |
/// | `Streaming` | −40 dB | 1.5 | 4 dB | −7 dB |
/// | `Speech` | −40 dB | 2 | 6 dB | −9 dB |
///
/// - **Target and ceiling** are those of the standard: EBU R 128 (−23 LUFS,
///   max true peak −1 dBTP), ATSC A/85 (−24 LKFS, −2 dBTP), the AES TD1004/
///   TD1008 range for streaming (−16 LUFS, −1 dBTP) and strom-dsp's own
///   speech target (−18 LUFS, −1 dBTP).
/// - **AGC range and rates.** Every preset gets ±24 dB of range: the AGC
///   must cover the input-to-target distance (12 dB for −35 LUFS input to
///   −23 LUFS) *plus* the level the multiband and limiter take away after
///   it, which the output-referenced trim adds to the AGC target (up to
///   6 dB). With ±12 dB, −35 LUFS input to `EbuR128` used the whole range
///   just to reach the target and had no room left for that make-up.
///   `EbuR128` keeps 1 dB/s up, 3 dB/s down: a slow "fader", not a
///   compressor (well below the syllable rate), cutting faster than
///   boosting so a sudden loud item comes down within seconds while a quiet
///   one rises gently. The other presets sit further from typical delivery
///   levels (streaming −16 LUFS is 7 LU above broadcast-level input; ATSC
///   A/85 needs more than 12 dB of cut for −10 LUFS input), so they also
///   get 3 / 6 dB/s: still a slow
///   rider (a 12 dB correction takes 2 … 4 s), fast enough to land within
///   the first 10 s of a programme at −35 … −10 LUFS input. All four pass
///   the chain acceptance test (60 s programme at −35, −23
///   and −10 LUFS input, integrated within ±1 LU of the target after 10 s;
///   `tests/broadcast.rs`).
/// - **AGC gate** −50 LUFS for programme: room tone and pauses sit below it,
///   while even quiet programme material delivered 12 LU under target
///   (−35 LUFS) stays well above it. **Speech: −40 LUFS** (the "faster AGC
///   gate"): 10 LU closer to speech, so after a talker stops the AGC freezes
///   earlier in the decay of its loudness detector and lifts less during the
///   pause; quiet speech (−30 … −35 LUFS input) still stays above it.
/// - **Band threshold = target + 5 dB** (programme) against each band's
///   sine-calibrated level: after the AGC the programme sits at the target,
///   so only loud passages and peaks of a band are compressed, at a low
///   textbook ratio of 2:1. It follows the target so that all programme
///   presets compress the same relative to their own loudness. **Speech:
///   target + 7 dB at 1.5:1** ("gentler multiband"): speech is already
///   levelled by the AGC; the bands only tame the loudest syllables.
/// - **Expander** (per band, against the same band level as the band
///   threshold): programme presets **target − 24 dB, 1:1.5, at most 4 dB**;
///   speech **target − 22 dB, 1:2, at most 6 dB**. After the AGC, room tone
///   and noise between words of a speech studio sit near or below
///   target − 25 dB (a −50 dBFS floor under speech at −23 LUFS, lifted by
///   whatever the AGC adds), while even quiet word endings stay above
///   target − 20 dB. Modest on purpose: the floor limits what a misjudged
///   threshold can do on quiet music (a fade or a soft passage loses at most
///   4 dB, slowly), and speech gets a little more.
/// - **Bass clip threshold = target + 9 dB**: see the common table.
/// - **Limiter release**: 200 ms for programme, long compared with bass
///   periods (no low-frequency distortion) and short enough to recover
///   between phrases (the same reasoning as [`LimiterParams::broadcast`]).
///   Speech: 150 ms, shorter than the syllable spacing (200 … 250 ms) so a
///   limited plosive does not hold down the next syllable.
///
/// # Crossovers and band time constants per band count
///
/// The same for every preset (see [`broadcast_crossovers`] and
/// [`broadcast_band_times`]):
///
/// | Bands | crossovers (Hz) | attack / release per band, low → high (ms) |
/// |---|---|---|
/// | 1 | (none) | 30/300 |
/// | 2 | 200 | 30/300, 10/150 |
/// | 3 | 200, 3000 | 30/300, 10/150, 5/100 |
/// | 4 | 120, 800, 4000 | 30/300, 15/200, 8/150, 5/100 |
/// | 5 | 120, 800, 4000, 10000 | 30/300, 15/200, 8/150, 5/100, 3/80 |
///
/// Time constants: a band's attack is at least about one period of the
/// lowest frequency it carries (a bass band reaching down to ~40 Hz:
/// 25 ms → 30 ms; a band starting at 120 … 200 Hz: 5 … 8 ms → 10 … 15 ms),
/// so the compressor does not modulate the waveform it follows; its release
/// is long compared with those periods (≥ 10 of them) and close to the
/// syllable spacing (200 … 250 ms) for the bands that carry the body of
/// speech and music. Higher bands recover faster, after sibilants and
/// transients. One band is a wideband compressor that still carries the
/// bass, so it uses the bass band's times.
///
/// Crossovers: 2 bands split off the bass at 200 Hz (the energy of kick,
/// bass and the fundamentals of voices lies below it, so bass peaks no
/// longer pump everything else); 3 bands add 3 kHz, the region where the
/// ear is most sensitive (ISO 226 equal-loudness contours), so presence and
/// sibilance are controlled apart from the body; 4 and 5 bands use the
/// element's defaults (120, 800, 4000, 10000 Hz). Crossovers that a
/// band count does not use keep ascending filler values. Bands at or
/// above the band count are inactive; they get the 5-band values of their
/// index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BroadcastPreset {
    /// EBU R 128 broadcast delivery: −23 LUFS, −1 dBTP.
    EbuR128,
    /// ATSC A/85 (US broadcast): −24 LKFS, −2 dBTP.
    AtscA85,
    /// Online/streaming (AES TD1004/TD1008 range): −16 LUFS, −1 dBTP.
    Streaming,
    /// Talk and commentary: −18 LUFS, −1 dBTP, gentler multiband, faster
    /// AGC gate.
    Speech,
}

impl BroadcastPreset {
    /// Every preset, in element enum order.
    pub const ALL: [Self; 4] = [Self::EbuR128, Self::AtscA85, Self::Streaming, Self::Speech];

    /// The element's enum nick (`ebu-r128`, `atsc-a85`, `streaming`, `speech`).
    pub fn nick(self) -> &'static str {
        match self {
            Self::EbuR128 => "ebu-r128",
            Self::AtscA85 => "atsc-a85",
            Self::Streaming => "streaming",
            Self::Speech => "speech",
        }
    }

    /// The preset with this nick, if any (`custom` is not a preset).
    pub fn from_nick(nick: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|p| p.nick() == nick)
    }

    /// (target LUFS, ceiling dBTP, AGC gate LUFS, band threshold offset dB
    /// re target, band ratio, limiter release ms). See the type docs.
    /// (agc max gain dB, agc max cut dB, rate up dB/s, rate down dB/s).
    /// See the type docs.
    fn agc_range(self) -> (f32, f32, f32, f32) {
        match self {
            Self::EbuR128 => (24.0, 24.0, 1.0, 3.0),
            Self::AtscA85 | Self::Streaming | Self::Speech => (24.0, 24.0, 3.0, 6.0),
        }
    }

    /// (expander threshold re target dB, ratio, floor dB). See the type docs.
    fn expander(self) -> (f32, f32, f32) {
        match self {
            Self::EbuR128 | Self::AtscA85 | Self::Streaming => (-24.0, 1.5, 4.0),
            Self::Speech => (-22.0, 2.0, 6.0),
        }
    }

    fn row(self) -> (f32, f32, f32, f32, f32, f32) {
        match self {
            Self::EbuR128 => (-23.0, -1.0, -50.0, 5.0, 2.0, 200.0),
            Self::AtscA85 => (-24.0, -2.0, -50.0, 5.0, 2.0, 200.0),
            Self::Streaming => (-16.0, -1.0, -50.0, 5.0, 2.0, 200.0),
            Self::Speech => (-18.0, -1.0, -40.0, 7.0, 1.5, 150.0),
        }
    }
}

/// Spike guard threshold (dB above the programme envelope) of every
/// broadcast preset.
pub const SPIKE_THRESHOLD_DB: f32 = 8.0;

/// Soft-knee width (dB) of every band in every broadcast preset.
pub const BROADCAST_BAND_KNEE_DB: f32 = 6.0;

/// Crossover frequencies (Hz) of the broadcast presets for `num_bands`
/// (clamped to 1 … 5); see [`BroadcastPreset`]. Strictly ascending; only the
/// first `num_bands − 1` are used.
pub fn broadcast_crossovers(num_bands: usize) -> [f32; MAX_CROSSOVERS] {
    match num_bands.clamp(1, MAX_BANDS) {
        2 => [200.0, 800.0, 4000.0, 10_000.0],
        3 => [200.0, 3000.0, 4000.0, 10_000.0],
        _ => [120.0, 800.0, 4000.0, 10_000.0],
    }
}

/// (attack ms, release ms) of every band of the broadcast presets for
/// `num_bands` (clamped to 1 … 5); see [`BroadcastPreset`]. Bands at or above
/// `num_bands` get the 5-band values of their index.
pub fn broadcast_band_times(num_bands: usize) -> [(f32, f32); MAX_BANDS] {
    const FIVE: [(f32, f32); MAX_BANDS] = [
        (30.0, 300.0),
        (15.0, 200.0),
        (8.0, 150.0),
        (5.0, 100.0),
        (3.0, 80.0),
    ];
    let n = num_bands.clamp(1, MAX_BANDS);
    let active: &[(f32, f32)] = match n {
        1 => &[(30.0, 300.0)],
        2 => &[(30.0, 300.0), (10.0, 150.0)],
        3 => &[(30.0, 300.0), (10.0, 150.0), (5.0, 100.0)],
        4 => &FIVE[..4],
        _ => &FIVE,
    };
    core::array::from_fn(|k| active.get(k).copied().unwrap_or(FIVE[k]))
}

impl BroadcastParams {
    /// The parameters of `preset` for `num_bands` bands, with `enabled` true
    /// and the default 5 ms limiter lookahead (the fields a preset does not
    /// set). See [`BroadcastPreset`] for every value and its reason.
    pub fn preset(preset: BroadcastPreset, num_bands: usize) -> Self {
        let mut p = Self {
            enabled: true,
            num_bands: num_bands.clamp(1, MAX_BANDS),
            limiter_lookahead_ms: 5.0,
            ..Self::default()
        };
        p.apply_preset(preset);
        p
    }

    /// Set every field of `preset` for the current `num_bands`, except
    /// `enabled`, `num_bands`, `spike_guard_lookahead_ms` and
    /// `limiter_lookahead_ms`, which a preset never changes.
    pub fn apply_preset(&mut self, preset: BroadcastPreset) {
        let (target, ceiling, gate, offset, ratio, limiter_release) = preset.row();
        let times = broadcast_band_times(self.num_bands);
        self.input_gain_db = 0.0;
        self.spike_guard_enabled = true;
        self.spike_threshold_db = SPIKE_THRESHOLD_DB;
        self.target_loudness_lufs = target;
        self.true_peak_ceiling_dbtp = ceiling;
        self.agc_enabled = true;
        let (max_gain, max_cut, rate_up, rate_down) = preset.agc_range();
        self.agc_max_gain_db = max_gain;
        self.agc_max_cut_db = max_cut;
        self.agc_rate_up_db_s = rate_up;
        self.agc_rate_down_db_s = rate_down;
        self.agc_gate_lufs = gate;
        self.multiband_enabled = true;
        self.multiband_drive_db = 0.0;
        self.crossover_hz = broadcast_crossovers(self.num_bands);
        self.bands = core::array::from_fn(|k| MultibandBandParams {
            threshold_db: target + offset,
            ratio,
            attack_ms: times[k].0,
            release_ms: times[k].1,
            makeup_db: 0.0,
            knee_db: BROADCAST_BAND_KNEE_DB,
        });
        self.limiter_enabled = true;
        self.limiter_release_ms = limiter_release;
        let (e_offset, e_ratio, e_floor) = preset.expander();
        self.band_expanders = [BandExpanderParams {
            threshold_db: target + e_offset,
            ratio: e_ratio,
            floor_db: e_floor,
        }; MAX_BANDS];
        self.expander_hold_ms = BROADCAST_EXPANDER_HOLD_MS;
        self.bass_clipper_enabled = true;
        self.bass_clip_threshold_db = target + BROADCAST_BASS_CLIP_OFFSET_DB;
    }
}

/// Expander hold of every broadcast preset (ms).
pub const BROADCAST_EXPANDER_HOLD_MS: f32 = 100.0;
/// Bass clip threshold of every broadcast preset, relative to the target (dB).
pub const BROADCAST_BASS_CLIP_OFFSET_DB: f32 = 9.0;

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

    /// Every broadcast preset, at every band count, is in the element's
    /// ranges, has strictly ascending crossovers (≥ 1 Hz
    /// apart, the element's rule), and is gentle.
    #[test]
    fn broadcast_presets_are_in_range_and_gentle() {
        let r = |v: f32, lo: f32, hi: f32| (lo..=hi).contains(&v);
        for preset in BroadcastPreset::ALL {
            assert_eq!(BroadcastPreset::from_nick(preset.nick()), Some(preset));
            for n in 1..=MAX_BANDS {
                let p = BroadcastParams::preset(preset, n);
                let ctx = format!("{preset:?} @ {n}");
                assert!(p.enabled && p.num_bands == n, "{ctx}");
                assert_eq!(p.limiter_lookahead_ms, 5.0, "{ctx}");
                assert!(r(p.input_gain_db, -24.0, 24.0), "{ctx}");
                assert!(r(p.target_loudness_lufs, -40.0, -5.0), "{ctx}");
                assert!(r(p.true_peak_ceiling_dbtp, -12.0, 0.0), "{ctx}");
                assert!(r(p.agc_max_gain_db, 0.0, 30.0), "{ctx}");
                assert!(r(p.agc_max_cut_db, 0.0, 30.0), "{ctx}");
                assert!(r(p.agc_rate_up_db_s, 0.1, 20.0), "{ctx}");
                assert!(r(p.agc_rate_down_db_s, 0.1, 40.0), "{ctx}");
                assert!(p.agc_rate_down_db_s > p.agc_rate_up_db_s, "{ctx}");
                assert!(r(p.agc_gate_lufs, -80.0, -20.0), "{ctx}");
                assert!(p.agc_gate_lufs < p.target_loudness_lufs - 15.0, "{ctx}");
                assert_eq!(p.multiband_drive_db, 0.0, "{ctx}: modest drive");
                assert!(r(p.limiter_release_ms, 1.0, 2000.0), "{ctx}");
                assert!(r(p.crossover_hz[0], 20.0, 20_000.0), "{ctx}");
                for w in p.crossover_hz.windows(2) {
                    assert!(w[1] - w[0] >= 1.0 && r(w[1], 20.0, 20_000.0), "{ctx}");
                }
                for b in &p.bands {
                    assert_eq!(b.clamped(), *b, "{ctx}: band in range");
                    assert!(b.ratio <= 2.0, "{ctx}: gentle ratio");
                    assert_eq!(b.makeup_db, 0.0, "{ctx}");
                }
                for (e, b) in p.band_expanders.iter().zip(&p.bands) {
                    assert_eq!(e.clamped(), *e, "{ctx}: expander in range");
                    assert!(
                        e.ratio <= 2.0 && e.floor_db <= 6.0,
                        "{ctx}: modest expander"
                    );
                    assert!(e.threshold_db < b.threshold_db - 20.0, "{ctx}");
                }
                assert!(p.bass_clipper_enabled, "{ctx}");
                assert!(p.bass_clip_threshold_db > p.bands[0].threshold_db, "{ctx}");
                assert!(r(p.bass_clip_threshold_db, -40.0, 0.0), "{ctx}");
            }
        }
        // Speech is gentler than programme.
        let s = BroadcastParams::preset(BroadcastPreset::Speech, 4);
        let e = BroadcastParams::preset(BroadcastPreset::EbuR128, 4);
        assert!(s.bands[0].ratio < e.bands[0].ratio);
        assert!(s.agc_gate_lufs > e.agc_gate_lufs, "faster AGC gate");
    }

    /// Targets and ceilings are those of the standards.
    #[test]
    fn broadcast_preset_targets() {
        for (preset, t, c) in [
            (BroadcastPreset::EbuR128, -23.0, -1.0),
            (BroadcastPreset::AtscA85, -24.0, -2.0),
            (BroadcastPreset::Streaming, -16.0, -1.0),
            (BroadcastPreset::Speech, -18.0, -1.0),
        ] {
            let p = BroadcastParams::preset(preset, 4);
            assert_eq!((p.target_loudness_lufs, p.true_peak_ceiling_dbtp), (t, c));
        }
    }

    /// The default preset (`ebu-r128`, 4 bands) matches the element's
    /// documented default of every top-level property.
    #[test]
    fn ebu_r128_matches_element_defaults() {
        let p = BroadcastParams::preset(BroadcastPreset::EbuR128, 4);
        assert_eq!(p.input_gain_db, 0.0);
        assert_eq!(p.target_loudness_lufs, -23.0);
        assert_eq!(p.true_peak_ceiling_dbtp, -1.0);
        assert!(p.agc_enabled && p.multiband_enabled && p.limiter_enabled);
        assert_eq!((p.agc_max_gain_db, p.agc_max_cut_db), (24.0, 24.0));
        assert_eq!((p.agc_rate_up_db_s, p.agc_rate_down_db_s), (1.0, 3.0));
        assert_eq!(p.agc_gate_lufs, -50.0);
        assert_eq!(p.multiband_drive_db, 0.0);
        assert_eq!(p.crossover_hz, [120.0, 800.0, 4000.0, 10_000.0]);
        assert_eq!(p.limiter_lookahead_ms, 5.0);
    }

    /// `apply_preset` leaves `enabled`, `num_bands` and the lookahead alone
    /// and uses the current band count.
    #[test]
    fn apply_preset_keeps_excluded_fields() {
        let mut p = BroadcastParams::preset(BroadcastPreset::EbuR128, 4);
        p.enabled = false;
        p.num_bands = 2;
        p.limiter_lookahead_ms = 12.0;
        p.input_gain_db = 5.0;
        p.apply_preset(BroadcastPreset::Speech);
        assert!(!p.enabled);
        assert_eq!(p.num_bands, 2);
        assert_eq!(p.limiter_lookahead_ms, 12.0);
        let mut q = BroadcastParams::preset(BroadcastPreset::Speech, 2);
        q.enabled = false;
        q.limiter_lookahead_ms = 12.0;
        assert_eq!(p, q);
        assert_eq!(p.crossover_hz[0], 200.0);
        assert_eq!((p.bands[1].attack_ms, p.bands[1].release_ms), (10.0, 150.0));
        // Inactive bands get the 5-band values of their index.
        assert_eq!((p.bands[4].attack_ms, p.bands[4].release_ms), (3.0, 80.0));
    }
}
