//! Long programme runs of the broadcast chain per preset, moved here from
//! `strom-dsp/tests/broadcast.rs` (60 … 90 s of programme at −35, −23 and
//! −10 LUFS input each). All signals are synthetic with fixed seeds.
//!
//! - [`acceptance`]: music-like programme, integrated loudness within ±1 LU
//!   of the target, true peak ≤ ceiling + 0.1 dB.
//! - [`peaky_acceptance`]: hot, peaky sport-like programme with a 10 s
//!   pause: the same bounds, and no short-term overshoot after the pause.
//! - [`dense_clicks`]: a click every 300 ms: on target, the limiter barely
//!   touches the clicks, and the spike guard is what makes it pass.
//!
//! Per change, the harness baseline guards the same readings at the
//! target level, per preset: `loudness_error_lu` and `max_true_peak_dbtp`
//! on `music` and `peaky`; on `peaky_clicks_300ms` (and its low-latency
//! twin) also `limiter_gr_peaks_mean_db` and the spike guard's numbers.
//! The default suite keeps a short pause test
//! (`broadcast::no_overshoot_after_a_pause`). Only these runs check other
//! input levels (the AGC has to travel) and a 10 s pause.

use strom_dsp::broadcast::{BroadcastParams, BroadcastProcessor, SILENCE_DB};
use strom_dsp::meter::{ChannelLayout, LoudnessConfig, LoudnessMeter};
use strom_dsp::Processor;
use strom_dsp_harness::agc_support::{scale_to_lufs, Looper};
use strom_dsp_harness::programme_support;

const FS: f32 = 48_000.0;
/// Frames per buffer (10 ms at 48 kHz).
const BLOCK: usize = 480;

fn processor(fs: f32, ch: usize, p: &BroadcastParams) -> BroadcastProcessor {
    let mut bp = BroadcastProcessor::new();
    bp.set_params(p);
    bp.configure(fs, ch);
    bp
}

fn db(x: f64) -> f64 {
    20.0 * x.max(1e-30).log10()
}

/// 60 s programmes per preset (`strom_dsp::presets`, at the default 4 bands)
/// at three input levels: the integrated loudness (excluding the first
/// 10 s) lands within ±1 LU of the preset target, and the output true peak
/// (BS.1770, `ebur128`) stays at or below the ceiling + 0.1 dB.
///
/// With these gentle presets this programme's output true peak stays 2 …
/// 9 dB below the ceiling, so it does not stress the true-peak limiter (its
/// own tests in `tests/truepeak.rs` do); [`peaky_acceptance`] does.
mod acceptance {
    use super::*;
    use strom_dsp::presets::BroadcastPreset;

    const SECONDS: usize = 60;
    const SKIP_SECONDS: usize = 10;
    /// Allowed true-peak overshoot (0.1 dB, the true-peak limiter's guarantee).
    const TP_MARGIN_DB: f64 = 0.1;

    /// The preset with this nick, at the default band count.
    fn preset(name: &str) -> BroadcastParams {
        BroadcastParams::preset(BroadcastPreset::from_nick(name).unwrap(), 4)
    }

    /// The music-like programme of [`programme_support::music_programme`].
    fn programme(ch: usize, fs: f64) -> Vec<f32> {
        programme_support::music_programme(ch, fs)
    }

    /// Run the three input levels; assert ±1 LU and true peak ≤ ceiling +
    /// `tp_margin_db`.
    fn check(name: &str, tp_margin_db: f64) {
        let ch = 2;
        let fs = FS as f64;
        let p = preset(name);
        let src = programme(ch, fs);
        for input_lufs in [-35.0, -23.0, -10.0] {
            let mut x = src.clone();
            scale_to_lufs(&mut x, ch, fs, input_lufs);
            let mut bp = processor(FS, ch, &p);
            let mut meter = LoudnessMeter::new(&LoudnessConfig {
                sample_rate: FS,
                channels: ch,
                layout: ChannelLayout::Unknown,
                true_peak: true,
            })
            .unwrap();
            let mut looper = Looper::new(&x);
            let mut buf = vec![0.0f32; BLOCK * ch];
            let blocks = SECONDS * FS as usize / BLOCK;
            let skip = SKIP_SECONDS * FS as usize / BLOCK;
            for i in 0..blocks {
                looper.fill(&mut buf);
                bp.process(&mut buf);
                if i >= skip {
                    meter.process(&buf);
                }
            }
            let integrated = meter.integrated();
            let tp = db((0..ch).map(|c| meter.true_peak(c)).fold(0.0, f64::max));
            println!(
                "{name} at {input_lufs} LUFS: integrated {integrated:.2} LUFS (target {}), \
                 true peak {tp:.2} dBTP (ceiling {})",
                p.target_loudness_lufs, p.true_peak_ceiling_dbtp
            );
            assert!(
                (integrated - p.target_loudness_lufs as f64).abs() <= 1.0,
                "{name} at {input_lufs} LUFS: integrated {integrated:.2} LUFS, target {}",
                p.target_loudness_lufs
            );
            assert!(
                tp <= p.true_peak_ceiling_dbtp as f64 + tp_margin_db,
                "{name} at {input_lufs} LUFS: true peak {tp:.2} dBTP, ceiling {}",
                p.true_peak_ceiling_dbtp
            );
        }
    }

    #[test]
    #[ignore = "long programme run; runs in the extended workflow"]
    fn preset_ebu_r128() {
        check("ebu-r128", TP_MARGIN_DB);
    }

    #[test]
    #[ignore = "long programme run; runs in the extended workflow"]
    fn preset_atsc_a85() {
        check("atsc-a85", TP_MARGIN_DB);
    }

    #[test]
    #[ignore = "long programme run; runs in the extended workflow"]
    fn preset_streaming() {
        check("streaming", TP_MARGIN_DB);
    }

    #[test]
    #[ignore = "long programme run; runs in the extended workflow"]
    fn preset_speech() {
        check("speech", TP_MARGIN_DB);
    }
}

/// A hot, peaky, sport-like programme through the full chain, per preset at
/// −35, −23 and −10 LUFS input:
///
/// - integrated loudness (from 10 s to 60 s) within ±1 LU of the target;
/// - output true peak ≤ ceiling + 0.1 dB over the whole run;
/// - no overshoot after a 10 s pause: from the end of the pause on, the
///   output short-term loudness never exceeds the target by more than 1 LU.
///
/// The multiband and the limiter take a large share of this programme's
/// level (dense hits and clicks), so it checks that the chain makes up what
/// the stages after the AGC remove. Both measured with `ebur128`.
mod peaky_acceptance {
    use super::*;
    use strom_dsp::presets::BroadcastPreset;

    /// Programme before the pause (s); integrated loudness is measured from
    /// `SKIP_SECONDS` to here.
    const PROGRAMME_SECONDS: usize = 60;
    const SKIP_SECONDS: usize = 10;
    const PAUSE_SECONDS: usize = 10;
    /// Programme after the pause (s).
    const RETURN_SECONDS: usize = 20;
    const TP_MARGIN_DB: f64 = 0.1;
    const LOUDNESS_TOL_LU: f64 = 1.0;
    const OVERSHOOT_TOL_LU: f64 = 1.0;

    fn preset(name: &str) -> BroadcastParams {
        BroadcastParams::preset(BroadcastPreset::from_nick(name).unwrap(), 4)
    }

    /// A sport-like programme (a ~10.9 s loop, fixed seed): a pink-noise
    /// crowd bed; percussive hits every ~250 ms (a decaying white-noise
    /// burst plus a decaying 60 Hz thump); a 5 ms fs/4 sine burst at 45°
    /// phase (sample peaks 3 dB under its true peak) every ~2.3 s; and a
    /// single-sample click every ~1.1 s.
    fn programme(ch: usize, fs: f64) -> Vec<f32> {
        programme_with_clicks(ch, fs, 1.1)
    }

    /// [`programme`] with a single-sample click every `click_s` seconds
    /// instead of every 1.1 s (same bed, hits and bursts).
    pub(super) fn programme_with_clicks(ch: usize, fs: f64, click_s: f64) -> Vec<f32> {
        programme_support::peaky_programme(ch, fs, click_s)
    }

    /// Run one preset at three input levels and check all three bounds;
    /// every failed check is reported.
    fn check(name: &str) {
        let ch = 2;
        let fs = FS as f64;
        let p = preset(name);
        let target = p.target_loudness_lufs as f64;
        let ceiling = p.true_peak_ceiling_dbtp as f64;
        let src = programme(ch, fs);
        let mut failures = Vec::new();
        for input_lufs in [-35.0, -23.0, -10.0] {
            let mut x = src.clone();
            scale_to_lufs(&mut x, ch, fs, input_lufs);
            let mut bp = processor(FS, ch, &p);
            let cfg = LoudnessConfig {
                sample_rate: FS,
                channels: ch,
                layout: ChannelLayout::Unknown,
                true_peak: true,
            };
            // Integrated loudness over 10 … 60 s; short-term and true peak
            // over the whole run.
            let mut integrated_meter = LoudnessMeter::new(&cfg).unwrap();
            let mut meter = LoudnessMeter::new(&cfg).unwrap();
            let mut looper = Looper::new(&x);
            let mut buf = vec![0.0f32; BLOCK * ch];
            let per_s = FS as usize / BLOCK;
            let skip = SKIP_SECONDS * per_s;
            let pause_start = PROGRAMME_SECONDS * per_s;
            let pause_end = pause_start + PAUSE_SECONDS * per_s;
            let end = pause_end + RETURN_SECONDS * per_s;
            let mut max_short_term_after_pause = SILENCE_DB;
            for i in 0..end {
                if (pause_start..pause_end).contains(&i) {
                    buf.fill(0.0);
                } else {
                    looper.fill(&mut buf);
                }
                bp.process(&mut buf);
                meter.process(&buf);
                if (skip..pause_start).contains(&i) {
                    integrated_meter.process(&buf);
                }
                if i >= pause_end {
                    max_short_term_after_pause = max_short_term_after_pause.max(meter.short_term());
                }
            }
            let integrated = integrated_meter.integrated();
            let tp = db((0..ch).map(|c| meter.true_peak(c)).fold(0.0, f64::max));
            println!(
                "peaky {name} at {input_lufs} LUFS: integrated {integrated:.2} LUFS \
                 (target {target}), true peak {tp:.2} dBTP (ceiling {ceiling}), \
                 max short-term after pause {max_short_term_after_pause:.2} LUFS"
            );
            if (integrated - target).abs() > LOUDNESS_TOL_LU {
                failures.push(format!(
                    "{name} at {input_lufs} LUFS: integrated {integrated:.2} LUFS, target {target}"
                ));
            }
            if tp > ceiling + TP_MARGIN_DB {
                failures.push(format!(
                    "{name} at {input_lufs} LUFS: true peak {tp:.2} dBTP, ceiling {ceiling}"
                ));
            }
            if max_short_term_after_pause > target + OVERSHOOT_TOL_LU {
                failures.push(format!(
                    "{name} at {input_lufs} LUFS: short-term {max_short_term_after_pause:.2} \
                     LUFS after the pause, target {target}"
                ));
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    #[test]
    #[ignore = "long programme run; runs in the extended workflow"]
    fn preset_ebu_r128() {
        check("ebu-r128");
    }

    #[test]
    #[ignore = "long programme run; runs in the extended workflow"]
    fn preset_atsc_a85() {
        check("atsc-a85");
    }

    #[test]
    #[ignore = "long programme run; runs in the extended workflow"]
    fn preset_streaming() {
        check("streaming");
    }

    #[test]
    #[ignore = "long programme run; runs in the extended workflow"]
    fn preset_speech() {
        check("speech");
    }
}

/// The [`peaky_acceptance`] programme with a click every 300 ms instead of
/// every 1.1 s, through the `streaming` preset (4 bands) at −35, −23 and
/// −10 LUFS input:
///
/// - integrated output loudness (10 … 60 s) within ±1 LU of the target;
/// - limiter gain reduction on the clicks below 3 dB: for each click from
///   10 s on, the largest reduction from the limiter lookahead (+1 ms)
///   before it to 2 ms after it (in output time), averaged over the clicks.
///   The largest single value is printed: a click that lands on a hit
///   also gets the hit's own reduction.
///
/// The clicks carry much of the input's measured loudness. Before the
/// spike guard and the final clipper the limiter took about 13 dB off
/// them, the output-referenced trim ran into its +6 dB bound, and the
/// integrated output missed the target by 1.1 … 1.7 LU (−17.69, −17.15 and
/// −17.11 LUFS at the three input levels). With the spike guard off today
/// the final clipper takes most of each click and the output lands within
/// 1 LU (−16.44 … −16.32 LUFS), but the limiter still takes 4.0 … 4.1 dB
/// off the clicks; with it on, 0.2 dB (−16.14 … −16.09 LUFS).
///
/// Guard: the same run with the spike guard off must fail one of the two
/// bounds (today the limiter reduction on the clicks), so the test fails
/// if the spike guard stops catching the clicks.
mod dense_clicks {
    use super::*;
    use strom_dsp::presets::BroadcastPreset;

    const SECONDS: usize = 60;
    const SKIP_SECONDS: usize = 10;
    const CLICK_S: f64 = 0.3;
    const LOUDNESS_TOL_LU: f64 = 1.0;
    const CLICK_GR_DB: f64 = 3.0;
    /// Buffer length (1 ms), so the limiter meter is read every 1 ms.
    const MS: usize = 48;

    /// (integrated LUFS from 10 s, mean and max limiter GR on clicks).
    fn run_case(p: &BroadcastParams, x: &[f32], ch: usize) -> (f64, f64, f64) {
        let mut bp = processor(FS, ch, p);
        let lat = bp.latency_frames();
        let mut meter = LoudnessMeter::new(&LoudnessConfig {
            sample_rate: FS,
            channels: ch,
            layout: ChannelLayout::Unknown,
            true_peak: false,
        })
        .unwrap();
        let mut looper = Looper::new(x);
        let mut buf = vec![0.0f32; MS * ch];
        let blocks = SECONDS * FS as usize / MS;
        let skip = SKIP_SECONDS * FS as usize / MS;
        let mut gr = Vec::with_capacity(blocks);
        for i in 0..blocks {
            looper.fill(&mut buf);
            bp.process(&mut buf);
            gr.push(bp.take_meters().limiter_gain_reduction_db);
            if i >= skip {
                meter.process(&buf);
            }
        }
        // Click frames over the whole run (the loop repeats every 2^19).
        let fs = FS as f64;
        let one = programme_support::click_frames(fs, CLICK_S);
        let la = (p.limiter_lookahead_ms as f64 * 0.001 * fs).round() as usize;
        let mut per_click = Vec::new();
        let mut base = 0;
        while base < blocks * MS {
            for &m in &one {
                let at = base + m + lat;
                if at < skip * MS || at + 2 * MS >= blocks * MS {
                    continue;
                }
                let from = (at - la - MS) / MS;
                let to = (at + 2 * MS) / MS;
                per_click.push(gr[from..=to].iter().fold(0.0f64, |a, &b| a.max(b)));
            }
            base += programme_support::PROGRAMME_FRAMES;
        }
        let mean = per_click.iter().sum::<f64>() / per_click.len() as f64;
        let max = per_click.iter().fold(0.0f64, |a, &b| a.max(b));
        (meter.integrated(), mean, max)
    }

    #[test]
    #[ignore = "long programme run; runs in the extended workflow"]
    fn streaming_with_clicks_every_300_ms() {
        let ch = 2;
        let fs = FS as f64;
        let preset = BroadcastParams::preset(BroadcastPreset::Streaming, 4);
        assert!(preset.spike_guard_enabled);
        let target = preset.target_loudness_lufs as f64;
        let src = peaky_acceptance::programme_with_clicks(ch, fs, CLICK_S);
        let mut failures = Vec::new();
        for input_lufs in [-35.0, -23.0, -10.0] {
            let mut x = src.clone();
            scale_to_lufs(&mut x, ch, fs, input_lufs);
            // Standard (0.5 ms) and low-latency (no) spike guard lookahead.
            for (mode, la) in [("standard", 0.5f32), ("low latency", 0.0)] {
                let p = BroadcastParams {
                    spike_guard_lookahead_ms: la,
                    ..preset
                };
                let (integrated, gr_mean, gr_max) = run_case(&p, &x, ch);
                println!(
                    "clicks every 300 ms, streaming at {input_lufs} LUFS, {mode}: integrated \
                     {integrated:.2} LUFS (target {target}), limiter GR on clicks mean \
                     {gr_mean:.2} dB, max {gr_max:.2} dB"
                );
                if (integrated - target).abs() > LOUDNESS_TOL_LU {
                    failures.push(format!(
                        "at {input_lufs} LUFS, {mode}: integrated {integrated:.2} LUFS, \
                         target {target}"
                    ));
                }
                if gr_mean >= CLICK_GR_DB {
                    failures.push(format!(
                        "at {input_lufs} LUFS, {mode}: limiter GR on clicks {gr_mean:.2} dB"
                    ));
                }
            }
            let off = BroadcastParams {
                spike_guard_enabled: false,
                ..preset
            };
            let (integrated_off, gr_mean_off, _) = run_case(&off, &x, ch);
            println!(
                "clicks every 300 ms, streaming at {input_lufs} LUFS, spike guard off: \
                 {integrated_off:.2} LUFS, GR mean {gr_mean_off:.2} dB"
            );
            if (integrated_off - target).abs() <= LOUDNESS_TOL_LU && gr_mean_off < CLICK_GR_DB {
                failures.push(format!(
                    "at {input_lufs} LUFS: the case passes without the spike guard \
                     ({integrated_off:.2} LUFS, {gr_mean_off:.2} dB): not a guard"
                ));
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }
}
