//! Generic metrics over interleaved `f32` signals.
//!
//! Each function takes plain sample slices (and, where needed, a frame
//! offset or a list of event frames), so any stage can be measured: hand it
//! the stage's input and output, or a meter series.
//!
//! | Metric | Function | Reference |
//! |---|---|---|
//! | integrated loudness, LRA, max true peak | [`loudness`] | ITU-R BS.1770 / EBU Tech 3342, through `ebur128` |
//! | gain reduction on peaks / on programme | [`split_by_marks`] | own definition (below) |
//! | clip depth | [`clip_depth`] | own definition (below) |
//! | aliasing spur | [`alias_spur_db`] | non-harmonic spectral spur of a tone |
//! | SMPTE IMD | [`smpte_imd_db`] | SMPTE RP 120 (60 Hz + 7 kHz, 4:1) |
//! | CCIF IMD | [`ccif_imd_db`] | CCIF twin tone (19 + 20 kHz) |
//! | sidebands | [`sideband_dbc`] | sideband level around a carrier |
//! | NMR | [`nmr`] | Brandenburg 1987 (NMR), Zwicker & Fastl (Bark bands, spreading) |
//!
//! The masking model in [`nmr`] is a **test metric only**. Nothing in the
//! processing or control path of strom-dsp may use it.

use strom_dsp::meter::{ChannelLayout, LoudnessConfig, LoudnessMeter};

use crate::spectrum::Spectrum;

/// Value used for "nothing measured" in dB (never −∞ or NaN).
pub const NONE_DB: f64 = -200.0;

fn db10(x: f64) -> f64 {
    if x > 0.0 {
        (10.0 * x.log10()).max(NONE_DB)
    } else {
        NONE_DB
    }
}

// ---------------------------------------------------------------------------
// Loudness and true peak.

/// Loudness numbers of a signal (BS.1770 / EBU Tech 3342, `ebur128`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Loudness {
    /// Gated integrated loudness (LUFS).
    pub integrated_lufs: f64,
    /// Loudness range (LU).
    pub lra_lu: f64,
    /// Largest true peak over all channels (dBTP).
    pub max_true_peak_dbtp: f64,
}

/// [`Loudness`] of interleaved `x` (`ch` channels) at `fs`.
pub fn loudness(x: &[f32], ch: usize, fs: f64) -> Loudness {
    let mut m = LoudnessMeter::new(&LoudnessConfig {
        sample_rate: fs as f32,
        channels: ch,
        layout: ChannelLayout::Unknown,
        true_peak: true,
    })
    .expect("loudness meter");
    m.process(x);
    let tp = (0..ch).map(|c| m.true_peak(c)).fold(0.0, f64::max);
    Loudness {
        integrated_lufs: m.integrated().max(NONE_DB),
        lra_lu: if m.loudness_range().is_finite() {
            m.loudness_range()
        } else {
            0.0
        },
        max_true_peak_dbtp: if tp > 0.0 {
            (20.0 * tp.log10()).max(NONE_DB)
        } else {
            NONE_DB
        },
    }
}

// ---------------------------------------------------------------------------
// Gain reduction split into "on peaks" and "on programme".

/// Mean and maximum of a series, split by event windows.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplitStats {
    /// Mean over events of the largest value in each event window
    /// (0 without events).
    pub peaks_mean: f64,
    /// Largest value in any event window (0 without events).
    pub peaks_max: f64,
    /// Mean over the blocks outside every exclusion window.
    pub programme_mean: f64,
    /// Largest value outside every exclusion window.
    pub programme_max: f64,
}

/// Split a per-block series (`series[i]` covers frames
/// `i·block .. (i+1)·block`) by event frames `marks` (in the series' time
/// base).
///
/// - **On peaks:** for each mark `m`, the largest value over the blocks
///   touching `m − before .. m + after`.
/// - **On programme:** every block outside `m − before .. m + exclude` for
///   all marks (`exclude ≥ after` also leaves out the recovery after an
///   event).
pub fn split_by_marks(
    series: &[f64],
    block: usize,
    marks: &[usize],
    before: usize,
    after: usize,
    exclude: usize,
) -> SplitStats {
    let block = block.max(1);
    let n = series.len();
    let blocks = |from: usize, to: usize| (from / block).min(n)..(to / block + 1).min(n);
    let mut excluded = vec![false; n];
    let (mut sum, mut max, mut count) = (0.0, 0.0f64, 0usize);
    for &m in marks {
        let r = blocks(m.saturating_sub(before), m + after);
        if r.is_empty() {
            continue;
        }
        let v = series[r].iter().fold(0.0f64, |a, &b| a.max(b));
        sum += v;
        max = max.max(v);
        count += 1;
        for e in &mut excluded[blocks(m.saturating_sub(before), m + exclude)] {
            *e = true;
        }
    }
    let rest: Vec<f64> = series
        .iter()
        .zip(&excluded)
        .filter(|(_, &e)| !e)
        .map(|(&v, _)| v)
        .collect();
    SplitStats {
        peaks_mean: if count > 0 { sum / count as f64 } else { 0.0 },
        peaks_max: max,
        programme_mean: if rest.is_empty() {
            0.0
        } else {
            rest.iter().sum::<f64>() / rest.len() as f64
        },
        programme_max: rest.iter().fold(0.0f64, |a, &b| a.max(b)),
    }
}

// ---------------------------------------------------------------------------
// Clip depth.

/// What a clipper (or any stage that only removes peaks) took away.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClipDepth {
    /// Removed energy relative to the programme energy over the whole
    /// signal (dB).
    pub overall_db: f64,
    /// Median, 90th and 99th percentile and maximum of the per-block depth
    /// (dB) over the 10 ms blocks where anything was removed
    /// ([`NONE_DB`] if none).
    pub p50_db: f64,
    pub p90_db: f64,
    pub p99_db: f64,
    pub max_db: f64,
    /// Share of 10 ms blocks where anything was removed (0 … 1).
    pub blocks_share: f64,
    /// Share of samples that differ from the reference (0 … 1).
    pub samples_share: f64,
}

/// Clip depth of `processed` against `reference` (same length, same
/// alignment: pass the stage input delayed by the stage latency).
///
/// Per 10 ms block: `10·log10(Σ(reference − processed)² / Σ reference²)`
/// over all channels. A sample counts as touched when it differs from the
/// reference by more than 1e-9.
pub fn clip_depth(reference: &[f32], processed: &[f32], ch: usize, fs: f64) -> ClipDepth {
    let block = ((0.01 * fs).round() as usize).max(1) * ch;
    let mut depths = Vec::new();
    let (mut removed_all, mut energy_all) = (0.0f64, 0.0f64);
    let (mut touched, mut total, mut blocks) = (0usize, 0usize, 0usize);
    for (r, p) in reference.chunks(block).zip(processed.chunks(block)) {
        let (mut removed, mut energy, mut hit) = (0.0f64, 0.0f64, 0usize);
        for (&a, &b) in r.iter().zip(p) {
            let d = a as f64 - b as f64;
            removed += d * d;
            energy += a as f64 * a as f64;
            if d.abs() > 1e-9 {
                hit += 1;
            }
            total += 1;
        }
        blocks += 1;
        touched += hit;
        removed_all += removed;
        energy_all += energy;
        if hit > 0 {
            depths.push(db10(removed / energy.max(1e-30)));
        }
    }
    depths.sort_by(f64::total_cmp);
    let pct = |q: f64| {
        if depths.is_empty() {
            NONE_DB
        } else {
            depths[((depths.len() - 1) as f64 * q).round() as usize]
        }
    };
    ClipDepth {
        overall_db: db10(removed_all / energy_all.max(1e-30)),
        p50_db: pct(0.5),
        p90_db: pct(0.9),
        p99_db: pct(0.99),
        max_db: pct(1.0),
        blocks_share: depths.len() as f64 / blocks.max(1) as f64,
        samples_share: touched as f64 / total.max(1) as f64,
    }
}

// ---------------------------------------------------------------------------
// Spectral metrics.

/// FFT length used by the spectral metrics (1.37 s at 48 kHz, 0.73 Hz
/// bins).
pub const FFT_LEN: usize = 1 << 16;

/// Largest non-harmonic spur (dB re full-scale sine) in the spectrum of
/// channel 0 of `x` from frame `from` ([`FFT_LEN`] frames), for a tone at
/// `f0`. Left out: below 20 Hz, and `±guard_hz` around every harmonic
/// `k·f0` below Nyquist (gain modulation sidebands of a slow stage are not
/// aliasing). Everything else that a memoryless nonlinearity makes from a
/// tone at `f0` is an alias, folded back from above Nyquist.
pub fn alias_spur_db(x: &[f32], ch: usize, fs: f64, from: usize, f0: f64, guard_hz: f64) -> f64 {
    let s = Spectrum::of(x, ch, 0, fs, from, FFT_LEN);
    let nyq = 0.5 * fs;
    s.max_db_where(|f| {
        if f < 20.0 {
            return false;
        }
        let mut k = 1.0;
        while k * f0 < nyq {
            if (f - k * f0).abs() <= guard_hz {
                return false;
            }
            k += 1.0;
        }
        true
    })
}

/// SMPTE RP 120 intermodulation of the 60 Hz + 7 kHz (4:1) pair, in dB:
/// the RMS sum of the sidebands `7 kHz ± n·60 Hz` (n = 1, 2, 3) relative to
/// the 7 kHz carrier.
pub fn smpte_imd_db(x: &[f32], ch: usize, fs: f64, from: usize) -> f64 {
    let s = Spectrum::of(x, ch, 0, fs, from, FFT_LEN);
    let carrier = 10f64.powf(s.level_db(7_000.0) / 20.0);
    let side: f64 = (1..=3)
        .flat_map(|n| [7_000.0 - 60.0 * n as f64, 7_000.0 + 60.0 * n as f64])
        .map(|f| 10f64.powf(s.level_db(f) / 10.0))
        .sum();
    db10(side / (carrier * carrier).max(1e-30))
}

/// CCIF twin-tone intermodulation of the 19 + 20 kHz pair, in dB: the
/// largest of the difference tone (1 kHz) and the third-order products
/// (18 and 21 kHz) relative to the mean amplitude of the two tones.
pub fn ccif_imd_db(x: &[f32], ch: usize, fs: f64, from: usize) -> f64 {
    let s = Spectrum::of(x, ch, 0, fs, from, FFT_LEN);
    let a =
        0.5 * (10f64.powf(s.level_db(19_000.0) / 20.0) + 10f64.powf(s.level_db(20_000.0) / 20.0));
    let products = [1_000.0, 18_000.0, 21_000.0]
        .into_iter()
        .filter(|&f| f < 0.5 * fs)
        .map(|f| s.level_db(f))
        .fold(NONE_DB, f64::max);
    products - 20.0 * a.max(1e-15).log10()
}

/// Largest sideband level (dBc) at `carrier ± offset_hz` relative to the
/// carrier.
pub fn sideband_dbc(
    x: &[f32],
    ch: usize,
    fs: f64,
    from: usize,
    carrier: f64,
    offset_hz: f64,
) -> f64 {
    let s = Spectrum::of(x, ch, 0, fs, from, FFT_LEN);
    let c = s.level_db(carrier);
    s.level_db(carrier - offset_hz)
        .max(s.level_db(carrier + offset_hz))
        - c
}

// ---------------------------------------------------------------------------
// Noise-to-mask ratio.

/// NMR frame length (frames) and hop.
const NMR_FRAME: usize = 2048;
const NMR_HOP: usize = 1024;
/// Distance of the masked threshold below the (spread) masker energy, dB.
const MASK_OFFSET_DB: f64 = 10.0;
/// Spreading slopes (dB per Bark) towards higher and lower bands.
const SPREAD_UP_DB: f64 = 10.0;
const SPREAD_DOWN_DB: f64 = 25.0;
/// Bark bands 0 … 24.
const BARK_BANDS: usize = 25;
/// Frames whose reference is quieter than this (dBFS, mean square) are
/// left out.
const NMR_MIN_LEVEL_DB: f64 = -70.0;
/// Threshold floor per band (energy relative to a full-scale sine's frame
/// energy, dB): a stand-in for the threshold in quiet.
const MASK_FLOOR_DB: f64 = -110.0;

/// NMR numbers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Nmr {
    /// `10·log10` of the mean over frames and bands of noise/mask (dB).
    pub mean_db: f64,
    /// Largest per-frame NMR (dB).
    pub max_frame_db: f64,
}

/// Critical-band rate (Bark) of `f` Hz (Zwicker & Terhardt 1980, as given in
/// Zwicker & Fastl, *Psychoacoustics*).
pub fn bark(f: f64) -> f64 {
    13.0 * (0.00076 * f).atan() + 3.5 * (f / 7_500.0).powi(2).atan()
}

/// Noise-to-mask ratio of `processed` against `reference` (both
/// interleaved, `ch` channels, aligned), from frame `from`, mean over
/// channels.
///
/// The reference should carry the processor's linear behaviour (its delay
/// and any all-pass phase, e.g. the input through the same chain with every
/// dynamic stage neutral), so that only what the dynamics did is compared.
///
/// Own simple model (Brandenburg 1987, Zwicker & Fastl), **test metric
/// only**:
/// 1. Hann frames of 2048, hop 1024; frames whose reference is below
///    −70 dBFS are left out.
/// 2. Power spectra summed into integer Bark bands 0 … 24. Per band the
///    reference is gain matched to the output (least-squares real gain over
///    the band's bins), so a slow per-band gain (AGC, multiband) is not
///    counted as error; the error is what remains,
///    `Σ |Y − g_b·R|²` over the band.
/// 3. The gain-matched reference band energy, spread over bands
///    (−10 dB/Bark upwards, −25 dB/Bark downwards from each masker band)
///    and lowered by 10 dB, is the masked threshold, with a floor 110 dB
///    under a full-scale sine.
/// 4. Frame NMR = mean over bands of error/threshold; the result is the
///    mean over frames, in dB.
pub fn nmr(reference: &[f32], processed: &[f32], ch: usize, fs: f64, from: usize) -> Nmr {
    use realfft::RealFftPlanner;
    let frames = (reference.len().min(processed.len())) / ch;
    let mut planner = RealFftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(NMR_FRAME);
    let win: Vec<f64> = (0..NMR_FRAME)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / NMR_FRAME as f64).cos())
        .collect();
    let wsum2: f64 = win.iter().map(|w| w * w).sum();
    let band_of: Vec<usize> = (0..=NMR_FRAME / 2)
        .map(|k| {
            let f = k as f64 * fs / NMR_FRAME as f64;
            (bark(f).floor() as usize).min(BARK_BANDS - 1)
        })
        .collect();
    let spread: Vec<Vec<f64>> = (0..BARK_BANDS)
        .map(|b| {
            (0..BARK_BANDS)
                .map(|j| {
                    let dz = b as f64 - j as f64;
                    let att = if dz >= 0.0 {
                        SPREAD_UP_DB * dz
                    } else {
                        SPREAD_DOWN_DB * -dz
                    };
                    10f64.powf(-(att + MASK_OFFSET_DB) / 10.0)
                })
                .collect()
        })
        .collect();
    // Band energies are normalised to the frame mean square (Parseval with
    // the window power), so a full-scale sine is ½.
    let norm = 2.0 / (NMR_FRAME as f64 * wsum2);
    let floor = 0.5 * 10f64.powf(MASK_FLOOR_DB / 10.0);
    let min_level = 10f64.powf(NMR_MIN_LEVEL_DB / 10.0);

    let mut ratio_sum = 0.0;
    let mut n_frames = 0usize;
    let mut max_frame = NONE_DB;
    let mut rb = fft.make_input_vec();
    let mut pb = fft.make_input_vec();
    let mut rs = fft.make_output_vec();
    let mut ps = fft.make_output_vec();
    for c in 0..ch {
        let mut start = from;
        while start + NMR_FRAME <= frames {
            let mut rr = 0.0f64;
            for i in 0..NMR_FRAME {
                rb[i] = reference[(start + i) * ch + c] as f64 * win[i];
                pb[i] = processed[(start + i) * ch + c] as f64 * win[i];
                rr += rb[i] * rb[i];
            }
            if rr / wsum2 < min_level {
                start += NMR_HOP;
                continue;
            }
            fft.process(&mut rb, &mut rs).expect("FFT");
            fft.process(&mut pb, &mut ps).expect("FFT");
            // Per band: Σ|R|², Σ Re(Y·R*), Σ|Y|².
            let mut r2 = [0.0f64; BARK_BANDS];
            let mut yr = [0.0f64; BARK_BANDS];
            let mut y2 = [0.0f64; BARK_BANDS];
            for k in 1..=NMR_FRAME / 2 {
                let b = band_of[k];
                r2[b] += rs[k].norm_sqr();
                yr[b] += (ps[k] * rs[k].conj()).re;
                y2[b] += ps[k].norm_sqr();
            }
            let mut mb = [0.0f64; BARK_BANDS];
            let mut nb = [0.0f64; BARK_BANDS];
            for b in 0..BARK_BANDS {
                // Least squares: g = Σ Re(Y R*) / Σ|R|²; the residual
                // energy is Σ|Y|² − g·Σ Re(Y R*) (≥ 0 up to rounding).
                let g = if r2[b] > 0.0 { yr[b] / r2[b] } else { 0.0 };
                mb[b] = g * g * r2[b] * norm;
                nb[b] = (y2[b] - g * yr[b]).max(0.0) * norm;
            }
            let mut frame = 0.0;
            for b in 0..BARK_BANDS {
                let t: f64 = (0..BARK_BANDS).map(|j| mb[j] * spread[b][j]).sum();
                frame += nb[b] / t.max(floor);
            }
            frame /= BARK_BANDS as f64;
            ratio_sum += frame;
            n_frames += 1;
            max_frame = max_frame.max(db10(frame));
            start += NMR_HOP;
        }
    }
    Nmr {
        mean_db: if n_frames > 0 {
            db10(ratio_sum / n_frames as f64)
        } else {
            NONE_DB
        },
        max_frame_db: max_frame,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn tone(frames: usize, fs: f64, parts: &[(f64, f64)]) -> Vec<f32> {
        (0..frames)
            .map(|n| {
                let t = n as f64 / fs;
                parts
                    .iter()
                    .map(|&(f, a)| a * (2.0 * PI * f * t).sin())
                    .sum::<f64>() as f32
            })
            .collect()
    }

    #[test]
    fn split_by_marks_separates_events() {
        let mut s = vec![1.0; 100];
        s[50] = 9.0;
        s[51] = 4.0;
        let st = split_by_marks(&s, 10, &[505], 10, 10, 30);
        assert_eq!(st.peaks_max, 9.0);
        assert_eq!(st.peaks_mean, 9.0);
        assert_eq!(st.programme_max, 1.0);
        assert_eq!(st.programme_mean, 1.0);
        let none = split_by_marks(&s, 10, &[], 10, 10, 30);
        assert_eq!(none.peaks_max, 0.0);
        assert_eq!(none.programme_max, 9.0);
    }

    #[test]
    fn clip_depth_of_known_removal() {
        let fs = 48_000.0;
        let x = tone(4800, fs, &[(1000.0, 1.0)]);
        // Remove exactly 10 % of the amplitude everywhere: depth −20 dB.
        let y: Vec<f32> = x.iter().map(|v| v * 0.9).collect();
        let d = clip_depth(&x, &y, 1, fs);
        assert!((d.overall_db + 20.0).abs() < 0.01, "{d:?}");
        assert!((d.p50_db + 20.0).abs() < 0.05, "{d:?}");
        assert!(d.blocks_share == 1.0 && d.samples_share > 0.9, "{d:?}");
        // Nothing removed.
        let d = clip_depth(&x, &x, 1, fs);
        assert_eq!(d.overall_db, NONE_DB);
        assert_eq!(d.max_db, NONE_DB);
        assert_eq!(d.samples_share, 0.0);
    }

    #[test]
    fn alias_spur_finds_a_folded_component() {
        let fs = 48_000.0;
        let n = FFT_LEN;
        // 15 kHz tone plus a −70 dB "alias" at 3 kHz (45 kHz folded).
        let x = tone(
            n,
            fs,
            &[(15_000.0, 0.5), (3_000.0, 10f64.powf(-70.0 / 20.0))],
        );
        let a = alias_spur_db(&x, 1, fs, 0, 15_000.0, 200.0);
        assert!((a + 70.0).abs() < 1.0, "{a}");
        let clean = tone(n, fs, &[(15_000.0, 0.5)]);
        assert!(alias_spur_db(&clean, 1, fs, 0, 15_000.0, 200.0) < -120.0);
    }

    #[test]
    fn imd_of_known_sidebands() {
        let fs = 48_000.0;
        let n = FFT_LEN;
        // 7 kHz carrier 0.2 with ±60 Hz sidebands at 1 % of it each:
        // RMS sum √2 % → −37 dB.
        let x = tone(
            n,
            fs,
            &[
                (60.0, 0.8),
                (7_000.0, 0.2),
                (6_940.0, 0.002),
                (7_060.0, 0.002),
            ],
        );
        let d = smpte_imd_db(&x, 1, fs, 0);
        assert!((d + 36.99).abs() < 0.5, "{d}");
        let x = tone(n, fs, &[(19_000.0, 0.5), (20_000.0, 0.5), (1_000.0, 0.005)]);
        assert!((ccif_imd_db(&x, 1, fs, 0) + 40.0).abs() < 0.5);
        let x = tone(
            n,
            fs,
            &[(3_000.0, 0.1), (2_950.0, 0.001), (3_100.0, 0.0001)],
        );
        assert!((sideband_dbc(&x, 1, fs, 0, 3_000.0, 50.0) + 40.0).abs() < 0.5);
        let s100 = sideband_dbc(&x, 1, fs, 0, 3_000.0, 100.0);
        assert!((s100 + 60.0).abs() < 1.0, "{s100}");
    }

    #[test]
    fn nmr_orders_errors() {
        let fs = 48_000.0;
        let n = 48_000;
        let x = tone(n, fs, &[(440.0, 0.3), (2_500.0, 0.1)]);
        // A gain change alone is no error.
        let gained: Vec<f32> = x.iter().map(|v| v * 0.5).collect();
        let g = nmr(&x, &gained, 1, fs, 0);
        assert!(g.mean_db < -100.0, "{g:?}");
        // Small and large added distortion (cubic): larger error, larger NMR.
        let distort = |k: f32| -> Vec<f32> { x.iter().map(|v| v + k * v * v * v).collect() };
        let small = nmr(&x, &distort(0.01), 1, fs, 0);
        let large = nmr(&x, &distort(0.3), 1, fs, 0);
        assert!(large.mean_db > small.mean_db + 20.0, "{small:?} {large:?}");
        assert!(bark(1000.0) > 8.0 && bark(1000.0) < 9.0);
    }
}
