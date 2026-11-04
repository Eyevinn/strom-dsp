//! The declipper on the test corpus: clean speech and music clipped 3, 6
//! and 10 dB below their peak, then declipped, and compared with the
//! original.
//!
//! Two measurements:
//!
//! - **Never worse** ([`corpus_cases`]): every clip of the listening kit's
//!   declipper groups ([`DECLIP_CLEAN_CLIPS`]), prepared as the kit does
//!   (stereo, scaled to [`REFERENCE_LUFS`](crate::listen::REFERENCE_LUFS),
//!   cut to [`MAX_SECONDS`]), through the declipper at its defaults. The
//!   repair must not move the clipped samples further from the original
//!   than the clipping did (SDR gain on them ≥ 0 dB), and the output's
//!   sample peak must stay within [`PEAK_MARGIN_DB`] of the original's.
//! - **Speech quality** ([`speech_dsdrc`]): the clean speech clips mixed to
//!   mono, normalised to a peak of 1, clipped, declipped; the mean SDR
//!   improvement on the clipped samples (ΔSDRc, Záviška et al. 2021) per
//!   depth, which must stay within [`SPEECH_DSDRC_MARGIN_DB`] of
//!   [`SPEECH_DSDRC_REFERENCE`].
//!
//! [`failures`] turns both into the acceptance check of the extended
//! workflow (`tests/declipper_corpus.rs`).

use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use strom_dsp::declipper::{Declipper, DeclipperParams, DeclipperStats};
use strom_dsp::Processor;

use crate::chain::FS;
use crate::listen::{
    audio, clip_below_peak, masked_sdr, run_declipper, set_original, Clip, DECLIP_CLEAN_CLIPS,
};
use crate::signals::sample_peak;

/// Clip depths below the peak (dB).
pub const DEPTHS_DB: [f64; 3] = [3.0, 6.0, 10.0];
/// Clips are cut to this length (s), as in the listening kit.
pub const MAX_SECONDS: f64 = 30.0;
/// Most the output's sample peak may exceed the original's (dB). The
/// rebuild is smooth, so where the true waveform is flat-topped (glottal
/// pulses of a low voice clipped 3 dB) it rises above the true peak:
/// measured up to 2.1 dB (vctk_p225 at 3 dB); every case clipped 6 or 10 dB
/// stays at or below the original's peak. A rebuild that runs away (the
/// 6 dB cap over a 3 dB clip) exceeds this bound.
pub const PEAK_MARGIN_DB: f64 = 2.5;
/// Speech ΔSDRc (dB) at [`DEPTHS_DB`] of the declipper without the
/// plausibility check and with a 12 dB cap (otherwise default settings).
pub const SPEECH_DSDRC_REFERENCE: [f64; 3] = [16.5, 12.8, 8.7];
/// Most the speech ΔSDRc may fall below [`SPEECH_DSDRC_REFERENCE`] (dB).
pub const SPEECH_DSDRC_MARGIN_DB: f64 = 1.0;
/// Corpus category of the clean speech clips.
pub const SPEECH_CATEGORY: &str = "1_speech_clean";

/// One clip at one depth.
#[derive(Debug, Clone)]
pub struct Case {
    pub clip: String,
    pub category: String,
    pub depth_db: f64,
    /// Samples (both channels) the clipping changed.
    pub clipped: usize,
    /// SDR gain on the clipped samples (dB): SDR of the output against the
    /// original minus that of the clipped input.
    pub sdr_gain_db: f64,
    /// Sample peaks (dBFS).
    pub peak_original_db: f64,
    pub peak_clipped_db: f64,
    pub peak_output_db: f64,
    pub stats: DeclipperStats,
}

fn db(v: f64) -> f64 {
    20.0 * v.max(1e-12).log10()
}

/// `f` over `items` on `jobs` threads, results in order.
fn par_map<T: Sync, R: Send>(items: &[T], jobs: usize, f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let next = AtomicUsize::new(0);
    let out: Mutex<Vec<(usize, R)>> = Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..jobs.clamp(1, items.len().max(1)) {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(item) = items.get(i) else { break };
                let r = f(item);
                out.lock().expect("results").push((i, r));
            });
        }
    });
    let mut out = out.into_inner().expect("results");
    out.sort_by_key(|r| r.0);
    out.into_iter().map(|r| r.1).collect()
}

/// The declipper's case for one prepared original (stereo, 48 kHz) at
/// `depth_db`.
pub fn case(clip: &Clip, original: &[f32], depth_db: f64, params: &DeclipperParams) -> Case {
    let input = clip_below_peak(original, depth_db);
    let mask: Vec<bool> = original
        .iter()
        .zip(&input)
        .map(|(a, b)| a.to_bits() != b.to_bits())
        .collect();
    let clipped = mask.iter().filter(|&&m| m).count();
    let (output, stats, _) = run_declipper(&input, params);
    let sdr_gain_db = if clipped == 0 {
        0.0
    } else {
        masked_sdr(original, &output, &mask) - masked_sdr(original, &input, &mask)
    };
    Case {
        clip: clip.id.clone(),
        category: clip.category.clone(),
        depth_db,
        clipped,
        sdr_gain_db,
        peak_original_db: db(sample_peak(original)),
        peak_clipped_db: db(sample_peak(&input)),
        peak_output_db: db(sample_peak(&output)),
        stats,
    }
}

/// Every clip of [`DECLIP_CLEAN_CLIPS`] (those in the corpus at `corpus`)
/// at every depth of [`DEPTHS_DB`].
pub fn corpus_cases(corpus: &Path, params: &DeclipperParams, jobs: usize) -> io::Result<Vec<Case>> {
    let manifest = crate::listen::read_manifest(&corpus.join("manifest.tsv"))?;
    let clips: Vec<&Clip> = DECLIP_CLEAN_CLIPS
        .iter()
        .filter_map(|id| manifest.iter().find(|c| c.id == *id))
        .collect();
    let originals = par_map(&clips, jobs, |c| set_original(corpus, c, MAX_SECONDS));
    let mut work = Vec::new();
    for (c, o) in clips.iter().zip(originals) {
        let o = o?.0;
        for &d in &DEPTHS_DB {
            work.push((*c, o.clone(), d));
        }
    }
    Ok(par_map(&work, jobs, |(c, o, d)| case(c, o, *d, params)))
}

/// One clean speech clip as the speech measurement uses it: the mean of
/// its channels, normalised to a peak of 1.
pub fn speech_original(corpus: &Path, clip: &Clip) -> io::Result<Vec<f64>> {
    let native = audio::decode(&corpus.join(&clip.file), clip.channels)?;
    let ch = clip.channels.max(1);
    let mut x: Vec<f64> = native
        .chunks_exact(ch)
        .map(|f| f.iter().map(|&v| f64::from(v)).sum::<f64>() / ch as f64)
        .collect();
    let peak = x.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    if peak > 0.0 {
        for v in &mut x {
            *v /= peak;
        }
    }
    Ok(x)
}

/// ΔSDRc (dB) of mono `x` (peak 1) clipped `depth_db` below its peak, or
/// `None` when nothing clips.
pub fn speech_case(x: &[f64], depth_db: f64, params: &DeclipperParams) -> Option<f64> {
    let c = 10f64.powf(-depth_db / 20.0);
    let xc: Vec<f32> = x.iter().map(|v| v.clamp(-c, c) as f32).collect();
    let truth: Vec<bool> = x.iter().map(|v| v.abs() > c).collect();
    if !truth.iter().any(|&t| t) {
        return None;
    }
    let mut d = Declipper::new();
    d.set_params(params);
    d.configure(FS as f32, 1);
    let lat = d.latency_frames();
    let mut y = xc.clone();
    y.extend(std::iter::repeat_n(0.0, lat));
    for b in y.chunks_mut(480) {
        d.process(b);
    }
    let y = &y[lat..];
    let sdr = |z: &[f32]| {
        let (mut s, mut e) = (0.0, 0.0);
        for ((&a, &b), &m) in x.iter().zip(z).zip(&truth) {
            if m {
                s += a * a;
                e += (a - f64::from(b)).powi(2);
            }
        }
        10.0 * (s / e.max(1e-300)).log10()
    };
    Some(sdr(y) - sdr(&xc))
}

/// Mean speech ΔSDRc (dB) at each depth of [`DEPTHS_DB`] over the clips of
/// [`SPEECH_CATEGORY`].
pub fn speech_dsdrc(corpus: &Path, params: &DeclipperParams, jobs: usize) -> io::Result<[f64; 3]> {
    let manifest = crate::listen::read_manifest(&corpus.join("manifest.tsv"))?;
    let clips: Vec<&Clip> = manifest
        .iter()
        .filter(|c| c.category == SPEECH_CATEGORY)
        .collect();
    let originals = par_map(&clips, jobs, |c| speech_original(corpus, c));
    let mut work = Vec::new();
    for o in originals {
        let o = o?;
        for k in 0..DEPTHS_DB.len() {
            work.push((o.clone(), k));
        }
    }
    let r = par_map(&work, jobs, |(o, k)| {
        (*k, speech_case(o, DEPTHS_DB[*k], params))
    });
    let mut sum = [0.0; 3];
    let mut n = [0usize; 3];
    for (k, v) in r {
        if let Some(v) = v {
            sum[k] += v;
            n[k] += 1;
        }
    }
    Ok(std::array::from_fn(|k| sum[k] / n[k].max(1) as f64))
}

/// The acceptance check: one line per broken bound.
pub fn failures(cases: &[Case], speech: &[f64; 3]) -> Vec<String> {
    let mut f = Vec::new();
    for c in cases {
        if c.sdr_gain_db < 0.0 {
            f.push(format!(
                "{} {} dB: SDR gain on the clipped samples {:.2} dB < 0",
                c.clip, c.depth_db, c.sdr_gain_db
            ));
        }
        if c.peak_output_db > c.peak_original_db + PEAK_MARGIN_DB {
            f.push(format!(
                "{} {} dB: output peak {:.2} dBFS > original {:.2} + {PEAK_MARGIN_DB} dB",
                c.clip, c.depth_db, c.peak_output_db, c.peak_original_db
            ));
        }
    }
    for (k, &d) in DEPTHS_DB.iter().enumerate() {
        let min = SPEECH_DSDRC_REFERENCE[k] - SPEECH_DSDRC_MARGIN_DB;
        if speech[k] < min {
            f.push(format!(
                "speech {d} dB: mean ΔSDRc {:.2} dB < {min:.1}",
                speech[k]
            ));
        }
    }
    f
}

/// The cases and the speech ΔSDRc as a table.
pub fn table(cases: &[Case], speech: &[f64; 3]) -> String {
    let mut s = format!(
        "{:<40} {:>5} {:>8} {:>8} {:>9} {:>9} {:>9} {:>6} {:>8}\n",
        "clip", "depth", "clipped", "gain dB", "peak orig", "clipped", "output", "runs", "rejected"
    );
    for c in cases {
        s += &format!(
            "{:<40} {:>5.0} {:>8} {:>8.2} {:>9.2} {:>9.2} {:>9.2} {:>6} {:>8}\n",
            c.clip,
            c.depth_db,
            c.clipped,
            c.sdr_gain_db,
            c.peak_original_db,
            c.peak_clipped_db,
            c.peak_output_db,
            c.stats.runs_repaired,
            c.stats.runs_rejected,
        );
    }
    s += &format!(
        "speech ΔSDRc (dB) at 3/6/10 dB: {:.2} / {:.2} / {:.2} (reference {:.1} / {:.1} / {:.1})\n",
        speech[0],
        speech[1],
        speech[2],
        SPEECH_DSDRC_REFERENCE[0],
        SPEECH_DSDRC_REFERENCE[1],
        SPEECH_DSDRC_REFERENCE[2]
    );
    s
}
