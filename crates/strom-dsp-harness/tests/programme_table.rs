//! The broadcast chain on programme at three input levels, per preset: a
//! report, not a regression check (run it on purpose; it prints a table).
//!
//! For each preset, input level (−35, −23, −10 LUFS) and programme (the
//! peaky programme with a click every 1.1 s, the same with a click every
//! 300 ms, the music-like programme, speech-like noise band-limited to
//! 7 kHz): 60 s through the preset at 4 bands, measured from 10 s.
//! Columns: integrated output against the target (LU), the governor's
//! loudness deficit and budget excess at the end, the final clipper's clip depth
//! (energy-weighted mean of the per-buffer values, dB), the limiter's mean
//! gain reduction (dB), the largest output true peak (dBTP), and the
//! governor's density and state at the end.

use strom_dsp::broadcast::{BroadcastParams, BroadcastProcessor};
use strom_dsp::presets::BroadcastPreset;
use strom_dsp::Processor;
use strom_dsp_harness::agc_support::scale_to_lufs;
use strom_dsp_harness::metrics;
use strom_dsp_harness::signals::{SignalKind, CH};

const FS: f64 = 48_000.0;
const SECONDS: f64 = 60.0;
const SKIP_SECONDS: f64 = 10.0;
const BLOCK: usize = 480;

/// One row of the table.
pub struct Row {
    pub error_lu: f64,
    pub deficit_lu: f64,
    pub excess_db: f64,
    pub density_db: f64,
    pub clip_depth_db: f64,
    pub limiter_gr_db: f64,
    pub true_peak_dbtp: f64,
    pub state: &'static str,
}

fn run(p: &BroadcastParams, kind: SignalKind, input_lufs: f64) -> Row {
    let frames = (SECONDS * FS) as usize;
    let mut x = kind.generate(frames, FS).data;
    scale_to_lufs(&mut x, CH, FS, input_lufs);
    let mut bp = BroadcastProcessor::new();
    bp.set_params(p);
    bp.configure(FS as f32, CH);
    let skip = (SKIP_SECONDS * FS) as usize / BLOCK;
    let (mut depth, mut gr, mut n) = (0.0f64, 0.0f64, 0usize);
    let mut last = None;
    for (i, c) in x.chunks_mut(BLOCK * CH).enumerate() {
        bp.process(c);
        let m = bp.take_meters();
        if i >= skip {
            depth += 10f64.powf(m.clip_depth_db / 10.0);
            gr += m.limiter_gain_reduction_db;
            n += 1;
        }
        last = Some(m);
    }
    let l = metrics::loudness(&x[(SKIP_SECONDS * FS) as usize * CH..], CH, FS);
    let m = last.expect("ran");
    Row {
        error_lu: l.integrated_lufs - p.target_loudness_lufs as f64,
        deficit_lu: m.loudness_deficit_lu,
        excess_db: m.budget_excess_db,
        density_db: bp.governor().density_db(),
        clip_depth_db: (10.0 * (depth / n as f64).max(1e-30).log10()).max(-120.0),
        limiter_gr_db: gr / n as f64,
        true_peak_dbtp: l.max_true_peak_dbtp,
        state: m.governor_state.nick(),
    }
}

#[test]
#[ignore = "long report, prints a table"]
fn programme_table() {
    let programmes = [
        ("peaky", SignalKind::Peaky(1.1)),
        ("clicky 300 ms", SignalKind::Peaky(0.3)),
        ("music", SignalKind::Music),
        ("speech-like", SignalKind::SpeechNoise(7_000.0)),
    ];
    let rows: Vec<(String, Row)> = std::thread::scope(|s| {
        let handles: Vec<_> = BroadcastPreset::ALL
            .into_iter()
            .map(|preset| {
                s.spawn(move || {
                    let p = BroadcastParams::preset(preset, 4);
                    let mut out = Vec::new();
                    for (name, kind) in programmes {
                        for input in [-35.0, -23.0, -10.0] {
                            let r = run(&p, kind, input);
                            out.push((format!("| {} | {name} | {input} |", preset.nick()), r));
                        }
                    }
                    out
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("preset run"))
            .collect()
    });
    println!(
        "| Preset | Programme | Input (LUFS) | Output − target (LU) | Deficit (LU) | Budget excess (dB) \
         | Clip depth (dB) | Limiter GR (dB) | True peak (dBTP) | Density (dB) | Governor |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|");
    for (head, r) in rows {
        println!(
            "{head} {:+.2} | {:.2} | {:.2} | {:.1} | {:.2} | {:.2} | {:.2} | {} |",
            r.error_lu,
            r.deficit_lu,
            r.excess_db,
            r.clip_depth_db,
            r.limiter_gr_db,
            r.true_peak_dbtp,
            r.density_db,
            r.state
        );
    }
}
