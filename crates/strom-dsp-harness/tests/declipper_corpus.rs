//! The declipper on the test corpus: never worse than its clipped input,
//! no runaway peaks, speech quality kept ([`strom_dsp_harness::declip`]).
//!
//! Needs the corpus (`scripts/fetch-corpus.sh`) and ffmpeg; a measurement,
//! not a default test:
//!
//! ```sh
//! cargo test --release -p strom-dsp-harness --test declipper_corpus -- --ignored --nocapture
//! ```
//!
//! `STROM_DSP_CORPUS` points at another unpacked corpus (default
//! `corpus/data` in the repository).

use std::path::PathBuf;

use strom_dsp::declipper::DeclipperParams;
use strom_dsp_harness::declip::{corpus_cases, failures, speech_dsdrc, table};

#[test]
#[ignore = "needs the test corpus and ffmpeg; run in the extended workflow"]
fn declipper_on_the_corpus() {
    let corpus = std::env::var_os("STROM_DSP_CORPUS").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../corpus/data"),
        PathBuf::from,
    );
    let jobs = std::thread::available_parallelism().map_or(4, |n| n.get());
    let params = DeclipperParams::default();
    let cases = corpus_cases(&corpus, &params, jobs).expect("corpus clips");
    assert_eq!(
        cases.len(),
        strom_dsp_harness::listen::DECLIP_CLEAN_CLIPS.len() * 3,
        "every clip of the declipper groups is in the corpus"
    );
    let speech = speech_dsdrc(&corpus, &params, jobs).expect("speech clips");
    print!("{}", table(&cases, &speech));
    let f = failures(&cases, &speech);
    assert!(f.is_empty(), "{}", f.join("\n"));
}
