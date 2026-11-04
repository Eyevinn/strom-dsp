//! Acceptance check "no measurable crackle on speech" (see
//! [`strom_dsp_harness::crackle`]): every preset × latency mode over the
//! corpus speech and phone clips, worst-frame NMR under the bound, with the
//! known failures listed there.
//!
//! It needs the test corpus (`scripts/fetch-corpus.sh`; `STROM_DSP_CORPUS`
//! points elsewhere) and runs about 15 s on 8 threads in release, so it is
//! ignored in the default suite and runs in the extended workflow:
//!
//! ```sh
//! scripts/fetch-corpus.sh
//! cargo test --release -p strom-dsp-harness --test crackle -- --ignored
//! ```

use strom_dsp_harness::crackle::{acceptance, corpus_dir, verdict};

#[test]
#[ignore = "needs the corpus (scripts/fetch-corpus.sh); runs in the extended workflow"]
fn no_measurable_crackle_on_speech() {
    let corpus = corpus_dir();
    let jobs = std::thread::available_parallelism().map_or(4, |n| n.get());
    let runs = acceptance(&corpus, jobs, |_| {}).unwrap_or_else(|e| {
        panic!(
            "corpus at {}: {e} (run scripts/fetch-corpus.sh)",
            corpus.display()
        )
    });
    let (table, problems) = verdict(&runs);
    println!("{table}");
    assert!(problems.is_empty(), "\n{}", problems.join("\n"));
}
