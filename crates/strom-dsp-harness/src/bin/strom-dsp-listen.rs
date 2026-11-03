//! Listening kit: render processor configurations over the test corpus and
//! build the static listening page.
//!
//! ```text
//! strom-dsp-listen render --corpus DIR --out DIR [--jobs N] [--max-seconds S]
//!                         [--no-wav] [--groups a,b] [--clips x,y]
//! strom-dsp-listen page --out DIR
//! strom-dsp-listen list
//! ```
//!
//! `scripts/listen.sh` runs both steps after fetching the corpus.

use std::path::PathBuf;
use std::process::ExitCode;

use strom_dsp_harness::listen::{self, page, ClipSet, RenderOptions};

const USAGE: &str = "usage:
  strom-dsp-listen render --corpus DIR --out DIR [--jobs N] [--max-seconds S]
                          [--no-wav] [--groups a,b] [--clips x,y]
  strom-dsp-listen page --out DIR
  strom-dsp-listen list

render  runs every group's configurations over its corpus clips into DIR
        (MP4 and, unless --no-wav, 24-bit WAV per version; metrics.json per
        set; listen-data.json). Needs ffmpeg. --jobs defaults to 4,
        --max-seconds (clip length cap) to 30.
page    writes DIR/index.html and each set's audio.js from listen-data.json.
list    prints the groups, their configurations and clips.";

fn main() -> ExitCode {
    match run(std::env::args().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: Vec<String>) -> Result<(), String> {
    let mut it = args.into_iter();
    let cmd = it.next().ok_or(USAGE)?;
    let mut corpus = None;
    let mut out = None;
    let mut jobs = 4;
    let mut max_seconds = 30.0;
    let mut wav = true;
    let mut groups = None;
    let mut clips = None;
    let list = |s: String| Some(s.split(',').map(str::to_string).collect::<Vec<_>>());
    while let Some(a) = it.next() {
        let mut val = || it.next().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--corpus" => corpus = Some(PathBuf::from(val()?)),
            "--out" => out = Some(PathBuf::from(val()?)),
            "--jobs" => jobs = val()?.parse().map_err(|e| format!("--jobs: {e}"))?,
            "--max-seconds" => {
                max_seconds = val()?.parse().map_err(|e| format!("--max-seconds: {e}"))?
            }
            "--no-wav" => wav = false,
            "--groups" => groups = list(val()?),
            "--clips" => clips = list(val()?),
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            _ => return Err(format!("unknown argument {a}\n{USAGE}")),
        }
    }
    match cmd.as_str() {
        "render" => {
            let opts = RenderOptions {
                corpus: corpus.ok_or("render needs --corpus")?,
                out: out.ok_or("render needs --out")?,
                jobs,
                max_seconds,
                wav,
                groups,
                clips,
            };
            let data = listen::render(&opts).map_err(|e| e.to_string())?;
            eprintln!(
                "rendered {} sets in {} s into {}",
                data["sets"].as_array().map_or(0, Vec::len),
                data["render_seconds"],
                opts.out.display()
            );
            Ok(())
        }
        "page" => {
            let out = out.ok_or("page needs --out")?;
            page::build(&out).map_err(|e| e.to_string())?;
            eprintln!("wrote {}", out.join("index.html").display());
            Ok(())
        }
        "list" => {
            for g in listen::groups() {
                println!("{}: {}", g.id, g.label);
                for c in &g.configs {
                    println!("  config {}: {}", c.id, c.about);
                }
                match g.clips {
                    ClipSet::Ids(ids) => println!("  clips: {}", ids.join(", ")),
                    ClipSet::Categories(c) => println!("  clips: categories {}", c.join(", ")),
                }
            }
            Ok(())
        }
        "-h" | "--help" => {
            println!("{USAGE}");
            Ok(())
        }
        _ => Err(format!("unknown command {cmd}\n{USAGE}")),
    }
}
