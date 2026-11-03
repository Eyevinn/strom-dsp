//! Knob-turning guard test: changing any property while audio flows must not
//! click.
//!
//! # Method
//!
//! Every element runs in a `gst_check::Harness` on stereo F32 at 48 kHz, in
//! 10 ms buffers, with a steady sine on both channels (1 kHz at amplitude 0.25,
//! −12 dBFS, unless a scenario needs another tone). The scenario's setup puts
//! the processor in an *active* state (gate closed or open as needed,
//! compressor above threshold, EQ band on the tone, limiter limiting, ducker
//! ducking, …). After [`CHANGE_AT`] buffers (500 ms, so everything is
//! settled) the property is changed, between two buffers, exactly as an
//! application thread would.
//!
//! The output of each channel goes through the **sinusoid annihilator**
//! `r[n] = y[n] − 2·cos(ω)·y[n−1] + y[n−2]`, which is exactly 0 for any sine of
//! frequency ω, whatever its amplitude and phase. A processor that is linear
//! and time-invariant in steady state turns a sine into a sine, so `r` only
//! sees *changes*. For a slow gain change `g[n]` on the sine the residual is
//! about `2·g'·sin ω + g''` (per sample), so it measures how abrupt a change
//! is: a hard switch leaves a residual of the order of the step itself, a
//! smooth fade leaves almost nothing.
//!
//! The figure reported per scenario is the peak `|r|` from 20 ms before to
//! 200 ms after the change, in dB relative to the **louder of the two settled
//! output amplitudes** (the peak in the 20 ms before the analysis window, and
//! in the last 20 ms of the run). Using the louder one, not the level after
//! the change, is deliberate: a gate that fades from −12 dBFS to −72 dBFS
//! within its 100 ms release necessarily leaves a residual far above a level
//! 60 dB down, but what is heard is the residual against the signal that was
//! there; normalising to the quiet end would fail every audible-length fade.
//!
//! # Bound and its calibration
//!
//! [`HARD_LIMIT_DB`] = −50 dB is the pass/fail bound, [`TARGET_DB`] = −60 dB
//! the design target. Calibration, for a full-scale swing (amplitude `A` → 0
//! or 0 → `A`) of a 1 kHz tone at 48 kHz (`sin ω = 0.13`):
//!
//! | change | peak residual |
//! |---|---|
//! | hard switch | up to 0 dB (phase dependent, typically −20 … 0 dB) |
//! | linear ramp over 10 ms (slope kinks at both ends) | ≈ −54 dB |
//! | one-pole with τ = 5 ms (kink at the start) | ≈ −48 dB |
//! | raised-cosine fade over 5 ms | ≈ −55 dB |
//! | raised-cosine fade over 10 ms | ≈ −61 dB |
//! | raised-cosine fade over 20 ms | ≈ −67 dB |
//!
//! The steady-state floor of the elements is −90 … −150 dB (f32 rounding).
//! The Strom team's listening reading was "above −40 dB is a click, −40 to
//! −60 dB is worth a listen". −50 dB fails every hard switch by a wide
//! margin, and passes any fade that is smooth (no slope kink) and at least a
//! few ms long. It is not tight enough to demand the −60 dB target from
//! every change: a dynamics processor's own attack (a one-pole by design, the
//! user's choice of speed) is allowed to be a fast but smooth change.
//!
//! # Scenarios
//!
//! The scenarios are chosen so that a correctly smoothed change stays below
//! the bound: the test is about hard discontinuities, not about changes that
//! are meant to be heard. Where a time the user sets decides the speed (a
//! gate's attack, a dynamics processor's attack), the scenario uses a
//! moderate setting and says so. Exemptions are explicit, per scenario, with a
//! reason ([`Case::exempt`]).
//!
//! Every element's properties are enumerated from their param specs: every
//! writable, not construct-only property flagged `GST_PARAM_MUTABLE_PLAYING`
//! must be covered. Properties without an explicit scenario get generic ones
//! (default → a quarter of the way to each end of the range; booleans
//! toggled; every enum value) in the element's active base setup, so a newly
//! added property is tested from the day it appears.
//!
//! The meter is a passthrough (trivially clean, still measured). The ducker
//! is a `GstAggregator`: it is driven on both pads (programme and a steady
//! key on `sidechain`) like its own harness test.

mod util;

use std::collections::BTreeSet;
use std::f64::consts::PI;
use std::fmt::Write as _;

use gst::glib;
use gst::prelude::*;
use gst_check::Harness;

const RATE: u32 = 48_000;
const CH: usize = 2;
/// Frames per buffer (10 ms).
const FRAMES: usize = 480;
/// The property changes before buffer `CHANGE_AT` is pushed (between buffers
/// 49 and 50), after 500 ms of steady signal.
const CHANGE_AT: usize = 50;
/// Total buffers: 500 ms after the change, so every fade has settled.
const BUFFERS: usize = 100;
/// Analysis window: 20 ms before …
const PRE: usize = 960;
/// … to 200 ms after the change.
const POST: usize = 9600;
/// Length of the windows the settled amplitudes are measured in (20 ms).
const SETTLE: usize = 960;

/// Pass/fail bound (dB relative to the settled amplitude), see the module docs.
const HARD_LIMIT_DB: f64 = -50.0;
/// Design target: scenarios above it are reported, not failed.
const TARGET_DB: f64 = -60.0;

/// Elements and the GObject type name of each.
const ELEMENTS: [&str; 10] = [
    "stromdsp-gate",
    "stromdsp-compressor",
    "stromdsp-equalizer",
    "stromdsp-limiter",
    "stromdsp-deesser",
    "stromdsp-cleanup",
    "stromdsp-stereo",
    "stromdsp-delay",
    "stromdsp-meter",
    "stromdsp-ducker",
];

/// Test tone: one frequency, per-channel amplitude and phase.
#[derive(Debug, Clone, Copy)]
struct Tone {
    freq: f64,
    amp: [f64; CH],
    phase: [f64; CH],
}

const TONE: Tone = Tone {
    freq: 1000.0,
    amp: [0.25, 0.25],
    phase: [0.0, 0.0],
};

impl Tone {
    fn samples(&self, frames: usize) -> Vec<f32> {
        let w = 2.0 * PI * self.freq / RATE as f64;
        (0..frames)
            .flat_map(|n| {
                (0..CH).map(move |c| (self.amp[c] * (w * n as f64 + self.phase[c]).sin()) as f32)
            })
            .collect()
    }
}

/// One knob-turning scenario.
#[derive(Debug, Clone)]
struct Case {
    factory: &'static str,
    /// Properties set before PLAYING (string form, `set_property_from_str`).
    setup: Vec<(&'static str, String)>,
    /// Properties changed at [`CHANGE_AT`], in order. The first is the
    /// property the scenario covers.
    change: Vec<(&'static str, String)>,
    tone: Tone,
    /// Ducker only: amplitude of the steady 1 kHz key on `sidechain`.
    key_amp: f64,
    /// Bound for this scenario (dB).
    bound: f64,
    /// Why the bound is relaxed, if it is.
    exemption: Option<&'static str>,
    /// See [`Case::exempt_steady`].
    steady_margin: Option<f64>,
    /// Generated from the param spec rather than written out.
    generic: bool,
}

fn owned(v: &[(&'static str, &str)]) -> Vec<(&'static str, String)> {
    v.iter().map(|(k, s)| (*k, s.to_string())).collect()
}

/// A scenario on `factory`: the element's active base setup plus `setup`,
/// then the `change` at [`CHANGE_AT`].
fn case(
    factory: &'static str,
    setup: &[(&'static str, &str)],
    change: &[(&'static str, &str)],
) -> Case {
    let mut s = owned(&base_setup(factory));
    s.extend(owned(setup));
    Case {
        factory,
        setup: s,
        change: owned(change),
        tone: base_tone(factory),
        key_amp: 0.5,
        bound: HARD_LIMIT_DB,
        exemption: None,
        steady_margin: None,
        generic: false,
    }
}

impl Case {
    fn tone(mut self, freq: f64, amp: f64) -> Self {
        self.tone.freq = freq;
        self.tone.amp = [amp; CH];
        self
    }

    /// Relax the bound for this one scenario, with the reason.
    fn exempt(mut self, bound: f64, why: &'static str) -> Self {
        self.bound = bound;
        self.exemption = Some(why);
        self
    }

    /// The new setting makes the processor follow the waveform (an "instant"
    /// time constant, no lookahead): its output is no longer a pure sine, so
    /// the annihilator reads the new *steady state*, not a click. The change
    /// itself must not add more than 3 dB on top of that steady residual
    /// (measured in the last 100 ms); [`HARD_LIMIT_DB`] still applies if the
    /// steady residual is lower.
    fn exempt_steady(mut self, why: &'static str) -> Self {
        self.steady_margin = Some(3.0);
        self.exemption = Some(why);
        self
    }

    fn effective_bound(&self, m: &Measure) -> f64 {
        match self.steady_margin {
            Some(margin) => self.bound.max(m.post_floor_db + margin),
            None => self.bound,
        }
    }

    fn label(&self) -> String {
        let mut s = format!("{}:", self.factory.trim_start_matches("stromdsp-"));
        for (k, v) in &self.change {
            let from = self
                .setup
                .iter()
                .rev()
                .find(|(n, _)| n == k)
                .map(|(_, v)| v.as_str())
                .unwrap_or("default");
            let _ = write!(s, " {k} {from} -> {v}");
        }
        s
    }
}

/// Active base setup of each element (see the module docs).
fn base_setup(factory: &str) -> Vec<(&'static str, &'static str)> {
    match factory {
        // Open on the −12 dBFS tone; attack 20 ms so that an opening gate is a
        // smooth change (the default 5 ms one-pole starts with a slope kink of
        // about −48 dB: an intended fast attack, see the module docs).
        "stromdsp-gate" => vec![
            ("open-threshold", "-40"),
            ("close-threshold", "-50"),
            ("attack", "20"),
        ],
        // 4:1 at −20 dBFS (threshold is linear), tone 8 dB above it.
        "stromdsp-compressor" => vec![("threshold", "0.1"), ("ratio", "4")],
        // One peaking band, +12 dB on the tone.
        "stromdsp-equalizer" => vec![
            ("num-bands", "1"),
            ("band0-type", "7"),
            ("band0-frequency", "1000"),
            ("band0-gain", "12"),
            ("band0-q", "1"),
        ],
        // Limiting the −12 dBFS tone by 9 dB.
        "stromdsp-limiter" => vec![("threshold", "-21")],
        // Tone in the sibilance band, 28 dB above threshold.
        "stromdsp-deesser" => vec![("threshold", "-40"), ("attack", "5")],
        "stromdsp-cleanup" => vec![("highpass", "80"), ("dc-block", "true")],
        "stromdsp-stereo" => vec![],
        "stromdsp-delay" => vec![("delay", "5")],
        "stromdsp-meter" => vec![],
        // Steady key well above threshold: ducking by `depth`.
        "stromdsp-ducker" => vec![("threshold", "-30"), ("depth", "-15")],
        _ => unreachable!("{factory}"),
    }
}

fn base_tone(factory: &str) -> Tone {
    match factory {
        "stromdsp-deesser" => Tone {
            freq: 7000.0,
            ..TONE
        },
        // Different levels and phases per channel, so width and balance act.
        "stromdsp-stereo" => Tone {
            amp: [0.25, 0.15],
            phase: [0.0, 0.7],
            ..TONE
        },
        "stromdsp-cleanup" => Tone {
            freq: 130.0,
            ..TONE
        },
        _ => TONE,
    }
}

/// The explicit scenarios.
fn cases() -> Vec<Case> {
    let mut v = Vec::new();

    // ---- gate ----
    let g = "stromdsp-gate";
    let closed = [("open-threshold", "-6"), ("close-threshold", "-10")];
    v.push(case(g, &closed, &[("enabled", "false")]));
    v.push(case(
        g,
        &[closed[0], closed[1], ("enabled", "false")],
        &[("enabled", "true")],
    ));
    v.push(case(g, &[], &[("enabled", "false")]));
    v.push(case(g, &[("enabled", "false")], &[("enabled", "true")]));
    // The gate closes (default 100 ms release) …
    v.push(case(
        g,
        &[
            ("open-threshold", "-60"),
            ("close-threshold", "-70"),
            ("attack", "5"),
        ],
        &[("open-threshold", "-6"), ("close-threshold", "-6")],
    ));
    // … and opens again (20 ms attack).
    v.push(case(g, &closed, &[("open-threshold", "-40")]));
    v.push(case(g, &[], &[("close-threshold", "-20")]));
    for a in ["0.01", "200"] {
        v.push(case(g, &[], &[("attack", a)]));
    }
    for h in ["0", "1000"] {
        v.push(case(g, &[], &[("hold", h)]));
    }
    for r in ["1", "2000"] {
        v.push(case(g, &closed, &[("release", r)]));
    }
    v.push(case(g, &closed, &[("reduction", "-20")]));
    v.push(case(g, &closed, &[("reduction", "-80")]));
    v.push(case(g, &closed, &[("mode", "expander")]));
    v.push(case(
        g,
        &[closed[0], closed[1], ("mode", "expander")],
        &[("ratio", "20")],
    ));
    v.push(case(
        g,
        &[closed[0], closed[1], ("mode", "expander")],
        &[("ratio", "1")],
    ));

    // ---- compressor ----
    let c = "stromdsp-compressor";
    v.push(case(c, &[], &[("enabled", "false")]));
    v.push(case(c, &[("enabled", "false")], &[("enabled", "true")]));
    v.push(case(c, &[], &[("threshold", "0.01")]));
    v.push(case(c, &[], &[("threshold", "1")]));
    v.push(case(c, &[], &[("ratio", "1")]));
    v.push(case(c, &[], &[("ratio", "100")]));
    v.push(case(c, &[], &[("attack", "1")]));
    v.push(case(c, &[], &[("attack", "10000")]));
    v.push(case(c, &[], &[("release", "10")]));
    v.push(case(c, &[], &[("release", "10000")]));
    // 0.01 ms is "instant" by definition: the gain reduction jumps to the
    // detector's momentary value and then follows the waveform.
    const INSTANT: &str = "a 0.01 ms time constant is an instant change by definition; \
                           afterwards the gain follows the waveform (steady residual \
                           -59/-54 dB), so this measures the setting, not a click";
    v.push(case(c, &[], &[("attack", "0.01")]).exempt(-20.0, INSTANT));
    v.push(case(c, &[], &[("release", "0.01")]).exempt(-20.0, INSTANT));
    v.push(case(c, &[], &[("makeup-gain", "4")]));
    v.push(case(c, &[], &[("makeup-gain", "0.25")]));
    v.push(case(c, &[], &[("knee", "1")]));
    v.push(case(c, &[], &[("knee", "0")]));

    // ---- equalizer ----
    let e = "stromdsp-equalizer";
    v.push(case(e, &[], &[("enabled", "false")]));
    v.push(case(e, &[("enabled", "false")], &[("enabled", "true")]));
    v.push(case(e, &[], &[("band0-gain", "-12")]));
    v.push(case(e, &[("band0-gain", "0")], &[("band0-gain", "12")]));
    v.push(case(e, &[], &[("band0-gain", "-60")]));
    v.push(case(e, &[("band0-gain", "-60")], &[("band0-gain", "60")]));
    v.push(case(
        e,
        &[("band0-frequency", "250")],
        &[("band0-frequency", "4000")],
    ));
    v.push(case(e, &[], &[("band0-frequency", "24000")]));
    v.push(case(e, &[], &[("band0-frequency", "20")]));
    v.push(case(e, &[("band0-q", "0.5")], &[("band0-q", "8")]));
    v.push(case(e, &[], &[("band0-q", "100")]));
    v.push(case(e, &[], &[("band0-q", "0.01")]));
    v.push(case(e, &[], &[("band0-enabled", "false")]));
    v.push(case(e, &[], &[("band0-type", "0")]));
    v.push(case(e, &[], &[("band0-type", "8")]));
    v.push(case(
        e,
        &[("band0-type", "2"), ("band0-frequency", "100")],
        &[("band0-frequency", "500")],
    ));
    v.push(case(
        e,
        &[("band0-type", "8"), ("band0-frequency", "2000")],
        &[("band0-gain", "-12")],
    ));
    // Bands beyond num-bands are inactive: changing them does nothing.
    v.push(case(e, &[], &[("band5-gain", "20")]));

    // ---- limiter ----
    let l = "stromdsp-limiter";
    v.push(case(l, &[], &[("enabled", "false")]));
    v.push(case(l, &[("enabled", "false")], &[("enabled", "true")]));
    v.push(case(l, &[("threshold", "-3")], &[("threshold", "-15")]));
    v.push(case(l, &[], &[("threshold", "0")]));
    v.push(case(l, &[], &[("attack", "0.1")]));
    v.push(case(l, &[], &[("attack", "50")]));
    v.push(case(l, &[], &[("release", "1")]));
    v.push(case(l, &[], &[("release", "2000")]));
    v.push(case(l, &[], &[("lookahead", "10")]));
    v.push(case(l, &[], &[("lookahead", "2")]));
    const WAVEFORM: &str = "with a 0.01 ms attack or no lookahead the limiter shapes \
                            each peak of the 9 dB-limited sine: the steady output is not \
                            a pure sine";
    v.push(case(l, &[], &[("attack", "0.01")]).exempt_steady(WAVEFORM));
    v.push(case(l, &[], &[("lookahead", "0")]).exempt_steady(WAVEFORM));

    // ---- de-esser (7 kHz tone) ----
    let d = "stromdsp-deesser";
    v.push(case(d, &[], &[("enabled", "false")]));
    v.push(case(d, &[("enabled", "false")], &[("enabled", "true")]));
    v.push(case(d, &[("threshold", "-10")], &[("threshold", "-40")]));
    v.push(case(d, &[], &[("threshold", "0")]));
    v.push(case(d, &[], &[("frequency", "2000")]));
    v.push(case(d, &[], &[("max-reduction", "0")]));
    v.push(case(d, &[], &[("attack", "20")]));
    v.push(case(d, &[], &[("release", "500")]));
    v.push(case(d, &[], &[("mode", "wideband")]));

    // ---- cleanup (150 Hz tone) ----
    let k = "stromdsp-cleanup";
    v.push(case(k, &[], &[("enabled", "false")]));
    v.push(case(k, &[("enabled", "false")], &[("enabled", "true")]));
    v.push(case(k, &[("highpass", "0")], &[("highpass", "200")]).tone(100.0, 0.25));
    v.push(case(k, &[], &[("highpass", "0")]));
    v.push(case(k, &[], &[("highpass", "400")]));
    v.push(case(k, &[], &[("highpass-order", "8")]));
    v.push(case(k, &[], &[("dc-block", "false")]));
    v.push(case(k, &[], &[("hum", "50")]));
    v.push(case(k, &[("hum", "50")], &[("hum-harmonics", "8")]));
    v.push(case(k, &[("hum", "50")], &[("hum-q", "100")]));

    // ---- stereo ----
    let s = "stromdsp-stereo";
    v.push(case(s, &[("width", "1.5")], &[("enabled", "false")]));
    v.push(case(
        s,
        &[("width", "1.5"), ("enabled", "false")],
        &[("enabled", "true")],
    ));
    v.push(case(s, &[], &[("balance", "0.8")]));
    v.push(case(s, &[], &[("balance", "-1")]));
    v.push(case(s, &[], &[("width", "0")]));
    v.push(case(s, &[], &[("width", "2")]));
    v.push(case(s, &[], &[("invert-left", "true")]));
    v.push(case(s, &[], &[("invert-right", "true")]));
    v.push(case(s, &[], &[("mode", "mono")]));
    v.push(case(s, &[], &[("mode", "swap")]));
    v.push(case(s, &[("mode", "mono")], &[("mono-law", "minus-6db")]));

    // ---- delay ----
    let y = "stromdsp-delay";
    v.push(case(y, &[], &[("enabled", "false")]));
    v.push(case(y, &[("enabled", "false")], &[("enabled", "true")]));
    v.push(case(y, &[], &[("delay", "250")]));
    v.push(case(y, &[], &[("delay", "0")]));
    v.push(case(y, &[], &[("channel-delays", "<0.0, 2.5>")]));

    // ---- meter (passthrough) ----
    let m = "stromdsp-meter";
    v.push(case(m, &[], &[("interval", "10")]));

    // ---- ducker (programme + steady key) ----
    let u = "stromdsp-ducker";
    v.push(case(u, &[], &[("enabled", "false")]));
    v.push(case(u, &[("enabled", "false")], &[("enabled", "true")]));
    v.push(case(u, &[], &[("threshold", "0")]));
    v.push(case(u, &[("threshold", "0")], &[("threshold", "-30")]));
    v.push(case(u, &[], &[("depth", "-40")]));
    v.push(case(u, &[], &[("depth", "0")]));
    v.push(case(u, &[], &[("attack", "1")]));
    v.push(case(u, &[], &[("hold", "0")]));
    v.push(case(u, &[], &[("release", "2000")]));
    v.push(case(u, &[], &[("key-highpass", "2000")]));
    v.push(case(u, &[], &[("range-mode", "proportional")]));
    v.push(case(u, &[], &[("meter-interval", "0")]));

    v
}

/// Writable, not construct-only, mutable-in-PLAYING properties of the
/// element itself (not inherited from GstObject/BaseTransform/Aggregator).
fn knobs(element: &gst::Element) -> Vec<glib::ParamSpec> {
    element
        .list_properties()
        .iter()
        .filter(|p| {
            let f = p.flags();
            p.owner_type() == element.type_()
                && f.contains(glib::ParamFlags::WRITABLE)
                && !f.contains(glib::ParamFlags::CONSTRUCT_ONLY)
                && f.contains(gst::PARAM_FLAG_MUTABLE_PLAYING)
        })
        .cloned()
        .collect()
}

fn fmt_f(v: f64) -> String {
    format!("{v}")
}

/// Generic scenarios for a property without an explicit one: default → a
/// quarter of the way to each end of its range; booleans toggled; every other
/// enum value.
fn generic_cases(factory: &'static str, p: &glib::ParamSpec) -> Vec<Case> {
    let name: &'static str = Box::leak(p.name().to_string().into_boxed_str());
    let mut values = Vec::new();
    let quarter = |d: f64, lo: f64, hi: f64| [d + 0.25 * (hi - d), d - 0.25 * (d - lo)];
    if let Some(f) = p.downcast_ref::<glib::ParamSpecFloat>() {
        let (d, lo, hi) = (
            f.default_value() as f64,
            f.minimum() as f64,
            f.maximum() as f64,
        );
        values.extend(quarter(d, lo, hi).iter().map(|&v| fmt_f(v)));
    } else if let Some(f) = p.downcast_ref::<glib::ParamSpecDouble>() {
        values.extend(
            quarter(f.default_value(), f.minimum(), f.maximum())
                .iter()
                .map(|&v| fmt_f(v)),
        );
    } else if let Some(u) = p.downcast_ref::<glib::ParamSpecUInt>() {
        let q = quarter(
            u.default_value() as f64,
            u.minimum() as f64,
            u.maximum() as f64,
        );
        values.extend(q.iter().map(|v| format!("{}", v.round() as u32)));
    } else if let Some(i) = p.downcast_ref::<glib::ParamSpecInt>() {
        let q = quarter(
            i.default_value() as f64,
            i.minimum() as f64,
            i.maximum() as f64,
        );
        values.extend(q.iter().map(|v| format!("{}", v.round() as i32)));
    } else if let Some(b) = p.downcast_ref::<glib::ParamSpecBoolean>() {
        values.push((!b.default_value()).to_string());
    } else if let Some(en) = p.downcast_ref::<glib::ParamSpecEnum>() {
        let class = en.enum_class();
        for ev in class.values() {
            if ev.value() != en.default_value_as_i32() {
                values.push(ev.nick().to_string());
            }
        }
    } else {
        panic!(
            "{factory}: no generic scenario for {} of type {}; add an explicit one",
            p.name(),
            p.value_type()
        );
    }
    values.sort();
    values.dedup();
    values
        .into_iter()
        .map(|val| {
            let mut c = case(factory, &[], &[]);
            c.change = vec![(name, val)];
            c.generic = true;
            c
        })
        .collect()
}

fn to_bytes(s: &[f32]) -> Vec<u8> {
    s.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn from_bytes(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn buffer(samples: &[f32], i: usize) -> gst::Buffer {
    let dur = gst::ClockTime::from_mseconds(10);
    let mut b = gst::Buffer::from_mut_slice(to_bytes(samples));
    {
        let b = b.get_mut().unwrap();
        b.set_pts(dur * i as u64);
        b.set_duration(dur);
        b.set_offset((i * FRAMES) as u64);
        b.set_offset_end(((i + 1) * FRAMES) as u64);
    }
    b
}

/// Run one scenario and return the interleaved output.
fn run(case: &Case) -> Vec<f32> {
    util::init();
    let input = case.tone.samples(BUFFERS * FRAMES);
    let mut h = Harness::new(case.factory);
    let element = h.element().unwrap();
    for (k, v) in &case.setup {
        element.set_property_from_str(k, v);
    }
    let ducker = case.factory == "stromdsp-ducker";
    h.set_src_caps_str(&util::caps_str(RATE, CH));
    let mut hk = ducker.then(|| {
        h.set_live(false);
        let mut hk = Harness::with_element(&element, Some("sidechain"), None);
        hk.set_live(false);
        hk.set_src_caps_str(&util::caps_str(RATE, 1));
        hk
    });
    let key = Tone {
        freq: 1000.0,
        amp: [case.key_amp; CH],
        phase: [0.3; CH],
    }
    .samples(BUFFERS * FRAMES);
    h.play();
    let mut out = Vec::with_capacity(input.len());
    for (i, chunk) in input.chunks(FRAMES * CH).enumerate() {
        if i == CHANGE_AT {
            for (k, v) in &case.change {
                element.set_property_from_str(k, v);
            }
        }
        assert_eq!(h.push(buffer(chunk, i)), Ok(gst::FlowSuccess::Ok));
        if let Some(hk) = hk.as_mut() {
            // Mono key: every other sample of the stereo tone.
            let ks: Vec<f32> = key[i * FRAMES * CH..(i + 1) * FRAMES * CH]
                .iter()
                .step_by(CH)
                .copied()
                .collect();
            assert_eq!(hk.push(buffer(&ks, i)), Ok(gst::FlowSuccess::Ok));
        }
        let o = h.pull().expect("no output buffer");
        assert_eq!(o.size(), chunk.len() * 4);
        out.extend(from_bytes(&o.map_readable().unwrap()));
    }
    out
}

/// Annihilator measurement of one run (see the module docs).
#[derive(Debug, Clone, Copy)]
struct Measure {
    /// Peak residual around the change, dB re the louder settled amplitude.
    residual_db: f64,
    /// The same, dB re the settled amplitude after the change.
    re_after_db: f64,
    /// Steady-state residual before the change (scenario sanity).
    floor_db: f64,
    /// Steady-state residual in the last 100 ms (the processor after the
    /// change; not a pure sine when a setting makes it follow the waveform).
    post_floor_db: f64,
    before: f64,
    after: f64,
}

fn measure(out: &[f32], freq: f64) -> Measure {
    let c2 = 2.0 * (2.0 * PI * freq / RATE as f64).cos();
    let frames = out.len() / CH;
    let at = CHANGE_AT * FRAMES;
    let y = |n: usize, c: usize| out[n * CH + c] as f64;
    let r = |n: usize, c: usize| y(n, c) - c2 * y(n - 1, c) + y(n - 2, c);
    let peak = |range: std::ops::Range<usize>, f: &dyn Fn(usize, usize) -> f64| {
        range
            .flat_map(|n| (0..CH).map(move |c| (n, c)))
            .map(|(n, c)| f(n, c).abs())
            .fold(0.0f64, f64::max)
    };
    let before = peak(at - PRE - SETTLE..at - PRE, &y);
    let after = peak(frames - SETTLE..frames, &y);
    let resid = peak(at - PRE..at + POST, &r);
    let floor = peak(at - 4 * PRE..at - PRE, &r);
    let post_floor = peak(frames - 5 * SETTLE..frames, &r);
    let reference = before.max(after).max(1e-9);
    let db = |v: f64, re: f64| 20.0 * (v.max(1e-15) / re.max(1e-12)).log10();
    Measure {
        residual_db: db(resid, reference),
        re_after_db: db(resid, after),
        floor_db: db(floor, reference),
        post_floor_db: db(post_floor, reference),
        before,
        after,
    }
}

#[test]
fn every_knob_is_click_free() {
    util::init();
    let mut all = cases();

    // Coverage: every mutable-in-PLAYING property of every element.
    for factory in ELEMENTS {
        let element = gst::ElementFactory::make(factory).build().unwrap();
        let covered: BTreeSet<&str> = all
            .iter()
            .filter(|c| c.factory == factory)
            .filter_map(|c| c.change.first().map(|(k, _)| *k))
            .collect();
        let mut generic = Vec::new();
        for p in knobs(&element) {
            // Equalizer bands 1..7 are the same code as band 0: covered by
            // band0's scenarios (and band5 for an inactive band).
            let band_alias = p
                .name()
                .strip_prefix("band")
                .and_then(|r| r.split_once('-'))
                .map(|(_, f)| format!("band0-{f}"));
            let is_covered = covered.contains(p.name())
                || band_alias.is_some_and(|a| covered.contains(a.as_str()));
            if !is_covered {
                generic.extend(generic_cases(factory, &p));
            }
        }
        all.extend(generic);
    }

    let mut report = String::new();
    let mut failures = Vec::new();
    let mut above_target = 0;
    for c in &all {
        let out = run(c);
        assert!(
            out.iter().all(|v| v.is_finite()),
            "{}: non-finite",
            c.label()
        );
        let m = measure(&out, c.tone.freq);
        let bound = c.effective_bound(&m);
        let verdict = if m.residual_db > bound {
            failures.push(format!(
                "{} ({:.1} dB > {:.1} dB)",
                c.label(),
                m.residual_db,
                bound
            ));
            "FAIL"
        } else if m.residual_db > TARGET_DB {
            above_target += 1;
            "above target"
        } else {
            "ok"
        };
        // A scenario must be steady before the change, or it measures nothing.
        if m.floor_db > c.bound - 10.0 {
            failures.push(format!(
                "{}: not steady before the change (floor {:.1} dB)",
                c.label(),
                m.floor_db
            ));
        }
        let _ = writeln!(
            report,
            "{:>7.1} dB  (re after {:>6.1}, floors {:>6.1}/{:>6.1} dB, amp {:.4} -> {:.4})  {:<12} {}{}{}",
            m.residual_db,
            m.re_after_db,
            m.floor_db,
            m.post_floor_db,
            m.before,
            m.after,
            verdict,
            c.label(),
            if c.generic { "  [generic]" } else { "" },
            c.exemption
                .map(|w| format!("  [exempt to {:.0} dB: {w}]", c.bound))
                .unwrap_or_default(),
        );
    }
    println!("{report}");
    println!(
        "{} scenarios, {} above the {TARGET_DB} dB target, {} failures",
        all.len(),
        above_target,
        failures.len()
    );
    assert!(
        failures.is_empty(),
        "knob changes above {HARD_LIMIT_DB} dB:\n{}",
        failures.join("\n")
    );
}
