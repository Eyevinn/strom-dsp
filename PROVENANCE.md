# Provenance

`strom-dsp` and `gst-plugin-stromdsp` are a **clean-room implementation**. Every
line of code in this repository was written new, from public descriptions of
algorithms (textbooks, papers, public-domain notes) and from the documentation
of permissively licensed dependencies.

## Clean-room rules followed

- **Not accessed, in any form:** LSP Plugins (and any fork, mirror or quotation
  of it), lsp-plugins-rs (source, checkouts or built plugin), the Strom
  repository, and source code under copyleft or unclear licences (GPL, LGPL,
  AGPL, MPL, CC-BY-SA, unlicensed), including other DSP plugin collections,
  GStreamer's own C audio filters and Stack Overflow code listings.
- **Allowed sources only:** algorithm descriptions and equations from
  textbooks, papers and standards; public-domain material (RBJ Audio EQ
  Cookbook); docs and source of permissively licensed dependencies; the
  gstreamer-rs bindings and GStreamer API docs.
- **No transcription from memory:** each algorithm was derived from the
  references listed below, in our own structure and naming.
- **No comparison** of output, behaviour, performance or code with the plugin
  this project replaces. Correctness is verified against theory (analytic gain
  curves, transfer functions, time constants) and invariants only.
- The only thing taken from the replaced plugin is the *interface* Strom
  relies on: property names, GLib types, units, ranges and defaults.

## Algorithms

### Shared helpers (`crates/strom-dsp/src/util.rs`)

- dB/linear conversion: `20·log10(x)` and `10^(dB/20)`, with a −200 dB floor so
  results are always finite.
- One-pole smoother `y[n] = a·y[n−1] + (1−a)·x[n]`, `a = exp(−1/(τ·fs))`: the
  standard discretisation of a first-order RC low-pass (impulse-invariant
  mapping of `e^(−t/τ)`); the step response reaches 1 − 1/e ≈ 63.2 % after τ.
  Reference: J. O. Smith III, *Introduction to Digital Filters*, one-pole
  filters; Giannoulis/Massberg/Reiss 2012, section on level detection.
- Denormals: recursive states are flushed to exactly zero below 1e-15
  (≈ −300 dB). Own design choice.

<!-- Per-processor sections follow. -->

## Dependencies

| Crate | Used by | Licence |
|---|---|---|
| (none at runtime) | `strom-dsp` | |
| `gstreamer`, `gstreamer-base`, `gstreamer-audio` 0.25 | plugin | MIT OR Apache-2.0 |
| `gst-plugin-version-helper` 0.8 (build) | plugin | MIT OR Apache-2.0 |
| `gstreamer-check` 0.25 (dev) | plugin tests | MIT OR Apache-2.0 |
| `proptest` 1 (dev) | library tests | MIT OR Apache-2.0 |
| `criterion` 0.8 (dev) | benchmark | MIT OR Apache-2.0 |

Transitive dependencies are checked against the allowlist in `deny.toml` by
`cargo deny check licenses`. The GStreamer C libraries (LGPL) are system
libraries linked dynamically; they are not crates and not part of this check.
