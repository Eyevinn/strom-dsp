# Element Guidelines

Rules every strom-dsp element follows.

These are rules, not a description of the code. The code is the source of truth for how each element implements them.

strom-dsp elements are **real-time** processors. An operator changes them while audio is live on air. Every rule below follows from that.

## Bypass (true bypass)
- Every processing element has an `enabled` property, mutable in PLAYING.
- Toggling `enabled` **fades** between the processed and the dry signal over a few milliseconds (a raised-cosine or equal-gain crossfade). It never switches hard.
- Once the fade has finished, the output is the input sample for sample, delayed by exactly the element's reported latency. For an element without latency that means bit-exact with no delay.
- Toggling `enabled` never changes the reported latency and never shifts the audio in time: an element with latency (lookahead, oversampling, STFT) delays the dry path by the same amount.

## Live parameter changes
- Every property that is mutable in PLAYING changes **without an audible click**, from any value to any other value, including extremes, while signal flows.
- Gains are smoothed. Filter coefficients are interpolated, or glide at a limited rate, so a large jump (frequency, Q, gain) is spread over time. Thresholds and ratios move without steps in the gain.
- A property that cannot change click-free while playing is marked mutable only in NULL/READY, and its documentation says why.

## Real-time safety
- The processing path does not allocate, lock, log, do I/O or format strings. Everything is allocated when the caps or structural properties are set.
- No NaN, Inf or denormal slow paths at any property value within its range, at any supported sample rate.

## Formats
- Elements accept any sample rate and channel count their processing supports, and say in the pad template and docs what they don't.

## Latency
- An element that doesn't need latency adds none: no internal buffering, and it leaves the `LATENCY` query unchanged.
- An element that needs latency (lookahead, oversampling filters, STFT) reports all of it in the `LATENCY` query, keeps it constant for a given configuration, and posts a `Latency` message when a property change alters it.
- Latency is kept to what the processing needs. Where it is a trade-off (e.g. lookahead), it is a property with a documented default.
- Latency adds up along a live chain (processing, mixing, encoding, network), and two-way use (commentary, talkback, calls) feels every millisecond of it. Treat latency as a cost to minimise, not a budget to fill.
- Stages in a chain share the work instead of each keeping full margin. A later safety stage gets a shorter lookahead when an earlier stage already handles most of what it catches, e.g. a limiter after a clipper.
- An element or stage with latency makes it configurable, documents its default and the shortest setting that still keeps its guarantees, and its tests measure what a shorter setting costs (overshoot, distortion, true-peak margin).
- An element built from several stages states a total latency budget on its page, for its default and its low-latency configuration, and a test fails when a configuration exceeds it.
- Each element's page under [docs/elements/](elements/README.md) gives its latency, or its formula.
- An intentional content delay (e.g. `stromdsp-delay`) is not latency: it is not reported, and timestamps are unchanged.

## Defaults
- Every property has a default that is sensible on its own for live broadcast use (speech and programme audio). Each default and its reason are documented on the element's page under [docs/elements/](elements/README.md).
- Defaults and presets are gentle. The target is TV broadcast and streaming: consistent loudness and a clean, natural sound, not maximum loudness. Under-processing is the safer failure; an operator can always push harder.

## Protective stages
- A stage that protects against events (a peak, a click, clipping on the input) is transparent when there is nothing to protect against: its output is its input, sample for sample, delayed by its reported latency.

## Tests
Each rule has a test that fails if the rule is broken:
- **Knob test:** every PLAYING-mutable property of every element jumps between extremes and typical values while the processor is active, checked against a documented click bound.
- **Bypass test:** toggling `enabled` fades and is click-free; after the fade, the output equals the input delayed by exactly the reported latency, sample for sample.
- **Latency test:** an impulse comes out delayed by exactly the reported latency, with `enabled` true and false, and the `LATENCY` query matches. Zero-latency elements report and add zero.
- **Real-time safety:** a counting-allocator test, and RealtimeSanitizer where available, over the processing path.
- **Interface test:** names, types, ranges, defaults and mutability flags of every property.
- **Latency budget test:** for an element built from several stages, the reported latency of each named configuration stays within the budget on its page.
- **Transparency test:** a protective stage fed clean programme changes no sample.

A new element is not done until it passes all of these.

Keep the default test suite fast: `cargo test` should finish in about a minute on a laptop.
- Default tests use short signals (a fraction of a second where the property allows it) and check correctness against theory: frequency responses, gain curves, ceilings, latency, interface.
- Long programme runs (tens of seconds of audio, preset × level × programme grids, loudness and distortion figures) are measurements, not tests. They live in the measurement harness and the extended CI workflow, not in the default suite.
- RealtimeSanitizer runs once over each element's processing path, not once per internal component.
- Property tests use a small number of cases by default and more in the extended workflow.

## Provenance
- strom-dsp is an independent implementation from public knowledge. It is not a traditional clean room (no separate specification team); what it holds to is the source rules in `PROVENANCE.md`. Every algorithm is written from textbooks, papers and standards, and `PROVENANCE.md` names the reference for each one.
- Never read, copy or port code from other DSP projects or products. Never tune an element to match the output of a specific product.
- A crate is reused only for what is hard to get right (FFT, standards-compliant measurement, resampling), and only under a licence on the `deny.toml` allowlist.
