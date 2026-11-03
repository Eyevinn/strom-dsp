# Element Guidelines

Rules every strom-dsp element follows. They apply to elements written by people and by AI agents alike, so both read the same text.

These are rules, not a description of the code. The code is the source of truth for how each element implements them.

strom-dsp elements are **real-time** processors. An operator changes them while audio is live on air. Every rule below follows from that.

## Bypass
- Every processing element has an `enabled` property, mutable in PLAYING.
- Toggling `enabled` **fades** between the processed and the dry signal over a few milliseconds (a raised-cosine or equal-gain crossfade). It never switches hard.
- The disabled steady state is bit-exact passthrough, once the fade has finished.
- Bypass does not change latency. An element with latency (lookahead, oversampling, STFT) delays the dry path by the same amount, so toggling never shifts the audio in time.

## Live parameter changes
- Every property that is mutable in PLAYING changes **without an audible click**, from any value to any other value, including extremes, while signal flows.
- Gains are smoothed. Filter coefficients are interpolated, or glide at a limited rate, so a large jump (frequency, Q, gain) is spread over time. Thresholds and ratios move without steps in the gain.
- A property that cannot change click-free while playing is marked mutable only in NULL/READY, and its documentation says why.

## Real-time safety
- The processing path does not allocate, lock, log, do I/O or format strings. Everything is allocated when the caps or structural properties are set.
- No NaN, Inf or denormal slow paths at any property value within its range, at any supported sample rate.

## Formats and latency
- Elements accept any sample rate and channel count their processing supports, and say in the pad template and docs what they don't.
- Latency is reported in the `LATENCY` query and stays constant for a given configuration. When a property changes the latency, the element posts a `Latency` message.

## Defaults
- Every property has a default that is sensible on its own for live broadcast use (speech and programme audio). Each default and its reason are documented in the README.

## Tests
Each rule has a test that fails if the rule is broken:
- **Knob test:** every PLAYING-mutable property of every element jumps between extremes and typical values while the processor is active, checked against a documented click bound.
- **Bypass test:** toggling `enabled` fades, is click-free, ends bit-exact, and doesn't change latency.
- **Real-time safety:** a counting-allocator test, and RealtimeSanitizer where available, over the processing path.
- **Interface test:** names, types, ranges, defaults and mutability flags of every property.

A new element is not done until it passes all of these.

## Provenance
- strom-dsp is a clean-room implementation. Every algorithm is written from textbooks, papers and standards, and `PROVENANCE.md` names the reference for each one.
- Never read, copy or port code from other DSP projects or products. Never tune an element to match the output of a specific product.
- A crate is reused only for what is hard to get right (FFT, standards-compliant measurement, resampling), and only under a licence on the `deny.toml` allowlist.
