# Element Guidelines

Rules every strom-dsp element follows. They apply to elements written by people and by AI agents alike, so both read the same text.

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
- The README lists each element's latency, or its formula.
- An intentional content delay (e.g. `stromdsp-delay`) is not latency: it is not reported, and timestamps are unchanged.

## Defaults
- Every property has a default that is sensible on its own for live broadcast use (speech and programme audio). Each default and its reason are documented in the README.

## Tests
Each rule has a test that fails if the rule is broken:
- **Knob test:** every PLAYING-mutable property of every element jumps between extremes and typical values while the processor is active, checked against a documented click bound.
- **Bypass test:** toggling `enabled` fades and is click-free; after the fade, the output equals the input delayed by exactly the reported latency, sample for sample.
- **Latency test:** an impulse comes out delayed by exactly the reported latency, with `enabled` true and false, and the `LATENCY` query matches. Zero-latency elements report and add zero.
- **Real-time safety:** a counting-allocator test, and RealtimeSanitizer where available, over the processing path.
- **Interface test:** names, types, ranges, defaults and mutability flags of every property.

A new element is not done until it passes all of these.

## Provenance
- strom-dsp is a clean-room implementation. Every algorithm is written from textbooks, papers and standards, and `PROVENANCE.md` names the reference for each one.
- Never read, copy or port code from other DSP projects or products. Never tune an element to match the output of a specific product.
- A crate is reused only for what is hard to get right (FFT, standards-compliant measurement, resampling), and only under a licence on the `deny.toml` allowlist.
