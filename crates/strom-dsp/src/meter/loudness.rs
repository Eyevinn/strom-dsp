//! BS.1770 / EBU R 128 loudness and peak measurement, wrapping `ebur128`.

use ebur128::{Channel, EbuR128, Mode};

/// Largest channel count [`LoudnessMeter`] supports (the limit of `ebur128`).
pub const MAX_LOUDNESS_CHANNELS: usize = 64;

/// How one channel contributes to the BS.1770 loudness sum.
///
/// The weight `G` multiplies the channel's mean-square K-weighted energy
/// before the channels are summed (ITU-R BS.1770, eq. 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelRole {
    /// Front or other non-surround loudspeaker (L, R, C, rear ±135°…180°,
    /// height, unknown): `G = 1.0`.
    Front,
    /// Surround loudspeaker at 60° ≤ |azimuth| ≤ 120° (Ls/Rs of 5.x, side
    /// surrounds of 7.x): `G = 1.41`.
    Surround,
    /// Low-frequency effects: excluded (`G = 0`), as BS.1770 and EBU Tech 3341
    /// section 2.10 require.
    Lfe,
    /// Any other channel to leave out of the loudness sum (`G = 0`). Its peaks
    /// are still measured.
    Excluded,
    /// A mono signal that is reproduced on two loudspeakers: `G = 2.0`
    /// (counted twice). Only valid for a 1-channel meter.
    DualMono,
}

impl ChannelRole {
    /// The BS.1770 channel weight `G` of this role.
    pub fn weight(self) -> f64 {
        match self {
            ChannelRole::Front => 1.0,
            ChannelRole::Surround => 1.41,
            ChannelRole::Lfe | ChannelRole::Excluded => 0.0,
            ChannelRole::DualMono => 2.0,
        }
    }

    /// The `ebur128` channel whose built-in weight equals ours.
    fn to_ebur128(self) -> Channel {
        match self {
            ChannelRole::Front => Channel::Left,
            ChannelRole::Surround => Channel::LeftSurround,
            ChannelRole::Lfe | ChannelRole::Excluded => Channel::Unused,
            ChannelRole::DualMono => Channel::DualMono,
        }
    }
}

/// A channel layout, mapped to per-channel [`ChannelRole`]s by
/// [`ChannelLayout::role`].
///
/// | Layout | Channel order | Roles |
/// |---|---|---|
/// | `Unknown` (default) | any | every channel `Front` (1.0) |
/// | `Mono` | C | `Front` |
/// | `DualMono` | C | `DualMono` (2.0) |
/// | `Stereo` | L R | `Front`, `Front` |
/// | `Surround50` | L R C Ls Rs | 1.0, 1.0, 1.0, 1.41, 1.41 |
/// | `Surround51` | L R C LFE Ls Rs | 1.0, 1.0, 1.0, LFE, 1.41, 1.41 |
/// | `Surround71` | L R C LFE Lb Rb Ls Rs | 1.0, 1.0, 1.0, LFE, 1.0, 1.0, 1.41, 1.41 |
///
/// The 5.1 and 7.1 orders are the WAVE/SMPTE orders (also GStreamer's
/// default positions). In 7.1, the back pair (Lb/Rb, ±135°…150°) is outside the
/// 60°…120° surround zone and gets 1.0; the side pair (±90°) gets 1.41.
///
/// **Unknown layouts:** every channel counts with weight 1.0 and nothing is
/// excluded, because guessing that a channel is an LFE could hide real
/// programme. The broadcast chain's AGC detector does the same. If the
/// channel count does not match the layout, the meter falls back to `Unknown`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChannelLayout {
    /// Unknown positions: all channels weight 1.0.
    #[default]
    Unknown,
    /// One channel, weight 1.0.
    Mono,
    /// One channel reproduced on two loudspeakers, weight 2.0.
    DualMono,
    /// L, R.
    Stereo,
    /// L, R, C, Ls, Rs.
    Surround50,
    /// L, R, C, LFE, Ls, Rs.
    Surround51,
    /// L, R, C, LFE, Lb, Rb, Ls, Rs.
    Surround71,
}

impl ChannelLayout {
    /// Number of channels this layout describes (`None` for `Unknown`).
    pub fn channels(self) -> Option<usize> {
        match self {
            ChannelLayout::Unknown => None,
            ChannelLayout::Mono | ChannelLayout::DualMono => Some(1),
            ChannelLayout::Stereo => Some(2),
            ChannelLayout::Surround50 => Some(5),
            ChannelLayout::Surround51 => Some(6),
            ChannelLayout::Surround71 => Some(8),
        }
    }

    /// Role of channel `index` in a stream of `channels` channels. A layout
    /// whose channel count differs from `channels` is treated as `Unknown`.
    pub fn role(self, index: usize, channels: usize) -> ChannelRole {
        use ChannelRole::{DualMono, Front, Lfe, Surround};
        if self.channels() != Some(channels) {
            return Front;
        }
        match (self, index) {
            (ChannelLayout::DualMono, 0) => DualMono,
            (ChannelLayout::Surround50, 3 | 4) => Surround,
            (ChannelLayout::Surround51, 3) => Lfe,
            (ChannelLayout::Surround51, 4 | 5) => Surround,
            (ChannelLayout::Surround71, 3) => Lfe,
            (ChannelLayout::Surround71, 6 | 7) => Surround,
            _ => Front,
        }
    }
}

/// Setup parameters of a [`LoudnessMeter`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoudnessConfig {
    /// Sample rate in Hz (rounded to an integer; 8 000 … 192 000 tested).
    pub sample_rate: f32,
    /// Channel count, 1 … [`MAX_LOUDNESS_CHANNELS`].
    pub channels: usize,
    /// Channel layout, mapped to BS.1770 weights.
    pub layout: ChannelLayout,
    /// Measure true peak (4× oversampling below 96 kHz, 2× below 192 kHz; at
    /// 192 kHz and above the true peak equals the sample peak). Costs CPU.
    pub true_peak: bool,
}

impl Default for LoudnessConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48_000.0,
            channels: 2,
            layout: ChannelLayout::Unknown,
            true_peak: true,
        }
    }
}

/// Why a [`LoudnessMeter`] could not be set up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeterError {
    /// Channel count is 0 or above [`MAX_LOUDNESS_CHANNELS`].
    UnsupportedChannelCount(usize),
    /// Sample rate is not finite or outside 16 Hz … 2.8 MHz.
    UnsupportedSampleRate,
    /// A channel-role list has the wrong length, or uses
    /// [`ChannelRole::DualMono`] on a multichannel meter.
    InvalidChannelRoles,
}

impl std::fmt::Display for MeterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MeterError::UnsupportedChannelCount(n) => {
                write!(
                    f,
                    "unsupported channel count {n} (1 … {MAX_LOUDNESS_CHANNELS})"
                )
            }
            MeterError::UnsupportedSampleRate => write!(f, "unsupported sample rate"),
            MeterError::InvalidChannelRoles => write!(f, "invalid channel roles"),
        }
    }
}

impl std::error::Error for MeterError {}

/// A snapshot of all loudness values. Loudness in LUFS, range in LU, peaks as
/// linear amplitude (1.0 = 0 dBFS / 0 dBTP).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoudnessReading {
    /// Momentary loudness (last 400 ms, ungated). −∞ for digital silence.
    pub momentary: f64,
    /// Short-term loudness (last 3 s, ungated). −∞ for digital silence.
    pub short_term: f64,
    /// Integrated (gated) loudness since the last reset. −∞ until a 400 ms
    /// block above −70 LUFS has been measured.
    pub integrated: f64,
    /// Loudness range (EBU Tech 3342) since the last reset; 0 without data.
    pub loudness_range: f64,
    /// Highest sample peak over all channels since the last reset.
    pub sample_peak: f64,
    /// Highest true peak over all channels since the last reset (equals the
    /// sample peak when true peak is disabled).
    pub true_peak: f64,
}

/// EBU R 128 loudness meter: BS.1770 momentary, short-term and integrated
/// loudness, EBU Tech 3342 loudness range, sample peak and true peak.
///
/// This is a thin wrapper around the `ebur128` crate, always run in its
/// **histogram mode**: gating blocks are counted in 1000 fixed bins of
/// 0.1 LU (−70 … +30 LUFS) instead of being stored, so memory stays constant
/// however long the programme runs, and integrated loudness and LRA are
/// quantised to the bin grid (well within the EBU tolerances; see the tests).
///
/// # Real-time behaviour
///
/// [`LoudnessMeter::new`] and [`LoudnessMeter::set_channel_roles`] allocate.
/// Everything else is real-time safe (no allocation, locking or I/O; checked
/// with a counting allocator and RealtimeSanitizer):
///
/// - [`process`](Self::process): K-weighting filter, peak scan and optional
///   4× true-peak interpolation per sample, plus one histogram update per
///   100 ms. Cost is linear in the block length.
/// - [`momentary`](Self::momentary) sums the stored 400 ms of filtered audio,
///   [`short_term`](Self::short_term) the stored 3 s: the cost is proportional
///   to the window, not to the block (about 0.4·fs resp. 3·fs multiply-adds
///   per channel per call). Read them at the rate you need, e.g. every
///   10–100 ms.
/// - [`integrated`](Self::integrated) and [`loudness_range`](Self::loudness_range)
///   scan the 1000-bin histograms (constant cost; LRA uses an 8 KiB stack
///   array).
/// - Peaks are O(channels).
///
/// Measured at 48 kHz stereo (`benches/meter.rs`, Apple M5 Pro): `process`
/// of a 480-frame block takes about 7 µs (10 µs with true peak), i.e. 0.07 %
/// (0.10 %) of one core. A momentary read costs about 20 µs, a short-term
/// read about 150 µs, integrated and LRA under 1 µs each.
///
/// Do not use `ebur128`'s default (non-histogram) mode in a real-time path:
/// its block history is a `VecDeque` that reallocates after 500 s of
/// programme, and its LRA allocates and sorts on every read (see
/// `tests/meter.rs`).
#[derive(Debug)]
pub struct LoudnessMeter {
    inner: EbuR128,
    config: LoudnessConfig,
    roles: Vec<ChannelRole>,
    /// Per-channel peak hold since [`LoudnessMeter::clear_interval_peaks`].
    interval_sample_peak: Vec<f64>,
    interval_true_peak: Vec<f64>,
}

impl LoudnessMeter {
    /// Create a meter. Allocates (setup only).
    pub fn new(config: &LoudnessConfig) -> Result<Self, MeterError> {
        let channels = config.channels;
        if channels == 0 || channels > MAX_LOUDNESS_CHANNELS {
            return Err(MeterError::UnsupportedChannelCount(channels));
        }
        let rate = config.sample_rate;
        if !rate.is_finite() || !(16.0..=2_822_400.0).contains(&rate) {
            return Err(MeterError::UnsupportedSampleRate);
        }
        let mut mode = Mode::I | Mode::LRA | Mode::SAMPLE_PEAK | Mode::HISTOGRAM;
        if config.true_peak {
            mode |= Mode::TRUE_PEAK;
        }
        let inner = EbuR128::new(channels as u32, rate.round() as u32, mode)
            .map_err(|_| MeterError::UnsupportedSampleRate)?;
        let roles: Vec<ChannelRole> = (0..channels)
            .map(|c| config.layout.role(c, channels))
            .collect();
        let mut meter = Self {
            inner,
            config: *config,
            roles: Vec::new(),
            interval_sample_peak: vec![0.0; channels],
            interval_true_peak: vec![0.0; channels],
        };
        meter.set_channel_roles(&roles)?;
        Ok(meter)
    }

    /// The configuration this meter was built with.
    pub fn config(&self) -> &LoudnessConfig {
        &self.config
    }

    /// Per-channel roles currently in effect.
    pub fn channel_roles(&self) -> &[ChannelRole] {
        &self.roles
    }

    /// Override the per-channel roles (one per channel). Setup only: may
    /// allocate. Does not reset the measurement.
    pub fn set_channel_roles(&mut self, roles: &[ChannelRole]) -> Result<(), MeterError> {
        if roles.len() != self.config.channels
            || (roles.len() > 1 && roles.contains(&ChannelRole::DualMono))
        {
            return Err(MeterError::InvalidChannelRoles);
        }
        let map: Vec<Channel> = roles.iter().map(|r| r.to_ebur128()).collect();
        self.inner
            .set_channel_map(&map)
            .map_err(|_| MeterError::InvalidChannelRoles)?;
        self.roles.clear();
        self.roles.extend_from_slice(roles);
        Ok(())
    }

    /// Measure interleaved frames (`samples.len()` should be a multiple of the
    /// channel count; a trailing partial frame is ignored). Real-time safe.
    pub fn process(&mut self, samples: &[f32]) {
        let ch = self.config.channels;
        let whole = samples.len() - samples.len() % ch;
        if whole == 0 {
            return;
        }
        // Cannot fail: the length is a whole number of frames.
        let _ = self.inner.add_frames_f32(&samples[..whole]);
        let tp = self.config.true_peak;
        for c in 0..ch {
            let sp = self.inner.prev_sample_peak(c as u32).unwrap_or(0.0);
            let held = &mut self.interval_sample_peak[c];
            *held = held.max(sp);
            let tpv = if tp {
                self.inner.prev_true_peak(c as u32).unwrap_or(sp)
            } else {
                sp
            };
            let held = &mut self.interval_true_peak[c];
            *held = held.max(tpv);
        }
    }

    /// Restart every measurement (integrated, LRA, windows, filters, peaks).
    /// Real-time safe.
    pub fn reset(&mut self) {
        self.inner.reset();
        self.clear_interval_peaks();
    }

    /// Momentary loudness in LUFS (400 ms window). −∞ for silence.
    pub fn momentary(&self) -> f64 {
        self.inner.loudness_momentary().unwrap_or(f64::NEG_INFINITY)
    }

    /// Short-term loudness in LUFS (3 s window). −∞ for silence.
    pub fn short_term(&self) -> f64 {
        self.inner.loudness_shortterm().unwrap_or(f64::NEG_INFINITY)
    }

    /// Integrated (gated) loudness in LUFS since the last reset; −∞ until a
    /// gating block above −70 LUFS exists.
    pub fn integrated(&self) -> f64 {
        self.inner.loudness_global().unwrap_or(f64::NEG_INFINITY)
    }

    /// Loudness range in LU (EBU Tech 3342) since the last reset; 0 without data.
    pub fn loudness_range(&self) -> f64 {
        self.inner.loudness_range().unwrap_or(0.0)
    }

    /// Highest sample peak of `channel` since the last reset (linear).
    /// 0 for an out-of-range channel.
    pub fn sample_peak(&self, channel: usize) -> f64 {
        self.inner.sample_peak(channel as u32).unwrap_or(0.0)
    }

    /// Highest true peak of `channel` since the last reset (linear); the
    /// sample peak when true peak is disabled. 0 for an out-of-range channel.
    pub fn true_peak(&self, channel: usize) -> f64 {
        if self.config.true_peak {
            self.inner.true_peak(channel as u32).unwrap_or(0.0)
        } else {
            self.sample_peak(channel)
        }
    }

    /// Highest sample peak of `channel` since the last
    /// [`clear_interval_peaks`](Self::clear_interval_peaks) (linear).
    pub fn interval_sample_peak(&self, channel: usize) -> f64 {
        self.interval_sample_peak
            .get(channel)
            .copied()
            .unwrap_or(0.0)
    }

    /// Highest true peak of `channel` since the last
    /// [`clear_interval_peaks`](Self::clear_interval_peaks) (linear).
    pub fn interval_true_peak(&self, channel: usize) -> f64 {
        self.interval_true_peak.get(channel).copied().unwrap_or(0.0)
    }

    /// Restart the per-interval peak hold (e.g. after posting a meter
    /// message). Does not touch the since-reset peaks. Real-time safe.
    pub fn clear_interval_peaks(&mut self) {
        self.interval_sample_peak.fill(0.0);
        self.interval_true_peak.fill(0.0);
    }

    /// All values at once. Costs about as much as [`short_term`](Self::short_term)
    /// plus [`momentary`](Self::momentary). Real-time safe.
    pub fn reading(&self) -> LoudnessReading {
        let ch = self.config.channels;
        let sample_peak = (0..ch).map(|c| self.sample_peak(c)).fold(0.0, f64::max);
        let true_peak = (0..ch).map(|c| self.true_peak(c)).fold(0.0, f64::max);
        LoudnessReading {
            momentary: self.momentary(),
            short_term: self.short_term(),
            integrated: self.integrated(),
            loudness_range: self.loudness_range(),
            sample_peak,
            true_peak,
        }
    }
}
