//! Audio file I/O for the listening kit: decoding and AAC encoding through
//! the `ffmpeg` command-line tool, and a small WAV writer.

use std::io::{self, Write};
use std::path::Path;
use std::process::{Command, Stdio};

/// Decode `path` to interleaved `f32` at 48 kHz with its own channel count
/// (`channels`), through `ffmpeg`.
pub fn decode(path: &Path, channels: usize) -> io::Result<Vec<f32>> {
    let out = Command::new("ffmpeg")
        .args(["-nostdin", "-v", "error", "-i"])
        .arg(path)
        .args(["-ar", "48000", "-ac"])
        .arg(channels.to_string())
        .args(["-f", "f32le", "-acodec", "pcm_f32le", "-"])
        .stdin(Stdio::null())
        .output()?;
    if !out.status.success() {
        return Err(io::Error::other(format!(
            "ffmpeg could not decode {}: {}",
            path.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        )));
    }
    Ok(out
        .stdout
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

/// Write interleaved `x` (`channels`, 48 kHz) as a 24-bit PCM WAV file,
/// scaled by `gain` (linear). Samples beyond full scale are clamped.
pub fn write_wav24(path: &Path, x: &[f32], channels: usize, gain: f64) -> io::Result<()> {
    let data_len = (x.len() * 3) as u32;
    let mut b = Vec::with_capacity(44 + x.len() * 3);
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + data_len).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes()); // PCM
    b.extend_from_slice(&(channels as u16).to_le_bytes());
    b.extend_from_slice(&48_000u32.to_le_bytes());
    b.extend_from_slice(&(48_000 * 3 * channels as u32).to_le_bytes());
    b.extend_from_slice(&(3 * channels as u16).to_le_bytes());
    b.extend_from_slice(&24u16.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&data_len.to_le_bytes());
    const FULL: f64 = 8_388_607.0;
    for &v in x {
        let s = (v as f64 * gain * FULL).round().clamp(-FULL - 1.0, FULL) as i32;
        b.extend_from_slice(&s.to_le_bytes()[..3]);
    }
    std::fs::File::create(path)?.write_all(&b)
}

/// Encode the WAV file `wav` as AAC in MP4 (`mp4`) through `ffmpeg`:
/// 160 kbit/s for stereo, 96 kbit/s for mono.
pub fn encode_mp4(wav: &Path, mp4: &Path, channels: usize) -> io::Result<()> {
    let rate = if channels == 1 { "96k" } else { "160k" };
    let out = Command::new("ffmpeg")
        .args(["-nostdin", "-v", "error", "-y", "-i"])
        .arg(wav)
        .args(["-c:a", "aac", "-b:a", rate, "-movflags", "+faststart"])
        .arg(mp4)
        .stdin(Stdio::null())
        .output()?;
    if !out.status.success() {
        return Err(io::Error::other(format!(
            "ffmpeg could not encode {}: {}",
            mp4.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        )));
    }
    Ok(())
}
