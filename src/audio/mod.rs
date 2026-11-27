//! Audio decoding to the project-wide PCM contract: 16 kHz mono f32 in [-1, 1].
//!
//! WAV that is already 16 kHz mono (i16 or f32) is decoded natively with
//! hound; everything else (other rates, stereo, webm/ogg/mp3/...) goes
//! through system libav (demux -> decode -> swresample). The libav path is
//! either linked through the `transcribe-decoder` crate or, under
//! `lazy-ffmpeg`, loaded as a shared plugin on the first request that needs
//! it. No subprocesses are ever spawned.

// `lazy-ffmpeg` is the alternative to the default eager `audio-ffmpeg`; both
// at once would silently pick the eager path and leave libloading unused.
#[cfg(all(feature = "audio-ffmpeg", feature = "lazy-ffmpeg"))]
compile_error!(
    "enable either `audio-ffmpeg` (default) or `lazy-ffmpeg`, not both; \
     build the lazy variant with --no-default-features --features lazy-ffmpeg"
);

use std::io::Cursor;

pub const TARGET_SR: usize = 16_000;

/// Sample count -> seconds at the project-wide sample rate. The one place
/// that conversion happens, so a sample index and a timestamp can never drift
/// apart by a stray factor.
pub fn samples_to_sec(samples: usize) -> f32 {
    samples as f32 / TARGET_SR as f32
}

#[derive(thiserror::Error, Debug)]
pub enum AudioError {
    #[error("audio decoder unavailable")]
    Unavailable,
    #[error("unsupported or corrupt audio: {0}")]
    Decode(String),
}

/// Any container/codec -> 16 kHz mono f32 [-1,1].
pub fn decode_to_pcm_16k(data: &[u8]) -> Result<Vec<f32>, AudioError> {
    if let Some(pcm) = wav_fastpath(data) {
        return Ok(pcm);
    }
    decode_via_ffmpeg(data)
}

/// Native decode for WAV already matching the target format: 16 kHz mono,
/// i16 or f32 samples. Returns None (fall through to libav) for anything else.
fn wav_fastpath(data: &[u8]) -> Option<Vec<f32>> {
    if !data.starts_with(b"RIFF") {
        return None;
    }
    let mut reader = hound::WavReader::new(Cursor::new(data)).ok()?;
    let spec = reader.spec();
    if spec.sample_rate as usize != TARGET_SR || spec.channels != 1 {
        return None;
    }
    match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .map(|s| s.map(|v| f32::from(v) / 32768.0))
            .collect::<Result<Vec<f32>, _>>()
            .ok(),
        (hound::SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .collect::<Result<Vec<f32>, _>>()
            .ok(),
        _ => None,
    }
}

#[cfg(feature = "audio-ffmpeg")]
fn decode_via_ffmpeg(data: &[u8]) -> Result<Vec<f32>, AudioError> {
    transcribe_decoder::decode(data).map_err(AudioError::Decode)
}

#[cfg(all(feature = "lazy-ffmpeg", not(feature = "audio-ffmpeg")))]
mod plugin;
#[cfg(all(feature = "lazy-ffmpeg", not(feature = "audio-ffmpeg")))]
use plugin::decode_via_ffmpeg;

#[cfg(not(any(feature = "audio-ffmpeg", feature = "lazy-ffmpeg")))]
fn decode_via_ffmpeg(_data: &[u8]) -> Result<Vec<f32>, AudioError> {
    Err(AudioError::Decode("built without ffmpeg support".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "audio-ffmpeg")]
    fn fixture(name: &str) -> Vec<u8> {
        let path = format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), name);
        std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"))
    }

    #[cfg(feature = "audio-ffmpeg")]
    fn rms(pcm: &[f32]) -> f32 {
        (pcm.iter().map(|s| s * s).sum::<f32>() / pcm.len() as f32).sqrt()
    }

    fn sine_wav(sample_rate: u32, channels: u16) -> Vec<u8> {
        let spec = hound::WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut cursor = Cursor::new(Vec::new());
        {
            let mut writer = hound::WavWriter::new(&mut cursor, spec).unwrap();
            for k in 0..sample_rate as usize {
                let v = (2.0 * std::f64::consts::PI * 440.0 * k as f64 / sample_rate as f64).sin()
                    * 0.5;
                let s = (v * 32767.0) as i16;
                for _ in 0..channels {
                    writer.write_sample(s).unwrap();
                }
            }
            writer.finalize().unwrap();
        }
        cursor.into_inner()
    }

    #[test]
    fn wav_16k_mono_i16_fastpath() {
        let data = sine_wav(TARGET_SR as u32, 1);
        let pcm = decode_to_pcm_16k(&data).expect("decode 16k mono wav");
        assert_eq!(pcm.len(), TARGET_SR);
        for k in [0usize, 1, 100, 8_000, 15_999] {
            let expected =
                (2.0 * std::f64::consts::PI * 440.0 * k as f64 / TARGET_SR as f64).sin() * 0.5;
            assert!(
                (pcm[k] - expected as f32).abs() < 1e-3,
                "sample {k}: got {}, expected {expected}",
                pcm[k]
            );
        }
    }

    // The lazy build has no plugin in the test environment; its fallback path
    // is covered by the Unavailable branch of the garbage-bytes test.
    #[cfg(feature = "audio-ffmpeg")]
    #[test]
    fn wav_44k_resamples_to_16k() {
        let pcm = decode_to_pcm_16k(&fixture("beep-44k.wav")).expect("decode 44.1k wav");
        let lo = TARGET_SR * 95 / 100;
        let hi = TARGET_SR * 105 / 100;
        assert!(
            (lo..=hi).contains(&pcm.len()),
            "expected ~{TARGET_SR} samples, got {}",
            pcm.len()
        );
        assert!(rms(&pcm) > 0.05, "silent output, rms = {}", rms(&pcm));
    }

    #[cfg(feature = "audio-ffmpeg")]
    #[test]
    fn wav_16k_stereo_falls_through_to_ffmpeg() {
        let data = sine_wav(TARGET_SR as u32, 2);
        let pcm = decode_to_pcm_16k(&data).expect("decode 16k stereo wav");
        let lo = TARGET_SR * 95 / 100;
        let hi = TARGET_SR * 105 / 100;
        assert!(
            (lo..=hi).contains(&pcm.len()),
            "expected ~{TARGET_SR} samples, got {}",
            pcm.len()
        );
        assert!(rms(&pcm) > 0.05, "silent output, rms = {}", rms(&pcm));
    }

    #[cfg(feature = "audio-ffmpeg")]
    #[test]
    fn webm_and_ogg_decode_to_about_one_second() {
        for name in ["beep.webm", "beep.ogg"] {
            let pcm = decode_to_pcm_16k(&fixture(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert!(!pcm.is_empty(), "{name}: empty output");
            let lo = TARGET_SR * 90 / 100;
            let hi = TARGET_SR * 110 / 100;
            assert!(
                (lo..=hi).contains(&pcm.len()),
                "{name}: expected ~1 s (~{TARGET_SR} samples), got {}",
                pcm.len()
            );
            assert!(
                rms(&pcm) > 0.05,
                "{name}: silent output, rms = {}",
                rms(&pcm)
            );
        }
    }

    #[test]
    fn samples_to_sec_converts_at_the_target_rate() {
        assert_eq!(samples_to_sec(0), 0.0);
        assert_eq!(samples_to_sec(TARGET_SR), 1.0);
        assert_eq!(samples_to_sec(TARGET_SR / 2), 0.5);
        assert_eq!(samples_to_sec(TARGET_SR * 3600), 3600.0);
    }

    #[test]
    fn garbage_bytes_return_an_error() {
        let err = decode_to_pcm_16k(b"not audio").unwrap_err();
        // Under lazy-ffmpeg the plugin may be absent, which is Unavailable
        // rather than a decode failure.
        assert!(matches!(
            err,
            AudioError::Decode(_) | AudioError::Unavailable
        ));
    }
}
