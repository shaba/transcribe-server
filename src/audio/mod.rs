//! Audio decoding to the project-wide PCM contract: 16 kHz mono f32 in [-1, 1].
//!
//! WAV that is already 16 kHz mono (i16 or f32) is decoded natively with
//! hound; everything else (other rates, stereo, webm/ogg/mp3/...) goes
//! through system libav (ffmpeg-next: demux -> decode -> swresample).
//! No subprocesses are ever spawned.

use std::io::Cursor;

pub const TARGET_SR: usize = 16_000;

#[derive(thiserror::Error, Debug)]
pub enum AudioError {
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
    use ffmpeg_next as ff;
    use std::io::Write;

    static FFMPEG_INIT: std::sync::Once = std::sync::Once::new();
    FFMPEG_INIT.call_once(|| {
        ff::init().expect("ffmpeg init failed");
        // Keep libav's own stderr chatter (codec warnings etc.) quiet.
        ff::util::log::set_level(ff::util::log::Level::Error);
    });

    fn dec_err(e: impl std::fmt::Display) -> AudioError {
        AudioError::Decode(e.to_string())
    }

    // libav wants seekable input; a tempfile avoids unsafe in-memory AVIO.
    let mut tmp = tempfile::NamedTempFile::new().map_err(dec_err)?;
    tmp.write_all(data).map_err(dec_err)?;

    let mut ictx = ff::format::input(&tmp.path()).map_err(dec_err)?;
    let stream = ictx
        .streams()
        .best(ff::media::Type::Audio)
        .ok_or_else(|| AudioError::Decode("no audio stream".into()))?;
    let stream_index = stream.index();
    let ctx = ff::codec::context::Context::from_parameters(stream.parameters()).map_err(dec_err)?;
    let mut decoder = ctx.decoder().audio().map_err(dec_err)?;

    let dst_format = ff::format::Sample::F32(ff::format::sample::Type::Packed);
    let dst_layout = ff::channel_layout::ChannelLayout::MONO;

    // Built lazily from the first decoded frame: pre-decode parameters (and
    // WAV frames with an unspecified layout order) would make swresample
    // reject frames with "input changed".
    let mut resampler: Option<ff::software::resampling::Context> = None;
    let mut pcm: Vec<f32> = Vec::new();

    let mut drain =
        |decoder: &mut ff::decoder::Audio, pcm: &mut Vec<f32>| -> Result<(), AudioError> {
            let mut frame = ff::frame::Audio::empty();
            while decoder.receive_frame(&mut frame).is_ok() {
                if frame.channel_layout().is_empty() {
                    let n = frame.channel_layout().channels();
                    frame.set_channel_layout(ff::channel_layout::ChannelLayout::default(n));
                }
                let resampler = match resampler.as_mut() {
                    Some(r) => r,
                    None => resampler.insert(
                        ff::software::resampling::Context::get(
                            frame.format(),
                            frame.channel_layout(),
                            frame.rate(),
                            dst_format,
                            dst_layout,
                            TARGET_SR as u32,
                        )
                        .map_err(dec_err)?,
                    ),
                };
                let mut resampled = ff::frame::Audio::empty();
                resampler.run(&frame, &mut resampled).map_err(dec_err)?;
                if resampled.samples() > 0 {
                    pcm.extend_from_slice(resampled.plane::<f32>(0));
                }
            }
            Ok(())
        };

    for (s, packet) in ictx.packets() {
        if s.index() != stream_index {
            continue;
        }
        // Skip undecodable packets; container-level garbage already failed above.
        if decoder.send_packet(&packet).is_ok() {
            drain(&mut decoder, &mut pcm)?;
        }
    }
    let _ = decoder.send_eof();
    drain(&mut decoder, &mut pcm)?;

    // Drain samples buffered inside the resampler.
    if let Some(mut resampler) = resampler {
        loop {
            let mut resampled = ff::frame::Audio::new(dst_format, 4096, dst_layout);
            let delay = resampler.flush(&mut resampled).map_err(dec_err)?;
            if resampled.samples() > 0 {
                pcm.extend_from_slice(&resampled.plane::<f32>(0)[..resampled.samples()]);
            }
            if delay.is_none() || resampled.samples() == 0 {
                break;
            }
        }
    }

    if pcm.is_empty() {
        return Err(AudioError::Decode("no audio samples decoded".into()));
    }
    Ok(pcm)
}

#[cfg(not(feature = "audio-ffmpeg"))]
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
    fn garbage_bytes_return_decode_error() {
        let err = decode_to_pcm_16k(b"not audio").unwrap_err();
        assert!(matches!(err, AudioError::Decode(_)));
    }
}
