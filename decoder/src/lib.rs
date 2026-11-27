//! FFmpeg fallback decoder, linked directly or loaded as a versioned C plugin.
pub fn decode(data: &[u8]) -> Result<Vec<f32>, String> {
    use ffmpeg_next as ff;
    use std::io::Write;

    static FFMPEG_INIT: std::sync::Once = std::sync::Once::new();
    FFMPEG_INIT.call_once(|| {
        ff::init().expect("ffmpeg init failed");
        // Keep libav's own stderr chatter (codec warnings etc.) quiet.
        ff::util::log::set_level(ff::util::log::Level::Error);
    });

    fn dec_err(e: impl std::fmt::Display) -> String {
        e.to_string()
    }

    // libav wants seekable input; a tempfile avoids unsafe in-memory AVIO.
    let mut tmp = tempfile::NamedTempFile::new().map_err(dec_err)?;
    tmp.write_all(data).map_err(dec_err)?;

    let mut ictx = ff::format::input(&tmp.path()).map_err(dec_err)?;
    let stream = ictx
        .streams()
        .best(ff::media::Type::Audio)
        .ok_or_else(|| String::from("no audio stream"))?;
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

    let mut drain = |decoder: &mut ff::decoder::Audio, pcm: &mut Vec<f32>| -> Result<(), String> {
        let mut frame = ff::frame::Audio::empty();
        while decoder.receive_frame(&mut frame).is_ok() {
            // swresample needs a concrete channel layout, and PCM in RIFF/WAV
            // carries no channel mask, so libavcodec hands out frames with an
            // unspecified layout that has to be stamped with the default one
            // for the channel count. That count must come from the frame (or
            // the decoder), never from the unspecified layout itself: on
            // FFmpeg < 7.0 ffmpeg-next maps ChannelLayout onto the legacy
            // AVFrame.channel_layout bitmask, where "unspecified" is 0 and so
            // reports 0 channels; the default layout for 0 is empty again and
            // swr_init then fails with EINVAL. FFmpeg >= 7.0 keeps the channel
            // count in the layout even when unspecified, which is why this only
            // broke against 6.x system libs.
            if frame.channel_layout().is_empty() {
                let n = match frame.channels() {
                    0 => decoder.channels(),
                    n => n,
                };
                let layout = ff::channel_layout::ChannelLayout::default(i32::from(n));
                frame.set_channel_layout(layout);
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
                        16_000,
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
        return Err(String::from("no audio samples decoded"));
    }
    Ok(pcm)
}

// The callback copies the borrowed result before this call returns. Neither
// side frees memory allocated by the other, or passes Rust types over the ABI.
type Output = unsafe extern "C" fn(*mut std::ffi::c_void, *const f32, usize, *const u8, usize);

/// # Safety
/// Input must reference `len` readable bytes. The callback and context must
/// remain valid for this synchronous call and the callback must not unwind.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ts_decode_16k_v1(
    input: *const u8,
    len: usize,
    ctx: *mut std::ffi::c_void,
    output: Output,
) {
    let data = unsafe { std::slice::from_raw_parts(input, len) };
    let result = std::panic::catch_unwind(|| decode(data))
        .unwrap_or_else(|_| Err("decoder panicked".into()));
    unsafe {
        match result {
            Ok(pcm) => output(ctx, pcm.as_ptr(), pcm.len(), std::ptr::null(), 0),
            Err(msg) => output(ctx, std::ptr::null(), 0, msg.as_ptr(), msg.len()),
        }
    }
}
