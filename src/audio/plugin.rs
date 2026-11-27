use super::AudioError;
use libloading::Library;
use std::ffi::c_void;
use std::path::Path;
use std::sync::OnceLock;

type Emit = unsafe extern "C" fn(*mut c_void, *const f32, usize, *const u8, usize);
type Decode = unsafe extern "C" fn(*const u8, usize, *mut c_void, Emit);
struct Decoder {
    call: Decode,
    _library: Library,
}

fn load() -> Result<Decoder, String> {
    // Absolute, build-time path: never search cwd or accept a request-supplied library.
    let path = Path::new(
        option_env!("TRANSCRIBE_DECODER_PATH")
            .unwrap_or("/opt/transcribe/lib/libtranscribe_decoder.so"),
    );
    if !path.is_absolute() {
        return Err("decoder path must be absolute".into());
    }
    // SAFETY: deployment supplies the trusted decoder built with this binary;
    // the versioned entry point must implement the synchronous C ABI below.
    unsafe {
        let library = Library::new(path).map_err(|e| format!("loading {}: {e}", path.display()))?;
        let call = *library
            .get::<Decode>(b"ts_decode_16k_v1\0")
            .map_err(|e| e.to_string())?;
        Ok(Decoder {
            call,
            _library: library,
        })
    }
}

type ResultSlot = Option<Result<Vec<f32>, AudioError>>;
unsafe extern "C" fn receive(
    ctx: *mut c_void,
    pcm: *const f32,
    n: usize,
    msg: *const u8,
    len: usize,
) {
    // SAFETY: the caller supplies a live ResultSlot; the plugin supplies one
    // borrowed slice, valid until the callback returns. Never retain its pointers.
    unsafe {
        *ctx.cast::<ResultSlot>() = Some(if msg.is_null() {
            Ok(std::slice::from_raw_parts(pcm, n).to_vec())
        } else {
            Err(AudioError::Decode(
                String::from_utf8_lossy(std::slice::from_raw_parts(msg, len)).into_owned(),
            ))
        });
    }
}

pub(super) fn decode_via_ffmpeg(data: &[u8]) -> Result<Vec<f32>, AudioError> {
    static DECODER: OnceLock<Result<Decoder, String>> = OnceLock::new();
    let decoder = DECODER.get_or_init(load).as_ref().map_err(|e| {
        tracing::error!(%e,"FFmpeg decoder unavailable");
        AudioError::Unavailable
    })?;
    let mut slot: ResultSlot = None;
    unsafe {
        (decoder.call)(data.as_ptr(), data.len(), (&raw mut slot).cast(), receive);
    }
    slot.unwrap_or(Err(AudioError::Unavailable))
}
