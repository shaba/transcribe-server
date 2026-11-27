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
    /// Pins the verified inode while dlopen runs (Linux opens the plugin
    /// through this descriptor, so a rename cannot swap it in between).
    _file: Option<std::fs::File>,
}

#[cfg(unix)]
fn open_verified(path: &Path) -> Result<std::fs::File, String> {
    use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};

    let file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)
        .map_err(|e| format!("opening {}: {e}", path.display()))?;
    let meta = file
        .metadata()
        .map_err(|e| format!("inspecting {}: {e}", path.display()))?;
    if !meta.is_file() {
        return Err(format!("{} is not a regular file", path.display()));
    }
    if meta.permissions().mode() & 0o022 != 0 {
        return Err(format!(
            "{} must not be group- or world-writable",
            path.display()
        ));
    }
    // Running as root is fine; otherwise the file must be the server user's
    // own, not something another account can replace.
    let euid = unsafe { libc::geteuid() };
    if meta.uid() != 0 && meta.uid() != euid {
        return Err(format!(
            "{} must be owned by root or the server user",
            path.display()
        ));
    }
    Ok(file)
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
    #[cfg(unix)]
    let (library_path, file) = {
        let file = open_verified(path)?;
        // Linux only: dlopen the already-verified descriptor instead of the
        // path, so a rename between the check and the load cannot swap it.
        #[cfg(target_os = "linux")]
        let library_path = {
            use std::os::fd::AsRawFd;
            std::path::PathBuf::from(format!("/proc/self/fd/{}", file.as_raw_fd()))
        };
        #[cfg(not(target_os = "linux"))]
        let library_path = path.to_path_buf();
        (library_path, Some(file))
    };
    #[cfg(not(unix))]
    let (library_path, file) = (path.to_path_buf(), None);

    // SAFETY: deployment supplies the trusted decoder built with this binary;
    // the versioned entry point must implement the synchronous C ABI below.
    unsafe {
        let library = Library::new(&library_path)
            .map_err(|e| format!("loading {}: {e}", library_path.display()))?;
        let call = *library
            .get::<Decode>(b"ts_decode_16k_v1\0")
            .map_err(|e| e.to_string())?;
        Ok(Decoder {
            call,
            _library: library,
            _file: file,
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
    let decoder = DECODER
        .get_or_init(|| {
            let loaded = load();
            if let Err(e) = &loaded {
                // Once per process, not once per request.
                tracing::error!(%e, "FFmpeg decoder unavailable");
            }
            loaded
        })
        .as_ref()
        .map_err(|_| AudioError::Unavailable)?;
    let mut slot: ResultSlot = None;
    unsafe {
        (decoder.call)(data.as_ptr(), data.len(), (&raw mut slot).cast(), receive);
    }
    slot.unwrap_or(Err(AudioError::Unavailable))
}
