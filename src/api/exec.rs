//! Shared engine execution discipline: bounded wait for an engine slot,
//! then run the blocking transcribe call on the blocking pool.

use std::ops::Range;
use std::sync::Arc;
use std::time::Duration;

use crate::api::error::ApiError;
use crate::audio::samples_to_sec;
use crate::engine::cancel::CancelOnDrop;
use crate::engine::{CancelFlag, EngineError, TranscribeOptions, Transcript};
use crate::server::AppState;

/// How long a request may wait for a free engine slot before 503.
const QUEUE_TIMEOUT: Duration = Duration::from_secs(60);

/// Transcribe pcm[range]: acquire a semaphore permit (bounded wait),
/// then call the engine via spawn_blocking. pcm and the options are shared so
/// callers can transcribe several ranges without copying samples or rebuilding
/// the same request options per chunk.
///
/// The engine times its rows from the start of the slice it was handed, so
/// this is where `range.start` is added back: the returned transcript is
/// timed relative to the start of `pcm`, whatever sub-range was transcribed.
/// Applying the offset here rather than in the callers keeps it applied
/// exactly once, on every path.
pub(crate) async fn transcribe_range(
    state: &AppState,
    pcm: Arc<Vec<f32>>,
    range: Range<usize>,
    options: Arc<TranscribeOptions>,
) -> Result<Transcript, ApiError> {
    let offset = samples_to_sec(range.start);
    // Owned, and moved into the blocking task below: a permit tied to this
    // future would be released the moment a client hangs up, while the run it
    // paid for is still winding down (and on a family that ignores the abort
    // callback, still running to the end). The next request would then walk
    // past --parallel and its bounded wait straight into the model's own lock.
    let permit = tokio::time::timeout(QUEUE_TIMEOUT, Arc::clone(&state.sem).acquire_owned())
        .await
        .map_err(|_| ApiError::busy())?
        .map_err(|e| ApiError::internal(format!("semaphore closed: {e}")))?;
    let engine = Arc::clone(&state.engine);
    // A blocking task runs to completion whatever the runtime does with the
    // future waiting on it, so a client that hangs up mid-request would keep a
    // core and the engine slot busy transcribing an answer nobody will read.
    // The guard turns dropping this future into a cancel the engine can act on.
    let cancel = CancelFlag::new();
    let guard = CancelOnDrop::new(cancel.clone());
    let mut transcript = tokio::task::spawn_blocking(move || {
        let result = engine.transcribe(&pcm[range], &options, &cancel);
        drop(permit);
        result
    })
    .await
    .map_err(|e| ApiError::internal(format!("transcription task failed: {e}")))?
    .map_err(engine_error)?;
    guard.disarm();
    transcript.shift(offset);
    Ok(transcript)
}

fn engine_error(err: EngineError) -> ApiError {
    match err {
        EngineError::UnknownModel(_) | EngineError::Unsupported(_) => {
            ApiError::bad_request(err.to_string())
        }
        // Only reachable when the caller is already gone, so this response is
        // built for the log and never delivered; 499 is the status the rest of
        // the world uses for it.
        EngineError::Cancelled => ApiError::cancelled(),
        EngineError::Failed(_) => ApiError::internal(err.to_string()),
    }
}
