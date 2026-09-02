//! Shared engine execution discipline: bounded wait for an engine slot,
//! then run the blocking transcribe call on the blocking pool.

use std::ops::Range;
use std::sync::Arc;
use std::time::Duration;

use crate::api::error::ApiError;
use crate::audio::samples_to_sec;
use crate::engine::{EngineError, TranscribeOptions, Transcript};
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
    let permit = tokio::time::timeout(QUEUE_TIMEOUT, state.sem.acquire())
        .await
        .map_err(|_| ApiError::busy())?
        .map_err(|e| ApiError::internal(format!("semaphore closed: {e}")))?;
    let engine = Arc::clone(&state.engine);
    let mut transcript =
        tokio::task::spawn_blocking(move || engine.transcribe(&pcm[range], &options))
            .await
            .map_err(|e| ApiError::internal(format!("transcription task failed: {e}")))?
            .map_err(engine_error)?;
    drop(permit);
    transcript.shift(offset);
    Ok(transcript)
}

fn engine_error(err: EngineError) -> ApiError {
    match err {
        EngineError::UnknownModel(_) => ApiError::bad_request(err.to_string()),
        EngineError::Failed(_) => ApiError::internal(err.to_string()),
    }
}
