//! Shared engine execution discipline: bounded wait for an engine slot,
//! then run the blocking transcribe call on the blocking pool.

use std::ops::Range;
use std::sync::Arc;
use std::time::Duration;

use crate::api::error::ApiError;
use crate::engine::EngineError;
use crate::server::AppState;

/// How long a request may wait for a free engine slot before 503.
const QUEUE_TIMEOUT: Duration = Duration::from_secs(60);

/// Transcribe pcm[range]: acquire a semaphore permit (bounded wait),
/// then call the engine via spawn_blocking. pcm is shared so callers
/// can transcribe several ranges without copying samples.
pub(crate) async fn transcribe_range(
    state: &AppState,
    pcm: Arc<Vec<f32>>,
    range: Range<usize>,
    model: Option<String>,
    language: Option<String>,
) -> Result<String, ApiError> {
    let permit = tokio::time::timeout(QUEUE_TIMEOUT, state.sem.acquire())
        .await
        .map_err(|_| ApiError::busy())?
        .map_err(|e| ApiError::internal(format!("semaphore closed: {e}")))?;
    let engine = Arc::clone(&state.engine);
    let text = tokio::task::spawn_blocking(move || {
        engine.transcribe(&pcm[range], model.as_deref(), language.as_deref())
    })
    .await
    .map_err(|e| ApiError::internal(format!("transcription task failed: {e}")))?
    .map_err(engine_error)?;
    drop(permit);
    Ok(text)
}

fn engine_error(err: EngineError) -> ApiError {
    match err {
        EngineError::UnknownModel(_) => ApiError::bad_request(err.to_string()),
        EngineError::Failed(_) => ApiError::internal(err.to_string()),
    }
}
