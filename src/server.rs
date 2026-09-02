//! AppState and router assembly.

use std::sync::Arc;

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::middleware;
use axum::routing::{get, post};

use crate::auth::{AuthKeys, require_api_key};
use crate::config::Config;
use crate::engine::SttEngine;

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<dyn SttEngine>,
    pub cfg: Arc<Config>,
    pub sem: Arc<tokio::sync::Semaphore>, // permits = cfg.parallel
}

impl AppState {
    pub fn new(engine: Arc<dyn SttEngine>, cfg: Arc<Config>) -> Self {
        let sem = Arc::new(tokio::sync::Semaphore::new(cfg.parallel));
        Self { engine, cfg, sem }
    }

    /// Chunk length to use for a request naming `model`: `--chunk-max-sec`,
    /// lowered to the model's own limit when it reports a shorter one.
    ///
    /// The flag is the operator's choice and is never raised past it -- a
    /// model that accepts an hour of audio is still chunked at the configured
    /// length, because chunking is also what keeps latency and memory bounded.
    /// Lowering it is not optional though: a chunk past the model's window is
    /// either refused outright (hard-context-cap families answer
    /// INPUT_TOO_LONG, which reaches the client as a 500) or accepted at a
    /// quality the model was not trained for (the soft-window families, which
    /// only log a warning). The chunk lands strictly inside the window, see
    /// [`MODEL_WINDOW_MARGIN_SEC`].
    pub fn chunk_max_sec(&self, model: Option<&str>) -> f32 {
        self.chunk_max_sec_for(self.engine.resolve_model(model).as_ref())
    }

    /// [`AppState::chunk_max_sec`] for a caller that already resolved the
    /// model, so a request path does not clone a ModelInfo twice.
    pub fn chunk_max_sec_for(&self, model: Option<&crate::engine::ModelInfo>) -> f32 {
        let configured = self.cfg.chunk_max_sec;
        match model.and_then(|m| m.max_audio_sec) {
            Some(limit) if limit > 0.0 => {
                let usable = window_inside(limit);
                if usable < configured {
                    usable
                } else {
                    configured
                }
            }
            _ => configured,
        }
    }
}

/// Seconds to stay clear of a model's advertised limit.
///
/// The number the library reports is derived by inverting a frame count, and
/// the front-end that later counts frames rounds the other way, so a chunk of
/// exactly the advertised length can still come out one frame too long -- and
/// the families that enforce the limit answer that with an error, not a
/// shorter transcript. One 30 ms frame would do arithmetically; half a second
/// costs nothing and covers the families that derive the number from a
/// representative prompt length rather than from the request's own.
const MODEL_WINDOW_MARGIN_SEC: f32 = 0.5;

/// The longest chunk that safely fits a model advertising `limit` seconds.
/// Never less than half the limit, so a small window stays usable.
fn window_inside(limit: f64) -> f32 {
    (limit - f64::from(MODEL_WINDOW_MARGIN_SEC)).max(limit * 0.5) as f32
}

/// Auth applies to /v1/* only; /health is open (llama.cpp style).
pub fn build_router(state: AppState, keys: AuthKeys) -> Router {
    let v1 = Router::new()
        .route("/v1/models", get(crate::api::models::list_models))
        .route(
            "/v1/audio/transcriptions",
            post(crate::api::transcriptions::transcribe),
        )
        .route(
            "/v1/audio/translations",
            post(crate::api::transcriptions::translate),
        )
        .route("/v1/audio/stream", get(crate::api::stream::stream))
        .layer(middleware::from_fn_with_state(keys, require_api_key));
    Router::new()
        .route("/health", get(crate::api::health::health))
        .merge(v1)
        .layer(DefaultBodyLimit::max(state.cfg.max_upload_mb * 1024 * 1024))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{CancelFlag, EngineError, ModelInfo, TranscribeOptions, Transcript};
    use clap::Parser;

    /// Engine reporting exactly the model limits a test asks it for.
    struct Limits(Vec<ModelInfo>);

    impl SttEngine for Limits {
        fn transcribe(
            &self,
            _pcm: &[f32],
            _options: &TranscribeOptions,
            _cancel: &CancelFlag,
        ) -> Result<Transcript, EngineError> {
            Ok(Transcript::default())
        }

        fn models(&self) -> Vec<ModelInfo> {
            self.0.clone()
        }

        fn backend(&self) -> String {
            "test".to_string()
        }
    }

    fn model(id: &str, max_audio_sec: Option<f64>) -> ModelInfo {
        ModelInfo {
            id: id.to_string(),
            max_audio_sec,
            ..ModelInfo::default()
        }
    }

    fn app(models: Vec<ModelInfo>, chunk_max_sec: &str) -> AppState {
        let cfg =
            Config::try_parse_from(["ts", "--chunk-max-sec", chunk_max_sec]).expect("parse config");
        AppState::new(Arc::new(Limits(models)), Arc::new(cfg))
    }

    #[test]
    fn a_shorter_model_limit_lowers_the_configured_chunk() {
        let state = app(vec![model("short", Some(8.0))], "25");
        assert_eq!(state.chunk_max_sec(Some("short")), 7.5);
    }

    /// The chunk must land strictly inside the model's window: a chunk of
    /// exactly the advertised length can still be one frame too long once the
    /// front-end counts frames, and the families that enforce the limit answer
    /// that with an error rather than a shorter transcript.
    #[test]
    fn the_model_limit_is_never_used_to_its_last_second() {
        for limit in [2.0f64, 8.0, 30.0, 600.0] {
            let state = app(vec![model("m", Some(limit))], "100000");
            let chunk = f64::from(state.chunk_max_sec(Some("m")));
            assert!(chunk < limit, "{chunk} does not stay inside {limit}");
            assert!(chunk >= limit * 0.5, "{chunk} wastes half of {limit}");
        }
    }

    #[test]
    fn a_longer_model_limit_does_not_raise_it() {
        let state = app(vec![model("long", Some(600.0))], "25");
        assert_eq!(state.chunk_max_sec(Some("long")), 25.0);
    }

    /// A model whose limit equals the configured chunk length still has to be
    /// chunked inside its window, not exactly at it.
    #[test]
    fn a_limit_equal_to_the_configured_chunk_still_lowers_it() {
        let state = app(vec![model("same", Some(25.0))], "25");
        assert_eq!(state.chunk_max_sec(Some("same")), 24.5);
    }

    #[test]
    fn an_unreported_limit_keeps_the_configured_chunk() {
        let state = app(vec![model("any", None)], "25");
        assert_eq!(state.chunk_max_sec(Some("any")), 25.0);
        // Zero is how the library spells "no limit of my own".
        let state = app(vec![model("any", Some(0.0))], "25");
        assert_eq!(state.chunk_max_sec(Some("any")), 25.0);
    }

    /// An unknown alias runs on the default model, so its limit is the one
    /// that applies -- the same fallback the engine itself makes.
    #[test]
    fn an_unknown_alias_uses_the_default_models_limit() {
        let state = app(
            vec![model("first", Some(8.0)), model("second", Some(20.0))],
            "25",
        );
        assert_eq!(state.chunk_max_sec(Some("no-such-model")), 7.5);
        assert_eq!(state.chunk_max_sec(None), 7.5);
        assert_eq!(state.chunk_max_sec(Some("second")), 19.5);
    }

    #[test]
    fn no_models_loaded_keeps_the_configured_chunk() {
        let state = app(Vec::new(), "25");
        assert_eq!(state.chunk_max_sec(None), 25.0);
    }
}
