//! Real STT engine backed by the transcribe-cpp crate (transcribe.cpp / ggml).
//!
//! Concurrency: the 0.x C library allows at most one in-flight compute per
//! model, and the crate's `Session` is Send but not Sync (`run` takes
//! `&mut self`). We therefore keep one long-lived `Session` per model behind a
//! `Mutex`: session init allocates decoder/KV state, so creating one per call
//! would be wasted work, and with a single model `--parallel > 1` serializes
//! on the model's compute lock anyway. Real parallelism would need one loaded
//! model copy per worker; not worth the memory until proven needed.

use std::sync::Mutex;

use transcribe_cpp::{Backend, Model, ModelOptions, RunOptions, Session, SessionOptions};

use crate::config::ModelSpec;
use crate::engine::{EngineError, SttEngine};

struct LoadedModel {
    alias: String,
    model: Model,
    session: Mutex<Session>,
}

/// Engine holding one loaded model per configured alias; the first spec is
/// the default model (used when the request names no model or an unknown one).
pub struct TranscribeCppEngine {
    models: Vec<LoadedModel>,
}

impl TranscribeCppEngine {
    pub fn load(
        specs: &[ModelSpec],
        no_gpu: bool,
        threads: Option<usize>,
    ) -> Result<Self, EngineError> {
        if specs.is_empty() {
            return Err(EngineError::Failed(
                "no models configured (use --model)".to_string(),
            ));
        }

        let model_options = ModelOptions {
            backend: if no_gpu { Backend::Cpu } else { Backend::Auto },
            ..ModelOptions::default()
        };
        let session_options = SessionOptions {
            // 0 = library default thread count.
            n_threads: threads.map_or(0, |t| i32::try_from(t).unwrap_or(i32::MAX)),
            ..SessionOptions::default()
        };

        let mut models = Vec::with_capacity(specs.len());
        for spec in specs {
            let model = Model::load_with(&spec.path, &model_options).map_err(|e| {
                EngineError::Failed(format!(
                    "loading model '{}' from {}: {e}",
                    spec.alias,
                    spec.path.display()
                ))
            })?;
            let session = model.session_with(&session_options).map_err(|e| {
                EngineError::Failed(format!("creating session for model '{}': {e}", spec.alias))
            })?;
            tracing::info!(
                alias = %spec.alias,
                arch = %model.arch(),
                backend = %model.backend(),
                "model loaded"
            );
            models.push(LoadedModel {
                alias: spec.alias.clone(),
                model,
                session: Mutex::new(session),
            });
        }
        Ok(TranscribeCppEngine { models })
    }

    /// Requested alias if known, otherwise the default (first) model.
    fn resolve(&self, alias: Option<&str>) -> &LoadedModel {
        alias
            .and_then(|a| self.models.iter().find(|m| m.alias == a))
            .unwrap_or(&self.models[0])
    }
}

impl SttEngine for TranscribeCppEngine {
    fn transcribe(
        &self,
        pcm: &[f32],
        model: Option<&str>,
        language: Option<&str>,
    ) -> Result<String, EngineError> {
        let entry = self.resolve(model);
        let options = RunOptions {
            language: language.map(str::to_string),
            ..RunOptions::default()
        };
        // A poisoned lock means a previous run panicked; the session itself
        // stays usable (the crate recovers its own locks the same way).
        let mut session = entry
            .session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        session
            .run(pcm, &options)
            .map(|transcript| transcript.text)
            .map_err(|e| EngineError::Failed(format!("model '{}': {e}", entry.alias)))
    }

    fn models(&self) -> Vec<String> {
        self.models.iter().map(|m| m.alias.clone()).collect()
    }

    fn backend(&self) -> String {
        // What the library actually bound for the default model, e.g.
        // "cpu", "metal", "cuda" (detects CPU fallback after a GPU request).
        self.models[0].model.backend()
    }
}

#[cfg(test)]
mod tests {
    use super::TranscribeCppEngine;
    use crate::config::ModelSpec;
    use crate::engine::SttEngine;

    #[test]
    fn load_without_specs_fails() {
        let err = TranscribeCppEngine::load(&[], true, None)
            .err()
            .expect("load must fail without specs");
        assert!(err.to_string().contains("no models configured"));
    }

    #[test]
    fn load_with_missing_file_fails() {
        let specs = [ModelSpec {
            alias: "missing".to_string(),
            path: "/nonexistent/model.gguf".into(),
        }];
        let err = TranscribeCppEngine::load(&specs, true, None)
            .err()
            .expect("load must fail for a missing file");
        let msg = err.to_string();
        assert!(msg.contains("missing"), "unexpected error: {msg}");
    }

    /// Real-model smoke test: TS_TEST_MODEL=/path/to/model.gguf cargo test \
    ///   --features engine-transcribe -- --ignored
    #[test]
    #[ignore = "needs TS_TEST_MODEL pointing at a real GGUF model"]
    fn test_real_model_smoke() {
        let path = std::env::var("TS_TEST_MODEL").expect("TS_TEST_MODEL not set");
        let specs = [ModelSpec {
            alias: "test".to_string(),
            path: path.into(),
        }];
        let engine = TranscribeCppEngine::load(&specs, true, None).expect("load model");
        assert_eq!(engine.models(), vec!["test".to_string()]);
        assert!(!engine.backend().is_empty());
        // 1 second of silence at 16 kHz; any text (possibly empty) is fine.
        let pcm = vec![0.0f32; 16000];
        let text = engine
            .transcribe(&pcm, Some("test"), None)
            .expect("transcribe silence");
        // Unknown alias falls back to the default (first) model.
        let text2 = engine
            .transcribe(&pcm, Some("no-such-alias"), None)
            .expect("transcribe with unknown alias");
        assert_eq!(text.is_empty(), text2.is_empty());
    }
}
