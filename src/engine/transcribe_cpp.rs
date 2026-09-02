//! Real STT engine backed by the transcribe-cpp crate (transcribe.cpp / ggml).
//!
//! Concurrency: the 0.x C library allows at most one in-flight compute per
//! model, and the crate's `Session` is Send but not Sync (`run` takes
//! `&mut self`). We therefore keep one long-lived `Session` per model behind a
//! `Mutex`: session init allocates decoder/KV state, so creating one per call
//! would be wasted work, and with a single model `--parallel > 1` serializes
//! on the model's compute lock anyway. Real parallelism would need one loaded
//! model copy per worker; not worth the memory until proven needed.
//!
//! Backend modules: when libtranscribe is a dynamic-backend build (the ggml
//! compute backends are loadable modules rather than compiled in — how the
//! distro packages ship it), the host MUST register them before the first
//! model load, or every load fails with `TRANSCRIBE_ERR_BACKEND` and the
//! library logs "failed to initialize CPU backend". `init_backends_default()`
//! is the portable way to do that: it is a no-op returning `Ok(())` in a
//! compiled-in build, and in a dynamic build it loads the modules from the
//! directory the build pinned (or, failing that, from the directory holding
//! libtranscribe itself). `TranscribeCppEngine::load` does that registration.
//!
//! Logging: `init_logging` redirects the library's own stderr sink into
//! `tracing`; `main` calls it once at startup.

use std::sync::Mutex;

use transcribe_cpp::Task as CppTask;
use transcribe_cpp::{
    Backend, Feature, Itn, Model, ModelOptions, Pnc, RunOptions, Session, SessionOptions,
    init_backends_default,
};

use crate::config::ModelSpec;
use crate::engine::{
    EngineError, ModelInfo, SttEngine, Task, TimedSpan, TranscribeOptions, Transcript,
};

struct LoadedModel {
    alias: String,
    model: Model,
    session: Mutex<Session>,
    /// Read once at load: the library's capabilities are immutable for a
    /// loaded model, and building them costs a String per supported language
    /// (a whisper model lists a hundred). /health, every transcription and
    /// every stream ask for this, so it must not be an FFI read plus an
    /// allocation storm each time.
    info: ModelInfo,
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

        Self::init_backends()?;

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
            let info = model_info(&spec.alias, &model);
            models.push(LoadedModel {
                alias: spec.alias.clone(),
                model,
                session: Mutex::new(session),
                info,
            });
        }
        Ok(TranscribeCppEngine { models })
    }

    /// Register the ggml backend modules, once per process, before any model
    /// load. Idempotent in the library, so calling it again (a second engine,
    /// a test) is harmless. The error is reported with the device count so a
    /// misconfigured module directory is obvious from the log line alone.
    fn init_backends() -> Result<(), EngineError> {
        init_backends_default().map_err(|e| {
            EngineError::Failed(format!(
                "registering ggml backend modules: {e} ({} compute device(s) available); \
                 a dynamic-backend libtranscribe needs its backend modules installed \
                 where the build expects them",
                transcribe_cpp::device_count()
            ))
        })
    }

    /// Requested alias if known, otherwise the default (first) model. Must
    /// stay the same rule as `SttEngine::resolve_model`, which is what the
    /// HTTP layer consults about the model a request will run on.
    fn resolve(&self, alias: Option<&str>) -> &LoadedModel {
        alias
            .and_then(|a| self.models.iter().find(|m| m.alias == a))
            .unwrap_or(&self.models[0])
    }
}

/// Map a crate transcript onto the engine-level one: milliseconds relative to
/// the audio handed to `run` become seconds relative to the same audio.
///
/// Only the segment and word rows are carried over. The crate also exposes
/// token rows (id, per-token confidence `p`, times, owning segment/word), but
/// the OpenAI verbose_json shape this feeds has nowhere to put them, and `p`
/// is NaN for families that produce no confidence.
fn convert(transcript: transcribe_cpp::Transcript) -> Transcript {
    fn spans(rows: impl IntoIterator<Item = (i64, i64, String)>) -> Vec<TimedSpan> {
        rows.into_iter()
            .map(|(t0_ms, t1_ms, text)| TimedSpan {
                start: t0_ms as f32 / 1000.0,
                end: t1_ms as f32 / 1000.0,
                text,
            })
            .collect()
    }
    Transcript {
        text: transcript.text,
        language: transcript.language,
        segments: spans(
            transcript
                .segments
                .into_iter()
                .map(|s| (s.t0_ms, s.t1_ms, s.text)),
        ),
        words: spans(
            transcript
                .words
                .into_iter()
                .map(|w| (w.t0_ms, w.t1_ms, w.text)),
        ),
    }
}

/// Route libtranscribe and ggml diagnostics into `tracing` instead of the
/// library's own stderr sink, once per process. The crate logs through the
/// `log` facade and `tracing-subscriber` installs the log-to-tracing bridge,
/// so the lines arrive under the `transcribe_cpp` target and `RUST_LOG` can
/// filter them (`RUST_LOG=transcribe_cpp=warn` quiets the model-load chatter).
///
/// Called from `main` next to the subscriber setup rather than from
/// [`TranscribeCppEngine::load`]: displacing the library's stderr sink is only
/// an improvement where a subscriber exists to receive the messages, and in a
/// process without one -- a test binary, say -- `log` drops them entirely,
/// which would hide exactly the diagnostics a failing model load needs.
///
/// `transcribe_log_set` is a process-global the library documents as
/// once-at-startup, hence the `Once`.
pub fn init_logging() {
    static LOGGING: std::sync::Once = std::sync::Once::new();
    LOGGING.call_once(transcribe_cpp::init_logging);
}

/// What the library reports about a freshly loaded model.
fn model_info(alias: &str, model: &Model) -> ModelInfo {
    let caps = model.capabilities();
    ModelInfo {
        id: alias.to_string(),
        arch: model.arch(),
        languages: caps.languages,
        supports_translate: caps.supports_translate,
        translate_target_languages: caps.translate_target_languages,
        // The library reports 0 for "no limit of its own".
        max_audio_sec: (caps.max_audio_ms > 0).then(|| caps.max_audio_ms as f64 / 1000.0),
    }
}

/// Resolve a tri-state request toggle against what the loaded model can
/// actually switch at run time, so a family without the switch is asked for
/// its own default.
///
/// The library tolerates the explicit value either way -- it transcribes with
/// the family default and logs one WARN per run telling the caller to
/// pre-check `transcribe_model_supports`. This is that pre-check: it keeps a
/// server-wide `--pnc` from filling the log with one warning per request for
/// every model that has no such switch.
fn resolve_pnc(requested: Option<bool>, supported: bool) -> Pnc {
    match (requested, supported) {
        (Some(true), true) => Pnc::On,
        (Some(false), true) => Pnc::Off,
        _ => Pnc::Default,
    }
}

/// [`resolve_pnc`] for the ITN switch; the two crate enums are distinct types.
fn resolve_itn(requested: Option<bool>, supported: bool) -> Itn {
    match (requested, supported) {
        (Some(true), true) => Itn::On,
        (Some(false), true) => Itn::Off,
        _ => Itn::Default,
    }
}

impl SttEngine for TranscribeCppEngine {
    fn transcribe(
        &self,
        pcm: &[f32],
        request: &TranscribeOptions,
    ) -> Result<Transcript, EngineError> {
        let entry = self.resolve(request.model.as_deref());
        // Probed per call rather than cached at load: the probe is a plain
        // struct read behind the FFI, invisible next to the inference it
        // guards.
        let pnc = resolve_pnc(request.pnc, entry.model.supports(Feature::Pnc));
        let itn = resolve_itn(request.itn, entry.model.supports(Feature::Itn));
        let task = match request.task {
            Task::Transcribe => CppTask::Transcribe,
            Task::Translate => {
                // Asked of the cached capabilities, not of the library: the
                // answer cannot change for a loaded model, and rebuilding it
                // allocates a String per supported language.
                if let Some(refusal) = entry.info.translation_refusal(
                    &format!("model '{}'", entry.alias),
                    request.target_language.as_deref(),
                ) {
                    return Err(EngineError::Unsupported(refusal));
                }
                CppTask::Translate
            }
        };
        // RunOptions::default() asks for TimestampKind::Auto, i.e. the richest
        // granularity the family supports (token-level for GigaAM, segment for
        // whisper). Which rows actually come back is family-dependent, so the
        // mapping below copies whatever is populated and never assumes.
        let options = RunOptions {
            task,
            language: request.language.clone(),
            // Only meaningful for a translation; the library documents its
            // meaning on a transcribe run as undefined, so it never travels
            // with one whatever the caller put in the options.
            target_language: match request.task {
                Task::Translate => request.target_language.clone(),
                Task::Transcribe => None,
            },
            pnc,
            itn,
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
            .map(convert)
            .map_err(|e| EngineError::Failed(format!("model '{}': {e}", entry.alias)))
    }

    fn models(&self) -> Vec<ModelInfo> {
        self.models.iter().map(|entry| entry.info.clone()).collect()
    }

    fn resolve_model(&self, alias: Option<&str>) -> Option<ModelInfo> {
        // Overridden so the common path clones one entry instead of the whole
        // list, and so the fallback stays literally the one `resolve` applies.
        Some(self.resolve(alias).info.clone())
    }

    fn backend(&self) -> String {
        // What the library actually bound for the default model, e.g.
        // "cpu", "metal", "cuda" (detects CPU fallback after a GPU request).
        self.models[0].model.backend()
    }
}

#[cfg(test)]
mod tests {
    use super::{TranscribeCppEngine, convert, resolve_itn, resolve_pnc};
    use crate::config::ModelSpec;
    use crate::engine::{SttEngine, TranscribeOptions};
    use transcribe_cpp::{Itn, Pnc};

    #[test]
    fn toggles_apply_only_to_models_that_switch_them() {
        assert_eq!(resolve_pnc(Some(true), true), Pnc::On);
        assert_eq!(resolve_pnc(Some(false), true), Pnc::Off);
        assert_eq!(resolve_itn(Some(true), true), Itn::On);
        assert_eq!(resolve_itn(Some(false), true), Itn::Off);
        // Unset stays the family default whatever the model supports, and an
        // explicit value on a family without the switch does too.
        for supported in [true, false] {
            assert_eq!(resolve_pnc(None, supported), Pnc::Default);
            assert_eq!(resolve_itn(None, supported), Itn::Default);
        }
        for requested in [Some(true), Some(false)] {
            assert_eq!(resolve_pnc(requested, false), Pnc::Default);
            assert_eq!(resolve_itn(requested, false), Itn::Default);
        }
    }

    #[test]
    fn convert_maps_milliseconds_to_seconds() {
        let crate_transcript = transcribe_cpp::Transcript {
            text: "hello world".to_string(),
            language: Some("en".to_string()),
            segments: vec![transcribe_cpp::Segment {
                t0_ms: 40,
                t1_ms: 1_240,
                text: "hello world".to_string(),
                ..Default::default()
            }],
            words: vec![
                transcribe_cpp::Word {
                    t0_ms: 40,
                    t1_ms: 520,
                    text: "hello".to_string(),
                    ..Default::default()
                },
                transcribe_cpp::Word {
                    t0_ms: 560,
                    t1_ms: 1_240,
                    text: "world".to_string(),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let result = convert(crate_transcript);
        assert_eq!(result.text, "hello world");
        assert_eq!(result.language.as_deref(), Some("en"));
        assert_eq!(result.segments.len(), 1);
        assert_eq!(result.segments[0].start, 0.04);
        assert_eq!(result.segments[0].end, 1.24);
        assert_eq!(result.segments[0].text, "hello world");
        let words: Vec<(f32, f32, &str)> = result
            .words
            .iter()
            .map(|w| (w.start, w.end, w.text.as_str()))
            .collect();
        assert_eq!(
            words,
            [(0.04, 0.52, "hello"), (0.56, 1.24, "world")],
            "word rows must keep order and text"
        );
    }

    #[test]
    fn convert_without_timestamp_rows_yields_text_only() {
        let result = convert(transcribe_cpp::Transcript {
            text: "no rows".to_string(),
            ..Default::default()
        });
        assert_eq!(result.text, "no rows");
        assert!(result.segments.is_empty());
        assert!(result.words.is_empty());
        assert!(result.language.is_none());
    }

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

    /// A dynamic-backend libtranscribe registers no compute device until the
    /// host asks for it, and every model load then fails with a bare "backend
    /// error (status 8)". `load` must do that registration itself: after it
    /// runs, the process has at least one device. Repeated calls stay fine
    /// (the library call is idempotent), which is what lets `load` run it
    /// unconditionally rather than behind a `Once` of its own.
    #[test]
    fn load_registers_backend_modules() {
        // Any load attempt reaches init_backends() — a missing model file
        // fails later, at model load, not before.
        let specs = [ModelSpec {
            alias: "missing".to_string(),
            path: "/nonexistent/model.gguf".into(),
        }];
        let _ = TranscribeCppEngine::load(&specs, true, None);
        assert!(
            transcribe_cpp::device_count() > 0,
            "no compute device registered after load(); backend modules were not loaded"
        );
        TranscribeCppEngine::init_backends().expect("init_backends must be idempotent");
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
        let models = engine.models();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "test");
        assert!(!models[0].arch.is_empty());
        assert!(!engine.backend().is_empty());
        // 1 second of silence at 16 kHz; any text (possibly empty) is fine.
        let pcm = vec![0.0f32; 16000];
        let options = |alias: &str| TranscribeOptions {
            model: Some(alias.to_string()),
            ..TranscribeOptions::default()
        };
        let result = engine
            .transcribe(&pcm, &options("test"))
            .expect("transcribe silence");
        // Unknown alias falls back to the default (first) model.
        let result2 = engine
            .transcribe(&pcm, &options("no-such-alias"))
            .expect("transcribe with unknown alias");
        assert_eq!(result.text.is_empty(), result2.text.is_empty());
        // Timestamp rows, when the family produces them, stay inside the audio.
        for span in result.segments.iter().chain(result.words.iter()) {
            assert!(
                span.start >= 0.0 && span.end <= 1.1 && span.start <= span.end,
                "row outside the 1 s buffer: {span:?}"
            );
        }
    }
}
