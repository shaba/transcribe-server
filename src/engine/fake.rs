use super::{
    CancelFlag, EngineError, ModelInfo, SttEngine, Task, TimedSpan, TranscribeOptions, Transcript,
};
use crate::audio::samples_to_sec;

pub struct FakeEngine;

impl SttEngine for FakeEngine {
    /// Echoes the model alias and the sample count, and produces alignment
    /// rows spanning exactly the PCM it was given, so the timestamp plumbing
    /// (including the chunk offsets applied by callers) is exercised without
    /// a real model. The reported language is the requested one: the fake
    /// "detects" whatever it was told.
    ///
    /// A toggle the caller set is echoed too (`...:pnc=off`), as is a
    /// translation request (`...:translate=en`), which is what lets the HTTP
    /// tests assert that a request field reached the engine. Anything left at
    /// its default adds nothing, so the plain text shape is unchanged.
    fn transcribe(
        &self,
        pcm: &[f32],
        options: &TranscribeOptions,
        cancel: &CancelFlag,
    ) -> Result<Transcript, EngineError> {
        if cancel.is_cancelled() {
            return Err(EngineError::Cancelled);
        }
        let mut text = format!(
            "fake:{}:{}",
            options.model.as_deref().unwrap_or("default"),
            pcm.len()
        );
        if options.task == Task::Translate {
            text.push_str(&format!(
                ":translate={}",
                options.target_language.as_deref().unwrap_or("auto")
            ));
        }
        if let Some(pnc) = options.pnc {
            text.push_str(&format!(":pnc={}", on_off(pnc)));
        }
        if let Some(itn) = options.itn {
            text.push_str(&format!(":itn={}", on_off(itn)));
        }
        let duration = samples_to_sec(pcm.len());
        // The fake text carries no spaces, so its colon-separated parts stand
        // in for words: enough rows to catch a word list that is mis-ordered
        // or shifted by the wrong offset.
        let parts: Vec<&str> = text.split(':').collect();
        let n = parts.len() as f32;
        let words = parts
            .iter()
            .enumerate()
            .map(|(i, part)| TimedSpan {
                start: duration * i as f32 / n,
                end: duration * (i + 1) as f32 / n,
                text: (*part).to_string(),
            })
            .collect();
        Ok(Transcript {
            segments: vec![TimedSpan {
                start: 0.0,
                end: duration,
                text: text.clone(),
            }],
            words,
            language: options.language.clone(),
            text,
        })
    }

    fn models(&self) -> Vec<ModelInfo> {
        vec![ModelInfo {
            id: "fake-model".to_string(),
            arch: "fake".to_string(),
            languages: vec!["en".to_string(), "ru".to_string()],
            supports_translate: true,
            translate_target_languages: vec!["en".to_string()],
            max_audio_sec: None,
        }]
    }

    fn backend(&self) -> String {
        "fake".to_string()
    }
}

fn on_off(value: bool) -> &'static str {
    if value { "on" } else { "off" }
}

#[cfg(test)]
mod tests {
    use super::FakeEngine;
    use crate::engine::{CancelFlag, SttEngine, TranscribeOptions};

    fn options(model: Option<&str>, language: Option<&str>) -> TranscribeOptions {
        TranscribeOptions {
            model: model.map(str::to_string),
            language: language.map(str::to_string),
            ..TranscribeOptions::default()
        }
    }

    #[test]
    fn transcribe_without_model_uses_default() {
        let engine = FakeEngine;
        let pcm = [0.0f32, 0.1, -0.1];
        let result = engine
            .transcribe(&pcm, &TranscribeOptions::default(), &CancelFlag::new())
            .expect("transcribe ok");
        assert_eq!(result.text, "fake:default:3");
        assert!(result.language.is_none());
    }

    #[test]
    fn set_toggles_are_echoed_and_unset_ones_are_not() {
        let engine = FakeEngine;
        let both = TranscribeOptions {
            pnc: Some(false),
            itn: Some(true),
            ..options(Some("ru"), None)
        };
        let result = engine
            .transcribe(&[], &both, &CancelFlag::new())
            .expect("transcribe ok");
        assert_eq!(result.text, "fake:ru:0:pnc=off:itn=on");
        let one = TranscribeOptions {
            pnc: Some(true),
            ..TranscribeOptions::default()
        };
        let result = engine
            .transcribe(&[], &one, &CancelFlag::new())
            .expect("transcribe ok");
        assert_eq!(result.text, "fake:default:0:pnc=on");
    }

    #[test]
    fn transcribe_with_model_and_empty_pcm() {
        let engine = FakeEngine;
        let result = engine
            .transcribe(&[], &options(Some("ru"), Some("ru")), &CancelFlag::new())
            .expect("transcribe ok");
        assert_eq!(result.text, "fake:ru:0");
        assert_eq!(result.language.as_deref(), Some("ru"));
        // Zero-length audio still yields rows, all of zero length.
        assert!(
            result
                .segments
                .iter()
                .all(|s| s.start == 0.0 && s.end == 0.0)
        );
        assert!(result.words.iter().all(|w| w.start == 0.0 && w.end == 0.0));
    }

    #[test]
    fn segment_spans_the_whole_buffer() {
        let engine = FakeEngine;
        let pcm = vec![0.0f32; 32_000]; // 2 s at 16 kHz
        let result = engine
            .transcribe(&pcm, &TranscribeOptions::default(), &CancelFlag::new())
            .expect("transcribe ok");
        assert_eq!(result.segments.len(), 1);
        assert_eq!(result.segments[0].start, 0.0);
        assert_eq!(result.segments[0].end, 2.0);
        assert_eq!(result.segments[0].text, result.text);
    }

    #[test]
    fn words_are_ordered_and_cover_the_buffer() {
        let engine = FakeEngine;
        let pcm = vec![0.0f32; 16_000]; // 1 s at 16 kHz
        let result = engine
            .transcribe(&pcm, &TranscribeOptions::default(), &CancelFlag::new())
            .expect("transcribe ok");
        assert_eq!(result.words.len(), 3, "{:?}", result.words);
        assert_eq!(result.words[0].start, 0.0);
        assert_eq!(result.words.last().expect("last word").end, 1.0);
        for pair in result.words.windows(2) {
            assert!(pair[0].start < pair[0].end, "empty word: {:?}", pair[0]);
            assert_eq!(pair[0].end, pair[1].start, "words must be contiguous");
        }
    }

    #[test]
    fn models_lists_fake_model() {
        let engine = FakeEngine;
        let models = engine.models();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "fake-model");
        assert!(models[0].supports_translate);
    }

    #[test]
    fn backend_is_fake() {
        let engine = FakeEngine;
        assert_eq!(engine.backend(), "fake");
    }

    #[test]
    fn a_cancelled_call_does_no_work() {
        let cancel = CancelFlag::new();
        cancel.cancel();
        let err = FakeEngine
            .transcribe(&[0.0; 16], &TranscribeOptions::default(), &cancel)
            .expect_err("a cancelled call must not transcribe");
        assert!(matches!(err, crate::engine::EngineError::Cancelled));
    }

    #[test]
    fn engine_is_object_safe() {
        let engine: std::sync::Arc<dyn SttEngine> = std::sync::Arc::new(FakeEngine);
        assert_eq!(engine.backend(), "fake");
    }
}
