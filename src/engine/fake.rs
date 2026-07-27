use super::{EngineError, SttEngine};

// TODO(server tasks): drop allow(dead_code) once handlers consume the engine.
#[allow(dead_code)]
pub struct FakeEngine;

impl SttEngine for FakeEngine {
    fn transcribe(
        &self,
        pcm: &[f32],
        model: Option<&str>,
        _language: Option<&str>,
    ) -> Result<String, EngineError> {
        Ok(format!("fake:{}:{}", model.unwrap_or("default"), pcm.len()))
    }

    fn models(&self) -> Vec<String> {
        vec!["fake-model".to_string()]
    }

    fn backend(&self) -> String {
        "fake".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::FakeEngine;
    use crate::engine::SttEngine;

    #[test]
    fn transcribe_without_model_uses_default() {
        let engine = FakeEngine;
        let pcm = [0.0f32, 0.1, -0.1];
        let text = engine.transcribe(&pcm, None, None).expect("transcribe ok");
        assert_eq!(text, "fake:default:3");
    }

    #[test]
    fn transcribe_with_model_and_empty_pcm() {
        let engine = FakeEngine;
        let text = engine
            .transcribe(&[], Some("ru"), Some("ru"))
            .expect("transcribe ok");
        assert_eq!(text, "fake:ru:0");
    }

    #[test]
    fn models_lists_fake_model() {
        let engine = FakeEngine;
        assert_eq!(engine.models(), vec!["fake-model".to_string()]);
    }

    #[test]
    fn backend_is_fake() {
        let engine = FakeEngine;
        assert_eq!(engine.backend(), "fake");
    }

    #[test]
    fn engine_is_object_safe() {
        let engine: std::sync::Arc<dyn SttEngine> = std::sync::Arc::new(FakeEngine);
        assert_eq!(engine.backend(), "fake");
    }
}
