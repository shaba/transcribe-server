pub mod fake;

// TODO(server tasks): drop allow(dead_code) once handlers consume the engine.
#[allow(dead_code)]
#[derive(thiserror::Error, Debug)]
pub enum EngineError {
    #[error("unknown model: {0}")]
    UnknownModel(String),
    #[error("transcription failed: {0}")]
    Failed(String),
}

#[allow(dead_code)]
pub trait SttEngine: Send + Sync {
    /// pcm: 16 kHz mono f32 [-1,1]; model: request alias or None (default model)
    fn transcribe(
        &self,
        pcm: &[f32],
        model: Option<&str>,
        language: Option<&str>,
    ) -> Result<String, EngineError>;
    fn models(&self) -> Vec<String>;
    fn backend(&self) -> String; // "fake" | "cpu" | "cuda" | ...
}
