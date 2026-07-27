pub mod fake;
#[cfg(feature = "engine-transcribe")]
pub mod transcribe_cpp;

#[derive(thiserror::Error, Debug)]
pub enum EngineError {
    #[error("unknown model: {0}")]
    UnknownModel(String),
    #[error("transcription failed: {0}")]
    Failed(String),
}

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
