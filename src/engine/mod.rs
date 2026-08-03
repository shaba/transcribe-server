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

/// One timed row of a transcript - a segment or a word. Times are seconds
/// from the start of the PCM buffer the engine was handed, not absolute
/// times in the original recording.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TimedSpan {
    pub start: f32,
    pub end: f32,
    pub text: String,
}

/// The result of one engine call: text plus whatever alignment rows the model
/// produced. Both row vectors may be empty - a model without timestamp
/// support, or a chunk the model returned nothing for.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Transcript {
    pub text: String,
    /// Language the model reports for this audio, when it reports one.
    pub language: Option<String>,
    pub segments: Vec<TimedSpan>,
    pub words: Vec<TimedSpan>,
}

impl Transcript {
    /// Text-only transcript, for models that produce no alignment data.
    pub fn text_only(text: impl Into<String>) -> Self {
        Transcript {
            text: text.into(),
            ..Transcript::default()
        }
    }

    /// Move every timed row `offset` seconds later. Long audio is transcribed
    /// chunk by chunk and every engine times a chunk from that chunk's own
    /// start, so the offset of the chunk inside the recording has to be added
    /// back or every chunk restarts at zero.
    pub fn shift(&mut self, offset: f32) {
        for span in self.segments.iter_mut().chain(self.words.iter_mut()) {
            span.start += offset;
            span.end += offset;
        }
    }
}

pub trait SttEngine: Send + Sync {
    /// pcm: 16 kHz mono f32 [-1,1]; model: request alias or None (default model)
    fn transcribe(
        &self,
        pcm: &[f32],
        model: Option<&str>,
        language: Option<&str>,
    ) -> Result<Transcript, EngineError>;
    fn models(&self) -> Vec<String>;
    fn backend(&self) -> String; // "fake" | "cpu" | "cuda" | ...
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(start: f32, end: f32, text: &str) -> TimedSpan {
        TimedSpan {
            start,
            end,
            text: text.to_string(),
        }
    }

    #[test]
    fn text_only_has_no_rows() {
        let t = Transcript::text_only("hello");
        assert_eq!(t.text, "hello");
        assert!(t.language.is_none());
        assert!(t.segments.is_empty());
        assert!(t.words.is_empty());
    }

    #[test]
    fn shift_moves_segments_and_words() {
        let mut t = Transcript {
            text: "a b".to_string(),
            language: Some("ru".to_string()),
            segments: vec![span(0.0, 2.0, "a b")],
            words: vec![span(0.0, 1.0, "a"), span(1.5, 2.0, "b")],
        };
        t.shift(10.5);
        assert_eq!(t.segments, vec![span(10.5, 12.5, "a b")]);
        assert_eq!(t.words, vec![span(10.5, 11.5, "a"), span(12.0, 12.5, "b")]);
        // Text and language are untouched.
        assert_eq!(t.text, "a b");
        assert_eq!(t.language.as_deref(), Some("ru"));
    }

    #[test]
    fn shift_by_zero_is_identity() {
        let mut t = Transcript {
            text: "x".to_string(),
            language: None,
            segments: vec![span(1.0, 2.0, "x")],
            words: vec![span(1.0, 2.0, "x")],
        };
        let before = t.clone();
        t.shift(0.0);
        assert_eq!(t, before);
    }

    #[test]
    fn shift_of_empty_transcript_is_fine() {
        let mut t = Transcript::text_only("");
        t.shift(3.0);
        assert!(t.segments.is_empty());
        assert!(t.words.is_empty());
    }
}
