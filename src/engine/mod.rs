pub mod cancel;
pub mod fake;
#[cfg(feature = "engine-transcribe")]
pub mod transcribe_cpp;

pub use cancel::CancelFlag;

#[derive(thiserror::Error, Debug)]
pub enum EngineError {
    #[error("unknown model: {0}")]
    UnknownModel(String),
    /// The request asked the model for something it cannot do at all, as
    /// opposed to a knob it merely has no runtime switch for. Caller error,
    /// not a server fault.
    #[error("{0}")]
    Unsupported(String),
    /// The caller went away and the run was aborted on purpose. Not a
    /// failure: nobody is left to read the answer.
    #[error("transcription cancelled")]
    Cancelled,
    #[error("transcription failed: {0}")]
    Failed(String),
}

/// What the model should do with the audio.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Task {
    /// Transcribe speech in its source language.
    #[default]
    Transcribe,
    /// Translate speech into `TranscribeOptions::target_language`.
    Translate,
}

impl Task {
    /// The value the OpenAI `verbose_json` body reports as `task`.
    pub fn as_str(self) -> &'static str {
        match self {
            Task::Transcribe => "transcribe",
            Task::Translate => "translate",
        }
    }
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

/// Everything one engine call needs besides the audio: which model to use and
/// the knobs the HTTP layer lets a caller turn.
///
/// The toggles are tri-state on purpose. `None` means "whatever this model
/// family ships with", which is not the same as `Some(false)`: a family whose
/// published accuracy was measured with punctuation on must keep producing it
/// unless the caller actually asked for plain text.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TranscribeOptions {
    /// Request alias, or None for the default (first) model.
    pub model: Option<String>,
    /// Transcribe the speech or translate it.
    pub task: Task,
    /// Language hint for the audio, or None to let the model decide.
    pub language: Option<String>,
    /// Language to translate into; only meaningful for [`Task::Translate`],
    /// where None leaves the choice to the model (whisper only ever produces
    /// English).
    pub target_language: Option<String>,
    /// Punctuation and capitalization.
    pub pnc: Option<bool>,
    /// Inverse text normalization ("twenty five" -> "25").
    pub itn: Option<bool>,
}

/// What one loaded model is, as far as the HTTP layer needs to know. Every
/// field is what the engine reports about the model actually loaded, not what
/// the configuration asked for.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModelInfo {
    /// Request alias, the id a client passes as `model`.
    pub id: String,
    /// Model architecture, e.g. "gigaam" or "whisper"; empty when unknown.
    pub arch: String,
    /// Language codes the model handles; empty when it does not enumerate any
    /// (a monolingual model usually does not).
    pub languages: Vec<String>,
    /// Whether the model can translate at all.
    pub supports_translate: bool,
    /// Languages it can translate into; empty means "whatever it was trained
    /// to produce" (whisper: English) rather than "none".
    pub translate_target_languages: Vec<String>,
    /// Longest audio one call may hand it, in seconds; None when unbounded or
    /// unreported. f64 because the value is milliseconds divided by 1000 and
    /// goes straight into JSON: an f32 widened on the way out turns 12345 ms
    /// into 12.345000267028809.
    pub max_audio_sec: Option<f64>,
}

impl ModelInfo {
    /// Why this model cannot serve this translation, if it cannot.
    ///
    /// The rule lives here, next to the fields it reads, because two layers
    /// apply it: the HTTP handler as a cheap pre-check before decoding an
    /// upload, and the engine as the authority right before the run. Two
    /// copies of it would eventually disagree, and the subtle half -- that a
    /// model advertising no targets has not claimed it has none -- is exactly
    /// the half someone would get wrong.
    ///
    /// `named` is how the caller wants the model spelled in the message: the
    /// HTTP layer says which alias resolved to it, the engine just names it.
    pub fn translation_refusal(&self, named: &str, target: Option<&str>) -> Option<String> {
        if !self.supports_translate {
            return Some(format!("{named} cannot translate"));
        }
        let target = target?;
        let targets = &self.translate_target_languages;
        // An empty list is an information gap, not a claim of no targets, so
        // the request goes through and the library decides.
        if targets.is_empty() || targets.iter().any(|l| language_eq(l, target)) {
            return None;
        }
        Some(format!(
            "{named} cannot translate into '{target}' (supported: {})",
            targets.join(", ")
        ))
    }
}

impl ModelInfo {
    /// The model's own spelling of `target`, when it lists one that names the
    /// same language. The library takes a short language code and decides what
    /// it accepts; handing it the caller's capitalization and spacing is
    /// asking it to reject something we just accepted.
    pub fn canonical_translate_target<'a>(&'a self, target: &'a str) -> &'a str {
        self.translate_target_languages
            .iter()
            .find(|l| language_eq(l, target))
            .map_or_else(|| target.trim(), String::as_str)
    }
}

/// Compare two language codes the way a caller would expect: `EN` and ` en `
/// name the same language as `en`. Codes are ASCII, so this is not a Unicode
/// case-folding problem.
pub fn language_eq(a: &str, b: &str) -> bool {
    a.trim().eq_ignore_ascii_case(b.trim())
}

pub trait SttEngine: Send + Sync {
    /// pcm: 16 kHz mono f32 [-1,1]. A toggle the loaded model has no runtime
    /// switch for is ignored, not an error: a server holding several model
    /// families would otherwise reject requests for the one family that
    /// cannot honor a server-wide default.
    fn transcribe(
        &self,
        pcm: &[f32],
        options: &TranscribeOptions,
        cancel: &CancelFlag,
    ) -> Result<Transcript, EngineError>;
    /// Every loaded model, in configuration order: the first one is the
    /// default the server falls back to.
    fn models(&self) -> Vec<ModelInfo>;

    /// The model a request naming `alias` would actually run on: that alias
    /// when it is loaded, the default model otherwise. None only when no model
    /// is loaded at all.
    ///
    /// This is the same fallback [`SttEngine::transcribe`] applies, exposed so
    /// the HTTP layer can consult the model it is about to use without
    /// second-guessing which one that is.
    fn resolve_model(&self, alias: Option<&str>) -> Option<ModelInfo> {
        let models = self.models();
        alias
            .and_then(|alias| models.iter().find(|m| m.id == alias).cloned())
            .or_else(|| models.first().cloned())
    }
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
