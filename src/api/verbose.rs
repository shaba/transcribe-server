//! Assembly of the OpenAI `verbose_json` response body.
//!
//! Long audio is transcribed chunk by chunk, so the body is stitched from
//! several transcripts. The times in them are already absolute (relative to
//! the start of the recording, see `exec::transcribe_range`); this module
//! only concatenates, renumbers and rounds.

use std::ops::Range;

use serde_json::{Value, json};

use crate::engine::Transcript;

/// One transcribed chunk: where it sits in the recording (seconds) and what
/// the engine returned for it, already timed relative to the recording.
pub(crate) struct Chunk {
    pub span: Range<f32>,
    pub transcript: Transcript,
}

/// Chunk texts joined exactly like the `json` and `text` formats join them.
pub(crate) fn joined_text(chunks: &[Chunk]) -> String {
    let parts: Vec<String> = chunks
        .iter()
        .map(|c| c.transcript.text.clone())
        .collect::<Vec<_>>();
    super::join_parts(&parts)
}

/// Build the `verbose_json` body: OpenAI's shape, with `words` present only
/// when the model produced word rows.
///
/// `duration` is the length of the decoded audio in seconds;
/// `requested_language` is the language the caller asked for, used only when
/// the model reports none of its own.
pub(crate) fn body(chunks: &[Chunk], duration: f32, requested_language: Option<&str>) -> Value {
    let language = chunks
        .iter()
        .find_map(|c| c.transcript.language.clone())
        .or_else(|| requested_language.map(str::to_string))
        .unwrap_or_else(|| "unknown".to_string());

    let mut segments: Vec<Value> = Vec::new();
    for chunk in chunks {
        if chunk.transcript.segments.is_empty() {
            // A model that reports no segment rows still gets one row per
            // chunk: a caller building a timeline needs some anchor, and the
            // chunk bounds are a true (if coarse) one.
            if !chunk.transcript.text.is_empty() {
                let id = segments.len();
                segments.push(segment(
                    id,
                    chunk.span.start,
                    chunk.span.end,
                    &chunk.transcript.text,
                ));
            }
            continue;
        }
        for row in &chunk.transcript.segments {
            if row.text.is_empty() {
                continue;
            }
            let id = segments.len();
            segments.push(segment(id, row.start, row.end, &row.text));
        }
    }

    let words: Vec<Value> = chunks
        .iter()
        .flat_map(|c| c.transcript.words.iter())
        .filter(|w| !w.text.is_empty())
        .map(|w| json!({"word": w.text, "start": sec(w.start), "end": sec(w.end)}))
        .collect();

    let mut body = json!({
        "task": "transcribe",
        "language": language,
        "duration": sec(duration),
        "text": joined_text(chunks),
        "segments": segments,
    });
    if !words.is_empty() {
        body["words"] = Value::Array(words);
    }
    body
}

fn segment(id: usize, start: f32, end: f32, text: &str) -> Value {
    json!({"id": id, "start": sec(start), "end": sec(end), "text": text})
}

/// Seconds rounded to milliseconds. f32 accumulation leaves artefacts like
/// 1.2000000476837158 that would otherwise end up verbatim in the JSON, and
/// no engine here resolves finer than a millisecond anyway.
fn sec(seconds: f32) -> f64 {
    (f64::from(seconds) * 1000.0).round() / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::TimedSpan;

    fn span(start: f32, end: f32, text: &str) -> TimedSpan {
        TimedSpan {
            start,
            end,
            text: text.to_string(),
        }
    }

    /// Two chunks, the second one already shifted into absolute time.
    fn two_chunks() -> Vec<Chunk> {
        vec![
            Chunk {
                span: 0.0..2.0,
                transcript: Transcript {
                    text: "one two".to_string(),
                    language: Some("ru".to_string()),
                    segments: vec![span(0.0, 2.0, "one two")],
                    words: vec![span(0.0, 1.0, "one"), span(1.0, 2.0, "two")],
                },
            },
            Chunk {
                span: 2.0..3.5,
                transcript: Transcript {
                    text: "three".to_string(),
                    language: Some("ru".to_string()),
                    segments: vec![span(2.0, 3.5, "three")],
                    words: vec![span(2.0, 3.5, "three")],
                },
            },
        ]
    }

    #[test]
    fn body_has_the_openai_verbose_shape() {
        let body = body(&two_chunks(), 3.5, None);
        assert_eq!(body["task"], "transcribe");
        assert_eq!(body["language"], "ru");
        assert_eq!(body["duration"], 3.5);
        assert_eq!(body["text"], "one two three");
        let segments = body["segments"].as_array().expect("segments array");
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0]["id"], 0);
        assert_eq!(segments[0]["start"], 0.0);
        assert_eq!(segments[0]["end"], 2.0);
        assert_eq!(segments[0]["text"], "one two");
        assert_eq!(segments[1]["id"], 1);
        assert_eq!(segments[1]["start"], 2.0);
        assert_eq!(segments[1]["end"], 3.5);
    }

    #[test]
    fn words_are_concatenated_in_order() {
        let body = body(&two_chunks(), 3.5, None);
        let words = body["words"].as_array().expect("words array");
        let seen: Vec<(&str, f64)> = words
            .iter()
            .map(|w| {
                (
                    w["word"].as_str().expect("word"),
                    w["start"].as_f64().expect("start"),
                )
            })
            .collect();
        assert_eq!(seen, [("one", 0.0), ("two", 1.0), ("three", 2.0)]);
    }

    #[test]
    fn words_are_omitted_when_the_model_produced_none() {
        let chunks = vec![Chunk {
            span: 0.0..1.0,
            transcript: Transcript {
                text: "only text".to_string(),
                segments: vec![span(0.0, 1.0, "only text")],
                ..Transcript::default()
            },
        }];
        let body = body(&chunks, 1.0, None);
        assert!(body.get("words").is_none(), "{body}");
    }

    #[test]
    fn chunk_without_segment_rows_falls_back_to_the_chunk_span() {
        let chunks = vec![
            Chunk {
                span: 0.0..25.0,
                transcript: Transcript::text_only("first"),
            },
            Chunk {
                span: 25.0..40.0,
                transcript: Transcript::text_only("second"),
            },
        ];
        let body = body(&chunks, 40.0, None);
        let segments = body["segments"].as_array().expect("segments array");
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0]["start"], 0.0);
        assert_eq!(segments[0]["end"], 25.0);
        assert_eq!(segments[0]["text"], "first");
        assert_eq!(segments[1]["id"], 1);
        assert_eq!(segments[1]["start"], 25.0);
        assert_eq!(segments[1]["end"], 40.0);
    }

    #[test]
    fn silent_chunks_produce_no_segments_and_ids_stay_dense() {
        let chunks = vec![
            Chunk {
                span: 0.0..1.0,
                transcript: Transcript::text_only(""),
            },
            Chunk {
                span: 1.0..2.0,
                transcript: Transcript {
                    text: "spoken".to_string(),
                    segments: vec![span(1.0, 2.0, "spoken")],
                    ..Transcript::default()
                },
            },
        ];
        let body = body(&chunks, 2.0, None);
        let segments = body["segments"].as_array().expect("segments array");
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0]["id"], 0);
        assert_eq!(segments[0]["text"], "spoken");
        assert_eq!(body["text"], "spoken");
    }

    #[test]
    fn language_falls_back_to_the_requested_one_then_to_unknown() {
        let chunks = vec![Chunk {
            span: 0.0..1.0,
            transcript: Transcript::text_only("x"),
        }];
        assert_eq!(body(&chunks, 1.0, Some("ru"))["language"], "ru");
        assert_eq!(body(&chunks, 1.0, None)["language"], "unknown");
        // A language the model detected wins over the requested one.
        let detected = vec![Chunk {
            span: 0.0..1.0,
            transcript: Transcript {
                text: "x".to_string(),
                language: Some("en".to_string()),
                ..Transcript::default()
            },
        }];
        assert_eq!(body(&detected, 1.0, Some("ru"))["language"], "en");
    }

    #[test]
    fn times_are_rounded_to_milliseconds() {
        // 0.1 + 0.2 in f32 is 0.30000001192092896 as f64.
        let chunks = vec![Chunk {
            span: 0.0..1.0,
            transcript: Transcript {
                text: "x".to_string(),
                segments: vec![span(0.1 + 0.2, 1.0 / 3.0, "x")],
                ..Transcript::default()
            },
        }];
        let body = body(&chunks, 1.0 / 3.0, None);
        assert_eq!(body["segments"][0]["start"], 0.3);
        assert_eq!(body["segments"][0]["end"], 0.333);
        assert_eq!(body["duration"], 0.333);
    }

    #[test]
    fn no_chunks_yields_an_empty_but_well_formed_body() {
        let body = body(&[], 0.0, None);
        assert_eq!(body["text"], "");
        assert_eq!(body["duration"], 0.0);
        assert_eq!(body["segments"].as_array().expect("array").len(), 0);
        assert!(body.get("words").is_none());
    }

    #[test]
    fn joined_text_matches_the_plain_formats() {
        assert_eq!(joined_text(&two_chunks()), "one two three");
        assert_eq!(joined_text(&[]), "");
    }
}
