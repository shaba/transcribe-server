//! Integration tests for the HTTP API, running against a real server
//! bound to an ephemeral port with FakeEngine.

mod common;

use std::sync::Arc;

use clap::Parser;
use common::{auth_keys, spawn_app_with_cfg, spawn_fake_app};
use transcribe_server::config::Config;
use transcribe_server::engine::fake::FakeEngine;

#[tokio::test]
async fn health_reports_ok_backend_and_models() {
    let base = spawn_fake_app(auth_keys(&[])).await;
    let resp = reqwest::get(format!("{base}/health")).await.expect("GET");
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.expect("JSON body");
    assert_eq!(json["status"], "ok");
    assert_eq!(json["backend"], "fake");
    assert_eq!(json["models"], serde_json::json!(["fake-model"]));
}

#[tokio::test]
async fn models_returns_openai_style_list() {
    let base = spawn_fake_app(auth_keys(&[])).await;
    let resp = reqwest::get(format!("{base}/v1/models"))
        .await
        .expect("GET");
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.expect("JSON body");
    assert_eq!(json["object"], "list");
    assert_eq!(json["data"][0]["id"], "fake-model");
    assert_eq!(json["data"][0]["object"], "model");
    assert_eq!(json["data"][0]["owned_by"], "transcribe-server");
    assert_eq!(json["data"].as_array().expect("data array").len(), 1);
}

#[tokio::test]
async fn models_without_key_is_401_when_keys_configured() {
    let base = spawn_fake_app(auth_keys(&["secret"])).await;
    let resp = reqwest::get(format!("{base}/v1/models"))
        .await
        .expect("GET");
    assert_eq!(resp.status(), 401);
    let json: serde_json::Value = resp.json().await.expect("JSON body");
    assert_eq!(json["error"]["type"], "authentication_error");
}

#[tokio::test]
async fn models_with_valid_key_is_200() {
    let base = spawn_fake_app(auth_keys(&["secret"])).await;
    let resp = reqwest::Client::new()
        .get(format!("{base}/v1/models"))
        .header("authorization", "Bearer secret")
        .send()
        .await
        .expect("GET");
    assert_eq!(resp.status(), 200);
}

#[tokio::test]
async fn health_stays_open_when_keys_configured() {
    let base = spawn_fake_app(auth_keys(&["secret"])).await;
    let resp = reqwest::get(format!("{base}/health")).await.expect("GET");
    assert_eq!(resp.status(), 200);
}

/// In-memory 16 kHz mono i16 WAV with a 440 Hz tone (WAV fastpath input).
fn wav_16k_mono(seconds: f32) -> Vec<u8> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cursor = std::io::Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec).expect("wav writer");
        for k in 0..(seconds * 16_000.0) as usize {
            let v = (2.0 * std::f64::consts::PI * 440.0 * k as f64 / 16_000.0).sin() * 0.5;
            writer.write_sample((v * 32767.0) as i16).expect("sample");
        }
        writer.finalize().expect("finalize wav");
    }
    cursor.into_inner()
}

fn transcription_form(file: Vec<u8>, file_name: &str) -> reqwest::multipart::Form {
    reqwest::multipart::Form::new().part(
        "file",
        reqwest::multipart::Part::bytes(file).file_name(file_name.to_string()),
    )
}

async fn post_transcription(
    base: &str,
    form: reqwest::multipart::Form,
    bearer: Option<&str>,
) -> reqwest::Response {
    let mut req = reqwest::Client::new().post(format!("{base}/v1/audio/transcriptions"));
    if let Some(key) = bearer {
        req = req.header("authorization", format!("Bearer {key}"));
    }
    req.multipart(form).send().await.expect("POST")
}

/// The pnc/itn request fields are a documented extension over the OpenAI
/// shape; FakeEngine echoes whatever reached it, so this asserts the whole
/// path from multipart field to engine options.
#[tokio::test]
async fn transcriptions_pnc_and_itn_fields_reach_the_engine() {
    let base = spawn_fake_app(auth_keys(&[])).await;
    let form = transcription_form(wav_16k_mono(0.1), "a.wav")
        .text("pnc", "off")
        .text("itn", "true");
    let resp = post_transcription(&base, form, None).await;
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.expect("JSON body");
    let text = json["text"].as_str().expect("text field");
    assert!(text.contains(":pnc=off"), "unexpected text: {text}");
    assert!(text.contains(":itn=on"), "unexpected text: {text}");
}

/// Unset request fields fall back to the server-wide flags, and a request
/// field overrides them.
#[tokio::test]
async fn transcriptions_pnc_falls_back_to_the_server_default() {
    let cfg = Arc::new(
        Config::try_parse_from(["ts", "--pnc", "on", "--itn", "off"]).expect("parse config"),
    );
    let base = spawn_app_with_cfg(Arc::new(FakeEngine), auth_keys(&[]), cfg).await;

    let resp =
        post_transcription(&base, transcription_form(wav_16k_mono(0.1), "a.wav"), None).await;
    let json: serde_json::Value = resp.json().await.expect("JSON body");
    let text = json["text"].as_str().expect("text field").to_string();
    assert!(text.contains(":pnc=on"), "unexpected text: {text}");
    assert!(text.contains(":itn=off"), "unexpected text: {text}");

    let form = transcription_form(wav_16k_mono(0.1), "a.wav").text("pnc", "off");
    let resp = post_transcription(&base, form, None).await;
    let json: serde_json::Value = resp.json().await.expect("JSON body");
    let text = json["text"].as_str().expect("text field").to_string();
    assert!(text.contains(":pnc=off"), "unexpected text: {text}");
}

/// Field order is the client's, so a bad toggle can arrive before the audio,
/// and validation must not depend on which came first. (What this pins is the
/// ordering-independent answer; the failure mode it guards against -- the
/// handler returning while the peer is still streaming a large upload, which
/// costs the client the error JSON and gets it a reset connection instead --
/// needs a body far larger than a test wants to push through loopback.)
#[tokio::test]
async fn transcriptions_bad_toggle_before_the_file_still_answers_json() {
    let base = spawn_fake_app(auth_keys(&[])).await;
    let form = reqwest::multipart::Form::new().text("pnc", "maybe").part(
        "file",
        reqwest::multipart::Part::bytes(wav_16k_mono(5.0)).file_name("a.wav".to_string()),
    );
    let resp = post_transcription(&base, form, None).await;
    assert_eq!(resp.status(), 400);
    let json: serde_json::Value = resp.json().await.expect("JSON body");
    let message = json["error"]["message"].as_str().expect("message");
    assert!(message.contains("pnc"), "unexpected message: {message}");
}

#[tokio::test]
async fn transcriptions_unparsable_toggle_is_400() {
    let base = spawn_fake_app(auth_keys(&[])).await;
    let form = transcription_form(wav_16k_mono(0.1), "a.wav").text("pnc", "maybe");
    let resp = post_transcription(&base, form, None).await;
    assert_eq!(resp.status(), 400);
    let json: serde_json::Value = resp.json().await.expect("JSON body");
    assert_eq!(json["error"]["type"], "invalid_request_error");
    let message = json["error"]["message"].as_str().expect("message");
    assert!(message.contains("pnc"), "unexpected message: {message}");
}

/// Open WebUI contract: multipart model + file, Bearer auth, no
/// response_format; response is JSON {"text": ...}. Must never break.
#[cfg(feature = "audio-ffmpeg")]
#[tokio::test]
async fn owui_contract_webm_multipart_returns_json_text() {
    let base = spawn_fake_app(auth_keys(&["k1"])).await;
    let webm = std::fs::read(format!(
        "{}/tests/fixtures/beep.webm",
        env!("CARGO_MANIFEST_DIR")
    ))
    .expect("read beep.webm");
    let form = reqwest::multipart::Form::new()
        .text("model", "whatever")
        .part(
            "file",
            reqwest::multipart::Part::bytes(webm).file_name("audio.webm"),
        );
    let resp = post_transcription(&base, form, Some("k1")).await;
    assert_eq!(resp.status(), 200);
    assert_eq!(
        resp.headers()
            .get("content-type")
            .expect("content-type")
            .to_str()
            .expect("header str"),
        "application/json"
    );
    let json: serde_json::Value = resp.json().await.expect("JSON body");
    let text = json["text"].as_str().expect("text field");
    assert!(!text.is_empty());
    assert!(
        text.starts_with("fake:whatever:"),
        "unexpected text: {text}"
    );
}

#[tokio::test]
async fn transcriptions_response_format_text_is_plain_text() {
    let base = spawn_fake_app(auth_keys(&[])).await;
    let form = transcription_form(wav_16k_mono(1.0), "audio.wav").text("response_format", "text");
    let resp = post_transcription(&base, form, None).await;
    assert_eq!(resp.status(), 200);
    let content_type = resp
        .headers()
        .get("content-type")
        .expect("content-type")
        .to_str()
        .expect("header str")
        .to_string();
    assert!(
        content_type.starts_with("text/plain"),
        "unexpected content-type: {content_type}"
    );
    let body = resp.text().await.expect("text body");
    assert_eq!(body, "fake:default:16000");
}

#[tokio::test]
async fn transcriptions_without_file_is_400() {
    let base = spawn_fake_app(auth_keys(&[])).await;
    let form = reqwest::multipart::Form::new().text("model", "whatever");
    let resp = post_transcription(&base, form, None).await;
    assert_eq!(resp.status(), 400);
    let json: serde_json::Value = resp.json().await.expect("JSON body");
    assert_eq!(json["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn transcriptions_garbage_file_is_400() {
    let base = spawn_fake_app(auth_keys(&[])).await;
    let form = transcription_form(b"definitely not audio".to_vec(), "audio.webm");
    let resp = post_transcription(&base, form, None).await;
    assert_eq!(resp.status(), 400);
    let json: serde_json::Value = resp.json().await.expect("JSON body");
    assert_eq!(json["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn transcriptions_wav_fastpath_through_full_http_path() {
    let base = spawn_fake_app(auth_keys(&[])).await;
    let form = transcription_form(wav_16k_mono(1.0), "audio.wav").text("model", "m1");
    let resp = post_transcription(&base, form, None).await;
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.expect("JSON body");
    assert_eq!(json["text"], "fake:m1:16000");
}

/// verbose_json: the OpenAI shape, with timestamps for a single-chunk file.
#[tokio::test]
async fn transcriptions_verbose_json_has_openai_shape() {
    let base = spawn_fake_app(auth_keys(&[])).await;
    let form = transcription_form(wav_16k_mono(1.0), "audio.wav")
        .text("model", "m1")
        .text("language", "ru")
        .text("response_format", "verbose_json");
    let resp = post_transcription(&base, form, None).await;
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.expect("JSON body");
    assert_eq!(json["task"], "transcribe");
    assert_eq!(json["language"], "ru");
    assert_eq!(json["duration"], 1.0);
    assert_eq!(json["text"], "fake:m1:16000");
    let segments = json["segments"].as_array().expect("segments array");
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0]["id"], 0);
    assert_eq!(segments[0]["start"], 0.0);
    assert_eq!(segments[0]["end"], 1.0);
    assert_eq!(segments[0]["text"], "fake:m1:16000");
    let words = json["words"].as_array().expect("words array");
    assert!(!words.is_empty());
    assert_eq!(words[0]["start"], 0.0);
    assert_eq!(
        words.last().expect("last word")["end"],
        serde_json::json!(1.0)
    );
}

/// The chunker's silent killer: every chunk is transcribed on its own, so its
/// timestamps start at zero and must be shifted by the chunk's offset in the
/// file. A 2 s file with a 0.5 s chunk window is cut into four chunks (a pure
/// sine has no silence to cut at, so the cuts are exactly at the window edge),
/// and the segments must run 0.0, 0.5, 1.0, 1.5 -- not 0.0 four times.
#[tokio::test]
async fn transcriptions_verbose_json_timestamps_are_absolute_across_chunks() {
    let cfg = Arc::new(
        Config::try_parse_from(["ts", "--parallel", "2", "--chunk-max-sec", "0.5"])
            .expect("parse config"),
    );
    let base = spawn_app_with_cfg(Arc::new(FakeEngine), auth_keys(&[]), cfg).await;
    let form =
        transcription_form(wav_16k_mono(2.0), "audio.wav").text("response_format", "verbose_json");
    let resp = post_transcription(&base, form, None).await;
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.expect("JSON body");

    assert_eq!(json["duration"], 2.0);
    let segments = json["segments"].as_array().expect("segments array");
    assert_eq!(segments.len(), 4, "expected four chunks: {segments:?}");
    let bounds: Vec<(f64, f64)> = segments
        .iter()
        .map(|s| {
            (
                s["start"].as_f64().expect("start"),
                s["end"].as_f64().expect("end"),
            )
        })
        .collect();
    assert_eq!(bounds, [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0)]);
    for (i, segment) in segments.iter().enumerate() {
        assert_eq!(segment["id"], i, "ids must be dense and ordered");
    }

    // Words are shifted by the same offset and stay ordered across chunks.
    let words = json["words"].as_array().expect("words array");
    assert_eq!(words.len(), 12, "3 fake words per chunk");
    assert_eq!(words[0]["start"], 0.0);
    assert_eq!(
        words.last().expect("last word")["end"],
        serde_json::json!(2.0)
    );
    let starts: Vec<f64> = words
        .iter()
        .map(|w| w["start"].as_f64().expect("start"))
        .collect();
    assert!(
        starts.windows(2).all(|p| p[0] < p[1]),
        "word starts must increase across chunks: {starts:?}"
    );
}

/// Default (no response_format) stays the bare {"text": ...} Open WebUI relies
/// on: no segments, no words, no duration.
#[tokio::test]
async fn transcriptions_default_json_carries_no_timestamps() {
    let base = spawn_fake_app(auth_keys(&[])).await;
    let form = transcription_form(wav_16k_mono(1.0), "audio.wav");
    let resp = post_transcription(&base, form, None).await;
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.expect("JSON body");
    assert_eq!(json["text"], "fake:default:16000");
    assert_eq!(
        json.as_object().expect("object").len(),
        1,
        "the default json body must stay a single text field: {json}"
    );
}

#[tokio::test]
async fn transcriptions_unknown_response_format_is_400() {
    let base = spawn_fake_app(auth_keys(&[])).await;
    let form = transcription_form(wav_16k_mono(1.0), "audio.wav").text("response_format", "srt");
    let resp = post_transcription(&base, form, None).await;
    assert_eq!(resp.status(), 400);
    let json: serde_json::Value = resp.json().await.expect("JSON body");
    assert_eq!(json["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn transcriptions_without_key_is_401_when_keys_configured() {
    let base = spawn_fake_app(auth_keys(&["secret"])).await;
    let form = transcription_form(wav_16k_mono(0.1), "audio.wav");
    let resp = post_transcription(&base, form, None).await;
    assert_eq!(resp.status(), 401);
    let json: serde_json::Value = resp.json().await.expect("JSON body");
    assert_eq!(json["error"]["type"], "authentication_error");
}

#[tokio::test]
async fn transcriptions_oversized_body_is_413_with_api_error_shape() {
    let cfg = Arc::new(
        Config::try_parse_from(["ts", "--parallel", "2", "--max-upload-mb", "1"])
            .expect("parse config"),
    );
    let base = spawn_app_with_cfg(Arc::new(FakeEngine), auth_keys(&[]), cfg).await;
    // ~70 s of 16-bit PCM is ~2.2 MB, over the 1 MB limit.
    let form = transcription_form(wav_16k_mono(70.0), "audio.wav");
    let resp = post_transcription(&base, form, None).await;
    assert_eq!(resp.status(), 413);
    let json: serde_json::Value = resp.json().await.expect("JSON body");
    assert_eq!(json["error"]["type"], "invalid_request_error");
    assert!(
        json["error"]["message"]
            .as_str()
            .expect("message")
            .contains("1 MB")
    );
}
