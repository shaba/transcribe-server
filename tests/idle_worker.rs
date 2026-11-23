//! Exercise the on-demand listener and real child lifecycle with FakeEngine.
#![cfg(unix)]

use std::process::Stdio;
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use reqwest::{Client, StatusCode, multipart};
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::net::TcpStream;
use tokio::process::Command;
use tokio::time::{sleep, timeout};
use tokio_tungstenite::tungstenite::{Message, client::IntoClientRequest};

const KEY: &str = "test-idle-worker";
const WAIT: Duration = Duration::from_secs(10);

#[cfg(target_os = "linux")]
fn assert_no_child(parent: u32) {
    let children =
        std::fs::read_to_string(format!("/proc/{parent}/task/{parent}/children")).unwrap();
    assert!(children.trim().is_empty(), "unexpected child: {children}");
}

async fn next_listener(
    logs: &mut tokio::io::Lines<BufReader<tokio::process::ChildStdout>>,
) -> String {
    timeout(WAIT, async {
        while let Some(line) = logs.next_line().await.expect("read server log") {
            if let Some((_, suffix)) = line.split_once("listening on http://") {
                return suffix.split_whitespace().next().unwrap().to_string();
            }
        }
        panic!("server exited before listening");
    })
    .await
    .expect("server should start listening")
}

#[tokio::test]
async fn idle_supervisor_exits_on_sigterm_without_a_connection() {
    let mut server = Command::new(env!("CARGO_BIN_EXE_transcribe-server"))
        .args([
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--engine",
            "fake",
            "--idle-timeout",
            "1",
        ])
        .stdout(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    let mut logs = BufReader::new(server.stdout.take().unwrap()).lines();
    next_listener(&mut logs).await;
    let pid = server.id().unwrap();
    #[cfg(target_os = "linux")]
    assert_no_child(pid);
    assert_eq!(unsafe { libc::kill(pid as libc::pid_t, libc::SIGTERM) }, 0);
    assert!(
        timeout(WAIT, server.wait())
            .await
            .unwrap()
            .unwrap()
            .success()
    );
}

async fn direct_worker(args: &[&str]) -> (tokio::process::Child, String) {
    let mut worker = Command::new(env!("CARGO_BIN_EXE_transcribe-server"))
        .args([
            "--worker-mode",
            "direct",
            "--port",
            "0",
            "--host",
            "127.0.0.1",
            "--engine",
            "fake",
        ])
        .args(args)
        .env("TRANSCRIBE_IDLE_TIMEOUT", "1")
        .env("RUST_LOG", "info")
        .env("NO_COLOR", "1")
        .stdout(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    let mut logs = BufReader::new(worker.stdout.take().unwrap()).lines();
    let addr = next_listener(&mut logs).await;
    tokio::spawn(async move { while let Ok(Some(_)) = logs.next_line().await {} });
    (worker, format!("http://{addr}"))
}

#[tokio::test]
async fn unloaded_override_keeps_direct_worker_until_loaded_then_exits() {
    let (mut worker, base) = direct_worker(&["--idle-timeout-unloaded", "0"]).await;
    let client = Client::builder().no_proxy().timeout(WAIT).build().unwrap();
    sleep(Duration::from_millis(2200)).await;
    assert!(worker.try_wait().unwrap().is_none());
    #[cfg(target_os = "linux")]
    assert_no_child(worker.id().unwrap());
    client
        .get(format!("{base}/v1/models?wakeup"))
        .send()
        .await
        .unwrap()
        .error_for_status()
        .unwrap();
    assert!(
        timeout(WAIT, worker.wait())
            .await
            .unwrap()
            .unwrap()
            .success()
    );
}

#[tokio::test]
async fn loaded_override_keeps_loaded_worker_past_default_timeout() {
    let (mut worker, base) = direct_worker(&["--idle-timeout-loaded", "0"]).await;
    let client = Client::builder().no_proxy().timeout(WAIT).build().unwrap();
    client
        .get(format!("{base}/v1/models?wakeup"))
        .send()
        .await
        .unwrap()
        .error_for_status()
        .unwrap();
    sleep(Duration::from_millis(2200)).await;
    assert!(worker.try_wait().unwrap().is_none());
    worker.kill().await.unwrap();
}

#[tokio::test]
async fn worker_wakes_serves_requests_and_unloads() {
    let tmp = tempfile::tempdir().expect("test tempdir");
    let mut server = Command::new(env!("CARGO_BIN_EXE_transcribe-server"))
        .args([
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--engine",
            "fake",
            "--idle-timeout",
            "1",
            "--api-key",
            KEY,
        ])
        .env("TMPDIR", tmp.path())
        .env("RUST_LOG", "info")
        .env("NO_COLOR", "1")
        .stdout(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .expect("start server");
    let mut logs = BufReader::new(server.stdout.take().unwrap()).lines();
    let addr = next_listener(&mut logs).await;
    #[cfg(target_os = "linux")]
    assert_no_child(server.id().unwrap());
    let client = Client::builder().no_proxy().timeout(WAIT).build().unwrap();
    let base = format!("http://{addr}");
    let models = format!("{base}/v1/models");
    let wakeup = format!("{models}?wakeup");

    let health: Value = client
        .get(format!("{base}/health"))
        .send()
        .await
        .unwrap()
        .error_for_status()
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(health["backend"], "on-demand");
    assert_eq!(
        client.get(&wakeup).send().await.unwrap().status(),
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        client
            .head(&wakeup)
            .bearer_auth(KEY)
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::OK
    );
    let listed: Value = client
        .get(&models)
        .bearer_auth(KEY)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(listed["data"][0]["id"], "fake-model");
    assert!(listed["data"][0].get("arch").is_none());
    // The first connection started a worker, but none of these probes may
    // load the engine. Wait for the supervisor to reap the probe-only child.
    timeout(WAIT, async {
        loop {
            let line = logs.next_line().await.unwrap().expect("supervisor exited");
            assert!(!line.contains("loading inference engine"), "{line}");
            if line.contains("(waiting for connection)") {
                break;
            }
        }
    })
    .await
    .expect("probe-only child did not retire");
    #[cfg(target_os = "linux")]
    assert_no_child(server.id().unwrap());
    assert_eq!(tmp.path().read_dir().unwrap().count(), 0);

    let wake = || client.get(&wakeup).bearer_auth(KEY).send();
    let (first, second) = tokio::join!(wake(), wake());
    assert_eq!(second.unwrap().status(), StatusCode::OK);
    let ready = first.unwrap();
    assert_eq!(ready.status(), StatusCode::OK);
    assert_eq!(ready.headers()["cache-control"], "no-store");
    assert_eq!(
        ready.json::<Value>().await.unwrap()["data"][0]["arch"],
        "fake"
    );
    let mut req = format!("ws://{addr}/v1/audio/stream")
        .into_client_request()
        .unwrap();
    req.headers_mut()
        .insert("authorization", format!("Bearer {KEY}").parse().unwrap());
    let (mut ws, _) = timeout(WAIT, tokio_tungstenite::connect_async(req))
        .await
        .unwrap()
        .unwrap();
    ws.send(Message::Text(r#"{"type":"start"}"#.into()))
        .await
        .unwrap();
    // The reaper polls once per second, so two ticks would have unloaded it.
    sleep(Duration::from_millis(2200)).await;
    ws.send(Message::Binary(vec![0; 32_000].into()))
        .await
        .unwrap();
    ws.send(Message::Text(r#"{"type":"stop"}"#.into()))
        .await
        .unwrap();
    let frame = timeout(WAIT, ws.next()).await.unwrap().unwrap().unwrap();
    let final_msg: Value = serde_json::from_str(frame.to_text().unwrap()).unwrap();
    assert_eq!(final_msg["type"], "final");
    assert_eq!(final_msg["text"], "fake:default:16000");
    drop(ws);

    assert_eq!(next_listener(&mut logs).await, addr);
    #[cfg(target_os = "linux")]
    assert_no_child(server.id().unwrap());
    let audio =
        multipart::Part::bytes(&include_bytes!("fixtures/beep-16k.wav")[..]).file_name("beep.wav");
    let transcription: Value = client
        .post(format!("{base}/v1/audio/transcriptions"))
        .bearer_auth(KEY)
        .multipart(multipart::Form::new().part("file", audio))
        .send()
        .await
        .unwrap()
        .error_for_status()
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(transcription["text"].as_str().unwrap().starts_with("fake:"));
    // Even a hard kill of the supervisor must release the inherited listener.
    server.kill().await.unwrap();
    timeout(WAIT, async {
        while TcpStream::connect(&addr).await.is_ok() {
            sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .expect("orphan worker kept listener open");
}
