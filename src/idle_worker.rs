//! The supervisor only owns the listener and reaps children. All HTTP, async
//! runtime and native library initialization happens after fork, in the child.
use std::io::{self, Read};
use std::net::TcpListener;
use std::os::fd::AsRawFd;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::extract::{DefaultBodyLimit, Request, State, WebSocketUpgrade};
use axum::handler::Handler;
use axum::http::{HeaderValue, Method, Uri, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router, middleware};
use serde_json::{Value, json};
use tokio::sync::watch;
use transcribe_server::api::{
    error::ApiError,
    models::{list_models, model_object},
    stream::stream_with_guard,
    transcriptions,
};
use transcribe_server::auth::require_api_key;
use transcribe_server::config::{Config, WorkerMode};
use transcribe_server::engine::{ModelInfo, SttEngine, fake::FakeEngine};
use transcribe_server::server::AppState;

pub(super) fn run(cfg: Config, mode: WorkerMode) -> Result<(), String> {
    let aliases = match cfg.engine.as_str() {
        "fake" => FakeEngine.models().into_iter().map(|m| m.id).collect(),
        "transcribe" if cfg.model.is_empty() => {
            return Err("no model specified: pass -m /path/to/model.gguf".into());
        }
        "transcribe" if !cfg!(feature = "engine-transcribe") => {
            return Err(
                "on-demand transcription requires the engine-transcribe build feature".into(),
            );
        }
        "transcribe" => cfg.model_specs().into_iter().map(|s| s.alias).collect(),
        other => return Err(format!("unknown engine: {other}")),
    };
    let listener = TcpListener::bind((cfg.host.as_str(), cfg.port))
        .map_err(|e| format!("binding listener: {e}"))?;
    listener.set_nonblocking(true).map_err(|e| e.to_string())?;
    match mode {
        WorkerMode::Supervised => parent(cfg, aliases, listener),
        WorkerMode::Direct => {
            super::init_tracing(cfg.verbose);
            tracing::info!(
                "listening on http://{} (direct worker)",
                listener.local_addr().map_err(|e| e.to_string())?
            );
            serve(cfg, aliases, listener)
        }
    }
    .map_err(|e| format!("on-demand server: {e}"))
}

// No child exists here. Unblock termination during the indefinite wait; the
// installed handler exits even if the signal arrives just before poll(). Block
// again before fork so the parent can forward signals and reap a live child.
fn wait_for_connection(listener: &TcpListener, signals: &libc::sigset_t) -> io::Result<()> {
    let mut fd = libc::pollfd {
        fd: listener.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };
    if unsafe { libc::sigprocmask(libc::SIG_UNBLOCK, signals, std::ptr::null_mut()) } < 0 {
        return Err(io::Error::last_os_error());
    }
    loop {
        let rc = unsafe { libc::poll(&mut fd, 1, -1) };
        if rc < 0 {
            let err = io::Error::last_os_error();
            if err.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(err);
        }
        if fd.revents & libc::POLLIN != 0 {
            break;
        }
        return Err(io::Error::other(format!(
            "listener poll error: {}",
            fd.revents
        )));
    }
    if unsafe { libc::sigprocmask(libc::SIG_BLOCK, signals, std::ptr::null_mut()) } < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

fn parent(cfg: Config, aliases: Vec<String>, listener: TcpListener) -> io::Result<()> {
    // Synchronous signal waiting avoids handlers, threads and the race between
    // checking a flag and sleeping in waitpid. Children restore the old mask.
    extern "C" fn caught(sig: libc::c_int) {
        if sig == libc::SIGTERM || sig == libc::SIGINT {
            // Only unblocked while idle. _exit is async-signal-safe.
            unsafe {
                libc::_exit(0);
            }
        }
    }
    let (signals, old) = unsafe {
        let mut set = std::mem::zeroed();
        let mut old = std::mem::zeroed();
        libc::sigemptyset(&mut set);
        for sig in [libc::SIGCHLD, libc::SIGTERM, libc::SIGINT] {
            libc::sigaddset(&mut set, sig);
        }
        if libc::sigprocmask(libc::SIG_BLOCK, &set, &mut old) < 0 {
            return Err(io::Error::last_os_error());
        }
        // Explicit dispositions also let a container PID 1 receive TERM/INT
        // and undo an inherited SIGCHLD ignore (which would auto-reap).
        for sig in [libc::SIGCHLD, libc::SIGTERM, libc::SIGINT] {
            if libc::signal(sig, caught as *const () as libc::sighandler_t) == libc::SIG_ERR {
                return Err(io::Error::last_os_error());
            }
        }
        (set, old)
    };
    let local = listener.local_addr()?;
    loop {
        println!("listening on http://{local} (waiting for connection)");
        wait_for_connection(&listener, &signals)?;
        // Linked native libraries can start threads before main. Don't run
        // Rust in the child of a known multithreaded process.
        #[cfg(target_os = "linux")]
        if std::fs::read_dir("/proc/self/task")?.count() != 1 {
            return Err(io::Error::other(
                "on-demand fork requires a single-threaded supervisor",
            ));
        }
        let (mut death, writer) = io::pipe()?;
        // SAFETY: this binary enters parent() directly from main, before
        // starting any threads or initializing tracing, Tokio or the engine.
        let pid = unsafe { libc::fork() };
        if pid < 0 {
            return Err(io::Error::last_os_error());
        }
        if pid == 0 {
            drop(writer);
            for sig in [libc::SIGCHLD, libc::SIGTERM, libc::SIGINT] {
                unsafe {
                    libc::signal(sig, libc::SIG_DFL);
                }
            }
            if unsafe { libc::sigprocmask(libc::SIG_SETMASK, &old, std::ptr::null_mut()) } < 0 {
                return Err(io::Error::last_os_error());
            }
            std::thread::Builder::new()
                .name("parent-watch".into())
                .spawn(move || {
                    loop {
                        match death.read(&mut [0]) {
                            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                            _ => break,
                        }
                    }
                    // Do not run native atexit hooks during active inference.
                    unsafe {
                        libc::_exit(0);
                    }
                })?;
            super::init_tracing(cfg.verbose);
            return serve(cfg, aliases, listener);
        }
        drop(death);
        let mut stopping = false;
        loop {
            let mut status = 0;
            let rc = unsafe { libc::waitpid(pid, &mut status, libc::WNOHANG) };
            if rc == pid {
                if stopping {
                    return Ok(());
                }
                if !libc::WIFEXITED(status) || libc::WEXITSTATUS(status) != 0 {
                    let detail = if libc::WIFEXITED(status) {
                        format!("exit code {}", libc::WEXITSTATUS(status))
                    } else if libc::WIFSIGNALED(status) {
                        format!("signal {}", libc::WTERMSIG(status))
                    } else {
                        format!("wait status {status}")
                    };
                    eprintln!("worker exited unexpectedly ({detail}); backing off");
                    std::thread::sleep(Duration::from_secs(1));
                }
                break;
            }
            if rc < 0 {
                let e = io::Error::last_os_error();
                if e.kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(e);
            }
            let mut sig = 0;
            let rc = unsafe { libc::sigwait(&signals, &mut sig) };
            if rc != 0 {
                return Err(io::Error::from_raw_os_error(rc));
            }
            if sig != libc::SIGCHLD {
                stopping = true;
                unsafe {
                    libc::kill(pid, libc::SIGTERM);
                }
            }
        }
        drop(writer);
    }
}

type LoadAttempt = watch::Receiver<Option<Result<AppState, String>>>;

/// How long a failed model load is cached before the next request retries it.
const LOAD_RETRY_DELAY: Duration = Duration::from_secs(5);

struct Worker {
    cfg: Arc<Config>,
    aliases: Vec<String>,
    ready: Mutex<Option<LoadAttempt>>,
    failed_at: Mutex<Option<Instant>>,
    activity: Mutex<Activity>,
}
struct Activity {
    active: usize,
    idle_since: Instant,
    models_loaded: bool,
    retiring: bool,
}
struct Lease(Arc<Worker>);
impl Drop for Lease {
    fn drop(&mut self) {
        let mut a = self.0.activity.lock().unwrap();
        a.active -= 1;
        if a.active == 0 {
            a.idle_since = Instant::now();
        }
    }
}
impl Worker {
    fn lease(self: &Arc<Self>) -> Result<Arc<Lease>, ApiError> {
        let mut a = self.activity.lock().unwrap();
        if a.retiring {
            return Err(ApiError::busy());
        }
        a.active += 1;
        Ok(Arc::new(Lease(self.clone())))
    }
    async fn acquire(self: &Arc<Self>) -> Result<(AppState, Arc<Lease>), ApiError> {
        let lease = self.lease()?;
        // The load belongs to the process, not the first HTTP request. A
        // disconnected caller must neither cancel nor duplicate a native load.
        let mut rx = {
            let mut slot = self.ready.lock().unwrap();
            // A failed load is retried, but not on every request: a persistent
            // misconfiguration would otherwise run a full native load per call.
            let stale = match slot.as_ref() {
                Some(rx) => match &*rx.borrow() {
                    Some(Err(_)) => {
                        let recent = self
                            .failed_at
                            .lock()
                            .unwrap()
                            .is_some_and(|at| at.elapsed() < LOAD_RETRY_DELAY);
                        if recent {
                            return Err(ApiError::unavailable(
                                "model load failed; retry shortly",
                            ));
                        }
                        true
                    }
                    None => rx.has_changed().is_err(),
                    Some(Ok(_)) => false,
                },
                None => false,
            };
            if stale {
                *slot = None;
            }
            slot.get_or_insert_with(|| {
                let (tx, rx) = watch::channel(None);
                let cfg = self.cfg.clone();
                let loading = lease.clone();
                tokio::task::spawn_blocking(move || {
                    super::init_native_logging();
                    tracing::info!("loading inference engine");
                    let result = super::build_engine(&cfg).map(|e| AppState::new(e, cfg));
                    match &result {
                        Ok(_) => {
                            loading.0.activity.lock().unwrap().models_loaded = true;
                            *loading.0.failed_at.lock().unwrap() = None;
                        }
                        Err(e) => {
                            tracing::error!(%e, "engine initialization failed");
                            *loading.0.failed_at.lock().unwrap() = Some(Instant::now());
                        }
                    }
                    tx.send_replace(Some(result));
                    drop(loading);
                });
                rx
            })
            .clone()
        };
        let ready = rx
            .wait_for(|value| value.is_some())
            .await
            .map_err(|_| ApiError::unavailable("model load failed"))?;
        Ok((
            ready
                .as_ref()
                .unwrap()
                .as_ref()
                .map_err(|_| ApiError::unavailable("model load failed"))?
                .clone(),
            lease,
        ))
    }
    async fn idle(&self) {
        let unloaded = self.cfg.idle_seconds(false);
        let loaded = self.cfg.idle_seconds(true);
        if unloaded == 0 && loaded == 0 {
            return std::future::pending().await;
        }
        loop {
            tokio::time::sleep(Duration::from_secs(1)).await;
            let mut a = self.activity.lock().unwrap();
            if a.active != 0 {
                continue;
            }
            let seconds = if a.models_loaded { loaded } else { unloaded };
            if seconds != 0 && a.idle_since.elapsed() >= Duration::from_secs(seconds) {
                a.retiring = true;
                return;
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum StopReason {
    Idle,
    Signal,
}

#[tokio::main(flavor = "current_thread")]
async fn serve(cfg: Config, aliases: Vec<String>, listener: TcpListener) -> io::Result<()> {
    // Keys are read here, after fork, so a supervised parent never holds them
    // beyond what Config already carries from the command line.
    let keys = super::auth_keys(&cfg).map_err(io::Error::other)?;
    if keys.0.is_empty() {
        tracing::warn!("no API keys configured, authentication is disabled");
    }
    let listener = tokio::net::TcpListener::from_std(listener)?;
    let limit = DefaultBodyLimit::max(cfg.max_upload_mb * 1024 * 1024);
    let worker = Arc::new(Worker {
        cfg: Arc::new(cfg),
        aliases,
        ready: Mutex::default(),
        failed_at: Mutex::default(),
        activity: Mutex::new(Activity {
            active: 0,
            models_loaded: false,
            idle_since: Instant::now(),
            retiring: false,
        }),
    });
    let v1 = Router::new()
        .route("/v1/models", get(models))
        .route("/v1/audio/transcriptions", post(infer_transcribe))
        .route("/v1/audio/translations", post(infer_translate))
        .route("/v1/audio/stream", get(stream))
        .layer(middleware::from_fn_with_state(keys, require_api_key));
    let router = Router::new()
        .route("/health", get(health))
        .merge(v1)
        .layer(limit)
        .with_state(worker.clone());
    tracing::info!(pid = std::process::id(), "HTTP worker ready");

    // Graceful shutdown waits for every connection, and a stalled client can
    // hold one open forever. Cap the wait so the supervisor can fork a fresh
    // worker: idle retirement has nothing legitimate left to drain, while a
    // signal may be waiting for an in-flight transcription.
    let (stop_tx, mut stop_rx) = watch::channel(None::<StopReason>);
    let graceful = async move {
        let reason = tokio::select! {
            _ = super::shutdown_signal() => StopReason::Signal,
            _ = worker.idle() => StopReason::Idle,
        };
        let _ = stop_tx.send(Some(reason));
    };
    let hard_stop = async move {
        let reason = match stop_rx.wait_for(|value| value.is_some()).await {
            Ok(value) => (*value).unwrap(),
            Err(_) => StopReason::Signal,
        };
        let limit = match reason {
            StopReason::Idle => Duration::from_secs(5),
            StopReason::Signal => Duration::from_secs(60),
        };
        tokio::time::sleep(limit).await;
        tracing::warn!(
            ?reason,
            "graceful shutdown did not finish; exiting without draining"
        );
        unsafe { libc::_exit(0) }
    };
    tokio::select! {
        result = axum::serve(listener, router).with_graceful_shutdown(graceful) => result,
        _ = hard_stop => unreachable!(),
    }
}
async fn health(State(w): State<Arc<Worker>>) -> Json<Value> {
    Json(json!({"status":"ok","backend":"on-demand","models":w.aliases}))
}
async fn models(State(w): State<Arc<Worker>>, method: Method, uri: Uri) -> Response {
    if method == Method::GET
        && uri
            .query()
            .is_some_and(|q| q.split('&').any(|p| p == "wakeup"))
    {
        let mut res = match w.acquire().await {
            Ok((s, _lease)) => list_models(State(s)).await.into_response(),
            Err(e) => e.into_response(),
        };
        res.headers_mut()
            .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
        return res;
    }
    // Same shape as the loaded list, so a client can parse it before the
    // first wake-up; only the capability fields are absent until load.
    let data: Vec<_> = w
        .aliases
        .iter()
        .map(|id| {
            model_object(&ModelInfo {
                id: id.clone(),
                ..ModelInfo::default()
            })
        })
        .collect();
    Json(json!({"object":"list","data":data})).into_response()
}
async fn infer_transcribe(
    State(w): State<Arc<Worker>>,
    req: Request,
) -> Result<Response, ApiError> {
    let (s, _lease) = w.acquire().await?;
    Ok(transcriptions::transcribe.call(req, s).await)
}
async fn infer_translate(
    State(w): State<Arc<Worker>>,
    req: Request,
) -> Result<Response, ApiError> {
    let (s, _lease) = w.acquire().await?;
    Ok(transcriptions::translate.call(req, s).await)
}
async fn stream(State(w): State<Arc<Worker>>, ws: WebSocketUpgrade) -> Result<Response, ApiError> {
    let (s, lease) = w.acquire().await?;
    Ok(stream_with_guard(s, ws, lease))
}
