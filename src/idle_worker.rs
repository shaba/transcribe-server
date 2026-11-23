//! The supervisor only owns the listener and reaps children. All HTTP, async
//! runtime and native library initialization happens after fork, in the child.
use std::io::{self, Read};
use std::net::TcpListener;
use std::os::fd::AsRawFd;
use std::sync::{Arc, Mutex, OnceLock};
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
    error::ApiError, models::list_models, stream::stream_with_guard, transcriptions,
};
use transcribe_server::auth::{AuthKeys, require_api_key};
use transcribe_server::config::Config;
use transcribe_server::engine::{SttEngine, fake::FakeEngine};
use transcribe_server::server::AppState;

pub(super) fn supervise(cfg: Config) -> Result<(), String> {
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
    let keys = super::auth_keys(&cfg)?;
    if keys.0.is_empty() {
        eprintln!("no API keys configured, authentication is disabled");
    }
    let listener = TcpListener::bind((cfg.host.as_str(), cfg.port))
        .map_err(|e| format!("binding listener: {e}"))?;
    listener.set_nonblocking(true).map_err(|e| e.to_string())?;
    parent(cfg, keys, aliases, listener).map_err(|e| format!("supervisor: {e}"))
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

fn parent(
    cfg: Config,
    keys: AuthKeys,
    aliases: Vec<String>,
    listener: TcpListener,
) -> io::Result<()> {
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
            return serve(cfg, keys, aliases, listener);
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
                    eprintln!("worker exited unexpectedly ({status}); backing off");
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

struct Worker {
    cfg: Arc<Config>,
    aliases: Vec<String>,
    ready: OnceLock<watch::Receiver<Option<Result<AppState, String>>>>,
    activity: Mutex<Activity>,
}
struct Activity {
    active: usize,
    idle_since: Instant,
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
        let mut rx = self
            .ready
            .get_or_init(|| {
                let (tx, rx) = watch::channel(None);
                let cfg = self.cfg.clone();
                let loading = lease.clone();
                tokio::task::spawn_blocking(move || {
                    super::init_native_logging();
                    tracing::info!("loading inference engine");
                    let result = super::build_engine(&cfg).map(|e| AppState::new(e, cfg));
                    if let Err(e) = &result {
                        tracing::error!(%e, "engine initialization failed");
                    }
                    tx.send_replace(Some(result));
                    drop(loading);
                });
                rx
            })
            .clone();
        let ready = rx
            .wait_for(|value| value.is_some())
            .await
            .map_err(|_| ApiError::busy())?;
        Ok((
            ready
                .as_ref()
                .unwrap()
                .as_ref()
                .map_err(|_| ApiError::busy())?
                .clone(),
            lease,
        ))
    }
    async fn idle(&self) {
        let timeout = Duration::from_secs(self.cfg.idle_worker_timeout);
        loop {
            tokio::time::sleep(Duration::from_secs(1)).await;
            let mut a = self.activity.lock().unwrap();
            if a.active == 0 && a.idle_since.elapsed() >= timeout {
                a.retiring = true;
                return;
            }
        }
    }
}

#[tokio::main(flavor = "current_thread")]
async fn serve(
    cfg: Config,
    keys: AuthKeys,
    aliases: Vec<String>,
    listener: TcpListener,
) -> io::Result<()> {
    let listener = tokio::net::TcpListener::from_std(listener)?;
    let limit = DefaultBodyLimit::max(cfg.max_upload_mb * 1024 * 1024);
    let worker = Arc::new(Worker {
        cfg: Arc::new(cfg),
        aliases,
        ready: OnceLock::new(),
        activity: Mutex::new(Activity {
            active: 0,
            idle_since: Instant::now(),
            retiring: false,
        }),
    });
    let v1 = Router::new()
        .route("/v1/models", get(models))
        .route("/v1/audio/transcriptions", post(infer))
        .route("/v1/audio/translations", post(infer))
        .route("/v1/audio/stream", get(stream))
        .layer(middleware::from_fn_with_state(keys, require_api_key));
    let router = Router::new()
        .route("/health", get(health))
        .merge(v1)
        .layer(limit)
        .with_state(worker.clone());
    tracing::info!(pid = std::process::id(), "HTTP worker ready");
    axum::serve(listener, router)
        .with_graceful_shutdown(async move {
            tokio::select! { _=super::shutdown_signal()=>{}, _=worker.idle()=>{} }
        })
        .await
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
    let data: Vec<_> = w
        .aliases
        .iter()
        .map(|id| json!({"id":id,"object":"model","owned_by":"transcribe-server"}))
        .collect();
    Json(json!({"object":"list","data":data})).into_response()
}
async fn infer(State(w): State<Arc<Worker>>, req: Request) -> Result<Response, ApiError> {
    let (s, _lease) = w.acquire().await?;
    Ok(if req.uri().path() == "/v1/audio/translations" {
        transcriptions::translate.call(req, s).await
    } else {
        transcriptions::transcribe.call(req, s).await
    })
}
async fn stream(State(w): State<Arc<Worker>>, ws: WebSocketUpgrade) -> Result<Response, ApiError> {
    let (s, lease) = w.acquire().await?;
    Ok(stream_with_guard(s, ws, lease))
}
