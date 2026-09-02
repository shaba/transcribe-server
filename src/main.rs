use std::process::ExitCode;
use std::sync::Arc;

use clap::Parser;
use tracing_subscriber::EnvFilter;

use transcribe_server::auth::AuthKeys;
use transcribe_server::config::Config;
use transcribe_server::engine::SttEngine;
use transcribe_server::engine::fake::FakeEngine;
use transcribe_server::server::{AppState, build_router};

fn main() -> ExitCode {
    let cfg = Config::parse();
    init_tracing(cfg.verbose);
    init_native_logging();
    match run(cfg) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            tracing::error!("{message}");
            ExitCode::FAILURE
        }
    }
}

/// Send the native library's diagnostics through the subscriber installed
/// above, so they carry a level and RUST_LOG can filter them. A no-op without
/// the engine, which is the only thing that has a native library to quiet.
fn init_native_logging() {
    #[cfg(feature = "engine-transcribe")]
    transcribe_server::engine::transcribe_cpp::init_logging();
}

/// RUST_LOG overrides everything; otherwise -v selects debug, default info.
fn init_tracing(verbose: bool) {
    let default = if verbose { "debug" } else { "info" };
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default));
    tracing_subscriber::fmt().with_env_filter(filter).init();
}

#[tokio::main]
async fn run(cfg: Config) -> Result<(), String> {
    tracing::debug!("config: {cfg:#?}");
    let keys: std::collections::HashSet<String> = cfg
        .all_api_keys()
        .map_err(|e| format!("reading API keys: {e}"))?
        .into_iter()
        .collect();
    if keys.is_empty() {
        tracing::warn!("no API keys configured, authentication is disabled");
    }
    let keys = AuthKeys(Arc::new(keys));

    let engine = build_engine(&cfg)?;
    tracing::info!(
        backend = %engine.backend(),
        // Ids only: a ModelInfo carries every language the model enumerates,
        // which is a hundred of them for a whisper model.
        models = ?engine.models().iter().map(|m| &m.id).collect::<Vec<_>>(),
        "engine ready"
    );

    let state = AppState::new(engine, Arc::new(cfg));
    let addr = (state.cfg.host.as_str(), state.cfg.port);
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| format!("binding {}:{}: {e}", addr.0, addr.1))?;
    let local = listener
        .local_addr()
        .map_err(|e| format!("local_addr: {e}"))?;
    tracing::info!("listening on http://{local}");

    axum::serve(listener, build_router(state, keys))
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|e| format!("server error: {e}"))
}

fn build_engine(cfg: &Config) -> Result<Arc<dyn SttEngine>, String> {
    match cfg.engine.as_str() {
        "fake" => Ok(Arc::new(FakeEngine)),
        "transcribe" => {
            #[cfg(feature = "engine-transcribe")]
            {
                if cfg.model.is_empty() {
                    return Err(
                        "no model specified: pass -m /path/to/model.gguf (or --engine fake \
                         for testing)"
                            .to_string(),
                    );
                }
                let engine = transcribe_server::engine::transcribe_cpp::TranscribeCppEngine::load(
                    &cfg.model_specs(),
                    cfg.no_gpu,
                    cfg.threads,
                )
                .map_err(|e| e.to_string())?;
                Ok(Arc::new(engine))
            }
            #[cfg(not(feature = "engine-transcribe"))]
            {
                Err(
                    "this binary was built without the engine-transcribe feature; \
                     use --engine fake or rebuild with --features engine-transcribe"
                        .to_string(),
                )
            }
        }
        other => Err(format!(
            "unknown engine: {other} (expected \"transcribe\" or \"fake\")"
        )),
    }
}

async fn shutdown_signal() {
    if let Err(e) = tokio::signal::ctrl_c().await {
        tracing::error!("failed to install shutdown signal handler: {e}");
        return;
    }
    tracing::info!("shutdown signal received, draining connections");
}
