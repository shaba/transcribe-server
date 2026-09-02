use axum::Json;
use axum::extract::State;
use serde_json::{Value, json};

use crate::server::AppState;

/// GET /health -> {"status":"ok","backend":"...","models":[...]}
pub async fn health(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "status": "ok",
        "backend": state.engine.backend(),
        "models": state.engine.models().iter().map(|m| &m.id).collect::<Vec<_>>(),
    }))
}
