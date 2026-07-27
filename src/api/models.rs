use axum::Json;
use axum::extract::State;
use serde_json::{Value, json};

use crate::server::AppState;

/// GET /v1/models -> OpenAI-style model list.
pub async fn list_models(State(state): State<AppState>) -> Json<Value> {
    let data: Vec<Value> = state
        .engine
        .models()
        .iter()
        .map(|id| {
            json!({
                "id": id,
                "object": "model",
                "owned_by": "transcribe-server",
            })
        })
        .collect();
    Json(json!({"object": "list", "data": data}))
}
