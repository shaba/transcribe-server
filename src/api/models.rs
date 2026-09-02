use axum::Json;
use axum::extract::State;
use serde_json::{Map, Value, json};

use crate::engine::ModelInfo;
use crate::server::AppState;

/// GET /v1/models -> OpenAI-style model list.
///
/// The three OpenAI keys are always there; what the library reports about the
/// loaded model is added next to them, so a client can pick a model by what it
/// can actually do instead of by naming convention. A field the model does not
/// report is left out rather than sent as an empty value.
pub async fn list_models(State(state): State<AppState>) -> Json<Value> {
    let data: Vec<Value> = state.engine.models().iter().map(model_object).collect();
    Json(json!({"object": "list", "data": data}))
}

fn model_object(model: &ModelInfo) -> Value {
    let mut object = Map::new();
    object.insert("id".to_string(), json!(model.id));
    object.insert("object".to_string(), json!("model"));
    object.insert("owned_by".to_string(), json!("transcribe-server"));
    if !model.arch.is_empty() {
        object.insert("arch".to_string(), json!(model.arch));
    }
    if !model.languages.is_empty() {
        object.insert("languages".to_string(), json!(model.languages));
    }
    object.insert(
        "supports_translate".to_string(),
        json!(model.supports_translate),
    );
    if !model.translate_target_languages.is_empty() {
        object.insert(
            "translate_target_languages".to_string(),
            json!(model.translate_target_languages),
        );
    }
    if let Some(seconds) = model.max_audio_sec {
        object.insert("max_audio_sec".to_string(), json!(seconds));
    }
    Value::Object(object)
}

#[cfg(test)]
mod tests {
    use super::model_object;
    use crate::engine::ModelInfo;

    #[test]
    fn openai_keys_are_always_present() {
        let object = model_object(&ModelInfo {
            id: "ru".to_string(),
            ..ModelInfo::default()
        });
        assert_eq!(object["id"], "ru");
        assert_eq!(object["object"], "model");
        assert_eq!(object["owned_by"], "transcribe-server");
    }

    #[test]
    fn unreported_capabilities_are_omitted_not_emptied() {
        let object = model_object(&ModelInfo {
            id: "ru".to_string(),
            ..ModelInfo::default()
        });
        for key in [
            "arch",
            "languages",
            "translate_target_languages",
            "max_audio_sec",
        ] {
            assert!(object.get(key).is_none(), "{key} should be omitted");
        }
        // A false capability is a real answer, not a missing one.
        assert_eq!(object["supports_translate"], false);
    }

    #[test]
    fn reported_capabilities_are_carried_over() {
        let object = model_object(&ModelInfo {
            id: "ru".to_string(),
            arch: "gigaam".to_string(),
            languages: vec!["ru".to_string()],
            supports_translate: true,
            translate_target_languages: vec!["en".to_string()],
            max_audio_sec: Some(30.0),
        });
        assert_eq!(object["arch"], "gigaam");
        assert_eq!(object["languages"], serde_json::json!(["ru"]));
        assert_eq!(object["supports_translate"], true);
        assert_eq!(
            object["translate_target_languages"],
            serde_json::json!(["en"])
        );
        assert_eq!(object["max_audio_sec"], 30.0);
    }
}
