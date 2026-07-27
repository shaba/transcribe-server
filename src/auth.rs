//! Optional Bearer API-key auth middleware (llama.cpp style).

/// Shared set of accepted API keys. An empty set disables auth entirely.
#[derive(Clone, Default)]
pub struct AuthKeys(pub std::sync::Arc<std::collections::HashSet<String>>);

// TODO(task 8): drop allow(dead_code) once build_router layers this middleware.
#[allow(dead_code)]
pub async fn require_api_key(
    axum::extract::State(keys): axum::extract::State<AuthKeys>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Result<axum::response::Response, crate::api::error::ApiError> {
    if keys.0.is_empty() {
        return Ok(next.run(req).await);
    }
    let authorized = req
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .is_some_and(|key| keys.0.contains(key));
    if authorized {
        Ok(next.run(req).await)
    } else {
        Err(crate::api::error::ApiError::unauthorized())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::Router;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use axum::middleware;
    use axum::routing::get;
    use tower::ServiceExt;

    fn app(keys: &[&str]) -> Router {
        let keys = AuthKeys(std::sync::Arc::new(
            keys.iter().map(|k| k.to_string()).collect(),
        ));
        Router::new()
            .route("/ping", get(|| async { "pong" }))
            .layer(middleware::from_fn_with_state(keys, require_api_key))
    }

    async fn send(app: Router, auth: Option<&str>) -> axum::response::Response {
        let mut builder = Request::builder().uri("/ping");
        if let Some(value) = auth {
            builder = builder.header("authorization", value);
        }
        app.oneshot(builder.body(Body::empty()).unwrap())
            .await
            .unwrap()
    }

    async fn body_string(resp: axum::response::Response) -> String {
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("read body");
        String::from_utf8(bytes.to_vec()).expect("utf8 body")
    }

    #[tokio::test]
    async fn empty_key_set_allows_all_requests() {
        let resp = send(app(&[]), None).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(body_string(resp).await, "pong");
    }

    #[tokio::test]
    async fn missing_header_yields_401_openai_error_json() {
        let resp = send(app(&["k1"]), None).await;
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        let json: serde_json::Value =
            serde_json::from_str(&body_string(resp).await).expect("JSON body");
        assert_eq!(json["error"]["message"], "invalid or missing API key");
        assert_eq!(json["error"]["type"], "authentication_error");
    }

    #[tokio::test]
    async fn wrong_key_yields_401() {
        let resp = send(app(&["k1"]), Some("Bearer nope")).await;
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn correct_bearer_key_passes() {
        let resp = send(app(&["k1", "k2"]), Some("Bearer k1")).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(body_string(resp).await, "pong");
    }

    #[tokio::test]
    async fn lowercase_bearer_scheme_is_rejected() {
        let resp = send(app(&["k1"]), Some("bearer k1")).await;
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }
}
