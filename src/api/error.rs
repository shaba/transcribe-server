use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

/// OpenAI-style API error: rendered as {"error":{"message":...,"type":...}}.
pub struct ApiError {
    pub status: StatusCode,
    pub message: String,
    pub error_type: &'static str,
}

impl ApiError {
    pub fn bad_request(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: msg.into(),
            error_type: "invalid_request_error",
        }
    }

    pub fn unauthorized() -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            message: "invalid or missing API key".to_string(),
            error_type: "authentication_error",
        }
    }

    pub fn too_large(limit_mb: usize) -> Self {
        Self {
            status: StatusCode::PAYLOAD_TOO_LARGE,
            message: format!("request body exceeds the {limit_mb} MB limit"),
            error_type: "invalid_request_error",
        }
    }

    pub fn busy() -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: "server is busy, try again later".to_string(),
            error_type: "server_error",
        }
    }

    /// The caller closed the connection before the answer was ready. Nginx's
    /// 499: no standard status says it, and no client ever reads this one.
    pub fn cancelled() -> Self {
        Self {
            status: StatusCode::from_u16(499).expect("499 is a valid status code"),
            message: "client closed the request".to_string(),
            error_type: "server_error",
        }
    }

    pub fn internal(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: msg.into(),
            error_type: "server_error",
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = json!({
            "error": {
                "message": self.message,
                "type": self.error_type,
            }
        });
        (self.status, axum::Json(body)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;
    use axum::response::IntoResponse;

    async fn body_json(resp: axum::response::Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("read body");
        serde_json::from_slice(&bytes).expect("body is JSON")
    }

    #[tokio::test]
    async fn bad_request_produces_openai_error_json() {
        let resp = ApiError::bad_request("x").into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json = body_json(resp).await;
        assert_eq!(json["error"]["message"], "x");
        assert_eq!(json["error"]["type"], "invalid_request_error");
    }

    #[tokio::test]
    async fn constructors_map_to_expected_status_and_type() {
        let cases = [
            (
                ApiError::unauthorized(),
                StatusCode::UNAUTHORIZED,
                "authentication_error",
            ),
            (
                ApiError::too_large(25),
                StatusCode::PAYLOAD_TOO_LARGE,
                "invalid_request_error",
            ),
            (
                ApiError::busy(),
                StatusCode::SERVICE_UNAVAILABLE,
                "server_error",
            ),
            (
                ApiError::internal("boom"),
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
            ),
        ];
        for (err, status, error_type) in cases {
            let resp = err.into_response();
            assert_eq!(resp.status(), status);
            let json = body_json(resp).await;
            assert_eq!(json["error"]["type"], error_type);
            assert!(json["error"]["message"].is_string());
        }
    }

    #[tokio::test]
    async fn too_large_message_mentions_limit() {
        let resp = ApiError::too_large(25).into_response();
        let json = body_json(resp).await;
        let msg = json["error"]["message"].as_str().unwrap();
        assert!(
            msg.contains("25"),
            "message should mention the limit: {msg}"
        );
    }
}
