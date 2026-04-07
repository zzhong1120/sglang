use std::sync::Arc;

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
    response::IntoResponse,
    routing::post,
    Json, Router,
};
use serde_json::json;
use smg::{
    config::RouterConfig,
    core::{BasicWorkerBuilder, WorkerType},
    routers::RouterFactory,
};
use tower::ServiceExt;

use crate::common::{create_test_context, test_app};

struct UpstreamServer {
    url: String,
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
    handle: tokio::task::JoinHandle<()>,
}

impl UpstreamServer {
    async fn start(route: &'static str) -> Self {
        async fn engine_handler() -> impl IntoResponse {
            (
                StatusCode::OK,
                Json(json!({
                    "id": "resp_engine_path",
                    "object": "test",
                    "route": "matched"
                })),
            )
        }

        let app = Router::new().route(route, post(engine_handler));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();

        let handle = tokio::spawn(async move {
            let server = axum::serve(listener, app).with_graceful_shutdown(async move {
                let _ = shutdown_rx.await;
            });
            server.await.unwrap();
        });

        Self {
            url: format!("http://{}", addr),
            shutdown_tx: Some(shutdown_tx),
            handle,
        }
    }

    async fn shutdown(mut self) {
        if let Some(shutdown_tx) = self.shutdown_tx.take() {
            let _ = shutdown_tx.send(());
        }
        let _ = tokio::time::timeout(tokio::time::Duration::from_secs(5), self.handle).await;
    }
}

fn base_test_config() -> RouterConfig {
    RouterConfig::builder()
        .regular_mode(vec![])
        .random_policy()
        .host("127.0.0.1")
        .port(3900)
        .max_payload_size(256 * 1024 * 1024)
        .request_timeout_secs(30)
        .worker_startup_timeout_secs(1)
        .worker_startup_check_interval_secs(1)
        .max_concurrent_requests(64)
        .queue_timeout_secs(60)
        .build_unchecked()
}

async fn build_regular_app(worker_url: &str) -> axum::Router {
    let app_context = create_test_context(base_test_config()).await;
    let worker = BasicWorkerBuilder::new(worker_url)
        .worker_type(WorkerType::Regular)
        .build();
    app_context.worker_registry.register(Arc::from(worker));

    let router = Arc::from(RouterFactory::create_router(&app_context).await.unwrap());
    test_app::create_test_app_with_context(router, app_context)
}

async fn build_pd_app(prefill_url: &str, decode_url: &str) -> axum::Router {
    let config = RouterConfig::builder()
        .prefill_decode_mode(vec![], vec![])
        .random_policy()
        .host("127.0.0.1")
        .port(3901)
        .max_payload_size(256 * 1024 * 1024)
        .request_timeout_secs(30)
        .worker_startup_timeout_secs(1)
        .worker_startup_check_interval_secs(1)
        .max_concurrent_requests(64)
        .queue_timeout_secs(60)
        .build_unchecked();

    let app_context = create_test_context(config).await;
    let prefill_worker = BasicWorkerBuilder::new(prefill_url)
        .worker_type(WorkerType::Prefill {
            bootstrap_port: None,
        })
        .build();
    let decode_worker = BasicWorkerBuilder::new(decode_url)
        .worker_type(WorkerType::Decode)
        .build();

    app_context.worker_registry.register(Arc::from(prefill_worker));
    app_context.worker_registry.register(Arc::from(decode_worker));

    let router = Arc::from(RouterFactory::create_router(&app_context).await.unwrap());
    test_app::create_test_app_with_context(router, app_context)
}

#[tokio::test]
async fn regular_router_passes_through_engine_chat_path() {
    let upstream = UpstreamServer::start("/v1/engines/roma/chat/completions").await;
    let app = build_regular_app(&upstream.url).await;

    let payload = json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": false
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/engines/roma/chat/completions")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    upstream.shutdown().await;
}

#[tokio::test]
async fn regular_router_passes_through_engine_completion_path() {
    let upstream = UpstreamServer::start("/v1/engines/roma/completions").await;
    let app = build_regular_app(&upstream.url).await;

    let payload = json!({
        "model": "test-model",
        "prompt": "hello",
        "stream": false
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/engines/roma/completions")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    upstream.shutdown().await;
}

#[tokio::test]
async fn pd_router_passes_through_engine_chat_path() {
    let prefill = UpstreamServer::start("/v1/engines/roma/chat/completions").await;
    let decode = UpstreamServer::start("/v1/engines/roma/chat/completions").await;
    let app = build_pd_app(&prefill.url, &decode.url).await;

    let payload = json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": false
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/engines/roma/chat/completions")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    prefill.shutdown().await;
    decode.shutdown().await;
}

#[tokio::test]
async fn pd_router_passes_through_engine_completion_path() {
    let prefill = UpstreamServer::start("/v1/engines/roma/completions").await;
    let decode = UpstreamServer::start("/v1/engines/roma/completions").await;
    let app = build_pd_app(&prefill.url, &decode.url).await;

    let payload = json!({
        "model": "test-model",
        "prompt": "hello",
        "stream": false
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/engines/roma/completions")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    prefill.shutdown().await;
    decode.shutdown().await;
}
