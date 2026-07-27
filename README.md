# transcribe-server

OpenAI-compatible speech-to-text server written in Rust on top of the
[transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) library
(ggml runtime supporting multiple ASR model families such as GigaAM, Whisper
and Parakeet). It exposes the familiar `/v1/audio/transcriptions` API so
clients like Open WebUI can use it as a drop-in STT backend.

Why: llama.cpp-style single-binary deployment for speech-to-text. No Python
stack, no subprocess calls — audio decoding happens in-process via system
libav, inference via transcribe.cpp/ggml (CPU or CUDA). One binary plus a
GGUF model file is the whole install.

Status: early development (0.0.1)

## Building

Prerequisites:

- Rust 1.93+ (`rust-version` in `Cargo.toml`)
- System libav development packages for audio decoding
  (feature `audio-ffmpeg`, on by default). On ALT Linux / Sisyphus:

  ```sh
  apt-get install libavformat-devel libavcodec-devel libavutil-devel libswresample-devel
  ```

  (Debian/Ubuntu: `libavformat-dev libavcodec-dev libavutil-dev
  libswresample-dev`; Fedora: `ffmpeg-free-devel` or RPM Fusion
  `ffmpeg-devel`.)

Default build (no real inference engine — serves the built-in fake engine,
useful for API integration testing only):

```sh
cargo build --release
```

Real engine build:

```sh
cargo build --release --features engine-transcribe
```

The `engine-transcribe` feature pulls in the `transcribe-cpp` crate, whose
`transcribe-cpp-sys` builds the bundled transcribe.cpp C++ library from
source via CMake. That means you additionally need `cmake` and a C++
toolchain (`g++`/`clang++`); the first build compiles ggml and is slow. There
is currently no system-library linking mode in the sys crate (0.1.x) — the
library is always built from the bundled sources. Extra CMake options can be
passed through the `TRANSCRIBE_CMAKE_ARGS` environment variable as an escape
hatch. Linking against a system/prebuilt libtranscribe is a possible future
improvement.

CUDA build (GPU inference):

```sh
cargo build --release --features engine-transcribe,transcribe-cpp/cuda
```

This requires the CUDA toolkit at build time. At runtime the backend is
auto-selected; `--no-gpu` forces CPU.

## Running

```sh
# Real engine: serve one model (alias derived from the file stem)
transcribe-server -m /var/lib/transcribe/models/gigaam-v3-e2e-rnnt-Q8_0.gguf --api-key secret

# Multiple models with explicit aliases; the first one is the default
transcribe-server -m ru=/models/gigaam-v3.gguf -m en=/models/whisper-large-v3.gguf

# Fake engine (no model needed, echoes input length) for integration testing
transcribe-server --engine fake
```

### CLI flags

Every flag can also be set through its environment variable (flag wins).

| Flag | Env | Default | Description |
| --- | --- | --- | --- |
| `--host` | `TRANSCRIBE_HOST` | `127.0.0.1` | Address to bind the HTTP server to |
| `--port` | `TRANSCRIBE_PORT` | `8010` | Port to listen on |
| `-m`, `--model` | `TRANSCRIBE_MODEL` | (none) | Model to serve: `alias=path` or `path` (repeatable; first is the default model) |
| `-t`, `--threads` | `TRANSCRIBE_THREADS` | library default | Number of CPU threads for inference |
| `--parallel` | `TRANSCRIBE_PARALLEL` | `1` | Max number of transcriptions processed in parallel |
| `--api-key` | `TRANSCRIBE_API_KEY` | (none) | API key required in the Authorization header (repeatable) |
| `--api-key-file` | `TRANSCRIBE_API_KEY_FILE` | (none) | File with API keys, one per line |
| `--language` | `TRANSCRIBE_LANGUAGE` | auto-detect | Default transcription language |
| `--chunk-max-sec` | `TRANSCRIBE_CHUNK_MAX_SEC` | `25` | Max chunk length in seconds for long-form audio |
| `--vad-threshold` | `TRANSCRIBE_VAD_THRESHOLD` | `0.01` | Energy VAD threshold for chunk splitting |
| `--max-upload-mb` | `TRANSCRIBE_MAX_UPLOAD_MB` | `64` | Max upload size in megabytes |
| `--no-gpu` | `TRANSCRIBE_NO_GPU` | off | Disable GPU inference |
| `--engine` | `TRANSCRIBE_ENGINE` | `transcribe` | Inference engine (`transcribe`, or `fake` for testing) |
| `-v`, `--verbose` | `TRANSCRIBE_VERBOSE` | off | Verbose (debug) logging |

Logging uses `tracing`; `RUST_LOG` overrides the level chosen by `-v`.

## API

### Authentication

If at least one API key is configured (`--api-key` / `--api-key-file`), all
`/v1/*` endpoints require `Authorization: Bearer <key>`; `/health` is always
open. With no keys configured, authentication is disabled entirely
(llama.cpp style).

Errors use the OpenAI JSON shape:

```json
{"error": {"message": "...", "type": "invalid_request_error"}}
```

### POST /v1/audio/transcriptions

OpenAI-compatible multipart transcription. Fields:

| Field | Required | Description |
| --- | --- | --- |
| `file` | yes | Audio file (WAV fast path; anything libav can decode: mp3, ogg, webm, m4a, ...) |
| `model` | no | Model alias; unknown or missing alias falls back to the default (first) model |
| `language` | no | Language hint (e.g. `ru`); default from `--language`, else auto-detect |
| `response_format` | no | `json` (default) or `text` |

Unknown extra fields (`temperature`, `prompt`, ...) are ignored, like the
OpenAI API does. Long audio is split into chunks (max `--chunk-max-sec`
seconds) at energy-VAD silence points and the chunk transcripts are joined.

```sh
curl -s -H "Authorization: Bearer secret" \
  -F model=ru -F file=@speech.ogg \
  http://127.0.0.1:8010/v1/audio/transcriptions
# {"text":"..."}

curl -s -H "Authorization: Bearer secret" \
  -F file=@speech.wav -F response_format=text \
  http://127.0.0.1:8010/v1/audio/transcriptions
# plain text transcript
```

### WS /v1/audio/stream

Streaming transcription over WebSocket. Control frames are JSON text:

```
client -> {"type":"start","model":"<alias>"?,"language":"ru"?}
client -> binary PCM16LE mono 16 kHz frames (any framing)
client -> {"type":"stop"}
server -> {"type":"partial","text":"..."}   per drained chunk
server -> {"type":"final","text":"..."}     on stop, then server closes
server -> {"type":"error","message":"..."}  on any error, then closes
```

Binary frames are only valid after `start` and must contain a whole number of
PCM16LE samples: a dangling odd byte is a protocol error. Partials are
emitted whenever a full chunk window (`--chunk-max-sec`) is buffered, cut at
a VAD-detected silence when there is one. The final text is every partial
text plus the remainder joined with a single space (empty chunk transcripts
are skipped).

### GET /v1/models

OpenAI-style model list:

```json
{"object":"list","data":[{"id":"ru","object":"model","owned_by":"transcribe-server"}]}
```

### GET /health

Unauthenticated liveness/info endpoint:

```json
{"status":"ok","backend":"cuda","models":["ru"]}
```

## Open WebUI

In Admin Panel -> Settings -> Audio -> Speech-to-Text:

- Speech-to-Text Engine: `openai`
- API Base URL: `http://host.containers.internal:8010/v1`
  (Open WebUI in a container reaching transcribe-server on the host; use the
  real host/port otherwise)
- API key: one of the configured keys (any non-empty value if auth is
  disabled)
- STT Model: a configured model alias (an unknown value falls back to the
  default model, so it can be left as-is)

Open WebUI sends standard OpenAI multipart requests (`file` + `model`), which
is exactly the contract above. If the server listens on `127.0.0.1` only, a
containerized Open WebUI needs the host gateway mapping
(`--add-host=host.containers.internal:host-gateway` for podman/docker).

## Deployment (systemd)

Host deployment in the same style as llama-server. Files in `packaging/`.
The unit is hardened: it runs as a dynamic unprivileged user
(`DynamicUser=yes`, `NoNewPrivileges=true`, `ProtectSystem=strict`), so model
files under `/var/lib/transcribe/models` must be readable by that user
(world-readable models are fine). The api-keys file stays root-owned `0640`:
systemd hands the service a copy via `LoadCredential`.

```sh
# 1. Build and install the binary
cargo build --release --features engine-transcribe
install -m 755 target/release/transcribe-server /usr/bin/transcribe-server

# 2. Config: environment file and API keys
mkdir -p /etc/transcribe /var/lib/transcribe/models
install -m 640 packaging/transcribe-server.env.example /etc/transcribe/transcribe-server.env
# edit /etc/transcribe/transcribe-server.env (model path, host, port)
touch /etc/transcribe/api-keys && chmod 640 /etc/transcribe/api-keys
# put one API key per line into /etc/transcribe/api-keys

# 3. Unit
install -m 644 packaging/transcribe-server.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now transcribe-server
curl -s http://127.0.0.1:8010/health
```

## License

MIT — see [LICENSE](LICENSE).
