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

Status: early development (0.0.2)

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

The `engine-transcribe` feature pulls in the `transcribe-cpp` crate (safe
wrapper) and its `transcribe-cpp-sys` crate (FFI plus the native build).
Where the native transcribe.cpp library comes from is chosen by one of three
linking modes.

### Linking modes

#### Mode 1 — crates.io (default)

```sh
cargo build --release --features engine-transcribe
```

`transcribe-cpp-sys` from crates.io compiles the vendored transcribe.cpp C++
tree via CMake. You additionally need `cmake` and a C++ toolchain
(`g++`/`clang++`); the first build compiles ggml and is slow. Extra CMake
options can be forwarded through the `TRANSCRIBE_CMAKE_ARGS` environment
variable as an escape hatch.

CUDA build (GPU inference):

```sh
cargo build --release --features engine-transcribe,transcribe-cpp/cuda
```

This requires the CUDA toolkit at build time. At runtime the backend is
auto-selected; `--no-gpu` forces CPU.

#### Mode 2 — patched/bundled checkout (`[patch.crates-io]`)

The same source build as mode 1, but the crates come from a git or local
checkout of transcribe.cpp instead of crates.io. Use this to pick up
unreleased upstream fixes — for example `TRANSCRIBE_DIR` support (mode 3)
until it lands in a crates.io release. Uncomment and adapt ONE of the two
variants from the example block at the bottom of `Cargo.toml` (uncommenting
both is invalid TOML — duplicate keys):

```toml
[patch.crates-io]
transcribe-cpp = { git = "https://github.com/handy-computer/transcribe.cpp", rev = "<commit>" }
transcribe-cpp-sys = { git = "https://github.com/handy-computer/transcribe.cpp", rev = "<commit>" }
```

or point at a local checkout (mind the upstream layout: the
`transcribe-cpp-sys` manifest sits at the transcribe.cpp repo root, the safe
wrapper lives in `bindings/rust/transcribe-cpp`):

```toml
[patch.crates-io]
transcribe-cpp = { path = "/path/to/transcribe.cpp/bindings/rust/transcribe-cpp" }
transcribe-cpp-sys = { path = "/path/to/transcribe.cpp" }
```

A patch only applies when the patched version satisfies the `[dependencies]`
requirement. The current upstream checkout is 0.2.0 while this project
requires `^0.1.3`, so also bump the requirement to
`transcribe-cpp = { version = "0.2.0", optional = true }` — otherwise cargo
warns `patch ... was not used in the crate graph` and silently keeps the
crates.io version. The server compiles against the 0.2.0 wrapper unmodified.

#### Mode 3 — system/prebuilt prefix (`TRANSCRIBE_DIR`)

Skip the vendored source build entirely and link an existing transcribe.cpp
install prefix (OPENSSL_DIR-style). Produce the prefix once from a
transcribe.cpp checkout:

```sh
cmake -B build -DTRANSCRIBE_INSTALL=ON -DTRANSCRIBE_BUILD_SHARED=ON
cmake --build build -j
cmake --install build --prefix /opt/transcribe
```

then build the server against it:

```sh
TRANSCRIBE_DIR=/opt/transcribe cargo build --release --features engine-transcribe
```

Notes:

- The prefix must contain the installed `lib*/transcribe-link.json` manifest;
  any `cmake --install` of a `TRANSCRIBE_INSTALL=ON` build provides it. The
  link line is reconstructed from the manifest, so static and shared prefixes
  both work.
- Build features (`transcribe-cpp/cuda`, `transcribe-cpp/vulkan`, ...) are
  inert in this mode: the prebuilt library already fixed its configuration
  (backends, static vs shared) and the manifest records it. Choose backends
  when configuring the prefix build instead (e.g. `-DGGML_CUDA=ON`).
- Shared prefixes (`TRANSCRIBE_BUILD_SHARED=ON`) need the prefix lib dir on
  the loader path at runtime, e.g. `LD_LIBRARY_PATH=/opt/transcribe/lib64`
  (or `lib`, depending on the platform). For systemd deployments uncomment
  the `Environment=LD_LIBRARY_PATH=...` line in
  `packaging/transcribe-server@.service`. Static prefixes (the upstream
  default) need nothing at runtime.
- `TRANSCRIBE_DIR` support is not in the crates.io release yet
  (`transcribe-cpp-sys` 0.1.3 predates it): until upstream releases it, mode
  3 additionally requires the mode 2 patch pointing at a checkout that
  contains it (upstream branch `rust-sys-system-prefix`).

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
| `--max-upload-mb` | `TRANSCRIBE_MAX_UPLOAD_MB` | `256` | Max upload size in megabytes ([why 256](#upload-size-and-memory)) |
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
| `response_format` | no | `json` (default), `verbose_json` or `text` |

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

#### `response_format=verbose_json`

Adds the timestamps the model produced, in the OpenAI `verbose_json` shape:

```json
{
  "task": "transcribe",
  "language": "ru",
  "duration": 41.28,
  "text": "...",
  "segments": [
    {"id": 0, "start": 0.0, "end": 4.32, "text": "..."},
    {"id": 1, "start": 4.32, "end": 9.8, "text": "..."}
  ],
  "words": [
    {"word": "...", "start": 0.04, "end": 0.36}
  ]
}
```

- Times are seconds from the start of the **file**, not of the chunk the
  segment happened to fall into: long audio is transcribed chunk by chunk and
  each chunk's times are shifted back by that chunk's offset.
- `duration` is the length of the decoded audio, which can differ slightly
  from the duration the container advertises.
- `segments` is always present. Models that report no segment rows of their
  own get one segment per chunk instead, spanning that chunk.
- `words` is present only when the model produced word rows. Which
  granularity a model produces is family-specific: GigaAM aligns down to
  40 ms, whisper models report segments only. The server asks for the finest
  granularity the loaded model advertises and passes on whatever comes back.
- Per-token confidence is available from the library but is not exposed:
  the OpenAI shape has no field for it.

```sh
curl -s -H "Authorization: Bearer secret" \
  -F file=@meeting.wav -F response_format=verbose_json \
  http://127.0.0.1:8010/v1/audio/transcriptions
```

#### Upload size and memory

`--max-upload-mb` defaults to 256. The sizing case is a full meeting or
lecture recording: one hour of 16 kHz mono 16-bit WAV — what a conference
recorder such as BigBlueButton hands over — is about 110 MB, so 256 MB covers
roughly two hours of the bulkiest input the server accepts. The same length in
a compressed container (opus, mp3, m4a) is an order of magnitude smaller, so
the limit only ever binds on raw WAV.

The limit is also the memory guard, because the request body is buffered
whole: the multipart field is collected into one buffer, and everything except
the 16 kHz mono WAV fast path is then written to a tempfile for libav. Peak
resident memory per in-flight upload is therefore roughly the body size plus
the decoded PCM (16 kHz mono f32 = 3.8 MB per minute, ~230 MB per hour). Raise
the limit only together with the RAM to back it; the alternative — streaming
the upload straight to a tempfile so the body never has to fit in memory — is
a TODO, not something the limit currently assumes.

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
The unit is a systemd template (`transcribe-server@.service`): every instance
`<name>` is configured by `/etc/transcribe/<name>.env`, so several models can
run side by side on different ports. The env files are not shipped by the
packages — the administrator creates them, starting from the commented
`packaging/gigaam.env.example` (the ALT package installs it as documentation
under `/usr/share/doc/transcribe-server`).

The unit is hardened: it runs as a dynamic unprivileged user
(`DynamicUser=yes`, `NoNewPrivileges=true`, `ProtectSystem=strict`), so model
files under `/var/lib/transcribe/models` must be readable by that user
(world-readable models are fine).

API keys are optional. `/etc/transcribe/<name>.api-keys` (one key per line)
stays root-owned `0640` and systemd hands the service a copy via
`LoadCredential=`; the unit also carries `SetCredential=api-keys:\n`, which
systemd uses as the fallback when that file does not exist. So an instance
without a key file starts fine and serves anonymously — the default `127.0.0.1`
bind. Adding or removing the file only needs a restart of the instance.

The fallback value is a single newline rather than nothing on purpose: systemd
ignores a `SetCredential=` with an empty value outright, and the instance then
fails to start with `status=243/CREDENTIALS` when the key file is absent. A
newline is skipped by the key parser, so it configures no key.

```sh
# 1. Build and install the binary
cargo build --release --features engine-transcribe
install -m 755 target/release/transcribe-server /usr/bin/transcribe-server

# 2. Unit
install -m 644 packaging/transcribe-server@.service /etc/systemd/system/
systemctl daemon-reload

# 3. Per-instance config: environment file
mkdir -p /etc/transcribe /var/lib/transcribe/models
install -m 640 packaging/gigaam.env.example /etc/transcribe/gigaam.env
# edit /etc/transcribe/gigaam.env (model path, host, port)

# 4. Optional: require API keys for this instance (skip for anonymous access)
install -m 640 /dev/null /etc/transcribe/gigaam.api-keys
# put one API key per line into /etc/transcribe/gigaam.api-keys

# 5. Start
systemctl enable --now transcribe-server@gigaam
curl -s http://127.0.0.1:8010/health
```

A second instance is just another env file with a different
`TRANSCRIBE_PORT`:

```sh
# /etc/transcribe/whisper.env -> TRANSCRIBE_PORT=8011, Whisper model path
systemctl enable --now transcribe-server@whisper
```

## License

MIT — see [LICENSE](LICENSE).
