# AGENTS.md - transcribe-server

Practical reference for coding agents working in this repository.

## Project

OpenAI-compatible speech-to-text server in Rust on top of the
[transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) library
(ggml runtime, multiple ASR model families: GigaAM, Whisper, Parakeet, ...).
Primary consumer: Open WebUI (STT engine "openai", multipart).

Key invariants:
- The `SttEngine` trait (`src/engine/mod.rs`) isolates the transcribe-cpp
  crate. HTTP code never touches the crate directly.
- PCM contract everywhere: 16 kHz mono f32 in [-1, 1].
- The Open WebUI request contract (see README, "Open WebUI") must not break:
  `POST /v1/audio/transcriptions`, multipart fields `file`/`model`/`language`,
  no `response_format` -> JSON `{"text": "..."}` response.
- Audio decoding uses system libav via ffmpeg-next. No subprocess execs.

## Build and test

```sh
cargo build                          # default: FakeEngine + libav decode
cargo build --features engine-transcribe   # real engine (bundled build needs cmake, g++)
cargo test                           # unit + integration (no GPU/model needed)
cargo test -- --ignored              # real-model tests (need TS_TEST_MODEL=path.gguf)
```

Feature matrix: `audio-ffmpeg` (default; system libavformat/avcodec/avutil/swresample
devel packages required), `engine-transcribe` (off by default).

## Pre-commit gates (all must pass before EVERY commit)

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

Never suppress or skip a failing test to make a commit pass. Analyze first;
if the test is wrong, fix the test in the same commit and say so in the message.

## Code style

- rustfmt defaults; clippy clean at `-D warnings`.
- Code, comments, commit messages: English, ASCII only (no em-dash, arrows,
  unicode punctuation). README and docs are free-form.
- Comments are concise and explain non-obvious invariants, not what the code
  already says. Do not comment copied code that had no comment at the origin.
- Prefer reusing existing infrastructure over adding new components or deps.
  New dependencies: latest versions, added via `cargo add`, and only when
  clearly needed.

## Commits

- Conventional prefixes: `feat:`, `fix:`, `test:`, `chore:`, `refactor:`, `docs:`.
- Each commit is a logically complete unit (code + its tests together).
- Concise messages, no AI trailers (no Assisted-by / Co-authored-by).
- Do NOT `git push` or create PRs unless the user explicitly asks.

## Repository hygiene

Only code, tests, fixtures, packaging, README.md and this AGENTS.md belong in
the repo. Design notes, specs, plans, spike write-ups live OUTSIDE the repo
(maintainer keeps them in `~/notes/transcribe-server/`). Do not commit them.

## Layout

```
src/config.rs      clap CLI (whisper.cpp/llama.cpp flag style), ModelSpec
src/auth.rs        optional Bearer auth middleware (llama.cpp style)
src/server.rs      AppState { engine, cfg, semaphore }, build_router
src/api/           error (OpenAI-style JSON), health, models,
                   transcriptions (multipart), stream (WebSocket)
src/audio/         decode_to_pcm_16k: WAV fastpath (hound) + libav fallback
src/chunk/         energy VAD chunker for long-form audio (~25-30 s window)
src/engine/        SttEngine trait, FakeEngine, transcribe_cpp (feature-gated)
tests/             integration tests against FakeEngine; fixtures in tests/fixtures/
packaging/         systemd unit + env example (host deploy, like llama-server)
```

## Gotchas

- `audio-ffmpeg` needs system devel packages (ALT: `apt-get install
  libavformat-devel libavcodec-devel libavutil-devel libswresample-devel`).
- `engine-transcribe` bundled build compiles ggml: needs cmake and a C++
  toolchain; first build is slow.
- transcribe-cpp crate is 0.x: check the actual API before assuming
  signatures; keep all crate-specific code inside `src/engine/transcribe_cpp.rs`.
- WS protocol and CLI flags are documented in README; keep both in sync with
  the code in the same commit.
