# NEWS

Notable changes per release. The release page on each forge is this file's
section for that tag, published by CI; see `tools/release-notes.sh`.

Written by hand: what matters in a release is usually the sentence a commit
list cannot produce -- which library version this one now requires, which
default changed under you. Commits carry the detail, this carries the point.

## 0.0.5

Built on transcribe.cpp 0.5.1. This jumps from 0.2: the Rust bindings split
their plain-data types into `transcribe-cpp-types`, model roles grew beyond
ASR, and GigaAM learned to window long audio internally.

### Requires

- `transcribe-cpp` 0.5.1 (libtranscribe 0.5.x); `transcribe-cpp-types` 0.5.1
  comes with it.
- Rust 1.93 or newer, as before.
- Unchanged system audio dependencies: libavformat / libavcodec / libavutil /
  libswresample.
- Model weights are memory-mapped by default. Replace a model file by writing
  a new file and renaming it over the old one; `TRANSCRIBE_NO_MMAP=1` forces a
  copy.

### New

- WebSocket streaming has a native path. A model whose `/v1/models` entry says
  `supports_streaming` is fed PCM directly; its committed/tentative split is
  sent as `partial` and `tentative` frames. Models without a native stream
  (GigaAM, whisper) use a bounded commit window: `--stream-window-sec`
  defaults to 5 s and `--stream-hop-sec` to 1 s, and only the uncommitted tail
  is re-decoded.
- GigaAM long-form input. transcribe.cpp 0.5 reports no audio maximum and
  encodes past the 25 s training window in overlapping windows internally.
  The default remains the external 25 s chunker: measured 5-14% faster than
  one long-form run from 33.8 s to 60 min with the same WER. Raise
  `--chunk-max-sec` above the input length to opt into one run.
- `/v1/models` reports `supports_streaming` and `supports_long_form`.
- On-demand inference: `--worker-mode`, `--idle-timeout`,
  `--idle-timeout-loaded` and `--idle-timeout-unloaded` keep the port bound
  without holding a model in memory between requests.
- Opt-in `lazy-ffmpeg` build: the FFmpeg fallback decoder is a shared plugin
  loaded on the first request that needs it.
- CPU and CUDA Docker images with a non-root user and a health check.

### Changed

- A native engine `Busy` answer is a 503; an ASR-incapable model is refused at
  load with `UnsupportedRole` instead of starting and failing every request.
- Decoded PCM containing NaN or infinity is rejected as a 400 before it
  reaches the engine.
- Request bodies have a ten-minute deadline and concurrent uploads are
  bounded; WebSockets have a start handshake timeout, an idle read timeout
  and a 4 MiB frame cap.
- `--features lazy-ffmpeg` must be combined with `--no-default-features`; the
  two decoder modes cannot be enabled together.

### Fixed

- A model-load failure is no longer reported as a generic 500 or echoed with
  its internal path; the detail stays in the log.
- The decoder plugin is opened with `O_NOFOLLOW`, checked for ownership and
  mode, and loaded through the verified descriptor on Linux.

## 0.0.4

Built on transcribe.cpp 0.2, and requires it: 0.2 removed the 0.1 device
symbols (`transcribe_backend_device_count` and friends), so this release links
against libtranscribe 0.2.x and will not link against 0.1.x.

### Requires

- `transcribe-cpp` 0.2.3 (libtranscribe 0.2.x)
- Unchanged system audio dependencies: libavformat / libavcodec / libavutil /
  libswresample. FFmpeg 7.1 and 8.1 are both supported.

### New

- `POST /v1/audio/translations`, the second OpenAI speech-to-text route. Beyond
  the OpenAI shape, which always translates into English, it accepts
  `target_language` and a `language` source hint; a model that cannot
  translate, or cannot translate into the requested target, is refused before
  the upload is decoded.
- `pnc` / `itn` toggles for punctuation-capitalization and inverse text
  normalization: `--pnc`/`--itn` set the deployment default, request fields
  (and the equivalent keys in the WebSocket start frame) override it per
  request. Unset keeps each model family's shipped behavior, and a family with
  no runtime switch is transcribed with its own default rather than refused.
- `GET /v1/models` reports what the library says about each loaded model:
  architecture, languages, translation targets, and the longest audio one call
  takes. A property the model does not report is omitted rather than sent
  empty.
- Chunking is bounded by the model's own window: `--chunk-max-sec` is lowered
  to the loaded model's maximum when that is shorter. Past that window a model
  either refuses the chunk outright or transcribes it at a quality it was never
  trained for.
- A client that hangs up mid-request has its run aborted, and its engine slot
  is released only when the run actually stops -- the slot no longer goes to
  the next request while the abandoned inference is still holding the model.
- `--device` and `--list-devices` for exact compute-device selection, which
  0.2's opaque device handles finally make expressible. An out-of-range index,
  or one that contradicts `--no-gpu`, is refused rather than silently falling
  back to automatic selection. `--device` is a flag with no environment
  variable on purpose: registry indices are rebuilt per process.
- libtranscribe and ggml diagnostics go through `tracing` under the
  `transcribe_cpp` target, so `RUST_LOG=info,transcribe_cpp=warn` quiets the
  model-load chatter.

### Also

- Dependencies refreshed: ffmpeg-next 9.0 (its resampler flush now drains
  buffered tail samples that used to be dropped), clap, thiserror,
  futures-util.
- `TRANSCRIBE_DIR` linking (mode 3 in the README) is in the crates.io release
  now, so building against a system libtranscribe no longer needs a downstream
  patch.

## 0.0.3

- Licensed MIT, copyright ALT Linux Team.

## 0.0.2

- `response_format=verbose_json` returns the timestamps the model produced, in
  the OpenAI shape: segments always, words when the model aligns them. Times
  are relative to the file, not to the chunk they fell into.
- The default upload limit is 256 MB, which covers roughly two hours of the
  bulkiest input the server accepts (16 kHz mono 16-bit WAV).

## 0.0.1

- First release: OpenAI-compatible `/v1/audio/transcriptions`, `/v1/models`
  and a WebSocket streaming endpoint on top of transcribe.cpp, with optional
  Bearer authentication and in-process audio decoding through system libav.
