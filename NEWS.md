# NEWS

Notable changes per release. The release page on each forge is this file's
section for that tag, published by CI; see `tools/release-notes.sh`.

Written by hand: what matters in a release is usually the sentence a commit
list cannot produce -- which library version this one now requires, which
default changed under you. Commits carry the detail, this carries the point.

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
