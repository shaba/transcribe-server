# transcribe-server

OpenAI-compatible speech-to-text server written in Rust on top of the
[transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) library
(ggml runtime supporting multiple ASR model families such as GigaAM, Whisper
and Parakeet). It exposes the familiar `/v1/audio/transcriptions` API so
clients like Open WebUI can use it as a drop-in STT backend.

Status: early development (0.0.1)
