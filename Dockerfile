FROM rust:1.93-bookworm AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake libclang-dev libavcodec-dev libavformat-dev libavutil-dev libswresample-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY decoder ./decoder
# Don't tune the distributed binary for the build host's instruction set.
RUN TRANSCRIBE_CMAKE_ARGS="-DTRANSCRIBE_X86_CONSERVATIVE=ON -DGGML_NATIVE=OFF" \
    cargo build --release --locked --features engine-transcribe --bin transcribe-server

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        libavcodec59 libavformat59 libavutil57 libswresample4 libstdc++6 curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/target/release/transcribe-server /usr/local/bin/
USER 65532:65532
ENV TRANSCRIBE_HOST=0.0.0.0
EXPOSE 8010
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s \
    CMD curl -fsS http://127.0.0.1:8010/health || exit 1
ENTRYPOINT ["/usr/local/bin/transcribe-server"]
