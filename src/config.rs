use std::io::BufRead;
use std::path::PathBuf;

/// Command-line configuration, whisper.cpp/llama.cpp flag style.
#[derive(clap::Parser, Debug, Clone)]
#[command(name = "transcribe-server", version, about)]
pub struct Config {
    /// Address to bind the HTTP server to
    #[arg(long, default_value = "127.0.0.1", env = "TRANSCRIBE_HOST")]
    pub host: String,

    /// Port to listen on
    #[arg(long, default_value_t = 8010, env = "TRANSCRIBE_PORT")]
    pub port: u16,

    /// Model to serve: "alias=path" or "path" (repeatable)
    #[arg(short = 'm', long, env = "TRANSCRIBE_MODEL")]
    pub model: Vec<String>,

    /// Number of CPU threads for inference
    #[arg(short = 't', long, env = "TRANSCRIBE_THREADS")]
    pub threads: Option<usize>,

    /// Max number of transcriptions processed in parallel
    #[arg(long, default_value_t = 1, env = "TRANSCRIBE_PARALLEL")]
    pub parallel: usize,

    /// API key required in the Authorization header (repeatable)
    #[arg(long, env = "TRANSCRIBE_API_KEY")]
    pub api_key: Vec<String>,

    /// File with API keys, one per line
    #[arg(long, env = "TRANSCRIBE_API_KEY_FILE")]
    pub api_key_file: Option<PathBuf>,

    /// Default transcription language (default: auto-detect)
    #[arg(long, env = "TRANSCRIBE_LANGUAGE")]
    pub language: Option<String>,

    /// Max chunk length in seconds for long-form audio
    #[arg(long, default_value_t = 25.0, env = "TRANSCRIBE_CHUNK_MAX_SEC")]
    pub chunk_max_sec: f32,

    /// Energy VAD threshold for chunk splitting
    #[arg(long, default_value_t = 0.01, env = "TRANSCRIBE_VAD_THRESHOLD")]
    pub vad_threshold: f32,

    /// Max upload size in megabytes
    #[arg(long, default_value_t = 64, env = "TRANSCRIBE_MAX_UPLOAD_MB")]
    pub max_upload_mb: usize,

    /// Disable GPU inference
    #[arg(long, env = "TRANSCRIBE_NO_GPU")]
    pub no_gpu: bool,

    /// Inference engine to use
    // Hidden alternative value "fake" (test engine) is accepted but not advertised.
    #[arg(long, default_value = "transcribe", env = "TRANSCRIBE_ENGINE")]
    pub engine: String,

    /// Verbose logging
    #[arg(short = 'v', long, env = "TRANSCRIBE_VERBOSE")]
    pub verbose: bool,
}

/// A model to load: user-facing alias plus path to the model file.
#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub alias: String,
    pub path: PathBuf,
}

impl Config {
    /// Parse `--model` values: "alias=path" keeps the alias,
    /// a bare "path" uses the file stem as the alias.
    pub fn model_specs(&self) -> Vec<ModelSpec> {
        self.model
            .iter()
            .map(|m| match m.split_once('=') {
                Some((alias, path)) => ModelSpec {
                    alias: alias.to_string(),
                    path: PathBuf::from(path),
                },
                None => {
                    let path = PathBuf::from(m);
                    let alias = path
                        .file_stem()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_else(|| m.clone());
                    ModelSpec { alias, path }
                }
            })
            .collect()
    }

    /// All API keys: `--api-key` flags plus lines from `--api-key-file`,
    /// trimmed, empty lines skipped.
    pub fn all_api_keys(&self) -> std::io::Result<Vec<String>> {
        let mut keys = self.api_key.clone();
        if let Some(path) = &self.api_key_file {
            let file = std::fs::File::open(path)?;
            for line in std::io::BufReader::new(file).lines() {
                let line = line?;
                let key = line.trim();
                if !key.is_empty() {
                    keys.push(key.to_string());
                }
            }
        }
        Ok(keys)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use std::io::Write;

    #[test]
    fn defaults() {
        let cfg = Config::try_parse_from(["ts"]).unwrap();
        assert_eq!(cfg.host, "127.0.0.1");
        assert_eq!(cfg.port, 8010);
        assert_eq!(cfg.parallel, 1);
        assert_eq!(cfg.chunk_max_sec, 25.0);
        assert_eq!(cfg.vad_threshold, 0.01);
        assert_eq!(cfg.max_upload_mb, 64);
        assert_eq!(cfg.engine, "transcribe");
        assert!(cfg.model.is_empty());
        assert!(cfg.threads.is_none());
        assert!(cfg.api_key.is_empty());
        assert!(cfg.api_key_file.is_none());
        assert!(cfg.language.is_none());
        assert!(!cfg.no_gpu);
        assert!(!cfg.verbose);
    }

    #[test]
    fn model_specs_alias_and_stem() {
        let cfg =
            Config::try_parse_from(["ts", "-m", "ru=/x/g.gguf", "-m", "/y/whisper.gguf"]).unwrap();
        let specs = cfg.model_specs();
        let aliases: Vec<&str> = specs.iter().map(|s| s.alias.as_str()).collect();
        assert_eq!(aliases, ["ru", "whisper"]);
        assert_eq!(specs[0].path, std::path::PathBuf::from("/x/g.gguf"));
        assert_eq!(specs[1].path, std::path::PathBuf::from("/y/whisper.gguf"));
    }

    #[test]
    fn all_api_keys_merges_flags_and_file() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(file, "k2").unwrap();
        writeln!(file, "  k3  ").unwrap();
        writeln!(file).unwrap();
        let cfg = Config::try_parse_from([
            "ts",
            "--api-key",
            "k1",
            "--api-key-file",
            file.path().to_str().unwrap(),
        ])
        .unwrap();
        let keys = cfg.all_api_keys().unwrap();
        assert_eq!(keys, ["k1", "k2", "k3"]);
    }

    #[test]
    fn all_api_keys_without_file() {
        let cfg = Config::try_parse_from(["ts", "--api-key", "k1"]).unwrap();
        assert_eq!(cfg.all_api_keys().unwrap(), ["k1"]);
    }
}
