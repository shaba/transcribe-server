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
    #[arg(
        long,
        default_value_t = 1,
        env = "TRANSCRIBE_PARALLEL",
        value_parser = at_least_one
    )]
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
    #[arg(
        long,
        default_value_t = 25.0,
        env = "TRANSCRIBE_CHUNK_MAX_SEC",
        value_parser = positive_f32
    )]
    pub chunk_max_sec: f32,

    /// Energy VAD threshold for chunk splitting
    #[arg(long, default_value_t = 0.01, env = "TRANSCRIBE_VAD_THRESHOLD")]
    pub vad_threshold: f32,

    /// Max upload size in megabytes
    // 256 MB is ~2 h of 16 kHz mono 16-bit WAV (an hour is ~110 MB), which is
    // the bulkiest input a meeting recorder hands over; compressed containers
    // of the same length are an order of magnitude smaller. The limit is also
    // the memory guard: a request body is buffered whole (see README).
    #[arg(long, default_value_t = 256, env = "TRANSCRIBE_MAX_UPLOAD_MB")]
    pub max_upload_mb: usize,

    /// Punctuation and capitalization: on|off (default: model family default)
    #[arg(long, env = "TRANSCRIBE_PNC", value_parser = toggle)]
    pub pnc: Option<bool>,

    /// Inverse text normalization ("twenty five" -> "25"): on|off
    #[arg(long, env = "TRANSCRIBE_ITN", value_parser = toggle)]
    pub itn: Option<bool>,

    /// Disable GPU inference
    #[arg(long, env = "TRANSCRIBE_NO_GPU")]
    pub no_gpu: bool,

    /// Select an exact compute device by registry index (see --list-devices).
    // Indices come from transcribe-cpp's process-local device registry, which
    // is only populated after backend registration; unset keeps today's
    // automatic backend policy (Backend::Auto, or Backend::Cpu with --no-gpu).
    //
    // Deliberately the one flag with no environment variable: the registry is
    // rebuilt per process and per backend build, so an index that persists in
    // an env file outlives its meaning and would silently bind a model to a
    // device nobody chose.
    #[arg(long)]
    pub device: Option<usize>,

    /// Print the enumerated compute devices and exit, without loading a model
    #[arg(long)]
    pub list_devices: bool,

    /// Inference engine to use
    // Hidden alternative value "fake" (test engine) is accepted but not advertised.
    #[arg(long, default_value = "transcribe", env = "TRANSCRIBE_ENGINE")]
    pub engine: String,

    /// Verbose logging
    #[arg(short = 'v', long, env = "TRANSCRIBE_VERBOSE")]
    pub verbose: bool,
}

fn at_least_one(s: &str) -> Result<usize, String> {
    let v: usize = s.parse().map_err(|_| format!("invalid number: {s}"))?;
    if v >= 1 {
        Ok(v)
    } else {
        Err("must be at least 1".to_string())
    }
}

/// The one spelling of a boolean knob in this server: the CLI value parser
/// and the equivalent request fields share it, so `--pnc off` and
/// `-F pnc=off` accept exactly the same words.
pub fn toggle(s: &str) -> Result<bool, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "on" | "true" | "1" | "yes" => Ok(true),
        "off" | "false" | "0" | "no" => Ok(false),
        other => Err(format!("expected on or off, got: {other}")),
    }
}

fn positive_f32(s: &str) -> Result<f32, String> {
    let v: f32 = s.parse().map_err(|_| format!("invalid number: {s}"))?;
    if v > 0.0 {
        Ok(v)
    } else {
        Err("must be greater than 0".to_string())
    }
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
        assert_eq!(cfg.max_upload_mb, 256);
        assert_eq!(cfg.engine, "transcribe");
        assert!(cfg.model.is_empty());
        assert!(cfg.threads.is_none());
        assert!(cfg.api_key.is_empty());
        assert!(cfg.api_key_file.is_none());
        assert!(cfg.language.is_none());
        assert!(!cfg.no_gpu);
        assert!(!cfg.verbose);
        assert!(cfg.pnc.is_none());
        assert!(cfg.itn.is_none());
        assert!(cfg.device.is_none());
        assert!(!cfg.list_devices);
    }

    #[test]
    fn toggles_accept_on_off_spellings_and_reject_the_rest() {
        for (text, expected) in [
            ("on", true),
            ("ON", true),
            ("true", true),
            ("1", true),
            ("yes", true),
            ("off", false),
            ("false", false),
            ("0", false),
            ("no", false),
        ] {
            assert_eq!(toggle(text), Ok(expected), "{text}");
        }
        assert!(toggle("maybe").is_err());
        assert!(toggle("").is_err());
    }

    #[test]
    fn pnc_and_itn_are_tri_state_on_the_command_line() {
        let cfg = Config::try_parse_from(["ts", "--pnc", "off", "--itn", "on"]).unwrap();
        assert_eq!(cfg.pnc, Some(false));
        assert_eq!(cfg.itn, Some(true));
        let err = Config::try_parse_from(["ts", "--pnc", "maybe"]).unwrap_err();
        assert!(err.to_string().contains("expected on or off"), "{err}");
    }

    /// Registry indices are process-local, so --device must stay a flag: an
    /// env var invites it into the env files the systemd unit reads.
    #[test]
    fn device_flag_parses_index_and_has_no_env_var() {
        let cfg = Config::try_parse_from(["ts", "--device", "1"]).unwrap();
        assert_eq!(cfg.device, Some(1));
        unsafe { std::env::set_var("TRANSCRIBE_DEVICE", "3") };
        let cfg = Config::try_parse_from(["ts"]).unwrap();
        unsafe { std::env::remove_var("TRANSCRIBE_DEVICE") };
        assert!(
            cfg.device.is_none(),
            "--device must not read the environment"
        );

        let cfg = Config::try_parse_from(["ts", "--list-devices"]).unwrap();
        assert!(cfg.list_devices);
        // --list-devices must not require -m: no model is fine at parse time.
        assert!(cfg.model.is_empty());
    }

    /// The limit is there to bound memory, but it must not reject the
    /// workload the server exists for. One hour of the bulkiest input we
    /// accept -- 16 kHz mono 16-bit WAV, what a conference recorder hands
    /// over -- is about 110 MiB, so anything at or below that is too small
    /// to be a sane default.
    #[test]
    fn default_upload_limit_covers_an_hour_of_raw_wav() {
        let hour_mib = 16_000 * 2 * 3600 / (1024 * 1024);
        let cfg = Config::try_parse_from(["ts"]).unwrap();
        assert!(
            cfg.max_upload_mb > hour_mib,
            "default limit {} MB does not fit an hour of 16 kHz mono WAV ({hour_mib} MB)",
            cfg.max_upload_mb
        );
    }

    #[test]
    fn parallel_zero_is_rejected() {
        let err = Config::try_parse_from(["ts", "--parallel", "0"]).unwrap_err();
        assert!(err.to_string().contains("must be at least 1"), "{err}");
        assert!(Config::try_parse_from(["ts", "--parallel", "1"]).is_ok());
    }

    #[test]
    fn chunk_max_sec_must_be_positive() {
        for bad in ["0", "-1.5", "nan"] {
            let arg = format!("--chunk-max-sec={bad}");
            let err = Config::try_parse_from(["ts", &arg]).unwrap_err();
            assert!(err.to_string().contains("must be greater than 0"), "{err}");
        }
        assert!(Config::try_parse_from(["ts", "--chunk-max-sec", "0.5"]).is_ok());
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
