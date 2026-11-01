//! Transport primitives for the W&B 0.21 service protocol.
//!
//! W&B owns the service protocol schema. This module keeps the process and
//! framing boundary in Rust; callers encode/decode the protobuf payloads.

use std::io::{Read, Write};
use std::net::{Shutdown, SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail, ensure};
use fs2::FileExt;
use prost::Message;
use rand::distr::{Alphanumeric, SampleString};
use sha2::{Digest, Sha256};
use tempfile::{NamedTempFile, TempDir};

const FRAME_MAGIC: u8 = b'W';
const MAX_FRAME_BYTES: usize = 64 * 1024 * 1024;

/// Find the official `wandb-core` shipped with the user's W&B installation.
///
/// `WANDB_CORE_PATH` remains the explicit override. Otherwise, search the
/// active virtual environment, nearby project environments, and `PATH`, then
/// lazily install the pinned official W&B wheel into the Rust cache. This
/// avoids requiring a Python installation just to provide the service binary.
pub fn resolve_core_binary() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("WANDB_CORE_PATH") {
        let path = PathBuf::from(path);
        ensure!(
            path.is_file(),
            "WANDB_CORE_PATH does not name an existing file: {}",
            path.display()
        );
        return Ok(path);
    }

    let mut candidates = Vec::new();
    if let Some(venv) = std::env::var_os("VIRTUAL_ENV") {
        add_venv_candidates(&mut candidates, Path::new(&venv));
    }
    for start in [std::env::current_dir().ok(), std::env::current_exe().ok()]
        .into_iter()
        .flatten()
    {
        for ancestor in start.ancestors() {
            for name in [".venv", "venv"] {
                add_venv_candidates(&mut candidates, &ancestor.join(name));
            }
        }
    }
    for directory in std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default()) {
        for name in core_binary_names() {
            candidates.push(directory.join(name));
        }
    }

    candidates.dedup();
    if let Some(path) = candidates.into_iter().find(|path| path.is_file()) {
        return Ok(path);
    }
    install_core_binary()
}

fn install_core_binary() -> Result<PathBuf> {
    let (wheel_url, expected_sha256) = pinned_wheel().context(
        "automatic wandb-core download is not available for this operating system and architecture; install W&B 0.21.3 or set WANDB_CORE_PATH",
    )?;
    let cache = core_cache_dir()?.join("wandb-core").join("0.21.3");
    std::fs::create_dir_all(&cache).context("create wandb-core cache directory")?;
    let binary = cache.join(core_binary_names()[0]);
    let lock_path = cache.join("install.lock");
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&lock_path)
        .with_context(|| format!("open W&B cache lock {}", lock_path.display()))?;
    lock.lock_exclusive().context("lock wandb-core cache")?;
    if binary.is_file() {
        return Ok(binary);
    }

    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
        .context("create W&B wheel download client")?;
    let wheel = client
        .get(wheel_url)
        .send()
        .context("download official W&B 0.21.3 wheel")?
        .error_for_status()
        .context("download official W&B 0.21.3 wheel")?
        .bytes()
        .context("read official W&B 0.21.3 wheel")?;
    let actual_sha256 = format!("{:x}", Sha256::digest(&wheel));
    ensure!(
        actual_sha256 == expected_sha256,
        "official W&B wheel SHA-256 mismatch (expected {expected_sha256}, got {actual_sha256})"
    );

    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(wheel))
        .context("open official W&B wheel archive")?;
    let mut entry = archive
        .by_name("wandb/bin/wandb-core")
        .context("official W&B wheel does not contain wandb/bin/wandb-core")?;
    let mut staged = NamedTempFile::new_in(&cache).context("stage wandb-core executable")?;
    std::io::copy(&mut entry, &mut staged).context("extract wandb-core executable")?;
    ensure!(
        staged.as_file().metadata()?.len() > 0,
        "wandb-core executable is empty"
    );
    staged
        .flush()
        .context("flush cached wandb-core executable")?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(staged.path(), std::fs::Permissions::from_mode(0o755))
            .context("make cached wandb-core executable runnable")?;
    }
    staged
        .persist(&binary)
        .map_err(|error| error.error)
        .with_context(|| format!("publish cached wandb-core at {}", binary.display()))?;
    Ok(binary)
}

fn core_cache_dir() -> Result<PathBuf> {
    #[cfg(windows)]
    let root = std::env::var_os("LOCALAPPDATA")
        .or_else(|| {
            std::env::var_os("USERPROFILE")
                .map(|home| PathBuf::from(home).join("AppData/Local").into_os_string())
        })
        .context("LOCALAPPDATA or USERPROFILE is required to cache wandb-core")?;
    #[cfg(not(windows))]
    let root = std::env::var_os("XDG_CACHE_HOME")
        .or_else(|| {
            std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache").into_os_string())
        })
        .context("XDG_CACHE_HOME or HOME is required to cache wandb-core")?;
    Ok(PathBuf::from(root).join("rust-agent"))
}

fn pinned_wheel() -> Option<(&'static str, &'static str)> {
    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    return Some((
        "https://files.pythonhosted.org/packages/d8/2d/7ef56e25f78786e59fefd9b19867c325f9686317d9f7b93b5cb340360a3e/wandb-0.21.3-py3-none-win_amd64.whl",
        "56d5a5697766f552a9933d8c6a564202194768eb0389bd5f9fe9a99cd4cee41e",
    ));
    #[cfg(all(target_os = "windows", target_arch = "x86"))]
    return Some((
        "https://files.pythonhosted.org/packages/75/d3/faa6ddb792a158c154fb704b25c96d0478e71eabf96e3f17529fb23b6894/wandb-0.21.3-py3-none-win32.whl",
        "45aa3d8ad53c6ee06f37490d7a329ed7d0f5ca4dbd5d05bb0c01d5da22f14691",
    ));
    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    return Some((
        "https://files.pythonhosted.org/packages/aa/e8/b5bfbbc7f76c11fd0665b92be8a38c6a83b27f353552233b9959b21be488/wandb-0.21.3-py3-none-macosx_10_14_x86_64.whl",
        "f85bac45b4482742ec9ff190af38eb00a877ddeb4875475e7e487dc19300ff03",
    ));
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    return Some((
        "https://files.pythonhosted.org/packages/59/a3/03f0fcde49609df1cb3a382fb5053f601b88da448bcd415ed7f75272eee7/wandb-0.21.3-py3-none-macosx_12_0_arm64.whl",
        "8a2b3ba419b91d47edead2755f04cef54f9e3c4496ee0c9854c3cfeff4216dd3",
    ));
    #[cfg(all(target_os = "linux", target_arch = "x86_64", not(target_env = "musl")))]
    return Some((
        "https://files.pythonhosted.org/packages/5b/af/a3252e5afac98a036f83c65ec92cadf6677ccdaacbbb2151da29f694d136/wandb-0.21.3-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "2ff24b6b8e0f9da840b6bd5c7f60b0a5507bd998db40c9c2d476f9a340bec8ed",
    ));
    #[cfg(all(target_os = "linux", target_arch = "aarch64", not(target_env = "musl")))]
    return Some((
        "https://files.pythonhosted.org/packages/ea/7f/805c3d2fa9e3b8b6bf2bc534887c9ed97bdf22007ca8ba59424a1c8bb360/wandb-0.21.3-py3-none-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
        "2d9cf8588cb090a2a41f589037fda72c57c9e23edfbd2ad829e575f1305d942c",
    ));
    #[cfg(all(target_os = "linux", target_arch = "x86_64", target_env = "musl"))]
    return Some((
        "https://files.pythonhosted.org/packages/ff/32/9580f42899e54f3d0b4ea619b6f6a54980a4e36fd0675d58c09f0a08d3f6/wandb-0.21.3-py3-none-musllinux_1_2_x86_64.whl",
        "514a0aad40ecc0bdb757b1dc86e4ac98f61d2d760445b6e1f555291562320f2d",
    ));
    #[cfg(all(target_os = "linux", target_arch = "aarch64", target_env = "musl"))]
    return Some((
        "https://files.pythonhosted.org/packages/4d/f9/4404b5a24bfd4ba027c19d30152b0fc7ebca8c49b202dee6ecb7f316082c/wandb-0.21.3-py3-none-musllinux_1_2_aarch64.whl",
        "4975dec19e2b343e23ed6e60f7e1290120553719f82e87a22205bede758416ad",
    ));
    #[allow(unreachable_code)]
    None
}

fn add_venv_candidates(candidates: &mut Vec<PathBuf>, root: &Path) {
    for package_dir in [
        root.join("Lib/site-packages"),
        root.join("lib/site-packages"),
    ] {
        for name in core_binary_names() {
            candidates.push(package_dir.join("wandb/bin").join(name));
        }
    }
    if let Ok(entries) = std::fs::read_dir(root.join("lib")) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir()
                && path
                    .file_name()
                    .is_some_and(|name| name.to_string_lossy().starts_with("python3."))
            {
                for name in core_binary_names() {
                    candidates.push(path.join("site-packages/wandb/bin").join(name));
                }
            }
        }
    }
}

fn core_binary_names() -> &'static [&'static str] {
    #[cfg(windows)]
    {
        &["wandb-core", "wandb-core.exe"]
    }
    #[cfg(not(windows))]
    {
        &["wandb-core"]
    }
}

/// A running `wandb-core` process and its localhost service connection.
pub struct WandbCore {
    child: Child,
    stream: ServiceStream,
    _port_dir: TempDir,
    stderr_file: NamedTempFile,
}

enum ServiceStream {
    Tcp(TcpStream),
    #[cfg(unix)]
    Unix(std::os::unix::net::UnixStream),
}

impl Read for ServiceStream {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        match self {
            Self::Tcp(stream) => stream.read(buffer),
            #[cfg(unix)]
            Self::Unix(stream) => stream.read(buffer),
        }
    }
}

impl Write for ServiceStream {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        match self {
            Self::Tcp(stream) => stream.write(buffer),
            #[cfg(unix)]
            Self::Unix(stream) => stream.write(buffer),
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            Self::Tcp(stream) => stream.flush(),
            #[cfg(unix)]
            Self::Unix(stream) => stream.flush(),
        }
    }
}

impl ServiceStream {
    fn set_read_timeout(&self, timeout: Option<Duration>) -> std::io::Result<()> {
        match self {
            Self::Tcp(stream) => stream.set_read_timeout(timeout),
            #[cfg(unix)]
            Self::Unix(stream) => stream.set_read_timeout(timeout),
        }
    }

    fn set_write_timeout(&self, timeout: Option<Duration>) -> std::io::Result<()> {
        match self {
            Self::Tcp(stream) => stream.set_write_timeout(timeout),
            #[cfg(unix)]
            Self::Unix(stream) => stream.set_write_timeout(timeout),
        }
    }

    fn shutdown(&self) -> std::io::Result<()> {
        match self {
            Self::Tcp(stream) => stream.shutdown(Shutdown::Both),
            #[cfg(unix)]
            Self::Unix(stream) => stream.shutdown(Shutdown::Both),
        }
    }
}

enum ServiceEndpoint {
    Tcp(SocketAddr),
    #[cfg(unix)]
    Unix(PathBuf),
}

impl WandbCore {
    /// Start the official W&B service binary and connect to its local endpoint.
    ///
    /// On Windows, W&B requires `--listen-on-localhost` because Unix sockets
    /// are unavailable. On Unix, prefer its default Unix domain socket. The
    /// port file and child are kept alive together.
    pub fn launch(binary: impl AsRef<Path>, timeout: Duration) -> Result<Self> {
        let port_dir = TempDir::new().context("create W&B service port directory")?;
        let stderr_file = NamedTempFile::new().context("create W&B service stderr file")?;
        let port_path = port_dir.path().join("service.port");
        let mut command = Command::new(binary.as_ref());
        command
            .arg("--port-filename")
            .arg(&port_path)
            .arg("--pid")
            .arg(std::process::id().to_string())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::from(
                stderr_file
                    .reopen()
                    .context("open W&B service stderr file")?,
            ));
        #[cfg(windows)]
        command.arg("--listen-on-localhost");

        let mut child = command
            .spawn()
            .with_context(|| format!("start W&B service {}", binary.as_ref().display()))?;
        let deadline = Instant::now() + timeout;
        let connection = (|| -> Result<ServiceStream> {
            let endpoint = loop {
                if let Some(status) = child.try_wait().context("check W&B service process")? {
                    let details = std::fs::read_to_string(stderr_file.path())
                        .unwrap_or_default()
                        .trim()
                        .to_owned();
                    bail!(
                        "wandb-core {} exited before opening its service socket ({status}){}",
                        binary.as_ref().display(),
                        if details.is_empty() {
                            String::new()
                        } else {
                            format!(": {details}")
                        }
                    );
                }
                if let Some(endpoint) = read_service_endpoint(&port_path)? {
                    break endpoint;
                }
                if Instant::now() >= deadline {
                    bail!("timed out waiting for wandb-core service port");
                }
                thread::sleep(Duration::from_millis(50));
            };

            let stream = match endpoint {
                ServiceEndpoint::Tcp(endpoint) => {
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    ServiceStream::Tcp(
                        TcpStream::connect_timeout(
                            &endpoint,
                            remaining.max(Duration::from_millis(1)),
                        )
                        .with_context(|| format!("connect to wandb-core at {endpoint}"))?,
                    )
                }
                #[cfg(unix)]
                ServiceEndpoint::Unix(path) => ServiceStream::Unix(
                    std::os::unix::net::UnixStream::connect(&path)
                        .with_context(|| format!("connect to wandb-core at {}", path.display()))?,
                ),
            };
            stream
                .set_read_timeout(Some(timeout))
                .context("set W&B socket read timeout")?;
            stream
                .set_write_timeout(Some(timeout))
                .context("set W&B socket write timeout")?;
            Ok(stream)
        })();
        let stream = match connection {
            Ok(stream) => stream,
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(error);
            }
        };
        Ok(Self {
            child,
            stream,
            _port_dir: port_dir,
            stderr_file,
        })
    }

    /// Send one protobuf-encoded `ServerRequest` payload.
    pub fn send_frame(&mut self, payload: &[u8]) -> Result<()> {
        ensure!(payload.len() <= MAX_FRAME_BYTES, "W&B frame exceeds 64 MiB");
        let length = u32::try_from(payload.len()).context("W&B frame length exceeds u32")?;
        self.stream.write_all(&[FRAME_MAGIC])?;
        self.stream.write_all(&length.to_le_bytes())?;
        self.stream.write_all(payload)?;
        self.stream.flush().context("flush W&B service request")
    }

    /// Read one protobuf-encoded `ServerResponse` payload.
    pub fn read_frame(&mut self) -> Result<Vec<u8>> {
        let mut header = [0; 5];
        self.stream.read_exact(&mut header).with_context(|| {
            let details = std::fs::read_to_string(self.stderr_file.path())
                .unwrap_or_default()
                .trim()
                .to_owned();
            if details.is_empty() {
                "read W&B service frame header".to_owned()
            } else {
                format!("read W&B service frame header: {details}")
            }
        })?;
        ensure!(header[0] == FRAME_MAGIC, "invalid W&B service frame marker");
        let length = u32::from_le_bytes(header[1..5].try_into().unwrap()) as usize;
        ensure!(length <= MAX_FRAME_BYTES, "W&B frame exceeds 64 MiB");
        let mut payload = vec![0; length];
        self.stream
            .read_exact(&mut payload)
            .context("read W&B service frame payload")?;
        Ok(payload)
    }
}

impl Drop for WandbCore {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn read_service_endpoint(path: &Path) -> Result<Option<ServiceEndpoint>> {
    let contents = match std::fs::read_to_string(path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error).context("read W&B service port file"),
    };
    let lines: Vec<_> = contents.lines().collect();
    if lines.last().copied() != Some("EOF") {
        return Ok(None);
    }
    if let Some(port) = lines.iter().find_map(|line| line.strip_prefix("sock=")) {
        let port = port.parse::<u16>().context("parse W&B service TCP port")?;
        ensure!(port != 0, "W&B service reported port zero");
        return Ok(Some(ServiceEndpoint::Tcp(SocketAddr::from((
            [127, 0, 0, 1],
            port,
        )))));
    }
    #[cfg(unix)]
    if let Some(socket) = lines.iter().find_map(|line| line.strip_prefix("unix=")) {
        ensure!(
            !socket.is_empty(),
            "W&B service reported an empty Unix socket"
        );
        return Ok(Some(ServiceEndpoint::Unix(PathBuf::from(socket))));
    }
    bail!("W&B service port file has no supported socket endpoint")
}

/// A live W&B run backed by the official `wandb-core` service.
pub struct Run {
    core: WandbCore,
    id: String,
    finished: bool,
    resumed: bool,
    resume_file: Option<PathBuf>,
}

impl Run {
    /// Start a run and publish its initial run record.
    pub fn start(
        binary: impl AsRef<Path>,
        project: &str,
        name: &str,
        config: &serde_json::Value,
        timeout: Duration,
    ) -> Result<Self> {
        let root_dir = std::env::var_os("WANDB_DIR")
            .map(PathBuf::from)
            .unwrap_or(std::env::current_dir().context("get current directory")?);
        let root = if root_dir.join(".wandb").is_dir() {
            root_dir.join(".wandb")
        } else {
            root_dir.join("wandb")
        };
        std::fs::create_dir_all(&root)
            .with_context(|| format!("create W&B directory {}", root.display()))?;
        let resume = std::env::var("WANDB_RESUME").ok().filter(|s| !s.is_empty());
        if let Some(value) = &resume {
            ensure!(
                matches!(value.as_str(), "allow" | "must" | "never" | "auto"),
                "invalid WANDB_RESUME value {value:?}; expected allow, must, never, or auto"
            );
        }
        let resume_file = root.join("wandb-resume.json");
        let env_id = std::env::var("WANDB_RUN_ID").ok();
        let id = if let Some(id) = env_id {
            validate_run_id(&id)?;
            id
        } else if resume.as_deref() == Some("auto") {
            read_auto_resume_id(&resume_file)?.unwrap_or_else(generate_run_id)
        } else {
            generate_run_id()
        };
        if resume.as_deref() == Some("auto") {
            std::fs::write(&resume_file, serde_json::json!({"run_id": id}).to_string())
                .with_context(|| format!("write W&B auto-resume file {}", resume_file.display()))?;
        }
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let mode = std::env::var("WANDB_MODE").unwrap_or_else(|_| "online".to_owned());
        let run_mode = if matches!(mode.as_str(), "offline" | "dryrun") {
            "offline-run"
        } else {
            "run"
        };
        let sync_dir = root.join(format!("{run_mode}-{stamp}-{id}"));
        let files_dir = sync_dir.join("files");
        let log_dir = sync_dir.join("logs");
        std::fs::create_dir_all(&files_dir).context("create W&B run files directory")?;
        std::fs::create_dir_all(&log_dir).context("create W&B run logs directory")?;
        let settings = Settings {
            api_key: std::env::var("WANDB_API_KEY")
                .ok()
                .map(|api_key| StringValue::new(&api_key)),
            project: Some(StringValue::new(project)),
            entity: std::env::var("WANDB_ENTITY")
                .ok()
                .map(|entity| StringValue::new(&entity)),
            base_url: Some(StringValue::new(
                &std::env::var("WANDB_BASE_URL")
                    .ok()
                    .filter(|url| !url.is_empty())
                    .unwrap_or_else(|| "https://api.wandb.ai".to_owned()),
            )),
            run_id: Some(StringValue::new(&id)),
            resume: resume.as_deref().map(StringValue::new),
            run_name: Some(StringValue::new(name)),
            mode: Some(StringValue::new(&mode)),
            offline: matches!(mode.as_str(), "offline" | "dryrun")
                .then_some(BoolValue { value: true }),
            run_mode: Some(StringValue::new(run_mode)),
            wandb_dir: Some(StringValue::new(&root.to_string_lossy())),
            sync_dir: Some(StringValue::new(&sync_dir.to_string_lossy())),
            sync_file: Some(StringValue::new(
                &sync_dir.join(format!("run-{id}.wandb")).to_string_lossy(),
            )),
            files_dir: Some(StringValue::new(&files_dir.to_string_lossy())),
            log_dir: Some(StringValue::new(&log_dir.to_string_lossy())),
        };
        let mut core = WandbCore::launch(binary, timeout)?;
        let info = Some(RecordInfo {
            stream_id: id.clone(),
        });
        core.send_message(&ServerRequest {
            request_id: String::new(),
            server_request_type: Some(server_request::ServerRequestType::InformInit(
                ServerInformInitRequest {
                    settings: Some(settings.clone()),
                    info: info.clone(),
                },
            )),
        })?;
        let mut run_record = RunRecord {
            run_id: id.clone(),
            entity: std::env::var("WANDB_ENTITY").unwrap_or_default(),
            project: project.to_owned(),
            display_name: name.to_owned(),
            config: Some(ConfigRecord {
                update: run_config_items(config)?,
            }),
            starting_step: 0,
            resumed: false,
        };
        let initialized = core.send_record_and_wait(Record {
            record_type: Some(record::RecordType::Run(run_record.clone())),
            info: info.clone(),
            ..Default::default()
        })?;
        let initialized = initialized
            .run_result
            .context("W&B service omitted the run initialization result")?;
        if let Some(error) = initialized.error {
            bail!("W&B run initialization failed: {}", error.message);
        }
        let initialized_run = initialized
            .run
            .context("W&B service omitted the initialized run")?;
        ensure!(
            initialized_run.run_id == id,
            "W&B service initialized unexpected run id {}",
            initialized_run.run_id
        );
        let resumed = initialized_run.resumed;
        run_record.starting_step = initialized_run.starting_step;
        run_record.resumed = resumed;
        core.send_message(&ServerRequest {
            request_id: String::new(),
            server_request_type: Some(server_request::ServerRequestType::InformStart(
                ServerInformStartRequest {
                    settings: Some(settings),
                    info: info.clone(),
                },
            )),
        })?;
        let _started = core.send_record_and_wait(Record {
            record_type: Some(record::RecordType::Request(Request {
                request_type: Some(request::RequestType::RunStart(RunStartRequest {
                    run: Some(run_record),
                    info: Some(RequestInfo {
                        stream_id: id.clone(),
                    }),
                })),
            })),
            info,
            control: Some(Control::default()),
        })?;
        Ok(Self {
            core,
            id,
            finished: false,
            resumed,
            resume_file: (resume.as_deref() == Some("auto")).then_some(resume_file),
        })
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    /// Whether the server attached this session to an existing run.
    pub fn was_resumed(&self) -> bool {
        self.resumed
    }

    /// Append scalar or structured JSON metrics to the current history row.
    pub fn log(&mut self, values: &serde_json::Map<String, serde_json::Value>) -> Result<()> {
        let mut item = values
            .iter()
            .map(|(key, value)| {
                Ok(HistoryItem {
                    key: key.clone(),
                    value_json: serde_json::to_string(value)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if !values.contains_key("_timestamp") {
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64();
            item.push(HistoryItem {
                key: "_timestamp".to_owned(),
                value_json: serde_json::to_string(&timestamp)?,
            });
        }
        self.core.send_message(&ServerRequest {
            request_id: String::new(),
            server_request_type: Some(server_request::ServerRequestType::RecordPublish(Record {
                record_type: Some(record::RecordType::Request(Request {
                    request_type: Some(request::RequestType::PartialHistory(
                        PartialHistoryRequest { item },
                    )),
                })),
                info: Some(RecordInfo {
                    stream_id: self.id.clone(),
                }),
                ..Default::default()
            })),
        })
    }

    pub fn finish(&mut self) -> Result<()> {
        if self.finished {
            return Ok(());
        }
        self.finish_with_exit_code(0)
    }

    fn finish_with_exit_code(&mut self, exit_code: i32) -> Result<()> {
        if self.finished {
            return Ok(());
        }
        self.core.send_record_and_wait(Record {
            record_type: Some(record::RecordType::Exit(RunExitRecord {
                exit_code,
                info: Some(RecordInfo {
                    stream_id: self.id.clone(),
                }),
            })),
            info: Some(RecordInfo {
                stream_id: self.id.clone(),
            }),
            ..Default::default()
        })?;
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            let result = self.core.send_record_and_wait(Record {
                record_type: Some(record::RecordType::Request(Request {
                    request_type: Some(request::RequestType::PollExit(PollExitRequest {
                        info: Some(RequestInfo {
                            stream_id: self.id.clone(),
                        }),
                    })),
                })),
                info: Some(RecordInfo {
                    stream_id: self.id.clone(),
                }),
                ..Default::default()
            })?;
            let response = result
                .response
                .and_then(|response| response.response_type)
                .context("wandb-core poll-exit response omitted completion status")?;
            let response::ResponseType::PollExit(poll_exit) = response else {
                bail!("wandb-core returned an unexpected poll-exit response")
            };
            let done = poll_exit.done;
            if done {
                break;
            }
            if Instant::now() >= deadline {
                bail!("wandb-core did not finish syncing the run within 30 seconds");
            }
            thread::sleep(Duration::from_millis(100));
        }
        self.core.send_message(&ServerRequest {
            request_id: String::new(),
            server_request_type: Some(server_request::ServerRequestType::InformFinish(
                ServerInformFinishRequest {
                    info: Some(RecordInfo {
                        stream_id: self.id.clone(),
                    }),
                },
            )),
        })?;
        self.core
            .close_and_wait(Duration::from_secs(60), exit_code)?;
        self.finished = true;
        if exit_code == 0
            && let Some(path) = &self.resume_file
        {
            clear_auto_resume_id(path, &self.id)?;
        }
        Ok(())
    }
}

fn run_config_items(config: &serde_json::Value) -> Result<Vec<ConfigItem>> {
    let values = config.as_object();
    let wandb_config = values
        .and_then(|values| values.get("_wandb"))
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    let mut items = vec![ConfigItem {
        key: "_wandb".to_owned(),
        value_json: serde_json::to_string(&wandb_config)?,
    }];
    if let Some(values) = values {
        items.extend(
            values
                .iter()
                .filter(|(key, _)| key.as_str() != "_wandb")
                .map(|(key, value)| {
                    Ok(ConfigItem {
                        key: key.clone(),
                        value_json: serde_json::to_string(value)?,
                    })
                })
                .collect::<Result<Vec<_>>>()?,
        );
    }
    Ok(items)
}

fn generate_run_id() -> String {
    Alphanumeric
        .sample_string(&mut rand::rng(), 8)
        .to_ascii_lowercase()
}

fn validate_run_id(id: &str) -> Result<()> {
    ensure!(!id.is_empty(), "WANDB_RUN_ID cannot be empty");
    ensure!(
        id.trim() == id,
        "WANDB_RUN_ID cannot start or end with whitespace"
    );
    ensure!(
        !id.chars().any(|c| ":;,#?/'".contains(c)),
        "WANDB_RUN_ID contains a reserved character"
    );
    Ok(())
}

fn read_auto_resume_id(path: &Path) -> Result<Option<String>> {
    let contents = match std::fs::read_to_string(path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error).with_context(|| format!("read {}", path.display())),
    };
    let parsed: serde_json::Value = match serde_json::from_str(&contents) {
        Ok(value) => value,
        Err(_) => return Ok(None),
    };
    let Some(id) = parsed.get("run_id").and_then(serde_json::Value::as_str) else {
        return Ok(None);
    };
    validate_run_id(id)?;
    Ok(Some(id.to_owned()))
}

fn clear_auto_resume_id(path: &Path, run_id: &str) -> Result<()> {
    if read_auto_resume_id(path)?.as_deref() == Some(run_id) {
        std::fs::remove_file(path)
            .with_context(|| format!("remove W&B auto-resume file {}", path.display()))?;
    }
    Ok(())
}

impl Drop for Run {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.finish_with_exit_code(1);
        }
    }
}

impl WandbCore {
    fn send_message<M: Message>(&mut self, message: &M) -> Result<()> {
        self.send_frame(&message.encode_to_vec())
    }

    fn send_record_and_wait(&mut self, mut record: Record) -> Result<ResultMessage> {
        let mailbox = Alphanumeric.sample_string(&mut rand::rng(), 16);
        record.control = Some(Control {
            req_resp: true,
            mailbox_slot: mailbox.clone(),
            ..record.control.unwrap_or_default()
        });
        self.send_message(&ServerRequest {
            request_id: mailbox,
            server_request_type: Some(server_request::ServerRequestType::RecordPublish(record)),
        })?;
        let response = ServerResponse::decode(self.read_frame()?.as_slice())
            .context("decode W&B service response")?;
        match response.server_response_type {
            Some(server_response::ServerResponseType::ResultCommunicate(result)) => Ok(result),
            _ => bail!("wandb-core returned an unexpected response to a communicated record"),
        }
    }

    fn close_and_wait(&mut self, timeout: Duration, exit_code: i32) -> Result<()> {
        self.send_message(&ServerRequest {
            request_id: String::new(),
            server_request_type: Some(server_request::ServerRequestType::InformTeardown(
                ServerInformTeardownRequest { exit_code },
            )),
        })?;
        let _ = self.stream.shutdown();
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(status) = self.child.try_wait().context("wait for wandb-core")? {
                ensure!(status.success(), "wandb-core exited with {status}");
                return Ok(());
            }
            if Instant::now() >= deadline {
                self.child.kill().context("stop idle wandb-core service")?;
                let _ = self.child.wait().context("reap wandb-core service")?;
                bail!("timed out waiting for wandb-core to finish and exit");
            }
            thread::sleep(Duration::from_millis(50));
        }
    }
}

#[derive(Clone, PartialEq, Message)]
struct StringValue {
    #[prost(string, tag = "1")]
    value: String,
}

#[derive(Clone, PartialEq, Message)]
struct BoolValue {
    #[prost(bool, tag = "1")]
    value: bool,
}

impl StringValue {
    fn new(value: &str) -> Self {
        Self {
            value: value.to_owned(),
        }
    }
}

#[derive(Clone, PartialEq, Message)]
struct Settings {
    #[prost(message, optional, tag = "55")]
    api_key: Option<StringValue>,
    #[prost(message, optional, tag = "97")]
    project: Option<StringValue>,
    #[prost(message, optional, tag = "69")]
    entity: Option<StringValue>,
    #[prost(message, optional, tag = "57")]
    base_url: Option<StringValue>,
    #[prost(message, optional, tag = "107")]
    run_id: Option<StringValue>,
    #[prost(message, optional, tag = "102")]
    resume: Option<StringValue>,
    #[prost(message, optional, tag = "110")]
    run_name: Option<StringValue>,
    #[prost(message, optional, tag = "92")]
    mode: Option<StringValue>,
    #[prost(message, optional, tag = "30")]
    offline: Option<BoolValue>,
    #[prost(message, optional, tag = "109")]
    run_mode: Option<StringValue>,
    #[prost(message, optional, tag = "142")]
    wandb_dir: Option<StringValue>,
    #[prost(message, optional, tag = "133")]
    sync_dir: Option<StringValue>,
    #[prost(message, optional, tag = "134")]
    sync_file: Option<StringValue>,
    #[prost(message, optional, tag = "70")]
    files_dir: Option<StringValue>,
    #[prost(message, optional, tag = "85")]
    log_dir: Option<StringValue>,
}

#[derive(Clone, PartialEq, Message)]
struct RecordInfo {
    #[prost(string, tag = "1")]
    stream_id: String,
}

#[derive(Clone, PartialEq, Message)]
struct RequestInfo {
    #[prost(string, tag = "1")]
    stream_id: String,
}

#[derive(Clone, PartialEq, Message)]
struct ServerInformInitRequest {
    #[prost(message, optional, tag = "1")]
    settings: Option<Settings>,
    #[prost(message, optional, tag = "200")]
    info: Option<RecordInfo>,
}

#[derive(Clone, PartialEq, Message)]
struct ServerInformFinishRequest {
    #[prost(message, optional, tag = "200")]
    info: Option<RecordInfo>,
}

#[derive(Clone, PartialEq, Message)]
struct ServerInformStartRequest {
    #[prost(message, optional, tag = "1")]
    settings: Option<Settings>,
    #[prost(message, optional, tag = "200")]
    info: Option<RecordInfo>,
}

#[derive(Clone, PartialEq, Message)]
struct ServerInformTeardownRequest {
    #[prost(int32, tag = "1")]
    exit_code: i32,
}

#[derive(Clone, PartialEq, Message)]
struct ServerRequest {
    #[prost(string, tag = "10")]
    request_id: String,
    #[prost(oneof = "server_request::ServerRequestType", tags = "1, 2, 3, 4, 7, 8")]
    server_request_type: Option<server_request::ServerRequestType>,
}

mod server_request {
    use super::{
        Record, ServerInformFinishRequest, ServerInformInitRequest, ServerInformStartRequest,
        ServerInformTeardownRequest,
    };
    use prost::Oneof;

    #[derive(Clone, PartialEq, Oneof)]
    pub enum ServerRequestType {
        #[prost(message, tag = "1")]
        RecordPublish(Record),
        #[prost(message, tag = "2")]
        RecordCommunicate(Record),
        #[prost(message, tag = "3")]
        InformInit(ServerInformInitRequest),
        #[prost(message, tag = "4")]
        InformFinish(ServerInformFinishRequest),
        #[prost(message, tag = "7")]
        InformTeardown(ServerInformTeardownRequest),
        #[prost(message, tag = "8")]
        InformStart(ServerInformStartRequest),
    }
}

#[derive(Clone, PartialEq, Message)]
struct Record {
    #[prost(oneof = "record::RecordType", tags = "17, 18, 100")]
    record_type: Option<record::RecordType>,
    #[prost(message, optional, tag = "200")]
    info: Option<RecordInfo>,
    #[prost(message, optional, tag = "16")]
    control: Option<Control>,
}

mod record {
    use super::{Request, RunExitRecord, RunRecord};
    use prost::Oneof;

    #[derive(Clone, PartialEq, Oneof)]
    pub enum RecordType {
        #[prost(message, tag = "17")]
        Run(RunRecord),
        #[prost(message, tag = "18")]
        Exit(RunExitRecord),
        #[prost(message, tag = "100")]
        Request(Request),
    }
}

#[derive(Clone, PartialEq, Message)]
struct RunRecord {
    #[prost(string, tag = "1")]
    run_id: String,
    #[prost(string, tag = "2")]
    entity: String,
    #[prost(string, tag = "3")]
    project: String,
    #[prost(string, tag = "8")]
    display_name: String,
    #[prost(message, optional, tag = "4")]
    config: Option<ConfigRecord>,
    #[prost(int64, tag = "14")]
    starting_step: i64,
    #[prost(bool, tag = "18")]
    resumed: bool,
}

#[derive(Clone, PartialEq, Message)]
struct ConfigRecord {
    #[prost(message, repeated, tag = "1")]
    update: Vec<ConfigItem>,
}

#[derive(Clone, PartialEq, Message)]
struct ConfigItem {
    #[prost(string, tag = "1")]
    key: String,
    #[prost(string, tag = "16")]
    value_json: String,
}

#[derive(Clone, PartialEq, Message)]
struct RunExitRecord {
    #[prost(int32, tag = "1")]
    exit_code: i32,
    #[prost(message, optional, tag = "200")]
    info: Option<RecordInfo>,
}

#[derive(Clone, PartialEq, Message)]
struct RunUpdateResult {
    #[prost(message, optional, tag = "1")]
    run: Option<RunRecord>,
    #[prost(message, optional, tag = "2")]
    error: Option<ErrorInfo>,
}

#[derive(Clone, PartialEq, Message)]
struct ErrorInfo {
    #[prost(string, tag = "1")]
    message: String,
    #[prost(int32, tag = "2")]
    code: i32,
}

#[derive(Clone, PartialEq, Message)]
struct Request {
    #[prost(oneof = "request::RequestType", tags = "8, 10, 11")]
    request_type: Option<request::RequestType>,
}

mod request {
    use super::{PartialHistoryRequest, PollExitRequest, RunStartRequest};
    use prost::Oneof;

    #[derive(Clone, PartialEq, Oneof)]
    pub enum RequestType {
        #[prost(message, tag = "8")]
        PollExit(PollExitRequest),
        #[prost(message, tag = "10")]
        PartialHistory(PartialHistoryRequest),
        #[prost(message, tag = "11")]
        RunStart(RunStartRequest),
    }
}

#[derive(Clone, PartialEq, Message)]
struct RunStartRequest {
    #[prost(message, optional, tag = "1")]
    run: Option<RunRecord>,
    #[prost(message, optional, tag = "200")]
    info: Option<RequestInfo>,
}

#[derive(Clone, PartialEq, Message)]
struct PartialHistoryRequest {
    #[prost(message, repeated, tag = "1")]
    item: Vec<HistoryItem>,
}

#[derive(Clone, PartialEq, Message)]
struct PollExitRequest {
    #[prost(message, optional, tag = "200")]
    info: Option<RequestInfo>,
}

#[derive(Clone, PartialEq, Message)]
struct PollExitResponse {
    #[prost(bool, tag = "1")]
    done: bool,
}

#[derive(Clone, PartialEq, Message)]
struct Response {
    #[prost(oneof = "response::ResponseType", tags = "26, 28")]
    response_type: Option<response::ResponseType>,
}

mod response {
    use super::{PollExitResponse, RunStartResponse};
    use prost::Oneof;

    #[derive(Clone, PartialEq, Oneof)]
    pub enum ResponseType {
        #[prost(message, tag = "26")]
        PollExit(PollExitResponse),
        #[prost(message, tag = "28")]
        RunStart(RunStartResponse),
    }
}

#[derive(Clone, PartialEq, Message)]
struct RunStartResponse {}

#[derive(Clone, PartialEq, Message)]
struct HistoryItem {
    #[prost(string, tag = "1")]
    key: String,
    #[prost(string, tag = "16")]
    value_json: String,
}

#[derive(Clone, PartialEq, Message)]
struct Control {
    #[prost(bool, tag = "1")]
    req_resp: bool,
    #[prost(bool, tag = "2")]
    local: bool,
    #[prost(string, tag = "4")]
    mailbox_slot: String,
}

#[derive(Clone, PartialEq, Message)]
struct ServerResponse {
    #[prost(oneof = "server_response::ServerResponseType", tags = "2")]
    server_response_type: Option<server_response::ServerResponseType>,
}

#[derive(Clone, PartialEq, Message)]
struct ResultMessage {
    #[prost(message, optional, tag = "17")]
    run_result: Option<RunUpdateResult>,
    #[prost(message, optional, tag = "18")]
    exit_result: Option<RunExitResult>,
    #[prost(message, optional, tag = "100")]
    response: Option<Response>,
}

#[derive(Clone, PartialEq, Message)]
struct RunExitResult {}

mod server_response {
    use super::ResultMessage;
    use prost::Oneof;

    #[derive(Clone, PartialEq, Oneof)]
    pub enum ServerResponseType {
        #[prost(message, tag = "2")]
        ResultCommunicate(ResultMessage),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ServiceEndpoint, clear_auto_resume_id, read_auto_resume_id, read_service_endpoint,
        run_config_items, validate_run_id,
    };
    use tempfile::tempdir;

    /// Exercises authenticated initialization, config and metric sync, and
    /// resuming the same server-side run. Run manually with a real API key:
    /// `cargo test live_config_logging_and_server_resume -- --ignored`.
    #[test]
    #[ignore = "requires W&B credentials and RUST_AGENT_WANDB_TEST_PROJECT"]
    fn live_config_logging_and_server_resume() {
        let project = std::env::var("RUST_AGENT_WANDB_TEST_PROJECT")
            .expect("set RUST_AGENT_WANDB_TEST_PROJECT to a writable W&B project");
        let run_id = std::env::var("WANDB_RUN_ID")
            .expect("set WANDB_RUN_ID to a unique run id for this test");
        assert_eq!(std::env::var("WANDB_RESUME").as_deref(), Ok("allow"));

        let core = super::resolve_core_binary().unwrap();
        let config = serde_json::json!({
            "suite": "rust-rewrite-validation",
            "depth": 2,
            "config_probe": "passed"
        });
        let timeout = std::time::Duration::from_secs(60);

        let mut initial = super::Run::start(
            &core,
            &project,
            "rust-agent rewrite W&B validation",
            &config,
            timeout,
        )
        .unwrap();
        assert_eq!(initial.id(), run_id);
        assert!(
            !initial.was_resumed(),
            "first W&B session unexpectedly resumed"
        );
        initial
            .log(
                &serde_json::json!({
                    "step": 0,
                    "phase": "initial",
                    "rust_agent_wandb_probe": 1
                })
                .as_object()
                .unwrap()
                .clone(),
            )
            .unwrap();
        initial.finish().unwrap();

        let mut resumed = super::Run::start(
            &core,
            &project,
            "rust-agent rewrite W&B validation",
            &config,
            timeout,
        )
        .unwrap();
        assert_eq!(resumed.id(), run_id);
        assert!(
            resumed.was_resumed(),
            "second W&B session did not resume the run"
        );
        resumed
            .log(
                &serde_json::json!({
                    "step": 1,
                    "phase": "resumed",
                    "rust_agent_wandb_probe": 2
                })
                .as_object()
                .unwrap()
                .clone(),
            )
            .unwrap();
        resumed.finish().unwrap();
        let stderr = std::fs::read_to_string(resumed.core.stderr_file.path()).unwrap_or_default();
        let api_key = std::env::var("WANDB_API_KEY").unwrap_or_default();
        let stderr = if api_key.is_empty() {
            stderr
        } else {
            stderr.replace(&api_key, "[redacted]")
        };
        for line in stderr.lines().filter(|line| {
            let line = line.to_ascii_lowercase();
            ["error", "warn", "fail", "panic", "offline"]
                .iter()
                .any(|needle| line.contains(needle))
        }) {
            eprintln!("wandb-core: {line}");
        }
    }

    /// Resume and finish an existing server-side run after an interrupted
    /// client session. Run manually with a real API key and run id:
    /// `cargo test live_resume_existing_server_run_and_finish -- --ignored`.
    #[test]
    #[ignore = "requires W&B credentials and an existing WANDB_RUN_ID"]
    fn live_resume_existing_server_run_and_finish() {
        let project = std::env::var("RUST_AGENT_WANDB_TEST_PROJECT")
            .expect("set RUST_AGENT_WANDB_TEST_PROJECT to a writable W&B project");
        let run_id = std::env::var("WANDB_RUN_ID").expect("set WANDB_RUN_ID to an existing run id");
        assert_eq!(std::env::var("WANDB_RESUME").as_deref(), Ok("allow"));

        let core = super::resolve_core_binary().unwrap();
        let config = serde_json::json!({
            "suite": "rust-rewrite-validation",
            "depth": 2,
            "config_probe": "passed"
        });
        let mut run = super::Run::start(
            &core,
            &project,
            "rust-agent rewrite W&B validation",
            &config,
            std::time::Duration::from_secs(60),
        )
        .unwrap();
        assert_eq!(run.id(), run_id);
        assert!(
            run.was_resumed(),
            "W&B service did not resume the existing run"
        );
        run.log(
            &serde_json::json!({
                "step": 2,
                "phase": "recovered",
                "rust_agent_wandb_probe": 3
            })
            .as_object()
            .unwrap()
            .clone(),
        )
        .unwrap();
        run.finish().unwrap();
    }

    #[test]
    fn run_id_validation_matches_wandb_reserved_characters() {
        assert!(validate_run_id("abc-123").is_ok());
        assert!(validate_run_id("").is_err());
        assert!(validate_run_id(" abc").is_err());
        assert!(validate_run_id("abc?123").is_err());
    }

    #[test]
    fn run_config_includes_sdk_internal_config_without_changing_user_values() {
        let config = serde_json::json!({
            "depth": 12,
            "learning_rate": 0.001,
            "_wandb": {"user_setting": true}
        });
        let items = run_config_items(&config).unwrap();
        let values = items
            .into_iter()
            .map(|item| {
                (
                    item.key,
                    serde_json::from_str::<serde_json::Value>(&item.value_json).unwrap(),
                )
            })
            .collect::<std::collections::BTreeMap<_, _>>();
        assert_eq!(values.get("depth"), Some(&serde_json::json!(12)));
        assert_eq!(values.get("learning_rate"), Some(&serde_json::json!(0.001)));
        assert_eq!(
            values.get("_wandb"),
            Some(&serde_json::json!({"user_setting": true}))
        );

        let defaults = run_config_items(&serde_json::json!({"depth": 2})).unwrap();
        let default_wandb = defaults.iter().find(|item| item.key == "_wandb").unwrap();
        assert_eq!(default_wandb.value_json, "{}");
    }

    #[test]
    fn auto_resume_reads_id_and_ignores_missing_or_malformed_files() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("wandb-resume.json");
        assert_eq!(read_auto_resume_id(&path).unwrap(), None);
        std::fs::write(&path, "not json").unwrap();
        assert_eq!(read_auto_resume_id(&path).unwrap(), None);
        std::fs::write(&path, r#"{"run_id":"resume-42"}"#).unwrap();
        assert_eq!(
            read_auto_resume_id(&path).unwrap().as_deref(),
            Some("resume-42")
        );
        clear_auto_resume_id(&path, "different-id").unwrap();
        assert_eq!(
            read_auto_resume_id(&path).unwrap().as_deref(),
            Some("resume-42")
        );
        clear_auto_resume_id(&path, "resume-42").unwrap();
        assert_eq!(read_auto_resume_id(&path).unwrap(), None);
    }

    #[test]
    fn service_endpoint_reads_tcp_and_unix_port_files() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("service.port");

        std::fs::write(&path, "sock=4312\nEOF\n").unwrap();
        assert!(matches!(
            read_service_endpoint(&path).unwrap(),
            Some(ServiceEndpoint::Tcp(address)) if address.port() == 4312
        ));

        #[cfg(unix)]
        {
            std::fs::write(&path, "unix=/tmp/wandb-test/socket\nEOF\n").unwrap();
            assert!(matches!(
                read_service_endpoint(&path).unwrap(),
                Some(ServiceEndpoint::Unix(path))
                    if path == std::path::Path::new("/tmp/wandb-test/socket")
            ));
        }
    }
}
