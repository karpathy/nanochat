//! Discovery helpers for nanochat's model-tag / numbered-checkpoint layout.

use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail, ensure};

#[derive(Debug, serde::Serialize)]
pub struct CheckpointSelection {
    pub model_tag: String,
    pub step: Option<u64>,
    pub model_file: PathBuf,
    pub metadata_file: Option<PathBuf>,
}

/// Resolve a nanochat checkpoint using its source, model tag, and optional
/// step, following the selection rules from `checkpoint_manager.py`.
pub fn select_checkpoint(
    source: &str,
    base_dir: Option<&Path>,
    model_tag: Option<&str>,
    step: Option<u64>,
) -> Result<CheckpointSelection> {
    let source_directory = match source {
        "base" => "base_checkpoints",
        "sft" => "chatsft_checkpoints",
        "rl" => "chatrl_checkpoints",
        _ => bail!("unknown checkpoint source '{source}', choose base, sft, or rl"),
    };
    let base_dir = base_dir
        .map(Path::to_owned)
        .map_or_else(default_base_dir, Ok)?;
    let root = base_dir.join(source_directory);
    let tag_path = match model_tag {
        Some(tag) => root.join(tag),
        None => find_largest_model(&root)?,
    };
    ensure!(
        tag_path.is_dir(),
        "model tag directory does not exist: {}",
        tag_path.display()
    );
    let model_file = match step {
        Some(step) => {
            let prefix = format!("model_{step:06}");
            ["pt", "safetensors"]
                .into_iter()
                .map(|extension| tag_path.join(format!("{prefix}.{extension}")))
                .find(|path| path.is_file())
                .with_context(|| format!("model step {step} not found in {}", tag_path.display()))?
        }
        None => resolve_model_file(&tag_path)?,
    };
    let selected_step = parse_model_filename(
        &model_file
            .file_name()
            .context("model checkpoint has no filename")?
            .to_string_lossy(),
    )
    .map(|(step, _)| step)
    .or(step)
    .or_else(|| find_last_step(&tag_path).ok());
    let metadata_file = selected_step
        .map(|step| tag_path.join(format!("meta_{step:06}.json")))
        .filter(|path| path.is_file())
        .or_else(|| {
            let path = tag_path.join("meta.json");
            path.is_file().then_some(path)
        });
    let selected_step = selected_step.or_else(|| {
        let metadata_file = metadata_file.as_ref()?;
        let metadata = crate::json_compat::parse_value(&fs::read(metadata_file).ok()?).ok()?;
        metadata.get("steps")?.as_u64()
    });
    Ok(CheckpointSelection {
        model_tag: tag_path
            .file_name()
            .context("model tag directory has no name")?
            .to_string_lossy()
            .into_owned(),
        step: selected_step,
        model_file,
        metadata_file,
    })
}

fn default_base_dir() -> Result<PathBuf> {
    if let Some(path) = crate::paths::base_dir_override() {
        return Ok(PathBuf::from(path));
    }
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .context("HOME or USERPROFILE must be set to locate nanochat checkpoints")?;
    Ok(PathBuf::from(home).join(".cache").join("nanochat"))
}

/// Resolve either an explicit model file or a checkpoint directory to a model
/// file. Numbered Python checkpoints are preferred over an unnumbered Rust
/// checkpoint when both are present.
pub fn resolve_model_file(path: impl AsRef<Path>) -> Result<PathBuf> {
    let path = path.as_ref();
    if path.is_file() {
        return Ok(path.to_owned());
    }
    ensure!(
        path.is_dir(),
        "checkpoint path does not exist: {}",
        path.display()
    );
    let (latest_step, latest_file) = find_last_checkpoint(path)?;
    if let Some(file) = latest_file {
        return Ok(file);
    }
    if let Some(step) = latest_step {
        bail!(
            "checkpoint metadata in {} refers to missing model step {step}",
            path.display()
        );
    }
    bail!("no model checkpoints found in {}", path.display())
}

/// Find the largest `d<number>` model tag, matching nanochat's default model
/// selection. If no depth-style tag exists, choose the most recently modified
/// directory.
pub fn find_largest_model(checkpoints_dir: impl AsRef<Path>) -> Result<PathBuf> {
    let root = checkpoints_dir.as_ref();
    let directories = fs::read_dir(root)
        .with_context(|| format!("list checkpoint root {}", root.display()))?
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            // Python's os.path.isdir follows symlinks; Path::is_dir preserves
            // that behavior for linked model-tag directories.
            let path = entry.path();
            path.is_dir().then_some(path)
        })
        .collect::<Vec<_>>();
    ensure!(
        !directories.is_empty(),
        "no model tags found in {}",
        root.display()
    );

    let depth_tags = directories
        .iter()
        .filter_map(|path| {
            let name = path.file_name()?.to_str()?;
            // checkpoint_manager.py uses re.match(r"d(\d+)") rather than
            // anchoring the tag, so names such as `d12-experiment` also count.
            let suffix = name.strip_prefix('d')?;
            let digits = suffix
                .chars()
                .take_while(char::is_ascii_digit)
                .collect::<String>();
            if digits.is_empty() {
                return None;
            }
            // Python's int() has no fixed-width limit. Normalize leading
            // zeroes so decimal strings can be compared by length and then
            // lexicographically without overflowing.
            let depth = digits.trim_start_matches('0');
            let depth = if depth.is_empty() { "0" } else { depth }.to_owned();
            Some((depth, path.clone()))
        })
        .collect::<Vec<_>>();
    // Keep the first directory in case depths tie, matching Python's stable
    // descending sort over os.listdir order.
    let largest_depth: Option<&(String, PathBuf)> =
        depth_tags.iter().fold(None, |best, candidate| match best {
            Some((best_depth, _)) if compare_decimal(best_depth, &candidate.0).is_ge() => best,
            _ => Some(candidate),
        });
    if let Some((_, path)) = largest_depth {
        return Ok(path.clone());
    }

    // Python sorts newest-first stably, so preserve the first directory when
    // modification times tie as well.
    let mut newest = directories.into_iter();
    let first = newest.next().expect("checked non-empty");
    Ok(newest.fold(first, |best, candidate| {
        let best_modified = fs::metadata(&best)
            .and_then(|metadata| metadata.modified())
            .ok();
        let candidate_modified = fs::metadata(&candidate)
            .and_then(|metadata| metadata.modified())
            .ok();
        if candidate_modified > best_modified {
            candidate
        } else {
            best
        }
    }))
}

fn compare_decimal(left: &str, right: &str) -> std::cmp::Ordering {
    left.len().cmp(&right.len()).then_with(|| left.cmp(right))
}

/// Find the newest numbered checkpoint, preferring PyTorch files whenever any
/// are present because nanochat's reference resolver only scans `model_<step>.pt`.
/// Safetensors-only directories remain supported for Rust-native checkpoints.
pub fn find_last_step(checkpoint_dir: impl AsRef<Path>) -> Result<u64> {
    let (step, _) = find_last_checkpoint(checkpoint_dir.as_ref())?;
    step.context("no numbered model checkpoints found")
}

fn find_last_checkpoint(checkpoint_dir: &Path) -> Result<(Option<u64>, Option<PathBuf>)> {
    let entries = fs::read_dir(checkpoint_dir)
        .with_context(|| format!("list checkpoint directory {}", checkpoint_dir.display()))?;
    let mut latest_pt: Option<(u64, PathBuf)> = None;
    let mut latest_safetensors: Option<(u64, PathBuf)> = None;
    for entry in entries {
        let entry = entry?;
        // The Python resolver selects by filename from os.listdir without
        // checking the entry type or whether a symlink target exists. Preserve
        // that selection so a malformed newest checkpoint fails on load
        // instead of silently resuming from an older step.
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let parsed = parse_model_filename(&name)
            .or_else(|| parse_python_model_step(&name).map(|step| (step, "pt")));
        let Some((step, extension)) = parsed else {
            continue;
        };
        let latest = if extension == "pt" {
            &mut latest_pt
        } else {
            &mut latest_safetensors
        };
        if latest
            .as_ref()
            .is_none_or(|(best_step, _)| step > *best_step)
        {
            *latest = Some((step, entry.path()));
        }
    }
    if let Some((step, _)) = latest_pt {
        // Python discovers the maximum step, then reconstructs the canonical
        // zero-padded filename in load_checkpoint(). If duplicate filenames
        // encode the same step, this selects the same file regardless of
        // directory iteration order.
        return Ok((
            Some(step),
            Some(checkpoint_dir.join(format!("model_{step:06}.pt"))),
        ));
    }
    if let Some((step, path)) = latest_safetensors {
        return Ok((Some(step), Some(path)));
    }

    // Preserve single-checkpoint directories after preferring numbered
    // nanochat checkpoints, including unnumbered PyTorch exports.
    let unnumbered_python = checkpoint_dir.join("model.pt");
    if unnumbered_python.is_file() {
        return Ok((None, Some(unnumbered_python)));
    }

    // Rust's original single-checkpoint layout predates numbered checkpoint
    // directories.
    let unnumbered = checkpoint_dir.join("model.safetensors");
    if unnumbered.is_file() {
        return Ok((None, Some(unnumbered)));
    }

    // An older Rust checkpoint may carry its step only in meta.json.
    let metadata_path = checkpoint_dir.join("meta.json");
    if metadata_path.is_file() {
        let metadata =
            crate::json_compat::parse_value(&fs::read(&metadata_path).with_context(|| {
                format!("read checkpoint metadata {}", metadata_path.display())
            })?)?;
        if let Some(step) = metadata.get("steps").and_then(serde_json::Value::as_u64) {
            return Ok((Some(step), None));
        }
    }
    Ok((None, None))
}

fn parse_model_filename(name: &str) -> Option<(u64, &str)> {
    let stem = name.strip_prefix("model_")?;
    let (step, extension) = stem.rsplit_once('.')?;
    if !matches!(extension, "pt" | "safetensors") || step.is_empty() {
        return None;
    }
    Some((step.parse().ok()?, extension))
}

/// Python's checkpoint manager uses `re.search(r'model_(\d+)\.pt$', name)`,
/// so numbered PyTorch discovery accepts a prefix before `model_` and later
/// reconstructs the canonical `model_<step>.pt` path.
fn parse_python_model_step(name: &str) -> Option<u64> {
    let start = name.rfind("model_")? + "model_".len();
    let digits = name[start..].strip_suffix(".pt")?;
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::{
        find_largest_model, find_last_checkpoint, parse_python_model_step, resolve_model_file,
        select_checkpoint,
    };
    use std::path::PathBuf;

    #[test]
    fn python_checkpoint_step_search_accepts_prefixes_and_requires_terminal_pt() {
        assert_eq!(parse_python_model_step("backup_model_000123.pt"), Some(123));
        assert_eq!(parse_python_model_step("model_9.pt.tmp"), None);
        assert_eq!(parse_python_model_step("backup_model_x9.pt"), None);
    }

    #[test]
    fn malformed_newest_prefixed_checkpoint_does_not_fall_back_to_older_step() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("model_000010.pt"), b"older").unwrap();
        std::fs::write(directory.path().join("backup_model_000011.pt"), b"newer").unwrap();
        let (step, path) = find_last_checkpoint(directory.path()).unwrap();
        assert_eq!(step, Some(11));
        assert_eq!(
            path,
            Some(PathBuf::from(directory.path()).join("model_000011.pt"))
        );
        assert!(!path.unwrap().exists());
    }

    #[test]
    fn explicit_tag_resolves_unnumbered_python_checkpoint_and_meta_step() {
        let base = tempfile::tempdir().unwrap();
        let tag_dir = base.path().join("base_checkpoints").join("custom-model");
        std::fs::create_dir_all(&tag_dir).unwrap();
        let model = tag_dir.join("model.pt");
        std::fs::write(&model, b"model").unwrap();
        std::fs::write(tag_dir.join("meta.json"), br#"{"steps":42}"#).unwrap();

        let selection =
            select_checkpoint("base", Some(base.path()), Some("custom-model"), None).unwrap();
        assert_eq!(selection.model_file, model);
        assert_eq!(selection.step, Some(42));
        assert_eq!(resolve_model_file(&tag_dir).unwrap(), model);
    }

    #[test]
    fn largest_model_uses_depth_then_directory_mtime_fallback() {
        let root = tempfile::tempdir().unwrap();
        let shallow = root.path().join("d12-small");
        let deep = root.path().join("d100-experiment");
        std::fs::create_dir(&shallow).unwrap();
        std::fs::create_dir(&deep).unwrap();
        assert_eq!(find_largest_model(root.path()).unwrap(), deep);

        let fallback = tempfile::tempdir().unwrap();
        let older = fallback.path().join("alpha");
        let newer = fallback.path().join("beta");
        std::fs::create_dir(&older).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(30));
        std::fs::create_dir(&newer).unwrap();
        assert_eq!(find_largest_model(fallback.path()).unwrap(), newer);
    }
}
