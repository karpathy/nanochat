//! Shared path environment handling.

use std::{
    ffi::OsString,
    path::{Path, PathBuf},
};

/// Return the nanochat data-directory override when it is set to a non-empty
/// value, matching Python's truthiness check in `get_base_dir()`.
pub fn base_dir_override() -> Option<OsString> {
    std::env::var_os("NANOCHAT_BASE_DIR").filter(|value| !value.is_empty())
}

/// Prefer a native tokenizer JSON cache, while retaining Python pickle caches.
pub fn cached_tokenizer_path(base_dir: &Path) -> PathBuf {
    let tokenizer_dir = base_dir.join("tokenizer");
    let json_path = tokenizer_dir.join("tokenizer.json");
    let pickle_path = tokenizer_dir.join("tokenizer.pkl");
    if json_path.is_file() || !pickle_path.is_file() {
        json_path
    } else {
        pickle_path
    }
}

#[cfg(test)]
mod tests {
    use super::cached_tokenizer_path;

    #[test]
    fn tokenizer_cache_prefers_json_and_falls_back_to_python_pickle() {
        let base_dir = tempfile::tempdir().unwrap();
        let tokenizer_dir = base_dir.path().join("tokenizer");
        std::fs::create_dir_all(&tokenizer_dir).unwrap();
        let pickle = tokenizer_dir.join("tokenizer.pkl");
        std::fs::write(&pickle, b"legacy").unwrap();
        assert_eq!(cached_tokenizer_path(base_dir.path()), pickle);

        let json = tokenizer_dir.join("tokenizer.json");
        std::fs::write(&json, b"native").unwrap();
        assert_eq!(cached_tokenizer_path(base_dir.path()), json);
    }
}
