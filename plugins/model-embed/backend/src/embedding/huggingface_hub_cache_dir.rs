use std::env;
use std::path::PathBuf;

pub(crate) fn huggingface_hub_cache_dir() -> PathBuf {
    if let Some(path) = env::var_os("HUGGINGFACE_HUB_CACHE") {
        return PathBuf::from(path);
    }
    if let Some(path) = env::var_os("HF_HOME") {
        return PathBuf::from(path).join("hub");
    }
    if let Some(home) = env::var_os("HOME") {
        return PathBuf::from(home).join(".cache/huggingface/hub");
    }
    PathBuf::from(".cache/huggingface/hub")
}
