//! Shared engine-neutral template utilities.
pub use cc_model_kit::template::*;

/// Templates shipped with the plugin (see `templates/README.md`).
pub fn template_source(name: &str) -> Option<&'static str> {
    match name {
        "qwen3.8.jinja" => Some(include_str!("../templates/qwen3.8.jinja")),
        // CortexModelServer patch: the LFM2.5-1.2B variant's template (the knowledge app's, the model's own).
        "lfm2.5.jinja" => Some(include_str!("../templates/lfm2.5.jinja")),
        _ => None,
    }
}
