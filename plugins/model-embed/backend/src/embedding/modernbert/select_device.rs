use candle_core::Device;

/// The device the encoder runs on: the GPU when one is preferred and this
/// build can drive it, otherwise the CPU.
pub(super) fn select_device(prefer_gpu: bool) -> Device {
    #[cfg(target_os = "macos")]
    if prefer_gpu {
        match Device::new_metal(0) {
            Ok(device) => return device,
            Err(error) => {
                embed_common::embed_warn!(
                    "Metal is unavailable for the semantic encoder: {}; running it on the CPU",
                    error
                );
            }
        }
    }
    #[cfg(not(target_os = "macos"))]
    let _ = prefer_gpu;
    Device::Cpu
}
