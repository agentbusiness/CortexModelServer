// CortexModelServer: from CortexAgent (crates/model/src/gguf_meta.rs, lines 1-633, unchanged but
// for this header, `pub(crate)` -> `pub` and the dead-code allowance; its tests are not copied).
#![allow(dead_code)]

//! The attention geometry a GGUF image declares about itself.
//!
//! One number decides how much context a machine can afford: the bytes of KV cache one sequence
//! occupies per token. `llama-server` allocates `--ctx-size` tokens of it *in total*, and this
//! repository configures that as `slots × context_tokens` with one stream per slot, so the cost is
//! exactly linear in both the context and the slot count. The load-time fit
//! (`LlamaServerConfig::fit_to_memory`) divides measured memory by it.
//!
//! That number is not a judgement and must not be guessed at: every term of it is written in the
//! weight file's own header. This module reads the header — and nothing else, seeking past the
//! tokenizer arrays rather than buffering them — and reports the
//! geometry. A file that does not declare what is needed produces `None`, and the caller keeps the
//! native context rather than an invented estimate.
//!
//! Hybrid architectures keep a fixed-size state per sequence for their recurrent blocks instead of
//! a per-token cache on every layer, and they say so: `<arch>.attention.head_count_kv` is a
//! *per-layer array* whose zeros are the recurrent blocks. Reading only the scalar form and falling
//! back to the query-head count over-stated a 4B hybrid image fifty-two times (measured), so the
//! array is summed, and the recurrent blocks are charged their own fixed state from the `ssm.*`
//! dimensions the same header declares.
//!
//! Interleaved sliding-window images (Gemma 4) declare which layers attend only to a window
//! (`<arch>.attention.sliding_window_pattern`, one boolean per layer), how wide the window is
//! (`<arch>.attention.sliding_window`), the narrower head width those layers use
//! (`<arch>.attention.key_length_swa` / `value_length_swa`), and how many trailing layers reuse an
//! earlier layer's cache instead of owning one (`<arch>.attention.shared_kv_layers`). The installed
//! llama.cpp (b10360, `src/llama-kv-cache-iswa.cpp`) allocates the window layers
//! `pad256(min(context, window + ubatch))` cells per sequence unless `--swa-full` is passed, which
//! this repository never passes; the full-attention layers get the whole context.

use std::{
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom},
    path::Path,
};

/// GGUF versions this reader accepts. Version 1 encoded its counts as 32-bit and predates every
/// artifact this repository pins.
const SUPPORTED_VERSIONS: [u32; 2] = [2, 3];

/// Runaway guards. A real image declares a few dozen keys; the tokenizer is one array value, not
/// thousands of entries at the top level.
const MAX_METADATA_KEYS: u64 = 4_096;
const MAX_KEY_BYTES: u64 = 1024;
const MAX_STRING_BYTES: u64 = 64 * 1024 * 1024;

/// A recurrent block's state is held in FP32 by `llama.cpp`, whatever the KV cache type.
const RECURRENT_ELEMENT_BYTES: u64 = 4;

/// llama.cpp pads the sliding-window cache to this many cells (`llama-kv-cache-iswa.cpp`).
const SWA_CELL_PADDING: u64 = 256;

/// The element type `llama-server` stores the KV cache in, selected by the operator with
/// `CORTEXAGENT_KV_CACHE_TYPE` and passed as `--cache-type-k` / `--cache-type-v`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KvCacheType {
    /// Two bytes per element; llama.cpp's own default.
    F16,
    /// ggml `q8_0`: blocks of 32 elements, each 32 signed bytes plus one FP16 scale — 34 bytes.
    Q8_0,
}

impl KvCacheType {
    /// Every accepted setting, in the spelling llama-server and the environment variable use.
    pub const ALL: [Self; 2] = [Self::F16, Self::Q8_0];

    /// The spelling `llama-server` accepts for `--cache-type-k` / `--cache-type-v`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Q8_0 => "q8_0",
        }
    }

    /// Strict parse of the operator setting: exactly `f16` or `q8_0`, nothing inferred.
    pub fn parse(value: &str) -> std::result::Result<Self, String> {
        Self::ALL
            .into_iter()
            .find(|kind| kind.as_str() == value)
            .ok_or_else(|| {
                format!(
                    "CORTEXAGENT_KV_CACHE_TYPE={value:?} is not a supported KV cache type; \
                     use one of: f16, q8_0"
                )
            })
    }

    /// A quantized V cache is only implemented on llama.cpp's flash-attention path.
    #[must_use]
    pub const fn requires_flash_attention(self) -> bool {
        matches!(self, Self::Q8_0)
    }

    /// Bytes one cache row of `elements` values occupies (ggml row size).
    #[must_use]
    pub const fn row_bytes(self, elements: u64) -> u64 {
        match self {
            Self::F16 => elements.saturating_mul(2),
            Self::Q8_0 => elements.div_ceil(32).saturating_mul(34),
        }
    }

    const fn index(self) -> usize {
        match self {
            Self::F16 => 0,
            Self::Q8_0 => 1,
        }
    }
}

/// What one sequence costs this image, as the image declares it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KvGeometry {
    /// Blocks that own a per-token KV cache. Fewer than the total block count on a hybrid or on an
    /// image whose trailing layers share an earlier layer's cache.
    pub attention_blocks: u64,
    /// Of those, the blocks that attend over a sliding window rather than the whole context.
    pub sliding_blocks: u64,
    /// KV heads summed across the cache-owning blocks.
    pub kv_heads_total: u64,
    pub key_length: u64,
    pub value_length: u64,
    /// Width of the sliding window in tokens; zero when the image declares no sliding layers.
    pub sliding_window: u64,
    /// Blocks that keep a fixed recurrent state per sequence instead of a per-token cache.
    pub recurrent_blocks: u64,
    pub ssm_inner_size: u64,
    pub ssm_state_size: u64,
    pub ssm_conv_kernel: u64,
    pub ssm_group_count: u64,
    /// Per-token bytes of the full-context layers, indexed by [`KvCacheType`].
    full_row_bytes: [u64; 2],
    /// Per-token bytes of the sliding-window layers, indexed by [`KvCacheType`].
    sliding_row_bytes: [u64; 2],
}

impl KvGeometry {
    /// Bytes of KV cache each additional token of context costs one sequence: the full-context
    /// layers. Sliding-window layers stop growing at their window and are charged in
    /// [`Self::bytes_per_slot`].
    #[must_use]
    pub const fn bytes_per_token(&self, cache: KvCacheType) -> u64 {
        self.full_row_bytes[cache.index()]
    }

    /// Cells llama.cpp allocates per sequence for the sliding-window layers.
    #[must_use]
    pub const fn sliding_cells(&self, context_tokens: u64, ubatch_tokens: u64) -> u64 {
        if self.sliding_blocks == 0 {
            return 0;
        }
        let wanted = self.sliding_window.saturating_add(ubatch_tokens);
        let cells = if wanted < context_tokens { wanted } else { context_tokens };
        cells.div_ceil(SWA_CELL_PADDING).saturating_mul(SWA_CELL_PADDING)
    }

    /// Bytes one sequence holds no matter how long its conversation runs: the recurrent blocks'
    /// convolution and state tensors. Zero for a purely attentional image.
    #[must_use]
    pub const fn recurrent_bytes_per_sequence(&self) -> u64 {
        let state = self.ssm_inner_size.saturating_mul(self.ssm_state_size);
        let convolution = self.ssm_conv_kernel.saturating_sub(1).saturating_mul(
            self.ssm_inner_size.saturating_add(
                2u64.saturating_mul(self.ssm_group_count)
                    .saturating_mul(self.ssm_state_size),
            ),
        );
        self.recurrent_blocks
            .saturating_mul(state.saturating_add(convolution))
            .saturating_mul(RECURRENT_ELEMENT_BYTES)
    }

    /// Everything one decode slot costs at `context_tokens` of context with a physical batch of
    /// `ubatch_tokens`, stored as `cache`.
    #[must_use]
    pub const fn bytes_per_slot(
        &self,
        context_tokens: u64,
        ubatch_tokens: u64,
        cache: KvCacheType,
    ) -> u64 {
        self.bytes_per_token(cache)
            .saturating_mul(context_tokens)
            .saturating_add(self.checkpoint_bytes(context_tokens, ubatch_tokens, cache))
    }

    /// Bytes one context checkpoint holds: the state that cannot be rewound — the sliding-window
    /// cells and the recurrent blocks' state — which is exactly what llama.cpp copies into host RAM
    /// per checkpoint (`LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`). The full-context layers are not
    /// copied; they can be truncated in place. Zero for a purely full-attention image.
    #[must_use]
    pub const fn checkpoint_bytes(
        &self,
        context_tokens: u64,
        ubatch_tokens: u64,
        cache: KvCacheType,
    ) -> u64 {
        let sliding = if self.sliding_blocks == 0 {
            0
        } else {
            self.sliding_row_bytes[cache.index()]
                .saturating_mul(self.sliding_cells(context_tokens, ubatch_tokens))
        };
        sliding.saturating_add(self.recurrent_bytes_per_sequence())
    }
}

/// The metadata value types, numbered as the GGUF specification numbers them.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ValueType {
    Uint8,
    Int8,
    Uint16,
    Int16,
    Uint32,
    Int32,
    Float32,
    Bool,
    String,
    Array,
    Uint64,
    Int64,
    Float64,
}

impl ValueType {
    fn from_code(code: u32) -> Option<Self> {
        Some(match code {
            0 => Self::Uint8,
            1 => Self::Int8,
            2 => Self::Uint16,
            3 => Self::Int16,
            4 => Self::Uint32,
            5 => Self::Int32,
            6 => Self::Float32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::Uint64,
            11 => Self::Int64,
            12 => Self::Float64,
            _ => return None,
        })
    }

    /// Byte width of a fixed-size value, or `None` for the two that carry their own length.
    const fn fixed_width(self) -> Option<u64> {
        Some(match self {
            Self::Uint8 | Self::Int8 | Self::Bool => 1,
            Self::Uint16 | Self::Int16 => 2,
            Self::Uint32 | Self::Int32 | Self::Float32 => 4,
            Self::Uint64 | Self::Int64 | Self::Float64 => 8,
            Self::String | Self::Array => return None,
        })
    }
}

/// One value, reduced to the only things this reader wants from it.
enum Scalar {
    Number(u64),
    /// A fixed-width integer array, which is how a hybrid declares its per-layer shape.
    Numbers(Vec<u64>),
    Text(String),
    Other,
}

/// Bound on a per-layer array this reader will materialize. A block count past this is not a model.
const MAX_ARRAY_ELEMENTS: u64 = 4_096;

struct Reader<R> {
    inner: R,
}

impl<R: Read + Seek> Reader<R> {
    fn header(&mut self) -> Option<u64> {
        let mut magic = [0u8; 4];
        self.inner.read_exact(&mut magic).ok()?;
        if &magic != b"GGUF" || !SUPPORTED_VERSIONS.contains(&self.u32()?) {
            return None;
        }
        let _tensor_count = self.u64()?;
        let key_count = self.u64()?;
        (key_count <= MAX_METADATA_KEYS).then_some(key_count)
    }

    fn u32(&mut self) -> Option<u32> {
        let mut bytes = [0u8; 4];
        self.inner.read_exact(&mut bytes).ok()?;
        Some(u32::from_le_bytes(bytes))
    }

    fn u64(&mut self) -> Option<u64> {
        let mut bytes = [0u8; 8];
        self.inner.read_exact(&mut bytes).ok()?;
        Some(u64::from_le_bytes(bytes))
    }

    fn skip(&mut self, bytes: u64) -> Option<()> {
        let offset = i64::try_from(bytes).ok()?;
        self.inner.seek(SeekFrom::Current(offset)).ok()?;
        Some(())
    }

    fn string(&mut self, limit: u64) -> Option<String> {
        let length = self.u64()?;
        if length > limit {
            // Too long to be anything this reader wants. Skip it rather than allocating it.
            self.skip(length)?;
            return Some(String::new());
        }
        let mut bytes = vec![0u8; usize::try_from(length).ok()?];
        self.inner.read_exact(&mut bytes).ok()?;
        String::from_utf8(bytes).ok()
    }

    /// Read one value, returning what it is worth to this reader and leaving the cursor after it.
    fn value(&mut self, kind: ValueType) -> Option<Scalar> {
        match kind {
            ValueType::String => Some(Scalar::Text(self.string(MAX_STRING_BYTES)?)),
            ValueType::Array => {
                let element = ValueType::from_code(self.u32()?)?;
                let count = self.u64()?;
                match element.fixed_width() {
                    Some(width) => {
                        // A short integer array is the per-layer shape and is read; anything else
                        // is one seek, however many elements it has.
                        let integral = matches!(
                            element,
                            ValueType::Bool
                                | ValueType::Uint8
                                | ValueType::Int8
                                | ValueType::Uint16
                                | ValueType::Int16
                                | ValueType::Uint32
                                | ValueType::Int32
                                | ValueType::Uint64
                                | ValueType::Int64
                        );
                        if integral && count <= MAX_ARRAY_ELEMENTS {
                            let mut values = Vec::with_capacity(usize::try_from(count).ok()?);
                            for _ in 0..count {
                                match self.value(element)? {
                                    Scalar::Number(number) => values.push(number),
                                    _ => return None,
                                }
                            }
                            return Some(Scalar::Numbers(values));
                        }
                        self.skip(count.saturating_mul(width))?;
                    }
                    None => {
                        // Strings and nested arrays carry their own lengths, so each is walked.
                        for _ in 0..count {
                            self.value(element)?;
                        }
                    }
                }
                Some(Scalar::Other)
            }
            ValueType::Bool | ValueType::Uint8 => {
                let mut byte = [0u8; 1];
                self.inner.read_exact(&mut byte).ok()?;
                Some(Scalar::Number(u64::from(byte[0])))
            }
            ValueType::Int8 => {
                let mut byte = [0u8; 1];
                self.inner.read_exact(&mut byte).ok()?;
                Some(Scalar::Number(u64::from(byte[0])))
            }
            ValueType::Uint16 | ValueType::Int16 => {
                let mut bytes = [0u8; 2];
                self.inner.read_exact(&mut bytes).ok()?;
                Some(Scalar::Number(u64::from(u16::from_le_bytes(bytes))))
            }
            ValueType::Uint32 | ValueType::Int32 => Some(Scalar::Number(u64::from(self.u32()?))),
            ValueType::Float32 => {
                self.skip(4)?;
                Some(Scalar::Other)
            }
            ValueType::Uint64 | ValueType::Int64 => Some(Scalar::Number(self.u64()?)),
            ValueType::Float64 => {
                self.skip(8)?;
                Some(Scalar::Other)
            }
        }
    }
}

/// Read the attention geometry a GGUF image declares, or `None` when the file is not a readable
/// GGUF or does not declare enough of it.
///
/// Never fails loudly: this sizes a tunable, and a machine whose weight file cannot be introspected
/// still has to start.
#[must_use]
pub fn kv_geometry(model_file: &Path) -> Option<KvGeometry> {
    let file = File::open(model_file).ok()?;
    kv_geometry_from(BufReader::new(file))
}

/// Read the exact architecture identifier declared by the artifact, without loading its tensors
/// or tokenizer. Missing, non-string, empty, or oversized identifiers are unavailable.
pub fn architecture(path: &Path) -> Option<String> {
    architecture_from(BufReader::new(File::open(path).ok()?))
}

/// Native per-sequence context, selected by the artifact's exact architecture namespace.
pub fn context_length(path: &Path) -> Option<u32> {
    context_length_from(BufReader::new(File::open(path).ok()?))
}

fn context_length_from<R: Read + Seek>(mut source: R) -> Option<u32> {
    let architecture = architecture_from(&mut source)?;
    source.rewind().ok()?;
    let wanted = format!("{architecture}.context_length");
    let mut reader = Reader { inner: source };
    for _ in 0..reader.header()? {
        let key = reader.string(MAX_KEY_BYTES)?;
        let kind = ValueType::from_code(reader.u32()?)?;
        if key == wanted {
            if !matches!(kind, ValueType::Uint32 | ValueType::Uint64) {
                return None;
            }
            let Scalar::Number(value) = reader.value(kind)? else { return None; };
            return u32::try_from(value).ok().filter(|value| *value > 0);
        }
        reader.value(kind)?;
    }
    None
}

fn architecture_from<R: Read + Seek>(source: R) -> Option<String> {
    let mut reader = Reader { inner: source };
    for _ in 0..reader.header()? {
        let key = reader.string(MAX_KEY_BYTES)?;
        let kind = ValueType::from_code(reader.u32()?)?;
        if key == "general.architecture" {
            if kind != ValueType::String {
                return None;
            }
            let architecture = reader.string(MAX_KEY_BYTES)?;
            return (!architecture.is_empty()).then_some(architecture);
        }
        reader.value(kind)?;
    }
    None
}

fn kv_geometry_from<R: Read + Seek>(source: R) -> Option<KvGeometry> {
    let mut reader = Reader { inner: source };
    let key_count = reader.header()?;

    let mut architecture: Option<String> = None;
    let mut block_count: Option<u64> = None;
    let mut head_count_kv: Option<u64> = None;
    // The per-layer form, whose zeros are the recurrent blocks.
    let mut head_count_kv_layers: Option<Vec<u64>> = None;
    let mut recurrent_layers: Option<Vec<u64>> = None;
    let mut recurrent_all: Option<bool> = None;
    let mut head_count: Option<u64> = None;
    let mut embedding_length: Option<u64> = None;
    let mut key_length: Option<u64> = None;
    let mut value_length: Option<u64> = None;
    let mut key_length_swa: Option<u64> = None;
    let mut value_length_swa: Option<u64> = None;
    let mut sliding_window: Option<u64> = None;
    let mut sliding_pattern: Option<Vec<u64>> = None;
    let mut shared_kv_layers = 0u64;
    let mut ssm_inner_size = 0u64;
    let mut ssm_state_size = 0u64;
    let mut ssm_conv_kernel = 0u64;
    let mut ssm_group_count = 0u64;

    for _ in 0..key_count {
        let key = reader.string(MAX_KEY_BYTES)?;
        let kind = ValueType::from_code(reader.u32()?)?;
        let value = reader.value(kind)?;
        // Architecture-prefixed keys are matched by suffix: the prefix is the architecture's own
        // name, which is exactly the string `general.architecture` carries, and matching on the
        // suffix means a new architecture needs no entry anywhere.
        let number = match value {
            Scalar::Number(number) => Some(number),
            Scalar::Numbers(values) => {
                if key.ends_with(".attention.head_count_kv") {
                    head_count_kv_layers = Some(values);
                } else if key.ends_with(".attention.recurrent_layers") {
                    recurrent_layers = Some(values);
                } else if key.ends_with(".attention.sliding_window_pattern") {
                    sliding_pattern = Some(values);
                }
                None
            }
            Scalar::Text(ref text) => {
                if key == "general.architecture" {
                    architecture = Some(text.clone());
                }
                None
            }
            Scalar::Other => None,
        };
        let Some(number) = number else { continue };
        if key.ends_with(".block_count") {
            block_count = Some(number);
        } else if key.ends_with(".attention.recurrent_layers") {
            recurrent_all = Some(number != 0);
        } else if key.ends_with(".attention.head_count_kv") {
            head_count_kv = Some(number);
        } else if key.ends_with(".attention.head_count") {
            head_count = Some(number);
        } else if key.ends_with(".attention.key_length") {
            key_length = Some(number);
        } else if key.ends_with(".attention.value_length") {
            value_length = Some(number);
        } else if key.ends_with(".attention.key_length_swa") {
            key_length_swa = Some(number);
        } else if key.ends_with(".attention.value_length_swa") {
            value_length_swa = Some(number);
        } else if key.ends_with(".attention.sliding_window") {
            sliding_window = Some(number);
        } else if key.ends_with(".attention.shared_kv_layers") {
            shared_kv_layers = number;
        } else if key.ends_with(".embedding_length") {
            embedding_length = Some(number);
        } else if key.ends_with(".ssm.inner_size") {
            ssm_inner_size = number;
        } else if key.ends_with(".ssm.state_size") {
            ssm_state_size = number;
        } else if key.ends_with(".ssm.conv_kernel") {
            ssm_conv_kernel = number;
        } else if key.ends_with(".ssm.group_count") {
            ssm_group_count = number;
        }
    }
    // A header without an architecture is not a model geometry, whatever else it declares.
    architecture?;

    let block_count = block_count?;
    let layer_count = usize::try_from(block_count).ok()?;
    // `key_length`/`value_length` are optional because they are derivable: the head dimension is
    // the embedding width divided across the query heads.
    let head_dimension = match (embedding_length, head_count) {
        (Some(embedding), Some(heads)) if heads != 0 => Some(embedding / heads),
        _ => None,
    };
    let key_length = key_length.or(head_dimension)?;
    let value_length = value_length.or(head_dimension)?;

    // KV heads per layer, zero for a block that keeps no per-token cache. The per-layer array is
    // the whole truth when it is present: which blocks attend, and how widely. Without it, every
    // block attends — and a multi-head-attention image declaring no separate KV head count means
    // every head is a KV head, which is the format's own default rather than an assumption.
    let layer_heads: Vec<u64> = match head_count_kv_layers {
        Some(layers) => layers,
        None => {
            let heads = head_count_kv.or(head_count)?;
            // A recurrent mask (`.attention.recurrent_layers`, per layer or for all) names the
            // blocks that keep no per-token cache; without one, every block attends.
            if let Some(layers) = recurrent_layers {
                if layers.len() != layer_count {
                    return None;
                }
                layers.iter().map(|recurrent| if *recurrent == 0 { heads } else { 0 }).collect()
            } else if let Some(recurrent) = recurrent_all {
                vec![if recurrent { 0 } else { heads }; layer_count]
            } else {
                vec![heads; layer_count]
            }
        }
    };
    // A declared pattern must describe every block; a partial one is not a geometry.
    if sliding_pattern.as_ref().is_some_and(|pattern| pattern.len() != layer_count) {
        return None;
    }
    // Trailing layers that reuse an earlier layer's cache own none of their own
    // (`n_layer_kv_from_start` in llama.cpp).
    let owning_layers = block_count.checked_sub(shared_kv_layers)?;

    let mut attention_blocks = 0u64;
    let mut sliding_blocks = 0u64;
    let mut kv_heads_total = 0u64;
    let mut full_row_bytes = [0u64; 2];
    let mut sliding_row_bytes = [0u64; 2];
    for (layer, heads) in layer_heads.iter().copied().enumerate() {
        if heads == 0 || layer as u64 >= owning_layers {
            continue;
        }
        let sliding = sliding_pattern
            .as_ref()
            .is_some_and(|pattern| pattern[layer] != 0);
        let (key, value) = if sliding {
            (key_length_swa.unwrap_or(key_length), value_length_swa.unwrap_or(value_length))
        } else {
            (key_length, value_length)
        };
        if key == 0 || value == 0 {
            return None;
        }
        attention_blocks += 1;
        kv_heads_total = kv_heads_total.saturating_add(heads);
        let target = if sliding {
            sliding_blocks += 1;
            &mut sliding_row_bytes
        } else {
            &mut full_row_bytes
        };
        for cache in KvCacheType::ALL {
            let row = cache
                .row_bytes(heads.saturating_mul(key))
                .saturating_add(cache.row_bytes(heads.saturating_mul(value)));
            target[cache.index()] = target[cache.index()].saturating_add(row);
        }
    }
    let sliding_window = if sliding_blocks > 0 { sliding_window.filter(|window| *window > 0)? } else { 0 };
    if kv_heads_total == 0 || key_length == 0 || value_length == 0 {
        return None;
    }
    Some(KvGeometry {
        attention_blocks,
        sliding_blocks,
        kv_heads_total,
        key_length,
        value_length,
        sliding_window,
        recurrent_blocks: layer_heads.iter().filter(|heads| **heads == 0).count() as u64
            + block_count.saturating_sub(layer_heads.len() as u64),
        ssm_inner_size,
        ssm_state_size,
        ssm_conv_kernel,
        ssm_group_count,
        full_row_bytes,
        sliding_row_bytes,
    })
}
