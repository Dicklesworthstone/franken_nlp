//! Charged resident INT4 models and execution on the existing blocking host.
//! No new runtime, active-catalog path, uncharged model constructor or fallback.

use super::*;
use crate::native_engine::strict_int4::weights::snapshot::Int4SnapshotError;
mod snapshot;
mod classify;
mod extract;
mod source;
mod corpus;
pub use corpus::Int4SourceCorpusConfig;
mod prefill;
pub mod batch;
use crate::tasks::classify::grouped_int4::Int4ClassificationError;
use crate::{
    artifact::converter::{ConversionSourceManifest, ConverterError, prepare_nanbeige42_input},
    native_engine::{
        decode::DecodeStepControl,
        generation::grouped_int4::{Int4GenerationBudget, bind_candidate_model},
        int4::PackedInt4Kernel,
        quant_algebra::MAX_MODEL_K,
        strict_int4::{Int4MemoryBudget, Int4MemoryRequirement, Int4ModelError, Int4ModelIdentity,
            Int4ModelLoadBudget, Int4ModelLoadRequirement, Int4ModelWeights, Int4RunBudget, StrictInt4Engine, StrictInt4Error},
    },
    tasks::chat::grouped_int4::{Int4ChatError, Int4ChatResult, PreparedInt4Chat},
    tokenizer::embedded::{PINNED_TOKENIZER_MODEL_BYTES, PINNED_TOKENIZER_CONFIG_BYTES},
};

/// Explicit source preparation, complete model conversion and allocator
/// allowances. The source must be the pinned ten-file closure, not a fnlpq.
/// Source/metadata and allocator allowances are modeled reserves, not RSS caps.
#[derive(Clone, Copy, Debug)]
pub struct Int4LoadLimits {
    pub model: Int4ModelLoadBudget,
    /// At least 128 MiB for prepared source metadata/snapshots, hash workspace
    /// and their construction. The host must raise this allowance when needed.
    pub source_preparation_bytes: u64,
    pub allocator_reserve_bytes: u64,
    pub run: RunLimits,
}
/// The exact ledger reservations made by this host, exclusive of its existing
/// runtime/preparation and RunLimits.cleanup_reserve_bytes. Not measured RSS.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4LoadRequirements { pub resident_bytes: u64, pub staging_bytes: u64 }
impl Int4LoadLimits {
    /// Pure admission calculation shared with command-line preflight. No IO,
    /// allocation, runtime, source parse or hash is performed here.
    pub fn requirements(self, group_width: usize) -> Result<Int4LoadRequirements, HostedError> {
        self.run.validate()?;
        if self.source_preparation_bytes < 128 * 1024 * 1024 || self.allocator_reserve_bytes == 0 {
            return Err(HostedError::Limits("explicit source and allocator reserves required"));
        }
        let need = Int4ModelLoadRequirement::for_group_width(group_width).map_err(model_error)?;
        if need.peak_payload_bytes().map_err(model_error)? > self.model.max_payload_bytes
            || need.source_values > self.model.max_source_values || need.groups > self.model.max_groups
            || need.shard_revalidation_bytes > self.model.max_shard_revalidation_bytes {
            return Err(HostedError::Limits("whole INT4 model load budget"));
        }
        Ok(Int4LoadRequirements {
            resident_bytes: sum(&[need.weight_payload_bytes, self.allocator_reserve_bytes])?,
            staging_bytes: sum(&[self.source_preparation_bytes, need.scratch_payload_bytes])?,
        })
    }
}

/// Stage-specific causes remain available explicitly. Default diagnostics
/// intentionally exclude paths, prompts and nested arbitrary error messages.
pub enum Int4HostError {
    Source(ConverterError), Model(Int4ModelError), Native(StrictInt4Error), Chat(Int4ChatError),
    Snapshot(Int4SnapshotError), Classification(Int4ClassificationError),
    Extraction(crate::tasks::extract::grouped_int4::Int4ExtractError),
    SourceTask(crate::tasks::source_planning::grouped_int4::Int4SourceError),
}
impl fmt::Display for Int4HostError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Source(_) => "INT4 source preparation refused",
            Self::Model(_) => "INT4 complete model conversion refused",
            Self::Native(_) => "INT4 native engine construction refused",
            Self::Chat(_) => "INT4 chat execution or finalization refused",
            Self::Snapshot(_) => "INT4 local snapshot loading or persistence refused",
            Self::Classification(_) => "INT4 classification execution or finalization refused",
            Self::Extraction(_) => "INT4 schema extraction execution or finalization refused",
            Self::SourceTask(_) => "INT4 source task execution or finalization refused",
        })
    }
}
impl fmt::Debug for Int4HostError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Display::fmt(self, f) }
}
impl Error for Int4HostError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Source(e) => Some(e), Self::Model(e) => Some(e), Self::Native(e) => Some(e),
            Self::Chat(e) => Some(e), Self::Snapshot(e) => Some(e), Self::Classification(e) => Some(e),
            Self::Extraction(e) => Some(e), Self::SourceTask(e) => Some(e) }
    }
}
fn source_error(error: ConverterError) -> HostedError { HostedError::Int4(Int4HostError::Source(error)) }
fn model_error(error: Int4ModelError) -> HostedError { HostedError::Int4(Int4HostError::Model(error)) }
fn native_error(error: StrictInt4Error) -> HostedError { HostedError::Int4(Int4HostError::Native(error)) }
fn chat_error(error: Int4ChatError) -> HostedError { HostedError::Int4(Int4HostError::Chat(error)) }

struct ResidentModel {
    // Storage drops before its weight charge, then the runtime-domain lease.
    loaded: Charged<Int4ModelWeights>, lease: EngineLease,
}
/// Clones share one immutable model and one process-ledger charge. Models
/// cannot be imported from uncharged weights or used across resource domains.
#[derive(Clone)]
pub struct ResidentInt4 { inner: Arc<ResidentModel> }
impl ResidentInt4 {
    pub fn model_identity(&self) -> &Int4ModelIdentity { self.inner.loaded.value.identity() }
    pub fn resources(&self) -> &Arc<EngineResources> { self.inner.lease.resources() }
}

impl NlpEngine {
    /// Verify the pinned source closure and convert it on the real blocking
    /// coordinator. All reservations precede source access. The strict source
    /// directory is never searched for alternate weights, and no output file
    /// or canonical artifact is created. Preparation/hash/read calls remain
    /// synchronous; cancellation waits for actual physical cleanup.
    pub fn load_source_candidate_int4(&self, source_dir: PathBuf, source_manifest: PathBuf,
        group_width: usize, limits: Int4LoadLimits, cancellation: CancellationToken)
        -> Result<ResidentInt4, HostedError> {
        dispatch::preflight(self, limits.run)?;
        let needed = limits.requirements(group_width)?;
        if source_dir.as_os_str().is_empty() || source_manifest.as_os_str().is_empty() {
            return Err(HostedError::Limits("explicit source directory and pinned manifest required"));
        }
        let lease = self.resources().acquire_lease();
        let resident = Pending::reserve(&lease, MemoryClass::Weights, needed.resident_bytes)?;
        let staging = Pending::reserve(&lease, MemoryClass::Staging, needed.staging_bytes)?;
        dispatch::run(self, limits.run, cancellation, move |control| {
            let source = allocate(staging, || {
                let manifest = ConversionSourceManifest::load_pinned(&source_manifest).map_err(source_error)?;
                checkpoint(control)?;
                prepare_nanbeige42_input(&source_dir, manifest, true, (2 * MAX_MODEL_K) as u64).map_err(source_error)
            })?;
            checkpoint(control)?;
            let assets = source.value.materialized_sources();
            check_tokenizer(&assets.tokenizer_model, &assets.tokenizer_config)?;
            let loaded = allocate(resident, || Int4ModelWeights::from_prepared(&source.value,
                group_width, limits.model, control).map_err(model_error))?;
            drop(source);
            Ok(ResidentInt4 { inner: Arc::new(ResidentModel { loaded, lease }) })
        })
    }

    /// Execute a pinned chat/text request using an explicit hardware-checked
    /// kernel (pass PackedInt4Kernel::scalar() for the conservative default).
    /// The resident model, prepared transcript and all memory guards stay in
    /// the closure until its physical completion, even if its wrapper cancels.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int4_chat(&self, model: &ResidentInt4, prepared: PreparedInt4Chat,
        request_seq: u64, native: NativeLimits, max_sampler_bytes: u64, kernel: PackedInt4Kernel,
        cancellation: CancellationToken) -> Result<HostedOutput<Int4ChatResult>, HostedError> {
        prefill::execute(self, model, prepared, request_seq, native, max_sampler_bytes,
            kernel, None, cancellation)
    }
}
fn native_requirements(prepared: &PreparedInt4Chat, sequence: u64, native: NativeLimits,
    sampler: u64) -> Result<Int4MemoryRequirement, HostedError> {
    native.run.validate()?;
    if sequence == 0 || sampler == 0 || native.allocator_reserve_bytes == 0 {
        return Err(HostedError::Limits("sequence, sampler and allocator reserves required"));
    }
    let required = Int4MemoryRequirement::for_context(native.context_tokens, prepared.native_plan().group_width())
        .map_err(native_error)?;
    if prepared.planned_work().forward_positions > native.context_tokens as u64
        || required.kv_bytes > prepared.task_plan().ir().budget().max_kv_bytes
        || prepared.native_plan().sampler_bytes() > sampler {
        return Err(HostedError::Limits("INT4 native task preflight"));
    }
    Ok(required)
}
fn check_tokenizer(model: &[u8], config: &[u8]) -> Result<(), HostedError> {
    if model != PINNED_TOKENIZER_MODEL_BYTES || config != PINNED_TOKENIZER_CONFIG_BYTES {
        return Err(HostedError::ModelIdentity);
    }
    Ok(())
}
fn checkpoint(control: &mut RunControl) -> Result<(), HostedError> {
    match control.prefill_checkpoint(0) {
        Some(cause) => Err(model_error(Int4ModelError::Cancelled(cause))), None => Ok(()),
    }
}

#[cfg(test)] mod tests;
