//! Explicit current-candidate INT8 execution on the process-owned runtime.
//!
//! This is not catalog discovery or artifact activation. The existing forensic
//! candidate loader keeps its evidence grade. Immutable weights are shared by
//! Arc; native calls use the one configured blocking coordinator, real memory
//! reservations, finite checkpoints and an actual-completion handoff.

use std::{error::Error, fmt, path::PathBuf, sync::Arc, time::Duration};
use asupersync::{cx::ScopedCpuError, runtime::{state::SpawnError, task_handle::JoinError}};
use crate::{
    NlpEngine, EngineLease, EngineResources, MemoryClass, MemoryReservation,
    CommittedMemory, ReservationError,
    execution_identity::{ExecutionIdentity, Sha256Digest},
    batch::generation::GuardedOutput,
    native_engine::{
        artifact_bridge::{ArtifactLoadBudget, ArtifactIdentity},
        generation::quantized::Int8GenerationBudget,
        constrained_int8::Int8JsonBudget,
        strict_int8::{CurrentCandidateInt8Model, CurrentCandidateInt8Error,
            Int8MemoryBudget, Int8MemoryRequirement, Int8RunBudget, StrictInt8Error},
    },
    tasks::{chat::quantized::{PreparedInt8Chat, Int8ChatResult, Int8ChatError},
        extract::{ExtractionVocabulary, quantized::{Int8ExtractPlan, Int8ExtractRun, Int8ExtractError}}},
};
mod dispatch;
mod scored;
mod resolve;
pub use resolve::ResolveConfig;
mod source;
pub use source::SourceLimits;
mod source_map;
pub use source_map::SourceMapConfig;
pub mod corpus;
pub use dispatch::{CancellationToken, RunControl, RunStop};

/// Finite cooperative execution limits, including time spent awaiting the pool.
/// These are not a thread-preemption or bounded OS-I/O-latency promise.
#[derive(Clone, Copy, Debug)]
pub struct RunLimits {
    pub max_elapsed: Duration,
    pub max_checkpoints: u64,
    /// Separately retained memory for cancellation/drain bookkeeping.
    pub cleanup_reserve_bytes: u64,
}
impl RunLimits {
    fn validate(self) -> Result<(), HostedError> {
        if self.max_elapsed.is_zero() || self.max_checkpoints < 2 || self.cleanup_reserve_bytes == 0 {
            return Err(HostedError::Limits("finite duration, checkpoints and cleanup reserve required"));
        }
        Ok(())
    }
}

/// The caller explicitly prices tokenizer/metadata/allocator overhead. Native
/// weight and streaming payload caps come from the existing bridge. No field
/// is advertised as a measured resident-set limit.
#[derive(Clone, Copy, Debug)]
pub struct LoadLimits {
    pub artifact: ArtifactLoadBudget,
    pub tokenizer_and_metadata_bytes: u64,
    pub allocator_reserve_bytes: u64,
    pub run: RunLimits,
}

/// Temporary native allocation envelope. The exact KV/RoPE/reference scratch
/// payload is derived from context; the caller supplies non-payload headroom.
#[derive(Clone, Copy, Debug)]
pub struct NativeLimits {
    pub context_tokens: usize,
    pub allocator_reserve_bytes: u64,
    pub run: RunLimits,
}

/// Stable failure categories; Display/Debug never print model paths, prompt
/// text, exception payloads or arbitrary nested native messages. Typed causes
/// are retained for an embedding application's explicit diagnostic handling.
pub enum HostedError {
    Limits(&'static str),
    Reentrant,
    ResourceDomain,
    ModelIdentity,
    SingleCoordinatorRequired,
    MissingRuntimeContext,
    Reservation(ReservationError),
    Spawn(SpawnError),
    Join { source: JoinError, physical_error: Option<Box<HostedError>> },
    Scope { source: ScopedCpuError, physical_error: Option<Box<HostedError>> },
    Stopped { stop: RunStop, task_error: Option<Box<HostedError>> },
    Panicked,
    CompletionMissing,
    Model(CurrentCandidateInt8Error),
    Native(StrictInt8Error),
    Chat(Int8ChatError),
    Extraction(Int8ExtractError),
    Classification(crate::tasks::classify::quantized::Int8ClassificationError),
    Judge(crate::tasks::judge::quantized::Int8JudgeError),
    Source(crate::tasks::source_planning::quantized::Int8SourceError),
    SourceMap(crate::tasks::source_planning::quantized::long::Int8SourceMapError),
    Resolution(crate::corpus::native_resolve::quantized::Int8ResolveError),
    BatchSetup(crate::batch::BatchFault),
    Batch(crate::batch::BatchRunError),
}
impl fmt::Display for HostedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Limits(_) => "hosted execution limits refused",
            Self::Reentrant => "hosted synchronous re-entry refused",
            Self::ResourceDomain => "hosted model belongs to a different resource domain",
            Self::ModelIdentity => "hosted task does not match the resident model",
            Self::SingleCoordinatorRequired => "portable hosted execution requires one blocking coordinator",
            Self::MissingRuntimeContext => "hosted execution requires the configured runtime context",
            Self::Reservation(_) => "hosted process memory reservation refused",
            Self::Spawn(_) => "hosted blocking work admission refused",
            Self::Join { .. } => "hosted runtime wrapper failed after physical drain",
            Self::Scope { .. } => "hosted native scope failed after physical drain",
            Self::Stopped { .. } => "hosted execution cancelled or exhausted its budget",
            Self::Panicked => "hosted execution panicked and drained",
            Self::CompletionMissing => "hosted execution produced no physical completion",
            Self::Model(_) => "current-candidate model loading failed",
            Self::Native(_) => "hosted native engine construction failed",
            Self::Chat(_) => "hosted native generation or chat failed",
            Self::Extraction(_) => "hosted native extraction failed",
            Self::Classification(_) => "hosted native classification failed",
            Self::Judge(_) => "hosted native judgment failed",
            Self::Source(_) => "hosted native source task failed",
            Self::SourceMap(_) => "hosted native source map failed",
            Self::Resolution(_) => "hosted native entity resolution failed",
            Self::BatchSetup(_) => "hosted corpus setup refused",
            Self::Batch(_) => "hosted corpus failed; inspect the retained summary",
        })
    }
}
impl fmt::Debug for HostedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Display::fmt(self, f) }
}
impl Error for HostedError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Reservation(e) => Some(e), Self::Spawn(e) => Some(e),
            Self::Join { source, .. } => Some(source), Self::Scope { source, .. } => Some(source),
            Self::Model(e) => Some(e), Self::Native(e) => Some(e), Self::Chat(e) => Some(e),
            Self::SourceMap(e) => Some(e), Self::Resolution(e) => Some(e),
            Self::Source(e) => Some(e), Self::Judge(e) => Some(e), Self::Classification(e) => Some(e), Self::Extraction(e) => Some(e), Self::BatchSetup(e) => Some(e), Self::Batch(e) => Some(e), _ => None,
        }
    }
}
impl From<ReservationError> for HostedError { fn from(e: ReservationError) -> Self { Self::Reservation(e) } }

/// A reservation that is explicitly aborted on every non-commit path. Drop
/// does not trigger the underlying ledger's forgotten-obligation leak policy.
struct Pending(Option<MemoryReservation>);
impl Pending {
    fn reserve(lease: &EngineLease, class: MemoryClass, bytes: u64) -> Result<Self, HostedError> {
        if bytes == 0 { return Err(HostedError::Limits("zero memory reservation")); }
        Ok(Self(Some(lease.reserve(class, bytes)?)))
    }
    fn commit(mut self) -> Result<CommittedMemory, HostedError> {
        self.0.take().ok_or(HostedError::CompletionMissing)?.commit().map_err(Into::into)
    }
}
impl Drop for Pending {
    fn drop(&mut self) {
        if let Some(reservation) = self.0.take() {
            if reservation.abort().is_err() { eprintln!("HOSTED_MEMORY ABORT_INVARIANT_FAILURE"); }
        }
    }
}
/// Declaration order is the ownership proof: storage drops before its charge.
struct Charged<T> { value: T, _memory: CommittedMemory }
fn allocate<T>(claim: Pending, build: impl FnOnce() -> Result<T, HostedError>) -> Result<Charged<T>, HostedError> {
    let value = build()?;
    match claim.commit() {
        Ok(memory) => Ok(Charged { value, _memory: memory }),
        Err(error) => { drop(value); Err(error) }
    }
}

struct ResidentModel {
    loaded: Charged<CurrentCandidateInt8Model>,
    // Retain the real runtime domain even after the loading NlpEngine drops.
    lease: EngineLease,
}
/// Cloning shares the same weights AND their one ledger charge; it never copies
/// model tensors. There is no constructor from uncharged materialized weights.
#[derive(Clone)]
pub struct ResidentInt8 { inner: Arc<ResidentModel> }
impl ResidentInt8 {
    pub fn artifact_identity(&self) -> &ArtifactIdentity { self.inner.loaded.value.artifact_identity() }
    pub fn resources(&self) -> &Arc<EngineResources> { self.inner.lease.resources() }
}

/// The result retains output memory authority after temporary native buffers
/// drain. Serialize writes only T. There is deliberately no unguarded unwrap.
pub type HostedOutput<T> = GuardedOutput<T, CommittedMemory>;

impl NlpEngine {
    /// Stream an explicitly selected local candidate file on the real blocking
    /// pool. No download, active-catalog lookup, default-model substitution or
    /// publisher-authenticity claim is performed. Loading is not interruptible
    /// inside the existing reader; cancellation is observed before/after it,
    /// and the synchronous method still waits for its physical cleanup.
    pub fn load_current_candidate_int8(&self, path: PathBuf, limits: LoadLimits,
        cancellation: CancellationToken) -> Result<ResidentInt8, HostedError> {
        dispatch::preflight(self, limits.run)?;
        if limits.artifact.max_weight_bytes == 0 || limits.artifact.max_stream_chunk_bytes == 0
            || limits.artifact.max_resident_envelope_bytes != 0 || limits.tokenizer_and_metadata_bytes == 0
            || limits.allocator_reserve_bytes == 0 {
            return Err(HostedError::Limits("explicit streaming-only model allocation envelope required"));
        }
        let resident_bytes = sum(&[limits.artifact.max_weight_bytes, limits.tokenizer_and_metadata_bytes,
            limits.allocator_reserve_bytes])?;
        let lease = self.resources().acquire_lease();
        let resident = Pending::reserve(&lease, MemoryClass::Weights, resident_bytes)?;
        let staging = Pending::reserve(&lease, MemoryClass::Staging, limits.artifact.max_stream_chunk_bytes)?;
        dispatch::run(self, limits.run, cancellation, move |_control| {
            let loaded = allocate(resident, || CurrentCandidateInt8Model::load(path, limits.artifact, |_| {})
                .map_err(HostedError::Model))?;
            drop(staging);
            Ok(ResidentInt8 { inner: Arc::new(ResidentModel { loaded, lease }) })
        })
    }

    /// Execute an already compiled generate/chat plan. Plans and their pinned
    /// tokenizer remain caller-owned preparation allocations until transferred;
    /// this method does not re-tokenize or silently alter the admitted identity.
    pub fn execute_int8_chat(&self, model: &ResidentInt8, prepared: PreparedInt8Chat,
        request_seq: u64, native: NativeLimits, max_sampler_bytes: u64,
        cancellation: CancellationToken) -> Result<HostedOutput<Int8ChatResult>, HostedError> {
        dispatch::preflight(self, native.run)?;
        self.check_resident_domain(model)?;
        check_model_identity(model.artifact_identity(), prepared.execution_identity())?;
        if request_seq == 0 || max_sampler_bytes == 0 { return Err(HostedError::Limits("sequence/sampler")); }
        let required = requirements(native)?;
        let work = prepared.planned_work();
        if work.forward_positions > native.context_tokens as u64
            || required.kv_bytes > prepared.task_plan().ir().budget().max_kv_bytes
            || prepared.native_plan().sampler_bytes() > max_sampler_bytes {
            return Err(HostedError::Limits("native task preflight"));
        }
        let lease = self.resources().acquire_lease();
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
        let workspace = Pending::reserve(&lease, MemoryClass::ActivationScratch,
            sum(&[required.rope_bytes, required.scratch_payload_bound,
                max_sampler_bytes, native.allocator_reserve_bytes])?)?;
        let output = output_claim(&lease, prepared.task_plan().ir().budget().max_output_bytes,
            u64::from(prepared.task_plan().ir().budget().max_output_tokens))?;
        let model = model.clone();
        dispatch::run(self, native.run, cancellation, move |control| {
            let mut engine = allocate_native(kv, workspace, || model.inner.loaded.value
                .engine(native.context_tokens, memory_budget(required)).map_err(HostedError::Model))?;
            let budget = Int8GenerationBudget { native: Int8RunBudget::exact(work),
                max_kv_bytes: required.kv_bytes, max_sampler_bytes };
            let result = prepared.execute(prepared.execution_identity(), &mut engine.value, request_seq, budget, control)
                .map_err(HostedError::Chat)?;
            // Complete native storage drains before the output becomes visible.
            drop(engine);
            let committed = output.commit()?;
            drop(lease);
            Ok(GuardedOutput::new(result, committed))
        })
    }

    /// Execute a compiled schema/source task through the same pool and ledger.
    /// The vocabulary is caller-provided immutable preparation, not rebuilt for
    /// every document. It remains owned until the physical invocation drains.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int8_extract(&self, model: &ResidentInt8, prepared: Int8ExtractPlan,
        vocabulary: Arc<ExtractionVocabulary>, native: NativeLimits,
        mask_limits: crate::grammar::mask::MaskWorkLimits, max_mask_node_visits: u64,
        cancellation: CancellationToken) -> Result<HostedOutput<Int8ExtractRun>, HostedError> {
        dispatch::preflight(self, native.run)?;
        self.check_resident_domain(model)?;
        check_model_identity(model.artifact_identity(), prepared.execution_identity())?;
        let required = requirements(native)?;
        let work = prepared.planned_work();
        if work.forward_positions > native.context_tokens as u64 || max_mask_node_visits == 0
            || mask_limits.max_trie_node_visits == 0 || mask_limits.checkpoint_interval_nodes == 0 {
            return Err(HostedError::Limits("native extraction preflight"));
        }
        let lease = self.resources().acquire_lease();
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
        let workspace = Pending::reserve(&lease, MemoryClass::ActivationScratch,
            sum(&[required.rope_bytes, required.scratch_payload_bound, native.allocator_reserve_bytes])?)?;
        let output = output_claim(&lease, prepared.max_result_bytes(), prepared.options().max_new_tokens as u64)?;
        let model = model.clone();
        dispatch::run(self, native.run, cancellation, move |control| {
            let mut engine = allocate_native(kv, workspace, || model.inner.loaded.value
                .engine(native.context_tokens, memory_budget(required)).map_err(HostedError::Model))?;
            let budget = Int8JsonBudget { native: Int8RunBudget::exact(work), json: crate::native_engine::constrained::JsonWorkBudget {
                max_forward_positions: work.forward_positions, max_projected_logits: work.projected_logits,
                max_kv_bytes: required.kv_bytes, max_total_mask_node_visits: max_mask_node_visits, mask_limits,
            } };
            let result = prepared.execute(&mut engine.value, prepared.execution_identity(), &vocabulary, budget, control)
                .map_err(HostedError::Extraction)?;
            drop(engine);
            let committed = output.commit()?;
            drop(lease);
            Ok(GuardedOutput::new(result, committed))
        })
    }
    fn check_resident_domain(&self, model: &ResidentInt8) -> Result<(), HostedError> {
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        Ok(())
    }
}
fn requirements(native: NativeLimits) -> Result<Int8MemoryRequirement, HostedError> {
    if native.allocator_reserve_bytes == 0 { return Err(HostedError::Limits("allocator reserve")); }
    Int8MemoryRequirement::for_context(native.context_tokens).map_err(HostedError::Native)
}
fn memory_budget(required: Int8MemoryRequirement) -> Int8MemoryBudget {
    Int8MemoryBudget { max_kv_bytes: required.kv_bytes, max_rope_bytes: required.rope_bytes,
        max_scratch_payload_bytes: required.scratch_payload_bound }
}
fn sum(values: &[u64]) -> Result<u64, HostedError> {
    values.iter().try_fold(0_u64, |total, &value| total.checked_add(value).ok_or(HostedError::Limits("memory arithmetic")))
}
fn output_claim(lease: &EngineLease, bytes: u64, tokens: u64) -> Result<Pending, HostedError> {
    // Typed result payload plus bounded canonical tree/byte staging and token
    // vectors. Allocator slack is separately priced by NativeLimits. This is a
    // conservative modeled commitment, not an observed or enforced RSS limit.
    let bytes = bytes.checked_mul(4).ok_or(HostedError::Limits("output arithmetic"))?;
    let tokens = tokens.checked_mul(8).ok_or(HostedError::Limits("token arithmetic"))?;
    Pending::reserve(lease, MemoryClass::JobBuffers, sum(&[bytes, tokens, 4096])?)
}

// Keep the native allocation ahead of BOTH committed component charges.
struct ChargedNative<T> { value: T, _kv: CommittedMemory, _scratch: CommittedMemory }
fn allocate_native<T>(kv: Pending, scratch: Pending, build: impl FnOnce() -> Result<T, HostedError>)
    -> Result<ChargedNative<T>, HostedError> {
    let value = build()?;
    let kv = match kv.commit() { Ok(kv) => kv, Err(error) => { drop(value); return Err(error); } };
    match scratch.commit() {
        Ok(scratch) => Ok(ChargedNative { value, _kv: kv, _scratch: scratch }),
        Err(error) => { drop(value); drop(kv); Err(error) }
    }
}
fn check_model_identity(model: &ArtifactIdentity, identity: &ExecutionIdentity) -> Result<(), HostedError> {
    if model.model_id != "Nanbeige4.2-3B" || model.revision != identity.source_revision
        || model.recipe_id != identity.quant_recipe
        || Sha256Digest::from_hex(&model.logical_model_sha256).ok() != Some(identity.logical_model_digest) {
        return Err(HostedError::ModelIdentity);
    }
    Ok(())
}

#[cfg(test)] mod tests;
