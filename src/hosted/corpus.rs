//! Entire bounded corpus streams on one process-owned native INT8 invocation.
//! No user-defined admission provider, per-document runtime or model reload.

use super::*;
mod classify;
pub use classify::ClassificationCorpusConfig;
use std::io::{BufRead, Write};
use crate::{
    batch::{self, BatchCode, BatchItemFailure, BatchLimits, BatchSummary,
        generation::{GenerationBatchArgs, quantized::{Int8BatchLimits,
            Int8GenerationAdmission, Int8GenerationBatchAdmission, Int8GenerationBatchPlanner,
            NativeInt8GenerationBatch}},
        extract::quantized::{Int8ExtractionAdmission, Int8ExtractionBatchAdmission,
            Int8ExtractionBatchLimits, Int8ExtractionBatchPlanner, NativeInt8ExtractionBatch}},
    native_engine::{lmhead::NANBEIGE_VOCAB_SIZE, strict_int8::Int8Work},
    tasks::chat::quantized::Int8ChatPlanner,
};

/// Whole-run transport and preparation commitment. Native limits never renew
/// at a document or epoch boundary. Preparation/IO estimates must account for
/// the embedder's selected compiler and owned reader/writer; these values are
/// reservations, not an allocator interceptor or an observed RSS bound.
#[derive(Clone, Copy, Debug)]
pub struct CorpusLimits {
    pub native: NativeLimits,
    pub transport: BatchLimits,
    /// Caller-priced schema/source/TaskIR compilation and task-argument copies.
    pub preparation_reserve_bytes: u64,
    /// Buffers retained by the supplied reader/writer (including external
    /// buffered sinks). A cursor over a whole corpus must price the whole data.
    pub io_reserve_bytes: u64,
}
impl CorpusLimits {
    fn reservation_bytes(self) -> Result<u64, HostedError> {
        self.native.run.validate()?;
        self.transport.validate().map_err(HostedError::BatchSetup)?;
        if self.preparation_reserve_bytes == 0 || self.io_reserve_bytes == 0 {
            return Err(HostedError::Limits("explicit preparation and IO reservations required"));
        }
        let t = self.transport;
        // Fixed conservative staging model: input/JSON copies, canonical
        // output staging and epoch-id tree bookkeeping. Compiler/allocator
        // overhead remains explicit, never inferred from an input byte count.
        let lines = (t.max_line_bytes as u64).checked_mul(8);
        let output = (t.max_output_line_bytes as u64).checked_mul(4);
        let ids = (t.max_epoch_ids as u64).checked_mul(128);
        sum(&[lines.ok_or(HostedError::Limits("input staging arithmetic"))?,
            output.ok_or(HostedError::Limits("output staging arithmetic"))?,
            ids.ok_or(HostedError::Limits("ID staging arithmetic"))?,
            t.max_epoch_id_bytes as u64, self.preparation_reserve_bytes, self.io_reserve_bytes])
    }
}

struct StreamInput<P, R, W> { planner: Option<P>, reader: R, writer: W }

impl NlpEngine {
    /// Own a bounded ordered NDJSON generation/chat run. The supplied planner
    /// and defaults retain their pinned semantics. One native engine stays
    /// resident for the whole run; output guards are provided by this host.
    /// Readers/writers move into the blocking closure and are dropped before
    /// physical completion. Arbitrary blocking IO is not safely preemptible.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_chat<R, W>(&self, model: &ResidentInt8,
        planner: Arc<Int8ChatPlanner>, defaults: Option<GenerationBatchArgs>,
        native_work: Int8BatchLimits, limits: CorpusLimits, reader: R, writer: W,
        cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        dispatch::preflight(self, limits.native.run)?;
        self.check_resident_domain(model)?;
        validate_work(native_work.max_model_work)?;
        if native_work.max_sampler_bytes == 0 { return Err(HostedError::Limits("sampler reservation")); }
        let required = requirements(limits.native)?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, limits.reservation_bytes()?)?,
            || Ok(StreamInput { planner: Some((planner, defaults)), reader, writer }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
        let workspace = Pending::reserve(&lease, MemoryClass::ActivationScratch,
            sum(&[required.rope_bytes, required.scratch_payload_bound, native_work.max_sampler_bytes,
                limits.native.allocator_reserve_bytes])?)?;
        let model = model.clone();
        dispatch::run(self, limits.native.run, cancellation, move |control| {
            let mut input = input;
            let (planner, defaults) = input.value.planner.take().ok_or(HostedError::CompletionMissing)?;
            let compiler = Int8GenerationBatchPlanner::new(&planner, defaults).map_err(HostedError::BatchSetup)?;
            let mut engine = allocate_native(kv, workspace, || model.inner.loaded.value
                .engine(limits.native.context_tokens, memory_budget(required)).map_err(HostedError::Model))?;
            let admission = CorpusAdmission { lease: &lease, model: model.artifact_identity(),
                kv_bytes: required.kv_bytes, sampler_bytes: native_work.max_sampler_bytes,
                output_bytes: limits.transport.max_output_line_bytes as u64 };
            let result = {
                let mut processor = NativeInt8GenerationBatch::new(compiler, &mut engine.value, admission, native_work)
                    .map_err(HostedError::BatchSetup)?;
                batch::run_ndjson(&mut input.value.reader, &mut input.value.writer,
                    &mut processor, limits.transport, control).map_err(HostedError::Batch)
            };
            drop(engine);
            drop(planner);
            drop(input); // writer/reader before their memory charge and physical handoff
            drop(lease);
            result
        })
    }

    /// Schema/source extraction using a single resident engine/vocabulary.
    /// Exact decimal JSON, source occurrence validation and nonrenewable native
    /// and mask budgets stay with the existing typed extraction adapter.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_extract<R, W>(&self, model: &ResidentInt8,
        planner: Int8ExtractionBatchPlanner, vocabulary: Arc<ExtractionVocabulary>,
        native_work: Int8ExtractionBatchLimits, limits: CorpusLimits, reader: R, writer: W,
        cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        dispatch::preflight(self, limits.native.run)?;
        self.check_resident_domain(model)?;
        validate_work(native_work.max_model_work)?;
        let masks = native_work.masks;
        if masks.per_mask.max_trie_node_visits == 0 || masks.per_mask.checkpoint_interval_nodes == 0
            || masks.max_visits_per_item == 0 || masks.max_visits_per_run < masks.max_visits_per_item {
            return Err(HostedError::Limits("mask reservation"));
        }
        let required = requirements(limits.native)?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, limits.reservation_bytes()?)?,
            || Ok(StreamInput { planner: Some((planner, vocabulary)), reader, writer }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
        let workspace = Pending::reserve(&lease, MemoryClass::ActivationScratch,
            sum(&[required.rope_bytes, required.scratch_payload_bound, limits.native.allocator_reserve_bytes])?)?;
        let model = model.clone();
        dispatch::run(self, limits.native.run, cancellation, move |control| {
            let mut input = input;
            let (planner, vocabulary) = input.value.planner.take().ok_or(HostedError::CompletionMissing)?;
            let mut engine = allocate_native(kv, workspace, || model.inner.loaded.value
                .engine(limits.native.context_tokens, memory_budget(required)).map_err(HostedError::Model))?;
            let admission = CorpusAdmission { lease: &lease, model: model.artifact_identity(),
                kv_bytes: required.kv_bytes, sampler_bytes: 0,
                output_bytes: limits.transport.max_output_line_bytes as u64 };
            let result = {
                let mut processor = NativeInt8ExtractionBatch::new(planner, &mut engine.value, &vocabulary, admission, native_work)
                    .map_err(HostedError::BatchSetup)?;
                batch::run_ndjson(&mut input.value.reader, &mut input.value.writer,
                    &mut processor, limits.transport, control).map_err(HostedError::Batch)
            };
            drop(engine);
            drop(vocabulary);
            drop(input);
            drop(lease);
            result
        })
    }
}

fn validate_work(work: Int8Work) -> Result<(), HostedError> {
    if work.forward_positions == 0 || work.projected_logits == 0 || work.attention_pairs == 0
        || work.projections.dot_products == 0 || work.projections.multiply_accumulates == 0 {
        return Err(HostedError::Limits("complete native work reservation"));
    }
    Ok(())
}

// Both concrete adapters consume this same process-ledger authority. The
// engine's full KV/sampler workspace is ALREADY charged for this corpus, so
// per-document admission verifies it instead of repeatedly charging copies.
struct CorpusAdmission<'a> {
    lease: &'a EngineLease, model: &'a ArtifactIdentity,
    kv_bytes: u64, sampler_bytes: u64, output_bytes: u64,
}
impl CorpusAdmission<'_> {
    fn admit_output(&self, identity: &ExecutionIdentity, work: Int8Work,
        kv: u64, sampler: u64, output: u64) -> Result<(ExecutionIdentity, Pending), BatchItemFailure> {
        check_request(self.model, self.kv_bytes, self.sampler_bytes, self.output_bytes,
            identity, work, kv, sampler, output)?;
        let tokens = work.projected_logits / NANBEIGE_VOCAB_SIZE as u64;
        // Keep the output as a live reservation through allocation AND write/
        // flush. The adapter cannot prove allocation at admission, so it must
        // not fabricate a post-allocation commit. Pending's explicit Drop abort
        // discharges the reservation only after GuardedOutput drops its bytes.
        let guard = output_claim(self.lease, output, tokens).map_err(admission_failure)?;
        Ok((identity.clone(), guard))
    }
}
impl Int8GenerationBatchAdmission for CorpusAdmission<'_> {
    type Guard = Pending;
    fn admit(&mut self, request: Int8GenerationAdmission<'_>) -> Result<(ExecutionIdentity, Self::Guard), BatchItemFailure> {
        self.admit_output(request.identity, request.model_work, request.kv_reservation_bytes,
            request.sampler_bytes, request.max_result_bytes)
    }
}
impl Int8ExtractionBatchAdmission for CorpusAdmission<'_> {
    type Guard = Pending;
    fn admit(&mut self, request: Int8ExtractionAdmission<'_>) -> Result<(ExecutionIdentity, Self::Guard), BatchItemFailure> {
        if request.mask_node_visits == 0 || request.mask_limits.max_trie_node_visits == 0
            || request.mask_limits.checkpoint_interval_nodes == 0 {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        self.admit_output(request.identity, request.model_work, request.kv_reservation_bytes, 0, request.max_result_bytes)
    }
}
#[allow(clippy::too_many_arguments)]
fn check_request(model: &ArtifactIdentity, kv_ceiling: u64, sampler_ceiling: u64, output_ceiling: u64,
    identity: &ExecutionIdentity, work: Int8Work, kv: u64, sampler: u64, output: u64) -> Result<(), BatchItemFailure> {
    check_model_identity(model, identity).map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
    // Host authority is this one strict-INT8, thinking-off/no-tools backend.
    crate::native_engine::constrained_int8::check_profile(identity)
        .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
    validate_work(work).map_err(|_| BatchItemFailure::fatal(BatchCode::InvalidExecution))?;
    if work.projected_logits % NANBEIGE_VOCAB_SIZE as u64 != 0 {
        return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
    }
    if kv != kv_ceiling || sampler > sampler_ceiling {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    if output == 0 || output > output_ceiling {
        return Err(BatchItemFailure::reject(BatchCode::OutputLineLimit));
    }
    Ok(())
}
fn admission_failure(error: HostedError) -> BatchItemFailure {
    // Host failure is terminal. Continuing cannot create another independent
    // budget or conceal process-wide exhaustion as a string of document errors.
    match error {
        HostedError::Reservation(_) => BatchItemFailure::fatal(BatchCode::Admission),
        _ => BatchItemFailure::fatal(BatchCode::InvalidExecution),
    }
}

#[cfg(test)] mod tests;
