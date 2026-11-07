//! Complete-document schema extraction on the ordered live NDJSON runner.
//! Each item owns its whole lossless map; flushes never renew native, mask or
//! original-coordinate verification allowances. Durable jobs remain separate.
use super::*;
use crate::{
    batch::{BatchProcessor, BatchRequestContext, BatchWork},
    native_engine::{int4::PackedInt4Kernel,
        strict_int4::{Int4MemoryRequirement, Int4ModelIdentity, Int4ModelWeights}},
};
use super::cohort::{Int4ExtractionMapCohortBudget, Int4ExtractionMapCohortMemoryLimits,
    Int4ExtractionMapCohortRequirement};

pub mod memory;
mod backend;
use backend::Backend;

/// Frozen, reconstructed and admitted identity ownership per possible chunk.
/// Preparation separately owns every complete source/schema/native plan.
pub const EXTRACTION_DOCUMENT_CHUNK_METADATA_BYTES: u64 = 32_768;

/// Record arguments can only be an empty JSON object. In particular, serde's
/// sequence representation of an empty struct is not accepted as a policy.
#[derive(Default)]
pub struct ExtractionDocumentArgs {}
impl<'de> Deserialize<'de> for ExtractionDocumentArgs {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct EmptyMap;
        impl<'de> serde::de::Visitor<'de> for EmptyMap {
            type Value = ExtractionDocumentArgs;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("an empty object of document arguments")
            }
            fn visit_map<M: serde::de::MapAccess<'de>>(self, mut map: M) -> Result<Self::Value, M::Error> {
                if map.next_key::<String>()?.is_some() {
                    return Err(serde::de::Error::custom("document arguments cannot override the shared extraction policy"));
                }
                Ok(ExtractionDocumentArgs {})
            }
        }
        deserializer.deserialize_map(EmptyMap)
    }
}

/// Fixed exact schema and per-document policy, with independent whole-run caps.
/// The complete per-document verification allowance is reserved on every
/// execution attempt, including attempts that fail before producing output.
#[derive(Clone)]
pub struct Int4ExtractionDocumentBatchConfig {
    pub request: ExtractionBatchArgs,
    pub limits: Int4ExtractionMapLimits,
    pub max_model_work: Int4Work,
    pub max_mask_visits: u64,
    pub max_verification: GroundingBudget,
}
impl Int4ExtractionDocumentBatchConfig {
    pub(crate) fn validate(&self) -> Result<(), BatchFault> {
        self.request.budget.validate().map_err(|_| BatchCode::InvalidLimits)?;
        let m = self.limits.mapping; let v = self.limits.verification;
        m.chunks.effective_token_limit().map_err(|_| BatchCode::InvalidLimits)?;
        if self.request.schema.is_empty() || m.chunks.max_input_bytes == 0
            || !(1..=256).contains(&m.chunks.max_chunks) || !valid_reduction(m.reduction)
            || !positive_work(m.max_model_work) || !fits(m.max_model_work, self.max_model_work)
            || m.mask_limits.max_trie_node_visits == 0 || m.mask_limits.checkpoint_interval_nodes == 0
            || m.mask_visits_per_chunk == 0 || m.max_mask_visits < m.mask_visits_per_chunk
            || m.max_mask_visits > self.max_mask_visits
            || !(1..=1_000_000).contains(&v.max_fields) || !(1..=1_000_000).contains(&v.max_matches)
            || v.max_scan_steps == 0 || !verification_fits(v, self.max_verification) {
            return Err(BatchCode::InvalidLimits.into());
        }
        Ok(())
    }
}

/// Sealed factory for this live population, bound to actual loaded weights.
/// Records cannot replace the schema, grounding, compiler, budgets or schedule.
pub struct Int4ExtractionDocumentBatchPlanner {
    compiler: Int4ExtractionBatchPlanner,
    config: Int4ExtractionDocumentBatchConfig,
    chunk_cohorts: bool,
    owner: Arc<()>,
}
impl Int4ExtractionDocumentBatchPlanner {
    pub fn for_model(model: &Int4ModelIdentity, ceiling: TaskBudget, compiler: CompileLimits,
        source: SourceRuntimeLimits, config: Int4ExtractionDocumentBatchConfig) -> Result<Self, BatchFault> {
        config.validate()?;
        // The transferred schema keeps its original allocation. The base
        // compiler needs no second retained default-schema copy.
        let compiler = Int4ExtractionBatchPlanner::for_model(model, ceiling, compiler, source, None)?;
        Self::configured(compiler, config)
    }
    fn configured(compiler: Int4ExtractionBatchPlanner, config: Int4ExtractionDocumentBatchConfig)
        -> Result<Self, BatchFault> {
        config.validate()?;
        let ceiling = compiler.compiler.ceiling; let request = config.request.budget;
        ceiling.validate().map_err(|_| BatchCode::InvalidLimits)?;
        let kv = Int4MemoryRequirement::for_context(config.limits.mapping.chunks.context_tokens,
            compiler.group_width()).map_err(|_| BatchCode::InvalidLimits)?.kv_bytes;
        if compiler.compiler.defaults.is_some() || compiler.shared_defaults_only
            || config.request.schema.len() > compiler.compiler.compiler_limits.max_schema_bytes
            || request.max_input_tokens > ceiling.max_input_tokens || request.max_output_tokens > ceiling.max_output_tokens
            || request.max_output_bytes > ceiling.max_output_bytes || request.max_grammar_states > ceiling.max_grammar_states
            || request.max_kv_bytes != kv || ceiling.max_kv_bytes != kv {
            return Err(BatchCode::InvalidLimits.into());
        }
        Ok(Self { compiler, config, chunk_cohorts: false, owner: Arc::new(()) })
    }
    pub fn base_execution_identity(&self) -> &ExecutionIdentity { self.compiler.base_execution_identity() }
    pub fn group_width(&self) -> usize { self.compiler.group_width() }
    pub fn task_ceiling(&self) -> TaskBudget { self.compiler.compiler.ceiling }
    pub fn document_budget(&self) -> TaskBudget { self.config.request.budget }
    pub fn request(&self) -> &ExtractionBatchArgs { &self.config.request }
    pub fn document_limits(&self) -> Int4ExtractionMapLimits { self.config.limits }
    pub fn max_run_work(&self) -> Int4Work { self.config.max_model_work }
    pub fn max_mask_visits(&self) -> u64 { self.config.max_mask_visits }
    pub fn max_verification(&self) -> GroundingBudget { self.config.max_verification }
    pub fn uses_chunk_cohorts(&self) -> bool { self.chunk_cohorts }
    pub fn with_chunk_cohorts(mut self) -> Result<Self, BatchFault> {
        self.cohort_requirements()?;
        if !self.chunk_cohorts { self.chunk_cohorts = true; self.owner = Arc::new(()); }
        Ok(self)
    }
    /// The declared complete-window peak stays owned across short documents,
    /// final ragged windows, reduction, serialization and writer flush.
    pub fn cohort_requirements(&self) -> Result<Int4ExtractionMapCohortRequirement, BatchFault> {
        let m = self.config.limits.mapping;
        if m.reduction.map_batch_chunks > m.chunks.max_chunks { return Err(BatchCode::InvalidLimits.into()); }
        Int4ExtractionMapCohortMemoryLimits { max_chunks: m.reduction.map_batch_chunks,
            context_tokens: m.chunks.context_tokens,
            max_output_tokens_per_chunk: u64::from(self.document_budget().max_output_tokens),
            max_result_bytes_per_chunk: self.document_budget().max_output_bytes,
        }.requirements(self.group_width()).map_err(|e| map_failure(e, true).fault)
    }
    pub fn max_result_bytes(&self) -> u64 { self.config.limits.mapping.reduction.max_result_bytes as u64 }
    pub fn output_token_ceiling(&self) -> Result<u64, BatchFault> {
        self.output_tokens(self.config.limits.mapping.chunks.max_chunks)
    }
    /// Price the allocation actually transferred by the caller, before any
    /// clone could hide an oversized schema capacity from host admission.
    pub fn retained_defaults_bytes(&self) -> Result<u64, BatchFault> {
        self.config.request.schema.capacity().checked_add(std::mem::size_of::<ExtractionBatchArgs>())
            .and_then(|n| u64::try_from(n).ok()).ok_or_else(|| BatchCode::InvalidLimits.into())
    }
    fn output_tokens(&self, chunks: usize) -> Result<u64, BatchFault> {
        u64::try_from(chunks).ok().and_then(|n| n.checked_mul(u64::from(self.document_budget().max_output_tokens)))
            .ok_or_else(|| BatchCode::InvalidLimits.into())
    }
    fn plan<'s, C: DecodeStepControl>(&self, source: &'s str, control: &mut C, fatal: bool)
        -> Result<PreparedInt4ExtractionMap<'s>, BatchItemFailure> {
        poll(control)?;
        if source.is_empty() || source.len() > self.config.limits.mapping.chunks.max_input_bytes {
            return Err(if fatal { BatchItemFailure::fatal(BatchCode::DocumentLimit) }
                else { BatchItemFailure::reject(BatchCode::DocumentLimit) });
        }
        self.compiler.plan_document_with_control(source, self.request(), self.config.limits, control)
            .map_err(|e| map_failure(e, fatal))
    }
    fn snapshot<C: DecodeStepControl>(&self, plan: &PreparedInt4ExtractionMap<'_>, control: &mut C)
        -> Result<(Int4ExtractionMapPreflight, Vec<ExecutionIdentity>), BatchItemFailure> {
        let expected = plan.preflight_metadata();
        if expected.chunk_count() == 0 || expected.chunk_count() > self.config.limits.mapping.chunks.max_chunks
            || expected.group_width() != self.group_width() || plan.max_kv_bytes() != self.document_budget().max_kv_bytes
            || plan.max_result_bytes() != self.max_result_bytes() || plan.execution_identities().len() != expected.chunk_count()
            || !fits(expected.planned_work(), self.config.max_model_work)
            || expected.reserved_mask_visits() > self.config.max_mask_visits {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        let mut identities = Vec::new();
        identities.try_reserve_exact(expected.chunk_count()).map_err(|_| BatchItemFailure::fatal(BatchCode::Allocation))?;
        for identity in plan.execution_identities() { poll(control)?; identities.push(identity.clone()); }
        Ok((expected, identities))
    }
    fn owns(&self, prepared: &PreparedInt4ExtractionDocument) -> Result<(), BatchItemFailure> {
        if !Arc::ptr_eq(&self.owner, &prepared.owner) { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
        Ok(())
    }
    fn compare<C: DecodeStepControl>(&self, plan: &PreparedInt4ExtractionMap<'_>,
        prepared: &PreparedInt4ExtractionDocument, control: &mut C) -> Result<(), BatchItemFailure> {
        self.owns(prepared)?;
        if plan.preflight_metadata() != prepared.expected || plan.max_kv_bytes() != self.document_budget().max_kv_bytes
            || plan.max_result_bytes() != self.max_result_bytes()
            || plan.execution_identities().len() != prepared.identities.len() {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        for (actual, expected) in plan.execution_identities().zip(&prepared.identities) {
            poll(control)?;
            if actual != expected { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
        }
        poll(control)
    }
}

/// Exact original source plus private commitments to every ordered chunk.
/// Metadata alone cannot authorize another source, schema, factory or schedule.
pub struct PreparedInt4ExtractionDocument {
    source: String,
    expected: Int4ExtractionMapPreflight,
    identities: Vec<ExecutionIdentity>,
    owner: Arc<()>,
}
impl PreparedInt4ExtractionDocument {
    pub fn chunk_count(&self) -> usize { self.expected.chunk_count() }
    pub fn planned_work(&self) -> Int4Work { self.expected.planned_work() }
    pub fn reserved_mask_visits(&self) -> u64 { self.expected.reserved_mask_visits() }
    pub fn execution_identities(&self) -> &[ExecutionIdentity] { &self.identities }
}
pub struct Int4ExtractionDocumentAdmission<'a> {
    pub identities: &'a [ExecutionIdentity],
    pub model_work: Int4Work,
    pub mask_node_visits: u64,
    pub mask_limits: MaskWorkLimits,
    pub kv_reservation_bytes: u64,
    pub max_result_bytes: u64,
    pub max_output_tokens: u64,
    pub chunk_count: usize,
}
pub trait Int4ExtractionDocumentBatchAdmission {
    type Guard;
    fn admit(&mut self, request: Int4ExtractionDocumentAdmission<'_>)
        -> Result<(Vec<ExecutionIdentity>, Self::Guard), BatchItemFailure>;
}

/// One closed native schedule for the entire live population. No public driver
/// or completed-output injection can replace actual schema/source execution.
pub struct NativeInt4DocumentExtractionBatch<'p, 'e, 'w, 'v, A: Int4ExtractionDocumentBatchAdmission> {
    planner: &'p Int4ExtractionDocumentBatchPlanner,
    backend: Backend<'e, 'w>,
    vocabulary: &'v ExtractionVocabulary,
    admission: A,
    ledger: Ledger,
}
impl<'p, 'e, 'w, 'v, A: Int4ExtractionDocumentBatchAdmission>
    NativeInt4DocumentExtractionBatch<'p, 'e, 'w, 'v, A> {
    pub fn new(planner: &'p Int4ExtractionDocumentBatchPlanner, engine: &'e mut StrictInt4Engine<'w>,
        vocabulary: &'v ExtractionVocabulary, admission: A) -> Result<Self, BatchFault> {
        let backend = Backend::Serial(engine);
        backend.check_capacity(planner).map_err(|e| e.fault)?;
        Ok(Self { planner, backend, vocabulary, admission, ledger: Ledger::default() })
    }
    pub fn new_cohorts(planner: &'p Int4ExtractionDocumentBatchPlanner, weights: &'w Int4ModelWeights,
        vocabulary: &'v ExtractionVocabulary, budget: Int4ExtractionMapCohortBudget,
        kernel: PackedInt4Kernel, admission: A) -> Result<Self, BatchFault> {
        let backend = Backend::Cohorts { weights, budget, kernel };
        backend.check_capacity(planner).map_err(|e| e.fault)?;
        Ok(Self { planner, backend, vocabulary, admission, ledger: Ledger::default() })
    }
    pub fn reserved_model_work(&self) -> Int4Work { self.ledger.work }
    pub fn reserved_mask_visits(&self) -> u64 { self.ledger.masks }
    pub fn reserved_verification(&self) -> GroundingBudget { self.ledger.verification }
    pub fn is_poisoned(&self) -> bool { self.ledger.failed || !self.backend.clean() }
}
impl<A: Int4ExtractionDocumentBatchAdmission> BatchProcessor
    for NativeInt4DocumentExtractionBatch<'_, '_, '_, '_, A> {
    type Args = ExtractionDocumentArgs;
    type Prepared = PreparedInt4ExtractionDocument;
    type Output = GuardedOutput<Int4ExtractionMapRun, A::Guard>;
    fn prepare(&mut self, _: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        self.ledger.failed = true;
        Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>, control: &mut C)
        -> Result<Self::Prepared, BatchItemFailure> {
        self.ledger.ready()?;
        self.ledger.failed = true;
        let result = (|| {
            poll(control)?;
            self.backend.check_capacity(self.planner)?;
            let plan = self.planner.plan(&document.text, control, false)?;
            self.backend.preflight(self.planner, &plan, self.vocabulary, control)?;
            let (expected, identities) = self.planner.snapshot(&plan, control)?;
            drop(plan);
            poll(control)?;
            Ok(PreparedInt4ExtractionDocument { source: document.text, expected, identities,
                owner: Arc::clone(&self.planner.owner) })
        })();
        // Recoverable input/schema refusals consume no execution allowance.
        // Cancellation, native state faults and unwinds keep the latch closed.
        if !result.as_ref().is_err_and(|e: &BatchItemFailure| e.stop) && self.backend.clean() {
            self.ledger.failed = false;
        }
        result
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork {
        let work = prepared.planned_work();
        BatchWork { forward_positions: work.forward_positions, projected_logits: work.projected_logits }
    }
    fn execute<C: DecodeStepControl>(&mut self, _: Self::Prepared, _: &mut C) -> Result<Self::Output, BatchItemFailure> {
        self.ledger.failed = true;
        Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        // Charge all five actual native axes, every mask and the complete
        // verifier allowance atomically before replan, callbacks or forwarding.
        self.ledger.begin(&self.planner.config, prepared.expected, context)?;
        let result = (|| {
            self.planner.owns(&prepared)?;
            poll(control)?;
            let kv = self.backend.check_capacity(self.planner)?;
            let plan = self.planner.plan(&prepared.source, control, true)?;
            self.planner.compare(&plan, &prepared, control)?;
            self.backend.preflight(self.planner, &plan, self.vocabulary, control)?;
            let (admitted, guard) = self.admission.admit(Int4ExtractionDocumentAdmission {
                identities: &prepared.identities, model_work: prepared.planned_work(),
                mask_node_visits: prepared.reserved_mask_visits(), mask_limits: self.planner.config.limits.mapping.mask_limits,
                kv_reservation_bytes: kv, max_result_bytes: self.planner.max_result_bytes(),
                max_output_tokens: self.planner.output_tokens(prepared.chunk_count()).map_err(BatchItemFailure::fatal)?,
                chunk_count: prepared.chunk_count(),
            })?;
            poll(control)?;
            if admitted != prepared.identities { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
            let output = self.backend.execute(plan, &admitted, self.vocabulary, control)?;
            prepared.expected.verify_completed(&output).map_err(|e| map_failure(e, true))?;
            self.backend.check_capacity(self.planner)?;
            poll(control)?;
            Ok(GuardedOutput::new(output, guard))
        })();
        match result {
            Ok(output) => { self.ledger.failed = false; Ok(output) }
            Err(mut failure) => { failure.stop = true; Err(failure) }
        }
    }
}

fn verification_fits(a: GroundingBudget, b: GroundingBudget) -> bool {
    a.max_fields <= b.max_fields && a.max_matches <= b.max_matches && a.max_scan_steps <= b.max_scan_steps
}
fn add_verification(a: GroundingBudget, b: GroundingBudget) -> Option<GroundingBudget> {
    Some(GroundingBudget { max_fields: a.max_fields.checked_add(b.max_fields)?,
        max_matches: a.max_matches.checked_add(b.max_matches)?, max_scan_steps: a.max_scan_steps.checked_add(b.max_scan_steps)? })
}
struct Ledger { work: Int4Work, masks: u64, verification: GroundingBudget, sequence: u64, failed: bool }
impl Default for Ledger {
    fn default() -> Self {
        Self { work: Int4Work::default(), masks: 0,
            verification: GroundingBudget { max_fields: 0, max_matches: 0, max_scan_steps: 0 }, sequence: 0, failed: false }
    }
}
impl Ledger {
    fn ready(&self) -> Result<(), BatchItemFailure> {
        if self.failed { Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)) } else { Ok(()) }
    }
    fn begin(&mut self, config: &Int4ExtractionDocumentBatchConfig, expected: Int4ExtractionMapPreflight,
        context: BatchRequestContext) -> Result<(), BatchItemFailure> {
        self.ready()?;
        self.failed = true;
        if context.request_seq <= self.sequence || context.epoch == 0 || context.input_line == 0 {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        let work = add_work(self.work, expected.planned_work()).filter(|&w| fits(w, config.max_model_work))
            .ok_or_else(|| BatchItemFailure::fatal(BatchCode::WorkLimit))?;
        let masks = self.masks.checked_add(expected.reserved_mask_visits()).filter(|&n| n <= config.max_mask_visits)
            .ok_or_else(|| BatchItemFailure::fatal(BatchCode::WorkLimit))?;
        let verification = add_verification(self.verification, config.limits.verification)
            .filter(|&v| verification_fits(v, config.max_verification))
            .ok_or_else(|| BatchItemFailure::fatal(BatchCode::WorkLimit))?;
        self.work = work; self.masks = masks; self.verification = verification; self.sequence = context.request_seq;
        Ok(())
    }
}
fn poll(control: &mut impl DecodeStepControl) -> Result<(), BatchItemFailure> {
    match control.prefill_checkpoint(0) {
        Some(cause) => Err(BatchItemFailure::fatal(BatchFault::cancelled(cause))), None => Ok(()),
    }
}
fn map_failure(error: Int4ExtractionMapError, fatal: bool) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    let (code, stop) = match error {
        Int4ExtractionMapError::InvalidLimits => (BatchCode::InvalidLimits, true),
        Int4ExtractionMapError::EmptySource => (BatchCode::DocumentLimit, fatal),
        Int4ExtractionMapError::WorkLimit => (BatchCode::WorkLimit, fatal),
        Int4ExtractionMapError::Admission => (BatchCode::Admission, fatal),
        Int4ExtractionMapError::Allocation => (BatchCode::Allocation, true),
        Int4ExtractionMapError::Chunk(MapReduceError::AllocationRefused) => (BatchCode::Allocation, true),
        Int4ExtractionMapError::Chunk(MapReduceError::InvalidLimits) => (BatchCode::InvalidLimits, true),
        Int4ExtractionMapError::Chunk(MapReduceError::Cancelled | MapReduceError::InvalidSpan
            | MapReduceError::InvalidTokenCount | MapReduceError::Tokenizer) => (BatchCode::Planning, true),
        Int4ExtractionMapError::Chunk(_) => (BatchCode::Planning, fatal),
        Int4ExtractionMapError::Extraction(_) | Int4ExtractionMapError::Reduction(_) => (BatchCode::InvalidExecution, true),
    };
    if stop { BatchItemFailure::fatal(code) } else { BatchItemFailure::reject(code) }
}

#[cfg(test)] mod tests;
