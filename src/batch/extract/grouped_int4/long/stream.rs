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
use super::object::{Int4ObjectExtractionPlanner, Int4ObjectExtractionMapLimits,
    Int4ObjectExtractionMapError, Int4ObjectExtractionMapPreflight,
    Int4ObjectExtractionMapRun, PreparedInt4ObjectExtractionMap};
use crate::tasks::extract::object_reconcile::{RootObjectError, RootObjectLimits, RootObjectSchema};

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
        self.limits.root_array_workspace_bytes().map_err(|_| BatchCode::InvalidLimits)?;
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

/// The complete document retains its selected semantic contract. Untagged
/// serialization keeps the ordinary map wire exactly as before; object mode
/// carries the existing partial-map and independently verified object result.
#[derive(Serialize)]
#[serde(untagged)]
pub enum Int4ExtractionDocumentRun {
    Map(Int4ExtractionMapRun),
    Object(Int4ObjectExtractionMapRun),
}
impl Int4ExtractionDocumentRun {
    pub fn mapped(&self) -> &Int4ExtractionMapRun {
        match self { Self::Map(run) => run, Self::Object(run) => &run.partial_map }
    }
    /// Object reconciliation performs an additional original-source check.
    /// Its complete charge must not be replaced with the partial-map charge.
    pub fn verification_used(&self) -> ExtractionMapVerification {
        match self { Self::Map(run) => run.verification_used, Self::Object(run) => run.verification_used }
    }
}

/// Shared by the live and durable document hosts. The original required
/// schema is never compiled as an independent per-chunk task in object mode.
pub(crate) enum DocumentCompiler {
    Map(Int4ExtractionBatchPlanner),
    Object { planner: Int4ObjectExtractionPlanner, limits: RootObjectLimits },
}
impl DocumentCompiler {
    pub(crate) fn base_planner(&self) -> &Int4ExtractionBatchPlanner {
        match self { Self::Map(planner) => planner, Self::Object { planner, .. } => planner.base_planner() }
    }
    pub(crate) fn root_object_reconciliation(&self) -> Option<RootObjectLimits> {
        match self { Self::Map(_) => None, Self::Object { limits, .. } => Some(*limits) }
    }
    pub(crate) fn document_encoding(&self) -> &UntrustedEncoding { self.base_planner().document_encoding() }
    pub(crate) fn encoding_workspace_bytes(&self) -> Result<u64, BatchFault> {
        crate::batch::extract::grouped_int4::memory::encoding_workspace_bytes(self.document_encoding())
    }
    pub(crate) fn with_document_encoding(self, policy: UntrustedEncoding) -> Result<Self, BatchFault> {
        crate::batch::extract::grouped_int4::memory::encoding_workspace_bytes(&policy)?;
        match self {
            Self::Map(planner) => Ok(Self::Map(planner.with_document_encoding(policy)?)),
            object @ Self::Object { .. } if policy == UntrustedEncoding::ByteFallback => Ok(object),
            Self::Object { .. } => Err(BatchCode::InvalidLimits.into()),
        }
    }
    pub(crate) fn with_selected_rows(self, limits: Int4JsonSparseLimits) -> Result<Self, BatchFault> {
        match self {
            Self::Map(planner) => Ok(Self::Map(planner.with_selected_rows(limits)?)),
            Self::Object { planner, limits: reconciliation } => Ok(Self::Object {
                planner: planner.with_selected_rows(limits)?, limits: reconciliation }),
        }
    }
    pub(crate) fn with_root_object_reconciliation(self, request: &ExtractionBatchArgs,
        mapping: Int4ExtractionMapLimits, reconciliation: RootObjectLimits)
        -> Result<Self, Int4ObjectExtractionMapError> {
        let Self::Map(planner) = self else { return Err(Int4ObjectExtractionMapError::Admission); };
        planner.require_byte_encoding().map_err(|_| Int4ObjectExtractionMapError::Admission)?;
        let limits = Int4ObjectExtractionMapLimits { mapping, reconciliation };
        limits.reconciliation_workspace_bytes()?;
        // This bounded schema-only check occurs before any population read.
        // Every document later repeats it under its original cancellation
        // control and binds the exact original/partial declarations in proof.
        struct Continue;
        impl DecodeStepControl for Continue {
            fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
        }
        let schema = RootObjectSchema::derive(&request.schema, planner.compiler.compiler_limits,
            reconciliation, &mut Continue)?;
        if schema.requires_source() != (request.grounding == ExtractionBatchGrounding::SourceMembership) {
            return Err(Int4ObjectExtractionMapError::Admission);
        }
        Ok(Self::Object { planner: Int4ObjectExtractionPlanner::new(planner)?, limits: reconciliation })
    }
    pub(crate) fn plan<'s, C: DecodeStepControl>(&self, source: &'s str,
        request: &ExtractionBatchArgs, limits: Int4ExtractionMapLimits, control: &mut C)
        -> Result<DocumentPlan<'s>, Int4ObjectExtractionMapError> {
        match self {
            Self::Map(planner) => Ok(DocumentPlan::Map(planner.plan_document_with_control(source, request, limits, control)?)),
            Self::Object { planner, limits: reconciliation } => Ok(DocumentPlan::Object(
                planner.plan_document_with_control(source, request,
                    Int4ObjectExtractionMapLimits { mapping: limits, reconciliation: *reconciliation }, control)?)),
        }
    }
}

/// Private execution plans and complete proofs remain paired by enum variant.
/// Exposing the mapped rows internally allows the existing real native
/// preflight to inspect every partial-schema task before admission or forward.
pub(crate) enum DocumentPlan<'s> {
    Map(PreparedInt4ExtractionMap<'s>),
    Object(PreparedInt4ObjectExtractionMap<'s>),
}
impl DocumentPlan<'_> {
    pub(crate) fn mapped(&self) -> &PreparedInt4ExtractionMap<'_> {
        match self { Self::Map(plan) => plan, Self::Object(plan) => plan.mapped() }
    }
    pub(crate) fn preflight_metadata(&self) -> DocumentPreflight {
        match self { Self::Map(plan) => DocumentPreflight::Map(plan.preflight_metadata()),
            Self::Object(plan) => DocumentPreflight::Object(plan.preflight_metadata()) }
    }
    pub(crate) fn execution_identities(&self) -> impl ExactSizeIterator<Item = &ExecutionIdentity> {
        self.mapped().execution_identities()
    }
    pub(crate) fn max_kv_bytes(&self) -> u64 { self.mapped().max_kv_bytes() }
    pub(crate) fn max_result_bytes(&self) -> u64 { self.mapped().max_result_bytes() }
    pub(crate) fn cohort_requirements(&self, context_limit: usize)
        -> Result<Int4ExtractionMapCohortRequirement, Int4ObjectExtractionMapError> {
        Ok(self.mapped().cohort_requirements(context_limit)?)
    }
    pub(crate) fn execute_with_control<C: DecodeStepControl>(self, admitted: &[ExecutionIdentity],
        engine: &mut StrictInt4Engine<'_>, vocabulary: &ExtractionVocabulary, control: &mut C)
        -> Result<Int4ExtractionDocumentRun, Int4ObjectExtractionMapError> {
        match self {
            Self::Map(plan) => Ok(Int4ExtractionDocumentRun::Map(plan.execute_with_control(admitted, engine, vocabulary, control)?)),
            Self::Object(plan) => Ok(Int4ExtractionDocumentRun::Object(plan.execute_with_control(admitted, engine, vocabulary, control)?)),
        }
    }
    pub(crate) fn execute_cohorts_with_control<C: DecodeStepControl>(self, admitted: &[ExecutionIdentity],
        weights: &Int4ModelWeights, vocabulary: &ExtractionVocabulary, budget: Int4ExtractionMapCohortBudget,
        kernel: PackedInt4Kernel, control: &mut C)
        -> Result<Int4ExtractionDocumentRun, Int4ObjectExtractionMapError> {
        match self {
            Self::Map(plan) => Ok(Int4ExtractionDocumentRun::Map(
                plan.execute_cohorts_with_control(admitted, weights, vocabulary, budget, kernel, control)?.document)),
            Self::Object(plan) => Ok(Int4ExtractionDocumentRun::Object(
                plan.execute_cohorts_with_control(admitted, weights, vocabulary, budget, kernel, control)?.document)),
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub(crate) enum DocumentPreflight {
    Map(Int4ExtractionMapPreflight),
    Object(Int4ObjectExtractionMapPreflight),
}
impl fmt::Debug for DocumentPreflight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DocumentPreflight").field("mapped", &self.mapped())
            .field("root_object_reconciliation", &matches!(self, Self::Object(_))).finish()
    }
}
impl DocumentPreflight {
    pub(crate) fn mapped(&self) -> Int4ExtractionMapPreflight {
        match self { Self::Map(proof) => *proof, Self::Object(proof) => proof.mapped() }
    }
    pub(crate) fn chunk_count(&self) -> usize { self.mapped().chunk_count() }
    pub(crate) fn group_width(&self) -> usize { self.mapped().group_width() }
    pub(crate) fn planned_work(&self) -> Int4Work { self.mapped().planned_work() }
    pub(crate) fn reserved_mask_visits(&self) -> u64 { self.mapped().reserved_mask_visits() }
    pub(crate) fn selected_rows(&self) -> Option<Int4JsonSparseLimits> { self.mapped().selected_rows() }
    pub(crate) fn document_encoding(&self) -> UntrustedEncoding { self.mapped().document_encoding() }
    pub(crate) fn execution_version(&self) -> &'static str {
        match self { Self::Map(proof) => proof.execution_version(), Self::Object(proof) => proof.execution_version() }
    }
    pub(crate) fn verify_completed(&self, run: &Int4ExtractionDocumentRun) -> Result<(), Int4ObjectExtractionMapError> {
        match (self, run) {
            (Self::Map(proof), Int4ExtractionDocumentRun::Map(run)) => Ok(proof.verify_completed(run)?),
            (Self::Object(proof), Int4ExtractionDocumentRun::Object(run)) => proof.verify_completed(run),
            _ => Err(Int4ObjectExtractionMapError::Admission),
        }
    }
}

/// Sealed factory for this live population, bound to actual loaded weights.
/// Records cannot replace the schema, grounding, compiler, budgets or schedule.
pub struct Int4ExtractionDocumentBatchPlanner {
    compiler: DocumentCompiler,
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
        compiler.require_byte_encoding()?;
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
        if let Some(union) = config.limits.root_array_union {
            let declaration = crate::grammar::runtime::JsonValidationSchema::compile(
                &config.request.schema, compiler.compiler.compiler_limits).map_err(|_| BatchCode::Planning)?;
            union.validate_schema(declaration.root()).map_err(|_| BatchCode::Planning)?;
        }
        Ok(Self { compiler: DocumentCompiler::Map(compiler), config, chunk_cohorts: false, owner: Arc::new(()) })
    }
    pub fn base_execution_identity(&self) -> &ExecutionIdentity { self.compiler.base_planner().base_execution_identity() }
    pub fn group_width(&self) -> usize { self.compiler.base_planner().group_width() }
    pub fn task_ceiling(&self) -> TaskBudget { self.compiler.base_planner().compiler.ceiling }
    pub fn compiler_limits(&self) -> CompileLimits { self.compiler.base_planner().compiler.compiler_limits }
    pub fn document_budget(&self) -> TaskBudget { self.config.request.budget }
    pub fn request(&self) -> &ExtractionBatchArgs { &self.config.request }
    pub fn document_limits(&self) -> Int4ExtractionMapLimits { self.config.limits }
    pub fn max_run_work(&self) -> Int4Work { self.config.max_model_work }
    pub fn max_mask_visits(&self) -> u64 { self.config.max_mask_visits }
    pub fn max_verification(&self) -> GroundingBudget { self.config.max_verification }
    pub fn selected_rows(&self) -> Option<Int4JsonSparseLimits> { self.compiler.base_planner().selected_rows() }
    pub fn root_object_reconciliation(&self) -> Option<RootObjectLimits> { self.compiler.root_object_reconciliation() }
    pub fn document_encoding(&self) -> &UntrustedEncoding { self.compiler.document_encoding() }
    pub fn encoding_workspace_bytes(&self) -> Result<u64, BatchFault> { self.compiler.encoding_workspace_bytes() }
    /// Select the bounded source/schema encoding for every complete document.
    /// One preparation shares its work allowance across sizing, chunk probes
    /// and all final chunk plans. An earlier preparation cannot be transferred
    /// across policy changes, even when its resulting chunk geometry matches.
    pub fn with_document_encoding(mut self, policy: UntrustedEncoding) -> Result<Self, BatchFault> {
        self.compiler = self.compiler.with_document_encoding(policy)?;
        self.owner = Arc::new(());
        Ok(self)
    }
    /// Explicitly reconcile complementary root properties under the unchanged
    /// original schema. This installs the fixed partial-object prompt before
    /// any source planning and invalidates every earlier preparation owner.
    pub fn with_root_object_reconciliation(mut self, limits: RootObjectLimits) -> Result<Self, BatchFault> {
        self.compiler = self.compiler.with_root_object_reconciliation(&self.config.request, self.config.limits, limits)
            .map_err(|e| document_failure(e, true).fault)?;
        self.owner = Arc::new(());
        Ok(self)
    }
    pub fn uses_chunk_cohorts(&self) -> bool { self.chunk_cohorts }
    /// Bind the complete legal-row policy before any source is prepared. A
    /// prior preparation cannot be transferred even at the full-row ceiling,
    /// where conservative work allowances happen to coincide.
    pub fn with_selected_rows(mut self, limits: Int4JsonSparseLimits) -> Result<Self, BatchFault> {
        self.compiler = self.compiler.with_selected_rows(limits)?;
        self.owner = Arc::new(());
        Ok(self)
    }
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
        -> Result<DocumentPlan<'s>, BatchItemFailure> {
        poll(control)?;
        if source.is_empty() || source.len() > self.config.limits.mapping.chunks.max_input_bytes {
            return Err(if fatal { BatchItemFailure::fatal(BatchCode::DocumentLimit) }
                else { BatchItemFailure::reject(BatchCode::DocumentLimit) });
        }
        self.compiler.plan(source, self.request(), self.config.limits, control)
            .map_err(|e| document_failure(e, fatal))
    }
    fn snapshot<C: DecodeStepControl>(&self, plan: &DocumentPlan<'_>, control: &mut C)
        -> Result<(DocumentPreflight, Vec<ExecutionIdentity>), BatchItemFailure> {
        let expected = plan.preflight_metadata();
        if expected.chunk_count() == 0 || expected.chunk_count() > self.config.limits.mapping.chunks.max_chunks
            || expected.group_width() != self.group_width() || expected.selected_rows() != self.selected_rows()
            || &expected.document_encoding() != self.document_encoding()
            || &plan.mapped().document_encoding() != self.document_encoding()
            || plan.max_kv_bytes() != self.document_budget().max_kv_bytes
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
    fn compare<C: DecodeStepControl>(&self, plan: &DocumentPlan<'_>,
        prepared: &PreparedInt4ExtractionDocument, control: &mut C) -> Result<(), BatchItemFailure> {
        self.owns(prepared)?;
        if plan.preflight_metadata() != prepared.expected || plan.max_kv_bytes() != self.document_budget().max_kv_bytes
            || &prepared.expected.document_encoding() != self.document_encoding()
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
    expected: DocumentPreflight,
    identities: Vec<ExecutionIdentity>,
    owner: Arc<()>,
}
impl PreparedInt4ExtractionDocument {
    pub fn chunk_count(&self) -> usize { self.expected.chunk_count() }
    pub fn planned_work(&self) -> Int4Work { self.expected.planned_work() }
    pub fn reserved_mask_visits(&self) -> u64 { self.expected.reserved_mask_visits() }
    pub fn execution_identities(&self) -> &[ExecutionIdentity] { &self.identities }
    pub fn document_encoding(&self) -> UntrustedEncoding { self.expected.document_encoding() }
}
pub struct Int4ExtractionDocumentAdmission<'a> {
    pub identities: &'a [ExecutionIdentity],
    pub document_encoding: UntrustedEncoding,
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
    type Output = GuardedOutput<Int4ExtractionDocumentRun, A::Guard>;
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
                document_encoding: prepared.document_encoding(),
                mask_node_visits: prepared.reserved_mask_visits(), mask_limits: self.planner.config.limits.mapping.mask_limits,
                kv_reservation_bytes: kv, max_result_bytes: self.planner.max_result_bytes(),
                max_output_tokens: self.planner.output_tokens(prepared.chunk_count()).map_err(BatchItemFailure::fatal)?,
                chunk_count: prepared.chunk_count(),
            })?;
            poll(control)?;
            if admitted != prepared.identities { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
            let output = self.backend.execute(plan, &admitted, self.vocabulary, control)?;
            prepared.expected.verify_completed(&output).map_err(|e| document_failure(e, true))?;
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
    fn begin(&mut self, config: &Int4ExtractionDocumentBatchConfig, expected: DocumentPreflight,
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
        Int4ExtractionMapError::Extraction(_) | Int4ExtractionMapError::Reduction(_)
            | Int4ExtractionMapError::ArrayUnion(_) => (BatchCode::InvalidExecution, true),
    };
    if stop { BatchItemFailure::fatal(code) } else { BatchItemFailure::reject(code) }
}

fn document_failure(error: Int4ObjectExtractionMapError, fatal: bool) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    let code = match error {
        Int4ObjectExtractionMapError::Map(error) => return map_failure(error, fatal),
        Int4ObjectExtractionMapError::Admission => BatchCode::Admission,
        Int4ObjectExtractionMapError::Reconciliation(RootObjectError::InvalidLimits) => BatchCode::InvalidLimits,
        Int4ObjectExtractionMapError::Reconciliation(RootObjectError::Schema) => BatchCode::Planning,
        Int4ObjectExtractionMapError::Reconciliation(RootObjectError::Allocation) => BatchCode::Allocation,
        Int4ObjectExtractionMapError::Reconciliation(_) => BatchCode::InvalidExecution,
    };
    // Reconciliation is required document semantics. A conflict, missing
    // required field, bad source witness or exhausted global allowance never
    // turns already completed partial chunks into a successful document.
    BatchItemFailure::fatal(code)
}

#[cfg(test)] mod tests;
