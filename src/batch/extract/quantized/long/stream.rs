//! Complete-document extraction on the existing ordered NDJSON runner.
//! One resident native driver; no context-sized truncation or per-row runtime.
use super::*;
use crate::batch::{BatchRequestContext, BatchProcessor, BatchWork};
use super::object::{Int8ObjectExtractionPlanner, Int8ObjectExtractionMapLimits,
    Int8ObjectExtractionMapError, Int8ObjectExtractionMapPreflight,
    Int8ObjectExtractionMapRun, PreparedInt8ObjectExtractionMap};
use crate::tasks::extract::object_reconcile::{RootObjectError, RootObjectLimits, RootObjectSchema};

/// The schema, grounding policy and partition limits are fixed for the stream.
/// Each document's exact native work and mask reservation, plus its complete
/// verification allowance, are charged BEFORE any forward and never refunded.
#[derive(Clone)]
pub struct ExtractionDocumentBatchConfig {
    pub request: ExtractionBatchArgs,
    pub limits: Int8ExtractionMapLimits,
    pub max_model_work: Int8Work,
    pub max_mask_visits: u64,
    pub max_verification: GroundingBudget,
}

/// Records cannot replace the admitted schema, profile, chunking or budgets.
/// Accept only an empty object, including refusal of serde's sequence form.
#[derive(Default)]
pub struct ExtractionDocumentArgs {}
impl<'de> Deserialize<'de> for ExtractionDocumentArgs {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct EmptyMap;
        impl<'de> serde::de::Visitor<'de> for EmptyMap {
            type Value = ExtractionDocumentArgs;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("an empty object of document arguments")
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

/// The ordinary map keeps its exact existing wire. Object mode contains the
/// unchanged partial map plus the independently checked original-schema object.
#[derive(Serialize)]
#[serde(untagged)]
pub enum Int8ExtractionDocumentRun {
    Map(Int8ExtractionMapRun),
    Object(Int8ObjectExtractionMapRun),
}
impl Int8ExtractionDocumentRun {
    pub fn mapped(&self) -> &Int8ExtractionMapRun {
        match self { Self::Map(run) => run, Self::Object(run) => &run.partial_map }
    }
    /// Object mode spends additional original-source verification after every
    /// partial chunk has completed; that charge is part of the same allowance.
    pub fn verification_used(&self) -> ExtractionMapVerification {
        match self { Self::Map(run) => run.verification_used, Self::Object(run) => run.verification_used }
    }
}

/// Both live and retained factories hold the actual compiler and its complete
/// map-encoding ledger. A new record cannot create a new tokenizer allowance.
pub(crate) enum DocumentCompiler {
    Map(Int8ExtractionBatchPlanner),
    Object { planner: Int8ObjectExtractionPlanner, limits: RootObjectLimits },
}
impl DocumentCompiler {
    pub(crate) fn base_planner(&self) -> &Int8ExtractionBatchPlanner {
        match self { Self::Map(planner) => planner, Self::Object { planner, .. } => planner.base_planner() }
    }
    pub(crate) fn root_object_reconciliation(&self) -> Option<RootObjectLimits> {
        match self { Self::Map(_) => None, Self::Object { limits, .. } => Some(*limits) }
    }
    pub(crate) fn document_encoding(&self) -> &UntrustedEncoding { self.base_planner().document_encoding() }
    pub(crate) fn encoding_workspace_bytes(&self) -> Result<u64, BatchFault> {
        self.base_planner().map_document_encoding_workspace_bytes()
    }
    pub(crate) fn with_document_encoding(self, policy: UntrustedEncoding,
        total: Option<SafeBpeWorkLimits>) -> Result<Self, BatchFault> {
        policy.validate().map_err(|_| BatchCode::InvalidLimits)?;
        self.encoding_workspace_bytes()?;
        if self.document_encoding() == &policy && self.base_planner().map_document_encoding_limits() == total {
            return Ok(self);
        }
        if self.document_encoding() != &UntrustedEncoding::ByteFallback {
            return Err(BatchCode::InvalidExecution.into());
        }
        let total = total.ok_or(BatchCode::InvalidLimits)?;
        match self {
            Self::Map(planner) => Ok(Self::Map(planner.with_map_document_encoding(policy, total)?)),
            Self::Object { planner, limits } => Ok(Self::Object {
                planner: planner.with_map_document_encoding(policy, total)?, limits }),
        }
    }
    pub(crate) fn with_root_object_reconciliation(self, request: &ExtractionBatchArgs,
        mapping: Int8ExtractionMapLimits, reconciliation: RootObjectLimits)
        -> Result<Self, Int8ObjectExtractionMapError> {
        validate_object_schema(request, mapping, reconciliation, self.base_planner().compiler.compiler_limits)?;
        match self {
            Self::Map(planner) => Ok(Self::Object {
                planner: Int8ObjectExtractionPlanner::new(planner)?, limits: reconciliation }),
            Self::Object { planner, limits } if limits == reconciliation => Ok(Self::Object { planner, limits }),
            Self::Object { .. } => Err(Int8ObjectExtractionMapError::Admission),
        }
    }
    pub(crate) fn plan<'s, C: DecodeStepControl>(&self, source: &'s str,
        request: &ExtractionBatchArgs, limits: Int8ExtractionMapLimits, control: &mut C)
        -> Result<DocumentPlan<'s>, Int8ObjectExtractionMapError> {
        match self {
            Self::Map(planner) => Ok(DocumentPlan::Map(planner.plan_document_with_control(source, request, limits, control)?)),
            Self::Object { planner, limits: reconciliation } => Ok(DocumentPlan::Object(
                planner.plan_document_with_control(source, request,
                    Int8ObjectExtractionMapLimits { mapping: limits, reconciliation: *reconciliation }, control)?)),
        }
    }
}

fn validate_object_schema(request: &ExtractionBatchArgs, mapping: Int8ExtractionMapLimits,
    reconciliation: RootObjectLimits, compiler: CompileLimits) -> Result<(), Int8ObjectExtractionMapError> {
    Int8ObjectExtractionMapLimits { mapping, reconciliation }.reconciliation_workspace_bytes()?;
    // Fixed, bounded schema-only admission before a population is opened.
    // Real source preparation repeats it with the original caller's control.
    let schema = RootObjectSchema::derive(&request.schema, compiler, reconciliation, &mut VerificationOnly)?;
    if schema.requires_source() != (request.grounding == ExtractionBatchGrounding::SourceMembership) {
        return Err(Int8ObjectExtractionMapError::Admission);
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum CompilerRef<'p> {
    Map(&'p Int8ExtractionBatchPlanner),
    Configured(&'p DocumentCompiler),
    #[cfg(feature = "asupersync-runtime")]
    Object { planner: &'p Int8ObjectExtractionPlanner, limits: RootObjectLimits },
}
impl<'p> From<&'p Int8ExtractionBatchPlanner> for CompilerRef<'p> {
    fn from(planner: &'p Int8ExtractionBatchPlanner) -> Self { Self::Map(planner) }
}
impl<'p> From<&'p DocumentCompiler> for CompilerRef<'p> {
    fn from(planner: &'p DocumentCompiler) -> Self { Self::Configured(planner) }
}
impl CompilerRef<'_> {
    fn base_planner(&self) -> &Int8ExtractionBatchPlanner {
        match self { Self::Map(planner) => planner, Self::Configured(compiler) => compiler.base_planner(),
            #[cfg(feature = "asupersync-runtime")]
            Self::Object { planner, .. } => planner.base_planner() }
    }
    fn validate(&self, config: &ExtractionDocumentBatchConfig) -> Result<(), BatchFault> {
        config.validate()?;
        self.base_planner().map_document_encoding_workspace_bytes()?;
        let reconciliation = match self {
            Self::Map(_) => None,
            Self::Configured(compiler) => compiler.root_object_reconciliation(),
            #[cfg(feature = "asupersync-runtime")]
            Self::Object { limits, .. } => Some(*limits),
        };
        if let Some(limits) = reconciliation {
            validate_object_schema(&config.request, config.limits, limits,
                self.base_planner().compiler.compiler_limits).map_err(|e| document_failure(e, true).fault)?;
        }
        Ok(())
    }
    fn plan<'s, C: DecodeStepControl>(&self, source: &'s str,
        request: &ExtractionBatchArgs, limits: Int8ExtractionMapLimits, control: &mut C)
        -> Result<DocumentPlan<'s>, Int8ObjectExtractionMapError> {
        match self {
            Self::Map(planner) => Ok(DocumentPlan::Map(planner.plan_document_with_control(source, request, limits, control)?)),
            Self::Configured(compiler) => compiler.plan(source, request, limits, control),
            #[cfg(feature = "asupersync-runtime")]
            Self::Object { planner, limits: reconciliation } => Ok(DocumentPlan::Object(
                planner.plan_document_with_control(source, request,
                    Int8ObjectExtractionMapLimits { mapping: limits, reconciliation: *reconciliation }, control)?)),
        }
    }
}

pub(crate) enum DocumentPlan<'s> {
    Map(PreparedInt8ExtractionMap<'s>),
    Object(PreparedInt8ObjectExtractionMap<'s>),
}
impl DocumentPlan<'_> {
    pub(crate) fn mapped(&self) -> &PreparedInt8ExtractionMap<'_> {
        match self { Self::Map(plan) => plan, Self::Object(plan) => plan.mapped() }
    }
    pub(crate) fn preflight_metadata(&self) -> DocumentPreflight {
        match self { Self::Map(plan) => DocumentPreflight::Map(plan.preflight_metadata()),
            Self::Object(plan) => DocumentPreflight::Object(plan.preflight_metadata()) }
    }
    pub(crate) fn execution_identities(&self) -> impl ExactSizeIterator<Item = &ExecutionIdentity> {
        self.mapped().execution_identities()
    }
    pub(crate) fn max_result_bytes(&self) -> u64 { self.mapped().max_result_bytes() }
    fn execute_with_driver<D: Driver, C: DecodeStepControl>(self, admitted: &[ExecutionIdentity],
        driver: D, control: &mut C) -> Result<Int8ExtractionDocumentRun, Int8ObjectExtractionMapError> {
        match self {
            Self::Map(plan) => Ok(Int8ExtractionDocumentRun::Map(plan.execute_with_driver(admitted, driver, control)?)),
            Self::Object(plan) => Ok(Int8ExtractionDocumentRun::Object(plan.execute_with_driver(admitted, driver, control)?)),
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub(crate) enum DocumentPreflight {
    Map(Int8ExtractionMapPreflight),
    Object(Int8ObjectExtractionMapPreflight),
}
impl fmt::Debug for DocumentPreflight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DocumentPreflight").field("mapped", &self.mapped())
            .field("root_object_reconciliation", &matches!(self, Self::Object(_))).finish()
    }
}
impl DocumentPreflight {
    pub(crate) fn mapped(&self) -> Int8ExtractionMapPreflight {
        match self { Self::Map(proof) => *proof, Self::Object(proof) => proof.mapped() }
    }
    pub(crate) fn source_span(&self) -> VerifiedSourceSpan { self.mapped().source_span() }
    pub(crate) fn chunk_count(&self) -> usize { self.mapped().chunk_count() }
    pub(crate) fn planned_work(&self) -> Int8Work { self.mapped().planned_work() }
    pub(crate) fn reserved_mask_visits(&self) -> u64 { self.mapped().reserved_mask_visits() }
    pub(crate) fn verify_completed_with_control<C: DecodeStepControl>(&self,
        run: &Int8ExtractionDocumentRun, control: &mut C) -> Result<(), Int8ObjectExtractionMapError> {
        match (self, run) {
            (Self::Map(proof), Int8ExtractionDocumentRun::Map(run)) => Ok(proof.verify_completed_with_control(run, control)?),
            (Self::Object(proof), Int8ExtractionDocumentRun::Object(run)) => proof.verify_completed_with_control(run, control),
            _ => Err(Int8ObjectExtractionMapError::Admission),
        }
    }
}

/// Admission sees ALL exact chunk identities, not a representative first chunk
/// or a compiler identity that omits the actual source/schema binding.
pub struct ExtractionDocumentAdmission<'a> {
    pub identities: &'a [ExecutionIdentity],
    pub model_work: Int8Work,
    pub mask_node_visits: u64,
    pub mask_limits: MaskWorkLimits,
    pub kv_reservation_bytes: u64,
    pub max_result_bytes: u64,
}
pub trait ExtractionDocumentBatchAdmission {
    type Guard;
    fn admit(&mut self, request: ExtractionDocumentAdmission<'_>)
        -> Result<(Vec<ExecutionIdentity>, Self::Guard), BatchItemFailure>;
}

/// Owns the original document and a private snapshot of every compiled identity.
/// Rebuilding avoids a self-referential owner; equality is checked before native
/// execution. Neither the source nor this snapshot has a wire constructor.
pub struct PreparedExtractionDocument {
    source: String,
    expected: DocumentPreflight,
    identities: Vec<ExecutionIdentity>,
    owner: Arc<()>,
}

pub struct NativeInt8DocumentExtractionBatch<'p, 'e, 'w, 'v, A: ExtractionDocumentBatchAdmission> {
    planner: CompilerRef<'p>,
    config: ExtractionDocumentBatchConfig,
    driver: NativeDriver<'e, 'w, 'v>,
    admission: A,
    ledger: Ledger,
}
impl<'p, 'e, 'w, 'v, A: ExtractionDocumentBatchAdmission>
    NativeInt8DocumentExtractionBatch<'p, 'e, 'w, 'v, A> {
    pub fn new(planner: &'p Int8ExtractionBatchPlanner, config: ExtractionDocumentBatchConfig,
        engine: &'e mut StrictInt8Engine<'w>, vocabulary: &'v ExtractionVocabulary, admission: A)
        -> Result<Self, BatchFault> {
        // A compact stream requires the complete-map ledger, not merely a
        // record-local encoding policy. The same borrowed planner retains
        // every probe, chunk and reconstruction debit across all records.
        Self::configured(planner.into(), config, engine, vocabulary, admission, None)
    }
    /// The same engine and fixed prompt workspace span every immutable chunk
    /// in every original document. Whole-corpus work and exact reconstruction
    /// checks remain unchanged; flush never replenishes their allowances.
    pub fn new_layer_major(planner: &'p Int8ExtractionBatchPlanner, config: ExtractionDocumentBatchConfig,
        engine: &'e mut StrictInt8Engine<'w>, vocabulary: &'v ExtractionVocabulary, admission: A,
        prefill: Int8PrefillLimits) -> Result<Self, BatchFault> {
        Self::configured(planner.into(), config, engine, vocabulary, admission, Some(prefill))
    }
    pub(crate) fn new_scheduled(compiler: &'p DocumentCompiler, config: ExtractionDocumentBatchConfig,
        engine: &'e mut StrictInt8Engine<'w>, vocabulary: &'v ExtractionVocabulary, admission: A,
        prefill: Option<Int8PrefillLimits>) -> Result<Self, BatchFault> {
        Self::configured(compiler.into(), config, engine, vocabulary, admission, prefill)
    }
    #[cfg(feature = "asupersync-runtime")]
    pub(crate) fn new_object_scheduled(planner: &'p Int8ObjectExtractionPlanner,
        reconciliation: RootObjectLimits, config: ExtractionDocumentBatchConfig,
        engine: &'e mut StrictInt8Engine<'w>, vocabulary: &'v ExtractionVocabulary, admission: A,
        prefill: Option<Int8PrefillLimits>) -> Result<Self, BatchFault> {
        Self::configured(CompilerRef::Object { planner, limits: reconciliation },
            config, engine, vocabulary, admission, prefill)
    }
    fn configured(planner: CompilerRef<'p>, config: ExtractionDocumentBatchConfig,
        engine: &'e mut StrictInt8Engine<'w>, vocabulary: &'v ExtractionVocabulary, admission: A,
        prefill: Option<Int8PrefillLimits>) -> Result<Self, BatchFault> {
        if let Some(prefill) = prefill { prefill.validate().map_err(|_| BatchCode::InvalidLimits)?; }
        planner.validate(&config)?;
        let driver = NativeDriver { engine, vocabulary, prefill };
        capacity(&driver, &config).map_err(|e| e.fault)?;
        Ok(Self { planner, config, driver, admission, ledger: Ledger::default() })
    }
    pub fn reserved_model_work(&self) -> Int8Work { self.ledger.work }
    pub fn reserved_mask_visits(&self) -> u64 { self.ledger.masks }
    pub fn reserved_verification(&self) -> GroundingBudget { self.ledger.verification }
    pub fn is_poisoned(&self) -> bool { self.ledger.failed || !self.driver.clean() }
}
impl<A: ExtractionDocumentBatchAdmission> BatchProcessor
    for NativeInt8DocumentExtractionBatch<'_, '_, '_, '_, A> {
    type Args = ExtractionDocumentArgs;
    type Prepared = PreparedExtractionDocument;
    type Output = GuardedOutput<Int8ExtractionDocumentRun, A::Guard>;
    fn prepare(&mut self, _: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        self.ledger.failed = true;
        Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>,
        control: &mut C) -> Result<Self::Prepared, BatchItemFailure> {
        prepare_document(self.planner, &self.config, &self.driver, &mut self.ledger, document, control)
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork {
        let work = prepared.expected.planned_work();
        BatchWork { forward_positions: work.forward_positions, projected_logits: work.projected_logits }
    }
    fn execute<C: DecodeStepControl>(&mut self, _: Self::Prepared, _: &mut C) -> Result<Self::Output, BatchItemFailure> {
        self.ledger.failed = true;
        Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        execute_document(self.planner, &self.config, &mut self.driver, &mut self.admission,
            &mut self.ledger, prepared, context, control)
    }
}

impl ExtractionDocumentBatchConfig {
    pub(crate) fn validate(&self) -> Result<(), BatchFault> {
        let m = self.limits.mapping;
        let v = self.limits.verification;
        self.request.budget.validate().map_err(|_| BatchCode::InvalidLimits)?;
        m.chunks.effective_token_limit().map_err(|_| BatchCode::InvalidLimits)?;
        if self.request.schema.is_empty() || m.chunks.max_input_bytes == 0 || m.chunks.max_chunks > 256
            || !nonzero(m.max_model_work) || !fits(m.max_model_work, self.max_model_work)
            || m.mask_limits.max_trie_node_visits == 0 || m.mask_limits.checkpoint_interval_nodes == 0
            || m.mask_visits_per_chunk == 0 || m.max_mask_visits < m.mask_visits_per_chunk
            || m.max_mask_visits > self.max_mask_visits
            || m.reduction.max_result_bytes == 0 || m.reduction.max_live_value_bytes == 0
            || !(1..=1_000_000).contains(&v.max_fields) || !(1..=1_000_000).contains(&v.max_matches)
            || v.max_scan_steps == 0 || !verification_fits(v, self.max_verification) {
            return Err(BatchCode::InvalidLimits.into());
        }
        Ok(())
    }
}
fn nonzero(w: Int8Work) -> bool {
    w.forward_positions != 0 && w.projected_logits != 0 && w.attention_pairs != 0
        && w.projections.dot_products != 0 && w.projections.multiply_accumulates != 0
}
fn verification_fits(a: GroundingBudget, b: GroundingBudget) -> bool {
    a.max_fields <= b.max_fields && a.max_matches <= b.max_matches && a.max_scan_steps <= b.max_scan_steps
}
fn add_verification(a: GroundingBudget, b: GroundingBudget) -> Option<GroundingBudget> {
    Some(GroundingBudget { max_fields: a.max_fields.checked_add(b.max_fields)?,
        max_matches: a.max_matches.checked_add(b.max_matches)?, max_scan_steps: a.max_scan_steps.checked_add(b.max_scan_steps)? })
}
struct Ledger {
    work: Int8Work, masks: u64, verification: GroundingBudget, sequence: u64, failed: bool,
    owner: Arc<()>,
}
impl Default for Ledger {
    fn default() -> Self {
        Self { work: Int8Work::default(), masks: 0,
            verification: GroundingBudget { max_fields: 0, max_matches: 0, max_scan_steps: 0 },
            sequence: 0, failed: false, owner: Arc::new(()) }
    }
}
impl Ledger {
    fn ready(&self) -> Result<(), BatchItemFailure> {
        if self.failed { Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)) } else { Ok(()) }
    }
    fn begin(&mut self, config: &ExtractionDocumentBatchConfig, expected: DocumentPreflight,
        context: BatchRequestContext) -> Result<(), BatchItemFailure> {
        self.ready()?;
        self.failed = true; // Unwinding or ANY execution failure closes this adapter.
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
        // Atomic across all five native axes, masks and all verification axes.
        self.work = work; self.masks = masks; self.verification = verification; self.sequence = context.request_seq;
        Ok(())
    }
}
fn capacity(driver: &impl Driver, config: &ExtractionDocumentBatchConfig) -> Result<u64, BatchItemFailure> {
    if !driver.clean() || driver.capacity() < config.limits.mapping.chunks.context_tokens {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    (driver.capacity() as u64).checked_mul(KV_BYTES_PER_TOKEN as u64)
        .filter(|&n| n != 0 && n <= config.request.budget.max_kv_bytes)
        .ok_or_else(|| BatchItemFailure::fatal(BatchCode::Admission))
}
fn checkpoint(control: &mut impl DecodeStepControl) -> Result<(), BatchItemFailure> {
    step(control).map_err(|e| map_failure(Int8SourceMapError::Source(e), true))
}
fn map_failure(error: Int8SourceMapError, fatal: bool) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    let (code, terminal) = match error {
        Int8SourceMapError::Allocation => (BatchCode::Allocation, true),
        Int8SourceMapError::WorkLimit => (BatchCode::WorkLimit, fatal),
        Int8SourceMapError::EmptySource => (BatchCode::DocumentLimit, fatal),
        Int8SourceMapError::InvalidLimits => (BatchCode::InvalidLimits, true),
        Int8SourceMapError::Admission => (BatchCode::Admission, fatal),
        Int8SourceMapError::Chunk(_) => (BatchCode::Planning, fatal),
        Int8SourceMapError::Source(_) | Int8SourceMapError::Reduction(_) => (BatchCode::InvalidExecution, true),
    };
    if terminal { BatchItemFailure::fatal(code) } else { BatchItemFailure::reject(code) }
}
fn document_failure(error: Int8ObjectExtractionMapError, fatal: bool) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    let code = match error {
        Int8ObjectExtractionMapError::Map(error) => return map_failure(error, fatal),
        Int8ObjectExtractionMapError::Admission => BatchCode::Admission,
        Int8ObjectExtractionMapError::Reconciliation(RootObjectError::InvalidLimits) => BatchCode::InvalidLimits,
        Int8ObjectExtractionMapError::Reconciliation(RootObjectError::Schema) => BatchCode::Planning,
        Int8ObjectExtractionMapError::Reconciliation(RootObjectError::Allocation) => BatchCode::Allocation,
        Int8ObjectExtractionMapError::Reconciliation(RootObjectError::Limit) => BatchCode::WorkLimit,
        Int8ObjectExtractionMapError::Reconciliation(_) => BatchCode::InvalidExecution,
    };
    // Object reconciliation is required item semantics. No conflict, missing
    // required field or failed final verification can publish partial chunks.
    BatchItemFailure::fatal(code)
}
fn prepare_document<'p, D: Driver, C: DecodeStepControl>(planner: impl Into<CompilerRef<'p>>,
    config: &ExtractionDocumentBatchConfig, driver: &D, ledger: &mut Ledger,
    document: BatchDocument<ExtractionDocumentArgs>, control: &mut C)
    -> Result<PreparedExtractionDocument, BatchItemFailure> {
    let planner = planner.into();
    ledger.ready()?; ledger.failed = true;
    let result = (|| {
        checkpoint(control)?;
        capacity(driver, config)?;
        if document.text.is_empty() || document.text.len() > config.limits.mapping.chunks.max_input_bytes {
            return Err(BatchItemFailure::reject(BatchCode::DocumentLimit));
        }
        let plan = planner.plan(&document.text, &config.request, config.limits, control)
            .map_err(|e| document_failure(e, false))?;
        let expected = plan.preflight_metadata();
        if expected.source_span().byte_start != 0 || expected.source_span().byte_end != document.text.len()
            || plan.execution_identities().len() != expected.chunk_count() {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        let mut identities = Vec::new();
        identities.try_reserve_exact(expected.chunk_count()).map_err(|_| BatchItemFailure::fatal(BatchCode::Allocation))?;
        for chunk in &plan.mapped().plans {
            checkpoint(control)?;
            preflight(chunk, driver)?;
            identities.push(chunk.execution_identity().clone());
        }
        drop(plan);
        checkpoint(control)?;
        Ok(PreparedExtractionDocument { source: document.text, expected, identities, owner: Arc::clone(&ledger.owner) })
    })();
    if !result.as_ref().is_err_and(|e: &BatchItemFailure| e.stop) { ledger.failed = false; }
    result
}

// Only this closed helper can compose a test driver with a completed map. The
// public adapter always supplies NativeDriver. No public result-injection hook.
#[allow(clippy::too_many_arguments)]
fn execute_document<'p, D: Driver, A: ExtractionDocumentBatchAdmission, C: DecodeStepControl>(
    planner: impl Into<CompilerRef<'p>>, config: &ExtractionDocumentBatchConfig,
    driver: &mut D, admission: &mut A, ledger: &mut Ledger,
    prepared: PreparedExtractionDocument, context: BatchRequestContext, control: &mut C)
    -> Result<GuardedOutput<Int8ExtractionDocumentRun, A::Guard>, BatchItemFailure> {
    let planner = planner.into();
    // Preparation and reconstruction must spend THIS invocation's shared
    // compact allowance. Equal semantic identities do not transfer that debt
    // from another adapter or a prior durable invocation.
    if !Arc::ptr_eq(&prepared.owner, &ledger.owner) {
        ledger.failed = true;
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    ledger.begin(config, prepared.expected, context)?;
    let result = (|| {
        checkpoint(control)?;
        let kv = capacity(driver, config)?;
        let plan = planner.plan(&prepared.source, &config.request, config.limits, control)
            .map_err(|e| document_failure(e, true))?;
        if plan.preflight_metadata() != prepared.expected || plan.execution_identities().len() != prepared.identities.len() {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        for (chunk, expected) in plan.mapped().plans.iter().zip(&prepared.identities) {
            checkpoint(control)?;
            chunk.verify_identity(expected).map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
            preflight(chunk, driver)?;
        }
        let (admitted, guard) = admission.admit(ExtractionDocumentAdmission {
            identities: &prepared.identities, model_work: prepared.expected.planned_work(),
            mask_node_visits: prepared.expected.reserved_mask_visits(), mask_limits: config.limits.mapping.mask_limits,
            kv_reservation_bytes: kv, max_result_bytes: plan.max_result_bytes(),
        })?;
        checkpoint(control)?;
        // The map preflights EVERY returned identity before its first forward.
        // Borrowing the driver preserves the SAME resident engine across items.
        let output = plan.execute_with_driver(&admitted, BorrowedDriver(driver), control)
            .map_err(|e| document_failure(e, true))?;
        prepared.expected.verify_completed_with_control(&output, control).map_err(|e| document_failure(e, true))?;
        if !driver.clean() { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
        checkpoint(control)?;
        Ok(GuardedOutput::new(output, guard))
    })();
    // Even a nominally recoverable admission refusal has consumed this item;
    // never keep going after a half-executed document or renew its reservation.
    match result {
        Ok(output) => { ledger.failed = false; Ok(output) }
        Err(mut failure) => { failure.stop = true; Err(failure) }
    }
}
struct BorrowedDriver<'a, D>(&'a mut D);
impl<D: Driver> Driver for BorrowedDriver<'_, D> {
    fn capacity(&self) -> usize { self.0.capacity() }
    fn clean(&self) -> bool { self.0.clean() }
    fn execute<C: DecodeStepControl>(&mut self, prepared: &PreparedInt8BatchExtraction,
        identity: &ExecutionIdentity, budget: Int8JsonBudget, control: &mut C) -> Result<Int8ExtractRun, Int8ExtractError> {
        self.0.execute(prepared, identity, budget, control)
    }
}

#[cfg(test)] mod tests;
