//! Complete original-document classification on one resident native INT8 stream.
//! Preparation freezes every chunk and head before inference. Exact rebuilding
//! avoids self-referential owners, while native and compact-tokenizer work stay
//! charged across records, failed preparations and flush boundaries.
use super::*;
use std::sync::Arc;
use crate::{
    batch::BatchRequestContext,
    tasks::classify::{ClassificationLabel, ClassificationMode, ClassificationRequest,
        quantized::long::{Int8ClassificationMapError, Int8ClassificationMapLimits,
            Int8ClassificationMapPreflight, Int8ClassificationMapRun, PreparedInt8ClassificationMap}},
    tokenizer::untrusted::{SafeBpeWork, SafeBpeWorkLimits, UntrustedEncoding},
    native_engine::strict_int8::Int8MemoryRequirement,
};

/// Frozen per-chunk identities, their admission copies and source coordinates
/// coexist with the reconstructed map. Additional to the map's own quote.
pub const CLASSIFICATION_DOCUMENT_CHUNK_METADATA_BYTES: u64 = 32_768;

/// Immutable compiler/configuration owned by one stream. Separate streams
/// receive separate seals, even when every public configuration field agrees.
pub struct Int8ClassificationDocumentBatchPlanner {
    planner: Arc<ClassificationPlanner>,
    identity: ExecutionIdentity,
    ceiling: TaskBudget,
    planning: ClassificationLimits,
    mapping: Int8ClassificationMapLimits,
    defaults: Option<ClassificationBatchArgs>,
    max_model_work: Int8Work,
    max_tokenizer_work: Option<SafeBpeWorkLimits>,
    seal: Arc<()>,
}
impl Int8ClassificationDocumentBatchPlanner {
    #[allow(clippy::too_many_arguments)]
    pub fn new(planner: Arc<ClassificationPlanner>, identity: ExecutionIdentity,
        ceiling: TaskBudget, planning: ClassificationLimits, mapping: Int8ClassificationMapLimits,
        defaults: Option<ClassificationBatchArgs>, max_model_work: Int8Work,
        max_tokenizer_work: Option<SafeBpeWorkLimits>) -> Result<Self, BatchFault> {
        Self::validate_configuration(&planner, &identity, ceiling, planning, mapping,
            defaults.as_ref(), max_model_work, max_tokenizer_work)?;
        Ok(Self { planner, identity, ceiling, planning, mapping, defaults,
            max_model_work, max_tokenizer_work, seal: Arc::new(()) })
    }
    /// Validate fixed semantic and work settings before loading model weights.
    /// Private defaults are borrowed, never copied into a throwaway factory.
    #[allow(clippy::too_many_arguments)]
    pub fn validate_configuration(planner: &ClassificationPlanner, identity: &ExecutionIdentity,
        ceiling: TaskBudget, planning: ClassificationLimits, mapping: Int8ClassificationMapLimits,
        defaults: Option<&ClassificationBatchArgs>, max_model_work: Int8Work,
        max_tokenizer_work: Option<SafeBpeWorkLimits>) -> Result<(), BatchFault> {
        validate_limits(ceiling, planning, mapping, max_model_work)?;
        validate_identity_owner(identity)?;
        EncodingLedger::new(planner.document_encoding(), max_tokenizer_work)?;
        // Reuse the established INT8 compiler/profile checks. No source is
        // prepared and no tokenizer allowance is spent during this check.
        Int8ClassificationBatchPlanner::new_with_document_encoding(planner, identity.clone(),
            ceiling, planning, None)?;
        if let Some(args) = defaults {
            validate_args(args, ceiling, planning).map_err(|error| error.fault)?;
        }
        Ok(())
    }
    pub fn base_execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    pub fn task_budget(&self) -> TaskBudget { self.ceiling }
    pub fn planning_limits(&self) -> ClassificationLimits { self.planning }
    pub fn mapping_limits(&self) -> Int8ClassificationMapLimits { self.mapping }
    pub fn max_run_work(&self) -> Int8Work { self.max_model_work }
    pub fn document_encoding(&self) -> &UntrustedEncoding { self.planner.document_encoding() }
    pub fn max_tokenizer_work(&self) -> Option<SafeBpeWorkLimits> { self.max_tokenizer_work }

    /// Account for actual transferred capacities rather than logical lengths.
    pub fn retained_defaults_bytes(&self) -> Result<u64, BatchFault> {
        let Some(args) = &self.defaults else { return Ok(0); };
        let base = (args.labels.capacity() as u64)
            .checked_mul(std::mem::size_of::<ClassificationLabel>() as u64)
            .ok_or(BatchCode::InvalidLimits)?;
        args.labels.iter().try_fold(base, |bytes, label| bytes.checked_add(label.id.capacity() as u64)
            .and_then(|n| n.checked_add(label.description.capacity() as u64))
            .ok_or_else(|| BatchCode::InvalidLimits.into()))
    }

    fn request(&self, document: BatchDocument<ClassificationBatchArgs>)
        -> Result<ClassificationRequest, BatchItemFailure> {
        if document.text.is_empty() || document.text.len() > self.mapping.chunks.max_input_bytes {
            return Err(BatchItemFailure::reject(BatchCode::DocumentLimit));
        }
        // A supplied object replaces the complete default; no field merging.
        let args = document.task_args.or_else(|| self.defaults.clone())
            .ok_or_else(|| BatchItemFailure::reject(BatchCode::Planning))?;
        validate_args(&args, self.ceiling, self.planning)?;
        Ok(args.into_request(document.text))
    }

    fn plan<'a, C: DecodeStepControl>(&self, request: &'a ClassificationRequest,
        encoding_work: &mut SafeBpeWork, ledger: &mut EncodingLedger, control: &mut C,
        executing: bool) -> Result<PreparedInt8ClassificationMap<'a>, BatchItemFailure> {
        checkpoint(control)?;
        ledger.admit(self.document_encoding())?;
        let context = PlanContext::new(&self.identity, self.ceiling)
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        let before = *encoding_work;
        let result = self.planner.plan_int8_document_with_encoding_work(request, &context,
            self.planning, self.mapping, control, encoding_work);
        // This runs on BOTH success and failure. A later rejected chunk or
        // cancellation cannot discard already-spent tokenizer work.
        ledger.charge(encoding_work.checked_sub(before)
            .ok_or_else(|| BatchItemFailure::fatal(BatchCode::InvalidExecution))?)?;
        result.map_err(|error| map_failure(error, executing))
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare<C, F>(&self, document: BatchDocument<ClassificationBatchArgs>,
        remaining: Int8Work, ledger: &mut EncodingLedger, invocation: &Arc<()>, control: &mut C, preflight: F)
        -> Result<PreparedInt8ClassificationDocument, BatchItemFailure>
    where C: DecodeStepControl,
        F: FnOnce(&PreparedInt8ClassificationMap<'_>, &[ExecutionIdentity], &mut C)
            -> Result<(), BatchItemFailure> {
        checkpoint(control)?;
        let request = self.request(document)?;
        let mut encoding_work = SafeBpeWork::default();
        let plan = self.plan(&request, &mut encoding_work, ledger, control, false)?;
        subtract(remaining, plan.planned_work())?;
        let identities = plan.identities().map_err(|error| map_failure(error, false))?;
        preflight(&plan, &identities, control)?;
        let expected = plan.preflight_metadata();
        let required_context = plan.required_context();
        drop(plan);
        checkpoint(control)?;
        Ok(PreparedInt8ClassificationDocument { request, expected, identities,
            required_context, encoding_work, seal: Arc::clone(&self.seal), invocation: Arc::clone(invocation) })
    }

    fn verify(&self, prepared: &PreparedInt8ClassificationDocument, invocation: &Arc<()>)
        -> Result<(), BatchItemFailure> {
        if !Arc::ptr_eq(&self.seal, &prepared.seal) || !Arc::ptr_eq(invocation, &prepared.invocation) {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        Ok(())
    }
}

/// The exact original request remains private; public callers cannot replace
/// its source or submit scores/partial chunks as a completed document.
pub struct PreparedInt8ClassificationDocument {
    request: ClassificationRequest,
    expected: Int8ClassificationMapPreflight,
    identities: Vec<ExecutionIdentity>,
    required_context: usize,
    encoding_work: SafeBpeWork,
    seal: Arc<()>,
    invocation: Arc<()>,
}
impl PreparedInt8ClassificationDocument {
    pub fn execution_identities(&self) -> &[ExecutionIdentity] { &self.identities }
    pub fn planned_work(&self) -> Int8Work { self.expected.planned_work() }
    pub fn chunk_count(&self) -> usize { self.expected.chunk_count() }
    pub fn encoding_work(&self) -> SafeBpeWork { self.encoding_work }
}

pub struct Int8ClassificationDocumentAdmission<'a> {
    pub identities: &'a [ExecutionIdentity],
    pub model_work: Int8Work,
    pub required_context: usize,
    pub task_budget: TaskBudget,
    pub head_count: usize,
    pub kv_reservation_bytes: u64,
    pub max_result_bytes: u64,
}
pub trait Int8ClassificationDocumentBatchAdmission {
    type Guard;
    fn admit(&mut self, request: Int8ClassificationDocumentAdmission<'_>)
        -> Result<(Vec<ExecutionIdentity>, Self::Guard), BatchItemFailure>;
}

/// One complete document is one guarded transport output. There is no
/// callback through which a caller can supply native results or replacements.
pub struct NativeInt8ClassificationDocumentBatch<'e, 'w, A: Int8ClassificationDocumentBatchAdmission> {
    compiler: Arc<Int8ClassificationDocumentBatchPlanner>,
    engine: &'e mut StrictInt8Engine<'w>,
    admission: A,
    run: DocumentRunState,
    encoding: EncodingLedger,
    invocation: Arc<()>,
    prefill: Option<Int8PrefillLimits>,
}
impl<'e, 'w, A: Int8ClassificationDocumentBatchAdmission>
    NativeInt8ClassificationDocumentBatch<'e, 'w, A> {
    pub fn new(compiler: Arc<Int8ClassificationDocumentBatchPlanner>,
        engine: &'e mut StrictInt8Engine<'w>, admission: A) -> Result<Self, BatchFault> {
        capacity(&compiler, compiler.task_budget(), engine).map_err(|error| error.fault)?;
        let encoding = EncodingLedger::new(compiler.document_encoding(), compiler.max_tokenizer_work())?;
        let run = DocumentRunState::new(compiler.max_run_work());
        Ok(Self { compiler, engine, admission, run, encoding, invocation: Arc::new(()), prefill: None })
    }
    pub fn new_layer_major(compiler: Arc<Int8ClassificationDocumentBatchPlanner>,
        engine: &'e mut StrictInt8Engine<'w>, admission: A, prefill: Int8PrefillLimits)
        -> Result<Self, BatchFault> {
        prefill.validate().map_err(|_| BatchCode::InvalidLimits)?;
        let mut processor = Self::new(compiler, engine, admission)?;
        processor.prefill = Some(prefill);
        Ok(processor)
    }
    pub fn remaining_work(&self) -> Int8Work { self.run.remaining }
    pub fn tokenizer_work(&self) -> SafeBpeWork { self.encoding.used }
    pub fn is_poisoned(&self) -> bool { self.run.state != State::Ready || !clean(self.engine) }
}

impl<A: Int8ClassificationDocumentBatchAdmission> BatchProcessor
    for NativeInt8ClassificationDocumentBatch<'_, '_, A> {
    type Args = ClassificationBatchArgs;
    type Prepared = PreparedInt8ClassificationDocument;
    type Output = GuardedOutput<Int8ClassificationMapRun, A::Guard>;

    fn prepare(&mut self, _: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        self.run.state = State::Failed;
        Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }

    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>,
        control: &mut C) -> Result<Self::Prepared, BatchItemFailure> {
        self.run.enter()?;
        let compiler = &self.compiler;
        let engine = &*self.engine;
        let result = compiler.prepare(document, self.run.remaining, &mut self.encoding, &self.invocation, control,
            |plan, identities, control| {
                let kv = capacity(compiler, plan.task_budget(), engine)?;
                plan.preflight_with_control(identities, engine, document_budget(plan.planned_work(), kv), control)
                    .map_err(|error| map_failure(error, false))
            });
        self.run.finish(result, clean(self.engine), false)
    }

    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork {
        transport_work(prepared.planned_work())
    }

    fn execute<C: DecodeStepControl>(&mut self, _: Self::Prepared, _: &mut C)
        -> Result<Self::Output, BatchItemFailure> {
        self.run.state = State::Failed;
        Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }

    fn execute_with_context<C: DecodeStepControl>(&mut self, mut prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        self.run.enter()?;
        let result = (|| {
            self.compiler.verify(&prepared, &self.invocation)?;
            // Charge all five axes before reconstruction, admission or any
            // native forward. Every later failure keeps the debit.
            self.run.charge(prepared.planned_work(), context)?;
            checkpoint(control)?;
            let kv = capacity(&self.compiler, prepared.request.budget, self.engine)?;
            let expected_encoding_work = prepared.encoding_work;
            let plan = self.compiler.plan(&prepared.request, &mut prepared.encoding_work,
                &mut self.encoding, control, true)?;
            let actual_ids = plan.identities().map_err(|error| map_failure(error, true))?;
            if plan.preflight_metadata() != prepared.expected || actual_ids != prepared.identities
                || plan.required_context() != prepared.required_context
                || prepared.encoding_work.checked_sub(expected_encoding_work) != Some(expected_encoding_work) {
                return Err(BatchItemFailure::fatal(BatchCode::Admission));
            }
            drop(actual_ids);
            plan.preflight_with_control(&prepared.identities, self.engine,
                document_budget(prepared.expected.planned_work(), kv), control)
                .map_err(|error| map_failure(error, true))?;
            let (admitted, guard) = self.admission.admit(Int8ClassificationDocumentAdmission {
                identities: &prepared.identities, model_work: prepared.expected.planned_work(),
                required_context: prepared.required_context, task_budget: prepared.request.budget,
                head_count: plan.head_count(), kv_reservation_bytes: kv,
                max_result_bytes: self.compiler.mapping.max_result_bytes,
            })?;
            if admitted != prepared.identities {
                return Err(BatchItemFailure::fatal(BatchCode::Admission));
            }
            checkpoint(control)?;
            let budget = document_budget(prepared.expected.planned_work(), kv);
            let output = match self.prefill {
                Some(prefill) => plan.execute_layer_major_with_control(&admitted,
                    self.engine, budget, prefill, control),
                None => plan.execute_with_control(&admitted, self.engine, budget, control),
            }.map_err(|error| map_failure(error, true))?;
            prepared.expected.verify_completed_with_control(&output, control)
                .map_err(|error| map_failure(error, true))?;
            checkpoint(control)?;
            Ok(GuardedOutput::new(output, guard))
        })();
        self.run.finish(result, clean(self.engine), true)
    }
}

pub(crate) fn validate_limits(task: TaskBudget, planning: ClassificationLimits,
    mapping: Int8ClassificationMapLimits, total: Int8Work) -> Result<(), BatchFault> {
    mapping.validate(task).map_err(|_| BatchCode::InvalidLimits)?;
    planning.validate().map_err(|_| BatchCode::InvalidLimits)?;
    if !nonzero(total) || subtract(total, mapping.max_model_work).is_err()
        || mapping.chunks.max_input_bytes > planning.max_input_bytes
        || mapping.chunks.context_tokens > planning.max_context_tokens
        || task.max_output_bytes > mapping.max_retained_result_bytes {
        return Err(BatchCode::InvalidLimits.into());
    }
    Ok(())
}
// The map's fixed identity-copy quote is valid only for bounded transferred
// owners. Inspect capacities BEFORE any clone or canonical serialization so
// a short label with a huge retained allocation cannot hide inside the quote.
fn validate_identity_owner(identity: &ExecutionIdentity) -> Result<(), BatchFault> {
    let strings = [&identity.source_revision, &identity.artifact_format, &identity.quant_recipe,
        &identity.task_spec, &identity.grammar_compiler_version, &identity.kv_dtype,
        &identity.sampler_version, &identity.backend_semantic_version];
    let bytes = strings.into_iter().chain(identity.host_class.iter()).chain(identity.compiler_identity.iter())
        .try_fold(0_u64, |total, value| total.checked_add(value.capacity() as u64))
        .ok_or(BatchCode::InvalidLimits)?;
    if bytes > 1024 { return Err(BatchCode::InvalidLimits.into()); }
    let canonical = identity.canonical_json_bytes().map_err(|_| BatchCode::Admission)?;
    if canonical.len() > 16_384 { return Err(BatchCode::InvalidLimits.into()); }
    Ok(())
}
fn validate_args(args: &ClassificationBatchArgs, ceiling: TaskBudget, limits: ClassificationLimits)
    -> Result<(), BatchItemFailure> {
    let budget = args.budget;
    if budget.validate().is_err() || budget.max_input_tokens > ceiling.max_input_tokens
        || budget.max_output_tokens > ceiling.max_output_tokens || budget.max_output_bytes > ceiling.max_output_bytes
        || budget.max_grammar_states > ceiling.max_grammar_states || budget.max_kv_bytes > ceiling.max_kv_bytes
        || args.labels.is_empty() || args.labels.len() > limits.max_labels
        || (args.mode == ClassificationMode::Exclusive && args.labels.len() < 2)
        || args.policy.minimum_candidate_weight_ppm > 1_000_000
        || args.policy.minimum_margin_ppm > 1_000_000 {
        return Err(BatchItemFailure::reject(BatchCode::Planning));
    }
    let mut total = 0_usize;
    for (index, label) in args.labels.iter().enumerate() {
        total = total.checked_add(label.id.len()).and_then(|n| n.checked_add(label.description.len()))
            .ok_or_else(|| BatchItemFailure::reject(BatchCode::Planning))?;
        if label.id.trim().is_empty() || label.id.chars().any(char::is_control)
            || label.id.len() > limits.max_label_id_bytes || label.description.len() > limits.max_label_description_bytes
            || total > limits.max_total_label_bytes || args.labels[..index].iter().any(|old| old.id == label.id) {
            return Err(BatchItemFailure::reject(BatchCode::Planning));
        }
    }
    Ok(())
}
fn capacity(compiler: &Int8ClassificationDocumentBatchPlanner, task: TaskBudget,
    engine: &StrictInt8Engine<'_>) -> Result<u64, BatchItemFailure> {
    if !clean(engine) || engine.kv_cache().capacity_positions() < compiler.mapping.chunks.context_tokens {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    let required = Int8MemoryRequirement::for_context(engine.kv_cache().capacity_positions())
        .map_err(|_| BatchItemFailure::fatal(BatchCode::InvalidExecution))?;
    if required.kv_bytes > task.max_kv_bytes {
        return Err(BatchItemFailure::reject(BatchCode::Admission));
    }
    Ok(required.kv_bytes)
}
fn clean(engine: &StrictInt8Engine<'_>) -> bool {
    !engine.is_poisoned() && engine.kv_cache().all_slots_have_len(0)
}
fn document_budget(work: Int8Work, max_kv_bytes: u64) -> Int8ScoringBudget {
    Int8ScoringBudget { native: Int8RunBudget::exact(work), max_kv_bytes }
}
fn nonzero(work: Int8Work) -> bool {
    work.forward_positions != 0 && work.projected_logits != 0 && work.attention_pairs != 0
        && work.projections.dot_products != 0 && work.projections.multiply_accumulates != 0
}

struct DocumentRunState { remaining: Int8Work, last_sequence: u64, state: State }
impl DocumentRunState {
    fn new(remaining: Int8Work) -> Self { Self { remaining, last_sequence: 0, state: State::Ready } }
    fn enter(&mut self) -> Result<(), BatchItemFailure> {
        if self.state != State::Ready { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
        self.state = State::Running; Ok(())
    }
    fn charge(&mut self, work: Int8Work, context: BatchRequestContext) -> Result<(), BatchItemFailure> {
        if self.state != State::Running || context.request_seq <= self.last_sequence
            || context.epoch == 0 || context.input_line == 0 {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        self.remaining = subtract(self.remaining, work).map_err(|error| BatchItemFailure::fatal(error.fault))?;
        self.last_sequence = context.request_seq; Ok(())
    }
    fn finish<T>(&mut self, result: Result<T, BatchItemFailure>, clean: bool, executed: bool)
        -> Result<T, BatchItemFailure> {
        match result {
            Ok(value) if clean && self.state == State::Running => { self.state = State::Ready; Ok(value) }
            Ok(_) => { self.state = State::Failed; Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)) }
            Err(mut error) => {
                if executed || !clean || self.state != State::Running || error.fault.cancellation.is_some() {
                    error.stop = true;
                }
                self.state = if error.stop { State::Failed } else { State::Ready };
                Err(error)
            }
        }
    }
}

/// Admission conservatively fits the complete fixed preparation envelope
/// before work. Only actual work is charged, including failed/cancelled calls.
/// A small remaining quota never silently changes the semantic encoder policy.
struct EncodingLedger { limits: Option<SafeBpeWorkLimits>, used: SafeBpeWork }
impl EncodingLedger {
    fn new(policy: &UntrustedEncoding, limits: Option<SafeBpeWorkLimits>) -> Result<Self, BatchFault> {
        policy.validate().map_err(|_| BatchCode::InvalidLimits)?;
        match (policy, limits) {
            (UntrustedEncoding::ByteFallback, None) => {}
            (UntrustedEncoding::SafeBpeV1(call), Some(total)) => {
                total.validate().map_err(|_| BatchCode::InvalidLimits)?;
                if total.remaining_call(SafeBpeWork::default(), *call).ok() != Some(*call) {
                    return Err(BatchCode::InvalidLimits.into());
                }
            }
            _ => return Err(BatchCode::InvalidLimits.into()),
        }
        Ok(Self { limits, used: SafeBpeWork::default() })
    }
    fn admit(&self, policy: &UntrustedEncoding) -> Result<(), BatchItemFailure> {
        match (policy, self.limits) {
            (UntrustedEncoding::ByteFallback, None) => Ok(()),
            (UntrustedEncoding::SafeBpeV1(call), Some(total))
                if total.remaining_call(self.used, *call).ok() == Some(*call) => Ok(()),
            _ => Err(BatchItemFailure::fatal(BatchCode::WorkLimit)),
        }
    }
    fn charge(&mut self, work: SafeBpeWork) -> Result<(), BatchItemFailure> {
        let used = self.used.checked_add(work)
            .ok_or_else(|| BatchItemFailure::fatal(BatchCode::WorkLimit))?;
        let fits = match self.limits {
            None => used == SafeBpeWork::default(),
            Some(total) => used.input_bytes() as u64 <= total.max_input_bytes
                && used.output_tokens() as u64 <= total.max_output_tokens
                && used.pair_checks() <= total.max_pair_checks && used.pair_bytes() <= total.max_pair_bytes,
        };
        if !fits { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
        self.used = used; Ok(())
    }
}
fn map_failure(error: Int8ClassificationMapError, executing: bool) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    use Int8ClassificationMapError as M;
    use Int8ClassificationError as C;
    let mut failure = match error {
        M::EmptySource => BatchItemFailure::reject(BatchCode::DocumentLimit),
        M::Limits | M::Chunk(_) => BatchItemFailure::reject(BatchCode::Planning),
        M::Work | M::Classification(C::WorkBudget | C::Native(StrictInt8Error::Work)
            | C::Scoring(Int8ScoringError::Native(StrictInt8Error::Work)))
            => BatchItemFailure::reject(BatchCode::WorkLimit),
        M::Allocation | M::Classification(C::Scoring(Int8ScoringError::Allocation))
            => BatchItemFailure::fatal(BatchCode::Allocation),
        M::Output | M::Classification(C::Scoring(Int8ScoringError::OutputBudget))
            => BatchItemFailure::reject(BatchCode::OutputLineLimit),
        M::Serialization | M::Classification(C::Scoring(Int8ScoringError::Serialization))
            => BatchItemFailure::fatal(BatchCode::Serialization),
        M::Classification(C::Planning(error)) => super::super::planning_failure(error),
        M::Classification(C::Native(StrictInt8Error::Context | StrictInt8Error::Memory)
            | C::Scoring(Int8ScoringError::Native(StrictInt8Error::Context | StrictInt8Error::Memory)))
            => BatchItemFailure::reject(BatchCode::Admission),
        M::Admission | M::Classification(C::Identity) => BatchItemFailure::fatal(BatchCode::Admission),
        _ => BatchItemFailure::fatal(BatchCode::InvalidExecution),
    };
    if executing { failure.stop = true; }
    failure
}

#[cfg(test)] mod tests;
