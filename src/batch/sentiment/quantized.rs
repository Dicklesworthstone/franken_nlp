//! Complete INT8 sentiment documents on one resident invocation.
//!
//! Preparation seals the original source, every chunk identity and the exact
//! map proof. Replanning consumes the same document tokenizer allowance; every
//! pass, including rejected preparation, also spends the invocation ledger.
//! All five native counters are debited before an execution attempt.
use std::sync::Arc;
use crate::{
    batch::{BatchCode, BatchDocument, BatchFault, BatchItemFailure, BatchProcessor,
        BatchRequestContext, BatchWork, output::GuardedOutput},
    execution_identity::ExecutionIdentity,
    native_engine::{constrained_int8, decode::DecodeStepControl, kv::KV_BYTES_PER_TOKEN,
        portable_int8::ProjectionWork,
        strict_int8::{Int8RunBudget, Int8Work, StrictInt8Engine, StrictInt8Error,
            prefill::Int8PrefillLimits, scoring::{Int8ScoringBudget, Int8ScoringError}}},
    tasks::{ir::{PlanContext, TaskBudget}, sentiment::{SentimentAxis, SentimentError,
        SentimentLimits, SentimentPlanner, SentimentPlanningError, SentimentRequest,
        batch::SentimentBatchArgs,
        quantized::{Int8SentimentError, long::{Int8SentimentMapError,
            Int8SentimentMapLimits, Int8SentimentMapPreflight, Int8SentimentMapRun,
            PreparedInt8SentimentMap}}}},
    tokenizer::untrusted::{SafeBpeLimits, SafeBpeWork, SafeBpeWorkLimits, UntrustedEncoding},
};

/// Exact identities retained beside the one complete map during reconstruction.
/// Configuration bounds all identity text capacity before any per-chunk clone.
pub const INT8_SENTIMENT_DOCUMENT_CHUNK_METADATA_BYTES: u64 = 32_768;
const MAX_IDENTITY_TEXT_BYTES: u64 = 1024;

/// The planner fixes the score space, EOS, decision policy and source encoder.
/// Per-record arguments may replace only axes and monotonically tighter task
/// budgets. Mapping and invocation ceilings are separate, nonrenewable bounds.
/// The owned identity's String capacities together may not exceed 1,024 bytes.
/// No Debug/Serialize: execution identity may carry private commitments.
pub struct Int8SentimentDocumentBatchConfig {
    pub identity: ExecutionIdentity,
    pub task_ceiling: TaskBudget,
    pub planning: SentimentLimits,
    pub mapping: Int8SentimentMapLimits,
    pub defaults: Option<SentimentBatchArgs>,
    pub max_model_work: Int8Work,
    /// Required for compact input, absent for byte fallback. Counts accepted
    /// and rejected preparation plus exact reconstruction across every epoch.
    pub max_tokenizer_work: Option<SafeBpeWorkLimits>,
}
impl Int8SentimentDocumentBatchConfig {
    pub fn validate(&self, planner: &SentimentPlanner) -> Result<(), BatchFault> {
        self.mapping.validate(self.task_ceiling).map_err(|_| BatchCode::InvalidLimits)?;
        constrained_int8::check_profile(&self.identity).map_err(|_| BatchCode::Admission)?;
        validate_identity_storage(&self.identity)?;
        if self.identity.task_spec != "sentiment-v1"
            || self.identity.tokenizer_digest != planner.tokenizer_digest()
            || self.identity.template_digest != *planner.template_digest() {
            return Err(BatchCode::Admission.into());
        }
        if !nonzero(self.max_model_work) || !fits(self.mapping.max_model_work, self.max_model_work) {
            return Err(BatchCode::InvalidLimits.into());
        }
        let p = self.planning; let a = p.per_axis;
        if a.max_candidates == 0 || a.max_total_tokens == 0 || a.max_nodes == 0
            || a.max_depth == 0 || a.max_candidate_id_bytes == 0 || a.max_projected_logits == 0
            || p.max_total_prompt_tokens == 0 || p.max_total_candidate_tokens == 0
            || p.max_total_nodes == 0 || p.max_total_projected_logits == 0 || p.max_output_bytes == 0 {
            return Err(BatchCode::InvalidLimits.into());
        }
        validate_encoding(planner.document_encoding(), self.max_tokenizer_work)?;
        if let Some(args) = &self.defaults {
            validate_args(args, self.task_ceiling).map_err(|error| error.fault)?;
        }
        Ok(())
    }

    /// Price transferred vector capacity and the bounded physical-join
    /// keepalive copy, including overallocated embedding-caller defaults.
    pub fn retained_defaults_bytes(&self) -> Result<u64, BatchFault> {
        self.defaults.as_ref().map_or(Ok(0), |args| (args.axes.capacity() as u64)
            .checked_add(args.axes.len() as u64)
            .and_then(|axes| axes.checked_mul(std::mem::size_of::<SentimentAxis>() as u64))
            .ok_or_else(|| BatchCode::InvalidLimits.into()))
    }
}

// Fixed per-chunk metadata must remain valid for embedding callers too.
// Length alone is insufficient: the moved base identity retains String spare
// capacity even though its later clones may be compact.
fn validate_identity_storage(identity: &ExecutionIdentity) -> Result<(), BatchFault> {
    let required = [&identity.source_revision, &identity.artifact_format,
        &identity.quant_recipe, &identity.task_spec, &identity.grammar_compiler_version,
        &identity.kv_dtype, &identity.sampler_version, &identity.backend_semantic_version];
    let total = required.into_iter()
        .chain(identity.host_class.iter()).chain(identity.compiler_identity.iter())
        .try_fold(0_u64, |sum, value| sum.checked_add(value.capacity() as u64))
        .ok_or(BatchCode::InvalidLimits)?;
    if total > MAX_IDENTITY_TEXT_BYTES { return Err(BatchCode::InvalidLimits.into()); }
    Ok(())
}

pub struct Int8SentimentDocumentBatchPlanner<'p> {
    planner: &'p SentimentPlanner,
    config: Int8SentimentDocumentBatchConfig,
    seal: Arc<()>,
}
impl<'p> Int8SentimentDocumentBatchPlanner<'p> {
    pub fn new(planner: &'p SentimentPlanner, config: Int8SentimentDocumentBatchConfig)
        -> Result<Self, BatchFault> {
        config.validate(planner)?;
        Ok(Self { planner, config, seal: Arc::new(()) })
    }
    pub fn base_execution_identity(&self) -> &ExecutionIdentity { &self.config.identity }
    pub fn task_budget(&self) -> TaskBudget { self.config.task_ceiling }
    pub fn mapping_limits(&self) -> Int8SentimentMapLimits { self.config.mapping }
    pub fn max_run_work(&self) -> Int8Work { self.config.max_model_work }
    pub fn document_encoding(&self) -> &UntrustedEncoding { self.planner.document_encoding() }

    fn request(&self, document: BatchDocument<SentimentBatchArgs>)
        -> Result<SentimentRequest, BatchItemFailure> {
        if document.text.is_empty() || document.text.len() > self.config.mapping.chunks.max_input_bytes {
            return Err(BatchItemFailure::reject(BatchCode::DocumentLimit));
        }
        let args = document.task_args.or_else(|| self.config.defaults.clone())
            .ok_or_else(|| BatchItemFailure::reject(BatchCode::Planning))?;
        validate_args(&args, self.config.task_ceiling)?;
        let mut axes = args.axes;
        axes.sort_unstable();
        Ok(SentimentRequest { document: document.text, axes, budget: args.budget })
    }

    fn plan<'s, C: DecodeStepControl>(&self, request: &'s SentimentRequest,
        document_work: &mut SafeBpeWork, ledger: &mut RunState, control: &mut C, executing: bool)
        -> Result<PreparedInt8SentimentMap<'s>, BatchItemFailure> {
        let ceiling = ledger.encoding_ceiling(self.document_encoding(), *document_work)?;
        let context = PlanContext::new(&self.config.identity, self.config.task_ceiling)
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        let before = *document_work;
        let result = self.planner.plan_int8_document_with_encoding_ceiling(request, &context,
            self.config.planning, self.config.mapping, control, document_work, ceiling);
        // Charge partial work even when a later probe, chunk, capacity check or
        // cancellation rejects the document. There is no successful-only debit.
        let charged = document_work.checked_sub(before)
            .ok_or_else(|| BatchItemFailure::fatal(BatchCode::InvalidExecution))
            .and_then(|delta| ledger.charge_encoding(delta));
        if let Err(fault) = charged {
            if let Err(error) = &result {
                if let Some(cause) = error.cancellation() {
                    return Err(BatchItemFailure::fatal(BatchFault::cancelled(cause)));
                }
            }
            return Err(fault);
        }
        result.map_err(|error| {
            // An externally tightened limit is invocation authority. Do not
            // pretend its refusal is a fresh per-document allowance.
            let total_limited = match (self.document_encoding(), ceiling) {
                (UntrustedEncoding::SafeBpeV1(original), Some(actual)) => *original != actual,
                _ => false,
            };
            if total_limited && matches!(&error, Int8SentimentMapError::Sentiment(
                Int8SentimentError::Planning(SentimentPlanningError::Budget))) {
                BatchItemFailure::fatal(BatchCode::WorkLimit)
            } else { map_failure(error, executing) }
        })
    }
}

/// Original source and unforgeable factory/request proof. Wire data cannot
/// construct this type or supply synthetic chunk results for publication.
pub struct PreparedInt8SentimentDocument {
    request: SentimentRequest,
    expected: Int8SentimentMapPreflight,
    identities: Vec<ExecutionIdentity>,
    required_context: usize,
    head_count: usize,
    encoding_work: SafeBpeWork,
    seal: Arc<()>,
}
impl PreparedInt8SentimentDocument {
    pub fn execution_identities(&self) -> &[ExecutionIdentity] { &self.identities }
    pub fn planned_work(&self) -> Int8Work { self.expected.planned_work() }
    pub fn chunk_count(&self) -> usize { self.expected.chunk_count() }
    pub fn encoding_work(&self) -> SafeBpeWork { self.encoding_work }
}

/// Host admission covers every exact chunk and the complete result at once.
pub struct Int8SentimentDocumentAdmission<'a> {
    pub identities: &'a [ExecutionIdentity],
    pub model_work: Int8Work,
    pub required_context: usize,
    pub head_count: usize,
    pub task_budget: TaskBudget,
    pub kv_reservation_bytes: u64,
    pub max_result_bytes: u64,
}
pub trait Int8SentimentDocumentBatchAdmission {
    type Guard;
    fn admit(&mut self, request: Int8SentimentDocumentAdmission<'_>)
        -> Result<(Vec<ExecutionIdentity>, Self::Guard), BatchItemFailure>;
}

pub struct NativeInt8SentimentDocumentBatch<'p, 'e, 'w, A: Int8SentimentDocumentBatchAdmission> {
    compiler: Int8SentimentDocumentBatchPlanner<'p>,
    driver: NativeDriver<'e, 'w>,
    admission: A,
    run: RunState,
    prefill: Option<Int8PrefillLimits>,
}
impl<'p, 'e, 'w, A: Int8SentimentDocumentBatchAdmission>
    NativeInt8SentimentDocumentBatch<'p, 'e, 'w, A> {
    pub fn new(compiler: Int8SentimentDocumentBatchPlanner<'p>, engine: &'e mut StrictInt8Engine<'w>,
        admission: A) -> Result<Self, BatchFault> {
        let driver = NativeDriver(engine);
        if !driver.clean() || compiler.config.mapping.chunks.context_tokens > driver.capacity() {
            return Err(BatchCode::InvalidExecution.into());
        }
        capacity_bytes(driver.capacity(), compiler.config.task_ceiling, 4)
            .map_err(|error| error.fault)?;
        let run = RunState::new(compiler.config.max_model_work, compiler.config.max_tokenizer_work);
        Ok(Self { compiler, driver, admission, run, prefill: None })
    }
    /// One bounded prompt workspace is reused across all chunks and axes.
    pub fn new_layer_major(compiler: Int8SentimentDocumentBatchPlanner<'p>,
        engine: &'e mut StrictInt8Engine<'w>, admission: A, prefill: Int8PrefillLimits)
        -> Result<Self, BatchFault> {
        prefill.validate().map_err(|_| BatchCode::InvalidLimits)?;
        let mut processor = Self::new(compiler, engine, admission)?;
        processor.prefill = Some(prefill);
        Ok(processor)
    }
    pub fn remaining_work(&self) -> Int8Work { self.run.remaining }
    pub fn encoding_work(&self) -> SafeBpeWork { self.run.encoding }
    pub fn is_poisoned(&self) -> bool { self.run.state != State::Ready || !self.driver.clean() }
}
impl<A: Int8SentimentDocumentBatchAdmission> BatchProcessor
    for NativeInt8SentimentDocumentBatch<'_, '_, '_, A> {
    type Args = SentimentBatchArgs;
    type Prepared = PreparedInt8SentimentDocument;
    type Output = GuardedOutput<Int8SentimentMapRun, A::Guard>;

    fn prepare(&mut self, _: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        self.run.fail();
        Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>,
        control: &mut C) -> Result<Self::Prepared, BatchItemFailure> {
        prepare_document(&self.compiler, &mut self.driver, &mut self.run, document, control)
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork {
        let work = prepared.planned_work();
        BatchWork { forward_positions: work.forward_positions, projected_logits: work.projected_logits }
    }
    fn execute<C: DecodeStepControl>(&mut self, _: Self::Prepared, _: &mut C)
        -> Result<Self::Output, BatchItemFailure> {
        self.run.fail();
        Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        execute_document(&self.compiler, &mut self.driver, &mut self.admission, &mut self.run,
            self.prefill, prepared, context, control)
    }
}

trait Driver {
    fn clean(&self) -> bool;
    fn capacity(&self) -> usize;
    fn preflight<C: DecodeStepControl>(&mut self, plan: &PreparedInt8SentimentMap<'_>,
        identities: &[ExecutionIdentity], kv_bytes: u64, control: &mut C)
        -> Result<(), Int8SentimentMapError>;
    fn execute<C: DecodeStepControl>(&mut self, plan: &PreparedInt8SentimentMap<'_>,
        identities: &[ExecutionIdentity], kv_bytes: u64, prefill: Option<Int8PrefillLimits>,
        control: &mut C) -> Result<Int8SentimentMapRun, Int8SentimentMapError>;
}
struct NativeDriver<'e, 'w>(&'e mut StrictInt8Engine<'w>);
impl Driver for NativeDriver<'_, '_> {
    fn clean(&self) -> bool { !self.0.is_poisoned() && self.0.kv_cache().all_slots_have_len(0) }
    fn capacity(&self) -> usize { self.0.kv_cache().capacity_positions() }
    fn preflight<C: DecodeStepControl>(&mut self, plan: &PreparedInt8SentimentMap<'_>,
        identities: &[ExecutionIdentity], kv_bytes: u64, control: &mut C)
        -> Result<(), Int8SentimentMapError> {
        plan.preflight_with_control(identities, self.0, native_budget(plan, kv_bytes), control)
    }
    fn execute<C: DecodeStepControl>(&mut self, plan: &PreparedInt8SentimentMap<'_>,
        identities: &[ExecutionIdentity], kv_bytes: u64, prefill: Option<Int8PrefillLimits>,
        control: &mut C) -> Result<Int8SentimentMapRun, Int8SentimentMapError> {
        let budget = native_budget(plan, kv_bytes);
        match prefill {
            None => plan.execute_with_control(identities, self.0, budget, control),
            Some(prefill) => plan.execute_layer_major_with_control(identities, self.0, budget, prefill, control),
        }
    }
}
fn native_budget(plan: &PreparedInt8SentimentMap<'_>, kv_bytes: u64) -> Int8ScoringBudget {
    Int8ScoringBudget { native: Int8RunBudget::exact(plan.planned_work()), max_kv_bytes: kv_bytes }
}
fn preflight<D: Driver, C: DecodeStepControl>(driver: &mut D, plan: &PreparedInt8SentimentMap<'_>,
    identities: &[ExecutionIdentity], control: &mut C) -> Result<u64, BatchItemFailure> {
    if !driver.clean() { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
    let bytes = capacity_bytes(driver.capacity(), plan.task_budget(), plan.head_count())?;
    if plan.required_context() > driver.capacity() {
        return Err(BatchItemFailure::reject(BatchCode::Admission));
    }
    driver.preflight(plan, identities, bytes, control).map_err(|error| map_failure(error, false))?;
    Ok(bytes)
}
fn capacity_bytes(capacity: usize, task: TaskBudget, heads: usize) -> Result<u64, BatchItemFailure> {
    let bytes = (capacity as u64).checked_mul(KV_BYTES_PER_TOKEN as u64)
        .ok_or_else(|| BatchItemFailure::fatal(BatchCode::InvalidExecution))?;
    if !(1..=4).contains(&heads) || bytes == 0
        || bytes.checked_mul(heads as u64).is_none_or(|needed| needed > task.max_kv_bytes) {
        return Err(BatchItemFailure::reject(BatchCode::Admission));
    }
    Ok(bytes)
}
fn prepare_document<D: Driver, C: DecodeStepControl>(compiler: &Int8SentimentDocumentBatchPlanner<'_>,
    driver: &mut D, run: &mut RunState, document: BatchDocument<SentimentBatchArgs>, control: &mut C)
    -> Result<PreparedInt8SentimentDocument, BatchItemFailure> {
    run.enter()?;
    let result = (|| {
        checkpoint(control)?;
        let request = compiler.request(document)?;
        let mut encoding_work = SafeBpeWork::default();
        let plan = compiler.plan(&request, &mut encoding_work, run, control, false)?;
        subtract(run.remaining, plan.planned_work())?;
        let identities = plan.identities().map_err(|error| map_failure(error, false))?;
        preflight(driver, &plan, &identities, control)?;
        if plan.chunk_count() == 0 || identities.len() != plan.chunk_count()
            || plan.head_count() != request.axes.len() || plan.task_budget() != request.budget
            || plan.max_result_bytes() != compiler.config.mapping.max_result_bytes {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        let expected = plan.preflight_metadata();
        let required_context = plan.required_context();
        let head_count = plan.head_count();
        drop(plan);
        checkpoint(control)?;
        Ok(PreparedInt8SentimentDocument { request, expected, identities, required_context,
            head_count, encoding_work, seal: Arc::clone(&compiler.seal) })
    })();
    run.finish(result, driver.clean(), false)
}

#[allow(clippy::too_many_arguments)]
fn execute_document<D: Driver, A: Int8SentimentDocumentBatchAdmission, C: DecodeStepControl>(
    compiler: &Int8SentimentDocumentBatchPlanner<'_>, driver: &mut D, admission: &mut A,
    run: &mut RunState, prefill: Option<Int8PrefillLimits>, mut prepared: PreparedInt8SentimentDocument,
    context: BatchRequestContext, control: &mut C)
    -> Result<GuardedOutput<Int8SentimentMapRun, A::Guard>, BatchItemFailure> {
    run.enter()?;
    let result = (|| {
        if !Arc::ptr_eq(&prepared.seal, &compiler.seal) {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        // An admitted attempt is never refunded, even if reconstruction,
        // host admission or the first native checkpoint subsequently refuses.
        run.charge(prepared.planned_work(), context)?;
        checkpoint(control)?;
        let plan = compiler.plan(&prepared.request, &mut prepared.encoding_work, run, control, true)?;
        let identities = plan.identities().map_err(|error| map_failure(error, true))?;
        if plan.preflight_metadata() != prepared.expected || identities != prepared.identities
            || plan.required_context() != prepared.required_context || plan.head_count() != prepared.head_count {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        let kv_bytes = preflight(driver, &plan, &identities, control)?;
        let (admitted, guard) = admission.admit(Int8SentimentDocumentAdmission {
            identities: &prepared.identities, model_work: prepared.planned_work(),
            required_context: prepared.required_context, head_count: prepared.head_count,
            task_budget: prepared.request.budget, kv_reservation_bytes: kv_bytes,
            max_result_bytes: compiler.config.mapping.max_result_bytes,
        })?;
        if admitted != prepared.identities { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
        checkpoint(control)?;
        let result = driver.execute(&plan, &admitted, kv_bytes, prefill, control)
            .map_err(|error| map_failure(error, true))?;
        prepared.expected.verify_completed_with_control(&result, control)
            .map_err(|error| map_failure(error, true))?;
        checkpoint(control)?;
        Ok(GuardedOutput::new(result, guard))
    })();
    run.finish(result, driver.clean(), true)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum State { Ready, Running, Failed }
struct RunState {
    remaining: Int8Work,
    last_sequence: u64,
    state: State,
    encoding: SafeBpeWork,
    max_encoding: Option<SafeBpeWorkLimits>,
}
impl RunState {
    fn new(remaining: Int8Work, max_encoding: Option<SafeBpeWorkLimits>) -> Self {
        Self { remaining, last_sequence: 0, state: State::Ready,
            encoding: SafeBpeWork::default(), max_encoding }
    }
    fn fail(&mut self) { self.state = State::Failed; }
    fn enter(&mut self) -> Result<(), BatchItemFailure> {
        if self.state != State::Ready { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
        self.state = State::Running;
        Ok(())
    }
    fn charge(&mut self, work: Int8Work, context: BatchRequestContext) -> Result<(), BatchItemFailure> {
        if self.state != State::Running || context.request_seq <= self.last_sequence
            || context.epoch == 0 || context.input_line == 0 {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        let next = subtract(self.remaining, work).map_err(|error| BatchItemFailure::fatal(error.fault))?;
        self.remaining = next;
        self.last_sequence = context.request_seq;
        Ok(())
    }
    fn finish<T>(&mut self, result: Result<T, BatchItemFailure>, clean: bool, executing: bool)
        -> Result<T, BatchItemFailure> {
        match result {
            Ok(value) if clean && self.state == State::Running => {
                self.state = State::Ready;
                Ok(value)
            },
            Ok(_) => {
                self.fail();
                Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
            },
            Err(mut error) => {
                if executing || !clean || self.state != State::Running || error.fault.cancellation.is_some() {
                    error.stop = true;
                }
                self.state = if error.stop { State::Failed } else { State::Ready };
                Err(error)
            },
        }
    }
    fn charge_encoding(&mut self, delta: SafeBpeWork) -> Result<(), BatchItemFailure> {
        let next = self.encoding.checked_add(delta)
            .ok_or_else(|| BatchItemFailure::fatal(BatchCode::WorkLimit))?;
        match self.max_encoding {
            None if next != SafeBpeWork::default() => return Err(BatchItemFailure::fatal(BatchCode::Admission)),
            Some(limits) if !encoding_fits(next, limits) => return Err(BatchItemFailure::fatal(BatchCode::WorkLimit)),
            _ => {},
        }
        self.encoding = next;
        Ok(())
    }
    fn encoding_ceiling(&self, policy: &UntrustedEncoding, document: SafeBpeWork)
        -> Result<Option<SafeBpeLimits>, BatchItemFailure> {
        match (policy, self.max_encoding) {
            (UntrustedEncoding::ByteFallback, None) if document == SafeBpeWork::default() => Ok(None),
            (UntrustedEncoding::SafeBpeV1(original), Some(limits)) => {
                let remaining = limits.remaining_call(self.encoding, *original)
                    .map_err(|_| BatchItemFailure::fatal(BatchCode::WorkLimit))?;
                // The map receives a cumulative DOCUMENT ledger. Add only the
                // invocation's unspent capacity to its already charged prefix,
                // then meet it with the original per-document policy.
                let add = |used: u64, left: u64, cap: u64| {
                    used.checked_add(left).map(|value| value.min(cap))
                        .filter(|&value| value > used)
                        .ok_or_else(|| BatchItemFailure::fatal(BatchCode::WorkLimit))
                };
                Ok(Some(SafeBpeLimits {
                    max_input_bytes: usize::try_from(add(document.input_bytes() as u64,
                        remaining.max_input_bytes as u64, original.max_input_bytes as u64)?)
                        .map_err(|_| BatchItemFailure::fatal(BatchCode::WorkLimit))?,
                    max_output_tokens: usize::try_from(add(document.output_tokens() as u64,
                        remaining.max_output_tokens as u64, original.max_output_tokens as u64)?)
                        .map_err(|_| BatchItemFailure::fatal(BatchCode::WorkLimit))?,
                    max_pair_checks: add(document.pair_checks(), remaining.max_pair_checks, original.max_pair_checks)?,
                    max_pair_bytes: add(document.pair_bytes(), remaining.max_pair_bytes, original.max_pair_bytes)?,
                }))
            },
            _ => Err(BatchItemFailure::fatal(BatchCode::Admission)),
        }
    }
}
fn validate_encoding(policy: &UntrustedEncoding, limits: Option<SafeBpeWorkLimits>) -> Result<(), BatchFault> {
    policy.validate().map_err(|_| BatchCode::InvalidLimits)?;
    match (policy, limits) {
        (UntrustedEncoding::ByteFallback, None) => Ok(()),
        (UntrustedEncoding::SafeBpeV1(per_document), Some(total)) => total
            .remaining_call(SafeBpeWork::default(), *per_document).map(|_| ())
            .map_err(|_| BatchCode::InvalidLimits.into()),
        _ => Err(BatchCode::InvalidLimits.into()),
    }
}
fn encoding_fits(work: SafeBpeWork, limits: SafeBpeWorkLimits) -> bool {
    work.input_bytes() as u64 <= limits.max_input_bytes
        && work.output_tokens() as u64 <= limits.max_output_tokens
        && work.pair_checks() <= limits.max_pair_checks && work.pair_bytes() <= limits.max_pair_bytes
}
fn validate_args(args: &SentimentBatchArgs, ceiling: TaskBudget) -> Result<(), BatchItemFailure> {
    args.budget.validate().map_err(|_| BatchItemFailure::reject(BatchCode::Planning))?;
    let b = args.budget;
    if args.axes.is_empty() || args.axes.len() > 4
        || args.axes.iter().enumerate().any(|(index, axis)| args.axes[..index].contains(axis))
        || b.max_input_tokens > ceiling.max_input_tokens || b.max_output_tokens > ceiling.max_output_tokens
        || b.max_output_bytes > ceiling.max_output_bytes || b.max_grammar_states > ceiling.max_grammar_states
        || b.max_kv_bytes > ceiling.max_kv_bytes {
        return Err(BatchItemFailure::reject(BatchCode::Planning));
    }
    Ok(())
}
fn nonzero(work: Int8Work) -> bool {
    work.forward_positions != 0 && work.projected_logits != 0 && work.attention_pairs != 0
        && work.projections.dot_products != 0 && work.projections.multiply_accumulates != 0
}
fn fits(work: Int8Work, ceiling: Int8Work) -> bool {
    work.forward_positions <= ceiling.forward_positions && work.projected_logits <= ceiling.projected_logits
        && work.attention_pairs <= ceiling.attention_pairs && work.projections.fits(ceiling.projections)
}
fn subtract(available: Int8Work, work: Int8Work) -> Result<Int8Work, BatchItemFailure> {
    let sub = |a: u64, b: u64| a.checked_sub(b).ok_or_else(|| BatchItemFailure::reject(BatchCode::WorkLimit));
    Ok(Int8Work { forward_positions: sub(available.forward_positions, work.forward_positions)?,
        projected_logits: sub(available.projected_logits, work.projected_logits)?,
        attention_pairs: sub(available.attention_pairs, work.attention_pairs)?,
        projections: ProjectionWork { dot_products: sub(available.projections.dot_products, work.projections.dot_products)?,
            multiply_accumulates: sub(available.projections.multiply_accumulates, work.projections.multiply_accumulates)? } })
}
fn checkpoint<C: DecodeStepControl>(control: &mut C) -> Result<(), BatchItemFailure> {
    match control.prefill_checkpoint(0) {
        Some(cause) => Err(BatchItemFailure::fatal(BatchFault::cancelled(cause))),
        None => Ok(()),
    }
}
fn map_failure(error: Int8SentimentMapError, executing: bool) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() {
        return BatchItemFailure::fatal(BatchFault::cancelled(cause));
    }
    let (code, fatal) = match error {
        Int8SentimentMapError::Allocation
            | Int8SentimentMapError::Sentiment(Int8SentimentError::Allocation
                | Int8SentimentError::Planning(SentimentPlanningError::AllocationRefused)
                | Int8SentimentError::Task(SentimentError::AllocationRefused)
                | Int8SentimentError::Native(StrictInt8Error::Allocation)
                | Int8SentimentError::Scoring(Int8ScoringError::Allocation)) => (BatchCode::Allocation, true),
        Int8SentimentMapError::Admission
            | Int8SentimentMapError::Sentiment(Int8SentimentError::Identity
                | Int8SentimentError::Planning(SentimentPlanningError::Identity)) => (BatchCode::Admission, true),
        Int8SentimentMapError::Sentiment(Int8SentimentError::Accounting) => (BatchCode::InvalidExecution, true),
        Int8SentimentMapError::Serialization
            | Int8SentimentMapError::Sentiment(Int8SentimentError::Serialization) => (BatchCode::Serialization, true),
        Int8SentimentMapError::Work
            | Int8SentimentMapError::Sentiment(Int8SentimentError::WorkBudget) => (BatchCode::WorkLimit, false),
        Int8SentimentMapError::Output
            | Int8SentimentMapError::Sentiment(Int8SentimentError::OutputBudget) => (BatchCode::OutputLineLimit, false),
        _ if executing => (BatchCode::Execution, true),
        _ => (BatchCode::Planning, false),
    };
    if executing || fatal { BatchItemFailure::fatal(code) } else { BatchItemFailure::reject(code) }
}

#[cfg(test)]
mod tests;
