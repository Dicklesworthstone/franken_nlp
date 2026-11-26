//! Complete-source faithfulness populations on one resident native INT8 engine.
//! Every original claim/source pair is one guarded result. Preparation and
//! exact reconstruction share tokenizer work; native work never renews at flush.
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use crate::{
    batch::{BatchCode, BatchDocument, BatchFault, BatchItemFailure, BatchProcessor,
        BatchRequestContext, BatchWork, output::GuardedOutput},
    execution_identity::{ExecutionIdentity, NumericsProfile, ThinkingMode, ToolMode},
    native_engine::{decode::DecodeStepControl, portable_int8::ProjectionWork,
        strict_int8::{Int8MemoryRequirement, Int8RunBudget, Int8Work, StrictInt8Engine,
            StrictInt8Error, STRICT_INT8_EXECUTION, prefill::Int8PrefillLimits,
            scoring::{Int8ScoringBudget, Int8ScoringError}}},
    tasks::{ir::{PlanContext, TaskBudget}, judge::{FaithfulnessPolicy, JudgeError,
        JudgeLimits, JudgePlanner, JudgeRequest, quantized::{Int8JudgeError,
            faithfulness_map::{Int8FaithfulnessMapError, Int8FaithfulnessMapLimits,
                Int8FaithfulnessMapPreflight, Int8FaithfulnessMapRun,
                PreparedInt8FaithfulnessMap}}}},
    tokenizer::untrusted::{SafeBpeWork, SafeBpeWorkLimits, UntrustedEncoding},
};

/// Frozen per-chunk identities and admission copies coexist with the rebuilt
/// native map. This is additional to the complete map's own memory quote.
pub const FAITHFULNESS_DOCUMENT_CHUNK_METADATA_BYTES: u64 = 32_768;

/// The fixed policy and task budget belong to the invocation. An input record
/// can supply only its exact private claim beside its original source text.
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Int8FaithfulnessDocumentArgs { pub claim: String }

pub struct Int8FaithfulnessDocumentBatchPlanner {
    planner: Arc<JudgePlanner>,
    identity: ExecutionIdentity,
    ceiling: TaskBudget,
    planning: JudgeLimits,
    mapping: Int8FaithfulnessMapLimits,
    policy: FaithfulnessPolicy,
    max_model_work: Int8Work,
    max_tokenizer_work: Option<SafeBpeWorkLimits>,
    seal: Arc<()>,
}
impl Int8FaithfulnessDocumentBatchPlanner {
    #[allow(clippy::too_many_arguments)]
    pub fn new(planner: Arc<JudgePlanner>, identity: ExecutionIdentity,
        ceiling: TaskBudget, planning: JudgeLimits, mapping: Int8FaithfulnessMapLimits,
        policy: FaithfulnessPolicy, max_model_work: Int8Work,
        max_tokenizer_work: Option<SafeBpeWorkLimits>) -> Result<Self, BatchFault> {
        Self::validate_configuration(&planner, &identity, ceiling, planning, mapping,
            policy, max_model_work, max_tokenizer_work)?;
        Ok(Self { planner, identity, ceiling, planning, mapping, policy,
            max_model_work, max_tokenizer_work, seal: Arc::new(()) })
    }

    /// Fixed configuration only: no invented source and no tokenizer work.
    #[allow(clippy::too_many_arguments)]
    pub fn validate_configuration(planner: &JudgePlanner, identity: &ExecutionIdentity,
        ceiling: TaskBudget, planning: JudgeLimits, mapping: Int8FaithfulnessMapLimits,
        policy: FaithfulnessPolicy, total: Int8Work,
        encoding: Option<SafeBpeWorkLimits>) -> Result<(), BatchFault> {
        identity.validate().map_err(|_| BatchCode::Admission)?;
        validate_identity_owner(identity)?;
        if identity.task_spec != "judge-v1" || identity.template_digest != *planner.template_digest()
            || identity.tokenizer_digest != planner.tokenizer_digest()
            || identity.numerics_profile != (NumericsProfile::StrictQuantized { version: 1 })
            || identity.backend_semantic_version != STRICT_INT8_EXECUTION || identity.kv_dtype != "bf16"
            || identity.thinking_mode != ThinkingMode::Disabled || identity.tool_mode != ToolMode::None {
            return Err(BatchCode::Admission.into());
        }
        mapping.validate(ceiling).map_err(|_| BatchCode::InvalidLimits)?;
        policy.validate().map_err(|_| BatchCode::InvalidLimits)?;
        EncodingLedger::new(planner.document_encoding(), encoding)?;
        let h = planning.per_head;
        if !nonzero(total) || subtract(total, mapping.max_model_work).is_err()
            || h.max_candidates == 0 || h.max_total_tokens == 0 || h.max_nodes == 0
            || h.max_depth == 0 || h.max_candidate_id_bytes == 0 || h.max_projected_logits == 0
            || planning.max_total_prompt_tokens == 0 || planning.max_total_candidate_tokens == 0
            || planning.max_total_projected_logits == 0
            || !(1..=64 * 1024 * 1024).contains(&planning.max_output_bytes)
            || ceiling.max_output_bytes > mapping.max_retained_result_bytes {
            return Err(BatchCode::InvalidLimits.into());
        }
        Ok(())
    }
    pub fn base_execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    pub fn task_budget(&self) -> TaskBudget { self.ceiling }
    pub fn planning_limits(&self) -> JudgeLimits { self.planning }
    pub fn mapping_limits(&self) -> Int8FaithfulnessMapLimits { self.mapping }
    pub fn policy(&self) -> FaithfulnessPolicy { self.policy }
    pub fn max_run_work(&self) -> Int8Work { self.max_model_work }
    pub fn document_encoding(&self) -> &UntrustedEncoding { self.planner.document_encoding() }
    pub fn max_tokenizer_work(&self) -> Option<SafeBpeWorkLimits> { self.max_tokenizer_work }

    fn request(&self, document: BatchDocument<Int8FaithfulnessDocumentArgs>)
        -> Result<JudgeRequest, BatchItemFailure> {
        let args = document.task_args.ok_or_else(|| BatchItemFailure::reject(BatchCode::Planning))?;
        let cap = self.mapping.chunks.max_input_bytes;
        if document.text.is_empty() || args.claim.is_empty()
            || document.text.len().checked_add(args.claim.len()).is_none_or(|n| n > cap)
            || document.text.capacity().checked_add(args.claim.capacity())
                .zip(cap.checked_mul(4)).is_none_or(|(n, maximum)| n > maximum) {
            return Err(BatchItemFailure::reject(BatchCode::DocumentLimit));
        }
        Ok(JudgeRequest::Faithfulness { source: document.text, claim: args.claim,
            policy: self.policy, budget: self.ceiling })
    }

    fn plan<'a, C: DecodeStepControl>(&self, request: &'a JudgeRequest,
        document_work: &mut SafeBpeWork, ledger: &mut EncodingLedger, control: &mut C,
        executing: bool) -> Result<PreparedInt8FaithfulnessMap<'a>, BatchItemFailure> {
        checkpoint(control)?;
        ledger.admit(self.document_encoding())?;
        let context = PlanContext::new(&self.identity, self.ceiling)
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        let before = *document_work;
        let result = self.planner.plan_int8_faithfulness_document_with_encoding_work(request,
            &context, self.planning, self.mapping, control, document_work);
        // Includes work spent before a later chunk, planner or cancellation
        // refusal. Reconstruction cannot acquire a new document allowance.
        ledger.charge(document_work.checked_sub(before)
            .ok_or_else(|| BatchItemFailure::fatal(BatchCode::InvalidExecution))?)?;
        result.map_err(|error| map_failure(error, executing))
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare<C, F>(&self, document: BatchDocument<Int8FaithfulnessDocumentArgs>, remaining: Int8Work,
        ledger: &mut EncodingLedger, invocation: &Arc<()>, control: &mut C, preflight: F)
        -> Result<PreparedInt8FaithfulnessDocument, BatchItemFailure>
    where C: DecodeStepControl,
        F: FnOnce(&PreparedInt8FaithfulnessMap<'_>, &[ExecutionIdentity], &mut C)
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
        Ok(PreparedInt8FaithfulnessDocument { request, expected, identities,
            required_context, encoding_work, seal: Arc::clone(&self.seal),
            invocation: Arc::clone(invocation) })
    }
    fn verify(&self, prepared: &PreparedInt8FaithfulnessDocument, invocation: &Arc<()>)
        -> Result<(), BatchItemFailure> {
        if !Arc::ptr_eq(&self.seal, &prepared.seal) || !Arc::ptr_eq(invocation, &prepared.invocation) {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        Ok(())
    }
}

/// No public constructor, deserializer or source/result replacement surface.
pub struct PreparedInt8FaithfulnessDocument {
    request: JudgeRequest,
    expected: Int8FaithfulnessMapPreflight,
    identities: Vec<ExecutionIdentity>,
    required_context: usize,
    encoding_work: SafeBpeWork,
    seal: Arc<()>,
    invocation: Arc<()>,
}
impl PreparedInt8FaithfulnessDocument {
    pub fn execution_identities(&self) -> &[ExecutionIdentity] { &self.identities }
    pub fn planned_work(&self) -> Int8Work { self.expected.planned_work() }
    pub fn chunk_count(&self) -> usize { self.expected.chunk_count() }
    pub fn encoding_work(&self) -> SafeBpeWork { self.encoding_work }
}

pub struct Int8FaithfulnessDocumentAdmission<'a> {
    pub identities: &'a [ExecutionIdentity],
    pub model_work: Int8Work,
    pub required_context: usize,
    pub task_budget: TaskBudget,
    pub max_head_count: usize,
    pub kv_reservation_bytes: u64,
    pub max_result_bytes: u64,
}
pub trait Int8FaithfulnessDocumentBatchAdmission {
    type Guard;
    fn admit(&mut self, request: Int8FaithfulnessDocumentAdmission<'_>)
        -> Result<(Vec<ExecutionIdentity>, Self::Guard), BatchItemFailure>;
}

pub struct NativeInt8FaithfulnessDocumentBatch<'e, 'w, A: Int8FaithfulnessDocumentBatchAdmission> {
    compiler: Arc<Int8FaithfulnessDocumentBatchPlanner>,
    engine: &'e mut StrictInt8Engine<'w>,
    admission: A,
    run: RunState,
    encoding: EncodingLedger,
    invocation: Arc<()>,
    prefill: Option<Int8PrefillLimits>,
}
impl<'e, 'w, A: Int8FaithfulnessDocumentBatchAdmission> NativeInt8FaithfulnessDocumentBatch<'e, 'w, A> {
    pub fn new(compiler: Arc<Int8FaithfulnessDocumentBatchPlanner>, engine: &'e mut StrictInt8Engine<'w>,
        admission: A) -> Result<Self, BatchFault> {
        capacity(&compiler, engine).map_err(|error| error.fault)?;
        let encoding = EncodingLedger::new(compiler.document_encoding(), compiler.max_tokenizer_work())?;
        let run = RunState::new(compiler.max_run_work());
        Ok(Self { compiler, engine, admission, run, encoding, invocation: Arc::new(()), prefill: None })
    }
    pub fn new_layer_major(compiler: Arc<Int8FaithfulnessDocumentBatchPlanner>,
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
impl<A: Int8FaithfulnessDocumentBatchAdmission> BatchProcessor for NativeInt8FaithfulnessDocumentBatch<'_, '_, A> {
    type Args = Int8FaithfulnessDocumentArgs;
    type Prepared = PreparedInt8FaithfulnessDocument;
    type Output = GuardedOutput<Int8FaithfulnessMapRun, A::Guard>;
    fn prepare(&mut self, _: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        self.run.state = State::Failed;
        Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>,
        control: &mut C) -> Result<Self::Prepared, BatchItemFailure> {
        self.run.enter()?;
        let compiler = &self.compiler;
        let engine = &*self.engine;
        let result = compiler.prepare(document, self.run.remaining, &mut self.encoding, &self.invocation,
            control, |plan, identities, control| {
                let kv = capacity(compiler, engine)?;
                plan.preflight_with_control(identities, engine, budget(plan.planned_work(), kv), control)
                    .map_err(|error| map_failure(error, false))
            });
        self.run.finish(result, clean(self.engine), false)
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork {
        let work = prepared.planned_work();
        BatchWork { forward_positions: work.forward_positions, projected_logits: work.projected_logits }
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
            // Owner verification precedes this nonrefundable five-axis debit.
            self.run.charge(prepared.planned_work(), context)?;
            checkpoint(control)?;
            let kv = capacity(&self.compiler, self.engine)?;
            let before = prepared.encoding_work;
            let plan = self.compiler.plan(&prepared.request, &mut prepared.encoding_work,
                &mut self.encoding, control, true)?;
            let identities = plan.identities().map_err(|error| map_failure(error, true))?;
            if plan.preflight_metadata() != prepared.expected || identities != prepared.identities
                || plan.required_context() != prepared.required_context
                || prepared.encoding_work.checked_sub(before) != Some(before) {
                return Err(BatchItemFailure::fatal(BatchCode::Admission));
            }
            drop(identities);
            let native = budget(prepared.expected.planned_work(), kv);
            plan.preflight_with_control(&prepared.identities, self.engine, native, control)
                .map_err(|error| map_failure(error, true))?;
            let (admitted, guard) = self.admission.admit(Int8FaithfulnessDocumentAdmission {
                identities: &prepared.identities, model_work: prepared.expected.planned_work(),
                required_context: prepared.required_context, task_budget: self.compiler.ceiling,
                max_head_count: plan.max_head_count(), kv_reservation_bytes: kv,
                max_result_bytes: self.compiler.mapping.max_result_bytes,
            })?;
            if admitted != prepared.identities {
                return Err(BatchItemFailure::fatal(BatchCode::Admission));
            }
            checkpoint(control)?;
            let output = match self.prefill {
                Some(prefill) => plan.execute_layer_major_with_control(&admitted, self.engine, native, prefill, control),
                None => plan.execute_with_control(&admitted, self.engine, native, control),
            }.map_err(|error| map_failure(error, true))?;
            prepared.expected.verify_completed_with_control(&output, control)
                .map_err(|error| map_failure(error, true))?;
            checkpoint(control)?;
            Ok(GuardedOutput::new(output, guard))
        })();
        self.run.finish(result, clean(self.engine), true)
    }
}

fn validate_identity_owner(identity: &ExecutionIdentity) -> Result<(), BatchFault> {
    let strings = [&identity.source_revision, &identity.artifact_format, &identity.quant_recipe,
        &identity.task_spec, &identity.grammar_compiler_version, &identity.kv_dtype,
        &identity.sampler_version, &identity.backend_semantic_version];
    let bytes = strings.into_iter().chain(identity.host_class.iter()).chain(identity.compiler_identity.iter())
        .try_fold(0_usize, |total, value| total.checked_add(value.capacity()))
        .ok_or(BatchCode::InvalidLimits)?;
    if bytes > 1024 || identity.canonical_json_bytes().map_err(|_| BatchCode::Admission)?.len() > 16_384 {
        return Err(BatchCode::InvalidLimits.into());
    }
    Ok(())
}
fn capacity(compiler: &Int8FaithfulnessDocumentBatchPlanner, engine: &StrictInt8Engine<'_>)
    -> Result<u64, BatchItemFailure> {
    if !clean(engine) || engine.kv_cache().capacity_positions() < compiler.mapping.chunks.context_tokens {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    let native = Int8MemoryRequirement::for_context(engine.kv_cache().capacity_positions())
        .map_err(|_| BatchItemFailure::fatal(BatchCode::InvalidExecution))?;
    if native.kv_bytes > compiler.ceiling.max_kv_bytes {
        return Err(BatchItemFailure::reject(BatchCode::Admission));
    }
    Ok(native.kv_bytes)
}
fn clean(engine: &StrictInt8Engine<'_>) -> bool {
    !engine.is_poisoned() && engine.kv_cache().all_slots_have_len(0)
}
fn budget(work: Int8Work, max_kv_bytes: u64) -> Int8ScoringBudget {
    Int8ScoringBudget { native: Int8RunBudget::exact(work), max_kv_bytes }
}
fn checkpoint<C: DecodeStepControl>(control: &mut C) -> Result<(), BatchItemFailure> {
    match control.prefill_checkpoint(0) {
        Some(cause) => Err(BatchItemFailure::fatal(BatchFault::cancelled(cause))), None => Ok(()),
    }
}
fn nonzero(w: Int8Work) -> bool {
    w.forward_positions != 0 && w.projected_logits != 0 && w.attention_pairs != 0
        && w.projections.dot_products != 0 && w.projections.multiply_accumulates != 0
}
fn subtract(a: Int8Work, b: Int8Work) -> Result<Int8Work, BatchItemFailure> {
    let sub = |a: u64, b: u64| a.checked_sub(b).ok_or_else(|| BatchItemFailure::reject(BatchCode::WorkLimit));
    Ok(Int8Work { forward_positions: sub(a.forward_positions, b.forward_positions)?,
        projected_logits: sub(a.projected_logits, b.projected_logits)?, attention_pairs: sub(a.attention_pairs, b.attention_pairs)?,
        projections: ProjectionWork { dot_products: sub(a.projections.dot_products, b.projections.dot_products)?,
            multiply_accumulates: sub(a.projections.multiply_accumulates, b.projections.multiply_accumulates)? } })
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum State { Ready, Running, Failed }
struct RunState { remaining: Int8Work, last_sequence: u64, state: State }
impl RunState {
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
    fn charge(&mut self, delta: SafeBpeWork) -> Result<(), BatchItemFailure> {
        let next = self.used.checked_add(delta)
            .ok_or_else(|| BatchItemFailure::fatal(BatchCode::WorkLimit))?;
        let fits = match self.limits {
            None => next == SafeBpeWork::default(),
            Some(total) => next.input_bytes() as u64 <= total.max_input_bytes
                && next.output_tokens() as u64 <= total.max_output_tokens
                && next.pair_checks() <= total.max_pair_checks && next.pair_bytes() <= total.max_pair_bytes,
        };
        if !fits { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
        self.used = next; Ok(())
    }
}
fn map_failure(error: Int8FaithfulnessMapError, executing: bool) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    use Int8FaithfulnessMapError as M;
    use Int8JudgeError as J;
    let mut failure = match error {
        M::EmptySource => BatchItemFailure::reject(BatchCode::DocumentLimit),
        M::Limits | M::Chunk(_) | M::Judge(J::Task(JudgeError::Contract(_) | JudgeError::Limit(_)))
            => BatchItemFailure::reject(BatchCode::Planning),
        M::Work | M::Judge(J::WorkBudget | J::Native(StrictInt8Error::Work)
            | J::Scoring(Int8ScoringError::Native(StrictInt8Error::Work)))
            => BatchItemFailure::reject(BatchCode::WorkLimit),
        M::Allocation | M::Judge(J::Task(JudgeError::AllocationRefused) | J::Scoring(Int8ScoringError::Allocation))
            => BatchItemFailure::fatal(BatchCode::Allocation),
        M::Output | M::Judge(J::Scoring(Int8ScoringError::OutputBudget))
            => BatchItemFailure::reject(BatchCode::OutputLineLimit),
        M::Serialization | M::Judge(J::Task(JudgeError::Serialization) | J::Scoring(Int8ScoringError::Serialization))
            => BatchItemFailure::fatal(BatchCode::Serialization),
        M::Judge(J::Native(StrictInt8Error::Context | StrictInt8Error::Memory)
            | J::Scoring(Int8ScoringError::Native(StrictInt8Error::Context | StrictInt8Error::Memory)))
            => BatchItemFailure::reject(BatchCode::Admission),
        M::Admission | M::Judge(J::Identity) => BatchItemFailure::fatal(BatchCode::Admission),
        _ => BatchItemFailure::fatal(BatchCode::InvalidExecution),
    };
    if executing { failure.stop = true; }
    failure
}

#[cfg(test)] mod tests;
