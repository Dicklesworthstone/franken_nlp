//! One immutable question across every source chunk on one charged INT4 engine.
//! Passage answers and abstentions are retained without global answer selection.
use super::*;
use crate::tasks::{
    answer::ANSWER_TASK_VERSION,
    source_planning::grouped_int4::long::question::{Int4QuestionRun, SourceQuestion},
};
#[cfg(test)]
use crate::tasks::{answer::{PassageCitation, PassageOccurrence}, summarize::SourceCitation};
mod synthesis;
mod cohort;

impl<T: std::borrow::Borrow<SourceQuestion>> Int4SourceMapConfig<T> {
    /// Complete frontier plus native local, independent scan and lifted
    /// citation storage. This is a modeled reservation, not measured RSS.
    pub fn question_workspace_bytes(&self, chunks: usize) -> Result<u64, HostedError> {
        self.workspace().question(chunks, self.task.borrow())
    }
}

// Queued cancellation retains the complete private question/source capture
// until its storage drops, before releasing the invocation's lease.
struct QuestionAdmission {
    input: Charged<SourceMapInput<SourceQuestion>>,
    required: Int4MemoryRequirement,
    lease: EngineLease,
}

impl NlpEngine {
    /// Apply the same private question to every nonblank document chunk.
    /// Preserve all answers, abstentions and interior whitespace in source
    /// order with original-coordinate citations. Distinct texts are not votes
    /// or semantic contradiction labels. Empty/wholly blank documents refuse.
    /// An explicitly selected compact planner retains one cumulative encoder
    /// ledger across question sizing, source probes and every compiled passage.
    #[allow(clippy::too_many_arguments)]
    pub fn answer_int4_document(&self, model: &ResidentInt4, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int4SourceMapConfig<SourceQuestion>, kernel: PackedInt4Kernel,
        cancellation: CancellationToken) -> Result<HostedOutput<Int4QuestionRun>, HostedError> {
        let run = config.native.run;
        let admission = admit_question(self, model, source, planner, vocabulary, config)?;
        let model = model.clone();
        dispatch::run(self, run, cancellation, move |control| {
            execute_question(admission, model.weights(), kernel, control, None)
        })
    }

    /// Use one persistent numerical team across every nonblank document
    /// chunk. The same immutable question, exact source spans and complete
    /// answer/abstention validator remain shared with serial execution.
    /// Whitespace coverage and nonrenewable work ceilings do not change.
    #[allow(clippy::too_many_arguments)]
    pub fn answer_int4_document_with_threads(&self, model: &ResidentInt4, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int4SourceMapConfig<SourceQuestion>, kernel: PackedInt4Kernel,
        compute_threads: usize, cancellation: CancellationToken)
        -> Result<HostedInt4TeamResult<Int4QuestionRun>, HostedError> {
        dispatch::preflight_int4_team(self, config.native.run, compute_threads)?;
        team::team_requirements(&config, model.model_identity().group_width(), compute_threads)?;
        let run = config.native.run;
        let admission = admit_question(self, model, source, planner, vocabulary, config)?;
        dispatch::run_int4_team(self, run, cancellation, model.clone(), compute_threads,
            kernel, move |weights, control, team| {
                let output = execute_question(admission, weights, kernel, control, Some(&mut *team))?;
                // The native engine has released its executor borrow. The
                // dispatcher joins this same team before exposing the result.
                let compute = team.stats().into();
                Ok(HostedInt4TeamResult::new(output, compute))
            })
    }
}

fn admit_question(host: &NlpEngine, model: &ResidentInt4, source: String,
    planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
    config: Int4SourceMapConfig<SourceQuestion>) -> Result<QuestionAdmission, HostedError> {
    dispatch::preflight(host, config.native.run)?;
    if !Arc::ptr_eq(host.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
    constrained_int4::check_model(&config.identity, model.model_identity())
        .map_err(|_| HostedError::ModelIdentity)?;
    let required = question_requirements(&config, source.len(), model.model_identity().group_width(), &planner)?;
    let bytes = input_bytes(source.capacity(), &config)?;
    let lease = host.resources().acquire_lease();
    let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?,
        || Ok(SourceMapInput { source, planner, vocabulary, config }))?;
    Ok(QuestionAdmission { input, required, lease })
}

fn execute_question<'weights>(admission: QuestionAdmission, weights: &'weights Int4ModelWeights,
    kernel: PackedInt4Kernel, control: &mut RunControl,
    team: Option<&'weights mut (dyn Int4ProjectionExecutor + 'weights)>)
    -> Result<HostedOutput<Int4QuestionRun>, HostedError> {
    let QuestionAdmission { input, required, lease } = admission;
    let config = &input.value.config;
    let width = weights.group_width();
    checkpoint(control)?;
    let context = PlanContext::new(&config.identity, config.budget)
        .map_err(|_| HostedError::Limits("INT4 document question context"))?;
    let prepared = input.value.planner.plan_int4_question_with_control(&input.value.source,
        &config.task, config.budget, &context, config.planning, config.mapping, width, control)
        .map_err(map_error)?;
    let expected = prepared.preflight_metadata();
    if prepared.group_width() != width || expected.group_width() != width
        || expected.native_chunks() == 0
        || prepared.execution_identities().len() != expected.native_chunks()
        || required.kv_bytes > prepared.max_kv_bytes() {
        return Err(HostedError::ModelIdentity);
    }
    // Whitespace ranges occupy the result even without native plans.
    let reduction = Pending::reserve(&lease, MemoryClass::JobBuffers,
        config.question_workspace_bytes(expected.chunk_count())?)?;
    let output = output_claim(&lease, prepared.max_result_bytes(), 0)?;
    let mut admitted = Vec::new();
    admitted.try_reserve_exact(expected.native_chunks())
        .map_err(|_| map_error(Int4SourceMapError::Allocation))?;
    for identity in prepared.execution_identities() {
        checkpoint(control)?;
        constrained_int4::check_model(identity, weights.identity())
            .map_err(|_| HostedError::ModelIdentity)?;
        admitted.push(identity.clone());
    }
    let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
    let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch,
        sum(&[required.rope_bytes, required.scratch_payload_bound, config.native.allocator_reserve_bytes])?)?;
    let mut engine = allocate_native(kv, scratch, || {
        let engine = StrictInt4Engine::new(weights, config.native.context_tokens, Int4MemoryBudget {
            max_kv_bytes: required.kv_bytes, max_rope_bytes: required.rope_bytes,
            max_scratch_payload_bytes: required.scratch_payload_bound,
        }).map_err(native_error)?.with_kernel(kernel);
        match team {
            Some(team) => engine.with_projection_executor(team).map_err(native_error),
            None => Ok(engine),
        }
    })?;
    let result = allocate(output, || {
        let result = prepared.execute_with_control(&admitted,
            &mut engine.value, &input.value.vocabulary, control).map_err(map_error)?;
        expected.verify_completed_with_control(&result, control).map_err(map_error)?;
        check_clean(&engine.value)?;
        checkpoint(control)?;
        Ok(result)
    })?;
    drop(engine);
    drop(admitted);
    drop(input);
    drop(reduction);
    drop(lease);
    Ok(GuardedOutput::new(result.value, result._memory))
}

fn question_requirements(config: &Int4SourceMapConfig<SourceQuestion>, source_bytes: usize,
    width: usize, planner: &SourceTaskPlanner) -> Result<Int4MemoryRequirement, HostedError> {
    let required = common_requirements(config, source_bytes, width, planner)?;
    if config.identity.task_spec != ANSWER_TASK_VERSION { return Err(HostedError::ModelIdentity); }
    config.task.validate().map_err(map_error)?;
    // Byte fallback emits one token per byte. Compact input has a distinct
    // byte ceiling: controlled preparation checks its actual token count and
    // complete prompt/context, including the trusted schema and manifest.
    let byte_limit = match planner.document_encoding() {
        crate::tokenizer::untrusted::UntrustedEncoding::ByteFallback =>
            config.planning.max_input_bytes.min(config.budget.max_input_tokens as usize),
        crate::tokenizer::untrusted::UntrustedEncoding::SafeBpeV1(policy) =>
            config.planning.max_input_bytes.min(policy.max_input_bytes),
    };
    if config.task.question.len() > byte_limit {
        return Err(HostedError::Limits("INT4 document question input budget"));
    }
    config.question_workspace_bytes(config.mapping.chunks.max_chunks)?;
    Ok(required)
}
fn input_bytes(source_capacity: usize, config: &Int4SourceMapConfig<SourceQuestion>) -> Result<u64, HostedError> {
    sum(&[source_capacity as u64, config.task.question.capacity() as u64, config.preparation_reserve_bytes])
}

#[cfg(test)]
mod tests;
