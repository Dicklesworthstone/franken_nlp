//! Question-aware document QA on the existing charged process-owned host.
//! No extra model, runtime, retrieval, majority vote or global answer synthesis.
use super::*;
use crate::{native_engine::decode::DecodeStepControl,
    tasks::{answer::ANSWER_TASK_VERSION, source_planning::quantized::{Int8SourceError,
        long::question::{SourceQuestion, Int8QuestionRun}}}};

impl NlpEngine {
    /// Apply one question to every nonblank source chunk and retain all passage
    /// answers with original-document citations. SourceMapConfig's default task
    /// remains unchanged; this concrete specialization admits only question QA.
    /// The source and question are owned and charged through complete execution.
    #[allow(clippy::too_many_arguments)]
    pub fn answer_int8_document(&self, model: &ResidentInt8, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: SourceMapConfig<SourceQuestion>, cancellation: CancellationToken)
        -> Result<HostedOutput<Int8QuestionRun>, HostedError> {
        dispatch::preflight(self, config.native.run)?;
        self.check_resident_domain(model)?;
        check_model_identity(model.artifact_identity(), &config.identity)?;
        let required = requirements(config.native)?;
        validate_question(&config, source.len(), required.kv_bytes)?;
        check_assets(&config.identity, planner.tokenizer_digest(), *planner.template_digest())?;
        let bytes = input_bytes(source.capacity(), &config)?;
        let run = config.native.run;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?,
            || Ok(SourceMapInput { source, planner, vocabulary, config }))?;
        let model = model.clone();
        dispatch::run(self, run, cancellation, move |control| {
            // Capture the WHOLE charged input so field-disjoint closure capture
            // cannot release source/question ownership ahead of its charge.
            let input = input;
            let config = &input.value.config;
            let context = PlanContext::new(&config.identity, config.budget)
                .map_err(|_| HostedError::Limits("document question task context"))?;
            let prepared = input.value.planner.plan_int8_question_with_control(&input.value.source,
                &config.task, config.budget, &context, config.planning, config.mapping, control)
                .map_err(HostedError::SourceMap)?;
            let expected = prepared.preflight_metadata();
            // Price EVERY source chunk and the whole native/evidence frontier,
            // not only answered passages. Blank source ranges still occupy the
            // result; temporary native token buffers remain in this reserve.
            let reduction = Pending::reserve(&lease, MemoryClass::JobBuffers,
                reduction_bytes(config, expected.chunk_count())?)?;
            let output = output_claim(&lease, prepared.max_result_bytes(), 0)?;
            let mut admitted = Vec::new();
            admitted.try_reserve_exact(expected.native_chunks())
                .map_err(|_| HostedError::SourceMap(Int8SourceMapError::Allocation))?;
            for identity in prepared.execution_identities() {
                if let Some(cause) = control.prefill_checkpoint(0) {
                    return Err(HostedError::SourceMap(Int8SourceMapError::Source(Int8SourceError::Cancelled(cause))));
                }
                check_model_identity(model.artifact_identity(), identity)?;
                constrained_int8::check_profile(identity).map_err(|_| HostedError::ModelIdentity)?;
                admitted.push(identity.clone());
            }
            let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
            let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch,
                sum(&[required.rope_bytes, required.scratch_payload_bound, config.native.allocator_reserve_bytes])?)?;
            let mut engine = allocate_native(kv, scratch, || model.inner.loaded.value
                .engine(config.native.context_tokens, memory_budget(required)).map_err(HostedError::Model))?;
            let result = allocate(output, || prepared.execute_with_control(&admitted,
                &mut engine.value, &input.value.vocabulary, control).map_err(HostedError::SourceMap))?;
            drop(engine);
            drop(admitted);
            drop(input);
            drop(reduction);
            drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }
}
fn validate_question(config: &SourceMapConfig<SourceQuestion>, source_bytes: usize, kv_bytes: u64)
    -> Result<(), HostedError> {
    validate_common(config, source_bytes, kv_bytes)?;
    if config.identity.task_spec != ANSWER_TASK_VERSION { return Err(HostedError::ModelIdentity); }
    config.task.validate().map_err(HostedError::SourceMap)?;
    if config.task.question.len() > config.planning.max_input_bytes
        || config.task.question.len() > config.budget.max_input_tokens as usize {
        return Err(HostedError::Limits("document question input budget"));
    }
    Ok(())
}
fn input_bytes(source_capacity: usize, config: &SourceMapConfig<SourceQuestion>) -> Result<u64, HostedError> {
    sum(&[source_capacity as u64, config.task.question.capacity() as u64, config.preparation_reserve_bytes])
}

#[cfg(test)] mod tests;
