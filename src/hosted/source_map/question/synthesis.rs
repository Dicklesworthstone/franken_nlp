//! One charged native invocation for passage discovery and evidence-only synthesis.
use super::*;
use crate::tasks::source_planning::quantized::long::question::synthesis::{
    SourceQuestionSynthesis, Int8QuestionSynthesisRun, QuestionSynthesisLimits};
use crate::validation::grounded_fields::GroundingBudget;
mod hierarchy;

impl NlpEngine {
    /// The final pass is funded before discovery. Same resident model, engine,
    /// controller and original source; no callback accepts generated fake evidence.
    /// The output guard remains held through caller serialization and delivery.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_int8_document_answer(&self, model: &ResidentInt8, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: SourceMapConfig<SourceQuestionSynthesis>, cancellation: CancellationToken)
        -> Result<HostedOutput<Int8QuestionSynthesisRun>, HostedError> {
        dispatch::preflight(self, config.native.run)?;
        self.check_resident_domain(model)?;
        check_model_identity(model.artifact_identity(), &config.identity)?;
        let required = requirements(config.native)?;
        validate_synthesis(&config, source.len(), required.kv_bytes, planner.document_encoding())?;
        check_assets(&config.identity, planner.tokenizer_digest(), *planner.template_digest())?;
        let bytes = input_bytes(&config, source.capacity(), &planner)?;
        let run = config.native.run;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?,
            || Ok(SourceMapInput { source, planner, vocabulary, config }))?;
        let model = model.clone();
        dispatch::run(self, run, cancellation, move |control| {
            let input = input; // Capture the whole storage-before-charge package.
            let config = &input.value.config;
            let context = PlanContext::new(&config.identity, config.budget)
                .map_err(|_| HostedError::Limits("question synthesis task context"))?;
            let prepared = input.value.planner.plan_int8_question_synthesis_with_control(&input.value.source,
                &config.task, config.budget, &context, config.planning, config.mapping, control)
                .map_err(HostedError::SourceMap)?;
            let expected = prepared.preflight_metadata();
            // Map output, copied evidence/origin frontier and a final native
            // result coexist. Price them before executing either stage. Prepared
            // map grammars drain before the final grammar is constructed.
            let temporary = sum(&[reduction_bytes(config, expected.discovery().chunk_count())?,
                synthesis_bytes(config.task.limits, config.task.question.verification,
                    config.budget.max_output_bytes, u64::from(config.budget.max_output_tokens))?])?;
            let reduction = Pending::reserve(&lease, MemoryClass::JobBuffers, temporary)?;
            let output = output_claim(&lease, prepared.max_result_bytes(), 0)?;
            let mut admitted = Vec::new();
            admitted.try_reserve_exact(expected.discovery().native_chunks())
                .map_err(|_| HostedError::SourceMap(Int8SourceMapError::Allocation))?;
            for identity in prepared.execution_identities() {
                if let Some(cause) = control.prefill_checkpoint(0) {
                    return Err(HostedError::SourceMap(Int8SourceError::Cancelled(cause).into()));
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
            let result = allocate(output, || prepared.execute_with_control(&admitted, &input.value.planner,
                &mut engine.value, &input.value.vocabulary, control).map_err(HostedError::SourceMap))?;
            expected.verify_completed_with_control(&result.value, control).map_err(HostedError::SourceMap)?;
            drop(engine);
            drop(admitted);
            drop(input);
            drop(reduction);
            drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }
}
fn validate_synthesis(config: &SourceMapConfig<SourceQuestionSynthesis>, source_bytes: usize, kv_bytes: u64,
    encoding: &UntrustedEncoding)
    -> Result<(), HostedError> {
    validate_common(config, source_bytes, kv_bytes)?;
    if config.identity.task_spec != ANSWER_TASK_VERSION { return Err(HostedError::ModelIdentity); }
    config.task.question.validate().map_err(HostedError::SourceMap)?;
    let limits = config.task.limits;
    let byte_limit = match encoding {
        UntrustedEncoding::ByteFallback => config.planning.max_input_bytes.min(config.budget.max_input_tokens as usize),
        UntrustedEncoding::SafeBpeV1(policy) => config.planning.max_input_bytes.min(policy.max_input_bytes),
    };
    if config.task.question.question.len() > byte_limit
        || !(1..=1024).contains(&limits.max_evidence_passages) || limits.max_evidence_passages > config.planning.max_passages
        || limits.max_evidence_bytes == 0 || limits.max_evidence_bytes > config.planning.max_input_bytes {
        return Err(HostedError::Limits("question synthesis evidence or question limits"));
    }
    Ok(())
}
fn input_bytes(config: &SourceMapConfig<SourceQuestionSynthesis>, source_capacity: usize, planner: &SourceTaskPlanner)
    -> Result<u64, HostedError> {
    sum(&[super::super::input_bytes(config, source_capacity, planner)?,
        config.task.question.question.capacity() as u64])
}
fn synthesis_bytes(limits: QuestionSynthesisLimits, verification: GroundingBudget, output: u64, tokens: u64)
    -> Result<u64, HostedError> {
    let error = || HostedError::Limits("question synthesis memory arithmetic");
    sum(&[(limits.max_evidence_bytes as u64).checked_mul(4).ok_or_else(error)?,
        (limits.max_evidence_passages as u64).checked_mul(1024).ok_or_else(error)?,
        (verification.max_matches as u64).checked_mul(64).ok_or_else(error)?,
        output.checked_mul(4).ok_or_else(error)?, tokens.checked_mul(8).ok_or_else(error)?])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::untrusted::{SafeBpeLimits, SafeBpeWorkLimits};
    pub(super) fn config() -> SourceMapConfig<SourceQuestionSynthesis> {
        let base = super::super::tests::config();
        SourceMapConfig { identity: base.identity,
            task: SourceQuestionSynthesis { question: base.task,
                limits: QuestionSynthesisLimits { max_evidence_passages: 4, max_evidence_bytes: 1024 } },
            budget: base.budget, planning: base.planning, mapping: base.mapping, native: base.native,
            preparation_reserve_bytes: base.preparation_reserve_bytes,
            reduction_reserve_bytes: base.reduction_reserve_bytes }
    }
    #[test]
    fn compact_synthesis_admits_question_bytes_and_shared_workspace_before_native_work() {
        let policy = UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 4096,
            max_output_tokens: 4096, max_pair_checks: 1 << 24, max_pair_bytes: 1 << 28 });
        let work = SafeBpeWorkLimits { max_input_bytes: 65_536, max_output_tokens: 65_536,
            max_pair_checks: 1 << 26, max_pair_bytes: 1 << 30 };
        let byte = super::super::tests::planner();
        let compact = super::super::tests::planner().with_map_document_encoding(policy.clone(), work).unwrap();
        let mut c = config(); c.task.question.question = " the".repeat(32);
        assert!(c.task.question.question.len() > c.budget.max_input_tokens as usize);
        validate_synthesis(&c, 10, 1024, &policy).unwrap();
        assert!(validate_synthesis(&c, 10, 1024, &UntrustedEncoding::ByteFallback).is_err());
        assert_eq!(input_bytes(&c, 1024, &compact).unwrap() - input_bytes(&c, 1024, &byte).unwrap(),
            compact.map_document_encoding_workspace_bytes().unwrap());
        let unbudgeted = super::super::tests::planner().with_document_encoding(policy.clone()).unwrap();
        assert!(input_bytes(&c, 1024, &unbudgeted).is_err());
        c.task.question.question = "q".repeat(4097);
        assert!(validate_synthesis(&c, 10, 1024, &policy).is_err());
        c.preparation_reserve_bytes = u64::MAX;
        assert!(input_bytes(&c, 1024, &compact).is_err());
    }
    #[test]
    fn byte_synthesis_preserves_original_question_capacity_and_preparation_charge() {
        let mut c = config();
        let mut question = String::with_capacity(8192); question.push_str("Who?");
        c.task.question.question = question;
        let planner = super::super::tests::planner();
        validate_synthesis(&c, 10, 1024, &UntrustedEncoding::ByteFallback).unwrap();
        assert_eq!(input_bytes(&c, 1024, &planner).unwrap(),
            1024 + c.task.question.question.capacity() as u64 + c.preparation_reserve_bytes);
    }
    #[test]
    fn evidence_origin_fanout_and_final_native_storage_are_additional_to_the_map() {
        let l = QuestionSynthesisLimits { max_evidence_passages: 32, max_evidence_bytes: 16384 };
        let v = GroundingBudget { max_fields: 4096, max_matches: 16384, max_scan_steps: 1 << 26 };
        assert_eq!(synthesis_bytes(l, v, 1 << 20, 128).unwrap(), 16384 * 4 + 32 * 1024 + 16384 * 64 + 4 * (1 << 20) + 128 * 8);
        let mut doubled = v; doubled.max_matches *= 2;
        assert_eq!(synthesis_bytes(l, doubled, 1 << 20, 128).unwrap() - synthesis_bytes(l, v, 1 << 20, 128).unwrap(), 16384 * 64);
    }
    #[test]
    fn temporary_memory_arithmetic_never_wraps_to_a_small_reservation() {
        let l = QuestionSynthesisLimits { max_evidence_passages: 1, max_evidence_bytes: 1 };
        assert!(synthesis_bytes(l, GroundingBudget::default(), u64::MAX, 1).is_err());
        assert!(synthesis_bytes(l, GroundingBudget::default(), 1, u64::MAX).is_err());
    }
}
