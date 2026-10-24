//! One owned native INT4 redaction invocation, including fresh verification.
use super::*;
mod long;
use crate::{
    native_engine::{constrained_int4, strict_int4::{Int4Work, STRICT_INT4_PROFILE}},
    tasks::{extract::ExtractionVocabulary, ner::NER_TASK_VERSION, source_planning::SourceTaskPlanner,
        redact::{RedactError, RedactionRequest, pseudonym::Pseudonyms,
            grouped_int4::{self as redact_task, Int4Redactor, Int4RedactionConfig,
                Int4RedactionError, Int4RedactionRun, INT4_REDACTION_EXECUTION}}},
};

struct RedactInput<D = Int4RedactionConfig> {
    source: String,
    planner: Arc<SourceTaskPlanner>,
    vocabulary: Arc<ExtractionVocabulary>,
    config: RedactConfig<D>,
    pseudonyms: Option<RedactionPseudonyms>,
}
impl NlpEngine {
    /// The original and optional transformed-source NER passes share one
    /// resident model, engine, deadline, detector scope and nonrenewable work
    /// ledger. The complete input/key and all reservations survive the actual
    /// physical invocation. Only a completed redaction result is published.
    #[allow(clippy::too_many_arguments)]
    pub fn redact_int4(&self, model: &ResidentInt4, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: RedactConfig<Int4RedactionConfig>, pseudonyms: Option<RedactionPseudonyms>,
        kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<Int4RedactionRun>, HostedError> {
        dispatch::preflight(self, config.native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        redact_task::check_model(&config.ner_identity, model.model_identity()).map_err(redaction_error)?;
        let width = model.model_identity().group_width();
        let required = requirements(&config, source.len(), width,
            planner.tokenizer_digest(), *planner.template_digest())?;
        let input_bytes = input_bytes(&source, &config, pseudonyms.as_ref())?;
        let temporary_bytes = temporary_bytes(&config)?;
        let run = config.native.run;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, input_bytes)?,
            || Ok(RedactInput { source, planner, vocabulary, config, pseudonyms }))?;
        let model = model.clone();
        dispatch::run(self, run, cancellation, move |control| {
            let input = input;
            let config = &input.value.config;
            let temporary = Pending::reserve(&lease, MemoryClass::JobBuffers, temporary_bytes)?;
            let redactor = Int4Redactor::new(&input.value.planner, config.ner_identity.clone(),
                config.detector.clone(), model.model_identity()).map_err(redaction_error)?;
            let pseudonyms = pseudonym_context(input.value.pseudonyms.as_ref(), &config.request)?;
            let output = output_claim(&lease, config.detector.max_result_bytes, 0)?;
            let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
            let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch,
                sum(&[required.rope_bytes, required.scratch_payload_bound, config.native.allocator_reserve_bytes])?)?;
            let mut engine = allocate_native(kv, scratch, || {
                StrictInt4Engine::new(&model.inner.loaded.value, config.native.context_tokens, Int4MemoryBudget {
                    max_kv_bytes: required.kv_bytes, max_rope_bytes: required.rope_bytes,
                    max_scratch_payload_bytes: required.scratch_payload_bound,
                }).map(|engine| engine.with_kernel(kernel)).map_err(native_error)
            })?;
            let result = allocate(output, || {
                let result = redactor.redact(&input.value.source, &config.request, pseudonyms.as_ref(),
                    &mut engine.value, &input.value.vocabulary, control).map_err(execution_error)?;
                if result.schema_version != 1 || result.execution != INT4_REDACTION_EXECUTION
                    || result.numerics_profile != STRICT_INT4_PROFILE || result.group_width != width
                    || result.ner_passes != 1 + usize::from(config.request.verify)
                    || !within(result.model_work, result.reserved_model_work)
                    || !within(result.reserved_model_work, config.detector.max_model_work)
                    || result.mask_node_visit_charge > result.reserved_mask_node_visits
                    || result.reserved_mask_node_visits > config.detector.max_mask_visits
                    || engine.value.is_poisoned() || !engine.value.kv_cache().all_slots_have_len(0) {
                    return Err(redaction_error(Int4RedactionError::InvalidResult));
                }
                checkpoint(control)?;
                Ok(result)
            })?;
            drop(engine);
            drop(pseudonyms);
            drop(redactor);
            drop(temporary);
            drop(input);
            drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }
}
fn redaction_error(error: Int4RedactionError) -> HostedError { HostedError::Int4(Int4HostError::Redaction(error)) }
fn detector_error(error: RedactError) -> HostedError { redaction_error(error.into()) }
fn execution_error(error: Int4RedactionError) -> HostedError {
    // A borrowed core caller may retain a bounded residual report. Hosted
    // errors carry only its count; discard coordinate vectors while the output
    // reservation still owns the failing finalization operation.
    redaction_error(match error {
        Int4RedactionError::Residual(report) =>
            RedactError::VerificationResidual { count: report.residuals.len() }.into(),
        error => error,
    })
}
fn pseudonym_context<'a>(secret: Option<&'a RedactionPseudonyms>, request: &RedactionRequest)
    -> Result<Option<Pseudonyms<'a>>, HostedError> {
    let context = secret.map(|s| Pseudonyms::full256(&s.key, &s.namespace,
        request.actions.expected_key_commitment.as_deref())).transpose().map_err(detector_error)?;
    request.actions.check_key(context.as_ref()).map_err(detector_error)?;
    Ok(context)
}
fn requirements(config: &RedactConfig<Int4RedactionConfig>, source_bytes: usize, width: usize,
    tokenizer: Sha256Digest, template: Sha256Digest) -> Result<Int4MemoryRequirement, HostedError> {
    config.native.run.validate()?;
    config.ner_identity.validate().map_err(|_| HostedError::ModelIdentity)?;
    constrained_int4::check_profile(&config.ner_identity, width).map_err(|_| HostedError::ModelIdentity)?;
    if config.ner_identity.task_spec != NER_TASK_VERSION || config.ner_identity.tokenizer_digest != tokenizer
        || config.ner_identity.template_digest != template { return Err(HostedError::ModelIdentity); }
    config.detector.ner.validate().map_err(|_| detector_error(RedactError::InvalidOptions))?;
    config.detector.per_pass.validate().map_err(|_| detector_error(RedactError::InvalidOptions))?;
    config.request.rules.validate().map_err(detector_error)?;
    if config.native.allocator_reserve_bytes == 0 || config.preparation_reserve_bytes == 0
        || config.edit_reserve_bytes == 0 || source_bytes > config.detector.planning.max_input_bytes
        || source_bytes > config.request.rule_budget.max_input_bytes
        || config.detector.planning.max_context_tokens > config.native.context_tokens
        || config.request.edit_budget.max_output_bytes as u64 > config.detector.max_result_bytes
        || !(1..=64 * 1024 * 1024).contains(&config.detector.max_result_bytes)
        || config.detector.mask_limits.max_trie_node_visits == 0
        || config.detector.mask_limits.checkpoint_interval_nodes == 0 {
        return Err(HostedError::Limits("INT4 redaction source, context, preparation or output"));
    }
    let required = Int4MemoryRequirement::for_context(config.native.context_tokens, width).map_err(native_error)?;
    if required.kv_bytes > config.detector.per_pass.max_kv_bytes {
        return Err(HostedError::Limits("INT4 redaction complete resident KV capacity"));
    }
    let passes = 1 + u64::from(config.request.verify);
    if config.detector.mask_visits_per_pass == 0
        || config.detector.mask_visits_per_pass.checked_mul(passes).is_none_or(|n| n > config.detector.max_mask_visits) {
        return Err(redaction_error(Int4RedactionError::WorkBudget));
    }
    Ok(required)
}
fn input_bytes<D>(source: &String, config: &RedactConfig<D>,
    secret: Option<&RedactionPseudonyms>) -> Result<u64, HostedError> {
    let secret_bytes = if let Some(secret) = secret {
        if secret.namespace.is_empty() || secret.namespace.len() > 256 {
            return Err(detector_error(RedactError::InvalidOptions));
        }
        sum(&[secret.namespace.capacity() as u64, 2048])?
    } else { 0 };
    sum(&[source.capacity() as u64, config.preparation_reserve_bytes, secret_bytes])
}
fn temporary_bytes(config: &RedactConfig<Int4RedactionConfig>) -> Result<u64, HostedError> {
    let result = config.detector.per_pass.max_output_bytes.checked_mul(4)
        .ok_or(HostedError::Limits("INT4 redaction intermediate result arithmetic"))?;
    let tokens = u64::from(config.detector.per_pass.max_output_tokens).checked_mul(8)
        .ok_or(HostedError::Limits("INT4 redaction intermediate token arithmetic"))?;
    sum(&[result, tokens, config.edit_reserve_bytes])
}
fn within(a: Int4Work, b: Int4Work) -> bool {
    a.forward_positions <= b.forward_positions && a.projected_logits <= b.projected_logits
        && a.attention_pairs <= b.attention_pairs && a.projections.fits(b.projections)
}

#[cfg(test)]
mod tests;

