//! One process-owned redaction invocation, including fresh NER verification.
use super::*;
use crate::{
    native_engine::constrained_int8,
    tasks::{ner::NER_TASK_VERSION, source_planning::SourceTaskPlanner,
        redact::{RedactError, RedactionRequest,
            pseudonym::{PseudonymKey, Pseudonyms},
            quantized::{Int8Redactor, Int8RedactionConfig, Int8RedactionError, Int8RedactionRun}}},
};

/// The two NER passes share one recipe, engine, deadline and work ledger.
/// Preparation must cover the planner/vocabulary, temporary encoded sources,
/// grammar state, configuration and allocator overhead. Editing headroom covers
/// rule/union vectors, replacement strings, maps and occurrence verification.
/// Both are explicit modeled reservations, not measured/enforced RSS limits.
pub struct RedactConfig {
    pub ner_identity: ExecutionIdentity,
    pub request: RedactionRequest,
    pub detector: Int8RedactionConfig,
    pub native: NativeLimits,
    pub preparation_reserve_bytes: u64,
    pub edit_reserve_bytes: u64,
}

/// Explicit full-256-bit HMAC mode. The same immutable key and namespace are
/// borrowed throughout this invocation. No key generation, argv parsing,
/// serialization, secret clone or implicit 128-bit preflight occurs here.
/// Other caller-held Arc copies remain the caller's lifetime responsibility.
pub struct RedactionPseudonyms {
    pub key: Arc<PseudonymKey>,
    pub namespace: String,
}
struct RedactInput {
    source: String,
    planner: Arc<SourceTaskPlanner>,
    vocabulary: Arc<ExtractionVocabulary>,
    config: RedactConfig,
    pseudonyms: Option<RedactionPseudonyms>,
}

impl NlpEngine {
    /// Reuse one real resident model/native engine for original-source NER and,
    /// when requested, transformed-source NER. Return only a completed guarded
    /// result after rules, occurrence verification and all edits have drained.
    /// No public neural CLI, batch daemon or artifact gate is activated here.
    #[allow(clippy::too_many_arguments)]
    pub fn redact_int8(&self, model: &ResidentInt8, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: RedactConfig, pseudonyms: Option<RedactionPseudonyms>, cancellation: CancellationToken)
        -> Result<HostedOutput<Int8RedactionRun>, HostedError> {
        dispatch::preflight(self, config.native.run)?;
        self.check_resident_domain(model)?;
        check_model_identity(model.artifact_identity(), &config.ner_identity)?;
        let required = requirements(config.native)?;
        validate(&config, source.len(), planner.tokenizer_digest(), *planner.template_digest(), required.kv_bytes)?;
        let input_bytes = input_bytes(&source, &config, pseudonyms.as_ref())?;
        let temporary_bytes = temporary_bytes(&config)?;
        let run = config.native.run;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, input_bytes)?,
            || Ok(RedactInput { source, planner, vocabulary, config, pseudonyms }))?;
        let model = model.clone();
        dispatch::run(self, run, cancellation, move |control| {
            // Keep the whole charged package, including any key Arc, until all
            // borrowing native/planner/pseudonym state has physically drained.
            let input = input;
            let config = &input.value.config;
            let temporary = Pending::reserve(&lease, MemoryClass::JobBuffers, temporary_bytes)?;
            let redactor = Int8Redactor::new(&input.value.planner, config.ner_identity.clone(), config.detector.clone())
                .map_err(HostedError::Redaction)?;
            let pseudonyms = pseudonym_context(input.value.pseudonyms.as_ref(), &config.request)?;
            let output = output_claim(&lease, config.detector.max_result_bytes, 0)?;
            let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
            let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch,
                sum(&[required.rope_bytes, required.scratch_payload_bound, config.native.allocator_reserve_bytes])?)?;
            let mut engine = allocate_native(kv, scratch, || model.inner.loaded.value
                .engine(config.native.context_tokens, memory_budget(required)).map_err(HostedError::Model))?;
            let result = allocate(output, || redactor.redact(&input.value.source, &config.request,
                pseudonyms.as_ref(), &mut engine.value, &input.value.vocabulary, control).map_err(execution_error))?;
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
fn redaction_error(error: RedactError) -> HostedError { HostedError::Redaction(error.into()) }
fn execution_error(error: Int8RedactionError) -> HostedError {
    // Errors do not retain the successful-output reservation. Drain potentially
    // large residual coordinate vectors under that reservation instead of
    // returning an uncharged report. The borrowed native API retains the full
    // report for embeddings that provide their own error/output ownership.
    HostedError::Redaction(match error {
        Int8RedactionError::Residual(report) => RedactError::VerificationResidual { count: report.residuals.len() }.into(),
        error => error,
    })
}
fn pseudonym_context<'a>(secret: Option<&'a RedactionPseudonyms>, request: &RedactionRequest)
    -> Result<Option<Pseudonyms<'a>>, HostedError> {
    let context = secret.map(|s| Pseudonyms::full256(&s.key, &s.namespace,
        request.actions.expected_key_commitment.as_deref())).transpose().map_err(redaction_error)?;
    request.actions.check_key(context.as_ref()).map_err(redaction_error)?;
    Ok(context)
}
fn validate(config: &RedactConfig, source_bytes: usize, tokenizer: Sha256Digest,
    template: Sha256Digest, kv_bytes: u64) -> Result<(), HostedError> {
    config.native.run.validate()?;
    config.ner_identity.validate().map_err(|_| HostedError::ModelIdentity)?;
    constrained_int8::check_profile(&config.ner_identity).map_err(|_| HostedError::ModelIdentity)?;
    if config.ner_identity.task_spec != NER_TASK_VERSION || config.ner_identity.tokenizer_digest != tokenizer
        || config.ner_identity.template_digest != template { return Err(HostedError::ModelIdentity); }
    config.request.rules.validate().map_err(redaction_error)?;
    if config.preparation_reserve_bytes == 0 || config.edit_reserve_bytes == 0
        || source_bytes > config.detector.planning.max_input_bytes || source_bytes > config.request.rule_budget.max_input_bytes
        || config.detector.planning.max_context_tokens > config.native.context_tokens
        || kv_bytes > config.detector.per_pass.max_kv_bytes
        || config.request.edit_budget.max_output_bytes as u64 > config.detector.max_result_bytes
        || !(1..=64 * 1024 * 1024).contains(&config.detector.max_result_bytes) {
        return Err(HostedError::Limits("redaction source, context, preparation or output"));
    }
    let passes = 1 + u64::from(config.request.verify);
    if config.detector.mask_visits_per_pass == 0
        || config.detector.mask_visits_per_pass.checked_mul(passes).is_none_or(|n| n > config.detector.max_mask_visits) {
        return Err(HostedError::Redaction(Int8RedactionError::WorkBudget));
    }
    Ok(())
}
fn input_bytes(source: &String, config: &RedactConfig, secret: Option<&RedactionPseudonyms>) -> Result<u64, HostedError> {
    let secret_bytes = if let Some(secret) = secret {
        if secret.namespace.is_empty() || secret.namespace.len() > 256 { return Err(redaction_error(RedactError::InvalidOptions)); }
        // Key block/id are bounded by PseudonymKey construction. This prices
        // the retained share plus derived HMAC identity metadata conservatively.
        sum(&[secret.namespace.capacity() as u64, 2048])?
    } else { 0 };
    sum(&[source.capacity() as u64, config.preparation_reserve_bytes, secret_bytes])
}
fn temporary_bytes(config: &RedactConfig) -> Result<u64, HostedError> {
    // The transformed output stays live while a fresh NER result is validated.
    // Price bounded result/canonical staging and token storage once at peak,
    // not two resident native engines. Rule/edit heap overhead is explicit.
    let result = config.detector.per_pass.max_output_bytes.checked_mul(4)
        .ok_or(HostedError::Limits("redaction intermediate result arithmetic"))?;
    let tokens = u64::from(config.detector.per_pass.max_output_tokens).checked_mul(8)
        .ok_or(HostedError::Limits("redaction intermediate token arithmetic"))?;
    sum(&[result, tokens, config.edit_reserve_bytes])
}

#[cfg(test)] mod tests;
