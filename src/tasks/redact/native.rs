//! Concrete native NER adapter for the redaction/verification pipeline.
//! One admitted engine, tokenizer, vocabulary and request control are reused.
//! The task factory supplies trusted instruction/scaffold tokens; each source
//! is freshly encoded and checked against the returned TaskIR before execution.

use std::collections::BTreeSet;
use crate::{
    execution_identity::{ExecutionIdentity, Sha256Digest},
    grammar::{CompileLimits, runtime::SourceRuntimeLimits},
    native_engine::{constrained::{JsonDecodeOptions, JsonWorkBudget},
        decode::DecodeStepControl, hf_bf16_eager::HfBf16EagerEngine},
    tasks::{extract::{SourceDocument, SourceDocumentEncoder, ExtractionVocabulary},
        ir::TaskPlan, ner::{EntityType, NerError, NerOptions, NerPlan, NerResult}},
    tokenizer::specials::TemplateControlIds,
};
use super::pipeline::NerPass;

pub struct NativeNerConfig {
    pub options: NerOptions,
    pub decode: JsonDecodeOptions,
    pub compiler: CompileLimits,
    pub source: SourceRuntimeLimits,
    pub max_source_bytes: usize,
    pub max_source_tokens: usize,
    /// AGGREGATE budget across original and verification passes, not a fresh
    /// budget for each pass. KV is a residency ceiling, not a consumed counter.
    pub total_work: JsonWorkBudget,
}

/// No Debug or Serialize: private plans, keys and source text do not belong in
/// telemetry. No new runtime, model loader, weight clone or unbounded retry.
pub struct NativeNerPass<'a, F, C> {
    engine: &'a mut HfBf16EagerEngine,
    encoder: &'a SourceDocumentEncoder,
    vocabulary: &'a ExtractionVocabulary,
    controls: &'a TemplateControlIds,
    control: &'a mut C,
    factory: F,
    config: NativeNerConfig,
    types: BTreeSet<EntityType>,
    remaining: JsonWorkBudget,
    failed: bool,
    contract: Option<Sha256Digest>,
}
impl<'a, F, C> NativeNerPass<'a, F, C>
where F: FnMut(&SourceDocument, &NerOptions) -> Result<(TaskPlan, ExecutionIdentity), NerError>,
      C: DecodeStepControl {
    pub fn new(
        engine: &'a mut HfBf16EagerEngine, encoder: &'a SourceDocumentEncoder,
        vocabulary: &'a ExtractionVocabulary, controls: &'a TemplateControlIds,
        control: &'a mut C, config: NativeNerConfig, factory: F,
    ) -> Result<Self, NerError> {
        config.options.validate()?;
        if config.max_source_bytes == 0 || config.max_source_tokens == 0 { return Err(NerError::InvalidOptions); }
        let types = config.options.types.iter().copied().collect();
        let remaining = config.total_work;
        Ok(Self { engine, encoder, vocabulary, controls, control, factory, config, types, remaining, failed: false, contract: None })
    }
    /// Remaining forward/mask work for the second pass or a bounded reuse.
    pub fn remaining_work(&self) -> JsonWorkBudget { self.remaining }
}
impl<F, C> NerPass for NativeNerPass<'_, F, C>
where F: FnMut(&SourceDocument, &NerOptions) -> Result<(TaskPlan, ExecutionIdentity), NerError>,
      C: DecodeStepControl {
    type Error = NerError;
    fn types(&self) -> &BTreeSet<EntityType> { &self.types }
    fn run(&mut self, source: &str) -> Result<NerResult, NerError> {
        if self.failed { return Err(NerError::InvalidOptions); }
        let document = self.encoder.encode(source, self.config.max_source_bytes, self.config.max_source_tokens)?;
        let (task, identity) = (self.factory)(&document, &self.config.options)?;
        // This constructor proves the factory used THIS source's exact token
        // segment and THIS options schema. Returning a stale original plan for
        // the verification text fails before entering the model.
        let plan = NerPlan::from_task_plan(&task, &document, self.config.options.clone(), self.config.decode.clone(),
            self.config.compiler, self.controls, self.config.source)?;
        let identity = plan.bind_identity(identity)?;
        // Source content may change; detector recipe, trusted scaffold,
        // model/artifact identity and policy may not silently change between
        // original and verification passes.
        let observed = pass_contract(&task, &identity)?;
        if self.contract.is_some_and(|saved| saved != observed) {
            self.failed = true;
            return Err(NerError::InvalidOptions);
        }
        self.contract = Some(observed);
        // Poison before crossing into fallible compute, including unwinding.
        // A partial native pass cannot be retried against uncharged work.
        self.failed = true;
        let result = plan.execute_eager(self.engine, &identity, self.vocabulary, self.remaining, self.control)?;
        self.remaining = charge(self.remaining, &result)?;
        self.failed = false;
        Ok(result)
    }
}
fn pass_contract(task: &TaskPlan, identity: &ExecutionIdentity) -> Result<Sha256Digest, NerError> {
    let mut invariant = identity.clone();
    let erased = Sha256Digest::of_bytes(b"redact-source-erasure-v1");
    invariant.prompt_digest = erased;
    invariant.taskir_digest = erased;
    let scaffold: Vec<_> = task.ir().prompt_segments().iter()
        .filter(|s| s.kind() != crate::tasks::ir::PromptSegmentKind::Document).collect();
    let identity_bytes = invariant.canonical_json_bytes().map_err(|_| NerError::Serialization)?;
    // This digest remains private, since scaffold tokens can contain private
    // application instructions. It is never exported as a content identifier.
    Ok(Sha256Digest::of_bytes(&crate::canonjson::canonical_bytes(
        &("redact-native-contract-v1", identity_bytes, scaffold, task.ir().budget())
    ).map_err(|_| NerError::Serialization)?))
}
fn charge(mut remaining: JsonWorkBudget, result: &NerResult) -> Result<JsonWorkBudget, NerError> {
    remaining.max_forward_positions = remaining.max_forward_positions.checked_sub(result.forward_positions).ok_or(NerError::InvalidResult)?;
    remaining.max_projected_logits = remaining.max_projected_logits.checked_sub(result.projected_logits).ok_or(NerError::InvalidResult)?;
    remaining.max_total_mask_node_visits = remaining.max_total_mask_node_visits.checked_sub(result.mask_node_visit_charge).ok_or(NerError::InvalidResult)?;
    Ok(remaining)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{grammar::mask::MaskWorkLimits, tasks::{extract::ExtractionGrounding, ir::ScoreSpace}};
    fn result() -> NerResult {
        NerResult { schema_version: 1, task_spec_version: "ner-v1".to_owned(), numerics_profile: "hf-bf16-eager".to_owned(),
            score_space: ScoreSpace::NotComputed, grounding: ExtractionGrounding::SourceMembership, entities: Vec::new(),
            generated_token_ids: Vec::new(), forward_positions: 2, projected_logits: 20, mask_node_visit_charge: 200 }
    }
    fn budget() -> JsonWorkBudget { JsonWorkBudget { max_forward_positions: 4, max_projected_logits: 40,
        max_total_mask_node_visits: 400, max_kv_bytes: 1234, mask_limits: MaskWorkLimits::default() } }
    #[test]
    fn both_passes_charge_the_same_aggregate_work_budget() {
        let left = charge(charge(budget(), &result()).unwrap(), &result()).unwrap();
        assert_eq!(left.max_forward_positions, 0); assert_eq!(left.max_projected_logits, 0);
        assert_eq!(left.max_total_mask_node_visits, 0); assert_eq!(left.max_kv_bytes, 1234);
        assert!(charge(left, &result()).is_err());
    }
    #[test]
    fn every_work_axis_refuses_underflow() {
        for axis in 0..3 {
            let mut b = budget();
            match axis { 0 => b.max_forward_positions = 1, 1 => b.max_projected_logits = 19, _ => b.max_total_mask_node_visits = 199 }
            assert!(charge(b, &result()).is_err());
        }
    }
}
