//! Full detector-union redaction and residual verification. A verification
//! failure returns coordinates only, never the partially redacted document.

use std::{collections::BTreeSet, error::Error, fmt};
use serde::{Deserialize, Serialize};
use crate::{tasks::ner::{EntityType, NerResult}, validation::grounded_fields::GroundingBudget};
use super::{RedactError, actions::{ActionPolicy, EditBudget, RedactionResult, VerificationStatus, apply},
    detectors::{RuleSet, RuleBudget}, pseudonym::Pseudonyms, union::{DetectedDocument, NerProfile, RedactionRegion}};

/// Static adapter contract, not a recipe/plugin execution hook. Each call must
/// plan from the supplied source; the pipeline independently checks returned
/// occurrences and requires the same type scope on every verification pass.
pub trait NerPass {
    type Error: Error + 'static;
    fn types(&self) -> &BTreeSet<EntityType>;
    fn run(&mut self, source: &str) -> Result<NerResult, Self::Error>;
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LeakReport {
    pub schema_version: u32,
    /// Coordinates refer to the transformed document, never the original.
    pub residuals: Vec<RedactionRegion>,
    pub rules: RuleSet,
    pub model_types: BTreeSet<EntityType>,
}

#[derive(Debug)]
pub enum PipelineError<E> { Redaction(RedactError), Model(E), Residual(LeakReport) }
impl<E: Error + 'static> fmt::Display for PipelineError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Redaction(e) => e.fmt(f),
            // Preserve the typed source, but do not dump model prompt/content.
            Self::Model(_) => f.write_str("redaction model pass failed"),
            Self::Residual(report) => write!(f, "redaction verification failed: {} residual regions", report.residuals.len()),
        }
    }
}
impl<E: Error + 'static> Error for PipelineError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Redaction(e) => Some(e), Self::Model(e) => Some(e), Self::Residual(_) => None }
    }
}
impl<E> From<RedactError> for PipelineError<E> { fn from(e: RedactError) -> Self { Self::Redaction(e) } }

#[derive(Clone, Debug, Serialize)]
pub struct RedactionRequest {
    pub rules: RuleSet,
    pub actions: ActionPolicy,
    pub rule_budget: RuleBudget,
    pub grounding_budget: GroundingBudget,
    pub edit_budget: EditBudget,
    /// Rerun exactly the originally selected union; no implicit rules-only
    /// substitute when the original run included a model.
    pub verify: bool,
}
impl Default for RedactionRequest {
    fn default() -> Self {
        Self { rules: RuleSet::default(), actions: ActionPolicy::default(), rule_budget: RuleBudget::default(),
            grounding_budget: GroundingBudget::default(), edit_budget: EditBudget::default(), verify: true }
    }
}

pub fn redact_rules(
    source: &str, request: &RedactionRequest, pseudonyms: Option<&Pseudonyms<'_>>,
) -> Result<RedactionResult, PipelineError<std::convert::Infallible>> {
    let document = DetectedDocument::rules_only(source, &request.rules, request.rule_budget)?;
    let mut output = apply(&document, &request.actions, pseudonyms, request.edit_budget)?;
    if request.verify {
        let residual = DetectedDocument::rules_only(&output.text, &request.rules, request.rule_budget)?;
        require_clean::<std::convert::Infallible>(residual, request.edit_budget.max_output_bytes)?;
        output.verification = VerificationStatus::CleanDeclaredUnion;
    }
    bind_request(&mut output, request)?;
    output.check_size(request.edit_budget.max_output_bytes)?;
    Ok(output)
}

pub fn redact_with_ner<P: NerPass>(
    source: &str, request: &RedactionRequest, pseudonyms: Option<&Pseudonyms<'_>>, model: &mut P,
) -> Result<RedactionResult, PipelineError<P::Error>> {
    redact_with_profile(source, request, pseudonyms, model, NerProfile::Eager)
}

pub(super) fn redact_with_profile<P: NerPass>(
    source: &str, request: &RedactionRequest, pseudonyms: Option<&Pseudonyms<'_>>, model: &mut P,
    profile: NerProfile,
) -> Result<RedactionResult, PipelineError<P::Error>> {
    request.actions.check_key(pseudonyms)?;
    request.rules.validate()?;
    if source.len() > request.rule_budget.max_input_bytes { return Err(RedactError::InputBudget.into()); }
    let types = model.types().clone();
    if types.is_empty() { return Err(RedactError::InvalidOptions.into()); }
    let first = model.run(source).map_err(PipelineError::Model)?;
    if model.types() != &types { return Err(RedactError::InvalidNerEvidence.into()); }
    let document = DetectedDocument::with_ner_profile(source, &request.rules, request.rule_budget, &first, &types, request.grounding_budget, profile)?;
    drop(first);
    let mut output = apply(&document, &request.actions, pseudonyms, request.edit_budget)?;
    drop(document);
    if request.verify {
        // Fresh source text goes through the model again. No old mention,
        // offset, or successful original-source receipt is reused.
        let second = model.run(&output.text).map_err(PipelineError::Model)?;
        if model.types() != &types { return Err(RedactError::InvalidNerEvidence.into()); }
        let residual = DetectedDocument::with_ner_profile(&output.text, &request.rules, request.rule_budget, &second, &types, request.grounding_budget, profile)?;
        require_clean::<P::Error>(residual, request.edit_budget.max_output_bytes)?;
        output.verification = VerificationStatus::CleanDeclaredUnion;
    }
    bind_request(&mut output, request)?;
    output.check_size(request.edit_budget.max_output_bytes)?;
    Ok(output)
}
fn bind_request(output: &mut RedactionResult, request: &RedactionRequest) -> Result<(), RedactError> {
    // Public policy data only; never hash private source text for export.
    let binding = ("redact-pipeline-v1", output.policy_digest, request);
    output.policy_digest = crate::execution_identity::Sha256Digest::of_bytes(
        &crate::canonjson::canonical_bytes(&binding).map_err(|_| RedactError::Serialization)?);
    Ok(())
}
fn require_clean<E>(document: DetectedDocument<'_>, report_cap: usize) -> Result<(), PipelineError<E>> {
    if !document.regions.is_empty() {
        let report = LeakReport { schema_version: 1, residuals: document.regions,
            rules: document.rule_set, model_types: document.model_types };
        if crate::canonjson::canonical_bytes(&report).map_err(|_| RedactError::Serialization)?.len() > report_cap {
            return Err(RedactError::OutputBudget.into());
        }
        return Err(PipelineError::Residual(report));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{tasks::{ner::NamedEntity, extract::ExtractionGrounding, ir::ScoreSpace},
        validation::grounded_fields::{SourceOccurrence, scan_occurrences}};
    struct Model { types: BTreeSet<EntityType>, seen: Vec<String>, leak: bool, fail: bool }
    impl NerPass for Model {
        type Error = std::io::Error;
        fn types(&self) -> &BTreeSet<EntityType> { &self.types }
        fn run(&mut self, text: &str) -> Result<NerResult, Self::Error> {
            self.seen.push(text.to_owned());
            if self.fail { return Err(std::io::Error::other("typed native failure")); }
            let mention = if self.seen.len() == 1 { "Alice" } else if self.leak { "redacted" } else { "never-present" };
            let entities = if text.contains(mention) {
                let spans = scan_occurrences(text, mention, &mut GroundingBudget::default()).unwrap();
                vec![NamedEntity { text: mention.to_owned(), entity_type: EntityType::Person,
                    occurrence: if spans.len() == 1 { SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous }, spans }]
            } else { Vec::new() };
            Ok(NerResult { schema_version: 1, task_spec_version: "ner-v1".to_owned(), numerics_profile: "hf-bf16-eager".to_owned(),
                score_space: ScoreSpace::NotComputed, grounding: ExtractionGrounding::SourceMembership, entities,
                generated_token_ids: Vec::new(), forward_positions: 0, projected_logits: 0, mask_node_visit_charge: 0 })
        }
    }
    fn model() -> Model { Model { types: [EntityType::Person].into_iter().collect(), seen: Vec::new(), leak: false, fail: false } }
    #[test]
    fn rules_only_redaction_and_verification_work_without_model() {
        let out = redact_rules("a@example.org +1 (212) 555-0199", &RedactionRequest::default(), None).unwrap();
        assert!(!out.text.contains("a@example.org")); assert_eq!(out.verification(), VerificationStatus::CleanDeclaredUnion);
        assert!(out.model_types.is_empty());
    }
    #[test]
    fn verification_reruns_both_detectors_on_transformed_text() {
        let mut m = model(); let source = "Alice a@example.org Alice";
        let out = redact_with_ner(source, &RedactionRequest::default(), None, &mut m).unwrap();
        assert_eq!(m.seen, vec![source.to_owned(), out.text.clone()]);
        assert_eq!(out.text, "[redacted:person] [redacted:email] [redacted:person]");
        assert_eq!(out.verification(), VerificationStatus::CleanDeclaredUnion);
    }
    #[test]
    fn model_residual_is_error_and_report_has_no_raw_source() {
        let mut m = model(); m.leak = true;
        let error = match redact_with_ner("Alice", &RedactionRequest::default(), None, &mut m) { Err(e) => e, Ok(_) => panic!("residual must fail") };
        let PipelineError::Residual(report) = error else { panic!("missing residual report") };
        assert!(!report.residuals.is_empty());
        let json = crate::canonjson::canonical_string(&report).unwrap();
        assert!(!json.contains("Alice")); assert!(!json.contains("text"));
    }
    #[test]
    fn model_failure_preserves_its_typed_cause() {
        let mut m = model(); m.fail = true;
        assert!(matches!(redact_with_ner("Alice", &RedactionRequest::default(), None, &mut m), Err(PipelineError::Model(_))));
    }
    #[test]
    fn verification_opt_out_is_reported_and_not_a_clean_pass() {
        let mut m = model(); let request = RedactionRequest { verify: false, ..RedactionRequest::default() };
        let out = redact_with_ner("Alice", &request, None, &mut m).unwrap();
        assert_eq!(m.seen.len(), 1); assert_eq!(out.verification(), VerificationStatus::NotRequested);
    }
    #[test]
    fn exact_rule_union_verification_detects_new_cross_edit_matches() {
        // Construct a candidate final text to exercise the same no-result gate:
        // completed replacement data are never exempted from rule scanning.
        let d = DetectedDocument::rules_only("x@y.org", &RuleSet::default(), RuleBudget::default()).unwrap();
        assert!(matches!(require_clean::<std::convert::Infallible>(d, 4096), Err(PipelineError::Residual(_))));
    }
}
