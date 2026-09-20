//! Rules-only batch redaction, using the same detector union, edit engine and
//! residual verification as single-document redaction. No model or key source
//! is inferred from request data. Private text is never placed in diagnostics.

use serde::{Deserialize, Serialize};

use crate::{
    batch::{BatchCode, BatchFault, BatchItemFailure},
    native_engine::decode::DecodeStepControl,
    tasks::redact::{
        RedactError, RedactionRequest, redact_rules,
        actions::{RedactionAction, RedactionResult},
        detectors::RuleSet,
        pipeline::PipelineError,
    },
    textutil::TextBudget,
};

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RulesOnlyAction {
    Mask,
    #[default]
    Placeholder,
}

/// A complete per-document policy. Dates are opt-in via rules.enabled. Empty
/// rules, NER types, pseudonymization and verification opt-outs are refused.
/// Supporting keyed pseudonyms would require a separate run-owned key source;
/// silently treating a missing key as permission to emit originals is forbidden.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RulesOnlyOptions {
    #[serde(default)]
    pub rules: RuleSet,
    #[serde(default)]
    pub action: RulesOnlyAction,
    /// Sensitive coordinate data are explicit result content, never telemetry.
    #[serde(default)]
    pub include_map: bool,
}
impl RulesOnlyOptions {
    pub fn validate(&self) -> Result<(), BatchFault> {
        self.rules.validate().map_err(|_| BatchCode::Planning)?;
        if self.rules.enabled.is_empty() { return Err(BatchCode::Planning.into()); }
        Ok(())
    }
}

/// Host-owned limits; documents cannot increase them through task_args.
#[derive(Clone, Copy, Debug)]
pub struct RedactionBatchLimits {
    pub max_detections: usize,
    /// Conservative work units for EACH of detection and residual verification.
    pub max_rule_work_per_scan: u64,
    /// Two complete scan ceilings are reserved before every attempted document,
    /// including failed attempts. Flush controls never replenish this allowance.
    pub max_total_rule_work: u64,
}
impl Default for RedactionBatchLimits {
    fn default() -> Self {
        Self { max_detections: 4096, max_rule_work_per_scan: 128 * 1024 * 1024,
            max_total_rule_work: 1_u64 << 40 }
    }
}
impl RedactionBatchLimits {
    pub fn validate(self) -> Result<(), BatchFault> {
        if !(1..=16_384).contains(&self.max_detections) || self.max_rule_work_per_scan == 0
            || self.max_rule_work_per_scan.checked_mul(2).is_none() || self.max_total_rule_work == 0 {
            return Err(BatchCode::InvalidLimits.into());
        }
        Ok(())
    }
}

pub(super) struct RedactionState {
    limits: RedactionBatchLimits,
    remaining: u64,
}
impl RedactionState {
    pub(super) fn new(limits: RedactionBatchLimits) -> Result<Self, BatchFault> {
        limits.validate()?;
        Ok(Self { limits, remaining: limits.max_total_rule_work })
    }
    pub(super) fn remaining(&self) -> u64 { self.remaining }
    pub(super) fn execute<C: DecodeStepControl>(&mut self, source: &str,
        options: &RulesOnlyOptions, budget: TextBudget, control: &mut C,
    ) -> Result<RulesOnlyOutput, BatchItemFailure> {
        options.validate().map_err(|fault| BatchItemFailure { fault, stop: false })?;
        if source.len() > budget.max_input_bytes { return Err(BatchItemFailure::reject(BatchCode::DocumentLimit)); }
        if budget.max_items == 0 { return Err(BatchItemFailure::reject(BatchCode::Admission)); }
        super::checkpoint(control)?;
        let reserved = self.limits.max_rule_work_per_scan.checked_mul(2)
            .ok_or_else(|| BatchItemFailure::fatal(BatchCode::InvalidLimits))?;
        self.remaining = self.remaining.checked_sub(reserved)
            .ok_or_else(|| BatchItemFailure::fatal(BatchCode::WorkLimit))?;
        let mut request = RedactionRequest::default();
        request.rules = options.rules.clone();
        request.actions.default_action = match options.action {
            RulesOnlyAction::Mask => RedactionAction::Mask,
            RulesOnlyAction::Placeholder => RedactionAction::Placeholder,
        };
        request.actions.include_map = options.include_map;
        request.verify = true;
        // Input itself has the narrower document cap. A replacement may grow,
        // so residual scanning is admitted against the complete output bound.
        request.rule_budget.max_input_bytes = budget.max_input_bytes.max(budget.max_output_bytes);
        request.rule_budget.max_detections = self.limits.max_detections.min(budget.max_items);
        request.rule_budget.max_work = self.limits.max_rule_work_per_scan;
        request.edit_budget.max_regions = request.rule_budget.max_detections;
        request.edit_budget.max_output_bytes = budget.max_output_bytes;
        let result = redact_rules(source, &request, None).map_err(failure)?;
        super::checkpoint(control)?;
        Ok(RulesOnlyOutput { scope: "rules_only", result, rule_work_ceiling: reserved,
            cumulative_rule_work_reserved: self.limits.max_total_rule_work - self.remaining })
    }
}

/// Flatten only a code-owned result. RedactionResult cannot be deserialized as
/// proof. Native forward work remains zero; rule work is separately named and
/// is a charged ceiling, not an observed operation count or performance claim.
#[derive(Serialize)]
pub(super) struct RulesOnlyOutput {
    scope: &'static str,
    #[serde(flatten)]
    result: RedactionResult,
    rule_work_ceiling: u64,
    cumulative_rule_work_reserved: u64,
}

fn failure(error: PipelineError<std::convert::Infallible>) -> BatchItemFailure {
    let error = match error {
        PipelineError::Redaction(error) => error,
        PipelineError::Model(never) => match never {},
        // Neither transformed text nor matched originals escape on failure.
        PipelineError::Residual(_) => return BatchItemFailure::reject(BatchCode::InvalidExecution),
    };
    BatchItemFailure::reject(match error {
        RedactError::InvalidOptions => BatchCode::Planning,
        RedactError::InputBudget => BatchCode::DocumentLimit,
        RedactError::WorkBudget => BatchCode::WorkLimit,
        RedactError::DetectionBudget | RedactError::CandidateBudget => BatchCode::Admission,
        RedactError::OutputBudget => BatchCode::OutputLineLimit,
        RedactError::AllocationRefused => BatchCode::Allocation,
        RedactError::Serialization => BatchCode::Serialization,
        RedactError::InvalidSpan | RedactError::InvalidNerEvidence | RedactError::MissingPolicy
        | RedactError::MissingKey | RedactError::KeyMismatch | RedactError::Collision
        | RedactError::VerificationResidual { .. } => BatchCode::InvalidExecution,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{batch::{self, BatchDocument, BatchLimits, BatchProcessor},
        text_batch::{CliControl, TextBatchOptions, TextBatchProcessor, TextBatchTask},
        tasks::redact::{PiiKind, actions::VerificationStatus}};
    use serde_json::{Value, json};

    fn processor() -> TextBatchProcessor {
        TextBatchProcessor::new(TextBatchOptions::for_task(TextBatchTask::RedactRules), TextBudget::default()).unwrap()
    }
    fn request(text: &str, options: RulesOnlyOptions) -> BatchDocument<TextBatchOptions> {
        BatchDocument { id: "safe-id".to_owned(), text: text.to_owned(),
            task_args: Some(TextBatchOptions::RedactRules { options }) }
    }
    #[test]
    fn batch_uses_real_rules_and_verifies_the_declared_scope() {
        let mut p = processor();
        let source = "a@example.org +1 (212) 555-0199";
        let item = p.prepare(request(source, RulesOnlyOptions::default())).unwrap();
        let output = p.execute(item, &mut CliControl).unwrap();
        let single = redact_rules(source, &RedactionRequest::default(), None).unwrap();
        assert_eq!(output.result["text"], single.text());
        assert_eq!(single.verification(), VerificationStatus::CleanDeclaredUnion);
        assert_eq!(output.result["verification"], "clean_declared_union");
        assert_eq!(output.result["scope"], "rules_only");
        assert_eq!(output.result["model_types"], json!([]));
        assert!(output.result.get("edits").is_none());
        assert!(!serde_json::to_string(&output).unwrap().contains("a@example.org"));
    }
    #[test]
    fn explicit_mask_maps_preserve_unicode_coordinates_without_original_values() {
        let mut p = processor();
        let options = RulesOnlyOptions { action: RulesOnlyAction::Mask, include_map: true, ..Default::default() };
        let item = p.prepare(request("é a@example.org 上海", options)).unwrap();
        let output = p.execute(item, &mut CliControl).unwrap();
        assert_eq!(output.result["text"], "é ************* 上海");
        let edit = &output.result["edits"][0];
        assert_eq!(edit["original"]["span"]["byte_start"], 3);
        assert_eq!(edit["original"]["span"]["scalar_start"], 2);
        assert_eq!(edit["output_scalar_start"], 2);
        assert!(!edit.to_string().contains("a@example.org"));
    }
    #[test]
    fn rule_selection_is_complete_and_never_leaks_into_the_next_document() {
        let mut p = processor();
        let options = RulesOnlyOptions { rules: RuleSet { enabled: [PiiKind::Date].into_iter().collect() }, ..Default::default() };
        let item = p.prepare(request("2024-02-29 a@example.org", options)).unwrap();
        let output = p.execute(item, &mut CliControl).unwrap();
        assert_eq!(output.result["text"], "[redacted:date] a@example.org");
        let item = p.prepare(BatchDocument { id: "next".into(), text: "2024-02-29 a@example.org".into(), task_args: None }).unwrap();
        let output = p.execute(item, &mut CliControl).unwrap();
        assert_eq!(output.result["text"], "2024-02-29 [redacted:email]");
    }
    #[test]
    fn unsupported_ner_empty_scope_and_keyed_actions_are_refused() {
        let mut p = processor();
        for enabled in [vec![], vec![PiiKind::Person], vec![PiiKind::Email, PiiKind::Organization]] {
            let options = RulesOnlyOptions { rules: RuleSet { enabled: enabled.into_iter().collect() }, ..Default::default() };
            assert!(p.prepare(request("private", options)).is_err());
        }
        for value in [json!({"action":"pseudonymize"}), json!({"key":"private"}), json!({"verify":false})] {
            assert!(serde_json::from_value::<RulesOnlyOptions>(value).is_err());
        }
    }
    #[test]
    fn a_refused_document_spends_its_work_ceiling_and_returns_no_text() {
        let mut p = processor();
        let before = p.rule_work_remaining();
        let private = format!("{}@example.org", "a".repeat(4097));
        let item = p.prepare(request(&private, RulesOnlyOptions::default())).unwrap();
        let error = match p.execute(item, &mut CliControl) { Err(error) => error, Ok(_) => panic!("oversized candidate") };
        assert_eq!(error.fault.code, BatchCode::Admission); assert!(!error.stop);
        let charged = 2 * RedactionBatchLimits::default().max_rule_work_per_scan;
        assert_eq!(p.rule_work_remaining(), before - charged);
        let item = p.prepare(request("safe text", RulesOnlyOptions::default())).unwrap();
        assert_eq!(p.execute(item, &mut CliControl).unwrap().result["text"], "safe text");
        assert!(!format!("{error:?}").contains(&private));
    }
    #[test]
    fn full_stream_rejects_a_private_bad_policy_then_delivers_later_redaction() {
        let input = concat!("{\"id\":\"bad\",\"text\":\"PRIVATE_MARKER\",\"task_args\":{\"kind\":\"redact_rules\",\"options\":{\"key\":\"SECRET_KEY\"}}}\n",
            "{\"id\":\"ok\",\"text\":\"a@example.org\"}\n");
        let mut output = Vec::new();
        let summary = batch::run_ndjson(&mut input.as_bytes(), &mut output, &mut processor(), BatchLimits::default(), &mut CliControl).unwrap();
        assert_eq!((summary.failed, summary.succeeded), (1, 1));
        let text = String::from_utf8(output).unwrap();
        for private in ["PRIVATE_MARKER", "SECRET_KEY", "a@example.org"] { assert!(!text.contains(private)); }
        assert!(text.contains("[redacted:email]"));
    }
    #[test]
    fn explicit_flush_never_replenishes_total_rule_work() {
        let limits = RedactionBatchLimits { max_detections: 8, max_rule_work_per_scan: 1_000_000, max_total_rule_work: 2_000_000 };
        let mut p = TextBatchProcessor::new_with_redaction_limits(TextBatchOptions::for_task(TextBatchTask::RedactRules), TextBudget::default(), limits).unwrap();
        let input = b"{\"id\":\"x\",\"text\":\"a@b.org\"}\n{\"flush\":true}\n{\"id\":\"x\",\"text\":\"a@b.org\"}\n{\"id\":\"never\",\"text\":\"unused\"}\n";
        let mut reader = std::io::Cursor::new(input); let mut output = Vec::new();
        let error = batch::run_ndjson(&mut reader, &mut output, &mut p, BatchLimits::default(), &mut CliControl).unwrap_err();
        assert_eq!(error.fault.code, BatchCode::WorkLimit);
        assert_eq!(p.rule_work_remaining(), 0); assert_eq!(error.summary.succeeded, 1);
        assert!(reader.position() < input.len() as u64);
        let rows: Vec<Value> = output.split(|b| *b == b'\n').filter(|s| !s.is_empty()).map(|s| serde_json::from_slice(s).unwrap()).collect();
        assert_eq!(rows.last().unwrap()["event"], "run_error");
        assert!(!rows.iter().any(|row| row["event"] == "run_complete"));
    }
    #[test]
    fn work_overflow_is_invalid_before_any_document_can_execute() {
        let limits = RedactionBatchLimits { max_rule_work_per_scan: u64::MAX, ..Default::default() };
        assert!(TextBatchProcessor::new_with_redaction_limits(TextBatchOptions::for_task(TextBatchTask::RedactRules), TextBudget::default(), limits).is_err());
    }
    #[test]
    fn cancellation_never_releases_a_result_or_spends_unstarted_work() {
        struct Cancel;
        impl DecodeStepControl for Cancel {
            fn checkpoint(&mut self, _: usize) -> Option<crate::native_engine::decode::DecodeCancellationKind> {
                Some(crate::native_engine::decode::DecodeCancellationKind::Deadline)
            }
        }
        let mut p = processor(); let before = p.rule_work_remaining();
        let item = p.prepare(request("a@example.org", RulesOnlyOptions::default())).unwrap();
        let error = match p.execute(item, &mut Cancel) { Err(error) => error, Ok(_) => panic!("cancelled redaction") };
        assert!(error.stop); assert_eq!(error.fault.code, BatchCode::Cancelled); assert_eq!(p.rule_work_remaining(), before);
    }
}
