//! Real pinned planning and the real transactional redaction pipeline. Native
//! receipts here are private synthetic fixtures, not inference/recall evidence.
use super::*;
use crate::{
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    native_engine::strict_int8::STRICT_INT8_EXECUTION,
    tasks::{extract::ExtractionGrounding, ir::ScoreSpace, ner::NamedEntity,
        redact::{actions::{ActionPolicy, RedactionAction, VerificationStatus},
            pseudonym::PseudonymKey, union::DetectedDocument}},
    template::{IM_START, IM_END, THINK_START, THINK_END},
    tokenizer::{bpe::EncodeOptions, embedded::EmbeddedTokenizer, specials::ArchivedControlRegistries},
    validation::grounded_fields::{GroundingBudget, SourceOccurrence, scan_occurrences},
};
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn planner() -> SourceTaskPlanner {
    let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    let entries: Vec<_> = [IM_START, IM_END, THINK_START, THINK_END].iter().map(|&surface| {
        let ids = tokenizer.tokenizer().encode_ids_with_options(surface, EncodeOptions { add_bos: false, add_eos: false }).unwrap();
        assert_eq!(ids.len(), 1);
        serde_json::json!({"id":ids[0],"special":surface == IM_START || surface == IM_END,"surface":surface})
    }).collect();
    let specials: Vec<_> = entries.iter().filter(|e| e["special"] == true).cloned().collect();
    let registry = ArchivedControlRegistries::from_archived_json(
        &serde_json::json!({"schema_version":1,"registry":"TokenizerSpecialIds","entries":specials}).to_string(),
        &serde_json::json!({"schema_version":1,"registry":"TemplateControlIds","entries":entries}).to_string()).unwrap();
    SourceTaskPlanner::pinned(registry.template_controls(), tokenizer.eos_token_id().unwrap()).unwrap()
}
fn config() -> Int8RedactionConfig {
    let mut cap = Int8Work::default();
    cap.forward_positions = u64::MAX; cap.projected_logits = u64::MAX; cap.attention_pairs = u64::MAX;
    cap.projections.dot_products = u64::MAX; cap.projections.multiply_accumulates = u64::MAX;
    Int8RedactionConfig { ner: NerOptions { types: vec![EntityType::Person], ..NerOptions::default() },
        per_pass: TaskBudget { max_input_tokens: 8192, max_output_tokens: 64, max_output_bytes: 1 << 20,
            max_grammar_states: 4096, max_kv_bytes: 1 << 31 }, planning: SourcePlanningLimits::default(),
        max_model_work: cap, mask_limits: MaskWorkLimits::default(), mask_visits_per_pass: 1000,
        max_mask_visits: 2000, max_result_bytes: 4 << 20 }
}
fn identity(p: &SourceTaskPlanner) -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"synthetic-redaction-fixture");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "int8-fixture".to_owned(), packing_set_digest: d,
        tokenizer_digest: p.tokenizer_digest(), template_digest: *p.template_digest(), task_spec: NER_TASK_VERSION.to_owned(),
        taskir_digest: d, prompt_digest: d, grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(), sampler_version: "fixture".to_owned(),
        thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None, calibration_digest: d, decision_policy_digest: d,
        backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(), host_class: None, compiler_identity: None }
}
fn redactor(p: &SourceTaskPlanner) -> Int8Redactor<'_> { Int8Redactor::new(p, identity(p), config()).unwrap() }
fn ner(plan: &PreparedInt8SourceTask, text: &str, mention: &str) -> Int8SourceTaskRun {
    let work = constrained_int8::planned_work(plan.prompt_tokens(), 2).unwrap();
    let entities = if text.contains(mention) {
        let spans = scan_occurrences(text, mention, &mut GroundingBudget::default()).unwrap();
        vec![NamedEntity { text: mention.to_owned(), entity_type: EntityType::Person,
            occurrence: if spans.len() == 1 { SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous }, spans }]
    } else { vec![] };
    Int8SourceTaskRun { schema_version: 1, execution: INT8_SOURCE_EXECUTION.to_owned(), model_work: work,
        result: SourceTaskResult::Ner(NerResult { schema_version: 1, task_spec_version: NER_TASK_VERSION.to_owned(),
            numerics_profile: STRICT_INT8_PROFILE.to_owned(), score_space: ScoreSpace::NotComputed,
            grounding: ExtractionGrounding::SourceMembership, entities, generated_token_ids: vec![1, 0],
            forward_positions: work.forward_positions, projected_logits: work.projected_logits, mask_node_visit_charge: 17 }) }
}
struct Fake<'a, 'p> {
    recipe: &'a Int8Redactor<'p>, ledger: Ledger, seen: Vec<String>,
    leak: bool, cancel_after: Option<usize>, corrupt: bool,
}
impl<'a, 'p> Fake<'a, 'p> {
    fn new(recipe: &'a Int8Redactor<'p>, verify: bool) -> Self {
        Self { recipe, ledger: Ledger::new(&recipe.config, verify).unwrap(), seen: vec![],
            leak: false, cancel_after: None, corrupt: false }
    }
}
impl NerPass for Fake<'_, '_> {
    type Error = Int8RedactionError;
    fn types(&self) -> &BTreeSet<EntityType> { &self.recipe.types }
    fn run(&mut self, source: &str) -> Result<NerResult, Self::Error> {
        if self.ledger.failed { return Err(Int8RedactionError::InvalidResult); }
        self.ledger.failed = true;
        let plan = self.recipe.prepare(source, &mut Continue)?;
        self.ledger.reserve(plan.planned_work())?;
        self.seen.push(source.to_owned());
        let mut raw = ner(&plan, source, if self.seen.len() == 1 { "Alice" } else if self.leak { "redacted" } else { "absent-fixture" });
        if self.corrupt { raw.model_work.attention_pairs -= 1; }
        let (out, work) = check_run(&plan, raw, self.recipe.config.mask_visits_per_pass)?;
        self.ledger.finish(work, out.mask_node_visit_charge)?;
        self.checkpoint()?;
        Ok(out)
    }
}
impl AccountedPass for Fake<'_, '_> {
    fn checkpoint(&mut self) -> Result<(), Int8RedactionError> {
        if self.cancel_after.is_some_and(|n| self.seen.len() >= n) { Err(Int8RedactionError::Cancelled(DecodeCancellationKind::Deadline)) }
        else { Ok(()) }
    }
    fn ledger(&self) -> &Ledger { &self.ledger }
}

#[test]
fn fixed_recipe_rejects_eager_other_task_and_asset_substitution() {
    let p = planner();
    for axis in 0..6 {
        let mut id = identity(&p);
        match axis { 0 => id.numerics_profile = NumericsProfile::HfBf16Eager,
            1 => id.backend_semantic_version = "other".to_owned(), 2 => id.task_spec = "redact-v1".to_owned(),
            3 => id.tokenizer_digest = Sha256Digest::of_bytes(b"other"),
            4 => id.template_digest = Sha256Digest::of_bytes(b"other"), _ => id.kv_dtype = "int8".to_owned() }
        assert!(Int8Redactor::new(&p, id, config()).is_err());
    }
}
#[test]
fn each_pass_is_replanned_from_its_own_exact_text() {
    let p = planner(); let r = redactor(&p);
    let a = r.prepare("Alice <think>", &mut Continue).unwrap();
    let b = r.prepare("[redacted:person] <think>", &mut Continue).unwrap();
    assert_ne!(a.execution_identity().prompt_digest, b.execution_identity().prompt_digest);
    assert_eq!(a.execution_identity().schema_digest, b.execution_identity().schema_digest);
    assert_eq!(a.execution_identity().template_digest, b.execution_identity().template_digest);
    assert!(a.verify_identity(b.execution_identity()).is_err());
}
#[test]
fn rules_and_all_unicode_ner_occurrences_are_edited_and_redetected() {
    let p = planner(); let r = redactor(&p); let mut f = Fake::new(&r, true);
    let mut request = RedactionRequest::default(); request.actions.include_map = true;
    let source = "é Alice a@example.org Alice\r\n上海";
    let out = r.run_pipeline(source, &request, None, &mut f).unwrap();
    assert_eq!(f.seen, vec![source.to_owned(), out.result.text().to_owned()]);
    assert_eq!(out.result.text(), "é [redacted:person] [redacted:email] [redacted:person]\r\n上海");
    assert_eq!(out.result.verification(), VerificationStatus::CleanDeclaredUnion);
    assert_eq!(out.result.edits().len(), 3); assert_eq!(out.result.edits()[0].original.span.byte_start, 3);
    assert_eq!(out.ner_passes, 2); assert_eq!(out.mask_node_visit_charge, 34); assert_eq!(out.reserved_mask_node_visits, 2000);
    assert_eq!(out.ner_template_digest, *p.template_digest());
    assert!(within(out.model_work, out.reserved_model_work));
    let wire = canonjson::canonical_string(&out).unwrap();
    for private in ["Alice", "a@example.org", "prompt_digest", "generated_token_ids"] { assert!(!wire.contains(private)); }
}
#[test]
fn residuals_return_coordinates_only_and_no_partially_redacted_output() {
    let p = planner(); let r = redactor(&p); let mut f = Fake::new(&r, true); f.leak = true;
    let e = r.run_pipeline("Alice", &RedactionRequest::default(), None, &mut f).err().unwrap();
    assert_eq!(f.seen.len(), 2);
    let Int8RedactionError::Residual(report) = &e else { panic!("residual expected") };
    assert!(!report.residuals.is_empty());
    assert!(!canonjson::canonical_string(report).unwrap().contains("Alice"));
    assert!(!format!("{e:?} {e}").contains("redacted:person"));
}
#[test]
fn verification_opt_out_never_claims_a_clean_second_pass() {
    let p = planner(); let r = redactor(&p); let mut f = Fake::new(&r, false);
    let request = RedactionRequest { verify: false, ..RedactionRequest::default() };
    let out = r.run_pipeline("Alice", &request, None, &mut f).unwrap();
    assert_eq!(out.ner_passes, 1); assert_eq!(out.result.verification(), VerificationStatus::NotRequested);
    assert_eq!(out.reserved_mask_node_visits, 1000);
}
#[test]
fn cancellation_before_between_and_after_model_passes_suppresses_success() {
    let p = planner(); let r = redactor(&p);
    for n in 0..=2 {
        let mut f = Fake::new(&r, true); f.cancel_after = Some(n);
        let e = r.run_pipeline("Alice", &RedactionRequest::default(), None, &mut f).err().unwrap();
        assert_eq!(e.cancellation(), Some(DecodeCancellationKind::Deadline)); assert_eq!(f.seen.len(), n);
    }
}
#[test]
fn model_work_failure_stops_before_redetection() {
    let p = planner(); let r = redactor(&p); let mut f = Fake::new(&r, true); f.corrupt = true;
    assert!(r.run_pipeline("Alice", &RedactionRequest::default(), None, &mut f).is_err());
    assert_eq!(f.seen.len(), 1); assert!(f.ledger.failed);
}

#[test]
fn late_cancellation_does_not_erase_an_already_returned_native_failure() {
    let p = planner(); let r = redactor(&p); let mut f = Fake::new(&r, true);
    f.corrupt = true; f.cancel_after = Some(1);
    assert!(matches!(r.run_pipeline("Alice", &RedactionRequest::default(), None, &mut f),
        Err(Int8RedactionError::InvalidResult)));
    assert_eq!(f.seen.len(), 1);
}
#[test]
fn second_pass_is_not_given_a_new_copy_of_the_model_budget() {
    let p = planner(); let r = redactor(&p);
    let first = r.prepare("Alice", &mut Continue).unwrap().planned_work();
    let mut c = config(); c.max_model_work = first;
    let r = Int8Redactor::new(&p, identity(&p), c).unwrap(); let mut f = Fake::new(&r, true);
    let e = r.run_pipeline("Alice", &RedactionRequest::default(), None, &mut f).err().unwrap();
    assert!(matches!(e, Int8RedactionError::WorkBudget)); assert_eq!(f.seen.len(), 1);
    assert_eq!(f.ledger.reserved, first); assert!(f.ledger.failed);
}
#[test]
fn model_and_mask_reservations_are_atomic_and_not_refunded_by_early_finish() {
    let work = Int8Work::for_sequence(0, 5, 100).unwrap();
    for axis in 0..5 {
        let mut c = config(); c.max_model_work = work;
        match axis { 0 => c.max_model_work.forward_positions -= 1, 1 => c.max_model_work.projected_logits -= 1,
            2 => c.max_model_work.attention_pairs -= 1, 3 => c.max_model_work.projections.dot_products -= 1,
            _ => c.max_model_work.projections.multiply_accumulates -= 1 }
        let mut l = Ledger::new(&c, true).unwrap(); assert!(l.reserve(work).is_err());
        assert_eq!(l.reserved, Int8Work::default()); assert_eq!(l.masks_reserved, 0); assert!(l.failed);
    }
    let mut l = Ledger::new(&config(), true).unwrap(); l.reserve(work).unwrap();
    l.finish(Int8Work::default(), 0).unwrap(); assert_eq!(l.reserved, work); assert_eq!(l.masks_reserved, 1000);
    l.reserve(work).unwrap(); l.finish(work, 1).unwrap(); assert!(l.reserve(work).is_err());
}
#[test]
fn missing_verification_mask_budget_refuses_before_the_first_pass() {
    let mut c = config(); c.max_mask_visits = c.mask_visits_per_pass;
    assert!(Ledger::new(&c, true).is_err()); assert!(Ledger::new(&c, false).is_ok());
    c.mask_visits_per_pass = u64::MAX; c.max_mask_visits = u64::MAX;
    assert!(Ledger::new(&c, true).is_err());
}
#[test]
fn missing_pseudonym_key_is_not_discovered_after_model_work() {
    let p = planner(); let r = redactor(&p); let mut f = Fake::new(&r, true);
    let mut req = RedactionRequest::default(); req.actions.default_action = RedactionAction::Pseudonymize;
    assert!(matches!(r.run_pipeline("Alice", &req, None, &mut f), Err(Int8RedactionError::Redaction(RedactError::MissingKey))));
    assert!(f.seen.is_empty());
}
#[test]
fn same_scope_hmac_is_reused_without_exporting_original_values_or_keys() {
    let p = planner(); let r = redactor(&p); let key = PseudonymKey::from_bytes(&[7; 32], "fixture").unwrap();
    let context = Pseudonyms::full256(&key, "test-scope", None).unwrap();
    let mut req = RedactionRequest::default(); req.rules.enabled.clear();
    req.actions = ActionPolicy { default_action: RedactionAction::Pseudonymize, ..ActionPolicy::default() };
    let mut f = Fake::new(&r, true); let out = r.run_pipeline("Alice Alice", &req, Some(&context), &mut f).unwrap();
    let expected = context.pseudonym(super::super::PiiKind::Person, "Alice").unwrap();
    assert_eq!(out.result.text(), format!("{expected} {expected}"));
    assert_eq!(out.result.verification(), VerificationStatus::CleanDeclaredUnion);
}
#[test]
fn eager_and_int8_evidence_never_cross_pipeline_boundaries() {
    let p = planner(); let r = redactor(&p); let plan = r.prepare("Alice Alice", &mut Continue).unwrap();
    let SourceTaskResult::Ner(mut result) = ner(&plan, "Alice Alice", "Alice").result else { unreachable!() };
    let req = RedactionRequest::default();
    assert!(DetectedDocument::with_ner("Alice Alice", &req.rules, req.rule_budget, &result, &r.types, req.grounding_budget).is_err());
    result.numerics_profile = "hf-bf16-eager".to_owned();
    assert!(DetectedDocument::with_ner_profile("Alice Alice", &req.rules, req.rule_budget, &result,
        &r.types, req.grounding_budget, NerProfile::Int8).is_err());
}
#[test]
fn int8_grounding_rechecks_every_occurrence_before_edits() {
    let p = planner(); let r = redactor(&p); let plan = r.prepare("é Alice Alice", &mut Continue).unwrap();
    let req = RedactionRequest::default();
    for partial in [true, false] {
        let SourceTaskResult::Ner(mut result) = ner(&plan, "é Alice Alice", "Alice").result else { unreachable!() };
        if partial { result.entities[0].spans.pop(); } else { result.entities[0].spans[0].scalar_start += 1; }
        assert!(DetectedDocument::with_ner_profile("é Alice Alice", &req.rules, req.rule_budget, &result,
            &r.types, req.grounding_budget, NerProfile::Int8).is_err());
    }
}
#[test]
fn complete_native_envelope_obeys_the_host_output_limit() {
    let p = planner(); let r = redactor(&p); let mut f = Fake::new(&r, true);
    let out = r.run_pipeline("Alice", &RedactionRequest::default(), None, &mut f).unwrap();
    let bytes = canonjson::canonical_bytes(&out).unwrap().len() as u64;
    let mut c = config(); c.max_result_bytes = bytes;
    let r = Int8Redactor::new(&p, identity(&p), c).unwrap();
    let mut req = RedactionRequest::default(); req.edit_budget.max_output_bytes = bytes as usize;
    // Changing the public policy binding changes the digest, not its length.
    assert!(r.run_pipeline("Alice", &req, None, &mut Fake::new(&r, true)).is_ok());
    let mut c = r.config.clone(); c.max_result_bytes -= 1;
    let r = Int8Redactor::new(&p, identity(&p), c).unwrap(); req.edit_budget.max_output_bytes -= 1;
    assert!(r.run_pipeline("Alice", &req, None, &mut Fake::new(&r, true)).is_err());
}
#[test]
fn native_receipt_validation_rejects_schema_profile_and_missing_work() {
    let p = planner(); let r = redactor(&p); let plan = r.prepare("Alice", &mut Continue).unwrap();
    for axis in 0..6 {
        let mut raw = ner(&plan, "Alice", "Alice");
        match axis { 0 => raw.schema_version = 2, 1 => raw.execution = "other".to_owned(),
            2 => raw.model_work.projections.multiply_accumulates -= 1,
            _ => { let SourceTaskResult::Ner(ref mut n) = raw.result else { unreachable!() };
                match axis { 3 => n.numerics_profile = "hf-bf16-eager".to_owned(),
                    4 => n.mask_node_visit_charge = 1001, _ => n.generated_token_ids.clear() } }
        }
        assert!(check_run(&plan, raw, 1000).is_err());
    }
}
