//! Real pinned planning and shared edit/verification semantics. Synthetic native
//! receipts exercise the private seam only; these are not inference/recall evidence.
use super::*;
use crate::{
    artifact::converter::PINNED_REVISION,
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    native_engine::{generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::int4::convert::GROUPED_INT4_RECIPE, strict_int4::STRICT_INT4_EXECUTION},
    tasks::{extract::ExtractionGrounding, ir::ScoreSpace, ner::NamedEntity,
        redact::{PiiKind, actions::{ActionPolicy, RedactionAction, VerificationStatus},
            pseudonym::{PseudonymEncoding, PseudonymKey}, union::DetectedDocument}},
    tokenizer::{embedded::EmbeddedTokenizer, pinned_controls},
    validation::grounded_fields::{GroundingBudget, SourceOccurrence, scan_occurrences},
};

struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
fn planner() -> SourceTaskPlanner {
    let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    let registry = pinned_controls::pinned().unwrap();
    SourceTaskPlanner::pinned(registry.template_controls(), tokenizer.eos_token_id().unwrap()).unwrap()
}
fn config() -> Int4RedactionConfig {
    let mut cap = Int4Work::default();
    cap.forward_positions = u64::MAX; cap.projected_logits = u64::MAX; cap.attention_pairs = u64::MAX;
    cap.projections.dot_products = u64::MAX; cap.projections.multiply_accumulates = u64::MAX;
    Int4RedactionConfig {
        ner: NerOptions { types: vec![EntityType::Person], ..NerOptions::default() },
        per_pass: TaskBudget { max_input_tokens: 8192, max_output_tokens: 64, max_output_bytes: 1 << 20,
            max_grammar_states: 4096, max_kv_bytes: 1 << 31 },
        planning: SourcePlanningLimits::default(), max_model_work: cap,
        mask_limits: MaskWorkLimits::default(), mask_visits_per_pass: 1000,
        max_mask_visits: 2000, max_result_bytes: 4 << 20,
    }
}
fn identity(p: &SourceTaskPlanner, width: usize) -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"synthetic-int4-redaction-fixture");
    ExecutionIdentity {
        schema_version: 1, source_revision: PINNED_REVISION.to_owned(), logical_model_digest: d,
        artifact_format: INT4_SOURCE_FORMAT.to_owned(), quant_recipe: format!("{GROUPED_INT4_RECIPE}:group={width}"),
        packing_set_digest: Sha256Digest::of_bytes(&canonjson::canonical_bytes(&(
            INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", d, width,
        )).unwrap()),
        tokenizer_digest: p.tokenizer_digest(), template_digest: *p.template_digest(),
        task_spec: NER_TASK_VERSION.to_owned(), taskir_digest: d, prompt_digest: d,
        grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None,
    }
}
fn redactor(p: &SourceTaskPlanner, width: usize) -> Int4Redactor<'_> {
    Int4Redactor::from_bound_parts(p, identity(p, width), config(), width).unwrap()
}
fn receipt(plan: &PreparedInt4SourceTask, source: &str, mention: &str) -> Int4SourceTaskRun {
    let work = constrained_int4::planned_work(plan.group_width(), plan.prompt_tokens(), 2).unwrap();
    let entities = if source.contains(mention) {
        let spans = scan_occurrences(source, mention, &mut GroundingBudget::default()).unwrap();
        vec![NamedEntity { text: mention.to_owned(), entity_type: EntityType::Person,
            occurrence: if spans.len() == 1 { SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous }, spans }]
    } else { Vec::new() };
    Int4SourceTaskRun {
        schema_version: 1, execution: INT4_SOURCE_EXECUTION.to_owned(), group_width: plan.group_width(), model_work: work,
        result: SourceTaskResult::Ner(NerResult {
            schema_version: 1, task_spec_version: NER_TASK_VERSION.to_owned(), numerics_profile: STRICT_INT4_PROFILE.to_owned(),
            score_space: ScoreSpace::NotComputed, grounding: ExtractionGrounding::SourceMembership, entities,
            generated_token_ids: vec![1, 0], forward_positions: work.forward_positions,
            projected_logits: work.projected_logits, mask_node_visit_charge: 17,
        }),
    }
}
struct FixturePass<'a, 'p> {
    recipe: &'a Int4Redactor<'p>, ledger: Ledger, seen: Vec<String>,
    first_mention: &'static str, leak: bool, corrupt: bool, cancel_after: Option<usize>,
}
impl<'a, 'p> FixturePass<'a, 'p> {
    fn new(recipe: &'a Int4Redactor<'p>, verify: bool) -> Self {
        Self { recipe, ledger: Ledger::new(&recipe.config, verify).unwrap(), seen: Vec::new(),
            first_mention: "Alice", leak: false, corrupt: false, cancel_after: None }
    }
}
impl NerPass for FixturePass<'_, '_> {
    type Error = Int4RedactionError;
    fn types(&self) -> &BTreeSet<EntityType> { &self.recipe.types }
    fn run(&mut self, source: &str) -> Result<NerResult, Self::Error> {
        if self.ledger.failed { return Err(Int4RedactionError::InvalidResult); }
        self.ledger.failed = true;
        let plan = self.recipe.prepare(source, &mut Continue)?;
        self.ledger.reserve(plan.planned_work())?;
        self.seen.push(source.to_owned());
        let mention = if self.seen.len() == 1 { self.first_mention }
            else if self.leak { "redacted" } else { "absent-fixture" };
        let mut run = receipt(&plan, source, mention);
        if self.corrupt { run.model_work.projections.dot_products -= 1; }
        let (result, actual) = check_run(&plan, run, self.recipe.config.mask_visits_per_pass)?;
        self.ledger.finish(actual, result.mask_node_visit_charge)?;
        self.checkpoint()?;
        Ok(result)
    }
}
impl AccountedPass for FixturePass<'_, '_> {
    fn checkpoint(&mut self) -> Result<(), Int4RedactionError> {
        if self.cancel_after.is_some_and(|n| self.seen.len() >= n) {
            Err(Int4RedactionError::Cancelled(DecodeCancellationKind::Deadline))
        } else { Ok(()) }
    }
    fn ledger(&self) -> &Ledger { &self.ledger }
}

#[test]
fn fixed_recipe_rejects_profile_task_group_and_pinned_asset_substitution() {
    let p = planner();
    for axis in 0..12 {
        let mut id = identity(&p, 127);
        match axis {
            0 => id.numerics_profile = NumericsProfile::HfBf16Eager,
            1 => id.numerics_profile = NumericsProfile::StrictQuantized { version: 1 },
            2 => id.backend_semantic_version = "other".to_owned(),
            3 => id.task_spec = "redact-v1".to_owned(),
            4 => id.tokenizer_digest = Sha256Digest::of_bytes(b"other"),
            5 => id.template_digest = Sha256Digest::of_bytes(b"other"),
            6 => id.kv_dtype = "int8".to_owned(),
            7 => id.thinking_mode = ThinkingMode::Enabled,
            8 => id.tool_mode = ToolMode::Json,
            9 => id.quant_recipe = format!("{GROUPED_INT4_RECIPE}:group=32"),
            10 => id.packing_set_digest = Sha256Digest::of_bytes(b"other"),
            _ => id.artifact_format = "int8-fixture".to_owned(),
        }
        assert!(Int4Redactor::from_bound_parts(&p, id, config(), 127).is_err());
    }
    for width in [0, 32, usize::MAX] {
        assert!(Int4Redactor::from_bound_parts(&p, identity(&p, 127), config(), width).is_err());
    }
}
#[test]
fn exact_transformed_source_mints_a_new_plan_with_the_same_detector_scope() {
    let p = planner(); let r = redactor(&p, 127);
    let a = r.prepare("Alice <think>é", &mut Continue).unwrap();
    let b = r.prepare("[redacted:person] <think>é", &mut Continue).unwrap();
    assert_ne!(a.execution_identity().prompt_digest, b.execution_identity().prompt_digest);
    assert_eq!(a.execution_identity().schema_digest, b.execution_identity().schema_digest);
    assert_eq!(a.execution_identity().template_digest, b.execution_identity().template_digest);
    assert_eq!(a.execution_identity().quant_recipe, b.execution_identity().quant_recipe);
    assert!(a.verify_identity(b.execution_identity()).is_err());
}
#[test]
fn partial_group_geometry_is_retained_in_both_reservations_and_receipts() {
    let p = planner(); let mut runs = Vec::new();
    for width in [32, 127] {
        let r = redactor(&p, width); let mut f = FixturePass::new(&r, true);
        let out = r.run_pipeline("Alice", &RedactionRequest::default(), None, &mut f).unwrap();
        assert_eq!(r.group_width(), width); assert_eq!(out.group_width, width);
        assert_eq!(out.execution, INT4_REDACTION_EXECUTION); assert_eq!(out.numerics_profile, STRICT_INT4_PROFILE);
        assert!(within(out.model_work, out.reserved_model_work));
        runs.push(out);
    }
    assert_eq!(runs[0].model_work.projections.multiply_accumulates, runs[1].model_work.projections.multiply_accumulates);
    assert!(runs[0].model_work.projections.dot_products > runs[1].model_work.projections.dot_products);
    assert!(runs[0].reserved_model_work.projections.dot_products > runs[1].reserved_model_work.projections.dot_products);
    assert_ne!(runs[0].result.policy_digest(), runs[1].result.policy_digest());
}
#[test]
fn every_unicode_occurrence_and_rule_hit_is_edited_then_freshly_redetected() {
    let p = planner(); let r = redactor(&p, 127); let mut f = FixturePass::new(&r, true);
    let mut req = RedactionRequest::default(); req.actions.include_map = true;
    let source = "é Alice a@example.org Alice\r\n上海";
    let out = r.run_pipeline(source, &req, None, &mut f).unwrap();
    assert_eq!(out.result.text(), "é [redacted:person] [redacted:email] [redacted:person]\r\n上海");
    assert_eq!(f.seen, vec![source.to_owned(), out.result.text().to_owned()]);
    assert_eq!(out.result.verification(), VerificationStatus::CleanDeclaredUnion);
    assert_eq!(out.result.edits().len(), 3);
    assert_eq!(out.result.edits()[0].original.span.byte_start, 3);
    assert_eq!(out.result.edits()[0].original.span.scalar_start, 2);
    assert_eq!(out.ner_passes, 2); assert_eq!(out.mask_node_visit_charge, 34);
    assert_eq!(out.reserved_mask_node_visits, 2000);
    let wire = canonjson::canonical_string(&out).unwrap();
    for private in ["Alice", "a@example.org", "generated_token_ids", "prompt_digest", "logical_model_digest"] {
        assert!(!wire.contains(private));
    }
}
#[test]
fn overlapping_rule_and_ner_regions_cover_the_entire_connected_component() {
    let p = planner(); let r = redactor(&p, 32); let mut f = FixturePass::new(&r, true);
    f.first_mention = "example.org tail";
    let mut req = RedactionRequest::default(); req.actions.include_map = true;
    let source = "Alice@example.org tail untouched";
    let out = r.run_pipeline(source, &req, None, &mut f).unwrap();
    assert_eq!(out.result.text(), "[redacted:overlap] untouched");
    assert_eq!(out.result.region_count(), 1);
    assert_eq!(out.result.edits()[0].original.span.byte_end, source.find(" untouched").unwrap());
    assert_eq!(out.result.edits()[0].original.kinds.len(), 2);
}
#[test]
fn mask_action_keeps_untouched_utf8_and_masks_scalar_counts() {
    let p = planner(); let r = redactor(&p, 127); let mut f = FixturePass::new(&r, true);
    let mut req = RedactionRequest::default(); req.actions.default_action = RedactionAction::Mask;
    let out = r.run_pipeline("é Alice a@example.org Alice\r\n上海", &req, None, &mut f).unwrap();
    assert_eq!(out.result.text(), "é ***** ************* *****\r\n上海");
    assert_eq!(out.result.verification(), VerificationStatus::CleanDeclaredUnion);
}
#[test]
fn full256_pseudonyms_reuse_one_key_and_scope_without_exporting_private_values() {
    let p = planner(); let r = redactor(&p, 127);
    let key = PseudonymKey::from_bytes(&[7; 32], "fixture-key").unwrap();
    let context = Pseudonyms::full256(&key, "private-test-scope", None).unwrap();
    let mut req = RedactionRequest::default(); req.rules.enabled.clear();
    req.actions = ActionPolicy { default_action: RedactionAction::Pseudonymize,
        expected_key_commitment: Some(key.commitment()), ..ActionPolicy::default() };
    let mut f = FixturePass::new(&r, true);
    let out = r.run_pipeline("Alice Alice", &req, Some(&context), &mut f).unwrap();
    let expected = context.pseudonym(PiiKind::Person, "Alice").unwrap();
    assert_eq!(expected.len(), "person".len() + 3 + 64);
    assert_eq!(context.identity().encoding, PseudonymEncoding::Full256);
    assert_eq!(out.result.text(), format!("{expected} {expected}"));
    assert_eq!(f.seen[1], out.result.text());
    assert_eq!(out.result.verification(), VerificationStatus::CleanDeclaredUnion);
    let wire = canonjson::canonical_string(&out).unwrap();
    assert!(!wire.contains("Alice")); assert!(!wire.contains("private-test-scope"));
}
#[test]
fn missing_and_wrong_key_commitments_fail_before_native_detection() {
    let p = planner(); let r = redactor(&p, 32);
    let key = PseudonymKey::from_bytes(&[7; 32], "fixture-key").unwrap();
    let context = Pseudonyms::full256(&key, "scope", None).unwrap();
    let mut req = RedactionRequest::default(); req.actions.default_action = RedactionAction::Pseudonymize;
    let mut f = FixturePass::new(&r, true);
    assert!(matches!(r.run_pipeline("Alice", &req, None, &mut f),
        Err(Int4RedactionError::Redaction(RedactError::MissingKey))));
    assert!(f.seen.is_empty()); assert_eq!(f.ledger.reserved, Int4Work::default());
    req.actions.expected_key_commitment = Some("wrong".to_owned());
    assert!(matches!(r.run_pipeline("Alice", &req, Some(&context), &mut f),
        Err(Int4RedactionError::Redaction(RedactError::KeyMismatch))));
    assert!(f.seen.is_empty());
}
#[test]
fn residuals_return_transformed_coordinates_and_never_a_partial_document() {
    let p = planner(); let r = redactor(&p, 127); let mut f = FixturePass::new(&r, true); f.leak = true;
    let error = r.run_pipeline("Alice", &RedactionRequest::default(), None, &mut f).err().unwrap();
    let Int4RedactionError::Residual(report) = &error else { panic!("residual expected") };
    assert_eq!(f.seen.len(), 2); assert_eq!(report.residuals.len(), 1);
    assert_eq!(report.residuals[0].span.byte_start, 1);
    assert_eq!(report.residuals[0].span.byte_end, 9);
    let wire = canonjson::canonical_string(report).unwrap();
    for private in ["Alice", "redacted:person", "\"text\""] { assert!(!wire.contains(private)); }
    assert!(!format!("{error:?} {error}").contains("redacted:person"));
}
#[test]
fn explicit_verification_opt_out_is_reported_as_unverified() {
    let p = planner(); let r = redactor(&p, 32); let mut f = FixturePass::new(&r, false);
    let req = RedactionRequest { verify: false, ..RedactionRequest::default() };
    let out = r.run_pipeline("Alice", &req, None, &mut f).unwrap();
    assert_eq!(out.ner_passes, 1); assert_eq!(out.result.verification(), VerificationStatus::NotRequested);
    assert_eq!(out.reserved_mask_node_visits, 1000); assert_eq!(out.mask_node_visit_charge, 17);
}
#[test]
fn expanded_verification_text_must_fit_its_own_source_and_context_limits() {
    let p = planner(); let r = redactor(&p, 32);
    let first = r.prepare("Alice", &mut Continue).unwrap();
    let second = r.prepare("[redacted:person]", &mut Continue).unwrap();
    assert!(second.prompt_tokens() > first.prompt_tokens());
    for source_cap in [true, false] {
        let mut c = config();
        if source_cap { c.planning.max_input_bytes = 8; }
        else { c.planning.max_context_tokens = first.prompt_tokens() + c.per_pass.max_output_tokens as usize; }
        let r = Int4Redactor::from_bound_parts(&p, identity(&p, 32), c, 32).unwrap();
        let mut f = FixturePass::new(&r, true);
        assert!(r.run_pipeline("Alice", &RedactionRequest::default(), None, &mut f).is_err());
        assert_eq!(f.seen.len(), 1); assert!(f.ledger.failed);
        assert!(matches!(f.run("Alice"), Err(Int4RedactionError::InvalidResult)));
        assert_eq!(f.seen.len(), 1);
    }
}
#[test]
fn all_five_native_axes_are_cumulative_and_early_finish_never_refunds_reservations() {
    let work = Int4Work::for_sequence(127, 0, 5, 100).unwrap();
    for axis in 0..5 {
        let mut c = config(); c.max_model_work = work.checked_add(work).unwrap();
        match axis {
            0 => c.max_model_work.forward_positions -= 1,
            1 => c.max_model_work.projected_logits -= 1,
            2 => c.max_model_work.attention_pairs -= 1,
            3 => c.max_model_work.projections.dot_products -= 1,
            _ => c.max_model_work.projections.multiply_accumulates -= 1,
        }
        let mut ledger = Ledger::new(&c, true).unwrap();
        ledger.reserve(work).unwrap(); ledger.finish(Int4Work::default(), 0).unwrap();
        assert_eq!(ledger.reserved, work); assert_eq!(ledger.masks_reserved, 1000);
        assert!(matches!(ledger.reserve(work), Err(Int4RedactionError::WorkBudget)));
        assert_eq!(ledger.reserved, work); assert_eq!(ledger.masks_reserved, 1000);
        assert_eq!(ledger.started, 1); assert!(ledger.failed);
    }
}
#[test]
fn each_actual_receipt_must_fit_its_own_pass_even_when_earlier_work_was_unused() {
    let large = Int4Work::for_sequence(127, 0, 5, 100).unwrap();
    let small = Int4Work::for_sequence(127, 0, 1, 1).unwrap();
    let mut ledger = Ledger::new(&config(), true).unwrap();
    ledger.reserve(large).unwrap(); ledger.finish(Int4Work::default(), 0).unwrap();
    ledger.reserve(small).unwrap();
    assert!(within(large, ledger.reserved));
    assert!(matches!(ledger.finish(large, 1), Err(Int4RedactionError::InvalidResult)));
    assert_eq!(ledger.completed, 1); assert!(ledger.failed);
}
#[test]
fn no_verification_mask_budget_is_recreated_or_overflowed() {
    let mut c = config(); c.max_mask_visits = c.mask_visits_per_pass;
    assert!(matches!(Ledger::new(&c, true), Err(Int4RedactionError::WorkBudget)));
    assert!(Ledger::new(&c, false).is_ok());
    c.mask_visits_per_pass = u64::MAX; c.max_mask_visits = u64::MAX;
    assert!(Ledger::new(&c, true).is_err());
    let work = Int4Work::for_sequence(127, 0, 1, 1).unwrap();
    let mut ledger = Ledger::new(&config(), true).unwrap(); ledger.reserve(work).unwrap();
    assert!(ledger.finish(work, 1001).is_err()); assert!(ledger.failed); assert_eq!(ledger.completed, 0);
}
#[test]
fn complete_first_plan_reservation_can_exhaust_the_second_pass_budget() {
    let p = planner(); let r = redactor(&p, 127);
    let first = r.prepare("Alice", &mut Continue).unwrap().planned_work();
    let mut c = config(); c.max_model_work = first;
    let r = Int4Redactor::from_bound_parts(&p, identity(&p, 127), c, 127).unwrap();
    let mut f = FixturePass::new(&r, true);
    assert!(matches!(r.run_pipeline("Alice", &RedactionRequest::default(), None, &mut f),
        Err(Int4RedactionError::WorkBudget)));
    assert_eq!(f.seen.len(), 1); assert_eq!(f.ledger.reserved, first); assert!(f.ledger.failed);
}
#[test]
fn receipt_failure_is_terminal_and_cannot_retry_or_reach_verification() {
    let p = planner(); let r = redactor(&p, 127); let mut f = FixturePass::new(&r, true); f.corrupt = true;
    assert!(r.run_pipeline("Alice", &RedactionRequest::default(), None, &mut f).is_err());
    assert_eq!(f.seen.len(), 1); assert!(f.ledger.failed); assert_eq!(f.ledger.completed, 0);
    assert!(matches!(f.run("Alice"), Err(Int4RedactionError::InvalidResult))); assert_eq!(f.seen.len(), 1);
}
#[test]
fn receipt_checks_refuse_wrong_group_profile_schema_work_and_output_counts() {
    let p = planner(); let r = redactor(&p, 127); let plan = r.prepare("Alice", &mut Continue).unwrap();
    for axis in 0..15 {
        let mut raw = receipt(&plan, "Alice", "Alice");
        match axis {
            0 => raw.schema_version = 2, 1 => raw.execution = "other".to_owned(),
            2 => raw.group_width = 32, 3 => raw.model_work.forward_positions -= 1,
            4 => raw.model_work.projected_logits -= 1, 5 => raw.model_work.attention_pairs -= 1,
            6 => raw.model_work.projections.dot_products -= 1,
            7 => raw.model_work.projections.multiply_accumulates -= 1,
            _ => {
                let SourceTaskResult::Ner(ref mut n) = raw.result else { unreachable!() };
                match axis {
                    8 => n.numerics_profile = "strict-quantized-v1".to_owned(),
                    9 => n.task_spec_version = "redact-v1".to_owned(),
                    10 => n.schema_version = 2,
                    11 => n.generated_token_ids.clear(),
                    12 => n.generated_token_ids.resize(65, 1),
                    13 => n.forward_positions -= 1,
                    _ => n.mask_node_visit_charge = 1001,
                }
            }
        }
        assert!(matches!(check_run(&plan, raw, 1000), Err(Int4RedactionError::InvalidResult)));
    }
}
#[test]
fn int4_evidence_never_crosses_eager_or_int8_union_boundaries() {
    let p = planner(); let r = redactor(&p, 127); let source = "Alice";
    let plan = r.prepare(source, &mut Continue).unwrap(); let req = RedactionRequest::default();
    let SourceTaskResult::Ner(mut n) = receipt(&plan, source, "Alice").result else { unreachable!() };
    for profile in [NerProfile::Eager, NerProfile::Int8] {
        assert!(DetectedDocument::with_ner_profile(source, &req.rules, req.rule_budget,
            &n, &r.types, req.grounding_budget, profile).is_err());
    }
    n.numerics_profile = "hf-bf16-eager".to_owned();
    assert!(DetectedDocument::with_ner_profile(source, &req.rules, req.rule_budget,
        &n, &r.types, req.grounding_budget, NerProfile::Int4).is_err());
}
#[test]
fn missing_occurrences_forged_unicode_offsets_and_undeclared_types_cannot_be_edits() {
    let p = planner(); let r = redactor(&p, 127); let source = "é Alice Alice";
    let plan = r.prepare(source, &mut Continue).unwrap(); let req = RedactionRequest::default();
    for axis in 0..3 {
        let SourceTaskResult::Ner(mut n) = receipt(&plan, source, "Alice").result else { unreachable!() };
        match axis {
            0 => { n.entities[0].spans.pop(); }
            1 => n.entities[0].spans[0].scalar_start += 1,
            _ => n.entities[0].entity_type = EntityType::Organization,
        }
        assert!(DetectedDocument::with_ner_profile(source, &req.rules, req.rule_budget,
            &n, &r.types, req.grounding_budget, NerProfile::Int4).is_err());
    }
}
#[test]
fn cancellation_before_between_and_after_passes_preserves_the_typed_cause() {
    let p = planner(); let r = redactor(&p, 127);
    for completed in 0..=2 {
        let mut f = FixturePass::new(&r, true); f.cancel_after = Some(completed);
        let error = r.run_pipeline("Alice", &RedactionRequest::default(), None, &mut f).err().unwrap();
        assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline));
        assert_eq!(f.seen.len(), completed);
    }
    for error in [
        Int4RedactionError::Source(Int4SourceError::Cancelled(DecodeCancellationKind::Deadline)),
        Int4RedactionError::Native(StrictInt4Error::Cancelled(DecodeCancellationKind::Deadline)),
    ] {
        assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline));
        assert!(error.source().is_some());
    }
}
#[test]
fn full_allocated_kv_is_checked_even_for_short_logical_sequences() {
    let bytes = 8192 * KV_BYTES_PER_TOKEN as u64;
    assert!(check_capacity(8192, bytes).is_ok());
    assert!(matches!(check_capacity(8192, bytes - 1), Err(Int4RedactionError::WorkBudget)));
    assert!(matches!(check_capacity(8192, KV_BYTES_PER_TOKEN as u64), Err(Int4RedactionError::WorkBudget)));
    assert!(check_capacity(usize::MAX, u64::MAX).is_err());
}
#[test]
fn the_complete_native_envelope_must_fit_the_result_byte_cap() {
    let p = planner(); let r = redactor(&p, 127); let mut f = FixturePass::new(&r, true);
    let out = r.run_pipeline("Alice", &RedactionRequest::default(), None, &mut f).unwrap();
    let bytes = canonjson::canonical_bytes(&out).unwrap().len() as u64;
    let mut c = config(); c.max_result_bytes = bytes;
    let r = Int4Redactor::from_bound_parts(&p, identity(&p, 127), c, 127).unwrap();
    let mut req = RedactionRequest::default(); req.edit_budget.max_output_bytes = bytes as usize;
    assert!(r.run_pipeline("Alice", &req, None, &mut FixturePass::new(&r, true)).is_ok());
    let mut c = r.config.clone(); c.max_result_bytes -= 1;
    let r = Int4Redactor::from_bound_parts(&p, identity(&p, 127), c, 127).unwrap();
    req.edit_budget.max_output_bytes -= 1;
    assert!(r.run_pipeline("Alice", &req, None, &mut FixturePass::new(&r, true)).is_err());
}
