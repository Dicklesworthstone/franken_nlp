//! Pinned preparation and private evidence-corruption fixtures. None executes
//! model weights or certifies NER/redaction recall, privacy or performance.
use super::*;
use crate::{execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    grammar::mask::MaskWorkLimits,
    native_engine::strict_int8::STRICT_INT8_EXECUTION,
    tasks::{extract::ExtractionGrounding, ir::ScoreSpace, mapreduce::{ChunkLimits, ExecutionLimits},
        ner::{EntityType, NamedEntity, NerResult}},
    tokenizer::pinned_controls,
    validation::grounded_fields::{SourceOccurrence, VerifiedSourceSpan, scan_occurrences}};
use super::super::{PiiKind, Detector, actions::{ActionPolicy, RedactionAction}};

struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn planner() -> SourceTaskPlanner {
    let registry = pinned_controls::pinned().unwrap();
    let controls = registry.template_controls();
    let eos = controls.entries().iter().find(|e| e.special && e.surface == crate::template::IM_END).unwrap().id;
    SourceTaskPlanner::pinned(controls, eos).unwrap()
}
fn identity(p: &SourceTaskPlanner) -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"long-redaction-model-free-fixture");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "int8-fixture".to_owned(), packing_set_digest: d,
        tokenizer_digest: p.tokenizer_digest(), template_digest: *p.template_digest(), task_spec: NER_TASK_VERSION.to_owned(),
        taskir_digest: d, prompt_digest: d, grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None }
}
fn config() -> LongRedactionConfig {
    LongRedactionConfig { ner: NerOptions::default(),
        per_chunk: TaskBudget { max_input_tokens: 8192, max_output_tokens: 64, max_output_bytes: 1 << 20,
            max_grammar_states: 4096, max_kv_bytes: 1 << 31 },
        planning: SourcePlanningLimits::default(),
        mapping: Int8SourceMapLimits {
            chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 6, max_chunk_tokens: 6,
                context_tokens: 8192, reserved_tokens: 64, max_chunks: 64, max_tokenizer_calls: 1024 },
            reduction: ExecutionLimits::default(), max_model_work: Int8Work::for_sequence(0, 16384, 100_000_000).unwrap(),
            mask_limits: MaskWorkLimits::default(), mask_visits_per_chunk: 1000, max_mask_visits: 64000 },
        max_result_bytes: 4 << 20 }
}
fn raw(source: &str, names: &[&str]) -> NerResult {
    NerResult { schema_version: 1, task_spec_version: NER_TASK_VERSION.to_owned(), numerics_profile: STRICT_INT8_PROFILE.to_owned(),
        score_space: ScoreSpace::NotComputed, grounding: ExtractionGrounding::SourceMembership,
        entities: names.iter().map(|&text| {
            let spans = scan_occurrences(source, text, &mut GroundingBudget::default()).unwrap();
            NamedEntity { text: text.to_owned(), entity_type: EntityType::Person,
                occurrence: if spans.len() == 1 { SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous }, spans }
        }).collect(), generated_token_ids: Vec::new(), forward_positions: 0, projected_logits: 0, mask_node_visit_charge: 0 }
}
fn extent(source: &str, start: usize, end: usize) -> VerifiedSourceSpan {
    VerifiedSourceSpan { byte_start: start, byte_end: end,
        scalar_start: source[..start].chars().count(), scalar_end: source[..end].chars().count() }
}
fn collect<'s>(source: &'s str, parts: &[(&str, &[&str])], request: &RedactionRequest,
    budget: &mut GroundingBudget) -> Result<DetectedDocument<'s>, RedactError> {
    let mut out = detection::Detections::new(source, request, &NerOptions::default())?;
    let mut offset = 0;
    for (index, (text, names)) in parts.iter().enumerate() {
        out.push(index, extent(source, offset, offset + text.len()), &raw(text, names), budget)?;
        offset += text.len();
    }
    out.finish(parts.len())
}

#[test]
fn whole_document_rules_see_email_spanning_neural_chunks() {
    let source = "a@example.org"; let request = RedactionRequest::default();
    let doc = collect(source, &[("a@exam", &[]), ("ple.org", &[])], &request, &mut GroundingBudget::default()).unwrap();
    assert!(doc.regions().iter().any(|r| r.kinds.contains(&PiiKind::Email)
        && r.span.byte_start == 0 && r.span.byte_end == source.len()));
    let result = actions::apply(&doc, &request.actions, None, request.edit_budget).unwrap();
    assert!(!result.text().contains(source));
    assert!(result.model_types().contains(&EntityType::Person));
}
#[test]
fn every_repeated_unicode_mention_is_lifted_before_single_transactional_edit() {
    let source = "éAéA éAéA"; let mut request = RedactionRequest::default();
    request.actions = ActionPolicy { default_action: RedactionAction::Mask, include_map: true, ..ActionPolicy::default() };
    let doc = collect(source, &[("éAéA ", &["é"]), ("éAéA", &["é"])], &request, &mut GroundingBudget::default()).unwrap();
    assert_eq!(doc.regions().len(), 4);
    assert_eq!(doc.regions()[2].span.byte_start, 7); assert_eq!(doc.regions()[2].span.scalar_start, 5);
    let result = actions::apply(&doc, &request.actions, None, request.edit_budget).unwrap();
    assert_eq!(result.text(), "*A*A *A*A"); assert_eq!(result.edits().len(), 4);
    for region in doc.regions() { assert_eq!(&source[region.span.byte_start..region.span.byte_end], "é"); }
}
#[test]
fn duplicated_proposals_are_checked_and_charged_before_overlap_union() {
    let request = RedactionRequest::default(); let source = "Alice Alice";
    let mut budget = GroundingBudget::default(); let before = budget;
    let doc = collect(source, &[(source, &["Alice", "Alice"])], &request, &mut budget).unwrap();
    assert_eq!(doc.regions().len(), 2); assert_eq!(doc.detection_count(), 4);
    assert_eq!(before.max_fields - budget.max_fields, 2); assert_eq!(before.max_matches - budget.max_matches, 4);
    let mut out = detection::Detections::new(source, &request, &NerOptions::default()).unwrap();
    let mut bad = raw(source, &["Alice", "Alice"]); bad.entities[1].spans.pop();
    assert!(out.push(0, extent(source, 0, source.len()), &bad, &mut GroundingBudget::default()).is_err());
    assert!(out.finish(1).is_err());
}
#[test]
fn corrupt_last_chunk_never_becomes_partial_redaction() {
    let source = "Alice éAéA"; let request = RedactionRequest::default();
    for axis in 0..6 {
        let mut out = detection::Detections::new(source, &request, &NerOptions::default()).unwrap();
        let mut budget = GroundingBudget::default();
        out.push(0, extent(source, 0, 6), &raw("Alice ", &["Alice"]), &mut budget).unwrap();
        let mut bad = raw("éAéA", &["é"]);
        match axis { 0 => { bad.entities[0].spans.pop(); }, 1 => bad.entities[0].spans[0].scalar_end += 1,
            2 => bad.entities[0].occurrence = SourceOccurrence::Anchored,
            3 => bad.numerics_profile = "hf-bf16-eager".to_owned(),
            4 => bad.entities[0].entity_type = EntityType::Event, _ => bad.entities[0].text = "absent".to_owned() }
        assert!(out.push(1, extent(source, 6, source.len()), &bad, &mut budget).is_err(), "{axis}");
        assert!(out.finish(2).is_err());
    }
}
#[test]
fn coverage_rejects_missing_reordered_overlapping_and_wrong_scalar_chunks() {
    let source = "Alice Bob"; let request = RedactionRequest::default();
    let first = raw("Alice ", &[]);
    for axis in 0..4 {
        let mut out = detection::Detections::new(source, &request, &NerOptions::default()).unwrap();
        let mut origin = extent(source, 0, 6);
        match axis { 0 => origin.byte_start = 1, 1 => origin.scalar_end += 1,
            2 => origin.byte_end = 0, _ => origin.scalar_start = 1 }
        assert!(out.push(0, origin, &first, &mut GroundingBudget::default()).is_err());
        assert!(out.finish(1).is_err());
    }
    let mut out = detection::Detections::new(source, &request, &NerOptions::default()).unwrap();
    assert!(out.push(1, extent(source, 0, 6), &first, &mut GroundingBudget::default()).is_err());
    let mut out = detection::Detections::new(source, &request, &NerOptions::default()).unwrap();
    out.push(0, extent(source, 0, 6), &first, &mut GroundingBudget::default()).unwrap();
    assert!(out.finish(1).is_err());
}
#[test]
fn rule_and_model_overlap_retains_both_detector_provenances() {
    let source = "a@example.org"; let request = RedactionRequest::default();
    let doc = collect(source, &[(source, &[source])], &request, &mut GroundingBudget::default()).unwrap();
    assert_eq!(doc.regions().len(), 1);
    assert!(doc.regions()[0].detectors.contains(&Detector::NerSourceV1));
    assert!(doc.regions()[0].detectors.contains(&Detector::EmailAsciiV1));
    assert_eq!(doc.regions()[0].span.byte_end, source.len());
}
#[test]
fn complete_detection_and_grounding_allowances_are_not_renewed_per_chunk_or_stage() {
    let mut request = RedactionRequest::default(); request.rule_budget.max_detections = 1;
    assert!(collect("Alice Bob", &[("Alice ", &["Alice"]), ("Bob", &["Bob"])],
        &request, &mut GroundingBudget::default()).is_err());
    let request = RedactionRequest::default();
    for axis in 0..2 {
        let mut budget = GroundingBudget::default();
        if axis == 0 { budget.max_fields = 1; } else { budget.max_matches = 1; }
        let first = collect("Alice", &[("Alice", &["Alice"])], &request, &mut budget).unwrap(); drop(first);
        assert!(collect("Bob", &[("Bob", &["Bob"])], &request, &mut budget).is_err());
    }
    let mut measured = GroundingBudget::default(); let original = measured;
    collect("Alice Bob", &[("Alice ", &["Alice"]), ("Bob", &["Bob"])], &request, &mut measured).unwrap();
    let used = original.max_scan_steps - measured.max_scan_steps;
    let mut short = GroundingBudget { max_scan_steps: used - 1, ..original };
    assert!(collect("Alice Bob", &[("Alice ", &["Alice"]), ("Bob", &["Bob"])], &request, &mut short).is_err());
}
#[test]
fn fresh_transformed_source_verification_rejects_residuals_without_exporting_text() {
    let source = "Alice Bob"; let request = RedactionRequest::default(); let mut budget = GroundingBudget::default();
    let first = collect(source, &[(source, &["Alice"])], &request, &mut budget).unwrap();
    let edited = actions::apply(&first, &request.actions, None, request.edit_budget).unwrap();
    assert!(!edited.text().contains("Alice")); assert!(edited.text().contains("Bob"));
    let remaining = collect(edited.text(), &[(edited.text(), &["Bob"])], &request, &mut budget).unwrap();
    let error = require_clean(remaining, 1 << 20).unwrap_err();
    let LongRedactionError::Redaction(Int8RedactionError::Residual(report)) = error else { panic!("residual") };
    let json = canonjson::canonical_string(&report).unwrap();
    assert!(!json.contains("Bob")); assert!(!json.contains("Alice")); assert_eq!(report.residuals.len(), 1);
    assert_eq!(&edited.text()[report.residuals[0].span.byte_start..report.residuals[0].span.byte_end], "Bob");
    let clean = collect("*****", &[("*****", &[])], &request, &mut budget).unwrap();
    require_clean(clean, 1 << 20).unwrap();
}
#[test]
fn actual_pinned_preflight_covers_every_chunk_and_binds_ner_profile() {
    let p = planner(); let c = config(); let redactor = Int8DocumentRedactor::new(&p, identity(&p), c.clone()).unwrap();
    let source = "éAéAéAéA";
    let sized = redactor.preflight(source, &mut Continue).unwrap();
    let plans = redactor.prepare(source, c.mapping, &mut Continue).unwrap();
    assert_eq!(sized.chunks, 2); assert_eq!(sized.source_bytes, source.len());
    assert_eq!(sized.source_scalars, 8); assert_eq!(sized.planned_model_work, plans.planned_work());
    assert_eq!(sized.reserved_mask_node_visits, 2000);
    for actual in plans.execution_identities() {
        assert_eq!(actual.task_spec, NER_TASK_VERSION); assert_eq!(actual.numerics_profile, NumericsProfile::StrictQuantized { version: 1 });
        assert_eq!(actual.logical_model_digest, identity(&p).logical_model_digest);
    }
}
#[test]
fn all_five_original_preflight_work_limits_fail_before_any_native_execution() {
    let p = planner(); let c = config(); let source = "éAéAéAéA";
    let exact = Int8DocumentRedactor::new(&p, identity(&p), c.clone()).unwrap().preflight(source, &mut Continue).unwrap();
    for axis in 0..6 {
        let mut c = c.clone(); c.mapping.max_model_work = exact.planned_model_work; c.mapping.max_mask_visits = exact.reserved_mask_node_visits;
        match axis { 0 => c.mapping.max_model_work.forward_positions -= 1, 1 => c.mapping.max_model_work.projected_logits -= 1,
            2 => c.mapping.max_model_work.attention_pairs -= 1, 3 => c.mapping.max_model_work.projections.dot_products -= 1,
            4 => c.mapping.max_model_work.projections.multiply_accumulates -= 1, _ => c.mapping.max_mask_visits -= 1 }
        let redactor = Int8DocumentRedactor::new(&p, identity(&p), c).unwrap();
        assert!(redactor.preflight(source, &mut Continue).is_err(), "{axis}");
    }
}
#[test]
fn early_eos_does_not_refund_original_reservations_to_verification() {
    let work = config().mapping.max_model_work; let mut limits = config().mapping; limits.max_mask_visits = 1000;
    let stage = LongRedactionPreflight { source_bytes: 6, source_scalars: 4, chunks: 1,
        planned_model_work: work, reserved_mask_node_visits: 1000 };
    let mut accounting = Accounting::default(); accounting.reserve(stage, limits).unwrap();
    accounting.finish(&LongRedactionStage { preflight: stage, model_work: Int8Work::default(), mask_node_visit_charge: 0 }).unwrap();
    assert_eq!(accounting.reserved, work); assert_eq!(accounting.masks_reserved, 1000);
    assert_eq!(subtract(limits.max_model_work, accounting.reserved).unwrap(), Int8Work::default());
    assert!(accounting.reserve(stage, limits).is_err());
    for axis in 0..5 {
        let mut above = work;
        match axis { 0 => above.forward_positions += 1, 1 => above.projected_logits += 1, 2 => above.attention_pairs += 1,
            3 => above.projections.dot_products += 1, _ => above.projections.multiply_accumulates += 1 }
        assert!(subtract(work, above).is_err());
    }
}
#[test]
fn cancellation_empty_input_and_wrong_identity_are_refused_without_inference() {
    struct Stop;
    impl DecodeStepControl for Stop { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::Deadline) } }
    let p = planner(); let redactor = Int8DocumentRedactor::new(&p, identity(&p), config()).unwrap();
    assert_eq!(redactor.preflight("Alice", &mut Stop).unwrap_err().cancellation(), Some(DecodeCancellationKind::Deadline));
    assert!(redactor.preflight("", &mut Continue).is_err());
    let mut wrong = identity(&p); wrong.task_spec = "summarize-v1".to_owned();
    assert!(Int8DocumentRedactor::new(&p, wrong, config()).is_err());
    let mut wrong = identity(&p); wrong.numerics_profile = NumericsProfile::HfBf16Eager;
    assert!(Int8DocumentRedactor::new(&p, wrong, config()).is_err());
}


fn compact_policy() -> crate::tokenizer::untrusted::UntrustedEncoding {
    use crate::tokenizer::untrusted::{SafeBpeLimits, UntrustedEncoding};
    UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 4096, max_output_tokens: 4096,
        max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24 })
}
fn compact_work() -> crate::tokenizer::untrusted::SafeBpeWorkLimits {
    crate::tokenizer::untrusted::SafeBpeWorkLimits { max_input_bytes: 65_536, max_output_tokens: 65_536,
        max_pair_checks: 1 << 24, max_pair_bytes: 1 << 28 }
}
fn compact_config() -> LongRedactionConfig {
    let mut c = config(); c.mapping.chunks.max_chunk_bytes = 64; c.mapping.chunks.max_chunk_tokens = 8; c
}
const COMPACT_SOURCE: &str = "The quick brown fox jumps over the lazy dog. 上海 <think> <|im_start|> ▁";

#[test]
fn compact_documents_require_cumulative_authority_and_exact_encoding_identity() {
    let one_call = planner().with_document_encoding(compact_policy()).unwrap();
    assert!(Int8DocumentRedactor::new(&one_call, identity(&one_call), compact_config()).is_err());
    let p = planner().with_map_document_encoding(compact_policy(), compact_work()).unwrap();
    assert!(p.map_document_encoding_workspace_bytes().unwrap() > 0);
    Int8DocumentRedactor::new(&p, identity(&p), compact_config()).unwrap();
    let mut wrong = identity(&p); wrong.template_digest = *planner().template_digest();
    assert!(Int8DocumentRedactor::new(&p, wrong, compact_config()).is_err());
}

#[test]
fn compact_preflight_uses_real_tokens_and_retains_complete_source_geometry() {
    let mut c = compact_config(); c.mapping.chunks.max_chunk_bytes = 4096; c.mapping.chunks.max_chunk_tokens = 4096;
    let byte = planner(); let old = Int8DocumentRedactor::new(&byte, identity(&byte), c.clone()).unwrap()
        .preflight(COMPACT_SOURCE, &mut Continue).unwrap();
    let p = planner().with_map_document_encoding(compact_policy(), compact_work()).unwrap();
    let r = Int8DocumentRedactor::new(&p, identity(&p), c.clone()).unwrap();
    let actual = r.prepare(COMPACT_SOURCE, c.mapping, &mut Continue).unwrap();
    let sized = metadata(COMPACT_SOURCE, &actual);
    assert_eq!(sized.chunks, 1); assert_eq!(sized.chunks, old.chunks);
    assert_eq!(sized.source_bytes, old.source_bytes); assert_eq!(sized.source_scalars, old.source_scalars);
    assert_eq!(actual.preflight_metadata().source_span(), extent(COMPACT_SOURCE, 0, COMPACT_SOURCE.len()));
    assert!(sized.planned_model_work.forward_positions < old.planned_model_work.forward_positions);
    assert_eq!(actual.execution_identities().next().unwrap().template_digest, *p.template_digest());
    let used = p.map_document_encoding_work().unwrap().unwrap();
    assert_eq!(used.input_bytes(), 2 * COMPACT_SOURCE.len(), "accepted probe plus actual prompt");
    assert!(used.output_tokens() < used.input_bytes()); assert!(used.pair_checks() > 0);
    assert!(byte.map_document_encoding_work().unwrap().is_none());
}

#[test]
fn compact_preflight_original_and_fresh_expanded_edit_share_all_four_work_ceilings() {
    let source = format!("é Alice\r\n{COMPACT_SOURCE} Alice");
    let mut request = RedactionRequest::default(); request.actions.include_map = true;
    let detected = collect(&source, &[(&source, &["Alice"])], &request, &mut GroundingBudget::default()).unwrap();
    let edited = actions::apply(&detected, &request.actions, None, request.edit_budget).unwrap();
    assert!(edited.text().len() > source.len()); assert!(!edited.text().contains("Alice"));
    let spans = scan_occurrences(&source, "Alice", &mut GroundingBudget::default()).unwrap();
    assert_eq!(edited.edits().len(), spans.len());
    for (edit, span) in edited.edits().iter().zip(spans) { assert_eq!(edit.original.span, span); }

    let probe = planner().with_map_document_encoding(compact_policy(), compact_work()).unwrap();
    let c = compact_config(); let r = Int8DocumentRedactor::new(&probe, identity(&probe), c.clone()).unwrap();
    let first = r.preflight(&source, &mut Continue).unwrap();
    let after_preflight = probe.map_document_encoding_work().unwrap().unwrap();
    assert!(first.chunks > 1);
    assert!(after_preflight.input_bytes() > 2 * source.len(), "rejected chunk-fit probes stay charged");
    let original = r.prepare(&source, c.mapping, &mut Continue).unwrap();
    assert_eq!(metadata(&source, &original), first); drop(original);
    let after_original = probe.map_document_encoding_work().unwrap().unwrap();
    assert_eq!(after_original, after_preflight.checked_add(after_preflight).unwrap());
    let verification = r.prepare(edited.text(), c.mapping, &mut Continue).unwrap();
    assert_eq!(verification.preflight_metadata().source_span(), extent(edited.text(), 0, edited.text().len()));
    let after_verification = probe.map_document_encoding_work().unwrap().unwrap();
    assert!(after_verification.input_bytes() > after_original.input_bytes());
    assert!(after_verification.output_tokens() > after_original.output_tokens());
    assert!(after_verification.pair_checks() > after_original.pair_checks());
    assert!(after_verification.pair_bytes() > after_original.pair_bytes());

    for axis in 0..4 {
        let mut cap = compact_work();
        match axis { 0 => cap.max_input_bytes = after_original.input_bytes() as u64,
            1 => cap.max_output_tokens = after_original.output_tokens() as u64,
            2 => cap.max_pair_checks = after_original.pair_checks(),
            _ => cap.max_pair_bytes = after_original.pair_bytes() }
        let p = planner().with_map_document_encoding(compact_policy(), cap).unwrap();
        let r = Int8DocumentRedactor::new(&p, identity(&p), c.clone()).unwrap();
        r.preflight(&source, &mut Continue).unwrap();
        drop(r.prepare(&source, c.mapping, &mut Continue).unwrap());
        assert_eq!(p.map_document_encoding_work().unwrap().unwrap(), after_original);
        assert!(r.prepare(edited.text(), c.mapping, &mut Continue).is_err(), "{axis}");
        assert_eq!(p.map_document_encoding_work().unwrap().unwrap(), after_original,
            "fresh verification cannot renew an exhausted operation axis");
    }
}

#[test]
fn compact_result_policy_binds_call_and_operation_limits_without_changing_byte_policy() {
    let previous = Sha256Digest::of_bytes(b"public-action-policy");
    let request = RedactionRequest::default(); let c = compact_config();
    let byte = planner(); let r = Int8DocumentRedactor::new(&byte, identity(&byte), c.clone()).unwrap();
    let established = Sha256Digest::of_bytes(&canonjson::canonical_bytes(&(
        LONG_REDACTION_EXECUTION, CHUNK_PROFILE, previous, &request, &c.ner, c.per_chunk, c.mapping.chunks,
    )).unwrap());
    assert_eq!(r.policy_digest(previous, &request).unwrap(), established);
    let p = planner().with_map_document_encoding(compact_policy(), compact_work()).unwrap();
    let compact = Int8DocumentRedactor::new(&p, identity(&p), c.clone()).unwrap()
        .policy_digest(previous, &request).unwrap();
    assert_ne!(compact, established);
    for axis in 0..5 {
        let mut work = compact_work(); let mut policy = compact_policy();
        match axis {
            0 => work.max_input_bytes -= 1, 1 => work.max_output_tokens -= 1,
            2 => work.max_pair_checks -= 1, 3 => work.max_pair_bytes -= 1,
            _ => { let crate::tokenizer::untrusted::UntrustedEncoding::SafeBpeV1(ref mut call) = policy
                else { unreachable!() }; call.max_output_tokens -= 1; }
        }
        let next = planner().with_map_document_encoding(policy, work).unwrap();
        let r = Int8DocumentRedactor::new(&next, identity(&next), c.clone()).unwrap();
        assert_ne!(r.policy_digest(previous, &request).unwrap(), compact, "{axis}");
        assert_ne!(next.template_digest(), p.template_digest());
    }
}

#[test]
fn compact_edited_stage_cancellation_preserves_cause_and_prior_work() {
    struct Stop(DecodeCancellationKind);
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(self.0) }
    }
    let p = planner().with_map_document_encoding(compact_policy(), compact_work()).unwrap();
    let c = compact_config(); let r = Int8DocumentRedactor::new(&p, identity(&p), c.clone()).unwrap();
    r.preflight("Alice", &mut Continue).unwrap();
    drop(r.prepare("Alice", c.mapping, &mut Continue).unwrap());
    let used = p.map_document_encoding_work().unwrap();
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let error = r.prepare("[redacted:person]", c.mapping, &mut Stop(cause)).err().unwrap();
        assert_eq!(error.cancellation(), Some(cause));
        assert_eq!(p.map_document_encoding_work().unwrap(), used);
    }
}
