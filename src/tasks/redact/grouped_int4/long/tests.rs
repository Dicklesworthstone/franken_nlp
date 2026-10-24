//! Pinned map preparation and shared destructive-offset/edit semantics.
//! Native stage receipts are private scripted fixtures, not inference evidence.
use super::*;
use crate::{
    artifact::converter::PINNED_REVISION,
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    grammar::mask::MaskWorkLimits,
    native_engine::{generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::int4::convert::GROUPED_INT4_RECIPE, strict_int4::STRICT_INT4_EXECUTION},
    tasks::{extract::{ExtractionGrounding, SourceDocumentEncoder}, ir::ScoreSpace,
        mapreduce::{ChunkLimits, ChunkPlan, ExecutionLimits, MapReduceError},
        ner::{EntityType, NamedEntity, NerResult},
        redact::{PiiKind, Detector, actions::{ActionPolicy, RedactionAction}, pseudonym::PseudonymKey}},
    tokenizer::pinned_controls,
    validation::grounded_fields::{SourceOccurrence, VerifiedSourceSpan, scan_occurrences},
};
struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
fn planner() -> SourceTaskPlanner {
    let registry = pinned_controls::pinned().unwrap();
    let controls = registry.template_controls();
    let eos = controls.entries().iter().find(|e| e.special && e.surface == crate::template::IM_END).unwrap().id;
    SourceTaskPlanner::pinned(controls, eos).unwrap()
}
fn identity(p: &SourceTaskPlanner, width: usize) -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"private-long-int4-redaction-fixture");
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
fn config() -> Int4LongRedactionConfig {
    let mut unlimited = Int4Work::default();
    unlimited.forward_positions = u64::MAX; unlimited.projected_logits = u64::MAX;
    unlimited.attention_pairs = u64::MAX; unlimited.projections.dot_products = u64::MAX;
    unlimited.projections.multiply_accumulates = u64::MAX;
    Int4LongRedactionConfig {
        ner: NerOptions { types: vec![EntityType::Person], ..NerOptions::default() },
        per_chunk: TaskBudget { max_input_tokens: 8192, max_output_tokens: 64, max_output_bytes: 1 << 20,
            max_grammar_states: 4096, max_kv_bytes: 1 << 31 },
        planning: SourcePlanningLimits::default(),
        mapping: Int4SourceMapLimits {
            chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 6, max_chunk_tokens: 6,
                context_tokens: 8192, reserved_tokens: 64, max_chunks: 256, max_tokenizer_calls: 1024 },
            reduction: ExecutionLimits::default(), max_model_work: unlimited,
            mask_limits: MaskWorkLimits::default(), mask_visits_per_chunk: 1000,
            max_mask_visits: 256_000, verification: GroundingBudget::default(),
        },
        max_result_bytes: 4 << 20,
    }
}
fn redactor(p: &SourceTaskPlanner, width: usize) -> Int4DocumentRedactor<'_> {
    Int4DocumentRedactor::from_bound_parts(p, identity(p, width), config(), width).unwrap()
}
fn raw(source: &str, names: &[&str]) -> NerResult {
    NerResult {
        schema_version: 1, task_spec_version: NER_TASK_VERSION.to_owned(), numerics_profile: STRICT_INT4_PROFILE.to_owned(),
        score_space: ScoreSpace::NotComputed, grounding: ExtractionGrounding::SourceMembership,
        entities: names.iter().filter(|name| source.contains(**name)).map(|&name| {
            let spans = scan_occurrences(source, name, &mut GroundingBudget::default()).unwrap();
            NamedEntity { text: name.to_owned(), entity_type: EntityType::Person,
                occurrence: if spans.len() == 1 { SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous }, spans }
        }).collect(),
        generated_token_ids: Vec::new(), forward_positions: 0, projected_logits: 0, mask_node_visit_charge: 0,
    }
}
fn extent(source: &str, start: usize, end: usize) -> VerifiedSourceSpan {
    VerifiedSourceSpan { byte_start: start, byte_end: end,
        scalar_start: source[..start].chars().count(), scalar_end: source[..end].chars().count() }
}
fn collect<'s>(source: &'s str, parts: &[(&str, &[&str])], request: &RedactionRequest,
    grounding: &mut GroundingBudget) -> Result<DetectedDocument<'s>, RedactError> {
    let mut detections = detection::Detections::with_profile(source, request, &config().ner, NerProfile::Int4)?;
    let mut offset = 0;
    for (index, (text, names)) in parts.iter().enumerate() {
        detections.push(index, extent(source, offset, offset + text.len()), &raw(text, names), grounding)?;
        offset += text.len();
    }
    detections.finish(parts.len())
}
fn exact_chunks<'s>(recipe: &Int4DocumentRedactor<'_>, source: &'s str, mapping: Int4SourceMapLimits) -> ChunkPlan<'s> {
    let context = PlanContext::new(&recipe.identity, recipe.config.per_chunk).unwrap();
    let capacity = recipe.planner.int4_map_capacity_with_control(&SourceMapTask::Ner(recipe.config.ner.clone()),
        recipe.config.per_chunk, &context, recipe.config.planning, recipe.group_width, &mut Continue).unwrap();
    let limits = capacity.constrain_chunks(mapping.chunks).unwrap();
    let registry = pinned_controls::pinned().unwrap();
    let encoder = SourceDocumentEncoder::pinned(registry.template_controls()).unwrap();
    ChunkPlan::build(source, limits, |text| encoder.encode(text, limits.max_chunk_bytes, limits.max_chunk_bytes)
        .map(|doc| doc.total_token_count()).map_err(|_| MapReduceError::Tokenizer)).unwrap()
}
struct FixtureDetection { seen: Vec<String>, names: Vec<&'static str>, residual: Option<&'static str>,
    corrupt_chunk: Option<usize>, cancel_after: Option<usize> }
impl FixtureDetection {
    fn new() -> Self { Self { seen: Vec::new(), names: vec!["Alice"], residual: None, corrupt_chunk: None, cancel_after: None } }
}
impl DetectionDriver for FixtureDetection {
    fn checkpoint(&mut self) -> Result<(), Int4LongRedactionError> {
        if self.cancel_after.is_some_and(|n| self.seen.len() >= n) {
            Err(Int4RedactionError::Cancelled(DecodeCancellationKind::Deadline).into())
        } else { Ok(()) }
    }
    fn detect<'s>(&mut self, recipe: &Int4DocumentRedactor<'_>, source: &'s str,
        request: &RedactionRequest, accounting: &mut Accounting, grounding: &mut GroundingBudget)
        -> Result<(DetectedDocument<'s>, Int4LongRedactionStage), Int4LongRedactionError> {
        let mapping = accounting.begin(recipe.config.mapping)?;
        let prepared = recipe.prepare(source, mapping, &mut Continue)?;
        let preflight = metadata(prepared.preflight_metadata());
        accounting.reserve(preflight, recipe.config.mapping)?;
        let chunks = exact_chunks(recipe, source, mapping);
        assert_eq!(preflight.chunks, chunks.chunks().len());
        let first = self.seen.is_empty(); self.seen.push(source.to_owned());
        let names = if first { self.names.clone() } else { self.residual.into_iter().collect() };
        let mut map_left = mapping.verification;
        let before = *grounding;
        let mut detections = detection::Detections::with_profile(source, request, &recipe.config.ner, NerProfile::Int4)?;
        for chunk in chunks.chunks() {
            let mut result = raw(chunk.text(), &names);
            // Model-free emulation of the map's separate occurrence-verifier
            // budget. Production consumes its sealed verify_completed proof.
            for entity in &result.entities {
                map_left.max_fields = map_left.max_fields.checked_sub(1).ok_or(Int4SourceMapError::WorkLimit)?;
                scan_occurrences(chunk.text(), &entity.text, &mut map_left).map_err(|_| Int4SourceMapError::WorkLimit)?;
            }
            if self.corrupt_chunk == Some(chunk.id()) { result.numerics_profile = "strict-quantized-v1".to_owned(); }
            detections.push(chunk.id(), chunk.span(), &result, grounding)?;
        }
        let document = detections.finish(preflight.chunks)?;
        let stage = Int4LongRedactionStage {
            preflight, model_work: preflight.planned_model_work, mask_node_visit_charge: preflight.chunks as u64 * 17,
            map_verification_used: verification_used(mapping.verification, map_left)?,
            detection_verification_used: verification_used(before, *grounding)?,
        };
        accounting.finish(&stage, recipe.config.mapping)?;
        Ok((document, stage))
    }
}
fn synthetic_stage(preflight: Int4LongRedactionPreflight) -> Int4LongRedactionStage {
    Int4LongRedactionStage { preflight, model_work: Int4Work::default(), mask_node_visit_charge: 0,
        map_verification_used: ZERO_VERIFICATION, detection_verification_used: ZERO_VERIFICATION }
}
fn one_stage(work: Int4Work) -> Int4LongRedactionPreflight {
    Int4LongRedactionPreflight { source_bytes: 5, source_scalars: 5, chunks: 1, group_width: 127,
        planned_model_work: work, reserved_mask_node_visits: 1000 }
}

#[test]
fn native_profile_actual_group_and_pinned_assets_cannot_be_substituted() {
    let p = planner();
    for axis in 0..8 {
        let mut id = identity(&p, 127);
        match axis {
            0 => id.numerics_profile = NumericsProfile::StrictQuantized { version: 1 },
            1 => id.task_spec = "redact-v1".to_owned(),
            2 => id.tokenizer_digest = Sha256Digest::of_bytes(b"other"),
            3 => id.template_digest = Sha256Digest::of_bytes(b"other"),
            4 => id.quant_recipe = format!("{GROUPED_INT4_RECIPE}:group=32"),
            5 => id.packing_set_digest = Sha256Digest::of_bytes(b"other"),
            6 => id.tool_mode = ToolMode::Json,
            _ => id.thinking_mode = ThinkingMode::Enabled,
        }
        assert!(Int4DocumentRedactor::from_bound_parts(&p, id, config(), 127).is_err());
    }
    assert!(Int4DocumentRedactor::from_bound_parts(&p, identity(&p, 127), config(), 32).is_err());
}
#[test]
fn pinned_preflight_covers_the_complete_source_and_preserves_partial_group_geometry() {
    let p = planner(); let source = "éAéAéAéA"; let mut work = Vec::new();
    for width in [32, 127] {
        let r = redactor(&p, width); let preflight = r.preflight(source, &mut Continue).unwrap();
        let prepared = r.prepare(source, r.config.mapping, &mut Continue).unwrap();
        assert_eq!(preflight.source_bytes, source.len()); assert_eq!(preflight.source_scalars, 8);
        assert_eq!(preflight.chunks, 2); assert_eq!(preflight.group_width, width); assert_eq!(r.group_width(), width);
        assert_eq!(preflight.planned_model_work, prepared.planned_work()); assert_eq!(preflight.reserved_mask_node_visits, 2000);
        for id in prepared.execution_identities() {
            assert_eq!(id.task_spec, NER_TASK_VERSION);
            assert_eq!(id.numerics_profile, NumericsProfile::StrictQuantized { version: 2 });
            assert_eq!(id.quant_recipe, identity(&p, width).quant_recipe);
        }
        work.push(preflight.planned_model_work);
    }
    assert_eq!(work[0].projections.multiply_accumulates, work[1].projections.multiply_accumulates);
    assert!(work[0].projections.dot_products > work[1].projections.dot_products);
}
#[test]
fn whole_source_rules_see_sensitive_values_crossing_neural_chunk_boundaries() {
    let source = "a@example.org"; let req = RedactionRequest::default();
    let document = collect(source, &[("a@exam", &[]), ("ple.org", &[])], &req, &mut GroundingBudget::default()).unwrap();
    assert_eq!(document.regions().len(), 1); assert_eq!(document.regions()[0].span.byte_end, source.len());
    assert!(document.regions()[0].kinds.contains(&PiiKind::Email));
    let out = actions::apply(&document, &req.actions, None, req.edit_budget).unwrap();
    assert_eq!(out.text(), "[redacted:email]");
}
#[test]
fn repeated_unicode_occurrences_are_lifted_before_one_transactional_edit() {
    let source = "éAéA éAéA"; let mut req = RedactionRequest::default();
    req.actions = ActionPolicy { default_action: RedactionAction::Mask, include_map: true, ..ActionPolicy::default() };
    let document = collect(source, &[("éAéA ", &["é"]), ("éAéA", &["é"])], &req, &mut GroundingBudget::default()).unwrap();
    assert_eq!(document.regions().len(), 4); assert_eq!(document.regions()[2].span.byte_start, 7);
    assert_eq!(document.regions()[2].span.scalar_start, 5);
    let result = actions::apply(&document, &req.actions, None, req.edit_budget).unwrap();
    assert_eq!(result.text(), "*A*A *A*A"); assert_eq!(result.edits().len(), 4);
}
#[test]
fn connected_rule_and_model_overlap_keeps_both_detector_provenances() {
    let req = RedactionRequest::default(); let source = "a@example.org tail";
    let doc = collect(source, &[(source, &["example.org tail"])], &req, &mut GroundingBudget::default()).unwrap();
    assert_eq!(doc.regions().len(), 1); assert_eq!(doc.regions()[0].span.byte_end, source.len());
    assert!(doc.regions()[0].detectors.contains(&Detector::EmailAsciiV1));
    assert!(doc.regions()[0].detectors.contains(&Detector::NerSourceV1));
    let out = actions::apply(&doc, &req.actions, None, req.edit_budget).unwrap();
    assert_eq!(out.text(), "[redacted:overlap]");
}
#[test]
fn transformed_source_is_freshly_partitioned_and_all_stage_receipts_are_retained() {
    let p = planner(); let r = redactor(&p, 127); let mut driver = FixtureDetection::new();
    let out = r.run_pipeline("Alice", &RedactionRequest::default(), None, &mut driver).unwrap();
    assert_eq!(driver.seen, vec!["Alice".to_owned(), "[redacted:person]".to_owned()]);
    assert_eq!(out.original.preflight.chunks, 1);
    assert!(out.verification.as_ref().unwrap().preflight.chunks > out.original.preflight.chunks);
    assert_eq!(out.result.verification(), VerificationStatus::CleanDeclaredUnion);
    assert_eq!(out.group_width, 127); assert_eq!(out.execution, INT4_LONG_REDACTION_EXECUTION);
    assert_eq!(out.numerics_profile, STRICT_INT4_PROFILE); assert_eq!(out.detector_scope, DETECTOR_SCOPE);
    let second = out.verification.as_ref().unwrap();
    assert_eq!(out.reserved_model_work, out.original.preflight.planned_model_work.checked_add(second.preflight.planned_model_work).unwrap());
    assert_eq!(out.model_work, out.original.model_work.checked_add(second.model_work).unwrap());
    assert_eq!(out.map_verification_used, add_verification(out.original.map_verification_used, second.map_verification_used).unwrap());
    assert_eq!(out.detection_verification_used,
        add_verification(out.original.detection_verification_used, second.detection_verification_used).unwrap());
    assert_eq!(out.verification_scan_steps, out.detection_verification_used.scan_steps);
    let wire = canonjson::canonical_string(&out).unwrap();
    for private in ["Alice", "prompt_digest", "generated_token_ids", "original_spans", "logical_model_digest"] {
        assert!(!wire.contains(private));
    }
}
#[test]
fn one_full256_context_is_reused_with_real_edit_and_fresh_verification() {
    let p = planner(); let r = redactor(&p, 127);
    let key = PseudonymKey::from_bytes(&[9; 32], "fixture-key").unwrap();
    let context = Pseudonyms::full256(&key, "private-namespace", None).unwrap();
    let mut req = RedactionRequest::default(); req.rules.enabled.clear();
    req.actions.default_action = RedactionAction::Pseudonymize;
    req.actions.expected_key_commitment = Some(key.commitment());
    let mut driver = FixtureDetection::new();
    let out = r.run_pipeline("Alice Alice", &req, Some(&context), &mut driver).unwrap();
    let expected = context.pseudonym(PiiKind::Person, "Alice").unwrap();
    assert_eq!(expected.len(), "person".len() + 3 + 64);
    assert_eq!(out.result.text(), format!("{expected} {expected}"));
    assert_eq!(driver.seen[1], out.result.text()); assert!(out.verification.is_some());
    let wire = canonjson::canonical_string(&out).unwrap();
    assert!(!wire.contains("private-namespace")); assert!(!wire.contains("Alice"));
}
#[test]
fn missing_or_mismatched_keys_refuse_before_any_native_stage() {
    let p = planner(); let r = redactor(&p, 127); let mut driver = FixtureDetection::new();
    let mut req = RedactionRequest::default(); req.actions.default_action = RedactionAction::Pseudonymize;
    assert!(matches!(r.run_pipeline("Alice", &req, None, &mut driver),
        Err(Int4LongRedactionError::Redaction(Int4RedactionError::Redaction(RedactError::MissingKey)))));
    assert!(driver.seen.is_empty());
    let key = PseudonymKey::from_bytes(&[9; 32], "fixture-key").unwrap();
    let context = Pseudonyms::full256(&key, "scope", None).unwrap();
    req.actions.expected_key_commitment = Some("wrong".to_owned());
    assert!(r.run_pipeline("Alice", &req, Some(&context), &mut driver).is_err());
    assert!(driver.seen.is_empty());
}
#[test]
fn residual_verification_returns_coordinates_without_partially_edited_text() {
    let p = planner(); let r = redactor(&p, 127); let mut driver = FixtureDetection::new(); driver.residual = Some("red");
    let error = r.run_pipeline("Alice", &RedactionRequest::default(), None, &mut driver).err().unwrap();
    let Int4LongRedactionError::Redaction(Int4RedactionError::Residual(report)) = &error else { panic!("residual") };
    assert_eq!(driver.seen.len(), 2); assert_eq!(report.residuals.len(), 1);
    assert_eq!(report.residuals[0].span.byte_start, 1); assert_eq!(report.residuals[0].span.byte_end, 4);
    let wire = canonjson::canonical_string(report).unwrap();
    assert!(!wire.contains("Alice")); assert!(!wire.contains("redacted:person"));
    assert!(!format!("{error:?} {error}").contains("redacted:person"));
}
#[test]
fn no_verify_does_not_fabricate_a_second_stage_or_clean_status() {
    let p = planner(); let r = redactor(&p, 127); let mut driver = FixtureDetection::new();
    let req = RedactionRequest { verify: false, ..RedactionRequest::default() };
    let out = r.run_pipeline("Alice", &req, None, &mut driver).unwrap();
    assert!(out.verification.is_none()); assert_eq!(out.result.verification(), VerificationStatus::NotRequested);
    assert_eq!(out.reserved_model_work, out.original.preflight.planned_model_work);
    assert_eq!(out.map_verification_used, out.original.map_verification_used);
    assert_eq!(driver.seen.len(), 1);
}
#[test]
fn expanded_transformed_source_must_fit_fresh_byte_and_chunk_limits() {
    let p = planner();
    for byte_limit in [true, false] {
        let mut c = config();
        if byte_limit { c.mapping.chunks.max_input_bytes = 8; } else { c.mapping.chunks.max_chunks = 1; }
        let r = Int4DocumentRedactor::from_bound_parts(&p, identity(&p, 127), c, 127).unwrap();
        let mut driver = FixtureDetection::new();
        assert!(r.run_pipeline("Alice", &RedactionRequest::default(), None, &mut driver).is_err());
        assert_eq!(driver.seen.len(), 1);
    }
}
#[test]
fn all_five_original_work_axes_and_masks_fail_during_real_pinned_preflight() {
    let p = planner(); let r = redactor(&p, 127); let source = "éAéAéAéA";
    let exact = r.preflight(source, &mut Continue).unwrap();
    for axis in 0..6 {
        let mut c = config(); c.mapping.max_model_work = exact.planned_model_work;
        c.mapping.max_mask_visits = exact.reserved_mask_node_visits;
        match axis {
            0 => c.mapping.max_model_work.forward_positions -= 1,
            1 => c.mapping.max_model_work.projected_logits -= 1,
            2 => c.mapping.max_model_work.attention_pairs -= 1,
            3 => c.mapping.max_model_work.projections.dot_products -= 1,
            4 => c.mapping.max_model_work.projections.multiply_accumulates -= 1,
            _ => c.mapping.max_mask_visits -= 1,
        }
        let r = Int4DocumentRedactor::from_bound_parts(&p, identity(&p, 127), c, 127).unwrap();
        assert!(r.preflight(source, &mut Continue).is_err());
    }
}
#[test]
fn reserved_work_and_masks_never_refill_after_early_native_completion() {
    let work = Int4Work::for_sequence(127, 0, 8, 100).unwrap(); let preflight = one_stage(work);
    for axis in 0..6 {
        let mut limits = config().mapping; limits.max_model_work = work.checked_add(work).unwrap();
        limits.max_mask_visits = 2000;
        match axis {
            0 => limits.max_model_work.forward_positions -= 1, 1 => limits.max_model_work.projected_logits -= 1,
            2 => limits.max_model_work.attention_pairs -= 1, 3 => limits.max_model_work.projections.dot_products -= 1,
            4 => limits.max_model_work.projections.multiply_accumulates -= 1, _ => limits.max_mask_visits -= 1,
        }
        let mut accounting = Accounting::default(); accounting.begin(limits).unwrap(); accounting.reserve(preflight, limits).unwrap();
        accounting.finish(&synthetic_stage(preflight), limits).unwrap();
        assert_eq!(accounting.reserved, work); assert_eq!(accounting.masks_reserved, 1000);
        let result = accounting.begin(limits).and_then(|_| accounting.reserve(preflight, limits));
        assert!(matches!(result, Err(Int4LongRedactionError::Map(Int4SourceMapError::WorkLimit))));
        assert!(accounting.failed); assert_eq!(accounting.reserved, work); assert_eq!(accounting.masks_reserved, 1000);
        assert!(accounting.begin(limits).is_err());
    }
}
#[test]
fn per_stage_actuals_cannot_borrow_unused_native_or_mask_allowances() {
    let work = Int4Work::for_sequence(127, 0, 8, 100).unwrap(); let preflight = one_stage(work);
    for axis in 0..7 {
        let limits = config().mapping; let mut accounting = Accounting::default();
        accounting.begin(limits).unwrap(); accounting.reserve(preflight, limits).unwrap();
        let mut stage = synthetic_stage(preflight); stage.model_work = work;
        match axis {
            0 => stage.model_work.forward_positions += 1, 1 => stage.model_work.projected_logits += 1,
            2 => stage.model_work.attention_pairs += 1, 3 => stage.model_work.projections.dot_products += 1,
            4 => stage.model_work.projections.multiply_accumulates += 1,
            5 => stage.mask_node_visit_charge = 1001, _ => stage.preflight.group_width = 32,
        }
        assert!(accounting.finish(&stage, limits).is_err()); assert!(accounting.failed);
        assert_eq!(accounting.completed, 0); assert_eq!(accounting.actual, Int4Work::default());
    }
}
#[test]
fn every_exactly_exhausted_allowance_is_a_work_error_for_the_next_stage() {
    let work = Int4Work::for_sequence(127, 0, 8, 100).unwrap(); let preflight = one_stage(work);
    for axis in 0..9 {
        let mut limits = config().mapping;
        match axis {
            0 => limits.max_model_work.forward_positions = work.forward_positions,
            1 => limits.max_model_work.projected_logits = work.projected_logits,
            2 => limits.max_model_work.attention_pairs = work.attention_pairs,
            3 => limits.max_model_work.projections.dot_products = work.projections.dot_products,
            4 => limits.max_model_work.projections.multiply_accumulates = work.projections.multiply_accumulates,
            5 => limits.max_mask_visits = 1000,
            6 => limits.verification.max_fields = 1,
            7 => limits.verification.max_matches = 1,
            _ => limits.verification.max_scan_steps = 1,
        }
        let mut accounting = Accounting::default(); accounting.begin(limits).unwrap(); accounting.reserve(preflight, limits).unwrap();
        let mut stage = synthetic_stage(preflight);
        stage.map_verification_used = SourceMapVerification { fields: 1, matches: 1, scan_steps: 1 };
        accounting.finish(&stage, limits).unwrap();
        assert!(matches!(accounting.begin(limits), Err(Int4LongRedactionError::Map(Int4SourceMapError::WorkLimit))));
        assert!(accounting.failed);
    }
}
#[test]
fn map_and_destructive_offset_grounding_pools_are_independently_nonrenewable() {
    let p = planner();
    let mut c = config(); c.mapping.verification.max_fields = 1;
    let r = Int4DocumentRedactor::from_bound_parts(&p, identity(&p, 127), c, 127).unwrap();
    let mut driver = FixtureDetection::new();
    assert!(matches!(r.run_pipeline("Alice", &RedactionRequest::default(), None, &mut driver),
        Err(Int4LongRedactionError::Map(Int4SourceMapError::WorkLimit))));
    assert_eq!(driver.seen.len(), 1);
    let r = redactor(&p, 127); let mut req = RedactionRequest::default(); req.grounding_budget.max_fields = 1;
    let mut driver = FixtureDetection::new(); driver.residual = Some("red");
    assert!(matches!(r.run_pipeline("Alice", &req, None, &mut driver),
        Err(Int4LongRedactionError::Redaction(Int4RedactionError::Redaction(RedactError::DetectionBudget)))));
    assert_eq!(driver.seen.len(), 2);
}
#[test]
fn forged_final_chunk_cannot_authorize_a_partial_edit() {
    let p = planner(); let r = redactor(&p, 127); let mut driver = FixtureDetection::new(); driver.corrupt_chunk = Some(1);
    assert!(r.run_pipeline("Alice Alice", &RedactionRequest::default(), None, &mut driver).is_err());
    assert_eq!(driver.seen, vec!["Alice Alice".to_owned()]);
}
#[test]
fn shared_detector_rejects_wrong_profile_missing_occurrences_and_unicode_offset_forgery() {
    let req = RedactionRequest::default(); let source = "éAéA";
    for axis in 0..5 {
        let mut detector = detection::Detections::with_profile(source, &req, &config().ner, NerProfile::Int4).unwrap();
        let mut result = raw(source, &["é"]);
        match axis {
            0 => result.numerics_profile = "strict-quantized-v1".to_owned(),
            1 => { result.entities[0].spans.pop(); }
            2 => result.entities[0].spans[0].scalar_end += 1,
            3 => result.entities[0].occurrence = SourceOccurrence::Anchored,
            _ => result.entities[0].entity_type = EntityType::Event,
        }
        assert!(detector.push(0, extent(source, 0, source.len()), &result, &mut GroundingBudget::default()).is_err());
        assert!(detector.finish(1).is_err());
    }
    let mut int8 = detection::Detections::new(source, &req, &config().ner).unwrap();
    assert!(int8.push(0, extent(source, 0, source.len()), &raw(source, &["é"]), &mut GroundingBudget::default()).is_err());
}
#[test]
fn complete_coverage_rejects_missing_reordered_overlapping_and_invalid_scalar_chunks() {
    let req = RedactionRequest::default(); let source = "Alice Bob";
    for axis in 0..5 {
        let mut detector = detection::Detections::with_profile(source, &req, &config().ner, NerProfile::Int4).unwrap();
        let mut span = extent(source, 0, 6); let mut id = 0;
        match axis { 0 => span.byte_start = 1, 1 => span.byte_end = 0, 2 => span.scalar_start = 1,
            3 => span.scalar_end += 1, _ => id = 1 }
        assert!(detector.push(id, span, &raw("Alice ", &[]), &mut GroundingBudget::default()).is_err());
        assert!(detector.finish(1).is_err());
    }
    let mut detector = detection::Detections::with_profile(source, &req, &config().ner, NerProfile::Int4).unwrap();
    detector.push(0, extent(source, 0, 6), &raw("Alice ", &[]), &mut GroundingBudget::default()).unwrap();
    assert!(detector.finish(1).is_err());
}
#[test]
fn cancellation_before_between_and_after_stages_never_publishes_success() {
    let p = planner(); let r = redactor(&p, 127);
    for at in 0..=2 {
        let mut driver = FixtureDetection::new(); driver.cancel_after = Some(at);
        let error = r.run_pipeline("Alice", &RedactionRequest::default(), None, &mut driver).err().unwrap();
        assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline)); assert_eq!(driver.seen.len(), at);
    }
    struct Stop;
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::Deadline) }
    }
    assert_eq!(r.preflight("Alice", &mut Stop).unwrap_err().cancellation(), Some(DecodeCancellationKind::Deadline));
    assert!(r.preflight("", &mut Continue).is_err());
}
#[test]
fn the_complete_document_receipt_is_checked_against_the_result_limit() {
    let p = planner(); let r = redactor(&p, 127);
    let out = r.run_pipeline("Alice", &RedactionRequest::default(), None, &mut FixtureDetection::new()).unwrap();
    let bytes = canonjson::canonical_bytes(&out).unwrap().len() as u64;
    let mut c = config(); c.max_result_bytes = bytes;
    let r = Int4DocumentRedactor::from_bound_parts(&p, identity(&p, 127), c, 127).unwrap();
    let mut req = RedactionRequest::default(); req.edit_budget.max_output_bytes = bytes as usize;
    assert!(r.run_pipeline("Alice", &req, None, &mut FixtureDetection::new()).is_ok());
    let mut c = r.config.clone(); c.max_result_bytes -= 1;
    let r = Int4DocumentRedactor::from_bound_parts(&p, identity(&p, 127), c, 127).unwrap();
    req.edit_budget.max_output_bytes -= 1;
    assert!(r.run_pipeline("Alice", &req, None, &mut FixtureDetection::new()).is_err());
}
