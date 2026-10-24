//! Pinned preflight, independent grounding and private synthetic native results.
//! These fixtures exercise composition contracts without model-quality claims.
use super::*;
use std::sync::OnceLock;
use crate::{
    artifact::converter::PINNED_REVISION,
    corpus::{native_resolve::{NativeResolveLimits, grouped_int4::INT4_RESOLUTION_EXECUTION},
        resolve::{BidirectionalScores, PairLogProbabilities}},
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    grammar::mask::MaskWorkLimits,
    native_engine::{generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::int4::convert::GROUPED_INT4_RECIPE,
        strict_int4::{STRICT_INT4_EXECUTION, scoring::INT4_SCORING_EXECUTION}},
    tasks::{extract::ExtractionGrounding, ir::{DependencyScope, ScoreSpace}, ner::{EntityType, NamedEntity}},
    tokenizer::pinned_controls,
    validation::{SourceSpan, validate_source_span, grounded_fields::{SourceOccurrence, scan_occurrences}},
};

struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn source() -> &'static SourceTaskPlanner {
    static PINNED: OnceLock<SourceTaskPlanner> = OnceLock::new();
    PINNED.get_or_init(|| {
        let registry = pinned_controls::pinned().unwrap();
        let eos = registry.template_controls().entries().iter()
            .find(|entry| entry.special && entry.surface == crate::template::IM_END).unwrap().id;
        SourceTaskPlanner::pinned(registry.template_controls(), eos).unwrap()
    })
}
fn identity(width: usize) -> ExecutionIdentity {
    let p = source(); let d = Sha256Digest::of_bytes(b"private-int4-entities-fixture");
    ExecutionIdentity { schema_version: 1, source_revision: PINNED_REVISION.to_owned(), logical_model_digest: d,
        artifact_format: INT4_SOURCE_FORMAT.to_owned(), quant_recipe: format!("{GROUPED_INT4_RECIPE}:group={width}"),
        packing_set_digest: Sha256Digest::of_bytes(&canonjson::canonical_bytes(&(
            INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", d, width,
        )).unwrap()), tokenizer_digest: p.tokenizer_digest(), template_digest: *p.template_digest(),
        task_spec: NER_TASK_VERSION.to_owned(), taskir_digest: d, prompt_digest: d,
        grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None }
}
fn unlimited() -> Int4Work {
    Int4Work { forward_positions: u64::MAX, projected_logits: u64::MAX, attention_pairs: u64::MAX,
        projections: ProjectionWork { dot_products: u64::MAX, multiply_accumulates: u64::MAX } }
}
fn config() -> Int4EntityConfig {
    Int4EntityConfig {
        ner: NerOptions::default(),
        ner_budget: TaskBudget { max_input_tokens: 8192, max_output_tokens: 64, max_output_bytes: 1 << 20,
            max_grammar_states: 4096, max_kv_bytes: 2 * 1024 * 1024 * 1024 },
        source_planning: SourcePlanningLimits::default(),
        masks: SourceMaskBudget { per_mask: MaskWorkLimits::default(),
            max_visits_per_item: 1000, max_visits_per_run: 1_000_000 },
        resolution: ResolveOptions::default(), graph: ResolveLimits::default(),
        scoring: Int4ResolveLimits { planning: NativeResolveLimits::default(), max_model_work: unlimited() },
        verification: GroundingBudget::default(), max_model_work: unlimited(), max_result_bytes: 16 << 20,
        chunks: None, max_snapshot_chunks: 4096,
    }
}
fn chunked() -> Int4EntityConfig {
    let mut c = config();
    c.chunks = Some(ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 16, max_chunk_tokens: 16,
        context_tokens: 8192, reserved_tokens: 64, max_chunks: 256, max_tokenizer_calls: 4096 });
    c
}
fn documents(items: &[(&str, &str)]) -> Vec<EntityDocument> {
    items.iter().map(|&(id, text)| EntityDocument { id: id.to_owned(), text: text.to_owned() }).collect()
}
fn prepared(items: &[(&str, &str)], c: &Int4EntityConfig, width: usize) -> CompiledInputs {
    compile_inputs(documents(items), source(), &identity(width), c, width, &mut Continue).unwrap()
}
fn mention(text: &str, quote: &str) -> NamedEntity {
    let spans = scan_occurrences(text, quote, &mut GroundingBudget::default()).unwrap();
    assert!(!spans.is_empty());
    NamedEntity { text: quote.to_owned(), entity_type: EntityType::Person,
        occurrence: if spans.len() == 1 { SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous }, spans }
}
fn ner(entities: Vec<NamedEntity>) -> NerResult {
    NerResult { schema_version: 1, task_spec_version: NER_TASK_VERSION.to_owned(), numerics_profile: STRICT_INT4_PROFILE.to_owned(),
        score_space: ScoreSpace::NotComputed, grounding: ExtractionGrounding::SourceMembership,
        entities, generated_token_ids: vec![1, 166_101], forward_positions: 0, projected_logits: 0, mask_node_visit_charge: 20 }
}
fn synthetic(text: &str, chunk: &ChunkWitness, width: usize, quote: Option<&str>) -> Int4SourceTaskRun {
    let mut result = ner(quote.map(|quote| vec![mention(text, quote)]).unwrap_or_default());
    let work = constrained_int4::planned_work(width, chunk.prompt_tokens, result.generated_token_ids.len()).unwrap();
    result.forward_positions = work.forward_positions; result.projected_logits = work.projected_logits;
    Int4SourceTaskRun { schema_version: 1, execution: INT4_SOURCE_EXECUTION.to_owned(), group_width: width,
        result: SourceTaskResult::Ner(result), model_work: work }
}
fn maps(compiled: CompiledInputs, c: &Int4EntityConfig, width: usize, quote: Option<&str>) -> Extracted {
    collect(compiled.inputs, c, width, &mut Continue, |text, chunk, _| {
        Ok(synthetic(text, chunk, width, quote.filter(|quote| text.contains(*quote))))
    }).unwrap()
}
// Synthetic numerical boundary only. The real source-validated graph finalizer
// still decides all pairs and clusters; this is not a native inference receipt.
fn resolution(maps: &Extracted, c: &Int4EntityConfig, width: usize) -> Int4ResolutionRun {
    let graph = ResolutionPlan::prepare(&maps.documents, c.resolution, c.graph, &mut Continue).unwrap();
    let same = PairLogProbabilities { same: -1.0, different: -4.0, uncertain: -4.0 };
    let scores = graph.pairs().map(|ticket| ticket.finish(BidirectionalScores { forward: same, reverse: same })).collect();
    let heads = graph.candidate_count() * 2;
    let work = if heads == 0 { Int4Work::default() } else {
        Int4Work::for_sequence(width, 0, heads, heads * 10).unwrap()
    };
    Int4ResolutionRun { schema_version: 1, execution: INT4_RESOLUTION_EXECUTION,
        numerics_profile: STRICT_INT4_PROFILE, scoring_execution: INT4_SCORING_EXECUTION, group_width: width,
        model_evaluated: heads != 0, head_count: heads, planned_model_work: work, model_work: work,
        rewound_positions: heads as u64 * 2, result: graph.finalize(scores, &mut Continue).unwrap() }
}
fn completed(items: &[(&str, &str)], c: &Int4EntityConfig, width: usize) -> Int4EntityRun {
    let p = prepared(items, c, width); let reserved = p.ner_work; let masks_reserved = p.mask_visits; let chunks = p.chunks;
    let maps = maps(p, c, width, Some("Smith")); let resolved = resolution(&maps, c, width);
    finish(maps, resolved, reserved, masks_reserved, chunks, c, width, &mut Continue).unwrap()
}

#[test]
fn every_document_and_chunk_is_prepared_before_any_native_execution() {
    let c = chunked(); let p = prepared(&[("b", "Smith Smith Smith Smith"), ("a", "é\r\nSmith 上海😀 Smith")], &c, 32);
    assert_eq!(p.inputs[0].document.id, "a"); assert_eq!(p.inputs[1].document.id, "b");
    assert!(p.chunks > p.inputs.len());
    let mut work = Int4Work::default(); let mut count = 0;
    for input in &p.inputs {
        let (mut byte, mut scalar) = (0, 0);
        for chunk in &input.chunks {
            let text = chunk_text(&input.document.text, chunk.span, byte, scalar).unwrap();
            let plan = source_plan(text, source(), &identity(32), &c, 32, &mut Continue).unwrap();
            check_rebuilt(chunk, &plan).unwrap();
            work = plus(work, chunk.work).unwrap(); count += 1;
            byte = chunk.span.byte_end; scalar = chunk.span.scalar_end;
        }
        assert_eq!(byte, input.document.text.len()); assert_eq!(scalar, input.document.text.chars().count());
    }
    assert_eq!(count, p.chunks); assert_eq!(work, p.ner_work);
    assert_eq!(p.mask_visits, p.chunks as u64 * c.masks.max_visits_per_item);
}

#[test]
fn chunking_allows_original_documents_beyond_each_source_chunk_limit() {
    let mut c = chunked(); c.source_planning.max_input_bytes = 16;
    let text = "Smith Smith Smith Smith Smith Smith";
    let p = prepared(&[("original", text)], &c, 32);
    assert!(p.chunks > 1); assert_eq!(p.inputs[0].document.text, text);
    c.chunks = None;
    assert!(matches!(compile_inputs(documents(&[("original", text)]), source(), &identity(32), &c, 32, &mut Continue),
        Err(Int4EntityError::WorkBudget)));
}

#[test]
fn actual_group_geometry_changes_ner_work_without_retokenizing_sources() {
    let c = chunked(); let items = [("doc", "Smith Smith Smith Smith")];
    let a = prepared(&items, &c, 16); let b = prepared(&items, &c, 32);
    assert_eq!(a.chunks, b.chunks); assert_eq!(a.ner_work.forward_positions, b.ner_work.forward_positions);
    assert_eq!(a.ner_work.projected_logits, b.ner_work.projected_logits);
    assert_eq!(a.ner_work.attention_pairs, b.ner_work.attention_pairs);
    assert_eq!(a.ner_work.projections.multiply_accumulates, b.ner_work.projections.multiply_accumulates);
    assert_eq!(a.ner_work.projections.dot_products, b.ner_work.projections.dot_products * 2);
    assert_ne!(a.inputs[0].chunks[0].identity, b.inputs[0].chunks[0].identity);
}

#[test]
fn five_aggregate_ner_work_axes_are_checked_during_preflight() {
    let items = [("a", "Smith"), ("b", "Smith")]; let base = config();
    let exact = prepared(&items, &base, 32).ner_work;
    for axis in 0..5 {
        let mut c = base.clone(); c.max_model_work = exact;
        match axis { 0 => c.max_model_work.forward_positions -= 1, 1 => c.max_model_work.projected_logits -= 1,
            2 => c.max_model_work.attention_pairs -= 1, 3 => c.max_model_work.projections.dot_products -= 1,
            _ => c.max_model_work.projections.multiply_accumulates -= 1 }
        assert!(matches!(compile_inputs(documents(&items), source(), &identity(32), &c, 32, &mut Continue),
            Err(Int4EntityError::WorkBudget)), "axis {axis}");
    }
}

#[test]
fn early_eos_never_refunds_reserved_ner_allowance_into_pair_scoring() {
    let mut c = config(); let p = prepared(&[("a", "Smith"), ("b", "Smith")], &c, 32); let reserved = p.ner_work;
    let maps = maps(p, &c, 32, Some("Smith"));
    assert!(maps.work.forward_positions < reserved.forward_positions);
    assert!(maps.work.projected_logits < reserved.projected_logits);
    c.max_model_work = reserved;
    assert_eq!(remaining_scoring(&c, reserved).unwrap().max_model_work, Int4Work::default());
    assert_ne!(remaining_scoring(&c, maps.work).unwrap().max_model_work, Int4Work::default());
}

#[test]
fn pair_stage_receives_only_the_meet_of_stage_cap_and_global_remainder() {
    let mut c = config(); let reserved = Int4Work::for_sequence(32, 0, 20, 100).unwrap();
    let remainder = Int4Work::for_sequence(32, 0, 8, 40).unwrap();
    c.max_model_work = plus(reserved, remainder).unwrap(); c.scoring.max_model_work = remainder;
    c.scoring.max_model_work.projected_logits -= 1;
    c.scoring.planning.max_pairs = 100; c.graph.max_candidate_pairs = 2;
    let pair = remaining_scoring(&c, reserved).unwrap();
    assert_eq!(pair.max_model_work, c.scoring.max_model_work); assert_eq!(pair.planning.max_pairs, 2);
    c.max_model_work.attention_pairs = reserved.attention_pairs - 1;
    assert!(matches!(remaining_scoring(&c, reserved), Err(Int4EntityError::WorkBudget)));
}

#[test]
fn repeated_surfaces_expand_all_independently_verified_occurrences() {
    let c = config(); let p = prepared(&[("original", "Smith Smith")], &c, 32);
    let maps = maps(p, &c, 32, Some("Smith"));
    assert_eq!(maps.documents[0].mentions.len(), 2);
    assert_eq!(maps.documents[0].mentions[0].span.byte_start, 0);
    assert_eq!(maps.documents[0].mentions[1].span.byte_start, 6);
    assert_eq!(maps.receipts[0].proposed_entities, 1); assert_eq!(maps.receipts[0].anchored_mentions, 2);
    assert_eq!(maps.verification_used.fields, 1); assert_eq!(maps.verification_used.matches, 2);
}

#[test]
fn duplicate_proposals_are_verified_before_deduplication() {
    let c = config(); let text = "Smith Smith";
    let entity = mention(text, "Smith");
    let mut verification = c.verification;
    let values = grounding::expand(text, ner(vec![entity.clone(), entity.clone()]), &c,
        &mut verification, &mut Continue).unwrap();
    assert_eq!(values.len(), 2);
    let mut corrupt = entity.clone(); corrupt.spans.pop();
    let mut verification = c.verification;
    assert!(grounding::expand(text, ner(vec![entity, corrupt]), &c,
        &mut verification, &mut Continue).is_err());
}

#[test]
fn fabricated_missing_reordered_and_mislabelled_occurrences_are_refused() {
    let c = config(); let text = "Smith Smith";
    for axis in 0..6 {
        let mut entity = mention(text, "Smith");
        match axis { 0 => { entity.spans.pop(); }, 1 => entity.spans.reverse(),
            2 => entity.spans[0].byte_end += 1, 3 => entity.spans[0].scalar_end += 1,
            4 => entity.occurrence = SourceOccurrence::Anchored, _ => entity.text = "Absent".to_owned() }
        let mut verification = c.verification;
        assert!(matches!(grounding::expand(text, ner(vec![entity]), &c, &mut verification, &mut Continue),
            Err(Int4EntityError::Graph(ResolveError::InvalidAnchor))), "axis {axis}");
    }
}

#[test]
fn grounding_fields_matches_and_scans_are_nonrenewable_across_documents() {
    let items = [("a", "Smith"), ("b", "Smith")];
    for axis in 0..3 {
        let mut c = config(); let p = prepared(&items, &c, 32);
        let first = maps(prepared(&items[..1], &c, 32), &c, 32, Some("Smith")).verification_used;
        match axis { 0 => c.verification.max_fields = first.fields,
            1 => c.verification.max_matches = first.matches, _ => c.verification.max_scan_steps = first.scan_steps }
        let mut calls = 0;
        let result = collect(p.inputs, &c, 32, &mut Continue, |text, chunk, _| {
            calls += 1; Ok(synthetic(text, chunk, 32, Some("Smith")))
        });
        assert!(result.is_err(), "axis {axis}"); assert_eq!(calls, 2);
    }
}

#[test]
fn masks_are_reserved_for_every_chunk_before_inference() {
    let items = [("a", "Smith Smith Smith Smith")]; let mut c = chunked();
    let p = prepared(&items, &c, 32); assert!(p.chunks > 1);
    c.masks.max_visits_per_run = p.mask_visits - 1;
    assert!(matches!(compile_inputs(documents(&items), source(), &identity(32), &c, 32, &mut Continue),
        Err(Int4EntityError::WorkBudget)));
}

#[test]
fn graph_expansion_cannot_exceed_global_mention_or_byte_limits() {
    let items = [("a", "Smith Smith"), ("b", "Smith")];
    for mention_cap in [true, false] {
        let mut c = config(); let p = prepared(&items, &c, 32);
        if mention_cap { c.graph.max_mentions = 2; }
        else { c.graph.max_input_bytes = items.iter().map(|(id, text)| id.len() + text.len()).sum(); }
        assert!(matches!(collect(p.inputs, &c, 32, &mut Continue, |text, chunk, _|
            Ok(synthetic(text, chunk, 32, Some("Smith")))), Err(Int4EntityError::WorkBudget)));
    }
}

#[test]
fn invalid_last_document_is_refused_before_the_first_native_call() {
    for items in [vec![("a", "Smith"), ("a", "Smith")], vec![("a", "Smith"), ("bad\n", "Smith")],
        vec![("a", "Smith"), ("b", "")]] {
        assert!(matches!(compile_inputs(documents(&items), source(), &identity(32), &config(), 32, &mut Continue),
            Err(Int4EntityError::InvalidInput)));
    }
}

#[test]
fn exact_scaffold_and_snapshot_chunk_limits_refuse_without_truncation() {
    let items = [("a", "Smith Smith Smith Smith Smith")];
    for axis in 0..3 {
        let mut c = chunked();
        match axis { 0 => c.max_snapshot_chunks = 1,
            1 => c.chunks.as_mut().unwrap().max_chunks = 1,
            _ => { c.chunks.as_mut().unwrap().context_tokens = 65; c.chunks.as_mut().unwrap().reserved_tokens = 64; } }
        assert!(compile_inputs(documents(&items), source(), &identity(32), &c, 32, &mut Continue).is_err());
    }
}

#[test]
fn reconstructed_chunk_witness_cannot_change_source_policy_or_geometry() {
    let c = config(); let p = prepared(&[("a", "Smith")], &c, 32); let chunk = &p.inputs[0].chunks[0];
    let plan = source_plan("Smith", source(), &identity(32), &c, 32, &mut Continue).unwrap();
    check_rebuilt(chunk, &plan).unwrap();
    let changed = source_plan("Other", source(), &identity(32), &c, 32, &mut Continue).unwrap();
    assert!(matches!(check_rebuilt(chunk, &changed), Err(Int4EntityError::Accounting)));
    let mut other = c.clone(); other.ner.max_entities -= 1;
    let changed = source_plan("Smith", source(), &identity(32), &other, 32, &mut Continue).unwrap();
    assert!(check_rebuilt(chunk, &changed).is_err());
    let changed = source_plan("Smith", source(), &identity(16), &c, 16, &mut Continue).unwrap();
    assert!(check_rebuilt(chunk, &changed).is_err());
}

#[test]
fn unicode_chunk_coordinates_lift_to_original_document_and_id() {
    let mut c = chunked(); c.chunks.as_mut().unwrap().max_chunk_bytes = 10;
    let items = [("original", "é\r\n上海 😀 上海")]; let p = prepared(&items, &c, 32);
    let maps = collect(p.inputs, &c, 32, &mut Continue, |text, chunk, _| {
        let quote = text.chars().find(|ch| !ch.is_whitespace()).unwrap().to_string();
        Ok(synthetic(text, chunk, 32, Some(&quote)))
    }).unwrap();
    assert_eq!(maps.documents.len(), 1); assert_eq!(maps.documents[0].id, "original");
    let doc = &maps.documents[0];
    for mention in &doc.mentions {
        let s = mention.span;
        validate_source_span(&doc.text, &mention.surface, SourceSpan::new(s.byte_start, s.byte_end, s.scalar_start, s.scalar_end)).unwrap();
    }
    assert!(maps.receipts[0].chunks.len() > 1);
    assert_eq!(maps.receipts[0].chunks.last().unwrap().source_span.byte_end, items[0].1.len());
}

#[test]
fn chunks_cannot_skip_repeat_or_corrupt_original_extents() {
    for axis in 0..4 {
        let c = chunked(); let mut p = prepared(&[("a", "Smith Smith Smith Smith")], &c, 32);
        assert!(p.inputs[0].chunks.len() > 1);
        match axis { 0 => p.inputs[0].chunks[1].span.byte_start += 1,
            1 => p.inputs[0].chunks[1].span.scalar_start += 1,
            2 => p.inputs[0].chunks[1].span.scalar_end += 1,
            _ => { p.inputs[0].chunks.pop(); } }
        assert!(matches!(collect(p.inputs, &c, 32, &mut Continue, |text, chunk, _|
            Ok(synthetic(text, chunk, 32, None))), Err(Int4EntityError::Accounting)));
    }
}

#[test]
fn all_ner_receipt_metadata_and_work_axes_are_checked() {
    let c = config(); let items = [("a", "Smith"), ("b", "Smith")];
    for axis in 0..17 {
        let p = prepared(&items, &c, 32); let mut calls = 0;
        let result = collect(p.inputs, &c, 32, &mut Continue, |text, chunk, _| {
            calls += 1; let mut run = synthetic(text, chunk, 32, Some("Smith"));
            match axis {
                0 => run.schema_version = 2, 1 => run.execution = "other".to_owned(), 2 => run.group_width = 16,
                3 => run.model_work.forward_positions += 1, 4 => run.model_work.projected_logits += 1,
                5 => run.model_work.attention_pairs += 1, 6 => run.model_work.projections.dot_products += 1,
                7 => run.model_work.projections.multiply_accumulates += 1,
                _ => { let SourceTaskResult::Ner(ner) = &mut run.result else { panic!("NER"); };
                    match axis { 8 => ner.schema_version = 2, 9 => ner.task_spec_version = "other".to_owned(),
                        10 => ner.numerics_profile = "strict-quantized-v1".to_owned(),
                        11 => ner.grounding = ExtractionGrounding::NotRequested,
                        12 => ner.forward_positions += 1, 13 => ner.projected_logits += 1,
                        14 => ner.mask_node_visit_charge = c.masks.max_visits_per_item + 1,
                        15 => ner.generated_token_ids.clear(), _ => ner.entities[0].entity_type = EntityType::Money }
                }
            }
            Ok(run)
        });
        assert!(result.is_err(), "axis {axis}"); assert_eq!(calls, 1);
    }
}

#[test]
fn stage_identity_normalizes_only_task_fields_and_keeps_every_model_fact() {
    let a = identity(32); let mut b = a.clone();
    b.task_spec = "resolve-v1".to_owned(); b.template_digest = Sha256Digest::of_bytes(b"resolve-template");
    b.taskir_digest = Sha256Digest::of_bytes(b"different-task"); same_model(&a, &b).unwrap();
    for axis in 0..9 {
        let mut changed = b.clone(); let digest = Sha256Digest::of_bytes(b"changed");
        match axis { 0 => changed.logical_model_digest = digest, 1 => changed.packing_set_digest = digest,
            2 => changed.source_revision = "other".to_owned(), 3 => changed.quant_recipe = "other".to_owned(),
            4 => changed.tokenizer_digest = digest, 5 => changed.artifact_format = "other".to_owned(),
            6 => changed.backend_semantic_version = "other".to_owned(),
            7 => changed.numerics_profile = NumericsProfile::StrictQuantized { version: 1 },
            _ => changed.kv_dtype = "int8".to_owned() }
        assert!(matches!(same_model(&a, &changed), Err(Int4EntityError::Identity)), "axis {axis}");
    }
}

#[test]
fn canonical_input_order_survives_discovery_and_final_clustering() {
    let c = config();
    let a = completed(&[("b", "Smith"), ("a", "Smith Smith")], &c, 32);
    let b = completed(&[("a", "Smith Smith"), ("b", "Smith")], &c, 32);
    assert_eq!(canonjson::canonical_bytes(&a).unwrap(), canonjson::canonical_bytes(&b).unwrap());
    assert_eq!(a.documents[0].document_id, "a");
    assert_eq!(a.resolution.result.dependency_scope, DependencyScope::CorpusGlobal);
    assert_eq!(a.resolution.result.clusters.len(), 1);
}

#[test]
fn complete_output_includes_chunk_lineage_and_honest_work_and_quality_status() {
    let c = chunked(); let run = completed(&[("original", "Smith Smith Smith Smith")], &c, 32);
    assert_eq!(run.group_width, 32); assert_eq!(run.chunk_profile, Some(CHUNK_PROFILE));
    assert_eq!(run.chunk_count, run.documents[0].chunks.len());
    assert_eq!(run.documents[0].document_id, "original");
    assert_eq!(run.reserved_model_work, plus(run.ner_reserved_work, run.resolution.planned_model_work).unwrap());
    assert_eq!(run.model_work, plus(run.ner_model_work, run.resolution.model_work).unwrap());
    assert_eq!(run.resolution.result.calibration, "uncalibrated");
    let public = canonjson::canonical_string(&run).unwrap();
    for private in ["taskir_digest", "prompt_digest", "logical_model_digest", "generated_token_ids", "source_root_sha256"] {
        assert!(!public.contains(private));
    }
}

#[test]
fn no_entities_still_reports_ner_work_and_preserves_all_original_documents() {
    let c = config(); let p = prepared(&[("a", "Smith"), ("b", "Smith")], &c, 16);
    let reserved = p.ner_work; let masks_reserved = p.mask_visits; let chunks = p.chunks;
    let maps = maps(p, &c, 16, None); let resolved = resolution(&maps, &c, 16);
    let run = finish(maps, resolved, reserved, masks_reserved, chunks, &c, 16, &mut Continue).unwrap();
    assert_eq!(run.documents.len(), 2); assert!(run.resolution.result.mentions.is_empty());
    assert!(!run.resolution.model_evaluated); assert_eq!(run.resolution.head_count, 0);
    assert!(run.ner_model_work.forward_positions > 0); assert_eq!(run.model_work, run.ner_model_work);
}

#[test]
fn empty_snapshot_has_zero_ner_work_but_still_checks_source_configuration() {
    let c = config(); let p = prepared(&[], &c, 32);
    assert_eq!(p.chunks, 0); assert_eq!(p.ner_work, Int4Work::default()); assert_eq!(p.mask_visits, 0);
    let mut wrong = identity(32); wrong.numerics_profile = NumericsProfile::StrictQuantized { version: 1 };
    assert!(compile_inputs(Vec::new(), source(), &wrong, &c, 32, &mut Continue).is_err());
    let mut bad = c.clone(); bad.ner.types.clear();
    assert!(compile_inputs(Vec::new(), source(), &identity(32), &bad, 32, &mut Continue).is_err());
}

#[test]
fn output_budget_covers_the_whole_entity_and_resolution_envelope() {
    let items = [("a", "Smith"), ("b", "Smith")]; let mut c = config();
    let bytes = canonjson::canonical_bytes(&completed(&items, &c, 32)).unwrap().len();
    c.max_result_bytes = bytes;
    assert_eq!(canonjson::canonical_bytes(&completed(&items, &c, 32)).unwrap().len(), bytes);
    c.max_result_bytes -= 1;
    let p = prepared(&items, &c, 32); let reserved = p.ner_work; let masks_reserved = p.mask_visits; let chunks = p.chunks;
    let maps = maps(p, &c, 32, Some("Smith")); let resolved = resolution(&maps, &c, 32);
    assert!(matches!(finish(maps, resolved, reserved, masks_reserved, chunks, &c, 32, &mut Continue),
        Err(Int4EntityError::Graph(ResolveError::OutputBudget))));
}

#[test]
fn finish_rejects_missing_chunk_wrong_group_or_overbudget_result() {
    for axis in 0..3 {
        let mut c = config(); let p = prepared(&[("a", "Smith")], &c, 32);
        let reserved = p.ner_work; let masks_reserved = p.mask_visits; let mut chunks = p.chunks;
        let maps = maps(p, &c, 32, Some("Smith")); let mut resolved = resolution(&maps, &c, 32);
        match axis { 0 => chunks += 1, 1 => resolved.group_width = 16, _ => c.max_model_work = Int4Work::default() }
        assert!(matches!(finish(maps, resolved, reserved, masks_reserved, chunks, &c, 32, &mut Continue),
            Err(Int4EntityError::Accounting)));
    }
}

#[test]
fn actual_string_and_witness_capacities_are_retained_in_the_memory_quote() {
    let c = chunked(); let mut docs = documents(&[("a", "Smith Smith Smith Smith")]);
    docs[0].text.reserve(8192); docs[0].id.reserve(1024);
    let p = compile_inputs(docs, source(), &identity(32), &c, 32, &mut Continue).unwrap();
    let input = &p.inputs[0];
    let expected = (p.inputs.capacity() * size_of::<Input>() + input.document.id.capacity()
        + input.document.text.capacity() + input.chunks.capacity() * size_of::<ChunkWitness>()) as u64;
    assert_eq!(retained_input_bytes(&p).unwrap(), expected);
    assert!(expected > p.input_bytes as u64);
}

#[test]
fn cancellation_after_one_chunk_prevents_every_later_chunk_and_pair() {
    struct Control(bool);
    impl DecodeStepControl for Control {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { self.0.then_some(DecodeCancellationKind::Deadline) }
    }
    let c = chunked(); let p = prepared(&[("a", "Smith Smith Smith Smith")], &c, 32); let mut calls = 0;
    let result = collect(p.inputs, &c, 32, &mut Control(false), |text, chunk, control| {
        calls += 1; control.0 = true; Ok(synthetic(text, chunk, 32, None))
    });
    assert_eq!(calls, 1); assert_eq!(result.err().unwrap().cancellation(), Some(DecodeCancellationKind::Deadline));
}

#[test]
fn cancellation_during_partition_preserves_its_typed_cause() {
    struct Stop;
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::PollQuota) }
    }
    let c = chunked();
    let error = partition("Smith Smith", source(), c.chunks.unwrap(), &mut Stop).err().unwrap();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::PollQuota));
}

#[test]
fn cancellation_and_private_error_text_survive_every_stage() {
    for cause in [DecodeCancellationKind::User, DecodeCancellationKind::Timeout,
        DecodeCancellationKind::Deadline, DecodeCancellationKind::PollQuota,
        DecodeCancellationKind::CostBudget, DecodeCancellationKind::FailFast,
        DecodeCancellationKind::RaceLost, DecodeCancellationKind::ParentCancelled,
        DecodeCancellationKind::ResourceUnavailable, DecodeCancellationKind::Shutdown,
        DecodeCancellationKind::LinkedExit] {
        for error in [Int4EntityError::Graph(ResolveError::Cancelled(cause)),
            Int4EntityError::Source(Int4SourceError::Cancelled(cause)),
            Int4EntityError::Resolution(Int4ResolveError::Native(StrictInt4Error::Cancelled(cause))),
            Int4EntityError::Native(StrictInt4Error::Cancelled(cause))] {
            assert_eq!(error.cancellation(), Some(cause)); assert_eq!(format!("{error}"), format!("{error:?}"));
        }
    }
}

#[test]
fn mask_and_grounding_configuration_axes_are_bounded_even_without_input() {
    for axis in 0..8 {
        let mut c = config();
        match axis { 0 => c.masks.max_visits_per_item = 0, 1 => c.masks.per_mask.max_trie_node_visits = 0,
            2 => c.masks.per_mask.checkpoint_interval_nodes = 0, 3 => c.verification.max_fields = 0,
            4 => c.verification.max_matches = 0, 5 => c.verification.max_scan_steps = 0,
            6 => c.max_snapshot_chunks = 0, _ => c.max_result_bytes = 0 }
        assert!(matches!(c.validate(), Err(Int4EntityError::InvalidLimits)), "axis {axis}");
    }
}
