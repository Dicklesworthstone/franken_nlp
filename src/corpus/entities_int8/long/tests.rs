//! Pinned planning and private fault injection only; no fixture is native or
//! quality evidence. Synthetic NER cannot enter the public API.
use super::*;
use crate::{execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    native_engine::strict_int8::STRICT_INT8_EXECUTION,
    tasks::{extract::ExtractionGrounding, ir::ScoreSpace, ner::{EntityType, NamedEntity}},
    tokenizer::pinned_controls,
    validation::grounded_fields::{SourceOccurrence, scan_occurrences}};
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
struct Stop;
impl DecodeStepControl for Stop {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::Deadline) }
}
fn config() -> Int8DocumentEntityConfig {
    let cap = Int8Work::for_sequence(0, 16_384, 16_384 * 166_144).unwrap();
    Int8DocumentEntityConfig { entities: Int8EntityConfig {
        ner: NerOptions::default(), ner_budget: TaskBudget { max_input_tokens: 8192, max_output_tokens: 64,
            max_output_bytes: 1 << 20, max_grammar_states: 100_000, max_kv_bytes: 2 * 1024 * 1024 * 1024 },
        source_planning: SourcePlanningLimits::default(), masks: SourceMaskBudget { per_mask: Default::default(),
            max_visits_per_item: 1_000_000_000, max_visits_per_run: 1_000_000_000_000 },
        resolution: ResolveOptions::default(), graph: ResolveLimits::default(),
        scoring: Int8ResolveLimits { planning: Default::default(), max_model_work: cap },
        verification: GroundingBudget::default(), max_model_work: cap, max_result_bytes: 16 << 20 },
        chunks: ChunkLimits { max_input_bytes: 1 << 20, max_chunk_bytes: 6, max_chunk_tokens: 6,
            context_tokens: 8192, reserved_tokens: 64, max_chunks: 64, max_tokenizer_calls: 1024 },
        max_snapshot_chunks: 128 }
}
fn doc(id: &str, text: &str) -> EntityDocument { EntityDocument { id: id.to_owned(), text: text.to_owned() } }
fn planners() -> (Arc<SourceTaskPlanner>, ExecutionIdentity, Arc<ResolutionPlanner>, ExecutionIdentity) {
    let registry = pinned_controls::pinned().unwrap(); let controls = registry.template_controls();
    let eos = controls.entries().iter().find(|e| e.special && e.surface == crate::template::IM_END).unwrap().id;
    let source = Arc::new(SourceTaskPlanner::pinned(controls, eos).unwrap());
    let resolver = Arc::new(ResolutionPlanner::pinned(controls, eos).unwrap());
    let d = Sha256Digest::of_bytes(b"document-entity-model-free-fixture");
    let a = ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "fixture-int8".to_owned(), packing_set_digest: d,
        tokenizer_digest: source.tokenizer_digest(), template_digest: *source.template_digest(), task_spec: NER_TASK_VERSION.to_owned(),
        taskir_digest: d, prompt_digest: d, grammar_compiler_version: "fixture".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None };
    let mut b = a.clone(); b.task_spec = resolve::RESOLVE_VERSION.to_owned(); b.template_digest = resolver.template_digest();
    (source, a, resolver, b)
}
fn prepare(documents: Vec<EntityDocument>, config: Int8DocumentEntityConfig) -> Result<PreparedInt8DocumentEntityCorpus, Int8EntityError> {
    let (p, a, r, b) = planners(); prepare_int8_document_entities(documents, p, a, r, b, config, &mut Continue)
}
fn input(id: &str, parts: &[&str]) -> DocumentInput {
    let mut text = String::new(); let mut chunks = Vec::new(); let mut scalar = 0;
    for part in parts {
        let start = text.len(); text.push_str(part); let next = scalar + part.chars().count();
        chunks.push(ChunkWitness { span: VerifiedSourceSpan { byte_start: start, byte_end: text.len(),
            scalar_start: scalar, scalar_end: next }, identity: Sha256Digest::of_bytes(b"private-fixture"), prompt_tokens: 10,
            work: constrained_int8::planned_work(10, 64).unwrap() });
        scalar = next;
    }
    DocumentInput { document: doc(id, &text), chunks }
}
fn run(text: &str, surface: Option<&str>) -> Int8SourceTaskRun {
    let entities = surface.into_iter().map(|surface| {
        let spans = scan_occurrences(text, surface, &mut GroundingBudget::default()).unwrap();
        NamedEntity { text: surface.to_owned(), entity_type: EntityType::Person,
            occurrence: if spans.len() == 1 { SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous }, spans }
    }).collect();
    let work = constrained_int8::planned_work(10, 3).unwrap();
    Int8SourceTaskRun { schema_version: 1, execution: INT8_SOURCE_EXECUTION.to_owned(), model_work: work,
        result: SourceTaskResult::Ner(NerResult { schema_version: 1, task_spec_version: NER_TASK_VERSION.to_owned(),
            numerics_profile: STRICT_INT8_PROFILE.to_owned(), score_space: ScoreSpace::NotComputed,
            grounding: ExtractionGrounding::SourceMembership, entities, generated_token_ids: vec![1, 2, 0],
            forward_positions: work.forward_positions, projected_logits: work.projected_logits, mask_node_visit_charge: 10 }) }
}
#[test]
fn document_that_exceeds_one_context_preflights_as_real_pinned_chunks() {
    let mut c = config(); c.entities.source_planning.max_context_tokens = 1024; c.entities.ner_budget.max_input_tokens = 1024;
    c.chunks.context_tokens = 1024; c.chunks.max_chunk_bytes = 1024; c.chunks.max_chunk_tokens = 128;
    let text = " Alice".repeat(2048); let (p, a, r, b) = planners();
    assert!(source_plan(&text, &p, &a, &c.entities, &mut Continue).is_err());
    let prepared = prepare_int8_document_entities(vec![doc("original", &text)], p, a, r, b, c, &mut Continue).unwrap();
    assert_eq!(prepared.document_count(), 1); assert!(prepared.chunk_count() > 1);
    assert!(prepared.required_ner_context_tokens() <= 1024);
    assert_eq!(prepared.reserved_mask_visits(), prepared.chunk_count() as u64 * 1_000_000_000);
    assert!(prepared.retained_input_bytes().unwrap() > text.len() as u64);
    let work = prepared.inputs[0].chunks.iter().fold(Int8Work::default(), |sum, c| plus(sum, c.work).unwrap());
    assert_eq!(work, prepared.ner_reserved_work());
}
#[test]
fn snapshot_order_is_canonical_and_every_rebuilt_chunk_keeps_its_identity() {
    let x = prepare(vec![doc("z", "Bob"), doc("a", "é Alice")], config()).unwrap();
    let y = prepare(vec![doc("a", "é Alice"), doc("z", "Bob")], config()).unwrap();
    assert_eq!(x.inputs[0].document.id, "a"); assert_eq!(x.ner_reserved_work(), y.ner_reserved_work());
    for (left, right) in x.inputs.iter().zip(&y.inputs) {
        for (l, r) in left.chunks.iter().zip(&right.chunks) { assert_eq!(l.span, r.span); assert_eq!(l.identity, r.identity); }
    }
    let d = &x.inputs[0]; let original = ResolutionDocument { id: d.document.id.clone(), text: d.document.text.clone(), mentions: Vec::new() };
    let mut local = local_input(&original, &d.chunks[0], 0, 0).unwrap();
    let plan = source_plan(&local.document.text, &x.source, &x.source_identity, &x.config.entities, &mut Continue).unwrap();
    check_rebuilt(&local, &plan).unwrap();
    local.witness = Sha256Digest::of_bytes(b"drift"); assert!(check_rebuilt(&local, &plan).is_err());
}
#[test]
fn complete_snapshot_chunks_and_mask_allowances_do_not_reset_at_document_boundaries() {
    let mut c = config(); c.max_snapshot_chunks = 3;
    assert!(prepare(vec![doc("a", "Alice Alice"), doc("b", "Alice Alice")], c).is_err());
    let mut c = config(); c.entities.masks.max_visits_per_run = c.entities.masks.max_visits_per_item;
    assert!(prepare(vec![doc("a", "Alice"), doc("b", "Bob")], c).is_err());
}
#[test]
fn every_native_axis_is_reserved_before_inference_and_early_eos_does_not_refund_pairs() {
    for axis in 0..5 {
        let mut c = config(); let w = &mut c.entities.max_model_work;
        match axis { 0 => w.forward_positions = 1, 1 => w.projected_logits = 1, 2 => w.attention_pairs = 1,
            3 => w.projections.dot_products = 1, _ => w.projections.multiply_accumulates = 1 }
        assert!(matches!(prepare(vec![doc("a", "Alice Alice")], c), Err(Int8EntityError::WorkBudget)));
    }
    let c = config(); let prepared = prepare(vec![doc("a", "Alice Alice")], c.clone()).unwrap();
    let scoring = remaining_scoring(&c.entities, prepared.ner_reserved_work()).unwrap();
    assert_eq!(plus(scoring.max_model_work, prepared.ner_reserved_work()).unwrap(), c.entities.max_model_work);
}
#[test]
fn unicode_mentions_return_to_original_documents_before_all_pairs_are_planned() {
    let c = config();
    let (maps, geometry) = collect_chunks(vec![input("a", &["é Alice", " Alice"]), input("b", &["Alice"])], &c.entities,
        &mut Continue, |i, _| Ok(run(&i.document.text, Some("Alice")))).unwrap();
    assert_eq!(maps.documents.len(), 2); assert_eq!(maps.documents[0].text, "é Alice Alice");
    assert_eq!(maps.documents[0].mentions.len(), 2); assert_eq!(geometry[0].ner_chunks, 2);
    let span = maps.documents[0].mentions[1].span;
    assert_eq!((span.byte_start, span.byte_end, span.scalar_start, span.scalar_end), (9, 14, 8, 13));
    assert_eq!(maps.verification_used.fields, 3); assert_eq!(maps.verification_used.matches, 3);
    let plan = ResolutionPlan::prepare(&maps.documents, c.entities.resolution, c.entities.graph, &mut Continue).unwrap();
    assert_eq!(plan.mentions().len(), 3);
    assert_eq!(plan.mentions()[0].document_id, "a"); assert_eq!(plan.mentions()[1].document_id, "a");
    assert_eq!(plan.mentions()[0].context(), "é Alice Alice");
    assert_eq!(plan.mentions()[1].context(), "é Alice Alice");
    let (_, _, resolver, identity) = planners();
    let scored = resolver.prepare_int8(&plan, &identity, c.entities.scoring, &mut Continue).unwrap();
    assert_eq!(scored.pair_count(), 3);
}
#[test]
fn a_local_entity_proposal_does_not_relabel_equal_text_in_another_chunk() {
    let (maps, _) = collect_chunks(vec![input("a", &["é Alice", " Alice"])], &config().entities, &mut Continue,
        |i, _| Ok(run(&i.document.text, i.document.text.starts_with('é').then_some("Alice")))).unwrap();
    assert_eq!(maps.documents[0].mentions.len(), 1);
    assert_eq!(maps.documents[0].mentions[0].span.byte_start, 3);
}
#[test]
fn repeated_occurrences_and_duplicated_proposals_are_verified_before_deduplication() {
    let (maps, _) = collect_chunks(vec![input("a", &["Alice Alice", " Alice"])], &config().entities, &mut Continue, |i, _| {
        let mut result = run(&i.document.text, Some("Alice"));
        let SourceTaskResult::Ner(ner) = &mut result.result else { panic!("fixture") };
        ner.entities.push(ner.entities[0].clone()); Ok(result)
    }).unwrap();
    assert_eq!(maps.documents[0].mentions.len(), 3); assert_eq!(maps.receipts[0].proposed_entities, 4);
    assert_eq!(maps.verification_used.fields, 4); assert_eq!(maps.verification_used.matches, 6);
}
#[test]
fn corrupt_last_chunk_never_yields_a_partial_graph_or_native_transcript() {
    for axis in 0..4 {
        let error = collect_chunks(vec![input("a", &["Alice", " Alice"])], &config().entities, &mut Continue, |i, _| {
            let mut result = run(&i.document.text, Some("Alice"));
            if i.document.text.starts_with(' ') {
                let SourceTaskResult::Ner(ner) = &mut result.result else { panic!("fixture") };
                match axis { 0 => ner.entities[0].spans[0].scalar_end += 1,
                    1 => ner.entities[0].spans.clear(), 2 => ner.numerics_profile = "hf-bf16-eager".to_owned(),
                    _ => ner.generated_token_ids.push(0) }
            }
            Ok(result)
        }).err().unwrap();
        assert!(!format!("{error:?}").contains("Alice"));
    }
}
#[test]
fn gaps_overlaps_bad_utf8_scalar_drift_and_incomplete_partitions_are_refused() {
    for axis in 0..5 {
        let mut d = input("a", &["é Alice", " Alice"]);
        match axis { 0 => d.chunks[0].span.byte_start = 1, 1 => d.chunks[0].span.byte_end = 1,
            2 => d.chunks[1].span.byte_start -= 1, 3 => d.chunks[1].span.scalar_start += 1,
            _ => { d.chunks.pop(); } }
        assert!(collect_chunks(vec![d], &config().entities, &mut Continue,
            |i, _| Ok(run(&i.document.text, Some("Alice")))).is_err());
    }
}
#[test]
fn occurrence_and_expanded_graph_budgets_are_shared_across_all_chunks() {
    for axis in 0..4 {
        let mut c = config().entities;
        match axis { 0 => c.verification.max_fields = 1, 1 => c.verification.max_matches = 1,
            2 => c.graph.max_mentions = 1, _ => c.graph.max_input_bytes = 26 }
        assert!(collect_chunks(vec![input("a", &["é Alice", " Alice"])], &c, &mut Continue,
            |i, _| Ok(run(&i.document.text, Some("Alice")))).is_err());
    }
}
#[test]
fn cancellation_during_partition_and_after_an_earlier_chunk_retains_its_cause() {
    let (source, _, _, _) = planners();
    let error = partition("Alice", &source, config().chunks, &mut Stop).err().unwrap();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline));
    let error = collect_chunks(vec![input("a", &["Alice", " Alice"])], &config().entities, &mut Continue, |i, _| {
        if i.document.text.starts_with(' ') { return Err(ResolveError::Cancelled(DecodeCancellationKind::Deadline).into()); }
        Ok(run(&i.document.text, Some("Alice")))
    }).err().unwrap();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline));
}
#[test]
fn only_empty_snapshots_can_finish_without_ner_and_the_complete_wrapper_is_bounded() {
    let nonempty = prepare(vec![doc("a", "Alice")], config()).unwrap();
    assert!(nonempty.finalize_without_model(&mut Continue).is_err());
    assert!(matches!(prepare(vec![doc("a", "")], config()), Err(Int8EntityError::InvalidInput)));
    let empty = prepare(Vec::new(), config()).unwrap().finalize_without_model(&mut Continue).unwrap();
    assert!(empty.document_chunks.is_empty()); assert_eq!(empty.output.model_work, Int8Work::default());
    assert!(!empty.output.resolution.model_evaluated);
    assert!(finish_document(empty.output, Vec::new(), 0, 1, &mut Continue).is_err());
}
#[test]
fn invalid_partition_authority_and_duplicate_original_ids_fail_before_neural_work() {
    assert!(prepare(vec![doc("same", "Alice"), doc("same", "Bob")], config()).is_err());
    for axis in 0..4 {
        let mut c = config();
        match axis { 0 => c.chunks.max_chunks = 257, 1 => c.max_snapshot_chunks = 0,
            2 => c.chunks.max_chunk_bytes = 1, _ => c.chunks.context_tokens = 16_384 }
        assert!(prepare(Vec::new(), c).is_err());
    }
}
