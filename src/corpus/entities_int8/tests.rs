//! Model-free planning and private corruption fixtures, NOT neural success.
use super::*;
use crate::{
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    native_engine::strict_int8::STRICT_INT8_EXECUTION,
    tasks::{extract::ExtractionGrounding, ir::ScoreSpace, ner::{EntityType, NamedEntity}},
    tokenizer::pinned_controls,
    validation::grounded_fields::{SourceOccurrence, scan_occurrences},
};
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
struct Stop;
impl DecodeStepControl for Stop {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::Deadline) }
}
fn config() -> Int8EntityConfig {
    let cap = Int8Work::for_sequence(0, 16_384, 16_384 * 166_144).unwrap();
    Int8EntityConfig {
        ner: NerOptions::default(),
        ner_budget: TaskBudget { max_input_tokens: 8192, max_output_tokens: 64,
            max_output_bytes: 1024 * 1024, max_grammar_states: 100_000, max_kv_bytes: 2 * 1024 * 1024 * 1024 },
        source_planning: SourcePlanningLimits::default(),
        masks: SourceMaskBudget { per_mask: Default::default(), max_visits_per_item: 1_000_000_000,
            max_visits_per_run: 1_000_000_000_000 },
        resolution: ResolveOptions::default(), graph: ResolveLimits::default(),
        scoring: Int8ResolveLimits { planning: Default::default(), max_model_work: cap },
        verification: GroundingBudget::default(), max_model_work: cap, max_result_bytes: 16 * 1024 * 1024,
    }
}
fn doc(id: &str, text: &str) -> EntityDocument { EntityDocument { id: id.to_owned(), text: text.to_owned() } }
fn identity() -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"entity-int8-model-free-fixture");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "fixture-int8".to_owned(), packing_set_digest: d,
        tokenizer_digest: d, template_digest: d, task_spec: NER_TASK_VERSION.to_owned(),
        taskir_digest: d, prompt_digest: d, grammar_compiler_version: "fixture".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None }
}
fn planners() -> (Arc<SourceTaskPlanner>, ExecutionIdentity, Arc<ResolutionPlanner>, ExecutionIdentity) {
    let registry = pinned_controls::pinned().unwrap(); let controls = registry.template_controls();
    let eos = controls.entries().iter().find(|e| e.special && e.surface == crate::template::IM_END).unwrap().id;
    let source = Arc::new(SourceTaskPlanner::pinned(controls, eos).unwrap());
    let resolver = Arc::new(ResolutionPlanner::pinned(controls, eos).unwrap());
    let mut a = identity(); a.tokenizer_digest = source.tokenizer_digest(); a.template_digest = *source.template_digest();
    let mut b = a.clone(); b.task_spec = resolve::RESOLVE_VERSION.to_owned(); b.template_digest = resolver.template_digest();
    (source, a, resolver, b)
}
fn input(id: &str, text: &str) -> Input {
    Input { document: doc(id, text), witness: Sha256Digest::of_bytes(b"private-fixture"), prompt_tokens: 10,
        work: constrained_int8::planned_work(10, config().ner_budget.max_output_tokens as usize).unwrap() }
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
fn ner_mut(run: &mut Int8SourceTaskRun) -> &mut NerResult {
    let SourceTaskResult::Ner(ner) = &mut run.result else { panic!("fixture") }; ner
}
fn axis(work: &mut Int8Work, axis: usize) -> &mut u64 {
    match axis { 0 => &mut work.forward_positions, 1 => &mut work.projected_logits, 2 => &mut work.attention_pairs,
        3 => &mut work.projections.dot_products, _ => &mut work.projections.multiply_accumulates }
}

#[test]
fn all_five_axes_have_checked_nonrenewable_remainders() {
    let config = config(); let ner = constrained_int8::planned_work(10, 64).unwrap();
    let remaining = remaining_scoring(&config, ner).unwrap();
    assert_eq!(plus(remaining.max_model_work, ner).unwrap(), config.max_model_work);
    for i in 0..5 {
        let mut cap = ner; *axis(&mut cap, i) -= 1;
        assert!(matches!(subtract(cap, ner), Err(Int8EntityError::WorkBudget)));
        let mut a = Int8Work::default(); *axis(&mut a, i) = u64::MAX;
        let mut b = Int8Work::default(); *axis(&mut b, i) = 1;
        assert!(plus(a, b).is_err());
    }
    // Early EOS does not return the conservative reservation to pair scoring.
    let actual = constrained_int8::planned_work(10, 3).unwrap();
    assert_ne!(subtract(config.max_model_work, actual).unwrap(), remaining.max_model_work);
}
#[test]
fn canonical_sources_have_identical_private_witnesses_and_real_pinned_plans() {
    let (p, a, r, b) = planners();
    let x = prepare_int8_entities(vec![doc("z", "Bob"), doc("a", "é Alice")], p.clone(), a.clone(),
        r.clone(), b.clone(), config(), &mut Continue).unwrap();
    let y = prepare_int8_entities(vec![doc("a", "é Alice"), doc("z", "Bob")], p, a, r, b, config(), &mut Continue).unwrap();
    assert_eq!(x.document_count(), 2); assert_eq!(x.inputs[0].document.id, "a");
    assert_eq!(x.ner_reserved_work(), y.ner_reserved_work()); assert_eq!(x.reserved_mask_visits(), 2_000_000_000);
    for (left, right) in x.inputs.iter().zip(&y.inputs) {
        assert_eq!(left.witness, right.witness); assert_eq!(left.work, right.work);
    }
    let rebuilt = source_plan(&x.inputs[0].document.text, &x.source, &x.source_identity, &x.config, &mut Continue).unwrap();
    check_rebuilt(&x.inputs[0], &rebuilt).unwrap();
    let mut changed = input("a", "é Alice"); changed.work = x.inputs[0].work; changed.prompt_tokens = x.inputs[0].prompt_tokens;
    assert!(matches!(check_rebuilt(&changed, &rebuilt), Err(Int8EntityError::Accounting)));
}
#[test]
fn every_ner_axis_and_whole_mask_cap_fail_before_inference() {
    let (p, a, r, b) = planners();
    for i in 0..6 {
        let mut c = config();
        if i < 5 { *axis(&mut c.max_model_work, i) = 1; } else { c.masks.max_visits_per_run = 1; }
        assert!(matches!(prepare_int8_entities(vec![doc("a", "Alice")], p.clone(), a.clone(), r.clone(), b.clone(), c,
            &mut Continue), Err(Int8EntityError::WorkBudget)));
    }
}
#[test]
fn empty_snapshot_is_real_zero_work_but_nonempty_cannot_skip_ner() {
    let (p, a, r, b) = planners();
    let nonempty = prepare_int8_entities(vec![doc("a", "")], p.clone(), a.clone(), r.clone(), b.clone(), config(), &mut Continue).unwrap();
    assert!(matches!(nonempty.finalize_without_model(&mut Continue), Err(Int8EntityError::Accounting)));
    let empty = prepare_int8_entities(Vec::new(), p, a, r, b, config(), &mut Continue).unwrap();
    let result = empty.finalize_without_model(&mut Continue).unwrap();
    assert_eq!(result.model_work, Int8Work::default()); assert_eq!(result.reserved_mask_node_visits, 0);
    assert!(result.documents.is_empty()); assert!(!result.resolution.model_evaluated);
    assert!(result.resolution.result.mentions.is_empty()); assert!(result.resolution.result.clusters.is_empty());
}
#[test]
fn incompatible_stage_models_and_eager_profiles_are_refused_even_for_empty_input() {
    let (p, a, r, b) = planners();
    for i in 0..4 {
        let mut changed = b.clone();
        match i { 0 => changed.logical_model_digest = Sha256Digest::of_bytes(b"different"),
            1 => changed.backend_semantic_version = "different".to_owned(),
            2 => changed.numerics_profile = NumericsProfile::HfBf16Eager,
            _ => changed.packing_set_digest = Sha256Digest::of_bytes(b"different") }
        assert!(matches!(prepare_int8_entities(Vec::new(), p.clone(), a.clone(), r.clone(), changed, config(), &mut Continue),
            Err(Int8EntityError::Identity)));
    }
}
#[test]
fn all_unicode_occurrences_are_recovered_without_relocating_or_merging_names() {
    let maps = collect(vec![input("a", "é Alice Alice"), input("b", "Bob")], &config(), &mut Continue,
        |i, _| Ok(run(&i.document.text, Some(if i.document.id == "a" { "Alice" } else { "Bob" })))).unwrap();
    assert_eq!(maps.documents[0].mentions.len(), 2);
    let span = maps.documents[0].mentions[1].span;
    assert_eq!((span.byte_start, span.byte_end, span.scalar_start, span.scalar_end), (9, 14, 8, 13));
    assert_eq!(maps.verification_used.fields, 2); assert_eq!(maps.verification_used.matches, 3);
    assert_eq!(maps.mask_visits, 20);
    assert_eq!(maps.work, plus(constrained_int8::planned_work(10, 3).unwrap(), constrained_int8::planned_work(10, 3).unwrap()).unwrap());
}
#[test]
fn corrupt_last_document_aborts_complete_collection() {
    let mut calls = 0;
    let result = collect(vec![input("a", "Alice"), input("b", "Bob")], &config(), &mut Continue, |i, _| {
        calls += 1; let mut result = run(&i.document.text, Some(&i.document.text));
        if i.document.id == "b" { ner_mut(&mut result).entities[0].spans[0].scalar_end += 1; }
        Ok(result)
    });
    assert_eq!(calls, 2); assert!(matches!(result, Err(Int8EntityError::Graph(ResolveError::InvalidAnchor))));
}
#[test]
fn omission_ambiguous_evidence_wrong_type_and_profile_are_not_accepted() {
    for corruption in 0..5 {
        let result = collect(vec![input("a", "Alice Alice")], &config(), &mut Continue, |i, _| {
            let mut result = run(&i.document.text, Some("Alice")); let n = ner_mut(&mut result);
            match corruption { 0 => { n.entities[0].spans.pop(); },
                1 => n.entities[0].occurrence = SourceOccurrence::Anchored,
                2 => n.entities[0].entity_type = EntityType::Money,
                3 => n.numerics_profile = "hf-bf16-eager".to_owned(),
                _ => { n.entities[0].text = "absent".to_owned(); n.entities[0].spans.clear(); } }
            Ok(result)
        });
        assert!(result.is_err());
    }
}
#[test]
fn every_native_receipt_axis_is_independently_verified() {
    for i in 0..5 {
        let result = collect(vec![input("a", "Alice")], &config(), &mut Continue, |_, _| {
            let mut result = run("Alice", Some("Alice")); *axis(&mut result.model_work, i) += 1; Ok(result)
        });
        assert!(matches!(result, Err(Int8EntityError::Accounting)));
    }
}
#[test]
fn verification_fields_and_expanded_mentions_are_not_renewed_per_document() {
    for budget in 0..3 {
        let mut c = config();
        match budget { 0 => c.verification.max_fields = 1, 1 => c.graph.max_mentions = 1, _ => c.graph.max_input_bytes = 20 }
        let result = collect(vec![input("a", "Alice"), input("b", "Bob")], &c, &mut Continue,
            |i, _| Ok(run(&i.document.text, Some(&i.document.text))));
        assert!(result.is_err());
    }
}
#[test]
fn duplicate_proposals_are_verified_before_deduplication_and_empty_results_keep_documents() {
    let maps = collect(vec![input("a", "Alice"), input("b", "no names")], &config(), &mut Continue, |i, _| {
        let mut result = run(&i.document.text, (i.document.id == "a").then_some("Alice"));
        if i.document.id == "a" { let n = ner_mut(&mut result); n.entities.push(n.entities[0].clone()); }
        Ok(result)
    }).unwrap();
    assert_eq!(maps.documents.len(), 2); assert_eq!(maps.documents[0].mentions.len(), 1);
    assert!(maps.documents[1].mentions.is_empty()); assert_eq!(maps.verification_used.fields, 2);
    assert_eq!(maps.receipts[0].proposed_entities, 2); assert_eq!(maps.receipts[1].anchored_mentions, 0);
}
#[test]
fn cancellation_aborts_preparation_and_collection_without_invoking_another_pass() {
    let (p, a, r, b) = planners();
    let error = match prepare_int8_entities(vec![doc("a", "Alice")], p, a, r, b, config(), &mut Stop) {
        Err(error) => error, Ok(_) => panic!("must cancel"),
    };
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline));
    let mut calls = 0;
    assert!(collect(vec![input("a", "Alice")], &config(), &mut Stop, |_, _| {
        calls += 1; Ok(run("Alice", Some("Alice")))
    }).is_err());
    assert_eq!(calls, 0);
}
#[test]
fn duplicate_ids_and_input_caps_are_checked_without_changing_source_bytes() {
    let mut sources = vec![doc("z", ""), doc("a", "é\r\nAlice")];
    assert_eq!(validate_documents(&mut sources, &config()).unwrap(), 11);
    assert_eq!(sources[0].text, "é\r\nAlice");
    assert!(matches!(validate_documents(&mut [doc("a", "Alice"), doc("a", "Bob")], &config()), Err(Int8EntityError::InvalidInput)));
    let mut c = config(); c.graph.max_input_bytes = 1;
    assert!(validate_documents(&mut sources, &c).is_err());
}
#[test]
fn raw_input_capacities_not_just_lengths_are_retained_for_host_pricing() {
    let (p, a, r, b) = planners(); let mut document = doc("a", "Alice"); document.text.reserve(8192);
    let capacity = document.text.capacity() as u64;
    let prepared = prepare_int8_entities(vec![document], p, a, r, b, config(), &mut Continue).unwrap();
    assert!(prepared.retained_input_bytes().unwrap() >= capacity);
    assert_eq!(prepared.input_bytes(), 6);
}
