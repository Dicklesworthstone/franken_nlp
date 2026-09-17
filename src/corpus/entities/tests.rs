use super::*;
use std::{cell::Cell, rc::Rc};
use crate::{
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    native_engine::{decode::DecodeCancellationKind, hf_bf16_eager::HF_BF16_EAGER_PROFILE},
    tasks::{extract::ExtractionGrounding, ir::ScoreSpace, ner::{EntityType, NamedEntity}},
    template::{IM_START, IM_END, THINK_START, THINK_END},
    tokenizer::{bpe::EncodeOptions, embedded::EmbeddedTokenizer, specials::ArchivedControlRegistries},
    validation::grounded_fields::{SourceOccurrence, scan_occurrences},
};
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn config() -> EntityCorpusConfig {
    EntityCorpusConfig {
        ner: NerOptions::default(),
        ner_budget: TaskBudget { max_input_tokens: 8192, max_output_tokens: 512,
            max_output_bytes: 1024 * 1024, max_grammar_states: 100_000, max_kv_bytes: 2 * 1024 * 1024 * 1024 },
        source_planning: SourcePlanningLimits::default(),
        masks: SourceMaskBudget { per_mask: Default::default(), max_visits_per_item: 1_000_000,
            max_visits_per_run: 100_000_000 },
        resolution: ResolveOptions::default(), corpus: ResolveLimits::default(), native: NativeResolveLimits::default(),
        verification: GroundingBudget::default(),
        max_work: BatchWork { forward_positions: 100_000, projected_logits: 100_000 * 166_144 },
        max_result_bytes: 16 * 1024 * 1024, max_retained_guard_bytes: 1024 * 1024,
    }
}
fn doc(id: &str, text: &str) -> EntityDocument { EntityDocument { id: id.to_owned(), text: text.to_owned() } }
fn base_identity() -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"entity-fixture");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "bf16-verbatim".to_owned(), packing_set_digest: d,
        tokenizer_digest: d, template_digest: d, task_spec: NER_TASK_VERSION.to_owned(),
        taskir_digest: d, prompt_digest: d, grammar_compiler_version: "fixture".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::HfBf16Eager, kv_dtype: "bf16".to_owned(), sampler_version: "fixture".to_owned(),
        thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None, calibration_digest: d, decision_policy_digest: d,
        backend_semantic_version: "fixture".to_owned(), host_class: None, compiler_identity: None }
}
fn planners() -> (SourceTaskPlanner, ResolutionPlanner, ExecutionIdentity, ExecutionIdentity) {
    // Synthetic census ONLY for model-free source tests, not release authority.
    let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    let entries: Vec<_> = [IM_START, IM_END, THINK_START, THINK_END].iter().map(|&surface| {
        let ids = tokenizer.tokenizer().encode_ids_with_options(surface, EncodeOptions { add_bos: false, add_eos: false }).unwrap();
        assert_eq!(ids.len(), 1);
        serde_json::json!({"id":ids[0],"special":surface == IM_START || surface == IM_END,"surface":surface})
    }).collect();
    let eos = entries[1]["id"].as_u64().unwrap() as u32;
    let specials: Vec<_> = entries.iter().filter(|e| e["special"] == true).cloned().collect();
    let controls = ArchivedControlRegistries::from_archived_json(
        &serde_json::json!({"schema_version":1,"registry":"TokenizerSpecialIds","entries":specials}).to_string(),
        &serde_json::json!({"schema_version":1,"registry":"TemplateControlIds","entries":entries}).to_string()).unwrap();
    let source = SourceTaskPlanner::pinned(controls.template_controls(), eos).unwrap();
    let resolver = ResolutionPlanner::pinned(controls.template_controls(), eos).unwrap();
    let mut a = base_identity(); a.tokenizer_digest = source.tokenizer_digest(); a.template_digest = *source.template_digest();
    let mut b = a.clone(); b.task_spec = resolve::RESOLVE_VERSION.to_owned(); b.template_digest = resolver.template_digest();
    (source, resolver, a, b)
}
fn input(id: &str, text: &str, c: &EntityCorpusConfig) -> PreparedDocument {
    let positions = 10 + u64::from(c.ner_budget.max_output_tokens) - 1;
    PreparedDocument { document: doc(id, text), identity: Sha256Digest::of_bytes(b"private-fixture"),
        work: BatchWork { forward_positions: positions, projected_logits: positions * 166_144 } }
}
fn ner(text: &str, surface: Option<&str>) -> NerResult {
    let entities = surface.into_iter().map(|s| {
        let spans = scan_occurrences(text, s, &mut GroundingBudget::default()).unwrap();
        NamedEntity { text: s.to_owned(), entity_type: EntityType::Person,
            occurrence: if spans.len() == 1 { SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous }, spans }
    }).collect();
    NerResult { schema_version: 1, task_spec_version: NER_TASK_VERSION.to_owned(),
        numerics_profile: HF_BF16_EAGER_PROFILE.to_owned(), score_space: ScoreSpace::NotComputed,
        grounding: ExtractionGrounding::SourceMembership, entities, generated_token_ids: vec![1, 2, 0],
        forward_positions: 12, projected_logits: 12 * 166_144, mask_node_visit_charge: 10 }
}
struct Guard(Rc<Cell<usize>>);
impl Guard { fn new(n: &Rc<Cell<usize>>) -> Self { n.set(n.get() + 1); Self(Rc::clone(n)) } }
impl Drop for Guard { fn drop(&mut self) { self.0.set(self.0.get() - 1); } }

#[test]
fn canonical_document_order_preserves_original_unicode_and_empty_documents() {
    let mut docs = vec![doc("z", ""), doc("a", "é\r\nAlice")];
    assert_eq!(validate_documents(&mut docs, &config()).unwrap(), 11);
    assert_eq!(docs[0].id, "a"); assert_eq!(docs[0].text, "é\r\nAlice"); assert!(docs[1].text.is_empty());
}
#[test]
fn raw_ids_duplicates_and_aggregate_source_bytes_are_checked() {
    for mut docs in [vec![doc("", "Alice")], vec![doc("x\n", "Alice")], vec![doc("x", ""), doc("x", "different")]] {
        assert!(matches!(validate_documents(&mut docs, &config()), Err(EntityCorpusError::InvalidInput)));
    }
    let mut c = config(); c.corpus.max_input_bytes = 7;
    assert!(matches!(validate_documents(&mut [doc("a", "Alice"), doc("b", "Bob")], &c), Err(EntityCorpusError::WorkBudget)));
}
#[test]
fn model_backend_and_host_identity_cannot_drift_between_stages() {
    let a = base_identity(); let d = Sha256Digest::of_bytes(b"changed");
    for axis in 0..12 {
        let mut b = a.clone(); b.task_spec = resolve::RESOLVE_VERSION.to_owned();
        match axis { 0 => b.logical_model_digest = d, 1 => b.packing_set_digest = d, 2 => b.tokenizer_digest = d,
            3 => b.source_revision = "other".to_owned(), 4 => b.backend_semantic_version = "other".to_owned(),
            5 => b.kv_dtype = "int8".to_owned(), 6 => b.quant_recipe = "other".to_owned(),
            7 => b.artifact_format = "other".to_owned(), 8 => b.host_class = Some("other".to_owned()),
            9 => b.compiler_identity = Some("other".to_owned()), 10 => b.thinking_mode = ThinkingMode::Enabled,
            _ => b.tool_mode = ToolMode::Xml }
        assert!(same_engine_identity(&a, &b).is_err());
    }
}
#[test]
fn only_explicit_task_owned_identity_differences_are_allowed() {
    let a = base_identity(); let mut b = a.clone(); let d = Sha256Digest::of_bytes(b"task-owned");
    b.task_spec = resolve::RESOLVE_VERSION.to_owned(); b.template_digest = d; b.prompt_digest = d; b.taskir_digest = d;
    b.grammar_compiler_version = "none".to_owned(); b.schema_digest = d; b.sampler_version = "scored-eos".to_owned();
    b.calibration_digest = d; b.decision_policy_digest = d;
    assert!(same_engine_identity(&a, &b).is_ok());
}
#[test]
fn both_work_axes_share_one_nonrenewable_remainder() {
    let cap = BatchWork { forward_positions: 100, projected_logits: 1000 };
    let ner = BatchWork { forward_positions: 70, projected_logits: 400 };
    assert_eq!(subtract(cap, ner).unwrap(), BatchWork { forward_positions: 30, projected_logits: 600 });
    assert!(subtract(cap, BatchWork { forward_positions: 101, projected_logits: 1 }).is_err());
    assert!(subtract(cap, BatchWork { forward_positions: 1, projected_logits: 1001 }).is_err());
    assert!(add(cap, BatchWork { forward_positions: u64::MAX, projected_logits: 0 }).is_err());
    assert!(add(cap, BatchWork { forward_positions: 0, projected_logits: u64::MAX }).is_err());
}
#[test]
fn expanded_mention_and_byte_charges_are_aggregate_and_atomic() {
    let d = document_from_ner("a".to_owned(), "Alice Alice".to_owned(), ner("Alice Alice", Some("Alice")),
        config().corpus, &mut GroundingBudget::default(), &mut Continue).unwrap();
    let mut limits = config().corpus; limits.max_mentions = 3;
    let (mut bytes, mut mentions) = (0, 0);
    charge_document(&d, &mut bytes, &mut mentions, limits).unwrap();
    let prior = (bytes, mentions);
    assert!(charge_document(&d, &mut bytes, &mut mentions, limits).is_err()); assert_eq!((bytes, mentions), prior);
    limits.max_mentions = 9; limits.max_input_bytes = bytes;
    assert!(charge_document(&d, &mut bytes, &mut mentions, limits).is_err()); assert_eq!((bytes, mentions), prior);
}
#[test]
fn all_ner_work_is_preflighted_and_canonical_before_execution() {
    let (p, r, a, b) = planners(); let c = config();
    let x = prepare_entity_corpus(vec![doc("z", "Bob"), doc("a", "Alice")], &p, &a, &r, &b, c.clone(), &mut Continue).unwrap();
    let y = prepare_entity_corpus(vec![doc("a", "Alice"), doc("z", "Bob")], &p, &a, &r, &b, c, &mut Continue).unwrap();
    assert_eq!(x.document_count(), 2); assert_eq!(x.documents[0].document.id, "a");
    assert_eq!(x.ner_reserved_work(), y.ner_reserved_work()); assert_eq!(x.reserved_mask_visits(), 2_000_000);
    for (left, right) in x.documents.iter().zip(&y.documents) { assert_eq!(left.identity, right.identity); assert_eq!(left.work, right.work); }
}
#[test]
fn whole_ner_and_mask_limits_fail_during_model_free_preparation() {
    let (p, r, a, b) = planners();
    for axis in 0..3 {
        let mut c = config();
        match axis { 0 => c.max_work.forward_positions = 1, 1 => c.max_work.projected_logits = 1, _ => c.masks.max_visits_per_run = 1 }
        assert!(matches!(prepare_entity_corpus(vec![doc("a", "Alice")], &p, &a, &r, &b, c, &mut Continue), Err(EntityCorpusError::WorkBudget)));
    }
}
#[test]
fn rebuilt_ner_plan_must_match_its_complete_private_witness() {
    let (p, r, a, b) = planners();
    let plan = prepare_entity_corpus(vec![doc("a", "Alice")], &p, &a, &r, &b, config(), &mut Continue).unwrap();
    let mut input = plan.documents.into_iter().next().unwrap();
    let actual = plan.compiler.prepare(batch_document(&input.document, &plan.config).unwrap()).unwrap();
    check_prepared(&input, &actual).unwrap(); input.identity = Sha256Digest::of_bytes(b"foreign");
    assert!(matches!(check_prepared(&input, &actual), Err(EntityCorpusError::Accounting)));
}
#[test]
fn repeated_unicode_mentions_are_expanded_and_all_guards_stay_owned() {
    let alive = Rc::new(Cell::new(0)); let c = config();
    let maps = collect_extractions(vec![input("a", "é Alice Alice", &c), input("b", "Bob", &c)], &c, &mut Continue, |input, _| {
        let name = if input.document.id == "a" { "Alice" } else { "Bob" };
        Ok(GuardedOutput::new(SourceTaskResult::Ner(ner(&input.document.text, Some(name))), Guard::new(&alive)))
    }).unwrap();
    assert_eq!(alive.get(), 2); assert_eq!(maps.documents[0].mentions.len(), 2);
    assert_eq!(maps.documents[0].mentions[1].span.byte_start, 9);
    assert_eq!(maps.documents[0].mentions[1].span.scalar_start, 8);
    assert_eq!(maps.verification_used.matches, 3); assert_eq!(maps.verification_used.fields, 2);
    assert_eq!(maps.actual_work.forward_positions, 24); assert_eq!(maps.mask_visits, 20);
    drop(maps); assert_eq!(alive.get(), 0);
}
#[test]
fn zero_entity_documents_are_retained_in_complete_snapshot_receipts() {
    let c = config(); let maps = collect_extractions(vec![input("a", "no names", &c)], &c, &mut Continue, |i, _| {
        Ok(GuardedOutput::new(SourceTaskResult::Ner(ner(&i.document.text, None)), ()))
    }).unwrap();
    assert_eq!(maps.documents.len(), 1); assert_eq!(maps.receipts.len(), 1);
    assert_eq!(maps.receipts[0].anchored_mentions, 0); assert_eq!(maps.verification_used.fields, 0);
}
#[test]
fn corrupt_last_document_never_yields_partial_corpus_and_releases_guards() {
    let alive = Rc::new(Cell::new(0)); let calls = Cell::new(0); let c = config();
    let result = collect_extractions(vec![input("a", "Alice", &c), input("b", "Bob", &c)], &c, &mut Continue, |i, _| {
        calls.set(calls.get() + 1); let mut n = ner(&i.document.text, Some(&i.document.text));
        if i.document.id == "b" { n.entities[0].spans[0].scalar_end += 1; }
        Ok(GuardedOutput::new(SourceTaskResult::Ner(n), Guard::new(&alive)))
    });
    assert!(matches!(result, Err(EntityCorpusError::Resolution(ResolveError::InvalidAnchor))));
    assert_eq!(calls.get(), 2); assert_eq!(alive.get(), 0);
}
#[test]
fn occurrence_verification_allowance_is_not_renewed_per_document() {
    let alive = Rc::new(Cell::new(0)); let mut c = config(); c.verification.max_fields = 1;
    let result = collect_extractions(vec![input("a", "Alice", &c), input("b", "Bob", &c)], &c, &mut Continue, |i, _| {
        Ok(GuardedOutput::new(SourceTaskResult::Ner(ner(&i.document.text, Some(&i.document.text))), Guard::new(&alive)))
    });
    assert!(matches!(result, Err(EntityCorpusError::Resolution(ResolveError::InputBudget)))); assert_eq!(alive.get(), 0);
}
#[test]
fn cancellation_between_native_result_and_verification_keeps_exact_cause() {
    struct Cancel(Cell<bool>);
    impl DecodeStepControl for Cancel {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { self.0.get().then_some(DecodeCancellationKind::Deadline) }
    }
    let alive = Rc::new(Cell::new(0)); let c = config(); let mut control = Cancel(Cell::new(false)); let calls = Cell::new(0);
    let result = collect_extractions(vec![input("a", "Alice", &c), input("b", "Bob", &c)], &c, &mut control, |i, control| {
        calls.set(calls.get() + 1); control.0.set(true);
        Ok(GuardedOutput::new(SourceTaskResult::Ner(ner(&i.document.text, Some(&i.document.text))), Guard::new(&alive)))
    });
    assert!(matches!(result, Err(EntityCorpusError::Resolution(ResolveError::Cancelled(DecodeCancellationKind::Deadline)))));
    assert_eq!(calls.get(), 1); assert_eq!(alive.get(), 0);
}
#[test]
fn every_native_work_axis_is_checked_before_accepting_mentions() {
    let c = config(); let expected = input("a", "Alice", &c).work;
    for axis in 0..5 {
        let mut n = ner("Alice", Some("Alice"));
        match axis { 0 => n.forward_positions += 1, 1 => n.projected_logits += 1,
            2 => n.generated_token_ids.clear(), 3 => n.mask_node_visit_charge = c.masks.max_visits_per_item + 1,
            _ => n.generated_token_ids.resize(c.ner_budget.max_output_tokens as usize + 1, 1) }
        assert!(matches!(observed_ner(&n, expected, &c), Err(EntityCorpusError::Accounting)));
    }
}
#[test]
fn inline_guard_storage_cannot_overflow_or_escape_its_cap() {
    assert_eq!(guard_bytes::<u64>(4, 32).unwrap(), 32);
    assert!(guard_bytes::<u64>(5, 32).is_err()); assert!(guard_bytes::<u64>(usize::MAX, usize::MAX).is_err());
    assert_eq!(guard_bytes::<()>(100, 0).unwrap(), 0);
}
