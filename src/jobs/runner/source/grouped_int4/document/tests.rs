//! UNRUN pinned-planning and contract fixtures. No native inference, accuracy,
//! crash-execution, calibration or release-readiness claims follow from these.
use super::*;
use crate::{
    artifact::converter::PINNED_REVISION, canonjson,
    jobs::{FrozenManifest, JobContract, JobId, JobInput, JobSecret},
    native_engine::{decode::DecodeCancellationKind, generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::{ProjectionWork, int4::convert::GROUPED_INT4_RECIPE}},
    tasks::{keyphrases::KeyphraseOptions, ner::NerOptions},
    tokenizer::{embedded::EmbeddedTokenizer, pinned_controls},
};

const WIDTH: usize = 32;
const SOURCE: &str = "éabc----βdef";
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 8192, max_output_tokens: 64, max_output_bytes: 65536,
        max_grammar_states: 8192, max_kv_bytes: Int4MemoryRequirement::for_context(8192, WIDTH).unwrap().kv_bytes }
}
fn maximum_work() -> Int4Work {
    let n = u64::MAX / 4;
    Int4Work { forward_positions: n, projected_logits: n, attention_pairs: n,
        projections: ProjectionWork { dot_products: n, multiply_accumulates: n } }
}
fn mapping() -> Int4SourceMapLimits {
    Int4SourceMapLimits {
        chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 8, max_chunk_tokens: 8,
            context_tokens: 8192, reserved_tokens: 64, max_chunks: 8, max_tokenizer_calls: 1024 },
        reduction: ExecutionLimits { map_batch_chunks: 1, reduce_fan_in: 2,
            max_value_bytes: 4 << 20, ..ExecutionLimits::default() },
        max_model_work: maximum_work(), mask_limits: MaskWorkLimits::default(),
        mask_visits_per_chunk: 1000, max_mask_visits: 8000, verification: GroundingBudget::default(),
    }
}
fn limits() -> JobLimits {
    JobLimits { max_items: 2, max_id_bytes: 128, max_input_bytes_per_item: 4096,
        max_snapshot_bytes: 65536, max_result_bytes: 4 << 20, max_spool_bytes: 16 << 20,
        max_materialized_bytes: 16 << 20, max_journal_bytes: 1 << 20, max_attempts: 4,
        max_work: JobWork::from_int4(maximum_work(), 32_000) }
}
fn lifetime() -> Int4SourceDocumentLifetime {
    Int4SourceDocumentLifetime { verification: GroundingBudget {
        max_fields: 16_384, max_matches: 65_536, max_scan_steps: 256 * 1024 * 1024 },
        summary_scan_steps: 2 * 1024 * 1024 * 1024 }
}
fn summary_request() -> SourceSummarySynthesis {
    SourceSummarySynthesis { map_options: SummaryOptions::default(), synthesis_options: SummaryOptions::default(),
        limits: SummarySynthesisLimits { max_evidence_segments: 16, max_evidence_bytes: 4096,
            verification: GroundingBudget::default() } }
}
fn question() -> SourceQuestion {
    SourceQuestion { question: "Who is mentioned? PRIVATE_QUESTION".to_owned(),
        options: AnswerOptions::default(), verification: GroundingBudget::default() }
}
fn question_request() -> SourceQuestionSynthesis {
    SourceQuestionSynthesis { question: question(),
        limits: QuestionSynthesisLimits { max_evidence_passages: 4, max_evidence_bytes: 4096 } }
}
fn task(index: usize) -> Int4SourceDocumentTask {
    match index {
        0 => Int4SourceDocumentTask::Map(SourceMapTask::Ner(NerOptions::default())),
        1 => Int4SourceDocumentTask::Summary { options: SummaryOptions::default(), limits: Int4SummaryLimits::default() },
        2 => Int4SourceDocumentTask::Question(question()),
        3 => Int4SourceDocumentTask::SummarySynthesis(summary_request()),
        4 => Int4SourceDocumentTask::SummaryHierarchy { request: summary_request(),
            limits: SummaryHierarchyLimits { max_passes: 4, ..SummaryHierarchyLimits::default() } },
        5 => Int4SourceDocumentTask::QuestionSynthesis(question_request()),
        6 => Int4SourceDocumentTask::QuestionHierarchy { request: question_request(),
            limits: QuestionHierarchyLimits { max_passes: 4, ..QuestionHierarchyLimits::default() } },
        _ => unreachable!(),
    }
}
fn identity(planner: &SourceTaskPlanner, task: BuiltInTask) -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"private-document-job-fixture-not-model-authority");
    let packing = Sha256Digest::of_bytes(&canonjson::canonical_bytes(&(
        INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", d, WIDTH)).unwrap());
    ExecutionIdentity { schema_version: 1, source_revision: PINNED_REVISION.to_owned(), logical_model_digest: d,
        artifact_format: INT4_SOURCE_FORMAT.to_owned(), quant_recipe: format!("{GROUPED_INT4_RECIPE}:group={WIDTH}"),
        packing_set_digest: packing, tokenizer_digest: planner.tokenizer_digest(), template_digest: *planner.template_digest(),
        task_spec: task.spec().identity(), taskir_digest: d, prompt_digest: d,
        grammar_compiler_version: "uncompiled".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "uncompiled".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None }
}
fn build(task: Int4SourceDocumentTask, mapping: Int4SourceMapLimits,
    job: JobLimits, lifetime: Int4SourceDocumentLifetime) -> Result<Int4SourceDocumentJobPlanner, BatchFault> {
    let controls = pinned_controls::pinned().unwrap();
    let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    let planner = Arc::new(SourceTaskPlanner::pinned(controls.template_controls(), tokenizer.eos_token_id().unwrap()).unwrap());
    let identity = identity(&planner, task.kind());
    let source = Int4SourceJobPlanner::configured(planner, identity, WIDTH, budget(), SourcePlanningLimits::default(),
        None, Int4SourceBatchLimits { max_model_work: job.max_work.int4_model_work(),
            masks: SourceMaskBudget { per_mask: mapping.mask_limits,
                max_visits_per_item: mapping.mask_visits_per_chunk, max_visits_per_run: job.max_work.mask_node_visits } })?;
    Int4SourceDocumentJobPlanner::configured(source, task, mapping, job, lifetime)
}
fn factory(index: usize) -> Int4SourceDocumentJobPlanner { build(task(index), mapping(), limits(), lifetime()).unwrap() }
fn document(text: &str) -> BatchDocument<SourceDocumentArgs> {
    BatchDocument { id: "whole-document".to_owned(), text: text.to_owned(), task_args: None }
}
fn prepare(factory: &Int4SourceDocumentJobPlanner, text: &str) -> PreparedInt4SourceDocument {
    factory.prepare_with_control(document(text), &mut Continue).unwrap()
}
fn freeze(factory: &Int4SourceDocumentJobPlanner) -> FrozenManifest {
    let input = br#"{"id":"whole-document","text":"private complete source"}"#;
    FrozenManifest::freeze(&JobSecret::from_bytes([7; 32]), JobContract { job_id: JobId([9; 16]),
        execution: factory.execution_identity(), recipe: factory.job_recipe(), limits: factory.job_limits() },
        [JobInput { id: "whole-document", original: input, normalized: input }], &mut Continue).unwrap()
}

#[test]
fn all_seven_routes_prepare_real_complete_pinned_discovery() {
    for index in 0..7 {
        let factory = factory(index); let prepared = prepare(&factory, SOURCE);
        assert_eq!(prepared.chunk_count(), 2);
        assert_eq!(prepared.execution_identities().len(), 2);
        for identity in prepared.execution_identities() {
            assert_eq!(identity.task_spec, factory.execution_identity().task_spec);
            assert_eq!(identity.numerics_profile, NumericsProfile::StrictQuantized { version: 2 });
            assert_ne!(identity.prompt_digest, factory.execution_identity().prompt_digest);
        }
        assert_eq!(prepared.work, prepared.expected.work().unwrap());
        factory.plan(SOURCE, &mut Continue).unwrap().compare(&prepared, &mut Continue).unwrap();
    }
}
#[test]
fn keyphrase_and_independent_summary_maps_keep_their_task_identities() {
    for task in [SourceMapTask::Keyphrases(KeyphraseOptions::default()), SourceMapTask::Summarize(SummaryOptions::default())] {
        let factory = build(Int4SourceDocumentTask::Map(task), mapping(), limits(), lifetime()).unwrap();
        let prepared = prepare(&factory, SOURCE);
        assert!(matches!(prepared.expected, planning::Expected::Map(_)));
        assert!(prepared.execution_identities().iter().all(|id| id.task_spec == factory.execution_identity().task_spec));
    }
}
#[test]
fn synthesis_and_hierarchies_debit_all_permitted_passes_before_native_work() {
    for (single, hierarchy) in [(3, 4), (5, 6)] {
        let single = prepare(&factory(single), SOURCE); let hierarchy = prepare(&factory(hierarchy), SOURCE);
        let one = single.planned_work(); let many = hierarchy.planned_work();
        assert_eq!(one.mask_node_visits, 3000);
        assert_eq!(many.mask_node_visits, 6000);
        assert!(one.fits(many));
        assert!(many.model.forward_positions > one.model.forward_positions);
        assert!(many.model.projected_logits > one.model.projected_logits);
        assert!(many.model.attention_pairs > one.model.attention_pairs);
        assert!(many.model.projections.dot_products > one.model.projections.dot_products);
        assert!(many.model.projections.multiply_accumulates > one.model.projections.multiply_accumulates);
    }
}
#[test]
fn every_discovery_identity_and_full_receipt_survive_reconstruction() {
    let factory = factory(4); let mut prepared = prepare(&factory, SOURCE);
    prepared.identities.last_mut().unwrap().prompt_digest = Sha256Digest::of_bytes(b"changed final chunk");
    assert!(factory.plan(SOURCE, &mut Continue).unwrap().compare(&prepared, &mut Continue).is_err());
    let mut prepared = prepare(&factory, SOURCE);
    prepared.work.model.projections.dot_products -= 1;
    assert!(factory.plan(SOURCE, &mut Continue).unwrap().compare(&prepared, &mut Continue).is_err());
    let prepared = prepare(&factory, SOURCE);
    assert!(factory.plan("changed source", &mut Continue).unwrap().compare(&prepared, &mut Continue).is_err());
}
#[test]
fn prepared_documents_cannot_cross_factories_with_identical_public_policies() {
    let first = factory(0); let second = factory(0); let prepared = prepare(&first, SOURCE);
    first.owns(&prepared).unwrap();
    assert!(second.owns(&prepared).is_err());
}
#[test]
fn question_whitespace_ranges_are_not_native_plans_or_abstentions() {
    let factory = factory(2); let prepared = prepare(&factory, "Alice   \n        \nBob");
    assert!(prepared.chunk_count() > prepared.execution_identities().len());
    assert!(factory.prepare_with_control(document(" \n\t  "), &mut Continue).is_err());
    factory.plan(&prepared.source, &mut Continue).unwrap().compare(&prepared, &mut Continue).unwrap();
}
#[test]
fn each_document_work_axis_and_mask_ceiling_must_fit_immutable_lifetime_authority() {
    for axis in 0..6 {
        let mut job = limits();
        let maximum = JobWork::from_int4(mapping().max_model_work, mapping().max_mask_visits);
        job.max_work = maximum;
        match axis {
            0 => job.max_work.model.forward_positions -= 1,
            1 => job.max_work.model.projected_logits -= 1,
            2 => job.max_work.model.attention_pairs -= 1,
            3 => job.max_work.model.projections.dot_products -= 1,
            4 => job.max_work.model.projections.multiply_accumulates -= 1,
            _ => job.max_work.mask_node_visits -= 1,
        }
        assert!(build(task(0), mapping(), job, lifetime()).is_err(), "axis {axis}");
    }
}
#[test]
fn lifetime_verification_covers_every_failed_or_uncertain_attempt_on_all_axes() {
    for axis in 0..4 {
        let mut cap = lifetime();
        match axis { 0 => cap.verification.max_fields -= 1, 1 => cap.verification.max_matches -= 1,
            2 => cap.verification.max_scan_steps -= 1, _ => cap.summary_scan_steps -= 1 }
        assert!(build(task(1), mapping(), limits(), cap).is_err(), "axis {axis}");
    }
    let mut cap = lifetime(); cap.verification.max_scan_steps = u64::MAX;
    let mut per = GroundingBudget::default(); per.max_scan_steps = u64::MAX;
    assert!(verify_lifetime(per, 0, cap, 2).is_err());
}
#[test]
fn question_and_synthesis_lifetime_proof_uses_the_effective_verifier() {
    let mut map = mapping(); map.verification.max_fields = 1;
    let mut cap = lifetime(); cap.verification.max_fields = 4;
    assert!(build(task(0), map, limits(), cap).is_ok());
    for index in [2, 3, 4, 5, 6] { assert!(build(task(index), map, limits(), cap).is_err()); }
}
#[test]
fn every_immutable_job_limit_remains_equal_when_opening_or_resuming() {
    let factory = factory(0); let exact = factory.job_limits(); factory.check_job_limits(exact).unwrap();
    for axis in 0..7 {
        let mut changed = exact;
        match axis { 0 => changed.max_attempts += 1, 1 => changed.max_items += 1,
            2 => changed.max_input_bytes_per_item += 1, 3 => changed.max_result_bytes -= 1,
            4 => changed.max_work.model.projections.dot_products -= 1,
            5 => changed.max_work.mask_node_visits -= 1, _ => changed.max_spool_bytes -= 1 }
        assert!(factory.check_job_limits(changed).is_err());
    }
}
#[test]
fn complete_private_question_and_hierarchy_policy_are_keyed_recipe_authority() {
    let original = factory(6); let baseline = freeze(&original);
    let mut changed = question_request(); changed.question.question.push('!');
    let changed = build(Int4SourceDocumentTask::QuestionHierarchy { request: changed,
        limits: QuestionHierarchyLimits { max_passes: 4, ..QuestionHierarchyLimits::default() } },
        mapping(), limits(), lifetime()).unwrap();
    assert!(baseline.binding.compare(&freeze(&changed).binding).is_err());
    let changed = build(Int4SourceDocumentTask::QuestionHierarchy { request: question_request(),
        limits: QuestionHierarchyLimits { max_passes: 3, ..QuestionHierarchyLimits::default() } },
        mapping(), limits(), lifetime()).unwrap();
    assert!(baseline.binding.compare(&freeze(&changed).binding).is_err());
    let projected = serde_json::to_value(original.job_recipe()).unwrap();
    assert_eq!(projected["task"]["request"]["question"]["question"], question().question);
    assert_eq!(projected["task"]["limits"]["max_passes"], 4);
    assert_eq!(projected["dependency_scope"], "item-local-complete-document");
}
#[test]
fn mapping_verification_and_lifetime_changes_invalidate_resume_commitments() {
    let baseline = freeze(&factory(0));
    let mut changed = mapping(); changed.verification.max_fields -= 1;
    assert!(baseline.binding.compare(&freeze(&build(task(0), changed, limits(), lifetime()).unwrap()).binding).is_err());
    let mut changed = mapping(); changed.chunks.max_tokenizer_calls += 1;
    assert!(baseline.binding.compare(&freeze(&build(task(0), changed, limits(), lifetime()).unwrap()).binding).is_err());
    let mut changed = lifetime(); changed.verification.max_matches += 1;
    assert!(baseline.binding.compare(&freeze(&build(task(0), mapping(), limits(), changed).unwrap()).binding).is_err());
}
#[test]
fn exact_summary_can_store_a_smaller_final_result_than_native_and_map_envelopes() {
    let mut job = limits(); job.max_result_bytes = 32_768;
    let summary = Int4SummaryLimits { max_result_bytes: 32_768, ..Int4SummaryLimits::default() };
    let factory = build(Int4SourceDocumentTask::Summary { options: SummaryOptions::default(), limits: summary },
        mapping(), job, lifetime()).unwrap();
    assert_eq!(factory.max_result_bytes(), 32_768);
    assert!(factory.task_budget().max_output_bytes > factory.max_result_bytes());
    assert!(factory.document_limits().reduction.max_result_bytes as u64 > factory.max_result_bytes());
    assert!(build(task(0), mapping(), job, lifetime()).is_err());
}
#[test]
fn retained_token_arrays_are_priced_per_output_shape() {
    let expected = [8 * 64, 0, 0, 9 * 64, 12 * 64, 0, 4 * 64];
    for (index, tokens) in expected.into_iter().enumerate() {
        assert_eq!(factory(index).output_token_ceiling().unwrap(), tokens);
    }
}
#[test]
fn empty_oversized_cancelled_or_task_override_documents_never_prepare() {
    let factory = factory(0);
    assert!(factory.prepare_with_control(document(""), &mut Continue).is_err());
    assert!(factory.prepare_with_control(document(&"x".repeat(4097)), &mut Continue).is_err());
    struct Cancel;
    impl DecodeStepControl for Cancel { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        Some(DecodeCancellationKind::Deadline)
    } }
    assert!(factory.prepare_with_control(document(SOURCE), &mut Cancel).is_err());
    assert!(serde_json::from_str::<SourceDocumentArgs>(r#"{"question":"override"}"#).is_err());
    assert!(serde_json::from_str::<SourceDocumentArgs>(r#"{"max_output_tokens":1}"#).is_err());
    assert!(serde_json::from_str::<SourceDocumentArgs>("{}").is_ok());
}
struct Admission;
impl Int4SourceDocumentBatchAdmission for Admission {
    type Guard = ();
    fn admit(&mut self, _: Int4SourceDocumentAdmission<'_>) -> Result<(Vec<ExecutionIdentity>, ()), BatchItemFailure> {
        Err(BatchItemFailure::fatal(BatchCode::Admission))
    }
}
#[test]
fn real_native_processor_and_owned_factory_satisfy_durable_host_types() {
    fn durable<P: DurableBatchProcessor>() {}
    fn send<T: Send + 'static>() {}
    durable::<Int4SourceDocumentJobProcessor<'static, 'static, 'static, 'static, Admission>>();
    send::<Int4SourceDocumentJobPlanner>(); send::<PreparedInt4SourceDocument>();
}
