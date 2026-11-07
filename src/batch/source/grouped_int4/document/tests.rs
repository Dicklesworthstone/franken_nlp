//! UNRUN source/identity fixtures. These exercise the pinned planner only;
//! native parity, accuracy and throughput require separately retained evidence.
use super::*;
use crate::{
    artifact::converter::PINNED_REVISION,
    native_engine::{decode::DecodeCancellationKind, generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::{ProjectionWork, int4::convert::GROUPED_INT4_RECIPE}},
    tasks::{mapreduce::{ChunkLimits, ExecutionLimits}, ner::NerOptions,
        source_planning::grouped_int4::long::{summary::synthesis::SummarySynthesisLimits,
            question::synthesis::QuestionSynthesisLimits}},
    tokenizer::{embedded::EmbeddedTokenizer, pinned_controls},
};

const WIDTH: usize = 32;
pub(super) const SOURCE: &str = "éabc----βdef";
pub(super) struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 8192, max_output_tokens: 64, max_output_bytes: 65_536,
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
        reduction: ExecutionLimits { map_batch_chunks: 3, reduce_fan_in: 2,
            max_value_bytes: 4 << 20, ..ExecutionLimits::default() },
        max_model_work: maximum_work(), mask_limits: MaskWorkLimits::default(),
        mask_visits_per_chunk: 1000, max_mask_visits: 8000, verification: GroundingBudget::default(),
    }
}
fn run_limits() -> Int4SourceDocumentBatchLimits {
    Int4SourceDocumentBatchLimits { max_model_work: maximum_work(), max_mask_visits: 32_000,
        max_verification: GroundingBudget { max_fields: 16_384, max_matches: 65_536, max_scan_steps: 256 * 1024 * 1024 },
        max_summary_scan_steps: 2 * 1024 * 1024 * 1024, max_keyphrase_scan_work: u64::MAX / 4 }
}
fn summary() -> SourceSummarySynthesis {
    SourceSummarySynthesis { map_options: SummaryOptions::default(), synthesis_options: SummaryOptions::default(),
        limits: SummarySynthesisLimits { max_evidence_segments: 16, max_evidence_bytes: 4096,
            verification: GroundingBudget::default() } }
}
fn question() -> SourceQuestion {
    SourceQuestion { question: "Who is mentioned? PRIVATE_FIXED_QUESTION".to_owned(),
        options: AnswerOptions::default(), verification: GroundingBudget::default() }
}
fn question_request() -> SourceQuestionSynthesis {
    SourceQuestionSynthesis { question: question(),
        limits: QuestionSynthesisLimits { max_evidence_passages: 4, max_evidence_bytes: 4096 } }
}
fn task(index: usize) -> Int4SourceDocumentTask {
    match index {
        0 => Int4SourceDocumentTask::Map(SourceMapTask::Ner(NerOptions::default())),
        1 => Int4SourceDocumentTask::Map(SourceMapTask::Keyphrases(KeyphraseOptions::default())),
        2 => Int4SourceDocumentTask::Keyphrases { options: KeyphraseOptions::default(),
            limits: Int4KeyphraseLimits::default(), max_lifetime_scan_work: u64::MAX / 4 },
        3 => Int4SourceDocumentTask::Summary { options: SummaryOptions::default(), limits: Int4SummaryLimits::default() },
        4 => Int4SourceDocumentTask::SummarySynthesis(summary()),
        5 => Int4SourceDocumentTask::SummaryHierarchy { request: summary(),
            limits: SummaryHierarchyLimits { max_passes: 4, ..SummaryHierarchyLimits::default() } },
        6 => Int4SourceDocumentTask::Question(question()),
        7 => Int4SourceDocumentTask::QuestionSynthesis(question_request()),
        8 => Int4SourceDocumentTask::QuestionHierarchy { request: question_request(),
            limits: QuestionHierarchyLimits { max_passes: 4, ..QuestionHierarchyLimits::default() } },
        9 => Int4SourceDocumentTask::Map(SourceMapTask::Summarize(SummaryOptions::default())),
        _ => unreachable!(),
    }
}
fn identity(planner: &SourceTaskPlanner, task: BuiltInTask) -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"private source corpus fixture; not model authority");
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
pub(super) fn factory(index: usize) -> Int4SourceDocumentBatchPlanner {
    let controls = pinned_controls::pinned().unwrap();
    let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    let planner = Arc::new(SourceTaskPlanner::pinned(controls.template_controls(), tokenizer.eos_token_id().unwrap()).unwrap());
    let task = task(index); let identity = identity(&planner, task.kind());
    Int4SourceDocumentBatchPlanner::configured(planner, identity, WIDTH, task, budget(), SourcePlanningLimits::default(),
        mapping(), run_limits()).unwrap()
}
pub(super) fn document(text: &str) -> BatchDocument<SourceDocumentArgs> {
    BatchDocument { id: "whole-document".to_owned(), text: text.to_owned(), task_args: None }
}

#[test]
fn all_complete_source_modes_prepare_exact_original_discovery() {
    for index in 0..10 {
        let factory = factory(index);
        let prepared = factory.prepare_with_control(document(SOURCE), &mut Continue).unwrap();
        assert_eq!(prepared.chunk_count(), 2, "mode {index}");
        assert_eq!(prepared.execution_identities().len(), 2, "mode {index}");
        assert_ne!(prepared.execution_identities()[0].prompt_digest, prepared.execution_identities()[1].prompt_digest);
        assert!(prepared.execution_identities().iter().all(|id| id.task_spec == factory.execution_identity().task_spec));
        let plan = factory.configuration().plan(SOURCE, &mut Continue).unwrap();
        assert_eq!(plan.expected().work().unwrap().model, prepared.planned_work());
    }
}
#[test]
fn schedules_preserve_native_work_and_include_all_hierarchy_reservations() {
    for index in 0..10 {
        let serial = factory(index);
        let before = serial.prepare_with_control(document(SOURCE), &mut Continue).unwrap();
        let cohorts = serial.with_chunk_cohorts().unwrap();
        let after = cohorts.prepare_with_control(document(SOURCE), &mut Continue).unwrap();
        assert_eq!(before.planned_work(), after.planned_work());
        assert_eq!(before.reserved_mask_visits(), after.reserved_mask_visits());
        assert!(cohorts.cohort_requirements().unwrap().native.kv_bytes > cohorts.task_budget().max_kv_bytes);
    }
    for index in [5, 8] {
        let hierarchy = factory(index).with_hierarchy_cohorts(2).unwrap().with_chunk_cohorts().unwrap();
        let prepared = hierarchy.prepare_with_control(document(SOURCE), &mut Continue).unwrap();
        assert_eq!(prepared.reserved_mask_visits(), 6000);
        assert!(hierarchy.hierarchy_cohort_schedule().unwrap().discovery_cohorts);
    }
    assert!(factory(0).with_hierarchy_cohorts(2).is_err());
}
#[test]
fn every_per_document_work_and_verification_axis_must_fit_run_authority() {
    for axis in 0..9 {
        let map = mapping(); let mut limits = run_limits();
        limits.max_model_work = map.max_model_work; limits.max_mask_visits = map.max_mask_visits;
        limits.max_verification = map.verification;
        match axis {
            0 => limits.max_model_work.forward_positions -= 1,
            1 => limits.max_model_work.projected_logits -= 1,
            2 => limits.max_model_work.attention_pairs -= 1,
            3 => limits.max_model_work.projections.dot_products -= 1,
            4 => limits.max_model_work.projections.multiply_accumulates -= 1,
            5 => limits.max_mask_visits -= 1,
            6 => limits.max_verification.max_fields -= 1,
            7 => limits.max_verification.max_matches -= 1,
            _ => limits.max_verification.max_scan_steps -= 1,
        }
        assert!(validate_limits(&task(0), budget(), SourcePlanningLimits::default(), map, limits).is_err(), "axis {axis}");
    }
    let mut limits = run_limits(); limits.max_summary_scan_steps = task(3).summary_scan_steps() - 1;
    assert!(validate_limits(&task(3), budget(), SourcePlanningLimits::default(), mapping(), limits).is_err());
    let mut limits = run_limits(); limits.max_keyphrase_scan_work = task(2).keyphrase_scan_work() - 1;
    assert!(validate_limits(&task(2), budget(), SourcePlanningLimits::default(), mapping(), limits).is_err());
    let mut limits = run_limits(); limits.max_keyphrase_scan_work += 1;
    assert!(validate_limits(&task(2), budget(), SourcePlanningLimits::default(), mapping(), limits).is_err());
}
#[test]
fn fixed_policy_records_accept_only_an_empty_object_or_absence() {
    for input in [r#"{"id":"one","text":"source"}"#, r#"{"id":"one","text":"source","task_args":null}"#,
        r#"{"id":"one","text":"source","task_args":{}}"#] {
        assert!(serde_json::from_str::<BatchDocument<SourceDocumentArgs>>(input).is_ok());
    }
    for args in ["[]", "[{}]", "true", "\"policy\"", r#"{"question":"override"}"#,
        r#"{"max_output_tokens":1}"#, r#"{"options":{}}"#] {
        assert!(serde_json::from_str::<SourceDocumentArgs>(args).is_err(), "{args}");
    }
}
#[test]
fn retention_quotes_actual_private_capacity_and_output_shapes() {
    let mut fixed = question(); fixed.question.reserve(4096);
    let actual = fixed.question.capacity() as u64;
    assert_eq!(Int4SourceDocumentTask::Question(fixed).retained_bytes().unwrap(), actual);
    let mut options = NerOptions::default(); options.types.reserve(512);
    let actual = (options.types.capacity() * std::mem::size_of::<crate::tasks::ner::EntityType>()) as u64;
    assert_eq!(Int4SourceDocumentTask::Map(SourceMapTask::Ner(options)).retained_bytes().unwrap(), actual);
    let expected = [8 * 64, 8 * 64, 0, 0, 9 * 64, 12 * 64, 0, 0, 4 * 64, 8 * 64];
    for (index, tokens) in expected.into_iter().enumerate() {
        assert_eq!(factory(index).output_token_ceiling().unwrap(), tokens);
    }
}
#[test]
fn question_whitespace_chunks_are_not_fake_native_heads_or_abstentions() {
    let factory = factory(6);
    let prepared = factory.prepare_with_control(document("Alice   \n        \nBob"), &mut Continue).unwrap();
    assert!(prepared.chunk_count() > prepared.execution_identities().len());
    assert!(factory.prepare_with_control(document(" \n\t  "), &mut Continue).is_err());
}
