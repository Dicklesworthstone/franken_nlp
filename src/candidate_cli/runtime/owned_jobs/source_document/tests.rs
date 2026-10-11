//! Real pinned source planning at the CLI boundary, not native inference.
use super::*;
use crate::{canonjson, batch::BatchDocument, jobs::runner::source::document::SourceDocumentArgs};

struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn facts() -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".into(), revision: "fixture".into(),
        recipe_id: "nanbeige42-int8-v1".into(), source_root_sha256: "0".repeat(64), logical_model_sha256: "1".repeat(64) }
}
fn args(task: &str) -> JobArgs {
    let mut a = command::tests::args(task, &["--source-document-config", "d.json", "--preparation-mib", "1024"]);
    a.host.context_tokens = 8192; a.host.max_new_tokens = 64; a.host.memory_mib = 16384; a
}
fn lifetime() -> JobLimits { let mut l = command::tests::lifetime(); l.max_result_bytes = 4 << 20; l }
fn settings(reserve: u64) -> String {
    serde_json::json!({"schema_version":1,"max_chunks":8,"max_chunk_bytes":8,"max_tokenizer_calls":1024,
        "max_result_bytes":4 << 20,"max_live_value_bytes":16 << 20,"max_total_value_bytes":64 << 20,
        "reduction_reserve_mib":reserve}).to_string()
}
fn defaults(a: &JobArgs, l: Limits) -> Defaults {
    command::load_defaults(a, a.host.task_budget(l), None, None).unwrap()
}
fn factory(a: &JobArgs, reserve: u64, defaults: Defaults) -> Int8SourceDocumentJobPlanner {
    let (_, limits) = a.validate().unwrap();
    let (prepared, _) = prepare(a, &facts(), defaults, &settings(reserve), lifetime(), limits, &mut Continue).unwrap();
    let PreparedJob::SourceDocument { planner, reduction_reserve_bytes } = prepared else { panic!("wrong source document route") };
    assert_eq!(reduction_reserve_bytes, reserve * MIB); planner
}

#[test]
fn all_cli_source_families_reach_the_sealed_complete_document_factory() {
    for task in ["ner", "keyphrases", "summarize"] {
        let a = args(task); let (_, limits) = a.validate().unwrap();
        let p = factory(&a, 64, defaults(&a, limits));
        assert_eq!(p.execution_identity().task_spec, format!("{task}-v1"));
        assert_eq!(p.document_limits().max_model_work, lifetime().max_work.model);
        assert!(p.document_limits().chunks.reserved_tokens > a.host.max_new_tokens);
        let prepared = p.prepare_with_control(BatchDocument::<SourceDocumentArgs> {
            id: "original".into(), text: "é Alice Bob 上海".into(), task_args: None }, &mut Continue).unwrap();
        assert!(prepared.chunk_count() > 1);
        assert!(prepared.planned_work().fits(lifetime().max_work));
        assert_eq!(prepared.planned_work().mask_node_visits,
            a.host.max_mask_node_visits * prepared.chunk_count() as u64);
        assert!(prepared.execution_identities().iter().all(|id| id.task_spec == p.execution_identity().task_spec));
    }
}
#[test]
fn fixed_task_defaults_are_required_and_missing_configuration_cannot_use_single_context() {
    let a = args("ner"); let (_, limits) = a.validate().unwrap();
    let error = super::super::prepare(&a, &facts(), a.host.task_budget(limits), defaults(&a, limits),
        command::native_limits(&a, lifetime()).unwrap(), &mut Continue).err().unwrap();
    assert_eq!(error.code, "source_document_job_configuration_missing");
    for defaults in [Defaults::Source(None), Defaults::Extract(None)] {
        let error = prepare(&a, &facts(), defaults, &settings(64), lifetime(), limits, &mut Continue).err().unwrap();
        assert_eq!(error.code, "source_document_job_requires_fixed_task");
    }
    let other = args("keyphrases");
    let error = prepare(&a, &facts(), defaults(&other, limits), &settings(64), lifetime(), limits, &mut Continue).err().unwrap();
    assert_eq!(error.code, "source_document_job_task_mismatch");
}
#[test]
fn semantic_task_options_are_frozen_but_host_memory_headroom_does_not_renew_work() {
    let a = args("keyphrases"); let (_, limits) = a.validate().unwrap();
    let first = factory(&a, 64, defaults(&a, limits));
    let other_memory = factory(&a, 65, defaults(&a, limits));
    assert_eq!(canonjson::canonical_bytes(first.job_recipe()).unwrap(),
        canonjson::canonical_bytes(other_memory.job_recipe()).unwrap());
    let Defaults::Source(Some(SourceBatchArgs::Keyphrases { mut options, budget })) = defaults(&a, limits)
        else { panic!("keyphrase defaults") };
    options.max_phrases += 1;
    let changed = factory(&a, 64, Defaults::Source(Some(SourceBatchArgs::Keyphrases { options, budget })));
    assert_ne!(canonjson::canonical_bytes(first.job_recipe()).unwrap(),
        canonjson::canonical_bytes(changed.job_recipe()).unwrap());
    first.check_job_limits(lifetime()).unwrap(); other_memory.check_job_limits(lifetime()).unwrap();
}
#[test]
fn cancellation_is_preserved_before_configuration_and_inside_factory_error_mapping() {
    struct Stop;
    impl DecodeStepControl for Stop { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        Some(DecodeCancellationKind::Deadline)
    } }
    let a = args("ner"); let (_, limits) = a.validate().unwrap();
    let error = prepare(&a, &facts(), Defaults::Source(None), "private malformed config", lifetime(), limits, &mut Stop)
        .err().unwrap();
    assert_eq!(error.code, "job_execution_budget");
    let error = planner_failure(crate::batch::BatchFault::cancelled(DecodeCancellationKind::Deadline));
    assert_eq!(error.exit, ErrorCode::BudgetOrTimeout); assert_eq!(error.code, "job_execution_budget");
}
