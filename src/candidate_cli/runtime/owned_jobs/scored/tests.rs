//! Host configuration and real pinned durable planning, without weights or job IO.
use super::*;
use crate::{batch::BatchDocument,
    candidate_cli::jobs::scored::tests::args,
    jobs::runner::scored::{Int8ClassificationJobPlanner, Int8SentimentJobPlanner},
    native_engine::decode::{DecodeCancellationKind, DecodeStepControl}};
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn facts() -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(), revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
        recipe_id: "scored-job-planning-fixture".to_owned(), source_root_sha256: "ab".repeat(32), logical_model_sha256: "cd".repeat(32) }
}
const LABELS: &str = r#"{"labels":[{"id":"legal","description":"law"},{"id":"other"}],"mode":"multi_label"}"#;
fn corpus(task: &str) -> Corpus {
    let a = args(task, &[]); let (_, limits) = a.validate().unwrap();
    let defaults = a.parse_defaults((task == "classify").then_some(LABELS), a.host.budget(limits)).unwrap();
    scored_batch::configure_scored(a.kind().unwrap(), &a.host, &facts(), limits, defaults).unwrap()
}
#[test]
fn classification_cli_settings_reach_the_real_durable_multihead_compiler() {
    let Corpus::Classify(p, c) = corpus("classify") else { panic!("classification") };
    let base = c.identity.clone();
    let factory = Int8ClassificationJobPlanner::new(&p, c.identity, c.task_ceiling, c.planning,
        c.defaults, c.max_model_work).unwrap();
    let plan = factory.prepare_with_control(BatchDocument { id: "a".to_owned(),
        text: "legal é <tool_call> 上海".to_owned(), task_args: None }, &mut Continue).unwrap();
    assert_eq!(plan.head_count(), 2);
    assert_eq!(plan.execution_identity().task_spec, "classify-v1");
    assert_eq!(plan.execution_identity().logical_model_digest, base.logical_model_digest);
    assert_eq!(plan.execution_identity().template_digest, base.template_digest);
    assert_ne!(plan.execution_identity().prompt_digest, base.prompt_digest);
    assert!(plan.model_work().projected_logits > 0);
}
#[test]
fn sentiment_cli_settings_reach_the_real_complete_axis_compiler() {
    let Corpus::Sentiment(p, c) = corpus("sentiment") else { panic!("sentiment") };
    let base = c.identity.clone(); let factory = Int8SentimentJobPlanner::new(&p, c).unwrap();
    let plan = factory.prepare_with_control(BatchDocument { id: "a".to_owned(),
        text: "A welcome result. é 上海".to_owned(), task_args: None }, &mut Continue).unwrap();
    assert_eq!(plan.head_count(), crate::tasks::sentiment::SentimentAxis::ALL.len());
    assert_eq!(plan.execution_identity().task_spec, "sentiment-v1");
    assert_eq!(plan.execution_identity().logical_model_digest, base.logical_model_digest);
    assert_eq!(plan.execution_identity().template_digest, base.template_digest);
    assert!(plan.model_work().projections.multiply_accumulates > 0);
}
fn lifetime() -> JobLimits {
    JobLimits { max_items: 10, max_id_bytes: 128, max_input_bytes_per_item: 65536,
        max_snapshot_bytes: 1 << 20, max_result_bytes: 1 << 20, max_spool_bytes: 16 << 20,
        max_materialized_bytes: 16 << 20, max_journal_bytes: 1 << 20, max_attempts: 20,
        max_work: JobWork { model: args("classify", &[]).host.work_ceiling(), mask_node_visits: 0 } }
}
#[test]
fn native_result_ceiling_must_fit_immutable_job_limits_before_any_io() {
    let a = args("classify", &[]); let mut l = lifetime(); l.validate().unwrap();
    result_ceiling(&a, l).unwrap();
    l.max_result_bytes -= 1;
    assert_eq!(result_ceiling(&a, l).unwrap_err().code, "score_job_result_ceiling");
}
#[test]
fn metadata_report_retains_all_lifetime_counters_without_private_scores() {
    let limits = lifetime(); let id = JobId([9; 16]);
    let progress = JobProgress { job_id: id, items: 2, committed: 2, attempts: 3,
        reserved_work: limits.max_work, spool_bytes: 4096, materialized: true };
    let report = progress_report(progress, id, limits, true,
        RunMode::Resume { discard_uncommitted: false }, "classify").unwrap();
    let value = serde_json::to_value(report).unwrap();
    assert_eq!(value["attempts"], 3); assert_eq!(value["committed"], 2);
    assert_eq!(value["reserved_work"]["mask_node_visits"], 0);
    assert_eq!(value["reserved_work"]["model"]["projections"]["multiply_accumulates"],
        limits.max_work.model.projections.multiply_accumulates);
    let text = value.to_string();
    for private in ["labels", "task_args", "dimensions", "candidate_weight", "source_text", "private legal category"] {
        assert!(!text.contains(private));
    }
}
