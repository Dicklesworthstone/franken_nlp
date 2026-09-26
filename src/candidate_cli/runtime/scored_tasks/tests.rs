//! Real pinned planning without model weights; no simulated inference success.
use super::*;
use crate::{candidate_cli::scored::tests::command,
    native_engine::decode::DecodeCancellationKind};
struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
struct Stop;
impl DecodeStepControl for Stop {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::Deadline) }
}
fn facts() -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(), revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
        recipe_id: "metadata-only-unit-fixture".to_owned(), source_root_sha256: "ab".repeat(32),
        logical_model_sha256: "cd".repeat(32) }
}
fn input(c: &ScoredCommand, json: &str) -> Request {
    let (_, limits) = c.args.common().unwrap();
    command::request(c.kind, json, c.args.budget(limits), c.args.max_input_bytes).unwrap()
}
fn identity(p: &Prepared) -> &ExecutionIdentity {
    match p { Prepared::Classify(p) => p.execution_identity(), Prepared::Sentiment(p) => p.execution_identity() }
}
#[test]
fn both_classification_modes_and_sentiment_compile_before_model_loading() {
    for (name, json, count) in [
        ("classify", r#"{"document":"é <tool_call> 上海","labels":[{"id":"a"},{"id":"b"}]}"#, 1),
        ("classify", r#"{"document":"é <tool_call> 上海","labels":[{"id":"a"},{"id":"b"}],"mode":"multi_label"}"#, 2),
        ("sentiment", r#"{"document":"é <tool_call> 上海"}"#, 4)] {
        let c = command(name, &[]); let request = input(&c, json);
        let prepared = prepare(&request, &facts(), &c.args, &mut Continue).unwrap();
        let id = identity(&prepared);
        assert_eq!(id.task_spec, format!("{name}-v1"));
        assert_eq!(id.logical_model_digest.to_hex(), facts().logical_model_sha256);
        assert_eq!(id.quant_recipe, facts().recipe_id);
        assert_eq!(id.numerics_profile, NumericsProfile::StrictQuantized { version: 1 });
        assert_ne!(id.prompt_digest, Sha256Digest::of_bytes(b"null"));
        assert_ne!(id.taskir_digest, Sha256Digest::of_bytes(b"null"));
        assert_ne!(id.decision_policy_digest, Sha256Digest::of_bytes(b"null"));
        match prepared {
            Prepared::Classify(p) => { assert_eq!(p.head_count(), count); p.verify_identity(p.execution_identity()).unwrap(); },
            Prepared::Sentiment(p) => { assert_eq!(p.head_count(), count); p.verify_identity(p.execution_identity()).unwrap(); },
        }
    }
}
#[test]
fn small_complete_work_ceiling_refuses_the_plan_without_loading_weights() {
    for name in ["classify", "sentiment"] {
        let c = command(name, &["--max-multiply-accumulates", "1"]);
        let json = if name == "classify" { r#"{"document":"x","labels":[{"id":"a"},{"id":"b"}]}"# }
            else { r#"{"document":"x"}"# };
        assert!(matches!(prepare(&input(&c, json), &facts(), &c.args, &mut Continue), Err(CandidateError::Planning)));
    }
}
#[test]
fn cancelled_preparation_never_falls_through_to_native_or_model_loading() {
    let c = command("sentiment", &[]); let request = input(&c, r#"{"document":"x"}"#);
    assert!(matches!(prepare(&request, &facts(), &c.args, &mut Stop), Err(CandidateError::Planning)));
}
#[test]
fn duplicate_budget_fields_are_not_a_route_around_the_host_ceiling() {
    let c = command("classify", &[]); let (_, limits) = c.args.common().unwrap();
    for json in [r#"{"document":"x","labels":[{"id":"a"},{"id":"b"}],"budget":{"max_input_tokens":9999999}}"#,
        r#"{"document":"x","labels":[{"id":"a"},{"id":"b"}],"numerics_profile":"hf-bf16-eager"}"#] {
        assert!(command::request(c.kind, json, c.args.budget(limits), c.args.max_input_bytes).is_err());
    }
}
#[test]
fn sentiment_axis_and_threshold_policy_are_bound_to_different_identities() {
    let c = command("sentiment", &[]);
    let all = prepare(&input(&c, r#"{"document":"x"}"#), &facts(), &c.args, &mut Continue).unwrap();
    for json in [r#"{"document":"x","axes":["valence"]}"#,
        r#"{"document":"x","policy":{"minimum_peak_weight_ppm":800000,"maximum_normalized_entropy_ppm":1000000}}"#] {
        let other = prepare(&input(&c, json), &facts(), &c.args, &mut Continue).unwrap();
        assert_ne!(identity(&all), identity(&other));
        assert_eq!(identity(&all).logical_model_digest, identity(&other).logical_model_digest);
    }
}
