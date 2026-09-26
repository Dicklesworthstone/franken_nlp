//! Actual pinned corpus planning, without loaded weights or native-success fixtures.
use super::*;
use crate::{batch::{BatchDocument, classify::ClassificationBatchArgs},
    candidate_cli::scored_batch::tests::command,
    native_engine::decode::DecodeCancellationKind,
    tasks::sentiment::{SentimentAxis, batch::{Int8SentimentBatchPlanner, SentimentBatchArgs}}};
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn facts() -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(), revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
        recipe_id: "metadata-only-test-fixture".to_owned(), source_root_sha256: "ab".repeat(32), logical_model_sha256: "cd".repeat(32) }
}
fn doc<A>() -> BatchDocument<A> { BatchDocument { id: "x".to_owned(), text: "Great café <tool_call>".to_owned(), task_args: None } }
#[test]
fn classification_configuration_and_real_item_plan_keep_the_selected_model() {
    let c = command("classify", &[]); let (_, l, _) = c.validate().unwrap(); let b = c.host.budget(l);
    let defaults = c.parse_defaults(Some(r#"{"labels":[{"id":"happy"},{"id":"sad"}]}"#), b).unwrap();
    let Corpus::Classify(p, cfg) = configure(&c, &facts(), l, defaults).unwrap() else { panic!("wrong task") };
    assert_eq!(cfg.max_model_work, c.host.work_ceiling());
    let compiler = Int8ClassificationBatchPlanner::new(&p, cfg.identity, b, cfg.planning, cfg.defaults).unwrap();
    let plan = compiler.prepare_with_control(doc::<ClassificationBatchArgs>(), &mut Continue).unwrap();
    assert_eq!(plan.head_count(), 1);
    assert_eq!(plan.execution_identity().logical_model_digest.to_hex(), facts().logical_model_sha256);
    assert_eq!(plan.execution_identity().task_spec, "classify-v1");
    assert_ne!(plan.execution_identity().prompt_digest, Sha256Digest::of_bytes(b"null"));
}
#[test]
fn sentiment_item_override_does_not_change_the_next_records_default_axes() {
    let c = command("sentiment", &[]); let (_, l, _) = c.validate().unwrap(); let b = c.host.budget(l);
    let defaults = c.parse_defaults(None, b).unwrap();
    let Corpus::Sentiment(p, cfg) = configure(&c, &facts(), l, defaults).unwrap() else { panic!("wrong task") };
    assert_eq!(cfg.max_model_work, c.host.work_ceiling());
    let compiler = Int8SentimentBatchPlanner::new(&p, cfg).unwrap();
    let mut one = doc(); one.task_args = Some(SentimentBatchArgs { axes: vec![SentimentAxis::Valence], budget: b });
    let a = compiler.prepare_with_control(one, &mut Continue).unwrap();
    let z = compiler.prepare_with_control(doc(), &mut Continue).unwrap();
    assert_eq!(a.head_count(), 1); assert_eq!(z.head_count(), 4);
    assert_ne!(a.execution_identity().taskir_digest, z.execution_identity().taskir_digest);
    assert_eq!(z.execution_identity().logical_model_digest.to_hex(), facts().logical_model_sha256);
    assert_eq!(z.execution_identity().numerics_profile, NumericsProfile::StrictQuantized { version: 1 });
}
#[test]
fn sentiment_policy_is_bound_to_the_run_not_accepted_as_an_item_override() {
    let c = command("sentiment", &[]); let (_, l, _) = c.validate().unwrap(); let b = c.host.budget(l);
    let a = c.parse_defaults(None, b).unwrap();
    let z = c.parse_defaults(Some(r#"{"policy":{"minimum_peak_weight_ppm":900000,"maximum_normalized_entropy_ppm":500000}}"#), b).unwrap();
    let Corpus::Sentiment(_, a) = configure(&c, &facts(), l, a).unwrap() else { panic!("wrong task") };
    let Corpus::Sentiment(_, z) = configure(&c, &facts(), l, z).unwrap() else { panic!("wrong task") };
    assert_ne!(a.identity.template_digest, z.identity.template_digest);
    assert!(serde_json::from_value::<SentimentBatchArgs>(serde_json::json!({
        "axes":["valence"],"budget":b,"policy":{"minimum_peak_weight_ppm":0}
    })).is_err());
}
#[test]
fn wrong_task_settings_or_model_revision_never_become_a_stream_configuration() {
    let c = command("sentiment", &[]); let (_, l, _) = c.validate().unwrap(); let b = c.host.budget(l);
    assert!(configure(&c, &facts(), l, Defaults::Classify(None)).is_err());
    let mut changed = facts(); changed.revision = "other".to_owned();
    assert!(configure(&c, &changed, l, c.parse_defaults(None, b).unwrap()).is_err());
}

#[test]
fn an_item_allowance_cannot_exceed_the_whole_sentiment_stream_allowance() {
    let c = command("sentiment", &[]); let (_, l, _) = c.validate().unwrap(); let b = c.host.budget(l);
    let defaults = c.parse_defaults(None, b).unwrap();
    let Corpus::Sentiment(p, mut cfg) = configure(&c, &facts(), l, defaults).unwrap() else { panic!("wrong task") };
    cfg.max_item_work.attention_pairs = cfg.max_model_work.attention_pairs + 1;
    assert!(cfg.validate(&p).is_err());
    cfg.max_item_work = cfg.max_model_work;
    cfg.max_item_work.projections.multiply_accumulates += 1;
    assert!(cfg.validate(&p).is_err());
}
