//! Actual source validation and pinned pair plans; no synthetic neural results.
use super::*;
use crate::candidate_cli::resolve::tests::{command, input};
use crate::native_engine::decode::{DecodeStepControl, DecodeCancellationKind};
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn facts() -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(), revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
        recipe_id: "fixture-only".to_owned(), source_root_sha256: "ab".repeat(32), logical_model_sha256: "cd".repeat(32) }
}
#[test]
fn every_candidate_has_both_orders_and_native_identity_is_resolve_not_classify() {
    let cmd = command(&[]); let (_, limits) = cmd.validate().unwrap(); let source = cmd.input(&input("Alice")).unwrap();
    let graph = ResolutionPlan::prepare(&source.documents, source.options, cmd.graph(), &mut Continue).unwrap();
    let (p,id) = planner(&facts()).unwrap(); assert_eq!(id.task_spec,"resolve-v1");
    let prepared = p.prepare_int8(&graph, &id, cmd.scoring(limits), &mut Continue).unwrap();
    assert_eq!(prepared.pair_count(),1);
    assert_eq!(prepared.planned_work().projected_logits, 8 * crate::native_engine::lmhead::NANBEIGE_VOCAB_SIZE as u64);
    cmd.host.admit_plan(prepared.required_context_tokens(), prepared.planned_work()).unwrap();
    for identity in prepared.execution_identities() {
        assert_eq!(identity.task_spec, "resolve-v1"); assert_eq!(identity.logical_model_digest,id.logical_model_digest);
        assert_ne!(identity.prompt_digest,id.prompt_digest);
    }
    assert!(prepared.finalize_without_model(&mut Continue).is_err());
}
#[test]
fn empty_candidate_graph_finalizes_singletons_with_no_neural_work() {
    let cmd = command(&[]); let (_, limits) = cmd.validate().unwrap(); let source = cmd.input(&input("Bob")).unwrap();
    let graph = ResolutionPlan::prepare(&source.documents, source.options, cmd.graph(), &mut Continue).unwrap();
    let (p,id) = planner(&facts()).unwrap();
    let prepared = p.prepare_int8(&graph, &id, cmd.scoring(limits), &mut Continue).unwrap();
    assert_eq!(prepared.pair_count(),0);
    let result = prepared.finalize_without_model(&mut Continue).unwrap();
    let expected = Expected { documents:2, mentions:2, pairs:0, work:Int8Work::default() };
    check_result(&result,expected).unwrap();
    assert!(!result.model_evaluated); assert_eq!(result.head_count,0);
    assert_eq!(result.result.clusters.len(),2); assert!(result.result.judgments.is_empty());
    assert!(check_result(&result,Expected { pairs:1,..expected }).is_err());
}
#[test]
fn invalid_anchors_duplicate_ids_and_incomplete_pair_limits_fail_before_weights() {
    let cmd = command(&[]); let original: serde_json::Value = serde_json::from_str(&input("Alice")).unwrap();
    for field in ["byte_start","byte_end","scalar_start","scalar_end"] {
        let mut changed = original.clone(); changed["documents"][0]["mentions"][0]["span"][field] = serde_json::json!(999);
        let source = cmd.input(&changed.to_string()).unwrap();
        assert!(ResolutionPlan::prepare(&source.documents,source.options,cmd.graph(),&mut Continue).is_err());
    }
    let mut changed = original; changed["documents"][1]["id"] = serde_json::json!("a");
    let source = cmd.input(&changed.to_string()).unwrap();
    assert!(ResolutionPlan::prepare(&source.documents,source.options,cmd.graph(),&mut Continue).is_err());
    let zero = command(&["--max-pairs","0"]); let source = zero.input(&input("Alice")).unwrap();
    assert!(ResolutionPlan::prepare(&source.documents,source.options,zero.graph(),&mut Continue).is_err());
}
#[test]
fn every_whole_snapshot_work_axis_can_prevent_native_admission() {
    let (p,id) = planner(&facts()).unwrap();
    for flag in ["--max-forward-positions","--max-projected-logits","--max-attention-pairs","--max-dot-products","--max-multiply-accumulates"] {
        let cmd = command(&[flag,"1"]); let (_, limits) = cmd.validate().unwrap(); let source = cmd.input(&input("Alice")).unwrap();
        let graph = ResolutionPlan::prepare(&source.documents,source.options,cmd.graph(),&mut Continue).unwrap();
        assert!(p.prepare_int8(&graph,&id,cmd.scoring(limits),&mut Continue).is_err(),"{flag}");
    }
}
#[test]
fn cancellation_applies_to_source_verification_and_pair_preparation() {
    struct Stop;
    impl DecodeStepControl for Stop { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::Deadline) } }
    let cmd = command(&[]); let (_, limits) = cmd.validate().unwrap(); let source = cmd.input(&input("Alice")).unwrap();
    assert!(ResolutionPlan::prepare(&source.documents,source.options,cmd.graph(),&mut Stop).is_err());
    let graph = ResolutionPlan::prepare(&source.documents,source.options,cmd.graph(),&mut Continue).unwrap();
    let (p,id) = planner(&facts()).unwrap();
    assert!(p.prepare_int8(&graph,&id,cmd.scoring(limits),&mut Stop).is_err());
}
