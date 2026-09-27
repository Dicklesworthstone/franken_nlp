//! Real pinned preparation and genuine empty finalization, without model weights.
use super::*;
use crate::{candidate_cli::resolve::tests::command,
    native_engine::decode::{DecodeStepControl, DecodeCancellationKind}};
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn facts() -> ArtifactIdentity {
    ArtifactIdentity { model_id:"Nanbeige4.2-3B".to_owned(),revision:"f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
        recipe_id:"model-free-fixture".to_owned(),source_root_sha256:"ab".repeat(32),logical_model_sha256:"cd".repeat(32) }
}
fn request(empty: bool) -> String {
    serde_json::json!({"documents":if empty { vec![] } else { vec![serde_json::json!({"id":"a","text":"é Alice"}),
        serde_json::json!({"id":"b","text":"Alice <tool_call>"})] },
        "options":{"blocking":"ascii_word_overlap","context_scalars":32,"minimum_margin_milli":1000}}).to_string()
}
fn prepare(cmd: &ResolveCommand, empty: bool) -> Result<PreparedInt8EntityCorpus, CandidateError> {
    let (_,limits) = cmd.validate()?; let raw = cmd.discovery.input(cmd,&request(empty))?;
    let (source,_) = super::super::super::source_tasks::planner()?;
    let a = super::super::super::source_tasks::source_identity(&facts(),&source,BuiltInTask::Ner)?;
    let (resolver,b) = super::super::planner(&facts())?;
    let config = cmd.discovery.config(cmd,limits,raw.ner,raw.options);
    entities_int8::prepare_int8_entities(raw.documents,Arc::new(source),a,Arc::new(resolver),b,config,&mut Continue)
        .map_err(|_| CandidateError::Planning)
}
#[test]
fn raw_cli_builds_both_real_planners_and_preflights_every_ner_pass() {
    let cmd = command(&["--discover-entities","--max-ner-tokens","64"]);
    let prepared = prepare(&cmd,false).unwrap();
    assert_eq!(prepared.document_count(),2); assert_eq!(prepared.source_identity().task_spec,"ner-v1");
    assert_eq!(prepared.resolution_identity().task_spec,"resolve-v1");
    assert_eq!(prepared.source_identity().logical_model_digest,prepared.resolution_identity().logical_model_digest);
    assert_eq!(prepared.ner_reserved_work().projected_logits,2*64*166_144);
    cmd.host.admit_plan(prepared.required_ner_context_tokens(),prepared.ner_reserved_work()).unwrap();
    assert!(prepared.finalize_without_model(&mut Continue).is_err());
}
#[test]
fn cli_empty_snapshot_is_zero_work_and_complete_result_checks_reject_substitution() {
    let cmd = command(&["--discover-entities"]); let prepared = prepare(&cmd,true).unwrap(); let expected = Expected::of(&prepared);
    let result = prepared.finalize_without_model(&mut Continue).unwrap(); check_result(&result,expected).unwrap();
    assert_eq!(result.model_work,Int8Work::default()); assert_eq!(result.resolution.head_count,0);
    assert!(check_result(&result,Expected { documents:1,..expected }).is_err());
    let json = crate::canonjson::canonical_string(&result).unwrap();
    assert!(!json.contains("prompt_digest")); assert!(!json.contains("generated_token_ids"));
}
#[test]
fn any_whole_model_axis_can_refuse_all_ner_preflight_before_weight_loading() {
    for flag in ["--max-forward-positions","--max-projected-logits","--max-attention-pairs","--max-dot-products","--max-multiply-accumulates"] {
        let cmd = command(&["--discover-entities",flag,"1"]);
        assert!(prepare(&cmd,false).is_err(),"{flag}");
    }
}
