//! Real pinned CLI preparation and genuine empty finalization, without weights.
//! These tests cannot manufacture a nonempty native execution receipt.
use super::*;
use crate::{candidate_cli::resolve::tests::command,
    native_engine::{decode::{DecodeStepControl, DecodeCancellationKind}, lmhead::NANBEIGE_VOCAB_SIZE}};
struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
fn facts() -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(),
        revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
        recipe_id: "model-free-fixture".to_owned(), source_root_sha256: "ab".repeat(32),
        logical_model_sha256: "cd".repeat(32) }
}
fn request(empty: bool) -> String {
    let documents = if empty { vec![] } else {
        vec![serde_json::json!({"id":"a","text":"é Alice"}),
            serde_json::json!({"id":"b","text":"Alice <tool_call>"})]
    };
    serde_json::json!({"documents":documents,
        "options":{"blocking":"ascii_word_overlap","context_scalars":32,"minimum_margin_milli":1000}}).to_string()
}
fn prepare(cmd: &ResolveCommand, empty: bool) -> Result<PreparedInt8DocumentEntityCorpus, CandidateError> {
    let (_, limits) = cmd.validate()?;
    let raw = cmd.discovery.input(cmd, &request(empty))?;
    let (source, _) = crate::candidate_cli::runtime::source_tasks::planner()?;
    let a = crate::candidate_cli::runtime::source_tasks::source_identity(&facts(), &source, BuiltInTask::Ner)?;
    let (resolver, b) = crate::candidate_cli::runtime::resolution::planner(&facts())?;
    let base = cmd.discovery.config(cmd, limits, raw.ner, raw.options);
    let config = cmd.discovery.long.configuration(cmd, base)?;
    prepare_int8_document_entities(raw.documents, Arc::new(source), a, Arc::new(resolver), b,
        config, &mut Continue).map_err(|_| CandidateError::Planning)
}
#[test]
fn cli_preflights_real_chunk_plans_and_keeps_original_document_cardinality() {
    let cmd = command(&["--discover-entities", "--chunked", "--max-ner-tokens", "64", "--max-ner-chunk-bytes", "6"]);
    let prepared = prepare(&cmd, false).unwrap();
    assert_eq!(prepared.document_count(), 2); assert!(prepared.chunk_count() > prepared.document_count());
    assert_eq!(prepared.source_identity().task_spec, "ner-v1");
    assert_eq!(prepared.resolution_identity().task_spec, "resolve-v1");
    assert_eq!(prepared.source_identity().logical_model_digest, prepared.resolution_identity().logical_model_digest);
    assert_eq!(prepared.ner_reserved_work().projected_logits,
        prepared.chunk_count() as u64 * 64 * NANBEIGE_VOCAB_SIZE as u64);
    assert_eq!(prepared.reserved_mask_visits(), prepared.chunk_count() as u64
        * prepared.config().entities.masks.max_visits_per_item);
    cmd.host.admit_plan(prepared.required_ner_context_tokens(), prepared.ner_reserved_work()).unwrap();
    assert!(prepared.finalize_without_model(&mut Continue).is_err());
}
#[test]
fn genuine_empty_output_rejects_wrong_mode_geometry_and_reserved_work() {
    let cmd = command(&["--discover-entities", "--chunked"]);
    let prepared = prepare(&cmd, true).unwrap(); let expected = DocumentExpected::of(&prepared);
    let mut result = prepared.finalize_without_model(&mut Continue).unwrap();
    check_document_result(&result, expected).unwrap();
    assert_eq!(result.output.model_work, Int8Work::default());
    assert_eq!(result.output.resolution.head_count, 0);
    assert!(check_document_result(&result, DocumentExpected { chunks: 1, ..expected }).is_err());
    assert!(check_document_result(&result, DocumentExpected {
        base: Expected { documents: 1, ..expected.base }, ..expected }).is_err());
    result.execution = "wrong-mode";
    assert!(check_document_result(&result, expected).is_err());
    result.execution = INT8_DOCUMENT_ENTITY_EXECUTION;
    result.output.ner_reserved_work.forward_positions = 1;
    assert!(check_document_result(&result, expected).is_err());
    result.output.ner_reserved_work.forward_positions = 0;
    let json = crate::canonjson::canonical_string(&result).unwrap();
    assert!(!json.contains("prompt_digest")); assert!(!json.contains("generated_token_ids"));
    assert!(json.contains("ner_chunk_boundaries_may_split_entities"));
}
#[test]
fn all_five_native_ceilings_apply_to_chunk_discovery_before_weight_loading() {
    for flag in ["--max-forward-positions", "--max-projected-logits", "--max-attention-pairs",
        "--max-dot-products", "--max-multiply-accumulates"] {
        let cmd = command(&["--discover-entities", "--chunked", "--max-ner-tokens", "64", flag, "1"]);
        assert!(prepare(&cmd, false).is_err(), "{flag}");
    }
}
#[test]
fn actual_snapshot_chunks_and_masks_are_checked_not_just_document_lower_bounds() {
    for extra in [vec!["--max-snapshot-ner-chunks", "2"], vec!["--max-ner-chunks", "1"],
        vec!["--max-snapshot-mask-node-visits", "2000000000"]] {
        let mut flags = vec!["--discover-entities", "--chunked", "--max-ner-tokens", "64", "--max-ner-chunk-bytes", "6"];
        flags.extend(extra);
        let cmd = command(&flags); cmd.validate().unwrap();
        // Two original documents pass input's lower bound, but the exact
        // partition requires more chunks and cannot borrow a renewed budget.
        cmd.discovery.input(&cmd, &request(false)).unwrap();
        assert!(prepare(&cmd, false).is_err());
    }
}
