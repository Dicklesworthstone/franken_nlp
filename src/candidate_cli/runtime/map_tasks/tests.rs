//! Pinned partition/planning comparisons only. No synthetic native success.
use super::*;
use crate::{candidate_cli::map::tests::command,
    native_engine::decode::DecodeCancellationKind,
    tasks::{mapreduce::ChunkLimits, source_planning::quantized::long::SourceMapTask}};

struct Control { calls: usize, stop_at: usize }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.calls += 1;
        (self.calls >= self.stop_at).then_some(DecodeCancellationKind::Deadline)
    }
}
fn go() -> Control { Control { calls: 0, stop_at: usize::MAX } }
fn facts() -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(),
        revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
        recipe_id: "planning-fixture-not-an-artifact".to_owned(),
        source_root_sha256: "ab".repeat(32), logical_model_sha256: "cd".repeat(32) }
}
fn assets(command: &MapCommand) -> (SourceTaskPlanner, ExecutionIdentity, TaskBudget, SourceMapTask,
    Int8SourceMapCapacity, Int8SourceMapLimits) {
    let (_, limits) = command.validate().unwrap();
    let controls = pinned_controls::pinned().unwrap();
    let p = SourceTaskPlanner::pinned(controls.template_controls(), 166_101).unwrap();
    let id = source_identity(&facts(), &p, command.kind().unwrap()).unwrap();
    let b = command.host.task_budget(limits); let task = command.map_task(None).unwrap();
    let cap = p.int8_map_capacity_with_control(&task, b, &PlanContext::new(&id, b).unwrap(),
        command.host.planning(), &mut go()).unwrap();
    let mapping = command.mapping(cap).unwrap();
    (p, id, b, task, cap, mapping)
}
#[test]
fn long_document_preflight_equals_actual_native_plans_for_each_task() {
    for name in ["ner", "keyphrases", "summarize"] {
        let cmd = command(name, &[]); let (p, id, b, task, cap, mapping) = assets(&cmd);
        let text = "Alice <tool_call> é 上海 😀\r\n".repeat(100);
        assert!(text.len() > cmd.host.context_tokens);
        let expected = preflight(&text, &p, &cmd, cap, mapping, b, &mut go()).unwrap();
        let ctx = PlanContext::new(&id, b).unwrap();
        let actual = p.plan_int8_map_with_control(&text, &task, b, &ctx,
            cmd.host.planning(), mapping, &mut go()).unwrap();
        assert!(actual.chunk_count() > 1);
        assert_eq!(expected.chunks, actual.chunk_count());
        assert_eq!(expected.work, actual.planned_work());
        assert_eq!(expected.masks, actual.reserved_mask_visits());
        assert_eq!(expected.source_bytes, text.len());
        assert_eq!(expected.source_scalars, text.chars().count());
        for identity in actual.execution_identities() {
            assert_eq!(identity.logical_model_digest.to_hex(), facts().logical_model_sha256);
            assert_eq!(identity.task_spec, cmd.kind().unwrap().spec().identity());
        }
    }
}
#[test]
fn automatic_limits_preserve_every_source_byte_and_crlf_boundary() {
    let cmd = command("ner", &["--max-chunk-bytes", "16"]);
    let (p, _, b, _, cap, mapping) = assets(&cmd);
    let text = "é <think>\r\n😀 e\u{301} 上海\r\n".repeat(8);
    let chunks = ChunkPlan::build(&text, mapping.chunks, |text| {
        p.source_encoder().encode(text, 16, 16).map(|s| s.total_token_count()).map_err(|_| MapReduceError::Tokenizer)
    }).unwrap();
    assert_eq!(chunks.chunks().iter().map(|c| c.text()).collect::<String>(), text);
    let mut byte_end = 0; let mut scalar_end = 0;
    for c in chunks.chunks() {
        let span = c.span();
        assert_eq!((span.byte_start, span.scalar_start), (byte_end, scalar_end));
        assert_eq!(&text[span.byte_start..span.byte_end], c.text());
        assert!(c.tokens() <= mapping.chunks.effective_token_limit().unwrap());
        assert!(!(text.as_bytes().get(span.byte_end.wrapping_sub(1)) == Some(&b'\r')
            && text.as_bytes().get(span.byte_end) == Some(&b'\n')));
        byte_end = span.byte_end; scalar_end = span.scalar_end;
    }
    assert_eq!((byte_end, scalar_end), (text.len(), text.chars().count()));
    let result = preflight(&text, &p, &cmd, cap, mapping, b, &mut go()).unwrap();
    assert_eq!(result.chunks, chunks.chunks().len());
}
#[test]
fn exceeding_chunk_cardinality_refuses_the_entire_document() {
    let cmd = command("ner", &["--max-chunks", "1"]);
    let (p, _, b, _, cap, mapping) = assets(&cmd);
    let source = "a".repeat(cap.max_source_tokens() + 1);
    assert!(preflight(&source, &p, &cmd, cap, mapping, b, &mut go()).is_err());
}
#[test]
fn one_less_than_actual_work_or_mask_allowance_refuses_before_loading() {
    let mut cmd = command("ner", &[]); let (p, _, b, _, cap, mapping) = assets(&cmd);
    let text = "Alice\r\n".repeat(100);
    let expected = preflight(&text, &p, &cmd, cap, mapping, b, &mut go()).unwrap();
    cmd.max_forward_positions = expected.work.forward_positions - 1;
    assert!(preflight(&text, &p, &cmd, cap, mapping, b, &mut go()).is_err());
    cmd.max_forward_positions = expected.work.forward_positions;
    cmd.max_total_mask_node_visits = expected.masks - 1;
    assert!(preflight(&text, &p, &cmd, cap, mapping, b, &mut go()).is_err());
    cmd.max_total_mask_node_visits = expected.masks;
    assert!(preflight(&text, &p, &cmd, cap, mapping, b, &mut go()).is_ok());
}
#[test]
fn empty_oversized_and_cancelled_documents_have_no_preflight_success() {
    let cmd = command("ner", &[]); let (p, _, b, _, cap, mapping) = assets(&cmd);
    assert!(preflight("", &p, &cmd, cap, mapping, b, &mut go()).is_err());
    let too_big = "a".repeat(cmd.host.max_input_bytes + 1);
    assert!(preflight(&too_big, &p, &cmd, cap, mapping, b, &mut go()).is_err());
    let text = "é上海\r\n".repeat(50); let mut count = go();
    preflight(&text, &p, &cmd, cap, mapping, b, &mut count).unwrap();
    for stop_at in [1, 2, count.calls] {
        assert!(preflight(&text, &p, &cmd, cap, mapping, b,
            &mut Control { calls: 0, stop_at }).is_err());
    }
}
#[test]
fn caller_chunk_byte_ceiling_is_retained_without_guessing_source_tokens() {
    let cmd = command("ner", &["--max-chunk-bytes", "2048"]);
    let (_, _, _, _, cap, mapping) = assets(&cmd);
    assert_eq!(mapping.chunks.max_chunk_bytes, 2048);
    assert_eq!(mapping.chunks.reserved_tokens, cap.reserved_tokens());
    assert_eq!(mapping.chunks.max_chunk_tokens, cap.max_source_tokens());
    let tiny = ChunkLimits { context_tokens: cap.reserved_tokens(), ..mapping.chunks };
    assert!(cap.constrain_chunks(tiny).is_err());
}
