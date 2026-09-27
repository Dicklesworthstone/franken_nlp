//! Transport fixtures only. These do not substitute for native model runs.
use super::*;
use crate::{native_engine::{generation::GenerationWork, strict_int8::Int8Work}, tasks::chat::ChatResult};

fn facts() -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(),
        revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(), recipe_id: "stream-framing-fixture".to_owned(),
        source_root_sha256: "ab".repeat(32), logical_model_sha256: "cd".repeat(32) }
}
fn bounds(scores: bool) -> StreamBounds {
    StreamBounds { tokens: 4, content_bytes: 64, capture_logprobs: scores,
        event_bytes: 4352, terminal_bytes: 16 * 1024, stream_bytes: 256 * 1024, sink_memory_bytes: 1 << 20 }
}
fn sink(scores: bool) -> TokenSink<Vec<u8>> {
    TokenSink::new(Vec::new(), Task::Generate, bounds(scores), &facts(), 166101, None).unwrap()
}
fn event(index: usize, id: u32, bytes: &[u8], scores: bool) -> DecodeTokenEvent {
    DecodeTokenEvent { schema_version: DECODE_TOKEN_EVENT_SCHEMA_VERSION, request_seq: 1,
        token_index: index, token_id: id, decoded_bytes: bytes.to_vec(), logprob: scores.then_some(-0.5) }
}
fn push<W: Write>(s: &mut TokenSink<W>, event: DecodeTokenEvent) {
    let permit = s.reserve(&event).unwrap(); s.permit(permit, event).unwrap();
}
fn completion<W: Write>(s: &TokenSink<W>, finish: GenerationFinish) -> Int8ChatResult {
    let proposals = s.token_ids.len() + usize::from(finish == GenerationFinish::ByteLimit);
    let positions = 2 + proposals - 1;
    Int8ChatResult { schema_version: 1, result: ChatResult { schema_version: 1,
        task: s.task.to_owned(), execution: INT8_GENERATION_VERSION.to_owned(), numerics_profile: STRICT_INT8_PROFILE.to_owned(),
        request_seq: 1, sample_index: 0, content: String::from_utf8(s.bytes.clone()).unwrap(),
        token_ids: s.token_ids.clone(), finish_reason: finish, effective_seed: s.expected_seed.clone(),
        token_logprobs: s.scores.clone(), logprob_score_space: s.bounds.capture_logprobs.then_some(DecodeScoreSpace::FullVocabularyLogSoftmax),
        native_work: GenerationWork { forward_positions: positions as u64,
            projected_logits: (proposals * NANBEIGE_VOCAB_SIZE) as u64,
            sampled_steps: if s.expected_seed.is_some() { proposals as u64 } else { 0 } } },
        model_work: Int8Work::for_sequence(0, positions, proposals * NANBEIGE_VOCAB_SIZE).unwrap() }
}
fn rows(bytes: &[u8]) -> Vec<serde_json::Value> {
    std::str::from_utf8(bytes).unwrap().lines().map(|line| serde_json::from_str(line).unwrap()).collect()
}
#[test]
fn reservation_is_invisible_and_only_permit_publishes_start_and_token() {
    let mut s = sink(false); let e = event(0, 1, b"a", false);
    let permit = s.reserve(&e).unwrap();
    assert!(s.writer.is_empty()); assert!(s.token_ids.is_empty());
    s.permit(permit, e).unwrap();
    let r = rows(&s.writer); assert_eq!(r.len(), 2);
    assert_eq!(r[0]["event"], "run_start"); assert_eq!(r[1]["event"], "token");
    assert_eq!(r[1]["data"]["token_index"], 0); assert_eq!(r[1]["data"]["decoded_bytes"], serde_json::json!([97]));
    assert!(r.iter().all(|r| r["provisional"] == true));
    assert!(r.iter().all(|r| r["protocol"] == PROTOCOL && r["evidence"] == "non_authoritative"));
    assert!(!std::str::from_utf8(&s.writer).unwrap().contains("run_complete"));
}
#[test]
fn split_utf8_bytes_and_empty_eos_reconcile_without_lossy_text_conversion() {
    let mut s = sink(true);
    push(&mut s, event(0, 1, &[0xc3], true));
    push(&mut s, event(1, 2, &[0xa9], true));
    push(&mut s, event(2, 166101, b"", true));
    let result = completion(&s, GenerationFinish::Eos); s.finish(&result).unwrap();
    let r = rows(&s.writer); assert_eq!(r.len(), 5);
    assert_eq!(r[4]["event"], "run_complete"); assert_eq!(r[4]["provisional"], false);
    assert_eq!(r[4]["data"]["result"]["content"], "é");
    let bytes: Vec<u8> = r[1..4].iter().flat_map(|r| r["data"]["decoded_bytes"].as_array().unwrap())
        .map(|v| v.as_u64().unwrap() as u8).collect();
    assert_eq!(bytes, "é".as_bytes());
    assert_eq!(r[3]["data"]["decoded_bytes"], serde_json::json!([]));
    assert_eq!(r[4]["data"], serde_json::to_value(&result).unwrap());
}
#[test]
fn permits_are_exact_single_use_and_cannot_cross_identical_sinks() {
    let mut a = sink(false); let mut b = sink(false); let e = event(0, 1, b"a", false);
    let pa = a.reserve(&e).unwrap(); let pb = b.reserve(&e).unwrap();
    assert!(b.permit(pa, e.clone()).is_err()); assert!(b.writer.is_empty());
    drop(pb);
    let r = completion(&a, GenerationFinish::ByteLimit);
    assert!(a.finish(&r).is_err()); assert!(a.writer.is_empty());
    let mut s = sink(false); let p = s.reserve(&e).unwrap();
    let mut changed = e; changed.decoded_bytes[0] = b'b';
    assert!(s.permit(p, changed.clone()).is_err());
    assert!(s.reserve(&changed).is_err()); assert!(s.writer.is_empty());
}
#[test]
fn abandoned_or_overlapping_reservations_never_become_completed_output() {
    let mut s = sink(false); let e = event(0, 1, b"a", false);
    drop(s.reserve(&e).unwrap());
    assert!(s.finish(&completion(&s, GenerationFinish::ByteLimit)).is_err()); assert!(s.writer.is_empty());
    let mut s = sink(false); let p = s.reserve(&e).unwrap();
    assert!(s.reserve(&e).is_err()); assert!(s.permit(p, e).is_err()); assert!(s.writer.is_empty());
}
#[test]
fn malformed_or_oversized_events_refuse_before_any_bytes_are_visible() {
    for axis in 0..8 {
        let mut s = sink(true); let mut e = event(0, 1, b"a", true);
        match axis { 0 => e.request_seq = 2, 1 => e.schema_version += 1,
            2 => e.token_index = 1, 3 => e.token_id = NANBEIGE_VOCAB_SIZE as u32,
            4 => e.logprob = Some(f32::NAN), 5 => e.logprob = Some(0.5),
            6 => e.decoded_bytes = vec![0; 65], _ => e.logprob = None }
        assert!(s.reserve(&e).is_err(), "axis {axis}"); assert!(s.writer.is_empty());
        assert!(s.reserve(&event(0, 1, b"a", true)).is_err());
    }
    let mut s = sink(false);
    assert!(s.reserve(&event(0, 1, b"a", true)).is_err());
    let mut s = sink(false);
    assert!(s.reserve(&event(0, 166101, b"eos must be empty", false)).is_err());
}
#[test]
fn final_result_mutations_cannot_launder_a_different_stream_into_success() {
    for axis in 0..11 {
        let mut s = sink(true); push(&mut s, event(0, 1, b"a", true)); push(&mut s, event(1, 166101, b"", true));
        let before = s.writer.clone(); let mut r = completion(&s, GenerationFinish::Eos);
        match axis { 0 => r.result.content.push('b'), 1 => r.result.token_ids[0] = 2,
            2 => r.result.token_logprobs.as_mut().unwrap()[0] = -0.7,
            3 => r.result.task = "chat-v1".to_owned(), 4 => r.result.request_seq = 2,
            5 => r.result.sample_index = 1, 6 => r.result.effective_seed = Some("01".repeat(32)),
            7 => r.result.numerics_profile = "hf-bf16-eager".to_owned(),
            8 => r.result.execution = "different".to_owned(), 9 => r.schema_version += 1,
            _ => r.result.finish_reason = GenerationFinish::TokenLimit }
        assert!(s.finish(&r).is_err(), "axis {axis}"); assert_eq!(s.writer, before);
        assert!(s.finish(&completion(&s, GenerationFinish::Eos)).is_err());
    }
}
#[test]
fn content_limit_can_finish_without_tokens_but_token_limit_requires_all_tokens() {
    let mut s = sink(false); let r = completion(&s, GenerationFinish::ByteLimit);
    s.finish(&r).unwrap(); let r = rows(&s.writer);
    assert_eq!(r.len(), 2); assert_eq!(r[0]["event"], "run_start"); assert_eq!(r[1]["event"], "run_complete");
    let mut s = sink(false); push(&mut s, event(0, 1, b"a", false));
    let r = completion(&s, GenerationFinish::TokenLimit); assert!(s.finish(&r).is_err());
    let mut s = sink(false);
    for i in 0..4 { push(&mut s, event(i, 1, b"a", false)); }
    s.finish(&completion(&s, GenerationFinish::TokenLimit)).unwrap();
    let n = s.writer.len(); assert!(s.finish(&completion(&s, GenerationFinish::TokenLimit)).is_err());
    assert_eq!(s.writer.len(), n);
}
#[test]
fn eos_and_token_ceiling_close_admission_even_before_final_publication() {
    let mut s = sink(false); push(&mut s, event(0, 166101, b"", false));
    let n = s.writer.len(); assert!(s.reserve(&event(1, 1, b"a", false)).is_err()); assert_eq!(s.writer.len(), n);
    let mut s = sink(false);
    for i in 0..4 { push(&mut s, event(i, 1, b"a", false)); }
    let n = s.writer.len(); assert!(s.reserve(&event(4, 1, b"a", false)).is_err()); assert_eq!(s.writer.len(), n);
}
#[test]
fn terminal_transport_capacity_is_reserved_before_each_token() {
    let mut s = sink(false);
    let e = event(0, 1, b"a", false);
    let start = s.start().unwrap().len(); let token = s.frame("token", true, &e, s.bounds.event_bytes).unwrap().len();
    s.bounds.stream_bytes = (start + token + s.bounds.terminal_bytes) as u64 - 1;
    assert!(s.reserve(&e).is_err()); assert!(s.writer.is_empty());
    let mut s = sink(false); s.bounds.event_bytes = 10;
    assert!(s.reserve(&e).is_err()); assert!(s.writer.is_empty());
}
struct Broken {
    bytes: Vec<u8>, calls: usize, flushes: usize, fail_flush: bool,
}
impl Write for Broken {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.calls += 1;
        if !self.fail_flush && self.calls > 1 { return Err(io::Error::other("private external message")); }
        let count = if self.fail_flush { bytes.len() } else { bytes.len().min(7) };
        self.bytes.extend_from_slice(&bytes[..count]); Ok(count)
    }
    fn flush(&mut self) -> io::Result<()> { self.flushes += 1; Err(io::Error::other("private external message")) }
}
#[test]
fn partial_write_and_flush_failures_are_permanent_and_never_retried() {
    for fail_flush in [false, true] {
        let writer = Broken { bytes: Vec::new(), calls: 0, flushes: 0, fail_flush };
        let mut s = TokenSink::new(writer, Task::Generate, bounds(false), &facts(), 166101, None).unwrap();
        let e = event(0, 1, b"a", false); let p = s.reserve(&e).unwrap();
        let error = s.permit(p, e.clone()).unwrap_err(); assert_eq!(error.code, "stream_output_io");
        assert!(!error.to_string().contains("private"));
        let calls = (s.writer.calls, s.writer.flushes, s.writer.bytes.len());
        assert!(s.reserve(&e).is_err()); assert!(s.finish(&completion(&s, GenerationFinish::ByteLimit)).is_err());
        assert_eq!(calls, (s.writer.calls, s.writer.flushes, s.writer.bytes.len()));
        assert!(s.token_ids.is_empty()); assert!(s.bytes.is_empty());
    }
}
#[test]
fn seeded_chat_keeps_provenance_and_final_seed_without_exposing_prompt_identity() {
    let mut s = TokenSink::new(Vec::new(), Task::Chat, bounds(false), &facts(), 166101, Some("07".repeat(32))).unwrap();
    push(&mut s, event(0, 166101, b"", false)); s.finish(&completion(&s, GenerationFinish::Eos)).unwrap();
    let r = rows(&s.writer); assert_eq!(r[2]["data"]["result"]["effective_seed"], "07".repeat(32));
    assert!(r.iter().all(|row| row["task"] == "chat-v1" && row["source_root_sha256"] == "ab".repeat(32)));
    for row in r { assert!(row.get("prompt_digest").is_none()); assert!(row.get("input").is_none()); }
}
