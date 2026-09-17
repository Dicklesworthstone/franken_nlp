use super::*;
use std::{cell::Cell, io::{self, Cursor}, rc::Rc};
use crate::{corpus::resolve::{BidirectionalScores, PairLogProbabilities, ResolutionResult},
    native_engine::decode::DecodeCancellationKind, tasks::ner::{EntityType, NamedEntity},
    validation::grounded_fields::VerifiedSourceSpan};
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn doc(id: &str) -> ResolutionDocument {
    ResolutionDocument { id: id.to_owned(), text: "Smith".to_owned(), mentions: vec![MentionInput {
        entity_type: "person".to_owned(), surface: "Smith".to_owned(), span: VerifiedSourceSpan {
            byte_start: 0, byte_end: 5, scalar_start: 0, scalar_end: 5 } }] }
}
fn line(id: &str) -> String { format!("{}\n", serde_json::to_string(&doc(id)).unwrap()) }
struct Synthetic { calls: usize }
impl SnapshotResolver for Synthetic {
    type Output = ResolutionResult; type Error = ResolveError;
    fn resolve<C: DecodeStepControl>(&mut self, plan: &ResolutionPlan<'_>, control: &mut C) -> Result<Self::Output, Self::Error> {
        self.calls += 1;
        let p = PairLogProbabilities { same: -0.1, different: -4.0, uncertain: -5.0 };
        plan.finalize(plan.pairs().map(|t| t.finish(BidirectionalScores { forward: p, reverse: p })).collect(), control)
    }
}
fn run(input: &[u8], output: &mut Vec<u8>, resolver: &mut Synthetic, limits: ResolutionStreamLimits)
    -> Result<ResolutionStreamSummary, ResolutionStreamError<ResolveError>> {
    run_ndjson(&mut Cursor::new(input), output, resolver, ResolveOptions::default(), ResolveLimits::default(), limits, &mut Continue)
}
fn ner(source: &str, surface: &str) -> NerResult {
    let spans = scan_occurrences(source, surface, &mut GroundingBudget::default()).unwrap();
    NerResult { schema_version: 1, task_spec_version: NER_TASK_VERSION.to_owned(), numerics_profile: HF_BF16_EAGER_PROFILE.to_owned(),
        score_space: ScoreSpace::NotComputed, grounding: ExtractionGrounding::SourceMembership,
        entities: vec![NamedEntity { text: surface.to_owned(), entity_type: EntityType::Person,
            occurrence: if spans.len() == 1 { SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous }, spans }],
        generated_token_ids: vec![1, 0], forward_positions: 2, projected_logits: 3, mask_node_visit_charge: 4 }
}
#[test]
fn native_ner_occurrences_become_separate_mentions_not_an_assumed_entity() {
    let source = "é Smith Smith"; let raw = ner(source, "Smith");
    let d = document_from_ner("x".to_owned(), source.to_owned(), raw, ResolveLimits::default(), &mut GroundingBudget::default(), &mut Continue).unwrap();
    assert_eq!(d.mentions.len(), 2); assert_eq!(d.mentions[0].span.byte_start, 3); assert_eq!(d.mentions[1].span.scalar_start, 8);
    assert_eq!(ResolutionPlan::prepare(&[d], ResolveOptions::default(), ResolveLimits::default(), &mut Continue).unwrap().candidate_count(), 1);
}
#[test]
fn duplicate_ner_proposals_are_deduplicated_only_after_proof_validation() {
    let mut raw = ner("Smith Smith", "Smith"); raw.entities.push(raw.entities[0].clone());
    let d = document_from_ner("x".to_owned(), "Smith Smith".to_owned(), raw.clone(), ResolveLimits::default(), &mut GroundingBudget::default(), &mut Continue).unwrap();
    assert_eq!(d.mentions.len(), 2);
    raw.entities[1].spans.pop();
    assert!(document_from_ner("x".to_owned(), "Smith Smith".to_owned(), raw, ResolveLimits::default(), &mut GroundingBudget::default(), &mut Continue).is_err());
}
#[test]
fn source_swap_missing_occurrence_and_forged_ner_headers_fail() {
    for mode in 0..4 {
        let mut raw = ner("Smith Smith", "Smith");
        match mode { 0 => raw.task_spec_version = "answer-v1".to_owned(), 1 => { raw.entities[0].spans.pop(); },
            2 => raw.entities[0].spans[0].scalar_start += 1, _ => raw.grounding = ExtractionGrounding::NotRequested }
        assert!(document_from_ner("x".to_owned(), "Smith Smith".to_owned(), raw, ResolveLimits::default(), &mut GroundingBudget::default(), &mut Continue).is_err());
    }
    assert!(document_from_ner("x".to_owned(), "Jones".to_owned(), ner("Smith", "Smith"), ResolveLimits::default(), &mut GroundingBudget::default(), &mut Continue).is_err());
}
#[test]
fn input_arrival_order_does_not_change_snapshot_json() {
    let mut a = Vec::new(); let mut b = Vec::new();
    run(format!("{}{}", line("b"), line("a")).as_bytes(), &mut a, &mut Synthetic { calls: 0 }, ResolutionStreamLimits::default()).unwrap();
    run(format!("{}{}", line("a"), line("b")).as_bytes(), &mut b, &mut Synthetic { calls: 0 }, ResolutionStreamLimits::default()).unwrap();
    assert_eq!(a, b);
    let value: serde_json::Value = serde_json::from_slice(&a).unwrap(); assert_eq!(value["epoch"], 1);
    assert_eq!(value["result"]["clusters"][0]["mentions"], serde_json::json!([0, 1]));
}
#[test]
fn explicit_flush_and_eof_are_distinct_complete_snapshots() {
    let input = format!("{}{}{{\"flush\":true}}\n{}", line("b"), line("a"), line("a"));
    let mut output = Vec::new(); let mut resolver = Synthetic { calls: 0 };
    let summary = run(input.as_bytes(), &mut output, &mut resolver, ResolutionStreamLimits::default()).unwrap();
    assert_eq!(summary.snapshots, 2); assert_eq!(summary.documents, 3); assert_eq!(resolver.calls, 2);
    let rows: Vec<_> = std::str::from_utf8(&output).unwrap().lines().map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap()).collect();
    assert_eq!(rows[0]["epoch"], 1); assert_eq!(rows[1]["epoch"], 2);
    assert_eq!(rows[0]["result"]["mentions"].as_array().unwrap().len(), 2);
    assert_eq!(rows[1]["result"]["mentions"].as_array().unwrap().len(), 1);
}
#[test]
fn malformed_or_duplicate_record_aborts_unpublished_epoch() {
    for bad in ["{broken}\n", "{\"flush\":true,\"other\":1}\n", "{\"id\":\"x\",\"id\":\"y\"}\n", " \n"] {
        let mut output = Vec::new(); let mut r = Synthetic { calls: 0 };
        assert!(run(format!("{}{bad}", line("a")).as_bytes(), &mut output, &mut r, ResolutionStreamLimits::default()).is_err());
        assert!(output.is_empty()); assert_eq!(r.calls, 0);
    }
    let mut output = Vec::new(); let mut r = Synthetic { calls: 0 };
    assert!(run(format!("{}{}", line("a"), line("a")).as_bytes(), &mut output, &mut r, ResolutionStreamLimits::default()).is_err());
    assert!(output.is_empty());
}
#[test]
fn crlf_blank_lines_and_unterminated_last_record_are_bounded() {
    let input = format!("\r\n{}", line("a").replace('\n', "\r\n"));
    let mut output = Vec::new(); let mut r = Synthetic { calls: 0 };
    let summary = run(input.as_bytes(), &mut output, &mut r, ResolutionStreamLimits::default()).unwrap();
    assert_eq!(summary.input_bytes, input.len() as u64); assert_eq!(summary.records, 1);
    let last = line("a"); run(last.trim_end().as_bytes(), &mut Vec::new(), &mut r, ResolutionStreamLimits::default()).unwrap();
    assert_eq!(r.calls, 2);
}
#[test]
fn line_total_input_records_and_snapshot_limits_do_not_publish_partial_corpus() {
    let input = format!("{}{}", line("a"), line("b"));
    for axis in 0..3 {
        let mut limits = ResolutionStreamLimits::default();
        match axis { 0 => limits.max_line_bytes = 3, 1 => limits.max_input_bytes = 3, _ => limits.max_records = 1 }
        let mut output = Vec::new(); assert!(run(input.as_bytes(), &mut output, &mut Synthetic { calls: 0 }, limits).is_err()); assert!(output.is_empty());
    }
    let input = format!("{}{{\"flush\":true}}\n{}{{\"flush\":true}}\n", line("a"), line("b"));
    let mut output = Vec::new(); let mut r = Synthetic { calls: 0 };
    assert!(run(input.as_bytes(), &mut output, &mut r, ResolutionStreamLimits { max_snapshots: 1, ..ResolutionStreamLimits::default() }).is_err());
    assert_eq!(r.calls, 1); assert_eq!(output.iter().filter(|&&b| b == b'\n').count(), 1);
}
#[test]
fn ner_input_shape_and_nonrenewable_verification_budget_cross_flushes() {
    let record = serde_json::json!({"id":"x","text":"Smith","ner":ner("Smith", "Smith")}).to_string();
    let input = format!("{record}\n{{\"flush\":true}}\n{record}\n");
    let limits = ResolutionStreamLimits { ner_verification: GroundingBudget { max_fields: 1, ..GroundingBudget::default() }, ..ResolutionStreamLimits::default() };
    let mut output = Vec::new(); let mut r = Synthetic { calls: 0 };
    assert!(run(input.as_bytes(), &mut output, &mut r, limits).is_err()); assert_eq!(r.calls, 1);
}
#[test]
fn native_work_reservation_never_renews_or_partially_mutates_on_failure() {
    let mut remaining = BatchWork { forward_positions: 20, projected_logits: 40 };
    let work = BatchWork { forward_positions: 10, projected_logits: 20 };
    reserve(&mut remaining, work).unwrap(); reserve(&mut remaining, work).unwrap();
    assert_eq!(remaining, BatchWork::default()); assert!(reserve(&mut remaining, work).is_err());
    let mut remaining = BatchWork { forward_positions: 10, projected_logits: 19 }; let before = remaining;
    assert!(reserve(&mut remaining, work).is_err()); assert_eq!(remaining, before);
}
struct LiveOutput { alive: Rc<Cell<bool>> }
impl Serialize for LiveOutput {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> { serializer.serialize_bool(true) }
}
impl Drop for LiveOutput { fn drop(&mut self) { self.alive.set(false); } }
struct GuardedResolver { alive: Rc<Cell<bool>>, calls: usize }
impl SnapshotResolver for GuardedResolver {
    type Output = LiveOutput; type Error = ResolveError;
    fn resolve<C: DecodeStepControl>(&mut self, _: &ResolutionPlan<'_>, _: &mut C) -> Result<LiveOutput, ResolveError> {
        self.calls += 1; self.alive.set(true); Ok(LiveOutput { alive: self.alive.clone() })
    }
}
struct FailedSink { alive: Rc<Cell<bool>>, calls: usize, fail_flush: bool }
impl Write for FailedSink {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        assert!(self.alive.get()); self.calls += 1;
        if self.fail_flush { Ok(bytes.len()) } else { Err(io::Error::other("fixture")) }
    }
    fn flush(&mut self) -> io::Result<()> { assert!(self.alive.get()); Err(io::Error::other("fixture")) }
}
#[test]
fn output_guard_survives_write_and_flush_errors_without_reading_next_epoch() {
    for fail_flush in [false, true] {
        let input = format!("{}{{\"flush\":true}}\n{}", line("a"), line("b"));
        let alive = Rc::new(Cell::new(false)); let mut r = GuardedResolver { alive: alive.clone(), calls: 0 };
        let mut sink = FailedSink { alive: alive.clone(), calls: 0, fail_flush }; let mut reader = Cursor::new(input.as_bytes());
        assert!(matches!(run_ndjson(&mut reader, &mut sink, &mut r, ResolveOptions::default(), ResolveLimits::default(),
            ResolutionStreamLimits::default(), &mut Continue), Err(ResolutionStreamError::OutputIo)));
        assert_eq!(r.calls, 1); assert_eq!(sink.calls, 1); assert!(!alive.get()); assert!(reader.position() < input.len() as u64);
    }
}
#[test]
fn result_envelope_limits_are_checked_before_any_write() {
    let mut output = Vec::new();
    assert!(run(line("a").as_bytes(), &mut output, &mut Synthetic { calls: 0 }, ResolutionStreamLimits {
        max_output_line_bytes: 16, ..ResolutionStreamLimits::default() }).is_err()); assert!(output.is_empty());
    assert!(run(line("a").as_bytes(), &mut output, &mut Synthetic { calls: 0 }, ResolutionStreamLimits {
        max_output_bytes: 1, ..ResolutionStreamLimits::default() }).is_err()); assert!(output.is_empty());
}
#[test]
fn cancelled_stream_does_not_masquerade_as_empty_success() {
    struct Cancel;
    impl DecodeStepControl for Cancel { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::Deadline) } }
    let mut output = Vec::new(); let mut r = Synthetic { calls: 0 };
    assert!(matches!(run_ndjson(&mut Cursor::new(line("a")), &mut output, &mut r, ResolveOptions::default(), ResolveLimits::default(),
        ResolutionStreamLimits::default(), &mut Cancel), Err(ResolutionStreamError::Resolution(ResolveError::Cancelled(DecodeCancellationKind::Deadline)))));
    assert!(output.is_empty()); assert_eq!(r.calls, 0);
}
