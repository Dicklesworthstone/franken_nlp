//! Cryptographic/format/population tests only. No fixture is native inference.
use super::*;
use crate::{execution_identity::{ExecutionIdentity, Sha256Digest, NumericsProfile, ThinkingMode, ToolMode},
    native_engine::{constrained_int8, decode::{DecodeCancellationKind, DecodeStepControl}}};
use std::io::Cursor;

pub(super) fn error<T>(value: Result<T, JobError>) -> JobError {
    match value { Err(error) => error, Ok(_) => panic!("expected job refusal") }
}
#[derive(Default)]
pub(super) struct Control { pub calls: u64, pub stop_at: Option<u64> }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.calls += 1;
        self.stop_at.filter(|&n| self.calls >= n).map(|_| DecodeCancellationKind::Deadline)
    }
}
pub(super) fn key() -> JobSecret { JobSecret::from_bytes([17; 32]) }
pub(super) fn work() -> JobWork {
    JobWork { model: constrained_int8::planned_work(128, 16).unwrap(), mask_node_visits: 1000 }
}
pub(super) fn limits() -> JobLimits {
    let mut cap = JobWork::default(); for _ in 0..16 { cap = cap.checked_add(work()).unwrap(); }
    JobLimits { max_items: 16, max_id_bytes: 128, max_input_bytes_per_item: 1 << 20,
        max_snapshot_bytes: 16 << 20, max_result_bytes: 16384, max_spool_bytes: 1 << 20,
        max_materialized_bytes: 1 << 20, max_journal_bytes: 32 << 20, max_attempts: 16, max_work: cap }
}
pub(super) fn identity() -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"owned-job-storage-fixture");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "fixture".to_owned(), packing_set_digest: d,
        tokenizer_digest: d, template_digest: d, task_spec: "job-fixture-v1".to_owned(), taskir_digest: d, prompt_digest: d,
        grammar_compiler_version: "none".to_owned(), schema_digest: d, numerics_profile: NumericsProfile::DiagnosticF32,
        kv_dtype: "bf16".to_owned(), sampler_version: "fixture-addressed-v1".to_owned(), thinking_mode: ThinkingMode::Disabled,
        tool_mode: ToolMode::None, calibration_digest: d, decision_policy_digest: d,
        backend_semantic_version: "fixture-v1".to_owned(), host_class: None, compiler_identity: None }
}
pub(super) fn input(n: usize) -> JobInput<'static> {
    match n {
        0 => JobInput { id: "alpha", original: b"private Alice source", normalized: b"private Alice source" },
        _ => JobInput { id: "beta", original: b"private Bob source", normalized: b"private Bob source" },
    }
}
pub(super) fn manifest_with(limits: JobLimits) -> FrozenManifest {
    FrozenManifest::freeze(&key(), JobContract { job_id: JobId([7; 16]), execution: &identity(),
        recipe: &"fixed item-local recipe; effective-seed=7", limits }, [input(0), input(1)], &mut Control::default()).unwrap()
}
pub(super) fn manifest() -> FrozenManifest { manifest_with(limits()) }
fn hex(bytes: &[u8]) -> String { bytes.iter().map(|b| format!("{b:02x}")).collect() }

#[test]
fn rfc4231_sha256_vectors_include_long_key_normalization() {
    assert_eq!(hex(&commitment::rfc_hmac(&[0x0b; 20], b"Hi There")),
        "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7");
    assert_eq!(hex(&commitment::rfc_hmac(b"Jefe", b"what do ya want for nothing?")),
        "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843");
    assert_eq!(hex(&commitment::rfc_hmac(&[0xaa; 131], b"Test Using Larger Than Block-Size Key - Hash Key First")),
        "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54");
}
#[test]
fn domains_tuple_boundaries_and_job_ids_are_separated() {
    let key = key();
    assert!(!key.commit(b"a", &[b"bc"]).matches(key.commit(b"ab", &[b"c"])));
    assert!(!key.commit(b"a", &[b"b", b"c"]).matches(key.commit(b"a", &[b"bc"])));
    let a = manifest();
    let b = FrozenManifest::freeze(&key, JobContract { job_id: JobId([8; 16]), execution: &identity(),
        recipe: &"fixed item-local recipe; effective-seed=7", limits: limits() }, [input(0), input(1)], &mut Control::default()).unwrap();
    assert!(!a.items[0].id.matches(b.items[0].id));
    assert!(!a.population_commitment().matches(b.population_commitment()));
}
#[test]
fn complete_population_uniqueness_precedes_any_execution_or_io() {
    let failure = error(FrozenManifest::freeze(&key(), JobContract { job_id: JobId([7; 16]), execution: &identity(),
        recipe: &"recipe", limits: limits() }, [input(0), input(1), input(0)], &mut Control::default()));
    assert_eq!(failure, JobError::DuplicateId);
}
#[test]
fn population_order_original_and_normalized_content_are_bound() {
    let a = manifest();
    let b = FrozenManifest::freeze(&key(), JobContract { job_id: JobId([7; 16]), execution: &identity(),
        recipe: &"fixed item-local recipe; effective-seed=7", limits: limits() }, [input(1), input(0)], &mut Control::default()).unwrap();
    assert_eq!(error(a.binding.compare(&b.binding)), JobError::Mismatch(MismatchField::Population));
    let changed = JobInput { normalized: b"changed normalization", ..input(0) };
    assert_eq!(error(a.verify_input(&key(), 0, &changed)), JobError::Mismatch(MismatchField::Population));
    assert_eq!(error(a.verify_input(&key(), 1, &input(0))), JobError::Mismatch(MismatchField::Population));
}
#[test]
fn exact_recipe_seed_execution_and_limits_mismatch_categories() {
    let a = manifest();
    let b = FrozenManifest::freeze(&key(), JobContract { job_id: JobId([7; 16]), execution: &identity(),
        recipe: &"fixed item-local recipe; effective-seed=8", limits: limits() }, [input(0), input(1)], &mut Control::default()).unwrap();
    assert_eq!(error(a.binding.compare(&b.binding)), JobError::Mismatch(MismatchField::Recipe));
    let mut id = identity(); id.sampler_version = "other-address-v2".to_owned();
    let b = FrozenManifest::freeze(&key(), JobContract { job_id: JobId([7; 16]), execution: &id,
        recipe: &"fixed item-local recipe; effective-seed=7", limits: limits() }, [input(0), input(1)], &mut Control::default()).unwrap();
    assert_eq!(error(a.binding.compare(&b.binding)), JobError::Mismatch(MismatchField::Execution));
    let mut cap = limits(); cap.max_attempts += 1; let b = manifest_with(cap);
    assert_eq!(error(a.binding.compare(&b.binding)), JobError::Mismatch(MismatchField::Limits));
}
#[test]
fn manifest_index_and_whole_snapshot_caps_fail_closed() {
    let mut cap = limits(); cap.max_items = 1;
    assert_eq!(error(FrozenManifest::freeze(&key(), JobContract { job_id: JobId([7; 16]), execution: &identity(),
        recipe: &"r", limits: cap }, [input(0), input(1)], &mut Control::default())), JobError::Limit);
    cap = limits(); cap.max_snapshot_bytes = 1;
    assert_eq!(error(FrozenManifest::freeze(&key(), JobContract { job_id: JobId([7; 16]), execution: &identity(),
        recipe: &"r", limits: cap }, [input(0)], &mut Control::default())), JobError::Limit);
}
#[test]
fn freeze_uses_the_existing_cancellation_control() {
    let e = error(FrozenManifest::freeze(&key(), JobContract { job_id: JobId([7; 16]), execution: &identity(),
        recipe: &"r", limits: limits() }, [input(0), input(1)], &mut Control { calls: 0, stop_at: Some(2) }));
    assert_eq!(e, JobError::Cancelled(DecodeCancellationKind::Deadline));
}
#[test]
fn retained_metadata_contains_neither_inputs_nor_unkeyed_input_hashes() {
    let m = manifest();
    let encoded = crate::canonjson::canonical_string(&(&m.binding, &m.items)).unwrap();
    for private in ["private Alice source", "private Bob source", "alpha", "beta"] { assert!(!encoded.contains(private)); }
    assert!(!encoded.contains(&Sha256Digest::of_bytes(input(0).original).to_hex()));
}
#[test]
fn all_six_work_axes_and_overflows_are_checked() {
    for axis in 0..6 {
        let mut bigger = work(); let cap = work();
        match axis {
            0 => bigger.model.forward_positions += 1, 1 => bigger.model.projected_logits += 1,
            2 => bigger.model.attention_pairs += 1, 3 => bigger.model.projections.dot_products += 1,
            4 => bigger.model.projections.multiply_accumulates += 1, _ => bigger.mask_node_visits += 1,
        }
        assert!(!bigger.fits(cap));
    }
    let mut max = JobWork::default(); max.mask_node_visits = u64::MAX;
    assert_eq!(error(max.checked_add(work())), JobError::WorkLimit);
    max = JobWork::default(); max.model.forward_positions = u64::MAX;
    assert_eq!(error(max.checked_add(work())), JobError::WorkLimit);
}
#[test]
fn canonical_framing_round_trip_and_cross_item_replay_refusal() {
    let m = manifest(); let mut spool = Cursor::new(Vec::new());
    let pointer = frame::append(&mut spool, &key(), &m.binding, &m.items[0], 0,
        &serde_json::json!({"z":1,"a":"é"}), 4096, 8192).unwrap();
    assert_eq!(frame::read(&mut spool, &key(), &m.binding, &m.items[0], &pointer, 4096).unwrap(), "{\"a\":\"é\",\"z\":1}".as_bytes());
    assert!(frame::read(&mut spool, &key(), &m.binding, &m.items[1], &pointer, 4096).is_err());
}
#[test]
fn every_header_or_payload_corruption_is_rejected() {
    let m = manifest(); let mut spool = Cursor::new(Vec::new());
    let pointer = frame::append(&mut spool, &key(), &m.binding, &m.items[0], 0, &"result", 4096, 8192).unwrap();
    let original = spool.into_inner();
    for offset in 0..original.len() {
        let mut bytes = original.clone(); bytes[offset] ^= 1;
        assert!(frame::read(&mut Cursor::new(bytes), &key(), &m.binding, &m.items[0], &pointer, 4096).is_err());
    }
    for n in 0..original.len() {
        assert!(frame::read(&mut Cursor::new(&original[..n]), &key(), &m.binding, &m.items[0], &pointer, 4096).is_err());
    }
}
#[test]
fn nonfinite_oversized_and_uncommitted_append_never_yield_a_pointer() {
    let m = manifest(); let mut spool = Cursor::new(Vec::new());
    assert_eq!(error(frame::append(&mut spool, &key(), &m.binding, &m.items[0], 0, &f64::NAN, 4096, 8192)), JobError::Serialization);
    assert_eq!(error(frame::append(&mut spool, &key(), &m.binding, &m.items[0], 0, &"large", 1, 8192)), JobError::Limit);
    assert!(spool.get_ref().is_empty());
    spool.get_mut().push(0);
    assert_eq!(error(frame::append(&mut spool, &key(), &m.binding, &m.items[0], 0, &1, 4096, 8192)), JobError::UncommittedTail);
}
