//! Host admission arithmetic over actual declared model/task types; no fake
//! neural result or alternate admission provider is exposed by the public API.
use super::*;
use crate::{execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    native_engine::strict_int8::STRICT_INT8_EXECUTION};

fn limits() -> CorpusLimits {
    CorpusLimits { native: NativeLimits { context_tokens: 32, allocator_reserve_bytes: 1024 * 1024,
        run: RunLimits { max_elapsed: Duration::from_secs(30), max_checkpoints: 10_000, cleanup_reserve_bytes: 65536 } },
        transport: BatchLimits::default(), preparation_reserve_bytes: 8 * 1024 * 1024, io_reserve_bytes: 64 * 1024 }
}
fn model() -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(), revision: "fixture".to_owned(),
        recipe_id: "nanbeige42-int8-v1".to_owned(), source_root_sha256: "0".repeat(64),
        logical_model_sha256: "1".repeat(64) }
}
fn identity() -> ExecutionIdentity {
    let model = model(); let d = Sha256Digest::of_bytes(b"fixture");
    ExecutionIdentity { schema_version: 1, source_revision: model.revision,
        logical_model_digest: Sha256Digest::from_hex(&model.logical_model_sha256).unwrap(),
        artifact_format: "fixture".to_owned(), quant_recipe: model.recipe_id, packing_set_digest: d,
        tokenizer_digest: d, template_digest: d, task_spec: "chat-v1".to_owned(), taskir_digest: d, prompt_digest: d,
        grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None }
}
fn work() -> Int8Work { Int8Work::for_sequence(0, 16, 4 * NANBEIGE_VOCAB_SIZE).unwrap() }
fn admit(id: &ExecutionIdentity, work: Int8Work, kv: u64, sampler: u64, output: u64) -> Result<(), BatchItemFailure> {
    check_request(&model(), 4096, 2048, 8192, id, work, kv, sampler, output)
}

#[test]
fn both_task_families_use_the_same_actual_model_and_workspace_authority() {
    let mut id = identity();
    for task in ["generate-v1", "chat-v1", "extract-v1"] {
        id.task_spec = task.to_owned();
        admit(&id, work(), 4096, if task == "extract-v1" { 0 } else { 2048 }, 8192).unwrap();
    }
}

#[test]
fn mismatched_physical_model_facts_refuse_before_any_output_claim() {
    for axis in 0..4 {
        let mut id = identity();
        match axis { 0 => id.source_revision.push('x'), 1 => id.quant_recipe.push('x'),
            2 => id.logical_model_digest = Sha256Digest::of_bytes(b"other"), _ => id.numerics_profile = NumericsProfile::HfBf16Eager }
        let error = admit(&id, work(), 4096, 2048, 8192).unwrap_err();
        assert!(error.stop); assert_eq!(error.fault.code, BatchCode::Admission);
    }
}

#[test]
fn backend_and_kv_profile_cannot_be_repaired_during_admission() {
    for axis in 0..3 {
        let mut id = identity();
        match axis { 0 => id.backend_semantic_version = "other".to_owned(),
            1 => id.kv_dtype = "int8".to_owned(), _ => id.numerics_profile = NumericsProfile::StrictQuantized { version: 2 } }
        assert!(admit(&id, work(), 4096, 0, 8192).unwrap_err().stop);
    }
}

#[test]
fn resident_kv_must_equal_the_actual_charge_not_merely_fit_under_it() {
    for kv in [0, 4095, 4097, u64::MAX] {
        let error = admit(&identity(), work(), kv, 0, 8192).unwrap_err();
        assert!(error.stop); assert_eq!(error.fault.code, BatchCode::Admission);
    }
    admit(&identity(), work(), 4096, 0, 8192).unwrap();
}

#[test]
fn sampler_cannot_consume_unreserved_scratch() {
    admit(&identity(), work(), 4096, 2048, 8192).unwrap();
    let error = admit(&identity(), work(), 4096, 2049, 8192).unwrap_err();
    assert!(error.stop); assert_eq!(error.fault.code, BatchCode::Admission);
}

#[test]
fn oversized_or_zero_result_is_a_bounded_document_refusal() {
    for bytes in [0, 8193, u64::MAX] {
        let error = admit(&identity(), work(), 4096, 0, bytes).unwrap_err();
        assert!(!error.stop); assert_eq!(error.fault.code, BatchCode::OutputLineLimit);
    }
}

#[test]
fn full_projection_geometry_and_all_native_work_axes_are_required() {
    for axis in 0..6 {
        let mut w = work();
        match axis { 0 => w.forward_positions = 0, 1 => w.projected_logits = 0,
            2 => w.attention_pairs = 0, 3 => w.projections.dot_products = 0,
            4 => w.projections.multiply_accumulates = 0, _ => w.projected_logits -= 1 }
        let error = admit(&identity(), w, 4096, 0, 8192).unwrap_err();
        assert!(error.stop); assert_eq!(error.fault.code, BatchCode::InvalidExecution);
    }
}

#[test]
fn transport_commitment_includes_ids_lines_output_and_owned_io() {
    let l = limits(); let t = l.transport;
    let expected = 8 * t.max_line_bytes as u64 + 4 * t.max_output_line_bytes as u64
        + 128 * t.max_epoch_ids as u64 + t.max_epoch_id_bytes as u64
        + l.preparation_reserve_bytes + l.io_reserve_bytes;
    assert_eq!(l.reservation_bytes().unwrap(), expected);
    let mut larger = l; larger.io_reserve_bytes += 17;
    assert_eq!(larger.reservation_bytes().unwrap(), expected + 17);
    larger = l; larger.preparation_reserve_bytes += 31;
    assert_eq!(larger.reservation_bytes().unwrap(), expected + 31);
}

#[test]
fn preparation_and_external_io_are_not_assumed_free() {
    let mut l = limits(); l.preparation_reserve_bytes = 0;
    assert!(l.reservation_bytes().is_err());
    l = limits(); l.io_reserve_bytes = 0;
    assert!(l.reservation_bytes().is_err());
}

#[test]
fn malformed_transport_and_every_reservation_overflow_fail_before_reading() {
    let mut l = limits(); l.transport.max_json_depth = 65;
    assert!(matches!(l.reservation_bytes(), Err(HostedError::BatchSetup(_))));
    for axis in 0..5 {
        let mut l = limits();
        match axis { 0 => l.preparation_reserve_bytes = u64::MAX, 1 => l.io_reserve_bytes = u64::MAX,
            2 => l.transport.max_line_bytes = usize::MAX, 3 => l.transport.max_output_line_bytes = usize::MAX,
            _ => l.transport.max_epoch_ids = usize::MAX }
        if usize::BITS == 64 || axis < 2 { assert!(l.reservation_bytes().is_err()); }
    }
}

#[test]
fn elapsed_checkpoint_and_cleanup_limits_remain_whole_run_requirements() {
    let mut l = limits(); l.native.run.max_elapsed = Duration::ZERO;
    assert!(l.reservation_bytes().is_err());
    l = limits(); l.native.run.max_checkpoints = 1;
    assert!(l.reservation_bytes().is_err());
    l = limits(); l.native.run.cleanup_reserve_bytes = 0;
    assert!(l.reservation_bytes().is_err());
}

#[test]
fn output_capacity_is_a_preflight_only_not_an_invented_success_receipt() {
    // This check cannot assert anything about neural result validity or actual
    // serialized size; the existing task finalizer and runner still do that.
    assert!(admit(&identity(), work(), 4096, 0, 1).is_ok());
    assert!(admit(&identity(), work(), 4096, 0, 8192).is_ok());
}

#[test]
fn batch_diagnostics_preserve_summary_without_echoing_private_content() {
    let summary = BatchSummary { succeeded: 2, failed: 1, ..BatchSummary::default() };
    let error = HostedError::Batch(batch::BatchRunError { fault: BatchCode::OutputIo.into(), summary });
    assert!(error.source().is_some());
    match error { HostedError::Batch(error) => { assert_eq!(error.summary, summary); assert_eq!(error.fault.code, BatchCode::OutputIo); }, _ => unreachable!() }
    let setup = HostedError::BatchSetup(BatchCode::InvalidLimits.into());
    assert!(setup.source().is_some());
    assert!(!format!("{setup} {setup:?}").contains("identity"));
}

#[test]
fn rejection_constructor_accepts_codes_and_complete_planning_faults() {
    let code = BatchItemFailure::reject(BatchCode::Planning);
    let fault = batch::BatchFault { code: BatchCode::Planning, cancellation: None };
    let complete = BatchItemFailure::reject(fault);
    assert_eq!(code.fault, complete.fault);
    assert!(!code.stop && !complete.stop);
    let cancellation = batch::BatchFault::cancelled(crate::native_engine::decode::DecodeCancellationKind::Deadline);
    let fatal = BatchItemFailure::fatal(cancellation);
    assert!(fatal.stop); assert_eq!(fatal.fault, cancellation);
}
