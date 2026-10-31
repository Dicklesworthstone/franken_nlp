//! Resource and complete-operation contracts without native model execution.
use super::*;
use crate::native_engine::portable_int8::ProjectionWork;

fn host() -> Int4AssayHostLimits {
    Int4AssayHostLimits {
        native: NativeLimits { context_tokens: 128, allocator_reserve_bytes: 65536,
            run: RunLimits { max_elapsed: Duration::from_secs(1), max_checkpoints: 1000,
                cleanup_reserve_bytes: 65536 } },
        task: TaskBudget { max_input_tokens: 128, max_output_tokens: 8, max_output_bytes: 100_000,
            max_grammar_states: 128,
            max_kv_bytes: Int4MemoryRequirement::for_context(128, 32).unwrap().kv_bytes },
        preparation_reserve_bytes: 128 * 1024 * 1024,
        max_model_work: Int4Work { forward_positions: u64::MAX, projected_logits: u64::MAX,
            attention_pairs: u64::MAX, projections: ProjectionWork {
                dot_products: u64::MAX, multiply_accumulates: u64::MAX } },
    }
}
fn row() -> Int4Work {
    Int4Work { forward_positions: 2, projected_logits: 3, attention_pairs: 4,
        projections: ProjectionWork { dot_products: 5, multiply_accumulates: 6 } }
}
fn axis(work: &mut Int4Work, index: usize) -> &mut u64 {
    match index { 0 => &mut work.forward_positions, 1 => &mut work.projected_logits,
        2 => &mut work.attention_pairs, 3 => &mut work.projections.dot_products,
        4 => &mut work.projections.multiply_accumulates, _ => panic!("unknown work axis") }
}
fn is_work_budget(error: HostedError) -> bool {
    matches!(error, HostedError::Int4(Int4HostError::Classification(Int4ClassificationError::WorkBudget)))
}

#[test]
fn paired_models_reuse_one_maximum_native_envelope_while_retaining_both_score_collections() {
    let limits = metrics::Limits::default(); let host = host();
    let left = host.requirements(&[3], limits).unwrap();
    let right = host.requirements(&[10752], limits).unwrap();
    let pair = host.requirements(&[3, 10752], limits).unwrap();
    let reversed = host.requirements(&[10752, 3], limits).unwrap();
    assert_eq!(pair, reversed);
    assert_eq!(pair.native, right.native);
    assert_eq!(pair.workspace_bytes, right.workspace_bytes);
    assert_eq!(pair.native.kv_bytes, left.native.kv_bytes);
    assert!(pair.native.kv_bytes < left.native.kv_bytes + right.native.kv_bytes);
    assert_eq!(pair.output_reservation_bytes, left.output_reservation_bytes);
    assert_eq!(pair.input_bytes - left.input_bytes,
        native_classify::required_replay_bytes(limits, 2).unwrap()
            - native_classify::required_replay_bytes(limits, 1).unwrap());
    assert!(pair.input_bytes > left.input_bytes);
}

#[test]
fn complete_source_score_row_and_report_storage_are_priced_before_execution() {
    let limits = metrics::Limits::default(); let host = host();
    let quote = host.requirements(&[127], limits).unwrap();
    let native = Int4MemoryRequirement::for_context(host.native.context_tokens, 127).unwrap();
    assert_eq!(quote.native, native);
    assert_eq!(quote.workspace_bytes,
        native.rope_bytes + native.scratch_payload_bound + host.native.allocator_reserve_bytes);
    assert_eq!(quote.input_bytes,
        8 * limits.max_input_bytes as u64 + 8 * limits.max_manifest_bytes as u64
        + host.preparation_reserve_bytes + native_classify::required_replay_bytes(limits, 1).unwrap()
        + 4 * host.task.max_output_bytes);
    assert_eq!(quote.output_reservation_bytes, 4 * limits.max_output_bytes as u64 + 4096);
    let mut more = host; more.task.max_output_bytes += 1;
    assert_eq!(more.requirements(&[127], limits).unwrap().input_bytes, quote.input_bytes + 4);
    more = host; more.preparation_reserve_bytes += 1;
    assert_eq!(more.requirements(&[127], limits).unwrap().input_bytes, quote.input_bytes + 1);
    more = host; more.native.allocator_reserve_bytes += 1;
    assert_eq!(more.requirements(&[127], limits).unwrap().workspace_bytes, quote.workspace_bytes + 1);
}

#[test]
fn both_arms_share_every_work_axis_and_failed_reservation_is_atomic() {
    let row = row(); let complete = row.checked_add(row).unwrap();
    for field in 0..5 {
        let mut ceiling = complete; *axis(&mut ceiling, field) -= 1;
        let mut spent = Int4Work::default();
        charge_work(&mut spent, row, ceiling).unwrap(); // Candidate has really spent this work.
        let before = spent;
        assert!(is_work_budget(charge_work(&mut spent, row, ceiling).unwrap_err()), "axis {field}");
        assert_eq!(spent, before); // Baseline cannot receive a renewed allowance.
    }
    let mut spent = Int4Work::default();
    charge_work(&mut spent, row, complete).unwrap();
    charge_work(&mut spent, row, complete).unwrap();
    assert_eq!(spent, complete);
    assert!(is_work_budget(charge_work(&mut spent, row, complete).unwrap_err()));
    assert_eq!(spent, complete);
}

#[test]
fn overflow_in_any_native_counter_never_partially_charges_another_axis() {
    for field in 0..5 {
        let mut spent = Int4Work::default(); *axis(&mut spent, field) = u64::MAX;
        let before = spent;
        assert!(is_work_budget(charge_work(&mut spent, row(), host().max_model_work).unwrap_err()));
        assert_eq!(spent, before, "axis {field}");
    }
    for field in 0..5 {
        let mut invalid = row(); *axis(&mut invalid, field) = 0;
        let mut spent = row(); let before = spent;
        assert!(matches!(charge_work(&mut spent, invalid, host().max_model_work),
            Err(HostedError::Int4(Int4HostError::Classification(Int4ClassificationError::Accounting)))));
        assert_eq!(spent, before);
    }
}

#[test]
fn paired_bootstrap_prices_both_arms_before_any_model_call() {
    let options = EvalOptions::default(); let mut limits = metrics::Limits::default();
    limits.max_bootstrap_draws = 30_000 * options.bootstrap.resamples as u64;
    validate_bootstrap(30_000, 1, options, limits).unwrap();
    assert!(matches!(validate_bootstrap(30_000, 2, options, limits),
        Err(HostedError::Assay(AssayError::InputLimit))));
    validate_bootstrap(15_000, 2, options, limits).unwrap();
    assert!(validate_bootstrap(15_001, 2, options, limits).is_err());
    assert!(validate_bootstrap(1, 0, options, limits).is_err());
    assert!(validate_bootstrap(1, 3, options, limits).is_err());
    #[cfg(target_pointer_width = "64")]
    assert!(validate_bootstrap(usize::MAX, 2, options, limits).is_err());
}

#[test]
fn complete_context_capacity_and_every_resource_input_are_admitted() {
    let limits = metrics::Limits::default();
    for field in 0..10 {
        let mut host = host();
        match field {
            0 => host.task.max_kv_bytes -= 1,
            1 => host.native.context_tokens = 0,
            2 => host.preparation_reserve_bytes = 0,
            3 => host.native.allocator_reserve_bytes = 0,
            4 => host.native.run.max_checkpoints = 1,
            _ => *axis(&mut host.max_model_work, field - 5) = 0,
        }
        assert!(host.requirements(&[127], limits).is_err(), "axis {field}");
    }
    for widths in [&[][..], &[0][..], &[10753][..], &[1, 2, 3][..]] {
        assert!(host().requirements(widths, limits).is_err());
    }
    let mut too_many_rows = limits; too_many_rows.max_rows += 1;
    assert!(host().requirements(&[127], too_many_rows).is_err());
}

#[test]
fn storage_overflow_is_refused_before_native_or_score_allocation() {
    for field in 0..3 {
        let mut host = host();
        match field { 0 => host.preparation_reserve_bytes = u64::MAX,
            1 => host.native.allocator_reserve_bytes = u64::MAX,
            _ => host.task.max_output_bytes = u64::MAX }
        assert!(host.requirements(&[127], metrics::Limits::default()).is_err());
    }
}

#[test]
fn report_scope_and_errors_do_not_expose_private_source_or_nested_diagnostics() {
    for scope in ["", "host name", "host\nname", "\t", &"a".repeat(257)] {
        assert!(matches!(native_classify::validate_scope("native-assay", Some(scope)), Err(AssayError::Scope)));
    }
    native_classify::validate_scope("native-assay", Some("owner-declared-host-v1")).unwrap();
    let error = HostedError::Assay(AssayError::Commitment);
    assert!(error.source().is_some());
    assert_eq!(error.to_string(), format!("{error:?}"));
    assert_eq!(error.to_string(), "hosted native evaluation or qualification refused");
}

#[test]
fn sealed_input_and_public_reports_cross_the_existing_physical_host_boundary() {
    fn send<T: Send + 'static>() {}
    send::<Input>(); send::<NativeReplay>(); send::<Evaluation>(); send::<QualificationReport>();
    let _evaluate = NlpEngine::evaluate_int4_classification;
    let _qualify = NlpEngine::qualify_int4_classification;
}
