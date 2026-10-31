//! Pure admission and accounting fixtures, not native/model-quality evidence.
use super::*;
use crate::{native_engine::{constrained::JsonDecodeOutput, portable_int8::ProjectionWork},
    tasks::{extract::ExtractionGrounding, ir::ScoreSpace},
    validation::grounded_fields::{SourceFieldEvidence, SourceOccurrence, VerifiedSourceSpan}};

fn host() -> Int4ExtractAssayHostLimits {
    let mut source = SourceRuntimeLimits::default();
    source.verification.max_fields = ExtractLimits::default().max_fields_per_row;
    Int4ExtractAssayHostLimits {
        native: NativeLimits { context_tokens: 128, allocator_reserve_bytes: 65_536,
            run: RunLimits { max_elapsed: Duration::from_secs(1), max_checkpoints: 1000,
                cleanup_reserve_bytes: 65_536 } },
        task: TaskBudget { max_input_tokens: 128, max_output_tokens: 8, max_output_bytes: 100_000,
            max_grammar_states: 128, max_kv_bytes: Int4MemoryRequirement::for_context(128, 32).unwrap().kv_bytes },
        compiler: CompileLimits { max_states: 128, max_output_bytes: 100_000, ..CompileLimits::default() },
        source, preparation_reserve_bytes: 512 * 1024 * 1024,
        max_model_work: Int4Work { forward_positions: u64::MAX, projected_logits: u64::MAX,
            attention_pairs: u64::MAX, projections: ProjectionWork { dot_products: u64::MAX,
                multiply_accumulates: u64::MAX } },
        mask_limits: MaskWorkLimits { max_trie_node_visits: 1000, checkpoint_interval_nodes: 16 },
        max_mask_node_visits: 1_000_000,
    }
}
fn work() -> Int4Work {
    Int4Work { forward_positions: 2, projected_logits: 3, attention_pairs: 4,
        projections: ProjectionWork { dot_products: 5, multiply_accumulates: 6 } }
}
fn axis(work: &mut Int4Work, index: usize) -> &mut u64 {
    match index { 0 => &mut work.forward_positions, 1 => &mut work.projected_logits,
        2 => &mut work.attention_pairs, 3 => &mut work.projections.dot_products,
        4 => &mut work.projections.multiply_accumulates, _ => panic!("unknown work axis") }
}
fn is_budget(error: HostedError) -> bool {
    matches!(error, HostedError::Int4(Int4HostError::Extraction(Int4ExtractError::Native(
        Int4JsonError::Decode(JsonDecodeError::BudgetExceeded(_))))))
}

#[test]
fn paired_replay_reuses_one_native_compiler_and_source_envelope() {
    let host = host(); let limits = metrics::Limits::default(); let extract = ExtractLimits::default();
    let narrow = host.requirements(&[3], limits, extract).unwrap();
    let wide = host.requirements(&[10752], limits, extract).unwrap();
    let pair = host.requirements(&[3, 10752], limits, extract).unwrap();
    assert_eq!(pair, host.requirements(&[10752, 3], limits, extract).unwrap());
    assert_eq!(pair.native, wide.native);
    assert_eq!(pair.workspace_bytes, wide.workspace_bytes);
    assert_eq!(pair.native.kv_bytes, narrow.native.kv_bytes);
    assert!(pair.native.kv_bytes < narrow.native.kv_bytes + wide.native.kv_bytes);
    assert_eq!(pair.input_bytes - narrow.input_bytes,
        native_extract::required_replay_bytes(limits, extract, 2).unwrap()
            - native_extract::required_replay_bytes(limits, extract, 1).unwrap());
    assert_eq!(pair.output_reservation_bytes, narrow.output_reservation_bytes);
    assert_eq!(pair.output_reservation_bytes, 4 * limits.max_output_bytes as u64 + 4096);
}

#[test]
fn gold_tree_uses_joint_wire_and_node_bounds_and_current_source_is_separate() {
    let host = host(); let limits = metrics::Limits::default(); let extract = ExtractLimits::default();
    let quote = host.requirements(&[127], limits, extract).unwrap();
    let mut changed = host; changed.source.index.max_index_bytes += 1;
    assert_eq!(changed.requirements(&[127], limits, extract).unwrap().input_bytes, quote.input_bytes + 2);
    changed = host; changed.compiler.max_transitions += 1;
    assert_eq!(changed.requirements(&[127], limits, extract).unwrap().input_bytes, quote.input_bytes + 128);
    let mut more = extract; more.max_occurrences_per_row += 1;
    assert_eq!(host.requirements(&[127], limits, more).unwrap().input_bytes, quote.input_bytes + 256);
    // At default row geometry the aggregate wire cap dominates retained gold.
    more = extract; more.max_json_nodes_per_row += 1;
    assert_eq!(host.requirements(&[127], limits, more).unwrap().input_bytes, quote.input_bytes + 128);
    // With three small trees, one additional node is priced per retained row
    // and once for the current independently parsed native observation.
    let limits = metrics::Limits { max_rows: 3, max_input_bytes: 65_536, ..limits };
    let extract = ExtractLimits { max_json_nodes_per_row: 128, ..extract };
    let quote = host.requirements(&[127], limits, extract).unwrap();
    let more = ExtractLimits { max_json_nodes_per_row: 129, ..extract };
    assert_eq!(host.requirements(&[127], limits, more).unwrap().input_bytes, quote.input_bytes + 4 * 128);
}

#[test]
fn shared_property_prefixes_are_charged_for_one_active_evidence_pair() {
    let mut host = host(); host.compiler.max_schema_bytes = 1000;
    let limits = metrics::Limits::default(); let extract = ExtractLimits::default();
    let quote = host.requirements(&[32], limits, extract).unwrap();
    host.compiler.max_schema_bytes += 1;
    let paths = 2 * host.source.verification.max_fields as u64 + 128;
    assert_eq!(host.requirements(&[32], limits, extract).unwrap().input_bytes,
        quote.input_bytes + 64 + 4 * paths);
    let pair = host.requirements(&[32, 127], limits, extract).unwrap();
    assert_eq!(pair.input_bytes - host.requirements(&[32], limits, extract).unwrap().input_bytes,
        native_extract::required_replay_bytes(limits, extract, 2).unwrap()
            - native_extract::required_replay_bytes(limits, extract, 1).unwrap());
}

#[test]
fn incomplete_and_overflowing_resource_envelopes_refuse_before_preparation() {
    let limits = metrics::Limits::default(); let extract = ExtractLimits::default();
    for widths in [&[][..], &[0][..], &[10753][..], &[1, 2, 3][..]] {
        assert!(host().requirements(widths, limits, extract).is_err());
    }
    for index in 0..12 {
        let mut bad = host();
        match index { 0 => bad.preparation_reserve_bytes = 0,
            1 => bad.mask_limits.max_trie_node_visits = 0, 2 => bad.mask_limits.checkpoint_interval_nodes = 0,
            3 => bad.max_mask_node_visits = 0, 4 => bad.task.max_output_tokens = bad.native.context_tokens as u32,
            5 => bad.task.max_kv_bytes -= 1, 6 => bad.compiler.max_output_bytes += 1,
            7 => bad.compiler.max_states += 1, 8 => bad.source.verification.max_scan_steps = 0,
            9 => bad.source.verification.max_scan_steps = extract.max_verification_steps + 1,
            10 => bad.source.verification.max_fields = extract.max_fields_per_row + 1,
            _ => bad.preparation_reserve_bytes = u64::MAX }
        assert!(bad.requirements(&[32], limits, extract).is_err(), "axis {index}");
    }
}

#[test]
fn all_five_native_axes_share_one_nonrenewable_operation_allowance() {
    let row = work(); let pair = row.checked_add(row).unwrap();
    for index in 0..5 {
        let mut cap = pair; *axis(&mut cap, index) -= 1;
        let spent = checked_work(Int4Work::default(), row, cap).unwrap();
        assert!(is_budget(checked_work(spent, row, cap).unwrap_err()), "axis {index}");
        let mut overflow = Int4Work::default(); *axis(&mut overflow, index) = u64::MAX;
        assert!(is_budget(checked_work(overflow, row, host().max_model_work).unwrap_err()));
    }
    let spent = checked_work(Int4Work::default(), row, pair).unwrap();
    assert_eq!(checked_work(spent, row, pair).unwrap(), pair);
    assert!(is_budget(checked_work(pair, row, pair).unwrap_err()));
    assert!(checked_work(Int4Work::default(), Int4Work::default(), pair).is_err());
}

fn result(spans: Vec<VerifiedSourceSpan>) -> ExtractResult {
    ExtractResult { schema_version: 2, task_spec_version: "extract-v1".to_owned(),
        score_space: ScoreSpace::NotComputed, grounding: ExtractionGrounding::SourceMembership,
        output: JsonDecodeOutput { schema_version: 1,
            numerics_profile: crate::native_engine::strict_int4::STRICT_INT4_PROFILE.to_owned(),
            token_ids: vec![1, 2], json: "\"Alice\"".to_owned(), forward_positions: 1,
            projected_logits: 1, mask_node_visit_charge: 1 },
        source_fields: vec![SourceFieldEvidence { json_pointer: "$".to_owned(),
            occurrence: SourceOccurrence::Ambiguous, spans }] }
}
fn mention() -> ExtractResult {
    result(vec![VerifiedSourceSpan { byte_start: 0, byte_end: 5, scalar_start: 0, scalar_end: 5 },
        VerifiedSourceSpan { byte_start: 10, byte_end: 15, scalar_start: 10, scalar_end: 15 }])
}

#[test]
fn gold_is_spent_once_and_each_native_row_preadmits_all_three_remaining_passes() {
    let mut host = host(); host.compiler.max_string_bytes = 5;
    host.source.verification.max_fields = 1; host.source.verification.max_matches = 2;
    host.source.index.max_matches = 2;
    let policy = host.source;
    let extract = ExtractLimits { max_fields_per_row: 1, max_occurrences_per_row: 2,
        max_verification_steps: 1000, ..ExtractLimits::default() };
    let mut remaining = verification_budget(3, 1, 2, extract,
        GroundingBudget { max_fields: 1, max_matches: 2, max_scan_steps: 168 }).unwrap();
    assert_eq!(remaining.max_scan_steps, 832);
    preflight_verification(15, 1, host, &remaining).unwrap();
    let native = charge_native_verification(15, &mention(), &mut remaining).unwrap();
    assert_eq!(native, 2 * 8 * (15 + 5 + 1));
    assert_eq!(remaining.max_fields, 6); assert_eq!(remaining.max_matches, 12);
    // Independent complete-output verification consumes the third pass.
    remaining.max_fields -= 1; remaining.max_matches -= 2; remaining.max_scan_steps -= native / 2;
    assert_eq!(remaining.max_scan_steps, 328);
    assert!(is_budget(preflight_verification(15, 1, host, &remaining).unwrap_err()));
    assert_eq!(host.source, policy);
    assert!(is_budget(verification_budget(3, 1, 2, extract,
        GroundingBudget { max_fields: 0, max_matches: 0, max_scan_steps: 1001 }).unwrap_err()));
}

#[test]
fn schema_reachable_fields_reduce_work_without_renewing_or_changing_policy() {
    let mut host = host(); host.compiler.max_string_bytes = 5;
    host.source.verification.max_matches = 2; host.source.index.max_matches = 2;
    let remaining = GroundingBudget { max_fields: 3, max_matches: 6, max_scan_steps: 504 };
    preflight_verification(15, 1, host, &remaining).unwrap();
    assert!(is_budget(preflight_verification(15, 2, host, &remaining).unwrap_err()));
    let exhausted = GroundingBudget { max_fields: 0, max_matches: 0, max_scan_steps: 0 };
    preflight_verification(usize::MAX, 0, host, &exhausted).unwrap();
}

#[test]
fn empty_verbatim_values_spend_real_scans_and_structural_values_spend_none() {
    let empty = result(vec![VerifiedSourceSpan { byte_start: 0, byte_end: 0, scalar_start: 0, scalar_end: 0 },
        VerifiedSourceSpan { byte_start: 1, byte_end: 1, scalar_start: 1, scalar_end: 1 }]);
    let mut remaining = GroundingBudget { max_fields: 2, max_matches: 4, max_scan_steps: 32 };
    assert_eq!(charge_native_verification(1, &empty, &mut remaining).unwrap(), 32);
    assert_eq!(remaining, GroundingBudget { max_fields: 0, max_matches: 0, max_scan_steps: 0 });
    let mut structural = empty; structural.source_fields.clear(); structural.grounding = ExtractionGrounding::NotRequested;
    assert_eq!(charge_native_verification(1, &structural, &mut remaining).unwrap(), 0);
}

#[test]
fn source_debits_fail_atomically_for_every_exhausted_axis() {
    for axis in 0..3 {
        let mut remaining = GroundingBudget { max_fields: 2, max_matches: 4, max_scan_steps: 336 };
        match axis { 0 => remaining.max_fields -= 1, 1 => remaining.max_matches -= 1,
            _ => remaining.max_scan_steps -= 1 }
        let before = remaining;
        assert!(is_budget(charge_native_verification(15, &mention(), &mut remaining).unwrap_err()));
        assert_eq!(remaining, before);
    }
}
