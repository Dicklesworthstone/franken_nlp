//! Pure admission/ledger fixtures, not native execution or model quality.
use super::*;
use crate::{grammar::{CompileLimits, runtime::SourceRuntimeLimits},
    native_engine::portable_int8::ProjectionWork,
    tasks::{extract::ExtractionGrounding, ir::ScoreSpace, ner::{EntityType, NamedEntity, NER_TASK_VERSION}},
    validation::grounded_fields::{SourceOccurrence, VerifiedSourceSpan}};

fn host() -> Int4NerAssayHostLimits {
    Int4NerAssayHostLimits {
        native: NativeLimits { context_tokens: 128, allocator_reserve_bytes: 65_536,
            run: RunLimits { max_elapsed: Duration::from_secs(1), max_checkpoints: 1000,
                cleanup_reserve_bytes: 65_536 } },
        task: TaskBudget { max_input_tokens: 128, max_output_tokens: 8,
            max_output_bytes: 100_000, max_grammar_states: 128,
            max_kv_bytes: Int4MemoryRequirement::for_context(128, 32).unwrap().kv_bytes },
        planning: SourcePlanningLimits { max_context_tokens: 128,
            compiler: CompileLimits { max_states: 128, max_output_bytes: 100_000, ..CompileLimits::default() },
            source: SourceRuntimeLimits::default(), ..SourcePlanningLimits::default() },
        preparation_reserve_bytes: 512 * 1024 * 1024,
        max_model_work: Int4Work { forward_positions: u64::MAX, projected_logits: u64::MAX,
            attention_pairs: u64::MAX, projections: ProjectionWork {
                dot_products: u64::MAX, multiply_accumulates: u64::MAX } },
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
    matches!(error, HostedError::Int4(Int4HostError::SourceTask(Int4SourceError::Extraction(
        crate::tasks::extract::grouped_int4::Int4ExtractError::Native(constrained_int4::Int4JsonError::Decode(
            crate::native_engine::constrained::JsonDecodeError::BudgetExceeded(_)))))))
}

#[test]
fn paired_arms_reuse_one_native_and_preparation_envelope() {
    let host = host(); let limits = metrics::Limits::default(); let ner = NerLimits::default();
    let small = host.requirements(&[3], limits, ner).unwrap();
    let wide = host.requirements(&[10752], limits, ner).unwrap();
    let pair = host.requirements(&[3, 10752], limits, ner).unwrap();
    assert_eq!(pair, host.requirements(&[10752, 3], limits, ner).unwrap());
    assert_eq!(pair.native, wide.native);
    assert_eq!(pair.workspace_bytes, wide.workspace_bytes);
    assert_eq!(pair.native.kv_bytes, small.native.kv_bytes);
    assert!(pair.native.kv_bytes < small.native.kv_bytes + wide.native.kv_bytes);
    assert_eq!(pair.output_reservation_bytes, small.output_reservation_bytes);
    assert_eq!(pair.input_bytes - small.input_bytes,
        native_ner::required_replay_bytes(limits, ner, 2).unwrap()
            - native_ner::required_replay_bytes(limits, ner, 1).unwrap());
}

#[test]
fn source_compiler_gold_and_candidate_intervals_are_priced_independently() {
    let host = host(); let limits = metrics::Limits::default(); let ner = NerLimits::default();
    let quote = host.requirements(&[127], limits, ner).unwrap();
    let mut changed = host; changed.planning.source.index.max_index_bytes += 1;
    assert_eq!(changed.requirements(&[127], limits, ner).unwrap().input_bytes, quote.input_bytes + 2);
    changed = host; changed.planning.compiler.max_schema_bytes += 1;
    assert_eq!(changed.requirements(&[127], limits, ner).unwrap().input_bytes, quote.input_bytes + 64);
    changed = host; changed.planning.compiler.max_transitions += 1;
    assert_eq!(changed.requirements(&[127], limits, ner).unwrap().input_bytes, quote.input_bytes + 128);
    let mut more = ner; more.max_spans_per_row += 1;
    assert_eq!(host.requirements(&[127], limits, more).unwrap().input_bytes,
        quote.input_bytes + 64 * limits.max_rows as u64 + 1024);
    more = ner; more.max_occurrences_per_row += 1;
    assert_eq!(host.requirements(&[127], limits, more).unwrap().input_bytes, quote.input_bytes + 256);
    assert_eq!(quote.output_reservation_bytes, 4 * limits.max_output_bytes as u64 + 4096);
}

#[test]
fn invalid_envelopes_refuse_without_planner_or_native_allocation() {
    let limits = metrics::Limits::default(); let ner = NerLimits::default();
    for widths in [&[][..], &[0][..], &[10753][..], &[1, 2, 3][..]] {
        assert!(host().requirements(widths, limits, ner).is_err());
    }
    for index in 0..9 {
        let mut bad = host();
        match index {
            0 => bad.preparation_reserve_bytes = 0,
            1 => bad.mask_limits.max_trie_node_visits = 0,
            2 => bad.mask_limits.checkpoint_interval_nodes = 0,
            3 => bad.max_mask_node_visits = 0,
            4 => bad.planning.max_context_tokens += 1,
            5 => bad.task.max_kv_bytes -= 1,
            6 => bad.planning.compiler.max_output_bytes += 1,
            7 => bad.planning.compiler.max_states += 1,
            _ => bad.planning.source.verification.max_scan_steps = 0,
        }
        assert!(bad.requirements(&[32], limits, ner).is_err(), "axis {index}");
    }
    let mut bad = host(); bad.preparation_reserve_bytes = u64::MAX;
    assert!(bad.requirements(&[32], limits, ner).is_err());
    let mut bad = ner; bad.max_occurrences_per_row = 1;
    assert!(host().requirements(&[32], limits, bad).is_err());
}

#[test]
fn all_five_native_work_axes_spend_one_nonrenewable_operation_budget() {
    let row = work(); let pair = row.checked_add(row).unwrap();
    for index in 0..5 {
        let mut cap = pair; *axis(&mut cap, index) -= 1;
        let spent = checked_work(Int4Work::default(), row, cap).unwrap();
        assert!(is_budget(checked_work(spent, row, cap).unwrap_err()), "axis {index}");
    }
    let spent = checked_work(Int4Work::default(), row, pair).unwrap();
    assert_eq!(checked_work(spent, row, pair).unwrap(), pair);
    assert!(is_budget(checked_work(pair, row, pair).unwrap_err()));
    for index in 0..5 {
        let mut overflow = Int4Work::default(); *axis(&mut overflow, index) = u64::MAX;
        assert!(is_budget(checked_work(overflow, row, host().max_model_work).unwrap_err()));
    }
}

fn mention() -> NerResult {
    NerResult { schema_version: 1, task_spec_version: NER_TASK_VERSION.to_owned(),
        numerics_profile: crate::native_engine::strict_int4::STRICT_INT4_PROFILE.to_owned(),
        score_space: ScoreSpace::NotComputed, grounding: ExtractionGrounding::SourceMembership,
        entities: vec![NamedEntity { text: "Alice".to_owned(), entity_type: EntityType::Person,
            occurrence: SourceOccurrence::Ambiguous, spans: vec![
                VerifiedSourceSpan { byte_start: 0, byte_end: 5, scalar_start: 0, scalar_end: 5 },
                VerifiedSourceSpan { byte_start: 10, byte_end: 15, scalar_start: 10, scalar_end: 15 }] }],
        generated_token_ids: vec![1, 2], forward_positions: 1, projected_logits: 1, mask_node_visit_charge: 1 }
}

#[test]
fn verification_admits_all_three_passes_and_preserves_declared_source_policy() {
    let mut host = host();
    host.planning.source.verification.max_fields = 1;
    host.planning.source.verification.max_matches = 2;
    host.planning.source.index.max_matches = 2;
    let original = host.planning.source;
    let ner = NerLimits { max_spans_per_row: 1, max_occurrences_per_row: 2,
        max_verification_steps: 1000, ..NerLimits::default() };
    let mut ledger = new_ledger(1, 2, ner).unwrap();
    preflight_verification(15, 1, 5, host, &ledger.verification).unwrap();
    let charged = charge_native_verification(15, &mention(), &mut ledger.verification).unwrap();
    assert_eq!(charged, 2 * 8 * (15 + 5 + 1));
    assert_eq!(ledger.verification.max_fields, 4);
    assert_eq!(ledger.verification.max_matches, 8);
    // The independent collector spends the third pass through its KMP path.
    ledger.verification.max_fields -= 1;
    ledger.verification.max_matches -= 2;
    ledger.verification.max_scan_steps -= charged / 2;
    assert_eq!(ledger.verification.max_scan_steps, 1000 - 3 * 168);
    assert!(is_budget(preflight_verification(15, 1, 5, host, &ledger.verification).unwrap_err()));
    assert_eq!(host.planning.source, original);
}

#[test]
fn verification_debit_failure_is_atomic_across_fields_matches_and_steps() {
    for axis in 0..3 {
        let mut remaining = GroundingBudget { max_fields: 2, max_matches: 4, max_scan_steps: 336 };
        match axis { 0 => remaining.max_fields -= 1, 1 => remaining.max_matches -= 1,
            _ => remaining.max_scan_steps -= 1 }
        let before = remaining;
        assert!(is_budget(charge_native_verification(15, &mention(), &mut remaining).unwrap_err()));
        assert_eq!(remaining, before);
    }
}
