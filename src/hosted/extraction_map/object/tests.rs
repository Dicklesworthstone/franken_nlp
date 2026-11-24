//! Full quote and typed failure regressions; these do not run a model.
use super::*;
use crate::{
    batch::extract::ExtractionBatchGrounding,
    native_engine::{decode::DecodeCancellationKind, strict_int8::Int8Work},
    tasks::{extract::object_reconcile::RootObjectError, ir::TaskBudget,
        mapreduce::{ChunkLimits, ExecutionLimits},
        source_planning::quantized::long::Int8SourceMapLimits},
    validation::grounded_fields::GroundingBudget,
};

fn config() -> Int8ObjectExtractionMapConfig {
    Int8ObjectExtractionMapConfig { mapping: ExtractionMapConfig {
        request: ExtractionBatchArgs {
            schema: r#"{"type":"object","additionalProperties":false,"properties":{"a":{"type":"integer"}},"required":["a"]}"#.to_owned(),
            grounding: ExtractionBatchGrounding::Structural,
            budget: TaskBudget { max_input_tokens: 64, max_output_tokens: 16, max_output_bytes: 4096,
                max_grammar_states: 4096, max_kv_bytes: 1024 } },
        limits: Int8ExtractionMapLimits {
            mapping: Int8SourceMapLimits {
                chunks: ChunkLimits { max_input_bytes: 1024, max_chunk_bytes: 16, max_chunk_tokens: 16,
                    context_tokens: 128, reserved_tokens: 32, max_chunks: 16, ..ChunkLimits::default() },
                reduction: ExecutionLimits { map_batch_chunks: 1, max_value_bytes: 4096,
                    max_live_value_bytes: 8192, max_result_bytes: 4096, ..ExecutionLimits::default() },
                max_model_work: Int8Work::for_sequence(0, 128, 128 * 166_144).unwrap(),
                mask_limits: Default::default(), mask_visits_per_chunk: 1000, max_mask_visits: 16_000 },
            verification: GroundingBudget { max_fields: 32, max_matches: 128, max_scan_steps: 10_000 } },
        native: NativeLimits { context_tokens: 128, allocator_reserve_bytes: 4096,
            run: RunLimits { max_elapsed: Duration::from_secs(1), max_checkpoints: 100, cleanup_reserve_bytes: 4096 } },
        preparation_reserve_bytes: 4096, reduction_reserve_bytes: 4096,
    }, reconciliation: RootObjectLimits { max_properties: 16, max_origins: 64, max_input_bytes: 65536,
        max_json_bytes: 1024, max_nodes: 4096, max_result_bytes: 4096 } }
}

#[test]
fn complete_object_quote_adds_both_schema_declarations_and_one_global_reducer() {
    let config = config();
    let compiler = CompileLimits::default();
    let schema = RootObjectSchema::workspace_bytes(compiler).unwrap();
    let reducer = config.reconciliation.workspace_bytes().unwrap();
    let work = config.mapping.limits.mapping.max_model_work;
    let budget = config.mapping.request.budget;
    let verification = config.mapping.limits.verification;
    for chunks in [1, 2, 16] {
        assert_eq!(config.workspace_bytes(chunks, compiler).unwrap(),
            reduction_bytes(&config.mapping, chunks).unwrap() + schema + reducer);
        assert_eq!(config.limits().mapping.mapping.max_model_work, work);
        assert_eq!(config.mapping.request.budget, budget);
        assert_eq!(config.limits().mapping.verification, verification);
    }
    assert_eq!(config.workspace_bytes(16, compiler).unwrap() - config.workspace_bytes(1, compiler).unwrap(),
        reduction_bytes(&config.mapping, 16).unwrap() - reduction_bytes(&config.mapping, 1).unwrap());
}

#[test]
fn small_final_object_does_not_erase_original_and_partial_schema_storage() {
    let mut config = config();
    config.reconciliation.max_input_bytes = 2;
    config.reconciliation.max_json_bytes = 2;
    config.reconciliation.max_result_bytes = 2;
    let compiler = CompileLimits::default();
    let larger = CompileLimits { max_schema_bytes: compiler.max_schema_bytes * 2, ..compiler };
    let base = reduction_bytes(&config.mapping, 1).unwrap() + config.reconciliation.workspace_bytes().unwrap();
    let first = config.workspace_bytes(1, compiler).unwrap();
    let second = config.workspace_bytes(1, larger).unwrap();
    assert_eq!(first - base, RootObjectSchema::workspace_bytes(compiler).unwrap());
    assert_eq!(second - base, RootObjectSchema::workspace_bytes(larger).unwrap());
    assert!(second > first);
    assert!(first > config.reconciliation.max_result_bytes as u64);
}

#[test]
fn complete_object_quote_refuses_invalid_limits_and_arithmetic_before_preparation() {
    for axis in 0..9 {
        let mut config = config();
        match axis {
            0 => config.mapping.limits.mapping.reduction.map_batch_chunks = 0,
            1 => config.reconciliation.max_properties = 0,
            2 => config.reconciliation.max_origins = 15,
            3 => config.reconciliation.max_input_bytes = 1,
            4 => config.reconciliation.max_json_bytes = 4097,
            5 => config.reconciliation.max_nodes = 0,
            6 => config.reconciliation.max_result_bytes = 4097,
            7 => config.reconciliation.max_result_bytes = config.reconciliation.max_json_bytes - 1,
            _ => config.mapping.reduction_reserve_bytes = u64::MAX,
        }
        assert!(config.workspace_bytes(1, CompileLimits::default()).is_err(), "axis {axis}");
    }
    for chunks in [0, 17, usize::MAX] {
        assert!(config().workspace_bytes(chunks, CompileLimits::default()).is_err());
    }
    for compiler in [
        CompileLimits { max_schema_bytes: 0, ..CompileLimits::default() },
        CompileLimits { max_states: 0, ..CompileLimits::default() },
    ] {
        assert!(config().workspace_bytes(1, compiler).is_err());
    }
    #[cfg(target_pointer_width = "64")]
    for compiler in [
        CompileLimits { max_schema_bytes: usize::MAX, ..CompileLimits::default() },
        CompileLimits { max_states: usize::MAX, ..CompileLimits::default() },
    ] {
        assert!(config().workspace_bytes(1, compiler).is_err());
    }
}

#[test]
fn object_workspace_stays_additive_to_coordinator_and_fixed_worker_storage() {
    let config = config();
    let object = config.object_workspace_bytes(CompileLimits::default()).unwrap();
    let required = requirements(config.mapping.native).unwrap();
    let serial = scratch_bytes(config.mapping.native, required, None).unwrap();
    for width in [2, 7, 64] {
        for rows in [1, 4, 64] {
            let extra = Int8PrefillLimits::required_extra_scratch_bytes(rows).unwrap();
            let prefill = Int8PrefillLimits { max_batch_rows: rows, max_extra_scratch_bytes: extra };
            let coordinator = scratch_bytes(config.mapping.native, required, Some(prefill)).unwrap();
            let workers = dispatch::int8_prefill_team_workspace_bytes(width, rows).unwrap();
            assert_eq!(sum(&[coordinator, workers, object]).unwrap(), serial + extra + workers + object);
            assert_eq!(config.object_workspace_bytes(CompileLimits::default()).unwrap(), object);
        }
    }
}

#[test]
fn object_host_wrapper_preserves_all_map_and_reconciliation_cancellation_causes() {
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        for error in [
            Int8ObjectExtractionMapError::Reconciliation(RootObjectError::Cancelled(cause)),
            Int8ObjectExtractionMapError::Map(Int8SourceMapError::Source(Int8SourceError::Cancelled(cause))),
        ] {
            let HostedError::Int8ObjectExtractionMap(error) = object_error(error)
                else { panic!("object failure classification changed"); };
            assert_eq!(error.cancellation(), Some(cause));
            assert!(std::error::Error::source(&error).is_some());
        }
    }
}
