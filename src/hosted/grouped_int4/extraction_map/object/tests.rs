//! Admission and typed error regressions; no model execution authority.
use super::*;
use crate::{batch::extract::ExtractionBatchGrounding,
    native_engine::decode::DecodeCancellationKind,
    tasks::extract::{array_union::RootArrayUnionLimits, object_reconcile::RootObjectError}};

fn config() -> Int4ObjectExtractionMapConfig {
    let mut mapping = super::super::tests::config();
    mapping.request.schema = r#"{"type":"object","additionalProperties":false,"properties":{"a":{"type":"integer"}},"required":["a"]}"#.to_owned();
    mapping.request.grounding = ExtractionBatchGrounding::Structural;
    mapping.limits.mapping.reduction.map_batch_chunks = 1;
    Int4ObjectExtractionMapConfig { mapping, reconciliation: RootObjectLimits {
        max_properties: 16, max_origins: 64, max_input_bytes: 65536,
        max_json_bytes: 1024, max_nodes: 4096, max_result_bytes: 4096,
    } }
}

#[test]
fn original_partial_schema_and_reducer_are_charged_once_beside_every_native_chunk() {
    let config = config();
    let compiler = CompileLimits::default();
    let schema = RootObjectSchema::workspace_bytes(compiler).unwrap();
    let reducer = config.reconciliation.workspace_bytes().unwrap();
    let work = config.mapping.limits.mapping.max_model_work;
    let task = config.mapping.request.budget;
    let verification = config.mapping.limits.verification;
    for chunks in [1, 2, 16] {
        let native_map = config.mapping.workspace_bytes(chunks).unwrap();
        assert_eq!(config.workspace_bytes(chunks, compiler).unwrap(), native_map + reducer + schema);
        assert_eq!(config.limits().mapping.mapping.max_model_work, work);
        assert_eq!(config.mapping.request.budget, task);
        assert_eq!(config.limits().mapping.verification, verification);
    }
    let chunk_growth = config.workspace_bytes(16, compiler).unwrap() - config.workspace_bytes(1, compiler).unwrap();
    assert_eq!(chunk_growth, config.mapping.workspace_bytes(16).unwrap() - config.mapping.workspace_bytes(1).unwrap());
}

#[test]
fn tiny_legal_output_cannot_hide_large_optional_schema_storage() {
    let mut config = config();
    config.reconciliation.max_input_bytes = 2;
    config.reconciliation.max_json_bytes = 2;
    config.reconciliation.max_result_bytes = 2;
    let compiler = CompileLimits::default();
    let larger = CompileLimits { max_schema_bytes: compiler.max_schema_bytes * 2, ..compiler };
    let independent = config.mapping.workspace_bytes(1).unwrap() + config.reconciliation.workspace_bytes().unwrap();
    let required = config.workspace_bytes(1, compiler).unwrap();
    let expanded = config.workspace_bytes(1, larger).unwrap();
    assert_eq!(required - independent, RootObjectSchema::workspace_bytes(compiler).unwrap());
    assert_eq!(expanded - independent, RootObjectSchema::workspace_bytes(larger).unwrap());
    assert!(expanded > required);
    assert!(required > config.reconciliation.max_result_bytes as u64);
}

#[test]
fn invalid_reducer_schedules_and_complete_output_bounds_refuse_the_object_quote() {
    for axis in 0..9 {
        let mut config = config();
        match axis {
            0 => config.mapping.limits.mapping.reduction.map_batch_chunks = 2,
            1 => config.mapping.limits.root_array_union = Some(RootArrayUnionLimits {
                max_records: 16, max_origins: 64, max_input_bytes: 65536,
                max_json_bytes: 1024, max_nodes: 4096, max_result_bytes: 4096,
            }),
            2 => config.reconciliation.max_properties = 0,
            3 => config.reconciliation.max_origins = 15,
            4 => config.reconciliation.max_input_bytes = 1,
            5 => config.reconciliation.max_json_bytes = 4097,
            6 => config.reconciliation.max_nodes = 0,
            7 => config.reconciliation.max_result_bytes = 4097,
            _ => config.reconciliation.max_result_bytes = config.reconciliation.max_json_bytes - 1,
        }
        assert!(config.workspace_bytes(1, CompileLimits::default()).is_err(), "axis {axis}");
    }
    for chunks in [0, 17, usize::MAX] {
        assert!(config().workspace_bytes(chunks, CompileLimits::default()).is_err());
    }
}

#[test]
fn schema_and_total_workspace_arithmetic_cannot_wrap_into_a_cheaper_admission() {
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
    let mut config = config();
    config.mapping.reduction_reserve_bytes = u64::MAX;
    assert!(config.workspace_bytes(1, CompileLimits::default()).is_err());
}

#[test]
fn object_error_wrapper_preserves_every_cancellation_cause_and_the_original_source_chain() {
    for cause in [DecodeCancellationKind::User, DecodeCancellationKind::Timeout,
        DecodeCancellationKind::Deadline, DecodeCancellationKind::PollQuota,
        DecodeCancellationKind::CostBudget, DecodeCancellationKind::FailFast,
        DecodeCancellationKind::RaceLost, DecodeCancellationKind::ParentCancelled,
        DecodeCancellationKind::ResourceUnavailable, DecodeCancellationKind::Shutdown,
        DecodeCancellationKind::LinkedExit] {
        let errors = [
            Int4ObjectExtractionMapError::Reconciliation(RootObjectError::Cancelled(cause)),
            Int4ObjectExtractionMapError::Map(Int4ExtractionMapError::Extraction(Int4ExtractError::Extraction(
                ExtractError::Decode(JsonDecodeError::Cancelled(cause))))),
        ];
        for error in errors {
            let HostedError::Int4(Int4HostError::ObjectExtractionMap(error)) = object_error(error)
                else { panic!("wrong hosted error classification"); };
            assert_eq!(error.cancellation(), Some(cause));
            assert!(std::error::Error::source(&error).is_some());
        }
    }
}
