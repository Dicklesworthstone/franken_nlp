//! Policy-aware host preflight remains separate from byte-only job factories.
use super::*;
use crate::tokenizer::untrusted::{SafeBpeLimits, UntrustedEncoding};

fn policy() -> UntrustedEncoding {
    UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 8192, max_output_tokens: 4096,
        max_pair_checks: 1 << 22, max_pair_bytes: 1 << 26 })
}

#[test]
fn compact_host_configuration_is_explicit_for_every_source_task_and_projection_policy() {
    let planner = planner().with_document_encoding(policy()).unwrap();
    for task in [BuiltInTask::Ner, BuiltInTask::Keyphrases, BuiltInTask::Summarize, BuiltInTask::Answer] {
        for selected in [None, Some(Int4JsonSparseLimits { max_rows_per_step: 7 }),
            Some(Int4JsonSparseLimits { max_rows_per_step: NANBEIGE_VOCAB_SIZE })] {
            let mut config = config(&planner, task, 32); config.selected_rows = selected;
            config.validate_planning(&planner, 32, config.task_ceiling.max_kv_bytes).unwrap();
            assert!(source_batch::check_configuration(&planner, &config.identity, config.task_ceiling,
                config.planning, config.defaults.as_ref(), 32).is_err());
            source_batch::check_configuration_with_document_encoding(&planner, &config.identity,
                config.task_ceiling, config.planning, config.defaults.as_ref(), 32).unwrap();
            let compiler = Int4SourceBatchPlanner::new_with_document_encoding(&planner, config.identity.clone(),
                config.task_ceiling, config.planning, None, 32).unwrap();
            assert_eq!(compiler.document_encoding(), &policy());
            assert!(compiler.require_byte_encoding().is_err());
        }
    }
}

#[test]
fn host_revalidation_rejects_stale_byte_and_changed_compact_policy_identities() {
    let byte = planner();
    let compact = planner().with_document_encoding(policy()).unwrap();
    let UntrustedEncoding::SafeBpeV1(mut encoding) = policy() else { unreachable!() };
    encoding.max_pair_checks -= 1;
    let changed = planner().with_document_encoding(UntrustedEncoding::SafeBpeV1(encoding)).unwrap();
    for task in [BuiltInTask::Ner, BuiltInTask::Keyphrases, BuiltInTask::Summarize, BuiltInTask::Answer] {
        let compact_config = config(&compact, task, 32);
        compact_config.validate_planning(&compact, 32, compact_config.task_ceiling.max_kv_bytes).unwrap();
        assert!(compact_config.validate_planning(&byte, 32, compact_config.task_ceiling.max_kv_bytes).is_err());
        assert!(compact_config.validate_planning(&changed, 32, compact_config.task_ceiling.max_kv_bytes).is_err());
        let byte_config = config(&byte, task, 32);
        byte_config.validate_planning(&byte, 32, byte_config.task_ceiling.max_kv_bytes).unwrap();
        assert!(byte_config.validate_planning(&compact, 32, byte_config.task_ceiling.max_kv_bytes).is_err());
        let changed_config = config(&changed, task, 32);
        changed_config.validate_planning(&changed, 32, changed_config.task_ceiling.max_kv_bytes).unwrap();
        assert_eq!(compact_config.identity.tokenizer_digest, changed_config.identity.tokenizer_digest);
        assert_ne!(compact_config.identity.template_digest, changed_config.identity.template_digest);
    }
}

#[test]
fn encoding_workspace_does_not_widen_task_context_output_or_resident_kv() {
    let planner = planner().with_document_encoding(policy()).unwrap();
    let baseline = config(&planner, BuiltInTask::Answer, 32);
    for axis in 0..5 {
        let mut invalid = config(&planner, BuiltInTask::Answer, 32);
        let mut allocated_kv = invalid.task_ceiling.max_kv_bytes;
        match axis {
            0 => invalid.planning.max_input_bytes = 0,
            1 => invalid.planning.max_context_tokens = 0,
            2 => invalid.task_ceiling.max_output_bytes = 0,
            3 => allocated_kv += 1,
            _ => invalid.selected_rows = Some(Int4JsonSparseLimits { max_rows_per_step: 0 }),
        }
        assert!(invalid.validate_planning(&planner, 32, allocated_kv).is_err(), "axis {axis}");
    }
    baseline.validate_planning(&planner, 32, baseline.task_ceiling.max_kv_bytes).unwrap();
}
