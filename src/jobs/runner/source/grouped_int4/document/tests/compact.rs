//! Compact complete-document recipes and per-attempt preparation ownership.
use super::*;
use crate::tokenizer::untrusted::{SafeBpeLimits, SafeBpeWork, SafeBpeWorkLimits, UntrustedEncoding};

fn call() -> SafeBpeLimits {
    SafeBpeLimits { max_input_bytes: 8192, max_output_tokens: 8192,
        max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24 }
}
fn allowance() -> SafeBpeWorkLimits {
    SafeBpeWorkLimits { max_input_bytes: 1 << 20, max_output_tokens: 1 << 20,
        max_pair_checks: 1 << 28, max_pair_bytes: 1 << 32 }
}
fn compact(index: usize) -> Int4SourceDocumentJobPlanner {
    factory(index).with_map_document_encoding(UntrustedEncoding::SafeBpeV1(call()), allowance()).unwrap()
}
fn rebuild(factory: &Int4SourceDocumentJobPlanner, prepared: &PreparedInt4SourceDocument)
    -> Result<(), BatchItemFailure> {
    factory.owns(prepared)?;
    factory.plan_with_operation(&prepared.source, &prepared.operation, &mut Continue)?.compare(prepared, &mut Continue)
}

#[test]
fn one_durable_attempt_keeps_its_encoder_through_preparation_and_reconstruction() {
    for index in 0..7 {
        let planner = compact(index);
        let before_freeze = planner.planner.map_document_encoding_work().unwrap();
        let manifest = freeze(&planner);
        assert_eq!(planner.planner.map_document_encoding_work().unwrap(), before_freeze);
        assert_eq!(before_freeze, Some(SafeBpeWork::default()));
        let first = prepare(&planner, SOURCE);
        let second = prepare(&planner, SOURCE);
        let once = first.map_document_encoding_work().unwrap().unwrap();
        assert_eq!(second.map_document_encoding_work().unwrap(), Some(once));
        assert!(!Arc::ptr_eq(&first.operation, &second.operation));
        rebuild(&planner, &first).unwrap();
        assert_eq!(first.map_document_encoding_work().unwrap(), Some(once.checked_add(once).unwrap()));
        assert_eq!(second.map_document_encoding_work().unwrap(), Some(once));
        assert_eq!(manifest.binding.compare(&freeze(&planner).binding), Ok(()));
        assert_eq!(planner.planner.map_document_encoding_work().unwrap(), before_freeze);
    }
}

#[test]
fn every_explicit_encoding_limit_changes_private_resume_authority() {
    for index in [0, 3, 4, 5, 6] {
        let byte = factory(index);
        assert_eq!(byte.encoding_workspace_bytes().unwrap(), 0);
        assert!(serde_json::to_value(byte.job_recipe()).unwrap().get("input_encoding").is_none());
        let original = compact(index); let original_binding = freeze(&original).binding;
        let recipe = canonjson::canonical_bytes(original.job_recipe()).unwrap();
        let projection = serde_json::to_value(original.job_recipe()).unwrap();
        assert_eq!(projection["input_encoding"]["ownership"], "one-original-document-prepare-and-execute-reconstruction-v1");
        assert_eq!(projection["input_encoding"]["policy"]["kind"], "safe_bpe_v1");
        assert_eq!(projection["input_encoding"]["complete_work_limits"]["max_input_bytes"], allowance().max_input_bytes);
        assert!(original_binding.compare(&freeze(&byte).binding).is_err());
        assert_eq!(original_binding.compare(&freeze(&compact(index)).binding), Ok(()));
        for axis in 0..8 {
            let mut call = call(); let mut total = allowance();
            match axis {
                0 => call.max_input_bytes -= 1, 1 => call.max_output_tokens -= 1,
                2 => call.max_pair_checks -= 1, 3 => call.max_pair_bytes -= 1,
                4 => total.max_input_bytes -= 1, 5 => total.max_output_tokens -= 1,
                6 => total.max_pair_checks -= 1, _ => total.max_pair_bytes -= 1,
            }
            let changed = factory(index).with_map_document_encoding(UntrustedEncoding::SafeBpeV1(call), total).unwrap();
            assert_ne!(canonjson::canonical_bytes(changed.job_recipe()).unwrap(), recipe, "mode {index}, axis {axis}");
            assert_ne!(changed.execution_identity().template_digest, original.execution_identity().template_digest);
            assert!(original_binding.compare(&freeze(&changed).binding).is_err(), "mode {index}, axis {axis}");
        }
    }
}

#[test]
fn exact_attempt_limits_do_not_renew_on_reconstruction_or_cross_prepared_owners() {
    for index in [2, 3, 4, 5, 6] {
        let probe = compact(index); let prepared = prepare(&probe, SOURCE);
        let once = prepared.map_document_encoding_work().unwrap().unwrap();
        let twice = once.checked_add(once).unwrap();
        let total = SafeBpeWorkLimits { max_input_bytes: twice.input_bytes() as u64,
            max_output_tokens: twice.output_tokens() as u64,
            max_pair_checks: twice.pair_checks(), max_pair_bytes: twice.pair_bytes() };
        let planner = factory(index).with_map_document_encoding(UntrustedEncoding::SafeBpeV1(call()), total).unwrap();
        let prepared = prepare(&planner, SOURCE);
        rebuild(&planner, &prepared).unwrap();
        assert_eq!(prepared.map_document_encoding_work().unwrap(), Some(twice));
        assert!(rebuild(&planner, &prepared).is_err());
        assert_eq!(prepared.map_document_encoding_work().unwrap(), Some(twice));
        // A later admitted retry owns a new attempt under exactly the same recipe.
        let next = prepare(&planner, SOURCE);
        assert_eq!(next.map_document_encoding_work().unwrap(), Some(once));
        let changed = planner.with_chunk_cohorts().unwrap();
        assert!(changed.owns(&next).is_err());
    }
}
