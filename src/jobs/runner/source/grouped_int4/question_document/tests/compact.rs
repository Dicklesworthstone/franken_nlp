//! Explicit compact private-question attempts bind live and retained planning.
use super::*;
use crate::tokenizer::untrusted::{SafeBpeLimits, SafeBpeWorkLimits, UntrustedEncoding};

pub(super) fn policy() -> UntrustedEncoding {
    UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 8192, max_output_tokens: 8192,
        max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24 })
}
pub(super) fn allowance() -> SafeBpeWorkLimits {
    SafeBpeWorkLimits { max_input_bytes: 1 << 20, max_output_tokens: 1 << 20,
        max_pair_checks: 1 << 28, max_pair_bytes: 1 << 32 }
}
#[test]
fn configured_compiler_and_job_builder_capture_identical_policy_and_recipe() {
    for mode in 0..3 {
        let config = config(mode); let limits = limits(config);
        let direct = Int4QuestionDocumentJobPlanner::new(
            compiler(config).with_map_document_encoding(policy(), allowance()).unwrap(), limits).unwrap();
        let decorated = factory(mode, false, None).with_map_document_encoding(policy(), allowance()).unwrap();
        assert_eq!(direct.execution_identity(), decorated.execution_identity());
        assert_eq!(canonjson::canonical_bytes(direct.job_recipe()).unwrap(), canonjson::canonical_bytes(decorated.job_recipe()).unwrap());
        assert_eq!(direct.document_encoding(), &policy());
        assert_eq!(direct.map_document_encoding_limits(), Some(allowance()));
        let UntrustedEncoding::SafeBpeV1(call) = policy() else { unreachable!() };
        assert_eq!(direct.encoding_workspace_bytes().unwrap(), call.requirements().unwrap().workspace_bytes + 1024);
        let first = direct.prepare_with_control(document(SOURCE, QUESTION), &mut Continue).unwrap();
        let second = direct.prepare_with_control(document(SOURCE, QUESTION), &mut Continue).unwrap();
        assert_eq!(first.map_document_encoding_work().unwrap(), second.map_document_encoding_work().unwrap());
        assert!(first.map_document_encoding_work().unwrap().unwrap().input_bytes() > SOURCE.len() + QUESTION.len());
        assert_eq!(decorated.durable_work(&first).unwrap_err().fault.code, BatchCode::Admission);
        let original = frozen(&direct, &record(SOURCE, QUESTION)).binding;
        assert_eq!(original.compare(&frozen(&decorated, &record(SOURCE, QUESTION)).binding), Ok(()));
        assert_eq!(original.compare(&frozen(&direct, &record(SOURCE, "Where is named? PRIVATE_QUESTION")).binding),
            Err(JobError::Mismatch(MismatchField::Population)));
    }
}

#[test]
fn byte_recipe_is_unchanged_and_all_compact_ceilings_are_authenticated() {
    for mode in 0..3 {
        let byte = factory(mode, false, None);
        assert!(serde_json::to_value(byte.job_recipe()).unwrap().get("input_encoding").is_none());
        assert_eq!(byte.encoding_workspace_bytes().unwrap(), 0);
        let original = byte.with_map_document_encoding(policy(), allowance()).unwrap();
        let input = record(SOURCE, QUESTION); let baseline = frozen(&original, &input).binding;
        let recipe = canonjson::canonical_bytes(original.job_recipe()).unwrap();
        for axis in 0..8 {
            let UntrustedEncoding::SafeBpeV1(mut call) = policy() else { unreachable!() };
            let mut total = allowance();
            match axis {
                0 => call.max_input_bytes -= 1, 1 => call.max_output_tokens -= 1,
                2 => call.max_pair_checks -= 1, 3 => call.max_pair_bytes -= 1,
                4 => total.max_input_bytes -= 1, 5 => total.max_output_tokens -= 1,
                6 => total.max_pair_checks -= 1, _ => total.max_pair_bytes -= 1,
            }
            let changed = factory(mode, false, None).with_map_document_encoding(UntrustedEncoding::SafeBpeV1(call), total).unwrap();
            assert_ne!(canonjson::canonical_bytes(changed.job_recipe()).unwrap(), recipe);
            assert!(baseline.compare(&frozen(&changed, &input).binding).is_err(), "mode {mode}, axis {axis}");
        }
        let prepared = original.prepare_with_control(document(SOURCE, QUESTION), &mut Continue).unwrap();
        let mut total = allowance(); total.max_pair_checks -= 1;
        let changed = original.with_map_document_encoding(policy(), total).unwrap();
        assert_eq!(changed.durable_work(&prepared).unwrap_err().fault.code, BatchCode::Admission);
    }
}
