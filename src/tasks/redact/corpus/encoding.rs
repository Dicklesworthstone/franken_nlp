//! One encoder ledger per original document, retained through edited-source verification.
use super::*;

pub(super) fn operation(factory: &SourceTaskPlanner)
    -> Result<Option<SourceTaskPlanner>, BatchItemFailure> {
    factory.map_document_encoding_workspace_bytes()
        .map_err(|_| BatchItemFailure::fatal(BatchCode::InvalidLimits))?;
    if factory.map_document_encoding_limits().is_none() { return Ok(None); }
    factory.fork_map_document_operation().map(Some)
        .map_err(|_| BatchItemFailure::fatal(BatchCode::InvalidLimits))
}

/// A prepared record retains the only mutable counter used for its preflight,
/// execution reconstruction and edited-source scan. The factory never lends
/// its own unused counter to a document or reports it as population work.
pub(super) fn retained<'p>(factory: &'p SourceTaskPlanner, operation: Option<&'p SourceTaskPlanner>)
    -> Result<&'p SourceTaskPlanner, BatchItemFailure> {
    let operation = match (factory.map_document_encoding_limits(), operation) {
        (None, None) => factory,
        (Some(_), Some(operation)) => operation,
        _ => return Err(BatchItemFailure::fatal(BatchCode::Admission)),
    };
    if operation.template_digest() != factory.template_digest()
        || operation.tokenizer_digest() != factory.tokenizer_digest()
        || operation.document_encoding() != factory.document_encoding()
        || operation.map_document_encoding_limits() != factory.map_document_encoding_limits() {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    Ok(operation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::{embedded::EmbeddedTokenizer, pinned_controls,
        untrusted::{SafeBpeLimits, SafeBpeWork, SafeBpeWorkLimits, UntrustedEncoding}};

    fn planner() -> SourceTaskPlanner {
        let tokenizer = EmbeddedTokenizer::pinned().unwrap();
        let controls = pinned_controls::pinned().unwrap();
        SourceTaskPlanner::pinned(controls.template_controls(), tokenizer.eos_token_id().unwrap()).unwrap()
    }
    fn call() -> SafeBpeLimits {
        SafeBpeLimits { max_input_bytes: 16, max_output_tokens: 16,
            max_pair_checks: 100_000, max_pair_bytes: 1_000_000 }
    }
    fn compact(bytes: u64) -> SourceTaskPlanner {
        planner().with_map_document_encoding(UntrustedEncoding::SafeBpeV1(call()),
            SafeBpeWorkLimits { max_input_bytes: bytes, max_output_tokens: 8192,
                max_pair_checks: 1_000_000, max_pair_bytes: 32_000_000 }).unwrap()
    }

    #[test]
    fn reconstruction_and_redetection_cannot_renew_the_retained_operation() {
        let factory = compact(15);
        let operation = operation(&factory).unwrap().unwrap();
        for expected in [5, 10, 15] {
            let retained = retained(&factory, Some(&operation)).unwrap();
            retained.source_encoder().encode_with_control("alpha", 16, 16, &mut || None).unwrap();
            assert_eq!(retained.map_document_encoding_work().unwrap().unwrap().input_bytes(), expected);
        }
        let before = operation.map_document_encoding_work().unwrap();
        assert!(retained(&factory, Some(&operation)).unwrap().source_encoder()
            .encode_with_control("alpha", 16, 16, &mut || None).is_err());
        assert_eq!(operation.map_document_encoding_work().unwrap(), before);
        assert_eq!(factory.map_document_encoding_work().unwrap(), Some(SafeBpeWork::default()));
        let next = super::operation(&factory).unwrap().unwrap();
        next.source_encoder().encode_with_control("alpha", 16, 16, &mut || None).unwrap();
        assert_eq!(next.map_document_encoding_work().unwrap().unwrap().input_bytes(), 5);
        assert_eq!(operation.map_document_encoding_work().unwrap(), before);
    }

    #[test]
    fn operation_policy_and_complete_limits_must_match_the_preparing_factory() {
        let factory = compact(15);
        let operation = operation(&factory).unwrap().unwrap();
        let changed = compact(16);
        assert_eq!(retained(&changed, Some(&operation)).err().unwrap().fault.code, BatchCode::Admission);
        assert_eq!(retained(&factory, None).err().unwrap().fault.code, BatchCode::Admission);
        assert_eq!(retained(&planner(), Some(&operation)).err().unwrap().fault.code, BatchCode::Admission);
    }

    #[test]
    fn byte_mode_reuses_its_planner_and_one_call_compact_mode_is_refused() {
        let byte = planner();
        assert!(operation(&byte).unwrap().is_none());
        assert!(std::ptr::eq(retained(&byte, None).unwrap(), &byte));
        let unbounded = planner().with_document_encoding(UntrustedEncoding::SafeBpeV1(call())).unwrap();
        assert_eq!(operation(&unbounded).err().unwrap().fault.code, BatchCode::InvalidLimits);
    }
}
