//! Explicit compact ownership for one original document, never each chunk.
use super::*;
use crate::tokenizer::untrusted::{SafeBpeWorkLimits, UntrustedEncoding};

/// This optional projection stays in private job recipes. Byte mode omits it
/// entirely, preserving existing authenticated recipe bytes and resume keys.
#[derive(Serialize)]
pub(crate) struct EncodingRecipe {
    ownership: &'static str,
    accounting: &'static str,
    policy: UntrustedEncoding,
    complete_work_limits: SafeBpeWorkLimits,
}
pub(crate) fn recipe(planner: &SourceTaskPlanner) -> Result<Option<EncodingRecipe>, BatchFault> {
    planner.map_document_encoding_workspace_bytes().map_err(|_| BatchCode::InvalidLimits)?;
    match planner.map_document_encoding_limits() {
        None => Ok(None),
        Some(complete_work_limits) => Ok(Some(EncodingRecipe {
            ownership: "one-original-document-prepare-and-execute-reconstruction-v1",
            accounting: "invocation-local-tokenizer-work-not-journal-debits-v1",
            policy: planner.document_encoding().clone(), complete_work_limits,
        })),
    }
}

pub(crate) fn supported(task: &Int4SourceDocumentTask) -> bool {
    matches!(task, Int4SourceDocumentTask::Map(_) | Int4SourceDocumentTask::Keyphrases { .. }
        | Int4SourceDocumentTask::Summary { .. } | Int4SourceDocumentTask::Question(_))
}
pub(crate) fn configure(planner: &SourceTaskPlanner, policy: UntrustedEncoding,
    complete: SafeBpeWorkLimits) -> Result<Arc<SourceTaskPlanner>, BatchFault> {
    let planner = planner.fork_map_document_operation().map_err(|_| BatchCode::InvalidLimits)?
        .with_map_document_encoding(policy, complete).map_err(|_| BatchCode::InvalidLimits)?;
    Ok(Arc::new(planner))
}

/// A prepared record retains this exact object for its second compilation.
/// The factory's policy remains immutable and its unused counter is never a
/// population-wide allowance or a source of successful-work telemetry.
pub(crate) fn operation(planner: &Arc<SourceTaskPlanner>) -> Result<Arc<SourceTaskPlanner>, BatchItemFailure> {
    if planner.map_document_encoding_limits().is_none() {
        return Ok(Arc::clone(planner));
    }
    let operation = planner.fork_map_document_operation()
        .map_err(|_| BatchItemFailure::fatal(BatchCode::InvalidLimits))?;
    Ok(Arc::new(operation))
}
pub(crate) fn verify_operation(factory: &SourceTaskPlanner, operation: &SourceTaskPlanner)
    -> Result<(), BatchItemFailure> {
    if operation.template_digest() != factory.template_digest()
        || operation.tokenizer_digest() != factory.tokenizer_digest()
        || operation.document_encoding() != factory.document_encoding()
        || operation.map_document_encoding_limits() != factory.map_document_encoding_limits() {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    Ok(())
}
