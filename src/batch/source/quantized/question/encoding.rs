//! One tokenizer operation owns the complete original source/question pair.
use super::*;
use serde::Serialize;

/// Private recipe projection; no question or source content is included.
#[derive(Serialize)]
pub(crate) struct EncodingRecipe {
    ownership: &'static str,
    accounting: &'static str,
    policy: UntrustedEncoding,
    complete_work_limits: SafeBpeWorkLimits,
}
pub(crate) fn recipe(planner: &SourceTaskPlanner) -> Result<Option<EncodingRecipe>, BatchFault> {
    planner.map_document_encoding_workspace_bytes().map_err(|_| BatchCode::InvalidLimits)?;
    Ok(planner.map_document_encoding_limits().map(|complete_work_limits| EncodingRecipe {
        ownership: "one-original-source-question-prepare-and-execute-reconstruction-v1",
        accounting: "invocation-local-tokenizer-work-not-journal-debits-v1",
        policy: planner.document_encoding().clone(), complete_work_limits,
    }))
}
pub(super) fn configure(planner: &SourceTaskPlanner, policy: UntrustedEncoding, complete: SafeBpeWorkLimits)
    -> Result<Arc<SourceTaskPlanner>, BatchFault> {
    let planner = planner.fork_map_document_operation().map_err(|_| BatchCode::InvalidLimits)?
        .with_map_document_encoding(policy, complete).map_err(|_| BatchCode::InvalidLimits)?;
    Ok(Arc::new(planner))
}
pub(super) fn operation(planner: &Arc<SourceTaskPlanner>) -> Result<Arc<SourceTaskPlanner>, BatchItemFailure> {
    if planner.map_document_encoding_limits().is_none() { return Ok(Arc::clone(planner)); }
    let operation = planner.fork_map_document_operation().map_err(|_| BatchItemFailure::fatal(BatchCode::InvalidLimits))?;
    Ok(Arc::new(operation))
}
pub(super) fn verify_operation(factory: &SourceTaskPlanner, operation: &SourceTaskPlanner) -> Result<(), BatchItemFailure> {
    if operation.template_digest() != factory.template_digest() || operation.tokenizer_digest() != factory.tokenizer_digest()
        || operation.document_encoding() != factory.document_encoding()
        || operation.map_document_encoding_limits() != factory.map_document_encoding_limits() {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    Ok(())
}
