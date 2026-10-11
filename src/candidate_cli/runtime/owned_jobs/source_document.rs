//! Native source-document jobs prepared before weight loading, with fixed tasks.
use super::*;
use crate::{
    candidate_cli::jobs::source_document as settings,
    batch::source::SourceBatchArgs,
    jobs::runner::source::document::Int8SourceDocumentJobPlanner,
    tasks::source_planning::quantized::long::SourceMapTask,
};

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare(args: &JobArgs, facts: &ArtifactIdentity, defaults: Defaults,
    text: &str, lifetime: JobLimits, limits: Limits, control: &mut impl DecodeStepControl)
    -> Result<(PreparedJob, Arc<ExtractionVocabulary>), Failure> {
    checkpoint(control)?;
    settings::validate_mode(args)?;
    let Defaults::Source(Some(request)) = defaults else {
        return Err(Failure::usage("source_document_job_requires_fixed_task"));
    };
    let (task, budget, name) = match request {
        SourceBatchArgs::Ner { options, budget } => (SourceMapTask::Ner(options), budget, "ner"),
        SourceBatchArgs::Keyphrases { options, budget } => (SourceMapTask::Keyphrases(options), budget, "keyphrases"),
        SourceBatchArgs::Summarize { options, budget } => (SourceMapTask::Summarize(options), budget, "summarize"),
        SourceBatchArgs::Answer { .. } => return Err(Failure::usage("source_document_job_requires_fixed_task")),
    };
    if name != args.task { return Err(Failure::usage("source_document_job_task_mismatch")); }
    let (mapping, reduction_reserve_bytes) = settings::load(text, args, budget, lifetime, limits)?;
    checkpoint(control)?;
    let registry = pinned_controls::pinned().map_err(|_| CandidateError::Identity)?;
    let controls = registry.template_controls();
    let eos = controls.entries().iter().find(|e| e.special && e.surface == crate::template::IM_END)
        .map(|e| e.id).ok_or(CandidateError::Identity)?;
    let mut identity = candidate_identity(facts)?;
    identity.task_spec = args.task_kind()?.spec().identity();
    // Factory binds the actual pinned tokenizer/template; no second planner or
    // temporary document is needed to derive the exact source scaffold budget.
    let planner = Int8SourceDocumentJobPlanner::pinned(controls, eos, identity, task, budget,
        args.host.planning(), mapping, lifetime, control).map_err(planner_failure)?;
    checkpoint(control)?;
    let vocabulary = Arc::new(ExtractionVocabulary::pinned(controls).map_err(|_| CandidateError::Planning)?);
    checkpoint(control)?;
    Ok((PreparedJob::SourceDocument { planner, reduction_reserve_bytes }, vocabulary))
}
fn planner_failure(fault: crate::batch::BatchFault) -> Failure {
    if let Some(cause) = fault.cancellation { return cancelled(cause); }
    Failure::usage("source_document_job_planning")
}

#[cfg(test)] mod tests;
