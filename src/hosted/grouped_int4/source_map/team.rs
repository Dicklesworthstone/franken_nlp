//! One sealed numerical team across complete source maps and exact reducers.
use super::*;
use super::{keyphrases::KeyphraseOperation, summary::SummaryOperation};
use crate::tasks::source_planning::grouped_int4::long::{
    keyphrases::{Int4CorpusKeyphraseRun, Int4KeyphraseLimits},
    summary::{Int4CorpusSummaryRun, Int4SummaryLimits},
};

impl NlpEngine {
    /// Execute every source chunk with one persistent output-row team. The
    /// original whole-document model/mask/verification budgets, exact source
    /// coordinates and result seals are shared with the serial route. Compact
    /// input, when explicitly selected by the planner, keeps its same one
    /// nonrenewable tokenizer ledger. Workers join before a result is returned.
    /// Independent chunk cohorts are a separate physical schedule.
    #[allow(clippy::too_many_arguments)]
    pub fn map_int4_source_with_threads(&self, model: &ResidentInt4, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int4SourceMapConfig, kernel: PackedInt4Kernel, compute_threads: usize,
        cancellation: CancellationToken) -> Result<HostedInt4TeamResult<Int4SourceMapRun>, HostedError> {
        execute_with_threads(self, model, source, planner, vocabulary, config,
            kernel, compute_threads, cancellation, MapOperation)
    }

    /// Keep one numerical team across every cited-summary map and the final
    /// exact bullet/evidence union. Every candidate remains present until the
    /// complete root applies its declared cap. This does not enable neural
    /// synthesis, hierarchical reasoning or a semantic-entailment claim.
    #[allow(clippy::too_many_arguments)]
    pub fn summarize_int4_source_with_threads(&self, model: &ResidentInt4, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int4SourceMapConfig, summary: Int4SummaryLimits, kernel: PackedInt4Kernel,
        compute_threads: usize, cancellation: CancellationToken)
        -> Result<HostedInt4TeamResult<Int4CorpusSummaryRun>, HostedError> {
        execute_with_threads(self, model, source, planner, vocabulary, config,
            kernel, compute_threads, cancellation, SummaryOperation(summary))
    }

    /// Keep one numerical team across every keyphrase map and the complete
    /// exact-text candidate/evidence union. Global top-k is applied only once;
    /// selecting a team cannot shrink the admitted frontier or renew a budget.
    #[allow(clippy::too_many_arguments)]
    pub fn reduce_int4_keyphrases_document_with_threads(&self, model: &ResidentInt4, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int4SourceMapConfig, keyphrases: Int4KeyphraseLimits, kernel: PackedInt4Kernel,
        compute_threads: usize, cancellation: CancellationToken)
        -> Result<HostedInt4TeamResult<Int4CorpusKeyphraseRun>, HostedError> {
        execute_with_threads(self, model, source, planner, vocabulary, config,
            kernel, compute_threads, cancellation, KeyphraseOperation(keyphrases))
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_with_threads<O: DocumentOperation>(host: &NlpEngine, model: &ResidentInt4, source: String,
    planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>, config: Int4SourceMapConfig,
    kernel: PackedInt4Kernel, compute_threads: usize, cancellation: CancellationToken, operation: O)
    -> Result<HostedInt4TeamResult<O::Output>, HostedError> {
    // Process width and complete native/team arithmetic are checked before
    // source capture. The dispatcher owns the one worker-scratch reservation
    // until every child has physically joined, including cancellation/panic.
    dispatch::preflight_int4_team(host, config.native.run, compute_threads)?;
    team_requirements(&config, model.model_identity().group_width(), compute_threads)?;
    let run = config.native.run;
    let admitted = admit_document(host, model, source, planner, vocabulary, config, &operation)?;
    dispatch::run_int4_team(host, run, cancellation, model.clone(), compute_threads,
        kernel, move |weights, control, team| {
            let output = execute_document(admitted, weights, kernel, control, Some(&mut *team), operation)?;
            // The native engine has dropped its executor borrow. These are
            // physical counters only; they never enter the semantic result.
            let compute = team.stats().into();
            Ok(HostedInt4TeamResult::new(output, compute))
        })
}

fn team_requirements(config: &Int4SourceMapConfig, group_width: usize, compute_threads: usize)
    -> Result<u64, HostedError> {
    if config.mapping.reduction.map_batch_chunks != 1 {
        return Err(HostedError::Limits("INT4 source document teams require serial chunk scheduling"));
    }
    let extra = int4_task_team_workspace_bytes(compute_threads, group_width)?;
    let native = Int4MemoryRequirement::for_context(config.native.context_tokens, group_width).map_err(native_error)?;
    sum(&[native.rope_bytes, native.scratch_payload_bound, config.native.allocator_reserve_bytes, extra])?;
    Ok(extra)
}

#[cfg(test)]
mod tests;
