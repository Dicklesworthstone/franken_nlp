//! Complete cited-summary reduction over the same charged INT4 map host.
use super::*;
use crate::tasks::source_planning::grouped_int4::long::summary::{
    Int4CorpusSummaryError, Int4CorpusSummaryRun, Int4SummaryLimits,
};
mod synthesis;

impl NlpEngine {
    /// Summarize every original chunk and return one deterministic ranked
    /// bullet/evidence union. Every candidate survives intermediate reduction;
    /// only the complete final root may apply the declared bullet cap.
    /// Citations prove source membership, not semantic entailment.
    #[allow(clippy::too_many_arguments)]
    pub fn summarize_int4_source(&self, model: &ResidentInt4, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int4SourceMapConfig, summary: Int4SummaryLimits,
        kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<Int4CorpusSummaryRun>, HostedError> {
        let run = config.native.run;
        let operation = SummaryOperation(summary);
        let admitted = admit_document(self, model, source, planner, vocabulary, config, &operation)?;
        let model = model.clone();
        dispatch::run(self, run, cancellation, move |control| {
            execute_document(admitted, model.weights(), kernel, control, None, operation)
        })
    }
}

pub(super) struct SummaryOperation(pub(super) Int4SummaryLimits);
impl DocumentOperation for SummaryOperation {
    type Output = Int4CorpusSummaryRun;
    fn validate(&self, config: &Int4SourceMapConfig) -> Result<(), HostedError> {
        if !matches!(&config.task, SourceMapTask::Summarize(_)) {
            return Err(HostedError::Limits("INT4 complete summary requires summary maps"));
        }
        self.0.validate(config.mapping).map_err(summary_error)
    }
    fn allocation(&self, config: &Int4SourceMapConfig, prepared: &PreparedInt4SourceMap<'_>)
        -> Result<DocumentAllocation, HostedError> {
        prepared.check_summary(self.0).map_err(summary_error)?;
        // The final union omits native token transcripts. The full native
        // frontier remains charged separately until all reduction finishes.
        Ok(DocumentAllocation { workspace_bytes: config.workspace_bytes(prepared.chunk_count())?,
            output_bytes: self.0.max_result_bytes as u64, output_tokens: 0 })
    }
    fn execute(self, prepared: PreparedInt4SourceMap<'_>, admitted: &[ExecutionIdentity],
        engine: &mut StrictInt4Engine<'_>, vocabulary: &ExtractionVocabulary,
        control: &mut RunControl) -> Result<Self::Output, HostedError> {
        let expected = prepared.preflight_metadata();
        let result = prepared.execute_summary_with_control(admitted, engine, vocabulary, self.0, control)
            .map_err(summary_error)?;
        expected.verify_summary_completed_with_limits(&result, self.0).map_err(summary_error)?;
        Ok(result)
    }
}
fn summary_error(error: Int4CorpusSummaryError) -> HostedError {
    HostedError::Int4(Int4HostError::SourceSummary(error))
}
