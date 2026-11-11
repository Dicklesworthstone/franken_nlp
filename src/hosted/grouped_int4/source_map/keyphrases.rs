//! Complete deterministic keyphrase ranking with all native maps still owned.
use super::*;
use crate::tasks::source_planning::grouped_int4::long::keyphrases::{
    Int4CorpusKeyphraseError, Int4CorpusKeyphraseRun, Int4KeyphraseLimits,
};

impl<T> Int4SourceMapConfig<T> {
    /// Complete native frontier plus the independent exact-text candidate and
    /// occurrence union. The final top-k cannot reduce the admitted frontier.
    /// These are conservative owner allowances, not an observed RSS promise.
    pub fn keyphrase_workspace_bytes(&self, chunks: usize, limits: Int4KeyphraseLimits)
        -> Result<u64, HostedError> {
        self.workspace().keyphrases(chunks, limits)
    }
}

impl NlpEngine {
    /// Extract from every original chunk, independently verify every occurrence
    /// and return one exact-text ranking. All candidates survive intermediate
    /// reductions; final ordering uses supporting-chunk count, local rank sum,
    /// first source position and exact text. Repeated occurrences within one
    /// chunk contribute one support vote. This is not neural reranking.
    #[allow(clippy::too_many_arguments)]
    pub fn reduce_int4_keyphrases_document(&self, model: &ResidentInt4, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int4SourceMapConfig, keyphrases: Int4KeyphraseLimits,
        kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<Int4CorpusKeyphraseRun>, HostedError> {
        let run = config.native.run;
        let operation = KeyphraseOperation(keyphrases);
        let admitted = admit_document(self, model, source, planner, vocabulary, config, &operation)?;
        let model = model.clone();
        dispatch::run(self, run, cancellation, move |control| {
            execute_document(admitted, model.weights(), kernel, control, None, operation)
        })
    }
}

pub(super) struct KeyphraseOperation(pub(super) Int4KeyphraseLimits);
impl DocumentOperation for KeyphraseOperation {
    type Output = Int4CorpusKeyphraseRun;
    fn validate(&self, config: &Int4SourceMapConfig) -> Result<(), HostedError> {
        if !matches!(&config.task, SourceMapTask::Keyphrases(_)) {
            return Err(HostedError::Limits("INT4 complete keyphrase ranking requires keyphrase maps"));
        }
        // Final top-k cannot reduce the admitted complete candidate frontier.
        config.keyphrase_workspace_bytes(config.mapping.chunks.max_chunks, self.0)?;
        Ok(())
    }
    fn allocation(&self, config: &Int4SourceMapConfig, prepared: &PreparedInt4SourceMap<'_>)
        -> Result<DocumentAllocation, HostedError> {
        prepared.check_keyphrases(self.0).map_err(keyphrase_error)?;
        // The final ranking retains no generated-token transcripts. All
        // temporary native tokens remain charged in the complete frontier.
        Ok(DocumentAllocation { workspace_bytes: config.keyphrase_workspace_bytes(prepared.chunk_count(), self.0)?,
            output_bytes: self.0.max_result_bytes as u64, output_tokens: 0 })
    }
    fn execute(self, prepared: PreparedInt4SourceMap<'_>, admitted: &[ExecutionIdentity],
        engine: &mut StrictInt4Engine<'_>, vocabulary: &ExtractionVocabulary,
        control: &mut RunControl) -> Result<Self::Output, HostedError> {
        let expected = prepared.preflight_metadata();
        let result = prepared.execute_keyphrases_with_control(admitted, engine, vocabulary, self.0, control)
            .map_err(keyphrase_error)?;
        expected.verify_keyphrases_completed_with_limits(&result, self.0).map_err(keyphrase_error)?;
        Ok(result)
    }
}

#[cfg(test)]
fn keyphrase_requirements(config: &Int4SourceMapConfig, source_bytes: usize, width: usize,
    planner: &SourceTaskPlanner, limits: Int4KeyphraseLimits) -> Result<Int4MemoryRequirement, HostedError> {
    let required = map_requirements(config, source_bytes, width, planner)?;
    KeyphraseOperation(limits).validate(config)?;
    Ok(required)
}
fn keyphrase_error(error: Int4CorpusKeyphraseError) -> HostedError {
    HostedError::Int4(Int4HostError::SourceKeyphrases(error))
}

#[cfg(test)]
mod tests;
