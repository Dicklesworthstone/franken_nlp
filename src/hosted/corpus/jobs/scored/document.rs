//! Complete-document INT8 scoring inside the existing authenticated job host.
//! Planner/configuration owners, population, journal and native claims survive
//! every callback error and cancellation until the physical workers join.
use super::*;
use crate::{
    batch::{classify::quantized::long::CLASSIFICATION_DOCUMENT_CHUNK_METADATA_BYTES,
        sentiment::quantized::INT8_SENTIMENT_DOCUMENT_CHUNK_METADATA_BYTES},
    hosted::{Int8ClassificationMapHostConfig, Int8ClassificationMapMemoryRequirement},
    jobs::{JobInputMemory, runner::scored::document::{Int8ClassificationDocumentJobPlanner,
        Int8ClassificationDocumentJobProcessor, Int8SentimentDocumentJobPlanner,
        Int8SentimentDocumentJobProcessor}},
    tasks::{classify::{ClassificationLimits, quantized::long::Int8ClassificationMapLimits},
        sentiment::{SentimentLimits, quantized::long::{Int8SentimentMapLimits,
            memory::{Int8SentimentMapMemoryLimits, Int8SentimentMapMemoryRequirement}}}},
    tokenizer::untrusted::UntrustedEncoding,
};

/// Complete simultaneous ownership before private population or model IO.
/// Includes journal/input staging, map frontier, frozen chunk identities and
/// complete output. Loaded weights, cleanup, prompt morsels and CPU workers
/// remain separate charges. Input retention changes only population storage.
#[allow(clippy::too_many_arguments)]
pub fn int8_classification_document_job_requirements(task: TaskBudget,
    planning: ClassificationLimits, mapping: Int8ClassificationMapLimits,
    encoding: &UntrustedEncoding, limits: JobHostLimits, job: JobLimits, input: JobInputMemory)
    -> Result<Int8ClassificationMapMemoryRequirement, HostedJobError> {
    check_document_envelope(job, mapping.chunks.max_input_bytes, mapping.max_result_bytes,
        mapping.max_model_work)?;
    let mut required = classification_map_requirements(task, planning, mapping, encoding, limits)?;
    required.input_bytes = sum(&[required.input_bytes,
        limits.required_buffer_bytes_for_input(job, input)?])?;
    sum(&[required.input_bytes, required.native.kv_bytes, required.workspace_bytes,
        required.output_reservation_bytes])?;
    Ok(required)
}

#[allow(clippy::too_many_arguments)]
pub fn int8_sentiment_document_job_requirements(task: TaskBudget,
    planning: SentimentLimits, mapping: Int8SentimentMapLimits,
    encoding: &UntrustedEncoding, limits: JobHostLimits, job: JobLimits, input: JobInputMemory)
    -> Result<Int8SentimentMapMemoryRequirement, HostedJobError> {
    check_document_envelope(job, mapping.chunks.max_input_bytes, mapping.max_result_bytes,
        mapping.max_model_work)?;
    let mut required = sentiment_map_requirements(task, planning, mapping, encoding, limits)?;
    required.input_bytes = sum(&[required.input_bytes,
        limits.required_buffer_bytes_for_input(job, input)?])?;
    sum(&[required.input_bytes, required.native.kv_bytes, required.workspace_bytes,
        required.output_reservation_bytes])?;
    Ok(required)
}

fn classification_map_requirements(task: TaskBudget, planning: ClassificationLimits,
    mapping: Int8ClassificationMapLimits, encoding: &UntrustedEncoding, limits: JobHostLimits)
    -> Result<Int8ClassificationMapMemoryRequirement, HostedError> {
    let mut required = Int8ClassificationMapHostConfig { native: limits.native, planning, mapping,
        preparation_reserve_bytes: limits.preparation_reserve_bytes }
        .requirements_with_encoding(task, encoding)?;
    required.input_bytes = sum(&[required.input_bytes, metadata(mapping.chunks.max_chunks,
        CLASSIFICATION_DOCUMENT_CHUNK_METADATA_BYTES)?])?;
    Ok(required)
}
fn sentiment_map_requirements(task: TaskBudget, planning: SentimentLimits,
    mapping: Int8SentimentMapLimits, encoding: &UntrustedEncoding, limits: JobHostLimits)
    -> Result<Int8SentimentMapMemoryRequirement, HostedError> {
    let mut required = Int8SentimentMapMemoryLimits { context_tokens: limits.native.context_tokens,
        preparation_reserve_bytes: limits.preparation_reserve_bytes,
        allocator_reserve_bytes: limits.native.allocator_reserve_bytes, max_axes: 4 }
        .requirements(task, planning, mapping)
        .and_then(|required| required.with_input_encoding(encoding))
        .map_err(HostedError::SentimentMap)?;
    required.input_bytes = sum(&[required.input_bytes, metadata(mapping.chunks.max_chunks,
        INT8_SENTIMENT_DOCUMENT_CHUNK_METADATA_BYTES)?])?;
    Ok(required)
}
fn metadata(chunks: usize, per_chunk: u64) -> Result<u64, HostedError> {
    (chunks as u64).checked_mul(per_chunk)
        .ok_or(HostedError::Limits("scored document job identity storage"))
}
fn check_document_envelope(job: JobLimits, input_bytes: usize, output_bytes: u64,
    work: Int8Work) -> Result<(), HostedError> {
    job.validate().map_err(|_| HostedError::Limits("scored document lifetime limits"))?;
    let cap = job.max_work.model;
    if input_bytes > job.max_input_bytes_per_item || output_bytes > job.max_result_bytes as u64
        || work.forward_positions > cap.forward_positions || work.projected_logits > cap.projected_logits
        || work.attention_pairs > cap.attention_pairs || !work.projections.fits(cap.projections) {
        return Err(HostedError::Limits("complete scoring document exceeds immutable job envelope"));
    }
    Ok(())
}

impl NlpEngine {
    /// Retain complete classification documents on one resident engine. Each
    /// original document commits atomically as one item, with all chunk scores
    /// and its exact source-weighted reducer. Resume authenticates the source,
    /// labels, encoding, chunk/reducer policy and cumulative work allowances.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_classification_documents_scheduled<R>(&self, model: &ResidentInt8,
        planner: Arc<ClassificationPlanner>, config: ClassificationDocumentCorpusConfig,
        request: SourceJobRequest, limits: JobHostLimits, prefill: Option<Int8PrefillLimits>,
        team: Option<(Int8Kernel, usize)>, reader: R, cancellation: CancellationToken)
        -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        let factory = Int8ClassificationDocumentJobPlanner::new(planner, config.identity,
            config.task_ceiling, config.planning, config.mapping, config.defaults,
            config.max_model_work, config.max_tokenizer_work).map_err(HostedError::BatchSetup)?;
        factory.check_job_limits(request.limits).map_err(HostedError::BatchSetup)?;
        let required = classification_map_requirements(factory.task_ceiling(), factory.planning_limits(),
            factory.mapping_limits(), factory.document_encoding(), limits)?;
        if factory.retained_defaults_bytes().map_err(HostedError::BatchSetup)?
            > limits.preparation_reserve_bytes {
            return Err(HostedError::Limits("classification document job defaults reservation").into());
        }
        run(self, model, ClassificationDocumentTask { factory: Arc::new(factory), required },
            request, limits, prefill, team, reader, cancellation)
    }

    /// Retain complete independent sentiment dimensions over every original
    /// chunk. Failed documents never commit partial aggregates; all attempts
    /// and native work remain charged under the original authenticated job.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_sentiment_documents_scheduled<R>(&self, model: &ResidentInt8,
        planner: Arc<SentimentPlanner>, config: SentimentDocumentCorpusConfig,
        request: SourceJobRequest, limits: JobHostLimits, prefill: Option<Int8PrefillLimits>,
        team: Option<(Int8Kernel, usize)>, reader: R, cancellation: CancellationToken)
        -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        let factory = Int8SentimentDocumentJobPlanner::new(planner, config).map_err(HostedError::BatchSetup)?;
        factory.check_job_limits(request.limits).map_err(HostedError::BatchSetup)?;
        let required = sentiment_map_requirements(factory.task_ceiling(), factory.planning_limits(),
            factory.mapping_limits(), factory.document_encoding(), limits)?;
        if factory.retained_defaults_bytes().map_err(HostedError::BatchSetup)?
            > limits.preparation_reserve_bytes {
            return Err(HostedError::Limits("sentiment document job defaults reservation").into());
        }
        run(self, model, SentimentDocumentTask { factory: Arc::new(factory), required },
            request, limits, prefill, team, reader, cancellation)
    }
}

struct ClassificationDocumentTask {
    factory: Arc<Int8ClassificationDocumentJobPlanner>,
    required: Int8ClassificationMapMemoryRequirement,
}
impl Int8JobTask for ClassificationDocumentTask {
    fn identity(&self) -> &ExecutionIdentity { self.factory.execution_identity() }
    fn retain_for_join(&self) -> Option<Box<dyn Send>> { Some(Box::new(Arc::clone(&self.factory))) }
    fn validate(&self, native: Int8MemoryRequirement, job: JobLimits) -> Result<(), HostedError> {
        self.factory.check_job_limits(job).map_err(HostedError::BatchSetup)?;
        if native != self.required.native {
            return Err(HostedError::Limits("classification document job native geometry differs"));
        }
        Ok(())
    }
    fn preparation_workspace_bytes(&self) -> Result<u64, HostedError> { Ok(self.required.input_bytes) }
    fn execute(self, engine: &mut StrictInt8Engine<'_>, admission: CorpusAdmission<'_>,
        request: SourceJobRequest, population: &Population<'_>, prefill: Option<Int8PrefillLimits>,
        control: &mut RunControl) -> Result<Result<JobProgress, JobRunError>, HostedError> {
        let admission = crate::hosted::corpus::classify::document_admission(admission,
            self.factory.mapping_limits(), self.factory.task_ceiling(), self.factory.planning_limits().max_labels)?;
        Ok(Int8ClassificationDocumentJobProcessor::scheduled(self.factory, engine, admission, prefill)
            .map_err(JobRunError::Processor)
            .and_then(|processor| run_population(request, population, processor, control)))
    }
}

struct SentimentDocumentTask {
    factory: Arc<Int8SentimentDocumentJobPlanner>,
    required: Int8SentimentMapMemoryRequirement,
}
impl Int8JobTask for SentimentDocumentTask {
    fn identity(&self) -> &ExecutionIdentity { self.factory.execution_identity() }
    fn retain_for_join(&self) -> Option<Box<dyn Send>> { Some(Box::new(Arc::clone(&self.factory))) }
    fn validate(&self, native: Int8MemoryRequirement, job: JobLimits) -> Result<(), HostedError> {
        self.factory.check_job_limits(job).map_err(HostedError::BatchSetup)?;
        if native != self.required.native {
            return Err(HostedError::Limits("sentiment document job native geometry differs"));
        }
        Ok(())
    }
    fn preparation_workspace_bytes(&self) -> Result<u64, HostedError> { Ok(self.required.input_bytes) }
    fn execute(self, engine: &mut StrictInt8Engine<'_>, admission: CorpusAdmission<'_>,
        request: SourceJobRequest, population: &Population<'_>, prefill: Option<Int8PrefillLimits>,
        control: &mut RunControl) -> Result<Result<JobProgress, JobRunError>, HostedError> {
        Ok(Int8SentimentDocumentJobProcessor::scheduled(&self.factory, engine, admission, prefill)
            .map_err(JobRunError::Processor)
            .and_then(|processor| run_population(request, population, processor, control)))
    }
}

#[cfg(test)]
mod tests;
