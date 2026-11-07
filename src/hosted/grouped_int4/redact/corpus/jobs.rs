//! Retained complete-document INT4 redaction through the existing owned-job
//! protocol. One item commits one final edit, never intermediate NER chunks.
use super::*;
use crate::{
    hosted::corpus::jobs::{run_population, HostedJobError, JobHostLimits, SourceJobRequest},
    jobs::{JobLimits, JobProgress, population::JobPopulation, runner::{JobRunError,
        redact::grouped_int4::{self as redact_job, Int4RedactionJobConfig,
            Int4RedactionJobProcessor, INT4_REDACTION_JOB_RECIPE_BYTES}}},
};

struct JobPreparation {
    planner: Arc<SourceTaskPlanner>,
    vocabulary: Arc<ExtractionVocabulary>,
    config: RedactionCorpusConfig<Int4RedactionJobConfig>,
    secret: Option<RedactionPseudonyms>,
    request: SourceJobRequest,
}
struct JobInput<R> { preparation: Option<JobPreparation>, reader: R }

impl NlpEngine {
    /// Create or explicitly resume an owned complete-document redaction job
    /// from the complete original NDJSON population. The configured original
    /// text bound is independent of the encoded input-envelope bound.
    ///
    /// The existing runner authenticates the complete population, actual
    /// model/group, detector/chunk recipe, both grounding pools, lifetime work
    /// and full256 key scope before tail repair or native execution. Its result
    /// guard survives spool sync and journal acknowledgement. This API emits
    /// only metadata; retained edited text and maps stay in the protected job.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int4_redact_document<R>(&self, model: &ResidentInt4,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: RedactionCorpusConfig<Int4RedactionJobConfig>, pseudonyms: Option<RedactionPseudonyms>,
        request: SourceJobRequest, limits: JobHostLimits, kernel: PackedInt4Kernel,
        reader: R, cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.job_int4_redact_document_for_schedule(model, planner, vocabulary, config,
            pseudonyms, request, limits, kernel, reader, cancellation, false)
    }

    /// Create or resume the explicit native chunk-cohort schedule. The frozen
    /// recipe binds its execution marker, window width and context capacity;
    /// serial jobs retain their existing recipe and semantic result wire.
    /// Both scans reuse one admitted native peak and full-attempt work debit.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int4_redact_document_cohorts<R>(&self, model: &ResidentInt4,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: RedactionCorpusConfig<Int4RedactionJobConfig>, pseudonyms: Option<RedactionPseudonyms>,
        request: SourceJobRequest, limits: JobHostLimits, kernel: PackedInt4Kernel,
        reader: R, cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.job_int4_redact_document_for_schedule(model, planner, vocabulary, config,
            pseudonyms, request, limits, kernel, reader, cancellation, true)
    }

    #[allow(clippy::too_many_arguments)]
    fn job_int4_redact_document_for_schedule<R>(&self, model: &ResidentInt4,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: RedactionCorpusConfig<Int4RedactionJobConfig>, pseudonyms: Option<RedactionPseudonyms>,
        request: SourceJobRequest, limits: JobHostLimits, kernel: PackedInt4Kernel,
        reader: R, cancellation: CancellationToken, cohorts: bool) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        dispatch::preflight(self, limits.native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) {
            return Err(HostedError::ResourceDomain.into());
        }
        if cohorts {
            redact_job::check_cohort_configuration(&planner, &config.batch, model.model_identity(),
                request.limits, limits.native.context_tokens).map_err(HostedError::BatchSetup)?;
        } else {
            redact_job::check_configuration(&planner, &config.batch, model.model_identity(), request.limits)
                .map_err(HostedError::BatchSetup)?;
        }
        let width = model.model_identity().group_width();
        let required = if cohorts {
            job_requirements_for_schedule(&config, request.limits, limits, width, pseudonyms.as_ref(), true)?
        } else {
            job_requirements(&config, request.limits, limits, width, pseudonyms.as_ref())?
        };
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes)?,
            || Ok(JobInput { preparation: Some(JobPreparation {
                planner, vocabulary, config, secret: pseudonyms, request }), reader }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let workspace = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let model = model.clone();
        let completed = dispatch::run(self, limits.native.run, cancellation, move |control| {
            // Capture the complete charged owner, including queued cancellation.
            let mut input = input;
            checkpoint(control)?;
            let JobPreparation { planner, vocabulary, config, secret, request } =
                input.value.preparation.take().ok_or(HostedError::CompletionMissing)?;
            // The actual pseudonym key commitment is checked before any
            // population IO; the independent job secret authenticates storage.
            let context = pseudonym_context(secret.as_ref(), &config.batch.batch.request)?;
            let population = match JobPopulation::read_ndjson(&mut input.value.reader, &request.key,
                request.job_id, request.limits, limits.transport, control) {
                Ok(population) => population,
                Err(error) => return Ok(Err(JobRunError::Storage(error))),
            };
            checkpoint(control)?;
            // Each document contains independent native contexts. A
            // concatenated-sequence admission would reject valid work.
            let admission = DocumentAdmission {
                lease: &lease, model: model.model_identity(),
                fixed: FixedItem::new(&config.batch.batch, width, required.native.kv_bytes,
                    request.limits.max_result_bytes as u64),
            };
            let result = if let Some(quote) = required.cohort {
                let result = (|| {
                    let processor = Int4RedactionJobProcessor::new_cohorts(&planner, config.batch,
                        &model.inner.loaded.value, &vocabulary, context.as_ref(),
                        native_cohort_budget(quote, limits.native.context_tokens), kernel, admission)
                        .map_err(JobRunError::Processor)?;
                    run_population(request, &population, processor, control)
                })();
                // All exact-sized windows drain before returning; keep the
                // full reservation live through spool and journal commits.
                drop(workspace); drop(kv);
                result
            } else {
                let mut engine = allocate_native(kv, workspace, || {
                    StrictInt4Engine::new(&model.inner.loaded.value, limits.native.context_tokens, Int4MemoryBudget {
                        max_kv_bytes: required.native.kv_bytes, max_rope_bytes: required.native.rope_bytes,
                        max_scratch_payload_bytes: required.native.scratch_payload_bound,
                    }).map(|engine| engine.with_kernel(kernel)).map_err(native_error)
                })?;
                let result = (|| {
                    let processor = Int4RedactionJobProcessor::new(&planner, config.batch,
                        &mut engine.value, &vocabulary, context.as_ref(), admission)
                        .map_err(JobRunError::Processor)?;
                    run_population(request, &population, processor, control)
                })();
                drop(engine);
                result
            };
            // Runner, journal, spool and guarded outputs finish first.
            drop(population);
            drop(context);
            drop(secret);
            drop(vocabulary);
            drop(planner);
            drop(input);
            drop(lease);
            Ok(result)
        })?;
        // Physical cancellation/drain failure takes precedence even if the
        // last durable result committed before its acknowledgement was lost.
        completed.map_err(HostedJobError::Job)
    }
}

struct JobRequirements {
    native: Int4MemoryRequirement,
    workspace_bytes: u64,
    input_bytes: u64,
    cohort: Option<Int4SourceMapCohortRequirement>,
}
fn job_requirements(config: &RedactionCorpusConfig<Int4RedactionJobConfig>,
    job: JobLimits, limits: JobHostLimits, width: usize, secret: Option<&RedactionPseudonyms>)
    -> Result<JobRequirements, HostedJobError> {
    job_requirements_for_schedule(config, job, limits, width, secret, false)
}
fn job_requirements_for_schedule(config: &RedactionCorpusConfig<Int4RedactionJobConfig>,
    job: JobLimits, limits: JobHostLimits, width: usize, secret: Option<&RedactionPseudonyms>, cohorts: bool)
    -> Result<JobRequirements, HostedJobError> {
    redact_job::check_job_limits(&config.batch, job).map_err(HostedError::BatchSetup)?;
    let detector = &config.batch.batch.detector;
    let row = Int4MemoryRequirement::for_context(limits.native.context_tokens, width).map_err(native_error)?;
    if config.edit_reserve_bytes == 0 || limits.native.allocator_reserve_bytes == 0
        || row.kv_bytes == 0 || row.kv_bytes > detector.per_chunk.max_kv_bytes
        || detector.planning.max_context_tokens > limits.native.context_tokens
        || detector.mapping.chunks.context_tokens > limits.native.context_tokens
        || detector.max_result_bytes == 0 || detector.max_result_bytes > job.max_result_bytes as u64 {
        return Err(HostedError::Limits("INT4 redaction job context, output or editing reservation").into());
    }
    let cohort = cohorts.then(|| detector.cohort_requirements(limits.native.context_tokens, width))
        .transpose().map_err(long::document_error)?;
    let native = cohort.map_or(row, |quote| quote.native);
    // Population, journal and canonical serializer storage already include the
    // retained-result cap. The independent output guard prices the actual
    // detector envelope, which may be smaller than that stored-frame ceiling.
    let input_bytes = sum(&[limits.required_buffer_bytes(job)?,
        long::document_workspace_bytes(detector, config.edit_reserve_bytes)?,
        secret_bytes(secret)?, 4 * INT4_REDACTION_JOB_RECIPE_BYTES as u64,
        cohort_temporary_bytes(cohort)?])?;
    let workspace_bytes = sum(&[native.rope_bytes, native.scratch_payload_bound,
        limits.native.allocator_reserve_bytes])?;
    sum(&[input_bytes, native.kv_bytes, workspace_bytes])?;
    Ok(JobRequirements { native, workspace_bytes, input_bytes, cohort })
}

#[cfg(test)]
mod tests;
