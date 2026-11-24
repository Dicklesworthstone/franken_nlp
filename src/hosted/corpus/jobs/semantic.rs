//! Authenticated INT8 jobs whose one result contains extraction and all review.
//! The existing runner owns durable attempts, replay and atomic acknowledgement.
use super::*;
use crate::{
    batch::extract::semantic::int8::Int8SemanticExtractionBatchLimits,
    hosted::corpus::semantic::{SemanticAdmission, SemanticAssets, encoding_ledger_bytes,
        semantic_host, team_bytes, validate_judge_planning},
    jobs::{JobWork, runner::extract::semantic::{
        Int8SemanticExtractionJobPlanner, Int8SemanticExtractionJobProcessor,
        SEMANTIC_EXTRACTION_JOB_RECIPE_BYTES}},
    tasks::{ir::TaskBudget, judge::JudgeLimits,
        extract::semantic::int8::Int8SemanticOptions},
    tokenizer::untrusted::{SafeBpeWorkLimits, UntrustedEncoding},
};

/// Complete job geometry. Preparation is part of input_bytes; resident weights,
/// caller-owned preparation and runtime cleanup remain separate admissions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int8SemanticJobRequirements {
    pub native: Int8MemoryRequirement,
    pub native_workspace_bytes: u64,
    pub preparation_bytes: u64,
    pub input_bytes: u64,
    pub output_bytes: u64,
    pub team_workspace_bytes: u64,
}
impl Int8SemanticJobRequirements {
    pub fn total_bytes(self) -> Result<u64, HostedError> {
        sum(&[self.native.kv_bytes, self.native_workspace_bytes, self.input_bytes,
            self.output_bytes, self.team_workspace_bytes])
    }
}

/// Pure full admission before private schema, key, population or model IO.
/// Every dynamic review plan coexists with the original extraction. Native and
/// mask ceilings are immutable lifetime authority; compact work is additionally
/// bounded per complete document and per explicitly admitted invocation.
#[allow(clippy::too_many_arguments)]
pub fn int8_semantic_job_requirements(extraction: TaskBudget,
    options: &Int8SemanticOptions, judge_planning: JudgeLimits,
    encoding: &UntrustedEncoding, encoding_work_limits: Option<SafeBpeWorkLimits>,
    work: Int8SemanticExtractionBatchLimits, limits: JobHostLimits, job: JobLimits,
    input: JobInputMemory, prefill: Option<Int8PrefillLimits>, compute_threads: Option<usize>)
    -> Result<Int8SemanticJobRequirements, HostedJobError> {
    options.validate().map_err(HostedError::SemanticExtraction)?;
    work.validate_item(options.max_model_work).map_err(HostedError::BatchSetup)?;
    validate_judge_planning(judge_planning)?;
    let ledger = encoding_ledger_bytes(encoding, encoding_work_limits)?;
    let job_buffers = limits.required_buffer_bytes_for_input(job, input)?;
    let lifetime = JobWork { model: work.max_model_work, mask_node_visits: work.masks.max_visits_per_run };
    if job.max_work != lifetime {
        return Err(HostedError::Limits("INT8 semantic job lifetime native and mask work").into());
    }
    if options.limits.max_output_bytes > job.max_result_bytes as u64 {
        return Err(HostedError::Limits("INT8 semantic job complete result envelope").into());
    }
    let required = semantic_host(limits.native, limits.preparation_reserve_bytes, work)
        .requirements_with_schedule(extraction, options.specification.judge_budget,
            options.limits, judge_planning, job.max_input_bytes_per_item, encoding, prefill)?;
    let preparation_bytes = preparation_with_recipe(required.preparation_bytes, ledger)?;
    let staging = job_buffers.checked_sub(limits.preparation_reserve_bytes)
        .ok_or(HostedError::Limits("INT8 semantic job preparation arithmetic"))?;
    let quote = Int8SemanticJobRequirements {
        native: required.native, native_workspace_bytes: required.native_workspace_bytes,
        preparation_bytes, input_bytes: sum(&[staging, preparation_bytes])?,
        output_bytes: required.output_bytes, team_workspace_bytes: team_bytes(prefill, compute_threads)?,
    };
    quote.total_bytes()?;
    Ok(quote)
}
fn preparation_with_recipe(semantic_bytes: u64, ledger: u64) -> Result<u64, HostedError> {
    // Bounded serialized and canonical private recipe buffers coexist while
    // binding the exact schema/review policy. Neither replaces compiler memory.
    let recipe = SEMANTIC_EXTRACTION_JOB_RECIPE_BYTES as u64;
    sum(&[semantic_bytes, ledger, recipe, recipe])
}

struct SemanticJob {
    factory: Int8SemanticExtractionJobPlanner,
    assets: Arc<SemanticAssets>,
    limits: JobHostLimits,
    required: Int8SemanticJobRequirements,
    max_source_bytes: usize,
}
impl NlpEngine {
    /// Start or resume a fixed-schema semantic extraction population. Each
    /// attempt durably reserves its full extraction-and-review allowance before
    /// inference; a failed stage acknowledges no partial result. Committed
    /// records are skipped under the existing authenticated resume protocol.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_semantic_extract<R>(&self, model: &ResidentInt8,
        factory: Int8SemanticExtractionJobPlanner, vocabulary: Arc<ExtractionVocabulary>,
        request: SourceJobRequest, limits: JobHostLimits, reader: R, cancellation: CancellationToken)
        -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.job_int8_semantic_extract_scheduled(model, factory, vocabulary, request, limits,
            None, None, reader, cancellation)
    }

    /// Physical prefill/team widths may change on authenticated resume. They
    /// do not change the private semantic recipe or renew lifetime attempts,
    /// native work or masks. Actual compilers, inputs and output authority stay
    /// owned through the worker join, including failure and unwinding paths.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_semantic_extract_scheduled<R>(&self, model: &ResidentInt8,
        factory: Int8SemanticExtractionJobPlanner, vocabulary: Arc<ExtractionVocabulary>,
        request: SourceJobRequest, limits: JobHostLimits, prefill: Option<Int8PrefillLimits>,
        team: Option<(Int8Kernel, usize)>, reader: R, cancellation: CancellationToken)
        -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        match team {
            None => dispatch::preflight(self, limits.native.run)?,
            Some((_, width)) => dispatch::preflight_int8_team(self, limits.native.run, width)?,
        }
        self.check_resident_domain(model)?;
        factory.check_job_limits(request.limits).map_err(HostedError::BatchSetup)?;
        for identity in [factory.execution_identity(), factory.judge_identity()] {
            check_model_identity(model.artifact_identity(), identity)?;
        }
        let config = factory.config();
        let required = int8_semantic_job_requirements(config.extraction.budget,
            &config.options, config.judge_planning, &config.encoding, config.encoding_work_limits,
            config.work, limits, request.limits, request.mode.input_memory(), prefill,
            team.map(|(_, width)| width))?;
        let assets = SemanticAssets::admit(self, factory.retain_compiler(), vocabulary, required.output_bytes)?;
        let task = SemanticJob { factory, assets, limits, required,
            max_source_bytes: request.limits.max_input_bytes_per_item };
        scored::run(self, model, task, request, limits, prefill, team, reader, cancellation)
    }
}
impl scored::Int8JobTask for SemanticJob {
    fn identity(&self) -> &ExecutionIdentity { self.factory.execution_identity() }
    fn retain_for_join(&self) -> Option<Box<dyn Send>> {
        Some(Box::new((Arc::clone(&self.assets), self.factory.retain_recipe())))
    }
    fn validate(&self, native: Int8MemoryRequirement, job: JobLimits) -> Result<(), HostedError> {
        self.factory.check_job_limits(job).map_err(HostedError::BatchSetup)?;
        if native != self.required.native || job.max_input_bytes_per_item != self.max_source_bytes {
            return Err(HostedError::Limits("INT8 semantic job admitted geometry"));
        }
        Ok(())
    }
    fn preparation_workspace_bytes(&self) -> Result<u64, HostedError> {
        self.required.preparation_bytes.checked_sub(self.limits.preparation_reserve_bytes)
            .ok_or(HostedError::Limits("INT8 semantic job extra preparation"))
    }
    fn execute(self, engine: &mut StrictInt8Engine<'_>, admission: CorpusAdmission<'_>,
        request: SourceJobRequest, population: &Population<'_>, prefill: Option<Int8PrefillLimits>,
        control: &mut RunControl) -> Result<Result<JobProgress, JobRunError>, HostedError> {
        let Self { factory, assets, limits, required, max_source_bytes } = self;
        let admission = SemanticAdmission { inner: admission, assets: Arc::clone(&assets),
            host: semantic_host(limits.native, limits.preparation_reserve_bytes, assets.planner.limits()),
            max_source_bytes, preparation_bytes: required.preparation_bytes,
            output_bytes: required.output_bytes, prefill };
        let result = (|| {
            let processor = match prefill {
                None => Int8SemanticExtractionJobProcessor::new(factory, engine, &assets.vocabulary, admission),
                Some(prefill) => Int8SemanticExtractionJobProcessor::new_layer_major(
                    factory, engine, &assets.vocabulary, admission, prefill),
            }.map_err(JobRunError::Processor)?;
            run_population(request, population, processor, control)
        })();
        Ok(result)
    }
}

#[cfg(test)] mod tests;
