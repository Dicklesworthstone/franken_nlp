//! Authenticated retained redaction. The existing native stream adapter owns
//! detection, editing, fresh verification, work accounting and output guards.
use super::*;
use crate::{
    batch::{BatchDocument, BatchProcessor, BatchRequestContext, BatchWork},
    jobs::{JobProgress, JobWork, population::JobPopulation, runner::{DurableBatchProcessor, JobRunError}},
    native_engine::decode::DecodeStepControl,
    hosted::corpus::jobs::{run_population, HostedJobError, JobHostLimits, SourceJobRequest},
    tasks::redact::batch::{RedactionBatchArgs, Int8RedactionBatchAdmission},
};
mod recipe;
use recipe::{RedactionJobRecipe, RECIPE_BYTES};

impl NlpEngine {
    /// Create or explicitly resume one item-local redaction job. Original
    /// inputs, model, all policies/limits and the ACTUAL pseudonym key/scope
    /// authenticate before repair or inference. Only fully verified edited
    /// documents may commit. The supplied request is explicit retention consent;
    /// stored results and optional coordinate maps remain sensitive content.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_redact<R>(&self, model: &ResidentInt8,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: RedactionCorpusConfig, pseudonyms: Option<RedactionPseudonyms>,
        request: SourceJobRequest, limits: JobHostLimits, reader: R,
        cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        dispatch::preflight(self, limits.native.run)?;
        self.check_resident_domain(model)?;
        check_model_identity(model.artifact_identity(), &config.batch.ner_identity)?;
        recipe::bounded(&config.batch.request)?;
        recipe::bounded(&config.batch.detector.ner)?;
        redaction_batch::check_configuration(&planner, &config.batch).map_err(HostedError::BatchSetup)?;
        let required = requirements(limits.native)?;
        capacity(config.batch.detector.planning.max_context_tokens, config.batch.detector.per_pass.max_kv_bytes,
            config.batch.detector.max_result_bytes, config.edit_reserve_bytes,
            limits.native.context_tokens, required.kv_bytes, request.limits.max_result_bytes)?;
        let buffers = sum(&[limits.required_buffer_bytes(request.limits)?, temporary_bytes(&config)?,
            secret_bytes(pseudonyms.as_ref())?, 4 * RECIPE_BYTES as u64])?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, buffers)?,
            || Ok(StreamInput { planner: Some((planner, vocabulary, config, pseudonyms, request)), reader, writer: () }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch,
            sum(&[required.rope_bytes, required.scratch_payload_bound, limits.native.allocator_reserve_bytes])?)?;
        let model = model.clone();
        let completed = dispatch::run(self, limits.native.run, cancellation, move |control| {
            // Capture the entire charged owner, also on queued cancellation.
            let mut input = input;
            let (planner, vocabulary, config, secret, request) = input.value.planner.take()
                .ok_or(HostedError::CompletionMissing)?;
            let context = pseudonym_context(secret.as_ref(), &config.batch.request)?;
            let identity = config.batch.ner_identity.clone();
            let recipe = RedactionJobRecipe::short(&config.batch, context.as_ref())?;
            let population = match JobPopulation::read_ndjson(&mut input.value.reader, &request.key,
                request.job_id, request.limits, limits.transport, control) {
                Ok(p) => p, Err(e) => return Ok(Err(JobRunError::Storage(e))),
            };
            let mut engine = allocate_native(kv, scratch, || model.inner.loaded.value
                .engine(limits.native.context_tokens, memory_budget(required)).map_err(HostedError::Model))?;
            let admission = CorpusAdmission { lease: &lease, model: model.artifact_identity(),
                kv_bytes: required.kv_bytes, sampler_bytes: 0, output_bytes: request.limits.max_result_bytes as u64 };
            let result = (|| {
                let native = NativeInt8RedactionBatch::new(&planner, config.batch,
                    &mut engine.value, &vocabulary, context.as_ref(), admission).map_err(JobRunError::Processor)?;
                run_population(request, &population, RedactionJob { native, identity, recipe }, control)
            })();
            // Runner/journal/processor and guarded outputs finish first. No
            // native text, secret, journal or live runtime escapes this call.
            drop(engine); drop(population); drop(context); drop(secret);
            drop(vocabulary); drop(planner); drop(input); drop(lease);
            Ok(result)
        })?;
        completed.map_err(HostedJobError::Job)
    }
}

#[allow(clippy::too_many_arguments)]
fn capacity(context: usize, kv_limit: u64, output: u64, edits: u64,
    actual_context: usize, actual_kv: u64, result_limit: usize) -> Result<(), HostedError> {
    if context == 0 || context > actual_context || actual_kv == 0 || actual_kv > kv_limit
        || edits == 0 || output == 0 || output > result_limit as u64 {
        return Err(HostedError::Limits("redaction job complete context, output and edit admission"));
    }
    Ok(())
}

// Private closed set of native adapters, not an external model/permit factory.
trait Native: BatchProcessor<Args = RedactionBatchArgs> { fn poisoned(&self) -> bool; }
impl<A: Int8RedactionBatchAdmission> Native for NativeInt8RedactionBatch<'_, '_, '_, '_, '_, '_, A> {
    fn poisoned(&self) -> bool { self.is_poisoned() }
}
struct RedactionJob<P: Native> { native: P, identity: ExecutionIdentity, recipe: RedactionJobRecipe }
impl<P: Native> BatchProcessor for RedactionJob<P> {
    type Args = RedactionBatchArgs;
    type Prepared = P::Prepared;
    type Output = P::Output;
    fn prepare(&mut self, document: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        self.native.prepare(document)
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>, control: &mut C)
        -> Result<Self::Prepared, BatchItemFailure> { self.native.prepare_with_control(document, control) }
    fn planned_work(&self, plan: &Self::Prepared) -> BatchWork { self.native.planned_work(plan) }
    fn execute<C: DecodeStepControl>(&mut self, plan: Self::Prepared, control: &mut C)
        -> Result<Self::Output, BatchItemFailure> { self.native.execute(plan, control) }
    fn execute_with_context<C: DecodeStepControl>(&mut self, plan: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        // Preserve durable ordinal/epoch: native adapters reject uncontrolled
        // execution and reused request sequence numbers. No per-item reset.
        self.native.execute_with_context(plan, context, control)
    }
}
impl<P: Native> DurableBatchProcessor for RedactionJob<P> {
    type Recipe = RedactionJobRecipe;
    fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    fn job_recipe(&self) -> &Self::Recipe { &self.recipe }
    fn durable_work(&self, plan: &Self::Prepared) -> Result<JobWork, BatchItemFailure> {
        self.recipe.check_work(self.native.planned_work(plan), self.native.poisoned())
    }
    fn max_result_bytes(&self, _: &Self::Prepared) -> u64 { self.recipe.max_result_bytes }
}

#[cfg(test)] mod tests;
