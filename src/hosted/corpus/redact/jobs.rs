//! Authenticated retained redaction. The existing native stream adapter owns
//! detection, editing, fresh verification, work accounting and output guards.
use super::*;
use crate::{
    batch::{BatchDocument, BatchProcessor, BatchRequestContext, BatchWork},
    jobs::{JobProgress, JobWork, runner::{DurableBatchProcessor, JobRunError}},
    native_engine::decode::DecodeStepControl,
    hosted::corpus::jobs::{read_population, run_population, HostedJobError, JobHostLimits, SourceJobRequest},
    tasks::redact::batch::{RedactionBatchArgs, Int8RedactionBatchAdmission},
};
mod recipe;
mod long;
mod team;
use recipe::{RedactionJobRecipe, RECIPE_BYTES};

impl NlpEngine {
    /// Create or explicitly resume one authenticated item-local redaction job.
    /// The owned runner retains the existing spool/journal commit protocol.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_redact<R>(&self, model: &ResidentInt8,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: RedactionCorpusConfig, pseudonyms: Option<RedactionPseudonyms>,
        request: SourceJobRequest, limits: JobHostLimits, reader: R,
        cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        team::run(self, model, planner, vocabulary, config, pseudonyms, request,
            limits, None, None, reader, cancellation)
    }

    /// Reuse one bounded prompt workspace across every pending original and
    /// edited-text pass. Physical scheduling does not change the saved recipe.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_redact_layer_major<R>(&self, model: &ResidentInt8,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: RedactionCorpusConfig, pseudonyms: Option<RedactionPseudonyms>,
        request: SourceJobRequest, limits: JobHostLimits, prefill: Int8PrefillLimits,
        reader: R, cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        team::run(self, model, planner, vocabulary, config, pseudonyms, request,
            limits, Some(prefill), None, reader, cancellation)
    }

    /// All pending attempts share one physical team. Workers join before any
    /// progress or failure returns; authenticated lifetime debits never renew.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_redact_with_threads<R>(&self, model: &ResidentInt8,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: RedactionCorpusConfig, pseudonyms: Option<RedactionPseudonyms>,
        request: SourceJobRequest, limits: JobHostLimits, kernel: Int8Kernel,
        compute_threads: usize, reader: R, cancellation: CancellationToken)
        -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        team::run(self, model, planner, vocabulary, config, pseudonyms, request,
            limits, None, Some((kernel, compute_threads)), reader, cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_redact_layer_major_with_threads<R>(&self, model: &ResidentInt8,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: RedactionCorpusConfig, pseudonyms: Option<RedactionPseudonyms>,
        request: SourceJobRequest, limits: JobHostLimits, kernel: Int8Kernel,
        compute_threads: usize, prefill: Int8PrefillLimits, reader: R, cancellation: CancellationToken)
        -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        team::run(self, model, planner, vocabulary, config, pseudonyms, request,
            limits, Some(prefill), Some((kernel, compute_threads)), reader, cancellation)
    }
}

impl team::Policy for RedactionCorpusConfig {
    fn identity(&self) -> &ExecutionIdentity { &self.batch.ner_identity }
    fn request(&self) -> &RedactionRequest { &self.batch.request }
    fn validate(&self, planner: &SourceTaskPlanner, request: &SourceJobRequest,
        limits: JobHostLimits, native: Int8MemoryRequirement) -> Result<u64, HostedError> {
        recipe::bounded(&self.batch.request)?; recipe::bounded(&self.batch.detector.ner)?;
        redaction_batch::check_configuration(planner, &self.batch).map_err(HostedError::BatchSetup)?;
        let d = &self.batch.detector;
        capacity(d.planning.max_context_tokens, d.per_pass.max_kv_bytes,
            d.max_result_bytes, self.edit_reserve_bytes, limits.native.context_tokens,
            native.kv_bytes, request.limits.max_result_bytes)?;
        temporary_bytes(self)
    }
    fn recipe(&self, _: &SourceTaskPlanner, context: Option<&Pseudonyms<'_>>)
        -> Result<RedactionJobRecipe, HostedError> {
        RedactionJobRecipe::short(&self.batch, context)
    }
    #[allow(clippy::too_many_arguments)]
    fn execute<C: DecodeStepControl>(self, planner: &SourceTaskPlanner,
        vocabulary: &ExtractionVocabulary, context: Option<&Pseudonyms<'_>>,
        engine: &mut StrictInt8Engine<'_>, admission: CorpusAdmission<'_>, prefill: Option<Int8PrefillLimits>,
        request: SourceJobRequest, population: &crate::hosted::corpus::jobs::Population<'_>,
        recipe: RedactionJobRecipe, control: &mut C) -> Result<JobProgress, JobRunError> {
        let identity = self.batch.ner_identity.clone();
        let native = match prefill {
            None => NativeInt8RedactionBatch::new(planner, self.batch, engine, vocabulary, context, admission),
            Some(prefill) => NativeInt8RedactionBatch::new_layer_major(
                planner, self.batch, engine, vocabulary, context, admission, prefill),
        }.map_err(JobRunError::Processor)?;
        run_population(request, population, RedactionJob { native, identity, recipe }, control)
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
trait Native: BatchProcessor<Args = RedactionBatchArgs> {
    fn poisoned(&self) -> bool;
    fn check_prepared(&self, _: &Self::Prepared) -> Result<(), BatchItemFailure> { Ok(()) }
}
impl<A: Int8RedactionBatchAdmission> Native for NativeInt8RedactionBatch<'_, '_, '_, '_, '_, '_, A> {
    fn poisoned(&self) -> bool { self.is_poisoned() }
}
struct RedactionJob<P: Native> { native: P, identity: ExecutionIdentity, recipe: RedactionJobRecipe }
impl<P: Native> BatchProcessor for RedactionJob<P> {
    type Args = RedactionBatchArgs;
    type Prepared = P::Prepared;
    type Output = P::Output;
    fn prepare(&mut self, document: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> { self.native.prepare(document) }
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
        self.native.check_prepared(plan)?;
        self.recipe.check_work(self.native.planned_work(plan), self.native.poisoned())
    }
    fn max_result_bytes(&self, _: &Self::Prepared) -> u64 { self.recipe.max_result_bytes }
}

#[cfg(test)] mod tests;
