//! Durable complete-document redaction with the real resident INT4 processor.
//! JobRunner authenticates the fixed policy/population and persists each full
//! attempt debit before execution. No native transcript or partial edit commits.
use crate::{
    batch::{BatchCode, BatchDocument, BatchFault, BatchItemFailure, BatchProcessor,
        BatchRequestContext, BatchWork, generation::GuardedOutput},
    execution_identity::ExecutionIdentity,
    jobs::{JobLimits, JobWork, runner::DurableBatchProcessor},
    native_engine::{decode::DecodeStepControl,
        strict_int4::{Int4ModelIdentity, StrictInt4Engine}},
    tasks::{extract::ExtractionVocabulary,
        redact::{pseudonym::Pseudonyms, grouped_int4::{
            corpus::{self as document_batch, Int4DocumentRedactionBatchConfig,
                Int4RedactionBatchAdmission, NativeInt4DocumentRedactionBatch,
                PreparedInt4DocumentRedaction, RedactionBatchArgs},
            long::Int4LongRedactionRun}},
        source_planning::SourceTaskPlanner},
};
use serde::Serialize;

mod recipe;
pub use recipe::Int4RedactionJobRecipe;

/// Leave room for JobRunner's owned protocol and input-profile envelope.
pub const INT4_REDACTION_JOB_RECIPE_BYTES: usize = 1024 * 1024 - 1024;

/// The original decoded source ceiling is separate from encoded JSON framing
/// and from the larger source envelope needed by expanded verification text.
pub struct Int4RedactionJobConfig {
    pub batch: Int4DocumentRedactionBatchConfig,
    pub max_source_bytes: usize,
}

/// Pure configuration/limit admission; performs no storage or model work.
/// JobLimits owns the lifetime ceiling, including failures before any resume.
pub fn check_job_limits(config: &Int4RedactionJobConfig, limits: JobLimits) -> Result<(), BatchFault> {
    check_static_limits(config)?;
    recipe::validate_job_limits(limits,
        JobWork::from_int4(config.batch.max_model_work, config.batch.max_mask_visits),
        JobWork::from_int4(config.batch.item_model_work(), config.batch.detector.mapping.max_mask_visits),
        config.batch.detector.max_result_bytes)
}

/// Complete source/model admission before snapshot saving or population IO.
pub fn check_configuration(planner: &SourceTaskPlanner, config: &Int4RedactionJobConfig,
    model: &Int4ModelIdentity, limits: JobLimits) -> Result<(), BatchFault> {
    check_job_limits(config, limits)?;
    document_batch::check_configuration(planner, &config.batch, model)
}
fn check_static_limits(config: &Int4RedactionJobConfig) -> Result<(), BatchFault> {
    let d = &config.batch.detector;
    let work = config.batch.item_model_work();
    if !(1..=64 * 1024 * 1024).contains(&config.max_source_bytes)
        || config.max_source_bytes > d.mapping.chunks.max_input_bytes
        || config.max_source_bytes > config.batch.request.rule_budget.max_input_bytes
        || work.forward_positions == 0 || work.projected_logits == 0 || work.attention_pairs == 0
        || work.projections.dot_products == 0 || work.projections.multiply_accumulates == 0
        || d.mapping.max_mask_visits == 0
        || !JobWork::from_int4(work, d.mapping.max_mask_visits)
            .fits(JobWork::from_int4(config.batch.max_model_work, config.batch.max_mask_visits))
        || !(1..=64 * 1024 * 1024).contains(&d.max_result_bytes) {
        return Err(BatchCode::InvalidLimits.into());
    }
    // Count before either the native adapter or recipe clones private settings.
    bounded(&config.batch.request)?;
    bounded(&d.ner)?;
    Ok(())
}
fn bounded<T: Serialize + ?Sized>(value: &T) -> Result<(), BatchFault> {
    struct Counter(usize);
    impl std::io::Write for Counter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0 = self.0.checked_sub(bytes.len()).ok_or_else(|| std::io::Error::other("private recipe bound"))?;
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
    }
    serde_json::to_writer(Counter(INT4_REDACTION_JOB_RECIPE_BYTES), value)
        .map_err(|_| BatchCode::InvalidLimits.into())
}

/// Sealed original-size metadata also prevents a prepared value from a wider
/// job instance bypassing this instance's original-source ceiling.
pub struct PreparedInt4RedactionJob {
    native: PreparedInt4DocumentRedaction,
    source_bytes: usize,
}

/// One immutable policy/key scope and one real native corpus processor.
/// Outputs keep their admission guard through spool sync and journal commit.
pub struct Int4RedactionJobProcessor<'p, 'e, 'w, 'v, 's, 'k, A: Int4RedactionBatchAdmission> {
    native: NativeInt4DocumentRedactionBatch<'p, 'e, 'w, 'v, 's, 'k, A>,
    identity: ExecutionIdentity,
    recipe: Int4RedactionJobRecipe,
}
impl<'p, 'e, 'w, 'v, 's, 'k, A: Int4RedactionBatchAdmission>
    Int4RedactionJobProcessor<'p, 'e, 'w, 'v, 's, 'k, A> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(planner: &'p SourceTaskPlanner, config: Int4RedactionJobConfig,
        engine: &'e mut StrictInt4Engine<'w>, vocabulary: &'v ExtractionVocabulary,
        pseudonyms: Option<&'s Pseudonyms<'k>>, admission: A) -> Result<Self, BatchFault> {
        check_static_limits(&config)?;
        document_batch::check_configuration(planner, &config.batch, engine.model_identity())?;
        let recipe = Int4RedactionJobRecipe::new(&config, engine.model_identity().group_width(), pseudonyms)?;
        let identity = config.batch.ner_identity.clone();
        let native = NativeInt4DocumentRedactionBatch::new(planner, config.batch,
            engine, vocabulary, pseudonyms, admission)?;
        Ok(Self { native, identity, recipe })
    }
    fn check_source(&self, bytes: usize) -> Result<(), BatchItemFailure> {
        check_source(bytes, self.recipe.max_source_bytes(), self.native.is_poisoned())
    }
}
impl<A: Int4RedactionBatchAdmission> BatchProcessor
    for Int4RedactionJobProcessor<'_, '_, '_, '_, '_, '_, A> {
    type Args = RedactionBatchArgs;
    type Prepared = PreparedInt4RedactionJob;
    type Output = GuardedOutput<Int4LongRedactionRun, A::Guard>;
    fn prepare(&mut self, document: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        let source_bytes = document.text.len();
        self.check_source(source_bytes)?;
        self.native.prepare(document).map(|native| PreparedInt4RedactionJob { native, source_bytes })
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>, control: &mut C)
        -> Result<Self::Prepared, BatchItemFailure> {
        checkpoint(control)?;
        let source_bytes = document.text.len();
        self.check_source(source_bytes)?;
        self.native.prepare_with_control(document, control).map(|native| PreparedInt4RedactionJob { native, source_bytes })
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork { self.native.planned_work(&prepared.native) }
    fn execute<C: DecodeStepControl>(&mut self, prepared: Self::Prepared, control: &mut C)
        -> Result<Self::Output, BatchItemFailure> {
        self.check_source(prepared.source_bytes)?;
        self.native.execute(prepared.native, control)
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        checkpoint(control)?;
        self.check_source(prepared.source_bytes)?;
        self.native.execute_with_context(prepared.native, context, control)
    }
}
impl<A: Int4RedactionBatchAdmission> DurableBatchProcessor
    for Int4RedactionJobProcessor<'_, '_, '_, '_, '_, '_, A> {
    type Recipe = Int4RedactionJobRecipe;
    fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    fn job_recipe(&self) -> &Self::Recipe { &self.recipe }
    fn validate_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> { self.recipe.check_job_limits(limits) }
    fn durable_work(&self, prepared: &Self::Prepared) -> Result<JobWork, BatchItemFailure> {
        self.check_source(prepared.source_bytes)?;
        // The unknown transformed source requires the complete item envelope.
        // A new native ledger after resume cannot refund the persisted debits.
        self.recipe.check_work(self.native.planned_work(&prepared.native), self.native.is_poisoned())
    }
    fn max_result_bytes(&self, _: &Self::Prepared) -> u64 { self.recipe.max_result_bytes() }
}
fn check_source(bytes: usize, max: usize, poisoned: bool) -> Result<(), BatchItemFailure> {
    if poisoned { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
    if bytes == 0 || bytes > max { return Err(BatchItemFailure::reject(BatchCode::DocumentLimit)); }
    Ok(())
}
fn checkpoint(control: &mut impl DecodeStepControl) -> Result<(), BatchItemFailure> {
    match control.prefill_checkpoint(0) {
        Some(cause) => Err(BatchItemFailure::fatal(BatchFault::cancelled(cause))), None => Ok(()),
    }
}

