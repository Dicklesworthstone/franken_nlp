//! A complete ordered source corpus on one charged grouped-INT4 model.

use super::*;
use crate::{
    batch::source::{SourceBatchArgs, grouped_int4::{self as source_batch,
        Int4SourceAdmission, Int4SourceBatchAdmission, Int4SourceBatchLimits,
        Int4SourceBatchPlanner, NativeInt4SourceBatch}},
    tasks::{ir::TaskBudget, source_planning::{SourcePlanningLimits, SourceTaskPlanner}},
};

/// One closed task/model contract for the whole stream. Per-line task arguments
/// may tighten its ceilings, but cannot switch task, model, backend or groups.
/// Defaults may contain private passages and are intentionally not Debug.
pub struct Int4SourceCorpusConfig {
    pub identity: ExecutionIdentity,
    pub task_ceiling: TaskBudget,
    pub planning: SourcePlanningLimits,
    pub defaults: Option<SourceBatchArgs>,
    pub native_work: Int4SourceBatchLimits,
}
impl Int4SourceCorpusConfig {
    fn validate(&self, planner: &SourceTaskPlanner, model: &Int4ModelIdentity,
        allocated_kv: u64) -> Result<(), HostedError> {
        constrained_int4::check_model(&self.identity, model).map_err(|_| HostedError::ModelIdentity)?;
        self.validate_planning(planner, model.group_width(), allocated_kv)
    }

    fn validate_planning(&self, planner: &SourceTaskPlanner, group_width: usize,
        allocated_kv: u64) -> Result<(), HostedError> {
        source_batch::check_configuration(planner, &self.identity, self.task_ceiling,
            self.planning, self.defaults.as_ref(), group_width).map_err(HostedError::BatchSetup)?;
        source_batch::validate_limits(self.native_work).map_err(HostedError::BatchSetup)?;
        if allocated_kv == 0 || allocated_kv > self.task_ceiling.max_kv_bytes {
            return Err(HostedError::Limits("INT4 source corpus complete KV reservation"));
        }
        Ok(())
    }
}

struct SourceInput<R, W> {
    preparation: Option<(Arc<SourceTaskPlanner>, Arc<ExtractionVocabulary>, Int4SourceCorpusConfig)>,
    reader: R,
    writer: W,
}

impl NlpEngine {
    /// Run bounded NER, keyphrases, cited summaries or passage QA as ordered
    /// NDJSON. For QA each line's `text` is the question and its typed passages
    /// supply the evidence. One model, engine, vocabulary and work reservation
    /// serve the entire stream, including all transport epochs.
    ///
    /// Preparation/IO reservations price the owned captures. Cancellation
    /// retains their charge through physical cleanup and writer disposal.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int4_source<R, W>(&self, model: &ResidentInt4,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int4SourceCorpusConfig, limits: CorpusLimits, kernel: PackedInt4Kernel,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        dispatch::preflight(self, limits.native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        let width = model.model_identity().group_width();
        let required = requirements(width, config.native_work, limits)?;
        config.validate(&planner, model.model_identity(), required.native.kv_bytes)?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes)?,
            || Ok(SourceInput { preparation: Some((planner, vocabulary, config)), reader, writer }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let workspace = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let model = model.clone();
        dispatch::run(self, limits.native.run, cancellation, move |control| {
            let mut input = input;
            checkpoint(control)?;
            let (planner, vocabulary, config) = input.value.preparation.take().ok_or(HostedError::CompletionMissing)?;
            let task_spec = config.identity.task_spec.clone();
            let compiler = Int4SourceBatchPlanner::new(&planner, config.identity, config.task_ceiling,
                config.planning, config.defaults, width).map_err(HostedError::BatchSetup)?;
            let mut engine = allocate_native(kv, workspace, || {
                StrictInt4Engine::new(&model.inner.loaded.value, limits.native.context_tokens, Int4MemoryBudget {
                    max_kv_bytes: required.native.kv_bytes, max_rope_bytes: required.native.rope_bytes,
                    max_scratch_payload_bytes: required.native.scratch_payload_bound,
                }).map(|engine| engine.with_kernel(kernel)).map_err(native_error)
            })?;
            let admission = SourceAdmission { inner: CorpusAdmission {
                lease: &lease, model: model.model_identity(), kv_bytes: required.native.kv_bytes,
                masks: config.native_work.masks, output_bytes: limits.transport.max_output_line_bytes as u64,
            }, task_spec: &task_spec };
            let result = {
                let mut processor = NativeInt4SourceBatch::new(compiler, &mut engine.value,
                    &vocabulary, admission, config.native_work).map_err(HostedError::BatchSetup)?;
                stream::run_ndjson(&mut input.value.reader, &mut input.value.writer,
                    &mut processor, limits.transport, control).map_err(HostedError::Batch)
            };
            drop(engine);
            drop(vocabulary);
            drop(planner);
            drop(input);
            drop(lease);
            result
        })
    }
}

struct SourceAdmission<'a> { inner: CorpusAdmission<'a>, task_spec: &'a str }
impl Int4SourceBatchAdmission for SourceAdmission<'_> {
    type Guard = Pending;
    fn admit(&mut self, request: Int4SourceAdmission<'_>)
        -> Result<(ExecutionIdentity, Self::Guard), BatchItemFailure> {
        self.inner.admit_for_task(request, self.task_spec)
    }
}

#[cfg(test)] mod tests;
