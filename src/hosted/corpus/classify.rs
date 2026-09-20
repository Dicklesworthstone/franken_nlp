//! Resident ordered classification with concrete, process-ledger admission.
use super::*;
use crate::{
    batch::classify::{ClassificationBatchArgs, quantized::{Int8ClassificationAdmission,
        Int8ClassificationBatchPlanner, NativeInt8ClassificationBatch}},
    tasks::{classify::{ClassificationLimits, ClassificationPlanner}, ir::TaskBudget},
};

/// Immutable-by-transfer settings for a bounded classification stream. Request
/// budgets may narrow `task_ceiling`, never enlarge it. The whole output ceiling
/// is conservatively reserved for each item because the existing admission
/// interface exposes identity/work, not the item's individual byte budget.
/// No Debug: defaults may contain private labels/descriptions.
pub struct ClassificationCorpusConfig {
    pub identity: ExecutionIdentity,
    pub task_ceiling: TaskBudget,
    pub planning: ClassificationLimits,
    pub defaults: Option<ClassificationBatchArgs>,
    /// Five nonrefundable native counters across all documents and flushes.
    pub max_model_work: Int8Work,
}

impl NlpEngine {
    /// Run exclusive or multi-label NDJSON requests on one resident INT8 engine.
    /// Uses the existing classifier, full candidate scorer and result finalizer;
    /// no replacement tokenizer, sampled-label path or caller admission trait.
    /// Planning and blocking IO retain the underlying adapters' cooperative
    /// cancellation granularity. Successful return does not imply zero rejects.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_classify<R, W>(&self, model: &ResidentInt8,
        planner: Arc<ClassificationPlanner>, config: ClassificationCorpusConfig,
        limits: CorpusLimits, reader: R, writer: W, cancellation: CancellationToken)
        -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        dispatch::preflight(self, limits.native.run)?;
        self.check_resident_domain(model)?;
        check_model_identity(model.artifact_identity(), &config.identity)?;
        validate_work(config.max_model_work)?;
        let output_bytes = output_ceiling(config.task_ceiling, limits.transport.max_output_line_bytes)?;
        let required = requirements(limits.native)?;
        if required.kv_bytes > config.task_ceiling.max_kv_bytes {
            return Err(HostedError::Limits("classification corpus whole KV allocation"));
        }
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, limits.reservation_bytes()?)?,
            || Ok(StreamInput { planner: Some((planner, config)), reader, writer }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
        let workspace = Pending::reserve(&lease, MemoryClass::ActivationScratch,
            sum(&[required.rope_bytes, required.scratch_payload_bound, limits.native.allocator_reserve_bytes])?)?;
        let model = model.clone();
        dispatch::run(self, limits.native.run, cancellation, move |control| {
            // Retain the complete drop-ordered input package even when queued
            // work is discarded before it runs; do not capture its fields alone.
            let mut input = input;
            let (planner, config) = input.value.planner.take().ok_or(HostedError::CompletionMissing)?;
            let compiler = Int8ClassificationBatchPlanner::new(&planner, config.identity,
                config.task_ceiling, config.planning, config.defaults).map_err(HostedError::BatchSetup)?;
            let mut engine = allocate_native(kv, workspace, || model.inner.loaded.value
                .engine(limits.native.context_tokens, memory_budget(required)).map_err(HostedError::Model))?;
            let admission = ClassificationAdmission { output_bytes, inner: CorpusAdmission {
                lease: &lease, model: model.artifact_identity(), kv_bytes: required.kv_bytes,
                sampler_bytes: 0, output_bytes: limits.transport.max_output_line_bytes as u64 } };
            let result = {
                let mut processor = NativeInt8ClassificationBatch::new(compiler, &mut engine.value,
                    admission, config.max_model_work).map_err(HostedError::BatchSetup)?;
                batch::run_ndjson(&mut input.value.reader, &mut input.value.writer,
                    &mut processor, limits.transport, control).map_err(HostedError::Batch)
            };
            drop(engine);
            drop(planner);
            drop(input);
            drop(lease);
            result
        })
    }
}

struct ClassificationAdmission<'a> { inner: CorpusAdmission<'a>, output_bytes: u64 }
impl Int8ClassificationAdmission for ClassificationAdmission<'_> {
    type Guard = Pending;
    fn admit(&mut self, identity: &ExecutionIdentity, work: Int8Work)
        -> Result<(ExecutionIdentity, Self::Guard), BatchItemFailure> {
        if identity.task_spec != "classify-v1" { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
        self.inner.admit_output(identity, work, self.inner.kv_bytes, 0, self.output_bytes)
    }
}
fn output_ceiling(task: TaskBudget, line_bytes: usize) -> Result<u64, HostedError> {
    task.validate().map_err(|_| HostedError::Limits("classification corpus task ceiling"))?;
    if task.max_output_bytes == 0 || task.max_output_bytes > line_bytes as u64 {
        return Err(HostedError::Limits("classification result exceeds transport line envelope"));
    }
    Ok(task.max_output_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn budget() -> TaskBudget {
        TaskBudget { max_input_tokens: 4096, max_output_tokens: 64,
            max_output_bytes: 8192, max_grammar_states: 4096, max_kv_bytes: 65536 }
    }
    #[test]
    fn admission_prices_task_ceiling_not_caller_ids_or_work_estimates() {
        assert_eq!(output_ceiling(budget(), 16384).unwrap(), 8192);
        assert_eq!(output_ceiling(budget(), 8192).unwrap(), 8192);
    }
    #[test]
    fn oversized_result_ceiling_refuses_before_streaming() {
        assert!(output_ceiling(budget(), 8191).is_err());
        assert!(output_ceiling(budget(), 0).is_err());
        let mut task = budget(); task.max_output_bytes = u64::MAX;
        assert!(output_ceiling(task, 8192).is_err());
    }
    #[test]
    fn zero_result_budget_cannot_bypass_reservation() {
        let mut task = budget(); task.max_output_bytes = 0;
        assert!(output_ceiling(task, 8192).is_err());
    }
    #[test]
    fn corpus_configuration_is_owned_send_data() {
        fn send<T: Send + 'static>() {}
        send::<ClassificationCorpusConfig>();
        send::<StreamInput<(Arc<ClassificationPlanner>, ClassificationCorpusConfig),
            std::io::Cursor<Vec<u8>>, Vec<u8>>>();
    }
}
