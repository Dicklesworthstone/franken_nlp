//! One process-owned invocation for a resident independent-axis sentiment stream.
use super::*;
use crate::tasks::sentiment::{SentimentPlanner, batch::{Int8SentimentBatchPlanner,
    NativeInt8SentimentBatch, Int8SentimentAdmission, Int8SentimentAdmissionRequest}};
pub use crate::tasks::sentiment::batch::SentimentBatchConfig as SentimentCorpusConfig;

impl NlpEngine {
    /// Run complete sentiment bundles through one actual blocking-pool/scoped
    /// invocation. No per-document model/runtime, caller-supplied no-op permit,
    /// collecting output buffer, retry or partial-axis response is introduced.
    /// Score-space and policy are fixed by the immutable pinned planner.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_sentiment<R, W>(&self, model: &ResidentInt8,
        planner: Arc<SentimentPlanner>, config: SentimentCorpusConfig,
        limits: CorpusLimits, reader: R, writer: W, cancellation: CancellationToken)
        -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        dispatch::preflight(self, limits.native.run)?;
        self.check_resident_domain(model)?;
        check_model_identity(model.artifact_identity(), &config.identity)?;
        config.validate(&planner).map_err(HostedError::BatchSetup)?;
        let required = requirements(limits.native)?;
        validate_capacity(config.task_ceiling, required.kv_bytes, limits.transport.max_output_line_bytes)?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, limits.reservation_bytes()?)?,
            || Ok(StreamInput { planner: Some((planner, config)), reader, writer }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch,
            sum(&[required.rope_bytes, required.scratch_payload_bound, limits.native.allocator_reserve_bytes])?)?;
        let model = model.clone();
        dispatch::run(self, limits.native.run, cancellation, move |control| {
            // Consume/capture the whole input aggregate so queued disposal also
            // drops stored planner/IO before their charge and completion signal.
            let mut input = input;
            let (planner, config) = input.value.planner.take().ok_or(HostedError::CompletionMissing)?;
            let compiler = Int8SentimentBatchPlanner::new(&planner, config).map_err(HostedError::BatchSetup)?;
            let mut engine = allocate_native(kv, scratch, || model.inner.loaded.value
                .engine(limits.native.context_tokens, memory_budget(required)).map_err(HostedError::Model))?;
            let admission = CorpusAdmission { lease: &lease, model: model.artifact_identity(),
                kv_bytes: required.kv_bytes, sampler_bytes: 0,
                output_bytes: limits.transport.max_output_line_bytes as u64 };
            let result = {
                let mut processor = NativeInt8SentimentBatch::new(compiler, &mut engine.value, admission)
                    .map_err(HostedError::BatchSetup)?;
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

impl Int8SentimentAdmission for CorpusAdmission<'_> {
    type Guard = Pending;
    fn admit(&mut self, request: Int8SentimentAdmissionRequest<'_>)
        -> Result<(ExecutionIdentity, Self::Guard), BatchItemFailure> {
        if request.identity.task_spec != "sentiment-v1" || request.kv_reservation_bytes != self.kv_bytes {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        check_model_identity(self.model, request.identity).map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        crate::native_engine::constrained_int8::check_profile(request.identity)
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        validate_work(request.model_work).map_err(|_| BatchItemFailure::fatal(BatchCode::InvalidExecution))?;
        if request.max_result_bytes == 0 || request.max_result_bytes > self.output_bytes {
            return Err(BatchItemFailure::reject(BatchCode::OutputLineLimit));
        }
        // No generated-token buffer exists here. The entire bounded result
        // already includes candidate strings/scores/distributions. Do not infer
        // a token count from projected rows: the library supports three named
        // score spaces with different row counts. The CLI selects full vocab.
        let guard = output_claim(self.lease, request.max_result_bytes, 0).map_err(admission_failure)?;
        Ok((request.identity.clone(), guard))
    }
}
fn validate_capacity(task: crate::tasks::ir::TaskBudget, kv_bytes: u64, line_bytes: usize)
    -> Result<(), HostedError> {
    task.validate().map_err(|_| HostedError::Limits("sentiment corpus task ceiling"))?;
    if kv_bytes == 0 || kv_bytes > task.max_kv_bytes || task.max_output_bytes > line_bytes as u64 {
        return Err(HostedError::Limits("sentiment corpus complete KV/output envelope"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::ir::TaskBudget;
    fn budget() -> TaskBudget {
        TaskBudget { max_input_tokens: 4096, max_output_tokens: 16,
            max_output_bytes: 8192, max_grammar_states: 4096, max_kv_bytes: 65536 }
    }
    #[test]
    fn full_native_kv_and_complete_result_must_fit_even_for_short_documents() {
        validate_capacity(budget(), 65536, 8192).unwrap();
        assert!(validate_capacity(budget(), 65537, 8192).is_err());
        assert!(validate_capacity(budget(), 65536, 8191).is_err());
        assert!(validate_capacity(budget(), 0, 8192).is_err());
    }
    #[test]
    fn zero_task_axes_cannot_be_used_to_skip_admission() {
        let mut b = budget(); b.max_output_bytes = 0;
        assert!(validate_capacity(b, 65536, 8192).is_err());
        let mut b = budget(); b.max_output_tokens = 0;
        assert!(validate_capacity(b, 65536, 8192).is_err());
    }
    #[test]
    fn corpus_configuration_and_captures_cross_the_owned_runtime_boundary() {
        fn send<T: Send + 'static>() {}
        send::<SentimentCorpusConfig>();
        send::<StreamInput<(Arc<SentimentPlanner>, SentimentCorpusConfig),
            std::io::Cursor<Vec<u8>>, Vec<u8>>>();
    }
}
