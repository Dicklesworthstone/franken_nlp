//! One resident native model and process-owned invocation for retained text.
use super::*;
use crate::{
    jobs::runner::generation::{Int8GenerationJobPlanner, Int8GenerationJobProcessor},
    tasks::ir::TaskBudget,
};

impl NlpEngine {
    /// Start or explicitly resume generation/chat over the COMPLETE original
    /// NDJSON population. The sealed planner owns the exact sampling policy;
    /// no caller-supplied admission provider or replacement native driver exists.
    /// Only completed validated outputs reach the authenticated result spool.
    /// Retention is explicit through SourceJobRequest, not a new default.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_text<R>(&self, model: &ResidentInt8, planner: Int8GenerationJobPlanner,
        request: SourceJobRequest, limits: JobHostLimits, reader: R, cancellation: CancellationToken)
        -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        dispatch::preflight(self, limits.native.run)?;
        self.check_resident_domain(model)?;
        check_model_identity(model.artifact_identity(), planner.execution_identity())?;
        let native = planner.native_limits();
        validate_work(native.max_model_work)?;
        let bytes = limits.required_buffer_bytes(request.limits)?;
        let required = requirements(limits.native)?;
        capacity(planner.task_budget(), required.kv_bytes, request.limits.max_result_bytes)?;
        let scratch_bytes = workspace(sum(&[required.rope_bytes, required.scratch_payload_bound,
            limits.native.allocator_reserve_bytes])?, native.max_sampler_bytes)?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?,
            || Ok(StreamInput { planner: Some((planner, request)), reader, writer: () }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, scratch_bytes)?;
        let model = model.clone();
        let completed = dispatch::run(self, limits.native.run, cancellation, move |control| {
            // Capture the whole storage-before-charge owner, including queued
            // cancellation. No reader, seed or planner escapes physical drain.
            let mut input = input;
            let (planner, request) = input.value.planner.take().ok_or(HostedError::CompletionMissing)?;
            let population = match JobPopulation::read_ndjson(&mut input.value.reader, &request.key,
                request.job_id, request.limits, limits.transport, control) {
                Ok(population) => population,
                Err(error) => return Ok(Err(JobRunError::Storage(error))),
            };
            let mut engine = allocate_native(kv, scratch, || model.inner.loaded.value
                .engine(limits.native.context_tokens, memory_budget(required)).map_err(HostedError::Model))?;
            let admission = CorpusAdmission { lease: &lease, model: model.artifact_identity(),
                kv_bytes: required.kv_bytes, sampler_bytes: native.max_sampler_bytes,
                output_bytes: request.limits.max_result_bytes as u64 };
            let result = (|| {
                let processor = Int8GenerationJobProcessor::new(&planner, &mut engine.value, admission)
                    .map_err(JobRunError::Processor)?;
                // The common runner authenticates the exact recipe/population
                // before repair or forwards. Failed-attempt debits survive it.
                run_population(request, &population, processor, control)
            })();
            // Every result guard already survived spool sync/acknowledgement.
            // Runner/processor borrows end before resident scratch or inputs.
            drop(engine); drop(population); drop(planner); drop(input); drop(lease);
            Ok(result)
        })?;
        completed.map_err(HostedJobError::Job)
    }
}
fn capacity(task: TaskBudget, kv: u64, output: usize) -> Result<(), HostedError> {
    task.validate().map_err(|_| HostedError::Limits("text job task ceiling"))?;
    if kv == 0 || kv > task.max_kv_bytes || task.max_output_bytes > output as u64 {
        return Err(HostedError::Limits("text job complete KV or immutable output envelope"));
    }
    Ok(())
}
fn workspace(ordinary: u64, sampler: u64) -> Result<u64, HostedError> {
    if sampler == 0 { return Err(HostedError::Limits("text job sampler reservation")); }
    sum(&[ordinary, sampler])
}

#[cfg(test)]
mod tests {
    use super::*;
    fn task() -> TaskBudget {
        TaskBudget { max_input_tokens: 2048, max_output_tokens: 16, max_output_bytes: 65536,
            max_grammar_states: 4096, max_kv_bytes: 1 << 30 }
    }
    #[test]
    fn complete_kv_and_serialized_output_fit_before_population_ingestion() {
        capacity(task(), 1 << 30, 65536).unwrap();
        assert!(capacity(task(), (1 << 30) + 1, 65536).is_err());
        assert!(capacity(task(), 0, 65536).is_err());
        assert!(capacity(task(), 1 << 30, 65535).is_err());
    }
    #[test]
    fn sampler_workspace_is_additional_not_a_discount_on_native_scratch() {
        assert_eq!(workspace(12345, 32 << 20).unwrap(), 12345 + (32 << 20));
        assert!(workspace(12345, 0).is_err());
        assert!(workspace(u64::MAX, 1).is_err());
        assert_eq!(workspace(u64::MAX - 1, 1).unwrap(), u64::MAX);
    }
    #[test]
    fn invalid_task_budgets_are_not_repaired_to_match_storage() {
        let mut budget = task(); budget.max_output_tokens = 0;
        assert!(capacity(budget, 1 << 30, 65536).is_err());
        let mut budget = task(); budget.max_output_bytes = 0;
        assert!(capacity(budget, 1 << 30, 65536).is_err());
    }
    #[test]
    fn owned_planner_input_and_result_cross_the_existing_blocking_boundary() {
        fn send<T: Send + 'static>() {}
        send::<StreamInput<(Int8GenerationJobPlanner, SourceJobRequest), std::io::Cursor<Vec<u8>>, ()>>();
        send::<crate::tasks::chat::quantized::Int8ChatResult>();
        let _entry = NlpEngine::job_int8_text::<std::io::Cursor<Vec<u8>>>;
    }
}
