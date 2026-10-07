//! Retained complete judgments using the SAME processor as the live corpus.
//! No public injected scorer, renewed task budget, or generated-label fallback.
use super::*;
use super::super::jobs::{run_population, HostedJobError, JobHostLimits, SourceJobRequest};
use crate::{
    jobs::{JobProgress, JobWork, population::JobPopulation,
        runner::{DurableBatchProcessor, JobRunError}},
    native_engine::lmhead::scoring::ScoringLimits,
    tasks::judge::{JUDGE_PROMPT_VERSION, quantized::INT8_JUDGE_EXECUTION},
};

impl NlpEngine {
    /// Start or explicitly resume item-local pairwise, rubric or faithfulness
    /// work. Original input and the private recipe authenticate before repair
    /// or inference. Each committed item is one COMPLETE judgment bundle.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_judge<R>(&self, model: &ResidentInt8, planner: Arc<JudgePlanner>,
        config: JudgeCorpusConfig, request: SourceJobRequest, limits: JobHostLimits,
        reader: R, cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.job_int8_judge_scheduled(model, planner, config, request, limits, None, reader, cancellation)
    }
    /// Explicit physical prompt scheduling, frozen in the private job recipe.
    /// The extra scratch charge is retained across ALL pending item attempts.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_judge_layer_major<R>(&self, model: &ResidentInt8, planner: Arc<JudgePlanner>,
        config: JudgeCorpusConfig, request: SourceJobRequest, limits: JobHostLimits,
        prefill: Int8PrefillLimits, reader: R, cancellation: CancellationToken)
        -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.job_int8_judge_scheduled(model, planner, config, request, limits, Some(prefill), reader, cancellation)
    }
    #[allow(clippy::too_many_arguments)]
    fn job_int8_judge_scheduled<R>(&self, model: &ResidentInt8, planner: Arc<JudgePlanner>,
        config: JudgeCorpusConfig, request: SourceJobRequest, limits: JobHostLimits,
        prefill: Option<Int8PrefillLimits>, reader: R, cancellation: CancellationToken)
        -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        dispatch::preflight(self, limits.native.run)?;
        self.check_resident_domain(model)?;
        check_model_identity(model.artifact_identity(), &config.identity)?;
        config.validate(&planner)?;
        let bytes = limits.required_buffer_bytes(request.limits)?;
        let required = requirements(limits.native)?;
        check_capacity(config.task_ceiling, required.kv_bytes, request.limits.max_result_bytes)?;
        let scratch_bytes = crate::hosted::scored::scoring_scratch(sum(&[required.rope_bytes,
            required.scratch_payload_bound, limits.native.allocator_reserve_bytes])?, prefill)?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?,
            || Ok(StreamInput { planner: Some((planner, config, request)), reader, writer: () }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, scratch_bytes)?;
        let model = model.clone();
        let completed = dispatch::run(self, limits.native.run, cancellation, move |control| {
            // Capture the entire storage-before-charge aggregate, including
            // queued cancellation. Recipe and population remain inside it.
            let mut input = input;
            let (planner, config, request) = input.value.planner.take().ok_or(HostedError::CompletionMissing)?;
            let recipe = JudgeJobRecipe::new(&config, prefill)?;
            let population = match JobPopulation::read_ndjson(&mut input.value.reader, &request.key,
                request.job_id, request.limits, limits.transport, control) {
                Ok(population) => population,
                Err(error) => return Ok(Err(JobRunError::Storage(error))),
            };
            let mut engine = allocate_native(kv, scratch, || model.inner.loaded.value
                .engine(limits.native.context_tokens, memory_budget(required)).map_err(HostedError::Model))?;
            let admission = CorpusAdmission { lease: &lease, model: model.artifact_identity(),
                kv_bytes: required.kv_bytes, sampler_bytes: 0, output_bytes: request.limits.max_result_bytes as u64 };
            let result = {
                let ledger = WorkLedger::new(config.max_model_work);
                let native = JudgeProcessor { planner: &planner, config, engine: &mut engine.value,
                    admission, ledger, prefill };
                run_population(request, &population, JudgeJobProcessor { native, recipe }, control)
            };
            // The common runner drops journal/lock/processor first. Every
            // output guard survived spool synchronization and acknowledgement.
            drop(engine); drop(population); drop(planner); drop(input); drop(lease);
            Ok(result)
        })?;
        completed.map_err(HostedJobError::Job)
    }
}

fn check_capacity(task: TaskBudget, kv: u64, result_bytes: usize) -> Result<(), HostedError> {
    task.validate().map_err(|_| HostedError::Limits("judge job task ceiling"))?;
    if kv == 0 || kv > task.max_kv_bytes || task.max_output_bytes > result_bytes as u64 {
        return Err(HostedError::Limits("judge job full KV and immutable result envelope"));
    }
    Ok(())
}

// Exhaustive destructuring makes future scoring/planning limit additions a
// compile-time obligation to freeze them, not an unnoticed resume loophole.
#[derive(Serialize)]
struct ScorerRecipe {
    max_candidates: usize, max_total_tokens: usize, max_nodes: usize,
    max_depth: usize, max_candidate_id_bytes: usize, max_projected_logits: u64,
}
impl From<ScoringLimits> for ScorerRecipe {
    fn from(limits: ScoringLimits) -> Self {
        let ScoringLimits { max_candidates, max_total_tokens, max_nodes, max_depth,
            max_candidate_id_bytes, max_projected_logits } = limits;
        Self { max_candidates, max_total_tokens, max_nodes, max_depth,
            max_candidate_id_bytes, max_projected_logits }
    }
}
#[derive(Serialize)]
struct PlanningRecipe {
    per_head: ScorerRecipe, max_total_prompt_tokens: usize, max_total_candidate_tokens: usize,
    max_total_projected_logits: u64, max_output_bytes: u64,
}
impl From<JudgeLimits> for PlanningRecipe {
    fn from(limits: JudgeLimits) -> Self {
        let JudgeLimits { per_head, max_total_prompt_tokens, max_total_candidate_tokens,
            max_total_projected_logits, max_output_bytes } = limits;
        Self { per_head: per_head.into(), max_total_prompt_tokens, max_total_candidate_tokens,
            max_total_projected_logits, max_output_bytes }
    }
}
#[derive(Serialize)]
struct PrefillRecipe { rows: usize, extra_scratch_bytes: u64 }
#[derive(Serialize)]
struct JudgeJobRecipe {
    version: u32, dependency_scope: &'static str, execution: &'static str,
    prompt_version: &'static str, task_ceiling: TaskBudget, planning: PlanningRecipe,
    max_model_work: Int8Work, defaults: Option<JudgeBatchArgs>, prefill: Option<PrefillRecipe>,
}
impl JudgeJobRecipe {
    fn new(config: &JudgeCorpusConfig, prefill: Option<Int8PrefillLimits>) -> Result<Self, HostedError> {
        let prefill = prefill.map(|limits| limits.validate().map(|extra_scratch_bytes|
            PrefillRecipe { rows: limits.max_batch_rows, extra_scratch_bytes }))
            .transpose().map_err(HostedError::Native)?;
        // Called only after fixed configuration validation and admission.
        // Defaults stay private: JobRunner persists their keyed commitment,
        // not this object, its plaintext or a public source hash.
        let recipe = Self { version: 1, dependency_scope: "item-local", execution: INT8_JUDGE_EXECUTION,
            prompt_version: JUDGE_PROMPT_VERSION, task_ceiling: config.task_ceiling,
            planning: config.planning.into(), max_model_work: config.max_model_work,
            defaults: config.defaults.clone(), prefill };
        planning::check_serialized_size(&recipe, 1024 * 1024 - 1024)?;
        Ok(recipe)
    }
}

struct JudgeJobProcessor<'p, 'e, 'w, 'a> {
    native: JudgeProcessor<'p, 'e, 'w, 'a>, recipe: JudgeJobRecipe,
}
impl BatchProcessor for JudgeJobProcessor<'_, '_, '_, '_> {
    type Args = JudgeBatchArgs;
    type Prepared = PreparedInt8Judge;
    type Output = GuardedOutput<Int8JudgeRun, Pending>;
    fn prepare(&mut self, document: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        self.native.prepare(document)
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>, control: &mut C)
        -> Result<Self::Prepared, BatchItemFailure> { self.native.prepare_with_control(document, control) }
    fn planned_work(&self, plan: &Self::Prepared) -> BatchWork { self.native.planned_work(plan) }
    fn execute<C: DecodeStepControl>(&mut self, plan: Self::Prepared, control: &mut C)
        -> Result<Self::Output, BatchItemFailure> { self.native.execute(plan, control) }
}
impl DurableBatchProcessor for JudgeJobProcessor<'_, '_, '_, '_> {
    type Recipe = JudgeJobRecipe;
    fn execution_identity(&self) -> &ExecutionIdentity { &self.native.config.identity }
    fn job_recipe(&self) -> &Self::Recipe { &self.recipe }
    fn durable_work(&self, plan: &Self::Prepared) -> Result<JobWork, BatchItemFailure> {
        self.native.ledger.ready()?;
        if self.native.engine.is_poisoned() || !self.native.engine.kv_cache().all_slots_have_len(0) {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        // Every criterion, order and evidence window is priced. The common
        // journal debits these five axes before execution, including failures.
        Ok(JobWork { model: plan.planned_work(), mask_node_visits: 0 })
    }
    fn max_result_bytes(&self, plan: &Self::Prepared) -> u64 { plan.max_result_bytes() }
}

#[cfg(test)] mod tests;
