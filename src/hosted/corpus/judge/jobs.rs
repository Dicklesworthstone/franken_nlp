//! Retained complete judgments using the SAME processor as the live corpus.
//! No public injected scorer, renewed task budget, or generated-label fallback.
use super::*;
use super::super::jobs::{run_population, HostedJobError, JobHostLimits, Population, SourceJobRequest,
    scored::{self as scored_job, Int8JobTask}};
use crate::{
    jobs::{JobLimits, JobProgress, JobWork,
        runner::{DurableBatchProcessor, JobRunError}},
    native_engine::{lmhead::scoring::ScoringLimits, simd::Int8Kernel},
    tasks::judge::{JUDGE_PROMPT_VERSION, quantized::INT8_JUDGE_EXECUTION},
};

impl NlpEngine {
    /// Authenticate the exact original population and private recipe before
    /// repair or inference. Each committed item is a COMPLETE judgment bundle.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_judge<R>(&self, model: &ResidentInt8, planner: Arc<JudgePlanner>,
        config: JudgeCorpusConfig, request: SourceJobRequest, limits: JobHostLimits,
        reader: R, cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.job_int8_judge_scheduled(model, planner, config, request, limits,
            None, None, None, reader, cancellation)
    }

    /// Preserve the existing frozen-prefill recipe contract while sharing
    /// the admitted prompt workspace across every pending judgment.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_judge_layer_major<R>(&self, model: &ResidentInt8, planner: Arc<JudgePlanner>,
        config: JudgeCorpusConfig, request: SourceJobRequest, limits: JobHostLimits,
        prefill: Int8PrefillLimits, reader: R, cancellation: CancellationToken)
        -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.job_int8_judge_scheduled(model, planner, config, request, limits,
            None, Some(prefill), None, reader, cancellation)
    }

    /// Reuse one fixed team for every comparison order, rubric criterion and
    /// evidence head. Width may change on resume without renewing any work.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_judge_with_threads<R>(&self, model: &ResidentInt8, planner: Arc<JudgePlanner>,
        config: JudgeCorpusConfig, request: SourceJobRequest, limits: JobHostLimits,
        kernel: Int8Kernel, compute_threads: usize, reader: R, cancellation: CancellationToken)
        -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.job_int8_judge_scheduled(model, planner, config, request, limits,
            None, None, Some((kernel, compute_threads)), reader, cancellation)
    }

    /// Compose the frozen prompt schedule with one admitted persistent team.
    /// Team width is physical; the pre-existing prompt recipe remains frozen.
    /// All workers join before progress or failure becomes observable.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_judge_layer_major_with_threads<R>(&self, model: &ResidentInt8,
        planner: Arc<JudgePlanner>, config: JudgeCorpusConfig, request: SourceJobRequest,
        limits: JobHostLimits, kernel: Int8Kernel, compute_threads: usize,
        prefill: Int8PrefillLimits, reader: R, cancellation: CancellationToken)
        -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.job_int8_judge_scheduled(model, planner, config, request, limits,
            None, Some(prefill), Some((kernel, compute_threads)), reader, cancellation)
    }

    /// Share the exact compact policy and one invocation tokenizer allowance
    /// throughout authenticated start/resume. Policy and all four total caps
    /// are frozen in the private recipe. Failed preparation remains charged;
    /// stored native work and attempts never reset when this invocation ends.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_judge_scheduled<R>(&self, model: &ResidentInt8, planner: Arc<JudgePlanner>,
        config: JudgeCorpusConfig, request: SourceJobRequest, limits: JobHostLimits,
        encoding_work_limits: Option<SafeBpeWorkLimits>, prefill: Option<Int8PrefillLimits>,
        team: Option<(Int8Kernel, usize)>, reader: R, cancellation: CancellationToken)
        -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        let encoding = EncodingWorkScope::new(planner.document_encoding(), encoding_work_limits)
            .map_err(HostedError::BatchSetup)?;
        scored_job::run(self, model, JudgeTask { planner, config, encoding }, request, limits,
            prefill, team, reader, cancellation)
    }
}

struct JudgeTask {
    planner: Arc<JudgePlanner>, config: JudgeCorpusConfig, encoding: Arc<EncodingWorkScope>,
}
impl Int8JobTask for JudgeTask {
    fn identity(&self) -> &ExecutionIdentity { &self.config.identity }
    fn retain_for_join(&self) -> Option<Box<dyn Send>> {
        Some(Box::new((Arc::clone(&self.planner), Arc::clone(&self.encoding))))
    }
    fn preparation_workspace_bytes(&self) -> Result<u64, HostedError> {
        self.encoding.workspace_bytes().map_err(HostedError::BatchSetup)
    }
    fn validate(&self, native: Int8MemoryRequirement, job: JobLimits) -> Result<(), HostedError> {
        let (policy, _, _) = self.config.encoding_requirements(&self.planner, self.encoding.limits())?;
        if &policy != self.encoding.policy() { return Err(HostedError::ModelIdentity); }
        check_capacity(self.config.task_ceiling, native.kv_bytes, job.max_result_bytes)
    }
    fn execute(self, engine: &mut StrictInt8Engine<'_>, admission: CorpusAdmission<'_>,
        request: SourceJobRequest, population: &Population<'_>, prefill: Option<Int8PrefillLimits>,
        control: &mut RunControl) -> Result<Result<JobProgress, JobRunError>, HostedError> {
        let Self { planner, config, encoding } = self;
        let recipe = JudgeJobRecipe::with_encoding(&config, prefill, &encoding)?;
        let ledger = WorkLedger::new(config.max_model_work);
        let native = JudgeProcessor { planner: &planner, config, engine, admission, ledger, prefill, encoding };
        Ok(run_population(request, population, JudgeJobProcessor { native, recipe }, control))
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
    #[serde(skip_serializing_if = "Option::is_none")]
    input_encoding: Option<UntrustedEncoding>,
    #[serde(skip_serializing_if = "Option::is_none")]
    encoding_work_limits: Option<SafeBpeWorkLimits>,
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
            defaults: config.defaults.clone(), prefill, input_encoding: None, encoding_work_limits: None };
        planning::check_serialized_size(&recipe, 1024 * 1024 - 1024)?;
        Ok(recipe)
    }

    fn with_encoding(config: &JudgeCorpusConfig, prefill: Option<Int8PrefillLimits>,
        encoding: &EncodingWorkScope) -> Result<Self, HostedError> {
        encoding.workspace_bytes().map_err(HostedError::BatchSetup)?;
        let mut recipe = Self::new(config, prefill)?;
        if matches!(encoding.policy(), UntrustedEncoding::SafeBpeV1(_)) {
            recipe.input_encoding = Some(encoding.policy().clone());
            recipe.encoding_work_limits = encoding.limits();
        }
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
    type Output = GuardedOutput<JudgeOutput, Pending>;
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
