//! One persistent INT8 row team for complete multi-head scoring requests.
//! The task planners own score spaces, full denominators and semantic output.
//! This module changes only the physical projection schedule and admission.

use super::*;
use crate::{
    hosted::sentiment::{self, SentimentHostLimits},
    native_engine::{simd::Int8Kernel, strict_int8::{Int8Work, StrictInt8Engine}},
    tasks::sentiment::quantized::{Int8SentimentError, Int8SentimentRun, PreparedInt8Sentiment},
};

#[derive(Clone, Copy)]
struct ScoredTaskQuote {
    task: TaskBudget,
    context: usize,
    work: Int8Work,
    heads: usize,
    result_bytes: u64,
}

/// Private implementations accept only the existing sealed native plans. No
/// caller can substitute a score producer or manufacture a completed result.
trait ScoredTeamPlan: Send + 'static {
    type Output: Send + 'static;
    fn identity(&self) -> &ExecutionIdentity;
    fn quote(&self) -> ScoredTaskQuote;
    fn execute(&self, engine: &mut StrictInt8Engine<'_>, budget: Int8ScoringBudget,
        prefill: Option<Int8PrefillLimits>, control: &mut RunControl)
        -> Result<Self::Output, HostedError>;
    fn accounting(output: &Self::Output) -> (Int8Work, usize);
    fn accounting_error() -> HostedError;
}

impl ScoredTeamPlan for PreparedInt8Classification {
    type Output = Int8ClassificationRun;
    fn identity(&self) -> &ExecutionIdentity { self.execution_identity() }
    fn quote(&self) -> ScoredTaskQuote {
        ScoredTaskQuote { task: self.task_budget(), context: self.required_context(),
            work: self.planned_work(), heads: self.head_count(),
            result_bytes: self.task_budget().max_output_bytes }
    }
    fn execute(&self, engine: &mut StrictInt8Engine<'_>, budget: Int8ScoringBudget,
        prefill: Option<Int8PrefillLimits>, control: &mut RunControl)
        -> Result<Self::Output, HostedError> {
        match prefill {
            Some(prefill) => self.execute_layer_major_with_control(self.execution_identity(),
                engine, budget, prefill, control),
            None => self.execute_with_control(self.execution_identity(), engine, budget, control),
        }.map_err(HostedError::Classification)
    }
    fn accounting(output: &Self::Output) -> (Int8Work, usize) { (output.model_work, output.head_count) }
    fn accounting_error() -> HostedError { HostedError::Classification(Int8ClassificationError::Accounting) }
}

impl ScoredTeamPlan for PreparedInt8Judge {
    type Output = Int8JudgeRun;
    fn identity(&self) -> &ExecutionIdentity { self.execution_identity() }
    fn quote(&self) -> ScoredTaskQuote {
        ScoredTaskQuote { task: self.task_budget(), context: self.required_context(),
            work: self.planned_work(), heads: self.head_count(), result_bytes: self.max_result_bytes() }
    }
    fn execute(&self, engine: &mut StrictInt8Engine<'_>, budget: Int8ScoringBudget,
        prefill: Option<Int8PrefillLimits>, control: &mut RunControl)
        -> Result<Self::Output, HostedError> {
        match prefill {
            Some(prefill) => self.execute_layer_major_with_control(self.execution_identity(),
                engine, budget, prefill, control),
            None => self.execute_with_control(self.execution_identity(), engine, budget, control),
        }.map_err(HostedError::Judge)
    }
    fn accounting(output: &Self::Output) -> (Int8Work, usize) { (output.model_work, output.head_count) }
    fn accounting_error() -> HostedError { HostedError::Judge(Int8JudgeError::Accounting) }
}

impl ScoredTeamPlan for PreparedInt8Sentiment {
    type Output = Int8SentimentRun;
    fn identity(&self) -> &ExecutionIdentity { self.execution_identity() }
    fn quote(&self) -> ScoredTaskQuote {
        ScoredTaskQuote { task: self.task_budget(), context: self.required_context(),
            work: self.planned_work(), heads: self.head_count(), result_bytes: self.max_result_bytes() }
    }
    fn execute(&self, engine: &mut StrictInt8Engine<'_>, budget: Int8ScoringBudget,
        prefill: Option<Int8PrefillLimits>, control: &mut RunControl)
        -> Result<Self::Output, HostedError> {
        match prefill {
            Some(prefill) => self.execute_layer_major_with_control(self.execution_identity(),
                engine, budget, prefill, control),
            None => self.execute_with_control(self.execution_identity(), engine, budget, control),
        }.map_err(HostedError::Sentiment)
    }
    fn accounting(output: &Self::Output) -> (Int8Work, usize) { (output.model_work, output.head_count) }
    fn accounting_error() -> HostedError { HostedError::Sentiment(Int8SentimentError::Accounting) }
}

/// Preserve each existing public preparation contract. Classification and
/// judgment keep caller-admitted prepared storage; sentiment transfers an
/// additional explicit host charge together with the complete plan.
#[derive(Clone, Copy)]
enum ScoredHostLimits { CallerPrepared(NativeLimits), Sentiment(SentimentHostLimits) }
impl ScoredHostLimits {
    fn native(self) -> NativeLimits {
        match self { Self::CallerPrepared(native) => native, Self::Sentiment(limits) => limits.native }
    }
    fn validate(self, quote: ScoredTaskQuote, kv_bytes: u64) -> Result<(), HostedError> {
        check_capacity(quote.context, quote.task, self.native().context_tokens, kv_bytes)?;
        if quote.heads == 0 || quote.result_bytes == 0 {
            return Err(HostedError::Limits("complete scored task admission"));
        }
        if let Self::Sentiment(limits) = self {
            sentiment::validate(limits, quote.task, quote.context, kv_bytes, quote.work)?;
        }
        Ok(())
    }
    fn input<P>(self, lease: &EngineLease, prepared: P) -> Result<ScoredInput<P>, HostedError> {
        match self {
            Self::CallerPrepared(_) => Ok(ScoredInput::CallerPrepared(prepared)),
            Self::Sentiment(limits) => allocate(
                Pending::reserve(lease, MemoryClass::JobBuffers, limits.preparation_reserve_bytes)?,
                || Ok(prepared)).map(ScoredInput::Charged),
        }
    }
}

enum ScoredInput<P> { CallerPrepared(P), Charged(Charged<P>) }
impl<P> ScoredInput<P> {
    fn prepared(&self) -> &P {
        match self { Self::CallerPrepared(prepared) => prepared, Self::Charged(input) => &input.value }
    }
}

/// Admission is dispatcher-owned outside the native scope. All reservations
/// survive an early return or unwind until the persistent children join.
/// Prepared storage drops before its own charge; native storage lives inside
/// the scope and drops before its KV/scratch charges.
struct ScoredTeamAdmission<P> {
    input: ScoredInput<P>,
    kv: Option<Pending>, scratch: Option<Pending>, output: Option<Pending>,
    _kv: Option<CommittedMemory>, _scratch: Option<CommittedMemory>, _lease: EngineLease,
}
impl<P> ScoredTeamAdmission<P> {
    fn commit_native(&mut self) -> Result<(), HostedError> {
        self._kv = Some(self.kv.take().ok_or(HostedError::Limits("INT8 scoring KV admission consumed"))?.commit()?);
        self._scratch = Some(self.scratch.take().ok_or(HostedError::Limits("INT8 scoring scratch admission consumed"))?.commit()?);
        Ok(())
    }
    fn finish<T>(&mut self, result: T, compute: Int8ComputeReceipt)
        -> Result<HostedInt8TeamResult<T>, HostedError> {
        let output = self.output.take().ok_or(HostedError::Limits("INT8 scoring output admission consumed"))?;
        match output.commit() {
            Ok(memory) => Ok(HostedInt8TeamResult::from_guarded(GuardedOutput::new(result, memory), compute)),
            Err(error) => { drop(result); Err(error) }
        }
    }
}

struct TeamRequirements { native: Int8MemoryRequirement, scratch_bytes: u64, max_batch_rows: usize }
fn team_requirements(limits: ScoredHostLimits, quote: ScoredTaskQuote, width: usize,
    prefill: Option<Int8PrefillLimits>) -> Result<TeamRequirements, HostedError> {
    let native = requirements(limits.native())?;
    limits.validate(quote, native.kv_bytes)?;
    let scratch_bytes = scoring_scratch(sum(&[native.rope_bytes, native.scratch_payload_bound,
        limits.native().allocator_reserve_bytes])?, prefill)?;
    let max_batch_rows = prefill.map_or(1, |limits| limits.max_batch_rows);
    // Fixed child capacity is additional to every coordinator/preparation
    // allocation. The dispatcher owns the corresponding separate reservation.
    sum(&[scratch_bytes, dispatch::int8_prefill_team_workspace_bytes(width, max_batch_rows)?])?;
    Ok(TeamRequirements { native, scratch_bytes, max_batch_rows })
}

fn complete_accounting(quote: ScoredTaskQuote, work: Int8Work, heads: usize,
    poisoned: bool, empty_kv: bool) -> bool {
    quote.heads > 0 && work == quote.work && heads == quote.heads && !poisoned && empty_kv
}

#[allow(clippy::too_many_arguments)]
fn execute_scored_team<P: ScoredTeamPlan>(host: &NlpEngine, model: &ResidentInt8, prepared: P,
    limits: ScoredHostLimits, kernel: Int8Kernel, width: usize, prefill: Option<Int8PrefillLimits>,
    cancellation: CancellationToken) -> Result<HostedInt8TeamResult<P::Output>, HostedError> {
    let native = limits.native();
    dispatch::preflight_int8_team(host, native.run, width)?;
    host.check_resident_domain(model)?;
    check_model_identity(model.artifact_identity(), prepared.identity())?;
    let quote = prepared.quote();
    let required = team_requirements(limits, quote, width, prefill)?;
    let lease = host.resources().acquire_lease();
    let input = limits.input(&lease, prepared)?;
    let output = output_claim(&lease, quote.result_bytes, u64::from(quote.task.max_output_tokens))?;
    let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
    let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.scratch_bytes)?;
    let admission = ScoredTeamAdmission { input, kv: Some(kv), scratch: Some(scratch), output: Some(output),
        _kv: None, _scratch: None, _lease: lease };
    dispatch::run_int8_prefill_team(host, native.run, cancellation, model.clone(), width, kernel,
        required.max_batch_rows, admission, move |model, admission, control, team| {
            let mut engine = model.engine(native.context_tokens, memory_budget(required.native))
                .map_err(HostedError::Model)?.with_kernel(kernel)
                .with_projection_executor(team).map_err(HostedError::Native)?;
            admission.commit_native()?;
            let prepared = admission.input.prepared();
            let budget = Int8ScoringBudget { native: Int8RunBudget::exact(quote.work),
                max_kv_bytes: required.native.kv_bytes };
            let result = prepared.execute(&mut engine, budget, prefill, control)?;
            let (work, heads) = P::accounting(&result);
            if !complete_accounting(quote, work, heads, engine.is_poisoned(), engine.kv_cache().all_slots_have_len(0)) {
                return Err(P::accounting_error());
            }
            drop(engine);
            let compute = team.stats().into();
            admission.finish(result, compute)
        })
}

impl NlpEngine {
    /// Score every exclusive or independent classification head with one
    /// persistent numerical team. Width includes the coordinator, is at least
    /// two and must fit the installed process envelope. Prepared allocations
    /// retain the existing caller-admitted contract. Each head keeps its exact
    /// full-vocabulary normalizers, EOS semantics and cumulative work.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int8_classify_with_threads(&self, model: &ResidentInt8,
        prepared: PreparedInt8Classification, native: NativeLimits, kernel: Int8Kernel,
        compute_threads: usize, cancellation: CancellationToken)
        -> Result<HostedInt8TeamResult<Int8ClassificationRun>, HostedError> {
        execute_scored_team(self, model, prepared, ScoredHostLimits::CallerPrepared(native),
            kernel, compute_threads, None, cancellation)
    }

    /// Share the same fixed row team across bounded layer-major prompt
    /// morsels and every classification continuation. Coordinator and child
    /// prompt storage are both admitted before worker formation.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int8_classify_layer_major_with_threads(&self, model: &ResidentInt8,
        prepared: PreparedInt8Classification, native: NativeLimits, kernel: Int8Kernel,
        compute_threads: usize, prefill: Int8PrefillLimits, cancellation: CancellationToken)
        -> Result<HostedInt8TeamResult<Int8ClassificationRun>, HostedError> {
        execute_scored_team(self, model, prepared, ScoredHostLimits::CallerPrepared(native),
            kernel, compute_threads, Some(prefill), cancellation)
    }

    /// Preserve the complete pairwise, rubric or faithfulness/evidence bundle
    /// while its native projections share one persistent row team. No failed
    /// head becomes a partial judgment. Every child joins before publication.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int8_judge_with_threads(&self, model: &ResidentInt8,
        prepared: PreparedInt8Judge, native: NativeLimits, kernel: Int8Kernel,
        compute_threads: usize, cancellation: CancellationToken)
        -> Result<HostedInt8TeamResult<Int8JudgeRun>, HostedError> {
        execute_scored_team(self, model, prepared, ScoredHostLimits::CallerPrepared(native),
            kernel, compute_threads, None, cancellation)
    }

    /// Combine bounded layer-major prompts and persistent workers throughout
    /// every judgment head. The original result, work and cancellation
    /// contracts are retained; width is physical metadata only.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int8_judge_layer_major_with_threads(&self, model: &ResidentInt8,
        prepared: PreparedInt8Judge, native: NativeLimits, kernel: Int8Kernel,
        compute_threads: usize, prefill: Int8PrefillLimits, cancellation: CancellationToken)
        -> Result<HostedInt8TeamResult<Int8JudgeRun>, HostedError> {
        execute_scored_team(self, model, prepared, ScoredHostLimits::CallerPrepared(native),
            kernel, compute_threads, Some(prefill), cancellation)
    }

    /// All independent sentiment axes reuse one fixed native row team and
    /// the existing largest-head KV allocation. The complete model allowance
    /// and transferred preparation charge cover every axis until join.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int8_sentiment_with_threads(&self, model: &ResidentInt8,
        prepared: PreparedInt8Sentiment, limits: SentimentHostLimits, kernel: Int8Kernel,
        compute_threads: usize, cancellation: CancellationToken)
        -> Result<HostedInt8TeamResult<Int8SentimentRun>, HostedError> {
        execute_scored_team(self, model, prepared, ScoredHostLimits::Sentiment(limits),
            kernel, compute_threads, None, cancellation)
    }

    /// Use bounded layer-major prompt morsels and the same persistent row
    /// team for all sentiment axes. This preserves the selected probability
    /// space and never treats independent dimensions as an exclusive class.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int8_sentiment_layer_major_with_threads(&self, model: &ResidentInt8,
        prepared: PreparedInt8Sentiment, limits: SentimentHostLimits, kernel: Int8Kernel,
        compute_threads: usize, prefill: Int8PrefillLimits, cancellation: CancellationToken)
        -> Result<HostedInt8TeamResult<Int8SentimentRun>, HostedError> {
        execute_scored_team(self, model, prepared, ScoredHostLimits::Sentiment(limits),
            kernel, compute_threads, Some(prefill), cancellation)
    }
}

#[cfg(test)]
mod tests;
