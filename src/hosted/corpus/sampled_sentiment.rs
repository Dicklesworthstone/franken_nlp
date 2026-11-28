//! Independent, item-addressed score/explanation bundles on one resident INT8 model.
use super::*;
use crate::{
    batch::{BatchDocument, BatchFault, BatchProcessor, BatchWork, sentiment::sampled::{
        SampledSentimentBatchArgs, sampled_sentiment_batch_preparation_bytes,
    }},
    grammar::mask::MaskWorkLimits,
    native_engine::{constrained::JsonWorkBudget, constrained_int8::{Int8JsonBudget,
            json_sampling_workspace_bytes, json_sampling_reuse_workspace_bytes},
        decode::DecodeStepControl, simd::Int8Kernel,
        strict_int8::{StrictInt8Engine, prefill::Int8PrefillLimits}},
    tasks::sentiment::{SentimentRequest, sampled::{SampledSentimentError, SampledSentimentVocabulary,
        quantized::{Int8SampledSentimentError, Int8SampledSentimentPlanner, Int8SampledSentimentRun,
            PreparedInt8SampledSentiment}}},
};

/// Work ceilings cover every admitted record, including failed attempts.
/// The transport's flush control starts a fresh ID epoch only.
#[derive(Clone, Copy, Debug)]
pub struct SampledSentimentCorpusConfig {
    pub max_model_work: Int8Work,
    pub mask_limits: MaskWorkLimits,
    pub max_mask_node_visits: u64,
}

impl NlpEngine {
    /// Reuse one vocabulary, compiler and native allocation through EOF. Each
    /// sample starts with empty logical KV. Optional layer-major scheduling
    /// changes prompt execution without changing the item's sampling address.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_sampled_sentiment<R, W>(&self, model: &ResidentInt8,
        planner: Int8SampledSentimentPlanner, vocabulary: Arc<SampledSentimentVocabulary>,
        config: SampledSentimentCorpusConfig, limits: CorpusLimits, prefill: Option<Int8PrefillLimits>,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        self.batch_int8_sampled_sentiment_scheduled(model, planner, vocabulary, config,
            limits, prefill, false, None, reader, writer, cancellation)
    }

    /// Keep one fixed row team for every record, axis and independently
    /// addressed sample, including optional bounded prompt morsels. The
    /// terminal completion frame is withheld until all children physically
    /// join; flush does not renew any native, mask or cancellation allowance.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_sampled_sentiment_with_threads<R, W>(&self, model: &ResidentInt8,
        planner: Int8SampledSentimentPlanner, vocabulary: Arc<SampledSentimentVocabulary>,
        config: SampledSentimentCorpusConfig, limits: CorpusLimits, prefill: Option<Int8PrefillLimits>,
        kernel: Int8Kernel, compute_threads: usize, reader: R, writer: W,
        cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        self.batch_int8_sampled_sentiment_scheduled(model, planner, vocabulary, config,
            limits, prefill, false, Some((kernel, compute_threads)), reader, writer, cancellation)
    }

    /// Reuse each record/axis prompt only inside its exclusive sampled
    /// session. Flushes and records never retain KV or replenish cold debits.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_sampled_sentiment_reusing_prefill<R, W>(&self, model: &ResidentInt8,
        planner: Int8SampledSentimentPlanner, vocabulary: Arc<SampledSentimentVocabulary>,
        config: SampledSentimentCorpusConfig, limits: CorpusLimits, prefill: Option<Int8PrefillLimits>,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        self.batch_int8_sampled_sentiment_scheduled(model, planner, vocabulary, config,
            limits, prefill, true, None, reader, writer, cancellation)
    }

    /// Prompt reuse, bounded prefill and one persistent numerical team compose
    /// without changing semantic addresses or complete-stream work authority.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_sampled_sentiment_reusing_prefill_with_threads<R, W>(&self, model: &ResidentInt8,
        planner: Int8SampledSentimentPlanner, vocabulary: Arc<SampledSentimentVocabulary>,
        config: SampledSentimentCorpusConfig, limits: CorpusLimits, prefill: Option<Int8PrefillLimits>,
        kernel: Int8Kernel, compute_threads: usize, reader: R, writer: W,
        cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        self.batch_int8_sampled_sentiment_scheduled(model, planner, vocabulary, config,
            limits, prefill, true, Some((kernel, compute_threads)), reader, writer, cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn batch_int8_sampled_sentiment_scheduled<R, W>(&self, model: &ResidentInt8,
        planner: Int8SampledSentimentPlanner, vocabulary: Arc<SampledSentimentVocabulary>,
        config: SampledSentimentCorpusConfig, limits: CorpusLimits, prefill: Option<Int8PrefillLimits>,
        reuse_prefill: bool, team: Option<(Int8Kernel, usize)>, reader: R, writer: W,
        cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        let sampler_bytes = if reuse_prefill {
            json_sampling_reuse_workspace_bytes(planner.sampling_options().samples)
        } else { json_sampling_workspace_bytes() }
            .map_err(|e| HostedError::SampledSentiment(e.into()))?;
        let task = SampledStream { planner: Arc::new(planner), vocabulary, config, sampler_bytes,
            preparation_bytes: limits.preparation_reserve_bytes, reuse_prefill };
        scored::run(self, model, task, limits, prefill, team, reader, writer, cancellation)
    }
}

struct SampledStream {
    planner: Arc<Int8SampledSentimentPlanner>,
    vocabulary: Arc<SampledSentimentVocabulary>,
    config: SampledSentimentCorpusConfig,
    sampler_bytes: u64,
    preparation_bytes: u64,
    reuse_prefill: bool,
}
impl scored::CorpusTask for SampledStream {
    fn identity(&self) -> &ExecutionIdentity { self.planner.execution_identity() }
    fn retain_for_join(&self) -> Option<Box<dyn Send>> {
        Some(Box::new((Arc::clone(&self.planner), Arc::clone(&self.vocabulary))))
    }
    fn validate(&self, _model: &ArtifactIdentity, limits: CorpusLimits, native: Int8MemoryRequirement)
        -> Result<u64, HostedError> {
        validate_work(self.config.max_model_work)?;
        let task = self.planner.task_budget();
        let prep = sampled_sentiment_batch_preparation_bytes(task, self.planner.limits(), self.planner.input_encoding())
            .map_err(HostedError::BatchSetup)?;
        if prep > limits.preparation_reserve_bytes || native.kv_bytes > task.max_kv_bytes
            || task.max_output_bytes.min(self.planner.limits().max_result_bytes) > limits.transport.max_output_line_bytes as u64
            || self.config.mask_limits.max_trie_node_visits == 0
            || self.config.mask_limits.checkpoint_interval_nodes == 0 || self.config.max_mask_node_visits == 0 {
            return Err(HostedError::Limits("sampled sentiment corpus admission"));
        }
        Ok(self.sampler_bytes)
    }
    fn execute<R: BufRead, W: Write>(self, engine: &mut StrictInt8Engine<'_>,
        mut admission: CorpusAdmission<'_>, prefill: Option<Int8PrefillLimits>, reader: &mut R,
        writer: &mut W, limits: BatchLimits, defer: bool, control: &mut RunControl)
        -> Result<scored::Completion, HostedError> {
        admission.sampler_bytes = self.sampler_bytes;
        let mut processor = Processor { stream: self, engine, admission, prefill, spent: Debit::default() };
        scored::execute_stream(reader, writer, &mut processor, limits, defer, control)
    }
}

#[derive(Default, Debug, Eq, PartialEq)]
struct Debit { work: Int8Work, mask: u64 }
impl Debit {
    fn after(&self, work: Int8Work, mask: u64, config: SampledSentimentCorpusConfig)
        -> Result<Self, BatchItemFailure> {
        let refused = || BatchItemFailure::reject(BatchCode::WorkLimit);
        let work = self.work.checked_add(work).map_err(|_| refused())?;
        let cap = config.max_model_work;
        if work.forward_positions > cap.forward_positions || work.projected_logits > cap.projected_logits
            || work.attention_pairs > cap.attention_pairs || !work.projections.fits(cap.projections) {
            return Err(refused());
        }
        let mask = self.mask.checked_add(mask).filter(|&n| n <= config.max_mask_node_visits)
            .ok_or_else(refused)?;
        Ok(Self { work, mask })
    }
}

struct Processor<'e, 'w, 'a> {
    stream: SampledStream,
    engine: &'e mut StrictInt8Engine<'w>,
    admission: CorpusAdmission<'a>,
    prefill: Option<Int8PrefillLimits>,
    spent: Debit,
}
struct Prepared {
    plan: PreparedInt8SampledSentiment,
    budget: Int8JsonBudget,
    mask: u64,
}
impl BatchProcessor for Processor<'_, '_, '_> {
    type Args = SampledSentimentBatchArgs;
    type Prepared = Prepared;
    type Output = GuardedOutput<Int8SampledSentimentRun, Pending>;

    fn prepare(&mut self, _: BatchDocument<Self::Args>) -> Result<Prepared, BatchItemFailure> {
        Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, doc: BatchDocument<Self::Args>, control: &mut C)
        -> Result<Prepared, BatchItemFailure> {
        if self.engine.is_poisoned() || !self.engine.kv_cache().all_slots_have_len(0) {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        let args = doc.task_args.unwrap_or_default();
        let request = SentimentRequest { document: doc.text, axes: args.axes,
            budget: args.budget.unwrap_or(self.stream.planner.task_budget()) };
        let plan = self.stream.planner.plan_for_item_with_control(&request, &doc.id, control)
            .map_err(|error| task_failure(error, false))?;
        if plan.required_context() > self.engine.kv_cache().capacity_positions()
            || plan.task_budget().max_kv_bytes < self.admission.kv_bytes
            || plan.preparation_requirement_bytes() > self.stream.preparation_bytes {
            return Err(BatchItemFailure::reject(BatchCode::Admission));
        }
        let config = self.stream.config;
        let work = plan.planned_work();
        let budget = Int8JsonBudget { native: Int8RunBudget::exact(work),
            json: JsonWorkBudget { max_forward_positions: work.forward_positions,
                max_projected_logits: work.projected_logits, max_kv_bytes: self.admission.kv_bytes,
                mask_limits: config.mask_limits, max_total_mask_node_visits: config.max_mask_node_visits } };
        let mask = plan.complete_mask_charge(&self.stream.vocabulary, budget, self.stream.sampler_bytes)
            .map_err(|error| task_failure(error, false))?;
        self.spent.after(work, mask, config)?;
        Ok(Prepared { plan, budget, mask })
    }
    fn planned_work(&self, prepared: &Prepared) -> BatchWork {
        let work = prepared.plan.planned_work();
        BatchWork { forward_positions: work.forward_positions, projected_logits: work.projected_logits }
    }
    fn execute<C: DecodeStepControl>(&mut self, prepared: Prepared, control: &mut C)
        -> Result<Self::Output, BatchItemFailure> {
        let Prepared { plan, budget, mask } = prepared;
        let work = plan.planned_work();
        self.spent = self.spent.after(work, mask, self.stream.config)?;
        let (identity, guard) = self.admission.admit_output(plan.execution_identity(), work,
            self.admission.kv_bytes, self.stream.sampler_bytes, plan.max_result_bytes())?;
        let vocabulary = &self.stream.vocabulary;
        let sampler = self.stream.sampler_bytes;
        let result = if self.stream.reuse_prefill {
            plan.execute_reusing_prefill_with_control(self.engine, &identity, vocabulary,
                budget, sampler, self.prefill, control)
        } else { match self.prefill {
            Some(prefill) => plan.execute_layer_major_with_control(self.engine, &identity, vocabulary,
                budget, sampler, prefill, control),
            None => plan.execute_with_control(self.engine, &identity, vocabulary, budget, sampler, control),
        } }.map_err(|error| task_failure(error, true))?;
        let execution = plan.execution_version(self.stream.reuse_prefill);
        if self.engine.is_poisoned() || !self.engine.kv_cache().all_slots_have_len(0)
            || result.execution != execution || result.mask_node_visit_charge > mask {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        Ok(GuardedOutput::new(result, guard))
    }
}

fn task_failure(error: Int8SampledSentimentError, executing: bool) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() {
        return BatchItemFailure::fatal(BatchFault::cancelled(cause));
    }
    use Int8SampledSentimentError::Shared;
    match error {
        Shared(SampledSentimentError::WorkBudget | SampledSentimentError::TokenizerWork) if !executing =>
            BatchItemFailure::reject(BatchCode::WorkLimit),
        // Each record may tighten the compiled grammar and output ceilings.
        // A model-free planning refusal leaves the resident engine clean.
        Shared(SampledSentimentError::Limits | SampledSentimentError::Schema
            | SampledSentimentError::Template | SampledSentimentError::Tokenizer) if !executing =>
            BatchItemFailure::reject(BatchCode::Planning),
        Shared(SampledSentimentError::OutputBudget) if !executing => BatchItemFailure::reject(BatchCode::OutputLineLimit),
        Shared(SampledSentimentError::Identity) => BatchItemFailure::fatal(BatchCode::Admission),
        Shared(SampledSentimentError::Allocation) => BatchItemFailure::fatal(BatchCode::Allocation),
        Shared(SampledSentimentError::Serialization) => BatchItemFailure::fatal(BatchCode::Serialization),
        Shared(SampledSentimentError::Accounting) => BatchItemFailure::fatal(BatchCode::InvalidExecution),
        _ => BatchItemFailure::fatal(BatchCode::Execution),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn a_rejected_record_grammar_preserves_the_following_population() {
        let failure = task_failure(SampledSentimentError::Schema.into(), false);
        assert_eq!(failure.fault.code, BatchCode::Planning);
        assert!(!failure.stop);
        assert!(task_failure(SampledSentimentError::Schema.into(), true).stop);
        assert!(task_failure(SampledSentimentError::Accounting.into(), false).stop);
    }
    #[test]
    fn refused_attempts_leave_the_complete_prior_debit_intact() {
        let work = crate::native_engine::constrained_int8::planned_work(32, 64).unwrap();
        let twice = work.checked_add(work).unwrap();
        let config = SampledSentimentCorpusConfig { max_model_work: twice, max_mask_node_visits: 100,
            mask_limits: MaskWorkLimits { max_trie_node_visits: 10, checkpoint_interval_nodes: 2 } };
        let spent = Debit::default().after(work, 50, config).unwrap();
        for axis in 0..6 {
            let mut less = config;
            match axis {
                0 => less.max_model_work.forward_positions -= 1,
                1 => less.max_model_work.projected_logits -= 1,
                2 => less.max_model_work.attention_pairs -= 1,
                3 => less.max_model_work.projections.dot_products -= 1,
                4 => less.max_model_work.projections.multiply_accumulates -= 1,
                _ => less.max_mask_node_visits -= 1,
            }
            assert!(spent.after(work, 50, less).is_err());
            assert_eq!(spent, Debit { work, mask: 50 });
        }
        assert_eq!(spent.after(work, 50, config).unwrap(), Debit { work: twice, mask: 100 });
    }
}
