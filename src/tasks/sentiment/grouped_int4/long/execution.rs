//! One complete map lifetime with exact native finalizers and no public driver.
use std::io;
use super::*;
use crate::native_engine::strict_int4::scoring::cohort::{task::document_layout,
    INT4_SCORING_COHORT_EXECUTION};

impl PreparedInt4SentimentMap<'_> {
    /// Check every original chunk against the actual resident engine before
    /// an outer corpus runner admits an attempt. This performs no forwards.
    pub fn preflight_with_control<C: DecodeStepControl>(&self, admitted: &[ExecutionIdentity],
        engine: &StrictInt4Engine<'_>, budget: Int4ScoringBudget, control: &mut C)
        -> Result<(), Int4SentimentMapError> {
        checkpoint(control)?; self.check_identities(admitted, engine.model_identity(), budget.native, control)?;
        for (plan, identity) in self.plans.iter().zip(admitted) {
            checkpoint(control)?;
            plan.preflight(identity, engine, Int4ScoringBudget {
                native: Int4RunBudget::exact(plan.planned_work()), max_kv_bytes: budget.max_kv_bytes })?;
        }
        checkpoint(control)
    }

    /// Every document and every counter is preflighted before any forward.
    /// The serial engine reuses one physical cache across all chunk axes.
    pub fn execute_with_control<C: DecodeStepControl>(&self, admitted: &[ExecutionIdentity],
        engine: &mut StrictInt4Engine<'_>, budget: Int4ScoringBudget, control: &mut C)
        -> Result<Int4SentimentMapRun, Int4SentimentMapError> {
        self.preflight_with_control(admitted, engine, budget, control)?;
        let mut collected = Collection::new(self)?;
        for (index, (plan, identity)) in self.plans.iter().zip(admitted).enumerate() {
            checkpoint(control)?;
            let run = plan.execute_with_control(identity, engine, Int4ScoringBudget {
                native: Int4RunBudget::exact(plan.planned_work()), max_kv_bytes: budget.max_kv_bytes }, control)?;
            if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) {
                return Err(Int4SentimentMapError::Admission);
            }
            collected.push(index, run, control)?;
        }
        collected.finish(control)
    }

    /// Validate every exact ragged window using the real model and complete
    /// cohort allowance, without constructing an engine or running a layer.
    pub fn preflight_cohorts_with_control<C: DecodeStepControl>(&self, admitted: &[ExecutionIdentity],
        weights: &Int4ModelWeights, cohort_chunks: usize, budget: Int4SentimentMapCohortBudget,
        control: &mut C) -> Result<(), Int4SentimentMapError> {
        checkpoint(control)?; self.check_identities(admitted, weights.identity(), budget.native, control)?;
        self.cohort_windows(admitted, cohort_chunks, budget, control)?;
        checkpoint(control)
    }

    /// Real simultaneous chunk × axis heads, with separate KV and original
    /// semantic finalizers. Exact geometry for EVERY future window is checked
    /// before the first cohort is allocated or forwarded.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_cohorts_with_control<C: DecodeStepControl>(&self, admitted: &[ExecutionIdentity],
        weights: &Int4ModelWeights, kernel: PackedInt4Kernel, cohort_chunks: usize,
        budget: Int4SentimentMapCohortBudget, control: &mut C)
        -> Result<Int4SentimentMapCohortRun, Int4SentimentMapError> {
        checkpoint(control)?; self.check_identities(admitted, weights.identity(), budget.native, control)?;
        let windows = self.cohort_windows(admitted, cohort_chunks, budget, control)?;
        let mut collected = Collection::new(self)?;
        let mut group_steps = 0_u64; let mut projection_groups = 0_u64;
        for window in &windows {
            checkpoint(control)?;
            let mut documents = reserved(window.end - window.start)?;
            for index in window.start..window.end { documents.push((&self.plans[index], &admitted[index])); }
            let mut engine = Int4CohortEngine::new(weights, &window.contexts, budget.memory)?.with_kernel(kernel);
            let run = PreparedInt4Sentiment::execute_document_cohort_with_control(&documents, &mut engine,
                Int4ScoringCohortBudget { native: Int4RunBudget::exact(window.work),
                    max_kv_bytes: budget.memory.max_kv_bytes, max_output_bytes: budget.max_output_bytes }, control)?;
            engine.check_idle()?;
            if run.schema_version != 1 || run.execution != INT4_SCORING_COHORT_EXECUTION
                || run.output.len() != window.end - window.start {
                return Err(Int4SentimentMapError::Admission);
            }
            group_steps = group_steps.checked_add(run.group_steps).ok_or(Int4SentimentMapError::Work)?;
            projection_groups = projection_groups.checked_add(run.projection_groups).ok_or(Int4SentimentMapError::Work)?;
            for (index, result) in (window.start..window.end).zip(run.output) { collected.push(index, result, control)?; }
            drop(engine); checkpoint(control)?;
        }
        let output = collected.finish(control)?;
        self.complete_cohorts(output, cohort_chunks, windows.len(), group_steps, projection_groups, control)
    }

    pub(super) fn complete_cohorts<C: DecodeStepControl>(&self, output: Int4SentimentMapRun,
        cohort_chunks: usize, chunk_cohorts: usize, group_steps: u64, projection_groups: u64,
        control: &mut C) -> Result<Int4SentimentMapCohortRun, Int4SentimentMapError> {
        checkpoint(control)?;
        let cap = self.limits.max_result_bytes.checked_add(COHORT_WRAPPER_BYTES).ok_or(Int4SentimentMapError::Limits)?;
        let mut run = Int4SentimentMapCohortRun { schema_version: 1, execution: INT4_SENTIMENT_MAP_COHORT_EXECUTION.to_owned(),
            chunk_cohorts, group_steps, projection_groups, output,
            executed_cohort_chunks: cohort_chunks.min(self.chunk_count()), result_binding: [0; 32] };
        run.result_binding = binding::cohort_with_control(&run, cap, control)?;
        self.expected.verify_completed_cohorts_with_control(&run, cohort_chunks, control)?;
        checkpoint(control)?; Ok(run)
    }

    fn check_identities<C: DecodeStepControl>(&self, admitted: &[ExecutionIdentity], model: &Int4ModelIdentity,
        native: Int4RunBudget, control: &mut C) -> Result<(), Int4SentimentMapError> {
        if admitted.len() != self.plans.len() || self.plans.len() != self.expected.chunks
            || self.plans.len() != self.chunks.chunks().len() || model.group_width() != self.group_width()
            || !fits(self.expected.work, self.limits.max_model_work) { return Err(Int4SentimentMapError::Admission); }
        check_work(self.expected.work, native)?;
        let mut total = Int4Work::default();
        for (plan, identity) in self.plans.iter().zip(admitted) {
            checkpoint(control)?; plan.verify_identity(identity)?; check_model(identity, model)?;
            if plan.group_width() != self.group_width() || plan.task_budget() != self.task
                || plan.head_count() != self.expected.heads { return Err(Int4SentimentMapError::Admission); }
            total = total.checked_add(plan.planned_work())?;
        }
        if total != self.expected.work { return Err(Int4SentimentMapError::Admission); }
        checkpoint(control)?; Ok(())
    }

    pub(super) fn cohort_windows<C: DecodeStepControl>(&self, admitted: &[ExecutionIdentity], chunks: usize,
        budget: Int4SentimentMapCohortBudget, control: &mut C) -> Result<Vec<Window>, Int4SentimentMapError> {
        if chunks == 0 || chunks > self.limits.chunks.max_chunks || self.head_count() == 0
            || chunks.checked_mul(self.head_count()).is_none_or(|heads| heads > MAX_INT4_SEQUENCES)
            || admitted.len() != self.plans.len() || !(1..=MAX_BYTES).contains(&budget.max_output_bytes)
            || budget.max_task_workspace_bytes == 0 { return Err(Int4SentimentMapError::Limits); }
        check_work(self.expected.work, budget.native)?;
        let mut windows = reserved(self.plans.len().div_ceil(chunks))?;
        for start in (0..self.plans.len()).step_by(chunks) {
            checkpoint(control)?; let end = start + chunks.min(self.plans.len() - start);
            let mut contexts = reserved((end - start) * self.head_count())?;
            let mut documents = reserved(end - start)?; let mut expected = Int4Work::default();
            for index in start..end {
                checkpoint(control)?;
                let plan = &self.plans[index]; plan.verify_identity(&admitted[index])?;
                contexts.extend(plan.cohort_contexts()?);
                expected = expected.checked_add(plan.planned_work())?;
                documents.push((plan, &admitted[index]));
            }
            let required = Int4MemoryRequirement::for_cohort(&contexts, self.group_width())?;
            if required.kv_bytes > budget.memory.max_kv_bytes || required.rope_bytes > budget.memory.max_rope_bytes
                || required.scratch_payload_bound > budget.memory.max_scratch_payload_bytes {
                return Err(Int4SentimentMapError::Admission);
            }
            let (_, work) = document_layout(&documents, &contexts, Int4ScoringCohortBudget {
                native: Int4RunBudget::exact(expected), max_kv_bytes: budget.memory.max_kv_bytes,
                max_output_bytes: budget.max_output_bytes })?;
            if work != expected || required_task_workspace_bytes(work, contexts.len(), budget.max_output_bytes)?
                > budget.max_task_workspace_bytes { return Err(Int4SentimentMapError::Admission); }
            windows.push(Window { start, end, contexts, work });
        }
        checkpoint(control)?; Ok(windows)
    }
}

pub(super) struct Window { pub start: usize, pub end: usize, pub contexts: Vec<usize>, pub work: Int4Work }

pub(super) struct Collection<'map, 'source> {
    prepared: &'map PreparedInt4SentimentMap<'source>,
    chunks: Vec<Int4SentimentMapChunk>,
    work: Int4Work,
    bytes: u64,
    failed: bool,
}
impl<'map, 'source> Collection<'map, 'source> {
    pub(super) fn new(prepared: &'map PreparedInt4SentimentMap<'source>) -> Result<Self, Int4SentimentMapError> {
        Ok(Self { prepared, chunks: reserved(prepared.chunk_count())?, work: Int4Work::default(), bytes: 0, failed: false })
    }
    pub(super) fn push<C: DecodeStepControl>(&mut self, index: usize, run: Int4SentimentRun, control: &mut C)
        -> Result<(), Int4SentimentMapError> {
        if self.failed { return Err(Int4SentimentMapError::Admission); }
        self.failed = true; checkpoint(control)?;
        let plan = self.prepared.plans.get(index).ok_or(Int4SentimentMapError::Admission)?;
        let source = self.prepared.chunks.chunks().get(index).ok_or(Int4SentimentMapError::Admission)?;
        if index != self.chunks.len() || run.schema_version != 1 || run.execution != INT4_SENTIMENT_EXECUTION
            || run.numerics_profile != STRICT_INT4_PROFILE || run.group_width != plan.group_width()
            || run.head_count != plan.head_count() || run.model_work != plan.planned_work()
            || run.result.dimensions.len() != plan.head_count() { return Err(Int4SentimentMapError::Admission); }
        let chunk = Int4SentimentMapChunk { chunk_id: index, source_span: source.span(), source_tokens: source.tokens(), run };
        let remaining = self.prepared.limits.max_retained_result_bytes.checked_sub(self.bytes).ok_or(Int4SentimentMapError::Output)?;
        let bytes = value_bytes(&chunk, remaining)?;
        self.bytes = self.bytes.checked_add(bytes).ok_or(Int4SentimentMapError::Output)?;
        self.work = self.work.checked_add(chunk.run.model_work)?;
        if !fits(self.work, self.prepared.planned_work()) { return Err(Int4SentimentMapError::Work); }
        self.chunks.push(chunk); checkpoint(control)?; self.failed = false; Ok(())
    }
    pub(super) fn finish<C: DecodeStepControl>(self, control: &mut C) -> Result<Int4SentimentMapRun, Int4SentimentMapError> {
        checkpoint(control)?;
        if self.failed || self.chunks.len() != self.prepared.chunk_count() || self.work != self.prepared.planned_work() {
            return Err(Int4SentimentMapError::Admission);
        }
        let span = self.prepared.expected.source_span;
        let aggregate = aggregate_with_control(self.chunks.iter().map(|chunk| SentimentChunkView {
            chunk_id: chunk.chunk_id, span: chunk.source_span, result: &chunk.run.result }), self.chunks.len(),
            span.byte_end, span.scalar_end, self.prepared.limits.aggregation, control)?;
        let mut run = Int4SentimentMapRun { schema_version: 1, execution: INT4_SENTIMENT_MAP_EXECUTION.to_owned(),
            numerics_profile: STRICT_INT4_PROFILE.to_owned(), group_width: self.prepared.group_width(),
            chunk_profile: CHUNK_PROFILE.to_owned(), semantics: SEMANTICS.to_owned(), single_context_equivalence: false,
            source_span: span, chunk_limits: self.prepared.chunks.limits(), tokenizer_checks: self.prepared.chunks.tokenizer_calls(),
            planned_model_work: self.prepared.planned_work(), model_work: self.work, retained_result_bytes: self.bytes,
            chunks: self.chunks, aggregate, input_encoding: self.prepared.expected.encoding_record(),
            request_binding: self.prepared.expected.request_binding,
            result_binding: [0; 32] };
        run.result_binding = binding::result_with_control(&run, self.prepared.limits.max_result_bytes, control)?;
        value_bytes(&run, self.prepared.limits.max_result_bytes)?; checkpoint(control)?; Ok(run)
    }
}

/// Count with a nonallocating writer before allocating canonical bytes. This
/// prevents oversized complete maps from allocating beyond their output cap.
pub(super) fn value_bytes(value: &impl Serialize, cap: u64) -> Result<u64, Int4SentimentMapError> {
    struct Counter { bytes: u64, cap: u64, exceeded: bool }
    impl io::Write for Counter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if bytes.len() as u64 > self.cap - self.bytes {
                self.exceeded = true; return Err(io::Error::other("native sentiment map output budget"));
            }
            self.bytes += bytes.len() as u64; Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> { Ok(()) }
    }
    let mut counter = Counter { bytes: 0, cap, exceeded: false };
    if serde_json::to_writer(&mut counter, value).is_err() {
        return Err(if counter.exceeded { Int4SentimentMapError::Output } else { Int4SentimentMapError::Serialization });
    }
    let bytes = canonjson::canonical_bytes(value).map_err(|_| Int4SentimentMapError::Serialization)?;
    if bytes.len() as u64 > cap { return Err(Int4SentimentMapError::Output); }
    Ok(bytes.len() as u64)
}
