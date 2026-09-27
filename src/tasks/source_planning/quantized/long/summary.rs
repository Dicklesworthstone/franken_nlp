//! Complete cited INT8 document summary over the existing exact chunk plans.
//!
//! Native maps plus deterministic evidence union, NOT a neural synthesis pass.
//! All candidates survive every intermediate reduction. Only the complete root
//! is ranked/truncated. One consumed plan, engine and cancellation controller.

use super::*;
use std::cell::RefCell;
use crate::corpus::summarize::{CorpusSummaryError, CorpusSummaryLimits, CorpusSummaryResult,
    CorpusSummaryTask, SummaryPass, check_bytes};
use crate::tasks::{mapreduce::CHUNK_PROFILE, summarize::SummaryResult};

pub const INT8_CORPUS_SUMMARY_EXECUTION: &str = "portable-int8-cited-summary-map-exact-reduce-v1";

/// These bounds constrain the complete document, independently of each native
/// chunk's SummaryOptions and TaskBudget. No intermediate top-k is permitted.
#[derive(Clone, Copy, Debug)]
pub struct Int8SummaryLimits {
    pub aggregation: CorpusSummaryLimits,
    pub max_bullets: usize,
    pub max_result_bytes: usize,
}
impl Default for Int8SummaryLimits {
    fn default() -> Self {
        Self { aggregation: CorpusSummaryLimits::default(), max_bullets: 16, max_result_bytes: 4 * 1024 * 1024 }
    }
}
impl Int8SummaryLimits {
    pub fn validate(self, mapping: Int8SourceMapLimits) -> Result<(), Int8CorpusSummaryError> {
        self.aggregation.validate()?;
        if !(1..=1024).contains(&self.max_bullets)
            || !(1..=64 * 1024 * 1024).contains(&self.max_result_bytes)
            || self.aggregation.max_value_bytes > mapping.reduction.max_value_bytes
            || self.max_result_bytes > mapping.reduction.max_result_bytes {
            return Err(Int8CorpusSummaryError::InvalidLimits);
        }
        Ok(())
    }
}

pub enum Int8CorpusSummaryError {
    InvalidLimits, Accounting,
    Map(Int8SourceMapError), Source(Int8SourceError), Summary(SummaryError),
    Execution(ExecutionError<CorpusSummaryError<Int8SourceError>>),
}
impl fmt::Display for Int8CorpusSummaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidLimits => "invalid int8 document summary limits or task",
            Self::Accounting => "int8 document summary complete accounting diverged",
            Self::Map(_) => "int8 document summary map admission refused",
            Self::Source(_) => "int8 document summary native pass refused",
            Self::Summary(_) => "int8 document summary evidence or output refused",
            Self::Execution(_) => "int8 document summary reduction refused",
        })
    }
}
impl fmt::Debug for Int8CorpusSummaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Display::fmt(self, f) }
}
impl Error for Int8CorpusSummaryError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Map(e) => Some(e), Self::Source(e) => Some(e),
            Self::Summary(e) => Some(e), Self::Execution(e) => Some(e), _ => None }
    }
}
impl From<Int8SourceMapError> for Int8CorpusSummaryError { fn from(e: Int8SourceMapError) -> Self { Self::Map(e) } }
impl From<Int8SourceError> for Int8CorpusSummaryError { fn from(e: Int8SourceError) -> Self { Self::Source(e) } }
impl From<SummaryError> for Int8CorpusSummaryError { fn from(e: SummaryError) -> Self { Self::Summary(e) } }
impl Int8CorpusSummaryError {
    pub fn cancellation(&self) -> Option<DecodeCancellationKind> {
        match self {
            Self::Map(e) => e.cancellation(), Self::Source(e) => e.cancellation(),
            Self::Execution(ExecutionError::Task { source: CorpusSummaryError::Pass(e), .. }) => e.cancellation(),
            _ => None,
        }
    }
}

#[derive(Serialize)]
pub struct Int8CorpusSummaryRun {
    pub schema_version: u32,
    pub execution: &'static str,
    pub numerics_profile: &'static str,
    pub chunk_profile: &'static str,
    pub semantics: &'static str,
    /// Full mapped source interval, INCLUDING chunks whose bullets lose top-k.
    pub source_span: VerifiedSourceSpan,
    pub map_batches: usize,
    pub reduce_calls: usize,
    pub reduction_levels: usize,
    pub planned_model_work: Int8Work,
    pub model_work: Int8Work,
    pub reserved_mask_node_visits: u64,
    pub mask_node_visit_charge: u64,
    pub verification_scan_steps: u64,
    pub summary: CorpusSummaryResult,
}

impl PreparedInt8SourceMap<'_> {
    /// Model-free check of the entire prepared task set and reduction limits.
    /// A NER/keyphrase plan or mixed summary options cannot enter this reducer.
    pub fn check_summary(&self, limits: Int8SummaryLimits) -> Result<(), Int8CorpusSummaryError> {
        self.summary_options(limits).map(|_| ())
    }
    fn summary_options(&self, limits: Int8SummaryLimits) -> Result<SummaryOptions, Int8CorpusSummaryError> {
        limits.validate(self.limits)?;
        let mut options = None;
        for plan in &self.plans {
            let Finalizer::Summarize(value) = &plan.finalizer else { return Err(Int8CorpusSummaryError::InvalidLimits); };
            if options.is_some_and(|old| old != *value) || plan.execution_identity().task_spec != SUMMARIZE_TASK_VERSION {
                return Err(Int8CorpusSummaryError::InvalidLimits);
            }
            options = Some(*value);
        }
        options.ok_or(Int8CorpusSummaryError::InvalidLimits)
    }

    /// Every identity is admitted before the first forward. The original source
    /// and all prepared plans remain owned by the caller's resource reservation;
    /// no uncharged model/runtime is constructed here. A failed run is consumed.
    pub fn execute_summary_with_control<C: DecodeStepControl>(self, admitted: &[ExecutionIdentity],
        engine: &mut StrictInt8Engine<'_>, vocabulary: &ExtractionVocabulary,
        limits: Int8SummaryLimits, control: &mut C) -> Result<Int8CorpusSummaryRun, Int8CorpusSummaryError> {
        checkpoint(control)?;
        self.check_summary(limits)?;
        self.preflight(admitted, engine)?;
        let driver = NativeDriver { engine, vocabulary, control, limits: self.limits };
        self.execute_summary_with_driver(admitted, limits, driver)
    }

    // Private test seam; public callers cannot replace native execution with
    // fabricated summaries or receipts. Shares the actual map receipt checker.
    fn execute_summary_with_driver<D: SourceDriver>(self, admitted: &[ExecutionIdentity],
        limits: Int8SummaryLimits, driver: D) -> Result<Int8CorpusSummaryRun, Int8CorpusSummaryError> {
        let options = self.summary_options(limits)?;
        verify_identities(&self.plans, admitted)?;
        let driver = RefCell::new(driver);
        let mut checkpoint_error = None;
        let pass = Pass { plans: &self.plans, admitted, driver: &driver, options,
            next_chunk: 0, work: Int8Work::default(), masks: 0, mask_cap: self.limits.mask_visits_per_chunk };
        let mut task = CorpusSummaryTask::new_int8(pass, limits.aggregation)?;
        // RefCell borrows never overlap: the coordinator and synchronous native
        // calls use the SAME driver/control. Preserve the exact cancellation
        // kind instead of replacing it with an untyped map/reduce checkpoint.
        let reduced = mapreduce::execute(&self.chunks, &mut task, self.limits.reduction, || {
            match driver.borrow_mut().checkpoint() {
                Ok(()) => Ok(()),
                Err(error) => { checkpoint_error = Some(error); Err(MapReduceError::Cancelled) }
            }
        });
        if let Some(error) = checkpoint_error { return Err(error.into()); }
        let reduced = reduced.map_err(Int8CorpusSummaryError::Execution)?;
        driver.borrow_mut().checkpoint()?;
        let metadata = (reduced.root().source_span(), reduced.map_batches(), reduced.reduce_calls(), reduced.reduction_levels());
        let scan_steps = limits.aggregation.max_scan_steps - task.scan_steps_remaining();
        let pass = task.into_pass();
        if pass.next_chunk != self.plans.len() || reduced.root().value().mapped_chunks() != self.plans.len()
            || !within(pass.work, self.work) || pass.masks > self.masks {
            return Err(Int8CorpusSummaryError::Accounting);
        }
        let summary = reduced.into_value().into_ranked(limits.max_bullets, limits.max_result_bytes)?;
        if summary.forward_positions != pass.work.forward_positions || summary.projected_logits != pass.work.projected_logits
            || summary.mask_node_visit_charge != pass.masks { return Err(Int8CorpusSummaryError::Accounting); }
        let output = Int8CorpusSummaryRun { schema_version: 1, execution: INT8_CORPUS_SUMMARY_EXECUTION,
            numerics_profile: STRICT_INT8_PROFILE, chunk_profile: CHUNK_PROFILE,
            semantics: "exact-bullet-evidence-union-no-neural-synthesis-v1",
            source_span: metadata.0, map_batches: metadata.1, reduce_calls: metadata.2, reduction_levels: metadata.3,
            planned_model_work: self.work, model_work: pass.work, reserved_mask_node_visits: self.masks,
            mask_node_visit_charge: pass.masks, verification_scan_steps: scan_steps, summary };
        check_bytes(&output, limits.max_result_bytes)?;
        driver.borrow_mut().checkpoint()?;
        Ok(output)
    }
}

struct Pass<'a, D> {
    plans: &'a [PreparedInt8SourceTask], admitted: &'a [ExecutionIdentity], driver: &'a RefCell<D>,
    options: SummaryOptions, next_chunk: usize, work: Int8Work, masks: u64, mask_cap: u64,
}
impl<D: SourceDriver> SummaryPass for Pass<'_, D> {
    type Error = Int8SourceError;
    fn options(&self) -> SummaryOptions { self.options }
    fn run(&mut self, chunk: &SourceChunk<'_>) -> Result<SummaryResult, Int8SourceError> {
        let mut driver = self.driver.borrow_mut();
        driver.checkpoint()?;
        if chunk.id() != self.next_chunk { return Err(Int8SourceError::InvalidResult); }
        let plan = self.plans.get(chunk.id()).ok_or(Int8SourceError::InvalidResult)?;
        let admitted = self.admitted.get(chunk.id()).ok_or(Int8SourceError::InvalidResult)?;
        let result = driver.run(plan, admitted)?;
        check_run(plan, &result, self.mask_cap)?;
        self.work = add_work(self.work, result.model_work).ok_or(Int8SourceError::InvalidResult)?;
        self.masks = self.masks.checked_add(mask_charge(&result.result)?).ok_or(Int8SourceError::InvalidResult)?;
        let SourceTaskResult::Summarize(summary) = result.result else { return Err(Int8SourceError::InvalidResult); };
        self.next_chunk += 1;
        driver.checkpoint()?;
        Ok(summary)
    }
}

#[cfg(test)] mod tests;
