//! Complete cited summaries by exact bullet/evidence union and final ranking.
//! Native maps consume every chunk. All candidates survive intermediate merges;
//! top-k runs only once at the complete root. This is not neural synthesis.
use super::*;
use super::execution::{SourceDriver, NativeDriver, check_run, mask_charge, original_spans, preflight};
use crate::{
    corpus::summarize::{CorpusSummaryError, CorpusSummaryLimits, CorpusSummaryResult, CorpusSummaryTask, SummaryPass, check_bytes},
    tasks::{mapreduce::CHUNK_PROFILE, summarize::{SummaryResult, CitationGuarantee, SummarySemanticSupport}},
};

pub const INT4_CORPUS_SUMMARY_EXECUTION: &str = "grouped-int4-cited-summary-map-exact-reduce-v1";
const SUMMARY_SEMANTICS: &str = "exact-bullet-evidence-union-no-neural-synthesis-v1";

#[derive(Clone, Copy, Debug)]
pub struct Int4SummaryLimits {
    pub aggregation: CorpusSummaryLimits,
    pub max_bullets: usize,
    pub max_result_bytes: usize,
}
impl Default for Int4SummaryLimits {
    fn default() -> Self {
        Self { aggregation: CorpusSummaryLimits::default(), max_bullets: 16, max_result_bytes: 4 * 1024 * 1024 }
    }
}
impl Int4SummaryLimits {
    pub fn validate(self, mapping: Int4SourceMapLimits) -> Result<(), Int4CorpusSummaryError> {
        self.aggregation.validate()?;
        if !(1..=1024).contains(&self.max_bullets) || !(1..=64 * 1024 * 1024).contains(&self.max_result_bytes)
            || self.aggregation.max_value_bytes > mapping.reduction.max_value_bytes
            || self.max_result_bytes > mapping.reduction.max_result_bytes {
            return Err(Int4CorpusSummaryError::InvalidLimits);
        }
        Ok(())
    }
}

pub enum Int4CorpusSummaryError {
    InvalidLimits, Accounting,
    Map(Int4SourceMapError), Source(Int4SourceError), Summary(SummaryError),
    Execution(ExecutionError<CorpusSummaryError<Int4SourceError>>),
}
impl fmt::Display for Int4CorpusSummaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidLimits => "invalid INT4 document summary limits or task",
            Self::Accounting => "INT4 document summary complete accounting diverged",
            Self::Map(_) => "INT4 document summary map admission refused",
            Self::Source(_) => "INT4 document summary native pass refused",
            Self::Summary(_) => "INT4 document summary evidence or output refused",
            Self::Execution(_) => "INT4 document summary reduction refused",
        })
    }
}
impl fmt::Debug for Int4CorpusSummaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Display::fmt(self, f) }
}
impl Error for Int4CorpusSummaryError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Map(e) => Some(e), Self::Source(e) => Some(e), Self::Summary(e) => Some(e),
            Self::Execution(e) => Some(e), _ => None }
    }
}
impl From<Int4SourceMapError> for Int4CorpusSummaryError { fn from(e: Int4SourceMapError) -> Self { Self::Map(e) } }
impl From<Int4SourceError> for Int4CorpusSummaryError { fn from(e: Int4SourceError) -> Self { Self::Source(e) } }
impl From<SummaryError> for Int4CorpusSummaryError { fn from(e: SummaryError) -> Self { Self::Summary(e) } }
impl Int4CorpusSummaryError {
    pub fn cancellation(&self) -> Option<DecodeCancellationKind> {
        match self {
            Self::Map(e) => e.cancellation(), Self::Source(e) => e.cancellation(),
            Self::Execution(ExecutionError::Task { source: CorpusSummaryError::Pass(e), .. }) => e.cancellation(),
            _ => None,
        }
    }
}
#[derive(Serialize)]
pub struct Int4CorpusSummaryRun {
    pub schema_version: u32,
    pub execution: &'static str,
    pub numerics_profile: &'static str,
    pub group_width: usize,
    pub chunk_profile: &'static str,
    pub semantics: &'static str,
    /// Includes every source chunk, including bullets omitted only at final top-k.
    pub source_span: VerifiedSourceSpan,
    pub map_batches: usize,
    pub reduce_calls: usize,
    pub reduction_levels: usize,
    pub planned_model_work: Int4Work,
    pub model_work: Int4Work,
    pub reserved_mask_node_visits: u64,
    pub mask_node_visit_charge: u64,
    /// The map's independent complete occurrence verification allowance.
    pub verification_used: SourceMapVerification,
    /// Additional independent evidence-union verification, cumulatively capped.
    pub verification_scan_steps: u64,
    pub summary: CorpusSummaryResult,
}
impl Int4SourceMapPreflight {
    pub fn verify_summary_completed(&self, run: &Int4CorpusSummaryRun) -> Result<(), Int4CorpusSummaryError> {
        let s = &run.summary;
        if self.task_spec != SUMMARIZE_TASK_VERSION || run.schema_version != 1
            || run.execution != INT4_CORPUS_SUMMARY_EXECUTION || run.numerics_profile != STRICT_INT4_PROFILE
            || run.group_width != self.group_width || run.chunk_profile != CHUNK_PROFILE || run.semantics != SUMMARY_SEMANTICS
            || run.source_span != self.source_span || run.map_batches == 0
            || run.planned_model_work != self.work || !fits(run.model_work, self.work)
            || run.reserved_mask_node_visits != self.masks || run.mask_node_visit_charge > self.masks
            || !run.verification_used.fits(self.verification) || s.schema_version != 1
            || s.task_spec_version != SUMMARIZE_TASK_VERSION || s.mapped_chunks != self.chunks
            || s.forward_positions != run.model_work.forward_positions || s.projected_logits != run.model_work.projected_logits
            || s.mask_node_visit_charge != run.mask_node_visit_charge || !(1..=1024).contains(&s.requested_max_bullets)
            || s.bullets.len() > s.requested_max_bullets || s.score_space != ScoreSpace::NotComputed
            || s.citation_guarantee != CitationGuarantee::StructuralSourceMembership
            || s.semantic_support != SummarySemanticSupport::NotAssessed {
            return Err(Int4CorpusSummaryError::Accounting);
        }
        Ok(())
    }
}
impl PreparedInt4SourceMap<'_> {
    pub fn check_summary(&self, limits: Int4SummaryLimits) -> Result<(), Int4CorpusSummaryError> {
        self.summary_options(limits).map(|_| ())
    }
    fn summary_options(&self, limits: Int4SummaryLimits) -> Result<SummaryOptions, Int4CorpusSummaryError> {
        limits.validate(self.limits)?;
        let mut options = None;
        for plan in &self.plans {
            let Finalizer::Summarize(value) = &plan.finalizer else { return Err(Int4CorpusSummaryError::InvalidLimits); };
            if options.is_some_and(|old| old != *value) || plan.execution_identity().task_spec != SUMMARIZE_TASK_VERSION {
                return Err(Int4CorpusSummaryError::InvalidLimits);
            }
            options = Some(*value);
        }
        options.ok_or(Int4CorpusSummaryError::InvalidLimits)
    }
    pub fn execute_summary_with_control<C: DecodeStepControl>(self, admitted: &[ExecutionIdentity],
        engine: &mut StrictInt4Engine<'_>, vocabulary: &ExtractionVocabulary,
        summary: Int4SummaryLimits, control: &mut C) -> Result<Int4CorpusSummaryRun, Int4CorpusSummaryError> {
        checkpoint(control)?;
        let limits = self.limits;
        self.execute_summary_with_driver(admitted, summary, NativeDriver { engine, vocabulary, control, limits })
    }
    fn execute_summary_with_driver<D: SourceDriver>(self, admitted: &[ExecutionIdentity],
        limits: Int4SummaryLimits, mut driver: D) -> Result<Int4CorpusSummaryRun, Int4CorpusSummaryError> {
        driver.checkpoint()?;
        let options = self.summary_options(limits)?;
        preflight(&self.plans, admitted, &driver)?;
        let driver = RefCell::new(driver);
        let pass = Pass { plans: &self.plans, admitted, driver: &driver, options,
            next_chunk: 0, work: Int4Work::default(), masks: 0, mask_cap: self.limits.mask_visits_per_chunk,
            remaining: self.limits.verification };
        let mut task = CorpusSummaryTask::new_int4(pass, limits.aggregation)?;
        let mut checkpoint_error = None;
        let reduced = mapreduce::execute(&self.chunks, &mut task, self.limits.reduction, || {
            match driver.borrow_mut().checkpoint() {
                Ok(()) => Ok(()),
                Err(error) => { checkpoint_error = Some(error); Err(MapReduceError::Cancelled) }
            }
        });
        if let Some(error) = checkpoint_error { return Err(error.into()); }
        let reduced = reduced.map_err(Int4CorpusSummaryError::Execution)?;
        driver.borrow_mut().checkpoint()?;
        let metadata = (reduced.root().source_span(), reduced.map_batches(), reduced.reduce_calls(), reduced.reduction_levels());
        let scan_steps = limits.aggregation.max_scan_steps.checked_sub(task.scan_steps_remaining())
            .ok_or(Int4CorpusSummaryError::Accounting)?;
        let pass = task.into_pass();
        if pass.next_chunk != self.plans.len() || reduced.root().value().mapped_chunks() != self.plans.len()
            || !fits(pass.work, self.expected.work) || pass.masks > self.expected.masks {
            return Err(Int4CorpusSummaryError::Accounting);
        }
        let summary = reduced.into_value().into_ranked(limits.max_bullets, limits.max_result_bytes)?;
        let run = Int4CorpusSummaryRun { schema_version: 1, execution: INT4_CORPUS_SUMMARY_EXECUTION,
            numerics_profile: STRICT_INT4_PROFILE, group_width: self.expected.group_width, chunk_profile: CHUNK_PROFILE,
            semantics: SUMMARY_SEMANTICS, source_span: metadata.0, map_batches: metadata.1, reduce_calls: metadata.2,
            reduction_levels: metadata.3, planned_model_work: self.expected.work, model_work: pass.work,
            reserved_mask_node_visits: self.expected.masks, mask_node_visit_charge: pass.masks,
            verification_used: SourceMapVerification::used(self.limits.verification, pass.remaining)?,
            verification_scan_steps: scan_steps, summary };
        self.expected.verify_summary_completed(&run)?;
        check_bytes(&run, limits.max_result_bytes)?;
        driver.borrow_mut().checkpoint()?;
        Ok(run)
    }
}
struct Pass<'a, D> {
    plans: &'a [PreparedInt4SourceTask],
    admitted: &'a [ExecutionIdentity],
    driver: &'a RefCell<D>,
    options: SummaryOptions,
    next_chunk: usize,
    work: Int4Work,
    masks: u64,
    mask_cap: u64,
    remaining: GroundingBudget,
}
impl<D: SourceDriver> SummaryPass for Pass<'_, D> {
    type Error = Int4SourceError;
    fn options(&self) -> SummaryOptions { self.options }
    fn run(&mut self, chunk: &SourceChunk<'_>) -> Result<SummaryResult, Self::Error> {
        let mut driver = self.driver.borrow_mut();
        driver.checkpoint()?;
        if !driver.clean() || chunk.id() != self.next_chunk { return Err(Int4SourceError::InvalidResult); }
        let plan = self.plans.get(chunk.id()).ok_or(Int4SourceError::InvalidResult)?;
        let identity = self.admitted.get(chunk.id()).ok_or(Int4SourceError::InvalidResult)?;
        let native = driver.run(plan, identity)?;
        check_run(plan, &native, self.mask_cap)?;
        if !driver.clean() { return Err(Int4SourceError::InvalidResult); }
        // Independently verify every local occurrence under the map's one
        // document allowance. The reducer separately verifies/lifts its own
        // complete evidence union; it receives no invented original offsets.
        drop(original_spans(chunk, &native.result, &mut self.remaining, &mut || driver.checkpoint())?);
        self.work = add_work(self.work, native.model_work).ok_or(Int4SourceError::InvalidResult)?;
        self.masks = self.masks.checked_add(mask_charge(&native.result)).ok_or(Int4SourceError::InvalidResult)?;
        let SourceTaskResult::Summarize(summary) = native.result else { return Err(Int4SourceError::InvalidResult); };
        self.next_chunk += 1;
        driver.checkpoint()?;
        Ok(summary)
    }
}
#[cfg(test)] mod tests;
