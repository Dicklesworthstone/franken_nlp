//! Complete document-wide keyphrase ranking over the real INT8 map plans.
//!
//! The existing corpus reducer verifies every local occurrence, lifts original
//! coordinates and preserves all candidates through every merge. Only the
//! completed root is ranked. This is not neural reranking or a recall guarantee.
use super::*;
use std::cell::RefCell;
use crate::tasks::{
    corpus_keyphrases::{CorpusKeyphraseLimits, CorpusKeyphraseResult, CorpusKeyphraseTask,
        KeyphrasePass, check_bytes},
    mapreduce::CHUNK_PROFILE,
};

pub const INT8_CORPUS_KEYPHRASE_EXECUTION: &str = "portable-int8-keyphrase-map-exact-reduce-v1";
pub const INT8_CORPUS_KEYPHRASE_SEMANTICS: &str = "exact-text-chunk-support-rank-no-neural-reranking-v1";

/// Complete-document limits, independent of each native chunk's options.
/// A small final top-k never permits truncating an oversized complete union.
#[derive(Clone, Copy, Debug)]
pub struct Int8KeyphraseLimits {
    pub aggregation: CorpusKeyphraseLimits,
    pub max_phrases: usize,
    pub max_result_bytes: usize,
}
impl Default for Int8KeyphraseLimits {
    fn default() -> Self {
        Self { aggregation: CorpusKeyphraseLimits::default(), max_phrases: 16, max_result_bytes: 4 * 1024 * 1024 }
    }
}
impl Int8KeyphraseLimits {
    pub fn validate(self, mapping: Int8SourceMapLimits) -> Result<(), Int8CorpusKeyphraseError> {
        self.aggregation.validate()?;
        if !(1..=4096).contains(&self.max_phrases)
            || !(1..=64 * 1024 * 1024).contains(&self.max_result_bytes)
            || self.aggregation.max_value_bytes > mapping.reduction.max_value_bytes
            || self.max_result_bytes > mapping.reduction.max_result_bytes {
            return Err(Int8CorpusKeyphraseError::InvalidLimits);
        }
        Ok(())
    }
}

pub enum Int8CorpusKeyphraseError {
    InvalidLimits, Accounting,
    Map(Int8SourceMapError), Source(Int8SourceError), Keyphrases(KeyphraseError),
    Execution(ExecutionError<KeyphraseError>),
}
impl fmt::Display for Int8CorpusKeyphraseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidLimits => "invalid int8 document keyphrase limits or task",
            Self::Accounting => "int8 document keyphrase complete accounting diverged",
            Self::Map(_) => "int8 document keyphrase map admission refused",
            Self::Source(_) => "int8 document keyphrase native pass refused",
            Self::Keyphrases(_) => "int8 document keyphrase evidence or output refused",
            Self::Execution(_) => "int8 document keyphrase reduction refused",
        })
    }
}
impl fmt::Debug for Int8CorpusKeyphraseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Display::fmt(self, f) }
}
impl Error for Int8CorpusKeyphraseError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Map(e) => Some(e), Self::Source(e) => Some(e),
            Self::Keyphrases(e) => Some(e), Self::Execution(e) => Some(e), _ => None }
    }
}
impl From<Int8SourceMapError> for Int8CorpusKeyphraseError { fn from(e: Int8SourceMapError) -> Self { Self::Map(e) } }
impl From<Int8SourceError> for Int8CorpusKeyphraseError { fn from(e: Int8SourceError) -> Self { Self::Source(e) } }
impl From<KeyphraseError> for Int8CorpusKeyphraseError { fn from(e: KeyphraseError) -> Self { Self::Keyphrases(e) } }
impl Int8CorpusKeyphraseError {
    pub fn cancellation(&self) -> Option<DecodeCancellationKind> {
        match self { Self::Map(e) => e.cancellation(), Self::Source(e) => e.cancellation(), _ => None }
    }
}

#[derive(Serialize)]
pub struct Int8CorpusKeyphraseRun {
    pub schema_version: u32,
    pub execution: &'static str,
    pub numerics_profile: &'static str,
    pub chunk_profile: &'static str,
    pub semantics: &'static str,
    /// Entire mapped interval, including chunks with no selected final phrase.
    pub source_span: VerifiedSourceSpan,
    pub map_batches: usize,
    pub reduce_calls: usize,
    pub reduction_levels: usize,
    pub requested_max_phrases: usize,
    pub planned_model_work: Int8Work,
    pub model_work: Int8Work,
    pub reserved_mask_node_visits: u64,
    pub mask_node_visit_charge: u64,
    pub verification_scan_work: u64,
    pub keyphrases: CorpusKeyphraseResult,
}

impl PreparedInt8SourceMap<'_> {
    /// Refuse mixed tasks/options before any native call. This inspects the
    /// actual compiled plans, not just the caller's task name or first chunk.
    pub fn check_keyphrases(&self, limits: Int8KeyphraseLimits) -> Result<(), Int8CorpusKeyphraseError> {
        self.keyphrase_options(limits).map(|_| ())
    }
    fn keyphrase_options(&self, limits: Int8KeyphraseLimits) -> Result<KeyphraseOptions, Int8CorpusKeyphraseError> {
        limits.validate(self.limits)?;
        let mut options = None;
        for plan in &self.plans {
            let Finalizer::Keyphrases(value) = &plan.finalizer else { return Err(Int8CorpusKeyphraseError::InvalidLimits); };
            if options.is_some_and(|old| old != *value) || plan.execution_identity().task_spec != KEYPHRASES_TASK_VERSION {
                return Err(Int8CorpusKeyphraseError::InvalidLimits);
            }
            options = Some(*value);
        }
        options.ok_or(Int8CorpusKeyphraseError::InvalidLimits)
    }

    /// Consume one whole-document commitment and use one admitted native engine
    /// and controller. No caller-supplied collection of inference receipts can
    /// enter this public path; all actual identities are checked before forward.
    pub fn execute_keyphrases_with_control<C: DecodeStepControl>(self, admitted: &[ExecutionIdentity],
        engine: &mut StrictInt8Engine<'_>, vocabulary: &ExtractionVocabulary,
        limits: Int8KeyphraseLimits, control: &mut C) -> Result<Int8CorpusKeyphraseRun, Int8CorpusKeyphraseError> {
        checkpoint(control)?;
        self.check_keyphrases(limits)?;
        self.preflight(admitted, engine)?;
        let driver = NativeDriver { engine, vocabulary, control, limits: self.limits };
        self.execute_keyphrases_with_driver(admitted, limits, driver)
    }

    // Private scripted-test seam. It uses the same receipt and evidence checks
    // as production; it cannot substitute a public fabricated-result backend.
    fn execute_keyphrases_with_driver<D: SourceDriver>(self, admitted: &[ExecutionIdentity],
        limits: Int8KeyphraseLimits, driver: D) -> Result<Int8CorpusKeyphraseRun, Int8CorpusKeyphraseError> {
        let options = self.keyphrase_options(limits)?;
        verify_identities(&self.plans, admitted)?;
        let driver = RefCell::new(driver);
        let pass = Pass { plans: &self.plans, admitted, driver: &driver, options,
            next_chunk: 0, work: Int8Work::default(), masks: 0,
            mask_cap: self.limits.mask_visits_per_chunk, native_error: None };
        let mut task = CorpusKeyphraseTask::new_int8(pass, limits.aggregation)?;
        let mut checkpoint_error = None;
        // Coordinator and native calls borrow the same control sequentially.
        let reduced = mapreduce::execute(&self.chunks, &mut task, self.limits.reduction, || {
            match driver.borrow_mut().checkpoint() {
                Ok(()) => Ok(()),
                Err(error) => { checkpoint_error = Some(error); Err(MapReduceError::Cancelled) }
            }
        });
        let scan_work = limits.aggregation.max_scan_work.checked_sub(task.scan_work_remaining())
            .ok_or(Int8CorpusKeyphraseError::Accounting)?;
        let pass = task.into_pass();
        if let Some(error) = checkpoint_error { return Err(error.into()); }
        // KeyphrasePass predates the INT8 host and has a fixed error type. Keep
        // the original typed native cause, including cancellation, out-of-band
        // inside this private consumed pass instead of stringifying or losing it.
        if let Some(error) = pass.native_error { return Err(error.into()); }
        let reduced = reduced.map_err(Int8CorpusKeyphraseError::Execution)?;
        driver.borrow_mut().checkpoint()?;
        let metadata = (reduced.root().source_span(), reduced.map_batches(), reduced.reduce_calls(), reduced.reduction_levels());
        if pass.next_chunk != self.plans.len() || reduced.root().value().mapped_chunks() != self.plans.len()
            || !within(pass.work, self.work) || pass.masks > self.masks {
            return Err(Int8CorpusKeyphraseError::Accounting);
        }
        let keyphrases = reduced.into_value().into_ranked(limits.max_phrases, limits.max_result_bytes)?;
        if keyphrases.forward_positions != pass.work.forward_positions || keyphrases.projected_logits != pass.work.projected_logits
            || keyphrases.mask_node_visit_charge != pass.masks { return Err(Int8CorpusKeyphraseError::Accounting); }
        let output = Int8CorpusKeyphraseRun { schema_version: 1, execution: INT8_CORPUS_KEYPHRASE_EXECUTION,
            numerics_profile: STRICT_INT8_PROFILE, chunk_profile: CHUNK_PROFILE, semantics: INT8_CORPUS_KEYPHRASE_SEMANTICS,
            source_span: metadata.0, map_batches: metadata.1, reduce_calls: metadata.2, reduction_levels: metadata.3,
            requested_max_phrases: limits.max_phrases, planned_model_work: self.work, model_work: pass.work,
            reserved_mask_node_visits: self.masks, mask_node_visit_charge: pass.masks,
            verification_scan_work: scan_work, keyphrases };
        check_bytes(&output, limits.max_result_bytes)?;
        driver.borrow_mut().checkpoint()?;
        Ok(output)
    }
}

struct Pass<'a, D> {
    plans: &'a [PreparedInt8SourceTask], admitted: &'a [ExecutionIdentity], driver: &'a RefCell<D>,
    options: KeyphraseOptions, next_chunk: usize, work: Int8Work, masks: u64, mask_cap: u64,
    native_error: Option<Int8SourceError>,
}
impl<D: SourceDriver> Pass<'_, D> {
    fn run_native(&mut self, chunk: &SourceChunk<'_>) -> Result<KeyphraseResult, Int8SourceError> {
        let mut driver = self.driver.borrow_mut();
        driver.checkpoint()?;
        if chunk.id() != self.next_chunk { return Err(Int8SourceError::InvalidResult); }
        let plan = self.plans.get(chunk.id()).ok_or(Int8SourceError::InvalidResult)?;
        let admitted = self.admitted.get(chunk.id()).ok_or(Int8SourceError::InvalidResult)?;
        let result = driver.run(plan, admitted)?;
        check_run(plan, &result, self.mask_cap)?;
        self.work = add_work(self.work, result.model_work).ok_or(Int8SourceError::InvalidResult)?;
        self.masks = self.masks.checked_add(mask_charge(&result.result)?).ok_or(Int8SourceError::InvalidResult)?;
        let SourceTaskResult::Keyphrases(keyphrases) = result.result else { return Err(Int8SourceError::InvalidResult); };
        self.next_chunk += 1;
        driver.checkpoint()?;
        Ok(keyphrases)
    }
}
impl<D: SourceDriver> KeyphrasePass for Pass<'_, D> {
    fn options(&self) -> KeyphraseOptions { self.options }
    fn run(&mut self, chunk: &SourceChunk<'_>) -> Result<KeyphraseResult, KeyphraseError> {
        if self.native_error.is_some() { return Err(KeyphraseError::InvalidResult); }
        match self.run_native(chunk) {
            Ok(result) => Ok(result),
            Err(error) => { self.native_error = Some(error); Err(KeyphraseError::InvalidResult) }
        }
    }
}

#[cfg(test)] mod tests;
