//! Complete document-wide keyphrase ranking over actual grouped-INT4 maps.
//! Every native candidate and occurrence survives the exact union or the whole
//! request fails its bounds. Ranking and top-k occur only at the complete root.
use super::*;
use super::execution::{SourceDriver, NativeDriver, check_run, mask_charge, original_spans, preflight};
use super::cohort::{Int4SourceMapCohortBudget, Int4SourceMapCohortStats, COHORT_ENVELOPE_BYTES};
use crate::native_engine::{int4::PackedInt4Kernel, strict_int4::Int4ModelWeights};
use crate::tasks::{
    corpus_keyphrases::{CorpusKeyphraseLimits, CorpusKeyphraseResult, CorpusKeyphraseTask,
        CorpusKeyphraseWarning, KeyphrasePass, CORPUS_KEYPHRASE_POLICY, check_bytes},
    keyphrases::KeyphraseResult,
    mapreduce::CHUNK_PROFILE,
};

pub const INT4_CORPUS_KEYPHRASE_EXECUTION: &str = "grouped-int4-keyphrase-map-exact-reduce-v1";
pub const INT4_CORPUS_KEYPHRASE_COHORT_EXECUTION: &str = "grouped-int4-keyphrase-cohorts-exact-reduce-v1";
pub const INT4_CORPUS_KEYPHRASE_SEMANTICS: &str = "exact-text-chunk-support-rank-no-neural-reranking-v1";

/// Complete-document union and final-output limits, independent of each map's
/// proposal count. A small final top-k never permits a truncated intermediate.
#[derive(Clone, Copy, Debug)]
pub struct Int4KeyphraseLimits {
    pub aggregation: CorpusKeyphraseLimits,
    pub max_phrases: usize,
    pub max_result_bytes: usize,
}
impl Default for Int4KeyphraseLimits {
    fn default() -> Self {
        Self { aggregation: CorpusKeyphraseLimits::default(), max_phrases: 16, max_result_bytes: 4 * 1024 * 1024 }
    }
}
impl Int4KeyphraseLimits {
    pub fn validate(self, mapping: Int4SourceMapLimits) -> Result<(), Int4CorpusKeyphraseError> {
        self.aggregation.validate()?;
        if !(1..=4096).contains(&self.max_phrases)
            || !(1..=64 * 1024 * 1024).contains(&self.max_result_bytes)
            || self.aggregation.max_value_bytes > mapping.reduction.max_value_bytes
            || self.max_result_bytes > mapping.reduction.max_result_bytes {
            return Err(Int4CorpusKeyphraseError::InvalidLimits);
        }
        Ok(())
    }
}

/// Default diagnostics omit source content and native compiler details.
pub enum Int4CorpusKeyphraseError {
    InvalidLimits, Accounting,
    Map(Int4SourceMapError), Source(Int4SourceError), Keyphrases(KeyphraseError),
    Execution(ExecutionError<KeyphraseError>),
}
impl fmt::Display for Int4CorpusKeyphraseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidLimits => "invalid INT4 document keyphrase limits or task",
            Self::Accounting => "INT4 document keyphrase complete accounting diverged",
            Self::Map(_) => "INT4 document keyphrase map admission refused",
            Self::Source(_) => "INT4 document keyphrase native pass refused",
            Self::Keyphrases(_) => "INT4 document keyphrase evidence or output refused",
            Self::Execution(_) => "INT4 document keyphrase reduction refused",
        })
    }
}
impl fmt::Debug for Int4CorpusKeyphraseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Display::fmt(self, f) }
}
impl Error for Int4CorpusKeyphraseError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Map(e) => Some(e), Self::Source(e) => Some(e), Self::Keyphrases(e) => Some(e),
            Self::Execution(e) => Some(e), _ => None }
    }
}
impl From<Int4SourceMapError> for Int4CorpusKeyphraseError { fn from(e: Int4SourceMapError) -> Self { Self::Map(e) } }
impl From<Int4SourceError> for Int4CorpusKeyphraseError { fn from(e: Int4SourceError) -> Self { Self::Source(e) } }
impl From<KeyphraseError> for Int4CorpusKeyphraseError { fn from(e: KeyphraseError) -> Self { Self::Keyphrases(e) } }
impl Int4CorpusKeyphraseError {
    pub fn cancellation(&self) -> Option<DecodeCancellationKind> {
        match self { Self::Map(e) => e.cancellation(), Self::Source(e) => e.cancellation(), _ => None }
    }
}

#[derive(Serialize)]
pub struct Int4CorpusKeyphraseRun {
    pub schema_version: u32,
    pub execution: &'static str,
    pub numerics_profile: &'static str,
    pub group_width: usize,
    pub chunk_profile: &'static str,
    pub semantics: &'static str,
    /// Entire original interval, including chunks without a final selected phrase.
    pub source_span: VerifiedSourceSpan,
    pub map_batches: usize,
    pub reduce_calls: usize,
    pub reduction_levels: usize,
    pub requested_max_phrases: usize,
    pub planned_model_work: Int4Work,
    pub model_work: Int4Work,
    pub reserved_mask_node_visits: u64,
    pub mask_node_visit_charge: u64,
    /// Independent complete occurrence checks under the map's one document cap.
    pub verification_used: SourceMapVerification,
    /// Additional independent corpus matching work, capped across every map.
    pub verification_scan_work: u64,
    pub keyphrases: CorpusKeyphraseResult,
}
#[derive(Serialize)]
pub struct Int4CorpusKeyphraseCohortRun {
    pub schema_version: u32,
    pub execution: &'static str,
    /// Original complete union and ranking, with physical map_batches.
    pub document: Int4CorpusKeyphraseRun,
    pub cohorts: Int4SourceMapCohortStats,
}
impl Int4SourceMapPreflight {
    pub fn verify_keyphrases_completed(&self, run: &Int4CorpusKeyphraseRun) -> Result<(), Int4CorpusKeyphraseError> {
        let result = &run.keyphrases;
        if self.task_spec != KEYPHRASES_TASK_VERSION || run.schema_version != 1
            || run.execution != INT4_CORPUS_KEYPHRASE_EXECUTION || run.numerics_profile != STRICT_INT4_PROFILE
            || run.group_width != self.group_width || run.chunk_profile != CHUNK_PROFILE
            || run.semantics != INT4_CORPUS_KEYPHRASE_SEMANTICS || run.source_span != self.source_span
            || run.map_batches == 0 || run.map_batches > self.chunks
            || run.planned_model_work != self.work || !fits(run.model_work, self.work)
            || run.reserved_mask_node_visits != self.masks || run.mask_node_visit_charge > self.masks
            || !run.verification_used.fits(self.verification) || result.schema_version != 1
            || result.task_spec_version != KEYPHRASES_TASK_VERSION || result.mapped_chunks != self.chunks
            || result.forward_positions != run.model_work.forward_positions
            || result.projected_logits != run.model_work.projected_logits
            || result.mask_node_visit_charge != run.mask_node_visit_charge
            || !(1..=4096).contains(&run.requested_max_phrases) || result.phrases.len() > run.requested_max_phrases
            || (result.omitted_candidates > 0 && result.phrases.len() != run.requested_max_phrases)
            || result.phrases.len().checked_add(result.omitted_candidates)
                .is_none_or(|count| count > run.verification_used.fields)
            || result.ranking_policy != CORPUS_KEYPHRASE_POLICY || result.score_space != ScoreSpace::NotComputed
            || result.grounding != ExtractionGrounding::SourceMembership
            || result.warnings != [CorpusKeyphraseWarning::ChunkBoundariesMaySplitPhrases,
                CorpusKeyphraseWarning::SingleContextEquivalenceNotEstablished,
                CorpusKeyphraseWarning::ModelSelectionIsNotRecallGuarantee] {
            return Err(Int4CorpusKeyphraseError::Accounting);
        }
        Ok(())
    }
}
impl PreparedInt4SourceMap<'_> {
    /// Check every compiled task and its options before any native forward.
    pub fn check_keyphrases(&self, limits: Int4KeyphraseLimits) -> Result<(), Int4CorpusKeyphraseError> {
        self.keyphrase_options(limits).map(|_| ())
    }
    fn keyphrase_options(&self, limits: Int4KeyphraseLimits) -> Result<KeyphraseOptions, Int4CorpusKeyphraseError> {
        limits.validate(self.limits)?;
        let mut options = None;
        for plan in &self.plans {
            let Finalizer::Keyphrases(value) = &plan.finalizer else { return Err(Int4CorpusKeyphraseError::InvalidLimits); };
            if options.is_some_and(|old| old != *value) || plan.execution_identity().task_spec != KEYPHRASES_TASK_VERSION {
                return Err(Int4CorpusKeyphraseError::InvalidLimits);
            }
            options = Some(*value);
        }
        options.ok_or(Int4CorpusKeyphraseError::InvalidLimits)
    }
    /// Consume the complete prepared map using one real admitted native engine
    /// and the caller's existing control. This accepts no injected map results.
    pub fn execute_keyphrases_with_control<C: DecodeStepControl>(self, admitted: &[ExecutionIdentity],
        engine: &mut StrictInt4Engine<'_>, vocabulary: &ExtractionVocabulary,
        limits: Int4KeyphraseLimits, control: &mut C) -> Result<Int4CorpusKeyphraseRun, Int4CorpusKeyphraseError> {
        checkpoint(control)?;
        let mapping = self.limits;
        self.execute_keyphrases_with_driver(admitted, limits, NativeDriver { engine, vocabulary, control, limits: mapping })
    }
    /// Execute complete native chunk cohorts into the same exact phrase union.
    /// Original per-chunk ranks and every occurrence are verified before one
    /// final document-wide ranking; no intermediate candidate is discarded.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_keyphrases_cohorts_with_control<C: DecodeStepControl>(self, admitted: &[ExecutionIdentity],
        weights: &Int4ModelWeights, vocabulary: &ExtractionVocabulary, limits: Int4KeyphraseLimits,
        budget: Int4SourceMapCohortBudget, kernel: PackedInt4Kernel, control: &mut C)
        -> Result<Int4CorpusKeyphraseCohortRun, Int4CorpusKeyphraseError> {
        checkpoint(control)?;
        self.check_keyphrases(limits)?;
        let mut cohorts = Int4SourceMapCohortStats::default();
        let driver = self.aggregate_driver(admitted, weights, vocabulary, budget, kernel, &mut cohorts, control)?;
        let document = self.execute_keyphrases_with_driver(admitted, limits, driver)?;
        finish_cohorts(document, cohorts, limits.max_result_bytes, control)
    }
    // Private scripted seam, sharing the actual native receipt and occurrence
    // validators. It is not an alternate public execution backend.
    pub(super) fn execute_keyphrases_with_driver<D: SourceDriver>(self, admitted: &[ExecutionIdentity],
        limits: Int4KeyphraseLimits, mut driver: D) -> Result<Int4CorpusKeyphraseRun, Int4CorpusKeyphraseError> {
        driver.checkpoint()?;
        let options = self.keyphrase_options(limits)?;
        preflight(&self.plans, admitted, &driver)?;
        let driver = RefCell::new(driver);
        let pass = Pass { plans: &self.plans, admitted, driver: &driver, options,
            next_chunk: 0, work: Int4Work::default(), masks: 0, mask_cap: self.limits.mask_visits_per_chunk,
            remaining: self.limits.verification, native_error: None };
        let mut task = CorpusKeyphraseTask::new_int4(pass, limits.aggregation)?;
        let mut checkpoint_error = None;
        let reduced = mapreduce::execute(&self.chunks, &mut task, self.limits.reduction, || {
            match driver.borrow_mut().checkpoint() {
                Ok(()) => Ok(()),
                Err(error) => { checkpoint_error = Some(error); Err(MapReduceError::Cancelled) }
            }
        });
        let scan_work = limits.aggregation.max_scan_work.checked_sub(task.scan_work_remaining())
            .ok_or(Int4CorpusKeyphraseError::Accounting)?;
        let pass = task.into_pass();
        if let Some(error) = checkpoint_error { return Err(error.into()); }
        // KeyphrasePass has a fixed legacy error type. Preserve the typed native
        // cancellation/failure in this private consumed pass without rewriting
        // the native receipt or weakening the corpus reducer's profile checks.
        if let Some(error) = pass.native_error { return Err(error.into()); }
        let reduced = reduced.map_err(Int4CorpusKeyphraseError::Execution)?;
        driver.borrow_mut().checkpoint()?;
        let metadata = (reduced.root().source_span(), reduced.map_batches(), reduced.reduce_calls(), reduced.reduction_levels());
        if pass.next_chunk != self.plans.len() || reduced.root().value().mapped_chunks() != self.plans.len()
            || !fits(pass.work, self.expected.work) || pass.masks > self.expected.masks || !driver.borrow().clean() {
            return Err(Int4CorpusKeyphraseError::Accounting);
        }
        let keyphrases = reduced.into_value().into_ranked(limits.max_phrases, limits.max_result_bytes)?;
        let run = Int4CorpusKeyphraseRun { schema_version: 1, execution: INT4_CORPUS_KEYPHRASE_EXECUTION,
            numerics_profile: STRICT_INT4_PROFILE, group_width: self.expected.group_width, chunk_profile: CHUNK_PROFILE,
            semantics: INT4_CORPUS_KEYPHRASE_SEMANTICS, source_span: metadata.0, map_batches: metadata.1,
            reduce_calls: metadata.2, reduction_levels: metadata.3, requested_max_phrases: limits.max_phrases,
            planned_model_work: self.expected.work, model_work: pass.work,
            reserved_mask_node_visits: self.expected.masks, mask_node_visit_charge: pass.masks,
            verification_used: SourceMapVerification::used(self.limits.verification, pass.remaining)?,
            verification_scan_work: scan_work, keyphrases };
        self.expected.verify_keyphrases_completed(&run)?;
        // The output cap includes the complete native/map/verification envelope.
        check_bytes(&run, limits.max_result_bytes)?;
        driver.borrow_mut().checkpoint()?;
        Ok(run)
    }
}

pub(super) fn finish_cohorts<C: DecodeStepControl>(document: Int4CorpusKeyphraseRun,
    cohorts: Int4SourceMapCohortStats, max_result_bytes: usize, control: &mut C)
    -> Result<Int4CorpusKeyphraseCohortRun, Int4CorpusKeyphraseError> {
    cohort::check_aggregate_stats(cohorts, document.keyphrases.mapped_chunks, document.map_batches,
        document.planned_model_work, document.model_work)?;
    let run = Int4CorpusKeyphraseCohortRun { schema_version: 1, execution: INT4_CORPUS_KEYPHRASE_COHORT_EXECUTION,
        document, cohorts };
    let cap = max_result_bytes.checked_add(COHORT_ENVELOPE_BYTES as usize).ok_or(Int4CorpusKeyphraseError::Accounting)?;
    check_bytes(&run, cap)?;
    checkpoint(control)?;
    Ok(run)
}

struct Pass<'a, D> {
    plans: &'a [PreparedInt4SourceTask],
    admitted: &'a [ExecutionIdentity],
    driver: &'a RefCell<D>,
    options: KeyphraseOptions,
    next_chunk: usize,
    work: Int4Work,
    masks: u64,
    mask_cap: u64,
    remaining: GroundingBudget,
    native_error: Option<Int4SourceError>,
}
impl<D: SourceDriver> Pass<'_, D> {
    fn finish(&mut self, chunk: &SourceChunk<'_>, native: Int4SourceTaskRun, driver: &mut D)
        -> Result<KeyphraseResult, Int4SourceError> {
        if chunk.id() != self.next_chunk { return Err(Int4SourceError::InvalidResult); }
        let plan = self.plans.get(chunk.id()).ok_or(Int4SourceError::InvalidResult)?;
        check_run(plan, &native, self.mask_cap)?;
        if !driver.clean() { return Err(Int4SourceError::InvalidResult); }
        // Verify every local occurrence independently using the original map's
        // cumulative field/match/scan budget, including candidates later omitted.
        // The corpus reducer then performs its own bounded complete scan/lifting.
        drop(original_spans(chunk, &native.result, &mut self.remaining, &mut || driver.checkpoint())?);
        self.work = add_work(self.work, native.model_work).ok_or(Int4SourceError::InvalidResult)?;
        self.masks = self.masks.checked_add(mask_charge(&native.result)).ok_or(Int4SourceError::InvalidResult)?;
        let SourceTaskResult::Keyphrases(keyphrases) = native.result else { return Err(Int4SourceError::InvalidResult); };
        self.next_chunk += 1;
        driver.checkpoint()?;
        Ok(keyphrases)
    }
    fn run_native(&mut self, chunk: &SourceChunk<'_>) -> Result<KeyphraseResult, Int4SourceError> {
        let driver = self.driver;
        let mut driver = driver.borrow_mut();
        driver.checkpoint()?;
        if !driver.clean() || chunk.id() != self.next_chunk { return Err(Int4SourceError::InvalidResult); }
        let plan = self.plans.get(chunk.id()).ok_or(Int4SourceError::InvalidResult)?;
        let identity = self.admitted.get(chunk.id()).ok_or(Int4SourceError::InvalidResult)?;
        let native = driver.run(plan, identity)?;
        self.finish(chunk, native, &mut *driver)
    }
    fn run_native_batch(&mut self, chunks: &[SourceChunk<'_>])
        -> Result<Option<Vec<MapOutput<KeyphraseResult>>>, Int4SourceError> {
        let end = self.next_chunk.checked_add(chunks.len()).ok_or(Int4SourceError::InvalidResult)?;
        if chunks.is_empty() || chunks.iter().enumerate().any(|(index, chunk)| chunk.id() != self.next_chunk + index) {
            return Err(Int4SourceError::InvalidResult);
        }
        let driver = self.driver;
        let mut driver = driver.borrow_mut();
        if !driver.clean() { return Err(Int4SourceError::InvalidResult); }
        let plans = self.plans.get(self.next_chunk..end).ok_or(Int4SourceError::InvalidResult)?;
        let admitted = self.admitted.get(self.next_chunk..end).ok_or(Int4SourceError::InvalidResult)?;
        let Some(rows) = driver.run_batch(plans, admitted)? else { return Ok(None); };
        if rows.len() != chunks.len() { return Err(Int4SourceError::InvalidResult); }
        let mut output = Vec::new(); output.try_reserve_exact(chunks.len()).map_err(|_| SourcePlanningError::AllocationRefused)?;
        for (chunk, native) in chunks.iter().zip(rows) {
            driver.checkpoint()?;
            output.push(MapOutput { chunk_id: chunk.id(), value: self.finish(chunk, native, &mut *driver)? });
        }
        Ok(Some(output))
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
    fn run_batch(&mut self, chunks: &[SourceChunk<'_>])
        -> Result<Option<Vec<MapOutput<KeyphraseResult>>>, KeyphraseError> {
        if self.native_error.is_some() { return Err(KeyphraseError::InvalidResult); }
        match self.run_native_batch(chunks) {
            Ok(result) => Ok(result),
            Err(error) => { self.native_error = Some(error); Err(KeyphraseError::InvalidResult) }
        }
    }
}

#[cfg(test)] mod tests;
