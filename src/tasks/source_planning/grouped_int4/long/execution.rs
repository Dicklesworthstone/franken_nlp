//! One native engine, complete preflight and independently verified lineage.
use super::*;
use crate::{native_engine::lmhead::NANBEIGE_VOCAB_SIZE,
    tasks::{keyphrases::KEYPHRASES_RANKING,
        summarize::{CitationGuarantee, SummarySemanticSupport}}};

pub(super) trait SourceDriver {
    fn capacity(&self) -> usize;
    /// Resident row capacity. Serial engines reserve their full fixed context;
    /// exact-sized cohort rows may retain only the sealed row's own context.
    fn context_for(&self, _: &PreparedInt4SourceTask) -> Result<usize, Int4SourceError> { Ok(self.capacity()) }
    fn group_width(&self) -> usize;
    fn clean(&self) -> bool;
    fn check_model(&self, identity: &ExecutionIdentity) -> Result<(), Int4SourceError>;
    fn checkpoint(&mut self) -> Result<(), Int4SourceError>;
    fn run(&mut self, plan: &PreparedInt4SourceTask, identity: &ExecutionIdentity)
        -> Result<Int4SourceTaskRun, Int4SourceError>;
    /// None performs no work. Existing serial passes preserve their original
    /// native-row -> independent-verification -> next-row failure ordering.
    fn run_batch(&mut self, _: &[PreparedInt4SourceTask], _: &[ExecutionIdentity])
        -> Result<Option<Vec<Int4SourceTaskRun>>, Int4SourceError> { Ok(None) }
}
pub(super) struct NativeDriver<'e, 'w, 'v, 'c, C> {
    pub(super) engine: &'e mut StrictInt4Engine<'w>,
    pub(super) vocabulary: &'v ExtractionVocabulary,
    pub(super) control: &'c mut C,
    pub(super) limits: Int4SourceMapLimits,
}
impl<C: DecodeStepControl> SourceDriver for NativeDriver<'_, '_, '_, '_, C> {
    fn capacity(&self) -> usize { self.engine.kv_cache().capacity_positions() }
    fn group_width(&self) -> usize { self.engine.model_identity().group_width() }
    fn clean(&self) -> bool { !self.engine.is_poisoned() && self.engine.kv_cache().all_slots_have_len(0) }
    fn check_model(&self, identity: &ExecutionIdentity) -> Result<(), Int4SourceError> {
        constrained_int4::check_model(identity, self.engine.model_identity()).map_err(Into::into)
    }
    fn checkpoint(&mut self) -> Result<(), Int4SourceError> { checkpoint(self.control) }
    fn run(&mut self, plan: &PreparedInt4SourceTask, identity: &ExecutionIdentity)
        -> Result<Int4SourceTaskRun, Int4SourceError> {
        if !self.clean() { return Err(Int4SourceError::InvalidResult); }
        let work = plan.planned_work();
        let result = plan.execute_with_control(identity, self.engine, self.vocabulary, Int4JsonBudget {
            native: Int4RunBudget::exact(work), json: JsonWorkBudget {
                max_forward_positions: work.forward_positions, max_projected_logits: work.projected_logits,
                max_kv_bytes: plan.max_kv_bytes(), max_total_mask_node_visits: self.limits.mask_visits_per_chunk,
                mask_limits: self.limits.mask_limits,
            },
        }, self.control)?;
        if !self.clean() { return Err(Int4SourceError::InvalidResult); }
        Ok(result)
    }
}

pub(super) fn preflight(plans: &[PreparedInt4SourceTask], admitted: &[ExecutionIdentity],
    driver: &impl SourceDriver) -> Result<(), Int4SourceMapError> {
    if plans.is_empty() || plans.len() != admitted.len() || plans.len() > 256 || !driver.clean() {
        return Err(Int4SourceMapError::Admission);
    }
    let width = driver.group_width();
    for (plan, identity) in plans.iter().zip(admitted) {
        plan.verify_identity(identity)?;
        driver.check_model(identity)?;
        let context = driver.context_for(plan)?;
        let required = Int4MemoryRequirement::for_context(context, width)
            .map_err(|_| Int4SourceMapError::Admission)?;
        if context > driver.capacity() || plan.group_width() != width || plan.planned_work().forward_positions > context as u64
            || required.kv_bytes > plan.max_kv_bytes() || required.kv_bytes > plan.task_budget().max_kv_bytes {
            return Err(Int4SourceMapError::Admission);
        }
    }
    Ok(())
}
impl PreparedInt4SourceMap<'_> {
    pub fn preflight(&self, admitted: &[ExecutionIdentity], engine: &StrictInt4Engine<'_>)
        -> Result<(), Int4SourceMapError> {
        if self.plans.is_empty() || self.plans.len() != admitted.len()
            || engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) {
            return Err(Int4SourceMapError::Admission);
        }
        let width = engine.model_identity().group_width();
        let capacity = engine.kv_cache().capacity_positions();
        let required = Int4MemoryRequirement::for_context(capacity, width).map_err(|_| Int4SourceMapError::Admission)?;
        for (plan, identity) in self.plans.iter().zip(admitted) {
            plan.verify_identity(identity)?;
            constrained_int4::check_model(identity, engine.model_identity()).map_err(Int4SourceError::from)?;
            if plan.group_width() != width || plan.planned_work().forward_positions > capacity as u64
                || required.kv_bytes > plan.max_kv_bytes() || required.kv_bytes > plan.task_budget().max_kv_bytes {
                return Err(Int4SourceMapError::Admission);
            }
        }
        Ok(())
    }
    pub fn execute_with_control<C: DecodeStepControl>(self, admitted: &[ExecutionIdentity],
        engine: &mut StrictInt4Engine<'_>, vocabulary: &ExtractionVocabulary, control: &mut C)
        -> Result<Int4SourceMapRun, Int4SourceMapError> {
        checkpoint(control)?;
        let limits = self.limits;
        self.execute_with_driver(admitted, NativeDriver { engine, vocabulary, control, limits })
    }
    pub(super) fn execute_with_driver<D: SourceDriver>(self, admitted: &[ExecutionIdentity], mut driver: D)
        -> Result<Int4SourceMapRun, Int4SourceMapError> {
        driver.checkpoint()?;
        preflight(&self.plans, admitted, &driver)?;
        let driver = RefCell::new(driver);
        let mut task = Executor { plans: &self.plans, admitted, driver: &driver, next_chunk: 0,
            work: Int4Work::default(), masks: 0, mask_cap: self.limits.mask_visits_per_chunk,
            remaining: self.limits.verification };
        let mut checkpoint_error = None;
        let mapped = mapreduce::execute(&self.chunks, &mut task, self.limits.reduction, || {
            match driver.borrow_mut().checkpoint() {
                Ok(()) => Ok(()),
                Err(error) => { checkpoint_error = Some(error); Err(MapReduceError::Cancelled) }
            }
        });
        if let Some(error) = checkpoint_error { return Err(error.into()); }
        let mapped = mapped.map_err(Int4SourceMapError::Reduction)?;
        driver.borrow_mut().checkpoint()?;
        if task.next_chunk != self.plans.len() { return Err(Int4SourceError::InvalidResult.into()); }
        let run = complete(self.expected, self.limits.verification, task.remaining,
            task.work, task.masks, mapped, self.max_result_bytes())?;
        driver.borrow_mut().checkpoint()?;
        Ok(run)
    }
}

struct Executor<'a, D> {
    plans: &'a [PreparedInt4SourceTask],
    admitted: &'a [ExecutionIdentity],
    driver: &'a RefCell<D>,
    next_chunk: usize,
    work: Int4Work,
    masks: u64,
    mask_cap: u64,
    remaining: GroundingBudget,
}
impl<D: SourceDriver> MapReduceTask for Executor<'_, D> {
    type Value = SourceMapValue;
    type Error = Int4SourceError;
    fn policy(&self) -> ReductionPolicy { policy() }
    fn map_batch(&mut self, chunks: &[SourceChunk<'_>]) -> Result<Vec<MapOutput<Self::Value>>, Self::Error> {
        let mut outputs = Vec::new();
        outputs.try_reserve_exact(chunks.len()).map_err(|_| SourcePlanningError::AllocationRefused)?;
        for chunk in chunks {
            let mut driver = self.driver.borrow_mut();
            driver.checkpoint()?;
            if !driver.clean() || chunk.id() != self.next_chunk { return Err(Int4SourceError::InvalidResult); }
            let plan = self.plans.get(chunk.id()).ok_or(Int4SourceError::InvalidResult)?;
            let identity = self.admitted.get(chunk.id()).ok_or(Int4SourceError::InvalidResult)?;
            let native = driver.run(plan, identity)?;
            if !driver.clean() { return Err(Int4SourceError::InvalidResult); }
            self.work = add_work(self.work, native.model_work).ok_or(Int4SourceError::InvalidResult)?;
            self.masks = self.masks.checked_add(mask_charge(&native.result)).ok_or(Int4SourceError::InvalidResult)?;
            outputs.push(map_output(chunk, plan, native, self.mask_cap,
                &mut self.remaining, &mut || driver.checkpoint())?);
            self.next_chunk += 1;
            driver.checkpoint()?;
        }
        Ok(outputs)
    }
    fn reduce(&mut self, input: ReduceInput<'_, Self::Value>) -> Result<Self::Value, Self::Error> {
        let mut driver = self.driver.borrow_mut();
        reduce(input, &mut || driver.checkpoint())
    }
}

pub(super) fn policy() -> ReductionPolicy {
    ReductionPolicy { id: "lossless-source-order-chunk-results-v1", may_discard_information: false }
}
pub(super) fn map_output(chunk: &SourceChunk<'_>, plan: &PreparedInt4SourceTask,
    native: Int4SourceTaskRun, mask_cap: u64, remaining: &mut GroundingBudget,
    checkpoint: &mut impl FnMut() -> Result<(), Int4SourceError>)
    -> Result<MapOutput<SourceMapValue>, Int4SourceError> {
    check_run(plan, &native, mask_cap)?;
    let original_spans = original_spans(chunk, &native.result, remaining, checkpoint)?;
    Ok(MapOutput { chunk_id: chunk.id(), value: SourceMapValue {
        chunks: vec![Arc::new(MappedSourceChunk { chunk_id: chunk.id(), source_span: chunk.span(), native, original_spans })],
    } })
}
pub(super) fn reduce(input: ReduceInput<'_, SourceMapValue>,
    checkpoint: &mut impl FnMut() -> Result<(), Int4SourceError>) -> Result<SourceMapValue, Int4SourceError> {
    checkpoint()?;
    let count = input.children.iter().try_fold(0_usize, |n, child| n.checked_add(child.value().len()))
        .ok_or(Int4SourceError::InvalidResult)?;
    let mut chunks = Vec::new();
    chunks.try_reserve_exact(count).map_err(|_| SourcePlanningError::AllocationRefused)?;
    let mut next = input.children.first().ok_or(Int4SourceError::InvalidResult)?.chunk_range().start;
    for child in input.children {
        if child.chunk_range().start != next { return Err(Int4SourceError::InvalidResult); }
        for chunk in &child.value().chunks {
            checkpoint()?;
            if chunk.chunk_id != next { return Err(Int4SourceError::InvalidResult); }
            chunks.push(Arc::clone(chunk)); next += 1;
        }
        if child.chunk_range().end != next { return Err(Int4SourceError::InvalidResult); }
    }
    checkpoint()?;
    Ok(SourceMapValue { chunks })
}
pub(super) fn complete(expected: Int4SourceMapPreflight, verification: GroundingBudget,
    remaining: GroundingBudget, work: Int4Work, masks: u64, mapped: MapReduceResult<SourceMapValue>, cap: u64)
    -> Result<Int4SourceMapRun, Int4SourceMapError> {
    let run = Int4SourceMapRun { schema_version: 1, execution: INT4_SOURCE_MAP_EXECUTION,
        numerics_profile: STRICT_INT4_PROFILE, group_width: expected.group_width,
        task_spec: expected.task_spec, semantics: SEMANTICS,
        planned_model_work: expected.work, model_work: work,
        reserved_mask_node_visits: expected.masks, mask_node_visit_charge: masks,
        verification_used: SourceMapVerification::used(verification, remaining)?,
        untrusted_fields: ["mapped.root.value"], mapped };
    expected.verify_completed(&run)?;
    check_size(&run, cap).map_err(Int4SourceError::from)?;
    Ok(run)
}

pub(super) fn check_run(plan: &PreparedInt4SourceTask, run: &Int4SourceTaskRun, mask_cap: u64)
    -> Result<(), Int4SourceError> {
    check_size(run, plan.max_result_bytes())?;
    let (version, spec, profile, ids, positions, logits) = match (&plan.finalizer, &run.result) {
        (Finalizer::Ner(options), SourceTaskResult::Ner(r)) => {
            if r.score_space != ScoreSpace::NotComputed || r.grounding != ExtractionGrounding::SourceMembership
                || r.entities.len() > options.max_entities || r.entities.iter().any(|e| e.text.is_empty()
                    || e.text.chars().count() > options.max_mention_scalars || !options.types.contains(&e.entity_type)) {
                return Err(Int4SourceError::InvalidResult);
            }
            (r.schema_version, &r.task_spec_version, &r.numerics_profile, &r.generated_token_ids, r.forward_positions, r.projected_logits)
        }
        (Finalizer::Keyphrases(options), SourceTaskResult::Keyphrases(r)) => {
            if r.score_space != ScoreSpace::NotComputed || r.grounding != ExtractionGrounding::SourceMembership
                || r.ranking_policy != KEYPHRASES_RANKING || r.phrases.len() > options.max_phrases {
                return Err(Int4SourceError::InvalidResult);
            }
            let mut seen = std::collections::BTreeSet::new();
            for (index, phrase) in r.phrases.iter().enumerate() {
                if phrase.rank != index + 1 || phrase.text.is_empty() || phrase.text.chars().count() > options.max_phrase_scalars
                    || !seen.insert(phrase.text.as_str()) { return Err(Int4SourceError::InvalidResult); }
            }
            (r.schema_version, &r.task_spec_version, &r.numerics_profile, &r.generated_token_ids, r.forward_positions, r.projected_logits)
        }
        (Finalizer::Summarize(options), SourceTaskResult::Summarize(r)) => {
            if r.score_space != ScoreSpace::NotComputed || r.citation_guarantee != CitationGuarantee::StructuralSourceMembership
                || r.semantic_support != SummarySemanticSupport::NotAssessed || r.bullets.len() > options.max_bullets {
                return Err(Int4SourceError::InvalidResult);
            }
            for bullet in &r.bullets {
                if !bullet.text.chars().any(|c| !c.is_whitespace()) || bullet.text.chars().count() > options.max_bullet_scalars
                    || bullet.citations.is_empty() || bullet.citations.len() > options.max_citations_per_bullet
                    || bullet.citations.iter().any(|c| c.quote.is_empty() || c.quote.chars().count() > options.max_quote_scalars) {
                    return Err(Int4SourceError::InvalidResult);
                }
            }
            (r.schema_version, &r.task_spec_version, &r.numerics_profile, &r.generated_token_ids, r.forward_positions, r.projected_logits)
        }
        (Finalizer::Answer(_, options), SourceTaskResult::Answer(r)) => {
            if r.citations.len() > options.max_citations {
                return Err(Int4SourceError::InvalidResult);
            }
            (r.schema_version, &r.task_spec_version, &r.numerics_profile, &r.generated_token_ids, r.forward_positions, r.projected_logits)
        }
        _ => return Err(Int4SourceError::InvalidResult),
    };
    let options = plan.extraction.options();
    let actual = constrained_int4::planned_work(plan.group_width(), plan.prompt_tokens(), ids.len())?;
    if run.schema_version != 1 || run.execution != INT4_SOURCE_EXECUTION || run.group_width != plan.group_width()
        || version != 1 || spec != &plan.execution_identity().task_spec || profile != STRICT_INT4_PROFILE
        || ids.is_empty() || ids.len() > plan.task_budget().max_output_tokens as usize
        || ids.last() != Some(&options.eos_token_id)
        || ids[..ids.len() - 1].iter().any(|id| *id == options.eos_token_id || *id as usize >= NANBEIGE_VOCAB_SIZE
            || options.excluded_token_ids.contains(id))
        || actual != run.model_work || positions != actual.forward_positions || logits != actual.projected_logits
        || !fits(actual, plan.planned_work()) || mask_charge(&run.result) > mask_cap {
        return Err(Int4SourceError::InvalidResult);
    }
    Ok(())
}
pub(super) fn mask_charge(result: &SourceTaskResult) -> u64 {
    match result { SourceTaskResult::Ner(r) => r.mask_node_visit_charge,
        SourceTaskResult::Keyphrases(r) => r.mask_node_visit_charge,
        SourceTaskResult::Summarize(r) => r.mask_node_visit_charge,
        SourceTaskResult::Answer(r) => r.mask_node_visit_charge }
}
pub(super) fn original_spans<F: FnMut() -> Result<(), Int4SourceError>>(chunk: &SourceChunk<'_>,
    result: &SourceTaskResult, remaining: &mut GroundingBudget, checkpoint: &mut F)
    -> Result<Vec<OriginalSourceSpans>, Int4SourceError> {
    let mut fields = Vec::new();
    let mut field = |kind, text: &str, occurrence, spans: &[VerifiedSourceSpan]| -> Result<(), Int4SourceError> {
        checkpoint()?;
        if text.is_empty() { return Err(Int4SourceError::InvalidResult); }
        remaining.max_fields = remaining.max_fields.checked_sub(1).ok_or(ExtractError::OutputBudgetExceeded)?;
        let mut checked = scan_occurrences(chunk.text(), text, remaining).map_err(|error| match error {
            FieldGroundingError::AllocationRefused => ExtractError::AllocationRefused,
            FieldGroundingError::FieldBudget | FieldGroundingError::MatchBudget | FieldGroundingError::WorkBudget =>
                ExtractError::OutputBudgetExceeded,
            _ => ExtractError::InvalidResult,
        })?;
        if checked.as_slice() != spans || occurrence != if checked.len() == 1 { SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous } {
            return Err(Int4SourceError::InvalidResult);
        }
        let origin = chunk.span();
        for span in &mut checked {
            checkpoint()?;
            span.byte_start = span.byte_start.checked_add(origin.byte_start).ok_or(ExtractError::InvalidResult)?;
            span.byte_end = span.byte_end.checked_add(origin.byte_start).ok_or(ExtractError::InvalidResult)?;
            span.scalar_start = span.scalar_start.checked_add(origin.scalar_start).ok_or(ExtractError::InvalidResult)?;
            span.scalar_end = span.scalar_end.checked_add(origin.scalar_start).ok_or(ExtractError::InvalidResult)?;
            if span.byte_end > origin.byte_end || span.scalar_end > origin.scalar_end { return Err(Int4SourceError::InvalidResult); }
        }
        fields.try_reserve(1).map_err(|_| SourcePlanningError::AllocationRefused)?;
        fields.push(OriginalSourceSpans { field: kind, spans: checked });
        Ok(())
    };
    match result {
        SourceTaskResult::Ner(r) => for (index, e) in r.entities.iter().enumerate() {
            field(SourceMapField::Entity { index }, &e.text, e.occurrence, &e.spans)?;
        },
        SourceTaskResult::Keyphrases(r) => for (index, p) in r.phrases.iter().enumerate() {
            field(SourceMapField::Keyphrase { index }, &p.text, p.occurrence, &p.spans)?;
        },
        SourceTaskResult::Summarize(r) => for (bullet, b) in r.bullets.iter().enumerate() {
            for (citation, c) in b.citations.iter().enumerate() {
                field(SourceMapField::SummaryCitation { bullet, citation }, &c.quote, c.occurrence, &c.spans)?;
            }
        },
        SourceTaskResult::Answer(_) => return Err(Int4SourceError::InvalidResult),
    }
    checkpoint()?;
    Ok(fields)
}
pub(super) fn statistics(value: &SourceMapValue) -> Result<(Int4Work, u64, usize, usize), Int4SourceError> {
    let (mut work, mut masks, mut fields, mut matches) = (Int4Work::default(), 0_u64, 0_usize, 0_usize);
    for chunk in value.chunks() {
        work = add_work(work, chunk.native.model_work).ok_or(Int4SourceError::InvalidResult)?;
        masks = masks.checked_add(mask_charge(&chunk.native.result)).ok_or(Int4SourceError::InvalidResult)?;
        fields = fields.checked_add(chunk.original_spans.len()).ok_or(Int4SourceError::InvalidResult)?;
        for field in &chunk.original_spans {
            matches = matches.checked_add(field.spans.len()).ok_or(Int4SourceError::InvalidResult)?;
        }
    }
    Ok((work, masks, fields, matches))
}
