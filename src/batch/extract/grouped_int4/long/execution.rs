//! One native engine/control, ordered immutable values and independently lifted fields.
use super::*;
use std::cell::RefCell;

impl PreparedInt4ExtractionMap<'_> {
    fn preflight_driver<D: Driver, C: DecodeStepControl>(&self, admitted: &[ExecutionIdentity],
        driver: &D, control: &mut C) -> Result<(), Int4ExtractionMapError> {
        if self.plans.is_empty() || self.plans.len() != admitted.len() || self.plans.len() != self.expected.chunks
            || self.chunks.chunks().len() != self.plans.len() { return Err(Int4ExtractionMapError::Admission); }
        for ((plan, identity), chunk) in self.plans.iter().zip(admitted).zip(self.chunks.chunks()) {
            step(control)?;
            plan.verify_identity(identity)?;
            if plan.source().text() != chunk.text() { return Err(Int4ExtractionMapError::Admission); }
            // Check even the FINAL chunk's full resident KV reservation before
            // the first forward. Caller identities cannot waive native capacity.
            preflight(plan, driver).map_err(batch_error)?;
        }
        step(control)?;
        Ok(())
    }
    pub fn execute_with_control<C: DecodeStepControl>(self, admitted: &[ExecutionIdentity],
        engine: &mut StrictInt4Engine<'_>, vocabulary: &ExtractionVocabulary, control: &mut C)
        -> Result<Int4ExtractionMapRun, Int4ExtractionMapError> {
        if admitted.len() != self.expected.chunks || admitted.len() != self.plans.len() {
            return Err(Int4ExtractionMapError::Admission);
        }
        // Actual weights, content fingerprint, partition and profile are
        // checked for even the final chunk before the first model forward.
        // The private scripted driver below never supplies this authority.
        for identity in admitted {
            step(control)?;
            constrained_int4::check_model(identity, engine.model_identity()).map_err(Int4ExtractError::from)?;
        }
        self.execute_with_driver(admitted, NativeDriver { engine, vocabulary }, control)
    }
    // Private fault-injection seam only; public execution always owns the real
    // native extraction driver and cannot accept a fabricated map result.
    pub(super) fn execute_with_driver<D: Driver, C: DecodeStepControl>(self, admitted: &[ExecutionIdentity],
        driver: D, control: &mut C) -> Result<Int4ExtractionMapRun, Int4ExtractionMapError> {
        step(control)?;
        self.preflight_driver(admitted, &driver, control)?;
        let control = RefCell::new(control);
        let mut task = Executor { plans: &self.plans, admitted, driver, control: &control,
            next: 0, work: Int4Work::default(), masks: 0, remaining: self.limits.verification,
            mask_budget: ExtractionMaskBudget { per_mask: self.limits.mapping.mask_limits,
                max_visits_per_item: self.limits.mapping.mask_visits_per_chunk,
                max_visits_per_run: self.limits.mapping.max_mask_visits } };
        let mut cancelled = None;
        let mapped = mapreduce::execute(&self.chunks, &mut task, self.limits.mapping.reduction, || {
            match step(&mut **control.borrow_mut()) {
                Ok(()) => Ok(()), Err(error) => { cancelled = Some(error); Err(MapReduceError::Cancelled) }
            }
        });
        if let Some(error) = cancelled { return Err(error.into()); }
        let mapped = mapped.map_err(Int4ExtractionMapError::Reduction)?;
        step(&mut **control.borrow_mut())?;
        if task.next != self.expected.chunks || !task.driver.clean() { return Err(ExtractError::InvalidResult.into()); }
        let remaining = task.remaining; let total = self.limits.verification;
        let run = Int4ExtractionMapRun { schema_version: 1, execution: INT4_EXTRACTION_MAP_EXECUTION,
            numerics_profile: STRICT_INT4_PROFILE, group_width: self.expected.group_width,
            semantics: SEMANTICS, grounding: self.expected.grounding,
            planned_model_work: self.expected.work, model_work: task.work, reserved_mask_node_visits: self.expected.masks,
            mask_node_visit_charge: task.masks, verification_used: ExtractionMapVerification {
                fields: total.max_fields.checked_sub(remaining.max_fields).ok_or(Int4ExtractionMapError::WorkLimit)?,
                matches: total.max_matches.checked_sub(remaining.max_matches).ok_or(Int4ExtractionMapError::WorkLimit)?,
                scan_steps: total.max_scan_steps.checked_sub(remaining.max_scan_steps).ok_or(Int4ExtractionMapError::WorkLimit)? },
            untrusted_fields: ["mapped.root.value"], mapped };
        self.expected.verify_completed(&run)?;
        check_size(&run, self.max_result_bytes())?;
        step(&mut **control.borrow_mut())?;
        Ok(run)
    }
}

struct Executor<'a, 'c, D, C> {
    plans: &'a [PreparedInt4BatchExtraction], admitted: &'a [ExecutionIdentity], driver: D,
    control: &'a RefCell<&'c mut C>, next: usize, work: Int4Work, masks: u64,
    remaining: GroundingBudget, mask_budget: ExtractionMaskBudget,
}
impl<D: Driver, C: DecodeStepControl> MapReduceTask for Executor<'_, '_, D, C> {
    type Value = ExtractionMapValue;
    type Error = Int4ExtractError;
    fn policy(&self) -> ReductionPolicy {
        ReductionPolicy { id: "lossless-independent-schema-values-source-order-v1", may_discard_information: false }
    }
    fn map_batch(&mut self, chunks: &[SourceChunk<'_>]) -> Result<Vec<MapOutput<Self::Value>>, Int4ExtractError> {
        let mut output = Vec::new();
        output.try_reserve_exact(chunks.len()).map_err(|_| ExtractError::AllocationRefused)?;
        for chunk in chunks {
            let mut control = self.control.borrow_mut();
            step(&mut **control)?;
            if chunk.id() != self.next { return Err(ExtractError::InvalidResult.into()); }
            let plan = self.plans.get(chunk.id()).ok_or(ExtractError::InvalidResult)?;
            let identity = self.admitted.get(chunk.id()).ok_or(ExtractError::InvalidResult)?;
            let native = self.driver.execute(plan, identity, native_budget(plan, self.mask_budget), &mut **control)?;
            if !self.driver.clean() || !valid_result(plan, &native, self.mask_budget.max_visits_per_item) {
                return Err(ExtractError::InvalidResult.into());
            }
            // Recompute schema/JSON/field/EOS invariants from the sealed plan,
            // not just plausible counters supplied alongside the native output.
            plan.plan.verify_completed(&native)?;
            let fields = original_fields(chunk, &native.result.source_fields, &mut self.remaining, &mut **control)?;
            self.work = add_work(self.work, native.model_work).ok_or(ExtractError::InvalidResult)?;
            self.masks = self.masks.checked_add(native.result.output.mask_node_visit_charge).ok_or(ExtractError::InvalidResult)?;
            output.push(MapOutput { chunk_id: chunk.id(), value: ExtractionMapValue { chunks: vec![Arc::new(MappedExtractionChunk {
                chunk_id: chunk.id(), source_span: chunk.span(), native, original_fields: fields })] } });
            self.next += 1;
            step(&mut **control)?;
        }
        Ok(output)
    }
    fn reduce(&mut self, input: ReduceInput<'_, Self::Value>) -> Result<Self::Value, Int4ExtractError> {
        let mut control = self.control.borrow_mut(); step(&mut **control)?;
        let count = input.children.iter().try_fold(0_usize, |n, child| n.checked_add(child.value().chunks.len()))
            .ok_or(ExtractError::InvalidResult)?;
        let mut chunks = Vec::new(); chunks.try_reserve_exact(count).map_err(|_| ExtractError::AllocationRefused)?;
        let mut next = input.children.first().ok_or(ExtractError::InvalidResult)?.chunk_range().start;
        for child in input.children {
            if child.chunk_range().start != next { return Err(ExtractError::InvalidResult.into()); }
            for chunk in &child.value().chunks {
                step(&mut **control)?;
                if chunk.chunk_id != next { return Err(ExtractError::InvalidResult.into()); }
                chunks.push(Arc::clone(chunk)); next += 1;
            }
            if child.chunk_range().end != next { return Err(ExtractError::InvalidResult.into()); }
        }
        step(&mut **control)?;
        Ok(ExtractionMapValue { chunks })
    }
}

fn original_fields<C: DecodeStepControl>(chunk: &SourceChunk<'_>, fields: &[SourceFieldEvidence],
    remaining: &mut GroundingBudget, control: &mut C) -> Result<Vec<SourceFieldEvidence>, Int4ExtractError> {
    let mut output = Vec::new(); output.try_reserve_exact(fields.len()).map_err(|_| ExtractError::AllocationRefused)?;
    for field in fields {
        step(control)?;
        remaining.max_fields = remaining.max_fields.checked_sub(1).ok_or(ExtractError::OutputBudgetExceeded)?;
        let first = field.spans.first().ok_or(ExtractError::InvalidResult)?;
        let text = chunk.text().get(first.byte_start..first.byte_end).ok_or(ExtractError::InvalidResult)?;
        let mut spans = scan_occurrences(chunk.text(), text, remaining).map_err(|error| match error {
            FieldGroundingError::AllocationRefused => Int4ExtractError::from(ExtractError::AllocationRefused),
            FieldGroundingError::FieldBudget | FieldGroundingError::MatchBudget | FieldGroundingError::WorkBudget =>
                Int4ExtractError::from(ExtractError::OutputBudgetExceeded),
            _ => Int4ExtractError::from(ExtractError::InvalidResult),
        })?;
        if spans != field.spans || field.occurrence != (if spans.len() == 1 {
            SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous }) { return Err(ExtractError::InvalidResult.into()); }
        let origin = chunk.span();
        for span in &mut spans {
            step(control)?;
            // Empty verbatim strings legitimately name scalar boundaries. Keep
            // those zero-length spans rather than dropping evidence or guessing.
            span.byte_start = span.byte_start.checked_add(origin.byte_start).ok_or(ExtractError::InvalidResult)?;
            span.byte_end = span.byte_end.checked_add(origin.byte_start).ok_or(ExtractError::InvalidResult)?;
            span.scalar_start = span.scalar_start.checked_add(origin.scalar_start).ok_or(ExtractError::InvalidResult)?;
            span.scalar_end = span.scalar_end.checked_add(origin.scalar_start).ok_or(ExtractError::InvalidResult)?;
            if span.byte_end > origin.byte_end || span.scalar_end > origin.scalar_end { return Err(ExtractError::InvalidResult.into()); }
        }
        let mut pointer = String::new(); pointer.try_reserve_exact(field.json_pointer.len()).map_err(|_| ExtractError::AllocationRefused)?;
        pointer.push_str(&field.json_pointer);
        output.push(SourceFieldEvidence { json_pointer: pointer, occurrence: field.occurrence, spans });
    }
    step(control)?;
    Ok(output)
}
pub(super) fn statistics(value: &ExtractionMapValue) -> Result<(Int4Work, u64, usize, usize), Int4ExtractError> {
    let (mut work, mut masks, mut fields, mut spans) = (Int4Work::default(), 0_u64, 0_usize, 0_usize);
    for chunk in value.chunks() {
        work = add_work(work, chunk.native.model_work).ok_or(ExtractError::InvalidResult)?;
        masks = masks.checked_add(chunk.native.result.output.mask_node_visit_charge).ok_or(ExtractError::InvalidResult)?;
        fields = fields.checked_add(chunk.original_fields.len()).ok_or(ExtractError::InvalidResult)?;
        for field in &chunk.original_fields { spans = spans.checked_add(field.spans.len()).ok_or(ExtractError::InvalidResult)?; }
    }
    Ok((work, masks, fields, spans))
}
