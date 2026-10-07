//! One token per pending head per tick; compatible head projections share weights.
use super::*;

struct Row<'a> {
    prompt: &'a [u32], cursor: CandidateScoreCursor<'a>, schedule: CandidateSchedule,
    max_output_bytes: u64, prompt_done: usize, prefix_done: usize,
    previous: Vec<u32>, prepared: bool, done: bool, rewound: u64,
}
impl<'a> Row<'a> {
    fn new(request: &CompiledRequest<'a>) -> Result<Self, Int8ScoringError> {
        let cursor = request.scorer.cursor(request.mode)?;
        if cursor.planned_work() != request.schedule.scoring { return Err(Int8ScoringError::Accounting); }
        Ok(Self { prompt: request.prompt, cursor, schedule: request.schedule, max_output_bytes: request.max_output_bytes,
            prompt_done: 0, prefix_done: 0, previous: reserve(request.schedule.max_prefix)?,
            prepared: false, done: false, rewound: 0 })
    }
    fn step<D: GroupDriver>(&mut self, slot: usize, driver: &mut D) -> Result<Option<CohortToken>, Int8ScoringError> {
        if self.done { return Ok(None); }
        let request = self.cursor.request()?.ok_or(Int8ScoringError::Accounting)?;
        if request.prefix.len() > self.schedule.max_prefix { return Err(Int8ScoringError::Traversal); }
        if !self.prepared {
            if request.ordinal == 0 {
                if !request.prefix.is_empty() || self.prompt_done != 0 || driver.position(slot)? != 0 {
                    return Err(Int8ScoringError::Traversal);
                }
                self.prefix_done = 0;
            } else {
                if self.prompt_done != self.prompt.len() || request.prefix <= self.previous.as_slice()
                    || driver.position(slot)? != self.prompt.len() + self.previous.len() {
                    return Err(Int8ScoringError::Traversal);
                }
                let common = self.previous.iter().zip(request.prefix).take_while(|(a, b)| a == b).count();
                // A newly visited prefix must recompute a descendant hidden.
                if common >= request.prefix.len() { return Err(Int8ScoringError::Traversal); }
                driver.rewind(slot, self.prompt.len() + common)?;
                self.rewound = add(self.rewound, (self.previous.len() - common) as u64)?;
                self.prefix_done = common;
            }
            self.prepared = true;
        }
        if driver.position(slot)? != self.prompt_done + self.prefix_done { return Err(Int8ScoringError::Traversal); }
        let token = if self.prompt_done < self.prompt.len() { Some(self.prompt[self.prompt_done]) }
            else { request.prefix.get(self.prefix_done).copied() };
        Ok(token.map(|token| CohortToken { sequence: slot, token }))
    }
    fn forwarded(&mut self) -> Result<(), Int8ScoringError> {
        if self.done || !self.prepared { return Err(Int8ScoringError::Traversal); }
        if self.prompt_done < self.prompt.len() { self.prompt_done += 1; }
        else {
            let request = self.cursor.request()?.ok_or(Int8ScoringError::Accounting)?;
            if self.prefix_done >= request.prefix.len() { return Err(Int8ScoringError::Traversal); }
            self.prefix_done += 1;
        }
        Ok(())
    }
    fn ready<D: GroupDriver>(&self, slot: usize, driver: &D) -> Result<bool, Int8ScoringError> {
        if self.done { return Ok(false); }
        let request = self.cursor.request()?.ok_or(Int8ScoringError::Accounting)?;
        if driver.position(slot)? != self.prompt_done + self.prefix_done { return Err(Int8ScoringError::Traversal); }
        Ok(self.prepared && self.prompt_done == self.prompt.len() && self.prefix_done == request.prefix.len())
    }
    fn accept(&mut self, logits: &[f32]) -> Result<(), Int8ScoringError> {
        let request = self.cursor.request()?.ok_or(Int8ScoringError::Accounting)?;
        let ordinal = request.ordinal;
        self.previous.clear(); self.previous.extend_from_slice(request.prefix);
        self.cursor.accept(ordinal, logits)?;
        self.done = self.cursor.request()?.is_none(); self.prepared = false;
        Ok(())
    }
}
pub(super) fn same_rows(a: ProjectionRows<'_>, b: ProjectionRows<'_>) -> bool {
    match (a, b) {
        (ProjectionRows::FullVocabulary { vocabulary_size: a }, ProjectionRows::FullVocabulary { vocabulary_size: b }) => a == b,
        (ProjectionRows::Selected(a), ProjectionRows::Selected(b)) => a == b,
        _ => false,
    }
}

pub(super) fn drive_with<D, T, E, F>(requests: &[CompiledRequest<'_>], budget: Int8ScoringCohortBudget,
    driver: &mut D, finalize: F) -> Result<T, E>
where D: GroupDriver, E: From<Int8ScoringError>, F: FnOnce(Int8CandidateCohortRun, &mut D::Control) -> Result<T, E> {
    let result = (|| {
        let run = drive(requests, budget, driver).map_err(E::from)?;
        checkpoint(driver.control()).map_err(E::from)?;
        let result = finalize(run, driver.control())?;
        checkpoint(driver.control()).map_err(E::from)?;
        Ok(result)
    })();
    if result.is_err() { driver.abort(); }
    result
}
fn drive<D: GroupDriver>(requests: &[CompiledRequest<'_>], budget: Int8ScoringCohortBudget, driver: &mut D)
    -> Result<Int8CandidateCohortRun, Int8ScoringError> {
    check_count(requests.len())?; check_output_limit(budget.max_output_bytes)?;
    let mut rows = reserve(requests.len())?; let mut expected = Int8Work::default();
    for request in requests {
        checkpoint(driver.control())?;
        check_prompt(request.prompt)?; check_output_limit(request.max_output_bytes)?;
        expected = expected.checked_add(request.schedule.model)?;
        rows.push(Row::new(request)?);
    }
    let mut steps = reserve(rows.len())?; let mut slots = reserve(rows.len())?;
    let mut ready = reserve(rows.len())?; ready.resize(rows.len(), false);
    let (mut group_steps, mut projection_groups) = (0_u64, 0_u64);
    while rows.iter().any(|row| !row.done) {
        checkpoint(driver.control())?; steps.clear();
        for (slot, row) in rows.iter_mut().enumerate() {
            checkpoint(driver.control())?;
            if let Some(step) = row.step(slot, driver)? { steps.push(step); }
        }
        if !steps.is_empty() {
            driver.append_group(&steps)?;
            group_steps = add(group_steps, 1)?;
            if group_steps > expected.forward_positions { return Err(Int8ScoringError::Accounting); }
            for step in &steps { rows[step.sequence].forwarded()?; }
        }
        for (slot, row) in rows.iter().enumerate() { ready[slot] = row.ready(slot, driver)?; }
        if steps.is_empty() && !ready.iter().any(|&value| value) { return Err(Int8ScoringError::Accounting); }
        while let Some(first) = ready.iter().position(|&value| value) {
            checkpoint(driver.control())?; slots.clear();
            let selection = rows[first].cursor.request()?.ok_or(Int8ScoringError::Accounting)?.rows;
            for (slot, row) in rows.iter().enumerate() {
                if ready[slot] && same_rows(selection, row.cursor.request()?.ok_or(Int8ScoringError::Accounting)?.rows) {
                    slots.push(slot);
                }
            }
            let selection = checked_rows(selection)?;
            let width = selection.checked_count(V).map_err(StrictInt8Error::from)?;
            // Only this one sequence-major buffer is live. Different legal
            // row sets are separate groups, never widened or renormalized.
            let logits = driver.logits_group(&slots, selection)?;
            if logits.len() != width.checked_mul(slots.len()).ok_or(Int8ScoringError::Accounting)? {
                return Err(Int8ScoringError::Accounting);
            }
            projection_groups = add(projection_groups, 1)?;
            for (&slot, values) in slots.iter().zip(logits.chunks_exact(width)) {
                checkpoint(driver.control())?;
                rows[slot].accept(values)?; ready[slot] = false;
            }
        }
    }
    let mut heads = reserve(rows.len())?; let mut model_work = Int8Work::default();
    for (slot, row) in rows.into_iter().enumerate() {
        checkpoint(driver.control())?;
        let work = driver.work(slot)?; let rewound = driver.rewound_positions(slot)?;
        if work != row.schedule.model || rewound != row.rewound || rewound != row.schedule.rewound {
            return Err(Int8ScoringError::Accounting);
        }
        let scores = row.cursor.finish()?;
        if scores.work != row.schedule.scoring { return Err(Int8ScoringError::Accounting); }
        let head = Int8CandidateRun { schema_version: 1, execution: INT8_SCORING_EXECUTION.to_owned(),
            numerics_profile: STRICT_INT8_PROFILE.to_owned(), scores, model_work: work, rewound_positions: rewound };
        check_output(&head, row.max_output_bytes)?;
        model_work = model_work.checked_add(work)?; heads.push(head);
    }
    if model_work != expected { return Err(Int8ScoringError::Accounting); }
    let run = Int8CandidateCohortRun { schema_version: 1, execution: INT8_SCORING_COHORT_EXECUTION.to_owned(),
        numerics_profile: STRICT_INT8_PROFILE.to_owned(), heads, group_steps, projection_groups, model_work };
    check_output(&run, budget.max_output_bytes)?;
    Ok(run)
}
