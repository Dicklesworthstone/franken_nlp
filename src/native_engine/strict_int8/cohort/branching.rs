//! Exact KV backtracking for finite-language cohorts, not independent forks.
//!
//! Keep this owner separate from monotone generation/refill sessions. A slot
//! has ONE live branch, no hidden-state history and no transferable work credit.
use super::*;

pub(crate) struct Int8BranchSession<'run, 'weights, C: DecodeStepControl> {
    inner: Int8CohortSession<'run, 'weights, C>,
    discarded: Vec<u64>,
}
impl<'weights> Int8CohortEngine<'weights> {
    pub(crate) fn branch_session<'run, C: DecodeStepControl>(&'run mut self,
        budgets: &[Int8RunBudget], control: &'run mut C)
        -> Result<Int8BranchSession<'run, 'weights, C>, StrictInt8Error> {
        self.check_idle()?;
        if budgets.len() != self.sequence_count() { return Err(StrictInt8Error::Input); }
        let mut discarded = reserve(budgets.len())?; discarded.resize(budgets.len(), 0);
        Ok(Int8BranchSession { inner: self.session(budgets, control)?, discarded })
    }
}
impl<C: DecodeStepControl> Int8BranchSession<'_, '_, C> {
    pub(crate) fn control(&mut self) -> &mut C { self.inner.control() }
    pub(crate) fn abort(&mut self) { self.inner.abort(); }
    pub(crate) fn position(&self, sequence: usize) -> Result<usize, StrictInt8Error> { self.inner.position(sequence) }
    pub(crate) fn work(&self, sequence: usize) -> Result<Int8Work, StrictInt8Error> {
        work_at(&self.inner, sequence, *self.discarded.get(sequence).ok_or(StrictInt8Error::Input)?)
    }
    pub(crate) fn rewound_positions(&self, sequence: usize) -> Result<u64, StrictInt8Error> {
        self.work(sequence)?;
        Ok(self.discarded[sequence])
    }
    pub(crate) fn append_group(&mut self, steps: &[CohortToken]) -> Result<(), StrictInt8Error> {
        self.inner.append_group(steps)
    }
    pub(crate) fn logits_group(&mut self, sequences: &[usize], rows: LinearRows<'_>) -> Result<Vec<f32>, StrictInt8Error> {
        self.inner.logits_group(sequences, rows)
    }
    /// Discard a coherent suffix from all 44 slots. No-op retains its matching
    /// hidden; a real rewind clears hidden so an ancestor cannot project a
    /// descendant's activation. Only a new forward can restore that hidden.
    pub(crate) fn rewind(&mut self, sequence: usize, retain: usize) -> Result<(), StrictInt8Error> {
        let work = self.work(sequence)?;
        let current = self.position(sequence)?;
        let discarded = quote(current, retain, self.discarded[sequence], work.forward_positions)?;
        if current == retain { return Ok(()); }
        self.inner.abort();
        poll(self.inner.control)?;
        truncate(&mut self.inner.engine.sequences[sequence], current, retain)?;
        self.discarded[sequence] = discarded;
        poll(self.inner.control)?;
        self.inner.engine.state.poisoned = false;
        Ok(())
    }
}

// Called with zero discarded positions by ordinary sessions, preserving their
// original strict equality rather than relaxing it to position <= work.
pub(super) fn work_at<C: DecodeStepControl>(session: &Int8CohortSession<'_, '_, C>,
    sequence: usize, discarded: u64) -> Result<Int8Work, StrictInt8Error> {
    session.engine.state.check()?;
    let sum = session.accounts.iter().try_fold(ProjectionWork::default(), |sum, row|
        sum.checked_add(row.work.projections))?;
    if sum != session.ledger.reserved() { return Err(StrictInt8Error::Boundary); }
    let row = session.accounts.get(sequence).ok_or(StrictInt8Error::Input)?;
    verify_position(session.position(sequence)?, discarded, row.work.forward_positions)?;
    Ok(row.work)
}
fn verify_position(position: usize, discarded: u64, forwards: u64) -> Result<(), StrictInt8Error> {
    if (position as u64).checked_add(discarded) != Some(forwards) { return Err(StrictInt8Error::Boundary); }
    Ok(())
}
fn quote(current: usize, retain: usize, discarded: u64, forwards: u64) -> Result<u64, StrictInt8Error> {
    verify_position(current, discarded, forwards)?;
    let removed = current.checked_sub(retain).ok_or(StrictInt8Error::Context)?;
    discarded.checked_add(removed as u64).ok_or(StrictInt8Error::Work)
}
fn truncate(row: &mut Sequence, current: usize, retain: usize) -> Result<(), StrictInt8Error> {
    if retain > current || !row.cache.all_slots_have_len(current) { return Err(StrictInt8Error::Cache); }
    if retain != current {
        row.hidden = None;
        row.cache.rewind_completed(retain).map_err(|_| StrictInt8Error::Cache)?;
    }
    Ok(())
}

#[cfg(test)] mod tests;
