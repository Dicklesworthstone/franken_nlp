//! Exact candidate-tree backtracking over independent grouped-INT4 KV slots.
//!
//! Generation/refill retain their monotone session contract. This separate
//! owner records discarded positions without refunding any completed work.
use super::*;

pub(crate) struct Int4BranchSession<'run, 'weights, C: DecodeStepControl> {
    inner: Int4CohortSession<'run, 'weights, C>,
    discarded: Vec<u64>,
}
impl<'weights> Int4CohortEngine<'weights> {
    pub(crate) fn branch_session<'run, C: DecodeStepControl>(&'run mut self,
        budgets: &[Int4RunBudget], control: &'run mut C)
        -> Result<Int4BranchSession<'run, 'weights, C>, StrictInt4Error> {
        self.check_idle()?;
        if budgets.len() != self.sequence_count() { return Err(StrictInt4Error::Input); }
        let mut discarded = reserve(budgets.len())?; discarded.resize(budgets.len(), 0);
        Ok(Int4BranchSession { inner: self.session(budgets, control)?, discarded })
    }
}
impl<C: DecodeStepControl> Int4BranchSession<'_, '_, C> {
    pub(crate) fn control(&mut self) -> &mut C { self.inner.control() }
    pub(crate) fn abort(&mut self) { self.inner.abort(); }
    pub(crate) fn position(&self, sequence: usize) -> Result<usize, StrictInt4Error> { self.inner.position(sequence) }
    pub(crate) fn work(&self, sequence: usize) -> Result<Int4Work, StrictInt4Error> {
        self.inner.engine.state.check()?;
        let sum = self.inner.accounts.iter().try_fold(ProjectionWork::default(), |sum, row|
            sum.checked_add(row.work.projections))?;
        if sum != self.inner.ledger.reserved() { return Err(StrictInt4Error::Work); }
        let row = self.inner.accounts.get(sequence).ok_or(StrictInt4Error::Input)?;
        let discarded = *self.discarded.get(sequence).ok_or(StrictInt4Error::Input)?;
        verify_position(self.position(sequence)?, discarded, row.work.forward_positions)?;
        Ok(row.work)
    }
    pub(crate) fn rewound_positions(&self, sequence: usize) -> Result<u64, StrictInt4Error> {
        self.work(sequence)?;
        Ok(self.discarded[sequence])
    }
    pub(crate) fn append_group(&mut self, steps: &[Int4CohortToken]) -> Result<(), StrictInt4Error> {
        self.inner.append_group(steps)
    }
    pub(crate) fn logits_group(&mut self, sequences: &[usize], rows: LinearRows<'_>) -> Result<Vec<f32>, StrictInt4Error> {
        self.inner.logits_group(sequences, rows)
    }
    /// A no-op preserves the current hidden state. A real rewind clears it,
    /// so an ancestor can never project a descendant's stale activation.
    pub(crate) fn rewind(&mut self, sequence: usize, retain: usize) -> Result<(), StrictInt4Error> {
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
fn verify_position(position: usize, discarded: u64, forwards: u64) -> Result<(), StrictInt4Error> {
    if (position as u64).checked_add(discarded) != Some(forwards) { return Err(StrictInt4Error::Boundary); }
    Ok(())
}
fn quote(current: usize, retain: usize, discarded: u64, forwards: u64) -> Result<u64, StrictInt4Error> {
    verify_position(current, discarded, forwards)?;
    let removed = current.checked_sub(retain).ok_or(StrictInt4Error::Context)?;
    discarded.checked_add(removed as u64).ok_or(StrictInt4Error::Work)
}
fn truncate(row: &mut Sequence, current: usize, retain: usize) -> Result<(), StrictInt4Error> {
    if retain > current || !row.cache.all_slots_have_len(current) { return Err(StrictInt4Error::Cache); }
    if retain != current {
        row.hidden = None;
        row.cache.rewind_completed(retain).map_err(|_| StrictInt4Error::Cache)?;
    }
    Ok(())
}

#[cfg(test)] mod tests;
