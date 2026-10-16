//! Bounded immutable request epochs on reusable INT4 sequence slots.
//! Only storage is reused: retired work stays debited and each queued budget
//! can be installed exactly once. This wrapper never exposes its inner session.
use super::*;

pub const INT4_REFILL_EXECUTION: &str = "grouped-int4-refilling-token-morsel-epoch-v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RefillAdmission { pub sequence: usize, pub request: usize }
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RefillRetirement { pub request: usize, pub work: Int4Work }

/// A finite epoch, not a renewable stream of work certificates. All request
/// ceilings and every physical slot capacity are checked before opening.
impl<'weights> Int4CohortEngine<'weights> {
    pub fn refill_session<'run, C: DecodeStepControl>(&'run mut self,
        budgets: &[Int4RunBudget], control: &'run mut C)
        -> Result<Int4RefillSession<'run, 'weights, C>, StrictInt4Error> {
        self.check_idle()?;
        let tickets = Tickets::new(budgets, self.sequence_count())?;
        let mut accounts = reserve(self.sequence_count())?;
        for sequence in 0..self.sequence_count() {
            if budgets.iter().any(|budget|
                budget.max_forward_positions > self.sequences[sequence].cache.capacity_positions() as u64) {
                return Err(StrictInt4Error::Context);
            }
            accounts.push(empty_account());
        }
        let maximum = epoch_budget(budgets)?;
        self.state.open()?;
        Ok(Int4RefillSession {
            inner: Int4CohortSession { engine: self, control, accounts,
                ledger: ProjectionLedger::new(maximum.max_projection_work) },
            tickets, retired: Int4Work::default(),
        })
    }
}

pub struct Int4RefillSession<'run, 'weights, C: DecodeStepControl> {
    inner: Int4CohortSession<'run, 'weights, C>, tickets: Tickets, retired: Int4Work,
}
impl<C: DecodeStepControl> Int4RefillSession<'_, '_, C> {
    pub(crate) fn control(&mut self) -> &mut C { self.inner.control() }
    pub(crate) fn abort(&mut self) { self.inner.abort(); }

    /// FIFO request admission into the lowest vacant slot. Retirement can free
    /// storage, but can never rewind the immutable request queue or mint budget.
    pub fn admit_next(&mut self) -> Result<Option<RefillAdmission>, StrictInt4Error> {
        self.reconcile()?;
        let Some(admission) = self.tickets.next_admission()? else { return Ok(None); };
        let row = &self.inner.engine.sequences[admission.sequence];
        if row.hidden.is_some() || !row.cache.all_slots_have_len(0)
            || self.inner.accounts[admission.sequence].work != Int4Work::default() {
            return Err(StrictInt4Error::Boundary);
        }
        let budget = self.tickets.budgets[admission.request];
        self.tickets.install(admission)?;
        self.inner.accounts[admission.sequence] = Account { budget, work: Int4Work::default() };
        Ok(Some(admission))
    }

    /// Complete one request and clear all 44 KV slots plus hidden state before
    /// reuse. Its performed work stays permanently charged to this finite epoch.
    pub fn retire(&mut self, sequence: usize) -> Result<RefillRetirement, StrictInt4Error> {
        self.reconcile()?;
        let request = self.tickets.request(sequence)?;
        let work = self.inner.accounts[sequence].work;
        if work.forward_positions == 0 || self.inner.engine.sequences[sequence].hidden.is_none() {
            return Err(StrictInt4Error::EmptyHidden);
        }
        let retired = self.retired.checked_add(work)?;
        self.inner.engine.state.poisoned = true;
        poll(self.inner.control)?;
        reset_storage(&mut self.inner.engine.sequences[sequence], &mut self.inner.accounts[sequence]);
        self.tickets.retire(sequence)?;
        self.retired = retired;
        self.inner.engine.state.poisoned = false;
        self.reconcile()?;
        Ok(RefillRetirement { request, work })
    }

    pub fn append_packed(&mut self, runs: &[Int4TokenRun<'_>], limits: Int4PrefillLimits)
        -> Result<(), StrictInt4Error> {
        self.reconcile()?;
        for run in runs { self.tickets.request(run.sequence)?; }
        self.inner.append_packed(runs, limits)
    }
    pub fn logits_group(&mut self, sequences: &[usize], rows: LinearRows<'_>)
        -> Result<Vec<f32>, StrictInt4Error> {
        self.reconcile()?;
        for &sequence in sequences { self.tickets.request(sequence)?; }
        self.inner.logits_group(sequences, rows)
    }
    pub fn work(&self, sequence: usize) -> Result<Int4Work, StrictInt4Error> {
        self.tickets.request(sequence)?;
        self.inner.work(sequence)
    }
    /// A success receipt exists only after every immutable ticket retired.
    pub fn completed_work(&self) -> Result<Int4Work, StrictInt4Error> {
        self.reconcile()?;
        if !self.tickets.complete() { return Err(StrictInt4Error::Boundary); }
        Ok(self.retired)
    }
    fn reconcile(&self) -> Result<(), StrictInt4Error> {
        self.inner.engine.state.check()?;
        let mut total = self.retired;
        for (slot, account) in self.inner.accounts.iter().enumerate() {
            if self.inner.position(slot)? as u64 != account.work.forward_positions {
                return Err(StrictInt4Error::Boundary);
            }
            if self.tickets.slots[slot].is_none() && (account.work != Int4Work::default()
                || self.inner.engine.sequences[slot].hidden.is_some()) {
                return Err(StrictInt4Error::Boundary);
            }
            total = total.checked_add(account.work)?;
        }
        if total.projections != self.inner.ledger.reserved() { return Err(StrictInt4Error::Boundary); }
        Ok(())
    }
}
impl<C: DecodeStepControl> Drop for Int4RefillSession<'_, '_, C> {
    fn drop(&mut self) {
        if !self.tickets.complete() { self.inner.abort(); }
        // Inner RAII then clears every physical slot, including during unwind.
    }
}

struct Tickets {
    budgets: Vec<Int4RunBudget>, slots: Vec<Option<usize>>, next: usize, completed: usize,
}
impl Tickets {
    fn new(budgets: &[Int4RunBudget], slots: usize) -> Result<Self, StrictInt4Error> {
        check_count(budgets.len())?; check_count(slots)?;
        if slots > budgets.len() || budgets.iter().any(|budget| budget.max_forward_positions == 0) {
            return Err(StrictInt4Error::Input);
        }
        let mut owned = reserve(budgets.len())?; owned.extend_from_slice(budgets);
        let mut vacant = reserve(slots)?; vacant.resize(slots, None);
        Ok(Self { budgets: owned, slots: vacant, next: 0, completed: 0 })
    }
    fn next_admission(&self) -> Result<Option<RefillAdmission>, StrictInt4Error> {
        if self.next == self.budgets.len() { return Ok(None); }
        let sequence = self.slots.iter().position(Option::is_none).ok_or(StrictInt4Error::Input)?;
        Ok(Some(RefillAdmission { sequence, request: self.next }))
    }
    fn install(&mut self, admission: RefillAdmission) -> Result<(), StrictInt4Error> {
        if self.next_admission()? != Some(admission) { return Err(StrictInt4Error::Boundary); }
        self.slots[admission.sequence] = Some(admission.request);
        self.next += 1;
        Ok(())
    }
    fn request(&self, sequence: usize) -> Result<usize, StrictInt4Error> {
        self.slots.get(sequence).copied().flatten().ok_or(StrictInt4Error::Input)
    }
    fn retire(&mut self, sequence: usize) -> Result<(), StrictInt4Error> {
        self.request(sequence)?;
        self.slots[sequence] = None;
        self.completed = self.completed.checked_add(1).ok_or(StrictInt4Error::Work)?;
        Ok(())
    }
    fn complete(&self) -> bool {
        self.next == self.budgets.len() && self.completed == self.budgets.len()
            && self.slots.iter().all(Option::is_none)
    }
}
fn empty_account() -> Account {
    Account { budget: Int4RunBudget::exact(Int4Work::default()), work: Int4Work::default() }
}
fn reset_storage(row: &mut Sequence, account: &mut Account) {
    row.hidden = None;
    row.cache.clear();
    *account = empty_account();
}
fn epoch_budget(budgets: &[Int4RunBudget]) -> Result<Int4RunBudget, StrictInt4Error> {
    let mut total = Int4RunBudget::exact(Int4Work::default());
    for budget in budgets {
        total.max_forward_positions = total.max_forward_positions.checked_add(budget.max_forward_positions)
            .ok_or(StrictInt4Error::Work)?;
        total.max_attention_pairs = total.max_attention_pairs.checked_add(budget.max_attention_pairs)
            .ok_or(StrictInt4Error::Work)?;
        total.max_projection_work = total.max_projection_work.checked_add(budget.max_projection_work)?;
    }
    Ok(total)
}

#[cfg(test)] mod tests;
