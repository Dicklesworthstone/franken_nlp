//! Bounded immutable request epochs on reusable INT8 sequence slots.
//! Only storage is reused: retired work stays debited and each queued budget
//! can be installed exactly once. This wrapper never exposes its inner session.
use super::*;

pub const INT8_REFILL_EXECUTION: &str = "portable-int8-refilling-token-morsel-epoch-v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RefillAdmission { pub sequence: usize, pub request: usize }
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RefillRetirement { pub request: usize, pub work: Int8Work }

/// A finite epoch, not an independently renewable stream of work certificates.
/// All request ceilings and all physical capacities are checked before opening.
impl<'weights> Int8CohortEngine<'weights> {
    pub fn refill_session<'run, C: DecodeStepControl>(&'run mut self,
        budgets: &[Int8RunBudget], control: &'run mut C)
        -> Result<Int8RefillSession<'run, 'weights, C>, StrictInt8Error> {
        self.check_idle()?;
        let tickets = Tickets::new(budgets, self.sequence_count())?;
        let mut accounts = reserve(self.sequence_count())?;
        for sequence in 0..self.sequence_count() {
            if budgets.iter().any(|budget| budget.max_forward_positions > self.sequences[sequence].cache.capacity_positions() as u64) {
                return Err(StrictInt8Error::Context);
            }
            accounts.push(empty_account());
        }
        let maximum = epoch_budget(budgets)?;
        self.state.open()?;
        Ok(Int8RefillSession {
            inner: Int8CohortSession { engine: self, control, accounts,
                ledger: ProjectionLedger::new(maximum.max_projection_work) },
            tickets, retired: Int8Work::default(),
        })
    }
}

pub struct Int8RefillSession<'run, 'weights, C: DecodeStepControl> {
    inner: Int8CohortSession<'run, 'weights, C>, tickets: Tickets, retired: Int8Work,
}
impl<C: DecodeStepControl> Int8RefillSession<'_, '_, C> {
    pub(crate) fn control(&mut self) -> &mut C { self.inner.control() }
    pub(crate) fn abort(&mut self) { self.inner.abort(); }

    /// FIFO request admission into the lowest vacant physical slot. Neither
    /// the caller nor slot retirement can replace, rewind or extend the queue.
    pub fn admit_next(&mut self) -> Result<Option<RefillAdmission>, StrictInt8Error> {
        self.reconcile()?;
        let Some(admission) = self.tickets.next_admission()? else { return Ok(None); };
        let row = &self.inner.engine.sequences[admission.sequence];
        if row.hidden.is_some() || !row.cache.all_slots_have_len(0)
            || self.inner.accounts[admission.sequence].work != Int8Work::default() {
            return Err(StrictInt8Error::Boundary);
        }
        let budget = self.tickets.budgets[admission.request];
        self.tickets.install(admission)?;
        self.inner.accounts[admission.sequence] = Account { budget, work: Int8Work::default() };
        Ok(Some(admission))
    }

    /// Complete one request and clear ALL logical KV/hidden state before its
    /// slot becomes reusable. Unused per-request work is not handed to a sibling.
    pub fn retire(&mut self, sequence: usize) -> Result<RefillRetirement, StrictInt8Error> {
        self.reconcile()?;
        let request = self.tickets.request(sequence)?;
        let work = self.inner.accounts[sequence].work;
        if work.forward_positions == 0 || self.inner.engine.sequences[sequence].hidden.is_none() {
            return Err(StrictInt8Error::EmptyHidden);
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

    pub fn append_packed(&mut self, runs: &[CohortTokenRun<'_>], limits: Int8PrefillLimits)
        -> Result<(), StrictInt8Error> {
        self.reconcile()?;
        for run in runs { self.tickets.request(run.sequence)?; }
        self.inner.append_packed(runs, limits)
    }
    pub fn logits_group(&mut self, sequences: &[usize], rows: LinearRows<'_>)
        -> Result<Vec<f32>, StrictInt8Error> {
        self.reconcile()?;
        for &sequence in sequences { self.tickets.request(sequence)?; }
        self.inner.logits_group(sequences, rows)
    }
    /// No partial epoch can present an aggregate success receipt. This does
    /// not finalize task output or release host-owned output reservations.
    pub fn completed_work(&self) -> Result<Int8Work, StrictInt8Error> {
        self.reconcile()?;
        if !self.tickets.complete() { return Err(StrictInt8Error::Boundary); }
        Ok(self.retired)
    }
    fn reconcile(&self) -> Result<(), StrictInt8Error> {
        self.inner.engine.state.check()?;
        let mut total = self.retired;
        for (slot, account) in self.inner.accounts.iter().enumerate() {
            if self.inner.position(slot)? as u64 != account.work.forward_positions {
                return Err(StrictInt8Error::Boundary);
            }
            if self.tickets.slots[slot].is_none() && (account.work != Int8Work::default()
                || self.inner.engine.sequences[slot].hidden.is_some()) { return Err(StrictInt8Error::Boundary); }
            total = total.checked_add(account.work)?;
        }
        if total.projections != self.inner.ledger.reserved() { return Err(StrictInt8Error::Boundary); }
        Ok(())
    }
}
impl<C: DecodeStepControl> Drop for Int8RefillSession<'_, '_, C> {
    fn drop(&mut self) {
        if !self.tickets.complete() { self.inner.abort(); }
        // The inner RAII session then clears EVERY slot, even during unwind.
    }
}

struct Tickets {
    budgets: Vec<Int8RunBudget>, slots: Vec<Option<usize>>, next: usize, completed: usize,
}
impl Tickets {
    fn new(budgets: &[Int8RunBudget], slots: usize) -> Result<Self, StrictInt8Error> {
        check_count(budgets.len())?; check_count(slots)?;
        if slots > budgets.len() || budgets.iter().any(|budget| budget.max_forward_positions == 0) {
            return Err(StrictInt8Error::Input);
        }
        let mut owned = reserve(budgets.len())?; owned.extend_from_slice(budgets);
        let mut vacant = reserve(slots)?; vacant.resize(slots, None);
        Ok(Self { budgets: owned, slots: vacant, next: 0, completed: 0 })
    }
    fn next_admission(&self) -> Result<Option<RefillAdmission>, StrictInt8Error> {
        if self.next == self.budgets.len() { return Ok(None); }
        let sequence = self.slots.iter().position(Option::is_none).ok_or(StrictInt8Error::Input)?;
        Ok(Some(RefillAdmission { sequence, request: self.next }))
    }
    fn install(&mut self, admission: RefillAdmission) -> Result<(), StrictInt8Error> {
        if self.next_admission()? != Some(admission) { return Err(StrictInt8Error::Boundary); }
        self.slots[admission.sequence] = Some(admission.request); self.next += 1; Ok(())
    }
    fn request(&self, sequence: usize) -> Result<usize, StrictInt8Error> {
        self.slots.get(sequence).copied().flatten().ok_or(StrictInt8Error::Input)
    }
    fn retire(&mut self, sequence: usize) -> Result<(), StrictInt8Error> {
        self.request(sequence)?; self.slots[sequence] = None; self.completed += 1; Ok(())
    }
    fn complete(&self) -> bool {
        self.next == self.budgets.len() && self.completed == self.budgets.len() && self.slots.iter().all(Option::is_none)
    }
}
fn empty_account() -> Account {
    Account { budget: Int8RunBudget::exact(Int8Work::default()), work: Int8Work::default() }
}
fn reset_storage(row: &mut Sequence, account: &mut Account) {
    row.hidden = None; row.cache.clear(); *account = empty_account();
}
fn epoch_budget(budgets: &[Int8RunBudget]) -> Result<Int8RunBudget, StrictInt8Error> {
    let mut total = Int8RunBudget::exact(Int8Work::default());
    for budget in budgets {
        total.max_forward_positions = total.max_forward_positions.checked_add(budget.max_forward_positions)
            .ok_or(StrictInt8Error::Work)?;
        total.max_attention_pairs = total.max_attention_pairs.checked_add(budget.max_attention_pairs)
            .ok_or(StrictInt8Error::Work)?;
        total.max_projection_work = total.max_projection_work.checked_add(budget.max_projection_work)?;
    }
    Ok(total)
}
#[cfg(test)] mod tests;
