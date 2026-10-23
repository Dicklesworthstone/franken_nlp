//! Shared-weight grouped-INT4 execution for a fixed, ragged sequence cohort.
//!
//! Each sequence owns 44 KV slots and a non-renewable work ceiling. Linear
//! projections share packed groups, never attention contexts. One caller-owned
//! controller governs this finite batch; any execution failure poisons ALL
//! rows. No dynamic membership, runtime, worker spawn or model-quality claim.
use super::*;
use crate::native_engine::portable_int8::int4::batch::MAX_INT4_BATCH_ROWS;
mod execution;
pub mod packed;

pub const INT4_COHORT_EXECUTION: &str = "grouped-int4-ragged-sequence-cohort-v1";
pub const MAX_INT4_SEQUENCES: usize = MAX_INT4_BATCH_ROWS;

impl Int4Work {
    /// Add independent sequences' or heads' work, not their lengths before
    /// computing attention triangles. No cross-document attention is priced.
    pub fn checked_add(self, other: Self) -> Result<Self, StrictInt4Error> {
        Ok(Self {
            forward_positions: self.forward_positions.checked_add(other.forward_positions).ok_or(StrictInt4Error::Work)?,
            projected_logits: self.projected_logits.checked_add(other.projected_logits).ok_or(StrictInt4Error::Work)?,
            attention_pairs: self.attention_pairs.checked_add(other.attention_pairs).ok_or(StrictInt4Error::Work)?,
            projections: self.projections.checked_add(other.projections)?,
        })
    }
}
impl Int4MemoryRequirement {
    /// Complete fixed cohort payload: ragged KV capacities, ONE RoPE table,
    /// ONE unpacking buffer, grouped BF16/activation rails and full head output.
    /// Model, samplers, retained results and allocator slack are separate.
    pub fn for_cohort(contexts: &[usize], group_width: usize) -> Result<Self, StrictInt4Error> {
        check_count(contexts.len())?;
        let mut kv_bytes = 0_u64; let mut longest = 0;
        for &context in contexts {
            let need = Self::for_context(context, group_width)?;
            kv_bytes = kv_bytes.checked_add(need.kv_bytes).ok_or(StrictInt4Error::Memory)?;
            longest = longest.max(context);
        }
        let base = Self::for_context(longest, group_width)?;
        let rails = prefill::Int4PrefillLimits::required_extra_scratch_bytes(contexts.len())?;
        let heads = (contexts.len() as u64).checked_mul((V * std::mem::size_of::<f32>() + 8192) as u64)
            .ok_or(StrictInt4Error::Memory)?;
        let scratch_payload_bound = base.scratch_payload_bound.checked_add(rails)
            .and_then(|n| n.checked_add(heads)).ok_or(StrictInt4Error::Memory)?;
        Ok(Self { kv_bytes, rope_bytes: base.rope_bytes, scratch_payload_bound })
    }
}
struct Sequence { cache: KvCache, hidden: Option<Vec<Bf16>> }

/// Borrows one immutable, complete source/snapshot model. Physical slots are
/// fixed at construction and never become sampler identities or refill quotas.
pub struct Int4CohortEngine<'weights> {
    weights: &'weights Int4ModelWeights, rope: RopeTablesF32,
    sequences: Vec<Sequence>, activations: Vec<ActivationBuffer>, unpack: Vec<i8>,
    kernel: PackedInt4Kernel, state: RunState,
}
impl<'weights> Int4CohortEngine<'weights> {
    pub fn new(weights: &'weights Int4ModelWeights, contexts: &[usize], budget: Int4MemoryBudget)
        -> Result<Self, StrictInt4Error> {
        Int4MemoryRequirement::for_cohort(contexts, weights.group_width())?.check(budget)?;
        let mut sequences = reserve(contexts.len())?; let mut activations = reserve(contexts.len())?;
        for &context in contexts {
            sequences.push(Sequence { cache: KvCache::try_with_capacity(context)
                .map_err(|_| StrictInt4Error::Allocation)?, hidden: None });
            activations.push(ActivationBuffer::try_new(I)?);
        }
        let longest = contexts.iter().copied().max().ok_or(StrictInt4Error::Context)?;
        Ok(Self { weights, rope: RopeTablesF32::nanbeige(longest).map_err(|_| StrictInt4Error::Rope)?,
            sequences, activations, unpack: filled(weights.group_width(), 0_i8)?,
            kernel: PackedInt4Kernel::scalar(), state: RunState::default() })
    }
    /// Only an already hardware-checked executable is accepted; no dispatch.
    pub fn with_kernel(mut self, kernel: PackedInt4Kernel) -> Self { self.kernel = kernel; self }
    pub fn kernel_kind(&self) -> Int8KernelKind { self.kernel.kind() }
    pub fn model_identity(&self) -> &Int4ModelIdentity { self.weights.identity() }
    pub fn sequence_count(&self) -> usize { self.sequences.len() }
    pub fn capacity(&self, sequence: usize) -> Result<usize, StrictInt4Error> {
        Ok(self.sequences.get(sequence).ok_or(StrictInt4Error::Input)?.cache.capacity_positions())
    }
    pub fn is_poisoned(&self) -> bool { self.state.poisoned }
    pub fn check_idle(&self) -> Result<(), StrictInt4Error> {
        if self.state.active || self.state.poisoned || self.sequences.iter()
            .any(|row| row.hidden.is_some() || !row.cache.all_slots_have_len(0)) {
            return Err(StrictInt4Error::EngineUnavailable);
        }
        Ok(())
    }
    pub fn session<'run, C: DecodeStepControl>(&'run mut self, budgets: &[Int4RunBudget], control: &'run mut C)
        -> Result<Int4CohortSession<'run, 'weights, C>, StrictInt4Error> {
        self.check_idle()?;
        if budgets.len() != self.sequence_count() { return Err(StrictInt4Error::Input); }
        let mut accounts = reserve(budgets.len())?; let mut total = ProjectionWork::default();
        for &budget in budgets {
            total = total.checked_add(budget.max_projection_work)?;
            accounts.push(Account { budget, work: Int4Work::default() });
        }
        self.state.open()?;
        Ok(Int4CohortSession { engine: self, control, accounts, ledger: ProjectionLedger::new(total) })
    }
}

/// Strictly increasing physical sequence slots; inactive/finished rows are
/// absent, never padded or advanced as a side effect of a neighbor's step.
#[derive(Clone, Copy, Debug)]
pub struct Int4CohortToken { pub sequence: usize, pub token: u32 }
struct Account { budget: Int4RunBudget, work: Int4Work }
impl Account {
    fn quote(&self, width: usize, start: usize, capacity: usize, positions: usize, heads: usize)
        -> Result<Int4Work, StrictInt4Error> {
        if start.checked_add(positions).is_none_or(|end| end > capacity) { return Err(StrictInt4Error::Context); }
        let delta = Int4Work::for_sequence(width, start, positions, heads)?;
        let total = self.work.checked_add(delta)?;
        if total.forward_positions > self.budget.max_forward_positions
            || total.attention_pairs > self.budget.max_attention_pairs
            || !total.projections.fits(self.budget.max_projection_work) { return Err(StrictInt4Error::Work); }
        Ok(delta)
    }
}

/// Exclusive batch lifetime. Preflight refuses without mutation; any failure
/// after native execution starts permanently poisons the entire engine. Drop
/// clears all sequences, including rows that completed before a later failure.
pub struct Int4CohortSession<'run, 'weights, C: DecodeStepControl> {
    engine: &'run mut Int4CohortEngine<'weights>, control: &'run mut C,
    accounts: Vec<Account>, ledger: ProjectionLedger,
}
impl<C: DecodeStepControl> Int4CohortSession<'_, '_, C> {
    pub(crate) fn control(&mut self) -> &mut C { &mut *self.control }
    pub(crate) fn abort(&mut self) { self.engine.state.poisoned = true; }
    pub fn position(&self, sequence: usize) -> Result<usize, StrictInt4Error> {
        self.engine.state.check()?;
        let cache = &self.engine.sequences.get(sequence).ok_or(StrictInt4Error::Input)?.cache;
        let position = cache.len_for_slot(0).map_err(|_| StrictInt4Error::Cache)?;
        if !cache.all_slots_have_len(position) { return Err(StrictInt4Error::Cache); }
        Ok(position)
    }
    pub fn preflight(&self, sequence: usize, positions: usize, head_rows: usize) -> Result<Int4Work, StrictInt4Error> {
        let start = self.position(sequence)?;
        let work = self.accounts[sequence].quote(self.engine.weights.group_width(), start,
            self.engine.capacity(sequence)?, positions, head_rows)?;
        self.ledger.preflight(work.projections)?; Ok(work)
    }
    /// Only complete work may be read. Reconcile the physical projection ledger
    /// with ALL row accounts, then independently check this row's causal work.
    pub fn work(&self, sequence: usize) -> Result<Int4Work, StrictInt4Error> {
        let position = self.position(sequence)?;
        let mut total = ProjectionWork::default();
        for account in &self.accounts { total = total.checked_add(account.work.projections)?; }
        if total != self.ledger.reserved() { return Err(StrictInt4Error::Work); }
        let work = self.accounts[sequence].work;
        let heads = usize::try_from(work.projected_logits).map_err(|_| StrictInt4Error::Work)?;
        if work != Int4Work::for_sequence(self.engine.weights.group_width(), 0, position, heads)? {
            return Err(StrictInt4Error::Work);
        }
        Ok(work)
    }
    pub fn append_group(&mut self, steps: &[Int4CohortToken]) -> Result<(), StrictInt4Error> {
        self.engine.state.check()?;
        check_indices(steps.iter().map(|step| step.sequence), self.accounts.len())?;
        if steps.iter().any(|step| step.token as usize >= V) { return Err(StrictInt4Error::Input); }
        let mut positions = reserve(steps.len())?; let mut quotes = reserve(steps.len())?;
        let mut total = ProjectionWork::default();
        for step in steps {
            positions.push(self.position(step.sequence)?);
            let quote = self.preflight(step.sequence, 1, 0)?;
            total = total.checked_add(quote.projections)?; quotes.push(quote);
        }
        self.ledger.preflight(total)?;
        self.engine.state.poisoned = true;
        poll(self.control)?;
        let mut hidden = reserve(steps.len())?;
        for step in steps {
            poll(self.control)?;
            let source = self.engine.weights.embeddings.row(step.token as usize).map_err(|_| StrictInt4Error::Input)?;
            let mut row = filled(H, Bf16::from_bits(0))?; row.copy_from_slice(source);
            finite(&row)?; hidden.push(row); self.engine.sequences[step.sequence].hidden = None;
        }
        let engine = &mut *self.engine;
        let runner = LoopRunner::from_layer_weights(&engine.weights.layers);
        let mut executor = execution::Executor { final_norm: &engine.weights.final_norm, rope: &engine.rope,
            sequences: &mut engine.sequences, activations: &mut engine.activations[..steps.len()],
            unpack: &mut engine.unpack, kernel: engine.kernel, steps, positions: &positions,
            ledger: &mut self.ledger, control: &mut *self.control, completed: 0, norms: 0 };
        runner.run_group(&mut executor, &mut hidden)?;
        if executor.completed != KV_SLOT_COUNT || executor.norms != 2 { return Err(StrictInt4Error::Boundary); }
        for (((step, position), row), quote) in steps.iter().zip(positions).zip(hidden).zip(quotes) {
            let target = &mut engine.sequences[step.sequence];
            if !target.cache.all_slots_have_len(position + 1) { return Err(StrictInt4Error::Cache); }
            finite(&row)?; target.hidden = Some(row);
            self.accounts[step.sequence].work = self.accounts[step.sequence].work.checked_add(quote)?;
        }
        poll(self.control)?; engine.state.poisoned = false; Ok(())
    }
    /// Sequence-selection-major output, with checked_count(V) contiguous
    /// values for each listed slot. Only requested vocabulary rows are read.
    /// Retaining multiple results requires extra caller-owned admission.
    pub fn logits_group(&mut self, sequences: &[usize], rows: LinearRows<'_>) -> Result<Vec<f32>, StrictInt4Error> {
        self.engine.state.check()?;
        check_indices(sequences.iter().copied(), self.accounts.len())?;
        let count = rows.checked_count(V)?;
        let mut quotes = reserve(sequences.len())?; let mut total = ProjectionWork::default();
        for &sequence in sequences {
            if self.engine.sequences[sequence].hidden.is_none() { return Err(StrictInt4Error::EmptyHidden); }
            let quote = self.preflight(sequence, 0, count)?;
            total = total.checked_add(quote.projections)?; quotes.push(quote);
        }
        self.ledger.preflight(total)?;
        let length = count.checked_mul(sequences.len()).ok_or(StrictInt4Error::Memory)?;
        let mut output = filled(length, 0.0_f32)?;
        self.engine.state.poisoned = true;
        for (input, &sequence) in self.engine.activations.iter_mut().zip(sequences) {
            poll(self.control)?;
            input.encode_bf16(self.engine.sequences[sequence].hidden.as_deref().ok_or(StrictInt4Error::EmptyHidden)?)?;
        }
        self.engine.weights.head.as_linear().with_kernel(self.engine.kernel).project_batch_f32_into(
            &self.engine.activations[..sequences.len()], rows, &mut output, &mut self.engine.unpack,
            &mut self.ledger, &mut *self.control)?;
        for (&sequence, quote) in sequences.iter().zip(quotes) {
            self.accounts[sequence].work = self.accounts[sequence].work.checked_add(quote)?;
        }
        poll(self.control)?; self.engine.state.poisoned = false; Ok(output)
    }
}
impl<C: DecodeStepControl> Drop for Int4CohortSession<'_, '_, C> {
    fn drop(&mut self) { clear(&mut self.engine.sequences, &mut self.engine.state); }
}
fn clear(sequences: &mut [Sequence], state: &mut RunState) {
    state.poisoned |= std::thread::panicking();
    for row in sequences { row.hidden = None; row.cache.clear(); }
    state.active = false;
}
fn check_count(count: usize) -> Result<(), StrictInt4Error> {
    if count == 0 || count > MAX_INT4_SEQUENCES { Err(StrictInt4Error::Input) } else { Ok(()) }
}
fn check_indices(indices: impl Iterator<Item = usize>, count: usize) -> Result<(), StrictInt4Error> {
    let mut previous = None; let mut length = 0;
    for index in indices {
        if index >= count || previous.is_some_and(|prior| prior >= index) { return Err(StrictInt4Error::Input); }
        previous = Some(index); length += 1;
    }
    check_count(length)
}
fn reserve<T>(count: usize) -> Result<Vec<T>, StrictInt4Error> {
    let mut values = Vec::new(); values.try_reserve_exact(count).map_err(|_| StrictInt4Error::Allocation)?; Ok(values)
}
#[cfg(test)] mod tests;
