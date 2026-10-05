//! Bounded cross-sequence INT8 execution with one immutable model binding.
//!
//! A step groups linear operators, not attention contexts. Each sequence owns
//! all 44 KV slots and its own position, hidden state and work ceiling. The
//! fixed cohort uses one caller-owned controller; any native/task failure is
//! fatal to the cohort. There is no partial-success retry or dynamic admission.
//! Source implementation only; model parity and performance remain unqualified.
use super::*;
use crate::native_engine::portable_int8::batch::MAX_BATCH_ROWS;
mod execution;

pub const INT8_COHORT_EXECUTION: &str = "portable-int8-ragged-sequence-cohort-v1";

impl Int8MemoryRequirement {
    /// Entire resident cohort, including ragged KV capacities, ONE shared RoPE
    /// table, grouped activation/MLP rails and a complete grouped head output.
    /// Weights, sampler/preparation/output retention and allocator slack remain
    /// separately admitted by the host. This is not a measured RSS guarantee.
    pub fn for_cohort(contexts: &[usize]) -> Result<Self, StrictInt8Error> {
        check_count(contexts.len())?;
        let mut kv_bytes = 0_u64;
        let mut longest = 0;
        for &context in contexts {
            let row = Self::for_context(context)?;
            kv_bytes = kv_bytes.checked_add(row.kv_bytes).ok_or(StrictInt8Error::Memory)?;
            longest = longest.max(context);
        }
        let base = Self::for_context(longest)?;
        let grouped = prefill::Int8PrefillLimits::required_extra_scratch_bytes(contexts.len())?;
        let heads = (contexts.len() as u64).checked_mul((V * size_of::<f32>() + 8192) as u64)
            .ok_or(StrictInt8Error::Memory)?;
        let scratch_payload_bound = base.scratch_payload_bound.checked_add(grouped)
            .and_then(|bytes| bytes.checked_add(heads)).ok_or(StrictInt8Error::Memory)?;
        Ok(Self { kv_bytes, rope_bytes: base.rope_bytes, scratch_payload_bound })
    }
}

struct Sequence { cache: KvCache, hidden: Option<Vec<Bf16>> }

/// No cloned model payloads or per-sequence model revalidation. Membership and
/// capacities are fixed before allocation; a session cannot recycle a finished
/// sequence's quota for a new document. No worker/runtime is created here.
pub struct Int8CohortEngine<'weights> {
    weights: Int8WeightView<'weights>, rope: RopeTablesF32,
    sequences: Vec<Sequence>, activations: Vec<ActivationBuffer>, state: RunState,
}
impl<'weights> Int8CohortEngine<'weights> {
    pub fn new(weights: Int8WeightView<'weights>, contexts: &[usize], memory: Int8MemoryBudget)
        -> Result<Self, StrictInt8Error> {
        Int8MemoryRequirement::for_cohort(contexts)?.check(memory)?;
        let mut sequences = reserve(contexts.len())?;
        let mut activations = reserve(contexts.len())?;
        for &context in contexts {
            sequences.push(Sequence { cache: KvCache::try_with_capacity(context)
                .map_err(|_| StrictInt8Error::Allocation)?, hidden: None });
            activations.push(ActivationBuffer::try_new(I)?);
        }
        let longest = contexts.iter().copied().max().ok_or(StrictInt8Error::Context)?;
        Ok(Self { weights, rope: RopeTablesF32::nanbeige(longest).map_err(|_| StrictInt8Error::Rope)?,
            sequences, activations, state: RunState::default() })
    }
    pub fn artifact_identity(&self) -> &ArtifactIdentity { self.weights.identity }
    pub fn sequence_count(&self) -> usize { self.sequences.len() }
    pub fn capacity(&self, sequence: usize) -> Result<usize, StrictInt8Error> {
        Ok(self.sequences.get(sequence).ok_or(StrictInt8Error::Input)?.cache.capacity_positions())
    }
    pub fn is_poisoned(&self) -> bool { self.state.poisoned }
    pub fn check_idle(&self) -> Result<(), StrictInt8Error> {
        if self.state.active || self.state.poisoned || self.sequences.iter()
            .any(|row| row.hidden.is_some() || !row.cache.all_slots_have_len(0)) {
            return Err(StrictInt8Error::EngineUnavailable);
        }
        Ok(())
    }
    /// Every row receives its own non-renewable ceiling. All allocations and
    /// budget sums are checked before marking the engine active.
    pub fn session<'run, C: DecodeStepControl>(&'run mut self, budgets: &[Int8RunBudget], control: &'run mut C)
        -> Result<Int8CohortSession<'run, 'weights, C>, StrictInt8Error> {
        self.check_idle()?;
        if budgets.len() != self.sequences.len() { return Err(StrictInt8Error::Input); }
        let mut accounts = reserve(budgets.len())?;
        let mut projections = ProjectionWork::default();
        for &budget in budgets {
            projections = projections.checked_add(budget.max_projection_work)?;
            accounts.push(Account { budget, work: Int8Work::default() });
        }
        self.state.open()?;
        Ok(Int8CohortSession { engine: self, control, accounts, ledger: ProjectionLedger::new(projections) })
    }
}
impl CurrentCandidateInt8Model {
    /// Bind the resident materialization ONCE for all cohort rows. This does
    /// not grant artifact authenticity or process-memory admission authority.
    pub fn cohort_engine(&self, contexts: &[usize], memory: Int8MemoryBudget)
        -> Result<Int8CohortEngine<'_>, CurrentCandidateInt8Error> {
        Int8MemoryRequirement::for_cohort(contexts)?.check(memory)?;
        let weights = Int8WeightView::from_materialized(&self.loaded.weights)?;
        Ok(Int8CohortEngine::new(weights, contexts, memory)?)
    }
}

/// One token for a stable physical sequence slot, not a sampler identity.
/// Steps must list strictly increasing slots; inactive slots are simply absent.
#[derive(Clone, Copy, Debug)]
pub struct CohortToken { pub sequence: usize, pub token: u32 }

struct Account { budget: Int8RunBudget, work: Int8Work }
impl Account {
    fn quote(&self, start: usize, capacity: usize, positions: usize, heads: usize)
        -> Result<Int8Work, StrictInt8Error> {
        if start.checked_add(positions).is_none_or(|end| end > capacity) { return Err(StrictInt8Error::Context); }
        let delta = Int8Work::for_sequence(start, positions, heads)?;
        let total = self.work.checked_add(delta)?;
        if total.forward_positions > self.budget.max_forward_positions
            || total.attention_pairs > self.budget.max_attention_pairs
            || !total.projections.fits(self.budget.max_projection_work) { return Err(StrictInt8Error::Work); }
        Ok(delta)
    }
}

/// One exclusive RAII session for the entire cohort. Preflight refusals do not
/// mutate KV. Once native work starts, cancellation, errors and unwinding poison
/// the engine; Drop clears every sequence's 44 slots and retained hidden state.
pub struct Int8CohortSession<'run, 'weights, C: DecodeStepControl> {
    engine: &'run mut Int8CohortEngine<'weights>, control: &'run mut C,
    accounts: Vec<Account>, ledger: ProjectionLedger,
}
impl<C: DecodeStepControl> Int8CohortSession<'_, '_, C> {
    pub(crate) fn control(&mut self) -> &mut C { &mut *self.control }
    pub(crate) fn abort(&mut self) { self.engine.state.poisoned = true; }
    pub fn position(&self, sequence: usize) -> Result<usize, StrictInt8Error> {
        self.engine.state.check()?;
        let cache = &self.engine.sequences.get(sequence).ok_or(StrictInt8Error::Input)?.cache;
        let position = cache.len_for_slot(0).map_err(|_| StrictInt8Error::Cache)?;
        if !cache.all_slots_have_len(position) { return Err(StrictInt8Error::Cache); }
        Ok(position)
    }
    pub fn preflight(&self, sequence: usize, positions: usize, head_rows: usize) -> Result<Int8Work, StrictInt8Error> {
        let start = self.position(sequence)?;
        let work = self.accounts[sequence].quote(start, self.engine.capacity(sequence)?, positions, head_rows)?;
        self.ledger.preflight(work.projections)?;
        Ok(work)
    }
    /// Complete, reconciled per-sequence work. Partial native failures cannot
    /// be presented as successful completed per-row execution receipts.
    pub fn work(&self, sequence: usize) -> Result<Int8Work, StrictInt8Error> {
        self.engine.state.check()?;
        let sum = self.accounts.iter().try_fold(ProjectionWork::default(), |sum, row|
            sum.checked_add(row.work.projections))?;
        if sum != self.ledger.reserved() { return Err(StrictInt8Error::Boundary); }
        let row = self.accounts.get(sequence).ok_or(StrictInt8Error::Input)?;
        if self.position(sequence)? as u64 != row.work.forward_positions { return Err(StrictInt8Error::Boundary); }
        Ok(row.work)
    }
    /// One shared-weight decoder step over any nonempty active subset. Prompt
    /// rows and decode rows can coexist because each carries its own causal KV
    /// position. Linear geometry, not equal context length, defines this group.
    pub fn append_group(&mut self, steps: &[CohortToken]) -> Result<(), StrictInt8Error> {
        self.engine.state.check()?;
        check_indices(steps.iter().map(|step| step.sequence), self.accounts.len())?;
        if steps.iter().any(|step| step.token as usize >= V) { return Err(StrictInt8Error::Input); }
        let mut positions = reserve(steps.len())?;
        let mut quotes = reserve(steps.len())?;
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
            let source = self.engine.weights.embeddings.row(step.token as usize).map_err(|_| StrictInt8Error::Input)?;
            let mut row = filled(H, Bf16::from_bits(0))?; row.copy_from_slice(source);
            finite(&row)?; hidden.push(row);
            self.engine.sequences[step.sequence].hidden = None;
        }
        let engine = &mut *self.engine;
        let runner = LoopRunner::from_layer_weights(&engine.weights.layers);
        let mut executor = execution::Executor { final_norm: engine.weights.final_norm, rope: &engine.rope,
            sequences: &mut engine.sequences, activations: &mut engine.activations[..steps.len()],
            steps, positions: &positions, ledger: &mut self.ledger, control: &mut *self.control, completed: 0, norms: 0 };
        runner.run_group(&mut executor, &mut hidden)?;
        if executor.completed != KV_SLOT_COUNT || executor.norms != 2 { return Err(StrictInt8Error::Boundary); }
        for (((step, position), row), quote) in steps.iter().zip(positions).zip(hidden).zip(quotes) {
            let target = &mut engine.sequences[step.sequence];
            if !target.cache.all_slots_have_len(position + 1) { return Err(StrictInt8Error::Cache); }
            finite(&row)?; target.hidden = Some(row);
            self.accounts[step.sequence].work = self.accounts[step.sequence].work.checked_add(quote)?;
        }
        poll(self.control)?; engine.state.poisoned = false; Ok(())
    }
    /// Shared lm-head over only rows that need selection. Output is sequence-
    /// selection-major, with checked_count(V) contiguous values per input slot.
    /// Retaining multiple returned outputs needs caller-owned extra admission.
    pub fn logits_group(&mut self, sequences: &[usize], rows: LinearRows<'_>) -> Result<Vec<f32>, StrictInt8Error> {
        self.engine.state.check()?;
        check_indices(sequences.iter().copied(), self.accounts.len())?;
        let count = rows.checked_count(V)?;
        let mut quotes = reserve(sequences.len())?;
        let mut total = ProjectionWork::default();
        for &sequence in sequences {
            if self.engine.sequences[sequence].hidden.is_none() { return Err(StrictInt8Error::EmptyHidden); }
            let quote = self.preflight(sequence, 0, count)?;
            total = total.checked_add(quote.projections)?; quotes.push(quote);
        }
        self.ledger.preflight(total)?;
        let length = count.checked_mul(sequences.len()).ok_or(StrictInt8Error::Memory)?;
        let mut output = filled(length, 0.0_f32)?;
        self.engine.state.poisoned = true;
        for (input, &sequence) in self.engine.activations.iter_mut().zip(sequences) {
            poll(self.control)?;
            input.encode_bf16(self.engine.sequences[sequence].hidden.as_deref().ok_or(StrictInt8Error::EmptyHidden)?)?;
        }
        self.engine.weights.head.project_batch_f32_into(&self.engine.activations[..sequences.len()],
            rows, &mut output, &mut self.ledger, &mut *self.control)?;
        for (&sequence, quote) in sequences.iter().zip(quotes) {
            self.accounts[sequence].work = self.accounts[sequence].work.checked_add(quote)?;
        }
        poll(self.control)?; self.engine.state.poisoned = false; Ok(output)
    }
}
impl<C: DecodeStepControl> Drop for Int8CohortSession<'_, '_, C> {
    fn drop(&mut self) { clear(&mut self.engine.sequences, &mut self.engine.state); }
}
fn clear(sequences: &mut [Sequence], state: &mut RunState) {
    state.poisoned |= std::thread::panicking();
    for row in sequences { row.hidden = None; row.cache.clear(); }
    state.active = false;
}
fn check_count(count: usize) -> Result<(), StrictInt8Error> {
    if count == 0 || count > MAX_BATCH_ROWS { Err(StrictInt8Error::Input) } else { Ok(()) }
}
fn check_indices(indices: impl Iterator<Item = usize>, count: usize) -> Result<(), StrictInt8Error> {
    let mut previous = None; let mut length = 0;
    for index in indices {
        if index >= count || previous.is_some_and(|prior| prior >= index) { return Err(StrictInt8Error::Input); }
        previous = Some(index); length += 1;
    }
    check_count(length)
}
fn reserve<T>(count: usize) -> Result<Vec<T>, StrictInt8Error> {
    let mut values = Vec::new(); values.try_reserve_exact(count).map_err(|_| StrictInt8Error::Allocation)?; Ok(values)
}
#[cfg(test)] mod tests;
