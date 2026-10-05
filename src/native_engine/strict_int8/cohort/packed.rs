//! Layer-major token morsels spanning independently addressed documents.
//!
//! A run may contain several consecutive prompt positions or one decode token.
//! Linear operators share the complete finite token pack; attention still
//! appends ONE token and reads only its own document's causal prefix. The host
//! separately admits Int8PrefillLimits on top of the resident cohort envelope.
use super::*;
use crate::native_engine::strict_int8::prefill::Int8PrefillLimits;

pub const INT8_PACKED_COHORT_EXECUTION: &str = "portable-int8-ragged-token-morsel-cohort-v1";

/// Stable sequence slot plus consecutive exact tokens. Runs must be nonempty
/// and strictly increasing by slot; the TOTAL number of tokens is bounded.
#[derive(Clone, Copy, Debug)]
pub struct CohortTokenRun<'a> { pub sequence: usize, pub tokens: &'a [u32] }

impl<C: DecodeStepControl> Int8CohortSession<'_, '_, C> {
    /// Append a finite ragged token pack through both complete decoder loops.
    /// No vocabulary projection is performed. All tokens, contexts, per-row
    /// work and aggregate work are preflighted before any native mutation.
    /// Native errors poison the entire ordinary cohort session; no partial
    /// row completion, fallback or independently renewable allowance exists.
    pub fn append_packed(&mut self, runs: &[CohortTokenRun<'_>], limits: Int8PrefillLimits)
        -> Result<(), StrictInt8Error> {
        self.engine.state.check()?;
        let count = check_runs(runs, self.accounts.len(), limits)?;
        let mut steps = reserve(count)?;
        let mut positions = reserve(count)?;
        let mut quotes = reserve(runs.len())?;
        let mut total = ProjectionWork::default();
        for run in runs {
            let start = self.position(run.sequence)?;
            let quote = self.preflight(run.sequence, run.tokens.len(), 0)?;
            total = total.checked_add(quote.projections)?;
            quotes.push(quote);
            for (offset, &token) in run.tokens.iter().enumerate() {
                steps.push(CohortToken { sequence: run.sequence, token });
                positions.push(start.checked_add(offset).ok_or(StrictInt8Error::Context)?);
            }
        }
        self.ledger.preflight(total)?;
        // This extra cohort is NOT the engine's one-activation-per-document
        // storage. Its complete modeled payload was explicitly admitted above.
        let mut activations = reserve(count)?;
        for _ in 0..count { activations.push(ActivationBuffer::try_new(I)?); }
        let mut hidden = reserve(count)?;
        self.engine.state.poisoned = true;
        poll(self.control)?;
        for run in runs { self.engine.sequences[run.sequence].hidden = None; }
        for step in &steps {
            poll(self.control)?;
            let source = self.engine.weights.embeddings.row(step.token as usize)
                .map_err(|_| StrictInt8Error::Input)?;
            let mut row = filled(H, Bf16::from_bits(0))?;
            row.copy_from_slice(source); finite(&row)?; hidden.push(row);
        }
        let engine = &mut *self.engine;
        let runner = LoopRunner::from_layer_weights(&engine.weights.layers);
        let mut executor = execution::Executor {
            final_norm: engine.weights.final_norm, rope: &engine.rope,
            sequences: &mut engine.sequences, activations: &mut activations,
            steps: &steps, positions: &positions, ledger: &mut self.ledger,
            control: &mut *self.control, completed: 0, norms: 0,
        };
        runner.run_group(&mut executor, &mut hidden)?;
        if executor.completed != KV_SLOT_COUNT || executor.norms != 2 { return Err(StrictInt8Error::Boundary); }
        // All 44 slots must have reached EACH document's final packed token.
        // Only the last hidden row of a document survives for future logits.
        let mut offset = 0;
        for (run, quote) in runs.iter().zip(quotes) {
            let end = positions[offset].checked_add(run.tokens.len()).ok_or(StrictInt8Error::Context)?;
            if !engine.sequences[run.sequence].cache.all_slots_have_len(end) { return Err(StrictInt8Error::Cache); }
            self.accounts[run.sequence].work = self.accounts[run.sequence].work.checked_add(quote)?;
            offset += run.tokens.len();
        }
        for (step, row) in steps.iter().zip(hidden) {
            finite(&row)?; engine.sequences[step.sequence].hidden = Some(row);
        }
        poll(self.control)?; engine.state.poisoned = false; Ok(())
    }
}

fn check_runs(runs: &[CohortTokenRun<'_>], sequences: usize, limits: Int8PrefillLimits)
    -> Result<usize, StrictInt8Error> {
    limits.validate()?;
    check_indices(runs.iter().map(|run| run.sequence), sequences)?;
    let mut total = 0_usize;
    for run in runs {
        if run.tokens.is_empty() { return Err(StrictInt8Error::Input); }
        total = total.checked_add(run.tokens.len()).filter(|&n| n <= limits.max_batch_rows)
            .ok_or(StrictInt8Error::Memory)?;
        if run.tokens.iter().any(|&token| token as usize >= V) { return Err(StrictInt8Error::Input); }
    }
    Ok(total)
}

#[cfg(test)] mod tests;
pub mod refill;
