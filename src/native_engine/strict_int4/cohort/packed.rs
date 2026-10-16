//! Bounded layer-major token packs spanning independently addressed INT4 sequences.
//! Linear stages share prompt tokens across documents, while attention appends
//! one token at a time to its own causal prefix. No new math program or runtime.
use super::*;
use crate::native_engine::strict_int4::prefill::Int4PrefillLimits;

pub const INT4_PACKED_COHORT_EXECUTION: &str = "grouped-int4-ragged-token-morsel-v1";

/// One physical sequence and its consecutive exact tokens. Runs are nonempty
/// and strictly ordered by sequence. The total token count, not the number of
/// documents, must fit the separately admitted prompt-workspace declaration.
#[derive(Clone, Copy, Debug)]
pub struct Int4TokenRun<'a> { pub sequence: usize, pub tokens: &'a [u32] }

impl<C: DecodeStepControl> Int4CohortSession<'_, '_, C> {
    /// Append one finite token pack through both loops and final norms. No head
    /// is projected here. The caller may mix prompt runs with one-token decode
    /// runs, but must not supply feedback that has not yet been sampled.
    ///
    /// Every run's tokens, context and remaining work are checked before any
    /// cache/hidden mutation. Once execution begins, any failure poisons the
    /// whole session; ordinary Drop clears every cache, including inactive rows.
    /// Extra packed rails are caller-admitted IN ADDITION to the cohort envelope.
    pub fn append_packed(&mut self, runs: &[Int4TokenRun<'_>], limits: Int4PrefillLimits)
        -> Result<(), StrictInt4Error> {
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
                steps.push(Int4CohortToken { sequence: run.sequence, token });
                positions.push(start.checked_add(offset).ok_or(StrictInt4Error::Context)?);
            }
        }
        self.ledger.preflight(total)?;
        // This storage is separate from the engine's one-activation-per-sequence
        // rails. It is bounded by the explicit extra workspace, not by KV count.
        let mut activations = reserve(count)?;
        for _ in 0..count { activations.push(ActivationBuffer::try_new(I)?); }
        let mut hidden = reserve(count)?;
        self.engine.state.poisoned = true;
        poll(self.control)?;
        for run in runs { self.engine.sequences[run.sequence].hidden = None; }
        for step in &steps {
            poll(self.control)?;
            let source = self.engine.weights.embeddings.row(step.token as usize)
                .map_err(|_| StrictInt4Error::Input)?;
            let mut row = filled(H, Bf16::from_bits(0))?;
            row.copy_from_slice(source); finite(&row)?; hidden.push(row);
        }
        let engine = &mut *self.engine;
        let runner = LoopRunner::from_layer_weights(&engine.weights.layers);
        let mut executor = execution::Executor {
            final_norm: &engine.weights.final_norm, rope: &engine.rope,
            sequences: &mut engine.sequences, activations: &mut activations,
            unpack: &mut engine.unpack, kernel: engine.kernel,
            steps: &steps, positions: &positions, ledger: &mut self.ledger,
            control: &mut *self.control, completed: 0, norms: 0,
        };
        runner.run_group(&mut executor, &mut hidden)?;
        if executor.completed != KV_SLOT_COUNT || executor.norms != 2 { return Err(StrictInt4Error::Boundary); }
        // Check final lengths, not every intermediate position against a cache
        // which now contains the entire pack. All 44 slots must agree per run.
        let mut offset = 0;
        for (run, quote) in runs.iter().zip(quotes) {
            let end = positions[offset].checked_add(run.tokens.len()).ok_or(StrictInt4Error::Context)?;
            if !engine.sequences[run.sequence].cache.all_slots_have_len(end) { return Err(StrictInt4Error::Cache); }
            self.accounts[run.sequence].work = self.accounts[run.sequence].work.checked_add(quote)?;
            offset += run.tokens.len();
        }
        // Ordered overwrite retains only the final hidden row per document.
        // Earlier hidden rows remain staging and can never reach a head call.
        for (step, row) in steps.iter().zip(hidden) {
            finite(&row)?; engine.sequences[step.sequence].hidden = Some(row);
        }
        poll(self.control)?; engine.state.poisoned = false; Ok(())
    }
}

fn check_runs(runs: &[Int4TokenRun<'_>], sequences: usize, limits: Int4PrefillLimits)
    -> Result<usize, StrictInt4Error> {
    limits.validate()?;
    check_indices(runs.iter().map(|run| run.sequence), sequences)?;
    let mut total = 0_usize;
    for run in runs {
        if run.tokens.is_empty() { return Err(StrictInt4Error::Input); }
        total = total.checked_add(run.tokens.len()).filter(|&n| n <= limits.max_batch_rows)
            .ok_or(StrictInt4Error::Memory)?;
        if run.tokens.iter().any(|&token| token as usize >= V) { return Err(StrictInt4Error::Input); }
    }
    Ok(total)
}

#[cfg(test)] mod tests;
pub mod refill;

