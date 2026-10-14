//! Explicit INT4 prompt scheduling with the existing generation cursor.
//! No change to semantic identity, addressed sampling, stops or output policy.
//! Callers must separately admit the extra native workspace before execution.
use super::*;
use crate::native_engine::strict_int4::prefill::Int4PrefillLimits;

impl Int4GenerationPlan {
    #[allow(clippy::too_many_arguments)]
    pub fn execute_layer_major<D: DecodeByteDecoder, C: DecodeStepControl>(&self,
        admitted: &ExecutionIdentity, engine: &mut StrictInt4Engine<'_>, decoder: &D,
        request_seq: u64, budget: Int4GenerationBudget, prefill: Int4PrefillLimits, control: &mut C)
        -> Result<Int4GenerationRun, Int4GenerationError> {
        self.execute_layer_major_with_sink(admitted, engine, decoder, request_seq, budget, prefill, &mut Discard, control)
    }
    /// Preflight precedes session creation. Prompt failure emits no token;
    /// later decoder/sink failure aborts the same session without a retry.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_layer_major_with_sink<D: DecodeByteDecoder, S: DecodeEventSink, C: DecodeStepControl>(&self,
        admitted: &ExecutionIdentity, engine: &mut StrictInt4Engine<'_>, decoder: &D,
        request_seq: u64, budget: Int4GenerationBudget, prefill: Int4PrefillLimits, sink: &mut S, control: &mut C)
        -> Result<Int4GenerationRun, Int4GenerationError> {
        prefill.validate()?;
        self.preflight(admitted, engine, budget)?;
        let mut session = engine.session(budget.native, control)?;
        self.drive_layer_major(decoder, request_seq, sink, &mut session, prefill)
    }
    fn drive_layer_major<D: DecodeByteDecoder, S: DecodeEventSink, B: PrefillDriver>(&self,
        decoder: &D, request_seq: u64, sink: &mut S, driver: &mut B, limits: Int4PrefillLimits)
        -> Result<Int4GenerationRun, Int4GenerationError> {
        limits.validate()?;
        let result = (|| {
            let mut row = cursor::Cursor::new(&self.plan, request_seq, INT4_GENERATION_VERSION)?;
            row.before_forward(driver.control())?;
            driver.append_prompt(&self.plan.prompt, limits)?;
            // Reconcile logical positions already executed by the native
            // morsels. No repeated forward, intermediate head or random draw.
            for index in 0..self.plan.prompt.len() {
                if index + 1 == self.plan.prompt.len() {
                    let logits = driver.logits()?;
                    check_logits(&logits)?;
                    row.record_forward(true)?;
                    row.emit_next(&logits, decoder, sink, driver.control())?;
                } else { row.record_forward(false)?; }
            }
            while !row.done {
                row.before_forward(driver.control())?;
                let (token, needs_selection) = row.next_token()?;
                if !needs_selection { return Err(GenerationError::Contract("INT4 prefill cursor reconciliation").into()); }
                driver.append(token)?;
                let logits = driver.logits()?;
                check_logits(&logits)?;
                row.record_forward(true)?;
                row.emit_next(&logits, decoder, sink, driver.control())?;
            }
            let sequence = row.finish()?;
            let model_work = driver.work();
            check_completed(self.group_width, sequence.native_work, model_work)?;
            Ok(Int4GenerationRun { schema_version: 1, sequence, model_work })
        })();
        if result.is_err() { driver.abort(); }
        result
    }
}
trait PrefillDriver: Driver {
    fn append_prompt(&mut self, tokens: &[u32], limits: Int4PrefillLimits) -> Result<(), Int4GenerationError>;
}
impl<C: DecodeStepControl> PrefillDriver for Int4Session<'_, '_, C> {
    fn append_prompt(&mut self, tokens: &[u32], limits: Int4PrefillLimits) -> Result<(), Int4GenerationError> {
        self.append_layer_major(tokens, limits).map_err(Into::into)
    }
}

#[cfg(test)] mod tests;
