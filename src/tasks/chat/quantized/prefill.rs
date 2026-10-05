//! Explicit prefill strategy; the pinned planner and finalizer stay unchanged.
use super::*;
use crate::native_engine::strict_int8::prefill::Int8PrefillLimits;

impl PreparedInt8Chat {
    /// Caller admits extra prefill scratch in addition to the native budget.
    /// Prepared input, generation identity and raw score space are unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_layer_major<C: DecodeStepControl>(&self, admitted: &ExecutionIdentity,
        engine: &mut StrictInt8Engine<'_>, request_seq: u64, budget: Int8GenerationBudget,
        prefill: Int8PrefillLimits, control: &mut C) -> Result<Int8ChatResult, Int8ChatError> {
        let raw = self.native.execute_layer_major(admitted, engine, self.tokenizer.tokenizer(),
            request_seq, self.task_budget(budget), prefill, control)?;
        self.finish(raw)
    }

    /// Provisional token events use exactly the ordinary reserve/permit seam.
    /// Only independent full task finalization can return successful content.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_layer_major_with_sink<S: DecodeEventSink, C: DecodeStepControl>(&self,
        admitted: &ExecutionIdentity, engine: &mut StrictInt8Engine<'_>, request_seq: u64,
        budget: Int8GenerationBudget, prefill: Int8PrefillLimits, sink: &mut S, control: &mut C)
        -> Result<Int8ChatResult, Int8ChatError> {
        let raw = self.native.execute_layer_major_with_sink(admitted, engine, self.tokenizer.tokenizer(),
            request_seq, self.task_budget(budget), prefill, sink, control)?;
        self.finish(raw)
    }
}
