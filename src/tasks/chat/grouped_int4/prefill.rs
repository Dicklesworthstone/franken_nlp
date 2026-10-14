//! The same pinned task completion/finalizer with explicitly admitted prefill.
use super::*;
use crate::native_engine::strict_int4::prefill::Int4PrefillLimits;

impl PreparedInt4Chat {
    #[allow(clippy::too_many_arguments)]
    pub fn execute_layer_major<C: DecodeStepControl>(&self, admitted: &ExecutionIdentity,
        engine: &mut StrictInt4Engine<'_>, request_seq: u64, budget: Int4GenerationBudget,
        prefill: Int4PrefillLimits, control: &mut C) -> Result<Int4ChatResult, Int4ChatError> {
        let raw = self.native.execute_layer_major(admitted, engine, self.tokenizer.tokenizer(),
            request_seq, self.task_budget(budget), prefill, control)?;
        self.finish(raw)
    }
    /// Events are provisional until independent decoding, UTF-8, termination,
    /// exact native work and complete task-size checks all pass in finish.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_layer_major_with_sink<S: DecodeEventSink, C: DecodeStepControl>(&self,
        admitted: &ExecutionIdentity, engine: &mut StrictInt4Engine<'_>, request_seq: u64,
        budget: Int4GenerationBudget, prefill: Int4PrefillLimits, sink: &mut S, control: &mut C)
        -> Result<Int4ChatResult, Int4ChatError> {
        let raw = self.native.execute_layer_major_with_sink(admitted, engine, self.tokenizer.tokenizer(),
            request_seq, self.task_budget(budget), prefill, sink, control)?;
        self.finish(raw)
    }
}
