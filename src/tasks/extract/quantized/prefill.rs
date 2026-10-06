//! Explicit scheduling for the same sealed extraction plan and finalizer.
use super::*;
use crate::native_engine::strict_int8::prefill::Int8PrefillLimits;

impl Int8ExtractPlan {
    /// Opt into bounded prompt morsels without rebinding the task or changing
    /// its full/selected-row head policy. The host must separately reserve the
    /// extra scratch allowance; this argument is not an admission permit.
    pub fn execute_layer_major<C: DecodeStepControl>(&self, engine: &mut StrictInt8Engine<'_>,
        admitted: &ExecutionIdentity, vocabulary: &ExtractionVocabulary, budget: Int8JsonBudget,
        prefill: Int8PrefillLimits, control: &mut C) -> Result<Int8ExtractRun, Int8ExtractError> {
        self.execute_layer_major_with(engine, admitted, vocabulary, budget, prefill, control, Ok)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_layer_major_with<C, T, E, F>(&self, engine: &mut StrictInt8Engine<'_>,
        admitted: &ExecutionIdentity, vocabulary: &ExtractionVocabulary, budget: Int8JsonBudget,
        prefill: Int8PrefillLimits, control: &mut C, finalize: F) -> Result<T, E>
    where C: DecodeStepControl, E: From<Int8ExtractError> + From<Int8JsonError>,
        F: FnOnce(Int8ExtractRun) -> Result<T, E> {
        self.execute_with_prefill(engine, admitted, vocabulary, budget, Some(prefill), control, finalize)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn execute_with_prefill<C, T, E, F>(&self, engine: &mut StrictInt8Engine<'_>,
        admitted: &ExecutionIdentity, vocabulary: &ExtractionVocabulary, mut budget: Int8JsonBudget,
        prefill: Option<Int8PrefillLimits>, control: &mut C, finalize: F) -> Result<T, E>
    where C: DecodeStepControl, E: From<Int8ExtractError> + From<Int8JsonError>,
        F: FnOnce(Int8ExtractRun) -> Result<T, E> {
        if let Some(limits) = prefill { limits.validate().map_err(Int8JsonError::from).map_err(E::from)?; }
        self.verify_identity(admitted).map_err(E::from)?;
        if vocabulary.controls != self.extraction.controls {
            return Err(E::from(Int8ExtractError::from(ExtractError::Contract(
                "vocabulary control registry differs from int8 plan"))));
        }
        budget.json.max_kv_bytes = budget.json.max_kv_bytes.min(self.extraction.max_kv_bytes);
        let finish = |run| finalize(self.finalize(run).map_err(E::from)?);
        match (self.selected_rows, prefill) {
            (Some(rows), Some(prefill)) => selected_head::prefill::decode_json_int8_sparse_layer_major_with(
                engine, admitted, &self.extraction.prompt, &self.extraction.program, &vocabulary.oracle,
                &self.extraction.options, budget, rows, prefill, control, finish),
            (None, Some(prefill)) => constrained_int8::prefill::decode_json_int8_layer_major_with(
                engine, admitted, &self.extraction.prompt, &self.extraction.program, &vocabulary.oracle,
                &self.extraction.options, budget, prefill, control, finish),
            (Some(rows), None) => selected_head::decode_json_int8_sparse_with(engine, admitted, &self.extraction.prompt,
                &self.extraction.program, &vocabulary.oracle, &self.extraction.options, budget, rows, control, finish),
            (None, None) => constrained_int8::decode_json_int8_with(engine, admitted, &self.extraction.prompt,
                &self.extraction.program, &vocabulary.oracle, &self.extraction.options, budget, control, finish),
        }
    }
}
