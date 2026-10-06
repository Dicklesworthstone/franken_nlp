//! Opt-in layer-major prompt processing for grammar-first selected-row JSON.
//!
//! Only prompt scheduling differs from the serial driver. Grammar traversal,
//! token selection, explicit EOS, source validation, work reconciliation and
//! finalization all run through the existing constrained decoder. The host
//! must retain a separate extra-scratch reservation through native completion.
//! No model-parity, performance or release-qualification claim is made here.

use super::*;
use crate::native_engine::strict_int8::prefill::Int8PrefillLimits;

/// Execute bounded prompt morsels using the existing INT8 batch kernels, then
/// continue single-token constrained decoding. Scheduling does not change the
/// semantic identity or output contract. The old serial entrypoint is unchanged.
#[allow(clippy::too_many_arguments)]
pub fn decode_json_int8_sparse_layer_major<C: DecodeStepControl>(
    engine: &mut StrictInt8Engine<'_>, identity: &ExecutionIdentity, prompt: &[u32],
    program: &JsonProgram, vocabulary: &VocabMaskOracle, options: &JsonDecodeOptions,
    budget: Int8JsonBudget, limits: Int8JsonSparseLimits, prefill: Int8PrefillLimits, control: &mut C,
) -> Result<Int8JsonRun, Int8JsonError> {
    decode_json_int8_sparse_layer_major_with(engine, identity, prompt, program, vocabulary,
        options, budget, limits, prefill, control, Ok)
}

/// Keep task-specific final validation inside the SAME RAII session. Native,
/// grammar, cancellation and finalizer failures all abort before guards leave
/// scope; no second session, model load, retry or partial-success path exists.
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_json_int8_sparse_layer_major_with<C, T, E, F>(
    engine: &mut StrictInt8Engine<'_>, identity: &ExecutionIdentity, prompt: &[u32],
    program: &JsonProgram, vocabulary: &VocabMaskOracle, options: &JsonDecodeOptions,
    budget: Int8JsonBudget, limits: Int8JsonSparseLimits, prefill: Int8PrefillLimits, control: &mut C, finalize: F,
) -> Result<T, E>
where C: DecodeStepControl, E: From<Int8JsonError>, F: FnOnce(Int8JsonRun) -> Result<T, E> {
    prefill.validate().map_err(Int8JsonError::from).map_err(E::from)?;
    check_request(engine, identity, prompt, vocabulary, options, budget, limits).map_err(E::from)?;
    let mut session = engine.session(budget.native, control).map_err(Int8JsonError::from).map_err(E::from)?;
    drive_layer_major(prompt, program, vocabulary, options, budget, limits, prefill, &mut session, finalize)
}

// Private static seam: external callers must supply a real admitted engine.
trait PromptDriver: Driver {
    fn append_layer_major(&mut self, prompt: &[u32], limits: Int8PrefillLimits) -> Result<(), Int8JsonError>;
}
impl<C: DecodeStepControl> PromptDriver for Int8Session<'_, '_, C> {
    fn append_layer_major(&mut self, prompt: &[u32], limits: Int8PrefillLimits) -> Result<(), Int8JsonError> {
        Int8Session::append_layer_major(self, prompt, limits).map_err(Into::into)
    }
}

struct LayerMajor<'a, D> { driver: &'a mut D, prefill: Int8PrefillLimits }
impl<D: PromptDriver> Driver for LayerMajor<'_, D> {
    type Control = D::Control;
    fn control(&mut self) -> &mut Self::Control { self.driver.control() }
    fn append(&mut self, token: u32) -> Result<(), Int8JsonError> { self.driver.append(token) }
    fn append_prompt(&mut self, prompt: &[u32]) -> Result<(), Int8JsonError> {
        // Preserve caller-defined prompt-position cancellation checks. The
        // native morsels additionally poll throughout projections/attention.
        for index in 0..prompt.len() {
            if let Some(cause) = self.driver.control().prefill_checkpoint(index) {
                return Err(JsonDecodeError::Cancelled(cause).into());
            }
        }
        self.driver.append_layer_major(prompt, self.prefill)
    }
    fn logits(&mut self, rows: &[u32]) -> Result<Vec<f32>, Int8JsonError> { self.driver.logits(rows) }
    fn work(&self) -> Int8Work { self.driver.work() }
    fn abort(&mut self) { self.driver.abort(); }
}

#[allow(clippy::too_many_arguments)]
fn drive_layer_major<V, D, T, E, F>(prompt: &[u32], program: &JsonProgram, vocabulary: &V,
    options: &JsonDecodeOptions, budget: Int8JsonBudget, limits: Int8JsonSparseLimits, prefill: Int8PrefillLimits,
    driver: &mut D, finalize: F) -> Result<T, E>
where V: Vocabulary, D: PromptDriver, E: From<Int8JsonError>, F: FnOnce(Int8JsonRun) -> Result<T, E> {
    prefill.validate().map_err(Int8JsonError::from).map_err(E::from)?;
    drive(prompt, program, vocabulary, options, budget, limits, &mut LayerMajor { driver, prefill }, finalize)
}

#[cfg(test)] mod tests;
