//! One physical scheduling choice for BOTH stages of automatic discovery.
//! Exact native/task identity and all complete-result checks remain in their
//! existing executors. No alternate model, runtime, logits or public test seam.
use super::*;
use crate::corpus::native_resolve::quantized::PreparedInt8Resolution;

pub(super) fn configure(current: &mut Option<Int8PrefillLimits>, limits: Int8PrefillLimits)
    -> Result<(), Int8EntityError> {
    limits.validate()?;
    if current.is_some() { return Err(Int8EntityError::InvalidLimits); }
    *current = Some(limits);
    Ok(())
}

pub(super) fn source<C: DecodeStepControl>(plan: &PreparedInt8SourceTask, engine: &mut StrictInt8Engine<'_>,
    vocabulary: &ExtractionVocabulary, budget: Int8JsonBudget, prefill: Option<Int8PrefillLimits>, control: &mut C)
    -> Result<Int8SourceTaskRun, Int8EntityError> {
    Ok(match prefill {
        Some(limits) => plan.execute_layer_major_with_control(plan.execution_identity(), engine, vocabulary, budget, limits, control)?,
        None => plan.execute_with_control(plan.execution_identity(), engine, vocabulary, budget, control)?,
    })
}

pub(super) fn resolution<C: DecodeStepControl>(plan: PreparedInt8Resolution<'_, '_, '_>, admitted: &[ExecutionIdentity],
    engine: &mut StrictInt8Engine<'_>, budget: Int8ScoringBudget, prefill: Option<Int8PrefillLimits>, control: &mut C)
    -> Result<Int8ResolutionRun, Int8EntityError> {
    Ok(match prefill {
        Some(limits) => plan.execute_layer_major_with_control(admitted, engine, budget, limits, control)?,
        None => plan.execute_with_control(admitted, engine, budget, control)?,
    })
}

#[cfg(test)] mod tests;
