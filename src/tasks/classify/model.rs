//! Explicit prompt-bound model seam for embedders and model-free wiring tests.
//! The concrete eager runner remains the only native-work receipt producer.

use crate::{
    execution_identity::ExecutionIdentity,
    native_engine::{decode::{DecodeCancellationKind, DecodeStepControl},
        lmhead::scoring::{CandidateLogits, ProjectionRows, ScoringMode}},
    tasks::ir::PromptSegment,
};
use super::{ClassificationError, ClassificationPlanningError, ClassificationTaskResult,
    PreparedClassification, planning};

/// Trusted embedding boundary. Every projection must use the exact supplied
/// prompt, continuation prefix and requested full-vocabulary rows. The head
/// index identifies a head within this immutable request, not cache authority.
/// Implementations own model/runtime admission; no external error text enters
/// the result. This interface is never deserialized from request JSON.
pub trait ClassificationLogits {
    type Error: std::fmt::Display;
    fn project(&mut self, head_index: usize, prompt: &[PromptSegment],
        prefix: &[u32], rows: ProjectionRows<'_>) -> Result<Vec<f32>, Self::Error>;
}

impl PreparedClassification {
    /// Semantic execution using the same compiled scorers and finalizers as
    /// native inference. All heads complete or no result is returned. This
    /// produces no native/KV/performance claim about the supplied backend.
    pub fn execute_with_logits<M: ClassificationLogits, C: DecodeStepControl>(&self,
        admitted: &ExecutionIdentity, model: &mut M, control: &mut C)
        -> Result<ClassificationTaskResult, ClassificationPlanningError> {
        self.verify_identity(admitted)?;
        planning::checkpoint(control)?;
        let mut index = 0;
        let result = self.execute_heads::<ClassificationPlanningError, _>(|head| {
            planning::checkpoint(control)?;
            let mut bound = Bound { model: &mut *model, control: &mut *control, index,
                prompt: head.task.ir().prompt_segments(), cancelled: None };
            let scores = head.classifier.scorer.score(&mut bound, ScoringMode::FullVocabulary);
            if let Some(reason) = bound.cancelled { return Err(ClassificationPlanningError::Cancelled(reason)); }
            let scores = scores.map_err(ClassificationError::from)?;
            index += 1;
            Ok(scores)
        })?;
        planning::checkpoint(control)?;
        Ok(result)
    }
}

struct Bound<'a, M, C> {
    model: &'a mut M, control: &'a mut C, index: usize,
    prompt: &'a [PromptSegment], cancelled: Option<DecodeCancellationKind>,
}
impl<M: ClassificationLogits, C: DecodeStepControl> CandidateLogits for Bound<'_, M, C> {
    type Error = &'static str;
    fn project(&mut self, prefix: &[u32], rows: ProjectionRows<'_>) -> Result<Vec<f32>, Self::Error> {
        if let Some(reason) = self.control.prefill_checkpoint(0) {
            self.cancelled = Some(reason); return Err("classification cancelled");
        }
        self.model.project(self.index, self.prompt, prefix, rows).map_err(|_| "classification projection failed")
    }
}
