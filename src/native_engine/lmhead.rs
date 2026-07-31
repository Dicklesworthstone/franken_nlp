//! Untied language-model-head projection and f32 logit export.

use super::tensor::Bf16;
use super::weights::{Bf16Matrix, WeightShapeError};

/// Nanbeige's fixed untied vocabulary projection width.
pub const NANBEIGE_VOCAB_SIZE: usize = 166_144;
/// Exported f32 logits for the fixed vocabulary: 166,144 × 4 bytes.
pub const NANBEIGE_F32_LOGIT_BYTES: usize = NANBEIGE_VOCAB_SIZE * size_of::<f32>();

/// Projects hidden state through the untied bf16 lm_head and exports f32 logits.
///
/// The pinned eager oracle's `nn.Linear` receives bf16 inputs and weights, so
/// the completed projection narrows to bf16 before `logits.float()` widens it
/// at the model's public output boundary.
pub fn export_logits_f32(
    hidden: &[Bf16],
    lm_head: &Bf16Matrix,
) -> Result<Vec<f32>, WeightShapeError> {
    lm_head.project_f32_accumulate_bf16_then_export(hidden)
}

/// Deterministic first-index-wins greedy argmax over exported f32 logits.
#[must_use]
pub fn greedy_argmax(logits: &[f32]) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (index, logit) in logits.iter().copied().enumerate() {
        let replaces_best = match best {
            None => true,
            Some((_, current)) => logit.total_cmp(&current).is_gt(),
        };
        if replaces_best {
            best = Some((index, logit));
        }
    }
    best.map(|(index, _)| index)
}
