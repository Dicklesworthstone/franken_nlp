//! Untied language-model-head projection and f32 logit export.

use super::tensor::Bf16;
use super::weights::{Bf16Matrix, WeightShapeError};

/// Nanbeige's fixed untied vocabulary projection width.
pub const NANBEIGE_VOCAB_SIZE: usize = 166_144;
/// Exported f32 logits for the fixed vocabulary: 166,144 × 4 bytes.
pub const NANBEIGE_F32_LOGIT_BYTES: usize = NANBEIGE_VOCAB_SIZE * size_of::<f32>();

/// Projects hidden state through the untied lm_head and keeps logits in f32.
pub fn export_logits_f32(
    hidden: &[Bf16],
    lm_head: &Bf16Matrix,
) -> Result<Vec<f32>, WeightShapeError> {
    lm_head.project_f32_export(hidden)
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
