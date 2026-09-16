//! Borrowed-weight validation without cloning or constructing a scalar engine.
//! The scalar engine's weight validator is private. Reuse each layer's public
//! validator and the same model dimension constants at this borrowing boundary.
use super::*;
pub(super) fn validate(weights: &HfBf16EagerWeights) -> Result<(), BatchError> {
    for (name, matrix) in [("embeddings", &weights.embeddings), ("lm_head", &weights.lm_head)] {
        if matrix.rows() != NANBEIGE_VOCAB_SIZE || matrix.columns() != NANBEIGE_HIDDEN_SIZE {
            return Err(HfBf16EagerError::ModelMatrixShape { name, rows: matrix.rows(), columns: matrix.columns(),
                expected_rows: NANBEIGE_VOCAB_SIZE, expected_columns: NANBEIGE_HIDDEN_SIZE }.into());
        }
    }
    if weights.final_norm.len() != NANBEIGE_HIDDEN_SIZE {
        return Err(HfBf16EagerError::ModelVectorShape { name: "final_norm", expected: NANBEIGE_HIDDEN_SIZE,
            actual: weights.final_norm.len() }.into());
    }
    for layer in &weights.layers { layer.validate().map_err(HfBf16EagerError::from)?; }
    Ok(())
}
