//! Shape-checked bf16 model-weight primitives for the reference profile.

use super::tensor::{Bf16, cast_f32_to_bf16};

/// A row-major bf16 matrix with no implicit bias vector.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Bf16Matrix {
    rows: usize,
    columns: usize,
    values: Vec<Bf16>,
}

/// Matrix construction/projection shape failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WeightShapeError {
    /// Matrix storage did not equal rows × columns.
    MatrixStorage {
        rows: usize,
        columns: usize,
        expected: usize,
        actual: usize,
    },
    /// An input activation length did not match the matrix column count.
    ProjectionInput { expected: usize, actual: usize },
    /// A requested row was outside the matrix's output dimension.
    RowOutOfRange { row: usize, rows: usize },
}

impl Bf16Matrix {
    /// Constructs a matrix only when its row-major storage has the exact shape.
    pub fn new(rows: usize, columns: usize, values: Vec<Bf16>) -> Result<Self, WeightShapeError> {
        let expected = rows
            .checked_mul(columns)
            .ok_or(WeightShapeError::MatrixStorage {
                rows,
                columns,
                expected: usize::MAX,
                actual: values.len(),
            })?;
        if values.len() != expected {
            return Err(WeightShapeError::MatrixStorage {
                rows,
                columns,
                expected,
                actual: values.len(),
            });
        }
        Ok(Self {
            rows,
            columns,
            values,
        })
    }

    /// Matrix output features.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Matrix input features.
    #[must_use]
    pub const fn columns(&self) -> usize {
        self.columns
    }

    /// Borrows one row without widening its bf16 values.
    pub fn row(&self, row: usize) -> Result<&[Bf16], WeightShapeError> {
        let start = row
            .checked_mul(self.columns)
            .ok_or(WeightShapeError::RowOutOfRange {
                row,
                rows: self.rows,
            })?;
        let end = start
            .checked_add(self.columns)
            .ok_or(WeightShapeError::RowOutOfRange {
                row,
                rows: self.rows,
            })?;
        self.values
            .get(start..end)
            .ok_or(WeightShapeError::RowOutOfRange {
                row,
                rows: self.rows,
            })
    }

    /// Reference f32 accumulation followed by an explicit bf16 activation cast.
    ///
    /// The model has no bias in any of these projections. Backend-specific
    /// kernels may replace the inner dot product only after L1 parity evidence.
    pub fn project_f32_accumulate_cast_back(
        &self,
        input: &[Bf16],
    ) -> Result<Vec<Bf16>, WeightShapeError> {
        if input.len() != self.columns {
            return Err(WeightShapeError::ProjectionInput {
                expected: self.columns,
                actual: input.len(),
            });
        }
        let output = self
            .values
            .chunks_exact(self.columns)
            .map(|row| {
                row.iter()
                    .zip(input)
                    .map(|(weight, activation)| weight.to_f32() * activation.to_f32())
                    .sum::<f32>()
            })
            .collect::<Vec<_>>();
        Ok(cast_f32_to_bf16(&output))
    }

    /// Pinned `lm_head` operation result widened for the f32 logit boundary.
    ///
    /// `NanbeigeForCausalLM.forward` passes bf16 hidden activations and bf16
    /// weights to `nn.Linear`, so the CPU eager linear operation returns bf16.
    /// It then calls `logits.float()`. Keep the f32 dot-product accumulation,
    /// but narrow the completed linear result to bf16 before that final export.
    pub fn project_f32_accumulate_bf16_then_export(
        &self,
        input: &[Bf16],
    ) -> Result<Vec<f32>, WeightShapeError> {
        if input.len() != self.columns {
            return Err(WeightShapeError::ProjectionInput {
                expected: self.columns,
                actual: input.len(),
            });
        }
        Ok(self
            .values
            .chunks_exact(self.columns)
            .map(|row| {
                let accumulator = row
                    .iter()
                    .zip(input)
                    .map(|(weight, activation)| weight.to_f32() * activation.to_f32())
                    .sum::<f32>();
                Bf16::from_f32(accumulator).to_f32()
            })
            .collect())
    }
}
