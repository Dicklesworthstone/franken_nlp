//! Checked `.fnlpq` tensor materialization for native-engine profiles.
//!
//! This module deliberately separates the checked envelope metadata from the
//! bytes used to materialize a tensor.  This is synthetic code-first
//! scaffolding, not an artifact activation path.  A future production source
//! must expose one mapping at a time and may not retain the envelope while the
//! native weight set is assembled.  The current owned-buffer reader is
//! deliberately not adapted here: artifact schema and transaction authority
//! remain unresolved.

use std::{collections::BTreeMap, error::Error, fmt};

use crate::artifact::converter::{BF16_VERBATIM_V1, PORTABLE_QUANT_V1, StorageStage};

use super::{
    tensor::Bf16,
    weights::{Bf16Matrix, WeightShapeError},
};

const NANBEIGE_MODEL_ID: &str = "Nanbeige4.2-3B";
const SYNTHETIC_SCOPE: &str = "scope=synthetic evidence=non_authoritative";

/// One Generic mapping selected from a checked tensor declaration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TensorMapping {
    /// Logical tensor payload bytes.
    Data,
    /// Per-output-row f32 quantization scales.
    Scale,
    /// Per-output-row i32 semantic sums.
    RowSum,
}

impl TensorMapping {
    const fn stage_name(self) -> &'static str {
        match self {
            Self::Data => "payload",
            Self::Scale => "scales",
            Self::RowSum => "row-sums",
        }
    }
}

/// Public, non-byte metadata required before a tensor may be materialized.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactTensorDescriptor {
    /// Source-census tensor name retained by the artifact schema.
    pub name: String,
    /// Closed canonical dtype spelling.
    pub canonical_dtype: String,
    /// Declared row-major logical shape.
    pub shape: Vec<u32>,
    /// Converter-selected representation identity.
    pub quantization: String,
    /// Exact declared Generic mapping byte counts.
    pub mapping_lengths: TensorMappingLengths,
}

/// Declared byte lengths for a tensor's three Generic mappings.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TensorMappingLengths {
    /// Logical payload bytes.
    pub data: u64,
    /// Quantization-scale bytes.
    pub scale: u64,
    /// Semantic row-sum bytes.
    pub row_sum: u64,
}

impl TensorMappingLengths {
    const fn for_mapping(self, mapping: TensorMapping) -> u64 {
        match mapping {
            TensorMapping::Data => self.data,
            TensorMapping::Scale => self.scale,
            TensorMapping::RowSum => self.row_sum,
        }
    }
}

/// Identity facts passed from the checked artifact boundary into native code.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactIdentity {
    /// One-model artifact identity.
    pub model_id: String,
    /// Immutable source revision declared by the envelope.
    pub revision: String,
    /// Converter recipe identity.
    pub recipe_id: String,
    /// Source-root digest declared by the envelope.
    pub source_root_sha256: String,
    /// Packing-independent logical model identity.
    pub logical_model_sha256: String,
}

/// A synthetic checked source that can expose only bounded portions of one tensor mapping.
///
/// Implementations are responsible for validating the envelope before they
/// expose bytes.  The bridge verifies that every callback contributes exactly
/// the mapping length declared in `tensors()`, so a source cannot truncate or
/// append a tensor silently.
pub trait CheckedArtifactSource {
    /// Checked immutable identity facts.
    fn identity(&self) -> &ArtifactIdentity;

    /// Checked tensor declarations, in any stable order.
    fn tensors(&self) -> &[ArtifactTensorDescriptor];

    /// Bytes already resident solely because of the source representation.
    ///
    /// A future ratified file-backed implementation reports zero.  No current
    /// `.fnlpq` reader implements this interface.
    fn resident_envelope_bytes(&self) -> u64;

    /// Visit one mapping in bounded chunks after the envelope is checked.
    fn stream_mapping(
        &self,
        tensor: &ArtifactTensorDescriptor,
        mapping: TensorMapping,
        visitor: &mut dyn FnMut(&[u8]) -> Result<(), ArtifactBridgeError>,
    ) -> Result<(), ArtifactBridgeError>;
}

/// Explicit allocation and streaming limits for synthetic bridge exercises.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArtifactLoadBudget {
    /// Maximum bytes in the completed native weight set.
    pub max_weight_bytes: u64,
    /// Maximum callback chunk accepted from the checked source.
    pub max_stream_chunk_bytes: u64,
    /// Maximum source-owned envelope bytes while native weights are built.
    /// All synthetic fixtures use zero.
    pub max_resident_envelope_bytes: u64,
}

impl ArtifactLoadBudget {
    /// Construct a budget that forbids envelope-plus-weights materialization.
    #[must_use]
    pub const fn streaming_only(max_weight_bytes: u64, max_stream_chunk_bytes: u64) -> Self {
        Self {
            max_weight_bytes,
            max_stream_chunk_bytes,
            max_resident_envelope_bytes: 0,
        }
    }
}

/// A typed tensor contract used only by synthetic bridge tests.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactTensorContract {
    /// Artifact-facing source tensor name.
    pub source_name: String,
    /// Native-engine-facing route name.
    pub internal_name: String,
    /// Exact frozen logical shape.
    pub shape: Vec<u32>,
    /// Required conversion storage stage.
    pub stage: StorageStage,
}

/// A bf16 tensor retained without an implicit conversion to f32.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ArtifactBf16Tensor {
    /// Rank-one norm scale.
    Vector(Vec<Bf16>),
    /// Rank-two embedding matrix.
    Matrix(Bf16Matrix),
}

/// One portable quantized matrix retained in its canonical logical layout.
#[derive(Clone, Debug, PartialEq)]
pub struct PortableQuantizedMatrix {
    rows: usize,
    columns: usize,
    values: Vec<i8>,
    row_scales: Vec<f32>,
    row_sums: Vec<i32>,
}

impl PortableQuantizedMatrix {
    /// Number of output rows.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Number of values in each row.
    #[must_use]
    pub const fn columns(&self) -> usize {
        self.columns
    }

    /// Canonical signed weight bytes in row-major order.
    #[must_use]
    pub fn values(&self) -> &[i8] {
        &self.values
    }

    /// Positive finite f32 scale for each output row.
    #[must_use]
    pub fn row_scales(&self) -> &[f32] {
        &self.row_scales
    }

    /// Canonical signed row sum for each output row.
    #[must_use]
    pub fn row_sums(&self) -> &[i32] {
        &self.row_sums
    }
}

/// The mixed native representation required by the first portable int8 recipe.
#[derive(Clone, Debug, PartialEq)]
pub struct ArtifactWeightSet {
    identity: ArtifactIdentity,
    bf16: BTreeMap<String, ArtifactBf16Tensor>,
    quantized: BTreeMap<String, PortableQuantizedMatrix>,
}

impl ArtifactWeightSet {
    /// Immutable artifact identity bound before any weight bytes were accepted.
    #[must_use]
    pub fn identity(&self) -> &ArtifactIdentity {
        &self.identity
    }

    /// Verbatim bf16 tensors, indexed by their native route names.
    #[must_use]
    pub fn bf16(&self) -> &BTreeMap<String, ArtifactBf16Tensor> {
        &self.bf16
    }

    /// Portable int8 matrices and their required scale/row-sum sidecars.
    #[must_use]
    pub fn quantized(&self) -> &BTreeMap<String, PortableQuantizedMatrix> {
        &self.quantized
    }
}

/// Declared synthetic-memory facts retained for a later process-RSS rehearsal gate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArtifactLoadReceipt {
    /// Sum of the tensor mappings declared by the synthetic contract.
    pub declared_mapping_bytes: u64,
    /// Largest source callback actually observed during this load.
    pub largest_stream_chunk_bytes: u64,
    /// Source representation residency declared by the synthetic source.
    pub declared_source_resident_bytes: u64,
}

impl ArtifactLoadReceipt {
    /// Declared mapping bytes plus one observed chunk and source residency.
    ///
    /// This omits allocator, `BTreeMap`, key, descriptor, contract, and vector
    /// capacity overhead.  It is neither a resident-set measurement nor an
    /// admissible peak-RSS assertion.
    #[must_use]
    pub fn modeled_declared_bytes_without_overhead(self) -> Option<u64> {
        self.declared_mapping_bytes
            .checked_add(self.largest_stream_chunk_bytes)?
            .checked_add(self.declared_source_resident_bytes)
    }
}

/// Loader failures with the responsible source tensor and stage preserved.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ArtifactBridgeError {
    /// The artifact did not identify the only supported model.
    ModelId { observed: String },
    /// Current OQ-31 schema and artifact-activation authority is unresolved.
    SchemaAuthority { detail: &'static str },
    /// A source descriptor did not match the explicit on-load census contract.
    Census { tensor: String, reason: String },
    /// A mapping's dtype, bytes, or values violated its selected stage.
    Tensor {
        tensor: String,
        stage: &'static str,
        reason: String,
    },
    /// The source disclosed an envelope or callback beyond admitted bounds.
    Memory {
        subject: &'static str,
        observed: u64,
        limit: u64,
    },
    /// A checked source could not expose the requested bounded mapping.
    Source { tensor: String, detail: String },
    /// Matrix construction exposed an impossible checked shape relationship.
    Matrix { tensor: String, detail: String },
}

impl fmt::Display for ArtifactBridgeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ModelId { observed } => {
                write!(
                    formatter,
                    "artifact engine bridge rejects model_id={observed:?}"
                )
            }
            Self::SchemaAuthority { detail } => {
                write!(
                    formatter,
                    "artifact engine bridge authority unavailable: {detail}"
                )
            }
            Self::Census { tensor, reason } => {
                write!(
                    formatter,
                    "artifact engine census tensor {tensor:?}: {reason}"
                )
            }
            Self::Tensor {
                tensor,
                stage,
                reason,
            } => write!(
                formatter,
                "artifact engine tensor {tensor:?} stage={stage}: {reason}"
            ),
            Self::Memory {
                subject,
                observed,
                limit,
            } => write!(
                formatter,
                "artifact engine memory {subject}: observed={observed} limit={limit}"
            ),
            Self::Source { tensor, detail } => {
                write!(
                    formatter,
                    "artifact engine source tensor {tensor:?}: {detail}"
                )
            }
            Self::Matrix { tensor, detail } => {
                write!(
                    formatter,
                    "artifact engine matrix tensor {tensor:?}: {detail}"
                )
            }
        }
    }
}

impl Error for ArtifactBridgeError {}

/// The result of one checked materialization pass.
#[derive(Clone, Debug, PartialEq)]
pub struct LoadedArtifactWeights {
    /// Synthetic mixed bf16/int8 materialization output.
    pub weights: ArtifactWeightSet,
    /// Declared synthetic-memory facts; never a real-data RSS receipt.
    pub receipt: ArtifactLoadReceipt,
}

/// Refuse one-model artifact activation until OQ-31 and xmy authority are ratified.
///
/// The model-id check keeps the refusal named; revision, recipe, source-root,
/// packing, tokenizer, transaction, and receipt binding must be defined by the
/// one ratified artifact authority before this function may materialize bytes.
pub fn load_nanbeige42<S, F>(
    source: &S,
    budget: ArtifactLoadBudget,
    mut stage_line: F,
) -> Result<LoadedArtifactWeights, ArtifactBridgeError>
where
    S: CheckedArtifactSource,
    F: FnMut(&str),
{
    if source.identity().model_id != NANBEIGE_MODEL_ID {
        return Err(ArtifactBridgeError::ModelId {
            observed: source.identity().model_id.clone(),
        });
    }
    let _ = budget;
    stage_line(
        "LOAD STAGE=activation scope=synthetic evidence=non_authoritative status=REFUSED reason=oq31-authority-unresolved",
    );
    Err(ArtifactBridgeError::SchemaAuthority {
        detail: "OQ-31 schema selection and artifact activation identity are unresolved",
    })
}

/// Materialize an explicit synthetic tensor contract.
///
/// This is non-production test scaffolding.  It is intentionally unable to
/// activate a `.fnlpq`, bind a real artifact identity, or establish L2 proof.
#[doc(hidden)]
pub fn load_synthetic_with_contract<S, F>(
    source: &S,
    budget: ArtifactLoadBudget,
    contract: &[ArtifactTensorContract],
    mut stage_line: F,
) -> Result<LoadedArtifactWeights, ArtifactBridgeError>
where
    S: CheckedArtifactSource,
    F: FnMut(&str),
{
    let resident_envelope_bytes = source.resident_envelope_bytes();
    if resident_envelope_bytes > budget.max_resident_envelope_bytes {
        return Err(ArtifactBridgeError::Memory {
            subject: "resident-envelope-bytes",
            observed: resident_envelope_bytes,
            limit: budget.max_resident_envelope_bytes,
        });
    }
    stage_line(&format!(
        "LOAD STAGE=preflight {SYNTHETIC_SCOPE} status=BEGIN"
    ));
    let descriptors = index_descriptors(source.tensors())?;
    validate_contract(&descriptors, contract)?;
    let weight_bytes = contract.iter().try_fold(0_u64, |total, expected| {
        let descriptor = descriptors
            .get(expected.source_name.as_str())
            .expect("validate_contract requires every contract descriptor");
        total
            .checked_add(descriptor.mapping_lengths.data)
            .and_then(|value| value.checked_add(descriptor.mapping_lengths.scale))
            .and_then(|value| value.checked_add(descriptor.mapping_lengths.row_sum))
            .ok_or_else(|| ArtifactBridgeError::Memory {
                subject: "declared-weight-bytes-overflow",
                observed: u64::MAX,
                limit: budget.max_weight_bytes,
            })
    })?;
    if weight_bytes > budget.max_weight_bytes {
        return Err(ArtifactBridgeError::Memory {
            subject: "declared-weight-bytes",
            observed: weight_bytes,
            limit: budget.max_weight_bytes,
        });
    }
    stage_line(&format!(
        "LOAD STAGE=census {SYNTHETIC_SCOPE} status=PASS tensors={} declared_mapping_bytes={weight_bytes}",
        contract.len()
    ));

    let mut bf16 = BTreeMap::new();
    let mut quantized = BTreeMap::new();
    let mut largest_stream_chunk_bytes = 0_u64;
    for expected in contract {
        let descriptor = descriptors
            .get(expected.source_name.as_str())
            .expect("validate_contract requires every contract descriptor");
        match expected.stage {
            StorageStage::Bf16Verbatim => {
                let value = decode_bf16_tensor(
                    source,
                    descriptor,
                    budget.max_stream_chunk_bytes,
                    &mut largest_stream_chunk_bytes,
                )?;
                bf16.insert(expected.internal_name.clone(), value);
            }
            StorageStage::Int8Stage2A | StorageStage::Int8Stage2B | StorageStage::Int8Stage2C => {
                let value = decode_quantized_matrix(
                    source,
                    descriptor,
                    budget.max_stream_chunk_bytes,
                    &mut largest_stream_chunk_bytes,
                )?;
                quantized.insert(expected.internal_name.clone(), value);
            }
        }
        stage_line(&format!(
            "LOAD STAGE=tensor {SYNTHETIC_SCOPE} status=PASS tensor={} route={} storage={}",
            expected.source_name,
            expected.internal_name,
            expected.stage.as_str()
        ));
    }
    let receipt = ArtifactLoadReceipt {
        declared_mapping_bytes: weight_bytes,
        largest_stream_chunk_bytes,
        declared_source_resident_bytes: resident_envelope_bytes,
    };
    stage_line(&format!(
        "LOAD STAGE=complete {SYNTHETIC_SCOPE} status=PASS bf16_tensors={} quantized_tensors={} modeled_declared_bytes_without_overhead={}",
        bf16.len(),
        quantized.len(),
        receipt
            .modeled_declared_bytes_without_overhead()
            .unwrap_or(u64::MAX)
    ));
    Ok(LoadedArtifactWeights {
        weights: ArtifactWeightSet {
            identity: source.identity().clone(),
            bf16,
            quantized,
        },
        receipt,
    })
}

fn index_descriptors<'a>(
    descriptors: &'a [ArtifactTensorDescriptor],
) -> Result<BTreeMap<&'a str, &'a ArtifactTensorDescriptor>, ArtifactBridgeError> {
    let mut indexed = BTreeMap::new();
    for descriptor in descriptors {
        if indexed
            .insert(descriptor.name.as_str(), descriptor)
            .is_some()
        {
            return Err(ArtifactBridgeError::Census {
                tensor: descriptor.name.clone(),
                reason: "duplicate checked tensor descriptor".to_owned(),
            });
        }
    }
    Ok(indexed)
}

fn validate_contract(
    descriptors: &BTreeMap<&str, &ArtifactTensorDescriptor>,
    contract: &[ArtifactTensorContract],
) -> Result<(), ArtifactBridgeError> {
    if descriptors.len() != contract.len() {
        return Err(ArtifactBridgeError::Census {
            tensor: "<census>".to_owned(),
            reason: format!(
                "expected {} tensors, observed {}",
                contract.len(),
                descriptors.len()
            ),
        });
    }
    for expected in contract {
        let actual = descriptors
            .get(expected.source_name.as_str())
            .ok_or_else(|| ArtifactBridgeError::Census {
                tensor: expected.source_name.clone(),
                reason: "missing expected tensor".to_owned(),
            })?;
        if actual.shape != expected.shape {
            return Err(ArtifactBridgeError::Census {
                tensor: expected.source_name.clone(),
                reason: format!(
                    "shape mismatch: expected {:?}, observed {:?}",
                    expected.shape, actual.shape
                ),
            });
        }
        let scalar_count = checked_scalar_count(actual)?;
        let rows = checked_rows(actual)?;
        let expected_lengths = match expected.stage {
            StorageStage::Bf16Verbatim => TensorMappingLengths {
                data: scalar_count
                    .checked_mul(2)
                    .ok_or_else(|| ArtifactBridgeError::Census {
                        tensor: actual.name.clone(),
                        reason: "bf16 byte length overflow".to_owned(),
                    })?,
                scale: 0,
                row_sum: 0,
            },
            StorageStage::Int8Stage2A | StorageStage::Int8Stage2B | StorageStage::Int8Stage2C => {
                let sidecar = rows
                    .checked_mul(4)
                    .ok_or_else(|| ArtifactBridgeError::Census {
                        tensor: actual.name.clone(),
                        reason: "quantized sidecar byte length overflow".to_owned(),
                    })?;
                TensorMappingLengths {
                    data: scalar_count,
                    scale: sidecar,
                    row_sum: sidecar,
                }
            }
        };
        let (expected_dtype, expected_quantization) = match expected.stage {
            StorageStage::Bf16Verbatim => ("bf16", BF16_VERBATIM_V1),
            StorageStage::Int8Stage2A | StorageStage::Int8Stage2B | StorageStage::Int8Stage2C => {
                ("i8", PORTABLE_QUANT_V1)
            }
        };
        if actual.canonical_dtype != expected_dtype
            || actual.quantization != expected_quantization
            || actual.mapping_lengths != expected_lengths
        {
            return Err(ArtifactBridgeError::Census {
                tensor: actual.name.clone(),
                reason: format!(
                    "expected dtype={expected_dtype} quantization={expected_quantization} mappings={expected_lengths:?}; observed dtype={} quantization={} mappings={:?}",
                    actual.canonical_dtype, actual.quantization, actual.mapping_lengths
                ),
            });
        }
    }
    Ok(())
}

fn checked_scalar_count(descriptor: &ArtifactTensorDescriptor) -> Result<u64, ArtifactBridgeError> {
    descriptor
        .shape
        .iter()
        .try_fold(1_u64, |count, &dimension| {
            count
                .checked_mul(u64::from(dimension))
                .ok_or_else(|| ArtifactBridgeError::Census {
                    tensor: descriptor.name.clone(),
                    reason: "shape scalar count overflow".to_owned(),
                })
        })
}

fn checked_rows(descriptor: &ArtifactTensorDescriptor) -> Result<u64, ArtifactBridgeError> {
    descriptor
        .shape
        .first()
        .copied()
        .map(u64::from)
        .filter(|rows| *rows > 0)
        .ok_or_else(|| ArtifactBridgeError::Census {
            tensor: descriptor.name.clone(),
            reason: "tensor has no nonzero leading row dimension".to_owned(),
        })
}

fn checked_usize(
    value: u64,
    tensor: &str,
    stage: &'static str,
) -> Result<usize, ArtifactBridgeError> {
    usize::try_from(value).map_err(|_| ArtifactBridgeError::Tensor {
        tensor: tensor.to_owned(),
        stage,
        reason: "declared byte count does not fit host usize".to_owned(),
    })
}

fn decode_bf16_tensor<S: CheckedArtifactSource>(
    source: &S,
    descriptor: &ArtifactTensorDescriptor,
    max_stream_chunk_bytes: u64,
    largest_stream_chunk_bytes: &mut u64,
) -> Result<ArtifactBf16Tensor, ArtifactBridgeError> {
    let values = decode_bf16_mapping(
        source,
        descriptor,
        max_stream_chunk_bytes,
        largest_stream_chunk_bytes,
    )?;
    match descriptor.shape.as_slice() {
        [length] => {
            let expected = checked_usize(u64::from(*length), &descriptor.name, "bf16-shape")?;
            if values.len() != expected {
                return Err(ArtifactBridgeError::Tensor {
                    tensor: descriptor.name.clone(),
                    stage: "bf16-shape",
                    reason: format!("expected {expected} values, observed {}", values.len()),
                });
            }
            Ok(ArtifactBf16Tensor::Vector(values))
        }
        [rows, columns] => Bf16Matrix::new(
            checked_usize(u64::from(*rows), &descriptor.name, "bf16-shape")?,
            checked_usize(u64::from(*columns), &descriptor.name, "bf16-shape")?,
            values,
        )
        .map(ArtifactBf16Tensor::Matrix)
        .map_err(|error| matrix_error(&descriptor.name, error)),
        _ => Err(ArtifactBridgeError::Tensor {
            tensor: descriptor.name.clone(),
            stage: "bf16-shape",
            reason: "only rank-one vectors and rank-two matrices are engine materializable"
                .to_owned(),
        }),
    }
}

fn decode_bf16_mapping<S: CheckedArtifactSource>(
    source: &S,
    descriptor: &ArtifactTensorDescriptor,
    max_stream_chunk_bytes: u64,
    largest_stream_chunk_bytes: &mut u64,
) -> Result<Vec<Bf16>, ArtifactBridgeError> {
    let expected_values =
        checked_usize(checked_scalar_count(descriptor)?, &descriptor.name, "bf16")?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(expected_values)
        .map_err(|error| ArtifactBridgeError::Tensor {
            tensor: descriptor.name.clone(),
            stage: "bf16",
            reason: format!("weight admission reservation failed: {error}"),
        })?;
    let mut pending = None;
    stream_mapping(
        source,
        descriptor,
        TensorMapping::Data,
        max_stream_chunk_bytes,
        largest_stream_chunk_bytes,
        |chunk| {
            let mut bytes = chunk;
            if let Some(first) = pending.take() {
                let second = *bytes.first().ok_or_else(|| ArtifactBridgeError::Tensor {
                    tensor: descriptor.name.clone(),
                    stage: "bf16",
                    reason: "empty chunk followed an incomplete bf16 word".to_owned(),
                })?;
                values.push(Bf16::from_bits(u16::from_le_bytes([first, second])));
                bytes = &bytes[1..];
            }
            let remainder = bytes.chunks_exact(2);
            for pair in remainder.clone() {
                values.push(Bf16::from_bits(u16::from_le_bytes([pair[0], pair[1]])));
            }
            pending = remainder.remainder().first().copied();
            if values.len() > expected_values {
                return Err(ArtifactBridgeError::Tensor {
                    tensor: descriptor.name.clone(),
                    stage: "bf16",
                    reason: format!("received more than declared {expected_values} values"),
                });
            }
            Ok(())
        },
    )?;
    if pending.is_some() || values.len() != expected_values {
        return Err(ArtifactBridgeError::Tensor {
            tensor: descriptor.name.clone(),
            stage: "bf16",
            reason: format!(
                "expected {expected_values} complete values, observed {} with trailing_byte={}",
                values.len(),
                pending.is_some()
            ),
        });
    }
    Ok(values)
}

fn decode_quantized_matrix<S: CheckedArtifactSource>(
    source: &S,
    descriptor: &ArtifactTensorDescriptor,
    max_stream_chunk_bytes: u64,
    largest_stream_chunk_bytes: &mut u64,
) -> Result<PortableQuantizedMatrix, ArtifactBridgeError> {
    let [rows, columns] = descriptor.shape.as_slice() else {
        return Err(ArtifactBridgeError::Tensor {
            tensor: descriptor.name.clone(),
            stage: "quantized-shape",
            reason: "portable quantized engine tensors must be rank-two matrices".to_owned(),
        });
    };
    let rows = checked_usize(u64::from(*rows), &descriptor.name, "quantized-shape")?;
    let columns = checked_usize(u64::from(*columns), &descriptor.name, "quantized-shape")?;
    let expected_values = rows
        .checked_mul(columns)
        .ok_or_else(|| ArtifactBridgeError::Tensor {
            tensor: descriptor.name.clone(),
            stage: "quantized-shape",
            reason: "matrix value count overflow".to_owned(),
        })?;
    let values = decode_i8_mapping(
        source,
        descriptor,
        max_stream_chunk_bytes,
        largest_stream_chunk_bytes,
        expected_values,
    )?;
    let row_scales = decode_f32_mapping(
        source,
        descriptor,
        max_stream_chunk_bytes,
        largest_stream_chunk_bytes,
        rows,
    )?;
    let row_sums = decode_i32_mapping(
        source,
        descriptor,
        max_stream_chunk_bytes,
        largest_stream_chunk_bytes,
        rows,
    )?;
    for (row_index, (row, &declared_sum)) in values.chunks_exact(columns).zip(&row_sums).enumerate()
    {
        let observed_sum = row.iter().fold(0_i64, |sum, &value| sum + i64::from(value));
        if observed_sum != i64::from(declared_sum) {
            return Err(ArtifactBridgeError::Tensor {
                tensor: descriptor.name.clone(),
                stage: "row-sums",
                reason: format!(
                    "row {row_index} declares {declared_sum}, recomputed {observed_sum}"
                ),
            });
        }
    }
    Ok(PortableQuantizedMatrix {
        rows,
        columns,
        values,
        row_scales,
        row_sums,
    })
}

fn decode_i8_mapping<S: CheckedArtifactSource>(
    source: &S,
    descriptor: &ArtifactTensorDescriptor,
    max_stream_chunk_bytes: u64,
    largest_stream_chunk_bytes: &mut u64,
    expected_values: usize,
) -> Result<Vec<i8>, ArtifactBridgeError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(expected_values)
        .map_err(|error| ArtifactBridgeError::Tensor {
            tensor: descriptor.name.clone(),
            stage: "payload",
            reason: format!("weight admission reservation failed: {error}"),
        })?;
    stream_mapping(
        source,
        descriptor,
        TensorMapping::Data,
        max_stream_chunk_bytes,
        largest_stream_chunk_bytes,
        |chunk| {
            values.extend(chunk.iter().map(|&value| i8::from_ne_bytes([value])));
            if values.len() > expected_values {
                return Err(ArtifactBridgeError::Tensor {
                    tensor: descriptor.name.clone(),
                    stage: "payload",
                    reason: format!("received more than declared {expected_values} values"),
                });
            }
            Ok(())
        },
    )?;
    if values.len() != expected_values {
        return Err(ArtifactBridgeError::Tensor {
            tensor: descriptor.name.clone(),
            stage: "payload",
            reason: format!(
                "expected {expected_values} values, observed {}",
                values.len()
            ),
        });
    }
    Ok(values)
}

fn decode_f32_mapping<S: CheckedArtifactSource>(
    source: &S,
    descriptor: &ArtifactTensorDescriptor,
    max_stream_chunk_bytes: u64,
    largest_stream_chunk_bytes: &mut u64,
    expected_values: usize,
) -> Result<Vec<f32>, ArtifactBridgeError> {
    decode_fixed_width_mapping(
        source,
        descriptor,
        TensorMapping::Scale,
        max_stream_chunk_bytes,
        largest_stream_chunk_bytes,
        expected_values,
        |bytes| {
            let value = f32::from_le_bytes(bytes);
            if !value.is_finite() || value <= 0.0 {
                return Err("must be finite and positive".to_owned());
            }
            Ok(value)
        },
    )
}

fn decode_i32_mapping<S: CheckedArtifactSource>(
    source: &S,
    descriptor: &ArtifactTensorDescriptor,
    max_stream_chunk_bytes: u64,
    largest_stream_chunk_bytes: &mut u64,
    expected_values: usize,
) -> Result<Vec<i32>, ArtifactBridgeError> {
    decode_fixed_width_mapping(
        source,
        descriptor,
        TensorMapping::RowSum,
        max_stream_chunk_bytes,
        largest_stream_chunk_bytes,
        expected_values,
        |bytes| Ok(i32::from_le_bytes(bytes)),
    )
}

fn decode_fixed_width_mapping<S, T, D>(
    source: &S,
    descriptor: &ArtifactTensorDescriptor,
    mapping: TensorMapping,
    max_stream_chunk_bytes: u64,
    largest_stream_chunk_bytes: &mut u64,
    expected_values: usize,
    mut decode: D,
) -> Result<Vec<T>, ArtifactBridgeError>
where
    S: CheckedArtifactSource,
    D: FnMut([u8; 4]) -> Result<T, String>,
{
    let mut values = Vec::new();
    values
        .try_reserve_exact(expected_values)
        .map_err(|error| ArtifactBridgeError::Tensor {
            tensor: descriptor.name.clone(),
            stage: mapping.stage_name(),
            reason: format!("weight admission reservation failed: {error}"),
        })?;
    let mut pending = [0_u8; 4];
    let mut pending_len = 0_usize;
    stream_mapping(
        source,
        descriptor,
        mapping,
        max_stream_chunk_bytes,
        largest_stream_chunk_bytes,
        |chunk| {
            for &byte in chunk {
                pending[pending_len] = byte;
                pending_len += 1;
                if pending_len == 4 {
                    let value = decode(pending).map_err(|reason| ArtifactBridgeError::Tensor {
                        tensor: descriptor.name.clone(),
                        stage: mapping.stage_name(),
                        reason,
                    })?;
                    values.push(value);
                    pending_len = 0;
                    if values.len() > expected_values {
                        return Err(ArtifactBridgeError::Tensor {
                            tensor: descriptor.name.clone(),
                            stage: mapping.stage_name(),
                            reason: format!("received more than declared {expected_values} values"),
                        });
                    }
                }
            }
            Ok(())
        },
    )?;
    if pending_len != 0 || values.len() != expected_values {
        return Err(ArtifactBridgeError::Tensor {
            tensor: descriptor.name.clone(),
            stage: mapping.stage_name(),
            reason: format!(
                "expected {expected_values} complete values, observed {} trailing_bytes={pending_len}",
                values.len()
            ),
        });
    }
    Ok(values)
}

fn stream_mapping<S, F>(
    source: &S,
    descriptor: &ArtifactTensorDescriptor,
    mapping: TensorMapping,
    max_stream_chunk_bytes: u64,
    largest_stream_chunk_bytes: &mut u64,
    mut consume: F,
) -> Result<(), ArtifactBridgeError>
where
    S: CheckedArtifactSource,
    F: FnMut(&[u8]) -> Result<(), ArtifactBridgeError>,
{
    let expected_bytes = descriptor.mapping_lengths.for_mapping(mapping);
    let mut observed_bytes = 0_u64;
    source.stream_mapping(descriptor, mapping, &mut |chunk| {
        let chunk_len = u64::try_from(chunk.len()).map_err(|_| ArtifactBridgeError::Tensor {
            tensor: descriptor.name.clone(),
            stage: mapping.stage_name(),
            reason: "stream chunk length does not fit u64".to_owned(),
        })?;
        if chunk_len > max_stream_chunk_bytes {
            return Err(ArtifactBridgeError::Memory {
                subject: "stream-chunk-bytes",
                observed: chunk_len,
                limit: max_stream_chunk_bytes,
            });
        }
        observed_bytes =
            observed_bytes
                .checked_add(chunk_len)
                .ok_or_else(|| ArtifactBridgeError::Tensor {
                    tensor: descriptor.name.clone(),
                    stage: mapping.stage_name(),
                    reason: "stream byte count overflow".to_owned(),
                })?;
        if observed_bytes > expected_bytes {
            return Err(ArtifactBridgeError::Tensor {
                tensor: descriptor.name.clone(),
                stage: mapping.stage_name(),
                reason: format!("stream exceeded declared {expected_bytes} bytes"),
            });
        }
        *largest_stream_chunk_bytes = (*largest_stream_chunk_bytes).max(chunk_len);
        consume(chunk)
    })?;
    if observed_bytes != expected_bytes {
        return Err(ArtifactBridgeError::Tensor {
            tensor: descriptor.name.clone(),
            stage: mapping.stage_name(),
            reason: format!("declared {expected_bytes} bytes, observed {observed_bytes}"),
        });
    }
    Ok(())
}

fn matrix_error(tensor: &str, error: WeightShapeError) -> ArtifactBridgeError {
    ArtifactBridgeError::Matrix {
        tensor: tensor.to_owned(),
        detail: format!("{error:?}"),
    }
}
