//! Measured, safe dispatch policy for the native integer-kernel family.
//!
//! Feature detection only determines which tiers may be considered.  It does
//! not choose a backend: a measured row (with retained provenance) does.  A
//! missing row is deliberately reported as a conservative fallback rather
//! than as invented performance evidence.

use std::{fmt, str::FromStr, sync::OnceLock};

use serde::{Deserialize, Serialize};

use super::int8::{self, Int8KernelError, MODEL_KS, MODEL_NS};

/// Fixed native-kernel entry points supported by the dispatcher.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KernelOperation {
    Int8Gemm,
    Int8Gemv,
    Int4Gemv,
}

impl KernelOperation {
    /// Stable identifier used in robot and benchmark records.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::Int8Gemm => "int8_gemm",
            Self::Int8Gemv => "int8_gemv",
            Self::Int4Gemv => "int4_gemv",
        }
    }
}

/// Workload regime.  The selection key always names the reduction regime.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DispatchRegime {
    PrefillGemm,
    DecodeGemv,
    BatchedDecodeSkinnyM,
}

impl DispatchRegime {
    /// All stable regime spellings.
    pub const ALL: [Self; 3] = [
        Self::PrefillGemm,
        Self::DecodeGemv,
        Self::BatchedDecodeSkinnyM,
    ];

    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::PrefillGemm => "prefill_gemm",
            Self::DecodeGemv => "decode_gemv",
            Self::BatchedDecodeSkinnyM => "batched_decode_skinny_m",
        }
    }

    #[must_use]
    pub const fn representative_m(self) -> usize {
        match self {
            Self::PrefillGemm => 8,
            Self::DecodeGemv => 1,
            Self::BatchedDecodeSkinnyM => 4,
        }
    }
}

/// Dynamic-M bucket carried by every measured dispatch key.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MBucket {
    One,
    TwoToFour,
    FiveToSixteen,
    SeventeenPlus,
}

impl MBucket {
    #[must_use]
    pub const fn for_m(m: usize) -> Self {
        match m {
            0 | 1 => Self::One,
            2..=4 => Self::TwoToFour,
            5..=16 => Self::FiveToSixteen,
            _ => Self::SeventeenPlus,
        }
    }
}

/// Runtime shape component. `k` and `n` are fixed model dimensions; `m` is
/// dynamic and represented by [`MBucket`] in the table key.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KernelShape {
    pub k: usize,
    pub m: usize,
    pub n: usize,
}

/// Complete lookup key.  Future tuning overlays must preserve this identity
/// rather than replacing it with a host-wide preference.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DispatchKey {
    pub m_bucket: MBucket,
    pub operation: KernelOperation,
    pub regime: DispatchRegime,
    pub shape: KernelShape,
}

impl DispatchKey {
    #[must_use]
    pub const fn new(
        operation: KernelOperation,
        regime: DispatchRegime,
        shape: KernelShape,
    ) -> Self {
        Self {
            m_bucket: MBucket::for_m(shape.m),
            operation,
            regime,
            shape,
        }
    }
}

/// CPU architecture family, intentionally not a model- or vendor-name table.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Architecture {
    Aarch64,
    X86_64,
    Other,
}

impl Architecture {
    #[must_use]
    pub const fn host() -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            return Self::Aarch64;
        }
        #[cfg(target_arch = "x86_64")]
        {
            return Self::X86_64;
        }
        #[allow(unreachable_code)]
        Self::Other
    }
}

/// The complete tier catalog.  A tier can be detected but not selected; that
/// distinction is the point of the measured table.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KernelTier {
    A1Smmla,
    A2Dotprod,
    A3Autovec,
    X1aAvx512VnniZmm,
    X1bAvx512VnniYmm,
    X2AvxVnni,
    X3aAvx2Low7HighBit,
    X3bAvx2WidenedI16,
    Scalar,
}

impl KernelTier {
    /// Stable registry order.  Earlier entries are wider conservative
    /// candidates only; this order is never a measurement claim.
    pub const ALL: [Self; 9] = [
        Self::A1Smmla,
        Self::A2Dotprod,
        Self::A3Autovec,
        Self::X1aAvx512VnniZmm,
        Self::X1bAvx512VnniYmm,
        Self::X2AvxVnni,
        Self::X3aAvx2Low7HighBit,
        Self::X3bAvx2WidenedI16,
        Self::Scalar,
    ];

    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::A1Smmla => "a1_smmla",
            Self::A2Dotprod => "a2_dotprod",
            Self::A3Autovec => "a3_autovec",
            Self::X1aAvx512VnniZmm => "x1a_avx512_vnni_zmm",
            Self::X1bAvx512VnniYmm => "x1b_avx512_vnni_ymm",
            Self::X2AvxVnni => "x2_avx_vnni",
            Self::X3aAvx2Low7HighBit => "x3a_avx2_low7_high_bit",
            Self::X3bAvx2WidenedI16 => "x3b_avx2_widened_i16",
            Self::Scalar => "s_scalar_i32",
        }
    }

    #[must_use]
    pub const fn kernel_id(self, operation: KernelOperation) -> &'static str {
        match (self, operation) {
            (Self::Scalar, KernelOperation::Int8Gemm) => "s_scalar_i32_int8_gemm",
            (Self::Scalar, KernelOperation::Int8Gemv) => "s_scalar_i32_int8_gemv",
            (Self::Scalar, KernelOperation::Int4Gemv) => "s_scalar_i32_int4_gemv",
            (Self::A1Smmla, _) => "a1_smmla_pending",
            (Self::A2Dotprod, _) => "a2_dotprod_pending",
            (Self::A3Autovec, _) => "a3_autovec_pending",
            (Self::X1aAvx512VnniZmm, _) => "x1a_avx512_vnni_zmm_pending",
            (Self::X1bAvx512VnniYmm, _) => "x1b_avx512_vnni_ymm_pending",
            (Self::X2AvxVnni, _) => "x2_avx_vnni_pending",
            (Self::X3aAvx2Low7HighBit, _) => "x3a_avx2_low7_high_bit_pending",
            (Self::X3bAvx2WidenedI16, _) => "x3b_avx2_widened_i16_pending",
        }
    }

    #[must_use]
    pub const fn is_implemented(self) -> bool {
        matches!(self, Self::Scalar)
    }

    #[must_use]
    pub fn is_detected(self, features: DetectedFeatures) -> bool {
        match self {
            Self::A1Smmla => {
                features.architecture == Architecture::Aarch64 && features.aarch64_i8mm
            }
            Self::A2Dotprod => {
                features.architecture == Architecture::Aarch64 && features.aarch64_dotprod
            }
            Self::A3Autovec => features.architecture == Architecture::Aarch64,
            Self::X1aAvx512VnniZmm => {
                features.architecture == Architecture::X86_64
                    && features.x86_avx512f
                    && features.x86_avx512vnni
            }
            Self::X1bAvx512VnniYmm => {
                features.architecture == Architecture::X86_64
                    && features.x86_avx512f
                    && features.x86_avx512vnni
                    && features.x86_avx512vl
            }
            Self::X2AvxVnni => {
                features.architecture == Architecture::X86_64 && features.x86_avxvnni
            }
            Self::X3aAvx2Low7HighBit | Self::X3bAvx2WidenedI16 => {
                features.architecture == Architecture::X86_64 && features.x86_avx2
            }
            Self::Scalar => true,
        }
    }
}

impl fmt::Display for KernelTier {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.id())
    }
}

/// Stable parser for `--force-tier` and `FNLP_FORCE_TIER` values.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParseTierError {
    value: String,
}

impl fmt::Display for ParseTierError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "unknown kernel tier {:?}", self.value)
    }
}

impl std::error::Error for ParseTierError {}

impl FromStr for KernelTier {
    type Err = ParseTierError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        for tier in Self::ALL {
            if value == tier.id() {
                return Ok(tier);
            }
        }
        Err(ParseTierError {
            value: value.to_owned(),
        })
    }
}

/// One-time platform feature snapshot.  Firmware/OS exposure is authoritative:
/// the dispatcher never runs a speculative instruction probe.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DetectedFeatures {
    pub aarch64_dotprod: bool,
    pub aarch64_i8mm: bool,
    pub architecture: Architecture,
    pub x86_avx2: bool,
    pub x86_avx512f: bool,
    pub x86_avx512vl: bool,
    pub x86_avx512vnni: bool,
    pub x86_avxvnni: bool,
}

impl DetectedFeatures {
    /// Detect only the target's reviewed standard-library feature surface.
    #[must_use]
    pub fn detect() -> Self {
        let mut detected = Self {
            aarch64_dotprod: false,
            aarch64_i8mm: false,
            architecture: Architecture::host(),
            x86_avx2: false,
            x86_avx512f: false,
            x86_avx512vl: false,
            x86_avx512vnni: false,
            x86_avxvnni: false,
        };

        #[cfg(target_arch = "aarch64")]
        {
            detected.aarch64_dotprod = std::arch::is_aarch64_feature_detected!("dotprod");
            detected.aarch64_i8mm = std::arch::is_aarch64_feature_detected!("i8mm");
        }
        #[cfg(target_arch = "x86_64")]
        {
            detected.x86_avx2 = std::arch::is_x86_feature_detected!("avx2");
            detected.x86_avx512f = std::arch::is_x86_feature_detected!("avx512f");
            detected.x86_avx512vl = std::arch::is_x86_feature_detected!("avx512vl");
            detected.x86_avx512vnni = std::arch::is_x86_feature_detected!("avx512vnni");
            detected.x86_avxvnni = std::arch::is_x86_feature_detected!("avxvnni");
        }
        detected
    }
}

static HOST_FEATURES: OnceLock<DetectedFeatures> = OnceLock::new();

/// Return the single feature snapshot used by all dispatch decisions in this
/// process.  Tests inject [`DetectedFeatures`] directly instead.
#[must_use]
pub fn host_features() -> DetectedFeatures {
    *HOST_FEATURES.get_or_init(DetectedFeatures::detect)
}

/// Tile geometry selected with a backend.  Scalar uses `1×1×1`; future SIMD
/// rows must carry their exact measured geometry rather than inherit it.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TileGeometry {
    pub k: usize,
    pub m: usize,
    pub n: usize,
}

impl TileGeometry {
    #[must_use]
    pub const fn scalar() -> Self {
        Self { k: 1, m: 1, n: 1 }
    }
}

/// Retained evidence for a measured choice.  All fields are required so a row
/// cannot be mistaken for a capability-derived default.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MeasurementProvenance {
    pub benchmark_id: String,
    pub host_class: String,
    pub recorded_on: String,
}

/// A wider or otherwise eligible candidate that lost a measured comparison.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MeasuredCandidate {
    pub median_ns: u64,
    pub tier: KernelTier,
}

/// A single measured selection row.  `wider_tier_losses` records actual
/// competing numbers, not a claim inferred from CPU feature bits.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DispatchRow {
    pub key: DispatchKey,
    pub provenance: MeasurementProvenance,
    pub selected_median_ns: u64,
    pub selected_tier: KernelTier,
    pub tile: TileGeometry,
    pub wider_tier_losses: Vec<MeasuredCandidate>,
}

/// A validated immutable collection of measured rows.
#[derive(Clone, Debug, Default)]
pub struct DispatchTable {
    rows: Vec<DispatchRow>,
}

impl DispatchTable {
    /// Reject duplicate, corrupt, or ambiguous rows before they can route a
    /// kernel.  Table data is deliberately inert until this succeeds.
    pub fn from_rows(rows: Vec<DispatchRow>) -> Result<Self, DispatchError> {
        for (index, row) in rows.iter().enumerate() {
            validate_row(row)?;
            if rows[..index].iter().any(|prior| prior.key == row.key) {
                return Err(DispatchError::DuplicateTableKey { key: row.key });
            }
        }
        Ok(Self { rows })
    }

    /// Parse a strict JSON table.  Serde rejects duplicate struct fields and
    /// `deny_unknown_fields` rejects accidental schema expansion.
    pub fn from_json(source: &str) -> Result<Self, DispatchError> {
        let rows = serde_json::from_str(source).map_err(|error| DispatchError::CorruptTable {
            detail: error.to_string(),
        })?;
        Self::from_rows(rows)
    }

    #[must_use]
    pub fn rows(&self) -> &[DispatchRow] {
        &self.rows
    }

    fn matching(&self, key: DispatchKey) -> Option<&DispatchRow> {
        self.rows.iter().find(|row| row.key == key)
    }
}

/// Why a selection is honest to report.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SelectionProvenance {
    Measured {
        benchmark_id: String,
        host_class: String,
        recorded_on: String,
        selected_median_ns: u64,
        wider_tier_losses: Vec<MeasuredCandidate>,
    },
    ConservativeDefault {
        detail: &'static str,
    },
}

/// A complete decision suitable for robot reporting and a later executor.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct DispatchSelection {
    pub candidates: Vec<KernelTier>,
    pub key: DispatchKey,
    pub kernel_id: &'static str,
    pub provenance: SelectionProvenance,
    pub tier: KernelTier,
    pub tile: TileGeometry,
}

/// Typed safe-dispatch failures.  In particular, unsupported forced requests
/// fail before an executor could reach an ISA-specific function.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DispatchError {
    CorruptTable {
        detail: String,
    },
    DuplicateTableKey {
        key: DispatchKey,
    },
    ForcedTierUnavailable {
        requested: KernelTier,
        detected: Vec<KernelTier>,
    },
    ForcedTierUnimplemented {
        requested: KernelTier,
    },
    InvalidMeasurement {
        detail: String,
    },
    ScalarKernel(Int8KernelError),
}

impl fmt::Display for DispatchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CorruptTable { detail } => write!(formatter, "corrupt dispatch table: {detail}"),
            Self::DuplicateTableKey { key } => write!(formatter, "duplicate dispatch key: {key:?}"),
            Self::ForcedTierUnavailable {
                requested,
                detected,
            } => write!(
                formatter,
                "forced tier {requested} is not OS-detected; detected candidates={detected:?}"
            ),
            Self::ForcedTierUnimplemented { requested } => {
                write!(
                    formatter,
                    "forced tier {requested} has no registered kernel implementation"
                )
            }
            Self::InvalidMeasurement { detail } => {
                write!(formatter, "invalid measured dispatch row: {detail}")
            }
            Self::ScalarKernel(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for DispatchError {}

impl From<Int8KernelError> for DispatchError {
    fn from(error: Int8KernelError) -> Self {
        Self::ScalarKernel(error)
    }
}

/// Dispatch policy with injected detection for deterministic tests.
#[derive(Clone, Debug)]
pub struct Dispatcher {
    features: DetectedFeatures,
    table: DispatchTable,
}

impl Dispatcher {
    #[must_use]
    pub fn new(features: DetectedFeatures, table: DispatchTable) -> Self {
        Self { features, table }
    }

    #[must_use]
    pub fn host(table: DispatchTable) -> Self {
        Self::new(host_features(), table)
    }

    #[must_use]
    pub const fn features(&self) -> DetectedFeatures {
        self.features
    }

    #[must_use]
    pub fn detected_tiers(&self) -> Vec<KernelTier> {
        candidate_tiers(self.features)
    }

    /// Select a tier.  A forced unsupported tier fails before table lookup;
    /// a forced detected-but-not-yet-registered SIMD tier also fails safely.
    pub fn select(
        &self,
        key: DispatchKey,
        forced: Option<KernelTier>,
    ) -> Result<DispatchSelection, DispatchError> {
        let candidates = self.detected_tiers();
        if let Some(requested) = forced {
            if !candidates.contains(&requested) {
                return Err(DispatchError::ForcedTierUnavailable {
                    requested,
                    detected: candidates,
                });
            }
            if !requested.is_implemented() {
                return Err(DispatchError::ForcedTierUnimplemented { requested });
            }
            return Ok(scalar_selection(key, candidates));
        }

        if let Some(row) = self.table.matching(key) {
            if candidates.contains(&row.selected_tier) && row.selected_tier.is_implemented() {
                return Ok(DispatchSelection {
                    candidates,
                    key,
                    kernel_id: row.selected_tier.kernel_id(key.operation),
                    provenance: SelectionProvenance::Measured {
                        benchmark_id: row.provenance.benchmark_id.clone(),
                        host_class: row.provenance.host_class.clone(),
                        recorded_on: row.provenance.recorded_on.clone(),
                        selected_median_ns: row.selected_median_ns,
                        wider_tier_losses: row.wider_tier_losses.clone(),
                    },
                    tier: row.selected_tier,
                    tile: row.tile,
                });
            }
        }

        Ok(scalar_selection(key, candidates))
    }

    /// Safe int8 GEMM entry point.  Tier beads register their exact kernels
    /// here only after their differential proof; today this executes S.
    pub fn int8_gemm(
        &self,
        activations: &[i8],
        m: usize,
        k: usize,
        weights: &[i8],
        n: usize,
        regime: DispatchRegime,
        forced: Option<KernelTier>,
    ) -> Result<Dispatched<Vec<i32>>, DispatchError> {
        let key = DispatchKey::new(KernelOperation::Int8Gemm, regime, KernelShape { k, m, n });
        let selection = self.select(key, forced)?;
        let output = int8::gemm_s8s8(activations, m, k, weights, n)?;
        Ok(Dispatched { output, selection })
    }

    /// Safe int8 GEMV entry point.
    pub fn int8_gemv(
        &self,
        input: &[i8],
        weights: &[i8],
        n: usize,
        regime: DispatchRegime,
        forced: Option<KernelTier>,
    ) -> Result<Dispatched<Vec<i32>>, DispatchError> {
        let key = DispatchKey::new(
            KernelOperation::Int8Gemv,
            regime,
            KernelShape {
                k: input.len(),
                m: 1,
                n,
            },
        );
        let selection = self.select(key, forced)?;
        let output = int8::gemv_s8s8(input, weights, n)?;
        Ok(Dispatched { output, selection })
    }

    /// Safe int4-to-int8 GEMV entry point.
    pub fn int4_gemv(
        &self,
        input: &[i8],
        packed_weights: &[u8],
        n: usize,
        regime: DispatchRegime,
        forced: Option<KernelTier>,
    ) -> Result<Dispatched<Vec<i32>>, DispatchError> {
        let key = DispatchKey::new(
            KernelOperation::Int4Gemv,
            regime,
            KernelShape {
                k: input.len(),
                m: 1,
                n,
            },
        );
        let selection = self.select(key, forced)?;
        let output = int8::gemv_int4_s8(input, packed_weights, n)?;
        Ok(Dispatched { output, selection })
    }
}

/// Result of a dispatched entry point, retaining the actual policy decision.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Dispatched<T> {
    pub output: T,
    pub selection: DispatchSelection,
}

/// Robot-safe report for the current detection vector and every fixed shape.
#[derive(Clone, Debug, Serialize)]
pub struct BackendReport {
    pub architecture: Architecture,
    pub detected_features: DetectedFeatures,
    pub registry: Vec<TierReport>,
    pub selections: Vec<DispatchSelection>,
}

/// Registry entry reported even before an ISA tier supplies a kernel body.
#[derive(Clone, Debug, Serialize)]
pub struct TierReport {
    pub detected: bool,
    pub implementation: &'static str,
    pub tier: KernelTier,
}

/// Generate the no-measurement startup report.  It intentionally does not
/// invent rows: every selection says `no measurement — conservative default`.
#[must_use]
pub fn host_backend_report() -> BackendReport {
    let dispatcher = Dispatcher::host(DispatchTable::default());
    let detected_features = dispatcher.features();
    let registry = KernelTier::ALL
        .into_iter()
        .map(|tier| TierReport {
            detected: tier.is_detected(detected_features),
            implementation: if tier.is_implemented() {
                "registered"
            } else {
                "awaiting_tier_bead"
            },
            tier,
        })
        .collect();
    let mut selections = Vec::new();
    for operation in [
        KernelOperation::Int8Gemm,
        KernelOperation::Int8Gemv,
        KernelOperation::Int4Gemv,
    ] {
        for regime in DispatchRegime::ALL {
            for &k in &MODEL_KS {
                for &n in &MODEL_NS {
                    let key = DispatchKey::new(
                        operation,
                        regime,
                        KernelShape {
                            k,
                            m: regime.representative_m(),
                            n,
                        },
                    );
                    // The empty table and scalar floor make this construction
                    // infallible; keep an explicit branch if that ever changes.
                    if let Ok(selection) = dispatcher.select(key, None) {
                        selections.push(selection);
                    }
                }
            }
        }
    }
    BackendReport {
        architecture: detected_features.architecture,
        detected_features,
        registry,
        selections,
    }
}

fn candidate_tiers(features: DetectedFeatures) -> Vec<KernelTier> {
    KernelTier::ALL
        .into_iter()
        .filter(|tier| tier.is_detected(features))
        .collect()
}

fn scalar_selection(key: DispatchKey, candidates: Vec<KernelTier>) -> DispatchSelection {
    DispatchSelection {
        candidates,
        key,
        kernel_id: KernelTier::Scalar.kernel_id(key.operation),
        provenance: SelectionProvenance::ConservativeDefault {
            detail: "no measurement — conservative default",
        },
        tier: KernelTier::Scalar,
        tile: TileGeometry::scalar(),
    }
}

fn validate_row(row: &DispatchRow) -> Result<(), DispatchError> {
    if row.selected_median_ns == 0 {
        return Err(DispatchError::InvalidMeasurement {
            detail: "selected_median_ns must be non-zero".to_owned(),
        });
    }
    if row.provenance.benchmark_id.is_empty()
        || row.provenance.host_class.is_empty()
        || row.provenance.recorded_on.is_empty()
    {
        return Err(DispatchError::InvalidMeasurement {
            detail: "benchmark_id, host_class, and recorded_on are required".to_owned(),
        });
    }
    if row.tile.k == 0 || row.tile.m == 0 || row.tile.n == 0 {
        return Err(DispatchError::InvalidMeasurement {
            detail: "tile dimensions must be non-zero".to_owned(),
        });
    }
    for (index, candidate) in row.wider_tier_losses.iter().enumerate() {
        if candidate.median_ns == 0 {
            return Err(DispatchError::InvalidMeasurement {
                detail: "losing candidate median_ns must be non-zero".to_owned(),
            });
        }
        if candidate.tier == row.selected_tier
            || row.wider_tier_losses[..index]
                .iter()
                .any(|prior| prior.tier == candidate.tier)
        {
            return Err(DispatchError::InvalidMeasurement {
                detail: "losing candidate tiers must be unique and differ from selected tier"
                    .to_owned(),
            });
        }
    }
    Ok(())
}
