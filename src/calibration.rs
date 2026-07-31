//! Calibration and uncertainty contracts for task-level decisions.
//!
//! Raw model scores are not confidence claims. This module keeps fitting,
//! evaluation, artifact identity, and shift fallback separate so a caller
//! cannot silently train on a locked test split or use a stale coverage claim.

use std::{collections::BTreeSet, error::Error, fmt};

use crate::{
    error::StructuredTaskStatus,
    execution_identity::{ExecutionIdentity, Sha256Digest},
};

/// Frozen schema version for calibration artifacts and reports.
pub const CALIBRATION_SCHEMA_VERSION: u32 = 1;

/// The preregistered number of equally spaced confidence bins for ECE.
pub const DEFAULT_ECE_BINS: usize = 10;

/// Errors that reject an invalid calibration workflow before it can make a claim.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CalibrationError {
    EmptyField(&'static str),
    EmptySplit(SplitName),
    DuplicateId {
        split: SplitName,
        id: String,
    },
    SplitOverlap {
        id: String,
        first: SplitName,
        second: SplitName,
    },
    UnexpectedId {
        id: String,
        expected: SplitName,
    },
    MissingId {
        id: String,
        split: SplitName,
    },
    InvalidProbability {
        id: String,
    },
    TemperatureNeedsOpenProbability {
        id: String,
    },
    InvalidThreshold,
    InvalidBootstrapResamples,
    InvalidConfidenceLevel,
    InvalidAlpha,
    MissingExchangeabilityMemo,
    InvalidValidityWindow,
    InvalidDate,
    DuplicateLabel(String),
    IdentityMismatch {
        expected: Sha256Digest,
        supplied: Sha256Digest,
    },
    IdentityKey(String),
}

impl fmt::Display for CalibrationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField(field) => {
                write!(formatter, "calibration field must not be empty: {field}")
            }
            Self::EmptySplit(split) => write!(
                formatter,
                "calibration split must not be empty: {}",
                split.name()
            ),
            Self::DuplicateId { split, id } => {
                write!(formatter, "duplicate id in {} split: {id}", split.name())
            }
            Self::SplitOverlap { id, first, second } => write!(
                formatter,
                "split overlap for id={id}: {} and {}",
                first.name(),
                second.name()
            ),
            Self::UnexpectedId { id, expected } => write!(
                formatter,
                "id={id} is not admitted to the {} split",
                expected.name()
            ),
            Self::MissingId { id, split } => {
                write!(
                    formatter,
                    "missing required id={id} from {} split",
                    split.name()
                )
            }
            Self::InvalidProbability { id } => {
                write!(
                    formatter,
                    "id={id} has a non-finite or out-of-range probability"
                )
            }
            Self::TemperatureNeedsOpenProbability { id } => write!(
                formatter,
                "temperature fitting requires 0 < probability < 1 for id={id}"
            ),
            Self::InvalidThreshold => write!(formatter, "threshold must be finite and in [0, 1]"),
            Self::InvalidBootstrapResamples => {
                write!(
                    formatter,
                    "bootstrap confidence intervals require at least two resamples"
                )
            }
            Self::InvalidConfidenceLevel => {
                write!(formatter, "confidence level must be finite and in (0, 1)")
            }
            Self::InvalidAlpha => write!(formatter, "conformal alpha must be finite and in (0, 1)"),
            Self::MissingExchangeabilityMemo => write!(
                formatter,
                "conformal calibration requires a written exchangeability memo and named population"
            ),
            Self::InvalidValidityWindow => {
                write!(formatter, "validity end must not precede validity start")
            }
            Self::InvalidDate => write!(formatter, "invalid Gregorian validity date"),
            Self::DuplicateLabel(label) => {
                write!(formatter, "duplicate calibration label: {label}")
            }
            Self::IdentityMismatch { expected, supplied } => write!(
                formatter,
                "execution identity calibration digest mismatch: expected={} supplied={}",
                expected.to_hex(),
                supplied.to_hex()
            ),
            Self::IdentityKey(error) => write!(formatter, "invalid execution identity: {error}"),
        }
    }
}

impl Error for CalibrationError {}

/// One of the three immutable, mutually exclusive task-evaluation splits.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum SplitName {
    Development,
    Calibration,
    LockedTest,
}

impl SplitName {
    /// Stable machine-readable split spelling.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Development => "development",
            Self::Calibration => "calibration",
            Self::LockedTest => "locked_test",
        }
    }
}

/// Frozen membership and digest of the development, calibration, and locked-test ids.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SplitMembership {
    development: BTreeSet<String>,
    calibration: BTreeSet<String>,
    locked_test: BTreeSet<String>,
}

impl SplitMembership {
    /// Construct disjoint, non-empty split membership sets.
    pub fn new(
        development: impl IntoIterator<Item = String>,
        calibration: impl IntoIterator<Item = String>,
        locked_test: impl IntoIterator<Item = String>,
    ) -> Result<Self, CalibrationError> {
        let development = collect_ids(SplitName::Development, development)?;
        let calibration = collect_ids(SplitName::Calibration, calibration)?;
        let locked_test = collect_ids(SplitName::LockedTest, locked_test)?;
        ensure_disjoint(
            &development,
            SplitName::Development,
            &calibration,
            SplitName::Calibration,
        )?;
        ensure_disjoint(
            &development,
            SplitName::Development,
            &locked_test,
            SplitName::LockedTest,
        )?;
        ensure_disjoint(
            &calibration,
            SplitName::Calibration,
            &locked_test,
            SplitName::LockedTest,
        )?;
        Ok(Self {
            development,
            calibration,
            locked_test,
        })
    }

    /// Return the domain-separated commitment for each frozen split.
    #[must_use]
    pub fn digests(&self) -> SplitDigests {
        SplitDigests {
            development: digest_ids(SplitName::Development, &self.development),
            calibration: digest_ids(SplitName::Calibration, &self.calibration),
            locked_test: digest_ids(SplitName::LockedTest, &self.locked_test),
        }
    }

    /// Partition scored rows into capability-restricted sets.
    pub fn partition(
        &self,
        development: Vec<LabeledScore>,
        calibration: Vec<LabeledScore>,
        locked_test: Vec<LabeledScore>,
    ) -> Result<CalibrationPartition, CalibrationError> {
        validate_rows(SplitName::Development, &self.development, &development)?;
        validate_rows(SplitName::Calibration, &self.calibration, &calibration)?;
        validate_rows(SplitName::LockedTest, &self.locked_test, &locked_test)?;
        Ok(CalibrationPartition {
            development: DevelopmentSet { rows: development },
            calibration: CalibrationSet { rows: calibration },
            locked_test: LockedTestSet { rows: locked_test },
            digests: self.digests(),
        })
    }
}

/// Stable commitments for every split, suitable for artifact and diagnostic records.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SplitDigests {
    pub development: Sha256Digest,
    pub calibration: Sha256Digest,
    pub locked_test: Sha256Digest,
}

/// One binary prediction with its human label, kept content-free except for its id.
#[derive(Clone, Debug, PartialEq)]
pub struct LabeledScore {
    id: String,
    probability: f64,
    positive: bool,
}

impl LabeledScore {
    /// Validate a probability score before it enters any split.
    pub fn new(
        id: impl Into<String>,
        probability: f64,
        positive: bool,
    ) -> Result<Self, CalibrationError> {
        let id = id.into();
        if id.is_empty() {
            return Err(CalibrationError::EmptyField("row id"));
        }
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err(CalibrationError::InvalidProbability { id });
        }
        Ok(Self {
            id,
            probability,
            positive,
        })
    }

    /// Stable item id used exclusively for split membership checks.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Raw positive-class probability; it is not a confidence claim until calibrated.
    #[must_use]
    pub const fn probability(&self) -> f64 {
        self.probability
    }

    /// Human binary label used only by the sealed split owner.
    #[must_use]
    pub const fn positive(&self) -> bool {
        self.positive
    }
}

/// A complete validated split partition.
#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationPartition {
    development: DevelopmentSet,
    calibration: CalibrationSet,
    locked_test: LockedTestSet,
    digests: SplitDigests,
}

impl CalibrationPartition {
    /// Development rows, which may tune a task recipe but never a calibrator.
    #[must_use]
    pub fn development(&self) -> &DevelopmentSet {
        &self.development
    }

    /// Calibration rows, the only rows admitted to parameter fitting.
    #[must_use]
    pub fn calibration(&self) -> &CalibrationSet {
        &self.calibration
    }

    /// Locked-test rows, the only rows admitted to reliability reporting.
    #[must_use]
    pub fn locked_test(&self) -> &LockedTestSet {
        &self.locked_test
    }

    /// Frozen split commitments used in artifacts and detailed diagnostic logging.
    #[must_use]
    pub const fn split_digests(&self) -> SplitDigests {
        self.digests
    }
}

/// Development rows. This type intentionally has no fitting API.
#[derive(Clone, Debug, PartialEq)]
pub struct DevelopmentSet {
    rows: Vec<LabeledScore>,
}

impl DevelopmentSet {
    /// Number of frozen development rows.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether the set has no rows. Valid partitions are never empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

/// Calibration rows. Only this type is accepted by fit routines.
#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationSet {
    rows: Vec<LabeledScore>,
}

impl CalibrationSet {
    /// Number of rows used to fit a parameterized calibration model.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether the set has no rows. Valid partitions are never empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

/// Locked test rows. Only this type is accepted by reliability reporting.
#[derive(Clone, Debug, PartialEq)]
pub struct LockedTestSet {
    rows: Vec<LabeledScore>,
}

impl LockedTestSet {
    /// Number of rows available for locked-test reporting.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether the set has no rows. Valid partitions are never empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

/// A fitted scalar temperature for binary probabilities.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TemperatureModel {
    temperature: f64,
    fitted_rows: usize,
}

impl TemperatureModel {
    /// Fit temperature only on the sealed calibration split with deterministic golden-section search.
    pub fn fit(calibration: &CalibrationSet) -> Result<Self, CalibrationError> {
        for row in &calibration.rows {
            if row.probability <= 0.0 || row.probability >= 1.0 {
                return Err(CalibrationError::TemperatureNeedsOpenProbability {
                    id: row.id.clone(),
                });
            }
        }

        let mut lower_beta = 0.05_f64;
        let mut upper_beta = 20.0_f64;
        let golden_ratio = 0.618_033_988_749_894_9_f64;
        let mut left = upper_beta - (upper_beta - lower_beta) * golden_ratio;
        let mut right = lower_beta + (upper_beta - lower_beta) * golden_ratio;
        let mut left_loss = temperature_loss(calibration, left);
        let mut right_loss = temperature_loss(calibration, right);
        for _ in 0..128 {
            if left_loss <= right_loss {
                upper_beta = right;
                right = left;
                right_loss = left_loss;
                left = upper_beta - (upper_beta - lower_beta) * golden_ratio;
                left_loss = temperature_loss(calibration, left);
            } else {
                lower_beta = left;
                left = right;
                left_loss = right_loss;
                right = lower_beta + (upper_beta - lower_beta) * golden_ratio;
                right_loss = temperature_loss(calibration, right);
            }
        }
        let beta = (lower_beta + upper_beta) / 2.0;
        Ok(Self {
            temperature: 1.0 / beta,
            fitted_rows: calibration.len(),
        })
    }

    /// The fitted temperature. Values greater than one soften the score distribution.
    #[must_use]
    pub const fn temperature(self) -> f64 {
        self.temperature
    }

    /// Number of calibration rows consumed during this explicit fit.
    #[must_use]
    pub const fn fitted_rows(self) -> usize {
        self.fitted_rows
    }

    /// Apply the fitted transform without mutating calibration state.
    pub fn calibrate(self, probability: f64) -> Result<f64, CalibrationError> {
        validate_probability("score", probability)?;
        if probability == 0.0 || probability == 1.0 {
            return Ok(probability);
        }
        Ok(sigmoid(logit(probability) / self.temperature))
    }

    /// Content-free fit diagnostic suitable for an evaluation receipt or test log.
    #[must_use]
    pub fn diagnostic_line(self) -> String {
        format!(
            "CALIBRATION_FIT method=temperature fitted_rows={} temperature={:.17}",
            self.fitted_rows, self.temperature
        )
    }
}

/// A monotonic isotonic regression fit represented by deterministic PAV blocks.
#[derive(Clone, Debug, PartialEq)]
pub struct IsotonicModel {
    blocks: Vec<IsotonicBlock>,
}

impl IsotonicModel {
    /// Fit a pool-adjacent-violators model only on the calibration split.
    pub fn fit(calibration: &CalibrationSet) -> Self {
        let mut rows = calibration.rows.clone();
        rows.sort_by(|left, right| {
            left.probability
                .total_cmp(&right.probability)
                .then_with(|| left.id.cmp(&right.id))
        });

        let mut blocks = Vec::<PavBlock>::new();
        let mut index = 0;
        while index < rows.len() {
            let score = rows[index].probability;
            let mut positive = 0_usize;
            let mut count = 0_usize;
            while index < rows.len() && rows[index].probability == score {
                positive += if rows[index].positive { 1 } else { 0 };
                count += 1;
                index += 1;
            }
            blocks.push(PavBlock {
                upper_score: score,
                positive,
                count,
            });
            while blocks.len() >= 2 {
                let last = blocks.len() - 1;
                if blocks[last - 1].mean() <= blocks[last].mean() {
                    break;
                }
                let right = blocks.pop().expect("length checked");
                let left = blocks.last_mut().expect("length checked");
                left.upper_score = right.upper_score;
                left.positive += right.positive;
                left.count += right.count;
            }
        }

        Self {
            blocks: blocks
                .into_iter()
                .map(|block| IsotonicBlock {
                    upper_score: block.upper_score,
                    calibrated_probability: block.mean(),
                })
                .collect(),
        }
    }

    /// Apply the monotonic piecewise-constant fit.
    pub fn calibrate(&self, probability: f64) -> Result<f64, CalibrationError> {
        validate_probability("score", probability)?;
        let mut chosen = self
            .blocks
            .first()
            .expect("validated calibration split is non-empty");
        for block in &self.blocks {
            if probability < block.upper_score {
                break;
            }
            chosen = block;
        }
        Ok(chosen.calibrated_probability)
    }

    /// Return the monotonic PAV knot sequence for audit and artifact materialization.
    #[must_use]
    pub fn blocks(&self) -> &[IsotonicBlock] {
        &self.blocks
    }
}

/// One auditable isotonic PAV knot.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IsotonicBlock {
    pub upper_score: f64,
    pub calibrated_probability: f64,
}

#[derive(Clone, Copy, Debug)]
struct PavBlock {
    upper_score: f64,
    positive: usize,
    count: usize,
}

impl PavBlock {
    fn mean(self) -> f64 {
        self.positive as f64 / self.count as f64
    }
}

/// Locked-test reliability metrics. They cannot be produced from a calibration set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReliabilityMetrics {
    pub ece: f64,
    pub brier: f64,
    pub rows: usize,
}

/// An acceptance-rate/risk point for a fixed, declared confidence threshold.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SelectiveRisk {
    pub threshold: f64,
    pub accepted: usize,
    pub abstained: usize,
    pub risk: Option<f64>,
}

/// Preregistered deterministic bootstrap parameters for a locked-test report.
///
/// The seed is logged with the result so confidence intervals are replayable;
/// callers bind it to their evaluation receipt rather than sampling ambient
/// process randomness.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BootstrapConfig {
    resamples: usize,
    confidence_level: f64,
    seed: u64,
}

impl BootstrapConfig {
    /// Validate a reproducible percentile-bootstrap configuration.
    pub fn new(
        resamples: usize,
        confidence_level: f64,
        seed: u64,
    ) -> Result<Self, CalibrationError> {
        if resamples < 2 {
            return Err(CalibrationError::InvalidBootstrapResamples);
        }
        if !confidence_level.is_finite() || !(0.0..1.0).contains(&confidence_level) {
            return Err(CalibrationError::InvalidConfidenceLevel);
        }
        Ok(Self {
            resamples,
            confidence_level,
            seed,
        })
    }

    /// Number of independently resampled locked-test replicas.
    #[must_use]
    pub const fn resamples(self) -> usize {
        self.resamples
    }

    /// Central coverage of the two-sided percentile interval.
    #[must_use]
    pub const fn confidence_level(self) -> f64 {
        self.confidence_level
    }

    /// Deterministic PRNG seed retained in the report diagnostic.
    #[must_use]
    pub const fn seed(self) -> u64 {
        self.seed
    }
}

/// A two-sided nonparametric percentile confidence interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BootstrapConfidenceInterval {
    pub lower: f64,
    pub upper: f64,
    pub confidence_level: f64,
    pub resamples: usize,
}

/// Confidence intervals for reliability metrics measured only on the locked test.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReliabilityConfidenceIntervals {
    pub ece: BootstrapConfidenceInterval,
    pub brier: BootstrapConfidenceInterval,
    /// `None` is explicit when any replica accepts no rows, so selective risk
    /// is not silently conditioned on a different population.
    pub selective_risk: Option<BootstrapConfidenceInterval>,
    pub bootstrap: BootstrapConfig,
}

impl ReliabilityMetrics {
    /// Content-free locked-test metric diagnostic; callers attach identity separately.
    #[must_use]
    pub fn diagnostic_line(self) -> String {
        format!(
            "CALIBRATION_METRICS split=locked_test rows={} ece={:.17} brier={:.17}",
            self.rows, self.ece, self.brier
        )
    }
}

impl SelectiveRisk {
    /// Content-free locked-test selective-risk diagnostic.
    #[must_use]
    pub fn diagnostic_line(self) -> String {
        let risk = self
            .risk
            .map_or_else(|| "not_computed".to_owned(), |risk| format!("{risk:.17}"));
        format!(
            "CALIBRATION_SELECTIVE_RISK split=locked_test threshold={:.17} accepted={} abstained={} risk={risk}",
            self.threshold, self.accepted, self.abstained
        )
    }
}

/// Report calibration reliability on locked-test rows only.
pub fn report_locked_test(
    locked_test: &LockedTestSet,
    calibrate: impl Fn(f64) -> Result<f64, CalibrationError>,
    acceptance_threshold: f64,
) -> Result<(ReliabilityMetrics, SelectiveRisk), CalibrationError> {
    let rows = calibrated_locked_rows(locked_test, calibrate)?;
    evaluate_locked_rows(&rows, acceptance_threshold)
}

/// Measure percentile-bootstrap uncertainty for a locked-test-only reliability report.
///
/// This is intentionally separate from fitting: every bootstrap replica draws
/// exclusively from the already-calibrated locked-test rows.  It cannot widen
/// the fit authority to development or calibration splits.
pub fn bootstrap_locked_test_confidence_intervals(
    locked_test: &LockedTestSet,
    calibrate: impl Fn(f64) -> Result<f64, CalibrationError>,
    acceptance_threshold: f64,
    bootstrap: BootstrapConfig,
) -> Result<ReliabilityConfidenceIntervals, CalibrationError> {
    let rows = calibrated_locked_rows(locked_test, calibrate)?;
    // Validate threshold even before sampling so an empty replica cannot hide
    // a malformed task policy.
    let _ = evaluate_locked_rows(&rows, acceptance_threshold)?;
    let mut state = bootstrap.seed;
    let mut ece = Vec::with_capacity(bootstrap.resamples);
    let mut brier = Vec::with_capacity(bootstrap.resamples);
    let mut selective_risk = Vec::with_capacity(bootstrap.resamples);
    let mut replica = Vec::with_capacity(rows.len());
    for _ in 0..bootstrap.resamples {
        replica.clear();
        for _ in 0..rows.len() {
            replica.push(rows[sampled_index(&mut state, rows.len())]);
        }
        let (metrics, selective) = evaluate_locked_rows(&replica, acceptance_threshold)?;
        ece.push(metrics.ece);
        brier.push(metrics.brier);
        selective_risk.push(selective.risk);
    }

    Ok(ReliabilityConfidenceIntervals {
        ece: percentile_interval(&mut ece, bootstrap),
        brier: percentile_interval(&mut brier, bootstrap),
        selective_risk: selective_risk
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .map(|mut risks| percentile_interval(&mut risks, bootstrap)),
        bootstrap,
    })
}

impl ReliabilityConfidenceIntervals {
    /// Content-free receipt line naming the method, locked-test scope, and seed.
    #[must_use]
    pub fn diagnostic_line(self) -> String {
        let risk = self.selective_risk.map_or_else(
            || "not_computed".to_owned(),
            |interval| format!("[{:.17},{:.17}]", interval.lower, interval.upper),
        );
        format!(
            "CALIBRATION_CONFIDENCE_INTERVALS split=locked_test method=nonparametric_bootstrap_percentile confidence_level={:.17} resamples={} seed={} ece=[{:.17},{:.17}] brier=[{:.17},{:.17}] selective_risk={risk}",
            self.bootstrap.confidence_level,
            self.bootstrap.resamples,
            self.bootstrap.seed,
            self.ece.lower,
            self.ece.upper,
            self.brier.lower,
            self.brier.upper,
        )
    }
}

#[derive(Clone, Copy, Debug)]
struct CalibratedLockedRow {
    probability: f64,
    positive: bool,
}

fn calibrated_locked_rows(
    locked_test: &LockedTestSet,
    calibrate: impl Fn(f64) -> Result<f64, CalibrationError>,
) -> Result<Vec<CalibratedLockedRow>, CalibrationError> {
    locked_test
        .rows
        .iter()
        .map(|row| {
            let probability = calibrate(row.probability)?;
            validate_probability(&row.id, probability)?;
            Ok(CalibratedLockedRow {
                probability,
                positive: row.positive,
            })
        })
        .collect()
}

fn evaluate_locked_rows(
    rows: &[CalibratedLockedRow],
    acceptance_threshold: f64,
) -> Result<(ReliabilityMetrics, SelectiveRisk), CalibrationError> {
    if !acceptance_threshold.is_finite() || !(0.0..=1.0).contains(&acceptance_threshold) {
        return Err(CalibrationError::InvalidThreshold);
    }
    let mut bins = [ReliabilityBin::default(); DEFAULT_ECE_BINS];
    let mut brier_sum = 0.0_f64;
    let mut accepted = 0_usize;
    let mut accepted_errors = 0_usize;
    for row in rows {
        let probability = row.probability;
        let label = if row.positive { 1.0 } else { 0.0 };
        brier_sum += (probability - label).powi(2);
        let bin = ((probability * DEFAULT_ECE_BINS as f64) as usize).min(DEFAULT_ECE_BINS - 1);
        bins[bin].count += 1;
        bins[bin].probability_sum += probability;
        bins[bin].positive_sum += label;
        if probability >= acceptance_threshold {
            accepted += 1;
            accepted_errors += if row.positive { 0 } else { 1 };
        }
    }
    let rows = rows.len();
    let ece = bins.into_iter().fold(0.0_f64, |total, bin| {
        if bin.count == 0 {
            total
        } else {
            let count = bin.count as f64;
            total
                + (count / rows as f64)
                    * ((bin.probability_sum / count) - (bin.positive_sum / count)).abs()
        }
    });
    Ok((
        ReliabilityMetrics {
            ece,
            brier: brier_sum / rows as f64,
            rows,
        },
        SelectiveRisk {
            threshold: acceptance_threshold,
            accepted,
            abstained: rows - accepted,
            risk: (accepted > 0).then(|| accepted_errors as f64 / accepted as f64),
        },
    ))
}

fn percentile_interval(
    values: &mut [f64],
    bootstrap: BootstrapConfig,
) -> BootstrapConfidenceInterval {
    values.sort_by(f64::total_cmp);
    let last = values.len() - 1;
    let lower_tail = (1.0 - bootstrap.confidence_level) / 2.0;
    let lower_index = (lower_tail * last as f64).floor() as usize;
    let upper_index = ((1.0 - lower_tail) * last as f64).ceil() as usize;
    BootstrapConfidenceInterval {
        lower: values[lower_index],
        upper: values[upper_index],
        confidence_level: bootstrap.confidence_level,
        resamples: bootstrap.resamples,
    }
}

fn sampled_index(state: &mut u64, upper: usize) -> usize {
    let upper = u64::try_from(upper).expect("usize always fits in u64 on supported targets");
    let limit = (u64::MAX / upper) * upper;
    loop {
        let value = splitmix64(state);
        if value < limit {
            return (value % upper) as usize;
        }
    }
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[derive(Clone, Copy, Debug, Default)]
struct ReliabilityBin {
    count: usize,
    probability_sum: f64,
    positive_sum: f64,
}

/// A written, named assumption required before a conformal coverage claim exists.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExchangeabilityMemo {
    pub named_population: String,
    pub unit_of_analysis: String,
    pub written_assumption: String,
}

impl ExchangeabilityMemo {
    /// Build a memo only when its named population and unit of analysis are explicit.
    pub fn new(
        named_population: impl Into<String>,
        unit_of_analysis: impl Into<String>,
        written_assumption: impl Into<String>,
    ) -> Result<Self, CalibrationError> {
        let memo = Self {
            named_population: named_population.into(),
            unit_of_analysis: unit_of_analysis.into(),
            written_assumption: written_assumption.into(),
        };
        if memo.named_population.trim().is_empty() {
            return Err(CalibrationError::EmptyField("named population"));
        }
        if memo.unit_of_analysis.trim().is_empty() {
            return Err(CalibrationError::EmptyField("unit of analysis"));
        }
        if memo.written_assumption.trim().is_empty() {
            return Err(CalibrationError::MissingExchangeabilityMemo);
        }
        Ok(memo)
    }
}

/// A binary prediction-set element.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BinaryLabel {
    Negative,
    Positive,
}

/// Split-conformal threshold fitted from calibration nonconformity scores.
#[derive(Clone, Debug, PartialEq)]
pub struct ConformalModel {
    alpha: f64,
    nonconformity_threshold: f64,
    memo: ExchangeabilityMemo,
    fitted_rows: usize,
}

impl ConformalModel {
    /// Fit an explicit split-conformal model. `None` is a typed refusal, not a default assumption.
    pub fn fit(
        calibration: &CalibrationSet,
        memo: Option<ExchangeabilityMemo>,
        alpha: f64,
    ) -> Result<Self, CalibrationError> {
        if !alpha.is_finite() || !(0.0..1.0).contains(&alpha) {
            return Err(CalibrationError::InvalidAlpha);
        }
        let memo = memo.ok_or(CalibrationError::MissingExchangeabilityMemo)?;
        let mut scores = calibration
            .rows
            .iter()
            .map(|row| {
                if row.positive {
                    1.0 - row.probability
                } else {
                    row.probability
                }
            })
            .collect::<Vec<_>>();
        scores.sort_by(f64::total_cmp);
        let rank = (((scores.len() + 1) as f64 * (1.0 - alpha)).ceil() as usize)
            .saturating_sub(1)
            .min(scores.len() - 1);
        Ok(Self {
            alpha,
            nonconformity_threshold: scores[rank],
            memo,
            fitted_rows: scores.len(),
        })
    }

    /// Return the finite-sample coverage claim's named population memo.
    #[must_use]
    pub fn memo(&self) -> &ExchangeabilityMemo {
        &self.memo
    }

    /// Nominal miscoverage rate supplied at the explicit fit event.
    #[must_use]
    pub const fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Number of calibration rows used by the conformal fit.
    #[must_use]
    pub const fn fitted_rows(&self) -> usize {
        self.fitted_rows
    }

    /// Produce a binary prediction set for one probability.
    pub fn prediction_set(&self, probability: f64) -> Result<Vec<BinaryLabel>, CalibrationError> {
        validate_probability("score", probability)?;
        let mut set = Vec::with_capacity(2);
        if probability <= self.nonconformity_threshold {
            set.push(BinaryLabel::Negative);
        }
        if 1.0 - probability <= self.nonconformity_threshold {
            set.push(BinaryLabel::Positive);
        }
        Ok(set)
    }

    /// Measure empirical coverage solely on the locked test split.
    pub fn coverage_on_locked_test(
        &self,
        locked_test: &LockedTestSet,
    ) -> Result<CoverageReport, CalibrationError> {
        let mut covered = 0_usize;
        for row in &locked_test.rows {
            let set = self.prediction_set(row.probability)?;
            let expected = if row.positive {
                BinaryLabel::Positive
            } else {
                BinaryLabel::Negative
            };
            covered += if set.contains(&expected) { 1 } else { 0 };
        }
        Ok(CoverageReport {
            named_population: self.memo.named_population.clone(),
            alpha: self.alpha,
            covered,
            total: locked_test.len(),
        })
    }
}

/// A finite-sample coverage measurement scoped to one named population.
#[derive(Clone, Debug, PartialEq)]
pub struct CoverageReport {
    pub named_population: String,
    pub alpha: f64,
    pub covered: usize,
    pub total: usize,
}

impl CoverageReport {
    /// Empirical coverage on the locked test split, not a universal guarantee.
    #[must_use]
    pub fn empirical_coverage(&self) -> f64 {
        self.covered as f64 / self.total as f64
    }

    /// Coverage is logged with its finite sample and named population, never as a universal claim.
    #[must_use]
    pub fn diagnostic_line(&self) -> String {
        format!(
            "CALIBRATION_COVERAGE split=locked_test population={} alpha={:.17} covered={} total={} empirical_coverage={:.17}",
            self.named_population,
            self.alpha,
            self.covered,
            self.total,
            self.empirical_coverage(),
        )
    }
}

/// A calendar date without a timezone, used to bind a calibration validity domain.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ValidityDate {
    year: u16,
    month: u8,
    day: u8,
}

impl ValidityDate {
    /// Construct a checked proleptic Gregorian date.
    pub fn new(year: u16, month: u8, day: u8) -> Result<Self, CalibrationError> {
        if year == 0 || month == 0 || month > 12 || day == 0 || day > days_in_month(year, month) {
            return Err(CalibrationError::InvalidDate);
        }
        Ok(Self { year, month, day })
    }
}

impl fmt::Display for ValidityDate {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{:04}-{:02}-{:02}",
            self.year, self.month, self.day
        )
    }
}

/// Inclusive validity interval for a fitted calibration artifact.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ValidityWindow {
    pub valid_from: ValidityDate,
    pub valid_through: ValidityDate,
}

impl ValidityWindow {
    /// Construct an inclusive window with no inverted dates.
    pub fn new(
        valid_from: ValidityDate,
        valid_through: ValidityDate,
    ) -> Result<Self, CalibrationError> {
        if valid_through < valid_from {
            return Err(CalibrationError::InvalidValidityWindow);
        }
        Ok(Self {
            valid_from,
            valid_through,
        })
    }

    /// Whether one explicitly observed date remains in the artifact's validity domain.
    #[must_use]
    pub fn contains(self, date: ValidityDate) -> bool {
        self.valid_from <= date && date <= self.valid_through
    }
}

/// Required action when a named shift indicator or expiry invalidates calibration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShiftPolicy {
    /// Return raw scores, explicitly marked `uncalibrated`.
    RawScoresUncalibrated,
    /// Return a successful typed abstention instead of extending a coverage claim.
    ConservativeAbstain,
}

impl ShiftPolicy {
    const fn name(self) -> &'static str {
        match self {
            Self::RawScoresUncalibrated => "raw_scores_uncalibrated",
            Self::ConservativeAbstain => "conservative_abstain",
        }
    }
}

/// The caller's explicit shift assessment; it cannot be silently omitted.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ShiftAssessment {
    InDistribution,
    Detected { indicator: String },
}

/// Immutable data whose digest must occupy `ExecutionIdentity::calibration_digest`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CalibrationArtifactSpec {
    labels: Vec<String>,
    pub validity: ValidityWindow,
    pub named_population: String,
    pub fitted_parameter_digest: Sha256Digest,
    pub split_digests: SplitDigests,
    pub shift_policy: ShiftPolicy,
}

impl CalibrationArtifactSpec {
    /// Build canonical artifact metadata. Label order is normalized because it is a set.
    pub fn new(
        labels: impl IntoIterator<Item = String>,
        validity: ValidityWindow,
        named_population: impl Into<String>,
        fitted_parameter_digest: Sha256Digest,
        split_digests: SplitDigests,
        shift_policy: ShiftPolicy,
    ) -> Result<Self, CalibrationError> {
        let mut labels = labels.into_iter().collect::<Vec<_>>();
        if labels.is_empty() {
            return Err(CalibrationError::EmptyField("label set"));
        }
        labels.sort();
        for pair in labels.windows(2) {
            if pair[0] == pair[1] {
                return Err(CalibrationError::DuplicateLabel(pair[0].clone()));
            }
        }
        if labels.iter().any(|label| label.trim().is_empty()) {
            return Err(CalibrationError::EmptyField("label"));
        }
        let named_population = named_population.into();
        if named_population.trim().is_empty() {
            return Err(CalibrationError::EmptyField("named population"));
        }
        Ok(Self {
            labels,
            validity,
            named_population,
            fitted_parameter_digest,
            split_digests,
            shift_policy,
        })
    }

    /// Sorted labels forming the artifact's declared task label set.
    #[must_use]
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Digest the calibration material that must be embedded in the execution identity.
    #[must_use]
    pub fn digest(&self) -> Sha256Digest {
        let mut bytes = Vec::new();
        append_field(
            &mut bytes,
            "schema_version",
            &CALIBRATION_SCHEMA_VERSION.to_string(),
        );
        for label in &self.labels {
            append_field(&mut bytes, "label", label);
        }
        append_field(
            &mut bytes,
            "valid_from",
            &self.validity.valid_from.to_string(),
        );
        append_field(
            &mut bytes,
            "valid_through",
            &self.validity.valid_through.to_string(),
        );
        append_field(&mut bytes, "named_population", &self.named_population);
        append_field(
            &mut bytes,
            "fitted_parameter_digest",
            &self.fitted_parameter_digest.to_hex(),
        );
        append_field(
            &mut bytes,
            "development_split_digest",
            &self.split_digests.development.to_hex(),
        );
        append_field(
            &mut bytes,
            "calibration_split_digest",
            &self.split_digests.calibration.to_hex(),
        );
        append_field(
            &mut bytes,
            "locked_test_split_digest",
            &self.split_digests.locked_test.to_hex(),
        );
        append_field(&mut bytes, "shift_policy", self.shift_policy.name());
        Sha256Digest::of_bytes(&bytes)
    }
}

/// An identity-bound calibration artifact with an explicit validity and shift policy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CalibrationArtifact {
    identity_key: Sha256Digest,
    calibration_digest: Sha256Digest,
    spec: CalibrationArtifactSpec,
}

impl CalibrationArtifact {
    /// Bind a calibration artifact to the only permitted artifact-key projection.
    pub fn new(
        identity: &ExecutionIdentity,
        spec: CalibrationArtifactSpec,
    ) -> Result<Self, CalibrationError> {
        let expected = spec.digest();
        if identity.calibration_digest != expected {
            return Err(CalibrationError::IdentityMismatch {
                expected,
                supplied: identity.calibration_digest,
            });
        }
        let identity_key = identity
            .calibration_artifact_key()
            .map_err(|error| CalibrationError::IdentityKey(error.to_string()))?;
        Ok(Self {
            identity_key,
            calibration_digest: expected,
            spec,
        })
    }

    /// The `ExecutionIdentity` calibration-artifact projection, never a partial lookalike key.
    #[must_use]
    pub const fn key(&self) -> Sha256Digest {
        self.identity_key
    }

    /// Digest of the metadata and fitted-parameter commitment embedded in the identity.
    #[must_use]
    pub const fn calibration_digest(&self) -> Sha256Digest {
        self.calibration_digest
    }

    /// Artifact metadata and its declared validity/shift domain.
    #[must_use]
    pub fn spec(&self) -> &CalibrationArtifactSpec {
        &self.spec
    }

    /// Apply expiration/shift/threshold semantics without silently extending calibration authority.
    pub fn decide(
        &self,
        calibrated_probability: f64,
        acceptance_threshold: f64,
        observed_date: ValidityDate,
        shift: ShiftAssessment,
    ) -> Result<CalibratedTaskDecision, CalibrationError> {
        validate_probability("calibrated score", calibrated_probability)?;
        if !acceptance_threshold.is_finite() || !(0.0..=1.0).contains(&acceptance_threshold) {
            return Err(CalibrationError::InvalidThreshold);
        }
        if !self.spec.validity.contains(observed_date) {
            return Ok(self.invalidated("expired calibration validity window"));
        }
        if let ShiftAssessment::Detected { indicator } = shift {
            if indicator.trim().is_empty() {
                return Err(CalibrationError::EmptyField("shift indicator"));
            }
            return Ok(self.invalidated(&format!("distribution shift: {indicator}")));
        }
        if calibrated_probability < acceptance_threshold {
            return Ok(CalibratedTaskDecision {
                status: StructuredTaskStatus::Abstained,
                calibration_state: CalibrationState::Calibrated,
                reason: "calibrated probability below declared acceptance threshold".to_owned(),
            });
        }
        Ok(CalibratedTaskDecision {
            status: StructuredTaskStatus::Completed,
            calibration_state: CalibrationState::Calibrated,
            reason: "calibrated probability met declared acceptance threshold".to_owned(),
        })
    }

    fn invalidated(&self, reason: &str) -> CalibratedTaskDecision {
        match self.spec.shift_policy {
            ShiftPolicy::RawScoresUncalibrated => CalibratedTaskDecision {
                status: StructuredTaskStatus::Completed,
                calibration_state: CalibrationState::Uncalibrated,
                reason: reason.to_owned(),
            },
            ShiftPolicy::ConservativeAbstain => CalibratedTaskDecision {
                status: StructuredTaskStatus::Abstained,
                calibration_state: CalibrationState::Invalidated,
                reason: reason.to_owned(),
            },
        }
    }

    /// Compact content-free audit logging with split digests and validity scope.
    #[must_use]
    pub fn diagnostic_line(&self) -> String {
        format!(
            "CALIBRATION artifact_key={} calibration_digest={} fitted_parameter_digest={} population={} valid_from={} valid_through={} development_split={} calibration_split={} locked_test_split={} shift_policy={}",
            self.identity_key.to_hex(),
            self.calibration_digest.to_hex(),
            self.spec.fitted_parameter_digest.to_hex(),
            self.spec.named_population,
            self.spec.validity.valid_from,
            self.spec.validity.valid_through,
            self.spec.split_digests.development.to_hex(),
            self.spec.split_digests.calibration.to_hex(),
            self.spec.split_digests.locked_test.to_hex(),
            self.spec.shift_policy.name(),
        )
    }
}

/// Whether a returned task status is calibrated, explicitly uncalibrated, or invalidated.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CalibrationState {
    Calibrated,
    Uncalibrated,
    Invalidated,
}

impl CalibrationState {
    /// Stable envelope spelling for task, robot, and receipt consumers.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Calibrated => "calibrated",
            Self::Uncalibrated => "uncalibrated",
            Self::Invalidated => "invalidated",
        }
    }
}

/// The typed result of applying a calibration artifact's acceptance policy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CalibratedTaskDecision {
    pub status: StructuredTaskStatus,
    pub calibration_state: CalibrationState,
    pub reason: String,
}

impl CalibratedTaskDecision {
    /// `abstained` remains a successful structured result with process exit zero.
    #[must_use]
    pub const fn is_success(&self) -> bool {
        self.status.exit_code().as_u8() == 0
    }
}

fn collect_ids(
    split: SplitName,
    ids: impl IntoIterator<Item = String>,
) -> Result<BTreeSet<String>, CalibrationError> {
    let mut collected = BTreeSet::new();
    for id in ids {
        if id.is_empty() {
            return Err(CalibrationError::EmptyField("split id"));
        }
        if !collected.insert(id.clone()) {
            return Err(CalibrationError::DuplicateId { split, id });
        }
    }
    if collected.is_empty() {
        return Err(CalibrationError::EmptySplit(split));
    }
    Ok(collected)
}

fn ensure_disjoint(
    left: &BTreeSet<String>,
    left_name: SplitName,
    right: &BTreeSet<String>,
    right_name: SplitName,
) -> Result<(), CalibrationError> {
    if let Some(id) = left.intersection(right).next() {
        return Err(CalibrationError::SplitOverlap {
            id: id.clone(),
            first: left_name,
            second: right_name,
        });
    }
    Ok(())
}

fn digest_ids(split: SplitName, ids: &BTreeSet<String>) -> Sha256Digest {
    let mut bytes = Vec::new();
    append_field(&mut bytes, "split", split.name());
    for id in ids {
        append_field(&mut bytes, "id", id);
    }
    Sha256Digest::of_bytes(&bytes)
}

fn validate_rows(
    split: SplitName,
    expected: &BTreeSet<String>,
    rows: &[LabeledScore],
) -> Result<(), CalibrationError> {
    let mut observed = BTreeSet::new();
    for row in rows {
        if !expected.contains(&row.id) {
            return Err(CalibrationError::UnexpectedId {
                id: row.id.clone(),
                expected: split,
            });
        }
        if !observed.insert(row.id.clone()) {
            return Err(CalibrationError::DuplicateId {
                split,
                id: row.id.clone(),
            });
        }
    }
    if let Some(id) = expected.difference(&observed).next() {
        return Err(CalibrationError::MissingId {
            id: id.clone(),
            split,
        });
    }
    Ok(())
}

fn temperature_loss(calibration: &CalibrationSet, beta: f64) -> f64 {
    calibration.rows.iter().fold(0.0_f64, |loss, row| {
        let probability = sigmoid(logit(row.probability) * beta).clamp(1e-15, 1.0 - 1e-15);
        if row.positive {
            loss - probability.ln()
        } else {
            loss - (1.0 - probability).ln()
        }
    })
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

fn logit(probability: f64) -> f64 {
    (probability / (1.0 - probability)).ln()
}

fn validate_probability(id: &str, probability: f64) -> Result<(), CalibrationError> {
    if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
        return Err(CalibrationError::InvalidProbability { id: id.to_owned() });
    }
    Ok(())
}

fn append_field(bytes: &mut Vec<u8>, name: &str, value: &str) {
    bytes.extend_from_slice(&(name.len() as u64).to_be_bytes());
    bytes.extend_from_slice(name.as_bytes());
    bytes.extend_from_slice(&(value.len() as u64).to_be_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

const fn days_in_month(year: u16, month: u8) -> u8 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => 0,
    }
}

const fn is_leap_year(year: u16) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}
