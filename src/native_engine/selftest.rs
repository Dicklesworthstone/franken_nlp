//! Finite, model-free execution checks for the installed CPU kernel bodies.
//! This is a diagnostic, not model parity, a benchmark, or release authority.
//! Only repository-defined synthetic inputs are used; no files, model, network,
//! process kernel policy or worker pool are opened or changed by this module.

use std::{error::Error, fmt};
use serde::Serialize;
use sha2::{Digest, Sha256};
use super::{dispatch::{self, DetectedFeatures, DispatchRegime, DispatchTable, Dispatcher,
    KernelOperation, KernelTier}, simd::{Int8Kernel, Int8KernelKind}};

pub const SELFTEST_SCHEMA_VERSION: u32 = 1;
pub const SELFTEST_SUITE_VERSION: &str = "cpu-int8-execution-v1";
const KS: [usize; 18] = [0, 1, 7, 8, 9, 31, 32, 33, 63, 64, 65, 127, 128, 129,
    3072, 6144, 10751, 10752];
const SHAPES: [(Operation, usize, usize); 5] = [
    (Operation::Gemv, 1, 3), (Operation::Gemm, 1, 3),
    (Operation::Gemm, 4, 3), (Operation::Gemm, 5, 3), (Operation::Cohort, 4, 1),
];
pub const CASES_PER_TIER: usize = KS.len() * SHAPES.len() * 2;

#[derive(Clone, Copy, Debug, Default)]
pub struct SelftestRequest { pub tier: Option<KernelTier> }
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Operation { Gemm, Gemv, Cohort }
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Pattern { Extremes, Mixed }
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TierStatus { Pass, Fail, SkippedUnsupportedIsa, NotImplemented }
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict { Pass, Fail, Unsupported }
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CaseFailure { Dispatch, Kernel, WrongKernel, OutputShape, IntegerMismatch }

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct Mismatch { pub row: usize, pub column: usize, pub expected: i32, pub actual: i32 }
#[derive(Clone, Debug, Serialize)]
pub struct CaseReport {
    pub operation: Operation, pub pattern: Pattern,
    pub m: usize, pub k: usize, pub n: usize,
    pub requested_kernel: &'static str,
    pub observed_kernel: Option<&'static str>,
    pub expected_sha256: String,
    pub actual_sha256: Option<String>,
    pub failure: Option<CaseFailure>,
    pub first_mismatch: Option<Mismatch>,
}
#[derive(Clone, Debug, Serialize)]
pub struct TierReport {
    pub tier: KernelTier, pub status: TierStatus,
    pub cases_run: usize, pub cases_passed: usize, pub cases: Vec<CaseReport>,
}
#[derive(Clone, Debug, Serialize)]
pub struct SelftestReport {
    pub schema_version: u32,
    pub suite_version: &'static str,
    pub scope: &'static str,
    pub detected_features: DetectedFeatures,
    pub requested_tier: Option<KernelTier>,
    pub cases_per_tier: usize,
    pub tiers: Vec<TierReport>,
    pub cases_run: usize,
    pub cases_passed: usize,
    pub tiers_skipped: usize,
    pub verdict: Verdict,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SelftestError { Allocation, OracleOverflow }
impl fmt::Display for SelftestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Allocation => "CPU selftest could not reserve its bounded scratch",
            Self::OracleOverflow => "CPU selftest reference exceeded its declared i32 domain",
        })
    }
}
impl Error for SelftestError {}

/// Run the fixed suite on all available tiers, or one explicitly requested tier.
/// Forced unavailable requests are reported Unsupported, not successful skips.
/// An advertised but unimplemented tier, wrong executed kernel, native refusal,
/// wrong output size, or any i64 differential failure prevents an overall Pass.
/// The largest input scratch is 5*10752 signed bytes plus 3*10752 weight bytes;
/// report population is bounded by 9*CASES_PER_TIER, independent of caller input.
pub fn run(request: SelftestRequest) -> Result<SelftestReport, SelftestError> {
    let features = dispatch::host_features();
    let dispatcher = Dispatcher::host(DispatchTable::default());
    let mut tiers = Vec::new();
    tiers.try_reserve_exact(KernelTier::ALL.len()).map_err(|_| SelftestError::Allocation)?;
    for tier in KernelTier::ALL {
        if request.tier.is_some_and(|requested| requested != tier) { continue; }
        tiers.push(run_tier(tier, features, &dispatcher)?);
    }
    let cases_run = tiers.iter().map(|tier| tier.cases_run).sum();
    let cases_passed = tiers.iter().map(|tier| tier.cases_passed).sum();
    let tiers_skipped = tiers.iter().filter(|tier| tier.status == TierStatus::SkippedUnsupportedIsa).count();
    let verdict = summarize(&tiers, request.tier.is_some());
    Ok(SelftestReport { schema_version: SELFTEST_SCHEMA_VERSION,
        suite_version: SELFTEST_SUITE_VERSION,
        scope: "finite_synthetic_integer_cases_only_not_model_or_performance_qualification",
        detected_features: features, requested_tier: request.tier, cases_per_tier: CASES_PER_TIER,
        tiers, cases_run, cases_passed, tiers_skipped, verdict })
}

fn summarize(tiers: &[TierReport], forced: bool) -> Verdict {
    if tiers.is_empty() || tiers.iter().any(|tier| matches!(tier.status, TierStatus::Fail | TierStatus::NotImplemented)) {
        return Verdict::Fail;
    }
    if forced && tiers.iter().any(|tier| tier.status == TierStatus::SkippedUnsupportedIsa) {
        return Verdict::Unsupported;
    }
    let mut tested = false;
    for tier in tiers {
        if tier.status == TierStatus::Pass {
            if tier.cases_run != CASES_PER_TIER || tier.cases_passed != CASES_PER_TIER
                || tier.cases.len() != CASES_PER_TIER || tier.cases.iter().any(|case| case.failure.is_some()) {
                return Verdict::Fail;
            }
            tested = true;
        } else if tier.cases_run != 0 || tier.cases_passed != 0 || !tier.cases.is_empty() {
            return Verdict::Fail;
        }
    }
    if tested { Verdict::Pass } else { Verdict::Unsupported }
}

fn run_tier(tier: KernelTier, features: DetectedFeatures, dispatcher: &Dispatcher)
    -> Result<TierReport, SelftestError> {
    let mut result = TierReport { tier, status: TierStatus::SkippedUnsupportedIsa,
        cases_run: 0, cases_passed: 0, cases: Vec::new() };
    if !tier.is_detected(features) { return Ok(result); }
    if !tier.supports_operation(KernelOperation::Int8Gemm)
        || !tier.supports_operation(KernelOperation::Int8Gemv) {
        result.status = TierStatus::NotImplemented; return Ok(result);
    }
    // Independently reacquire a real-host executable; never trust report metadata.
    let kernel = tier.id().parse::<Int8KernelKind>().ok().and_then(|kind| Int8Kernel::detect(kind).ok());
    let Some(kernel) = kernel else { result.status = TierStatus::Fail; return Ok(result); };
    if kernel.kind().id() != tier.id() { result.status = TierStatus::Fail; return Ok(result); }
    result.cases.try_reserve_exact(CASES_PER_TIER).map_err(|_| SelftestError::Allocation)?;
    for k in KS {
        for (operation, m, n) in SHAPES {
            for pattern in [Pattern::Extremes, Pattern::Mixed] {
                let a = fixture(m, k, pattern, false)?;
                let w = fixture(n, k, pattern, true)?;
                let expected = oracle(&a, m, k, &w, n)?;
                let requested_kernel = match operation {
                    Operation::Gemm => tier.kernel_id(KernelOperation::Int8Gemm),
                    Operation::Gemv => tier.kernel_id(KernelOperation::Int8Gemv),
                    Operation::Cohort => tier.id(),
                };
                let mut case = CaseReport { operation, pattern, m, k, n, requested_kernel,
                    observed_kernel: None, expected_sha256: output_digest(&expected),
                    actual_sha256: None, failure: None, first_mismatch: None };
                let executed = execute(dispatcher, kernel, tier, operation, &a, m, k, &w, n)?;
                match executed {
                    Err(failure) => case.failure = Some(failure),
                    Ok((actual, observed_tier, observed_kernel)) => {
                        case.observed_kernel = Some(observed_kernel);
                        case.actual_sha256 = Some(output_digest(&actual));
                        let (failure, mismatch) = compare(&expected, &actual, n,
                            observed_tier == tier && observed_kernel == requested_kernel);
                        case.failure = failure; case.first_mismatch = mismatch;
                    }
                }
                result.cases_run += 1;
                if case.failure.is_none() { result.cases_passed += 1; }
                result.cases.push(case);
            }
        }
    }
    result.status = if result.cases_passed == CASES_PER_TIER { TierStatus::Pass } else { TierStatus::Fail };
    Ok(result)
}

type Executed = Result<(Vec<i32>, KernelTier, &'static str), CaseFailure>;
fn execute(dispatcher: &Dispatcher, kernel: Int8Kernel, tier: KernelTier, operation: Operation,
    a: &[i8], m: usize, k: usize, w: &[i8], n: usize) -> Result<Executed, SelftestError> {
    match operation {
        Operation::Gemm => Ok(dispatcher.int8_gemm(a, m, k, w, n,
            DispatchRegime::PrefillGemm, Some(tier)).map(|run|
                (run.output, run.selection.tier, run.selection.kernel_id)).map_err(|_| CaseFailure::Dispatch)),
        Operation::Gemv => Ok(dispatcher.int8_gemv(a, w, n,
            DispatchRegime::DecodeGemv, Some(tier)).map(|run|
                (run.output, run.selection.tier, run.selection.kernel_id)).map_err(|_| CaseFailure::Dispatch)),
        Operation::Cohort => {
            let mut inputs: [&[i8]; 4] = [&[]; 4];
            for row in 0..m { inputs[row] = &a[row * k..(row + 1) * k]; }
            let mut output = zeros(m)?;
            Ok(kernel.dot_rows_into(&inputs[..m], w, &mut output)
                .map(|()| (output, tier, kernel.kind().id())).map_err(|_| CaseFailure::Kernel))
        }
    }
}

fn zeros<T: Default + Clone>(length: usize) -> Result<Vec<T>, SelftestError> {
    let mut values = Vec::new();
    values.try_reserve_exact(length).map_err(|_| SelftestError::Allocation)?;
    values.resize(length, T::default()); Ok(values)
}
fn fixture(rows: usize, k: usize, pattern: Pattern, weights: bool) -> Result<Vec<i8>, SelftestError> {
    let mut values = zeros(rows * k)?;
    for row in 0..rows {
        for column in 0..k {
            values[row * k + column] = match pattern {
                Pattern::Extremes if weights => if row % 2 == 0 { -128 } else { 127 },
                Pattern::Extremes => [-128, 127, -127, 0, 1][row % 5],
                Pattern::Mixed if weights => ((column * 29 + row * 71 + 173) % 256) as u8 as i8,
                Pattern::Mixed => if row % 3 == 2 { 0 } else {
                    ((column * 73 + row * 17 + 19) % 256) as u8 as i8
                },
            };
        }
    }
    Ok(values)
}
fn oracle(a: &[i8], m: usize, k: usize, w: &[i8], n: usize) -> Result<Vec<i32>, SelftestError> {
    let mut result = zeros(m * n)?;
    for row in 0..m {
        for column in 0..n {
            let mut sum = 0_i64;
            for index in 0..k { sum += i64::from(a[row * k + index]) * i64::from(w[column * k + index]); }
            result[row * n + column] = i32::try_from(sum).map_err(|_| SelftestError::OracleOverflow)?;
        }
    }
    Ok(result)
}
fn output_digest(values: &[i32]) -> String {
    let mut hash = Sha256::new();
    hash.update(b"fnlp/cpu-selftest/i32-le/v1\0");
    hash.update((values.len() as u64).to_le_bytes());
    for value in values { hash.update(value.to_le_bytes()); }
    format!("{:x}", hash.finalize())
}
fn compare(expected: &[i32], actual: &[i32], n: usize, correct_kernel: bool)
    -> (Option<CaseFailure>, Option<Mismatch>) {
    if !correct_kernel { return (Some(CaseFailure::WrongKernel), None); }
    if expected.len() != actual.len() { return (Some(CaseFailure::OutputShape), None); }
    for (index, (&expected, &actual)) in expected.iter().zip(actual).enumerate() {
        if expected != actual {
            return (Some(CaseFailure::IntegerMismatch), Some(Mismatch {
                row: index / n, column: index % n, expected, actual }));
        }
    }
    (None, None)
}

#[cfg(test)]
mod tests;
