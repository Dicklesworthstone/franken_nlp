#![deny(unsafe_code)]

//! L1 per-operation differential harness.
//!
//! This test owns comparison policy, not production math.  Every row names a
//! profile and threshold table entry, and missing/duplicate threshold rows are
//! a hard error.  The frozen Phase -1 corpus currently provides whole-forward
//! traces rather than independently replayable per-op input vectors, so this
//! harness records that distinction in its report: scalar-spec comparisons are
//! executable now, while an oracle per-op claim remains unavailable until its
//! inputs are frozen.

#[path = "spec_engine/mod.rs"]
mod spec_engine;

use std::{
    collections::BTreeSet,
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};

use franken_nlp::native_engine::{
    attention::{eager_attention_head, softmax_f32_cast_back, ATTENTION_SCALE},
    diagnostic_f32::{
        greedy_argmax as diagnostic_argmax, softmax_f32, DiagnosticF32Matrix,
        DiagnosticF32RopeTables,
    },
    lmhead::{export_logits_f32, greedy_argmax as eager_argmax},
    nn::{
        embedding_row_stays_bf16, residual_add_f32_cast_back, rms_norm_f32_reduce_cast_back,
        swiglu_f32_cast_back, RMS_NORM_EPSILON,
    },
    rope::RopeTablesF32,
    tensor::Bf16,
    weights::Bf16Matrix,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const THRESHOLD_TABLE: &str = include_str!("fixtures/l1_thresholds.json");
const ORACLE_FLOOR: &str = "docs/truth-pack/oracle_floor.json";
const REFERENCE_MANIFEST: &str = "tests/fixtures/reference/manifest.json";
const HF_BF16_EAGER: &str = "hf-bf16-eager";
const DIAGNOSTIC_F32: &str = "diagnostic-f32";

const FLOAT_OPS: &[&str] = &[
    "rms_norm",
    "rope",
    "softmax",
    "gqa_qk_scores",
    "gqa_weighted_values",
    "mlp_gate_gemv",
    "mlp_up_gemv",
    "swiglu",
    "mlp_down_gemv",
    "residual_add",
    "final_norm",
    "lm_head",
];
const INTEGER_OPS: &[&str] = &["embedding_gather", "argmax", "top_k"];
const PROFILES: &[&str] = &[HF_BF16_EAGER, DIAGNOSTIC_F32];

#[derive(Debug, Deserialize)]
struct ThresholdTable {
    schema_version: u32,
    oracle_floor: FloorProvenance,
    float_rows: Vec<FloatThreshold>,
    integer_rows: Vec<IntegerThreshold>,
}

#[derive(Debug, Deserialize)]
struct FloorProvenance {
    relative_path: String,
    sha256: String,
    hf_bf16_eager_run_count_per_thread: u32,
    hf_bf16_eager_thread_counts: Vec<u32>,
    hf_bf16_eager_margin_rule: String,
    diagnostic_f32_rule: String,
}

#[derive(Debug, Deserialize)]
struct FloatThreshold {
    operation: String,
    profile: String,
    max_abs: f32,
    max_rel: f32,
    max_ulp: u32,
    min_cosine: f32,
}

#[derive(Debug, Deserialize)]
struct IntegerThreshold {
    operation: String,
    profile: String,
}

#[derive(Debug, Serialize)]
struct MetricVector {
    elements: usize,
    max_abs: f32,
    max_rel: f32,
    max_ulp: u32,
    cosine: Option<f32>,
    worst_index: Option<usize>,
    exact_nan_elements: usize,
    exact_infinite_elements: usize,
    exact_signed_zero_elements: usize,
}

#[derive(Debug, Serialize)]
struct ReportRow {
    operation: String,
    profile: String,
    status: String,
    comparison_surface: String,
    metrics: Option<MetricVector>,
    fixture_manifest_sha256: String,
    oracle_floor_sha256: String,
}

#[derive(Debug, Serialize)]
struct L1Report {
    schema_version: u32,
    status: String,
    note: String,
    rows: Vec<ReportRow>,
}

fn load_thresholds() -> ThresholdTable {
    let table: ThresholdTable =
        serde_json::from_str(THRESHOLD_TABLE).expect("L1 threshold table must be valid JSON");
    assert_eq!(table.schema_version, 1, "unknown L1 threshold schema");
    assert_eq!(table.oracle_floor.relative_path, ORACLE_FLOOR);
    assert_eq!(
        sha256_path(repo_path(&table.oracle_floor.relative_path)),
        table.oracle_floor.sha256,
        "L1 threshold table must bind the measured oracle floor bytes"
    );
    assert_eq!(table.oracle_floor.hf_bf16_eager_run_count_per_thread, 5);
    assert_eq!(table.oracle_floor.hf_bf16_eager_thread_counts, vec![1, 10]);
    assert!(
        !table.oracle_floor.hf_bf16_eager_margin_rule.is_empty()
            && !table.oracle_floor.diagnostic_f32_rule.is_empty(),
        "threshold derivation must remain explained for both reference profiles"
    );
    table
}

fn repo_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(relative)
}

fn sha256_path(path: PathBuf) -> String {
    let bytes = fs::read(&path)
        .unwrap_or_else(|error| panic!("read {} for SHA-256: {error}", path.display()));
    format!("{:x}", Sha256::digest(bytes))
}

fn float_threshold<'a>(
    table: &'a ThresholdTable,
    operation: &str,
    profile: &str,
) -> &'a FloatThreshold {
    let matches = table
        .float_rows
        .iter()
        .filter(|row| row.operation == operation && row.profile == profile)
        .collect::<Vec<_>>();
    assert_eq!(
        matches.len(),
        1,
        "L1 must fail closed on a missing or duplicate float threshold row operation={operation} profile={profile}"
    );
    matches[0]
}

fn integer_threshold<'a>(
    table: &'a ThresholdTable,
    operation: &str,
    profile: &str,
) -> &'a IntegerThreshold {
    let matches = table
        .integer_rows
        .iter()
        .filter(|row| row.operation == operation && row.profile == profile)
        .collect::<Vec<_>>();
    assert_eq!(
        matches.len(),
        1,
        "L1 must fail closed on a missing or duplicate integer threshold row operation={operation} profile={profile}"
    );
    matches[0]
}

fn ordered_f32_bits(value: f32) -> u32 {
    let bits = value.to_bits();
    if bits & 0x8000_0000 == 0 {
        bits | 0x8000_0000
    } else {
        !bits
    }
}

fn metrics(reference: &[f32], observed: &[f32]) -> Result<MetricVector, String> {
    if reference.len() != observed.len() {
        return Err(format!(
            "length mismatch expected={} observed={}",
            reference.len(),
            observed.len()
        ));
    }
    if reference.is_empty() {
        return Err("empty vectors have no L1 metric vector".to_owned());
    }

    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    let mut max_ulp = 0_u32;
    let mut worst_index = None;
    // These are reporting accumulators, so retain more precision than the f32
    // values being compared. Independent f32 reductions can otherwise make
    // bit-identical vectors appear to have a cosine below one.
    let mut dot = 0.0_f64;
    let mut reference_norm = 0.0_f64;
    let mut observed_norm = 0.0_f64;
    let mut finite_elements = 0_usize;
    let mut exact_nan_elements = 0_usize;
    let mut exact_infinite_elements = 0_usize;
    let mut exact_signed_zero_elements = 0_usize;

    for (index, (&expected, &actual)) in reference.iter().zip(observed).enumerate() {
        if expected.is_nan() || actual.is_nan() {
            if expected.to_bits() != actual.to_bits() {
                return Err(format!(
                    "NaN behavior mismatch index={index} expected_bits={:#010x} observed_bits={:#010x}",
                    expected.to_bits(),
                    actual.to_bits()
                ));
            }
            exact_nan_elements += 1;
            continue;
        }
        if expected.is_infinite() || actual.is_infinite() {
            if expected.to_bits() != actual.to_bits() {
                return Err(format!(
                    "infinite behavior mismatch index={index} expected_bits={:#010x} observed_bits={:#010x}",
                    expected.to_bits(),
                    actual.to_bits()
                ));
            }
            exact_infinite_elements += 1;
            continue;
        }
        if expected == 0.0 && actual == 0.0 && expected.to_bits() != actual.to_bits() {
            return Err(format!(
                "signed-zero behavior mismatch index={index} expected_bits={:#010x} observed_bits={:#010x}",
                expected.to_bits(),
                actual.to_bits()
            ));
        }
        if expected == 0.0 && actual == 0.0 {
            exact_signed_zero_elements += 1;
        }

        let absolute = (expected - actual).abs();
        let relative = absolute / expected.abs().max(f32::MIN_POSITIVE);
        let ulp = ordered_f32_bits(expected).abs_diff(ordered_f32_bits(actual));
        if absolute > max_abs || worst_index.is_none() {
            max_abs = absolute;
            max_rel = relative;
            max_ulp = ulp;
            worst_index = Some(index);
        } else {
            max_rel = max_rel.max(relative);
            max_ulp = max_ulp.max(ulp);
        }
        let expected_f64 = f64::from(expected);
        let actual_f64 = f64::from(actual);
        dot += expected_f64 * actual_f64;
        reference_norm += expected_f64 * expected_f64;
        observed_norm += actual_f64 * actual_f64;
        finite_elements += 1;
    }

    let cosine = if max_ulp == 0 {
        Some(1.0)
    } else if finite_elements == 0 {
        None
    } else if reference_norm == 0.0 && observed_norm == 0.0 {
        Some(1.0)
    } else if reference_norm == 0.0 || observed_norm == 0.0 {
        Some(0.0)
    } else {
        Some((dot / (reference_norm.sqrt() * observed_norm.sqrt())) as f32)
    };
    Ok(MetricVector {
        elements: reference.len(),
        max_abs,
        max_rel,
        max_ulp,
        cosine,
        worst_index,
        exact_nan_elements,
        exact_infinite_elements,
        exact_signed_zero_elements,
    })
}

fn assert_float_row(
    table: &ThresholdTable,
    operation: &str,
    profile: &str,
    reference: &[f32],
    observed: &[f32],
    fixture_manifest_sha256: &str,
) -> ReportRow {
    let threshold = float_threshold(table, operation, profile);
    let vector = metrics(reference, observed).unwrap_or_else(|error| {
        panic!("L1 op={operation} profile={profile} special-value behavior: {error}")
    });
    let cosine = vector.cosine.unwrap_or_else(|| {
        panic!("L1 op={operation} profile={profile} has no finite values for cosine")
    });
    assert!(
        vector.max_abs <= threshold.max_abs
            && vector.max_rel <= threshold.max_rel
            && vector.max_ulp <= threshold.max_ulp
            && cosine >= threshold.min_cosine,
        "L1 op={operation} profile={profile} first_index={:?} max_abs={} threshold_abs={} max_rel={} threshold_rel={} max_ulp={} threshold_ulp={} cosine={} threshold_cosine={}",
        vector.worst_index,
        vector.max_abs,
        threshold.max_abs,
        vector.max_rel,
        threshold.max_rel,
        vector.max_ulp,
        threshold.max_ulp,
        cosine,
        threshold.min_cosine,
    );
    eprintln!(
        "L1 op={operation} profile={profile} RESULT=PASS max_abs={} max_rel={} max_ulp={} cosine={cosine}",
        vector.max_abs, vector.max_rel, vector.max_ulp
    );
    ReportRow {
        operation: operation.to_owned(),
        profile: profile.to_owned(),
        status: "PASS".to_owned(),
        comparison_surface: "scalar-spec; frozen fixture matrix bound as provenance; isolated oracle op input unavailable"
            .to_owned(),
        metrics: Some(vector),
        fixture_manifest_sha256: fixture_manifest_sha256.to_owned(),
        oracle_floor_sha256: table.oracle_floor.sha256.clone(),
    }
}

fn assert_integer_row<T: Eq + std::fmt::Debug>(
    table: &ThresholdTable,
    operation: &str,
    profile: &str,
    expected: T,
    observed: T,
    fixture_manifest_sha256: &str,
) -> ReportRow {
    let _ = integer_threshold(table, operation, profile);
    assert_eq!(
        observed, expected,
        "L1 integer op={operation} profile={profile} must be bit-exact"
    );
    eprintln!("L1 op={operation} profile={profile} RESULT=PASS exact=true");
    ReportRow {
        operation: operation.to_owned(),
        profile: profile.to_owned(),
        status: "PASS".to_owned(),
        comparison_surface: "bit-exact scalar-spec selection/gather".to_owned(),
        metrics: None,
        fixture_manifest_sha256: fixture_manifest_sha256.to_owned(),
        oracle_floor_sha256: table.oracle_floor.sha256.clone(),
    }
}

fn fixture_manifest_sha256(table: &ThresholdTable) -> String {
    let manifest_path = repo_path(REFERENCE_MANIFEST);
    let manifest: serde_json::Value = serde_json::from_slice(
        &fs::read(&manifest_path)
            .unwrap_or_else(|error| panic!("read {}: {error}", manifest_path.display())),
    )
    .expect("reference manifest must be valid JSON");
    assert_eq!(
        manifest["oracle_floor_sha256"].as_str(),
        Some(table.oracle_floor.sha256.as_str()),
        "L1 must bind the same immutable floor as the fixture matrix"
    );
    let fixtures = manifest["fixtures"]
        .as_array()
        .expect("reference manifest must contain fixture entries");
    for profile in PROFILES {
        assert!(
            fixtures
                .iter()
                .any(|fixture| fixture["profile"].as_str() == Some(*profile)),
            "reference matrix must retain a trace for profile={profile}"
        );
    }
    sha256_path(manifest_path)
}

fn bf16s(values: &[f32]) -> Vec<Bf16> {
    values.iter().copied().map(Bf16::from_f32).collect()
}

fn widen(values: &[Bf16]) -> Vec<f32> {
    values.iter().map(|value| value.to_f32()).collect()
}

fn scalar_softmax_bf16(scores: &[f32]) -> Vec<Bf16> {
    let maximum = scores.iter().copied().reduce(f32::max).unwrap();
    let exponentials = scores
        .iter()
        .map(|score| (*score - maximum).exp())
        .collect::<Vec<_>>();
    let denominator = exponentials.iter().sum::<f32>();
    exponentials
        .iter()
        .map(|value| Bf16::from_f32(*value / denominator))
        .collect()
}

fn scalar_attention_bf16(
    query: &[Bf16],
    keys: &[Bf16],
    values: &[Bf16],
    sequence_len: usize,
) -> Vec<Bf16> {
    let width = 128;
    let scores = keys
        .chunks_exact(width)
        .map(|key| {
            query
                .iter()
                .zip(key)
                .map(|(left, right)| left.to_f32() * right.to_f32())
                .sum::<f32>()
                * ATTENTION_SCALE
        })
        .collect::<Vec<_>>();
    assert_eq!(scores.len(), sequence_len);
    let probabilities = scalar_softmax_bf16(&scores);
    let mut output = vec![0.0_f32; width];
    for (probability, value) in probabilities.iter().zip(values.chunks_exact(width)) {
        for (destination, source) in output.iter_mut().zip(value) {
            *destination += probability.to_f32() * source.to_f32();
        }
    }
    bf16s(&output)
}

fn scalar_project_bf16(matrix: &Bf16Matrix, input: &[Bf16]) -> Vec<Bf16> {
    (0..matrix.rows())
        .map(|row_index| {
            let value = matrix
                .row(row_index)
                .unwrap()
                .iter()
                .zip(input)
                .map(|(weight, activation)| weight.to_f32() * activation.to_f32())
                .sum::<f32>();
            Bf16::from_f32(value)
        })
        .collect()
}

fn scalar_project_f32(matrix: &Bf16Matrix, input: &[Bf16]) -> Vec<f32> {
    (0..matrix.rows())
        .map(|row_index| {
            matrix
                .row(row_index)
                .unwrap()
                .iter()
                .zip(input)
                .map(|(weight, activation)| weight.to_f32() * activation.to_f32())
                .sum::<f32>()
        })
        .collect()
}

fn top_k_indices(logits: &[f32], count: usize) -> Vec<usize> {
    let mut indices = (0..logits.len()).collect::<Vec<_>>();
    indices.sort_by(|left, right| {
        logits[*right]
            .total_cmp(&logits[*left])
            .then_with(|| left.cmp(right))
    });
    indices.truncate(count.min(indices.len()));
    indices
}

fn write_report(rows: Vec<ReportRow>) {
    let report = L1Report {
        schema_version: 1,
        status: "PARTIAL".to_owned(),
        note: "The harness is executable for current public primitive surfaces. Whole-forward Phase -1 traces bind provenance, but they are not isolated oracle per-op inputs; a later fixture expansion is required before this report can be promoted to a complete external L1 gate."
            .to_owned(),
        rows,
    };
    let bytes = serde_json::to_vec_pretty(&report).expect("L1 report must serialize");
    if let Some(path) = std::env::var_os("FRANKEN_NLP_L1_REPORT").map(PathBuf::from) {
        let mut report_file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap_or_else(|error| {
                panic!(
                    "create new L1 report {} without replacing an existing artifact: {error}",
                    path.display()
                )
            });
        report_file
            .write_all(&bytes)
            .unwrap_or_else(|error| panic!("write L1 report {}: {error}", path.display()));
        report_file
            .sync_all()
            .unwrap_or_else(|error| panic!("sync L1 report {}: {error}", path.display()));
        eprintln!(
            "L1 RESULT=PARTIAL rows={} report={}",
            report.rows.len(),
            path.display()
        );
    } else {
        let report_json = String::from_utf8(bytes).expect("serde JSON output must be UTF-8");
        eprintln!("L1 REPORT={report_json}");
        eprintln!(
            "L1 RESULT=PARTIAL rows={} report=stderr-json",
            report.rows.len()
        );
    }
}

#[test]
fn threshold_table_is_complete_unique_and_floor_bound() {
    let table = load_thresholds();
    let mut float_keys = BTreeSet::new();
    for row in &table.float_rows {
        assert!(
            FLOAT_OPS.contains(&row.operation.as_str()) && PROFILES.contains(&row.profile.as_str()),
            "threshold table has an unknown float row operation={} profile={}",
            row.operation,
            row.profile
        );
        assert!(
            row.max_abs.is_finite()
                && row.max_rel.is_finite()
                && row.min_cosine.is_finite()
                && row.max_abs >= 0.0
                && row.max_rel >= 0.0
                && (-1.0..=1.0).contains(&row.min_cosine),
            "threshold row must contain finite, ordered metric bounds"
        );
        assert!(
            float_keys.insert((row.operation.as_str(), row.profile.as_str())),
            "duplicate float threshold row operation={} profile={}",
            row.operation,
            row.profile
        );
    }
    for profile in PROFILES {
        for operation in FLOAT_OPS {
            let _ = float_threshold(&table, operation, profile);
        }
    }

    let mut integer_keys = BTreeSet::new();
    for row in &table.integer_rows {
        assert!(
            INTEGER_OPS.contains(&row.operation.as_str())
                && PROFILES.contains(&row.profile.as_str()),
            "threshold table has an unknown integer row operation={} profile={}",
            row.operation,
            row.profile
        );
        assert!(
            integer_keys.insert((row.operation.as_str(), row.profile.as_str())),
            "duplicate integer threshold row operation={} profile={}",
            row.operation,
            row.profile
        );
    }
    for profile in PROFILES {
        for operation in INTEGER_OPS {
            let _ = integer_threshold(&table, operation, profile);
        }
    }
}

#[test]
fn metric_vector_rejects_unsafe_special_value_equivalence() {
    let clean = metrics(&[1.0, -0.0, f32::INFINITY], &[1.0, -0.0, f32::INFINITY]).unwrap();
    assert_eq!(clean.max_abs, 0.0);
    assert_eq!(clean.max_ulp, 0);
    assert_eq!(clean.exact_signed_zero_elements, 1);
    assert_eq!(clean.exact_infinite_elements, 1);
    assert_eq!(clean.cosine, Some(1.0));

    let exact = metrics(
        &[0.00390625, -123.75, 8192.5, -0.03125],
        &[0.00390625, -123.75, 8192.5, -0.03125],
    )
    .expect("bit-identical vectors are metric-compatible");
    assert_eq!(exact.max_ulp, 0);
    assert_eq!(
        exact.cosine,
        Some(1.0),
        "bit-identical vectors must report an exact cosine of one"
    );

    assert!(metrics(
        &[f32::from_bits(0x7fc0_0001)],
        &[f32::from_bits(0x7fc0_0002)]
    )
    .unwrap_err()
    .contains("NaN behavior mismatch"));
    assert!(metrics(&[-0.0], &[0.0])
        .unwrap_err()
        .contains("signed-zero behavior mismatch"));
    assert!(metrics(&[f32::INFINITY], &[f32::NEG_INFINITY])
        .unwrap_err()
        .contains("infinite behavior mismatch"));
    assert!(metrics(&[1.0], &[1.0, 2.0])
        .unwrap_err()
        .contains("length mismatch"));
}

#[test]
fn hf_bf16_eager_primitives_match_scalar_spec_and_emit_report() {
    let table = load_thresholds();
    let manifest_digest = fixture_manifest_sha256(&table);
    let mut rows = Vec::new();

    let input = bf16s(&[1.125, -2.25, 0.5, 3.75]);
    let scale = bf16s(&[1.0, 0.5, -1.25, 0.75]);
    let expected_norm = bf16s(
        &spec_engine::rms_norm(&widen(&input), &widen(&scale), RMS_NORM_EPSILON, "l1_rms").unwrap(),
    );
    let observed_norm = rms_norm_f32_reduce_cast_back(&input, &scale, RMS_NORM_EPSILON).unwrap();
    rows.push(assert_float_row(
        &table,
        "rms_norm",
        HF_BF16_EAGER,
        &widen(&expected_norm),
        &widen(&observed_norm),
        &manifest_digest,
    ));
    rows.push(assert_float_row(
        &table,
        "final_norm",
        HF_BF16_EAGER,
        &widen(&expected_norm),
        &widen(&observed_norm),
        &manifest_digest,
    ));

    let mut expected_rope = bf16s(&[1.0, -2.0, 3.0, -4.0]);
    let mut expected_rope_f32 = widen(&expected_rope);
    spec_engine::apply_rope_split_half(&mut expected_rope_f32, 1, 4, 3, 70_000_000.0).unwrap();
    expected_rope = bf16s(&expected_rope_f32);
    let mut observed_rope = bf16s(&[1.0, -2.0, 3.0, -4.0]);
    RopeTablesF32::new(4, 4, 70_000_000.0)
        .unwrap()
        .apply_split_half(3, &mut observed_rope)
        .unwrap();
    rows.push(assert_float_row(
        &table,
        "rope",
        HF_BF16_EAGER,
        &widen(&expected_rope),
        &widen(&observed_rope),
        &manifest_digest,
    ));

    let scores = [1.25, -2.0, 0.5, 3.0];
    let expected_softmax = scalar_softmax_bf16(&scores);
    let observed_softmax = softmax_f32_cast_back(&scores).unwrap();
    rows.push(assert_float_row(
        &table,
        "softmax",
        HF_BF16_EAGER,
        &widen(&expected_softmax),
        &widen(&observed_softmax),
        &manifest_digest,
    ));

    let query = bf16s(&vec![0.5; 128]);
    let keys = bf16s(
        &(0..256)
            .map(|index| (index % 7) as f32 - 3.0)
            .collect::<Vec<_>>(),
    );
    let values = bf16s(
        &(0..256)
            .map(|index| (index % 11) as f32 / 8.0)
            .collect::<Vec<_>>(),
    );
    let expected_attention = scalar_attention_bf16(&query, &keys, &values, 2);
    let observed_attention = eager_attention_head(&query, &keys, &values, 2).unwrap();
    rows.push(assert_float_row(
        &table,
        "gqa_weighted_values",
        HF_BF16_EAGER,
        &widen(&expected_attention),
        &widen(&observed_attention),
        &manifest_digest,
    ));

    let gate = Bf16Matrix::new(
        3,
        4,
        bf16s(&[
            1.0, 0.0, -1.0, 0.5, 0.5, 0.25, 0.0, -0.5, -1.0, 0.5, 0.25, 1.0,
        ]),
    )
    .unwrap();
    let up = Bf16Matrix::new(
        3,
        4,
        bf16s(&[
            0.5, 1.0, 0.0, -0.5, 1.0, -1.0, 0.5, 0.25, -0.25, 0.5, 1.0, 0.0,
        ]),
    )
    .unwrap();
    let down = Bf16Matrix::new(
        4,
        3,
        bf16s(&[
            1.0, 0.0, 0.5, -0.5, 1.0, 0.0, 0.25, 0.5, -1.0, 0.0, -0.25, 1.0,
        ]),
    )
    .unwrap();
    let expected_gate = scalar_project_bf16(&gate, &input);
    let observed_gate = gate.project_f32_accumulate_cast_back(&input).unwrap();
    rows.push(assert_float_row(
        &table,
        "mlp_gate_gemv",
        HF_BF16_EAGER,
        &widen(&expected_gate),
        &widen(&observed_gate),
        &manifest_digest,
    ));
    let expected_up = scalar_project_bf16(&up, &input);
    let observed_up = up.project_f32_accumulate_cast_back(&input).unwrap();
    rows.push(assert_float_row(
        &table,
        "mlp_up_gemv",
        HF_BF16_EAGER,
        &widen(&expected_up),
        &widen(&observed_up),
        &manifest_digest,
    ));
    let expected_swiglu = expected_gate
        .iter()
        .zip(&expected_up)
        .map(|(left, right)| {
            let activated_gate = Bf16::from_f32(left.to_f32() / (1.0 + (-left.to_f32()).exp()));
            Bf16::from_f32(activated_gate.to_f32() * right.to_f32())
        })
        .collect::<Vec<_>>();
    let observed_swiglu = swiglu_f32_cast_back(&observed_gate, &observed_up).unwrap();
    rows.push(assert_float_row(
        &table,
        "swiglu",
        HF_BF16_EAGER,
        &widen(&expected_swiglu),
        &widen(&observed_swiglu),
        &manifest_digest,
    ));
    let expected_down = scalar_project_bf16(&down, &expected_swiglu);
    let observed_down = down
        .project_f32_accumulate_cast_back(&observed_swiglu)
        .unwrap();
    rows.push(assert_float_row(
        &table,
        "mlp_down_gemv",
        HF_BF16_EAGER,
        &widen(&expected_down),
        &widen(&observed_down),
        &manifest_digest,
    ));

    let update = bf16s(&[-0.5, 1.25, 0.25, -2.0]);
    let expected_residual = input
        .iter()
        .zip(&update)
        .map(|(left, right)| Bf16::from_f32(left.to_f32() + right.to_f32()))
        .collect::<Vec<_>>();
    let observed_residual = residual_add_f32_cast_back(&input, &update).unwrap();
    rows.push(assert_float_row(
        &table,
        "residual_add",
        HF_BF16_EAGER,
        &widen(&expected_residual),
        &widen(&observed_residual),
        &manifest_digest,
    ));

    rows.push(assert_integer_row(
        &table,
        "embedding_gather",
        HF_BF16_EAGER,
        input.clone(),
        embedding_row_stays_bf16(&input),
        &manifest_digest,
    ));
    let expected_logits = scalar_project_f32(&down, &expected_swiglu);
    let observed_logits = export_logits_f32(&observed_swiglu, &down).unwrap();
    rows.push(assert_float_row(
        &table,
        "lm_head",
        HF_BF16_EAGER,
        &expected_logits,
        &observed_logits,
        &manifest_digest,
    ));
    let logits = [1.0, 3.0, 3.0, -4.0];
    rows.push(assert_integer_row(
        &table,
        "argmax",
        HF_BF16_EAGER,
        Some(1_usize),
        eager_argmax(&logits),
        &manifest_digest,
    ));
    rows.push(assert_integer_row(
        &table,
        "top_k",
        HF_BF16_EAGER,
        vec![1_usize, 2, 0],
        top_k_indices(&logits, 3),
        &manifest_digest,
    ));

    write_report(rows);
}

#[test]
fn diagnostic_f32_public_primitives_match_scalar_spec_and_bind_same_matrix() {
    let table = load_thresholds();
    let manifest_digest = fixture_manifest_sha256(&table);
    let mut rows = Vec::new();

    let matrix = DiagnosticF32Matrix::new(
        3,
        4,
        vec![
            1.0, 0.0, -1.0, 0.5, 0.5, 0.25, 0.0, -0.5, -1.0, 0.5, 0.25, 1.0,
        ],
    )
    .unwrap();
    let input = [1.125, -2.25, 0.5, 3.75];
    let expected_projection = (0..3)
        .map(|row| {
            spec_engine::stable_dot(matrix.row(row).unwrap(), &input, "l1_diagnostic_projection")
                .unwrap()
        })
        .collect::<Vec<_>>();
    let observed_projection = matrix.matvec(&input, "l1_diagnostic_projection").unwrap();
    rows.push(assert_float_row(
        &table,
        "mlp_gate_gemv",
        DIAGNOSTIC_F32,
        &expected_projection,
        &observed_projection,
        &manifest_digest,
    ));
    rows.push(assert_float_row(
        &table,
        "lm_head",
        DIAGNOSTIC_F32,
        &expected_projection,
        &observed_projection,
        &manifest_digest,
    ));

    let scores = [1.25, -2.0, 0.5, 3.0];
    let expected_softmax = {
        let maximum = scores.iter().copied().reduce(f32::max).unwrap();
        let numerators = scores
            .iter()
            .map(|score| (*score - maximum).exp())
            .collect::<Vec<_>>();
        let denominator = numerators.iter().sum::<f32>();
        numerators
            .iter()
            .map(|value| value / denominator)
            .collect::<Vec<_>>()
    };
    rows.push(assert_float_row(
        &table,
        "softmax",
        DIAGNOSTIC_F32,
        &expected_softmax,
        &softmax_f32(&scores).unwrap(),
        &manifest_digest,
    ));

    let mut expected_rope = vec![1.0, -2.0, 3.0, -4.0];
    spec_engine::apply_rope_split_half(&mut expected_rope, 1, 4, 3, 70_000_000.0).unwrap();
    let mut observed_rope = vec![1.0, -2.0, 3.0, -4.0];
    DiagnosticF32RopeTables::new(4, 4, 70_000_000.0)
        .unwrap()
        .apply_all_heads(3, &mut observed_rope)
        .unwrap();
    rows.push(assert_float_row(
        &table,
        "rope",
        DIAGNOSTIC_F32,
        &expected_rope,
        &observed_rope,
        &manifest_digest,
    ));

    let logits = [1.0, 3.0, 3.0, -4.0];
    rows.push(assert_integer_row(
        &table,
        "argmax",
        DIAGNOSTIC_F32,
        Some(1_usize),
        diagnostic_argmax(&logits),
        &manifest_digest,
    ));
    rows.push(assert_integer_row(
        &table,
        "top_k",
        DIAGNOSTIC_F32,
        vec![1_usize, 2, 0],
        top_k_indices(&logits, 3),
        &manifest_digest,
    ));

    write_report(rows);
}
