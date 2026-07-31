#![deny(unsafe_code)]

//! L2 44-execution eager-forward comparison gate.
//!
//! This target deliberately refuses to manufacture a comparison input from a
//! prompt string.  A source-backed forward is only comparable to the frozen
//! trace when that trace carries the exact prefill/append ids and a checked
//! ten-file source-root identity.  Until the fixture producer adds those
//! fields, this test reports a typed blocked state rather than a parity pass.

use std::{
    collections::BTreeMap,
    env, fs,
    path::{Path, PathBuf},
};

use franken_nlp::{
    artifact::converter::{verify_source_closure, ConversionSourceManifest},
    native_engine::{
        hf_bf16_eager::{
            HfBf16EagerEngine, HfBf16EagerForward, HfBf16EagerWeights, HF_BF16_EAGER_PROFILE,
        },
        kv::{KV_SLOT_COUNT, LOOP_COUNT, PHYSICAL_LAYER_COUNT},
        tensor::Bf16,
    },
};
use serde::Deserialize;
use sha2::{Digest, Sha256};

const PINNED_MODEL_SOURCE_ENV: &str = "FNLP_PINNED_MODEL_SOURCE";
const PINNED_SOURCE_MANIFEST: &str = "docs/truth-pack/nanbeige4.2-3b.source.json";
const TRACE_ROOT: &str = "tests/fixtures/reference/hf-bf16-eager/prompt-000";
const VERIFIED_FULL_SOURCE_STATUS: &str = "verified-full-ten-file-sha256";
const HIDDEN_SIZE: usize = 3_072;

#[derive(Debug, Deserialize)]
struct TraceBundle {
    profile: String,
    revision: String,
    #[serde(default)]
    source_closure_verification: Option<String>,
    #[serde(default)]
    source_root_sha256: Option<String>,
    #[serde(default)]
    prefill_input_ids: Option<Vec<u32>>,
    #[serde(default)]
    append_input_ids: Option<Vec<u32>>,
    prefill: TracePhase,
    append: TracePhase,
}

#[derive(Debug, Deserialize)]
struct TracePhase {
    phase: String,
    records: Vec<TraceTensor>,
}

#[derive(Debug, Deserialize)]
struct TraceTensor {
    tap_name: String,
    relative_path: PathBuf,
    sha256: String,
    byte_length: usize,
    dtype: String,
    element_size: usize,
    shape: Vec<usize>,
    #[serde(rename = "loop")]
    loop_index: Option<usize>,
    layer: Option<usize>,
}

struct PhaseRecords<'trace> {
    post_layers: BTreeMap<(usize, usize), &'trace TraceTensor>,
    post_loop_norms: BTreeMap<usize, &'trace TraceTensor>,
}

#[derive(Debug)]
struct TapMetric {
    phase: &'static str,
    position: usize,
    loop_index: usize,
    layer_index: Option<usize>,
    mismatches: usize,
    max_abs: f32,
    first_mismatch: Option<(usize, u16, u16)>,
}

fn repo_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(relative)
}

fn load_trace() -> TraceBundle {
    let root = repo_path(TRACE_ROOT);
    serde_json::from_slice(
        &fs::read(root.join("trace.json")).expect("read frozen hf-bf16 eager L2 trace"),
    )
    .expect("parse frozen hf-bf16 eager L2 trace")
}

fn load_trace_without_binding_field(field: &str) -> TraceBundle {
    let root = repo_path(TRACE_ROOT);
    let mut trace: serde_json::Value = serde_json::from_slice(
        &fs::read(root.join("trace.json")).expect("read bound hf-bf16 eager L2 trace"),
    )
    .expect("parse bound hf-bf16 eager L2 trace as a mutable synthetic refusal case");
    let object = trace
        .as_object_mut()
        .expect("frozen L2 trace must be a top-level JSON object");
    assert!(
        object.remove(field).is_some(),
        "synthetic refusal case must remove its declared binding field {field}"
    );
    serde_json::from_value(trace)
        .expect("a trace with one optional binding field removed remains parseable")
}

fn missing_l2_binding(trace: &TraceBundle) -> Vec<&'static str> {
    let mut missing = Vec::new();
    if trace.prefill_input_ids.as_ref().is_none_or(Vec::is_empty) {
        missing.push("prefill_input_ids");
    }
    if !matches!(trace.append_input_ids.as_deref(), Some([_])) {
        missing.push("append_input_ids");
    }
    if trace.source_closure_verification.as_deref() != Some(VERIFIED_FULL_SOURCE_STATUS) {
        missing.push("source_closure_verification");
    }
    if !trace.source_root_sha256.as_deref().is_some_and(|digest| {
        digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit())
    }) {
        missing.push("source_root_sha256");
    }
    missing
}

fn phase_records(phase: &TracePhase) -> PhaseRecords<'_> {
    let mut post_layers = BTreeMap::new();
    let mut post_loop_norms = BTreeMap::new();
    for record in &phase.records {
        match record.tap_name.as_str() {
            "post_layer" => {
                let key = (
                    record.loop_index.expect("post-layer trace requires loop"),
                    record.layer.expect("post-layer trace requires layer"),
                );
                assert!(
                    post_layers.insert(key, record).is_none(),
                    "{} phase repeats post-layer trace {key:?}",
                    phase.phase
                );
            }
            "post_loop_norm" => {
                let loop_index = record
                    .loop_index
                    .expect("post-loop norm trace requires loop");
                assert!(
                    post_loop_norms.insert(loop_index, record).is_none(),
                    "{} phase repeats post-loop norm trace loop={loop_index}",
                    phase.phase
                );
            }
            _ => {}
        }
    }
    assert_eq!(
        post_layers.len(),
        KV_SLOT_COUNT,
        "{} phase must retain all 44 L2 post-layer outputs",
        phase.phase
    );
    assert_eq!(
        post_loop_norms.len(),
        LOOP_COUNT,
        "{} phase must retain both post-loop norms",
        phase.phase
    );
    PhaseRecords {
        post_layers,
        post_loop_norms,
    }
}

fn expected_hidden_row(root: &Path, tensor: &TraceTensor, position: usize) -> Vec<Bf16> {
    assert_eq!(tensor.dtype, "bfloat16", "L2 hidden trace dtype");
    assert_eq!(tensor.element_size, 2, "L2 bf16 element width");
    assert_eq!(tensor.shape.last().copied(), Some(HIDDEN_SIZE));
    let row_count = tensor.shape[..tensor.shape.len() - 1]
        .iter()
        .try_fold(1_usize, |total, &dimension| total.checked_mul(dimension))
        .expect("L2 trace shape must not overflow usize");
    assert!(
        position < row_count,
        "L2 trace position={position} exceeds rows={row_count} for {}",
        tensor.relative_path.display()
    );
    let bytes = fs::read(root.join(&tensor.relative_path)).unwrap_or_else(|error| {
        panic!("read L2 tensor {}: {error}", tensor.relative_path.display())
    });
    assert_eq!(bytes.len(), tensor.byte_length);
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        tensor.sha256,
        "L2 tensor digest must match its trace descriptor"
    );
    let row_bytes = HIDDEN_SIZE * tensor.element_size;
    let start = position
        .checked_mul(row_bytes)
        .expect("L2 row offset must not overflow");
    let end = start
        .checked_add(row_bytes)
        .expect("L2 row end must not overflow");
    bytes[start..end]
        .chunks_exact(2)
        .map(|chunk| Bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
        .collect()
}

fn compare_hidden(
    phase: &'static str,
    position: usize,
    loop_index: usize,
    layer_index: Option<usize>,
    expected: &[Bf16],
    observed: &[Bf16],
) -> TapMetric {
    assert_eq!(expected.len(), HIDDEN_SIZE);
    assert_eq!(observed.len(), HIDDEN_SIZE);
    let mut metric = TapMetric {
        phase,
        position,
        loop_index,
        layer_index,
        mismatches: 0,
        max_abs: 0.0,
        first_mismatch: None,
    };
    for (index, (&expected, &observed)) in expected.iter().zip(observed).enumerate() {
        metric.max_abs = metric
            .max_abs
            .max((expected.to_f32() - observed.to_f32()).abs());
        if expected.to_bits() != observed.to_bits() {
            metric.mismatches += 1;
            metric
                .first_mismatch
                .get_or_insert((index, expected.to_bits(), observed.to_bits()));
        }
    }
    eprintln!(
        "L2_TAP phase={} position={} loop={} layer={:?} mismatches={} max_abs={} first_mismatch={:?}",
        metric.phase,
        metric.position,
        metric.loop_index,
        metric.layer_index,
        metric.mismatches,
        metric.max_abs,
        metric.first_mismatch,
    );
    metric
}

fn compare_forward(
    root: &Path,
    phase: &'static str,
    position: usize,
    expected: &PhaseRecords<'_>,
    observed: &HfBf16EagerForward,
    failures: &mut Vec<TapMetric>,
) {
    assert_eq!(observed.layer_outputs.len(), KV_SLOT_COUNT);
    for output in &observed.layer_outputs {
        assert_eq!(
            output.kv_slot,
            output.layer_index + output.loop_index * PHYSICAL_LAYER_COUNT,
            "native L2 output must retain the fixed slot mapping"
        );
        let tensor = expected
            .post_layers
            .get(&(output.loop_index, output.layer_index))
            .expect("frozen L2 trace must name every native layer output");
        let metric = compare_hidden(
            phase,
            position,
            output.loop_index,
            Some(output.layer_index),
            &expected_hidden_row(root, tensor, position),
            &output.hidden,
        );
        if metric.mismatches != 0 {
            failures.push(metric);
        }
    }
    for loop_index in 0..LOOP_COUNT {
        let tensor = expected
            .post_loop_norms
            .get(&loop_index)
            .expect("frozen L2 trace must name each post-loop norm");
        let metric = compare_hidden(
            phase,
            position,
            loop_index,
            None,
            &expected_hidden_row(root, tensor, position),
            &observed.post_loop_norms[loop_index],
        );
        if metric.mismatches != 0 {
            failures.push(metric);
        }
    }
}

#[test]
fn l2_trace_refuses_missing_replay_inputs_or_full_source_binding() {
    let bound = load_trace();
    assert_eq!(bound.profile, HF_BF16_EAGER_PROFILE);
    assert!(
        missing_l2_binding(&bound).is_empty(),
        "the frozen L2 trace must retain every replay binding; refusal cases are synthetic"
    );

    for field in [
        "prefill_input_ids",
        "append_input_ids",
        "source_closure_verification",
        "source_root_sha256",
    ] {
        let refusal = load_trace_without_binding_field(field);
        assert_eq!(refusal.profile, HF_BF16_EAGER_PROFILE);
        assert_eq!(
            missing_l2_binding(&refusal),
            vec![field],
            "removing {field} must draw its typed replay-binding refusal without relying on an incomplete repository fixture"
        );
        eprintln!(
            "L2 RESULT=PASS case=synthetic-refusal profile={} revision={} missing={field}",
            refusal.profile, refusal.revision,
        );
    }
}

/// Runs only after the producer records exact token ids and a source-root
/// digest.  The source directory itself is re-hashed through the canonical
/// ten-file manifest before the engine opens a single tensor range.
#[test]
fn armed_source_forward_compares_all_44_layers_and_two_norms() {
    let trace = load_trace();
    let missing = missing_l2_binding(&trace);
    if !missing.is_empty() {
        eprintln!(
            "L2 RESULT=BLOCKED profile={} reason=missing-fixture-binding fields={}",
            trace.profile,
            missing.join(","),
        );
        return;
    }
    let Some(source) = env::var_os(PINNED_MODEL_SOURCE_ENV).map(PathBuf::from) else {
        eprintln!(
            "L2 RESULT=SKIPPED_NO_MODEL profile={} reason={PINNED_MODEL_SOURCE_ENV}-unset",
            trace.profile,
        );
        return;
    };

    let manifest = ConversionSourceManifest::load_pinned(repo_path(PINNED_SOURCE_MANIFEST))
        .expect("L2 requires the authenticated pinned ten-file manifest");
    let closure = verify_source_closure(&source, manifest, false)
        .expect("L2 refuses a source directory before all ten members hash exactly");
    assert_eq!(
        trace.source_root_sha256.as_deref(),
        Some(closure.source_root_sha256.as_str()),
        "L2 trace source root must equal the re-verified source directory identity"
    );

    let prefill_ids = trace
        .prefill_input_ids
        .as_deref()
        .expect("binding preflight requires prefill input ids");
    let append_ids = trace
        .append_input_ids
        .as_deref()
        .expect("binding preflight requires append input ids");
    let weights = HfBf16EagerWeights::from_pinned_source(&source)
        .expect("the fully verified source closure must satisfy the eager tensor census");
    let mut engine = HfBf16EagerEngine::new(weights, prefill_ids.len() + append_ids.len())
        .expect("L2 context cap must provision all 44 K/V slots");
    let prefill = engine
        .prefill(prefill_ids)
        .expect("L2 source forward must execute the recorded prefill ids");
    let root = repo_path(TRACE_ROOT);
    let expected_prefill = phase_records(&trace.prefill);
    let expected_append = phase_records(&trace.append);
    let mut failures = Vec::new();
    for (position, forward) in prefill.iter().enumerate() {
        compare_forward(
            &root,
            "prefill",
            position,
            &expected_prefill,
            forward,
            &mut failures,
        );
    }
    let append = engine
        .decode(append_ids[0])
        .expect("L2 source forward must execute the recorded append id");
    compare_forward(&root, "append", 0, &expected_append, &append, &mut failures);

    assert!(
        failures.is_empty(),
        "L2 RESULT=FAIL profile={} mismatching_taps={} first={:?}",
        trace.profile,
        failures.len(),
        failures.first(),
    );
    eprintln!(
        "L2 RESULT=PASS profile={} prefill_positions={} append_positions={} taps_per_position={} norms_per_position={} source_root_sha256={}",
        trace.profile,
        prefill.len(),
        append_ids.len(),
        KV_SLOT_COUNT,
        LOOP_COUNT,
        closure.source_root_sha256,
    );
}
