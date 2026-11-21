//! Independent-snapshot replay of actual complete-document samples.
//! This is an ignored controller gate, never synthetic model-parity evidence.
use super::*;
use std::{collections::BTreeMap, io::{self, Cursor}, path::PathBuf, sync::Mutex};
use crate::{LeakResponsePolicy, ResourceHostConfig, RuntimePreset,
    native_engine::strict_int4::weights::snapshot::{Int4SnapshotBudget, Int4SnapshotPin, Int4SnapshotRequirements}};

struct Writer(Arc<Mutex<Vec<u8>>>);
impl Write for Writer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes); Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> { Ok(()) }
}
fn records(bytes: &[u8]) -> BTreeMap<String, serde_json::Value> {
    bytes.split(|&byte| byte == b'\n').filter(|line| !line.is_empty()).filter_map(|line| {
        let event: serde_json::Value = serde_json::from_slice(line).unwrap();
        if event["event"] != "doc" { return None; }
        Some((event["caller_id"].as_str().unwrap().to_owned(), event["result"].clone()))
    }).collect()
}
fn semantic_payload(run: &serde_json::Value) -> serde_json::Value {
    let dimensions: Vec<_> = run["chunks"].as_array().unwrap().iter().map(|chunk| {
        let axes: Vec<_> = chunk["run"]["dimensions"].as_array().unwrap().iter().map(|axis| {
            let samples: Vec<_> = axis["samples"].as_array().unwrap().iter().map(|sample|
                serde_json::json!({"sample_index":sample["sample_index"],"score":sample["score"],
                    "justification":sample["justification"],"generated_tokens":sample["generated_tokens"],
                    "mask_node_visit_charge":sample["mask_node_visit_charge"]})).collect();
            serde_json::json!({"axis":axis["axis"],"mean":axis["mean"],"median":axis["median"],
                "population_stddev":axis["population_stddev"],"samples":samples})
        }).collect();
        serde_json::json!({"chunk_id":chunk["chunk_id"],"source_span":chunk["source_span"],
            "source_tokens":chunk["source_tokens"],"dimensions":axes})
    }).collect();
    serde_json::json!({"source_span":run["source_span"],"samples_per_axis":run["samples_per_axis"],
        "aggregate":run["aggregate"],"chunks":dimensions,"generated_tokens":run["generated_tokens"],
        "mask_node_visit_charge":run["mask_node_visit_charge"]})
}

#[test]
#[ignore = "controller-only pinned real INT4 snapshot: complete source samples, stable item addresses and serial/team/prefill/reuse parity"]
fn real_sampled_document_stream_replays_original_items_across_order_flush_and_execution_schedules() {
    assert!(crate::installed_process_resources().is_none(), "requires a fresh process");
    let snapshot = PathBuf::from(std::env::var("FNLP_INT4_PREFILL_SNAPSHOT").expect("complete pinned snapshot required"));
    let digest = std::env::var("FNLP_INT4_PREFILL_MODEL_SHA256").expect("independent model-content pin required");
    Sha256Digest::from_hex(&digest).expect("canonical lowercase SHA-256 required");
    let width: usize = std::env::var("FNLP_INT4_TEST_GROUP_WIDTH").expect("explicit group width required").parse().unwrap();
    let memory: u64 = std::env::var("FNLP_INT4_TEST_MEMORY_BYTES").expect("explicit host ceiling required").parse().unwrap();
    let pin = Int4SnapshotPin { group_width: width, model_content_sha256: std::array::from_fn(|index|
        u8::from_str_radix(&digest[2 * index..2 * index + 2], 16).unwrap()) };
    let engine = NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 3,
        helper_threads: 0, thread_ceiling: 5, memory_ceiling_bytes: memory,
        leak_response_policy: LeakResponsePolicy::PanicInLabOrCi,
    }).build().unwrap();
    let run = RunLimits { max_elapsed: Duration::from_secs(86_400), max_checkpoints: 10_000_000_000,
        cleanup_reserve_bytes: 65_536 };
    let load = Int4SnapshotRequirements::for_group_width(width).unwrap();
    let model = engine.load_int4_snapshot(snapshot, pin, Int4SnapshotBudget {
        max_weight_bytes: load.weight_payload_bytes, max_scratch_bytes: load.scratch_bytes,
        max_io_bytes: load.read_io_bytes().unwrap(),
    }, 64 * MIB, run, CancellationToken::default()).unwrap();
    let baseline = engine.resources().memory_snapshot();
    let mut config = fixture(width);
    config.document.native.run = run;
    config.document.options.max_sample_tokens = 256;
    config.document.options.max_justification_chars = 8;
    config.task.max_output_tokens = 2048;
    config.document.mapping.chunks.max_chunk_bytes = 16;
    config.document.mapping.chunks.max_chunk_tokens = 16;
    let one = planned_work(width, config.task.max_input_tokens as usize, 256).unwrap();
    let item = (0..64).try_fold(Int4Work::default(), |sum, _| sum.checked_add(one)).unwrap();
    config.document.mapping.max_model_work = item;
    config.max_model_work = item.checked_add(item).unwrap();
    config.transport.max_work = BatchWork { forward_positions: config.max_model_work.forward_positions,
        projected_logits: config.max_model_work.projected_logits };
    config.transport.max_output_bytes = 32 * MIB;
    config.io_reserve_bytes = 64 * MIB;
    let a = "{\"id\":\"a\",\"text\":\"The staff were warm. The wait was awful.\",\"task_args\":{\"axes\":[\"valence\"]}}\n";
    let b = "{\"id\":\"b\",\"text\":\"Calm cafe. Crowded noisy waiting room.\",\"task_args\":{\"axes\":[\"valence\"]}}\n";
    let inputs = [format!("{a}{b}"), format!("{b}{{\"flush\":true}}\n{a}")];
    for encoding in [UntrustedEncoding::ByteFallback,
        UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 4096, max_output_tokens: 4096,
            max_pair_checks: 16 << 20, max_pair_bytes: 256 << 20 })] {
        config.encoding = encoding;
        let mut reference: Option<BTreeMap<String, serde_json::Value>> = None;
        for schedule in 0..6 {
            let mut selected = config.clone();
            selected.document.reuse_prefill = schedule >= 3;
            selected.document.prefill = (schedule % 3 == 1).then(|| prefill(4));
            let bytes = Arc::new(Mutex::new(Vec::new()));
            let reader = Cursor::new(inputs[schedule % 2].clone().into_bytes());
            let writer = Writer(Arc::clone(&bytes));
            let summary = if schedule % 3 == 2 {
                engine.batch_int4_sampled_sentiment_documents_with_threads(&model, selected,
                    PackedInt4Kernel::scalar(), 4, reader, writer, CancellationToken::default())
            } else { engine.batch_int4_sampled_sentiment_documents(&model, selected,
                PackedInt4Kernel::scalar(), reader, writer, CancellationToken::default()) }.unwrap();
            assert_eq!(summary.succeeded, 2); assert_eq!(summary.failed, 0);
            let actual = records(&bytes.lock().unwrap()); assert_eq!(actual.len(), 2);
            for result in actual.values() {
                assert!(result["chunks"].as_array().unwrap().len() > 1);
                assert_eq!(result["single_context_equivalence"], false);
                assert_eq!(result["samples_per_axis"], 2);
                let chunks = result["chunks"].as_array().unwrap();
                assert_eq!(chunks.first().unwrap()["source_span"]["byte_start"], 0);
                assert_eq!(chunks.last().unwrap()["source_span"]["byte_end"], result["source_span"]["byte_end"]);
            }
            if let Some(reference) = &reference {
                for (id, expected) in reference { assert_eq!(semantic_payload(&actual[id]), semantic_payload(expected)); }
            } else { reference = Some(actual); }
            assert_eq!(engine.resources().memory_snapshot(), baseline);
            assert_eq!(engine.resources().outstanding_closure_snapshot().active_closures, 0);
        }
    }
}
