//! Explicit model-present serial/cohort parity. No quality or speed claim.
use super::*;

#[test]
#[ignore = "requires controller-admitted pinned source, explicit kernel and multi-GB native memory"]
fn complete_native_classification_cohorts_match_serial_maps_in_both_modes_and_encodings() {
    use crate::{artifact::converter::{ConversionSourceManifest, prepare_nanbeige42_input},
        native_engine::{int4::PackedInt4Kernel, simd::Int8KernelKind,
            strict_int4::{Int4ModelLoadBudget, Int4ModelLoadRequirement, Int4ModelWeights,
                Int4MemoryBudget, Int4MemoryRequirement}}};
    let directory = std::env::var("FNLP_INT4_TEST_SOURCE_DIR").expect("explicit pinned source directory required");
    let manifest = std::env::var("FNLP_INT4_TEST_SOURCE_MANIFEST").expect("explicit pinned manifest required");
    let memory_mib: u64 = std::env::var("FNLP_INT4_TEST_MEMORY_MIB").expect("explicit admitted memory required")
        .parse().unwrap();
    let kernel: Int8KernelKind = std::env::var("FNLP_INT4_TEST_KERNEL").expect("explicit native kernel required")
        .parse().unwrap();
    let kernel = PackedInt4Kernel::detect(kernel).unwrap();
    let width = 32;
    let load = Int4ModelLoadRequirement::for_group_width(width).unwrap();
    let memory_bytes = memory_mib.checked_mul(1024 * 1024).unwrap();
    let row = Int4MemoryRequirement::for_context(1024, width).unwrap();
    let task = TaskBudget { max_input_tokens: 1008, max_output_tokens: 16, max_output_bytes: 1_000_000,
        max_grammar_states: 4096, max_kv_bytes: row.kv_bytes * 3 };
    let planning = ClassificationLimits { max_labels: 3, max_input_bytes: 4096,
        max_context_tokens: 1024, ..Default::default() };
    let mapping = Int4ClassificationMapLimits { max_model_work: Int4Work { forward_positions: 131_072,
        projected_logits: 100_000_000, attention_pairs: 100_000_000_000,
        projections: ProjectionWork { dot_products: u64::MAX, multiply_accumulates: u64::MAX } }, ..mapping() };
    let memory = memory::Int4ClassificationMapMemoryLimits { context_tokens: 1024,
        preparation_reserve_bytes: memory::MIN_CLASSIFICATION_MAP_PREPARATION_BYTES,
        allocator_reserve_bytes: 16 * 1024 * 1024 };
    let encoding = UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 262_144,
        max_output_tokens: 262_144, max_pair_checks: 1 << 25, max_pair_bytes: 1 << 28 });
    let mut quote = memory.cohort_requirements(width, task, planning, mapping, 3, 2).unwrap();
    // Admit the largest selected encoder workspace before the pinned model IO.
    quote.memory = quote.memory.with_input_encoding(&encoding).unwrap();
    let execution = load.weight_payload_bytes.checked_add(quote.memory.native.kv_bytes)
        .and_then(|bytes| bytes.checked_add(quote.memory.workspace_bytes))
        .and_then(|bytes| bytes.checked_add(quote.memory.input_bytes))
        .and_then(|bytes| bytes.checked_add(quote.memory.output_reservation_bytes))
        .and_then(|bytes| bytes.checked_add(quote.memory.output_reservation_bytes)).unwrap();
    let loading = load.peak_payload_bytes().unwrap().checked_add(128 * 1024 * 1024).unwrap();
    assert!(loading.max(execution) <= memory_bytes, "controller must admit the complete declared model and map peak");
    let source = prepare_nanbeige42_input(directory, ConversionSourceManifest::load_pinned(manifest).unwrap(),
        false, 65536).unwrap();
    let weights = Int4ModelWeights::from_prepared(&source, width, Int4ModelLoadBudget {
        max_payload_bytes: load.peak_payload_bytes().unwrap(), max_source_values: load.source_values,
        max_groups: load.groups, max_shard_revalidation_bytes: load.shard_revalidation_bytes,
    }, &mut Continue).unwrap();
    drop(source);
    for policy in [UntrustedEncoding::ByteFallback, encoding] {
    for mode in [ClassificationMode::Exclusive, ClassificationMode::MultiLabel] {
        let mut request = request(mode); request.document = "0123456789abcdefghijklmn".to_owned(); request.budget = task;
        let planner = Int4ClassificationPlanner::for_model(weights.identity(), task, planning).unwrap()
            .with_document_encoding(policy.clone()).unwrap();
        let prepared = planner.plan_document_with_control(&request, mapping, &mut Continue).unwrap();
        assert_eq!(prepared.chunk_count(), 3);
        let identities = prepared.identities().unwrap(); let expected = prepared.preflight_metadata();
        let native = Int4RunBudget::exact(prepared.planned_work());
        let budget = Int4ClassificationMapCohortBudget { native,
            memory: Int4MemoryBudget { max_kv_bytes: quote.memory.native.kv_bytes, max_rope_bytes: quote.memory.native.rope_bytes,
                max_scratch_payload_bytes: quote.memory.native.scratch_payload_bound },
            max_task_workspace_bytes: quote.task_workspace_bytes, max_output_bytes: quote.max_cohort_output_bytes };
        prepared.preflight_cohorts_with_control(&identities, weights.identity(), 2, budget, &mut Continue).unwrap();
        let mut under = budget; under.native.max_forward_positions -= 1;
        assert!(prepared.preflight_cohorts_with_control(&identities, weights.identity(), 2, under, &mut Continue).is_err());
        let mut foreign = identities.clone(); foreign.last_mut().unwrap().prompt_digest =
            crate::execution_identity::Sha256Digest::of_bytes(b"foreign-last-document-map-chunk");
        assert!(prepared.preflight_cohorts_with_control(&foreign, weights.identity(), 2, budget, &mut Continue).is_err());
        let mut serial_engine = StrictInt4Engine::new(&weights, 1024, Int4MemoryBudget {
            max_kv_bytes: row.kv_bytes, max_rope_bytes: row.rope_bytes,
            max_scratch_payload_bytes: row.scratch_payload_bound }).unwrap().with_kernel(kernel);
        let serial = prepared.execute_with_control(&identities, &mut serial_engine,
            Int4ScoringBudget { native, max_kv_bytes: row.kv_bytes }, &mut Continue).unwrap();
        assert!(!serial_engine.is_poisoned()); assert!(serial_engine.kv_cache().all_slots_have_len(0));
        drop(serial_engine);
        let grouped = prepared.execute_cohorts_with_control(&identities, &weights, kernel, 2, budget, &mut Continue).unwrap();
        assert_eq!(canonjson::canonical_bytes(&grouped.output).unwrap(), canonjson::canonical_bytes(&serial).unwrap());
        assert_eq!(grouped.chunk_cohorts, 2); assert!(grouped.group_steps > 0 && grouped.projection_groups > 0);
        expected.verify_completed_cohorts(&grouped, 2).unwrap();
        assert!(prepared.execute_cohorts_with_control(&identities, &weights, kernel, 2, under, &mut Continue).is_err());
        assert!(prepared.execute_cohorts_with_control(&foreign, &weights, kernel, 2, budget, &mut Continue).is_err());
    }
    }
}
