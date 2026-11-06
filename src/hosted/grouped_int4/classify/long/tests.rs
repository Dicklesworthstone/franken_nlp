//! Admission and ownership contracts, independent of model availability.
use super::*;
use crate::{native_engine::{lmhead::NANBEIGE_VOCAB_SIZE, strict_int4::Int4Work},
    tasks::{classify::{ClassificationPolicy, ClassificationAggregationPolicy, ClassificationPlanningError},
        mapreduce::ChunkLimits}};

fn fixture() -> (ClassificationRequest, Int4ClassificationMapHostConfig) {
    let task = TaskBudget { max_input_tokens: 768, max_output_tokens: 16, max_output_bytes: 65536,
        max_grammar_states: 4096, max_kv_bytes: Int4MemoryRequirement::for_context(1024, 32).unwrap().kv_bytes };
    let request = ClassificationRequest { document: "a complete short document".to_owned(),
        labels: ["alpha", "beta"].map(|id| ClassificationLabel { id: id.to_owned(), description: String::new() }).to_vec(),
        mode: ClassificationMode::Exclusive, policy: ClassificationPolicy::default(), budget: task };
    let config = Int4ClassificationMapHostConfig {
        native: NativeLimits { context_tokens: 1024, allocator_reserve_bytes: 65536,
            run: RunLimits { max_elapsed: Duration::from_secs(10), max_checkpoints: 1000, cleanup_reserve_bytes: 65536 } },
        planning: ClassificationLimits { max_input_bytes: 4096, max_labels: 4, max_total_label_bytes: 1024, ..Default::default() },
        mapping: Int4ClassificationMapLimits { chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 512,
            max_chunk_tokens: 768, context_tokens: 1024, reserved_tokens: 16, max_chunks: 8, max_tokenizer_calls: 1024 },
            max_model_work: Int4Work::for_sequence(32, 0, 2048, 20 * NANBEIGE_VOCAB_SIZE).unwrap(),
            max_retained_result_bytes: 1024 * 1024, max_result_bytes: 1024 * 1024,
            aggregation: ClassificationAggregationPolicy::default() },
        preparation_reserve_bytes: 128 * 1024 * 1024,
    };
    (request, config)
}
#[test]
fn host_uses_the_same_complete_pure_memory_quote_as_pre_io_callers() {
    let (request, config) = fixture();
    for width in [1, 3, 32, 128, 10752] {
        let expected = Int4ClassificationMapMemoryLimits { context_tokens: config.native.context_tokens,
            preparation_reserve_bytes: config.preparation_reserve_bytes,
            allocator_reserve_bytes: config.native.allocator_reserve_bytes }
            .requirements(width, request.budget, config.planning, config.mapping).unwrap();
        assert_eq!(config.requirements(width, request.budget).unwrap(), expected);
    }
    let mut wrong = request.budget; wrong.max_kv_bytes -= 1;
    assert!(config.requirements(32, wrong).is_err());
    let mut wrong = config; wrong.native.run.max_checkpoints = 1;
    assert!(wrong.requirements(32, request.budget).is_err());
}
#[test]
fn complete_request_semantics_are_rejected_before_planner_or_model_work() {
    for field in 0..8 {
        let (mut request, config) = fixture();
        match field { 0 => request.document.clear(),
            1 => request.document = "x".repeat(config.mapping.chunks.max_input_bytes + 1),
            2 => { request.labels.pop(); }, 3 => request.labels[1].id = request.labels[0].id.clone(),
            4 => request.labels[0].id = "\nprivate-label".to_owned(),
            5 => request.labels[0].description = "x".repeat(config.planning.max_total_label_bytes + 1),
            6 => request.policy.minimum_candidate_weight_ppm = 1_000_001,
            _ => request.policy.minimum_margin_ppm = 1_000_001 }
        assert!(validate_request(&request, config).is_err(), "field {field}");
    }
}
#[test]
fn transferred_spare_string_and_vector_capacities_remain_charged() {
    let (mut request, config) = fixture();
    let required = config.requirements(32, request.budget).unwrap();
    let mut document = String::with_capacity(config.mapping.chunks.max_input_bytes + 1024);
    document.push_str(&request.document); request.document = document;
    let mut labels = Vec::with_capacity(config.planning.max_labels + 4);
    labels.extend(request.labels); request.labels = labels;
    let mut id = String::with_capacity(2048); id.push_str(&request.labels[0].id); request.labels[0].id = id;
    let mut description = String::with_capacity(4096); description.push('x'); request.labels[0].description = description;
    validate_request(&request, config).unwrap();
    let text: usize = request.labels.iter().map(|label| label.id.capacity() + label.description.capacity()).sum();
    let excess = (request.document.capacity() - config.mapping.chunks.max_input_bytes) as u64
        + ((request.labels.capacity() - config.planning.max_labels) * std::mem::size_of::<ClassificationLabel>()) as u64
        + (text - config.planning.max_total_label_bytes) as u64;
    assert_eq!(owned_input_bytes(&request, config, required).unwrap(), required.input_bytes + excess);
    let cohort = config.cohort_requirements(32, request.budget, head_count(&request), 2).unwrap();
    assert_eq!(owned_input_bytes(&request, config, cohort.memory).unwrap(), cohort.memory.input_bytes + excess);
}
#[test]
fn all_native_cancellation_causes_survive_the_host_error_boundary() {
    use crate::native_engine::decode::DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost, ParentCancelled,
        ResourceUnavailable, Shutdown, LinkedExit] {
        let error = Int4ClassificationMapError::Classification(Int4ClassificationError::Planning(
            ClassificationPlanningError::Cancelled(cause)));
        let HostedError::Int4(Int4HostError::ClassificationMap(error)) = map_error(error) else { panic!("typed map error required") };
        assert_eq!(error.cancellation(), Some(cause));
        assert_eq!(error.to_string(), "native classification document task refused");
        let error = crate::native_engine::strict_int4::scoring::Int4ScoringError::Native(StrictInt4Error::Cancelled(cause));
        let HostedError::Int4(Int4HostError::ClassificationMap(error)) = map_error(error.into()) else {
            panic!("native cohort cause must retain the map boundary")
        };
        assert_eq!(error.cancellation(), Some(cause));
    }
}

#[test]
fn cohort_host_uses_the_same_shared_pure_simultaneous_memory_quote() {
    let (request, config) = fixture();
    for width in [1, 3, 32, 128, 10752] {
        for chunks in [1, 2, 3, 8] {
            let expected = Int4ClassificationMapMemoryLimits { context_tokens: config.native.context_tokens,
                preparation_reserve_bytes: config.preparation_reserve_bytes,
                allocator_reserve_bytes: config.native.allocator_reserve_bytes }
                .cohort_requirements(width, request.budget, config.planning, config.mapping, head_count(&request), chunks).unwrap();
            assert_eq!(config.cohort_requirements(width, request.budget, head_count(&request), chunks).unwrap(), expected);
        }
    }
    let mut invalid = config; invalid.native.run.max_checkpoints = 1;
    assert!(invalid.cohort_requirements(32, request.budget, head_count(&request), 2).is_err());
}

#[test]
fn actual_label_mode_cannot_silently_shrink_or_split_the_declared_cohort() {
    let (mut request, mut config) = fixture();
    config.planning.max_labels = 128; config.planning.max_total_label_bytes = 4096;
    config.mapping.chunks.max_chunks = 256;
    request.labels = (0..128).map(|index| ClassificationLabel { id: format!("label-{index}"),
        description: String::new() }).collect();
    // 128 competing candidates are ONE exclusive head, not 128 KV branches.
    validate_cohort_request(&request, config, 64).unwrap();
    request.mode = ClassificationMode::MultiLabel;
    assert!(validate_cohort_request(&request, config, 1).is_err());
    request.labels.truncate(32);
    validate_cohort_request(&request, config, 2).unwrap();
    assert!(validate_cohort_request(&request, config, 3).is_err());
    for chunks in [0, 65, usize::MAX] { assert!(validate_cohort_request(&request, config, chunks).is_err()); }
    request.mode = ClassificationMode::Exclusive;
    validate_cohort_request(&request, config, 64).unwrap();
}

#[test]
fn exclusive_and_multilabel_windows_keep_their_actual_independent_head_counts() {
    let (mut request, mut config) = fixture();
    config.mapping.chunks.max_chunks = 64;
    assert_eq!(head_count(&request), 1);
    validate_cohort_request(&request, config, 64).unwrap();
    assert!(validate_cohort_request(&request, config, 65).is_err());
    request.mode = ClassificationMode::MultiLabel;
    assert_eq!(head_count(&request), request.labels.len());
    validate_cohort_request(&request, config, 32).unwrap();
    assert!(validate_cohort_request(&request, config, 33).is_err());
    for chunks in [0, usize::MAX] { assert!(validate_cohort_request(&request, config, chunks).is_err()); }
    request.labels.clear();
    assert!(validate_cohort_request(&request, config, 1).is_err());
}

#[test]
fn hosted_cohorts_share_the_pure_quote_and_charge_excess_transferred_capacity() {
    let (mut request, config) = fixture();
    request.mode = ClassificationMode::MultiLabel;
    request.budget.max_kv_bytes *= request.labels.len() as u64;
    let required = config.cohort_requirements(32, request.budget, head_count(&request), 2).unwrap();
    let expected = Int4ClassificationMapMemoryLimits { context_tokens: config.native.context_tokens,
        preparation_reserve_bytes: config.preparation_reserve_bytes,
        allocator_reserve_bytes: config.native.allocator_reserve_bytes }
        .cohort_requirements(32, request.budget, config.planning, config.mapping, head_count(&request), 2).unwrap();
    assert_eq!(required, expected);
    let base = owned_input_bytes(&request, config, required.memory).unwrap();
    let mut document = String::with_capacity(config.mapping.chunks.max_input_bytes + 1234);
    document.push_str(&request.document); request.document = document;
    assert_eq!(owned_input_bytes(&request, config, required.memory).unwrap() - base,
        (request.document.capacity() - config.mapping.chunks.max_input_bytes) as u64);
    let mut short = request.budget; short.max_kv_bytes -= 1;
    assert!(config.cohort_requirements(32, short, head_count(&request), 2).is_err());
    fn send<T: Send + 'static>() {}
    send::<Int4ClassificationMapCohortRun>();
    let _cohorts = NlpEngine::map_int4_classify_cohorts;
}
