//! Real pinned INT4 source planning, shared rule edits and owned-job storage,
//! with a PRIVATE synthetic executor. No native model inference, NER recall,
//! throughput or crash-parity evidence is claimed by these fixtures.
use super::*;
use crate::{
    artifact::converter::PINNED_REVISION,
    canonjson,
    execution_identity::{NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    grammar::mask::MaskWorkLimits,
    jobs::{FrozenManifest, JobContract, JobError, JobId, JobInput, JobSecret, MismatchField,
        TailPolicy, runner::{JobRunError, JobRunner}},
    native_engine::{decode::DecodeCancellationKind, generation::grouped_int4::INT4_SOURCE_FORMAT,
        lmhead::NANBEIGE_VOCAB_SIZE, portable_int8::int4::convert::GROUPED_INT4_RECIPE,
        strict_int4::{Int4MemoryRequirement, Int4Work, STRICT_INT4_EXECUTION, STRICT_INT4_PROFILE}},
    tasks::{ir::{PlanContext, TaskBudget}, mapreduce::{ChunkLimits, ExecutionLimits},
        ner::{EntityType, NerOptions, NER_TASK_VERSION},
        redact::{PiiKind, RedactionRequest, redact_rules,
            actions::{RedactionAction, RedactionResult, VerificationStatus},
            grouped_int4::{corpus::Int4RedactionAdmission, long::{Int4LongRedactionConfig, INT4_LONG_REDACTION_EXECUTION}},
            pseudonym::{PseudonymBudget, PseudonymEncoding, PseudonymKey}},
        source_planning::{SourcePlanningLimits,
            grouped_int4::long::{Int4SourceMapLimits, Int4SourceMapPreflight, SourceMapTask}}},
    tokenizer::{embedded::EmbeddedTokenizer, pinned_controls},
    validation::grounded_fields::GroundingBudget,
};
use serde::Serializer;
use std::{cell::Cell, fs, os::unix::fs::DirBuilderExt, path::PathBuf, rc::Rc,
    sync::atomic::{AtomicU64, Ordering}};

const WIDTH: usize = 127;
fn planner() -> SourceTaskPlanner {
    let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    let controls = pinned_controls::pinned().unwrap();
    SourceTaskPlanner::pinned(controls.template_controls(), tokenizer.eos_token_id().unwrap()).unwrap()
}
fn identity(p: &SourceTaskPlanner, width: usize) -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"private-redaction-job-fixture-no-native-authority");
    ExecutionIdentity { schema_version: 1, source_revision: PINNED_REVISION.to_owned(), logical_model_digest: d,
        artifact_format: INT4_SOURCE_FORMAT.to_owned(), quant_recipe: format!("{GROUPED_INT4_RECIPE}:group={width}"),
        packing_set_digest: Sha256Digest::of_bytes(&canonjson::canonical_bytes(&(
            INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", d, width)).unwrap()),
        tokenizer_digest: p.tokenizer_digest(), template_digest: *p.template_digest(),
        task_spec: NER_TASK_VERSION.to_owned(), taskir_digest: d, prompt_digest: d,
        grammar_compiler_version: "uncompiled".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "uncompiled".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None }
}
fn config_at(width: usize) -> Int4RedactionJobConfig {
    let p = planner();
    let one = Int4Work::for_sequence(width, 0, 8192, 64 * NANBEIGE_VOCAB_SIZE).unwrap();
    let two = one.checked_add(one).unwrap(); let four = two.checked_add(two).unwrap();
    let item = four.checked_add(four).unwrap(); let twice = item.checked_add(item).unwrap();
    let mut request = RedactionRequest::default();
    request.rule_budget.max_input_bytes = 4096;
    request.edit_budget.max_output_bytes = 65536;
    request.rules.enabled = [PiiKind::Email].into_iter().collect();
    Int4RedactionJobConfig { max_source_bytes: 512,
        batch: Int4DocumentRedactionBatchConfig { ner_identity: identity(&p, width),
            detector: Int4LongRedactionConfig {
                ner: NerOptions { types: vec![EntityType::Person], max_entities: 4, max_mention_scalars: 32 },
                per_chunk: TaskBudget { max_input_tokens: 8192, max_output_tokens: 64,
                    max_output_bytes: 65536, max_grammar_states: 4096,
                    max_kv_bytes: Int4MemoryRequirement::for_context(8192, width).unwrap().kv_bytes },
                planning: SourcePlanningLimits::default(),
                mapping: Int4SourceMapLimits {
                    chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 32, max_chunk_tokens: 64,
                        context_tokens: 8192, reserved_tokens: 64, max_chunks: 256, max_tokenizer_calls: 4096 },
                    reduction: ExecutionLimits::default(), max_model_work: item,
                    mask_limits: MaskWorkLimits::default(), mask_visits_per_chunk: 1000,
                    max_mask_visits: 256000, verification: GroundingBudget::default() },
                max_result_bytes: 65536 },
            request, max_model_work: twice.checked_add(twice).unwrap(), max_mask_visits: 1024000 } }
}
fn config() -> Int4RedactionJobConfig { config_at(WIDTH) }
fn item(c: &Int4RedactionJobConfig) -> JobWork {
    JobWork::from_int4(c.batch.item_model_work(), c.batch.detector.mapping.max_mask_visits)
}
fn limits(c: &Int4RedactionJobConfig) -> JobLimits {
    JobLimits { max_items: 2, max_id_bytes: 128, max_input_bytes_per_item: 8192,
        max_snapshot_bytes: 65536, max_result_bytes: 65536, max_spool_bytes: 1 << 20,
        max_materialized_bytes: 1 << 20, max_journal_bytes: 1 << 20, max_attempts: 4,
        max_work: JobWork::from_int4(c.batch.max_model_work, c.batch.max_mask_visits) }
}
fn axis(work: &mut JobWork, index: usize) -> &mut u64 {
    match index { 0 => &mut work.model.forward_positions, 1 => &mut work.model.projected_logits,
        2 => &mut work.model.attention_pairs, 3 => &mut work.model.projections.dot_products,
        4 => &mut work.model.projections.multiply_accumulates, _ => &mut work.mask_node_visits }
}
fn native_axis(work: &mut Int4Work, index: usize) -> &mut u64 {
    match index { 0 => &mut work.forward_positions, 1 => &mut work.projected_logits,
        2 => &mut work.attention_pairs, 3 => &mut work.projections.dot_products,
        _ => &mut work.projections.multiply_accumulates }
}
fn batch_work(work: JobWork) -> BatchWork {
    BatchWork { forward_positions: work.model.forward_positions, projected_logits: work.model.projected_logits }
}
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
struct Stop(DecodeCancellationKind);
impl DecodeStepControl for Stop { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(self.0) } }
fn inputs() -> [JobInput<'static>; 2] {
    let a = br#"{"id":"alpha","text":"Alice private contact a@example.com"}"#;
    let b = br#"{"id":"beta","text":"Bob private contact b@example.com","task_args":{}}"#;
    [JobInput { id: "alpha", original: a, normalized: a },
        JobInput { id: "beta", original: b, normalized: b }]
}
fn key() -> JobSecret { JobSecret::from_bytes([7; 32]) }
fn recipe(c: &Int4RedactionJobConfig) -> Int4RedactionJobRecipe {
    Int4RedactionJobRecipe::new(c, WIDTH, None).unwrap()
}
fn freeze(c: &Int4RedactionJobConfig, recipe: &Int4RedactionJobRecipe) -> FrozenManifest {
    FrozenManifest::freeze(&key(), JobContract { job_id: JobId([9; 16]), execution: &c.batch.ner_identity,
        recipe, limits: limits(c) }, inputs(), &mut Continue).unwrap()
}
fn fault<T>(result: Result<T, BatchFault>) -> BatchFault {
    match result { Err(e) => e, Ok(_) => panic!("expected bounded admission refusal") }
}
fn error<T>(result: Result<T, JobRunError>) -> JobRunError {
    match result { Err(e) => e, Ok(_) => panic!("expected durable refusal") }
}

#[test]
fn exact_lifetime_authority_covers_every_axis_and_stored_result_cap() {
    let c = config(); let exact = limits(&c); let r = recipe(&c);
    check_job_limits(&c, exact).unwrap(); r.check_job_limits(exact).unwrap();
    for n in 0..6 {
        for increase in [false, true] {
            let mut changed = exact; let value = axis(&mut changed.max_work, n);
            if increase { *value += 1; } else { *value -= 1; }
            assert_eq!(check_job_limits(&c, changed).unwrap_err().code, BatchCode::InvalidLimits);
            assert_eq!(r.check_job_limits(changed).unwrap_err().code, BatchCode::InvalidLimits);
        }
    }
    let mut small = exact; small.max_result_bytes -= 1;
    assert_eq!(check_job_limits(&c, small).unwrap_err().code, BatchCode::InvalidLimits);
    let mut larger_storage = exact; larger_storage.max_result_bytes += 1;
    check_job_limits(&c, larger_storage).unwrap();
}
#[test]
fn complete_item_envelope_must_fit_every_lifetime_axis_before_storage() {
    for n in 0..6 {
        let mut c = config(); let mut quoted = item(&c); let mut cap = limits(&c).max_work;
        *axis(&mut cap, n) = *axis(&mut quoted, n) - 1;
        c.batch.max_model_work = cap.int4_model_work(); c.batch.max_mask_visits = cap.mask_node_visits;
        assert_eq!(check_job_limits(&c, limits(&c)).unwrap_err().code, BatchCode::InvalidLimits);
    }
    for n in 0..5 {
        let mut c = config(); *native_axis(&mut c.batch.detector.mapping.max_model_work, n) = 0;
        assert_eq!(check_job_limits(&c, limits(&c)).unwrap_err().code, BatchCode::InvalidLimits);
    }
    let mut c = config(); c.batch.detector.mapping.max_mask_visits = 0;
    assert_eq!(check_job_limits(&c, limits(&c)).unwrap_err().code, BatchCode::InvalidLimits);
}
#[test]
fn decoded_source_envelope_is_positive_bounded_and_distinct_from_expanded_text() {
    for limit in [0, 4097, 64 * 1024 * 1024 + 1] {
        let mut c = config(); c.max_source_bytes = limit;
        assert_eq!(check_job_limits(&c, limits(&c)).unwrap_err().code, BatchCode::InvalidLimits);
    }
    let mut c = config(); c.batch.request.rule_budget.max_input_bytes = 511;
    assert_eq!(check_job_limits(&c, limits(&c)).unwrap_err().code, BatchCode::InvalidLimits);
    let c = config(); let r = recipe(&c);
    assert_eq!(r.max_source_bytes(), 512);
    assert!(c.batch.detector.mapping.chunks.max_input_bytes > r.max_source_bytes());
    check_source(512, r.max_source_bytes(), false).unwrap();
    for bytes in [0, 513, 4096] {
        let refused = check_source(bytes, r.max_source_bytes(), false).unwrap_err();
        assert_eq!(refused.fault.code, BatchCode::DocumentLimit); assert!(!refused.stop);
    }
    let poisoned = check_source(1, 512, true).unwrap_err();
    assert_eq!(poisoned.fault.code, BatchCode::InvalidExecution); assert!(poisoned.stop);
}
#[test]
fn decoded_utf8_bytes_and_encoded_json_bytes_have_independent_limits() {
    let record = br#"{"id":"u","text":"\u4e0a\u6d77","task_args":{}}"#;
    let parsed: BatchDocument<RedactionBatchArgs> = serde_json::from_slice(record).unwrap();
    assert_eq!(parsed.text.chars().count(), 2); assert_eq!(parsed.text.len(), 6);
    let mut c = config(); c.max_source_bytes = 6; let r = recipe(&c);
    assert!(record.len() > r.max_source_bytes());
    check_job_limits(&c, limits(&c)).unwrap(); check_source(parsed.text.len(), 6, false).unwrap();
    assert_eq!(check_source(parsed.text.len(), 5, false).unwrap_err().fault.code, BatchCode::DocumentLimit);
    // Execution rechecks the sealed original size against this instance's cap.
    check_source(6, 64, false).unwrap();
    assert_eq!(check_source(6, 5, false).unwrap_err().fault.code, BatchCode::DocumentLimit);
}
#[test]
fn fractional_projection_capacity_never_becomes_an_extra_whole_vocab_row() {
    let mut c = config(); let expected = item(&c);
    c.batch.detector.mapping.max_model_work.projected_logits += NANBEIGE_VOCAB_SIZE as u64 - 1;
    let r = recipe(&c); check_job_limits(&c, limits(&c)).unwrap();
    assert_eq!(r.check_work(batch_work(expected), false).unwrap(), expected);
    assert_eq!(expected.model.projected_logits % NANBEIGE_VOCAB_SIZE as u64, 0);
    c.batch.detector.mapping.max_model_work.projected_logits = NANBEIGE_VOCAB_SIZE as u64 - 1;
    assert_eq!(check_job_limits(&c, limits(&c)).unwrap_err().code, BatchCode::InvalidLimits);
}
#[test]
fn prepared_quote_checks_both_transport_counters_and_never_refunds_other_axes() {
    let c = config(); let expected = item(&c); let r = recipe(&c); let work = batch_work(expected);
    assert_eq!(r.check_work(work, false).unwrap(), expected);
    for n in 0..2 {
        let mut corrupt = work;
        if n == 0 { corrupt.forward_positions -= 1; } else { corrupt.projected_logits -= 1; }
        let failed = r.check_work(corrupt, false).unwrap_err();
        assert!(failed.stop); assert_eq!(failed.fault.code, BatchCode::InvalidExecution);
    }
    assert!(r.check_work(work, true).unwrap_err().stop);
    assert_eq!(expected.int4_model_work(), c.batch.item_model_work());
}
#[test]
fn independent_context_allowances_preserve_actual_group_partition_dots() {
    let small = config_at(16); let large = config_at(127);
    let a = item(&small); let b = item(&large);
    assert_eq!(a.model.forward_positions, b.model.forward_positions);
    assert_eq!(a.model.projected_logits, b.model.projected_logits);
    assert_eq!(a.model.attention_pairs, b.model.attention_pairs);
    assert!(a.model.projections.dot_products > b.model.projections.dot_products);
    assert_eq!(a.model.projections.multiply_accumulates, b.model.projections.multiply_accumulates);
    // Eight independently reset contexts are not one concatenated context.
    let joined = Int4Work::for_sequence(WIDTH, 0, b.model.forward_positions as usize,
        b.model.projected_logits as usize).unwrap();
    assert!(joined.attention_pairs > b.model.attention_pairs);
    assert_eq!(b.int4_model_work(), large.batch.item_model_work());
}
#[test]
fn fixed_empty_record_arguments_cannot_replace_policy_key_or_native_contract() {
    for args in [r#"{"verify":false}"#, r#"{"max_source_bytes":9000}"#, r#"{"namespace":"secret"}"#,
        r#"{"key":"private"}"#, r#"{"actions":{}}"#, r#"{"ner":{"types":["money"]}}"#,
        r#"{"group_width":32}"#, r#"{"max_model_work":{}}"#, r#"{"grounding_budget":{}}"#] {
        let input = format!(r#"{{"id":"opaque","text":"private","task_args":{args}}}"#);
        assert!(serde_json::from_str::<BatchDocument<RedactionBatchArgs>>(&input).is_err());
    }
    for args in ["null", "{}"] {
        let input = format!(r#"{{"id":"opaque","text":"private","task_args":{args}}}"#);
        assert!(serde_json::from_str::<BatchDocument<RedactionBatchArgs>>(&input).is_ok());
    }
}
#[test]
fn complete_recipe_projection_authenticates_each_semantic_and_resource_setting() {
    let c = config(); let baseline = freeze(&c, &recipe(&c));
    for n in 0..74 {
        let mut changed = config(); let b = &mut changed.batch;
        match n {
            0 => changed.max_source_bytes += 1,
            1 => b.detector.ner.max_entities += 1,
            2 => b.detector.ner.max_mention_scalars += 1,
            3 => b.detector.ner.types.push(EntityType::Location),
            4 => b.detector.per_chunk.max_input_tokens -= 1,
            5 => b.detector.per_chunk.max_output_tokens -= 1,
            6 => b.detector.per_chunk.max_output_bytes -= 1,
            7 => b.detector.per_chunk.max_grammar_states -= 1,
            8 => b.detector.per_chunk.max_kv_bytes -= 1,
            9 => b.detector.planning.max_input_bytes += 1,
            10 => b.detector.planning.max_context_tokens -= 1,
            11 => b.detector.planning.max_passages += 1,
            12 => b.detector.planning.compiler.max_schema_bytes += 1,
            13 => b.detector.planning.compiler.max_string_bytes += 1,
            14 => b.detector.planning.compiler.max_array_items += 1,
            15 => b.detector.planning.compiler.max_output_bytes += 1,
            16 => b.detector.planning.compiler.max_states += 1,
            17 => b.detector.planning.compiler.max_transitions += 1,
            18 => b.detector.planning.compiler.max_mask_bytes += 1,
            19 => b.detector.planning.source.index.max_source_bytes += 1,
            20 => b.detector.planning.source.index.max_index_bytes += 1,
            21 => b.detector.planning.source.index.max_build_steps += 1,
            22 => b.detector.planning.source.index.max_matches += 1,
            23 => b.detector.planning.source.verification.max_fields += 1,
            24 => b.detector.planning.source.verification.max_matches += 1,
            25 => b.detector.planning.source.verification.max_scan_steps += 1,
            26 => b.detector.mapping.chunks.max_input_bytes += 1,
            27 => b.detector.mapping.chunks.max_chunk_bytes += 1,
            28 => b.detector.mapping.chunks.max_chunk_tokens += 1,
            29 => b.detector.mapping.chunks.context_tokens -= 1,
            30 => b.detector.mapping.chunks.reserved_tokens += 1,
            31 => b.detector.mapping.chunks.max_chunks -= 1,
            32 => b.detector.mapping.chunks.max_tokenizer_calls += 1,
            33 => b.detector.mapping.reduction.map_batch_chunks += 1,
            34 => b.detector.mapping.reduction.reduce_fan_in += 1,
            35 => b.detector.mapping.reduction.max_reduction_levels += 1,
            36 => b.detector.mapping.reduction.max_task_calls += 1,
            37 => b.detector.mapping.reduction.max_value_bytes += 1,
            38 => b.detector.mapping.reduction.max_live_value_bytes += 1,
            39 => b.detector.mapping.reduction.max_total_value_bytes -= 1,
            40 => b.detector.mapping.reduction.max_result_bytes += 1,
            41..=45 => *native_axis(&mut b.detector.mapping.max_model_work, n - 41) += 1,
            46 => b.detector.mapping.mask_limits.max_trie_node_visits += 1,
            47 => b.detector.mapping.mask_limits.checkpoint_interval_nodes += 1,
            48 => b.detector.mapping.mask_visits_per_chunk += 1,
            49 => b.detector.mapping.max_mask_visits += 1,
            50 => b.detector.mapping.verification.max_fields += 1,
            51 => b.detector.mapping.verification.max_matches += 1,
            52 => b.detector.mapping.verification.max_scan_steps += 1,
            53 => b.detector.max_result_bytes += 1,
            54..=58 => *native_axis(&mut b.max_model_work, n - 54) += 1,
            59 => b.max_mask_visits += 1,
            60 => { b.request.rules.enabled.insert(PiiKind::Date); }
            61 => b.request.actions.default_action = RedactionAction::Mask,
            62 => { b.request.actions.per_type.insert(PiiKind::Person, RedactionAction::Mask); }
            63 => b.request.actions.include_map = true,
            64 => b.request.rule_budget.max_input_bytes += 1,
            65 => b.request.rule_budget.max_detections += 1,
            66 => b.request.rule_budget.max_candidate_bytes += 1,
            67 => b.request.rule_budget.max_work += 1,
            68 => b.request.grounding_budget.max_fields += 1,
            69 => b.request.grounding_budget.max_matches += 1,
            70 => b.request.grounding_budget.max_scan_steps += 1,
            71 => b.request.edit_budget.max_regions -= 1,
            72 => b.request.edit_budget.max_output_bytes -= 1,
            73 => b.request.verify = false,
            _ => unreachable!(),
        }
        let current = freeze(&changed, &recipe(&changed));
        assert_eq!(baseline.binding.compare(&current.binding),
            Err(JobError::Mismatch(MismatchField::Recipe)), "setting {n}");
    }
}
#[test]
fn recipe_versions_bind_the_complete_document_and_both_grounding_pools() {
    let c = config(); let encoded = serde_json::to_value(recipe(&c)).unwrap();
    assert_eq!(encoded["version"], 1);
    assert_eq!(encoded["dependency_scope"], "item-local");
    assert_eq!(encoded["input_policy"], "fixed-redaction-policy-original-utf8-full256-v1");
    assert_eq!(encoded["execution"], INT4_LONG_REDACTION_EXECUTION);
    assert_eq!(encoded["numerics_profile"], STRICT_INT4_PROFILE);
    assert_eq!(encoded["group_width"], WIDTH);
    assert_eq!(encoded["detector"]["planning"]["source"],
        serde_json::to_value(c.batch.detector.planning.source).unwrap());
    assert_eq!(encoded["detector"]["mapping"]["verification"],
        serde_json::to_value(c.batch.detector.mapping.verification).unwrap());
    assert_eq!(encoded["request"]["grounding_budget"],
        serde_json::to_value(c.batch.request.grounding_budget).unwrap());
}
#[test]
fn source_profile_group_and_packing_are_separate_execution_authority() {
    let c = config(); let r = recipe(&c); let baseline = freeze(&c, &r);
    for n in 0..4 {
        let mut changed = config();
        match n { 0 => changed.batch.ner_identity.logical_model_digest = Sha256Digest::of_bytes(b"other"),
            1 => changed.batch.ner_identity.packing_set_digest = Sha256Digest::of_bytes(b"other"),
            2 => changed.batch.ner_identity.numerics_profile = NumericsProfile::StrictQuantized { version: 1 },
            _ => changed.batch.ner_identity.quant_recipe = format!("{GROUPED_INT4_RECIPE}:group=32") }
        assert_eq!(baseline.binding.compare(&freeze(&changed, &r).binding),
            Err(JobError::Mismatch(MismatchField::Execution)));
    }
    let changed = Int4RedactionJobRecipe::new(&c, 32, None).unwrap();
    assert_eq!(baseline.binding.compare(&freeze(&c, &changed).binding),
        Err(JobError::Mismatch(MismatchField::Recipe)));
}
#[test]
fn raw_policy_size_is_bounded_before_private_clones_or_hmac_preparation() {
    let mut c = config();
    c.batch.request.actions.expected_key_commitment = Some("private".repeat(INT4_REDACTION_JOB_RECIPE_BYTES / 7 + 1));
    assert_eq!(check_static_limits(&c).unwrap_err().code, BatchCode::InvalidLimits);
    assert_eq!(fault(Int4RedactionJobRecipe::new(&c, WIDTH, None)).code, BatchCode::InvalidLimits);
    let mut c = config(); c.batch.detector.ner.types = vec![EntityType::Organization; 100000];
    assert_eq!(fault(Int4RedactionJobRecipe::new(&c, WIDTH, None)).code, BatchCode::InvalidLimits);
}
#[test]
fn pseudonyms_require_the_full256_context_and_matching_actual_key() {
    let mut c = config(); c.batch.request.actions.default_action = RedactionAction::Pseudonymize;
    assert_eq!(fault(Int4RedactionJobRecipe::new(&c, WIDTH, None)).code, BatchCode::Admission);
    let k = PseudonymKey::from_bytes(&[31; 32], "redaction-key").unwrap();
    let full = Pseudonyms::full256(&k, "private-job-namespace", None).unwrap();
    let short = Pseudonyms::preflight128(&k, "private-job-namespace", &[(PiiKind::Email, "a@example.com")],
        PseudonymBudget::default(), None).unwrap();
    assert_eq!(fault(Int4RedactionJobRecipe::new(&c, WIDTH, Some(&short))).code, BatchCode::Admission);
    c.batch.request.actions.expected_key_commitment = Some(k.commitment());
    let r = Int4RedactionJobRecipe::new(&c, WIDTH, Some(&full)).unwrap();
    let value = serde_json::to_value(r).unwrap();
    assert_eq!(value["pseudonyms"]["encoding"], serde_json::to_value(PseudonymEncoding::Full256).unwrap());
    c.batch.request.actions.expected_key_commitment = Some("wrong-key".to_owned());
    assert_eq!(fault(Int4RedactionJobRecipe::new(&c, WIDTH, Some(&full))).code, BatchCode::Admission);
}
#[test]
fn key_bytes_scope_and_expected_commitment_each_change_authenticated_recipe() {
    let mut c = config(); c.batch.request.actions.default_action = RedactionAction::Pseudonymize;
    let a = PseudonymKey::from_bytes(&[65; 32], "same-public-key-id").unwrap();
    let b = PseudonymKey::from_bytes(&[66; 32], "same-public-key-id").unwrap();
    let first = Pseudonyms::full256(&a, "private-scope-one", None).unwrap();
    let rotated = Pseudonyms::full256(&b, "private-scope-one", None).unwrap();
    let scope = Pseudonyms::full256(&a, "private-scope-two", None).unwrap();
    let baseline = freeze(&c, &Int4RedactionJobRecipe::new(&c, WIDTH, Some(&first)).unwrap());
    for context in [&rotated, &scope] {
        let current = freeze(&c, &Int4RedactionJobRecipe::new(&c, WIDTH, Some(context)).unwrap());
        assert_eq!(baseline.binding.compare(&current.binding), Err(JobError::Mismatch(MismatchField::Recipe)));
    }
    c.batch.request.actions.expected_key_commitment = Some(a.commitment());
    assert_eq!(baseline.binding.compare(&freeze(&c,
        &Int4RedactionJobRecipe::new(&c, WIDTH, Some(&first)).unwrap()).binding),
        Err(JobError::Mismatch(MismatchField::Recipe)));
}
#[test]
fn recipe_and_retained_manifest_do_not_contain_raw_keys_namespaces_or_source() {
    let c = config(); let k = PseudonymKey::from_bytes(&[65; 32], "public-id").unwrap();
    let context = Pseudonyms::full256(&k, "private-namespace-unique", None).unwrap();
    let r = Int4RedactionJobRecipe::new(&c, WIDTH, Some(&context)).unwrap();
    let encoded = canonjson::canonical_string(&r).unwrap(); let frozen = freeze(&c, &r);
    let retained = canonjson::canonical_string(&(&frozen.binding, &frozen.items)).unwrap();
    for private in ["private-namespace-unique", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", "a@example.com", "Alice private"] {
        assert!(!encoded.contains(private)); assert!(!retained.contains(private));
    }
    assert!(!retained.contains("max_source_bytes")); assert!(!retained.contains("alpha"));
    assert!(encoded.contains(&context.identity().scope_commitment));
}
#[test]
fn every_typed_cancellation_remains_fatal_and_private() {
    for cause in [DecodeCancellationKind::User, DecodeCancellationKind::Timeout, DecodeCancellationKind::Deadline,
        DecodeCancellationKind::PollQuota, DecodeCancellationKind::CostBudget, DecodeCancellationKind::FailFast,
        DecodeCancellationKind::RaceLost, DecodeCancellationKind::ParentCancelled, DecodeCancellationKind::ResourceUnavailable,
        DecodeCancellationKind::Shutdown, DecodeCancellationKind::LinkedExit] {
        let refused = checkpoint(&mut Stop(cause)).unwrap_err();
        assert!(refused.stop); assert_eq!(refused.fault, BatchFault::cancelled(cause));
        assert!(!refused.fault.to_string().contains("private"));
    }
}
struct Admission;
impl Int4RedactionBatchAdmission for Admission {
    type Guard = ();
    fn admit(&mut self, _: Int4RedactionAdmission<'_>) -> Result<(ExecutionIdentity, ()), BatchItemFailure> {
        Err(BatchItemFailure::fatal(BatchCode::Admission))
    }
}
#[test]
fn concrete_native_processor_implements_durable_boundary_without_model_fixture_authority() {
    fn durable<P: DurableBatchProcessor>() {}
    durable::<Int4RedactionJobProcessor<'static, 'static, 'static, 'static, 'static, 'static, Admission>>();
}

// Private storage seam: real group-aware source planning, sealed production
// recipe/checks, shared rule edits and real authenticated owned-job storage.
// No native result is fabricated or accepted by a production entrypoint.
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Root(PathBuf);
impl Root {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("fnlp-int4-redaction-job-{}-{}",
            std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        fs::DirBuilder::new().mode(0o700).create(&path).unwrap(); Self(path)
    }
}
impl Drop for Root { fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); } }
#[derive(Default)]
struct Events { plans: Cell<usize>, executions: Cell<usize>, held: Cell<bool>,
    serializations: Cell<usize>, releases: Cell<usize>, held_polls: Cell<usize> }
struct Hold(Rc<Events>);
impl Drop for Hold {
    fn drop(&mut self) { assert!(self.0.held.replace(false)); self.0.releases.set(self.0.releases.get() + 1); }
}
struct Datum { ordinal: u64, result: RedactionResult, events: Rc<Events> }
impl Serialize for Datum {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        assert!(self.events.held.get()); self.events.serializations.set(self.events.serializations.get() + 1);
        ("private-rules-storage-fixture-no-native-inference", self.ordinal, &self.result).serialize(serializer)
    }
}
struct Prepared { text: String, original: Int4SourceMapPreflight }
struct Processor<'s, 'k> {
    planner: SourceTaskPlanner, config: Int4RedactionJobConfig, recipe: Int4RedactionJobRecipe,
    pseudonyms: Option<&'s Pseudonyms<'k>>, events: Rc<Events>, fail: bool,
}
impl<'s, 'k> Processor<'s, 'k> {
    fn new(config: Int4RedactionJobConfig, pseudonyms: Option<&'s Pseudonyms<'k>>, events: Rc<Events>) -> Self {
        check_static_limits(&config).unwrap();
        let recipe = Int4RedactionJobRecipe::new(&config, WIDTH, pseudonyms).unwrap();
        Self { planner: planner(), config, recipe, pseudonyms, events, fail: false }
    }
}
impl BatchProcessor for Processor<'_, '_> {
    type Args = RedactionBatchArgs;
    type Prepared = Prepared;
    type Output = GuardedOutput<Datum, Hold>;
    fn prepare(&mut self, _: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        panic!("durable planning must retain caller control")
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, input: BatchDocument<Self::Args>, control: &mut C)
        -> Result<Self::Prepared, BatchItemFailure> {
        checkpoint(control)?; check_source(input.text.len(), self.recipe.max_source_bytes(), false)?;
        self.events.plans.set(self.events.plans.get() + 1);
        let d = &self.config.batch.detector;
        let context = PlanContext::new(&self.config.batch.ner_identity, d.per_chunk)
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        let plan = self.planner.plan_int4_map_with_control(&input.text, &SourceMapTask::Ner(d.ner.clone()),
            d.per_chunk, &context, d.planning, d.mapping, WIDTH, control).map_err(|e| {
                match e.cancellation() {
                    Some(cause) => BatchItemFailure::fatal(BatchFault::cancelled(cause)),
                    None => BatchItemFailure::reject(BatchCode::Planning),
                }
            })?;
        let original = plan.preflight_metadata();
        assert_eq!(plan.max_kv_bytes(), d.per_chunk.max_kv_bytes);
        assert_eq!(original.group_width(), WIDTH);
        drop(plan);
        Ok(Prepared { text: input.text, original })
    }
    fn planned_work(&self, _: &Self::Prepared) -> BatchWork { batch_work(item(&self.config)) }
    fn execute<C: DecodeStepControl>(&mut self, _: Self::Prepared, _: &mut C) -> Result<Self::Output, BatchItemFailure> {
        panic!("durable execution must retain delivery coordinates")
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        checkpoint(control)?; check_source(prepared.text.len(), self.recipe.max_source_bytes(), false)?;
        assert_eq!(context.epoch, 1); assert_eq!(context.input_line, context.request_seq);
        assert_eq!(prepared.original.source_span().byte_end, prepared.text.len());
        assert_eq!(prepared.original.source_span().scalar_end, prepared.text.chars().count());
        self.events.executions.set(self.events.executions.get() + 1);
        assert!(!self.events.held.replace(true)); let guard = Hold(Rc::clone(&self.events));
        if self.fail { return Err(BatchItemFailure::reject(BatchCode::Execution)); }
        let result = redact_rules(&prepared.text, &self.config.batch.request, self.pseudonyms)
            .map_err(|_| BatchItemFailure::reject(BatchCode::Execution))?;
        assert_eq!(result.verification(), if self.config.batch.request.verify {
            VerificationStatus::CleanDeclaredUnion } else { VerificationStatus::NotRequested });
        Ok(GuardedOutput::new(Datum { ordinal: context.request_seq, result, events: Rc::clone(&self.events) }, guard))
    }
}
impl DurableBatchProcessor for Processor<'_, '_> {
    type Recipe = Int4RedactionJobRecipe;
    fn execution_identity(&self) -> &ExecutionIdentity { &self.config.batch.ner_identity }
    fn job_recipe(&self) -> &Self::Recipe { &self.recipe }
    fn validate_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> { self.recipe.check_job_limits(limits) }
    fn durable_work(&self, prepared: &Self::Prepared) -> Result<JobWork, BatchItemFailure> {
        check_source(prepared.text.len(), self.recipe.max_source_bytes(), false)?;
        self.recipe.check_work(self.planned_work(prepared), false)
    }
    fn max_result_bytes(&self, _: &Self::Prepared) -> u64 { self.recipe.max_result_bytes() }
}
struct CancelAfterSync(Rc<Events>);
impl DecodeStepControl for CancelAfterSync {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        if !self.0.held.get() { return None; }
        let next = self.0.held_polls.get() + 1; self.0.held_polls.set(next);
        (next == 2).then_some(DecodeCancellationKind::PollQuota)
    }
}
#[test]
fn decoded_source_refusal_precedes_real_planning_and_attempt_debit() {
    let root = Root::new(); let input = inputs(); let events = Rc::new(Events::default());
    let mut c = config(); c.max_source_bytes = 4; let lim = limits(&c);
    assert!(input[0].original.len() < lim.max_input_bytes_per_item);
    let mut run = JobRunner::create(&root.0, key(), JobId([9; 16]), lim, &input,
        Processor::new(c, None, Rc::clone(&events)), &mut Continue).unwrap();
    assert_eq!(error(run.step(&mut Continue)), JobRunError::Processor(BatchCode::DocumentLimit.into()));
    assert_eq!(run.progress().attempts, 0); assert_eq!(run.progress().reserved_work, JobWork::default());
    assert_eq!(events.plans.get(), 0); assert_eq!(events.executions.get(), 0);
}
#[test]
fn invalid_lifetime_ceiling_refuses_before_creating_owned_storage() {
    let root = Root::new(); let c = config(); let mut lim = limits(&c); lim.max_work.mask_node_visits += 1;
    let events = Rc::new(Events::default());
    assert_eq!(error(JobRunner::create(&root.0, key(), JobId([9; 16]), lim, &inputs(),
        Processor::new(c, None, Rc::clone(&events)), &mut Continue)),
        JobRunError::Processor(BatchCode::InvalidLimits.into()));
    assert_eq!(fs::read_dir(&root.0).unwrap().count(), 0);
    assert_eq!(events.plans.get(), 0); assert_eq!(events.executions.get(), 0);
}
#[test]
fn durable_envelope_exceeds_original_plan_and_cannot_be_repriced_as_early_completion() {
    let c = config(); let quote = item(&c); let mut processor = Processor::new(c, None, Rc::default());
    let parsed = serde_json::from_slice(inputs()[0].original).unwrap();
    let prepared = processor.prepare_with_control(parsed, &mut Continue).unwrap();
    assert!(prepared.original.planned_work().forward_positions < quote.model.forward_positions);
    assert!(prepared.original.reserved_mask_visits() < quote.mask_node_visits);
    assert_eq!(processor.durable_work(&prepared).unwrap(), quote);
    assert_eq!(processor.planned_work(&prepared), batch_work(quote));
}
#[test]
fn resumed_redaction_skips_commits_and_materializes_identical_ordered_content() {
    let root = Root::new(); let full = Root::new(); let input = inputs();
    let events = Rc::new(Events::default()); let c = config(); let lim = limits(&c); let quote = item(&c);
    let make = || Processor::new(config(), None, Rc::clone(&events));
    let mut run = JobRunner::create(&root.0, key(), JobId([9; 16]), lim, &input, make(), &mut Continue).unwrap();
    assert_eq!(run.step(&mut Continue).unwrap().unwrap().committed, 1); drop(run);
    let mut resumed = JobRunner::resume(&root.0, key(), JobId([9; 16]), lim, &input,
        make(), TailPolicy::Refuse, &mut Continue).unwrap();
    let progress = resumed.run(&mut Continue).unwrap();
    assert_eq!(progress.committed, 2); assert_eq!(progress.attempts, 2);
    assert_eq!(progress.reserved_work, quote.checked_add(quote).unwrap());
    assert_eq!(events.plans.get(), 2); assert_eq!(events.executions.get(), 2); assert_eq!(events.releases.get(), 2);
    assert!(events.serializations.get() >= 4); assert!(!events.held.get());
    resumed.verify(&mut Continue).unwrap(); resumed.materialize_ordered(&mut Continue).unwrap();
    let mut baseline = JobRunner::create(&full.0, key(), JobId([9; 16]), lim, &input,
        Processor::new(config(), None, Rc::default()), &mut Continue).unwrap();
    baseline.run(&mut Continue).unwrap(); baseline.materialize_ordered(&mut Continue).unwrap();
    let result = fs::read(root.0.join("materialized.ndjson")).unwrap();
    assert_eq!(result, fs::read(full.0.join("materialized.ndjson")).unwrap());
    let text = String::from_utf8(result).unwrap();
    assert!(!text.contains("a@example.com")); assert!(!text.contains("b@example.com"));
    assert_eq!(text.lines().count(), 2);
}
#[test]
fn every_failed_attempt_axis_stays_spent_after_a_fresh_processor_resume() {
    for n in 0..6 {
        let mut c = config(); let mut quote = item(&c); let mut cap = limits(&c).max_work;
        *axis(&mut cap, n) = *axis(&mut quote, n);
        c.batch.max_model_work = cap.int4_model_work(); c.batch.max_mask_visits = cap.mask_node_visits;
        let lim = limits(&c); let input = inputs(); let root = Root::new(); let events = Rc::new(Events::default());
        let mut processor = Processor::new(c, None, Rc::clone(&events)); processor.fail = true;
        let mut run = JobRunner::create(&root.0, key(), JobId([9; 16]), lim, &input, processor, &mut Continue).unwrap();
        assert_eq!(error(run.step(&mut Continue)), JobRunError::Processor(BatchCode::Execution.into()));
        assert_eq!(run.progress().reserved_work, quote); assert_eq!(run.progress().committed, 0);
        assert_eq!(run.progress().attempts, 1); assert!(!events.held.get()); drop(run);
        let mut c = config(); c.batch.max_model_work = cap.int4_model_work(); c.batch.max_mask_visits = cap.mask_node_visits;
        let mut resumed = JobRunner::resume(&root.0, key(), JobId([9; 16]), lim, &input,
            Processor::new(c, None, Rc::clone(&events)), TailPolicy::Refuse, &mut Continue).unwrap();
        assert_eq!(error(resumed.step(&mut Continue)), JobRunError::Storage(JobError::WorkLimit));
        assert_eq!(events.executions.get(), 1); assert_eq!(resumed.progress().reserved_work, quote);
        assert_eq!(resumed.progress().attempts, 1); assert_eq!(resumed.progress().committed, 0);
    }
}
#[test]
fn recoverable_document_failure_is_never_cached_as_success_or_retried_implicitly() {
    let root = Root::new(); let input = inputs(); let c = config(); let lim = limits(&c); let quote = item(&c);
    let events = Rc::new(Events::default()); let mut p = Processor::new(c, None, Rc::clone(&events)); p.fail = true;
    let mut run = JobRunner::create(&root.0, key(), JobId([9; 16]), lim, &input, p, &mut Continue).unwrap();
    assert_eq!(error(run.step(&mut Continue)), JobRunError::Processor(BatchCode::Execution.into()));
    assert!(run.step(&mut Continue).is_err()); assert_eq!(events.executions.get(), 1);
    assert_eq!(run.progress().committed, 0); assert_eq!(run.progress().attempts, 1);
    assert_eq!(error(run.materialize_ordered(&mut Continue)), JobRunError::Stopped);
    assert!(!root.0.join("materialized.ndjson").exists()); drop(run);
    let mut resumed = JobRunner::resume(&root.0, key(), JobId([9; 16]), lim, &input,
        Processor::new(config(), None, Rc::clone(&events)), TailPolicy::Refuse, &mut Continue).unwrap();
    assert_eq!(resumed.step(&mut Continue).unwrap().unwrap().reserved_work, quote.checked_add(quote).unwrap());
    assert_eq!(resumed.progress().committed, 1); assert_eq!(resumed.progress().attempts, 2);
}
#[test]
fn same_full256_scope_remains_stable_across_resume_and_guarded_materialization() {
    let root = Root::new(); let input = inputs(); let events = Rc::new(Events::default());
    let k = PseudonymKey::from_bytes(&[37; 32], "durable-redaction-key").unwrap();
    let context = Pseudonyms::full256(&k, "private-resume-namespace", None).unwrap();
    let make_config = || { let mut c = config(); c.batch.request.actions.default_action = RedactionAction::Pseudonymize;
        c.batch.request.actions.expected_key_commitment = Some(k.commitment()); c };
    let lim = limits(&make_config());
    let make = || Processor::new(make_config(), Some(&context), Rc::clone(&events));
    let mut run = JobRunner::create(&root.0, key(), JobId([9; 16]), lim, &input, make(), &mut Continue).unwrap();
    run.step(&mut Continue).unwrap(); drop(run);
    let mut resumed = JobRunner::resume(&root.0, key(), JobId([9; 16]), lim, &input,
        make(), TailPolicy::Refuse, &mut Continue).unwrap();
    resumed.run(&mut Continue).unwrap(); resumed.materialize_ordered(&mut Continue).unwrap();
    let text = fs::read_to_string(root.0.join("materialized.ndjson")).unwrap();
    for original in ["a@example.com", "b@example.com"] {
        assert!(!text.contains(original));
        assert!(text.contains(&context.pseudonym(PiiKind::Email, original).unwrap()));
    }
    assert!(!text.contains("private-resume-namespace")); assert_eq!(events.releases.get(), 2);
}
#[test]
fn cancellation_after_spool_sync_authenticates_policy_scope_before_tail_repair() {
    let root = Root::new(); let input = inputs(); let c = config(); let lim = limits(&c); let quote = item(&c);
    let events = Rc::new(Events::default());
    let k = PseudonymKey::from_bytes(&[29; 32], "tail-key").unwrap();
    let context = Pseudonyms::full256(&k, "private-original-scope", None).unwrap();
    let changed = Pseudonyms::full256(&k, "private-replacement-scope", None).unwrap();
    let make = || Processor::new(config(), Some(&context), Rc::clone(&events));
    let mut control = CancelAfterSync(Rc::clone(&events));
    let mut run = JobRunner::create(&root.0, key(), JobId([9; 16]), lim, &input, make(), &mut control).unwrap();
    assert_eq!(error(run.step(&mut control)), JobRunError::Storage(JobError::Cancelled(DecodeCancellationKind::PollQuota)));
    assert_eq!(run.progress().committed, 0); assert_eq!(run.progress().attempts, 1);
    assert_eq!(run.progress().reserved_work, quote); assert_eq!(events.releases.get(), 1);
    assert!(!events.held.get()); drop(run);
    assert_eq!(error(JobRunner::resume(&root.0, key(), JobId([9; 16]), lim, &input,
        make(), TailPolicy::Refuse, &mut Continue)), JobRunError::Storage(JobError::UncommittedTail));
    assert_eq!(error(JobRunner::resume(&root.0, key(), JobId([9; 16]), lim, &input,
        Processor::new(config(), Some(&changed), Rc::clone(&events)), TailPolicy::DiscardUncommitted, &mut Continue)),
        JobRunError::Storage(JobError::Mismatch(MismatchField::Recipe)));
    assert_eq!(error(JobRunner::resume(&root.0, key(), JobId([9; 16]), lim, &input,
        make(), TailPolicy::Refuse, &mut Continue)), JobRunError::Storage(JobError::UncommittedTail));
    let mut resumed = JobRunner::resume(&root.0, key(), JobId([9; 16]), lim, &input,
        make(), TailPolicy::DiscardUncommitted, &mut Continue).unwrap();
    assert_eq!(resumed.progress().reserved_work, quote);
    assert_eq!(resumed.step(&mut Continue).unwrap().unwrap().reserved_work, quote.checked_add(quote).unwrap());
    assert_eq!(resumed.run(&mut Continue).unwrap().committed, 2);
}
#[test]
fn cancellation_before_planning_creates_no_attempt_or_admission_guard() {
    let root = Root::new(); let input = inputs(); let c = config(); let lim = limits(&c);
    let events = Rc::new(Events::default());
    let mut run = JobRunner::create(&root.0, key(), JobId([9; 16]), lim, &input,
        Processor::new(c, None, Rc::clone(&events)), &mut Continue).unwrap();
    assert!(run.step(&mut Stop(DecodeCancellationKind::User)).is_err());
    assert_eq!(run.progress().attempts, 0); assert_eq!(run.progress().reserved_work, JobWork::default());
    assert_eq!(events.plans.get(), 0); assert_eq!(events.executions.get(), 0); assert_eq!(events.releases.get(), 0);
}
