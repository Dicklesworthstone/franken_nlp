//! Source regressions for native cohort routing/admission. Only the explicitly
//! ignored final test loads a real model; fixture identities are test data, not
//! production attestations. Rust tests require the repository's pinned toolchain.
use super::*;
use crate::{
    execution_identity::{Sha256Digest, ThinkingMode, ToolMode},
    native_engine::generation::{GenerationOptions, batched::BatchGenerationBudget},
    tasks::{chat::{ChatPlanner, ChatLimits}, ir::TaskBudget},
    tokenizer::{bpe::EncodeOptions, embedded::EmbeddedTokenizer, specials::ArchivedControlRegistries},
    template::{IM_START, IM_END, THINK_START, THINK_END},
};

fn fits_one_wave(capacities: &[usize], shapes: &[RowShape], row: usize, used: &mut [bool]) -> bool {
    if row == shapes.len() { return true; }
    for slot in 0..capacities.len() {
        if !used[slot] && shapes[row].fits(capacities[slot]) {
            used[slot] = true;
            if fits_one_wave(capacities, shapes, row + 1, used) { used[slot] = false; return true; }
            used[slot] = false;
        }
    }
    false
}
#[test]
fn exhaustive_interval_assignment_preserves_single_wave_feasibility_and_makes_bounded_progress() {
    let intervals: Vec<_> = (1..=4).flat_map(|minimum| (minimum..=4).map(move |maximum| RowShape { minimum, maximum })).collect();
    let mut cases = 0;
    for slot_count in 1..=3_usize {
        for code in 0..4_usize.pow(slot_count as u32) {
            let mut code = code;
            let capacities: Vec<_> = (0..slot_count).map(|_| { let value = code % 4 + 1; code /= 4; value }).collect();
            let slots: Vec<_> = (0..slot_count).collect();
            for row_count in 1..=3_usize {
                for code in 0..intervals.len().pow(row_count as u32) {
                    let mut code = code;
                    let shapes: Vec<_> = (0..row_count).map(|_| { let value = intervals[code % intervals.len()]; code /= intervals.len(); value }).collect();
                    let individually_feasible = shapes.iter().all(|shape| capacities.iter().any(|&cap| shape.fits(cap)));
                    let result = plan_waves(&capacities, &slots, &shapes);
                    assert_eq!(result.is_ok(), individually_feasible);
                    if let Ok(waves) = result {
                        assert!(!waves.is_empty() && waves.len() <= row_count);
                        let mut seen = vec![false; row_count];
                        for wave in &waves {
                            let mut occupied = vec![false; slot_count];
                            for assignment in wave {
                                assert!(!seen[assignment.row]); seen[assignment.row] = true;
                                assert!(!occupied[assignment.slot]); occupied[assignment.slot] = true;
                                assert!(shapes[assignment.row].fits(capacities[assignment.slot]));
                            }
                            assert!(wave.windows(2).all(|pair| pair[0].row < pair[1].row));
                        }
                        assert!(seen.into_iter().all(|v| v));
                        assert_eq!(waves.len() == 1, fits_one_wave(&capacities, &shapes, 0, &mut vec![false; slot_count]));
                    }
                    cases += 1;
                }
            }
        }
    }
    assert_eq!(cases, 93_240);
}
#[test]
fn repeated_long_documents_reuse_only_the_compatible_slot_in_later_waves() {
    let shapes = [RowShape { minimum: 3, maximum: 4 }, RowShape { minimum: 3, maximum: 4 }, RowShape { minimum: 1, maximum: 2 }];
    let waves = plan_waves(&[2, 4], &[1, 0], &shapes).unwrap();
    assert_eq!(waves, vec![vec![Assignment { row: 0, slot: 1 }, Assignment { row: 2, slot: 0 }], vec![Assignment { row: 1, slot: 1 }]]);
}
#[test]
fn per_task_upper_kv_limit_is_respected_even_when_a_larger_slot_is_free() {
    let shapes = [RowShape { minimum: 2, maximum: 100 }, RowShape { minimum: 1, maximum: 2 }];
    let waves = plan_waves(&[2, 100], &[0, 1], &shapes).unwrap();
    assert_eq!(waves, vec![vec![Assignment { row: 0, slot: 1 }, Assignment { row: 1, slot: 0 }]]);
    let restricted = plan_waves(&[1, 999, 4], &[2, 0], &[RowShape { minimum: 1, maximum: 4 }; 2]).unwrap();
    assert!(restricted.iter().flatten().all(|a| a.slot != 1));
}
#[test]
fn bad_slots_and_individually_impossible_shapes_fail_without_execution() {
    for slots in [vec![], vec![0, 0], vec![2]] { assert!(plan_waves(&[1, 2], &slots, &[RowShape { minimum: 1, maximum: 2 }]).is_err()); }
    for shape in [RowShape { minimum: 0, maximum: 1 }, RowShape { minimum: 3, maximum: 9 }, RowShape { minimum: 2, maximum: 1 }] {
        assert!(plan_waves(&[1, 2], &[0, 1], &[shape]).is_err());
    }
}
#[test]
fn all_waves_are_charged_together_but_the_shared_arena_is_not_double_charged() {
    let price = BatchGenerationRequirements { planned_work: GenerationWork { forward_positions: 1, projected_logits: 10, sampled_steps: 1 },
        native_payload_bytes: 100, sampler_payload_bytes: 50, output_payload_upper_bytes: 60 };
    let mut total = BatchGenerationRequirements { planned_work: GenerationWork::default(), native_payload_bytes: 100,
        sampler_payload_bytes: 0, output_payload_upper_bytes: 0 };
    add_price(&mut total, price).unwrap(); add_price(&mut total, price).unwrap();
    assert_eq!(total.native_payload_bytes, 100); assert_eq!(total.sampler_payload_bytes, 100); assert_eq!(total.output_payload_upper_bytes, 120);
    assert_eq!(total.planned_work.forward_positions, 2); assert_eq!(total.planned_work.sampled_steps, 2);
    let budget = BatchChatBudget { generation: BatchGenerationBudget { max_forward_positions: 2, max_projected_logits: 20,
        max_native_payload_bytes: 100, max_sampler_payload_bytes: 100, max_output_payload_bytes: 120 }, max_result_bytes: 1024 };
    check_budget(total, budget).unwrap();
    assert_eq!(check_budget(total, BatchChatBudget { generation: BatchGenerationBudget { max_forward_positions: 1, ..budget.generation }, ..budget }).unwrap_err().code, BatchCode::WorkLimit);
    assert_eq!(check_budget(total, BatchChatBudget { generation: BatchGenerationBudget { max_sampler_payload_bytes: 99, ..budget.generation }, ..budget }).unwrap_err().code, BatchCode::Admission);
    assert!(add_price(&mut total, BatchGenerationRequirements { native_payload_bytes: 99, ..price }).is_err());
}
#[test]
fn aggregate_arithmetic_does_not_wrap_and_allow_another_wave() {
    let mut total = BatchGenerationRequirements { planned_work: GenerationWork { forward_positions: u64::MAX, ..GenerationWork::default() },
        native_payload_bytes: 1, sampler_payload_bytes: 0, output_payload_upper_bytes: 0 };
    let wave = BatchGenerationRequirements { planned_work: GenerationWork { forward_positions: 1, ..GenerationWork::default() },
        native_payload_bytes: 1, sampler_payload_bytes: 0, output_payload_upper_bytes: 0 };
    assert_eq!(add_price(&mut total, wave).unwrap_err().code, BatchCode::WorkLimit);
}
fn context(sequence: u64, epoch: u64, line: u64, offset: u64) -> GroupedRequest<()> {
    GroupedRequest { prepared: (), context: BatchRequestContext { request_seq: sequence, epoch, input_line: line, byte_offset: offset } }
}
#[test]
fn delivery_coordinates_cannot_cross_flush_epochs_or_alias_prior_requests() {
    validate_contexts(&[context(1, 1, 1, 0), context(3, 1, 4, 100)]).unwrap();
    for pair in [vec![context(1, 1, 1, 0), context(2, 2, 2, 100)],
        vec![context(1, 1, 1, 0), context(1, 1, 2, 100)], vec![context(2, 1, 1, 0), context(1, 1, 2, 100)],
        vec![context(1, 1, 1, 0), context(2, 1, 1, 100)], vec![context(1, 1, 1, 0), context(2, 1, 2, 0)],
        vec![context(0, 1, 1, 0)]] { assert!(validate_contexts(&pair).is_err()); }
}
#[test]
fn only_ordinary_task_no_results_become_nonfatal_document_errors() {
    for (reason, code) in [(BatchChatNoResult::IncompleteUtf8, BatchCode::Execution), (BatchChatNoResult::ResultByteLimit, BatchCode::OutputLineLimit)] {
        let item = BatchChatItem::NoResult { request_seq: 7, sample_index: 0, reason };
        let error = translate(item, 7).unwrap().err().unwrap(); assert!(!error.stop); assert_eq!(error.fault.code, code);
    }
    let item = BatchChatItem::NoResult { request_seq: 8, sample_index: 0, reason: BatchChatNoResult::IncompleteUtf8 };
    assert_eq!(translate(item, 7).err().unwrap().code, BatchCode::InvalidExecution);
    assert_eq!(failure(ChatError::Native(GenerationError::Cancelled(DecodeCancellationKind::Deadline))).cancellation, Some(DecodeCancellationKind::Deadline));
    assert_eq!(failure(ChatError::Native(GenerationError::NoLegalToken)).code, BatchCode::Execution);
}

fn fixture() -> (ChatPlanner, GenerationBatchArgs) {
    let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    let entries: Vec<_> = [IM_START, IM_END, THINK_START, THINK_END].iter().map(|&surface| {
        let ids = tokenizer.tokenizer().encode_ids_with_options(surface, EncodeOptions { add_bos: false, add_eos: false }).unwrap();
        assert_eq!(ids.len(), 1);
        serde_json::json!({"id":ids[0],"special":surface == IM_START || surface == IM_END,"surface":surface})
    }).collect();
    let eos = entries[1]["id"].as_u64().unwrap() as u32;
    let specials: Vec<_> = entries.iter().filter(|e| e["special"] == true).cloned().collect();
    let controls = ArchivedControlRegistries::from_archived_json(
        &serde_json::json!({"schema_version":1,"registry":"TokenizerSpecialIds","entries":specials}).to_string(),
        &serde_json::json!({"schema_version":1,"registry":"TemplateControlIds","entries":entries}).to_string()).unwrap();
    let d = Sha256Digest::of_bytes(b"synthetic test identity; never production admission");
    let identity = ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "bf16-verbatim".to_owned(), packing_set_digest: d,
        tokenizer_digest: d, template_digest: d, task_spec: "generate-v1".to_owned(), taskir_digest: d, prompt_digest: d,
        grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::HfBf16Eager, kv_dtype: "bf16".to_owned(), sampler_version: "fixture".to_owned(),
        thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None, calibration_digest: d, decision_policy_digest: d,
        backend_semantic_version: "fixture".to_owned(), host_class: None, compiler_identity: None };
    let budget = TaskBudget { max_input_tokens: 8192, max_output_tokens: 64, max_output_bytes: 100000,
        max_grammar_states: 4096, max_kv_bytes: 1 << 30 };
    let planner = ChatPlanner::pinned(controls.template_controls(), eos, identity, budget, ChatLimits::default()).unwrap();
    (planner, GenerationBatchArgs::Generate { generation: GenerationOptions::greedy(4, 1024, eos), budget, sample_index: 0 })
}
fn prepared(compiler: &GenerationBatchPlanner<'_>) -> Vec<GroupedRequest<PreparedChat>> {
    [("a", "Hello"), ("b", "Name a color.")].into_iter().enumerate().map(|(i, (id, text))| GroupedRequest {
        prepared: compiler.prepare(BatchDocument { id: id.to_owned(), text: text.to_owned(), task_args: None }).unwrap(),
        context: BatchRequestContext { request_seq: i as u64 + 1, epoch: 1, input_line: i as u64 + 1, byte_offset: i as u64 * 100 },
    }).collect()
}
#[test]
fn every_admitted_identity_is_verified_before_execution_and_slots_do_not_change_semantic_keys() {
    let (planner, defaults) = fixture(); let compiler = GenerationBatchPlanner::new(&planner, Some(defaults)).unwrap();
    let requests = prepared(&compiler);
    let identities: Vec<_> = requests.iter().map(|r| r.prepared.execution_identity().clone()).collect();
    verify_identities(&requests, &identities).unwrap(); assert!(verify_identities(&requests, &identities[..1]).is_err());
    let mut changed = identities.clone(); changed[1].backend_semantic_version = "foreign".to_owned();
    assert!(verify_identities(&requests, &changed).is_err()); changed = identities.clone(); changed.swap(0, 1);
    assert!(verify_identities(&requests, &changed).is_err());
    let wave = [Assignment { row: 0, slot: 9 }, Assignment { row: 1, slot: 3 }];
    let rows = native_requests(&requests, &wave, Some(&identities)).unwrap();
    assert_eq!(rows[0].slot, 9); assert_eq!(rows[1].request_seq, 2);
    rows[0].prepared.native_plan().verify_identity(rows[0].admitted_identity).unwrap();
    assert_eq!(work(&requests[0].prepared).unwrap().projected_logits, 4 * NANBEIGE_VOCAB_SIZE as u64);
    assert!(requests[0].prepared.native_plan().planned_work().projected_logits > work(&requests[0].prepared).unwrap().projected_logits);
}

/// Explicitly expensive authenticated-source test. This fixture-only host clones
/// identities for mechanism testing; it MUST NOT be used by production code.
/// Run only with the complete pinned source closure and sufficient host memory.
#[test]
#[ignore = "requires FNLP_BATCH_SOURCE_DIR and full-model scalar BF16 execution"]
fn real_model_grouped_ndjson_matches_width_one_without_cloning_weights() {
    use crate::native_engine::{batchsched::BatchEnvelope, hf_bf16_eager::HfBf16EagerWeights};
    use std::{cell::Cell, io::Cursor, rc::Rc};
    let Some(source) = std::env::var_os("FNLP_BATCH_SOURCE_DIR") else {
        eprintln!("SKIPPED_NO_MODEL FNLP_BATCH_SOURCE_DIR"); return;
    };
    struct Host(Rc<Cell<bool>>);
    struct Guard(Rc<Cell<bool>>);
    impl Drop for Guard { fn drop(&mut self) { assert!(self.0.replace(false)); } }
    impl GenerationCohortHost for Host {
        type Guard = Guard;
        fn admit(&mut self, r: GenerationCohortAdmission<'_>) -> Result<(Vec<ExecutionIdentity>, Guard), BatchFault> {
            assert!(!self.0.replace(true));
            Ok((r.items.iter().map(|item| item.identity.clone()).collect(), Guard(Rc::clone(&self.0))))
        }
    }
    struct Control;
    impl DecodeStepControl for Control { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
    let weights = HfBf16EagerWeights::from_pinned_source(source).unwrap();
    let (planner, defaults) = fixture();
    let plans = prepared(&GenerationBatchPlanner::new(&planner, Some(defaults.clone())).unwrap());
    let loaded = plans[0].prepared.execution_identity().clone();
    let capacity = plans.iter().map(|p| p.prepared.planned_work().forward_positions as usize).max().unwrap();
    let mut wires = Vec::new();
    for width in [1, 2] {
        let envelope = BatchEnvelope::compile(&[capacity, capacity], u64::MAX).unwrap();
        let payload = envelope.payload().total_bytes;
        let mut engine = EagerBatchEngine::new(&weights, envelope).unwrap();
        let compiler = GenerationBatchPlanner::new(&planner, Some(defaults.clone())).unwrap();
        let active = Rc::new(Cell::new(false));
        let budget = BatchChatBudget { generation: BatchGenerationBudget { max_forward_positions: 1000, max_projected_logits: 100 * NANBEIGE_VOCAB_SIZE as u64,
            max_native_payload_bytes: payload, max_sampler_payload_bytes: 64 * 1024 * 1024, max_output_payload_bytes: 1024 * 1024 }, max_result_bytes: 1024 * 1024 };
        let mut processor = NativeGenerationGroups::new(compiler, &mut engine, loaded.clone(), vec![0, 1], Host(Rc::clone(&active)), budget).unwrap();
        let mut input = Cursor::new(b"{\"id\":\"a\",\"text\":\"Hello\"}\n{\"id\":\"b\",\"text\":\"Name a color.\"}\n");
        let mut output = Vec::new();
        let summary = super::super::run_ndjson(&mut input, &mut output, &mut processor, BatchLimits::default(),
            GroupLimits { max_records: width }, &mut Control).unwrap();
        assert_eq!(summary.succeeded, 2, "the real-model fixture must return two validated typed results");
        assert!(!active.get()); wires.push(output);
    }
    assert_eq!(wires[0], wires[1], "full canonical NDJSON, token IDs, bytes, work and ordering must agree");
}
