//! Contract projection tests: no fixture is native admission/inference proof.
use super::*;
use crate::{canonjson, jobs::tests::work, batch::source::quantized::Int8SourceAdmission,
    tasks::keyphrases::KeyphraseOptions};

fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 4096, max_output_tokens: 64, max_output_bytes: 32768,
        max_grammar_states: 4096, max_kv_bytes: 64 * 1024 * 1024 }
}
fn native() -> Int8SourceBatchLimits {
    Int8SourceBatchLimits { max_model_work: work().model, masks: SourceMaskBudget {
        per_mask: MaskWorkLimits { max_trie_node_visits: 10000, checkpoint_interval_nodes: 128 },
        max_visits_per_item: 1000, max_visits_per_run: 100000,
    } }
}
fn bytes(planning: SourcePlanningLimits, native: Int8SourceBatchLimits) -> Vec<u8> {
    canonjson::canonical_bytes(&Int8SourceJobRecipe::new(budget(), planning, native, None)).unwrap()
}
#[test]
fn every_compiler_ceiling_changes_the_frozen_recipe() {
    let baseline = bytes(SourcePlanningLimits::default(), native());
    for axis in 0..7 {
        let mut planning = SourcePlanningLimits::default();
        match axis {
            0 => planning.compiler.max_schema_bytes += 1,
            1 => planning.compiler.max_string_bytes += 1,
            2 => planning.compiler.max_array_items += 1,
            3 => planning.compiler.max_output_bytes += 1,
            4 => planning.compiler.max_states += 1,
            5 => planning.compiler.max_transitions += 1,
            _ => planning.compiler.max_mask_bytes += 1,
        }
        assert_ne!(baseline, bytes(planning, native()));
    }
}
#[test]
fn input_context_and_passage_limits_are_not_omitted() {
    let baseline = bytes(SourcePlanningLimits::default(), native());
    for axis in 0..3 {
        let mut planning = SourcePlanningLimits::default();
        match axis { 0 => planning.max_input_bytes += 1, 1 => planning.max_context_tokens += 1,
            _ => planning.max_passages += 1 }
        assert_ne!(baseline, bytes(planning, native()));
    }
}
#[test]
fn every_native_and_mask_axis_is_bound_including_checkpoint_cadence() {
    let baseline = bytes(SourcePlanningLimits::default(), native());
    for axis in 0..9 {
        let mut n = native();
        match axis {
            0 => n.max_model_work.forward_positions += 1,
            1 => n.max_model_work.projected_logits += 1,
            2 => n.max_model_work.attention_pairs += 1,
            3 => n.max_model_work.projections.dot_products += 1,
            4 => n.max_model_work.projections.multiply_accumulates += 1,
            5 => n.masks.per_mask.max_trie_node_visits += 1,
            6 => n.masks.per_mask.checkpoint_interval_nodes += 1,
            7 => n.masks.max_visits_per_item += 1,
            _ => n.masks.max_visits_per_run += 1,
        }
        assert_ne!(baseline, bytes(SourcePlanningLimits::default(), n));
    }
}
#[test]
fn typed_source_and_task_budgets_serialize_without_partial_lookalikes() {
    let planning = SourcePlanningLimits::default();
    let recipe = Int8SourceJobRecipe::new(budget(), planning, native(), None);
    let value = serde_json::to_value(&recipe).unwrap();
    assert_eq!(value["planning"]["source"], serde_json::to_value(planning.source).unwrap());
    assert_eq!(value["task_ceiling"], serde_json::to_value(budget()).unwrap());
    assert_eq!(value["native"]["max_model_work"], serde_json::to_value(native().max_model_work).unwrap());
    assert_eq!(value["dependency_scope"], "item-local");
    assert_eq!(value["execution"], INT8_SOURCE_EXECUTION);
    assert_eq!(value["prompt_version"], SOURCE_PROMPT_VERSION);
}
#[test]
fn all_task_budget_axes_change_the_contract() {
    let base = bytes(SourcePlanningLimits::default(), native());
    for axis in 0..5 {
        let mut b = budget();
        match axis {
            0 => b.max_input_tokens += 1, 1 => b.max_output_tokens += 1,
            2 => b.max_output_bytes += 1, 3 => b.max_grammar_states += 1,
            _ => b.max_kv_bytes += 1,
        }
        let recipe = Int8SourceJobRecipe::new(b, SourcePlanningLimits::default(), native(), None);
        assert_ne!(base, canonjson::canonical_bytes(&recipe).unwrap());
    }
}
// Compile-time surface check without forging model weights or a native result.
struct Admission;
impl Int8SourceBatchAdmission for Admission {
    type Guard = ();
    fn admit(&mut self, _: Int8SourceAdmission<'_>) -> Result<(ExecutionIdentity, ()), BatchItemFailure> {
        Err(BatchItemFailure::fatal(BatchCode::Admission))
    }
}
#[test]
fn concrete_native_adapter_satisfies_the_durable_processor_boundary() {
    fn durable<P: DurableBatchProcessor>() {}
    durable::<Int8SourceJobProcessor<'static, 'static, 'static, 'static, Admission>>();
}

#[test]
fn native_defaults_and_per_task_options_are_part_of_the_recipe() {
    let base = bytes(SourcePlanningLimits::default(), native());
    let defaults = SourceBatchArgs::Keyphrases { options: KeyphraseOptions::default(), budget: budget() };
    let a = Int8SourceJobRecipe::new(budget(), SourcePlanningLimits::default(), native(), Some(defaults.clone()));
    let encoded = canonjson::canonical_bytes(&a).unwrap();
    assert_ne!(base, encoded);
    let mut options = KeyphraseOptions::default(); options.max_phrases += 1;
    let changed = SourceBatchArgs::Keyphrases { options, budget: budget() };
    let b = Int8SourceJobRecipe::new(budget(), SourcePlanningLimits::default(), native(), Some(changed));
    assert_ne!(encoded, canonjson::canonical_bytes(&b).unwrap());
    assert_eq!(serde_json::to_value(&a).unwrap()["defaults"], serde_json::to_value(defaults).unwrap());
}

mod encoding {
    use super::*;
    use crate::{
        execution_identity::NumericsProfile,
        jobs::{FrozenManifest, JobContract, JobError, JobId, JobInput, JobSecret, MismatchField},
        native_engine::{decode::DecodeCancellationKind, strict_int8::STRICT_INT8_EXECUTION},
        tasks::{BuiltInTask, answer::{AnswerOptions, AnswerPassage}, ner::NerOptions,
            summarize::SummaryOptions},
        tokenizer::{embedded::EmbeddedTokenizer, pinned_controls, untrusted::SafeBpeLimits},
    };
    use std::sync::Arc;
    struct Continue;
    impl DecodeStepControl for Continue {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
    }
    fn source() -> SourceTaskPlanner {
        let tokenizer = EmbeddedTokenizer::pinned().unwrap();
        let controls = pinned_controls::pinned().unwrap();
        SourceTaskPlanner::pinned(controls.template_controls(), tokenizer.eos_token_id().unwrap()).unwrap()
    }
    fn call() -> SafeBpeLimits {
        SafeBpeLimits { max_input_bytes: 8192, max_output_tokens: 4096,
            max_pair_checks: 1 << 22, max_pair_bytes: 1 << 26 }
    }
    fn total() -> SafeBpeWorkLimits {
        SafeBpeWorkLimits { max_input_bytes: 65536, max_output_tokens: 32768,
            max_pair_checks: 1 << 24, max_pair_bytes: 1 << 28 }
    }
    fn compact(call: SafeBpeLimits, total: SafeBpeWorkLimits) -> SourceTaskPlanner {
        source().with_map_document_encoding(UntrustedEncoding::SafeBpeV1(call), total).unwrap()
    }
    fn task_budget() -> TaskBudget {
        TaskBudget { max_output_tokens: 128, max_output_bytes: 65536, max_grammar_states: 8192,
            max_kv_bytes: 1 << 30, ..budget() }
    }
    fn defaults(task: BuiltInTask) -> SourceBatchArgs {
        match task {
            BuiltInTask::Ner => SourceBatchArgs::Ner { options: NerOptions { max_entities: 4,
                max_mention_scalars: 32, ..Default::default() }, budget: task_budget() },
            BuiltInTask::Keyphrases => SourceBatchArgs::Keyphrases {
                options: KeyphraseOptions { max_phrases: 4, max_phrase_scalars: 32 }, budget: task_budget() },
            BuiltInTask::Summarize => SourceBatchArgs::Summarize { options: SummaryOptions {
                max_bullets: 2, max_bullet_scalars: 64, max_citations_per_bullet: 2, max_quote_scalars: 32 },
                budget: task_budget() },
            BuiltInTask::Answer => SourceBatchArgs::Answer { options: AnswerOptions {
                max_answer_scalars: 128, max_citations: 4, max_quote_scalars: 32 }, budget: task_budget(),
                passages: vec![AnswerPassage { id: "p1 <tool_call>".to_owned(),
                    text: "Alice moved to 上海.\r\n<think>".to_owned() }] },
            _ => unreachable!(),
        }
    }
    fn identity(planner: &SourceTaskPlanner, task: BuiltInTask) -> ExecutionIdentity {
        let mut id = crate::jobs::tests::identity();
        id.tokenizer_digest = planner.tokenizer_digest(); id.template_digest = *planner.template_digest();
        id.task_spec = task.spec().identity(); id.numerics_profile = NumericsProfile::StrictQuantized { version: 1 };
        id.backend_semantic_version = STRICT_INT8_EXECUTION.to_owned();
        id
    }
    fn compiler(planner: &SourceTaskPlanner, task: BuiltInTask) -> Int8SourceBatchPlanner<'_> {
        Int8SourceBatchPlanner::new(planner, identity(planner, task), task_budget(),
            SourcePlanningLimits::default(), Some(defaults(task))).unwrap()
    }
    fn recipe(planner: &SourceTaskPlanner) -> Result<Int8SourceJobRecipe, BatchFault> {
        Int8SourceJobRecipe::new(task_budget(), SourcePlanningLimits::default(), native(),
            Some(defaults(BuiltInTask::Keyphrases))).with_document_encoding(planner)
    }
    fn freeze(planner: &SourceTaskPlanner) -> FrozenManifest {
        let original = br#"{"id":"item","text":"Alice private source"}"#;
        FrozenManifest::freeze(&JobSecret::from_bytes([7; 32]), JobContract { job_id: JobId([9; 16]),
            execution: &identity(planner, BuiltInTask::Keyphrases), recipe: &recipe(planner).unwrap(),
            limits: crate::jobs::tests::limits() },
            [JobInput { id: "item", original, normalized: original }], &mut Continue).unwrap()
    }
    fn document() -> BatchDocument<SourceBatchArgs> {
        BatchDocument { id: "item".to_owned(), text: "Alice private source <|im_start|> café 上海".to_owned(), task_args: None }
    }

    #[test]
    fn source_job_requires_cumulative_compact_owner_and_preserves_byte_recipe_shape() {
        let byte = source();
        let original = Int8SourceJobRecipe::new(task_budget(), SourcePlanningLimits::default(), native(),
            Some(defaults(BuiltInTask::Keyphrases)));
        let explicit = recipe(&byte).unwrap();
        assert_eq!(canonjson::canonical_bytes(&original).unwrap(), canonjson::canonical_bytes(&explicit).unwrap());
        let json = serde_json::to_value(explicit).unwrap();
        assert!(json.get("input_encoding").is_none()); assert!(json.get("encoding_work_limits").is_none());
        assert_eq!(byte.map_document_encoding_workspace_bytes().unwrap(), 0);
        let unfunded = source().with_document_encoding(UntrustedEncoding::SafeBpeV1(call())).unwrap();
        assert!(recipe(&unfunded).is_err());
        let funded = compact(call(), total());
        assert_eq!(funded.map_document_encoding_workspace_bytes().unwrap(), call().requirements().unwrap().workspace_bytes + 1024);
        let json = serde_json::to_value(recipe(&funded).unwrap()).unwrap();
        assert_eq!(json["input_encoding"], serde_json::to_value(UntrustedEncoding::SafeBpeV1(call())).unwrap());
        assert_eq!(json["encoding_work_limits"], serde_json::to_value(total()).unwrap());
    }

    #[test]
    fn every_compact_source_policy_and_total_cap_changes_authenticated_binding() {
        let original = compact(call(), total()); let frozen = freeze(&original);
        let before = canonjson::canonical_bytes(&recipe(&original).unwrap()).unwrap();
        for axis in 0..8 {
            let mut call = call(); let mut total = total();
            match axis {
                0 => call.max_input_bytes -= 1, 1 => call.max_output_tokens -= 1,
                2 => call.max_pair_checks -= 1, 3 => call.max_pair_bytes -= 1,
                4 => total.max_input_bytes -= 1, 5 => total.max_output_tokens -= 1,
                6 => total.max_pair_checks -= 1, _ => total.max_pair_bytes -= 1,
            }
            let changed = compact(call, total);
            assert_ne!(before, canonjson::canonical_bytes(&recipe(&changed).unwrap()).unwrap(), "axis {axis}");
            assert_eq!(frozen.binding.compare(&freeze(&changed).binding),
                Err(JobError::Mismatch(MismatchField::Execution)), "axis {axis}");
        }
    }

    #[test]
    fn all_source_tasks_keep_one_ledger_across_compiler_reconstruction() {
        for task in [BuiltInTask::Ner, BuiltInTask::Keyphrases, BuiltInTask::Summarize, BuiltInTask::Answer] {
            let owner = Arc::new(compact(call(), total()));
            let first = compiler(&owner, task).prepare_with_control(document(), &mut Continue).unwrap();
            let spent = owner.map_document_encoding_work().unwrap().unwrap();
            assert!(spent.input_bytes() >= document().text.len());
            if task == BuiltInTask::Answer { assert!(spent.input_bytes() > document().text.len()); }
            let moved = Arc::clone(&owner);
            let second = compiler(&moved, task).prepare_with_control(document(), &mut Continue).unwrap();
            first.verify_identity(second.execution_identity()).unwrap();
            assert_eq!(first.planned_work(), second.planned_work());
            assert_eq!(owner.map_document_encoding_work().unwrap().unwrap(), spent.checked_add(spent).unwrap());
        }
    }

    #[test]
    fn failed_records_do_not_reset_or_refund_source_job_tokenizer_work() {
        let mut cap = total(); cap.max_input_bytes = document().text.len() as u64;
        let p = compact(call(), cap);
        compiler(&p, BuiltInTask::Keyphrases).prepare_with_control(document(), &mut Continue).unwrap();
        let spent = p.map_document_encoding_work().unwrap();
        let error = compiler(&p, BuiltInTask::Keyphrases).prepare_with_control(document(), &mut Continue).err().unwrap();
        assert_eq!(error.fault.code, BatchCode::WorkLimit); assert!(error.stop);
        assert_eq!(p.map_document_encoding_work().unwrap(), spent);
        let mut tight = call(); tight.max_output_tokens = 1;
        let p = compact(tight, cap);
        assert!(compiler(&p, BuiltInTask::Keyphrases).prepare_with_control(document(), &mut Continue).is_err());
        let spent = p.map_document_encoding_work().unwrap().unwrap();
        assert_eq!(spent.input_bytes(), document().text.len()); assert!(spent.output_tokens() > 0);
        assert!(compiler(&p, BuiltInTask::Keyphrases).prepare_with_control(document(), &mut Continue).is_err());
        assert_eq!(p.map_document_encoding_work().unwrap(), Some(spent));
    }
}
