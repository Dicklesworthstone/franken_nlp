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
