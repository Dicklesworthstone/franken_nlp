//! Real pinned task compilation against a tiny test-only model identity.
//! No decoder runs: the fixture weights cannot perform model inference.
use super::*;
use crate::{
    batch::classify::ClassificationBatchArgs,
    native_engine::{lmhead::{NANBEIGE_VOCAB_SIZE, scoring::ScoringMode},
        strict_int4::{Int4ModelIdentity, weights::projection_fixture}},
    tasks::{classify::{ClassificationLabel, ClassificationLimits, ClassificationMode, ClassificationPolicy,
            grouped_int4::Int4ClassificationPlanner},
        sentiment::{SentimentAxis, SentimentLimits, SentimentOptions, SentimentPolicy,
            batch::SentimentBatchArgs, grouped_int4::Int4SentimentPlanner}},
    tokenizer::untrusted::SafeBpeLimits,
};

fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 2048, max_output_tokens: 16, max_output_bytes: 1 << 20,
        max_grammar_states: 4096, max_kv_bytes: Int4MemoryRequirement::for_context(2048, 32).unwrap().kv_bytes }
}
fn native_limits() -> Int4ScoredBatchLimits {
    let work = Int4Work::for_sequence(32, 0, 4096, 64 * NANBEIGE_VOCAB_SIZE).unwrap();
    Int4ScoredBatchLimits { max_item_work: work, max_model_work: work.checked_add(work).unwrap() }
}
fn policy(bytes: usize) -> UntrustedEncoding {
    UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: bytes, max_output_tokens: 2048,
        max_pair_checks: 4 * 1024 * 1024, max_pair_bytes: 64 * 1024 * 1024 })
}
fn classify(model: &Int4ModelIdentity, policy: UntrustedEncoding) -> Int4ClassificationPlanner {
    Int4ClassificationPlanner::for_model(model, budget(), ClassificationLimits::default()).unwrap()
        .with_document_encoding(policy).unwrap()
}
fn labels() -> ClassificationBatchArgs {
    ClassificationBatchArgs { labels: vec![
        ClassificationLabel { id: "energy".to_owned(), description: "Energy sources and supply".to_owned() },
        ClassificationLabel { id: "homes".to_owned(), description: "Residential buildings".to_owned() }],
        mode: ClassificationMode::MultiLabel,
        policy: ClassificationPolicy { minimum_candidate_weight_ppm: 0, minimum_margin_ppm: 0 },
        budget: budget() }
}
fn source() -> String { "Solar energy powers local homes. <think> café 上海\n".repeat(8) }

#[test]
fn explicit_stream_compaction_preserves_all_heads_and_changes_complete_policy_identity() {
    let (model, _) = projection_fixture(32);
    let byte = Int4ScoredBatchPlanner::new(classify(model.identity(), UntrustedEncoding::ByteFallback),
        Some(labels()), native_limits()).unwrap();
    let selected = policy(4096);
    let compact = Int4ScoredBatchPlanner::new_with_document_encoding(classify(model.identity(), selected.clone()),
        Some(labels()), native_limits()).unwrap();
    assert_eq!(compact.document_encoding(), &selected);
    assert_eq!(byte.encoding_workspace_bytes(), 0);
    assert_eq!(compact.encoding_workspace_bytes(), memory::encoding_workspace_bytes(&selected).unwrap());
    let document = || BatchDocument { id: "original".to_owned(), text: source(), task_args: None };
    let byte_plan = byte.prepare(document(), &mut Continue).unwrap();
    let first = compact.prepare(document(), &mut Continue).unwrap();
    let second = compact.prepare(BatchDocument { id: "later".to_owned(), text: source(), task_args: Some(labels()) },
        &mut Continue).unwrap();
    assert_eq!(first.head_count(), 2);
    assert_eq!(first.head_count(), byte_plan.head_count());
    assert_eq!(first.model_work().projected_logits, byte_plan.model_work().projected_logits);
    assert!(first.model_work().forward_positions < byte_plan.model_work().forward_positions);
    assert_eq!(first.execution_identity(), second.execution_identity());
    assert_eq!(first.model_work(), second.model_work());
    assert_ne!(first.execution_identity().template_digest, byte_plan.execution_identity().template_digest);
    assert_ne!(first.execution_identity().prompt_digest, byte_plan.execution_identity().prompt_digest);
    let UntrustedEncoding::SafeBpeV1(mut changed) = selected else { panic!("compact policy required"); };
    changed.max_pair_checks += 1;
    let changed = Int4ScoredBatchPlanner::new_with_document_encoding(
        classify(model.identity(), UntrustedEncoding::SafeBpeV1(changed)), Some(labels()), native_limits()).unwrap();
    let changed = changed.prepare(document(), &mut Continue).unwrap();
    assert_ne!(first.execution_identity().template_digest, changed.execution_identity().template_digest);
    assert_eq!(first.execution_identity().prompt_digest, changed.execution_identity().prompt_digest);
}

#[test]
fn ordinary_batch_and_durable_factories_do_not_inherit_compact_admission() {
    let (model, _) = projection_fixture(32);
    for defaults in [None, Some(labels())] {
        assert!(Int4ScoredBatchPlanner::new(classify(model.identity(), policy(4096)), defaults, native_limits()).is_err());
    }
    #[cfg(all(feature = "metadata-store", target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        use crate::jobs::runner::scored::grouped_int4::Int4ScoredJobPlanner;
        assert!(Int4ScoredJobPlanner::new(classify(model.identity(), policy(4096)), Some(labels()), native_limits()).is_err());
    }
}

#[test]
fn label_metadata_spends_the_same_document_encoding_allowance_and_cancellation_is_fatal() {
    let (model, _) = projection_fixture(32);
    let text = source();
    let compiler = Int4ScoredBatchPlanner::new_with_document_encoding(
        classify(model.identity(), policy(text.len())), Some(labels()), native_limits()).unwrap();
    let failure = compiler.prepare(BatchDocument { id: "bounded".to_owned(), text, task_args: None },
        &mut Continue).err().expect("document plus label records exceed one byte ledger");
    assert_eq!(failure.fault.code, BatchCode::Planning);
    let compiler = Int4ScoredBatchPlanner::new_with_document_encoding(
        classify(model.identity(), policy(4096)), Some(labels()), native_limits()).unwrap();
    let failure = compiler.prepare(BatchDocument { id: "cancelled".to_owned(), text: source(), task_args: None },
        &mut StopAt { calls: 0, stop: 3 }).err().expect("cancelled preparation is not an item rejection");
    assert!(failure.stop);
    assert_eq!(failure.fault.cancellation, Some(DecodeCancellationKind::Deadline));
}

#[test]
fn sentiment_encodes_each_complete_document_once_but_charges_every_axis_prompt() {
    let (model, _) = projection_fixture(32);
    let text = source();
    let registry = crate::tokenizer::pinned_controls::pinned().unwrap();
    let eos = registry.template_controls().entries().iter()
        .find(|entry| entry.special && entry.surface == crate::template::IM_END).unwrap().id;
    let planner = |encoding| Int4SentimentPlanner::for_model(model.identity(), SentimentOptions {
        mode: ScoringMode::FullVocabulary, eos_token_id: eos,
        policy: SentimentPolicy { minimum_peak_weight_ppm: 0, maximum_normalized_entropy_ppm: 1_000_000 },
    }, budget(), SentimentLimits::default()).unwrap().with_document_encoding(encoding).unwrap();
    let defaults = SentimentBatchArgs { axes: vec![SentimentAxis::Valence, SentimentAxis::Arousal], budget: budget() };
    let compact = Int4ScoredBatchPlanner::new_with_document_encoding(planner(policy(text.len())),
        Some(defaults.clone()), native_limits()).unwrap();
    let byte = Int4ScoredBatchPlanner::new(planner(UntrustedEncoding::ByteFallback), Some(defaults), native_limits()).unwrap();
    let document = || BatchDocument { id: "axes".to_owned(), text: text.clone(), task_args: None };
    let compact = compact.prepare(document(), &mut Continue).unwrap();
    let byte = byte.prepare(document(), &mut Continue).unwrap();
    assert_eq!(compact.head_count(), 2);
    assert_eq!(compact.model_work().projected_logits, byte.model_work().projected_logits);
    assert!(compact.model_work().forward_positions < byte.model_work().forward_positions);
    assert!(compact.model_work().forward_positions > compact.plan.required_context() as u64);
}
