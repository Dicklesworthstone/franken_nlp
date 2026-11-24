//! Admission regressions; these do not substitute for native model gates.
use super::*;
use crate::{
    batch::{BatchWork, extract::ExtractionMaskBudget},
    grammar::mask::MaskWorkLimits,
    native_engine::portable_int8::ProjectionWork,
    tasks::{extract::semantic::{SemanticLimits, SemanticVerificationSpec}, judge::FaithfulnessPolicy},
    tokenizer::untrusted::SafeBpeLimits,
};

struct Fixture {
    extraction: TaskBudget,
    options: Int8SemanticOptions,
    planning: JudgeLimits,
    encoding: UntrustedEncoding,
    encoding_work: Option<SafeBpeWorkLimits>,
    work: Int8SemanticExtractionBatchLimits,
    limits: CorpusLimits,
}
fn work() -> Int8Work {
    Int8Work { forward_positions: 100_000, projected_logits: 1_000_000_000,
        attention_pairs: 10_000_000_000, projections: ProjectionWork {
            dot_products: 100_000_000_000, multiply_accumulates: 100_000_000_000_000 } }
}
fn fixture() -> Fixture {
    let kv = Int8MemoryRequirement::for_context(256).unwrap().kv_bytes;
    let extraction = TaskBudget { max_input_tokens: 224, max_output_tokens: 32,
        max_output_bytes: 65_536, max_grammar_states: 4096, max_kv_bytes: kv };
    let judge = TaskBudget { max_input_tokens: 240, max_output_tokens: 8,
        max_output_bytes: 16_384, max_grammar_states: 4096, max_kv_bytes: kv };
    let options = Int8SemanticOptions { specification: SemanticVerificationSpec {
        schema_version: 1, experimental_opt_in: true, revision: "semantic-population-fixture".into(),
        claims: vec![], judge_budget: judge, judge_policy: FaithfulnessPolicy {
            minimum_candidate_weight_ppm: 0, minimum_margin_milli: 0,
            evidence_window_bytes: 4096, max_evidence_windows: 8, max_evidence_spans: 8 },
    }, limits: SemanticLimits { max_fields: 3, max_claim_bytes: 128,
        max_total_claim_bytes: 1024, max_template_bytes: 1024, max_walk_nodes: 128,
        max_input_json_bytes: 8192, max_total_prompt_tokens: 2048,
        max_total_projected_logits: 1_000_000, max_output_bytes: 131_072 },
        max_model_work: work() };
    let run = work().checked_add(work()).unwrap();
    Fixture { extraction, options,
        planning: JudgeLimits { max_total_prompt_tokens: 2048, max_total_candidate_tokens: 256,
            max_total_projected_logits: 1_000_000, max_output_bytes: 16_384, ..JudgeLimits::default() },
        encoding: UntrustedEncoding::ByteFallback, encoding_work: None,
        work: Int8SemanticExtractionBatchLimits { max_model_work: run,
            masks: ExtractionMaskBudget {
                per_mask: MaskWorkLimits { max_trie_node_visits: 1000, checkpoint_interval_nodes: 16 },
                max_visits_per_item: 32_000, max_visits_per_run: 128_000 } },
        limits: CorpusLimits {
            native: NativeLimits { context_tokens: 256, allocator_reserve_bytes: 1 << 20,
                run: RunLimits { max_elapsed: Duration::from_secs(120), max_checkpoints: 1_000_000,
                    cleanup_reserve_bytes: 65_536 } },
            transport: BatchLimits { max_line_bytes: 4096, max_document_bytes: 1024,
                max_output_line_bytes: 135_168, max_output_bytes: 8 << 20,
                max_work: BatchWork { forward_positions: run.forward_positions,
                    projected_logits: run.projected_logits }, ..BatchLimits::default() },
            preparation_reserve_bytes: 512 << 20, io_reserve_bytes: 65_536,
        },
    }
}
impl Fixture {
    fn quote(&self, prefill: Option<Int8PrefillLimits>, threads: Option<usize>)
        -> Result<Int8SemanticCorpusRequirements, HostedError> {
        int8_semantic_corpus_requirements(self.extraction, &self.options, self.planning,
            &self.encoding, self.encoding_work, self.work, self.limits, prefill, threads)
    }
}
fn prefill(rows: usize) -> Int8PrefillLimits {
    Int8PrefillLimits { max_batch_rows: rows,
        max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(rows).unwrap() }
}

#[test]
fn complete_semantic_preparation_transport_and_output_are_charged_independently() {
    let f = fixture();
    let semantic = semantic_host(f.limits.native, f.limits.preparation_reserve_bytes, f.work)
        .requirements(f.extraction, f.options.specification.judge_budget, f.options.limits,
            f.planning, f.limits.transport.max_document_bytes).unwrap();
    let quote = f.quote(None, None).unwrap();
    assert_eq!(quote.native, semantic.native);
    assert_eq!(quote.native_workspace_bytes, semantic.native_workspace_bytes);
    assert_eq!(quote.preparation_bytes, semantic.preparation_bytes);
    assert_eq!(quote.input_bytes, semantic.preparation_bytes + f.limits.reservation_bytes().unwrap()
        - f.limits.preparation_reserve_bytes);
    assert_eq!(quote.output_bytes, semantic.output_bytes);
    assert!(quote.output_bytes > 4 * f.options.limits.max_output_bytes);
    assert_eq!(quote.total_bytes().unwrap(), quote.native.kv_bytes
        + quote.native_workspace_bytes + quote.input_bytes + quote.output_bytes);
}

#[test]
fn population_size_changes_work_authority_without_replicating_resident_memory() {
    let mut f = fixture(); let before = f.quote(None, None).unwrap();
    f.work.max_model_work = f.work.max_model_work.checked_add(f.work.max_model_work).unwrap();
    f.work.masks.max_visits_per_run *= 2;
    f.limits.transport.max_requests *= 2;
    f.limits.transport.max_input_bytes *= 2;
    assert_eq!(f.quote(None, None).unwrap(), before);
    f.limits.transport.max_document_bytes *= 2;
    let after = f.quote(None, None).unwrap();
    assert_eq!(after.native, before.native);
    assert_eq!(after.output_bytes, before.output_bytes);
    assert_eq!(after.preparation_bytes - before.preparation_bytes,
        (f.options.limits.max_fields as u64 + 2) * 1024);
}

#[test]
fn compact_work_ledger_coexists_with_all_field_plans_prompt_morsels_and_workers() {
    let mut f = fixture(); let byte = f.quote(None, None).unwrap();
    let call = SafeBpeLimits { max_input_bytes: 8192, max_output_tokens: 4096,
        max_pair_checks: 1_000_000, max_pair_bytes: 4_000_000 };
    f.encoding = UntrustedEncoding::SafeBpeV1(call);
    assert!(f.quote(None, None).is_err(), "compact populations require invocation limits");
    f.encoding_work = Some(SafeBpeWorkLimits { max_input_bytes: 65536, max_output_tokens: 32768,
        max_pair_checks: 8_000_000, max_pair_bytes: 32_000_000 });
    let compact = f.quote(None, None).unwrap();
    assert_eq!(compact.input_bytes - byte.input_bytes, call.requirements().unwrap().workspace_bytes + 1024);
    assert_eq!(compact.output_bytes, byte.output_bytes);
    for rows in [1, 3, 64] { for threads in [2, 4, 64] {
        let prompt = prefill(rows);
        let scheduled = f.quote(Some(prompt), Some(threads)).unwrap();
        assert_eq!(scheduled.native, compact.native);
        assert_eq!(scheduled.input_bytes, compact.input_bytes);
        assert_eq!(scheduled.output_bytes, compact.output_bytes);
        assert_eq!(scheduled.native_workspace_bytes - compact.native_workspace_bytes, prompt.validate().unwrap());
        assert_eq!(scheduled.team_workspace_bytes,
            dispatch::int8_prefill_team_workspace_bytes(threads, rows).unwrap());
        assert_eq!(scheduled.total_bytes().unwrap(), compact.total_bytes().unwrap()
            + prompt.validate().unwrap() + scheduled.team_workspace_bytes);
    } }
    f.encoding = UntrustedEncoding::ByteFallback;
    assert!(f.quote(None, None).is_err(), "byte policy cannot silently discard compact authority");
}

#[test]
fn invalid_full_attempt_or_preparation_limits_refuse_before_input_reads() {
    for axis in 0..18 {
        let mut f = fixture(); let document = f.options.max_model_work;
        match axis {
            0 => f.work.max_model_work.forward_positions = document.forward_positions - 1,
            1 => f.work.max_model_work.projected_logits = document.projected_logits - 1,
            2 => f.work.max_model_work.attention_pairs = document.attention_pairs - 1,
            3 => f.work.max_model_work.projections.dot_products = document.projections.dot_products - 1,
            4 => f.work.max_model_work.projections.multiply_accumulates = document.projections.multiply_accumulates - 1,
            5 => f.work.masks.max_visits_per_run = f.work.masks.max_visits_per_item - 1,
            6 => f.work.masks.max_visits_per_item = 0,
            7 => f.extraction.max_kv_bytes -= 1,
            8 => f.options.specification.judge_budget.max_kv_bytes -= 1,
            9 => f.limits.preparation_reserve_bytes = u64::MAX,
            10 => f.limits.io_reserve_bytes = 0,
            11 => f.limits.transport.max_document_bytes = 0,
            12 => f.limits.transport.max_output_line_bytes = f.options.limits.max_output_bytes as usize - 1,
            13 => f.options.specification.experimental_opt_in = false,
            14 => f.planning.per_head.max_nodes = 0,
            15 => f.work.masks.per_mask.max_trie_node_visits = 0,
            16 => f.planning.per_head.max_depth = 0,
            _ => f.limits.native.run.max_checkpoints = 1,
        }
        assert!(f.quote(None, None).is_err(), "axis {axis}");
    }
    let f = fixture();
    for threads in [0, 1, 65, usize::MAX] { assert!(f.quote(Some(prefill(3)), Some(threads)).is_err()); }
    let mut prompt = prefill(3); prompt.max_extra_scratch_bytes -= 1;
    assert!(f.quote(Some(prompt), Some(4)).is_err());
    let quote = f.quote(None, None).unwrap();
    assert!(Int8SemanticCorpusRequirements { input_bytes: u64::MAX, ..quote }.total_bytes().is_err());
}

#[test]
fn immutable_semantic_assets_and_combined_output_cross_the_owned_runtime_boundary() {
    fn send<T: Send + 'static>() {}
    fn sync<T: Sync>() {}
    send::<SemanticStream>(); send::<Arc<SemanticAssets>>();
    sync::<SemanticAssets>();
    send::<Int8SemanticExtractionBatchPlanner>();
}
