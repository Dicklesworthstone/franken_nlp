use super::*;
use std::rc::Rc;
use crate::{canonjson, batch::BatchItemFailure,
    execution_identity::{NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    native_engine::hf_bf16_eager::HF_BF16_EAGER_PROFILE,
    tasks::{ir::ScoreSpace, summarize::{CitationGuarantee, SummarySemanticSupport, CitedBullet, SourceCitation}},
    template::{IM_START, IM_END, THINK_START, THINK_END},
    tokenizer::{bpe::EncodeOptions, embedded::EmbeddedTokenizer, specials::ArchivedControlRegistries},
    validation::grounded_fields::{GroundingBudget, SourceOccurrence, scan_occurrences}};

struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn planner() -> SourceTaskPlanner {
    let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    // Synthetic fixture census, not production registry provenance.
    let entries: Vec<_> = [IM_START, IM_END, THINK_START, THINK_END].iter().map(|&surface| {
        let ids = tokenizer.tokenizer().encode_ids_with_options(surface, EncodeOptions { add_bos: false, add_eos: false }).unwrap();
        assert_eq!(ids.len(), 1);
        serde_json::json!({"id":ids[0], "special":surface == IM_START || surface == IM_END, "surface":surface})
    }).collect();
    let eos = entries[1]["id"].as_u64().unwrap() as u32;
    let specials: Vec<_> = entries.iter().filter(|e| e["special"] == true).cloned().collect();
    let registry = ArchivedControlRegistries::from_archived_json(
        &serde_json::json!({"schema_version":1,"registry":"TokenizerSpecialIds","entries":specials}).to_string(),
        &serde_json::json!({"schema_version":1,"registry":"TemplateControlIds","entries":entries}).to_string()).unwrap();
    SourceTaskPlanner::pinned(registry.template_controls(), eos).unwrap()
}
fn identity(p: &SourceTaskPlanner) -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"fixture");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "bf16-verbatim".to_owned(), packing_set_digest: d,
        tokenizer_digest: p.tokenizer_digest(), template_digest: *p.template_digest(), task_spec: "summarize-v1".to_owned(),
        taskir_digest: d, prompt_digest: d, grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::HfBf16Eager, kv_dtype: "bf16".to_owned(), sampler_version: "fixture".to_owned(),
        thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None, calibration_digest: d, decision_policy_digest: d,
        backend_semantic_version: "fixture".to_owned(), host_class: None, compiler_identity: None }
}
fn config() -> NativeCorpusSummaryConfig {
    NativeCorpusSummaryConfig {
        chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 16, max_chunk_tokens: 16,
            reserved_tokens: 1, context_tokens: 8192, ..ChunkLimits::default() },
        execution: ExecutionLimits::default(), aggregation: CorpusSummaryLimits::default(),
        options: SummaryOptions { max_bullets: 2, max_bullet_scalars: 32, max_citations_per_bullet: 2, max_quote_scalars: 8 },
        budget: TaskBudget { max_input_tokens: 4096, max_output_tokens: 32, max_output_bytes: 65536,
            max_grammar_states: 4096, max_kv_bytes: (8192 * KV_BYTES_PER_TOKEN) as u64 },
        planning: SourcePlanningLimits::default(),
        masks: SourceMaskBudget { per_mask: Default::default(), max_visits_per_item: 10000, max_visits_per_run: 10000000 },
        max_work: BatchWork { forward_positions: 1000000, projected_logits: u64::MAX },
        max_bullets: 4, max_result_bytes: 1000000, max_retained_guard_bytes: 1024 * 1024,
    }
}
#[test]
fn real_pinned_scaffold_and_source_counts_match_every_prepared_map() {
    let p = planner(); let id = identity(&p); let c = config();
    let source = "é alpha\r\n<think> 上海 repeated text and more text";
    let prepared = prepare_summary_corpus(source, &p, &id, c, &mut Continue).unwrap();
    assert_eq!(prepared.chunks().chunks().iter().map(|c| c.text()).collect::<String>(), source);
    assert!(prepared.scaffold_tokens() > c.chunks.reserved_tokens);
    let compiler = SourceBatchPlanner::new(&p, id, c.budget, c.planning, None).unwrap();
    let mut total = BatchWork::default();
    for chunk in prepared.chunks().chunks() {
        let map = compiler.prepare(BatchDocument { id: chunk.id().to_string(), text: chunk.text().to_owned(),
            task_args: Some(SourceBatchArgs::Summarize { options: c.options, budget: c.budget }) }).unwrap();
        assert_eq!(map.planned_work(), chunk_work(chunk.tokens(), prepared.scaffold_tokens(), c.budget.max_output_tokens).unwrap());
        total.forward_positions += map.planned_work().forward_positions;
        total.projected_logits += map.planned_work().projected_logits;
    }
    assert_eq!(total, prepared.planned_work());
}
#[test]
fn whole_run_forward_projection_and_mask_bounds_fail_before_model_admission() {
    let p = planner(); let id = identity(&p); let mut c = config(); let source = "a".repeat(64);
    let prepared = prepare_summary_corpus(&source, &p, &id, c, &mut Continue).unwrap();
    c.max_work = prepared.planned_work(); c.masks.max_visits_per_run = prepared.reserved_mask_visits();
    assert!(prepare_summary_corpus(&source, &p, &id, c, &mut Continue).is_ok());
    for axis in 0..3 {
        let mut narrow = c;
        match axis { 0 => narrow.max_work.forward_positions -= 1, 1 => narrow.max_work.projected_logits -= 1,
            _ => narrow.masks.max_visits_per_run -= 1 }
        assert!(matches!(prepare_summary_corpus(&source, &p, &id, narrow, &mut Continue), Err(NativeCorpusSummaryError::WorkBudget)));
    }
}
#[test]
fn chunk_limits_reserve_schema_output_prompt_context_and_kv_together() {
    for scaffold in 1..32 { for output in 1..8 { for context in 8..64 {
        let mut c = config(); c.budget.max_output_tokens = output;
        c.budget.max_input_tokens = context; c.planning.max_context_tokens = context as usize;
        c.chunks.context_tokens = context as usize; c.chunks.reserved_tokens = 1;
        c.chunks.max_chunk_tokens = 128; c.budget.max_kv_bytes = (context as u64 - 1) * KV_BYTES_PER_TOKEN as u64;
        if let Ok(limits) = effective_chunks(c, scaffold) {
            let tokens = limits.effective_token_limit().unwrap();
            assert!(tokens + scaffold <= c.budget.max_input_tokens as usize);
            assert!(tokens + scaffold + output as usize <= context as usize);
            assert!((tokens + scaffold + output as usize - 1) as u64 * KV_BYTES_PER_TOKEN as u64 <= c.budget.max_kv_bytes);
            assert!(tokens <= c.chunks.max_chunk_tokens);
        }
    } } }
}
#[test]
fn cancellation_uses_dedicated_prefill_hook_and_retains_exact_reason() {
    struct Stop;
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { panic!("wrong hook") }
        fn prefill_checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::Deadline) }
    }
    let p = planner(); let id = identity(&p);
    assert!(matches!(prepare_summary_corpus("source", &p, &id, config(), &mut Stop),
        Err(NativeCorpusSummaryError::Cancelled(DecodeCancellationKind::Deadline))));
    let mut stop = Stop; let cell = RefCell::new(&mut stop); let cause = Cell::new(None);
    assert_eq!(checkpoint(&cell, &cause), Err(MapReduceError::Cancelled));
    assert_eq!(cause.get(), Some(DecodeCancellationKind::Deadline));
}
#[test]
fn changed_task_or_template_and_oversize_sources_do_not_prepare() {
    let p = planner(); let mut id = identity(&p); let c = config();
    id.task_spec = "answer-v1".to_owned(); assert!(prepare_summary_corpus("a", &p, &id, c, &mut Continue).is_err());
    id = identity(&p); id.template_digest = Sha256Digest::of_bytes(b"changed");
    assert!(prepare_summary_corpus("a", &p, &id, c, &mut Continue).is_err());
    assert!(prepare_summary_corpus("", &p, &identity(&p), c, &mut Continue).is_err());
    assert!(prepare_summary_corpus(&"a".repeat(4097), &p, &identity(&p), c, &mut Continue).is_err());
}
#[test]
fn work_reservation_is_atomic_and_never_wraps() {
    let original = BatchWork { forward_positions: 10, projected_logits: 20 }; let mut remaining = original;
    assert!(reserve(&mut remaining, BatchWork { forward_positions: 1, projected_logits: 21 }).is_err());
    assert_eq!(remaining, original);
    reserve(&mut remaining, original).unwrap(); assert_eq!(remaining, BatchWork::default());
    assert!(reserve(&mut remaining, original).is_err());
    assert!(chunk_work(usize::MAX, usize::MAX, u32::MAX).is_err());
}

struct Guard(Rc<Cell<usize>>);
impl Drop for Guard { fn drop(&mut self) { self.0.set(self.0.get() - 1); } }
struct Mock { live: Rc<Cell<usize>>, calls: Rc<Cell<usize>>, fail: bool, wrong_work: bool }
impl BatchProcessor for Mock {
    type Args = SourceBatchArgs;
    type Prepared = BatchDocument<SourceBatchArgs>;
    type Output = GuardedOutput<SourceTaskResult, Guard>;
    fn prepare(&mut self, doc: Self::Prepared) -> Result<Self::Prepared, BatchItemFailure> { Ok(doc) }
    fn planned_work(&self, doc: &Self::Prepared) -> BatchWork {
        let mut w = chunk_work(doc.text.len(), 3, 32).unwrap();
        if self.wrong_work { w.forward_positions += 1; } w
    }
    fn execute<C: DecodeStepControl>(&mut self, doc: Self::Prepared, _: &mut C) -> Result<Self::Output, BatchItemFailure> {
        self.calls.set(self.calls.get() + 1);
        if self.fail { return Err(BatchItemFailure::fatal(BatchFault::cancelled(DecodeCancellationKind::User))); }
        let spans = scan_occurrences(&doc.text, "a", &mut GroundingBudget::default()).unwrap();
        let forward = doc.text.len() as u64 + 4;
        let summary = SummaryResult { schema_version: 1, task_spec_version: SUMMARIZE_TASK_VERSION.to_owned(),
            numerics_profile: HF_BF16_EAGER_PROFILE.to_owned(), citation_guarantee: CitationGuarantee::StructuralSourceMembership,
            semantic_support: SummarySemanticSupport::NotAssessed, score_space: ScoreSpace::NotComputed,
            bullets: vec![CitedBullet { text: "claim".to_owned(), citations: vec![SourceCitation {
                quote: "a".to_owned(), occurrence: SourceOccurrence::Ambiguous, spans }] }],
            generated_token_ids: vec![1, 0], forward_positions: forward, projected_logits: forward * NANBEIGE_VOCAB_SIZE as u64,
            mask_node_visit_charge: 2 };
        self.live.set(self.live.get() + 1);
        Ok(GuardedOutput::new(SourceTaskResult::Summarize(summary), Guard(self.live.clone())))
    }
}
fn chunks() -> ChunkPlan<'static> {
    ChunkPlan::build("aaaaaaaa", ChunkLimits { max_chunk_bytes: 4, max_chunk_tokens: 4,
        ..ChunkLimits::default() }, |s| Ok(s.len())).unwrap()
}
#[test]
fn all_map_guards_survive_reduction_serialization_and_successful_delivery() {
    let plan = chunks(); let live = Rc::new(Cell::new(0)); let calls = Rc::new(Cell::new(0));
    let mut control = Continue; let control = RefCell::new(&mut control);
    let one = chunk_work(4, 3, 32).unwrap();
    let pass = BatchSummaryPass { processor: Mock { live: live.clone(), calls: calls.clone(), fail: false, wrong_work: false },
        control: &control, options: config().options, budget: config().budget, scaffold_tokens: 3,
        remaining: BatchWork { forward_positions: one.forward_positions * 2, projected_logits: one.projected_logits * 2 },
        guards: Vec::with_capacity(2), failed: false };
    let mut task = CorpusSummaryTask::new(pass, CorpusSummaryLimits::default()).unwrap();
    let result = mapreduce::execute(&plan, &mut task, ExecutionLimits::default(), || Ok(())).unwrap();
    assert_eq!(live.get(), 2); assert_eq!(calls.get(), 2);
    let pass = task.into_pass(); assert_eq!(pass.remaining, BatchWork::default());
    let output = GuardedOutput::new(result.into_value().into_ranked(1, 100000).unwrap(), pass.guards);
    let bytes = canonjson::canonical_string(&output).unwrap();
    assert!(!bytes.contains("guard")); assert_eq!(live.get(), 2);
    drop(output); assert_eq!(live.get(), 0);
}
#[test]
fn cancelled_maps_keep_the_work_charge_and_cannot_retry() {
    let plan = chunks(); let live = Rc::new(Cell::new(0)); let calls = Rc::new(Cell::new(0));
    let mut control = Continue; let control = RefCell::new(&mut control);
    let charge = chunk_work(4, 3, 32).unwrap();
    let mut pass = BatchSummaryPass { processor: Mock { live: live.clone(), calls: calls.clone(), fail: true, wrong_work: false },
        control: &control, options: config().options, budget: config().budget, scaffold_tokens: 3,
        remaining: charge, guards: Vec::with_capacity(1), failed: false };
    assert_eq!(pass.run(&plan.chunks()[0]).unwrap_err().cancellation, Some(DecodeCancellationKind::User));
    assert_eq!(pass.remaining, BatchWork::default()); assert!(pass.run(&plan.chunks()[0]).is_err());
    assert_eq!(calls.get(), 1); assert_eq!(live.get(), 0);
}
#[test]
fn changed_planned_work_is_rejected_without_calling_the_processor() {
    let plan = chunks(); let calls = Rc::new(Cell::new(0)); let mut control = Continue; let control = RefCell::new(&mut control);
    let mut pass = BatchSummaryPass { processor: Mock { live: Rc::new(Cell::new(0)), calls: calls.clone(), fail: false, wrong_work: true },
        control: &control, options: config().options, budget: config().budget, scaffold_tokens: 3,
        remaining: chunk_work(4, 3, 32).unwrap(), guards: Vec::with_capacity(1), failed: false };
    assert_eq!(pass.run(&plan.chunks()[0]).unwrap_err().code, BatchCode::InvalidExecution);
    assert_eq!(calls.get(), 0);
}
#[test]
fn reduction_failure_drops_retained_guards_only_after_the_task_is_drained() {
    let plan = chunks(); let live = Rc::new(Cell::new(0)); let mut control = Continue; let control = RefCell::new(&mut control);
    let one = chunk_work(4, 3, 32).unwrap();
    let pass = BatchSummaryPass { processor: Mock { live: live.clone(), calls: Rc::new(Cell::new(0)), fail: false, wrong_work: false },
        control: &control, options: config().options, budget: config().budget, scaffold_tokens: 3,
        remaining: BatchWork { forward_positions: one.forward_positions * 2, projected_logits: one.projected_logits * 2 },
        guards: Vec::with_capacity(2), failed: false };
    let mut task = CorpusSummaryTask::new(pass, CorpusSummaryLimits::default()).unwrap();
    let result = mapreduce::execute(&plan, &mut task, ExecutionLimits { max_result_bytes: 1,
        ..ExecutionLimits::default() }, || Ok(()));
    assert!(result.is_err()); assert_eq!(live.get(), 2);
    drop(task); assert_eq!(live.get(), 0);
}
