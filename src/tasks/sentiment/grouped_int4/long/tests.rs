//! Contract fixtures use real pinned compilation and original finalizers with
//! private synthetic logits. They are not model execution or quality evidence.
use super::*;
use std::sync::OnceLock;
use crate::{native_engine::{lmhead::scoring::SequenceScoreRule,
        portable_int8::ProjectionWork,
        strict_int4::scoring::cohort::{Int4CandidateCohortRun, task::{self, fixtures}}},
    tasks::{ir::PromptSegmentKind, sentiment::{SentimentAxis, SentimentPolicy}},
    tokenizer::{embedded::EmbeddedTokenizer, pinned_controls}};
use fixtures::Continue;

mod capacity;
mod execution;
mod binding;
mod encoding;

fn task_budget() -> TaskBudget { fixtures::budget() }
fn fixture() -> (&'static SentimentPlanner, &'static ExecutionIdentity) {
    static FIXTURE: OnceLock<(SentimentPlanner, ExecutionIdentity)> = OnceLock::new();
    let (planner, identity) = FIXTURE.get_or_init(|| {
        let registry = pinned_controls::pinned().unwrap();
        let eos_token_id = EmbeddedTokenizer::pinned().unwrap().eos_token_id().unwrap();
        let planner = SentimentPlanner::pinned(registry.template_controls(), SentimentOptions {
            mode: ScoringMode::SequenceSoftmax { rule: SequenceScoreRule::SumLogits }, eos_token_id,
            policy: SentimentPolicy { minimum_peak_weight_ppm: 0, maximum_normalized_entropy_ppm: 1_000_000 },
        }).unwrap();
        let identity = fixtures::identity("sentiment-v1", planner.tokenizer_digest(), *planner.template_digest());
        (planner, identity)
    });
    (planner, identity)
}
fn source(text: &str, axes: &[SentimentAxis]) -> SentimentRequest {
    SentimentRequest { document: text.to_owned(), axes: axes.to_vec(), budget: task_budget() }
}
fn map_limits() -> Int4SentimentMapLimits {
    Int4SentimentMapLimits { chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 8,
        max_chunk_tokens: 1024, context_tokens: 1024, reserved_tokens: 1, max_chunks: 64,
        max_tokenizer_calls: 8192 },
        max_model_work: Int4Work { forward_positions: u64::MAX, projected_logits: u64::MAX,
            attention_pairs: u64::MAX, projections: ProjectionWork { dot_products: u64::MAX,
                multiply_accumulates: u64::MAX } },
        max_retained_result_bytes: 4 * 1024 * 1024, max_result_bytes: 8 * 1024 * 1024,
        aggregation: SentimentAggregationPolicy::default() }
}
fn prepare<'a>(request: &'a SentimentRequest, limits: Int4SentimentMapLimits)
    -> Result<PreparedInt4SentimentMap<'a>, Int4SentimentMapError> {
    prepare_with_control(request, limits, &mut Continue)
}
fn prepare_with_control<'a, C: DecodeStepControl>(request: &'a SentimentRequest, limits: Int4SentimentMapLimits,
    control: &mut C) -> Result<PreparedInt4SentimentMap<'a>, Int4SentimentMapError> {
    let (planner, identity) = fixture();
    let planning = SentimentLimits::default();
    let capacity = planner.source_capacity(&request.axes, request.budget, task_budget(), planning,
        limits.chunks.context_tokens).map_err(Int4SentimentError::from)?;
    PreparedInt4SentimentMap::compile(request, limits, capacity, 32,
        super::binding::Authority { ceiling: task_budget(), planning }, control, |row, control|
        planner.compile_int4(row, &PlanContext::new(identity, task_budget()).unwrap(), planning, 32, control))
}
fn heads(plan: &PreparedInt4Sentiment) -> Int4CandidateCohortRun {
    fixtures::run(plan.inner.heads.iter().zip(&plan.schedules).map(|(head, &schedule)|
        (head.scorer.score(&mut fixtures::Constant, plan.inner.options.mode).unwrap(), schedule)))
}
fn run(plan: &PreparedInt4Sentiment) -> Int4SentimentRun {
    let mut heads = heads(plan).heads.into_iter();
    plan.execute_heads(&mut Continue, |_, _, _, _| heads.next().ok_or(Int4SentimentError::Accounting)).unwrap()
}
fn finish(plan: &PreparedInt4SentimentMap<'_>) -> Result<Int4SentimentMapRun, Int4SentimentMapError> {
    let mut collected = super::execution::Collection::new(plan)?;
    for (index, chunk) in plan.plans.iter().enumerate() { collected.push(index, run(chunk), &mut Continue)?; }
    collected.finish(&mut Continue)
}
fn cohort_budget(plan: &PreparedInt4SentimentMap<'_>) -> Int4SentimentMapCohortBudget {
    Int4SentimentMapCohortBudget { native: Int4RunBudget::exact(plan.planned_work()),
        memory: Int4MemoryBudget { max_kv_bytes: u64::MAX, max_rope_bytes: u64::MAX,
            max_scratch_payload_bytes: u64::MAX }, max_task_workspace_bytes: u64::MAX,
        max_output_bytes: MAX_BYTES }
}
struct Stop(DecodeCancellationKind);
impl DecodeStepControl for Stop {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(self.0) }
}
