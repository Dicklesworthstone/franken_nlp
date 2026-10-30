//! Synthetic task-finalizer fixtures, never native execution or model evidence.
use super::*;
use crate::execution_identity::{ExecutionIdentity, NumericsProfile, Sha256Digest, ThinkingMode, ToolMode};
use crate::native_engine::lmhead::scoring::{CandidateLogits, CandidateScores, ProjectionRows};

pub(crate) struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
pub(crate) fn identity(task: &str, tokenizer: Sha256Digest, template: Sha256Digest) -> ExecutionIdentity {
    let digest = Sha256Digest::of_bytes(b"int4-task-cohort-synthetic-score-fixture");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture".into(), logical_model_digest: digest,
        artifact_format: "synthetic-only".into(), quant_recipe: "fixture-int4".into(), packing_set_digest: digest,
        tokenizer_digest: tokenizer, template_digest: template, task_spec: task.into(), taskir_digest: digest,
        prompt_digest: digest, grammar_compiler_version: "none".into(), schema_digest: digest,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".into(),
        sampler_version: "fixture".into(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: digest, decision_policy_digest: digest,
        backend_semantic_version: crate::native_engine::strict_int4::STRICT_INT4_EXECUTION.into(),
        host_class: None, compiler_identity: None }
}
pub(crate) struct Constant;
impl CandidateLogits for Constant {
    type Error = &'static str;
    fn project(&mut self, _: &[u32], rows: ProjectionRows<'_>) -> Result<Vec<f32>, Self::Error> {
        let width = match rows { ProjectionRows::FullVocabulary { vocabulary_size } => vocabulary_size,
            ProjectionRows::Selected(ids) => ids.len() };
        Ok(vec![0.0; width])
    }
}
pub(crate) fn run(heads: impl IntoIterator<Item = (CandidateScores, CandidateSchedule)>) -> Int4CandidateCohortRun {
    let mut work = Int4Work::default(); let mut projections = 0; let mut width = None;
    let heads: Vec<_> = heads.into_iter().map(|(scores, schedule)| {
        if let Some(width) = width { assert_eq!(width, schedule.group_width); }
        else { width = Some(schedule.group_width); }
        work = work.checked_add(schedule.model).unwrap(); projections += schedule.scoring.prefix_evaluations as u64;
        Int4CandidateRun { schema_version: 1, execution: INT4_SCORING_EXECUTION.into(),
            numerics_profile: STRICT_INT4_PROFILE.into(), group_width: schedule.group_width,
            scores, model_work: schedule.model, rewound_positions: schedule.rewound }
    }).collect();
    Int4CandidateCohortRun { schema_version: 1, execution: INT4_SCORING_COHORT_EXECUTION.into(),
        numerics_profile: STRICT_INT4_PROFILE.into(), group_width: width.expect("nonempty fixture heads"),
        heads, group_steps: work.forward_positions, projection_groups: projections, model_work: work }
}
pub(crate) fn budget() -> crate::tasks::ir::TaskBudget {
    crate::tasks::ir::TaskBudget { max_input_tokens: 4096, max_output_tokens: 16, max_output_bytes: 1_000_000,
        max_grammar_states: 4096, max_kv_bytes: 1 << 30 }
}
