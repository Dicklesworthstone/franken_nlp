//! Prefix-reuse CLI policy and admission regressions; these do not load a model.
use super::*;
use crate::native_engine::constrained_int8::{
    Int8JsonSamplingOptions, json_sampling_reuse_workspace_bytes,
    json_sampling_workspace_bytes,
};

fn assert_same_semantics(cold: &SampledSentimentCommand, reused: &SampledSentimentCommand) {
    assert_eq!(cold.options().unwrap(), reused.options().unwrap());
    assert_eq!(cold.work_ceiling(), reused.work_ceiling());
    assert_eq!(
        cold.prefill().unwrap().map(|limits|
            (limits.max_batch_rows, limits.max_extra_scratch_bytes)),
        reused.prefill().unwrap().map(|limits|
            (limits.max_batch_rows, limits.max_extra_scratch_bytes)),
    );
    let axes = crate::tasks::sentiment::SentimentAxis::ALL.len();
    let (_, cold_limits) = cold.common().unwrap();
    let (_, reused_limits) = reused.common().unwrap();
    let cold_budget = cold.budget(cold_limits, axes).unwrap();
    let reused_budget = reused.budget(reused_limits, axes).unwrap();
    assert_eq!(cold_budget.max_input_tokens, reused_budget.max_input_tokens);
    assert_eq!(cold_budget.max_output_tokens, reused_budget.max_output_tokens);
    assert_eq!(cold_budget.max_output_bytes, reused_budget.max_output_bytes);
    assert_eq!(cold_budget.max_grammar_states, reused_budget.max_grammar_states);
    assert_eq!(cold_budget.max_kv_bytes, reused_budget.max_kv_bytes);
}

#[test]
fn reuse_is_explicit_and_keeps_the_complete_cold_sample_policy() {
    let ordinary = json_sampling_workspace_bytes().unwrap();
    let default = command(&[]);
    assert!(!default.reuse_prefill);
    assert_eq!(default.sampler_bytes().unwrap(), ordinary);
    for samples in [1_usize, 2, 32] {
        let count = samples.to_string();
        let cold = command(&["--samples", &count, "--max-new-tokens", "2"]);
        let reused = command(&["--samples", &count, "--max-new-tokens", "2", "--reuse-prefill"]);
        assert!(!cold.reuse_prefill);
        assert!(reused.reuse_prefill);
        assert_same_semantics(&cold, &reused);
        assert_eq!(cold.sampler_bytes().unwrap(), ordinary);
        let expected = ordinary
            + (samples as u64) * (std::mem::size_of::<Int8JsonSamplingOptions>() as u64);
        assert_eq!(json_sampling_reuse_workspace_bytes(samples).unwrap(), expected);
        assert_eq!(reused.sampler_bytes().unwrap(), expected);
    }
}

#[test]
fn reuse_composes_with_declared_teams_and_layer_major_prompt_policy() {
    for threads in ["2", "4"] {
        let cold = command(&["--samples", "2", "--max-new-tokens", "2",
            "--compute-threads", threads, "--prefill-rows", "4"]);
        let reused = command(&["--samples", "2", "--max-new-tokens", "2",
            "--compute-threads", threads, "--prefill-rows", "4", "--reuse-prefill"]);
        assert_same_semantics(&cold, &reused);
        let prefill = reused.prefill().unwrap().unwrap();
        assert_eq!(prefill.max_batch_rows, 4);
        assert_eq!(prefill.max_extra_scratch_bytes,
            crate::native_engine::strict_int8::prefill::Int8PrefillLimits
                ::required_extra_scratch_bytes(4).unwrap());
        assert_eq!(reused.sampler_bytes().unwrap(),
            json_sampling_reuse_workspace_bytes(2).unwrap());
    }
}

#[test]
fn invalid_reused_sample_counts_refuse_before_input_or_output_io() {
    for count in ["0", "33"] {
        let requested = command(&["--samples", count, "--reuse-prefill"]);
        let mut input = NoIo;
        let mut output = NoIo;
        let failure = requested.execute(&mut input, &mut output).unwrap_err();
        assert_eq!(failure.exit, ErrorCode::Usage);
    }
}

#[test]
fn map_command_does_not_accept_unwired_prefill_reuse() {
    let failure = super::super::map::definition()
        .try_get_matches_from(["sentiment-sampled-map", "--reuse-prefill"])
        .unwrap_err();
    assert_eq!(failure.kind(), clap::error::ErrorKind::UnknownArgument);
}

#[test]
fn durable_job_rejects_reuse_until_its_recipe_supports_that_schedule() {
    let failure = super::super::job::definition()
        .try_get_matches_from(["sentiment-sampled-job", "start", "--reuse-prefill"])
        .unwrap_err();
    assert_eq!(failure.kind(), clap::error::ErrorKind::UnknownArgument);
}
