//! UNRUN pure admission fixtures; no allocation or model execution claims.
use super::*;
use super::super::tests::{budget, limits, options};
use crate::tokenizer::untrusted::SafeBpeLimits;

fn memory() -> SampledSentimentMapMemoryLimits {
    SampledSentimentMapMemoryLimits { context_tokens: 4096,
        preparation_reserve_bytes: MIN_SAMPLED_SENTIMENT_MAP_PREPARATION_BYTES,
        allocator_reserve_bytes: MIB, max_axes: 4, max_sampler_bytes: json_sampling_workspace_bytes().unwrap() }
}
#[test]
fn one_shared_compiler_prices_all_retained_programs_and_complete_output() {
    let planning = SampledSentimentLimits::default(); let mut mapping = limits(); mapping.chunks.max_chunks = 1;
    let one = memory().requirements(32, budget(), planning, mapping, &options(), &UntrustedEncoding::ByteFallback).unwrap();
    mapping.chunks.max_chunks = 2;
    let two = memory().requirements(32, budget(), planning, mapping, &options(), &UntrustedEncoding::ByteFallback).unwrap();
    let prompt_bytes = budget().max_input_tokens as u64 * 4 * PLAN_BYTES_PER_TOKEN;
    assert_eq!(two.input_bytes - one.input_bytes, prompt_bytes + grammar_bytes(planning).unwrap() + CHUNK_METADATA_BYTES);
    assert_eq!(two.native, one.native); assert_eq!(two.workspace_bytes, one.workspace_bytes);
    assert_eq!(two.max_output_tokens, 2 * one.max_output_tokens);
    assert_eq!(two.output_reservation_bytes, 4 * mapping.max_result_bytes + 8 * two.max_output_tokens + 4096);
    assert!(two.input_bytes < 2 * one.input_bytes);
}
#[test]
fn compact_encoding_adds_one_workspace_and_does_not_multiply_native_work() {
    let policy = UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 4096, max_output_tokens: 4096,
        max_pair_checks: 1_000_000, max_pair_bytes: 64 * MIB });
    let baseline = memory().requirements(32, budget(), SampledSentimentLimits::default(), limits(),
        &options(), &UntrustedEncoding::ByteFallback).unwrap();
    let compact = memory().requirements(32, budget(), SampledSentimentLimits::default(), limits(), &options(), &policy).unwrap();
    let UntrustedEncoding::SafeBpeV1(policy) = policy else { unreachable!() };
    assert_eq!(compact.input_bytes, baseline.input_bytes + policy.requirements().unwrap().workspace_bytes);
    assert_eq!(compact.native, baseline.native); assert_eq!(compact.workspace_bytes, baseline.workspace_bytes);
    assert_eq!(compact.output_reservation_bytes, baseline.output_reservation_bytes);
}
#[test]
fn missing_owned_workspace_context_and_arithmetic_are_refused() {
    for case in 0..5 {
        let mut selected = memory();
        match case { 0 => selected.preparation_reserve_bytes -= 1, 1 => selected.allocator_reserve_bytes = 0,
            2 => selected.max_sampler_bytes -= 1, 3 => selected.max_axes = 0, _ => selected.context_tokens = 128 }
        assert!(selected.requirements(32, budget(), SampledSentimentLimits::default(), limits(),
            &options(), &UntrustedEncoding::ByteFallback).is_err());
    }
    let mut planning = SampledSentimentLimits::default(); planning.compile.max_schema_bytes = usize::MAX;
    assert!(memory().requirements(32, budget(), planning, limits(), &options(), &UntrustedEncoding::ByteFallback).is_err());
    let mut task = budget(); task.max_kv_bytes = 1;
    assert!(memory().requirements(32, task, SampledSentimentLimits::default(), limits(), &options(),
        &UntrustedEncoding::ByteFallback).is_err());
}
