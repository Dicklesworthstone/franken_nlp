//! Pure admission checks: no native model or hosted execution is performed.
use super::*;
use super::super::super::tests::fixture;

#[test]
fn synthesis_phase_quotes_preserve_scalar_authority_and_the_complete_discovery_window() {
    for width in [16, 32, 128] {
        let (_, mut config) = fixture(SourceMapTask::Summarize(Default::default()), width);
        let budget = config.budget;
        let scalar = Int4MemoryRequirement::for_context(config.native.context_tokens, width).unwrap();
        for chunks in [1, 2, 4, 8] {
            config.mapping.reduction.map_batch_chunks = chunks;
            let discovery = config.cohort_requirements(width).unwrap();
            let phases = config.synthesis_cohort_requirements(width).unwrap();
            assert!(phases.native.kv_bytes >= scalar.kv_bytes);
            assert!(phases.native.rope_bytes >= scalar.rope_bytes);
            assert!(phases.native.scratch_payload_bound >= scalar.scratch_payload_bound);
            assert_eq!(phases.native.kv_bytes, chunks as u64 * scalar.kv_bytes);
            assert_eq!(phases.workspace_bytes, discovery.workspace_bytes);
            assert_eq!(phases.max_result_bytes, discovery.max_result_bytes);
            assert_eq!(config.budget, budget, "a wider physical window cannot expand per-task authority");
        }
    }
}

#[test]
fn discovery_quote_cannot_authorize_an_underfunded_synthesis_context() {
    let (_, mut config) = fixture(SourceMapTask::Summarize(Default::default()), 32);
    config.mapping.reduction.map_batch_chunks = 4;
    config.budget.max_kv_bytes -= 1;
    assert!(config.cohort_requirements(32).is_ok());
    assert!(config.synthesis_cohort_requirements(32).is_err());
    let (_, mut config) = fixture(SourceMapTask::Summarize(Default::default()), 32);
    config.native.context_tokens += 1;
    assert!(config.synthesis_cohort_requirements(32).is_err());
}
