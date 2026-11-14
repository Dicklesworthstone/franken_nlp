use super::*;

#[test]
fn all_cancellation_causes_keep_the_frozen_exit_category_through_task_and_host() {
    use DecodeCancellationKind::*;
    for (cause, exit) in [
        (Timeout, ErrorCode::BudgetOrTimeout), (Deadline, ErrorCode::BudgetOrTimeout),
        (PollQuota, ErrorCode::BudgetOrTimeout), (CostBudget, ErrorCode::BudgetOrTimeout),
        (User, ErrorCode::Cancelled), (ParentCancelled, ErrorCode::Cancelled),
        (Shutdown, ErrorCode::Cancelled), (ResourceUnavailable, ErrorCode::AdmissionOrResourceLimit),
        (FailFast, ErrorCode::Generic), (LinkedExit, ErrorCode::Generic), (RaceLost, ErrorCode::Generic),
    ] {
        for failure in [
            cancelled(cause), task_failure(SampledSentimentError::Cancelled(cause)),
            task_failure(SampledSentimentError::Native(Int4JsonError::Decode(JsonDecodeError::Cancelled(cause)))),
            host_failure(HostedError::Int4(Int4HostError::Native(StrictInt4Error::Cancelled(cause)))),
            host_failure(HostedError::Int4(Int4HostError::SampledSentiment(
                SampledSentimentError::Native(Int4JsonError::Native(StrictInt4Error::Cancelled(cause)))))),
        ] {
            assert_eq!(failure.exit, exit);
            assert_eq!(failure.cancellation, Some(cause));
        }
    }
}

#[test]
fn budget_resource_and_structured_no_result_failures_are_distinct() {
    use SampledSentimentError::*;
    assert_eq!(admission_failure(CandidateError::Planning).exit, ErrorCode::AdmissionOrResourceLimit);
    assert_eq!(host_failure(HostedError::Int4(Int4HostError::Native(StrictInt4Error::Memory))).exit,
        ErrorCode::AdmissionOrResourceLimit);
    for error in [Limits, WorkBudget, Allocation,
        Native(Int4JsonError::Decode(JsonDecodeError::AllocationRefused)),
        Native(Int4JsonError::Native(StrictInt4Error::Context)),
        Native(Int4JsonError::Native(StrictInt4Error::Memory))] {
        assert_eq!(task_failure(error).exit, ErrorCode::AdmissionOrResourceLimit);
    }
    for error in [OutputBudget,
        Native(Int4JsonError::Decode(JsonDecodeError::BudgetExceeded("output before EOS"))),
        Native(Int4JsonError::Decode(JsonDecodeError::Mask(MaskOracleError::WorkBudgetExceeded {
            max_trie_node_visits: 1 }))),
        Native(Int4JsonError::Native(StrictInt4Error::Work))] {
        assert_eq!(task_failure(error).exit, ErrorCode::BudgetOrTimeout);
    }
    for error in [JsonDecodeError::NoLegalToken, JsonDecodeError::IndependentValidation] {
        let failure = task_failure(Native(Int4JsonError::Decode(error)));
        assert_eq!(failure.exit, ErrorCode::StructuredTaskNoResult);
        assert_eq!(failure.cancellation, None);
    }
    assert_eq!(task_failure(Schema).exit, ErrorCode::SchemaOrRecipeCompile);
    assert_eq!(task_failure(NoResult).exit, ErrorCode::StructuredTaskNoResult);
    assert_eq!(task_failure(Identity).exit, ErrorCode::ArtifactIntegrityOrFormatOrVersion);
}

#[test]
fn invariant_or_completion_failures_do_not_become_ordinary_no_result() {
    for error in [SampledSentimentError::Accounting, SampledSentimentError::Serialization,
        SampledSentimentError::Native(Int4JsonError::WorkMismatch),
        SampledSentimentError::Native(Int4JsonError::Decode(JsonDecodeError::InvalidLogits)),
        SampledSentimentError::Native(Int4JsonError::Decode(JsonDecodeError::IllegalTransition))] {
        assert_eq!(task_failure(error).exit, ErrorCode::Generic);
    }
    for error in [HostedError::Panicked, HostedError::CompletionMissing] {
        assert_eq!(host_failure(error).exit, ErrorCode::Generic);
    }
    assert_eq!(host_failure(HostedError::Limits("PRIVATE" )).exit, ErrorCode::AdmissionOrResourceLimit);
}
