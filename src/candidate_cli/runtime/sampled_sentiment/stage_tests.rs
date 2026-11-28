//! Original typed stage stops survive a later invocation deadline.
use super::*;

#[test]
fn artifact_and_native_stops_survive_post_stage_deadline_checks() {
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast,
        RaceLost, ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        for error in [
            SessionStageError::Artifact(ArtifactBridgeError::Cancelled(cause)),
            SessionStageError::Host(HostedError::Native(StrictInt8Error::Cancelled(cause))),
        ] {
            let result: Result<(), Failure> = checked_stage(Err(error),
                || Err(CandidateError::Timeout), session_failure);
            let failure = result.unwrap_err();
            assert_eq!(failure.cancellation, Some(cause));
            assert_eq!(failure.exit, cancelled(cause).exit);
        }
    }
}

#[test]
fn a_later_deadline_wins_over_a_nonterminal_setup_failure() {
    let result: Result<(), Failure> = checked_stage(
        Err(SessionStageError::Setup(CandidateError::Model)),
        || Err(CandidateError::Timeout), session_failure);
    let failure = result.unwrap_err();
    assert_eq!(failure.cancellation, Some(DecodeCancellationKind::Deadline));
    assert_eq!(failure.exit, ErrorCode::BudgetOrTimeout);
}

#[test]
fn optional_selected_policy_preserves_default_output_shape_and_names_only_the_public_cap() {
    let base = serde_json::json!({"schema_version":1,"output":{"status":"fixture"}});
    let dense = ProjectionResponse { response: &base, max_selected_rows: None };
    assert_eq!(serde_json::to_value(dense).unwrap(), base);
    let selected = ProjectionResponse { response: &base, max_selected_rows: Some(4096) };
    let mut value = serde_json::to_value(selected).unwrap();
    assert_eq!(value.as_object_mut().unwrap().remove("max_selected_rows"), Some(serde_json::json!(4096)));
    assert_eq!(value, base);
}
