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
