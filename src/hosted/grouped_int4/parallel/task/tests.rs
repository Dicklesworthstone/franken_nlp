//! Shared team admission/ownership regressions, not model-quality evidence.
use super::*;
use crate::tasks::{classify::grouped_int4::Int4ClassificationRun,
    extract::grouped_int4::Int4ExtractRun, judge::grouped_int4::Int4JudgeRun,
    sentiment::grouped_int4::Int4SentimentRun, source_planning::grouped_int4::Int4SourceTaskRun};

#[test]
fn every_complete_task_result_retains_the_same_sendable_guarded_team_container() {
    fn send<T: Send + 'static>() {}
    send::<HostedInt4TeamResult<Int4ClassificationRun>>();
    send::<HostedInt4TeamResult<Int4ExtractRun>>();
    send::<HostedInt4TeamResult<Int4JudgeRun>>();
    send::<HostedInt4TeamResult<Int4SentimentRun>>();
    send::<HostedInt4TeamResult<Int4SourceTaskRun>>();
    fn old_chat_type(value: HostedInt4ChatTeamResult) -> HostedInt4TeamResult<Int4ChatResult> { value }
    let _compatible = old_chat_type;
    let _classify = NlpEngine::execute_int4_classify_with_threads;
    let _extract = NlpEngine::execute_int4_extract_with_threads;
    let _judge = NlpEngine::execute_int4_judge_with_threads;
    let _sentiment = NlpEngine::execute_int4_sentiment_with_threads;
    let _source = NlpEngine::execute_int4_source_task_with_threads;
}

#[test]
fn task_team_quote_matches_the_physical_dispatcher_for_every_supported_geometry() {
    for threads in [2, 3, 7, 32, 64] {
        for group_width in [1, 3, 32, 128, 10752] {
            assert_eq!(int4_task_team_workspace_bytes(threads, group_width).unwrap(),
                dispatch::int4_team_workspace_bytes(threads, group_width).unwrap());
            assert_eq!(int4_task_team_workspace_bytes(threads, group_width).unwrap(),
                int4_chat_team_workspace_bytes(threads, group_width).unwrap());
        }
    }
    for (threads, group_width) in [(0, 128), (1, 128), (65, 128),
        (usize::MAX, 128), (2, 0), (2, 10753), (2, usize::MAX)] {
        assert!(int4_task_team_workspace_bytes(threads, group_width).is_err());
    }
}

mod runtime;
