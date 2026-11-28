//! Typed completion failures and the independently compared CLI/host quote.
use super::*;
use crate::tasks::sentiment::sampled::long::SampledSentimentMapError as SharedMapError;

fn admitted() -> (SampledSentimentMapCommand, MapLimits) {
    let matches = crate::candidate_cli::definition().try_get_matches_from([
        "candidate", "sentiment-sampled-map", "--model", "never-opened.fnlpq", "--memory-mib", "65536",
        "--seed", "4545454545454545454545454545454545454545454545454545454545454545",
        "--compute-threads", "4", "--prefill-rows", "7", "--input-encoding", "safe-bpe-v1",
    ]).unwrap();
    let CandidateCommand::SampledSentimentMap(command) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("sampled document required"); };
    let limits = command.validate().unwrap();
    (command, limits)
}

#[test]
fn complete_document_quote_cannot_drop_encoder_prompt_or_output_ownership() {
    let (command, limits) = admitted();
    let host = config(&command, &limits).unwrap();
    check_host_quote(&limits, &host).unwrap();
    let mut no_prefill = host.clone();
    no_prefill.prefill = None;
    assert_eq!(check_host_quote(&limits, &no_prefill).unwrap_err().exit, ErrorCode::Usage);
    for case in 0..5 {
        let (_, mut wrong) = admitted();
        match case {
            0 => wrong.memory.input_bytes -= 1,
            1 => wrong.memory.workspace_bytes -= 1,
            2 => wrong.memory.output_reservation_bytes -= 1,
            3 => wrong.memory.max_output_tokens -= 1,
            _ => wrong.minimum_memory_bytes = wrong.shared.memory_bytes + 1,
        }
        assert_eq!(check_host_quote(&wrong, &host).unwrap_err().exit, ErrorCode::Usage);
    }
}

#[test]
fn selected_document_configuration_preserves_the_full_pure_quote() {
    let (mut command, limits) = admitted();
    let dense = config(&command, &limits).unwrap();
    command.sampled.max_selected_rows = Some(4096);
    let selected = config(&command, &limits).unwrap();
    assert_eq!(selected.selected_rows, command.sampled.selected_rows().unwrap());
    check_host_quote(&limits, &selected).unwrap();
    assert_eq!(dense.requirements_with_encoding(limits.task, 4, &limits.encoding).unwrap(),
        selected.requirements_with_encoding(limits.task, 4, &limits.encoding).unwrap());
}

#[test]
fn every_cancellation_kind_survives_map_and_shared_reducer_boundaries() {
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast,
        RaceLost, ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        for error in [Int8SampledSentimentMapError::Cancelled(cause),
            Int8SampledSentimentMapError::Shared(SharedMapError::Cancelled(cause)),
            Int8SampledSentimentMapError::Sampled(Int8SampledSentimentError::Shared(
                SampledSentimentError::Cancelled(cause)))] {
            let failure = map_host_failure(HostedError::SampledSentimentMap(error));
            assert_eq!(failure.cancellation, Some(cause));
            assert_eq!(failure.exit, cancelled(cause).exit);
        }
    }
}

#[test]
fn tokenizer_output_and_model_allowances_remain_typed_no_result_failures() {
    for error in [Int8SampledSentimentMapError::Output,
        Int8SampledSentimentMapError::Shared(SharedMapError::Output),
        Int8SampledSentimentMapError::Chunk(MapReduceError::TokenizerBudget)] {
        assert_eq!(map_failure(error).exit, ErrorCode::BudgetOrTimeout);
    }
    for error in [Int8SampledSentimentMapError::Work,
        Int8SampledSentimentMapError::Shared(SharedMapError::Work),
        Int8SampledSentimentMapError::Chunk(MapReduceError::AllocationRefused)] {
        assert_eq!(map_failure(error).exit, ErrorCode::AdmissionOrResourceLimit);
    }
    assert_eq!(map_failure(Int8SampledSentimentMapError::Sampled(
        Int8SampledSentimentError::Shared(SampledSentimentError::TokenizerWork))).exit,
        ErrorCode::BudgetOrTimeout);
}

#[test]
fn trusted_scaffold_reservation_tightens_chunk_limits_without_refusing_valid_completion() {
    let (_, admitted) = admitted();
    let declared = admitted.mapping.chunks;
    let mut effective = declared;
    effective.reserved_tokens += 96;
    effective.max_chunk_tokens -= 96;
    assert!(partition_within_declared(effective, declared));
    for case in 0..7 {
        let mut widened = effective;
        match case {
            0 => widened.max_input_bytes += 1,
            1 => widened.max_chunk_bytes += 1,
            2 => widened.context_tokens += 1,
            3 => widened.max_chunks += 1,
            4 => widened.max_tokenizer_calls += 1,
            5 => widened.max_chunk_tokens = declared.max_chunk_tokens + 1,
            _ => widened.reserved_tokens = declared.reserved_tokens - 1,
        }
        assert!(!partition_within_declared(widened, declared), "case {case}");
    }
    effective.reserved_tokens = effective.context_tokens;
    assert!(!partition_within_declared(effective, declared));
}
