//! Pure resident quote checks, including the actual serial/team distinction.
use super::*;
use crate::candidate_cli::sampled_sentiment::map::corpus::{self as command, Command};

fn admitted(extra: &[&str]) -> (SampledSentimentMapCommand, MapLimits, Envelope) {
    let mut argv = vec!["sentiment-sampled-corpus", "--model", "never-opened.fnlpq",
        "--memory-mib", "65536",
        "--seed", "4545454545454545454545454545454545454545454545454545454545454545"];
    argv.extend_from_slice(extra);
    Command::from_matches(&command::definition().try_get_matches_from(argv).unwrap())
        .unwrap().admit().unwrap()
}

#[test]
fn independent_host_quote_covers_transport_encoder_prefill_and_every_worker() {
    for width in ["1", "2", "4", "64"] {
        for rows in [None, Some("4")] {
            for encoding in ["byte-fallback", "safe-bpe-v1"] {
                let mut args = vec!["--compute-threads", width, "--input-encoding", encoding];
                if let Some(rows) = rows { args.extend(["--prefill-rows", rows]); }
                let (command, limits, envelope) = admitted(&args);
                let host = configuration(&command, &limits, &envelope).unwrap();
                check_quote(&limits, &envelope, &host).unwrap();
                let base = host.requirements().unwrap();
                assert_eq!(base.input_bytes, limits.memory.input_bytes + envelope.transport_bytes);
                let extra = limits.prefill.map(|p| p.validate().unwrap()).unwrap_or(0);
                assert_eq!(base.workspace_bytes, limits.memory.workspace_bytes + extra);
                if width != "1" {
                    let team = host.requirements_with_threads(width.parse().unwrap()).unwrap();
                    assert_eq!(team.workspace_bytes, base.workspace_bytes + limits.shared.compute.workspace_bytes);
                    assert_eq!(team.input_bytes, base.input_bytes);
                    assert_eq!(team.output_reservation_bytes, base.output_reservation_bytes);
                }
            }
        }
    }
}

#[test]
fn mismatched_admission_refuses_dropped_transport_output_or_prompt_storage() {
    let args = ["--compute-threads", "4", "--prefill-rows", "4", "--input-encoding", "safe-bpe-v1"];
    for axis in 0..7 {
        let (command, mut limits, mut envelope) = admitted(&args);
        let host = configuration(&command, &limits, &envelope).unwrap();
        match axis {
            0 => envelope.transport_bytes -= 1,
            1 => limits.memory.input_bytes -= 1,
            2 => limits.memory.workspace_bytes -= 1,
            3 => limits.memory.output_reservation_bytes -= 1,
            4 => limits.memory.max_output_tokens -= 1,
            5 => limits.memory.max_output_bytes -= 1,
            _ => limits.minimum_memory_bytes = limits.shared.memory_bytes + 1,
        }
        assert_eq!(check_quote(&limits, &envelope, &host).unwrap_err().exit,
            ErrorCode::Usage, "axis {axis}");
    }
    let (command, limits, envelope) = admitted(&args);
    let mut host = configuration(&command, &limits, &envelope).unwrap();
    host.document.prefill = None;
    assert_eq!(check_quote(&limits, &envelope, &host).unwrap_err().exit, ErrorCode::Usage);
}

#[test]
fn item_caps_and_population_allowance_are_not_interchanged_by_host_configuration() {
    let (command, limits, envelope) = admitted(&[
        "--max-document-forward-positions", "32768",
        "--max-document-mask-node-visits", "10000000"]);
    let host = configuration(&command, &limits, &envelope).unwrap();
    check_quote(&limits, &envelope, &host).unwrap();
    assert_eq!(host.document.mapping.max_model_work.forward_positions, 32768);
    assert_eq!(host.max_model_work.forward_positions, 1_048_576);
    assert_eq!(host.document.max_mask_node_visits, 10_000_000);
    assert_eq!(host.max_mask_node_visits, 1_000_000_000_000);
    assert_eq!(host.transport.max_work.forward_positions, host.max_model_work.forward_positions);
    assert_eq!(host.task, limits.task);
    assert_eq!(host.encoding, limits.encoding);
}

#[test]
fn terminal_cancellation_remains_typed_through_population_error_classification() {
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast,
        RaceLost, ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let error = HostedError::BatchSetup(crate::batch::BatchFault::cancelled(cause));
        let failure = map_host_failure(error);
        assert_eq!(failure.cancellation, Some(cause));
        assert_eq!(failure.exit, cancelled(cause).exit);
    }
}
