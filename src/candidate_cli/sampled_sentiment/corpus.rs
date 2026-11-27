//! Ordered, single-context sampled sentiment records with one fixed sampler.
use super::*;
use crate::{
    batch::{BatchLimits, BatchWork, sentiment::sampled::sampled_sentiment_batch_preparation_bytes},
    candidate_cli::batch::{FRAME_ALLOWANCE, IO_BUFFER_BYTES},
};

#[derive(Args)]
struct StreamArgs {
    /// Whole-stream nonempty records, including malformed input and flush controls.
    #[arg(long, default_value_t = 1000)]
    max_requests: u64,
    #[arg(long, default_value_t = 64)]
    max_input_mib: u64,
    /// Entire framed output, including errors, flushes and the final record.
    #[arg(long, default_value_t = 64)]
    max_output_mib: u64,
    #[arg(long, default_value_t = 1_048_576)]
    max_line_bytes: usize,
}

pub(in crate::candidate_cli) struct Command {
    sampled: SampledSentimentCommand,
    stream: StreamArgs,
}
pub(in crate::candidate_cli) fn definition() -> clap::Command {
    StreamArgs::augment_args(super::definition().name("sentiment-sampled-batch")
        .about("Generate independent sampled sentiment bundles from ordered NDJSON on one resident INT8 model")
        .long_about("Read {id,text,task_args?} records and {\"flush\":true} controls. Omitted axes select all four dimensions; an optional complete task_args.budget may tighten the per-record ceiling. Each item returns every requested score and explanation. Stable IDs address the draws independently of record order and flush epochs. Native work, mask visits, execution checkpoints and the deadline cover the stream. Safe-BPE limits apply per record. --compute-threads reuses one persistent INT8 row team across all records, axes and samples, together with optional bounded --prefill-rows prompts. Final success is published only after all workers physically join. A failed item makes the process exit nonzero. Results describe generated-score variation and contain unverified model explanations.")
        .mut_arg("input", |arg| arg.help("NDJSON file, or '-' for stdin"))
        .mut_arg("max_mask_node_visits", |arg| arg.help("Nonrenewable mask visits across all records and samples")))
}

pub(in crate::candidate_cli) struct Envelope {
    pub transport: BatchLimits,
    pub output_bytes: u64,
    pub io_bytes: u64,
}
impl Command {
    pub(in crate::candidate_cli) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        Ok(Self { sampled: SampledSentimentCommand::from_matches(matches)?,
            stream: StreamArgs::from_arg_matches(matches)? })
    }
    fn admit(&self) -> Result<(CandidateArgs, Limits, Envelope), CandidateError> {
        let s = &self.stream;
        if !(1..=100_000).contains(&s.max_requests)
            || !(1..=16_384).contains(&s.max_input_mib) || !(1..=16_384).contains(&s.max_output_mib)
            || !(1..=4 * 1024 * 1024).contains(&s.max_line_bytes) || s.max_line_bytes < self.sampled.max_input_bytes {
            return Err(CandidateError::Arguments);
        }
        let line = self.sampled.max_result_bytes.checked_add(4096).ok_or(CandidateError::Arguments)?;
        let output_bytes = s.max_output_mib * MIB;
        let framing = (s.max_requests + 4) * FRAME_ALLOWANCE;
        let work = self.sampled.work_ceiling();
        let transport = BatchLimits { max_line_bytes: s.max_line_bytes, max_document_bytes: self.sampled.max_input_bytes,
            max_id_bytes: 128, max_epoch_ids: 4096, max_epoch_id_bytes: 256 * 1024, max_json_depth: 16,
            max_input_bytes: s.max_input_mib * MIB, max_requests: s.max_requests, max_output_line_bytes: line,
            max_output_bytes: output_bytes.checked_sub(framing).ok_or(CandidateError::Arguments)?,
            max_work: BatchWork { forward_positions: work.forward_positions, projected_logits: work.projected_logits } };
        transport.validate().map_err(|_| CandidateError::Arguments)?;
        let io_bytes = (line as u64).checked_add(FRAME_ALLOWANCE + 2 * IO_BUFFER_BYTES as u64)
            .ok_or(CandidateError::Arguments)?;
        // CorpusLimits stages parser, output, ID bookkeeping and external IO
        // simultaneously with the one-shot request peak. Refuse before any IO.
        let staging = (line as u64).checked_mul(4)
            .and_then(|n| n.checked_add(8 * s.max_line_bytes as u64 + 128 * transport.max_epoch_ids as u64
                + transport.max_epoch_id_bytes as u64))
            .and_then(|n| n.checked_add(io_bytes)).ok_or(CandidateError::Arguments)?;
        let (common, limits) = self.sampled.common_with_staging(staging)?;
        let budget = self.sampled.budget(limits, SentimentAxis::ALL.len())?;
        let encoding = common.input_encoding_policy(limits)?;
        let preparation = sampled_sentiment_batch_preparation_bytes(budget, self.sampled.task_limits(), &encoding)
            .map_err(|_| CandidateError::Arguments)?;
        if preparation > limits.preparation_bytes { return Err(CandidateError::Arguments); }
        Ok((common, limits, Envelope { transport, output_bytes, io_bytes }))
    }

    pub(in crate::candidate_cli) fn run_owned<R: Read + Send + 'static, W: Write + Send + 'static>(
        self, input: R, output: W, diagnostics: &mut impl Write) -> ExitCode {
        let result = self.admit().map_err(|_| Failure::usage("int8_sampled_sentiment_batch_options"))
            .and_then(|(common, limits, envelope)| {
                #[cfg(feature = "asupersync-runtime")]
                { runtime::corpus::execute(self.sampled, common, limits, envelope, input, output) }
                #[cfg(not(feature = "asupersync-runtime"))]
                { let _ = (common, limits, envelope, input, output);
                    Err(Failure::usage("int8_sampled_sentiment_batch_profile_unavailable")) }
            });
        match result {
            Ok(()) => ExitCode::SUCCESS,
            Err(failure) => report_failure(failure, diagnostics),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn parse(extra: &[&str]) -> Command {
        let seed = "0".repeat(64);
        let mut argv = vec!["sentiment-sampled-batch", "--model", "absent.fnlpq",
            "--memory-mib", "16384", "--seed", &seed];
        argv.extend_from_slice(extra);
        Command::from_matches(&definition().try_get_matches_from(argv).unwrap()).unwrap()
    }
    #[test]
    fn native_and_transport_caps_are_separate_and_framing_is_priced() {
        let (_, _, envelope) = parse(&["--max-requests", "30", "--max-output-mib", "8"]).admit().unwrap();
        assert_eq!(envelope.transport.max_requests, 30);
        assert_eq!(envelope.transport.max_output_bytes + 34 * FRAME_ALLOWANCE, 8 * MIB);
        assert_eq!(envelope.transport.max_work.forward_positions, 131_072);
        assert!(parse(&["--max-requests", "300", "--max-output-mib", "1"]).admit().is_err());
    }
    #[test]
    fn one_team_and_prompt_schedule_leave_stream_work_and_draw_policy_unchanged() {
        let serial = parse(&[]);
        let (_, _, baseline) = serial.admit().unwrap();
        for width in ["2", "3", "64"] {
            let threaded = parse(&["--compute-threads", width, "--prefill-rows", "4"]);
            let (common, limits, admitted) = threaded.admit().unwrap();
            assert_eq!(limits.compute.threads, width.parse::<usize>().unwrap());
            assert_eq!(limits.compute, common.compute.admit_prefill(threaded.sampled.prefill().unwrap()).unwrap());
            assert!(limits.compute.workspace_bytes > 0);
            assert_eq!(threaded.sampled.options().unwrap(), serial.sampled.options().unwrap());
            assert_eq!(admitted.transport.max_work, baseline.transport.max_work);
            assert_eq!(admitted.transport.max_requests, baseline.transport.max_requests);
            assert_eq!(admitted.transport.max_output_bytes, baseline.transport.max_output_bytes);
        }
        for width in ["0", "65"] { assert!(parse(&["--compute-threads", width]).admit().is_err()); }
    }
}
