//! Prompt scheduling only, deliberately separate from shared corpus/job args.
//! No sampler, probability-space, candidate-language or identity override.
use clap::Args;
use crate::{candidate_cli::CandidateError, native_engine::strict_int8::prefill::Int8PrefillLimits};

#[derive(Args, Default)]
pub(in crate::candidate_cli) struct ScoringPrefillArgs {
    /// Opt into layer-major INT8 prompt processing, 1..=64 tokens per morsel.
    /// Each scoring head keeps its complete candidate language and work budget.
    /// Extra scratch is process-admitted; omitted keeps serial prompt execution.
    #[arg(long, value_name = "ROWS")]
    prefill_rows: Option<usize>,
}
impl ScoringPrefillArgs {
    pub(in crate::candidate_cli) fn limits(&self) -> Result<Option<Int8PrefillLimits>, CandidateError> {
        self.prefill_rows.map(|rows| {
            let bytes = Int8PrefillLimits::required_extra_scratch_bytes(rows)
                .map_err(|_| CandidateError::Arguments)?;
            Ok(Int8PrefillLimits { max_batch_rows: rows, max_extra_scratch_bytes: bytes })
        }).transpose()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate_cli::{definition, CandidateCommand, scored::ScoredArgs};
    use std::io::{self, Read};

    fn command(task: &str, extra: &[&str]) -> CandidateCommand {
        let mut argv = vec!["candidate", task, "--model", "missing.fnlpq", "--memory-mib", "8192"];
        argv.extend_from_slice(extra);
        CandidateCommand::from_matches(&definition().try_get_matches_from(argv).unwrap()).unwrap()
    }
    fn policy(command: &CandidateCommand) -> &ScoringPrefillArgs {
        match command {
            CandidateCommand::Scored(c) => &c.prefill,
            CandidateCommand::Judge(c) => &c.prefill,
            CandidateCommand::Resolve(c) => &c.prefill,
            _ => panic!("unexpected command"),
        }
    }
    #[test]
    fn every_scored_task_keeps_serial_defaults_and_exposes_exact_bounded_geometry() {
        for task in ["classify", "sentiment", "judge", "resolve"] {
            assert!(policy(&command(task, &[])).limits().unwrap().is_none());
            for width in ["1", "4", "64"] {
                let command = command(task, &["--prefill-rows", width]);
                let limits = policy(&command).limits().unwrap().unwrap();
                assert_eq!(limits.max_batch_rows, width.parse::<usize>().unwrap());
                assert_eq!(limits.validate().unwrap(), limits.max_extra_scratch_bytes);
            }
        }
    }
    #[test]
    fn invalid_geometry_refuses_before_input_or_model_io_for_every_scored_task() {
        struct NeverRead;
        impl Read for NeverRead {
            fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("validation must precede input IO") }
        }
        for task in ["classify", "sentiment", "judge", "resolve"] {
            for width in ["0", "65", "1000000"] {
                let command = command(task, &["--prefill-rows", width]); let mut output = Vec::new();
                let result = match command {
                    CandidateCommand::Scored(c) => c.execute(&mut NeverRead, &mut output),
                    CandidateCommand::Judge(c) => c.execute(&mut NeverRead, &mut output),
                    CandidateCommand::Resolve(c) => c.execute(&mut NeverRead, &mut output),
                    _ => unreachable!(),
                };
                assert!(matches!(result, Err(CandidateError::Arguments))); assert!(output.is_empty());
            }
        }
    }
    #[test]
    fn malformed_counts_never_choose_a_default_schedule() {
        for task in ["classify", "sentiment", "judge", "resolve"] {
            for count in ["-1", "NaN", "4.0", "999999999999999999999999999999999999"] {
                assert!(definition().try_get_matches_from(["candidate", task, "--model", "missing.fnlpq",
                    "--memory-mib", "8192", "--prefill-rows", count]).is_err());
            }
        }
    }
    #[test]
    fn scheduling_does_not_change_shared_work_context_or_task_budgets() {
        for task in ["classify", "sentiment"] {
            let CandidateCommand::Scored(serial) = command(task, &[]) else { unreachable!() };
            let CandidateCommand::Scored(grouped) = command(task, &["--prefill-rows", "4"]) else { unreachable!() };
            let (_, a) = serial.args.common().unwrap(); let (_, b) = grouped.args.common().unwrap();
            assert_eq!(serial.args.work_ceiling(), grouped.args.work_ceiling());
            assert_eq!(crate::canonjson::canonical_bytes(&serial.args.budget(a)).unwrap(),
                crate::canonjson::canonical_bytes(&grouped.args.budget(b)).unwrap());
            assert_eq!(serial.args.context_tokens, grouped.args.context_tokens);
        }
    }
    #[test]
    fn shared_batch_and_job_arguments_do_not_silently_inherit_scheduling() {
        let error = ScoredArgs::augment_args(clap::Command::new("shared"))
            .try_get_matches_from(["shared", "--model", "missing.fnlpq", "--memory-mib", "8192",
                "--prefill-rows", "4"]).unwrap_err();
        assert_eq!(error.kind(), clap::error::ErrorKind::UnknownArgument);
    }
    #[test]
    fn automatic_entity_discovery_cannot_ignore_an_unsupported_override() {
        let error = definition().try_get_matches_from(["candidate", "resolve", "--model", "missing.fnlpq",
            "--memory-mib", "8192", "--prefill-rows", "4", "--discover-entities"]).unwrap_err();
        assert_eq!(error.kind(), clap::error::ErrorKind::ArgumentConflict);
    }
    #[test]
    fn scored_scheduling_never_exposes_free_generation_controls() {
        for task in ["classify", "sentiment", "judge", "resolve"] {
            for switch in ["--stop", "--logit-bias", "--temperature-milli", "--seed"] {
                assert!(definition().try_get_matches_from(["candidate", task, "--model", "missing.fnlpq",
                    "--memory-mib", "8192", "--prefill-rows", "4", switch, "1"]).is_err());
            }
        }
    }
}
