//! Scheduling only: never inherit the free-text sampler's stop/bias policy.
use clap::Args;
use crate::{candidate_cli::CandidateError, native_engine::strict_int8::prefill::Int8PrefillLimits};

#[derive(Args, Default)]
pub(in crate::candidate_cli) struct SourcePrefillArgs {
    /// Opt into layer-major INT8 prompt processing, 1..=64 tokens per morsel.
    /// Omitted keeps sequential prefill. Composes with --selected-rows.
    /// Extra scratch is process-admitted; this is not document batching or a speed claim.
    #[arg(long, value_name = "ROWS")]
    prefill_rows: Option<usize>,
}
impl SourcePrefillArgs {
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
    use crate::candidate_cli::{definition, source::tests::command};
    use std::io::{self, Read};

    #[test]
    fn all_four_source_commands_have_serial_defaults_and_explicit_bounded_morsels() {
        for task in ["ner", "keyphrases", "summarize", "answer"] {
            assert!(command(task, &[]).args.prefill.limits().unwrap().is_none());
            for rows in ["1", "4", "64"] {
                let cmd = command(task, &["--prefill-rows", rows, "--selected-rows", "32"]);
                let prefill = cmd.args.prefill.limits().unwrap().unwrap();
                assert_eq!(prefill.max_batch_rows, rows.parse::<usize>().unwrap());
                assert_eq!(prefill.validate().unwrap(), prefill.max_extra_scratch_bytes);
                assert_eq!(cmd.args.head.limits().unwrap().unwrap().max_rows_per_step, 32);
                assert!(cmd.args.host.common(cmd.args.input.clone()).is_ok());
            }
        }
    }
    #[test]
    fn invalid_rows_refuse_before_reading_input_options_or_the_model() {
        struct NeverRead;
        impl Read for NeverRead {
            fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("prefill validation must precede input IO") }
        }
        for task in ["ner", "keyphrases", "summarize", "answer"] {
            for rows in ["0", "65", "1000000"] {
                let cmd = command(task, &["--prefill-rows", rows, "--options", "missing-options.json"]);
                let mut output = Vec::new();
                assert!(matches!(cmd.execute(&mut NeverRead, &mut output), Err(CandidateError::Arguments)));
                assert!(output.is_empty());
            }
        }
    }
    #[test]
    fn malformed_rows_never_select_a_default_schedule() {
        for rows in ["-1", "NaN", "4.0", "999999999999999999999999999999999999"] {
            assert!(definition().try_get_matches_from(["candidate", "ner", "--model", "missing.fnlpq",
                "--memory-mib", "8192", "--prefill-rows", rows]).is_err());
        }
    }
    #[test]
    fn cli_and_native_geometry_agree_at_every_boundary() {
        for rows in [0, 1, 4, 63, 64, 65, usize::MAX] {
            assert_eq!(SourcePrefillArgs { prefill_rows: Some(rows) }.limits().is_ok(),
                Int8PrefillLimits::required_extra_scratch_bytes(rows).is_ok());
        }
    }
}
