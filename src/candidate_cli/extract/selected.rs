//! Explicit structured-head choice shared ONLY by supported single-task routes.
use clap::Args;
use crate::{candidate_cli::CandidateError,
    native_engine::constrained_int8::sparse::Int8JsonSparseLimits,
    tasks::{extract::quantized::Int8ExtractPlan, source_planning::quantized::PreparedInt8SourceTask}};

#[derive(Args, Default)]
pub(in crate::candidate_cli) struct SelectedRowsArgs {
    /// Opt into grammar-first INT8 head projection, with 1..=166144 legal rows per step.
    /// All legal rows are scored; exceeding this cap refuses, never prunes or retries.
    /// Omitted keeps full-vocabulary decoding. This is not a measured speed claim.
    #[arg(long, value_name = "ROWS")]
    pub selected_rows: Option<usize>,
}
impl SelectedRowsArgs {
    pub fn limits(&self) -> Result<Option<Int8JsonSparseLimits>, CandidateError> {
        self.selected_rows.map(|max_rows_per_step| {
            let limits = Int8JsonSparseLimits { max_rows_per_step };
            limits.validate().map_err(|_| CandidateError::Arguments)?;
            Ok(limits)
        }).transpose()
    }
    /// Consume the original executable before model admission. Never reuse an
    /// admitted dense identity or route a selected plan through a dense driver.
    pub fn extraction(&self, plan: Int8ExtractPlan) -> Result<Int8ExtractPlan, CandidateError> {
        match self.limits()? {
            Some(limits) => plan.with_selected_rows(limits).map_err(|_| CandidateError::Planning),
            None => Ok(plan),
        }
    }
    pub fn source(&self, plan: PreparedInt8SourceTask) -> Result<PreparedInt8SourceTask, CandidateError> {
        match self.limits()? {
            Some(limits) => plan.with_selected_rows(limits).map_err(|_| CandidateError::Planning),
            None => Ok(plan),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate_cli::{self, CandidateCommand};
    use std::io::{self, Read};
    fn parsed(name: &str, cap: Option<&str>) -> Result<CandidateCommand, clap::Error> {
        let mut argv = vec!["candidate", name, "--model", "missing-local-model.fnlpq", "--memory-mib", "8192"];
        if name == "extract" { argv.extend(["--schema", "missing-schema.json"]); }
        if let Some(cap) = cap { argv.extend(["--selected-rows", cap]); }
        let matches = candidate_cli::definition().try_get_matches_from(argv)?;
        CandidateCommand::from_matches(&matches)
    }
    fn head(command: &CandidateCommand) -> &SelectedRowsArgs {
        match command { CandidateCommand::Extract(c) => &c.head, CandidateCommand::Source(c) => &c.args.head,
            _ => panic!("unexpected task") }
    }
    #[test]
    fn all_five_supported_commands_keep_dense_defaults_and_accept_explicit_legal_row_caps() {
        for name in ["extract", "ner", "keyphrases", "summarize", "answer"] {
            assert!(head(&parsed(name, None).unwrap()).limits().unwrap().is_none());
            for cap in ["1", "32", "1024", "166144"] {
                let command = parsed(name, Some(cap)).unwrap();
                assert_eq!(head(&command).limits().unwrap().unwrap().max_rows_per_step, cap.parse::<usize>().unwrap());
            }
        }
    }
    #[test]
    fn invalid_caps_are_rejected_before_any_schema_document_or_model_io() {
        struct NeverRead;
        impl Read for NeverRead { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("input must not be read") } }
        for name in ["extract", "ner", "keyphrases", "summarize", "answer"] {
            for cap in ["0", "166145"] {
                let command = parsed(name, Some(cap)).unwrap(); let mut output = Vec::new();
                let result = match command {
                    CandidateCommand::Extract(c) => c.execute(&mut NeverRead, &mut output),
                    CandidateCommand::Source(c) => c.execute(&mut NeverRead, &mut output),
                    _ => unreachable!(),
                };
                assert!(matches!(result, Err(CandidateError::Arguments))); assert!(output.is_empty());
            }
        }
    }
    #[test]
    fn malformed_row_counts_and_unsupported_text_routes_cannot_silently_select_a_mode() {
        for cap in ["-1", "not-a-count", "999999999999999999999999999999999999"] {
            assert!(parsed("extract", Some(cap)).is_err()); assert!(parsed("ner", Some(cap)).is_err());
        }
        for name in ["generate", "chat", "text-batch"] {
            let error = parsed(name, Some("32")).err().unwrap();
            assert_eq!(error.kind(), clap::error::ErrorKind::UnknownArgument);
        }
    }
    #[test]
    fn native_and_cli_geometry_checks_agree_on_both_boundaries() {
        for cap in [0, 1, 31, 32, 33, 166144, 166145, usize::MAX] {
            let head = SelectedRowsArgs { selected_rows: Some(cap) };
            assert_eq!(head.limits().is_ok(), Int8JsonSparseLimits { max_rows_per_step: cap }.validate().is_ok());
        }
    }
}
