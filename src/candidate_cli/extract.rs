//! Exact user-schema extraction, not a generated-JSON parse/retry shortcut.
use super::*;
use crate::{
    batch::extract::{ExtractionBatchArgs, ExtractionBatchGrounding},
    grammar::{CompileLimits, runtime::JsonProgram},
};
use source::SourceHostArgs;

pub(super) const SCHEMA_BYTES: usize = 64 * 1024;

#[derive(Args)]
pub(crate) struct ExtractCommand {
    /// Exact UTF-8 source document; '-' or no path reads stdin.
    #[arg(default_value = "-")]
    pub(super) input: PathBuf,
    #[command(flatten)]
    pub(super) host: SourceHostArgs,
    /// Bounded local JSON schema file, never a URL or stdin. Exact numbers are retained.
    #[arg(long, value_name = "FILE")]
    pub(super) schema: PathBuf,
    /// Enforce x-fnlp-source=verbatim annotations against this exact document.
    /// Requires at least one annotated field; this is not semantic verification.
    #[arg(long)]
    pub(super) source_membership: bool,
}

pub(super) fn definition() -> clap::Command {
    ExtractCommand::augment_args(clap::Command::new("extract")
        .about("Extract using an exact local JSON schema and constrained INT8 decoding")
        .long_about("Compile a bounded user schema and exact source before loading weights. The supported schema subset is the existing grammar compiler's; unsupported keywords fail closed. Schema validity does not prove extracted facts. Use --source-membership for explicit verbatim source fields. No sampling, repair retry, remote schema resolution or semantic-verification claim is available."))
}
impl ExtractCommand {
    pub(super) fn validate(&self) -> Result<(CandidateArgs, Limits), CandidateError> {
        check_schema_path(&self.schema)?;
        self.host.common(self.input.clone())
    }
    pub(super) fn execute(self, input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
        let (args, limits) = self.validate()?;
        #[cfg(feature = "asupersync-runtime")]
        { runtime::extraction::execute(self, args, limits, input, output) }
        #[cfg(not(feature = "asupersync-runtime"))]
        { let _ = (self, args, limits, input, output); Err(CandidateError::Unavailable) }
    }
}

pub(super) fn check_schema_path(path: &std::path::Path) -> Result<(), CandidateError> {
    if path.as_os_str().is_empty() || path.as_os_str() == "-" { Err(CandidateError::Arguments) } else { Ok(()) }
}
pub(super) fn compiler(host: &SourceHostArgs) -> CompileLimits {
    CompileLimits { max_schema_bytes: SCHEMA_BYTES, ..host.planning().compiler }
}
pub(super) fn arguments(schema: String, source_membership: bool, budget: crate::tasks::ir::TaskBudget)
    -> Result<ExtractionBatchArgs, CandidateError> {
    if schema.is_empty() || schema.len() > SCHEMA_BYTES { return Err(CandidateError::Input); }
    Ok(ExtractionBatchArgs { schema, budget,
        grounding: if source_membership { ExtractionBatchGrounding::SourceMembership }
            else { ExtractionBatchGrounding::Structural } })
}

/// Exercise the actual exact-number runtime with the actual document before
/// accessing model metadata. Drop this inspection program before tokenization;
/// the shared planner separately compiles and seals the executable identity.
/// Never parse the schema via serde_json::Value, normalize or stringify it.
pub(super) fn check_schema(args: &ExtractionBatchArgs, document: &str, host: &SourceHostArgs)
    -> Result<(), CandidateError> {
    if args.schema.is_empty() || args.schema.len() > SCHEMA_BYTES || document.len() > host.max_input_bytes {
        return Err(CandidateError::Input);
    }
    let program = match args.grounding {
        ExtractionBatchGrounding::Structural => JsonProgram::compile(&args.schema, compiler(host)),
        ExtractionBatchGrounding::SourceMembership => JsonProgram::compile_with_source(
            &args.schema, document, compiler(host), host.planning().source),
    }.map_err(|_| CandidateError::Planning)?;
    if (args.grounding == ExtractionBatchGrounding::SourceMembership) != program.requires_source() {
        return Err(CandidateError::Planning);
    }
    Ok(())
}

#[cfg(test)] pub(super) mod tests;
