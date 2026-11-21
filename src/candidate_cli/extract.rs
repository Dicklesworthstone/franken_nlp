//! Exact user-schema extraction, not a generated-JSON parse/retry shortcut.
use super::*;
use crate::{
    batch::extract::{ExtractionBatchArgs, ExtractionBatchGrounding},
    grammar::{CompileLimits, runtime::JsonProgram},
};
use source::SourceHostArgs;
pub(super) mod selected;

pub(super) const SCHEMA_BYTES: usize = 64 * 1024;

#[derive(Args)]
pub(crate) struct ExtractCommand {
    /// Exact UTF-8 source document; '-' or no path reads stdin.
    #[arg(default_value = "-")]
    pub(super) input: PathBuf,
    #[command(flatten)]
    pub(super) host: SourceHostArgs,
    #[command(flatten)]
    pub(super) head: selected::SelectedRowsArgs,
    /// Bounded local JSON schema file, never a URL or stdin. Exact numbers are retained.
    #[arg(long, value_name = "FILE")]
    pub(super) schema: PathBuf,
    /// Enforce x-fnlp-source=verbatim annotations against this exact document.
    /// Requires at least one annotated field; this is not semantic verification.
    #[arg(long)]
    pub(super) source_membership: bool,
    #[arg(skip)]
    pub(super) encoding: input_encoding::InputEncodingArgs,
}

pub(super) fn definition() -> clap::Command {
    input_encoding::definition(ExtractCommand::augment_args(clap::Command::new("extract")
        .about("Extract using an exact local JSON schema and constrained INT8 decoding")
        .long_about("Compile a bounded user schema and exact source before loading weights. The supported schema subset is the existing grammar compiler's; unsupported keywords fail closed. Schema validity does not prove extracted facts. Use --source-membership for explicit verbatim source fields. --selected-rows explicitly projects the complete legal grammar set instead of the full vocabulary, with a per-step row ceiling that refuses rather than prunes. EOS remains a scored choice; no speed or parity qualification is implied. No sampling, repair retry, remote schema resolution or semantic-verification claim is available.")))
        .after_long_help("--input-encoding safe-bpe-v1 explicitly compacts the exact schema and source while excluding template-control tokens. Both segments share ONE tokenizer allowance, including pair checks, examined pair bytes and actual untrusted tokens. The aggregate input ceiling is max-input-bytes plus the 64-KiB schema allowance, capped at 1 MiB; the separate schema and document byte ceilings still apply. Workspace for the complete allowance is admitted before either file or model IO. The policy binds execution identity and is reported on compact results. Exhaustion refuses without fallback. Original schema numbers, source bytes and citation offsets remain exact. Byte-fallback is the default. Compact input changes prompt tokens and makes no identical-output, quality or measured-speed claim.")
}
impl ExtractCommand {
    pub(super) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        let mut command = Self::from_arg_matches(matches)?;
        command.encoding = input_encoding::InputEncodingArgs::from_matches(matches)?;
        Ok(command)
    }
    pub(super) fn validate(&self) -> Result<(CandidateArgs, Limits), CandidateError> {
        self.head.limits()?;
        check_schema_path(&self.schema)?;
        let (args, limits) = self.host.common_with_encoding(self.input.clone(), &self.encoding)?;
        validate_preparation(&self.host, &self.encoding_policy(limits)?, limits)?;
        Ok((args, limits))
    }
    pub(super) fn encoding_policy(&self, limits: Limits)
        -> Result<crate::tokenizer::untrusted::UntrustedEncoding, CandidateError> {
        encoding_policy(&self.host, &self.encoding, limits)
    }
    pub(super) fn execute(self, input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
        let (args, limits) = self.validate()?;
        #[cfg(feature = "asupersync-runtime")]
        { runtime::extraction::execute(self, args, limits, input, output) }
        #[cfg(not(feature = "asupersync-runtime"))]
        { let _ = (self, args, limits, input, output); Err(CandidateError::Unavailable) }
    }
}

/// The same complete schema/source allowance applies to a single request,
/// each corpus record, and the per-attempt bound inside an owned job.
pub(super) fn encoding_policy(host: &SourceHostArgs, encoding: &input_encoding::InputEncodingArgs,
    limits: Limits) -> Result<crate::tokenizer::untrusted::UntrustedEncoding, CandidateError> {
    let bytes = host.max_input_bytes.checked_add(SCHEMA_BYTES).ok_or(CandidateError::Arguments)?
        .min(crate::tokenizer::untrusted::SafeBpeLimits::MAX_INPUT_BYTES);
    encoding.policy(bytes, limits.max_prompt_tokens)
}

pub(super) fn validate_preparation(host: &SourceHostArgs,
    encoding: &crate::tokenizer::untrusted::UntrustedEncoding, limits: Limits) -> Result<(), CandidateError> {
    let encoder_bytes = match encoding {
        crate::tokenizer::untrusted::UntrustedEncoding::ByteFallback => 0,
        crate::tokenizer::untrusted::UntrustedEncoding::SafeBpeV1(policy) => policy.requirements()
            .map_err(|_| CandidateError::Arguments)?.workspace_bytes,
    };
    // The source host prices a document-only encoder. The complete replacement
    // quote also retains the exact original schema beside source, grammar and
    // output allocations, before opening any private file or stream.
    let floor = (256 * MIB).checked_add((host.max_input_bytes as u64).checked_mul(32)
        .ok_or(CandidateError::Arguments)?)
        .and_then(|n| n.checked_add(SCHEMA_BYTES as u64 * 32))
        .and_then(|n| n.checked_add((host.max_result_bytes as u64).checked_mul(16)?))
        .and_then(|n| n.checked_add(encoder_bytes)).ok_or(CandidateError::Arguments)?;
    if limits.preparation_bytes < floor { return Err(CandidateError::Arguments); }
    Ok(())
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
