//! Bounded native INT4 NER/rule union with fresh transformed-source verification.
use super::*;
use crate::{
    candidate_cli::redact::{self as shared, Action, Rule},
    tasks::redact::{RedactionRequest, actions::RedactionAction, pseudonym::PseudonymKey},
};

#[cfg(feature = "asupersync-runtime")]
mod runtime;
#[cfg(test)]
mod tests;

#[derive(Args, Clone, Eq, PartialEq)]
struct RedactionFlags {
    #[arg(long, value_enum, default_value = "mask")]
    action: Action,
    #[arg(long, value_enum, value_delimiter = ',', default_value = "email,phone,url,ip-address,credit-card")]
    rules: Vec<Rule>,
    /// Opt out of fresh model AND rule verification on the actual transformed text.
    #[arg(long)]
    no_verify: bool,
    /// Include sensitive original/output coordinate mappings in the final result.
    #[arg(long)]
    include_map: bool,
    /// Raw 32..4096-byte high-entropy key via inherited stdin; source must be a file.
    #[arg(long)]
    key_stdin: bool,
    #[arg(long)]
    key_id: Option<String>,
    #[arg(long)]
    namespace: Option<String>,
    /// Saved HMAC commitment, never a raw key; checked before document input.
    #[arg(long)]
    expected_key_commitment: Option<String>,
    #[arg(long, default_value_t = 4096)]
    max_detections: usize,
    /// Conservative work ceiling for EACH of at most two complete rule scans.
    #[arg(long, default_value_t = 128 * 1024 * 1024)]
    max_rule_work: u64,
    /// Separately charged rule/edit/coordinate workspace retained by the native host.
    #[arg(long, default_value_t = 64)]
    edit_reserve_mib: u64,
}
#[derive(Clone, Eq, PartialEq)]
pub(super) struct RedactArgs { source: source::SourceArgs, flags: RedactionFlags }
impl std::fmt::Debug for RedactArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str("RedactArgs") }
}
#[derive(Clone, Copy)]
pub(super) struct RedactLimits { pub(super) common: Int4Limits, edit_bytes: u64 }

pub(super) fn definition() -> clap::Command {
    RedactionFlags::augment_args(source::definition(crate::candidate_cli::source::Kind::Ner)
        .name("redact")
        .about("Redact with native grouped INT4 NER plus rules and fresh verification")
        .long_about("Redact one exact bounded UTF-8 document with the selected NER types plus deterministic rules. The complete original source must fit the declared context. By default the actual edited text is replanned and rerun through the same model AND rules; residual findings fail without publishing the rejected text. --no-verify explicitly skips that second pass. Actions are mask, placeholder or full-256-bit HMAC pseudonymize. Raw key bytes are accepted only through inherited stdin with a separate document file; an expected key commitment is checked before document input. No raw key argv option, chunked/corpus mode, retained job, automatic download or backend fallback. Source conversion and independently pinned snapshots share the same path. Both passes share native work, mask, memory and deadline admission. The result contains only final edited text and explicitly requested coordinate maps, never NER transcripts. Verification covers the declared detector union; omissions can remain and pseudonyms are not anonymization. Blocking IO observes the deadline cooperatively.")
        .mut_arg("input", |arg| arg.help("Exact bounded UTF-8 document; '-' reads stdin unless stdin supplies the private key"))
        .mut_arg("options", |arg| arg.long("ner-options").help("Complete bounded NerOptions JSON file; omitted selects person, organization, location"))
        .mut_arg("max_result_bytes", |arg| arg.help("Complete final redaction and per-pass NER result byte ceiling")))
}
impl RedactArgs {
    pub(super) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        Ok(Self { source: source::SourceArgs::from_matches(matches)?,
            flags: RedactionFlags::from_arg_matches(matches)? })
    }
    pub(super) fn validate(&self, args: &Int4Args) -> Result<RedactLimits, CandidateError> {
        let mut common = self.source.validate(args)?;
        if self.flags.rules.is_empty() || !(1..=16_384).contains(&self.flags.max_detections)
            || !(1..=1_000_000_000_000).contains(&self.flags.max_rule_work) {
            return Err(CandidateError::Arguments);
        }
        shared::validate_key_scope(self.flags.action, &args.input, self.flags.key_stdin,
            self.flags.key_id.as_deref(), self.flags.namespace.as_deref(),
            self.flags.expected_key_commitment.as_deref())?;
        let edit_bytes = mib(self.flags.edit_reserve_mib)?;
        let floor = sum(&[self.flags.max_detections as u64 * 1024,
            common.result_bytes as u64 * 16, MIB])?;
        if edit_bytes < floor || edit_bytes > common.memory_bytes { return Err(CandidateError::Arguments); }
        let task = self.source.budget(args, common);
        let temporary = sum(&[task.max_output_bytes * 4, u64::from(task.max_output_tokens) * 8, edit_bytes])?;
        // read_input reserves cap+1. The host separately captures this source,
        // a second preparation owner and bounded namespace/key identity data.
        // Its temporary result/token/edit charge overlaps full resident KV.
        common.minimum_memory_bytes = sum(&[common.minimum_memory_bytes, common.preparation_bytes,
            args.max_input_bytes as u64 + 1, 4096, temporary])?;
        if common.minimum_memory_bytes > common.memory_bytes { return Err(CandidateError::Arguments); }
        Ok(RedactLimits { common, edit_bytes })
    }
    fn request(&self, args: &Int4Args, limits: RedactLimits) -> RedactionRequest {
        let mut request = RedactionRequest::default();
        request.rules.enabled = self.flags.rules.iter().map(|rule| rule.kind()).collect();
        request.actions.default_action = match self.flags.action {
            Action::Mask => RedactionAction::Mask, Action::Placeholder => RedactionAction::Placeholder,
            Action::Pseudonymize => RedactionAction::Pseudonymize,
        };
        request.actions.include_map = self.flags.include_map;
        request.actions.expected_key_commitment = self.flags.expected_key_commitment.clone();
        // Only transformed text may use this larger byte envelope; original
        // input remains bounded independently before any native preparation.
        request.rule_budget.max_input_bytes = args.max_input_bytes.max(limits.common.result_bytes);
        request.rule_budget.max_detections = self.flags.max_detections;
        request.rule_budget.max_work = self.flags.max_rule_work;
        request.edit_budget.max_regions = self.flags.max_detections;
        request.edit_budget.max_output_bytes = limits.common.result_bytes;
        request.verify = !self.flags.no_verify;
        request
    }
    fn key(&self, input: &mut impl Read) -> Result<Option<(PseudonymKey, String)>, CandidateError> {
        shared::read_key(input, self.flags.key_stdin, self.flags.key_id.as_deref(),
            self.flags.namespace.as_deref(), self.flags.expected_key_commitment.as_deref())
    }
    pub(super) fn execute(self, args: Int4Args, input: &mut impl Read, output: &mut impl Write)
        -> Result<(), CandidateError> {
        let limits = self.validate(&args)?;
        #[cfg(feature = "asupersync-runtime")]
        { runtime::execute(args, self, limits, input, output) }
        #[cfg(not(feature = "asupersync-runtime"))]
        { let _ = (self, args, limits, input, output); Err(CandidateError::Unavailable) }
    }
}
