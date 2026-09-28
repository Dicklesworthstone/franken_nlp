//! Model-backed detector-union redaction. Only completed edited text is exported.
use super::*;
pub(super) mod long;
pub(super) mod corpus;
use clap::ValueEnum;
use crate::tasks::{ner::NerOptions, redact::{PiiKind, RedactionRequest,
    actions::RedactionAction, pseudonym::PseudonymKey}};
use source::SourceHostArgs;

#[derive(Clone, Copy, Eq, PartialEq, ValueEnum)]
pub(super) enum Action { Mask, Placeholder, Pseudonymize }
#[derive(Clone, Copy, ValueEnum)]
enum Rule { Email, Phone, Url, IpAddress, CreditCard, Date }
impl Rule {
    fn kind(self) -> PiiKind {
        match self { Self::Email => PiiKind::Email, Self::Phone => PiiKind::Phone,
            Self::Url => PiiKind::Url, Self::IpAddress => PiiKind::IpAddress,
            Self::CreditCard => PiiKind::CreditCard, Self::Date => PiiKind::Date }
    }
}

#[derive(Args)]
pub(crate) struct RedactCommand {
    /// Exact UTF-8 source, or {id,text,task_args?} records with --ndjson.
    /// '-' reads stdin unless stdin is the private key source.
    #[arg(default_value = "-")]
    pub(super) input: PathBuf,
    #[command(flatten)]
    pub(super) host: SourceHostArgs,
    #[command(flatten)]
    pub(super) long: long::LongArgs,
    #[command(flatten)]
    pub(super) corpus: corpus::CorpusArgs,
    /// Mask never expands the detected source; placeholders/pseudonyms may expand it.
    #[arg(long, value_enum, default_value = "mask")]
    action: Action,
    #[arg(long, value_enum, value_delimiter = ',', default_value = "email,phone,url,ip-address,credit-card")]
    rules: Vec<Rule>,
    /// Complete typed NerOptions JSON; omitted selects person, organization, location.
    #[arg(long, value_name = "FILE")]
    pub(super) ner_options: Option<PathBuf>,
    /// Opt out of fresh model AND rule verification on the transformed document.
    #[arg(long)]
    no_verify: bool,
    /// Include sensitive original/output coordinate mappings in the result, not logs.
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
    /// Conservative work ceiling for EACH of at most two rule scans per document.
    #[arg(long, default_value_t = 128 * 1024 * 1024)]
    max_rule_work: u64,
    /// Separately modeled rule/edit/coordinate memory retained by the native host.
    #[arg(long, default_value_t = 64)]
    pub(super) edit_reserve_mib: u64,
}

pub(super) fn definition() -> clap::Command {
    RedactCommand::augment_args(clap::Command::new("redact")
        .about("Redact with native source-bound NER plus rules; verify the edited text afresh")
        .after_help("Verification covers only the selected detector union, not all PII. Model omissions and Unicode obfuscations remain possible. Pseudonyms are not anonymization. By default this candidate command emits one completed JSON object, never intermediate NER text. Pseudonymization always uses full 256-bit HMAC; raw key bytes have no argv option. Add --chunked for long documents: rules scan whole text while NER uses source-aligned chunks, then verification re-chunks the actual edited text. NER chunk boundaries may split entities; per-document work ceilings never renew per chunk. Add --ndjson for an owned corpus stream with one resident engine, fixed policy/key scope and independent whole-corpus ceilings. Completed document events remain valid if later records fail; any failed record yields a nonzero exit. Records cannot override actions, keys, types, chunking or verification. The independent fnlp redact --rules-only command remains model-free."))
}
impl RedactCommand {
    pub(super) fn validate(&self) -> Result<(CandidateArgs, Limits), CandidateError> {
        let (common, limits) = self.host.common(self.input.clone())?;
        self.long.validate(&self.host, limits)?;
        if self.rules.is_empty() || !(1..=16_384).contains(&self.max_detections)
            || self.max_rule_work == 0 || self.max_rule_work > 1_000_000_000_000
            || self.ner_options.as_ref().is_some_and(|p| p.as_os_str().is_empty() || p.as_os_str() == "-") {
            return Err(CandidateError::Arguments);
        }
        let edit = self.edit_reserve_mib.checked_mul(MIB).ok_or(CandidateError::Arguments)?;
        let floor = (self.max_detections as u64).checked_mul(1024)
            .and_then(|n| n.checked_add(self.host.max_result_bytes as u64 * 16))
            .and_then(|n| n.checked_add(MIB)).ok_or(CandidateError::Arguments)?;
        if edit < floor || edit > limits.memory_bytes { return Err(CandidateError::Arguments); }
        if self.action == Action::Pseudonymize {
            if !self.key_stdin || self.input.as_os_str() == "-"
                || self.key_id.as_ref().is_none_or(|id| id.is_empty() || id.len() > 128
                    || !id.bytes().all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b)))
                || self.namespace.as_ref().is_none_or(|s| s.is_empty() || s.len() > 256) {
                return Err(CandidateError::Arguments);
            }
        } else if self.key_stdin || self.key_id.is_some() || self.namespace.is_some() || self.expected_key_commitment.is_some() {
            return Err(CandidateError::Arguments);
        }
        if self.expected_key_commitment.as_ref().is_some_and(|s| s.len() != 64
            || !s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))) {
            return Err(CandidateError::Arguments);
        }
        self.corpus.validate(self, limits)?;
        Ok((common, limits))
    }
    pub(super) fn request(&self) -> RedactionRequest {
        let mut request = RedactionRequest::default();
        request.rules.enabled = self.rules.iter().map(|r| r.kind()).collect();
        request.actions.default_action = match self.action {
            Action::Mask => RedactionAction::Mask, Action::Placeholder => RedactionAction::Placeholder,
            Action::Pseudonymize => RedactionAction::Pseudonymize,
        };
        request.actions.include_map = self.include_map;
        request.actions.expected_key_commitment = self.expected_key_commitment.clone();
        // Verification scans actual transformed bytes, which may be larger than
        // the original. This does not increase original input admission.
        request.rule_budget.max_input_bytes = self.host.max_input_bytes.max(self.host.max_result_bytes);
        request.rule_budget.max_detections = self.max_detections;
        request.rule_budget.max_work = self.max_rule_work;
        request.edit_budget.max_regions = self.max_detections;
        request.edit_budget.max_output_bytes = self.host.max_result_bytes;
        request.verify = !self.no_verify;
        request
    }
    pub(super) fn key(&self, input: &mut impl Read) -> Result<Option<(PseudonymKey, String)>, CandidateError> {
        if !self.key_stdin { return Ok(None); }
        let id = self.key_id.as_deref().ok_or(CandidateError::Arguments)?;
        let key = PseudonymKey::from_reader(input, id).map_err(|_| CandidateError::Input)?;
        if let Some(expected) = &self.expected_key_commitment {
            key.require_commitment(expected).map_err(|_| CandidateError::Identity)?;
        }
        Ok(Some((key, self.namespace.clone().ok_or(CandidateError::Arguments)?)))
    }
    pub(super) fn execute(self, input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
        // Owned corpus IO must enter through run_stdio/run_owned, never a
        // borrowed handle, collecting adapter or detached forwarding thread.
        if self.corpus.ndjson { return Err(CandidateError::Arguments); }
        let (common, limits) = self.validate()?;
        #[cfg(feature = "asupersync-runtime")]
        { runtime::redaction::execute(self, common, limits, input, output) }
        #[cfg(not(feature = "asupersync-runtime"))]
        { let _ = (self, common, limits, input, output); Err(CandidateError::Unavailable) }
    }
}
pub(super) fn ner_options(json: Option<&str>) -> Result<NerOptions, CandidateError> {
    let options = match json {
        None => NerOptions::default(),
        Some(text) => {
            if text.len() > source::OPTIONS_BYTES { return Err(CandidateError::Input); }
            let value = canonjson::parse_str_with_limits(text, canonjson::ParseLimits {
                max_depth: 8, max_string_bytes: source::OPTIONS_BYTES,
            }).map_err(|_| CandidateError::Input)?;
            serde_json::from_value(value).map_err(|_| CandidateError::Input)?
        }
    };
    options.validate().map_err(|_| CandidateError::Planning)?;
    Ok(options)
}

#[cfg(test)] pub(super) mod tests;
