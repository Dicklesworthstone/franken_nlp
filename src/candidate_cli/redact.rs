//! Model-backed detector-union redaction. Only completed edited text is exported.
use super::*;
pub(super) mod long;
pub(super) mod corpus;
pub(super) mod retention;
use clap::ValueEnum;
use crate::tasks::{ner::NerOptions, redact::{PiiKind, RedactionRequest,
    actions::RedactionAction, pseudonym::PseudonymKey}};
use crate::tokenizer::untrusted::{SafeBpeWorkLimits, UntrustedEncoding};
use source::{SourceHostArgs, prefill::SourcePrefillArgs};

#[derive(Clone, Copy, Eq, PartialEq, ValueEnum)]
pub(super) enum Action { Mask, Placeholder, Pseudonymize }
#[derive(Clone, Copy, Eq, PartialEq, ValueEnum)]
pub(super) enum Rule { Email, Phone, Url, IpAddress, CreditCard, Date }
impl Rule {
    pub(super) fn kind(self) -> PiiKind {
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
    /// Source scheduling stays separate from free-text generation policy.
    #[command(flatten)]
    pub(super) prefill: SourcePrefillArgs,
    #[arg(skip)]
    pub(super) compute: compute::Args,
    /// Compact input is admitted only for one complete redaction operation.
    #[arg(skip)]
    pub(super) encoding: input_encoding::InputEncodingArgs,
    #[command(flatten)]
    pub(super) long: long::LongArgs,
    #[command(flatten)]
    pub(super) corpus: corpus::CorpusArgs,
    #[command(flatten)]
    pub(super) retention: retention::RetentionArgs,
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
    /// Raw 32..4096-byte high-entropy key via inherited stdin; source is a file or explicit retained input.
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
    compute::definition(input_encoding::definition(RedactCommand::augment_args(clap::Command::new("redact")
        .about("Redact with native source-bound NER plus rules; verify the edited text afresh")
        .after_help("Verification covers only the selected detector union, not all PII. Model omissions and Unicode obfuscations remain possible. Pseudonyms are not anonymization. By default this candidate command emits one completed JSON object, never intermediate NER text. Pseudonymization always uses full 256-bit HMAC; raw key bytes have no argv option. Add --chunked for long documents: rules scan whole text while NER uses source-aligned chunks, then verification re-chunks the actual edited text. NER chunk boundaries may split entities; per-document work ceilings never renew per chunk. Add --ndjson for an owned corpus stream with one resident engine, fixed policy/key scope and independent whole-corpus ceilings. Completed document events remain valid if later records fail; any failed record yields a nonzero exit. Records cannot override actions, keys, types, chunking or verification. Add --store-results with --job-dir, --job-id, --job-key-file and --job-limits for explicit private retention. Save original envelopes only with --spool-input --max-retained-input-mib N; --resume --use-retained-input --max-retained-input-mib N restores them without a population positional argument or population stdin read. Otherwise resume requires the complete original population. Policy and both key scopes remain unchanged. Retained mode emits metadata only, and an error may follow durable progress. It requires metadata-store on supported Linux targets. The independent fnlp redact --rules-only command remains model-free."))
    ))
        .mut_arg("compute_threads", |arg| arg.help("Coordinator plus persistent INT8 row workers, 1..=64; one team spans original detection and fresh edited-text verification, including every chunk. Runtime threads are additional. Composes with --prefill-rows. One-shot, live corpus and retained start/resume share this schedule; a population reuses one team for its complete invocation"))
        .mut_arg("preparation_mib", |arg| arg
            .default_value_if("input_encoding", clap::builder::ArgPredicate::Equals("safe-bpe-v1".into()), "1024")
            .help("Preparation memory in MiB; defaults to 512 for byte input or 1024 for compact redaction, including edited-text tokenizer workspace"))
        .mut_arg("max_tokenizer_pair_checks", |arg| arg.help("BPE comparison cap for a complete short operation or each chunked encoding call; chunked totals use --max-total-tokenizer-pair-checks; safe-bpe-v1 only, default 4194304"))
        .mut_arg("max_tokenizer_pair_bytes", |arg| arg.help("BPE examined-byte cap for a complete short operation or each chunked encoding call; chunked totals use --max-total-tokenizer-pair-bytes; safe-bpe-v1 only, default 67108864"))
        .after_long_help("--input-encoding safe-bpe-v1 preserves exact source bytes and excludes template-control tokens. One tokenizer ledger covers original preflight, execution planning and fresh verification of the actual edited text. For short documents, pair-work ceilings apply to their sum and byte/token totals cover two preparations, or three with verification. With --chunked, pair flags bound each encoding call and --max-total-tokenizer-* bound the complete operation, including accepted/rejected probes and both detector stages. Total byte/token defaults are eight times the sum of original and edited byte ceilings; total pair defaults are 268435456 checks and 4294967296 examined bytes. Every accepted chunk independently fits context. Admission prices possible edited expansion and the complete map frontier before private key, document or model IO. One-shot compact results report exact policy, complete limits and consumed work; population metadata reports policy, limits and per-document scope. The policy digest binds both limits. Exhaustion returns no successful result or fallback. Compact populations require --chunked: each original document attempt retains one allowance through reconstruction and verification, while separate original records receive independent allowances. Retained jobs authenticate the policy and limits; explicit resume may start new invocation-local tokenizer work without refunding durable attempts or native/mask work. This changes prompt tokens and makes no claim of identical model output, detector quality or measured speed.")
}
impl RedactCommand {
    pub(super) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        let mut command = Self::from_arg_matches(matches)?;
        command.encoding = input_encoding::InputEncodingArgs::from_matches(matches)?;
        command.compute = compute::Args::from_matches(matches)?;
        Ok(command)
    }
    pub(super) fn validate(&self) -> Result<(CandidateArgs, Limits), CandidateError> {
        let prefill = self.prefill.limits()?;
        let compute = self.compute.admit_prefill(prefill)?;
        if !self.encoding.is_default() && !self.long.chunked
            && (self.corpus.ndjson || self.retention.store_results) {
            return Err(CandidateError::Arguments);
        }
        // Long documents encode bounded probes and chunks, never the entire
        // original as one call. Price that workspace beside all retained plans.
        let (mut common, mut limits) = if self.long.chunked {
            self.host.common(self.input.clone())?
        } else { self.host.common_with_encoding(self.input.clone(), &self.encoding)? };
        common.input_encoding = self.encoding.clone();
        common.compute = self.compute;
        limits.compute = compute;
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
        self.retention.validate(self.corpus.ndjson, limits)?;
        self.retention.input_retention.validate(self.retention.mode(), &self.input)
            .map_err(|_| CandidateError::Arguments)?;
        let population_stdin = self.input.as_os_str() == "-" && !self.retention.input_retention.use_retained_input;
        validate_key_scope(self.action, population_stdin, self.key_stdin, self.key_id.as_deref(),
            self.namespace.as_deref(), self.expected_key_commitment.as_deref())?;
        self.corpus.validate(self, limits)?;
        let (_, complete) = self.encoding_policy(limits)?;
        if (complete.is_some() || compute.threads > 1 || prefill.is_some())
            && self.compact_peak_bytes(limits, edit)? > limits.memory_bytes {
            return Err(CandidateError::Arguments);
        }
        Ok((common, limits))
    }
    pub(super) fn encoding_policy(&self, limits: Limits)
        -> Result<(UntrustedEncoding, Option<SafeBpeWorkLimits>), CandidateError> {
        if self.long.chunked { return self.long.encoding_policy(&self.encoding, &self.host, limits); }
        // Verification can expand the source. Only its admitted encoding peak
        // grows; the original document reader still uses max_input_bytes.
        let input_bytes = self.host.max_input_bytes.max(self.host.max_result_bytes);
        let policy = self.encoding.policy(input_bytes, limits.max_prompt_tokens)?;
        let complete = match &policy {
            UntrustedEncoding::ByteFallback => None,
            UntrustedEncoding::SafeBpeV1(call) => {
                let passes = 2 + u64::from(!self.no_verify);
                let complete = SafeBpeWorkLimits {
                    max_input_bytes: (call.max_input_bytes as u64).checked_mul(passes)
                        .ok_or(CandidateError::Arguments)?,
                    max_output_tokens: (call.max_output_tokens as u64).checked_mul(passes)
                        .ok_or(CandidateError::Arguments)?,
                    max_pair_checks: call.max_pair_checks, max_pair_bytes: call.max_pair_bytes,
                };
                complete.validate().map_err(|_| CandidateError::Arguments)?;
                let workspace = call.requirements().map_err(|_| CandidateError::Arguments)?.workspace_bytes
                    .checked_add(1024).ok_or(CandidateError::Arguments)?;
                let floor = (256 * MIB).checked_add(input_bytes as u64 * 32)
                    .and_then(|n| n.checked_add(self.host.max_result_bytes as u64 * 16))
                    .and_then(|n| n.checked_add(workspace)).ok_or(CandidateError::Arguments)?;
                if limits.preparation_bytes < floor { return Err(CandidateError::Arguments); }
                Some(complete)
            }
        };
        Ok((policy, complete))
    }
    fn compact_peak_bytes(&self, limits: Limits, edit_bytes: u64) -> Result<u64, CandidateError> {
        if self.corpus.ndjson { return self.corpus.compact_base_peak_bytes(self, limits, edit_bytes); }
        let native = Int8MemoryRequirement::for_context(self.host.context_tokens)
            .map_err(|_| CandidateError::Arguments)?;
        // The caller preparation and the transferred planner/source owner
        // coexist with edited text, a fresh NER result, native KV and output.
        let peak = compute::request_peak_bytes(limits, native, self.host.max_result_bytes as u64,
            0, 0, limits.preparation_bytes)?;
        let temporary = if self.long.chunked {
            self.long.temporary_bytes(&self.host, edit_bytes)?
        } else {
            (self.host.max_result_bytes as u64).checked_mul(4)
                .and_then(|n| n.checked_add(self.host.max_new_tokens as u64 * 8))
                .and_then(|n| n.checked_add(edit_bytes)).ok_or(CandidateError::Arguments)?
        };
        let prefill_bytes = self.prefill.limits()?.map(|limits| limits.validate())
            .transpose().map_err(|_| CandidateError::Arguments)?.unwrap_or(0);
        [peak, self.host.max_input_bytes as u64 + 1, 4096, temporary, prefill_bytes]
            .into_iter().try_fold(0_u64, |sum, bytes| sum.checked_add(bytes))
            .ok_or(CandidateError::Arguments)
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
        read_key(input, self.key_stdin, self.key_id.as_deref(), self.namespace.as_deref(),
            self.expected_key_commitment.as_deref())
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
/// One private-key contract for the INT8 and grouped-INT4 redaction commands.
/// No backend may relax stdin separation or accept raw key bytes in argv.
pub(super) fn validate_key_scope(action: Action, population_stdin: bool, key_stdin: bool,
    key_id: Option<&str>, namespace: Option<&str>, expected: Option<&str>) -> Result<(), CandidateError> {
    if action == Action::Pseudonymize {
        if !key_stdin || population_stdin
            || key_id.is_none_or(|id| id.is_empty() || id.len() > 128
                || !id.bytes().all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b)))
            || namespace.is_none_or(|s| s.is_empty() || s.len() > 256) {
            return Err(CandidateError::Arguments);
        }
    } else if key_stdin || key_id.is_some() || namespace.is_some() || expected.is_some() {
        return Err(CandidateError::Arguments);
    }
    if expected.is_some_and(|s| s.len() != 64
        || !s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))) {
        return Err(CandidateError::Arguments);
    }
    Ok(())
}
pub(super) fn read_key(input: &mut impl Read, key_stdin: bool, key_id: Option<&str>,
    namespace: Option<&str>, expected: Option<&str>) -> Result<Option<(PseudonymKey, String)>, CandidateError> {
    if !key_stdin { return Ok(None); }
    let key = PseudonymKey::from_reader(input, key_id.ok_or(CandidateError::Arguments)?)
        .map_err(|_| CandidateError::Input)?;
    if let Some(expected) = expected { key.require_commitment(expected).map_err(|_| CandidateError::Identity)?; }
    Ok(Some((key, namespace.ok_or(CandidateError::Arguments)?.to_owned())))
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
