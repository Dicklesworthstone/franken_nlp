//! Explicit text/source tokenization, bounded per complete request/transcript.
//! Never an ambient/default change or a renewed allowance between segments.
use super::*;
use crate::tokenizer::untrusted::{SafeBpeLimits, UntrustedEncoding};

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(super) struct InputEncodingArgs {
    safe_bpe: bool,
    max_pair_checks: Option<u64>,
    max_pair_bytes: Option<u64>,
}

pub(super) fn definition(command: clap::Command) -> clap::Command {
    command.arg(clap::Arg::new("input_encoding").long("input-encoding")
        .value_parser(["byte-fallback", "safe-bpe-v1"]).default_value("byte-fallback")
        .help("Untrusted request-text encoding; safe-bpe-v1 is an explicit byte-preserving, control-excluding candidate policy"))
        .arg(clap::Arg::new("max_tokenizer_pair_checks").long("max-tokenizer-pair-checks")
            .value_parser(clap::value_parser!(u64)).value_name("COUNT")
            .help("BPE pair-comparison cap per complete request/transcript across all untrusted segments; safe-bpe-v1 only, default 4194304"))
        .arg(clap::Arg::new("max_tokenizer_pair_bytes").long("max-tokenizer-pair-bytes")
            .value_parser(clap::value_parser!(u64)).value_name("BYTES")
            .help("BPE examined-pair byte cap per complete request/transcript across all untrusted segments; safe-bpe-v1 only, default 67108864"))
}

impl InputEncodingArgs {
    pub(super) fn is_default(&self) -> bool { self == &Self::default() }

    pub(super) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        let safe_bpe = matches.get_one::<String>("input_encoding").is_some_and(|value| value == "safe-bpe-v1");
        let options = Self { safe_bpe,
            max_pair_checks: matches.get_one::<u64>("max_tokenizer_pair_checks").copied(),
            max_pair_bytes: matches.get_one::<u64>("max_tokenizer_pair_bytes").copied() };
        if !safe_bpe && (options.max_pair_checks.is_some() || options.max_pair_bytes.is_some()) {
            return Err(clap::Error::raw(clap::error::ErrorKind::ArgumentConflict,
                "tokenizer work ceilings require --input-encoding safe-bpe-v1"));
        }
        Ok(options)
    }
    pub(super) fn policy(&self, max_input_bytes: usize, max_output_tokens: usize)
        -> Result<UntrustedEncoding, CandidateError> {
        if !self.safe_bpe {
            if self.max_pair_checks.is_some() || self.max_pair_bytes.is_some() { return Err(CandidateError::Arguments); }
            return Ok(UntrustedEncoding::ByteFallback);
        }
        let policy = UntrustedEncoding::SafeBpeV1(SafeBpeLimits {
            max_input_bytes, max_output_tokens,
            max_pair_checks: self.max_pair_checks.unwrap_or(4 * 1024 * 1024),
            max_pair_bytes: self.max_pair_bytes.unwrap_or(64 * MIB),
        });
        policy.validate().map_err(|_| CandidateError::Arguments)?;
        Ok(policy)
    }
    pub(super) fn workspace_bytes(&self, max_input_bytes: usize, max_output_tokens: usize)
        -> Result<u64, CandidateError> {
        match self.policy(max_input_bytes, max_output_tokens)? {
            UntrustedEncoding::ByteFallback => Ok(0),
            UntrustedEncoding::SafeBpeV1(limits) => limits.requirements()
                .map(|required| required.workspace_bytes).map_err(|_| CandidateError::Arguments),
        }
    }
}

pub(super) fn reported_policy(policy: &UntrustedEncoding) -> Option<&UntrustedEncoding> {
    (policy != &UntrustedEncoding::ByteFallback).then_some(policy)
}

/// Complete-document planning counts every sizing probe and repeated schema
/// encoding. Its byte/token work ceilings are distinct from original input
/// bytes and from the context capacity of any one native chunk.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(super) struct DocumentInputEncodingArgs {
    encoding: InputEncodingArgs,
    max_input_bytes: Option<usize>,
    max_output_tokens: Option<usize>,
}

pub(super) const DOCUMENT_ENCODING_HELP: &str = "Compact document input: --input-encoding safe-bpe-v1 preserves exact source/schema bytes and excludes template-control tokens while encoding ordinary text compactly. One tokenizer ledger covers the preliminary schema sizing, every accepted or rejected chunk-size probe, and every final schema/source compilation in ONE complete document preparation. --max-tokenizer-input-bytes and --max-tokenizer-output-tokens bound cumulative preparation work, separately from original document bytes and each native context. The pair-work flags apply to the same complete preparation. Exceeding any cap rejects the document before native execution; there is no per-chunk reset or byte fallback. A preliminary preflight and the later owned execution preparation each receive the same fixed bounded policy; corpus documents and resumed job preparations use that same policy with a new per-preparation ledger, not a lifetime tokenizer-work promise. The exact policy is reported on compact results and authenticated in retained job recipes. One transient encoder workspace is admitted beside retained plans. This option supports independent exact-schema chunks and root-array union; root-object reconciliation remains byte encoded. It selects a different prompt token sequence and does not claim identical model output, tokenizer parity, task quality or measured speed.";

pub(super) fn document_definition(command: clap::Command) -> clap::Command {
    definition(command)
        .mut_arg("input_encoding", |arg| arg.help("Complete-document input policy; safe-bpe-v1 shares one bounded tokenizer ledger across sizing probes and all chunk/schema preparation"))
        .mut_arg("max_tokenizer_pair_checks", |arg| arg.help("BPE pair-comparison cap for one complete document preparation, including rejected sizing probes; safe-bpe-v1 only, default 67108864"))
        .mut_arg("max_tokenizer_pair_bytes", |arg| arg.help("BPE examined-pair byte cap for one complete document preparation; safe-bpe-v1 only, default 536870912"))
        .arg(clap::Arg::new("max_tokenizer_input_bytes").long("max-tokenizer-input-bytes")
            .value_parser(clap::value_parser!(usize)).value_name("BYTES")
            .help("Cumulative bytes encoded during one complete document preparation, including repeated schema/source probes; safe-bpe-v1 only, default and maximum 1048576"))
        .arg(clap::Arg::new("max_tokenizer_output_tokens").long("max-tokenizer-output-tokens")
            .value_parser(clap::value_parser!(usize)).value_name("TOKENS")
            .help("Cumulative untrusted tokens produced during one complete document preparation; separate from per-chunk context, safe-bpe-v1 only, default and maximum 1048576"))
}

impl DocumentInputEncodingArgs {
    pub(super) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        let options = Self { encoding: InputEncodingArgs::from_matches(matches)?,
            max_input_bytes: matches.get_one::<usize>("max_tokenizer_input_bytes").copied(),
            max_output_tokens: matches.get_one::<usize>("max_tokenizer_output_tokens").copied() };
        if !options.encoding.safe_bpe && (options.max_input_bytes.is_some() || options.max_output_tokens.is_some()) {
            return Err(clap::Error::raw(clap::error::ErrorKind::ArgumentConflict,
                "complete-document tokenizer ceilings require --input-encoding safe-bpe-v1"));
        }
        Ok(options)
    }

    pub(super) fn policy(&self) -> Result<UntrustedEncoding, CandidateError> {
        if !self.encoding.safe_bpe {
            if self.max_input_bytes.is_some() || self.max_output_tokens.is_some() {
                return Err(CandidateError::Arguments);
            }
            return self.encoding.policy(1, 1);
        }
        let mut encoding = self.encoding.clone();
        encoding.max_pair_checks = Some(encoding.max_pair_checks.unwrap_or(SafeBpeLimits::MAX_PAIR_CHECKS));
        encoding.max_pair_bytes = Some(encoding.max_pair_bytes.unwrap_or(SafeBpeLimits::MAX_PAIR_BYTES));
        encoding.policy(self.max_input_bytes.unwrap_or(SafeBpeLimits::MAX_INPUT_BYTES),
            self.max_output_tokens.unwrap_or(SafeBpeLimits::MAX_OUTPUT_TOKENS))
    }

    pub(super) fn workspace_bytes(&self) -> Result<u64, CandidateError> {
        match self.policy()? {
            UntrustedEncoding::ByteFallback => Ok(0),
            UntrustedEncoding::SafeBpeV1(limits) => limits.requirements()
                .map(|required| required.workspace_bytes).map_err(|_| CandidateError::Arguments),
        }
    }
}

/// Only the explicitly changed tokenization route adds provenance to the
/// existing result envelope; the default response remains byte-compatible.
#[derive(Serialize)]
pub(super) struct Response<'a, T: Serialize> {
    #[serde(flatten)]
    pub response: &'a T,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_encoding: Option<&'a UntrustedEncoding>,
}

#[cfg(test)]
mod tests {
    use super::*;
    fn parsed(args: &[&str]) -> Result<InputEncodingArgs, clap::Error> {
        let matches = definition(clap::Command::new("generate")).try_get_matches_from(args)?;
        InputEncodingArgs::from_matches(&matches)
    }
    #[test]
    fn byte_default_has_no_new_workspace_or_serialized_policy() {
        let args = parsed(&["generate"]).unwrap();
        assert_eq!(args.policy(1024, 1024).unwrap(), UntrustedEncoding::ByteFallback);
        assert_eq!(args.workspace_bytes(1024, 1024).unwrap(), 0);
        let value = serde_json::json!({"output":"fixture"});
        assert_eq!(serde_json::to_value(Response { response: &value, input_encoding: None }).unwrap(), value);
        assert!(parsed(&["generate", "--max-tokenizer-pair-checks", "1"]).is_err());
    }
    #[test]
    fn explicit_bpe_limits_are_validated_priced_and_reported() {
        let args = parsed(&["generate", "--input-encoding", "safe-bpe-v1",
            "--max-tokenizer-pair-checks", "1234", "--max-tokenizer-pair-bytes", "10000"]).unwrap();
        let policy = args.policy(1024, 512).unwrap();
        let UntrustedEncoding::SafeBpeV1(limits) = &policy else { panic!("wrong policy"); };
        assert_eq!(limits.max_pair_checks, 1234);
        assert_eq!(limits.max_pair_bytes, 10000);
        assert!(args.workspace_bytes(1024, 512).unwrap() >= 256 * 1024);
        let value = serde_json::json!({"output":"fixture"});
        let reported = serde_json::to_value(Response { response: &value, input_encoding: Some(&policy) }).unwrap();
        assert_eq!(reported["input_encoding"], serde_json::to_value(&policy).unwrap());
        for bad in ["0", "18446744073709551615"] {
            let options = parsed(&["generate", "--input-encoding", "safe-bpe-v1",
                "--max-tokenizer-pair-checks", bad]).unwrap();
            assert!(options.policy(1024, 512).is_err());
        }
    }

    fn document_args(args: &[&str]) -> Result<DocumentInputEncodingArgs, clap::Error> {
        let matches = document_definition(clap::Command::new("extract-map")).try_get_matches_from(args)?;
        DocumentInputEncodingArgs::from_matches(&matches)
    }

    #[test]
    fn complete_document_encoding_has_distinct_cumulative_limits_and_no_default_charge() {
        let default = document_args(&["extract-map"]).unwrap();
        assert_eq!(default.policy().unwrap(), UntrustedEncoding::ByteFallback);
        assert_eq!(default.workspace_bytes().unwrap(), 0);
        let compact = document_args(&["extract-map", "--input-encoding", "safe-bpe-v1"]).unwrap();
        let UntrustedEncoding::SafeBpeV1(limits) = compact.policy().unwrap() else { panic!("wrong encoding"); };
        assert_eq!(limits, SafeBpeLimits { max_input_bytes: SafeBpeLimits::MAX_INPUT_BYTES,
            max_output_tokens: SafeBpeLimits::MAX_OUTPUT_TOKENS,
            max_pair_checks: SafeBpeLimits::MAX_PAIR_CHECKS, max_pair_bytes: SafeBpeLimits::MAX_PAIR_BYTES });
        assert_eq!(compact.workspace_bytes().unwrap(), limits.requirements().unwrap().workspace_bytes);
        for flag in ["--max-tokenizer-input-bytes", "--max-tokenizer-output-tokens",
            "--max-tokenizer-pair-checks", "--max-tokenizer-pair-bytes"] {
            assert!(document_args(&["extract-map", flag, "1"]).is_err());
            for value in ["0", "18446744073709551615"] {
                let parsed = document_args(&["extract-map", "--input-encoding", "safe-bpe-v1", flag, value]);
                assert!(parsed.is_err() || parsed.unwrap().policy().is_err());
            }
        }
    }

    #[test]
    fn complete_document_all_caps_are_explicit_and_affect_policy_identity() {
        let custom = document_args(&["extract-map", "--input-encoding", "safe-bpe-v1",
            "--max-tokenizer-input-bytes", "4096", "--max-tokenizer-output-tokens", "2048",
            "--max-tokenizer-pair-checks", "7777", "--max-tokenizer-pair-bytes", "65536"]).unwrap();
        assert_eq!(custom.policy().unwrap(), UntrustedEncoding::SafeBpeV1(SafeBpeLimits {
            max_input_bytes: 4096, max_output_tokens: 2048, max_pair_checks: 7777, max_pair_bytes: 65536 }));
        assert!(custom.workspace_bytes().unwrap()
            < document_args(&["extract-map", "--input-encoding", "safe-bpe-v1"]).unwrap().workspace_bytes().unwrap());
    }
}
