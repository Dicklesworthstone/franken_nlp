//! Explicit one-request input tokenization. Never an ambient/default change.
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
        .help("Untrusted message encoding; safe-bpe-v1 is an explicit byte-preserving, control-excluding candidate policy"))
        .arg(clap::Arg::new("max_tokenizer_pair_checks").long("max-tokenizer-pair-checks")
            .value_parser(clap::value_parser!(u64)).value_name("COUNT")
            .help("Whole-transcript BPE pair-comparison cap; safe-bpe-v1 only, default 4194304"))
        .arg(clap::Arg::new("max_tokenizer_pair_bytes").long("max-tokenizer-pair-bytes")
            .value_parser(clap::value_parser!(u64)).value_name("BYTES")
            .help("Whole-transcript BPE examined-pair byte cap; safe-bpe-v1 only, default 67108864"))
}

impl InputEncodingArgs {
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
}
