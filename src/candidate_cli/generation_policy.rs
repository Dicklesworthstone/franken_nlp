//! Bounded CLI controls for the existing identity-bound generation processor.
//!
//! Structured/scored task commands do not expose these switches. Stops use
//! the native exact-suffix contract: bytes are retained and streaming never
//! retracts content. Minimum length suppresses EOS, not byte stops or budgets.
use super::{CandidateError, GenerationOptions};
use clap::Args;
use crate::native_engine::lmhead::NANBEIGE_VOCAB_SIZE;

#[derive(Args)]
pub(super) struct GenerationPolicyArgs {
    /// Minimum non-EOS output tokens. Stops and resource ceilings still apply.
    #[arg(long, default_value_t = 0)]
    min_new_tokens: usize,
    /// Exact UTF-8 suffix checked after each token; matching bytes are retained.
    /// Repeat for alternatives (at most 64, each 1..=4096 UTF-8 bytes).
    #[arg(long = "stop", value_name = "TEXT")]
    stop_suffixes: Vec<String>,
    /// Exclude a vocabulary token from generation, even if it has positive bias.
    #[arg(long = "ban-token", value_name = "ID")]
    banned_token_ids: Vec<u32>,
    /// Sign-aware repetition multiplier in thousandths; 1000 disables it.
    /// Counts include the prompt and committed output, not just generated text.
    #[arg(long, default_value_t = 1000)]
    repetition_penalty_milli: u32,
    /// Subtract once for an already-seen token, in thousandths (-10000..=10000).
    #[arg(long, default_value_t = 0, allow_hyphen_values = true)]
    presence_penalty_milli: i32,
    /// Subtract per occurrence, in thousandths (-10000..=10000).
    #[arg(long, default_value_t = 0, allow_hyphen_values = true)]
    frequency_penalty_milli: i32,
    /// Add a logit bias after penalties: canonical TOKEN=MILLI, repeatable.
    /// Duplicate tokens are rejected; the bias range is -100000..=100000.
    #[arg(long = "logit-bias", value_name = "TOKEN=MILLI", value_parser = parse_bias)]
    logit_bias: Vec<TokenBias>,
}

impl Default for GenerationPolicyArgs {
    fn default() -> Self {
        Self { min_new_tokens: 0, stop_suffixes: Vec::new(), banned_token_ids: Vec::new(),
            repetition_penalty_milli: 1000, presence_penalty_milli: 0,
            frequency_penalty_milli: 0, logit_bias: Vec::new() }
    }
}

#[derive(Clone, Debug)]
struct TokenBias { token: u32, milli: i32 }

fn parse_bias(value: &str) -> Result<TokenBias, &'static str> {
    let (token, milli) = value.split_once('=').ok_or("expected canonical TOKEN=MILLI")?;
    let id = token.parse::<u32>().map_err(|_| "invalid bias token")?;
    let bias = milli.parse::<i32>().map_err(|_| "invalid bias value")?;
    if id.to_string() != token || bias.to_string() != milli
        || id as usize >= NANBEIGE_VOCAB_SIZE || bias.unsigned_abs() > 100_000 {
        return Err("bias token or value is noncanonical or outside its allowed range");
    }
    Ok(TokenBias { token: id, milli: bias })
}

impl GenerationPolicyArgs {
    pub(super) fn apply(&self, options: &mut GenerationOptions) -> Result<(), CandidateError> {
        // Check variable-sized inputs BEFORE copying them into native options.
        // The host's existing preparation reserve covers this bounded policy.
        if self.min_new_tokens > options.max_new_tokens || self.stop_suffixes.len() > 64
            || self.stop_suffixes.iter().any(|s| s.is_empty() || s.len() > 4096)
            || self.banned_token_ids.len() > NANBEIGE_VOCAB_SIZE || self.logit_bias.len() > 4096
            || self.banned_token_ids.iter().any(|&id| id as usize >= NANBEIGE_VOCAB_SIZE)
            || !(100..=10_000).contains(&self.repetition_penalty_milli)
            || self.presence_penalty_milli.unsigned_abs() > 10_000
            || self.frequency_penalty_milli.unsigned_abs() > 10_000 {
            return Err(CandidateError::Arguments);
        }
        let mut biases = std::collections::BTreeMap::new();
        for entry in &self.logit_bias {
            if entry.token as usize >= NANBEIGE_VOCAB_SIZE || entry.milli.unsigned_abs() > 100_000
                || biases.insert(entry.token, entry.milli).is_some() {
                return Err(CandidateError::Arguments);
            }
        }
        options.min_new_tokens = self.min_new_tokens;
        options.stop_suffixes = self.stop_suffixes.iter().map(|s| s.as_bytes().to_vec()).collect();
        options.banned_token_ids = self.banned_token_ids.clone();
        options.repetition_penalty_milli = self.repetition_penalty_milli;
        options.presence_penalty_milli = self.presence_penalty_milli;
        options.frequency_penalty_milli = self.frequency_penalty_milli;
        options.logit_bias_milli = biases;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate_cli::{CandidateArgs, CandidateCommand, definition};
    use clap::FromArgMatches;

    fn args(task: &str, extra: &[&str]) -> CandidateArgs {
        let mut argv = vec!["candidate", task, "--model", "local.fnlpq", "--memory-mib", "8192"];
        argv.extend_from_slice(extra);
        let matches = definition().try_get_matches_from(argv).unwrap();
        let CandidateCommand::Text { args, .. } = CandidateCommand::from_matches(&matches).unwrap()
            else { panic!("expected a free-text command") };
        args
    }

    #[test]
    fn defaults_preserve_the_existing_native_greedy_policy() {
        let args = args("generate", &[]);
        let options = args.options(166_101).unwrap();
        assert!(options == GenerationOptions::greedy(128, 65_536, 166_101));
        assert!(args.validate().is_ok());
    }

    #[test]
    fn generate_and_chat_forward_all_policy_controls_without_rewriting_stops() {
        for task in ["generate", "chat"] {
            let args = args(task, &["--min-new-tokens", "8", "--stop", " FIN\n", "--stop", "終わり",
                "--ban-token", "7", "--ban-token", "11", "--repetition-penalty-milli", "1200",
                "--presence-penalty-milli", "-250", "--frequency-penalty-milli", "750",
                "--logit-bias", "42=-1500", "--logit-bias", "0=100000"]);
            assert!(args.validate().is_ok());
            let options = args.options(166_101).unwrap();
            assert_eq!(options.min_new_tokens, 8);
            assert_eq!(options.stop_suffixes, vec![b" FIN\n".to_vec(), "終わり".as_bytes().to_vec()]);
            assert_eq!(options.banned_token_ids, vec![7, 11]);
            assert_eq!(options.repetition_penalty_milli, 1200);
            assert_eq!(options.presence_penalty_milli, -250);
            assert_eq!(options.frequency_penalty_milli, 750);
            assert_eq!(options.logit_bias_milli.get(&42), Some(&-1500));
            assert_eq!(options.logit_bias_milli.get(&0), Some(&100000));
        }
    }

    #[test]
    fn streaming_uses_the_same_policy_and_budget_validation() {
        for task in ["generate", "chat"] {
            let matches = definition().try_get_matches_from(["candidate", "stream", task,
                "--model", "local.fnlpq", "--memory-mib", "8192", "--stop", "END",
                "--min-new-tokens", "2", "--logit-bias", "7=-1000"]).unwrap();
            let (_, stream) = matches.subcommand().unwrap();
            let (_, inner) = stream.subcommand().unwrap();
            let parsed = crate::candidate_cli::stream::StreamArgs::from_arg_matches(inner).unwrap();
            assert!(parsed.validate().is_ok());
            let options = parsed.common.options(166_101).unwrap();
            assert_eq!(options.stop_suffixes, vec![b"END".to_vec()]);
            assert_eq!(options.min_new_tokens, 2);
            assert_eq!(options.logit_bias_milli.get(&7), Some(&-1000));
        }
    }

    #[test]
    fn inconsistent_or_out_of_range_options_fail_before_input_or_model_io() {
        for extra in [vec!["--min-new-tokens", "129"], vec!["--stop", ""],
            vec!["--ban-token", "166144"], vec!["--repetition-penalty-milli", "99"],
            vec!["--repetition-penalty-milli", "10001"], vec!["--presence-penalty-milli", "-10001"],
            vec!["--frequency-penalty-milli", "10001"],
            vec!["--logit-bias", "7=100", "--logit-bias", "7=100"]] {
            assert!(args("generate", &extra).validate().is_err());
        }
    }

    #[test]
    fn policy_sets_are_bounded_in_bytes_and_entries() {
        let mut args = args("generate", &[]);
        args.policy.stop_suffixes = vec!["x".repeat(4096); 64];
        assert!(args.validate().is_ok());
        args.policy.stop_suffixes.push("x".into());
        assert!(args.validate().is_err());
        args.policy.stop_suffixes = vec!["é".repeat(2049)];
        assert!(args.validate().is_err());
        args.policy.stop_suffixes.clear();
        args.policy.logit_bias = (0..4097).map(|token| TokenBias { token, milli: 0 }).collect();
        assert!(args.validate().is_err());
    }

    #[test]
    fn bias_spelling_cannot_alias_a_token_or_hide_nonfinite_values() {
        for value in ["", "1", "01=2", "+1=2", "-1=2", "1=+2", "1=-0", "1=02",
            "1=NaN", "1=1.5", "1=100001", "166144=0", "1=2=3", "1=2147483648"] {
            assert!(parse_bias(value).is_err(), "accepted invalid bias spelling");
        }
        assert!(parse_bias("0=-100000").is_ok());
        assert!(parse_bias("166143=100000").is_ok());
    }

    #[test]
    fn structured_and_scored_tasks_cannot_inherit_free_text_controls() {
        for task in ["ner", "keyphrases", "summarize", "answer", "classify", "sentiment"] {
            assert!(definition().try_get_matches_from(["candidate", task,
                "--model", "local.fnlpq", "--memory-mib", "8192", "--stop", "END"]).is_err());
        }
    }
}
