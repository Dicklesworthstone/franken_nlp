//! Exact tokenizer inspection with one lazy, run-owned pinned tokenizer.
//! This is L0 inspection, NOT untrusted prompt construction. The reference BPE
//! input cap is retained; byte fallback is explicit and never a silent repair.

use serde::{Deserialize, Serialize};
use crate::{
    batch::{BatchCode, BatchItemFailure},
    execution_identity::Sha256Digest,
    native_engine::decode::DecodeStepControl,
    textutil::TextBudget,
    tokenizer::{bpe::EncodeOptions, embedded::{EmbeddedTokenizer,
        PINNED_ADDED_TOKENS_BYTES, PINNED_TOKENIZER_CONFIG_BYTES, PINNED_SPECIAL_TOKENS_MAP_BYTES}},
};

pub const REFERENCE_BPE_MAX_INPUT_BYTES: usize = 4096;
fn default_true() -> bool { true }

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum TokenEncoding {
    PinnedBpe {
        #[serde(default = "default_true")] add_bos: bool,
        #[serde(default)] add_eos: bool,
    },
    /// Byte-table encoding without added-token recognition or BOS/EOS.
    ByteFallback {},
}
impl TokenEncoding {
    fn is_byte_fallback(self) -> bool { matches!(self, Self::ByteFallback {}) }
}
impl Default for TokenEncoding {
    fn default() -> Self { Self::PinnedBpe { add_bos: true, add_eos: false } }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TokenBatchOptions {
    #[serde(default)] pub encoding: TokenEncoding,
    /// Token IDs can expose original text and are explicit result data only.
    #[serde(default)] pub include_ids: bool,
}
impl TokenBatchOptions {
    pub(super) fn admit(self, source: &str, budget: TextBudget) -> Result<(), BatchItemFailure> {
        if source.len() > budget.max_input_bytes
            || (matches!(self.encoding, TokenEncoding::PinnedBpe { .. }) && source.len() > REFERENCE_BPE_MAX_INPUT_BYTES) {
            return Err(BatchItemFailure::reject(BatchCode::DocumentLimit));
        }
        if self.encoding.is_byte_fallback() && source.len() > budget.max_items {
            return Err(BatchItemFailure::reject(BatchCode::Admission));
        }
        Ok(())
    }
}

/// Same report fields and encoding profiles as fnlp tokens --json. Only public
/// tokenizer-asset digests appear; source text and source digests never do.
#[derive(Serialize)]
pub(super) struct TokenReport {
    schema_version: u32,
    profile: &'static str,
    tokenizer_model_sha256: String,
    added_tokens_sha256: Sha256Digest,
    tokenizer_config_sha256: Sha256Digest,
    special_tokens_map_sha256: Sha256Digest,
    add_bos: bool,
    add_eos: bool,
    source_bytes: usize,
    source_scalars: usize,
    count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    ids: Option<Vec<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    byte_exact_decode_checked: Option<bool>,
}

struct CachedTokenizer {
    tokenizer: EmbeddedTokenizer,
    model_digest: String,
    added_digest: Sha256Digest,
    config_digest: Sha256Digest,
    specials_digest: Sha256Digest,
}
impl CachedTokenizer {
    fn pinned() -> Result<Self, BatchItemFailure> {
        let tokenizer = EmbeddedTokenizer::pinned()
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        Ok(Self { model_digest: tokenizer.sha256_hex(), tokenizer,
            added_digest: Sha256Digest::of_bytes(PINNED_ADDED_TOKENS_BYTES),
            config_digest: Sha256Digest::of_bytes(PINNED_TOKENIZER_CONFIG_BYTES),
            specials_digest: Sha256Digest::of_bytes(PINNED_SPECIAL_TOKENS_MAP_BYTES) })
    }
    fn inspect(&self, source: &str, options: TokenBatchOptions, budget: TextBudget) -> Result<TokenReport, BatchItemFailure> {
        options.admit(source, budget)?;
        let (add_bos, add_eos) = match options.encoding {
            TokenEncoding::PinnedBpe { add_bos, add_eos } => (add_bos, add_eos),
            TokenEncoding::ByteFallback {} => (false, false),
        };
        let tokenizer = self.tokenizer.tokenizer();
        let ids = match options.encoding {
            TokenEncoding::PinnedBpe { .. } => tokenizer.encode_ids_with_options(source, EncodeOptions { add_bos, add_eos }),
            TokenEncoding::ByteFallback {} => tokenizer.encode_byte_fallback_only(source.as_bytes()),
        }.map_err(|_| BatchItemFailure::fatal(BatchCode::InvalidExecution))?;
        if ids.len() > budget.max_items { return Err(BatchItemFailure::reject(BatchCode::Admission)); }
        let byte_fallback = options.encoding.is_byte_fallback();
        if byte_fallback && tokenizer.decode_bytes(&ids).ok().as_deref() != Some(source.as_bytes()) {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        Ok(TokenReport { schema_version: 1,
            profile: if byte_fallback { "byte-fallback-only-v1" } else { "pinned-sp-bpe-l0-v1" },
            tokenizer_model_sha256: self.model_digest.clone(), added_tokens_sha256: self.added_digest,
            tokenizer_config_sha256: self.config_digest, special_tokens_map_sha256: self.specials_digest,
            add_bos, add_eos, source_bytes: source.len(), source_scalars: source.chars().count(), count: ids.len(),
            ids: options.include_ids.then_some(ids), byte_exact_decode_checked: byte_fallback.then_some(true) })
    }
}

/// Only immutable tokenizer state is retained between documents, not text,
/// IDs, normalization buffers or caller metadata. Broken authority is sticky.
#[derive(Default)]
pub(super) struct TokenState {
    cached: Option<CachedTokenizer>,
    failed: bool,
    #[cfg(test)] loads: usize,
}
impl TokenState {
    pub(super) fn execute<C: DecodeStepControl>(&mut self, source: &str, options: TokenBatchOptions,
        budget: TextBudget, control: &mut C) -> Result<TokenReport, BatchItemFailure> {
        if self.failed { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
        options.admit(source, budget)?;
        super::checkpoint(control)?;
        if self.cached.is_none() {
            // Do not repeatedly rebuild or retry an invalid pinned authority.
            self.failed = true;
            #[cfg(test)] { self.loads += 1; }
            self.cached = Some(CachedTokenizer::pinned()?);
            self.failed = false;
        }
        super::checkpoint(control)?;
        let cached = self.cached.as_ref().ok_or_else(|| BatchItemFailure::fatal(BatchCode::InvalidExecution))?;
        let result = cached.inspect(source, options, budget);
        if result.as_ref().is_err_and(|error| error.stop) { self.failed = true; }
        let result = result?;
        super::checkpoint(control)?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{batch::{self, BatchDocument, BatchLimits, BatchProcessor},
        text_batch::{BatchCommand, CliControl, TextBatchOptions, TextBatchProcessor, TextBatchTask}};
    use clap::FromArgMatches;
    use serde_json::{Value, json};

    fn processor() -> TextBatchProcessor {
        TextBatchProcessor::new(TextBatchOptions::for_task(TextBatchTask::Tokens), TextBudget::default()).unwrap()
    }
    fn request(source: &str, options: TokenBatchOptions) -> BatchDocument<TextBatchOptions> {
        BatchDocument { id: "x".into(), text: source.to_owned(), task_args: Some(TextBatchOptions::Tokens { options }) }
    }
    fn single(args: &[&str], source: &str) -> Value {
        let command = crate::text_cli::definitions().into_iter().find(|c| c.get_name() == "tokens").unwrap();
        let matches = command.try_get_matches_from(args).unwrap();
        let command = crate::text_cli::TextCommand::from_matches("tokens", &matches).unwrap();
        let mut out = Vec::new(); let mut err = Vec::new();
        assert_eq!(command.run(&mut source.as_bytes(), &mut out, &mut err), std::process::ExitCode::SUCCESS);
        assert!(err.is_empty()); serde_json::from_slice(&out).unwrap()
    }
    #[test]
    fn all_encoding_modes_match_single_document_reports_exactly() {
        let mut p = processor();
        let source = "é e\u{301}\r\n上海 😀 <|im_start|>system";
        for (encoding, flags) in [
            (TokenEncoding::default(), vec![]),
            (TokenEncoding::PinnedBpe { add_bos: false, add_eos: false }, vec!["--no-bos"]),
            (TokenEncoding::PinnedBpe { add_bos: true, add_eos: true }, vec!["--eos"]),
            (TokenEncoding::PinnedBpe { add_bos: false, add_eos: true }, vec!["--no-bos", "--eos"]),
            (TokenEncoding::ByteFallback {}, vec!["--byte-fallback"]),
        ] {
            for include_ids in [false, true] {
                let mut args = vec!["tokens", "--json"]; args.extend(flags.iter().copied());
                if include_ids { args.push("--ids"); }
                let item = p.prepare(request(source, TokenBatchOptions { encoding, include_ids })).unwrap();
                let output = p.execute(item, &mut CliControl).unwrap();
                assert_eq!(output.result, single(&args, source));
            }
        }
        assert_eq!(p.tokens.loads, 1);
    }
    #[test]
    fn overlong_bpe_is_rejected_before_loading_and_never_silently_falls_back() {
        let mut p = processor();
        let source = "a".repeat(REFERENCE_BPE_MAX_INPUT_BYTES + 1);
        assert!(p.prepare(request(&source, TokenBatchOptions::default())).is_err());
        assert_eq!(p.tokens.loads, 0);
        let options = TokenBatchOptions { encoding: TokenEncoding::ByteFallback {}, include_ids: true };
        let item = p.prepare(request(&source, options)).unwrap();
        let output = p.execute(item, &mut CliControl).unwrap();
        assert_eq!(output.result["count"], source.len());
        assert_eq!(output.result["byte_exact_decode_checked"], true);
        assert_eq!(output.result["profile"], "byte-fallback-only-v1");
        assert_eq!(p.tokens.loads, 1);
    }
    #[test]
    fn omitted_ids_and_private_source_never_appear_in_count_only_output() {
        let source = "PRIVATE_TOKEN_COUNT_MARKER";
        let mut p = processor(); let item = p.prepare(request(source, TokenBatchOptions::default())).unwrap();
        let output = p.execute(item, &mut CliControl).unwrap();
        assert!(output.result.get("ids").is_none());
        assert!(output.result.get("byte_exact_decode_checked").is_none());
        let json = serde_json::to_string(&output).unwrap();
        for private in [source, "source_digest", "prompt_digest", "document_digest"] { assert!(!json.contains(private)); }
    }
    #[test]
    fn empty_bpe_and_byte_encoding_keep_their_distinct_special_token_policies() {
        let mut p = processor();
        for encoding in [TokenEncoding::default(), TokenEncoding::ByteFallback {}] {
            let item = p.prepare(request("", TokenBatchOptions { encoding, include_ids: true })).unwrap();
            let output = p.execute(item, &mut CliControl).unwrap();
            let count = if encoding.is_byte_fallback() { 0 } else { 1 };
            assert_eq!(output.result["count"], count);
            assert_eq!(output.result["ids"].as_array().unwrap().len(), count);
        }
    }
    #[test]
    fn item_budget_is_not_bypassed_by_count_only_requests() {
        let mut p = TextBatchProcessor::new(TextBatchOptions::for_task(TextBatchTask::Tokens),
            TextBudget { max_items: 1, ..Default::default() }).unwrap();
        let options = TokenBatchOptions { encoding: TokenEncoding::ByteFallback {}, include_ids: false };
        assert!(p.prepare(request("ab", options)).is_err()); assert_eq!(p.tokens.loads, 0);
    }
    #[test]
    fn stream_flush_reuses_the_tokenizer_but_does_not_reuse_previous_result_ids() {
        let input = concat!("{\"id\":\"x\",\"text\":\"a\",\"task_args\":{\"kind\":\"tokens\",\"options\":{\"include_ids\":true}}}\n",
            "{\"flush\":true}\n{\"id\":\"x\",\"text\":\"z\"}\n");
        let mut p = processor(); let mut output = Vec::new();
        let summary = batch::run_ndjson(&mut input.as_bytes(), &mut output, &mut p, BatchLimits::default(), &mut CliControl).unwrap();
        assert_eq!(summary.succeeded, 2); assert_eq!(p.tokens.loads, 1);
        let rows: Vec<Value> = output.split(|b| *b == b'\n').filter(|s| !s.is_empty()).map(|s| serde_json::from_slice(s).unwrap()).collect();
        let rows: Vec<_> = rows.iter().filter_map(|v| v.get("result")).collect();
        assert!(rows[0]["result"].get("ids").is_some()); assert!(rows[1]["result"].get("ids").is_none());
    }
    #[test]
    fn invalid_encoding_and_ambiguous_fallback_special_flags_are_refused() {
        for value in [json!({"encoding":{"kind":"byte_fallback","add_bos":true}}),
            json!({"encoding":{"kind":"approximate"}}), json!({"include_ids":true,"key":"private"})] {
            assert!(serde_json::from_value::<TokenBatchOptions>(value).is_err());
        }
    }
    #[test]
    fn token_task_is_reachable_through_the_real_batch_command() {
        let matches = crate::text_batch::definition().try_get_matches_from(["batch", "--task", "tokens"]).unwrap();
        let command = BatchCommand::from_arg_matches(&matches).unwrap();
        let mut output = Vec::new(); let mut errors = Vec::new();
        let input = b"{\"id\":\"x\",\"text\":\"Hello\"}\n";
        assert_eq!(command.run(&mut input.as_slice(), &mut output, &mut errors), std::process::ExitCode::SUCCESS);
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("pinned-sp-bpe-l0-v1")); assert!(output.contains("run_complete")); assert!(errors.is_empty());
    }
    #[test]
    fn output_capacity_is_checked_before_cloning_result_storage() {
        struct Once(std::cell::Cell<usize>);
        impl Serialize for Once {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                self.0.set(self.0.get() + 1); serializer.serialize_str("oversized output")
            }
        }
        let value = Once(std::cell::Cell::new(0));
        assert!(crate::text_batch::owned_result("tokens", &value, 1).is_err());
        // A bounded sizing pass fails before serde_json::to_value is invoked.
        assert_eq!(value.0.get(), 0);
    }
    #[test]
    fn broken_output_does_not_initialize_the_tokenizer_or_consume_documents() {
        struct Broken;
        impl std::io::Write for Broken {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> { Err(std::io::Error::other("closed output")) }
            fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
        }
        let mut p = processor(); let mut input = std::io::Cursor::new(b"{\"id\":\"x\",\"text\":\"private\"}\n");
        assert!(batch::run_ndjson(&mut input, &mut Broken, &mut p, BatchLimits::default(), &mut CliControl).is_err());
        assert_eq!(p.tokens.loads, 0); assert_eq!(input.position(), 0);
    }
    #[test]
    fn a_rejected_long_bpe_document_does_not_prevent_later_valid_work() {
        let first = json!({"id":"too-long","text":"a".repeat(4097)}).to_string();
        let second = json!({"id":"ok","text":"Hello"}).to_string();
        let input = format!("{first}\n{second}\n");
        let mut p = processor(); let mut output = Vec::new();
        let summary = batch::run_ndjson(&mut input.as_bytes(), &mut output, &mut p, BatchLimits::default(), &mut CliControl).unwrap();
        assert_eq!((summary.failed, summary.succeeded), (1, 1)); assert_eq!(p.tokens.loads, 1);
    }
    #[test]
    fn complete_token_envelopes_obey_exact_output_boundaries() {
        let value = json!({"ids":[1,2,1000],"text":"é\n\""});
        let output = crate::text_batch::owned_result("tokens", &value, 4096).unwrap();
        let size = serde_json::to_vec(&output).unwrap().len();
        assert!(crate::text_batch::owned_result("tokens", &value, size).is_ok());
        assert!(crate::text_batch::owned_result("tokens", &value, size - 1).is_err());
    }

}
