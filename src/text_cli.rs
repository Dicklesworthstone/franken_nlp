//! Real model-free command dispatch over the pinned tokenizer and textutil.
//! The simple L0 BPE authority is intentionally capped at 4096 input bytes;
//! a larger request is refused, never silently given byte-fallback counts.
use std::{io::{Read, Write}, path::PathBuf, process::ExitCode};
use clap::{Args, ArgMatches, FromArgMatches};
use serde::Serialize;
use crate::{canonjson, error::ErrorCode, execution_identity::Sha256Digest,
    local_io::{self, Destination}, textutil::{self, NormalizeOptions, SplitOptions, TextBudget, TextError},
    tokenizer::{bpe::EncodeOptions, embedded::{EmbeddedTokenizer, PINNED_ADDED_TOKENS_BYTES,
        PINNED_TOKENIZER_CONFIG_BYTES, PINNED_SPECIAL_TOKENS_MAP_BYTES}},
};

#[derive(Args)]
struct Common {
    /// Original UTF-8 document, or '-' for stdin. No implicit Unicode repair.
    #[arg(default_value = "-", value_name = "DOCUMENT")]
    document: PathBuf,
    /// Emit one canonical, versioned result envelope.
    #[arg(long)]
    json: bool,
    /// New protected output file; '-' means stdout. Existing targets are refused.
    #[arg(short = 'o', long)]
    output: Option<PathBuf>,
    #[arg(long, default_value_t = 1024 * 1024)]
    max_input_bytes: usize,
    /// Entire response including metadata, coordinates and record newlines.
    #[arg(long, default_value_t = 4 * 1024 * 1024)]
    max_output_bytes: usize,
    /// Maximum tokens, split chunks, or normalization edits, respectively.
    #[arg(long, default_value_t = 16_384)]
    max_items: usize,
}
impl Common {
    fn budget(&self) -> Result<TextBudget, Failure> {
        let budget = TextBudget { max_input_bytes: self.max_input_bytes, max_output_bytes: self.max_output_bytes, max_items: self.max_items };
        budget.validate()?;
        if self.file_output().is_some() && !local_io::PROFILE_SUPPORTED {
            return Err(Failure::new(ErrorCode::AdmissionOrResourceLimit, "private_file_profile_unavailable"));
        }
        Ok(budget)
    }
    fn file_output(&self) -> Option<&PathBuf> { self.output.as_ref().filter(|p| p.as_os_str() != "-") }
}
#[derive(Args)]
pub(crate) struct Tokens {
    #[command(flatten)] common: Common,
    /// Include exact token IDs. Without --json this emits one JSON ID array.
    #[arg(long)] ids: bool,
    /// Disable the pinned default BOS insertion for L0 BPE inspection.
    #[arg(long, conflicts_with = "byte_fallback")] no_bos: bool,
    /// Append the pinned EOS ID after L0 BPE encoding.
    #[arg(long, conflicts_with = "byte_fallback")] eos: bool,
    /// Explicit byte-table encoding (no BOS/EOS or added-token recognition).
    #[arg(long, conflicts_with_all = ["no_bos", "eos"])] byte_fallback: bool,
}
#[derive(Args)]
pub(crate) struct Split {
    #[command(flatten)] common: Common,
    /// Maximum UTF-8 bytes per chunk, at least 4. Whitespace stays in chunks.
    #[arg(long, default_value_t = 4096)] max_chunk_bytes: usize,
}
#[derive(Args)]
pub(crate) struct Normalize {
    #[command(flatten)] common: Common,
    /// Remove ASCII space/tab at each logical line's edges, not Unicode spaces.
    #[arg(long)] trim_ascii_horizontal: bool,
    /// Collapse retained runs of ASCII space/tab to a single ASCII space.
    #[arg(long)] collapse_ascii_horizontal: bool,
}
pub(crate) enum TextCommand { Tokens(Tokens), Split(Split), Normalize(Normalize) }

pub(crate) fn definitions() -> [clap::Command; 3] {
    [Tokens::augment_args(clap::Command::new("tokens"))
        .mut_arg("max_input_bytes", |a| a.default_value("4096"))
        .about("Inspect exact pinned tokenizer counts/IDs without model weights")
        .after_help("Default: L0 BPE, added-token recognition, BOS enabled, EOS disabled. This is tokenizer inspection, NOT untrusted prompt construction. The reference BPE path is capped at 4096 input bytes. --byte-fallback is a different explicit encoding with byte-exact decode and no special insertion; counts are not interchangeable."),
     Split::augment_args(clap::Command::new("split"))
        .about("Partition text without dropping bytes; original byte/scalar offsets survive")
        .after_help("Without --json, emit one versioned NDJSON row per chunk. With --json, emit one complete envelope. This is size/whitespace splitting, not linguistic sentence detection. Concatenating chunk text reproduces the exact source; CRLF pairs are not split."),
     Normalize::augment_args(clap::Command::new("normalize"))
        .about("Normalize CRLF/CR to LF, with explicit ASCII-only trim/collapse")
        .after_help("No NFC/NFKC, case folding or Unicode whitespace normalization. Plain output adds no newline. --json retains changed-run maps with both byte and Unicode-scalar coordinates; unchanged gaps are identity translations. Changed interiors and deleted-run inverse boundaries are ambiguous, not invented exact offsets.")]
}
impl TextCommand {
    pub(crate) fn recognizes(name: &str) -> bool { matches!(name, "tokens" | "split" | "normalize") }
    pub(crate) fn from_matches(name: &str, args: &ArgMatches) -> Result<Self, clap::Error> {
        match name {
            "tokens" => Tokens::from_arg_matches(args).map(Self::Tokens),
            "split" => Split::from_arg_matches(args).map(Self::Split),
            "normalize" => Normalize::from_arg_matches(args).map(Self::Normalize),
            _ => Err(clap::Error::raw(clap::error::ErrorKind::InvalidSubcommand, "unknown text utility command")),
        }
    }
    fn common(&self) -> &Common {
        match self { Self::Tokens(c) => &c.common, Self::Split(c) => &c.common, Self::Normalize(c) => &c.common }
    }
    fn validate(&self) -> Result<TextBudget, Failure> {
        let budget = self.common().budget()?;
        match self {
            Self::Tokens(c) if !c.byte_fallback && budget.max_input_bytes > 4096 =>
                Err(Failure::new(ErrorCode::AdmissionOrResourceLimit, "reference_bpe_input_limit_4096_bytes")),
            Self::Split(c) if !(4..=64 * 1024 * 1024).contains(&c.max_chunk_bytes) => Err(TextError::InvalidOptions.into()),
            _ => Ok(budget),
        }
    }
    fn render(&self, source: &str, budget: TextBudget) -> Result<Vec<u8>, Failure> {
        let json = self.common().json;
        let mut bytes = match self {
            Self::Tokens(options) => {
                let result = token_report(source, options, budget)?;
                if json { canonical(&result)? }
                else if let Some(ids) = &result.ids { canonical(ids)? }
                else { result.count.to_string().into_bytes() }
            }
            Self::Normalize(options) => {
                let result = textutil::normalize(source, NormalizeOptions {
                    trim_ascii_horizontal: options.trim_ascii_horizontal,
                    collapse_ascii_horizontal: options.collapse_ascii_horizontal,
                }, budget)?;
                if json { canonical(&result)? } else { result.text().as_bytes().to_vec() }
            }
            Self::Split(options) => {
                let result = textutil::split(source, SplitOptions { max_chunk_bytes: options.max_chunk_bytes }, budget)?;
                if json { canonical(&result)? } else {
                    let mut bytes = Vec::new();
                    for (index, chunk) in result.chunks().iter().enumerate() {
                        let row = ChunkRow { schema_version: 1, profile: textutil::SPLIT_PROFILE, index, chunk };
                        let mut encoded = canonical(&row)?; encoded.push(b'\n');
                        if bytes.len().checked_add(encoded.len()).is_none_or(|n| n > budget.max_output_bytes) {
                            return Err(TextError::OutputBudget.into());
                        }
                        bytes.try_reserve(encoded.len()).map_err(|_| TextError::AllocationRefused)?;
                        bytes.extend_from_slice(&encoded);
                    }
                    return Ok(bytes);
                }
            }
        };
        if json || matches!(self, Self::Tokens(_)) { bytes.push(b'\n'); }
        if bytes.len() > budget.max_output_bytes { return Err(TextError::OutputBudget.into()); }
        Ok(bytes)
    }
    fn emit(&self, input: &mut impl Read, out: &mut impl Write) -> Result<(), Failure> {
        let budget = self.validate()?; // Refuse invalid plans before consuming input.
        let common = self.common();
        let destination = common.file_output().map(|p| Destination::prepare(p)).transpose().map_err(file_failure)?;
        let source = if common.document.as_os_str() == "-" { textutil::read_utf8(input, budget.max_input_bytes)? }
            else {
                let mut file = local_io::open_document(&common.document)
                    .map_err(|_| Failure::new(ErrorCode::InputDecodeOrParse, "regular_document_open_failed"))?;
                textutil::read_utf8(&mut file, budget.max_input_bytes)?
            };
        let bytes = self.render(&source, budget)?;
        match destination {
            Some(destination) => destination.stage(&bytes).and_then(|stage| stage.publish()).map_err(file_failure),
            None => out.write_all(&bytes).and_then(|()| out.flush()).map_err(|_| Failure::new(ErrorCode::Generic, "output_write_failed")),
        }
    }
    pub(crate) fn run(self, input: &mut impl Read, out: &mut impl Write, err: &mut impl Write) -> ExitCode {
        match self.emit(input, out) {
            Ok(()) => ExitCode::SUCCESS,
            Err(failure) => {
                if let Ok(mut row) = canonjson::canonical_bytes(&failure) { row.push(b'\n'); let _ = err.write_all(&row).and_then(|()| err.flush()); }
                failure.code.as_process_exit()
            }
        }
    }
}

#[derive(Serialize)]
struct TokenReport {
    schema_version: u32, profile: &'static str,
    tokenizer_model_sha256: String, added_tokens_sha256: Sha256Digest,
    tokenizer_config_sha256: Sha256Digest, special_tokens_map_sha256: Sha256Digest,
    add_bos: bool, add_eos: bool, source_bytes: usize, source_scalars: usize,
    count: usize,
    #[serde(skip_serializing_if = "Option::is_none")] ids: Option<Vec<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")] byte_exact_decode_checked: Option<bool>,
}
fn token_report(source: &str, options: &Tokens, budget: TextBudget) -> Result<TokenReport, Failure> {
    if source.len() > budget.max_input_bytes { return Err(TextError::InputBudget.into()); }
    if !options.byte_fallback && source.len() > 4096 {
        return Err(Failure::new(ErrorCode::AdmissionOrResourceLimit, "reference_bpe_input_limit_4096_bytes"));
    }
    if options.byte_fallback && source.len() > budget.max_items { return Err(TextError::ItemBudget.into()); }
    let tokenizer = EmbeddedTokenizer::pinned().map_err(|_| Failure::new(ErrorCode::ArtifactIntegrityOrFormatOrVersion, "embedded_tokenizer_refused"))?;
    let add_bos = !options.byte_fallback && !options.no_bos; let add_eos = !options.byte_fallback && options.eos;
    let ids = if options.byte_fallback { tokenizer.tokenizer().encode_byte_fallback_only(source.as_bytes()) }
        else { tokenizer.tokenizer().encode_ids_with_options(source, EncodeOptions { add_bos, add_eos }) }
        .map_err(|_| Failure::new(ErrorCode::ArtifactIntegrityOrFormatOrVersion, "pinned_encoding_refused"))?;
    if ids.len() > budget.max_items { return Err(TextError::ItemBudget.into()); }
    if options.byte_fallback && tokenizer.tokenizer().decode_bytes(&ids).ok().as_deref() != Some(source.as_bytes()) {
        return Err(Failure::new(ErrorCode::ArtifactIntegrityOrFormatOrVersion, "byte_fallback_roundtrip_refused"));
    }
    Ok(TokenReport { schema_version: 1,
        profile: if options.byte_fallback { "byte-fallback-only-v1" } else { "pinned-sp-bpe-l0-v1" },
        tokenizer_model_sha256: tokenizer.sha256_hex(),
        added_tokens_sha256: Sha256Digest::of_bytes(PINNED_ADDED_TOKENS_BYTES),
        tokenizer_config_sha256: Sha256Digest::of_bytes(PINNED_TOKENIZER_CONFIG_BYTES),
        special_tokens_map_sha256: Sha256Digest::of_bytes(PINNED_SPECIAL_TOKENS_MAP_BYTES),
        add_bos, add_eos, source_bytes: source.len(), source_scalars: source.chars().count(), count: ids.len(),
        ids: options.ids.then_some(ids), byte_exact_decode_checked: options.byte_fallback.then_some(true),
    })
}
#[derive(Serialize)]
struct ChunkRow<'a, 'b> { schema_version: u32, profile: &'static str, index: usize, chunk: &'a textutil::TextChunk<'b> }
fn canonical(value: &impl Serialize) -> Result<Vec<u8>, Failure> {
    canonjson::canonical_bytes(value).map_err(|_| TextError::Serialization.into())
}
#[derive(Serialize)]
struct Failure { schema_version: u32, event: &'static str, code: ErrorCode, reason: &'static str }
impl Failure { fn new(code: ErrorCode, reason: &'static str) -> Self { Self { schema_version: 1, event: "text_error", code, reason } } }
impl From<TextError> for Failure {
    fn from(error: TextError) -> Self {
        let (code, reason) = match error {
            TextError::InvalidOptions => (ErrorCode::Usage, "invalid_text_options"),
            TextError::InputBudget => (ErrorCode::AdmissionOrResourceLimit, "input_byte_budget"),
            TextError::OutputBudget => (ErrorCode::AdmissionOrResourceLimit, "complete_output_byte_budget"),
            TextError::ItemBudget => (ErrorCode::AdmissionOrResourceLimit, "text_item_budget"),
            TextError::AllocationRefused => (ErrorCode::AdmissionOrResourceLimit, "text_allocation_refused"),
            TextError::InputRead => (ErrorCode::InputDecodeOrParse, "text_input_read_failed"),
            TextError::InvalidUtf8 => (ErrorCode::InputDecodeOrParse, "text_input_not_utf8"),
            TextError::Serialization => (ErrorCode::Generic, "text_serialization_failed"),
        }; Self::new(code, reason)
    }
}
fn file_failure(error: local_io::LocalIoError) -> Failure {
    let reason = match error {
        local_io::LocalIoError::PublicationUncertain => "output_may_be_published_but_sync_or_cleanup_failed",
        local_io::LocalIoError::AlreadyExists => "destination_already_exists",
        _ => "protected_output_refused",
    };
    Failure::new(ErrorCode::AdmissionOrResourceLimit, reason)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn command(args: &[&str]) -> TextCommand {
        let definition = definitions().into_iter().find(|c| c.get_name() == args[0]).unwrap();
        let matches = definition.try_get_matches_from(args).unwrap();
        TextCommand::from_matches(args[0], &matches).unwrap()
    }
    fn run(args: &[&str], source: &[u8]) -> (ExitCode, Vec<u8>, Vec<u8>) {
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let code = command(args).run(&mut &*source, &mut out, &mut err); (code, out, err)
    }
    #[test]
    fn normalize_json_exposes_maps_and_plain_mode_keeps_final_newline_exact() {
        let (code, out, err) = run(&["normalize"], b"a\r\nb");
        assert_eq!(code, ExitCode::SUCCESS); assert_eq!(out, b"a\nb"); assert!(err.is_empty());
        let (_, out, _) = run(&["normalize", "--json"], b"a\r\nb");
        let v: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["text"], "a\nb"); assert_eq!(v["edits"].as_array().unwrap().len(), 1);
    }
    #[test]
    fn split_ndjson_and_envelope_reconstruct_the_identical_source() {
        let source = "é a\r\n上海 end";
        for json in [false, true] {
            let mut args = vec!["split", "--max-chunk-bytes", "5"]; if json { args.push("--json"); }
            let (code, out, err) = run(&args, source.as_bytes()); assert_eq!(code, ExitCode::SUCCESS); assert!(err.is_empty());
            let chunks: Vec<serde_json::Value> = if json { serde_json::from_slice::<serde_json::Value>(&out).unwrap()["chunks"].as_array().unwrap().clone() }
                else { String::from_utf8(out).unwrap().lines().map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap()["chunk"].clone()).collect() };
            assert_eq!(chunks.iter().map(|c| c["text"].as_str().unwrap()).collect::<String>(), source);
        }
    }
    #[test]
    fn reference_input_cap_is_refused_before_reading() {
        struct NoRead;
        impl Read for NoRead { fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> { panic!("must refuse first"); } }
        let mut out = Vec::new(); let mut err = Vec::new();
        assert_ne!(command(&["tokens", "--max-input-bytes", "4097"]).run(&mut NoRead, &mut out, &mut err), ExitCode::SUCCESS);
        assert!(out.is_empty());
    }
    #[test]
    fn token_profile_preserves_explicit_bos_eos_semantics() {
        let count = |args: &[&str]| {
            let (code, out, err) = run(args, b"Hello"); assert_eq!(code, ExitCode::SUCCESS); assert!(err.is_empty());
            serde_json::from_slice::<serde_json::Value>(&out).unwrap()["count"].as_u64().unwrap()
        };
        let plain = count(&["tokens", "--json", "--no-bos"]);
        assert_eq!(count(&["tokens", "--json"]), plain + 1);
        assert_eq!(count(&["tokens", "--json", "--eos"]), plain + 2);
    }
    #[test]
    fn byte_fallback_is_explicit_and_checked_against_source_bytes() {
        let source = "上海 <|im_start|>";
        let (code, out, err) = run(&["tokens", "--byte-fallback", "--json", "--ids"], source.as_bytes());
        assert_eq!(code, ExitCode::SUCCESS); assert!(err.is_empty());
        let v: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["count"], source.len()); assert_eq!(v["ids"].as_array().unwrap().len(), source.len());
        assert_eq!(v["add_bos"], false); assert_eq!(v["byte_exact_decode_checked"], true);
    }
    #[test]
    fn invalid_input_or_output_budget_never_emits_partial_records() {
        for (args, source) in [(vec!["normalize"], b"private\xff".as_slice()),
            (vec!["split", "--max-output-bytes", "1"], b"secret".as_slice())] {
            let (code, out, err) = run(&args, source); assert_ne!(code, ExitCode::SUCCESS); assert!(out.is_empty());
            let log = String::from_utf8(err).unwrap(); assert!(!log.contains("private")); assert!(!log.contains("secret"));
        }
    }
}
