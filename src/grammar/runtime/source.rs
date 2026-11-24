//! Source × canonical JSON composition. The source cursor consumes logical
//! unescaped bytes, including UTF-8 pieces split across model token boundaries.
//! Canonical escape prefixes are checked for a real source continuation before
//! they enter a vocabulary mask. Final evidence is produced independently.

use serde::{Deserialize, Serialize};
use super::*;
use crate::grammar::source_index::{SourceBuildError, SourceLanguageLimits};
use crate::validation::grounded_fields::{
    EvidenceMode, GroundingBudget, GroundingScanError, SourceFieldEvidence, verify_source_fields_with_checkpoints,
};

pub(crate) enum SourceValidationError<E> { Schema(SchemaError), Cancelled(E) }

pub const SOURCE_JSON_RUNTIME_VERSION: &str = "canonical-source-json-runtime-v1";

/// Source compilation keeps request cancellation distinct from an invalid
/// schema or a source-language resource refusal.
#[derive(Debug)]
pub enum SourceCompileError<E> { Schema(SchemaError), Cancelled(E) }
impl<E> std::fmt::Display for SourceCompileError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Schema(error) => write!(f, "{error}"),
            Self::Cancelled(_) => f.write_str("source JSON compilation cancelled"),
        }
    }
}
impl<E: std::fmt::Debug> std::error::Error for SourceCompileError<E> {}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SourceRuntimeLimits {
    pub index: SourceLanguageLimits,
    pub verification: GroundingBudget,
}
#[derive(Clone, Debug)]
pub(super) struct SourceBinding {
    pub(super) language: SourceLanguage,
    pub(super) limits: SourceRuntimeLimits,
}

impl JsonProgram {
    /// Bind original source bytes before decoding. This does not accept an
    /// external authority digest in place of the actual source. Callers must
    /// separately bind these bytes to their prompt and ExecutionIdentity.
    pub fn compile_with_source(
        schema: &str, document: &str, limits: CompileLimits, source_limits: SourceRuntimeLimits,
    ) -> Result<Self, SchemaError> {
        match Self::compile_with_source_control(schema, document, limits, source_limits,
            &mut || None::<std::convert::Infallible>) {
            Ok(program) => Ok(program),
            Err(SourceCompileError::Schema(error)) => Err(error),
            Err(SourceCompileError::Cancelled(never)) => match never {},
        }
    }

    /// Source counting, copying, sorting and index filtering poll the original
    /// request callback at bounded intervals. Schema compilation is separately
    /// bounded by CompileLimits and bracketed by that same callback.
    pub fn compile_with_source_control<E, F: FnMut() -> Option<E>>(
        schema: &str, document: &str, limits: CompileLimits, mut source_limits: SourceRuntimeLimits,
        checkpoint: &mut F,
    ) -> Result<Self, SourceCompileError<E>> {
        if let Some(cause) = checkpoint() { return Err(SourceCompileError::Cancelled(cause)); }
        if schema.len() > limits.max_schema_bytes {
            return Err(SourceCompileError::Schema(resource("schema byte limit exceeded")));
        }
        source_limits.verification.max_matches = source_limits.verification.max_matches.min(source_limits.index.max_matches);
        let language = SourceLanguage::build_with_control(document, source_limits.index, checkpoint)
            .map_err(|error| match error {
                SourceBuildError::Source(error) => SourceCompileError::Schema(resource(&error.to_string())),
                SourceBuildError::Cancelled(cause) => SourceCompileError::Cancelled(cause),
            })?;
        let program = Self::compile_bound(schema, limits, Some(SourceBinding { language, limits: source_limits }));
        if let Some(cause) = checkpoint() { return Err(SourceCompileError::Cancelled(cause)); }
        program.map_err(SourceCompileError::Schema)
    }

    #[must_use]
    pub fn version(&self) -> &'static str {
        if self.source.is_some() { SOURCE_JSON_RUNTIME_VERSION } else { JSON_RUNTIME_VERSION }
    }
    #[must_use]
    pub fn source_limits(&self) -> Option<SourceRuntimeLimits> { self.source.as_ref().map(|b| b.limits) }
    #[must_use]
    pub fn requires_source(&self) -> bool { self.schema.requires_verbatim_source() }

    /// Borrow immutable structural/schema annotations for downstream field
    /// traversal. This exposes no automaton state, acceptance flag or source
    /// index, and does not replace validate_json (including scalar-length caps).
    #[must_use]
    pub fn declarative_schema(&self) -> &SchemaNode { self.schema.root() }

    /// Return all exact occurrences using the validation-owned matcher, not
    /// the suffix index used to constrain token selection. Failure is atomic.
    pub fn source_fields(&self, text: &str) -> Result<Vec<SourceFieldEvidence>, SchemaError> {
        match self.check_fields(text, EvidenceMode::Recover, || Ok::<(), std::convert::Infallible>(())) {
            Ok(fields) => Ok(fields),
            Err(SourceValidationError::Schema(error)) => Err(error),
            Err(SourceValidationError::Cancelled(never)) => match never {},
        }
    }
    /// Parse and validate the complete value, preserving the request's cancellation
    /// cause across the independent occurrence census for every bound field.
    pub(crate) fn validate_json_with_checkpoints<E>(&self, text: &str,
        checkpoint: impl FnMut() -> Result<(), E>) -> Result<(), SourceValidationError<E>> {
        self.check_fields(text, EvidenceMode::CheckOnly, checkpoint).map(drop)
    }

    fn check_fields<E>(&self, text: &str, mode: EvidenceMode,
        mut checkpoint: impl FnMut() -> Result<(), E>) -> Result<Vec<SourceFieldEvidence>, SourceValidationError<E>> {
        checkpoint().map_err(SourceValidationError::Cancelled)?;
        let value = self.validated_value(text).map_err(SourceValidationError::Schema)?;
        let Some(binding) = &self.source else {
            checkpoint().map_err(SourceValidationError::Cancelled)?;
            return Ok(Vec::new());
        };
        verify_source_fields_with_checkpoints(self.schema.root(), &value, binding.language.source(),
            binding.limits.verification, mode, checkpoint).map_err(|error| match error {
                GroundingScanError::Checkpoint(cause) => SourceValidationError::Cancelled(cause),
                GroundingScanError::Verification(error) => SourceValidationError::Schema(SchemaError::Validation {
                    pointer: "$".to_owned(), reason: error.to_string(),
                }),
            })
    }
}

const SHORT: [u8; 7] = [b'"', b'\\', 8, 9, 10, 12, 13];
fn unicode_control(byte: u8) -> bool { byte < 32 && ![8, 9, 10, 12, 13].contains(&byte) }
fn has_unicode(cursor: &SourceCursor<'_>, high: Option<u8>) -> bool {
    (0_u8..32).any(|b| unicode_control(b) && high.is_none_or(|h| b >> 4 == h) && cursor.can_push(b))
}
pub(super) fn escape_tail(cursor: &SourceCursor<'_>) -> Option<usize> {
    if SHORT.iter().any(|&b| cursor.can_push(b)) { Some(1) }
    else if has_unicode(cursor, None) { Some(5) } else { None }
}
pub(super) fn advance(cursor: &mut SourceCursor<'_>, mode: StringMode, byte: u8) -> bool {
    match mode {
        StringMode::Plain => match byte {
            b'"' => cursor.is_accepting(),
            b'\\' => escape_tail(cursor).is_some(),
            _ => cursor.push_byte(byte), // the JSON lexer checks the UTF-8 domain
        },
        StringMode::Escape => match byte {
            b'"' | b'\\' => cursor.push_byte(byte),
            b'b' => cursor.push_byte(8), b't' => cursor.push_byte(9),
            b'n' => cursor.push_byte(10), b'f' => cursor.push_byte(12), b'r' => cursor.push_byte(13),
            b'u' => has_unicode(cursor, None),
            _ => false,
        },
        StringMode::Unicode { position, high } => match (position, byte) {
            (0 | 1, b'0') => true,
            (2, b'0' | b'1') => has_unicode(cursor, Some(byte - b'0')),
            (3, b'0'..=b'9' | b'a'..=b'f') => {
                let low = if byte <= b'9' { byte - b'0' } else { byte - b'a' + 10 };
                let logical = high * 16 + low;
                unicode_control(logical) && cursor.push_byte(logical)
            }
            _ => false,
        },
        StringMode::Utf8 { .. } => cursor.push_byte(byte),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const STRING: &str = r#"{"type":"string","x-fnlp-source":"verbatim"}"#;
    fn program(schema: &str, source: &str) -> JsonProgram {
        JsonProgram::compile_with_source(schema, source, CompileLimits::default(), SourceRuntimeLimits::default()).unwrap()
    }
    fn accepts(p: &JsonProgram, text: &str) -> bool {
        let mut s = p.initial_state(); s.consume_bytes(text.as_bytes()) && s.is_accepting()
    }
    #[test]
    fn all_substrings_survive_json_escaping_and_utf8_token_splits() {
        let source = "é\n\"\\\u{1}上海𐀀";
        let p = program(STRING, source);
        let bounds: Vec<_> = source.char_indices().map(|(i, _)| i).chain(std::iter::once(source.len())).collect();
        for &a in &bounds {
            for &b in bounds.iter().filter(|&&b| b >= a) {
                let text = escape_json_string(&source[a..b]);
                let mut s = p.initial_state();
                for byte in text.bytes() { assert!(s.consume(byte), "{text:?}"); }
                assert!(s.is_accepting()); p.validate_json(&text).unwrap();
            }
        }
    }
    #[test]
    fn invented_text_cannot_enter_the_mask_or_final_result() {
        let p = program(STRING, "Alice Bob");
        for text in ["\"Carol\"", "\"alice\"", "\"AliceBob\""] {
            assert!(!accepts(&p, text)); assert!(p.validate_json(text).is_err());
        }
    }
    #[test]
    fn impossible_escape_prefixes_are_rejected_immediately() {
        let p = program(STRING, "abc"); assert!(!p.initial_state().consume_bytes(b"\"\\"));
        let p = program(STRING, "\n"); assert!(!p.initial_state().consume_bytes(b"\"\\u"));
        let p = program(STRING, "\u{1}"); assert!(!p.initial_state().consume_bytes(b"\"\\u001"));
        assert!(accepts(&p, r#""\u0001""#));
    }
    #[test]
    fn byte_budget_reserves_a_real_source_escape_completion() {
        let p = JsonProgram::compile_with_source(STRING, "\u{1}", CompileLimits { max_output_bytes: 7, ..CompileLimits::default() }, SourceRuntimeLimits::default()).unwrap();
        assert!(!p.initial_state().consume_bytes(b"\"\\"));
        assert!(accepts(&p, "\"\""));
    }
    #[test]
    fn arrays_reset_the_source_cursor_and_preserve_ambiguity() {
        let p = program(r#"{"type":"array","items":{"type":"string","x-fnlp-source":"verbatim"},"maxItems":3}"#, "Alice Bob Alice");
        let text = r#"["Alice","Bob"]"#; assert!(accepts(&p, text));
        let e = p.source_fields(text).unwrap();
        assert_eq!(e[0].json_pointer, "/0"); assert_eq!(e[0].spans.len(), 2);
        assert_eq!(e[1].spans.len(), 1);
    }
    #[test]
    fn ordinary_fields_are_not_accidentally_source_constrained() {
        let p = program(r#"{"type":"object","additionalProperties":false,"properties":{"a":{"type":"string","x-fnlp-source":"verbatim"},"b":{"type":"string"}},"required":["a","b"]}"#, "Alice");
        let text = r#"{"a":"Alice","b":"person"}"#;
        assert!(accepts(&p, text)); assert_eq!(p.source_fields(text).unwrap().len(), 1);
    }
    #[test]
    fn enums_intersect_the_source_language_without_free_generation() {
        let p = program(r#"{"type":"string","enum":["Alice","Bob"],"x-fnlp-source":"verbatim"}"#, "Alice");
        assert!(accepts(&p, "\"Alice\"")); assert!(!accepts(&p, "\"Bob\""));
        assert!(!accepts(&p, "\"Ali\""));
    }
    #[test]
    fn source_bindings_have_a_distinct_runtime_identity() {
        assert_eq!(program(STRING, "Alice").version(), SOURCE_JSON_RUNTIME_VERSION);
        assert_eq!(JsonProgram::compile(r#"{"type":"string"}"#, CompileLimits::default()).unwrap().version(), JSON_RUNTIME_VERSION);
        assert!(JsonProgram::compile(STRING, CompileLimits::default()).is_err());
    }
    #[test]
    fn source_compile_preserves_cancellation_and_language_authority() {
        let source = "ébanana".repeat(1025);
        let expected = program(STRING, &source);
        let mut calls = 0;
        let actual = JsonProgram::compile_with_source_control(STRING, &source,
            CompileLimits::default(), SourceRuntimeLimits::default(), &mut || {
                calls += 1; None::<u8>
            }).unwrap();
        assert!(calls > 20);
        assert_eq!(expected.sampling_digest(), actual.sampling_digest());
        assert_eq!(expected.source_fields("\"ébanana\"").unwrap(), actual.source_fields("\"ébanana\"").unwrap());
        for cancel_at in [1, calls / 2, calls] {
            let mut observed = 0;
            let result = JsonProgram::compile_with_source_control(STRING, &source,
                CompileLimits::default(), SourceRuntimeLimits::default(), &mut || {
                    observed += 1; (observed == cancel_at).then_some(7_u8)
                });
            assert!(matches!(result, Err(SourceCompileError::Cancelled(7))));
            assert_eq!(observed, cancel_at);
        }
    }
    #[test]
    fn scalar_caps_and_empty_sources_remain_exact() {
        let p = program(r#"{"type":"string","maxLength":1,"x-fnlp-source":"verbatim"}"#, "é𐀀");
        assert!(accepts(&p, "\"é\"")); assert!(!accepts(&p, "\"é𐀀\""));
        let p = program(STRING, ""); assert!(accepts(&p, "\"\"")); assert!(!accepts(&p, "\"a\""));
    }
}
