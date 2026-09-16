//! Model-free text transforms. No Unicode normalization, case folding, fuzzy
//! relocation or linguistic sentence-boundary claim is hidden in these APIs.
use std::{error::Error, fmt, io::Read};
use serde::{Deserialize, Serialize};
use crate::{canonjson, validation::grounded_fields::VerifiedSourceSpan};

pub const NORMALIZE_PROFILE: &str = "crlf-cr-lf-ascii-horizontal-v1";
pub const SPLIT_PROFILE: &str = "utf8-whitespace-partition-v1";
const MAX_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TextError {
    InvalidOptions, InputBudget, OutputBudget, ItemBudget, AllocationRefused,
    InputRead, InvalidUtf8, Serialization,
}
impl fmt::Display for TextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidOptions => "invalid text utility options", Self::InputBudget => "text input byte budget exceeded",
            Self::OutputBudget => "complete text result byte budget exceeded", Self::ItemBudget => "text item budget exceeded",
            Self::AllocationRefused => "text allocation refused", Self::InputRead => "text input read failed",
            Self::InvalidUtf8 => "text input is not UTF-8", Self::Serialization => "text result serialization failed",
        })
    }
}
impl Error for TextError {}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct TextBudget { pub max_input_bytes: usize, pub max_output_bytes: usize, pub max_items: usize }
impl Default for TextBudget {
    fn default() -> Self { Self { max_input_bytes: 1024 * 1024, max_output_bytes: 4 * 1024 * 1024, max_items: 16_384 } }
}
impl TextBudget {
    pub fn validate(self) -> Result<(), TextError> {
        if self.max_input_bytes > MAX_BYTES || self.max_output_bytes == 0 || self.max_output_bytes > MAX_BYTES
            || self.max_items > 1_000_000 { return Err(TextError::InvalidOptions); }
        Ok(())
    }
    fn admit(self, source: &str) -> Result<(), TextError> {
        self.validate()?;
        if source.len() > self.max_input_bytes { return Err(TextError::InputBudget); }
        Ok(())
    }
    fn result(self, value: &impl Serialize) -> Result<(), TextError> {
        if canonjson::canonical_bytes(value).map_err(|_| TextError::Serialization)?.len() > self.max_output_bytes {
            return Err(TextError::OutputBudget);
        }
        Ok(())
    }
}

/// Bounded UTF-8 admission. Consume at most cap+1 bytes, retry Interrupted, and
/// never return a truncated document or silently replace malformed UTF-8.
pub fn read_utf8(reader: &mut impl Read, cap: usize) -> Result<String, TextError> {
    if cap > MAX_BYTES { return Err(TextError::InvalidOptions); }
    let mut bytes = Vec::new(); let mut scratch = [0_u8; 8192];
    loop {
        let width = scratch.len().min(cap - bytes.len() + 1);
        let count = match reader.read(&mut scratch[..width]) {
            Ok(0) => break, Ok(n) if n <= width => n, Ok(_) => return Err(TextError::InputRead),
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => return Err(TextError::InputRead),
        };
        if count > cap - bytes.len() { return Err(TextError::InputBudget); }
        bytes.try_reserve(count).map_err(|_| TextError::AllocationRefused)?;
        bytes.extend_from_slice(&scratch[..count]);
    }
    String::from_utf8(bytes).map_err(|_| TextError::InvalidUtf8)
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NormalizeOptions {
    /// Trim only ASCII space and tab at each logical line's edges.
    pub trim_ascii_horizontal: bool,
    /// Replace a retained run of ASCII space/tab with one ASCII space.
    pub collapse_ascii_horizontal: bool,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NormalizationKind { LineEnding, TrimAsciiHorizontal, CollapseAsciiHorizontal }
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct NormalizationEdit {
    pub original: VerifiedSourceSpan,
    pub normalized: VerifiedSourceSpan,
    pub kind: NormalizationKind,
}

/// Maps are minted with their exact original and transformed strings. The
/// private source is not serialized; edits contain coordinates, not raw values.
/// Interior boundaries of changed runs have no invented point correspondence.
#[derive(Serialize)]
pub struct NormalizedText<'a> {
    schema_version: u32, profile: &'static str, options: NormalizeOptions,
    #[serde(skip)] source: &'a str,
    original_bytes: usize, original_scalars: usize,
    text: String, edits: Vec<NormalizationEdit>,
}
impl NormalizedText<'_> {
    pub fn text(&self) -> &str { &self.text }
    pub fn edits(&self) -> &[NormalizationEdit] { &self.edits }
    pub fn original_to_normalized(&self, byte: usize) -> Option<usize> {
        self.map_boundary(byte, false)
    }
    pub fn normalized_to_original(&self, byte: usize) -> Option<usize> {
        self.map_boundary(byte, true)
    }
    fn map_boundary(&self, byte: usize, reverse: bool) -> Option<usize> {
        let (from, to) = if reverse { (self.text.as_str(), self.source) } else { (self.source, self.text.as_str()) };
        if !from.is_char_boundary(byte) { return None; }
        let mut previous = (0, 0); let mut candidate = None;
        for edit in &self.edits {
            let (a, b) = if reverse { (edit.normalized, edit.original) } else { (edit.original, edit.normalized) };
            if byte < a.byte_start { break; }
            if byte > a.byte_end { previous = (a.byte_end, b.byte_end); continue; }
            // A deletion's inverse is an interval, not a uniquely chosen point.
            if a.byte_start == a.byte_end || (byte > a.byte_start && byte < a.byte_end) { return None; }
            let mapped = if byte == a.byte_start { b.byte_start } else { b.byte_end };
            if candidate.is_some_and(|old| old != mapped) { return None; }
            candidate = Some(mapped);
        }
        let mapped = candidate.or_else(|| byte.checked_sub(previous.0).and_then(|delta| previous.1.checked_add(delta)))?;
        to.is_char_boundary(mapped).then_some(mapped)
    }
}

/// Always convert CRLF/CR to LF. Horizontal trim/collapse is opt-in; all other
/// Unicode scalars, including NBSP, combining marks and zero-width text, survive.
pub fn normalize(source: &str, options: NormalizeOptions, budget: TextBudget) -> Result<NormalizedText<'_>, TextError> {
    budget.admit(source)?;
    let mut text = String::new();
    text.try_reserve_exact(source.len().min(budget.max_output_bytes)).map_err(|_| TextError::AllocationRefused)?;
    let mut edits = Vec::new(); let mut edit_bytes = 0_usize;
    let (mut byte, mut original_scalar, mut normalized_scalar) = (0, 0, 0);
    let mut line_start = true;
    while byte < source.len() {
        let start = byte;
        let (replacement, kind) = match source.as_bytes()[byte] {
            b'\r' => {
                byte += 1; if source.as_bytes().get(byte) == Some(&b'\n') { byte += 1; }
                line_start = true; ("\n", NormalizationKind::LineEnding)
            }
            b' ' | b'\t' => {
                while source.as_bytes().get(byte).is_some_and(|b| matches!(b, b' ' | b'\t')) { byte += 1; }
                let trailing = byte == source.len() || matches!(source.as_bytes()[byte], b'\r' | b'\n');
                if options.trim_ascii_horizontal && (line_start || trailing) { ("", NormalizationKind::TrimAsciiHorizontal) }
                else if options.collapse_ascii_horizontal { (" ", NormalizationKind::CollapseAsciiHorizontal) }
                else { (&source[start..byte], NormalizationKind::CollapseAsciiHorizontal) }
            }
            _ => {
                let ch = source[byte..].chars().next().ok_or(TextError::InvalidUtf8)?;
                byte += ch.len_utf8(); line_start = ch == '\n';
                (&source[start..byte], NormalizationKind::LineEnding)
            }
        };
        let original = &source[start..byte];
        let original_end = original_scalar + original.chars().count();
        let normalized_end = normalized_scalar + replacement.chars().count();
        let output_end = text.len().checked_add(replacement.len()).filter(|&n| n <= budget.max_output_bytes)
            .ok_or(TextError::OutputBudget)?;
        if original != replacement {
            if edits.len() == budget.max_items { return Err(TextError::ItemBudget); }
            edits.try_reserve(1).map_err(|_| TextError::AllocationRefused)?;
            let edit = NormalizationEdit {
                original: VerifiedSourceSpan { byte_start: start, byte_end: byte, scalar_start: original_scalar, scalar_end: original_end },
                normalized: VerifiedSourceSpan { byte_start: text.len(), byte_end: output_end, scalar_start: normalized_scalar, scalar_end: normalized_end }, kind,
            };
            edit_bytes = edit_bytes.checked_add(canonjson::canonical_bytes(&edit).map_err(|_| TextError::Serialization)?.len())
                .filter(|&n| n <= budget.max_output_bytes).ok_or(TextError::OutputBudget)?;
            edits.push(edit);
        }
        text.try_reserve(replacement.len()).map_err(|_| TextError::AllocationRefused)?;
        text.push_str(replacement); original_scalar = original_end; normalized_scalar = normalized_end;
    }
    let result = NormalizedText { schema_version: 1, profile: NORMALIZE_PROFILE, options, source,
        original_bytes: source.len(), original_scalars: original_scalar, text, edits };
    budget.result(&result)?; Ok(result)
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SplitOptions { pub max_chunk_bytes: usize }
impl Default for SplitOptions { fn default() -> Self { Self { max_chunk_bytes: 4096 } } }
#[derive(Clone, Debug, Serialize)]
pub struct TextChunk<'a> { pub span: VerifiedSourceSpan, pub text: &'a str }
#[derive(Serialize)]
pub struct SplitText<'a> {
    schema_version: u32, profile: &'static str, options: SplitOptions,
    original_bytes: usize, original_scalars: usize, chunks: Vec<TextChunk<'a>>,
}
impl<'a> SplitText<'a> { pub fn chunks(&self) -> &[TextChunk<'a>] { &self.chunks } }

/// Lossless contiguous partition, not a semantic sentence segmenter. Prefer
/// the last ASCII whitespace boundary in the latter half of a bounded chunk;
/// otherwise cut at a UTF-8 boundary. CRLF pairs stay together. No byte is
/// trimmed, duplicated or synthesized, and exhaustion fails the whole result.
pub fn split(source: &str, options: SplitOptions, budget: TextBudget) -> Result<SplitText<'_>, TextError> {
    budget.admit(source)?;
    if !(4..=MAX_BYTES).contains(&options.max_chunk_bytes) { return Err(TextError::InvalidOptions); }
    if source.len() > budget.max_output_bytes { return Err(TextError::OutputBudget); }
    let mut chunks = Vec::new(); let mut chunk_bytes = 0_usize; let (mut start, mut scalar) = (0, 0);
    while start < source.len() {
        if chunks.len() == budget.max_items { return Err(TextError::ItemBudget); }
        let mut end = start + options.max_chunk_bytes.min(source.len() - start);
        while !source.is_char_boundary(end) { end -= 1; }
        if end < source.len() {
            // Never cut a CRLF pair, even when it straddles the size boundary.
            if source.as_bytes()[end - 1] == b'\r' && source.as_bytes()[end] == b'\n' { end -= 1; }
            let preferred = source.as_bytes()[start..end].iter().enumerate().rev().find_map(|(i, &b)| {
                let point = start + i + 1;
                (point >= start + options.max_chunk_bytes / 2 && matches!(b, b' ' | b'\t' | b'\n' | b'\r')
                    && !(b == b'\r' && source.as_bytes().get(point) == Some(&b'\n'))).then_some(point)
            });
            if let Some(point) = preferred { end = point; }
        }
        if end <= start { return Err(TextError::InvalidOptions); }
        let text = &source[start..end]; let next_scalar = scalar + text.chars().count();
        chunks.try_reserve(1).map_err(|_| TextError::AllocationRefused)?;
        let chunk = TextChunk { span: VerifiedSourceSpan { byte_start: start, byte_end: end, scalar_start: scalar, scalar_end: next_scalar }, text };
        chunk_bytes = chunk_bytes.checked_add(canonjson::canonical_bytes(&chunk).map_err(|_| TextError::Serialization)?.len())
            .filter(|&n| n <= budget.max_output_bytes).ok_or(TextError::OutputBudget)?;
        chunks.push(chunk);
        start = end; scalar = next_scalar;
    }
    let result = SplitText { schema_version: 1, profile: SPLIT_PROFILE, options,
        original_bytes: source.len(), original_scalars: scalar, chunks };
    budget.result(&result)?; Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn default_normalization_changes_only_line_endings() {
        let source = "é\r\n\t e\u{301}\u{a0}\u{200d} \r𐀀\n";
        let result = normalize(source, NormalizeOptions::default(), TextBudget::default()).unwrap();
        assert_eq!(result.text(), "é\n\t e\u{301}\u{a0}\u{200d} \n𐀀\n");
        assert_eq!(result.edits().len(), 2);
        for edit in result.edits() { assert_eq!(edit.kind, NormalizationKind::LineEnding); }
    }
    #[test]
    fn opt_in_ascii_trim_and_collapse_preserve_non_ascii_spaces() {
        let result = normalize(" \tA\t  B  \r\n \t\rC\u{a0}  ", NormalizeOptions {
            trim_ascii_horizontal: true, collapse_ascii_horizontal: true,
        }, TextBudget::default()).unwrap();
        assert_eq!(result.text(), "A B\n\nC\u{a0}");
        let again = normalize(result.text(), NormalizeOptions { trim_ascii_horizontal: true, collapse_ascii_horizontal: true }, TextBudget::default()).unwrap();
        assert_eq!(again.text(), result.text()); assert!(again.edits().is_empty());
    }
    #[test]
    fn offset_maps_never_invent_changed_interior_or_deleted_inverse() {
        let result = normalize("é\r\n  X", NormalizeOptions { trim_ascii_horizontal: true, collapse_ascii_horizontal: false }, TextBudget::default()).unwrap();
        assert_eq!(result.text(), "é\nX");
        assert_eq!(result.original_to_normalized(1), None); // Interior UTF-8 byte.
        assert_eq!(result.original_to_normalized(3), None); // Interior of CRLF.
        assert_eq!(result.original_to_normalized(4), Some(3));
        assert_eq!(result.original_to_normalized(6), Some(3));
        assert_eq!(result.normalized_to_original(3), None); // Deleted run has two endpoints.
        assert_eq!(result.normalized_to_original(4), Some(7));
    }
    #[test]
    fn untouched_boundaries_roundtrip_in_both_coordinate_spaces() {
        let source = "é\r\n上海\r𐀀";
        let result = normalize(source, NormalizeOptions::default(), TextBudget::default()).unwrap();
        for byte in source.char_indices().map(|(i, _)| i).chain(std::iter::once(source.len())) {
            if let Some(mapped) = result.original_to_normalized(byte) { assert_eq!(result.normalized_to_original(mapped), Some(byte)); }
        }
    }
    #[test]
    fn split_reconstructs_source_with_exact_unicode_coordinates() {
        for source in ["", "abc def\r\nmore", "𐀀上海é\u{301} whitespace\t尾", "a\r\nb\r\nc\r\n"] {
            for width in 4..=19 {
                let result = split(source, SplitOptions { max_chunk_bytes: width }, TextBudget::default()).unwrap();
                assert_eq!(result.chunks().iter().map(|c| c.text).collect::<String>(), source);
                for c in result.chunks() {
                    assert!(c.text.len() <= width && !c.text.is_empty());
                    assert_eq!(&source[c.span.byte_start..c.span.byte_end], c.text);
                    assert_eq!(source[..c.span.byte_start].chars().count(), c.span.scalar_start);
                    assert_eq!(source[..c.span.byte_end].chars().count(), c.span.scalar_end);
                    assert!(!(source[..c.span.byte_end].ends_with('\r') && source[c.span.byte_end..].starts_with('\n')));
                }
            }
        }
    }
    #[test]
    fn count_and_complete_envelope_budgets_fail_without_truncating() {
        let budget = TextBudget { max_items: 0, ..TextBudget::default() };
        assert!(matches!(normalize("\r", NormalizeOptions::default(), budget), Err(TextError::ItemBudget)));
        assert!(matches!(split("document", SplitOptions::default(), budget), Err(TextError::ItemBudget)));
        let budget = TextBudget { max_output_bytes: 1, ..TextBudget::default() };
        assert!(matches!(normalize("", NormalizeOptions::default(), budget), Err(TextError::OutputBudget)));
        assert!(matches!(split("", SplitOptions::default(), budget), Err(TextError::OutputBudget)));
    }
    #[test]
    fn reader_retries_interruptions_and_stops_one_byte_after_limit() {
        struct Interrupted { first: bool, input: std::io::Cursor<Vec<u8>> }
        impl Read for Interrupted {
            fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
                if !self.first { self.first = true; return Err(std::io::ErrorKind::Interrupted.into()); }
                self.input.read(out)
            }
        }
        let mut input = Interrupted { first: false, input: std::io::Cursor::new(b"1234567".to_vec()) };
        assert_eq!(read_utf8(&mut input, 4), Err(TextError::InputBudget)); assert_eq!(input.input.position(), 5);
        assert_eq!(read_utf8(&mut &b"\xff"[..], 1), Err(TextError::InvalidUtf8));
    }
}
