//! Independent source membership and occurrence recovery for structured output.
//!
//! This module consumes only immutable schema data, the validator's parsed JSON,
//! and original UTF-8 text. It never uses a grammar cursor, suffix index, mask,
//! or acceptance flag. KMP finds overlapping matches; byte/scalar coordinates
//! come from a separate scan. Limits fail the whole result, never truncate it.

use std::{error::Error, fmt};
use serde::{Deserialize, Serialize};
use crate::grammar::{SchemaNode, SourceAnnotation};
use super::JsonValue;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GroundingBudget {
    pub max_fields: usize,
    pub max_matches: usize,
    pub max_scan_steps: u64,
}
impl Default for GroundingBudget {
    fn default() -> Self {
        Self { max_fields: 4096, max_matches: 16_384, max_scan_steps: 64 * 1024 * 1024 }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VerifiedSourceSpan {
    pub byte_start: usize,
    pub byte_end: usize,
    pub scalar_start: usize,
    pub scalar_end: usize,
}
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceOccurrence { Anchored, Ambiguous }

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SourceFieldEvidence {
    pub json_pointer: String,
    pub occurrence: SourceOccurrence,
    pub spans: Vec<VerifiedSourceSpan>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FieldGroundingError { Shape, Absent, FieldBudget, MatchBudget, WorkBudget, AllocationRefused }
impl fmt::Display for FieldGroundingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Shape => "source verification requires a schema-valid value",
            Self::Absent => "verbatim field is absent from the original source",
            Self::FieldBudget => "source field count exceeds verification budget",
            Self::MatchBudget => "source occurrence count exceeds verification budget",
            Self::WorkBudget => "source verification exceeds scan work budget",
            Self::AllocationRefused => "source verification allocation refused",
        })
    }
}
impl Error for FieldGroundingError {}

/// Keep the caller's original cancellation/error separate from an absent
/// source match. In particular, interrupted evidence is never an abstention.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GroundingScanError<E> { Verification(FieldGroundingError), Checkpoint(E) }
impl<E> From<FieldGroundingError> for GroundingScanError<E> {
    fn from(error: FieldGroundingError) -> Self { Self::Verification(error) }
}
impl<E> fmt::Display for GroundingScanError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Verification(error) => fmt::Display::fmt(error, f),
            Self::Checkpoint(_) => f.write_str("source verification checkpoint refused execution"),
        }
    }
}
impl<E: Error + 'static> Error for GroundingScanError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Verification(error) => Some(error), Self::Checkpoint(error) => Some(error) }
    }
}

/// Independently check every present source annotation, including annotations
/// below optional object keys and arrays. Returned JSON pointers locate values,
/// not schema nodes. Absence of an optional field requires no invented span.
pub fn verify_source_fields(
    schema: &SchemaNode, value: &JsonValue, source: &str, budget: GroundingBudget,
) -> Result<Vec<SourceFieldEvidence>, FieldGroundingError> {
    match verify_source_fields_with_checkpoints(schema, value, source, budget, EvidenceMode::Recover,
        || Ok::<(), std::convert::Infallible>(())) {
        Ok(fields) => Ok(fields),
        Err(GroundingScanError::Verification(error)) => Err(error),
        Err(GroundingScanError::Checkpoint(never)) => match never {},
    }
}

#[derive(Clone, Copy)]
pub(crate) enum EvidenceMode { Recover, CheckOnly }

pub(crate) fn verify_source_fields_with_checkpoints<E>(
    schema: &SchemaNode, value: &JsonValue, source: &str, budget: GroundingBudget,
    mode: EvidenceMode,
    mut checkpoint: impl FnMut() -> Result<(), E>,
) -> Result<Vec<SourceFieldEvidence>, GroundingScanError<E>> {
    let fields = match mode { EvidenceMode::Recover => Some(Vec::new()), EvidenceMode::CheckOnly => None };
    let mut state = Verification { source, budget, fields };
    state.walk(schema, value, "$", 0, &mut checkpoint)?;
    checkpoint().map_err(GroundingScanError::Checkpoint)?;
    Ok(state.fields.unwrap_or_default())
}

struct Verification<'a> {
    source: &'a str,
    budget: GroundingBudget,
    fields: Option<Vec<SourceFieldEvidence>>,
}
impl Verification<'_> {
    fn walk<E>(&mut self, schema: &SchemaNode, value: &JsonValue, path: &str, depth: usize,
        checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<(), GroundingScanError<E>> {
        checkpoint().map_err(GroundingScanError::Checkpoint)?;
        if depth > 128 { return Err(FieldGroundingError::Shape.into()); }
        match (schema, value) {
            (SchemaNode::String { source: SourceAnnotation::Verbatim, .. }, JsonValue::String(text)) => {
                self.budget.max_fields = self.budget.max_fields.checked_sub(1).ok_or(FieldGroundingError::FieldBudget)?;
                let mut spans = Vec::new();
                scan(self.source, text, &mut self.budget, self.fields.is_some().then_some(&mut spans), &mut *checkpoint)?;
                if let Some(fields) = &mut self.fields {
                    let occurrence = if spans.len() == 1 { SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous };
                    fields.try_reserve(1).map_err(|_| FieldGroundingError::AllocationRefused)?;
                    fields.push(SourceFieldEvidence { json_pointer: path.to_owned(), occurrence, spans });
                }
            }
            (SchemaNode::Object { properties, .. }, JsonValue::Object(values)) => {
                for (name, value) in values {
                    let child = properties.get(name).ok_or(FieldGroundingError::Shape)?;
                    let child_path = self.fields.is_some().then(|| pointer(path, &name.replace('~', "~0").replace('/', "~1")));
                    self.walk(child, value, child_path.as_deref().unwrap_or(""), depth + 1, checkpoint)?;
                }
            }
            (SchemaNode::Array { items, .. }, JsonValue::Array(values)) => {
                for (i, value) in values.iter().enumerate() {
                    let child_path = self.fields.is_some().then(|| pointer(path, &i.to_string()));
                    self.walk(items, value, child_path.as_deref().unwrap_or(""), depth + 1, checkpoint)?;
                }
            }
            (SchemaNode::String { .. }, JsonValue::String(_))
            | (SchemaNode::Number { .. }, JsonValue::Number(_))
            | (SchemaNode::Boolean { .. }, JsonValue::Boolean(_))
            | (SchemaNode::Null { .. }, JsonValue::Null) => {}
            _ => return Err(FieldGroundingError::Shape.into()),
        }
        Ok(())
    }
}
fn pointer(parent: &str, key: &str) -> String {
    if parent == "$" { format!("/{key}") } else { format!("{parent}/{key}") }
}

/// Independent exact matching, also useful for NER mention finalization.
/// Consumes aggregate budgets shared by all fields in a response.
pub fn scan_occurrences(source: &str, text: &str, budget: &mut GroundingBudget) -> Result<Vec<VerifiedSourceSpan>, FieldGroundingError> {
    match scan_occurrences_with_checkpoints(source, text, budget, || Ok::<(), std::convert::Infallible>(())) {
        Ok(spans) => Ok(spans),
        Err(GroundingScanError::Verification(error)) => Err(error),
        Err(GroundingScanError::Checkpoint(never)) => match never {},
    }
}

/// The same independent KMP/UTF-8 occurrence census with bounded cooperative
/// interruption. Poll before work, every 4096 scan/prefix-fallback steps, and
/// before returning either complete matches or absence. An attempted scan
/// retains its full original work debit on cancellation; it never publishes a
/// partial occurrence set or refunds work for an unsuccessful search.
pub fn scan_occurrences_with_checkpoints<E>(source: &str, text: &str, budget: &mut GroundingBudget,
    checkpoint: impl FnMut() -> Result<(), E>) -> Result<Vec<VerifiedSourceSpan>, GroundingScanError<E>> {
    let mut spans = Vec::new();
    scan(source, text, budget, Some(&mut spans), checkpoint)?;
    Ok(spans)
}

/// Always prove presence and enforce the complete overlapping-match budget.
/// Decoders checking only validity do not allocate a vector of coordinates.
fn scan<E>(source: &str, text: &str, budget: &mut GroundingBudget,
    mut spans: Option<&mut Vec<VerifiedSourceSpan>>, mut checkpoint: impl FnMut() -> Result<(), E>,
) -> Result<(), GroundingScanError<E>> {
    checkpoint().map_err(GroundingScanError::Checkpoint)?;
    let mut until_checkpoint = SCAN_CHECKPOINT_STEPS;
    let mut step = || -> Result<(), GroundingScanError<E>> {
        if until_checkpoint == 0 {
            checkpoint().map_err(GroundingScanError::Checkpoint)?;
            until_checkpoint = SCAN_CHECKPOINT_STEPS;
        }
        until_checkpoint -= 1;
        Ok(())
    };
    let cost = (source.len() as u64).checked_add(text.len() as u64)
        .and_then(|n| n.checked_add(1)).and_then(|n| n.checked_mul(8)).ok_or(FieldGroundingError::WorkBudget)?;
    budget.max_scan_steps = budget.max_scan_steps.checked_sub(cost).ok_or(FieldGroundingError::WorkBudget)?;
    let mut matches = 0;
    let mut push = |span| -> Result<(), FieldGroundingError> {
        if matches == budget.max_matches { return Err(FieldGroundingError::MatchBudget); }
        if let Some(output) = &mut spans {
            output.try_reserve(1).map_err(|_| FieldGroundingError::AllocationRefused)?;
            output.push(span);
        }
        matches += 1;
        Ok(())
    };
    if text.is_empty() {
        for (scalar, byte) in source.char_indices().map(|(i, _)| i).chain(std::iter::once(source.len())).enumerate() {
            step()?;
            push(VerifiedSourceSpan { byte_start: byte, byte_end: byte, scalar_start: scalar, scalar_end: scalar })?;
        }
    } else {
        let needle = text.as_bytes();
        let mut prefix = Vec::new();
        prefix.try_reserve_exact(needle.len()).map_err(|_| FieldGroundingError::AllocationRefused)?;
        // Grow initialized entries only as they are computed. A bulk resize or
        // a separate chars().count() would hide an uninterruptible needle scan.
        prefix.push(0_usize);
        let mut text_scalars = 1_usize;
        let mut matched = 0;
        for i in 1..needle.len() {
            step()?;
            if needle[i] & 0xc0 != 0x80 { text_scalars += 1; }
            while matched > 0 && needle[i] != needle[matched] { step()?; matched = prefix[matched - 1]; }
            if needle[i] == needle[matched] { matched += 1; }
            prefix.push(matched);
        }
        let mut scalars = 0_usize;
        matched = 0;
        for (i, byte) in source.bytes().enumerate() {
            step()?;
            if byte & 0xc0 != 0x80 { scalars += 1; }
            while matched > 0 && byte != needle[matched] { step()?; matched = prefix[matched - 1]; }
            if byte == needle[matched] { matched += 1; }
            if matched == needle.len() {
                let end = i + 1; let start = end - needle.len();
                // KMP's matched prefix already proves byte equality for this
                // entire interval. Re-comparing the substring here would cost
                // O(matches * needle.len()) on overlapping repetitions, outside
                // the linear scan allowance charged above. Only constant-time
                // UTF-8 boundary checks remain before publishing coordinates.
                if !source.is_char_boundary(start) || !source.is_char_boundary(end) {
                    return Err(FieldGroundingError::Shape.into());
                }
                push(VerifiedSourceSpan { byte_start: start, byte_end: end,
                    scalar_start: scalars.checked_sub(text_scalars).ok_or(FieldGroundingError::Shape)?, scalar_end: scalars })?;
                matched = prefix[matched - 1]; // retain overlapping matches
            }
        }
    }
    drop(push);
    drop(step);
    checkpoint().map_err(GroundingScanError::Checkpoint)?;
    if matches == 0 { return Err(FieldGroundingError::Absent.into()); }
    budget.max_matches -= matches;
    Ok(())
}

const SCAN_CHECKPOINT_STEPS: usize = 4096;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{grammar::{compile_json_schema, CompileLimits}, validation::{parse_json, validate_source_span, SourceSpan}};

    #[test]
    fn check_only_enforces_the_same_aggregate_field_match_and_scan_limits() {
        let schema = compile_json_schema(r#"{"type":"array","maxItems":4,"items":{"type":"string","x-fnlp-source":"verbatim"}}"#,
            CompileLimits::default()).unwrap();
        let source = "éaaaa";
        let value = parse_json(r#"["aa","é",""]"#).unwrap();
        // Three overlapping pairs, one accent, and six empty boundary spans.
        let exact = GroundingBudget { max_fields: 3, max_matches: 10,
            max_scan_steps: 8 * (3 * (source.len() as u64 + 1) + 4) };
        for (budget, expected) in [
            (exact, Ok(())),
            (GroundingBudget { max_matches: 9, ..exact }, Err(FieldGroundingError::MatchBudget)),
            (GroundingBudget { max_fields: 2, ..exact }, Err(FieldGroundingError::FieldBudget)),
            (GroundingBudget { max_scan_steps: exact.max_scan_steps - 1, ..exact }, Err(FieldGroundingError::WorkBudget)),
        ] {
            let recover = verify_source_fields(schema.root(), &value, source, budget);
            assert_eq!(recover.as_ref().map(|_| ()).map_err(|error| *error), expected);
            let checked = verify_source_fields_with_checkpoints(schema.root(), &value, source,
                budget, EvidenceMode::CheckOnly, || Ok::<(), ()>(()));
            assert_eq!(checked, expected.map(|_| Vec::new()).map_err(GroundingScanError::Verification));
        }
    }

    #[test]
    fn checkpointed_scans_preserve_complete_unicode_overlap_and_budget_results() {
        for source in ["", "éaaaa上海上海", "e\u{301}e\u{301}", "aaaaa"] {
            for text in ["", "é", "aa", "上海", "e\u{301}", "missing"] {
                let mut plain = GroundingBudget::default();
                let mut checked = plain;
                let expected = scan_occurrences(source, text, &mut plain);
                let actual = scan_occurrences_with_checkpoints(source, text, &mut checked,
                    || Ok::<(), ()>(()));
                assert_eq!(actual, expected.map_err(GroundingScanError::Verification));
                assert_eq!(checked, plain);
                // Independent character-boundary oracle, including overlaps
                // and the terminal empty-string occurrence.
                let spans = source.char_indices().map(|(byte, _)| byte)
                    .chain(std::iter::once(source.len())).enumerate()
                    .filter_map(|(scalar, byte)| source[byte..].starts_with(text).then(|| VerifiedSourceSpan {
                        byte_start: byte, byte_end: byte + text.len(), scalar_start: scalar,
                        scalar_end: scalar + text.chars().count(),
                    })).collect::<Vec<_>>();
                let oracle = if spans.is_empty() { Err(GroundingScanError::Verification(FieldGroundingError::Absent)) }
                    else { Ok(spans) };
                assert_eq!(actual, oracle);
            }
        }
    }

    #[test]
    fn every_terminal_cause_interrupts_large_source_and_empty_occurrence_scans() {
        use crate::native_engine::decode::DecodeCancellationKind::*;
        let source = "é".repeat(SCAN_CHECKPOINT_STEPS * 3);
        for text in ["", "é", "absent"] {
            for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
                ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
                let mut budget = GroundingBudget::default();
                let before = budget;
                let mut calls = 0;
                let result = scan_occurrences_with_checkpoints(&source, text, &mut budget, || {
                    calls += 1; if calls == 2 { Err(cause) } else { Ok(()) }
                });
                assert_eq!(result, Err(GroundingScanError::Checkpoint(cause)));
                assert_eq!(calls, 2);
                assert_eq!(budget.max_scan_steps,
                    before.max_scan_steps - ((source.len() + text.len() + 1) * 8) as u64);
                assert_eq!(budget.max_matches, before.max_matches);
            }
        }
    }

    #[test]
    fn prefix_construction_and_one_long_fallback_are_both_interruptible() {
        let needle = format!("{}b", "a".repeat(2 * SCAN_CHECKPOINT_STEPS));
        // The second poll interrupts prefix construction. The third occurs
        // inside the final byte's long KMP fallback, before source scanning.
        for stop in [2, 3] {
            let mut calls = 0;
            let result = scan_occurrences_with_checkpoints("", &needle, &mut GroundingBudget::default(), || {
                calls += 1; if calls == stop { Err("original stop") } else { Ok(()) }
            });
            assert_eq!(result, Err(GroundingScanError::Checkpoint("original stop")));
            assert_eq!(calls, stop);
        }
    }

    #[test]
    fn final_checkpoint_cannot_be_laundered_into_absence_or_partial_success() {
        for text in ["b", "absent"] {
            let mut calls = 0;
            let mut budget = GroundingBudget::default();
            let matches = budget.max_matches;
            let result = scan_occurrences_with_checkpoints("abc", text, &mut budget, || {
                calls += 1; if calls == 2 { Err(7) } else { Ok(()) }
            });
            assert_eq!(result, Err(GroundingScanError::Checkpoint(7)));
            assert_eq!(budget.max_matches, matches);
        }
    }

    #[test]
    fn scan_refuses_work_and_match_budgets_without_renewal() {
        let mut budget = GroundingBudget { max_fields: 1, max_matches: 2, max_scan_steps: 39 };
        assert_eq!(scan_occurrences_with_checkpoints("aaaa", "", &mut budget, || Ok::<(), ()>(())),
            Err(GroundingScanError::Verification(FieldGroundingError::WorkBudget)));
        assert_eq!(budget.max_scan_steps, 39);
        budget.max_scan_steps = 40;
        assert_eq!(scan_occurrences_with_checkpoints("aaaa", "", &mut budget, || Ok::<(), ()>(())),
            Err(GroundingScanError::Verification(FieldGroundingError::MatchBudget)));
        assert_eq!(budget.max_scan_steps, 0);
    }

    #[test]
    fn overlapping_occurrences_and_unicode_offsets_are_verified_independently() {
        let source = "éaaaa上海上海";
        for text in ["aa", "上海", "é", "", "aaaa上海"] {
            let spans = scan_occurrences(source, text, &mut GroundingBudget::default()).unwrap();
            for s in &spans {
                validate_source_span(source, text, SourceSpan::new(s.byte_start, s.byte_end, s.scalar_start, s.scalar_end)).unwrap();
            }
            let expected = source.char_indices().map(|(i, _)| i).chain(std::iter::once(source.len()))
                .filter(|&i| source[i..].starts_with(text)).count();
            assert_eq!(spans.len(), expected);
        }
    }
    #[test]
    fn exact_spans_match_an_independent_oracle_over_short_unicode_strings() {
        fn words(depth: usize) -> Vec<String> {
            let alphabet = ["a", "b", "é", "e\u{301}", "😀", "\r\n", "\0"];
            let mut all = vec![String::new()];
            let mut frontier = vec![String::new()];
            for _ in 0..depth {
                let mut next = Vec::new();
                for prefix in &frontier {
                    for suffix in alphabet { next.push(format!("{prefix}{suffix}")); }
                }
                all.extend(next.iter().cloned());
                frontier = next;
            }
            all
        }
        let needles = words(2);
        for source in words(3) {
            for text in &needles {
                // Deliberately simple independent reference: test each scalar
                // boundary with ordinary substring comparison, without KMP.
                let expected: Vec<_> = source.char_indices().map(|(byte, _)| byte)
                    .chain(std::iter::once(source.len())).filter_map(|start| {
                        if !source[start..].starts_with(text.as_str()) { return None; }
                        let end = start + text.len();
                        Some(VerifiedSourceSpan { byte_start: start, byte_end: end,
                            scalar_start: source[..start].chars().count(),
                            scalar_end: source[..end].chars().count() })
                    }).collect();
                let mut budget = GroundingBudget::default();
                let before = budget;
                let actual = scan_occurrences(&source, text, &mut budget);
                let matched = expected.len();
                if expected.is_empty() {
                    assert_eq!(actual, Err(FieldGroundingError::Absent), "source={source:?} text={text:?}");
                } else {
                    assert_eq!(actual.unwrap(), expected, "source={source:?} text={text:?}");
                }
                assert_eq!(budget.max_matches, before.max_matches - matched);
                assert_eq!(budget.max_fields, before.max_fields);
                assert_eq!(before.max_scan_steps - budget.max_scan_steps,
                    (source.len() as u64 + text.len() as u64 + 1) * 8);
            }
        }
    }
    #[test]
    fn repetitive_long_matches_keep_every_overlap_under_one_linear_work_charge() {
        let source_scalars = 65_536;
        let needle_scalars = 32_768;
        let source = "é".repeat(source_scalars);
        let text = "é".repeat(needle_scalars);
        let matches = source_scalars - needle_scalars + 1;
        let work = (source.len() as u64 + text.len() as u64 + 1) * 8;
        let mut budget = GroundingBudget { max_fields: 1, max_matches: matches, max_scan_steps: work };
        let spans = scan_occurrences(&source, &text, &mut budget).unwrap();
        assert_eq!(spans.len(), matches);
        for (start, span) in spans.iter().enumerate() {
            assert_eq!(*span, VerifiedSourceSpan { byte_start: start * 2,
                byte_end: (start + needle_scalars) * 2,
                scalar_start: start, scalar_end: start + needle_scalars });
        }
        assert_eq!(budget.max_scan_steps, 0);
        assert_eq!(budget.max_matches, 0);
        assert_eq!(scan_occurrences(&source, &text, &mut budget), Err(FieldGroundingError::WorkBudget));

        // An insufficient occurrence budget rejects the complete result and
        // retains the charged work; it cannot publish a truncated prefix.
        let mut limited = GroundingBudget { max_fields: 1, max_matches: matches - 1, max_scan_steps: work };
        assert_eq!(scan_occurrences(&source, &text, &mut limited), Err(FieldGroundingError::MatchBudget));
        assert_eq!(limited.max_scan_steps, 0);
        assert_eq!(limited.max_matches, matches - 1);
    }
    #[test]
    fn repeated_long_prefix_mismatches_are_absent_without_renewing_work() {
        let source = "abé".repeat(16_384);
        let text = format!("{}z", "abé".repeat(8_192));
        let work = (source.len() as u64 + text.len() as u64 + 1) * 8;
        let mut budget = GroundingBudget { max_scan_steps: work, ..GroundingBudget::default() };
        let matches = budget.max_matches;
        assert_eq!(scan_occurrences(&source, &text, &mut budget), Err(FieldGroundingError::Absent));
        assert_eq!(budget.max_scan_steps, 0);
        assert_eq!(budget.max_matches, matches);
        assert_eq!(scan_occurrences(&source, &text, &mut budget), Err(FieldGroundingError::WorkBudget));
    }
    #[test]
    fn nested_arrays_and_escaped_property_names_have_value_pointers() {
        let schema = compile_json_schema(r#"{"type":"object","additionalProperties":false,"properties":{"a/b":{"type":"array","items":{"type":"string","x-fnlp-source":"verbatim"},"maxItems":2}}}"#, CompileLimits::default()).unwrap();
        let v = parse_json(r#"{"a/b":["Alice","Bob"]}"#).unwrap();
        let e = verify_source_fields(schema.root(), &v, "Alice Bob Alice", GroundingBudget::default()).unwrap();
        assert_eq!(e[0].json_pointer, "/a~1b/0"); assert_eq!(e[0].occurrence, SourceOccurrence::Ambiguous);
        assert_eq!(e[1].occurrence, SourceOccurrence::Anchored);
    }
    #[test]
    fn aggregate_matches_cannot_be_reset_between_fields() {
        let schema = compile_json_schema(r#"{"type":"array","items":{"type":"string","x-fnlp-source":"verbatim"},"maxItems":2}"#, CompileLimits::default()).unwrap();
        let v = parse_json(r#"["a","a"]"#).unwrap();
        assert_eq!(verify_source_fields(schema.root(), &v, "a a", GroundingBudget { max_matches: 3, ..GroundingBudget::default() }), Err(FieldGroundingError::MatchBudget));
    }
    #[test]
    fn invented_or_normalized_strings_are_rejected() {
        for text in ["é", "Alice"] {
            assert_eq!(scan_occurrences("e\u{301} alice", text, &mut GroundingBudget::default()), Err(FieldGroundingError::Absent));
        }
    }
    #[test]
    fn work_limit_refuses_before_scan() {
        assert_eq!(scan_occurrences("secret", "secret", &mut GroundingBudget { max_scan_steps: 0, ..GroundingBudget::default() }), Err(FieldGroundingError::WorkBudget));
    }
    #[test]
    fn empty_source_has_one_empty_boundary() {
        assert_eq!(scan_occurrences("", "", &mut GroundingBudget::default()).unwrap().len(), 1);
        assert_eq!(scan_occurrences("", "x", &mut GroundingBudget::default()), Err(FieldGroundingError::Absent));
    }
}
