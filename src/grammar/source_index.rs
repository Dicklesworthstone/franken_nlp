//! Bounded request-owned substring language with constant-size branch cursors.
//!
//! Prefix doubling and counting sorts build a suffix index without enumerating
//! substrings or comparing unbounded suffix strings. Only UTF-8-boundary starts
//! enter the final index. A cursor borrows the immutable index and narrows an
//! interval with binary searches; vocabulary branches never clone the document
//! or a vector of source occurrences. No source-derived state is shared across
//! requests. Matches are recovered only after a complete logical string.

use std::{error::Error, fmt, mem::size_of};
use serde::{Deserialize, Serialize};
use super::source::SourceMatch;

pub const SOURCE_LANGUAGE_VERSION: &str = "utf8-substring-suffix-v1";
/// Maximum source bytes, initialized entries or sort-loop iterations between
/// cooperative checkpoints. Allocation itself is synchronous and not preempted.
pub const SOURCE_BUILD_CHECKPOINT_INTERVAL: usize = 4096;

/// Explicit resource ceilings, including all sorting scratch before allocation.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SourceLanguageLimits {
    pub max_source_bytes: usize,
    /// Requested vector/string storage; not a process-RSS measurement.
    pub max_index_bytes: usize,
    /// Conservative bound on indexed construction-loop operations.
    pub max_build_steps: u64,
    /// Refuse, never truncate, an occurrence list exceeding this count.
    pub max_matches: usize,
}
impl Default for SourceLanguageLimits {
    fn default() -> Self {
        Self { max_source_bytes: 1024 * 1024, max_index_bytes: 64 * 1024 * 1024,
            max_build_steps: 512 * 1024 * 1024, max_matches: 16_384 }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SourceLanguageEstimate {
    pub source_bytes: usize,
    pub scalar_boundaries: usize,
    pub requested_peak_bytes: usize,
    pub build_step_bound: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourceLanguageError {
    SourceLimit, IndexLimit, BuildLimit, MatchLimit, ArithmeticOverflow,
    AllocationRefused, NoMatch, IncompleteUtf8,
}
impl fmt::Display for SourceLanguageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::SourceLimit => "source language exceeds source byte limit",
            Self::IndexLimit => "source language exceeds index storage limit",
            Self::BuildLimit => "source language exceeds construction work limit",
            Self::MatchLimit => "source occurrence count exceeds result limit",
            Self::ArithmeticOverflow => "source language arithmetic overflow",
            Self::AllocationRefused => "source language allocation refused",
            Self::NoMatch => "string is absent from the source language",
            Self::IncompleteUtf8 => "source string ends inside a UTF-8 scalar",
        })
    }
}
impl Error for SourceLanguageError {}

/// Index construction preserves the caller's cancellation value separately
/// from source/resource refusals. No partial language escapes on either path.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SourceBuildError<E> {
    Source(SourceLanguageError),
    Cancelled(E),
}
impl<E> From<SourceLanguageError> for SourceBuildError<E> {
    fn from(error: SourceLanguageError) -> Self { Self::Source(error) }
}
impl<E> fmt::Display for SourceBuildError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Source(error) => write!(f, "{error}"),
            Self::Cancelled(_) => f.write_str("source language construction cancelled"),
        }
    }
}
impl<E: fmt::Debug> Error for SourceBuildError<E> {}

struct BuildControl<'a, F> { callback: &'a mut F, remaining: usize }
impl<F> BuildControl<'_, F> {
    fn checkpoint<E>(&mut self) -> Result<(), SourceBuildError<E>>
    where F: FnMut() -> Option<E> {
        if let Some(cause) = (self.callback)() { return Err(SourceBuildError::Cancelled(cause)); }
        self.remaining = SOURCE_BUILD_CHECKPOINT_INTERVAL;
        Ok(())
    }
    fn step<E>(&mut self) -> Result<(), SourceBuildError<E>>
    where F: FnMut() -> Option<E> {
        self.remaining -= 1;
        if self.remaining == 0 { self.checkpoint()?; }
        Ok(())
    }
}

fn uncontrolled<T>(result: Result<T, SourceBuildError<std::convert::Infallible>>)
    -> Result<T, SourceLanguageError> {
    match result {
        Ok(value) => Ok(value),
        Err(SourceBuildError::Source(error)) => Err(error),
        Err(SourceBuildError::Cancelled(never)) => match never {},
    }
}

#[derive(Clone)]
pub struct SourceLanguage {
    source: String,
    suffixes: Vec<usize>,
    boundaries: Vec<usize>,
    limits: SourceLanguageLimits,
    estimate: SourceLanguageEstimate,
}
// Debug metadata must not expose a document or a derived suffix ordering.
impl fmt::Debug for SourceLanguage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SourceLanguage").field("estimate", &self.estimate).finish_non_exhaustive()
    }
}

impl SourceLanguage {
    pub fn preflight(source: &str, limits: SourceLanguageLimits) -> Result<SourceLanguageEstimate, SourceLanguageError> {
        uncontrolled(Self::preflight_with_control(source, limits, &mut || None))
    }

    /// Count exact UTF-8 boundaries without an uninterruptible document scan.
    pub fn preflight_with_control<E, F: FnMut() -> Option<E>>(source: &str,
        limits: SourceLanguageLimits, checkpoint: &mut F)
        -> Result<SourceLanguageEstimate, SourceBuildError<E>> {
        let mut control = BuildControl { callback: checkpoint, remaining: SOURCE_BUILD_CHECKPOINT_INTERVAL };
        control.checkpoint()?;
        let estimate = Self::preflight_controlled(source, limits, &mut control)?;
        control.checkpoint()?;
        Ok(estimate)
    }

    fn preflight_controlled<E, F: FnMut() -> Option<E>>(source: &str,
        limits: SourceLanguageLimits, control: &mut BuildControl<'_, F>)
        -> Result<SourceLanguageEstimate, SourceBuildError<E>> {
        use SourceLanguageError::ArithmeticOverflow as Overflow;
        let n = source.len();
        if n > limits.max_source_bytes { return Err(SourceLanguageError::SourceLimit.into()); }
        let mut boundaries = 1_usize;
        for byte in source.bytes() {
            if byte & 0xc0 != 0x80 { boundaries = boundaries.checked_add(1).ok_or(Overflow)?; }
            control.step()?;
        }
        let counters = n.max(256).checked_add(1).ok_or(Overflow)?;
        // suffixes, temporary order, old ranks, new ranks, counters, boundaries.
        let words = n.checked_mul(4).and_then(|x| x.checked_add(counters))
            .and_then(|x| x.checked_add(boundaries)).ok_or(Overflow)?;
        let peak = words.checked_mul(size_of::<usize>()).and_then(|x| x.checked_add(n))
            .and_then(|x| x.checked_add(size_of::<Self>() + 4 * size_of::<Vec<usize>>())).ok_or(Overflow)?;
        if peak > limits.max_index_bytes { return Err(SourceLanguageError::IndexLimit.into()); }
        let rounds = if n < 2 { 0 } else { usize::BITS - (n - 1).leading_zeros() };
        let steps = (n as u64).checked_mul(16)
            .and_then(|x| (counters as u64).checked_mul(4).and_then(|c| x.checked_add(c)))
            .and_then(|x| x.checked_mul(u64::from(rounds)))
            .and_then(|x| (n as u64).checked_mul(8).and_then(|y| x.checked_add(y))).ok_or(Overflow)?;
        if steps > limits.max_build_steps { return Err(SourceLanguageError::BuildLimit.into()); }
        Ok(SourceLanguageEstimate { source_bytes: n, scalar_boundaries: boundaries,
            requested_peak_bytes: peak, build_step_bound: steps })
    }

    pub fn build(source: &str, limits: SourceLanguageLimits) -> Result<Self, SourceLanguageError> {
        uncontrolled(Self::build_with_control(source, limits, &mut || None))
    }

    /// Construct the same suffix language with bounded cooperative work.
    /// The callback is never replaced or reset, and its exact cancellation
    /// payload survives preflight, copies, all sort passes and completion.
    pub fn build_with_control<E, F: FnMut() -> Option<E>>(source: &str,
        limits: SourceLanguageLimits, checkpoint: &mut F) -> Result<Self, SourceBuildError<E>> {
        let mut control = BuildControl { callback: checkpoint, remaining: SOURCE_BUILD_CHECKPOINT_INTERVAL };
        control.checkpoint()?;
        let estimate = Self::preflight_controlled(source, limits, &mut control)?;
        control.checkpoint()?;
        let n = source.len();
        let mut owned = String::new();
        owned.try_reserve_exact(n).map_err(|_| SourceLanguageError::AllocationRefused)?;
        let mut start = 0;
        while start < n {
            control.checkpoint()?;
            let mut end = start.saturating_add(SOURCE_BUILD_CHECKPOINT_INTERVAL).min(n);
            while !source.is_char_boundary(end) { end -= 1; }
            owned.push_str(&source[start..end]);
            start = end;
        }
        control.checkpoint()?;
        let mut boundaries = reserved(estimate.scalar_boundaries)?;
        for (i, _) in source.char_indices() { boundaries.push(i); control.step()?; }
        boundaries.push(n);
        let mut suffixes = reserved(n)?;
        for i in 0..n { suffixes.push(i); control.step()?; }
        let mut temporary = zeroes(n, &mut control)?;
        let mut ranks = reserved(n)?;
        for byte in source.bytes() { ranks.push(usize::from(byte) + 1); control.step()?; }
        let mut next_ranks = zeroes(n, &mut control)?;
        let mut counters = zeroes(n.max(256) + 1, &mut control)?;
        let mut classes = 256;
        let mut width = 1;
        while width < n {
            counting_sort(&suffixes, &mut temporary, &ranks, width, &mut counters[..=classes], &mut control)?;
            counting_sort(&temporary, &mut suffixes, &ranks, 0, &mut counters[..=classes], &mut control)?;
            let mut count = 1;
            next_ranks[suffixes[0]] = count;
            for pair in suffixes.windows(2) {
                let key = |i: usize| (ranks[i], ranks.get(i + width).copied().unwrap_or(0));
                if key(pair[0]) != key(pair[1]) { count += 1; }
                next_ranks[pair[1]] = count;
                control.step()?;
            }
            std::mem::swap(&mut ranks, &mut next_ranks);
            classes = count;
            if classes == n { break; }
            width = width.checked_mul(2).ok_or(SourceLanguageError::ArithmeticOverflow)?;
        }
        // Reuse rank scratch as the boundary bitmap; no additional n-byte map.
        for rank in &mut ranks { *rank = 0; control.step()?; }
        for &start in boundaries.iter().take(boundaries.len() - 1) { ranks[start] = 1; control.step()?; }
        let mut retained = 0;
        for i in 0..suffixes.len() {
            if ranks[suffixes[i]] != 0 { suffixes[retained] = suffixes[i]; retained += 1; }
            control.step()?;
        }
        suffixes.truncate(retained);
        control.checkpoint()?;
        Ok(Self { source: owned, suffixes, boundaries, limits, estimate })
    }

    #[must_use]
    pub const fn estimate(&self) -> SourceLanguageEstimate { self.estimate }
    #[must_use]
    pub fn source(&self) -> &str { &self.source }
    #[must_use]
    pub const fn limits(&self) -> SourceLanguageLimits { self.limits }
    #[must_use]
    pub fn cursor(&self) -> SourceCursor<'_> {
        SourceCursor { index: self, lower: 0, upper: self.suffixes.len(), bytes: 0 }
    }
    pub fn matches(&self, text: &str) -> Result<Vec<SourceMatch>, SourceLanguageError> {
        self.matches_bounded(text, self.limits.max_matches)
    }
    /// The caller may tighten an aggregate result budget but never widen it.
    pub fn matches_bounded(&self, text: &str, remaining: usize) -> Result<Vec<SourceMatch>, SourceLanguageError> {
        let mut cursor = self.cursor();
        for byte in text.bytes() {
            if !cursor.push_byte(byte) { return Err(SourceLanguageError::NoMatch); }
        }
        cursor.finish(remaining.min(self.limits.max_matches))
    }
}

fn reserved<T>(count: usize) -> Result<Vec<T>, SourceLanguageError> {
    let mut out = Vec::new();
    out.try_reserve_exact(count).map_err(|_| SourceLanguageError::AllocationRefused)?;
    Ok(out)
}
fn zeroes<E, F: FnMut() -> Option<E>>(count: usize, control: &mut BuildControl<'_, F>)
    -> Result<Vec<usize>, SourceBuildError<E>> {
    let mut out = reserved(count)?;
    for _ in 0..count { out.push(0); control.step()?; }
    Ok(out)
}
fn counting_sort<E, F: FnMut() -> Option<E>>(input: &[usize], output: &mut [usize], ranks: &[usize],
    offset: usize, counts: &mut [usize], control: &mut BuildControl<'_, F>) -> Result<(), SourceBuildError<E>> {
    for count in counts.iter_mut() { *count = 0; control.step()?; }
    let key = |i: usize| ranks.get(i + offset).copied().unwrap_or(0);
    for &i in input { counts[key(i)] += 1; control.step()?; }
    let mut start = 0;
    for count in counts.iter_mut() { let size = *count; *count = start; start += size; control.step()?; }
    for &i in input { let k = key(i); output[counts[k]] = i; counts[k] += 1; control.step()?; }
    Ok(())
}

/// A borrowed suffix interval, not a copied occurrence vector. All mutations
/// are local to this branch; unsuccessful transitions permanently kill it.
#[derive(Clone, Copy)]
pub struct SourceCursor<'a> {
    index: &'a SourceLanguage,
    lower: usize,
    upper: usize,
    bytes: usize,
}
impl fmt::Debug for SourceCursor<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SourceCursor").field("matched_bytes", &self.bytes).finish_non_exhaustive()
    }
}
impl SourceCursor<'_> {
    fn range(&self, byte: u8) -> (usize, usize) {
        let suffixes = &self.index.suffixes[self.lower..self.upper];
        let at = |start: &usize| self.index.source.as_bytes().get(*start + self.bytes).copied();
        let lower = suffixes.partition_point(|i| at(i) < Some(byte));
        let upper = suffixes.partition_point(|i| at(i) <= Some(byte));
        (self.lower + lower, self.lower + upper)
    }
    #[must_use]
    pub fn can_push(&self, byte: u8) -> bool {
        let (lower, upper) = self.range(byte); lower < upper
    }
    pub fn push_byte(&mut self, byte: u8) -> bool {
        let (lower, upper) = self.range(byte);
        self.lower = lower; self.upper = upper;
        self.bytes += 1;
        lower < upper
    }
    #[must_use]
    pub fn is_accepting(&self) -> bool {
        if self.bytes == 0 { return true; }
        self.lower < self.upper && self.index.source.is_char_boundary(self.index.suffixes[self.lower] + self.bytes)
    }
    fn finish(self, maximum: usize) -> Result<Vec<SourceMatch>, SourceLanguageError> {
        if !self.is_accepting() { return Err(SourceLanguageError::IncompleteUtf8); }
        let count = if self.bytes == 0 { self.index.boundaries.len() } else { self.upper - self.lower };
        if count > maximum { return Err(SourceLanguageError::MatchLimit); }
        let mut output = reserved(count)?;
        if self.bytes == 0 {
            for (scalar, &byte) in self.index.boundaries.iter().enumerate() {
                output.push(SourceMatch { byte_start: byte, byte_end: byte, scalar_start: scalar, scalar_end: scalar });
            }
        } else {
            for &start in &self.index.suffixes[self.lower..self.upper] {
                let end = start + self.bytes;
                let scalar_start = self.index.boundaries.binary_search(&start).map_err(|_| SourceLanguageError::IncompleteUtf8)?;
                let scalar_end = self.index.boundaries.binary_search(&end).map_err(|_| SourceLanguageError::IncompleteUtf8)?;
                output.push(SourceMatch { byte_start: start, byte_end: end, scalar_start, scalar_end });
            }
            output.sort_unstable();
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn index(s: &str) -> SourceLanguage { SourceLanguage::build(s, SourceLanguageLimits::default()).unwrap() }
    #[test]
    fn suffix_sort_equals_naive_for_repeated_and_multilingual_sources() {
        for source in ["", "a", "aaaaaaa", "banana", "mississippi", "éé a\n𐀀上海", "\0\u{1}\\\"\n"] {
            let x = index(source);
            let mut expected: Vec<_> = source.char_indices().map(|(i, _)| i).collect();
            expected.sort_unstable_by_key(|&i| &source.as_bytes()[i..]);
            assert_eq!(x.suffixes, expected);
        }
    }
    #[test]
    fn every_unicode_substring_recovers_all_overlapping_occurrences() {
        let source = "é上海é上海aaa"; let x = index(source);
        for &a in &x.boundaries {
            for &b in x.boundaries.iter().filter(|&&b| b >= a) {
                let needle = &source[a..b];
                let observed = x.matches(needle).unwrap();
                let expected: Vec<_> = x.boundaries.iter().copied()
                    .filter(|&i| source[i..].starts_with(needle)).collect();
                assert_eq!(observed.iter().map(|m| m.byte_start).collect::<Vec<_>>(), expected);
                for m in observed {
                    assert_eq!(&source[m.byte_start..m.byte_end], needle);
                    assert_eq!(m.scalar_start, source[..m.byte_start].chars().count());
                    assert_eq!(m.scalar_end, source[..m.byte_end].chars().count());
                }
            }
        }
    }
    #[test]
    fn byte_split_unicode_is_live_but_not_accepting() {
        let x = index("é"); let mut c = x.cursor();
        assert!(c.push_byte(0xc3)); assert!(!c.is_accepting());
        assert!(c.push_byte(0xa9)); assert!(c.is_accepting());
        assert!(!x.cursor().push_byte(0xa9));
    }
    #[test]
    fn branch_cursors_are_constant_size_and_independent() {
        let x = index("ab ac"); let mut a = x.cursor(); assert!(a.push_byte(b'a'));
        let mut b = a; assert!(a.push_byte(b'b')); assert!(b.push_byte(b'c'));
        assert_eq!(size_of::<SourceCursor<'_>>(), 4 * size_of::<usize>());
        assert!(!a.push_byte(b'z')); assert!(!a.push_byte(b'a')); assert!(b.is_accepting());
    }
    #[test]
    fn occurrence_caps_refuse_without_truncating() {
        let x = index("aaaa");
        assert_eq!(x.matches_bounded("aa", 2), Err(SourceLanguageError::MatchLimit));
        assert_eq!(x.matches_bounded("aa", 3).unwrap().len(), 3);
        assert_eq!(x.matches_bounded("", 4), Err(SourceLanguageError::MatchLimit));
    }
    #[test]
    fn construction_bounds_refuse_before_index_allocation() {
        let base = SourceLanguageLimits::default(); let need = SourceLanguage::preflight("banana", base).unwrap();
        assert_eq!(SourceLanguage::preflight("banana", SourceLanguageLimits { max_source_bytes: 5, ..base }), Err(SourceLanguageError::SourceLimit));
        assert_eq!(SourceLanguage::preflight("banana", SourceLanguageLimits { max_index_bytes: need.requested_peak_bytes - 1, ..base }), Err(SourceLanguageError::IndexLimit));
        assert_eq!(SourceLanguage::preflight("banana", SourceLanguageLimits { max_build_steps: need.build_step_bound - 1, ..base }), Err(SourceLanguageError::BuildLimit));
    }
    #[test]
    fn controlled_construction_preserves_exact_unicode_language_and_estimates() {
        let source = format!("{}\0\n上海{}", "éa".repeat(1400), "banana".repeat(800));
        let expected = index(&source);
        let mut calls = 0;
        let actual = SourceLanguage::build_with_control(&source, SourceLanguageLimits::default(), &mut || {
            calls += 1; None::<()>
        }).unwrap();
        assert!(calls > 20, "the complete source must not be one uninterruptible sort");
        assert_eq!(actual.source, expected.source);
        assert_eq!(actual.suffixes, expected.suffixes);
        assert_eq!(actual.boundaries, expected.boundaries);
        assert_eq!(actual.estimate, expected.estimate);
        for text in ["éaé", "banana", "上海", "\0\n", "a"] {
            assert_eq!(actual.matches(text), expected.matches(text));
        }
    }
    #[test]
    fn every_construction_checkpoint_returns_the_original_cancellation() {
        let source = "éa".repeat(1025);
        let limits = SourceLanguageLimits::default();
        let mut calls = 0;
        SourceLanguage::build_with_control(&source, limits, &mut || {
            calls += 1; None::<usize>
        }).unwrap();
        assert!(calls > 10);
        // Covers preflight, initialization, all radix passes, filtering and
        // the final completion checkpoint, with a one-shot cancellation.
        for cancel_at in 0..calls {
            let mut observed = 0;
            let result = SourceLanguage::build_with_control(&source, limits, &mut || {
                let at = observed; observed += 1;
                (at == cancel_at).then_some(cancel_at)
            });
            assert!(matches!(result, Err(SourceBuildError::Cancelled(cause)) if cause == cancel_at));
            assert_eq!(observed, cancel_at + 1, "no repoll after the original cancellation");
        }
    }
    #[test]
    fn source_preflight_scans_and_resource_refusals_remain_bounded() {
        let source = "😀".repeat(SOURCE_BUILD_CHECKPOINT_INTERVAL);
        let mut calls = 0;
        assert!(matches!(SourceLanguage::preflight_with_control(&source,
            SourceLanguageLimits::default(), &mut || {
                calls += 1; (calls == 2).then_some("original cause")
            }), Err(SourceBuildError::Cancelled("original cause"))));
        assert_eq!(calls, 2);
        let limits = SourceLanguageLimits { max_source_bytes: source.len() - 1, ..SourceLanguageLimits::default() };
        let mut calls = 0;
        assert!(matches!(SourceLanguage::build_with_control(&source, limits, &mut || {
            calls += 1; None::<()>
        }), Err(SourceBuildError::Source(SourceLanguageError::SourceLimit))));
        assert_eq!(calls, 1, "an oversized source is refused before counting it");
    }
    #[test]
    fn build_control_never_extends_the_fixed_work_quantum() {
        let mut calls = 0;
        {
            let mut checkpoint = || { calls += 1; None::<()> };
            let mut control = BuildControl { callback: &mut checkpoint, remaining: SOURCE_BUILD_CHECKPOINT_INTERVAL };
            control.checkpoint().unwrap();
            for _ in 0..SOURCE_BUILD_CHECKPOINT_INTERVAL - 1 { control.step().unwrap(); }
            assert_eq!(control.remaining, 1);
            control.step().unwrap();
            assert_eq!(control.remaining, SOURCE_BUILD_CHECKPOINT_INTERVAL);
        }
        assert_eq!(calls, 2);
    }
    #[test]
    fn no_normalization_or_lossy_match_is_permitted() {
        let x = index("e\u{301} Alice");
        assert_eq!(x.matches("é"), Err(SourceLanguageError::NoMatch));
        assert_eq!(x.matches("alice"), Err(SourceLanguageError::NoMatch));
    }
    #[test]
    fn debug_does_not_expose_source_or_suffixes() {
        let x = index("private document"); let s = format!("{x:?} {:?}", x.cursor());
        assert!(!s.contains("private")); assert!(!s.contains("suffixes"));
    }
}
