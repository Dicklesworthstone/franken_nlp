//! Explicit control-free BPE for untrusted segments, distinct from L0 encoding.
//!
//! The first source scalar is emitted as raw BYTE pieces so the decoder's
//! initial-marker state is already consumed. The rest uses ordinary pieces
//! without a dummy prefix or added-token matching. Literal U+2581 is a raw-byte
//! boundary, never a whitespace marker; arbitrary invalid UTF-8 stays byte-only.
//! Merges are score-ordered with the existing leftmost tie rule, restricted to
//! IDs that are neither forbidden nor overridden by injected-token decoding.
//! Long segments retain ranked adjacent pairs across merges. Work accounting
//! preserves the V1 full-scan pair counts and byte limits on every segment.

use std::collections::{BTreeSet, BinaryHeap};

use super::{DecoderPiece, EncodeError, Merge, SpBpeTokenizer, Symbol, WHITESPACE_MARKER};
use crate::tokenizer::untrusted::{SafeBpeLimits, SafeBpeWork, UntrustedDocumentError, poll};

struct Encoder<'a, F> {
    tokenizer: &'a SpBpeTokenizer,
    forbidden: &'a BTreeSet<u32>,
    limits: SafeBpeLimits,
    work: &'a mut SafeBpeWork,
    checkpoint: &'a mut F,
    ids: Vec<u32>,
}

pub(super) fn encode<F: FnMut() -> bool>(
    tokenizer: &SpBpeTokenizer,
    input: &[u8],
    forbidden: &BTreeSet<u32>,
    limits: SafeBpeLimits,
    work: &mut SafeBpeWork,
    checkpoint: &mut F,
) -> Result<Vec<u32>, UntrustedDocumentError> {
    let remaining = limits.max_output_tokens.checked_sub(work.output_tokens())
        .ok_or(UntrustedDocumentError::EncodingBudget { resource: "output tokens" })?;
    let mut ids = Vec::new();
    ids.try_reserve_exact(input.len().min(remaining)).map_err(|_| UntrustedDocumentError::AllocationRefused)?;
    let mut encoder = Encoder { tokenizer, forbidden, limits, work, checkpoint, ids };
    let Ok(text) = std::str::from_utf8(input) else {
        encoder.bytes(input)?;
        return Ok(encoder.ids);
    };
    let Some(first) = text.chars().next() else { return Ok(encoder.ids); };
    // The leading token is exactly the first raw source byte. A complete
    // initial scalar keeps the remainder on a valid UTF-8 boundary.
    encoder.bytes(&input[..first.len_utf8()])?;
    let mut remainder = &text[first.len_utf8()..];
    while let Some(marker) = remainder.find(WHITESPACE_MARKER) {
        encoder.ordinary(&remainder[..marker])?;
        encoder.bytes(b"\xe2\x96\x81")?;
        remainder = &remainder[marker + WHITESPACE_MARKER.len_utf8()..];
    }
    encoder.ordinary(remainder)?;
    poll(encoder.checkpoint)?;
    Ok(encoder.ids)
}

impl<F: FnMut() -> bool> Encoder<'_, F> {
    fn allowed(&self, id: u32) -> bool {
        !self.forbidden.contains(&id) && !self.tokenizer.injected_by_id.contains_key(&id)
    }

    fn push(&mut self, id: u32) -> Result<(), UntrustedDocumentError> {
        if !self.allowed(id) { return Err(UntrustedDocumentError::UnsafeTokenId { id }); }
        self.work.token(self.limits)?;
        self.ids.try_reserve(1).map_err(|_| UntrustedDocumentError::AllocationRefused)?;
        self.ids.push(id);
        Ok(())
    }

    fn bytes(&mut self, input: &[u8]) -> Result<(), UntrustedDocumentError> {
        for (offset, &byte) in input.iter().enumerate() {
            if offset % 256 == 0 { poll(self.checkpoint)?; }
            let id = self.tokenizer.byte_pieces[usize::from(byte)]
                .ok_or(UntrustedDocumentError::Encode(EncodeError::ByteFallbackMissing { byte }))?;
            if !matches!(self.tokenizer.piece_for_id(id), Ok(DecoderPiece::Byte(value)) if *value == byte) {
                return Err(UntrustedDocumentError::UnsafeTokenId { id });
            }
            self.push(id)?;
        }
        Ok(())
    }

    fn ordinary(&mut self, text: &str) -> Result<(), UntrustedDocumentError> {
        if text.is_empty() { return Ok(()); }
        poll(self.checkpoint)?;
        // Source-byte limits were charged before any allocation. At most one
        // span exists per source byte and normalization expands a space 1→3.
        let mut normalized = String::new();
        normalized.try_reserve_exact(text.len().checked_mul(3)
            .ok_or(UntrustedDocumentError::AllocationRefused)?)
            .map_err(|_| UntrustedDocumentError::AllocationRefused)?;
        let mut symbols = Vec::new();
        symbols.try_reserve_exact(text.len()).map_err(|_| UntrustedDocumentError::AllocationRefused)?;
        for (index, character) in text.chars().enumerate() {
            if index % 256 == 0 { poll(self.checkpoint)?; }
            let start = normalized.len();
            normalized.push(if character == ' ' { WHITESPACE_MARKER } else { character });
            symbols.push(Symbol { start, end: normalized.len(), next: Some(index + 1) });
        }
        if let Some(last) = symbols.last_mut() { last.next = None; }

        if symbols.len() <= 32 {
            self.merge_rescan(&normalized, &mut symbols)?;
        } else {
            self.merge_frontier(&normalized, &mut symbols)?;
        }

        let mut current = Some(0);
        let mut emitted = 0_usize;
        while let Some(index) = current {
            if emitted % 256 == 0 { poll(self.checkpoint)?; }
            let surface = &normalized[symbols[index].start..symbols[index].end];
            if let Some(id) = self.tokenizer.normal_pieces.get(surface)
                .filter(|piece| self.allowed(piece.id)).map(|piece| piece.id) {
                self.push(id)?;
            } else {
                // Missing/forbidden singleton pieces fall back to ORIGINAL
                // bytes. A normalized marker here represents an ASCII space;
                // literal source markers were separated before this method.
                for character in surface.chars() {
                    if character == WHITESPACE_MARKER {
                        self.bytes(b" ")?;
                    } else {
                        let mut buffer = [0_u8; 4];
                        self.bytes(character.encode_utf8(&mut buffer).as_bytes())?;
                    }
                }
            }
            emitted += 1;
            current = symbols[index].next;
        }
        Ok(())
    }

    fn merge_rescan(&mut self, normalized: &str, symbols: &mut [Symbol]) -> Result<(), UntrustedDocumentError> {
        loop {
            let mut best: Option<(usize, usize, f32)> = None;
            let mut left = Some(0);
            while let Some(index) = left {
                let Some(right) = symbols[index].next else { break; };
                let pair = &normalized[symbols[index].start..symbols[right].end];
                self.work.pair(pair.len(), self.limits)?;
                if self.work.pair_checks() % 256 == 1 { poll(self.checkpoint)?; }
                if let Some(piece) = self.tokenizer.normal_pieces.get(pair).filter(|piece| self.allowed(piece.id)) {
                    if best.as_ref().is_none_or(|(_, _, score)| piece.score.total_cmp(score).is_gt()) {
                        best = Some((index, right, piece.score));
                    }
                }
                left = symbols[index].next;
            }
            let Some((left, right, _)) = best else { break; };
            symbols[left].end = symbols[right].end;
            symbols[left].next = symbols[right].next;
        }
        Ok(())
    }

    fn enqueue(&self, queue: &mut BinaryHeap<Merge>, normalized: &str, symbols: &[Symbol], left: usize)
        -> Result<(), UntrustedDocumentError> {
        if let Some(right) = symbols[left].next {
            let end = symbols[right].end;
            let pair = &normalized[symbols[left].start..end];
            if let Some(piece) = self.tokenizer.normal_pieces.get(pair).filter(|piece| self.allowed(piece.id)) {
                queue.try_reserve(1).map_err(|_| UntrustedDocumentError::AllocationRefused)?;
                queue.push(Merge { score: piece.score, left, end });
            }
        }
        Ok(())
    }

    fn merge_frontier(&mut self, normalized: &str, symbols: &mut [Symbol]) -> Result<(), UntrustedDocumentError> {
        let mut previous = Vec::new();
        previous.try_reserve_exact(symbols.len()).map_err(|_| UntrustedDocumentError::AllocationRefused)?;
        let mut queue = BinaryHeap::new();
        let mut tail = symbols.len() - 1;
        let mut live = symbols.len();
        for (left, symbol) in symbols.iter().enumerate() {
            previous.push(left.checked_sub(1));
            if let Some(right) = symbol.next {
                self.work.pair(symbols[right].end - symbol.start, self.limits)?;
                if self.work.pair_checks() % 256 == 1 { poll(self.checkpoint)?; }
                self.enqueue(&mut queue, normalized, symbols, left)?;
            }
        }

        let mut pops = 0_usize;
        while let Some(Merge { left, end, .. }) = queue.pop() {
            if pops % 256 == 0 { poll(self.checkpoint)?; }
            pops += 1;
            // A consumed symbol has no successor. Growing either side of an
            // old adjacency moves its combined end strictly to the right.
            let Some(right) = symbols[left].next else { continue; };
            if symbols[right].end != end { continue; }
            symbols[left].end = end;
            symbols[left].next = symbols[right].next.take();
            if let Some(after) = symbols[left].next {
                previous[after] = Some(left);
            } else {
                tail = left;
            }
            live -= 1;

            // The V1 scan visits each interior span twice and each boundary
            // span once. The segment is contiguous, so these are the exact
            // logical charges, including the last scan that discovers no merge.
            let pair_bytes = (normalized.len() - symbols[0].end) + symbols[tail].start;
            if !self.work.try_charge_pairs(live - 1, pair_bytes, self.limits) {
                // Only the final, unaffordable scan needs individual charges.
                // This preserves both the original refusal reason and all work
                // consumed before the first unaffordable pair.
                let mut index = 0;
                while let Some(next) = symbols[index].next {
                    self.work.pair(symbols[next].end - symbols[index].start, self.limits)?;
                    if self.work.pair_checks() % 256 == 1 { poll(self.checkpoint)?; }
                    index = next;
                }
            }
            if let Some(before) = previous[left] {
                self.enqueue(&mut queue, normalized, symbols, before)?;
            }
            self.enqueue(&mut queue, normalized, symbols, left)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::embedded::EmbeddedTokenizer;

    #[test]
    fn frontier_preserves_v1_merges_and_partial_budget_charges() {
        let pinned = EmbeddedTokenizer::pinned().unwrap();
        let tokenizer = pinned.tokenizer();
        let mut forbidden = BTreeSet::new();
        if let Some(piece) = tokenizer.normal_pieces.get("aa") {
            forbidden.insert(piece.id);
        }
        for normalized in ["a".repeat(71), "banana▁你好é".repeat(12), "abababa".repeat(11)] {
            for (checks, bytes) in [(100_000, 1_000_000), (17, 1000), (233, 1000), (850, 508), (90, 208)] {
                let limits = SafeBpeLimits {
                    max_input_bytes: 4096,
                    max_output_tokens: 4096,
                    max_pair_checks: checks,
                    max_pair_bytes: bytes,
                };
                let run = |frontier: bool| {
                    let mut work = SafeBpeWork::default();
                    work.pair(3, limits).unwrap();
                    let mut symbols = Symbol::from_text(&normalized);
                    let mut checkpoint = || true;
                    let mut encoder = Encoder {
                        tokenizer, forbidden: &forbidden, limits, work: &mut work,
                        checkpoint: &mut checkpoint, ids: Vec::new(),
                    };
                    let result = if frontier {
                        encoder.merge_frontier(&normalized, &mut symbols)
                    } else {
                        encoder.merge_rescan(&normalized, &mut symbols)
                    };
                    let mut live = Vec::new();
                    let mut current = Some(0);
                    while let Some(index) = current {
                        live.push((symbols[index].start, symbols[index].end));
                        current = symbols[index].next;
                    }
                    (result, work, live)
                };
                assert_eq!(run(true), run(false), "limits={limits:?} normalized={normalized}");
            }
        }
    }
}
