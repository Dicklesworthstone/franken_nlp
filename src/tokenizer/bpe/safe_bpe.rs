//! Explicit control-free BPE for untrusted segments, distinct from L0 encoding.
//!
//! The first source scalar is emitted as raw BYTE pieces so the decoder's
//! initial-marker state is already consumed. The rest uses ordinary pieces
//! without a dummy prefix or added-token matching. Literal U+2581 is a raw-byte
//! boundary, never a whitespace marker; arbitrary invalid UTF-8 stays byte-only.
//! Merges are score-ordered with the existing leftmost tie rule, restricted to
//! IDs that are neither forbidden nor overridden by injected-token decoding.
//! The simple quadratic search is explicit and bounded by comparisons AND
//! examined pair bytes. No new performance or tokenizer-parity claim is made.

use std::collections::BTreeSet;

use super::{DecoderPiece, EncodeError, SpBpeTokenizer, WHITESPACE_MARKER};
use crate::tokenizer::untrusted::{SafeBpeLimits, SafeBpeWork, UntrustedDocumentError, poll};

struct SymbolSpan {
    start: usize,
    end: usize,
    next: Option<usize>,
}

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
            symbols.push(SymbolSpan { start, end: normalized.len(), next: Some(index + 1) });
        }
        if let Some(last) = symbols.last_mut() { last.next = None; }

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
}
