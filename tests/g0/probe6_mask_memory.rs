//! G0-06 dense-mask and sparse-projection accounting probe.
//!
//! The result is an accounting baseline, not a dispatch choice. Production
//! routing remains BLOCKED on the ADR's host-specific crossover evidence.

use std::time::Instant;

const VOCAB_SIZE: usize = 166_144;
const MASK_BYTES: usize = VOCAB_SIZE.div_ceil(8);
const F32_LOGIT_BYTES: usize = VOCAB_SIZE * std::mem::size_of::<f32>();
const LEGAL_SET_SIZES: &[usize] = &[1, 8, 64, 512, 4_096, 32_768, VOCAB_SIZE];

fn deterministic_logits() -> Vec<f32> {
    (0..VOCAB_SIZE)
        .map(|index| (((index.wrapping_mul(17) ^ 0x5a5a) % 4096) as f32 / 64.0) - 32.0)
        .collect()
}

fn set_legal(mask: &mut [u8], index: usize) {
    mask[index / 8] |= 1_u8 << (index % 8);
}

fn is_legal(mask: &[u8], index: usize) -> bool {
    (mask[index / 8] & (1_u8 << (index % 8))) != 0
}

fn intersect_masks(left: &[u8], right: &[u8]) -> Vec<u8> {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(left, right)| left & right)
        .collect()
}

fn evenly_spaced_legal_rows(legal_count: usize) -> Vec<usize> {
    assert!((1..=VOCAB_SIZE).contains(&legal_count));
    (0..legal_count)
        .map(|offset| offset * VOCAB_SIZE / legal_count)
        .collect()
}

fn dense_mask_argmax(logits: &[f32], legal_mask: &[u8]) -> Option<(usize, f32)> {
    assert_eq!(logits.len(), VOCAB_SIZE);
    assert_eq!(legal_mask.len(), MASK_BYTES);
    logits
        .iter()
        .enumerate()
        .filter(|(index, _value)| is_legal(legal_mask, *index))
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, value)| (index, *value))
}

fn sparse_rows_argmax(logits: &[f32], legal_rows: &[usize]) -> Option<(usize, f32)> {
    legal_rows
        .iter()
        .map(|index| (*index, logits[*index]))
        .max_by(|left, right| left.1.total_cmp(&right.1))
}

#[test]
fn mask_and_projection_accounting_agree_for_every_preregistered_legal_set() {
    assert_eq!(
        MASK_BYTES, 20_768,
        "166,144 legal bits must cost 20,768 bytes"
    );
    assert_eq!(
        F32_LOGIT_BYTES, 664_576,
        "full f32 vocabulary logits byte count"
    );

    let logits = deterministic_logits();
    for legal_count in LEGAL_SET_SIZES {
        let legal_rows = evenly_spaced_legal_rows(*legal_count);
        let mut candidate_mask = vec![0_u8; MASK_BYTES];
        for index in &legal_rows {
            set_legal(&mut candidate_mask, *index);
        }
        let allow_all_mask = vec![u8::MAX; MASK_BYTES];
        let mask_started = Instant::now();
        let legal_mask = intersect_masks(&candidate_mask, &allow_all_mask);
        let mask_elapsed = mask_started.elapsed();
        let dense_started = Instant::now();
        let dense = dense_mask_argmax(&logits, &legal_mask);
        let dense_elapsed = dense_started.elapsed();
        let sparse_started = Instant::now();
        let sparse = sparse_rows_argmax(&logits, &legal_rows);
        let sparse_elapsed = sparse_started.elapsed();
        assert_eq!(
            dense, sparse,
            "dense bitset masking and row-sliced projection must retain exact argmax semantics"
        );
        println!(
            "G0_PROBE6 case=legal-set-{legal_count} RESULT=PASS mask_bytes={MASK_BYTES} full_logit_bytes={F32_LOGIT_BYTES} mask_and_ns={} dense_full_vocab_ns={} sparse_rows_ns={} authority=local-measurement-only",
            mask_elapsed.as_nanos(),
            dense_elapsed.as_nanos(),
            sparse_elapsed.as_nanos()
        );
    }
    println!(
        "G0_PROBE6 RESULT=PASS cases={} authority=measurement-baseline-only",
        LEGAL_SET_SIZES.len()
    );
}
