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

fn dense_mask_argmax(logits: &[f32], legal_count: usize) -> Option<(usize, f32)> {
    logits
        .iter()
        .take(legal_count)
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, value)| (index, *value))
}

fn sparse_rows_argmax(logits: &[f32], legal_count: usize) -> Option<(usize, f32)> {
    (0..legal_count)
        .map(|offset| (offset, logits[offset]))
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
        let dense_started = Instant::now();
        let dense = dense_mask_argmax(&logits, *legal_count);
        let dense_elapsed = dense_started.elapsed();
        let sparse_started = Instant::now();
        let sparse = sparse_rows_argmax(&logits, *legal_count);
        let sparse_elapsed = sparse_started.elapsed();
        assert_eq!(
            dense, sparse,
            "legal rows must retain exact argmax semantics"
        );
        println!(
            "G0_PROBE6 case=legal-set-{legal_count} RESULT=PASS mask_bytes={MASK_BYTES} full_logit_bytes={F32_LOGIT_BYTES} dense_ns={} sparse_ns={} authority=measurement-baseline-only",
            dense_elapsed.as_nanos(),
            sparse_elapsed.as_nanos()
        );
    }
    println!(
        "G0_PROBE6 RESULT=PASS cases={} authority=measurement-baseline-only",
        LEGAL_SET_SIZES.len()
    );
}
