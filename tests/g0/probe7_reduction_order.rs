//! G0-07 fixed row-order reduction probe.
//!
//! Each candidate batch width calls the one canonical per-row accumulator. The
//! test protects the bitwise contract needed for batch-M versus batch-1; it
//! does not claim that reassociated/vectorized reductions are equivalent.

const K_SHAPES: &[usize] = &[1_024, 3_072, 6_144, 10_752];
const BATCH_WIDTHS: &[usize] = &[1, 8, 64];

fn seeded_value(seed: u32) -> f32 {
    let mantissa = (seed.wrapping_mul(1_103_515_245).wrapping_add(12_345) >> 8) & 0x00ff_ffff;
    (mantissa as f32 / 8_388_608.0) - 1.0
}

fn canonical_dot(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), right.len());
    let mut accumulator = 0.0_f32;
    for (left, right) in left.iter().zip(right) {
        accumulator += left * right;
    }
    accumulator
}

fn reverse_dot(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), right.len());
    let mut accumulator = 0.0_f32;
    for (left, right) in left.iter().zip(right).rev() {
        accumulator += left * right;
    }
    accumulator
}

fn batch_canonical_dot(rows: &[Vec<f32>], right: &[f32]) -> Vec<f32> {
    rows.iter().map(|row| canonical_dot(row, right)).collect()
}

#[test]
fn canonical_per_row_order_is_bitwise_batch_invariant_at_model_k_shapes() {
    for k in K_SHAPES {
        let right = (0..*k)
            .map(|index| seeded_value(index as u32 ^ 0x17))
            .collect::<Vec<_>>();
        for batch_width in BATCH_WIDTHS {
            let rows = (0..*batch_width)
                .map(|row| {
                    (0..*k)
                        .map(|column| seeded_value((row * k + column) as u32 ^ 0xa5))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let batched = batch_canonical_dot(&rows, &right);
            for (row, result) in rows.iter().zip(&batched) {
                assert_eq!(
                    canonical_dot(row, &right).to_bits(),
                    result.to_bits(),
                    "canonical order changed for k={k} batch_width={batch_width}"
                );
            }
            let reverse_matches = rows
                .iter()
                .zip(&batched)
                .all(|(row, result)| reverse_dot(row, &right).to_bits() == result.to_bits());
            println!(
                "G0_PROBE7 case=k-{k}-m-{batch_width} RESULT=PASS reduction=canonical-row-major reverse_candidate={} authority=scalar-order-only",
                if reverse_matches {
                    "not-distinguished"
                } else {
                    "rejected"
                }
            );
        }
    }
    println!(
        "G0_PROBE7 RESULT=PASS cases={} authority=scalar-order-only",
        K_SHAPES.len() * BATCH_WIDTHS.len()
    );
}

#[test]
fn reassociated_order_has_a_named_bitwise_counterexample() {
    let mut left = vec![0.0_f32; 1_024];
    // 2^25: one power beyond f32's 24-bit mantissa, so -(2^25) + 1.0 rounds
    // back to -(2^25) and the reassociated order genuinely absorbs the unit.
    // (At 2^24 the sum -(2^24 - 1) is still exactly representable and the
    // counterexample vanishes.)
    left[..3].copy_from_slice(&[33_554_432.0, -33_554_432.0, 1.0]);
    let right = vec![1.0_f32; left.len()];
    let canonical = canonical_dot(&left, &right);
    let a = left[0] * right[0];
    let b = left[1] * right[1];
    let c = left[2] * right[2];
    let sequential = (a + b) + c;
    let reassociated = a + (b + c);

    assert_eq!(canonical.to_bits(), sequential.to_bits());
    assert_eq!(sequential, 1.0, "(a + b) + c keeps the unit term");
    assert_eq!(reassociated, 0.0, "a + (b + c) loses the unit term");
    assert_ne!(sequential.to_bits(), reassociated.to_bits());
    println!(
        "G0_PROBE7 case=reassociated-counterexample RESULT=PASS sequential_bits={:08x} reassociated_bits={:08x} sequential_value={} reassociated_value={} decision=reassociated-order-rejected authority=scalar-order-only",
        sequential.to_bits(),
        reassociated.to_bits(),
        sequential,
        reassociated
    );
}
