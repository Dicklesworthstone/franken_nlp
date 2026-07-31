//! G0-10 scalar proofs for the two required non-saturating AVX2 constructions.
//!
//! The operations model the arithmetic used by the eventual intrinsics. This
//! target is not a Zen-3 throughput run and deliberately cannot ratify AVX2
//! dispatch on the current host.

const K: usize = 10_752;
const LOW7_PAIR_ABS_BOUND: i64 = 2 * 127 * 128;
const HIGH_BIT_PAIR_ABS_BOUND: i64 = 2 * 128;
const CORRECTION_PAIR_ABS_BOUND: i64 = 2 * 128 * 128;

fn scalar_i8_dot(left: &[i8], right: &[i8]) -> i64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| i64::from(*left) * i64::from(*right))
        .sum()
}

fn low7_high_bit_dot(left: &[i8], right: &[i8]) -> i64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            let unsigned_left = i64::from(*left) + 128;
            let low7 = unsigned_left & 0x7f;
            let high_bit = unsigned_left >> 7;
            let signed_right = i64::from(*right);
            low7 * signed_right + (high_bit - 1) * 128 * signed_right
        })
        .sum()
}

fn widened_i16_dot(left: &[i8], right: &[i8]) -> i64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| i64::from(i16::from(*left) * i16::from(*right)))
        .sum()
}

#[test]
fn low7_high_bit_and_widened_i16_models_match_every_i8_pair() {
    for left in i8::MIN..=i8::MAX {
        for right in i8::MIN..=i8::MAX {
            let expected = scalar_i8_dot(&[left], &[right]);
            assert_eq!(low7_high_bit_dot(&[left], &[right]), expected);
            assert_eq!(widened_i16_dot(&[left], &[right]), expected);
        }
    }
    println!("G0_PROBE10 case=all-i8-pairs RESULT=PASS authority=scalar-model-only");
}

#[test]
fn full_k_adversarial_vectors_preserve_i64_exactness_and_pair_bounds() {
    let cases = [
        (i8::MIN, i8::MIN),
        (i8::MIN, i8::MAX),
        (i8::MAX, i8::MIN),
        (i8::MAX, i8::MAX),
        (-1, i8::MIN),
        (0, i8::MAX),
    ];
    for (left_value, right_value) in cases {
        let left = vec![left_value; K];
        let right = vec![right_value; K];
        let expected = scalar_i8_dot(&left, &right);
        assert_eq!(low7_high_bit_dot(&left, &right), expected);
        assert_eq!(widened_i16_dot(&left, &right), expected);

        for (left_pair, right_pair) in left.chunks_exact(2).zip(right.chunks_exact(2)) {
            let low7_pair = left_pair
                .iter()
                .zip(right_pair)
                .map(|(left, right)| ((i64::from(*left) + 128) & 0x7f) * i64::from(*right))
                .sum::<i64>();
            let high_bit_pair = left_pair
                .iter()
                .zip(right_pair)
                .map(|(left, right)| ((i64::from(*left) + 128) >> 7) * i64::from(*right))
                .sum::<i64>();
            let correction_pair = right_pair
                .iter()
                .map(|right| 128 * i64::from(*right))
                .sum::<i64>();
            assert!(low7_pair.abs() <= LOW7_PAIR_ABS_BOUND);
            assert!(high_bit_pair.abs() <= HIGH_BIT_PAIR_ABS_BOUND);
            assert!(correction_pair.abs() <= CORRECTION_PAIR_ABS_BOUND);
        }
        println!(
            "G0_PROBE10 case=full-k-left-{left_value}-right-{right_value} RESULT=PASS k={K} scalar={expected} authority=scalar-model-only"
        );
    }
    println!(
        "G0_PROBE10 RESULT=PASS cases={} authority=scalar-model-only",
        cases.len() + 1
    );
}
