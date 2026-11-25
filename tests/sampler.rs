use franken_nlp::native_engine::{
    lmhead,
    sampler::{
        DrawAddress, Nucleus, RankedToken, SAMPLER_VERSION, SamplerError, Seed256,
        StableRequestKey, TopK, addressed_uniform, draw_digest, greedy_argmax, uniform_from_digest,
    },
};

#[test]
fn seed_parser_freezes_lower_hex_and_decimal_u64_expansion() {
    let hex = "000000000000000000000000000000000000000000000000000000000000002a";
    assert_eq!(Seed256::parse_cli(hex).unwrap().to_lower_hex(), hex);
    assert_eq!(
        Seed256::parse_cli("42").unwrap().to_lower_hex(),
        "1b1b64cea33cada4ac32cdab77fe25b1444bb6ebc7ea1e7ea5f32c73c030930a"
    );
    assert!(Seed256::parse_cli("A".repeat(64).as_str()).is_err());
    assert!(Seed256::parse_cli("-1").is_err());
    assert!(Seed256::parse_cli("18446744073709551616").is_err());
}

#[test]
fn seed_parser_trims_surrounding_whitespace() {
    let hex = "000000000000000000000000000000000000000000000000000000000000002a";
    // Trailing newline (the most common shell here-doc and env-var case).
    assert_eq!(
        Seed256::parse_cli(&format!("{hex}\n")).unwrap().to_lower_hex(),
        hex,
    );
    // Leading and trailing whitespace.
    assert_eq!(
        Seed256::parse_cli(&format!("  {hex}  ")).unwrap().to_lower_hex(),
        hex,
    );
    // Trailing tab.
    assert_eq!(
        Seed256::parse_cli(&format!("{hex}\t")).unwrap().to_lower_hex(),
        hex,
    );
    // Decimal form with trailing newline.
    assert!(Seed256::parse_cli("42\n").is_ok());
    // Whitespace-only is still rejected.
    assert!(Seed256::parse_cli("   \n  ").is_err());
    // Genuinely wrong hex with whitespace is still rejected.
    assert!(Seed256::parse_cli(&format!("{}B\n", "a".repeat(63))).is_err());
}

#[test]
fn draw_digest_is_length_framed_and_addressable() {
    assert_eq!(SAMPLER_VERSION, "fnlp-sampler-v1");
    let mut seed_bytes = [0_u8; 32];
    seed_bytes[31] = 0x2a;
    let seed = Seed256::from(seed_bytes);
    let key = StableRequestKey::from_canonical_digest([0x11; 32]);
    let address = DrawAddress::new(3, 7, 11);
    let digest = draw_digest(seed, key, address);
    assert_eq!(
        hex(&digest),
        "d8968b4501277f7e0429127c50c0751ef7b0748a39c57373a80bfee31aedbda9"
    );
    assert_eq!(
        addressed_uniform(seed, key, address),
        uniform_from_digest(digest)
    );
    assert_ne!(
        draw_digest(seed, key, DrawAddress::new(3, 7, 12)),
        digest,
        "draw index must be an address field, not shared stream state"
    );
}

#[test]
fn seeded_draws_replay_across_batch_reordering_and_resume() {
    let seed = Seed256::from_u64(99);
    let key = StableRequestKey::from_canonical_digest([0x5a; 32]);
    let row_zero = DrawAddress::new(0, 4, 0);
    let row_one = DrawAddress::new(1, 4, 0);
    let original = [
        (
            row_zero.sample_index,
            addressed_uniform(seed, key, row_zero),
        ),
        (row_one.sample_index, addressed_uniform(seed, key, row_one)),
    ];
    let reordered = [
        (row_one.sample_index, addressed_uniform(seed, key, row_one)),
        (
            row_zero.sample_index,
            addressed_uniform(seed, key, row_zero),
        ),
    ];
    assert_eq!(original[0].1, reordered[1].1);
    assert_eq!(original[1].1, reordered[0].1);
    assert_eq!(
        addressed_uniform(seed, key, DrawAddress::new(1, 4, 3)),
        addressed_uniform(seed, key, DrawAddress::new(1, 4, 3)),
        "resuming the same semantic coordinate must reproduce the same draw"
    );
}

#[test]
fn digest_mapping_uses_exact_53_bit_half_open_interval() {
    assert_eq!(uniform_from_digest([0_u8; 32]), 0.0);
    let mut largest = [0_u8; 32];
    largest[..8].copy_from_slice(&0xffff_ffff_ffff_f800_u64.to_be_bytes());
    let expected = ((1_u64 << 53) - 1) as f64 / (1_u64 << 53) as f64;
    assert_eq!(uniform_from_digest(largest), expected);
    assert!(uniform_from_digest(largest) < 1.0);
}

#[test]
fn greedy_delegates_to_lmhead_without_random_state() {
    let cases = [
        vec![3.0, 3.0, 2.0],
        vec![f32::NAN, -1.0, 0.0],
        vec![f32::NEG_INFINITY, f32::INFINITY, 0.0],
    ];
    for logits in cases {
        assert_eq!(greedy_argmax(&logits), lmhead::greedy_argmax(&logits));
    }
}

#[test]
fn fixed_heap_matches_full_sort_and_lowest_id_tie_break() {
    let mut logits: Vec<f32> = (0..25).map(|value| value as f32).collect();
    logits[23] = 24.0;
    let selected = TopK::select(&logits, 20).unwrap();
    let ids: Vec<usize> = selected
        .as_slice()
        .iter()
        .copied()
        .map(RankedToken::token_id)
        .collect();
    assert_eq!(
        ids,
        vec![
            23, 24, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5
        ]
    );
    assert_eq!(selected.select_uniform(0.0), Ok(23));
    assert_eq!(
        selected.select_uniform(f64::from_bits(0x3fef_ffff_ffff_ffff)),
        Ok(5)
    );
    assert_eq!(
        TopK::select(&[1.0], 21),
        Err(SamplerError::InvalidTopK { requested: 21 })
    );
}

#[test]
fn exact_top_p_keeps_crossing_entry_and_uses_half_open_selection() {
    let nucleus = Nucleus::exact_top_p(&[(4.0_f32).ln(), (2.0_f32).ln(), 0.0], 0.8).unwrap();
    let retained: Vec<usize> = nucleus
        .as_slice()
        .iter()
        .map(|entry| entry.token().token_id())
        .collect();
    assert_eq!(retained, vec![0, 1]);
    assert!((nucleus.cutoff_mass() - (6.0 / 7.0)).abs() < 1.0e-6);
    assert_eq!(nucleus.select_uniform(0.0), Ok(0));
    assert_eq!(nucleus.select_uniform(0.7), Ok(1));
    assert_eq!(
        nucleus.select_uniform(1.0),
        Err(SamplerError::InvalidUniform)
    );
    assert_eq!(
        Nucleus::exact_top_p(&[2.0, 1.0, 0.0], 1.0)
            .unwrap()
            .as_slice()
            .len(),
        3,
        "top-p=1 retains the complete vocabulary"
    );
}

#[test]
fn stochastic_paths_reject_nonfinite_logits_without_panicking() {
    assert_eq!(
        TopK::select(&[0.0, f32::NAN], 1),
        Err(SamplerError::NonFiniteLogit { token_id: 1 })
    );
    assert_eq!(
        Nucleus::exact_top_p(&[f32::INFINITY], 0.95),
        Err(SamplerError::NonFiniteLogit { token_id: 0 })
    );
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}


#[derive(Default)]
struct SamplingControl {
    calls: usize,
    stop_at: Option<usize>,
    cause: Option<franken_nlp::native_engine::decode::DecodeCancellationKind>,
}
impl franken_nlp::native_engine::decode::DecodeStepControl for SamplingControl {
    fn checkpoint(&mut self, step: usize) -> Option<franken_nlp::native_engine::decode::DecodeCancellationKind> {
        assert_eq!(step, 7, "sampling must preserve the caller's semantic decode step");
        self.calls += 1;
        if self.stop_at == Some(self.calls) { self.cause } else { None }
    }
}

fn sampling_causes() -> [franken_nlp::native_engine::decode::DecodeCancellationKind; 11] {
    use franken_nlp::native_engine::decode::DecodeCancellationKind::*;
    [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit]
}

fn assert_nucleus_bits(actual: &Nucleus, expected: &Nucleus) {
    assert_eq!(actual.cutoff_mass().to_bits(), expected.cutoff_mass().to_bits());
    assert_eq!(actual.as_slice().len(), expected.as_slice().len());
    for (actual, expected) in actual.as_slice().iter().zip(expected.as_slice()) {
        assert_eq!(actual.token().token_id(), expected.token().token_id());
        assert_eq!(actual.token().logit().to_bits(), expected.token().logit().to_bits());
        assert_eq!(actual.probability().to_bits(), expected.probability().to_bits());
    }
}

#[test]
fn controlled_sampling_preserves_full_vocabulary_probability_bits_and_addressed_choices() {
    use franken_nlp::native_engine::lmhead::NANBEIGE_VOCAB_SIZE;
    for length in [1, 2, 3, 255, 256, 257, 1025, NANBEIGE_VOCAB_SIZE] {
        let logits: Vec<_> = (0..length).map(|index| match index % 17 {
            0 => -0.0, 1 => 0.0, 2 => f32::MIN_POSITIVE, 3 => -f32::MIN_POSITIVE,
            _ => ((index * 31 % 109) as f32 - 54.0) / 7.0,
        }).collect();
        for top_p in [0.000_001, 0.37, 0.95, 1.0] {
            let expected = Nucleus::exact_top_p(&logits, top_p).unwrap();
            let actual = Nucleus::exact_top_p_with_control(&logits, top_p,
                &mut SamplingControl::default(), 7).unwrap();
            assert_nucleus_bits(&actual, &expected);
            for uniform in [0.0, 0.5, f64::from_bits(0x3fef_ffff_ffff_ffff)] {
                assert_eq!(actual.select_uniform_with_control(uniform, &mut SamplingControl::default(), 7),
                    expected.select_uniform(uniform));
            }
        }
        for k in [1, 20] {
            let expected = TopK::select(&logits, k).unwrap();
            let actual = TopK::select_with_control(&logits, k, &mut SamplingControl::default(), 7).unwrap();
            for (actual, expected) in actual.as_slice().iter().zip(expected.as_slice()) {
                assert_eq!(actual.token_id(), expected.token_id());
                assert_eq!(actual.logit().to_bits(), expected.logit().to_bits());
            }
            assert_eq!(actual.as_slice().len(), expected.as_slice().len());
            assert_eq!(actual.select_uniform(0.73), expected.select_uniform(0.73));
        }
    }
    for logits in [vec![f32::MAX, -f32::MAX, 0.0], vec![-0.0, 0.0, -0.0, 0.0]] {
        assert_nucleus_bits(&Nucleus::exact_top_p_with_control(&logits, 1.0,
            &mut SamplingControl::default(), 7).unwrap(), &Nucleus::exact_top_p(&logits, 1.0).unwrap());
    }
}

#[test]
fn every_sampler_checkpoint_preserves_every_typed_cancellation_cause() {
    let logits: Vec<_> = (0..769).map(|index| ((index * 37 % 101) as f32 - 50.0) / 9.0).collect();
    let mut completed = SamplingControl::default();
    Nucleus::exact_top_p_with_control(&logits, 1.0, &mut completed, 7).unwrap();
    assert!(completed.calls > 10, "all numerical, sorting and cutoff phases must checkpoint");
    for cause in sampling_causes() {
        for stop_at in 1..=completed.calls {
            let mut control = SamplingControl { stop_at: Some(stop_at), cause: Some(cause), ..Default::default() };
            assert!(matches!(Nucleus::exact_top_p_with_control(&logits, 1.0, &mut control, 7),
                Err(SamplerError::Cancelled(observed)) if observed == cause));
            assert_eq!(control.calls, stop_at, "a stopped sampler must do no later checkpointed work");
        }
        let mut completed_top = SamplingControl::default();
        TopK::select_with_control(&logits, 20, &mut completed_top, 7).unwrap();
        for stop_at in 1..=completed_top.calls {
            let mut control = SamplingControl { stop_at: Some(stop_at), cause: Some(cause), ..Default::default() };
            assert!(matches!(TopK::select_with_control(&logits, 20, &mut control, 7),
                Err(SamplerError::Cancelled(observed)) if observed == cause));
        }
    }
}

#[test]
fn cancellation_bounds_validation_and_dense_nucleus_draw_traversals() {
    use franken_nlp::native_engine::sampler::SAMPLING_CHECKPOINT_STRIDE;
    let mut invalid = vec![0.0; SAMPLING_CHECKPOINT_STRIDE + 1];
    invalid[SAMPLING_CHECKPOINT_STRIDE] = f32::NAN;
    let nucleus = Nucleus::exact_top_p(&vec![0.0; 1025], 1.0).unwrap();
    let mut completed = SamplingControl::default();
    nucleus.select_uniform_with_control(0.999_999, &mut completed, 7).unwrap();
    for cause in sampling_causes() {
        let mut control = SamplingControl { stop_at: Some(2), cause: Some(cause), ..Default::default() };
        assert!(matches!(Nucleus::exact_top_p_with_control(&invalid, 1.0, &mut control, 7),
            Err(SamplerError::Cancelled(observed)) if observed == cause));
        let mut control = SamplingControl { stop_at: Some(2), cause: Some(cause), ..Default::default() };
        assert!(matches!(TopK::select_with_control(&invalid, 20, &mut control, 7),
            Err(SamplerError::Cancelled(observed)) if observed == cause));
        for stop_at in 1..=completed.calls {
            let mut control = SamplingControl { stop_at: Some(stop_at), cause: Some(cause), ..Default::default() };
            assert!(matches!(nucleus.select_uniform_with_control(0.999_999, &mut control, 7),
                Err(SamplerError::Cancelled(observed)) if observed == cause));
        }
    }
}
