//! Refill scheduling/accounting fixtures. Synthetic drivers never model logits.
use super::*;

#[test]
fn receipt_replay_refills_finished_slots_and_preserves_fifo_request_order() {
    assert_eq!(expected_steps(&[2, 7, 1], &[3, 9, 1], 2, 4).unwrap(), 5);
    assert_eq!(expected_steps(&[2, 7, 1], &[3, 9, 1], 1, 4).unwrap(), 7);
    assert_eq!(expected_steps(&[2, 7, 1], &[3, 9, 1], 3, 64).unwrap(), 3);
}

#[test]
fn replay_handles_more_than_sixty_four_requests_with_bounded_physical_slots() {
    let prompts: Vec<usize> = (0..257).map(|i| i % 7 + 1).collect();
    let forwards: Vec<u64> = prompts.iter().enumerate().map(|(i, p)| (*p + i % 3) as u64).collect();
    for slots in [1, 2, 8, 64] {
        for width in [1, 4, 16, 64] {
            let steps = expected_steps(&prompts, &forwards, slots.min(prompts.len()), width).unwrap();
            assert!(steps > 0);
            assert!(steps <= forwards.iter().sum::<u64>());
        }
    }
}

#[test]
fn replay_rejects_zero_prompt_impossible_forward_and_invalid_geometry() {
    for (prompts, forwards, slots, width) in [
        (vec![], vec![], 1, 1), (vec![0], vec![1], 1, 1),
        (vec![2], vec![1], 1, 1), (vec![1], vec![1], 0, 1),
        (vec![1], vec![1], 2, 1), (vec![1], vec![1], 1, 65),
        (vec![1, 2], vec![1], 1, 1),
    ] {
        assert!(expected_steps(&prompts, &forwards, slots, width).is_err());
    }
}

#[test]
fn active_sampler_bound_scales_with_slots_not_epoch_population() {
    for slots in [1_u64, 4, 64] {
        let largest = 8192_u64;
        let active = times(largest, slots).unwrap();
        let population = times(largest, 1000).unwrap();
        assert!(active <= population);
        if slots < 64 { assert!(active < population); }
    }
}
