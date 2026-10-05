//! Finite round-robin token allocation and receipt replay. No token contents,
//! request IDs, random draws, worker numbers or elapsed time enter this policy.
use super::*;

/// At most width grants, each to the next non-exhausted document. A decode row
/// advertises exactly ONE available token; prompt rows can advertise a morsel.
/// Returns per-document counts and the next fairness cursor, without mutation.
pub(super) fn allocate(available: &[usize], next: usize, width: usize)
    -> Result<([usize; MAX_BATCH_ROWS], usize), Int8GenerationError> {
    if available.is_empty() || available.len() > MAX_BATCH_ROWS || next >= available.len()
        || width == 0 || width > MAX_BATCH_ROWS || available.iter().all(|&n| n == 0) {
        return Err(StrictInt8Error::Input.into());
    }
    let mut counts = [0; MAX_BATCH_ROWS]; let mut cursor = next;
    for _ in 0..width {
        let slot = (0..available.len()).map(|offset| (cursor + offset) % available.len())
            .find(|&slot| counts[slot] < available[slot]);
        let Some(slot) = slot else { break; };
        counts[slot] += 1; cursor = (slot + 1) % available.len();
    }
    Ok((counts, cursor))
}

/// Reconstruct physical decoder invocations from immutable prompt lengths and
/// independently checked completed forward counts. This does not trust the
/// reported tick count or substitute logical token work for physical packs.
/// Bounds cap replay at 64 * admitted-context forwards with fixed stack arrays.
pub(crate) fn expected_group_steps(prompts: &[usize], forwards: &[u64], width: usize)
    -> Result<u64, Int8GenerationError> {
    if prompts.is_empty() || prompts.len() > MAX_BATCH_ROWS || prompts.len() != forwards.len()
        || width == 0 || width > MAX_BATCH_ROWS { return Err(Int8GenerationError::WorkMismatch); }
    let mut limits = [0_usize; MAX_BATCH_ROWS]; let mut positions = [0_usize; MAX_BATCH_ROWS];
    for (slot, (&prompt, &forward)) in prompts.iter().zip(forwards).enumerate() {
        let forward = usize::try_from(forward).map_err(|_| Int8GenerationError::WorkMismatch)?;
        if prompt == 0 || prompt > forward || forward > DEFAULT_ADMITTED_CONTEXT_CAP {
            return Err(Int8GenerationError::WorkMismatch);
        }
        limits[slot] = forward;
    }
    let mut steps = 0_u64; let mut next = 0;
    loop {
        let mut available = [0_usize; MAX_BATCH_ROWS];
        for slot in 0..prompts.len() {
            if positions[slot] < limits[slot] {
                available[slot] = if positions[slot] < prompts[slot] { prompts[slot] - positions[slot] } else { 1 };
            }
        }
        if available[..prompts.len()].iter().all(|&n| n == 0) { return Ok(steps); }
        let (counts, following) = allocate(&available[..prompts.len()], next, width)?;
        for slot in 0..prompts.len() { positions[slot] += counts[slot]; }
        steps = steps.checked_add(1).ok_or(Int8GenerationError::WorkMismatch)?;
        next = following;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn a_decode_row_gets_one_token_while_prompt_rows_fill_remaining_capacity() {
        let (counts, next) = allocate(&[1, 9, 2], 0, 8).unwrap();
        assert_eq!(&counts[..3], &[1, 5, 2]); assert_eq!(next, 2);
    }
    #[test]
    fn width_one_rotates_instead_of_starving_later_documents() {
        let mut next = 0; let mut order = Vec::new();
        for _ in 0..9 {
            let (counts, following) = allocate(&[100, 100, 100], next, 1).unwrap();
            order.push(counts.iter().position(|&n| n != 0).unwrap()); next = following;
        }
        assert_eq!(order, [0, 1, 2, 0, 1, 2, 0, 1, 2]);
    }
    #[test]
    fn finished_and_temporarily_absent_rows_never_consume_capacity() {
        let (counts, _) = allocate(&[0, 1, 0, 2], 2, 64).unwrap();
        assert_eq!(&counts[..4], &[0, 1, 0, 2]);
        assert_eq!(expected_group_steps(&[1, 7], &[1, 8], 4).unwrap(), 3);
    }
    #[test]
    fn receipt_replay_prices_prompt_packs_and_decode_dependencies_separately() {
        for width in [1, 2, 3, 4, 8, 64] {
            assert_eq!(expected_group_steps(&[7], &[10], width).unwrap(), 7_usize.div_ceil(width) as u64 + 3);
            assert_eq!(expected_group_steps(&[2, 5, 1], &[4, 7, 1], 1).unwrap(), 12);
        }
        assert_eq!(expected_group_steps(&[2, 5, 1], &[4, 7, 1], 64).unwrap(), 3);
    }
    #[test]
    fn invalid_geometry_cannot_create_an_empty_spin_or_unbounded_replay() {
        assert!(allocate(&[], 0, 1).is_err()); assert!(allocate(&[0, 0], 0, 1).is_err());
        assert!(allocate(&[1], 1, 1).is_err()); assert!(allocate(&[1], 0, 65).is_err());
        for (prompts, forwards) in [(vec![0], vec![1]), (vec![2], vec![1]),
            (vec![1], vec![u64::MAX]), (vec![1, 2], vec![3])] {
            assert!(expected_group_steps(&prompts, &forwards, 4).is_err());
        }
    }
}
