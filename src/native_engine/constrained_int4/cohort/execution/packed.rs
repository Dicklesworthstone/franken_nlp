//! Finite prompt packs share the native layer-major path with ready feedback.
//! Selection never speculates a generated token or pads a sequence's prefix.
use super::*;

pub(super) fn append<D: GroupDriver>(cursors: &mut [Cursor<'_>], requests: &[Int4JsonCohortRequest<'_>],
    limits: Int4PrefillLimits, next_slot: &mut usize, ready: &mut Vec<usize>, driver: &mut D)
    -> Result<(), Int4JsonError> {
    let count = cursors.len();
    let mut lengths = [0_usize; MAX_INT4_SEQUENCES];
    let mut remaining = limits.max_batch_rows;
    let start = *next_slot;
    // Decode takes priority, but each selected row contributes only its last
    // sampled token. Rotation also supports packs narrower than the cohort.
    for offset in 0..count {
        let slot = (start + offset) % count; let row = &cursors[slot];
        if remaining != 0 && !row.done && row.positions >= requests[slot].prompt.len() {
            lengths[slot] = 1; remaining -= 1; *next_slot = (slot + 1) % count;
        }
    }
    // Allocate remaining physical rows fairly over known prompt positions,
    // then coalesce each document's selected prefix into one borrowed run.
    while remaining != 0 {
        let before = remaining;
        for offset in 0..count {
            let slot = (start + offset) % count; let row = &cursors[slot];
            if remaining != 0 && !row.done
                && row.positions + lengths[slot] < requests[slot].prompt.len() {
                lengths[slot] += 1; remaining -= 1; *next_slot = (slot + 1) % count;
            }
        }
        if remaining == before { break; }
    }
    // Fixed bookkeeping fits the existing cursor reserve. The much larger
    // native packed activations are admitted separately by limits.validate().
    let mut feedback = [0_u32; MAX_INT4_SEQUENCES];
    for (slot, row) in cursors.iter().enumerate() {
        if lengths[slot] != 0 && row.positions >= requests[slot].prompt.len() {
            feedback[slot] = row.next(requests[slot].prompt)?;
        }
    }
    let mut runs = [Int4TokenRun { sequence: 0, tokens: &[] }; MAX_INT4_SEQUENCES];
    let mut run_count = 0;
    for (slot, row) in cursors.iter().enumerate() {
        let length = lengths[slot];
        if length == 0 { continue; }
        let tokens = if row.positions < requests[slot].prompt.len() {
            let end = row.positions.checked_add(length).ok_or(StrictInt4Error::Work)?;
            for position in row.positions..end {
                if let Some(cause) = driver.control().prefill_checkpoint(position) {
                    return Err(JsonDecodeError::Cancelled(cause).into());
                }
            }
            requests[slot].prompt.get(row.positions..end).ok_or(JsonDecodeError::IllegalTransition)?
        } else {
            poll(driver.control(), row.tokens.len())?;
            std::slice::from_ref(&feedback[slot])
        };
        runs[run_count] = Int4TokenRun { sequence: slot, tokens }; run_count += 1;
    }
    if run_count == 0 { return Err(JsonDecodeError::IllegalTransition.into()); }
    driver.append_packed(&runs[..run_count], limits)?;
    for (slot, row) in cursors.iter_mut().enumerate() {
        if lengths[slot] == 0 { continue; }
        row.positions = row.positions.checked_add(lengths[slot]).ok_or(StrictInt4Error::Work)?;
        // A ready but unselected row must never project its old hidden again.
        if row.positions >= requests[slot].prompt.len() { ready.push(slot); }
    }
    Ok(())
}
