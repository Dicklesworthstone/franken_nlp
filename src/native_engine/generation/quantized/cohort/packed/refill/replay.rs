//! Replay physical slot reuse from independently checked request forward counts.
use super::*;
pub(crate) fn expected_steps(prompts: &[usize], forwards: &[u64], slots: usize, width: usize)
    -> Result<u64, Int8GenerationError> {
    if prompts.is_empty() || prompts.len() > MAX_BATCH_ROWS || prompts.len() != forwards.len()
        || slots == 0 || slots > prompts.len() || width == 0 || width > MAX_BATCH_ROWS {
        return Err(Int8GenerationError::WorkMismatch);
    }
    let mut ends = [0_usize; MAX_BATCH_ROWS]; let mut positions = [0_usize; MAX_BATCH_ROWS];
    for (request, (&prompt, &forward)) in prompts.iter().zip(forwards).enumerate() {
        let end = usize::try_from(forward).map_err(|_| Int8GenerationError::WorkMismatch)?;
        if prompt == 0 || prompt > end || end > DEFAULT_ADMITTED_CONTEXT_CAP { return Err(Int8GenerationError::WorkMismatch); }
        ends[request] = end;
    }
    let mut live = [None; MAX_BATCH_ROWS]; let mut queued = 0; let mut next = 0; let mut steps = 0_u64;
    loop {
        for slot in &mut live[..slots] {
            if slot.is_none() && queued < prompts.len() { *slot = Some(queued); queued += 1; }
        }
        if live[..slots].iter().all(Option::is_none) {
            if queued != prompts.len() { return Err(Int8GenerationError::WorkMismatch); }
            return Ok(steps);
        }
        let mut available = [0_usize; MAX_BATCH_ROWS];
        for (slot, request) in live[..slots].iter().enumerate() {
            if let Some(request) = request {
                available[slot] = if positions[*request] < prompts[*request] { prompts[*request] - positions[*request] } else { 1 };
            }
        }
        let (counts, following) = schedule::allocate(&available[..slots], next, width)?;
        for (slot, request) in live[..slots].iter_mut().enumerate() {
            if let Some(index) = *request {
                positions[index] += counts[slot];
                if positions[index] > ends[index] { return Err(Int8GenerationError::WorkMismatch); }
                if positions[index] == ends[index] { *request = None; }
            }
        }
        steps = steps.checked_add(1).ok_or(Int8GenerationError::WorkMismatch)?; next = following;
    }
}
