//! Bounded head chunks avoid a second vocabulary-sized row-ID allocation.
//! The full legal count is checked before the FIRST native projection. Fixed
//! 32-ID stack storage plus one <=32-logit output fits the ordinary engine's
//! full-vocabulary head/scratch envelope; no new uncharged heap rail is added.
use super::*;
const ROWS: usize = 32;

#[allow(clippy::too_many_arguments)]
pub(super) fn project<D: Driver>(mask: &DenseTokenMask, accepting: bool, options: &JsonDecodeOptions,
    cap: usize, remaining: u64, driver: &mut D, step: usize) -> Result<(u32, usize), Int8JsonError> {
    let legal = |id: u32| if id == options.eos_token_id { accepting }
        else { mask.contains(id) && !options.excluded_token_ids.contains(&id) };
    let mut count = 0_usize;
    for index in 0..mask.vocab_size() {
        if index % 1024 == 0 { poll(driver.control(), step)?; }
        if legal(index as u32) {
            count += 1;
            if count > cap || count as u64 > remaining {
                return Err(JsonDecodeError::BudgetExceeded("complete legal row set").into());
            }
        }
    }
    if count == 0 { return Err(JsonDecodeError::NoLegalToken.into()); }
    let mut rows = [0_u32; ROWS];
    let mut used = 0; let mut projected = 0; let mut best = None;
    for index in 0..mask.vocab_size() {
        if index % 1024 == 0 { poll(driver.control(), step)?; }
        if !legal(index as u32) { continue; }
        rows[used] = index as u32; used += 1;
        if used == ROWS {
            block(&rows, driver, step, &mut best)?;
            projected += used; used = 0;
        }
    }
    if used != 0 { block(&rows[..used], driver, step, &mut best)?; projected += used; }
    if projected != count { return Err(Int8JsonError::WorkMismatch); }
    poll(driver.control(), step)?;
    best.map(|(id, _)| (id, count)).ok_or_else(|| JsonDecodeError::NoLegalToken.into())
}
fn block<D: Driver>(rows: &[u32], driver: &mut D, step: usize, best: &mut Option<(u32, f32)>)
    -> Result<(), Int8JsonError> {
    poll(driver.control(), step)?;
    let logits = driver.logits(rows)?;
    let id = select(rows, &logits)?;
    let index = rows.binary_search(&id).map_err(|_| JsonDecodeError::InvalidLogits)?;
    let value = logits[index];
    // Blocks arrive in ascending token-ID order, so equal scores retain the
    // first ID across block boundaries just as within one block.
    if best.is_none_or(|(_, score)| value > score) { *best = Some((id, value)); }
    poll(driver.control(), step)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)] struct Control { calls: usize, cancel: Option<usize> }
    impl DecodeStepControl for Control {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.calls += 1;
            (self.cancel == Some(self.calls)).then_some(DecodeCancellationKind::Deadline)
        }
        fn prefill_checkpoint(&mut self, position: usize) -> Option<DecodeCancellationKind> { self.checkpoint(position) }
    }
    #[derive(Default)] struct Script { rows: Vec<Vec<u32>>, control: Control, tie: bool, bad_tail: bool }
    impl Driver for Script {
        type Control = Control;
        fn control(&mut self) -> &mut Control { &mut self.control }
        fn append(&mut self, _: u32) -> Result<(), Int8JsonError> { unreachable!() }
        fn logits(&mut self, rows: &[u32]) -> Result<Vec<f32>, Int8JsonError> {
            assert!(rows.len() <= ROWS); self.rows.push(rows.to_vec());
            Ok(rows.iter().map(|&id| if self.bad_tail && id == 99 { f32::NAN }
                else if self.tie { 0.0 } else { id as f32 }).collect())
        }
        fn work(&self) -> Int8Work { unreachable!() }
        fn abort(&mut self) { unreachable!() }
    }
    fn fixture(width: usize) -> (DenseTokenMask, JsonDecodeOptions) {
        let mut mask = DenseTokenMask::empty(width);
        for id in 1..width { mask.set_legal(id as u32).unwrap(); }
        (mask, JsonDecodeOptions { max_new_tokens: 2, eos_token_id: 0, excluded_token_ids: Default::default() })
    }
    #[test]
    fn large_legal_sets_project_every_row_once_with_bounded_tail_and_global_ties() {
        let (mask, options) = fixture(100);
        for tie in [false, true] {
            let mut d = Script { tie, ..Script::default() };
            assert_eq!(project(&mask, false, &options, 99, 99, &mut d, 0).unwrap(),
                (if tie { 1 } else { 99 }, 99));
            assert_eq!(d.rows.iter().map(Vec::len).collect::<Vec<_>>(), [32, 32, 32, 3]);
            assert_eq!(d.rows.into_iter().flatten().collect::<Vec<_>>(), (1..100).collect::<Vec<u32>>());
        }
    }
    #[test]
    fn full_set_admission_and_late_nonfinite_failures_never_return_partial_argmax() {
        let (mask, options) = fixture(100);
        for (cap, remaining) in [(98, 99), (99, 98)] {
            let mut d = Script::default();
            assert!(project(&mask, false, &options, cap, remaining, &mut d, 0).is_err());
            assert!(d.rows.is_empty());
        }
        let mut d = Script { bad_tail: true, ..Script::default() };
        assert!(project(&mask, false, &options, 99, 99, &mut d, 0).is_err());
        assert_eq!(d.rows.len(), 4);
    }
    #[test]
    fn large_set_counting_can_cancel_before_the_first_projection() {
        let (mask, options) = fixture(4096);
        let mut d = Script::default(); d.control.cancel = Some(2);
        assert!(project(&mask, false, &options, 4096, 4096, &mut d, 0).unwrap_err().cancellation().is_some());
        assert!(d.rows.is_empty());
    }
}
