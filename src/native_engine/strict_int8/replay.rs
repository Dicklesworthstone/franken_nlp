//! Private same-session prompt replay for independently addressed samples.
//!
//! Completed KV positions remain owned by the one native session. Rewinding
//! never creates a prefix snapshot, hidden-state history or renewable budget.

use super::*;

impl<C: DecodeStepControl> Int8Session<'_, '_, C> {
    /// Retain a coherent prefix and invalidate the descendant hidden state.
    /// A new forward must complete before any further head projection. Work,
    /// projection reservations and both remaining ceilings stay unchanged.
    pub(crate) fn rewind_for_replay(&mut self, positions: usize) -> Result<(), StrictInt8Error> {
        rewind(&mut self.engine.cache, &mut self.engine.state, &mut self.hidden, positions, self.control)
    }
}

fn rewind<C: DecodeStepControl>(cache: &mut KvCache, state: &mut RunState,
    hidden: &mut Option<Vec<Bf16>>, positions: usize, control: &mut C) -> Result<(), StrictInt8Error> {
    state.check()?;
    let completed = cache.len_for_slot(0).map_err(|_| StrictInt8Error::Cache)?;
    if positions > completed || !cache.all_slots_have_len(completed) {
        return Err(StrictInt8Error::Cache);
    }
    // Once replay begins, cancellation before or after truncation is fatal to
    // this session. Its existing Drop guard clears all 44 logical slots.
    state.poisoned = true;
    poll(control)?;
    cache.rewind_completed(positions).map_err(|_| StrictInt8Error::Cache)?;
    *hidden = None;
    poll(control)?;
    state.poisoned = false;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Continue;
    impl DecodeStepControl for Continue {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
    }

    #[test]
    fn replay_retains_every_prefix_and_invalidates_the_descendant_hidden() {
        let mut cache = KvCache::try_with_capacity(4).unwrap();
        for position in 0..3 {
            for slot in 0..KV_SLOT_COUNT {
                cache.append(slot, position, &[(position * 100 + slot) as u16; K],
                    &[(position * 100 + slot + 1) as u16; K]).unwrap();
            }
        }
        let pointers: Vec<_> = (0..KV_SLOT_COUNT).map(|slot|
            (cache.key_at(slot, 0).unwrap().as_ptr(), cache.value_at(slot, 0).unwrap().as_ptr())).collect();
        let mut state = RunState::default(); state.open().unwrap();
        let mut hidden = Some(vec![Bf16::from_f32(3.0); H]);
        rewind(&mut cache, &mut state, &mut hidden, 1, &mut Continue).unwrap();
        assert!(hidden.is_none()); assert!(state.check().is_ok());
        assert!(cache.all_slots_have_len(1)); assert_eq!(cache.capacity_positions(), 4);
        for slot in 0..KV_SLOT_COUNT {
            assert_eq!(cache.key_at(slot, 0).unwrap(), &[slot as u16; K]);
            assert_eq!(cache.value_at(slot, 0).unwrap(), &[(slot + 1) as u16; K]);
            assert!(cache.key_at(slot, 1).is_err()); assert!(cache.value_at(slot, 2).is_err());
            assert_eq!(pointers[slot],
                (cache.key_at(slot, 0).unwrap().as_ptr(), cache.value_at(slot, 0).unwrap().as_ptr()));
            cache.append(slot, 1, &[900; K], &[901; K]).unwrap();
            assert_eq!(cache.key_at(slot, 1).unwrap(), &[900; K]);
        }
        rewind(&mut cache, &mut state, &mut hidden, 0, &mut Continue).unwrap();
        assert!(cache.all_slots_have_len(0)); assert!(hidden.is_none());
        end_session(&mut cache, &mut state);
        assert!(!state.active); assert!(!state.poisoned);
    }

    #[test]
    fn future_and_incomplete_replays_refuse_without_mutation() {
        let mut cache = KvCache::try_with_capacity(3).unwrap();
        for slot in 0..KV_SLOT_COUNT { cache.append(slot, 0, &[1; K], &[2; K]).unwrap(); }
        let mut state = RunState::default(); state.open().unwrap();
        let mut hidden = Some(vec![Bf16::from_f32(1.0); H]);
        let before = cache.clone(); let before_hidden = hidden.clone();
        assert_eq!(rewind(&mut cache, &mut state, &mut hidden, 2, &mut Continue), Err(StrictInt8Error::Cache));
        assert_eq!(cache, before); assert_eq!(hidden, before_hidden); assert!(state.check().is_ok());
        cache.append(0, 1, &[3; K], &[4; K]).unwrap();
        let before = cache.clone();
        assert_eq!(rewind(&mut cache, &mut state, &mut hidden, 0, &mut Continue), Err(StrictInt8Error::Cache));
        assert_eq!(cache, before); assert_eq!(hidden, before_hidden); assert!(state.check().is_ok());
    }

    #[test]
    fn cancellation_on_either_side_of_rewind_poison_cleans_without_changing_its_cause() {
        use DecodeCancellationKind::*;
        struct Cancel { at: usize, calls: usize, cause: DecodeCancellationKind }
        impl DecodeStepControl for Cancel {
            fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
                self.calls += 1; (self.calls == self.at).then_some(self.cause)
            }
        }
        for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
            ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
            for at in 1..=2 {
                let mut cache = KvCache::try_with_capacity(2).unwrap();
                for position in 0..2 {
                    for slot in 0..KV_SLOT_COUNT { cache.append(slot, position, &[1; K], &[2; K]).unwrap(); }
                }
                let mut state = RunState::default(); state.open().unwrap();
                let mut hidden = Some(vec![Bf16::from_f32(1.0); H]);
                let mut control = Cancel { at, calls: 0, cause };
                assert_eq!(rewind(&mut cache, &mut state, &mut hidden, 1, &mut control),
                    Err(StrictInt8Error::Cancelled(cause)));
                assert_eq!(control.calls, at); assert!(state.poisoned); assert!(state.check().is_err());
                assert!(cache.all_slots_have_len(if at == 1 { 2 } else { 1 }));
                assert_eq!(hidden.is_none(), at == 2);
                end_session(&mut cache, &mut state);
                assert!(cache.all_slots_have_len(0)); assert!(state.poisoned); assert!(state.open().is_err());
            }
        }
    }
}
