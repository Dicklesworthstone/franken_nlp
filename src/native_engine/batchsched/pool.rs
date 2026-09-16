//! Capacity-frozen ragged sequence state and failure-atomic group retirement.
use super::*;
use std::sync::atomic::{AtomicU64, Ordering};
static NEXT_DOMAIN: AtomicU64 = AtomicU64::new(1);

pub(super) struct Row { pub cache: KvCache, generation: u64, active: bool }
pub(super) struct SequencePool { domain: u64, pub rows: Vec<Row> }
impl SequencePool {
    pub fn new(capacities: &[usize]) -> Result<Self, BatchError> {
        BatchEnvelope::estimate(capacities)?;
        let domain = NEXT_DOMAIN.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| BatchError::Limit("sequence domains exhausted"))?;
        let mut rows = reserve(capacities.len())?;
        for &cap in capacities {
            rows.push(Row { cache: KvCache::try_with_capacity(cap).map_err(HfBf16EagerError::from)?, generation: 0, active: false });
        }
        Ok(Self { domain, rows })
    }
    pub fn preflight_slot(&self, slot: usize, required_positions: usize) -> Result<(), BatchError> {
        let row = self.rows.get(slot).ok_or(BatchError::StaleSequence)?;
        if row.active { return Err(BatchError::SequenceBusy); }
        if required_positions == 0 || required_positions > row.cache.capacity_positions() { return Err(BatchError::ContextFull); }
        if !row.cache.all_slots_have_len(0) { return Err(BatchError::Contract("idle cache not empty")); }
        Ok(())
    }
    pub fn validate(&self, sequence: BatchSequence) -> Result<usize, BatchError> {
        let row = self.rows.get(sequence.slot).ok_or(BatchError::StaleSequence)?;
        if sequence.domain != self.domain || !row.active || row.generation != sequence.generation {
            return Err(BatchError::StaleSequence);
        }
        Ok(sequence.slot)
    }
    pub fn open(&mut self, slot: usize) -> Result<BatchSequence, BatchError> {
        let row = self.rows.get_mut(slot).ok_or(BatchError::StaleSequence)?;
        if row.active { return Err(BatchError::SequenceBusy); }
        let generation = row.generation.checked_add(1).ok_or(BatchError::Limit("sequence generations exhausted"))?;
        if !row.cache.all_slots_have_len(0) { return Err(BatchError::Contract("idle cache not empty")); }
        row.generation = generation; row.active = true;
        Ok(BatchSequence { domain: self.domain, slot, generation })
    }
    pub fn close(&mut self, sequence: BatchSequence) -> Result<(), BatchError> {
        let slot = self.validate(sequence)?; self.retire(slot); Ok(())
    }
    pub fn retire(&mut self, slot: usize) { self.rows[slot].cache.clear(); self.rows[slot].active = false; }
    pub fn len(&self, sequence: BatchSequence) -> Result<usize, BatchError> {
        let slot = self.validate(sequence)?;
        let cache = &self.rows[slot].cache;
        let length = cache.len_for_slot(0).map_err(HfBf16EagerError::from)?;
        if !cache.all_slots_have_len(length) { return Err(BatchError::Contract("divergent logical cache slots")); }
        Ok(length)
    }
    pub fn preflight(&self, input: &[BatchToken]) -> Result<(Vec<usize>, Vec<usize>), BatchError> {
        if input.is_empty() || input.len() > self.rows.len() { return Err(BatchError::Contract("step width")); }
        let mut seen = [false; MAX_BATCH_ROWS];
        let mut slots = reserve(input.len())?; let mut positions = reserve(input.len())?;
        for token in input {
            let slot = self.validate(token.sequence)?;
            if seen[slot] { return Err(BatchError::DuplicateSequence); } seen[slot] = true;
            if token.token_id as usize >= NANBEIGE_VOCAB_SIZE { return Err(BatchError::Contract("token outside vocabulary")); }
            let position = self.len(token.sequence)?;
            if position >= self.rows[slot].cache.capacity_positions() { return Err(BatchError::ContextFull); }
            slots.push(slot); positions.push(position);
        }
        Ok((slots, positions))
    }
}

/// Armed before the first fallible model operation. This is deliberate discard,
/// not a false claim that failed attention can roll back to reusable prefixes.
pub(super) struct StepTransaction<'a, 'b> {
    pub pool: &'a mut SequencePool,
    slots: &'b [usize],
    committed: bool,
}
impl<'a, 'b> StepTransaction<'a, 'b> {
    pub fn new(pool: &'a mut SequencePool, slots: &'b [usize]) -> Self { Self { pool, slots, committed: false } }
    pub fn commit(&mut self, old_positions: &[usize]) -> Result<(), BatchError> {
        if old_positions.len() != self.slots.len() { return Err(BatchError::Contract("commit row count")); }
        for (&slot, &position) in self.slots.iter().zip(old_positions) {
            let next = position.checked_add(1).ok_or(BatchError::Limit("position overflow"))?;
            if !self.pool.rows[slot].cache.all_slots_have_len(next) { return Err(BatchError::Contract("incomplete layer group")); }
        }
        self.committed = true; Ok(())
    }
}
impl Drop for StepTransaction<'_, '_> {
    fn drop(&mut self) { if !self.committed { for &slot in self.slots { self.pool.retire(slot); } } }
}
