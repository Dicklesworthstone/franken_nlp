//! Shared cursor transitions after one of the two explicitly admitted heads.
//! Selected languages are never unioned, pruned, or silently widened to full.
use super::*;
use crate::native_engine::constrained_int4::sparse::selection;

pub(super) fn project_ready<V: Vocabulary, D: GroupDriver>(cursors: &mut [Cursor<'_>],
    requests: &[Int4JsonCohortRequest<'_>], vocabulary: &V, projection: Projection<'_>,
    ready: &[usize], driver: &mut D, groups: &mut u64) -> Result<(), Int4JsonError> {
    match projection {
        Projection::Full => {
            // Preserve the original shared full-head invocation, including
            // finiteness checks on every vocabulary logit, legal or illegal.
            let logits = driver.logits_group(ready)?;
            let expected = vocabulary.width().checked_mul(ready.len()).ok_or(StrictInt4Error::Memory)?;
            checked_logits(&logits, expected)?;
            *groups = groups.checked_add(1).ok_or(StrictInt4Error::Work)?;
            for (&slot, logits) in ready.iter().zip(logits.chunks_exact(vocabulary.width())) {
                let request = &requests[slot]; let row = &mut cursors[slot];
                poll(driver.control(), row.tokens.len())?;
                row.charge_projection(vocabulary.width(), request)?;
                let mask = row.mask(request, vocabulary, driver)?;
                let token = select(logits, &mask, row.grammar.is_accepting(), request.options)?;
                drop(mask);
                row.accept(token, request, vocabulary, driver)?;
            }
        }
        Projection::Selected(limits) => {
            for &slot in ready {
                let request = &requests[slot]; let row = &mut cursors[slot]; let step = row.tokens.len();
                poll(driver.control(), step)?;
                let mask = row.mask(request, vocabulary, driver)?;
                let remaining = request.budget.json.max_projected_logits.checked_sub(row.projected as u64)
                    .ok_or(Int4JsonError::WorkMismatch)?;
                let cap = limits.get(slot).ok_or(Int4JsonError::WorkMismatch)?.max_rows_per_step;
                // The common selector first admits this ENTIRE legal set,
                // then projects every ID once in stable, bounded chunks.
                let (token, count) = selection::project(&mask, row.grammar.is_accepting(), request.options,
                    cap, remaining, &mut RowHead { driver, slot, groups }, step)?;
                drop(mask);
                row.charge_projection(count, request)?;
                row.accept(token, request, vocabulary, driver)?;
            }
        }
    }
    Ok(())
}

struct RowHead<'a, D> { driver: &'a mut D, slot: usize, groups: &'a mut u64 }
impl<D: GroupDriver> selection::Head for RowHead<'_, D> {
    type Control = D::Control;
    fn control(&mut self) -> &mut Self::Control { self.driver.control() }
    fn logits(&mut self, rows: &[u32]) -> Result<Vec<f32>, Int4JsonError> {
        let logits = self.driver.logits_selected(self.slot, rows)?;
        *self.groups = self.groups.checked_add(1).ok_or(StrictInt4Error::Work)?;
        Ok(logits)
    }
}
