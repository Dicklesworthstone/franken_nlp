//! Replace only prompt appending. The existing driver owns masks, full-head
//! projection, scored EOS, exact work reconciliation and semantic finalization.
use super::*;

pub(super) trait PrefillDriver: Driver {
    fn append_layer_major(&mut self, prompt: &[u32], limits: Int4PrefillLimits) -> Result<(), Int4JsonError>;
}
impl<C: DecodeStepControl> PrefillDriver for Int4Session<'_, '_, C> {
    fn append_layer_major(&mut self, prompt: &[u32], limits: Int4PrefillLimits) -> Result<(), Int4JsonError> {
        Ok(Int4Session::append_layer_major(self, prompt, limits)?)
    }
}

struct LayerMajor<'a, D> { inner: &'a mut D, limits: Int4PrefillLimits }
impl<D: PrefillDriver> Driver for LayerMajor<'_, D> {
    type Control = D::Control;
    fn control(&mut self) -> &mut Self::Control { self.inner.control() }
    fn append(&mut self, token: u32) -> Result<(), Int4JsonError> { self.inner.append(token) }
    fn append_prompt(&mut self, prompt: &[u32]) -> Result<(), Int4JsonError> {
        self.inner.append_layer_major(prompt, self.limits)
    }
    fn logits(&mut self) -> Result<Vec<f32>, Int4JsonError> { self.inner.logits() }
    fn work(&self) -> Int4Work { self.inner.work() }
    fn abort(&mut self) { self.inner.abort(); }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn drive_layer_major<V, D, T, E, F>(group_width: usize, prompt: &[u32], program: &JsonProgram,
    vocabulary: &V, options: &JsonDecodeOptions, budget: Int4JsonBudget, driver: &mut D,
    limits: Int4PrefillLimits, finalize: F) -> Result<T, E>
where V: Vocabulary, D: PrefillDriver, E: From<Int4JsonError>, F: FnOnce(Int4JsonRun) -> Result<T, E> {
    // The public entrypoint validates before opening a session too. Keep this
    // private composition boundary unable to enter a driver on invalid limits.
    limits.validate().map_err(Int4JsonError::from).map_err(E::from)?;
    drive(group_width, prompt, program, vocabulary, options, budget,
        &mut LayerMajor { inner: driver, limits }, finalize)
}
