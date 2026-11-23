//! Private, bounded request/result commitments for complete-document QA.
//! These hashes never enter serialized output or diagnostic formatting.
use super::*;
use sha2::{Digest, Sha256};

#[allow(clippy::too_many_arguments)]
pub(super) fn request<C: DecodeStepControl>(source: &str, question: &SourceQuestion,
    budget: TaskBudget, context: &PlanContext<'_>, planning: SourcePlanningLimits,
    mapping: Int8SourceMapLimits, planner: &SourceTaskPlanner, chunks: &ChunkPlan<'_>,
    control: &mut C) -> Result<[u8; 32], Int8SourceMapError> {
    let mut check = || checkpoint(control).map_err(Int8SourceMapError::from);
    let mut writer = PrivateBinding::new(b"fnlp-int8-question-request-v1", u64::MAX, &mut check)?;
    let compiler = planning.compiler;
    let identity = context.execution_identity().receipt_identity().map_err(|_| Int8SourceMapError::Admission)?;
    writer.value(&(identity, source, &question.question, question.options,
        question.verification, budget, context.budget_ceiling()))?;
    writer.value(&(mapping.chunks, mapping.reduction, mapping.max_model_work,
        (mapping.mask_limits.max_trie_node_visits, mapping.mask_limits.checkpoint_interval_nodes),
        mapping.mask_visits_per_chunk, mapping.max_mask_visits))?;
    writer.value(&((planning.max_input_bytes, planning.max_context_tokens, planning.max_passages),
        (compiler.max_schema_bytes, compiler.max_string_bytes, compiler.max_array_items, compiler.max_output_bytes,
            compiler.max_states, compiler.max_transitions, compiler.max_mask_bytes),
        planning.source, planner.document_encoding(), planner.map_document_encoding_limits()))?;
    writer.value(&chunks.chunks().len())?;
    for chunk in chunks.chunks() {
        writer.poll()?;
        writer.value(&(chunk.id(), chunk.span(), chunk.tokens(), has_text(chunk.text())))?;
    }
    writer.finish()
}

pub(super) fn result_with_checkpoints<F>(run: &Int8QuestionRun, cap: u64, check: &mut F)
    -> Result<[u8; 32], Int8SourceMapError>
    where F: FnMut() -> Result<(), Int8SourceMapError> {
    let mut writer = PrivateBinding::new(b"fnlp-int8-question-result-v1", cap, check)?;
    writer.hash.update(run.request_binding);
    writer.value(run)?;
    writer.finish()
}

const CHECKPOINT_BYTES: usize = 64 * 1024;
struct PrivateBinding<'a, F> {
    hash: Sha256,
    remaining: u64,
    until_checkpoint: usize,
    check: &'a mut F,
    failure: Option<Int8SourceMapError>,
}
impl<'a, F: FnMut() -> Result<(), Int8SourceMapError>> PrivateBinding<'a, F> {
    fn new(domain: &[u8], cap: u64, check: &'a mut F) -> Result<Self, Int8SourceMapError> {
        check()?;
        let mut hash = Sha256::new(); hash.update(domain); hash.update([0]);
        Ok(Self { hash, remaining: cap, until_checkpoint: CHECKPOINT_BYTES, check, failure: None })
    }
    fn poll(&mut self) -> Result<(), Int8SourceMapError> { (self.check)() }
    fn value(&mut self, value: &(impl Serialize + ?Sized)) -> Result<(), Int8SourceMapError> {
        let encoded = serde_json::to_writer(&mut *self, value);
        if let Some(error) = self.failure.take() { return Err(error); }
        encoded.map_err(|_| Int8SourceMapError::Admission)?;
        self.hash.update([0]);
        Ok(())
    }
    fn finish(mut self) -> Result<[u8; 32], Int8SourceMapError> {
        self.poll()?;
        Ok(self.hash.finalize().into())
    }
}
impl<F: FnMut() -> Result<(), Int8SourceMapError>> std::io::Write for PrivateBinding<'_, F> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let Some(remaining) = self.remaining.checked_sub(bytes.len() as u64) else {
            self.failure = Some(Int8SourceError::from(SourcePlanningError::OutputBudget).into());
            return Err(std::io::Error::other("INT8 question private binding limit"));
        };
        self.remaining = remaining;
        let mut offset = 0;
        while offset < bytes.len() {
            if self.until_checkpoint == 0 {
                if let Err(error) = self.poll() {
                    self.failure = Some(error);
                    return Err(std::io::Error::other("INT8 question private binding stopped"));
                }
                self.until_checkpoint = CHECKPOINT_BYTES;
            }
            let count = self.until_checkpoint.min(bytes.len() - offset);
            self.hash.update(&bytes[offset..offset + count]);
            offset += count;
            self.until_checkpoint -= count;
        }
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn large_serialization_preserves_every_terminal_cancellation_cause() {
        use DecodeCancellationKind::*;
        let source = "x".repeat(3 * CHECKPOINT_BYTES);
        for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
            ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
            let mut polls = 0;
            let mut check = || {
                polls += 1;
                if polls == 2 { Err(Int8SourceError::Cancelled(cause).into()) } else { Ok(()) }
            };
            let error = {
                let mut writer = PrivateBinding::new(b"test", u64::MAX, &mut check).unwrap();
                writer.value(&source).unwrap_err()
            };
            assert_eq!(error.cancellation(), Some(cause));
            assert_eq!(polls, 2);
        }
    }
}
