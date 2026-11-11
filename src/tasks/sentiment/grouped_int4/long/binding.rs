//! Private complete-request and native-result seals, never wire authority.
use super::*;
use sha2::{Digest, Sha256};

#[derive(Clone, Copy)]
pub(super) struct Authority {
    pub ceiling: TaskBudget,
    pub planning: SentimentLimits,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn request<C: DecodeStepControl>(request: &SentimentRequest,
    limits: Int4SentimentMapLimits, authority: Authority, encoding: &UntrustedEncoding,
    work: SafeBpeWork, chunks: &ChunkPlan<'_>, plans: &[PreparedInt4Sentiment], control: &mut C) -> Result<[u8; 32], Int4SentimentMapError> {
    checkpoint(control)?;
    if chunks.chunks().len() != plans.len() { return Err(Int4SentimentMapError::Admission); }
    // Validated finite source/plan bounds precede this streaming pass. The
    // original request and declared ceilings matter even when not exhausted.
    let mut writer = ControlledDigest::new(b"fnlp-int4-sentiment-map-request-v1", u64::MAX, control);
    writer.value(request)?;
    writer.value(&(limits.chunks, limits.max_model_work, limits.max_retained_result_bytes,
        limits.max_result_bytes, limits.aggregation))?;
    let p = authority.planning; let s = p.per_axis;
    writer.value(&(authority.ceiling, p.max_total_prompt_tokens, p.max_total_candidate_tokens,
        p.max_total_nodes, p.max_total_projected_logits, p.max_output_bytes))?;
    writer.value(&(s.max_candidates, s.max_total_tokens, s.max_nodes, s.max_depth,
        s.max_candidate_id_bytes, s.max_projected_logits))?;
    writer.value(&(encoding, work, plans.len()))?;
    for (chunk, plan) in chunks.chunks().iter().zip(plans) {
        writer.checkpoint()?;
        let identity = plan.execution_identity().receipt_identity()
            .map_err(|_| Int4SentimentMapError::Admission)?;
        writer.value(&(chunk.id(), chunk.span(), chunk.tokens(), identity))?;
    }
    writer.checkpoint()?;
    Ok(writer.finish())
}

#[cfg(test)]
pub(super) fn result(run: &Int4SentimentMapRun, cap: u64) -> Result<[u8; 32], Int4SentimentMapError> {
    result_with_control(run, cap, &mut Continue)
}
pub(super) fn result_with_control<C: DecodeStepControl>(run: &Int4SentimentMapRun, cap: u64,
    control: &mut C) -> Result<[u8; 32], Int4SentimentMapError> {
    let mut writer = ControlledDigest::new(b"fnlp-int4-sentiment-map-result-v1", cap, control);
    writer.seal(&run.request_binding);
    writer.value(run)?;
    writer.checkpoint()?;
    Ok(writer.finish())
}
pub(super) fn cohort_with_control<C: DecodeStepControl>(run: &Int4SentimentMapCohortRun, cap: u64,
    control: &mut C) -> Result<[u8; 32], Int4SentimentMapError> {
    let mut writer = ControlledDigest::new(b"fnlp-int4-sentiment-map-cohort-result-v1", cap, control);
    // Legitimate results for different requests can have identical public
    // bytes. Bind the nested owner as well as all physical cohort counters.
    writer.seal(&run.output.request_binding);
    writer.seal(&run.output.result_binding);
    // Equal cohort counts do not identify the executed windows (5 chunks at
    // widths 3 and 4 both need 2 windows). This private fixed-width field does
    // not consume the public JSON byte allowance.
    writer.hash.update((run.executed_cohort_chunks as u64).to_le_bytes());
    writer.hash.update([0]);
    writer.value(run)?;
    writer.checkpoint()?;
    Ok(writer.finish())
}

const CHECKPOINT_BYTES: usize = 64 * 1024;

/// Hash repository-owned types in deterministic serde order without creating
/// another source/output-sized buffer. Poll inside large serializer writes;
/// string escaping cannot turn one private value into an unbounded interval.
struct ControlledDigest<'a, C> {
    hash: Sha256,
    remaining: u64,
    exceeded: bool,
    control: &'a mut C,
    until_checkpoint: usize,
    cancelled: Option<DecodeCancellationKind>,
}
impl<'a, C: DecodeStepControl> ControlledDigest<'a, C> {
    fn new(domain: &[u8], cap: u64, control: &'a mut C) -> Self {
        let mut hash = Sha256::new(); hash.update(domain); hash.update([0]);
        Self { hash, remaining: cap, exceeded: false, control, until_checkpoint: 0, cancelled: None }
    }
    fn seal(&mut self, binding: &[u8; 32]) {
        // Fixed-size private framing is excluded from the public byte cap.
        self.hash.update(binding); self.hash.update([0]);
    }
    fn checkpoint(&mut self) -> Result<(), Int4SentimentMapError> {
        if self.cancelled.is_none() { self.cancelled = self.control.prefill_checkpoint(0); }
        if let Some(cause) = self.cancelled { return Err(Int4SentimentError::Cancelled(cause).into()); }
        self.until_checkpoint = CHECKPOINT_BYTES;
        Ok(())
    }
    fn value(&mut self, value: &(impl Serialize + ?Sized)) -> Result<(), Int4SentimentMapError> {
        if serde_json::to_writer(&mut *self, value).is_err() {
            if let Some(cause) = self.cancelled { return Err(Int4SentimentError::Cancelled(cause).into()); }
            return Err(if self.exceeded { Int4SentimentMapError::Output }
                else { Int4SentimentMapError::Serialization });
        }
        // Frame adjacent typed JSON values without serializing private seals.
        self.hash.update([0]);
        Ok(())
    }
    fn finish(self) -> [u8; 32] { self.hash.finalize().into() }
}
impl<C: DecodeStepControl> std::io::Write for ControlledDigest<'_, C> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let Some(remaining) = self.remaining.checked_sub(bytes.len() as u64) else {
            self.exceeded = true;
            return Err(std::io::Error::other("INT4 sentiment private binding limit"));
        };
        let mut written = 0;
        while written < bytes.len() {
            if self.until_checkpoint == 0 {
                self.checkpoint().map_err(|_| std::io::Error::other("INT4 sentiment private binding cancelled"))?;
            }
            let count = self.until_checkpoint.min(bytes.len() - written);
            self.hash.update(&bytes[written..written + count]);
            self.until_checkpoint -= count;
            written += count;
        }
        self.remaining = remaining;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
}
pub(super) struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn large_private_values_poll_inside_one_write_and_preserve_cancellation_causes() {
        struct Stop { checks: usize, cause: DecodeCancellationKind }
        impl DecodeStepControl for Stop {
            fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
                self.checks += 1;
                (self.checks == 2).then_some(self.cause)
            }
        }
        let source = "x".repeat(3 * CHECKPOINT_BYTES);
        use DecodeCancellationKind::*;
        for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
            ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
            let mut control = Stop { checks: 0, cause };
            let mut writer = ControlledDigest::new(b"cancellation-test", u64::MAX, &mut control);
            assert_eq!(writer.value(&source).unwrap_err().cancellation(), Some(cause));
            drop(writer);
            assert_eq!(control.checks, 2);
        }
        let mut control = Continue;
        let mut writer = ControlledDigest::new(b"cancellation-test", u64::MAX, &mut control);
        writer.value(&source).unwrap();
        let mut reference = Sha256::new(); reference.update(b"cancellation-test\0");
        reference.update(serde_json::to_vec(&source).unwrap()); reference.update([0]);
        let expected: [u8; 32] = reference.finalize().into();
        assert_eq!(writer.finish(), expected);
    }
}
