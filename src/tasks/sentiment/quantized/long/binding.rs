//! Streaming private request/completion seals and bounded result sizing.
use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn request<C: DecodeStepControl>(request: &SentimentRequest,
    context: &PlanContext<'_>, planning: SentimentLimits, limits: Int8SentimentMapLimits,
    encoding: &UntrustedEncoding, work: SafeBpeWork, chunks: &ChunkPlan<'_>,
    plans: &[PreparedInt8Sentiment], control: &mut C) -> Result<[u8; 32], Int8SentimentMapError> {
    checkpoint(control)?;
    if chunks.chunks().len() != plans.len() { return Err(Int8SentimentMapError::Admission); }
    let mut writer = ControlledDigest::new(b"fnlp-int8-sentiment-map-request-v1", u64::MAX, control);
    writer.value(request)?;
    writer.value(&(context.execution_identity(), context.budget_ceiling()))?;
    writer.value(&(limits.chunks, limits.max_model_work, limits.max_retained_result_bytes,
        limits.max_result_bytes, limits.aggregation))?;
    let p = planning; let s = p.per_axis;
    writer.value(&(p.max_total_prompt_tokens, p.max_total_candidate_tokens, p.max_total_nodes,
        p.max_total_projected_logits, p.max_output_bytes))?;
    writer.value(&(s.max_candidates, s.max_total_tokens, s.max_nodes, s.max_depth,
        s.max_candidate_id_bytes, s.max_projected_logits))?;
    writer.value(&(encoding, work, plans.len()))?;
    for (chunk, plan) in chunks.chunks().iter().zip(plans) {
        writer.checkpoint()?;
        let identity = plan.execution_identity().receipt_identity().map_err(|_| Int8SentimentMapError::Admission)?;
        writer.value(&(chunk.id(), chunk.span(), chunk.tokens(), identity))?;
    }
    writer.checkpoint()?;
    Ok(writer.finish())
}

pub(super) fn result_with_control<C: DecodeStepControl>(run: &Int8SentimentMapRun, cap: u64,
    control: &mut C) -> Result<[u8; 32], Int8SentimentMapError> {
    let mut writer = ControlledDigest::new(b"fnlp-int8-sentiment-map-result-v1", cap, control);
    writer.hash.update(run.request_binding);
    writer.hash.update([0]);
    writer.value(run)?;
    writer.checkpoint()?;
    Ok(writer.finish())
}

/// Refuse over-cap serializations before an output-sized allocation, then
/// price the actual canonical bytes. Preserve the same caller control before
/// and after the bounded canonicalization, including its final deadline.
pub(super) fn value_bytes_with_control<C: DecodeStepControl>(value: &impl Serialize,
    cap: u64, control: &mut C) -> Result<u64, Int8SentimentMapError> {
    let mut writer = ControlledDigest::new(b"fnlp-int8-sentiment-map-size-v1", cap, control);
    writer.value(value)?;
    writer.checkpoint()?;
    drop(writer);
    let bytes = canonjson::canonical_bytes(value).map_err(|_| Int8SentimentMapError::Serialization)?;
    if bytes.len() as u64 > cap { return Err(Int8SentimentMapError::Output); }
    checkpoint(control)?;
    Ok(bytes.len() as u64)
}

const CHECKPOINT_BYTES: usize = 64 * 1024;
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
    fn checkpoint(&mut self) -> Result<(), Int8SentimentMapError> {
        if self.cancelled.is_none() { self.cancelled = self.control.prefill_checkpoint(0); }
        if let Some(cause) = self.cancelled { return Err(Int8SentimentError::Cancelled(cause).into()); }
        self.until_checkpoint = CHECKPOINT_BYTES;
        Ok(())
    }
    fn value(&mut self, value: &(impl Serialize + ?Sized)) -> Result<(), Int8SentimentMapError> {
        if serde_json::to_writer(&mut *self, value).is_err() {
            if let Some(cause) = self.cancelled { return Err(Int8SentimentError::Cancelled(cause).into()); }
            return Err(if self.exceeded { Int8SentimentMapError::Output } else { Int8SentimentMapError::Serialization });
        }
        self.hash.update([0]);
        Ok(())
    }
    fn finish(self) -> [u8; 32] { self.hash.finalize().into() }
}
impl<C: DecodeStepControl> std::io::Write for ControlledDigest<'_, C> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let Some(remaining) = self.remaining.checked_sub(bytes.len() as u64) else {
            self.exceeded = true;
            return Err(std::io::Error::other("INT8 sentiment private binding limit"));
        };
        let mut written = 0;
        while written < bytes.len() {
            if self.until_checkpoint == 0 {
                self.checkpoint().map_err(|_| std::io::Error::other("INT8 sentiment private binding cancelled"))?;
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
    fn large_private_serializations_poll_inside_writes_and_keep_all_cancel_kinds() {
        struct Stop { checks: usize, cause: DecodeCancellationKind }
        impl DecodeStepControl for Stop {
            fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
                self.checks += 1;
                (self.checks == 2).then_some(self.cause)
            }
        }
        use DecodeCancellationKind::*;
        let value = "x".repeat(3 * CHECKPOINT_BYTES);
        for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
            ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
            let mut control = Stop { checks: 0, cause };
            let mut writer = ControlledDigest::new(b"cancel-test", u64::MAX, &mut control);
            assert_eq!(writer.value(&value).unwrap_err().cancellation(), Some(cause));
            drop(writer);
            assert_eq!(control.checks, 2);
        }
    }
}
