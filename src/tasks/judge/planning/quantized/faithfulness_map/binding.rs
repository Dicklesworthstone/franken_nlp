//! Private request and complete-result association. This is not a publisher
//! signature or exported source fingerprint. Values stream without a JSON tree.
use super::*;
use std::io;

#[allow(clippy::too_many_arguments)]
pub(super) fn request<C: DecodeStepControl>(request: &JudgeRequest,
    context: &PlanContext<'_>, planning: JudgeLimits, limits: Int8FaithfulnessMapLimits,
    encoding: &UntrustedEncoding, work: SafeBpeWork, chunks: &ChunkPlan<'_>,
    plans: &[PreparedInt8Judge], control: &mut C) -> Result<[u8; 32], Int8FaithfulnessMapError> {
    checkpoint(control)?;
    if chunks.chunks().len() != plans.len() { return Err(Int8FaithfulnessMapError::Admission); }
    // The finite input and complete plan have already been admitted. Preserve
    // even unused ceilings and the caller's original model/context identity.
    let mut writer = BoundedDigest::new(b"fnlp-int8-faithfulness-map-request-v1", u64::MAX, control);
    writer.value(request)?;
    writer.value(&(context.execution_identity(), context.budget_ceiling()))?;
    writer.value(&(limits.chunks, limits.max_model_work, limits.max_retained_result_bytes,
        limits.max_result_bytes, limits.aggregation, encoding, work))?;
    let head = planning.per_head;
    writer.value(&(planning.max_total_prompt_tokens, planning.max_total_candidate_tokens,
        planning.max_total_projected_logits, planning.max_output_bytes))?;
    writer.value(&(head.max_candidates, head.max_total_tokens, head.max_nodes,
        head.max_depth, head.max_candidate_id_bytes, head.max_projected_logits))?;
    writer.value(&plans.len())?;
    for (chunk, plan) in chunks.chunks().iter().zip(plans) {
        writer.checkpoint()?;
        writer.value(&(chunk.id(), chunk.span(), chunk.tokens(), plan.execution_identity(), plan.planned_work()))?;
    }
    writer.checkpoint()?;
    Ok(writer.finish().0)
}
pub(super) fn result<C: DecodeStepControl>(run: &Int8FaithfulnessMapRun, cap: u64, control: &mut C)
    -> Result<([u8; 32], u64), Int8FaithfulnessMapError> {
    let mut writer = BoundedDigest::new(b"fnlp-int8-faithfulness-map-result-v1", cap, control);
    writer.hash.update(run.request_binding); writer.hash.update([0]);
    writer.value(run)?;
    writer.checkpoint()?;
    Ok(writer.finish())
}
pub(super) fn value_bytes<C: DecodeStepControl>(value: &impl Serialize, cap: u64, control: &mut C)
    -> Result<u64, Int8FaithfulnessMapError> {
    let mut writer = BoundedDigest::new(b"fnlp-int8-faithfulness-map-byte-count-v1", cap, control);
    writer.value(value)?;
    writer.checkpoint()?;
    Ok(writer.finish().1)
}

// These repository-owned values contain finite validated scores. JSON key
// ordering changes the private digest but never canonical serialized length.
const CHECKPOINT_BYTES: usize = 64 * 1024;
struct BoundedDigest<'a, C> {
    hash: Sha256,
    bytes: u64,
    cap: u64,
    exceeded: bool,
    control: &'a mut C,
    until_checkpoint: usize,
    cancelled: Option<DecodeCancellationKind>,
}
impl<'a, C: DecodeStepControl> BoundedDigest<'a, C> {
    fn new(domain: &[u8], cap: u64, control: &'a mut C) -> Self {
        let mut hash = Sha256::new(); hash.update(domain); hash.update([0]);
        Self { hash, bytes: 0, cap, exceeded: false, control, until_checkpoint: 0, cancelled: None }
    }
    fn checkpoint(&mut self) -> Result<(), Int8FaithfulnessMapError> {
        if self.cancelled.is_none() { self.cancelled = self.control.prefill_checkpoint(0); }
        if let Some(cause) = self.cancelled { return Err(Int8JudgeError::Cancelled(cause).into()); }
        self.until_checkpoint = CHECKPOINT_BYTES; Ok(())
    }
    fn value(&mut self, value: &(impl Serialize + ?Sized)) -> Result<(), Int8FaithfulnessMapError> {
        if serde_json::to_writer(&mut *self, value).is_err() {
            if let Some(cause) = self.cancelled { return Err(Int8JudgeError::Cancelled(cause).into()); }
            return Err(if self.exceeded { Int8FaithfulnessMapError::Output }
                else { Int8FaithfulnessMapError::Serialization });
        }
        // Zero is escaped in JSON, making adjacent values unambiguous.
        self.hash.update([0]); Ok(())
    }
    fn finish(self) -> ([u8; 32], u64) { (self.hash.finalize().into(), self.bytes) }
}
impl<C: DecodeStepControl> io::Write for BoundedDigest<'_, C> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let mut written = 0;
        while written < bytes.len() {
            if self.until_checkpoint == 0 {
                self.checkpoint().map_err(|_| io::Error::other("INT8 faithfulness serialization cancelled"))?;
            }
            let count = self.until_checkpoint.min(bytes.len() - written);
            let Some(next) = self.bytes.checked_add(count as u64).filter(|&next| next <= self.cap) else {
                self.exceeded = true; return Err(io::Error::other("INT8 faithfulness serialization limit"));
            };
            self.hash.update(&bytes[written..written + count]);
            self.bytes = next; self.until_checkpoint -= count; written += count;
        }
        Ok(written)
    }
    fn flush(&mut self) -> io::Result<()> { Ok(()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_bytes_match_canonical_escaping_and_cancel_inside_large_writes() {
        let value = serde_json::json!({"z":["é𐐀", "\"\\\u{0001}\n\r\t\u{0008}\u{000c}", -0.0, 1.25e40], "a":null});
        let exact = canonjson::canonical_bytes(&value).unwrap().len() as u64;
        assert_eq!(value_bytes(&value, exact, &mut Continue).unwrap(), exact);
        assert!(matches!(value_bytes(&value, exact - 1, &mut Continue), Err(Int8FaithfulnessMapError::Output)));
        struct Stop { checks: usize, cause: DecodeCancellationKind }
        impl DecodeStepControl for Stop {
            fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
                self.checks += 1; (self.checks == 2).then_some(self.cause)
            }
        }
        let source = "x".repeat(CHECKPOINT_BYTES * 3);
        use DecodeCancellationKind::*;
        for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
            ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
            let mut control = Stop { checks: 0, cause };
            let error = value_bytes(&source, u64::MAX, &mut control).unwrap_err();
            assert_eq!(error.cancellation(), Some(cause)); assert_eq!(control.checks, 2);
        }
    }
}
