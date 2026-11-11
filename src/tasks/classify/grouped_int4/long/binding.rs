//! Process-private request and completed-payload seals. No wire authority.
use super::*;

#[derive(Clone, Copy)]
pub(super) struct Authority {
    pub ceiling: TaskBudget,
    pub planning: ClassificationLimits,
}

/// Bind the exact source/taxonomy, every declared task and map allowance, and
/// the ordered compiled identities. Equal geometry, score-language sizes or
/// work counters cannot authorize results from a different admitted request.
#[allow(clippy::too_many_arguments)]
pub(super) fn request<C: DecodeStepControl>(request: &ClassificationRequest,
    limits: Int4ClassificationMapLimits, authority: Authority, encoding: &UntrustedEncoding,
    work: SafeBpeWork, chunks: &ChunkPlan<'_>, plans: &[PreparedInt4Classification], control: &mut C)
    -> Result<[u8; 32], Int4ClassificationMapError> {
    planning::checkpoint(control)?;
    if chunks.chunks().len() != plans.len() { return Err(Int4ClassificationMapError::Admission); }
    // Source, label and plan sizes have already passed their finite declared
    // ceilings. Streaming avoids a second source-sized serialization buffer.
    let mut writer = ControlledDigest::new(b"fnlp-int4-classification-map-request-v1", u64::MAX, control);
    writer.value(request)?;
    writer.value(&(limits.chunks, limits.max_model_work, limits.max_retained_result_bytes,
        limits.max_result_bytes, limits.aggregation, encoding, work))?;
    let p = authority.planning; let s = p.scoring;
    writer.value(&(authority.ceiling, p.max_labels, p.max_input_bytes, p.max_label_id_bytes,
        p.max_label_description_bytes, p.max_total_label_bytes, p.max_context_tokens,
        p.max_total_prompt_tokens, p.max_work))?;
    writer.value(&(s.max_candidates, s.max_total_tokens, s.max_nodes, s.max_depth,
        s.max_candidate_id_bytes, s.max_projected_logits))?;
    writer.value(&plans.len())?;
    for (chunk, plan) in chunks.chunks().iter().zip(plans) {
        writer.checkpoint()?;
        let identity = plan.execution_identity().receipt_identity()
            .map_err(|_| Int4ClassificationMapError::Admission)?;
        writer.value(&(chunk.id(), chunk.span(), chunk.tokens(), identity))?;
    }
    writer.checkpoint()?;
    Ok(writer.finish())
}

#[cfg(test)]
pub(super) fn result(run: &Int4ClassificationMapRun, cap: u64)
    -> Result<[u8; 32], Int4ClassificationMapError> {
    result_with_control(run, cap, &mut Continue)
}
pub(super) fn result_with_control<C: DecodeStepControl>(run: &Int4ClassificationMapRun, cap: u64,
    control: &mut C) -> Result<[u8; 32], Int4ClassificationMapError> {
    let mut writer = ControlledDigest::new(b"fnlp-int4-classification-map-result-v1", cap, control);
    writer.inner.seal(&run.request_binding);
    writer.value(run)?;
    writer.checkpoint()?;
    Ok(writer.finish())
}
#[cfg(test)]
pub(super) fn cohort(run: &Int4ClassificationMapCohortRun, cap: u64)
    -> Result<[u8; 32], Int4ClassificationMapError> {
    cohort_with_control(run, cap, &mut Continue)
}
pub(super) fn cohort_with_control<C: DecodeStepControl>(run: &Int4ClassificationMapCohortRun, cap: u64,
    control: &mut C) -> Result<[u8; 32], Int4ClassificationMapError> {
    let mut writer = ControlledDigest::new(b"fnlp-int4-classification-map-cohort-result-v1", cap, control);
    // Two different requests may legitimately produce identical public JSON.
    // Replacing the public `output` owner must not carry unrelated physical
    // completion facts into that other request's otherwise valid preflight.
    writer.inner.seal(&run.output.request_binding);
    writer.inner.seal(&run.output.result_binding);
    // Equal window counts do not identify the executed chunk grouping.
    writer.inner.width(run.effective_cohort_chunks());
    writer.value(run)?;
    writer.checkpoint()?;
    Ok(writer.finish())
}

/// Only repository-owned structs enter this private protocol. Every public
/// payload field is streamed in its deterministic serde order; hidden seals
/// are skipped, so they never become a public content/equality fingerprint.
#[cfg(test)]
pub(super) fn digest(domain: &[u8], value: &(impl Serialize + ?Sized), cap: u64)
    -> Result<[u8; 32], Int4ClassificationMapError> {
    let mut writer = PrivateDigest::new(domain, cap);
    writer.value(value)?;
    Ok(writer.finish())
}

struct PrivateDigest { hash: Sha256, remaining: u64, exceeded: bool }
impl PrivateDigest {
    fn new(domain: &[u8], cap: u64) -> Self {
        let mut hash = Sha256::new(); hash.update(domain); hash.update([0]);
        Self { hash, remaining: cap, exceeded: false }
    }
    fn seal(&mut self, binding: &[u8; 32]) {
        // Fixed-size private framing is not part of the public byte cap and
        // never passes through Serialize or Debug.
        self.hash.update(binding); self.hash.update([0]);
    }
    // Hidden fixed-size schedule framing is outside the public byte ceiling.
    fn width(&mut self, width: usize) {
        self.hash.update((width as u64).to_le_bytes()); self.hash.update([0]);
    }
    #[cfg(test)]
    fn value(&mut self, value: &(impl Serialize + ?Sized)) -> Result<(), Int4ClassificationMapError> {
        if serde_json::to_writer(&mut *self, value).is_err() {
            return Err(if self.exceeded { Int4ClassificationMapError::Output }
                else { Int4ClassificationMapError::Serialization });
        }
        // Frame each typed JSON value, including adjacent scalar records.
        self.hash.update([0]);
        Ok(())
    }
    fn finish(self) -> [u8; 32] { self.hash.finalize().into() }
}
impl std::io::Write for PrivateDigest {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let Some(remaining) = self.remaining.checked_sub(bytes.len() as u64) else {
            self.exceeded = true;
            return Err(std::io::Error::other("INT4 classification private binding limit"));
        };
        self.remaining = remaining; self.hash.update(bytes); Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
}

const CHECKPOINT_BYTES: usize = 64 * 1024;

/// Private strings can expand during escaping. Poll on serialized-byte
/// progress, including within a single large serde write, while keeping the
/// original stream/hash unchanged and never allocating a serialized copy.
struct ControlledDigest<'a, C> {
    inner: PrivateDigest,
    control: &'a mut C,
    until_checkpoint: usize,
    cancelled: Option<DecodeCancellationKind>,
}
impl<'a, C: DecodeStepControl> ControlledDigest<'a, C> {
    fn new(domain: &[u8], cap: u64, control: &'a mut C) -> Self {
        Self { inner: PrivateDigest::new(domain, cap), control,
            until_checkpoint: 0, cancelled: None }
    }
    fn checkpoint(&mut self) -> Result<(), Int4ClassificationMapError> {
        if self.cancelled.is_none() { self.cancelled = self.control.prefill_checkpoint(0); }
        if let Some(cause) = self.cancelled {
            return Err(ClassificationPlanningError::Cancelled(cause).into());
        }
        self.until_checkpoint = CHECKPOINT_BYTES;
        Ok(())
    }
    fn value(&mut self, value: &(impl Serialize + ?Sized)) -> Result<(), Int4ClassificationMapError> {
        if serde_json::to_writer(&mut *self, value).is_err() {
            if let Some(cause) = self.cancelled {
                return Err(ClassificationPlanningError::Cancelled(cause).into());
            }
            return Err(if self.inner.exceeded { Int4ClassificationMapError::Output }
                else { Int4ClassificationMapError::Serialization });
        }
        self.inner.hash.update([0]);
        Ok(())
    }
    fn finish(self) -> [u8; 32] { self.inner.finish() }
}
impl<C: DecodeStepControl> std::io::Write for ControlledDigest<'_, C> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let mut written = 0;
        while written < bytes.len() {
            if self.until_checkpoint == 0 {
                self.checkpoint().map_err(|_| std::io::Error::other("INT4 classification binding cancelled"))?;
            }
            let count = self.until_checkpoint.min(bytes.len() - written);
            std::io::Write::write(&mut self.inner, &bytes[written..written + count])?;
            self.until_checkpoint -= count;
            written += count;
        }
        Ok(written)
    }
    fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
}

pub(super) struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_large_request_write_polls_between_quanta_and_preserves_every_cancellation_cause() {
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
            let mut writer = ControlledDigest::new(b"request-cancellation-test", u64::MAX, &mut control);
            let error = writer.value(&source).err().unwrap();
            assert_eq!(error.cancellation(), Some(cause));
            assert_eq!(writer.inner.remaining, u64::MAX - CHECKPOINT_BYTES as u64);
            drop(writer);
            assert_eq!(control.checks, 2);
        }
        struct KeepGoing;
        impl DecodeStepControl for KeepGoing {
            fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
        }
        let mut control = KeepGoing;
        let mut writer = ControlledDigest::new(b"request-cancellation-test", u64::MAX, &mut control);
        writer.value(&source).unwrap();
        assert_eq!(writer.finish(), digest(b"request-cancellation-test", &source, u64::MAX).unwrap());
    }
}
