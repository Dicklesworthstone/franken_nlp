//! Private request/result association, never wire authentication or a public
//! source fingerprint. Hashing streams within admitted bounds and checkpoints.
use super::*;
use sha2::{Digest, Sha256};

pub(super) fn request<C: DecodeStepControl>(request: &JudgeRequest,
    limits: Int4FaithfulnessMapLimits, authority: (TaskBudget, JudgeLimits),
    chunks: &ChunkPlan<'_>, plans: &[PreparedInt4Judge], control: &mut C)
    -> Result<[u8; 32], Int4FaithfulnessMapError> {
    checkpoint(control)?;
    if chunks.chunks().len() != plans.len() { return Err(Int4FaithfulnessMapError::Admission); }
    // Input/plan sizes have already passed their finite ceilings. Do not
    // allocate another source-sized JSON copy just to associate this plan.
    let mut writer = ControlledDigest::new(b"fnlp-int4-faithfulness-map-request-v1", u64::MAX, control);
    writer.value(request)?;
    writer.value(&(limits.chunks, limits.max_model_work, limits.max_retained_result_bytes,
        limits.max_result_bytes, limits.aggregation))?;
    let (ceiling, planning) = authority;
    let head = planning.per_head;
    writer.value(&(ceiling, planning.max_total_prompt_tokens, planning.max_total_candidate_tokens,
        planning.max_total_projected_logits, planning.max_output_bytes))?;
    writer.value(&(head.max_candidates, head.max_total_tokens, head.max_nodes, head.max_depth,
        head.max_candidate_id_bytes, head.max_projected_logits))?;
    writer.value(&plans.len())?;
    for (chunk, plan) in chunks.chunks().iter().zip(plans) {
        writer.checkpoint()?;
        let identity = plan.execution_identity().receipt_identity()
            .map_err(|_| Int4FaithfulnessMapError::Admission)?;
        writer.value(&(chunk.id(), chunk.span(), chunk.tokens(), identity))?;
    }
    writer.checkpoint()?;
    Ok(writer.finish())
}

pub(super) fn result_with_control<C: DecodeStepControl>(run: &Int4FaithfulnessMapRun, cap: u64,
    control: &mut C) -> Result<[u8; 32], Int4FaithfulnessMapError> {
    let mut writer = ControlledDigest::new(b"fnlp-int4-faithfulness-map-result-v1", cap, control);
    writer.inner.seal(&run.request_binding);
    writer.value(run)?;
    writer.checkpoint()?;
    Ok(writer.finish())
}
pub(super) fn cohort_with_control<C: DecodeStepControl>(run: &Int4FaithfulnessMapCohortRun, cap: u64,
    control: &mut C) -> Result<[u8; 32], Int4FaithfulnessMapError> {
    let mut writer = ControlledDigest::new(b"fnlp-int4-faithfulness-map-cohort-result-v1", cap, control);
    // Identical public JSON can come from different sources/claims. Bind
    // the nested private owner too, so swapping whole valid outputs fails.
    writer.inner.seal(&run.output.request_binding);
    writer.inner.seal(&run.output.result_binding);
    writer.inner.width(run.effective_cohort_chunks());
    writer.value(run)?;
    writer.checkpoint()?;
    Ok(writer.finish())
}

/// Only repository-owned types enter this protocol. Their deterministic
/// serde order binds every public field; private seals are skipped entirely.
struct PrivateDigest { hash: Sha256, remaining: u64, exceeded: bool }
impl PrivateDigest {
    fn new(domain: &[u8], cap: u64) -> Self {
        let mut hash = Sha256::new(); hash.update(domain); hash.update([0]);
        Self { hash, remaining: cap, exceeded: false }
    }
    fn seal(&mut self, seal: &[u8; 32]) { self.hash.update(seal); self.hash.update([0]); }
    // Hidden fixed-size schedule framing is outside the public byte ceiling.
    fn width(&mut self, width: usize) { self.hash.update((width as u64).to_le_bytes()); self.hash.update([0]); }
    #[cfg(test)]
    fn value(&mut self, value: &(impl Serialize + ?Sized)) -> Result<(), Int4FaithfulnessMapError> {
        serde_json::to_writer(&mut *self, value).map_err(|_| self.error())?;
        self.hash.update([0]);
        Ok(())
    }
    fn error(&self) -> Int4FaithfulnessMapError {
        if self.exceeded { Int4FaithfulnessMapError::Output } else { Int4FaithfulnessMapError::Serialization }
    }
    fn finish(self) -> [u8; 32] { self.hash.finalize().into() }
}
impl std::io::Write for PrivateDigest {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let Some(remaining) = self.remaining.checked_sub(bytes.len() as u64) else {
            self.exceeded = true;
            return Err(std::io::Error::other("INT4 faithfulness private binding limit"));
        };
        self.remaining = remaining; self.hash.update(bytes); Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
}

const CHECKPOINT_BYTES: usize = 64 * 1024;
struct ControlledDigest<'a, C> {
    inner: PrivateDigest,
    control: &'a mut C,
    until_checkpoint: usize,
    cancelled: Option<DecodeCancellationKind>,
}
impl<'a, C: DecodeStepControl> ControlledDigest<'a, C> {
    fn new(domain: &[u8], cap: u64, control: &'a mut C) -> Self {
        Self { inner: PrivateDigest::new(domain, cap), control, until_checkpoint: 0, cancelled: None }
    }
    fn checkpoint(&mut self) -> Result<(), Int4FaithfulnessMapError> {
        if self.cancelled.is_none() { self.cancelled = self.control.prefill_checkpoint(0); }
        if let Some(cause) = self.cancelled { return Err(Int4JudgeError::Cancelled(cause).into()); }
        self.until_checkpoint = CHECKPOINT_BYTES;
        Ok(())
    }
    fn value(&mut self, value: &(impl Serialize + ?Sized)) -> Result<(), Int4FaithfulnessMapError> {
        if serde_json::to_writer(&mut *self, value).is_err() {
            if let Some(cause) = self.cancelled { return Err(Int4JudgeError::Cancelled(cause).into()); }
            return Err(self.inner.error());
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
                self.checkpoint().map_err(|_| std::io::Error::other("INT4 faithfulness binding cancelled"))?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escaped_content_and_large_writes_preserve_hashes_caps_and_all_cancellation_causes() {
        struct Stop { checks: usize, cause: Option<DecodeCancellationKind> }
        impl DecodeStepControl for Stop {
            fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
                self.checks += 1;
                if self.checks == 2 { self.cause } else { None }
            }
        }
        let source = format!("{}\n\t\"é𐐀", "x".repeat(3 * CHECKPOINT_BYTES));
        let wire = serde_json::to_vec(&source).unwrap();
        let mut plain = PrivateDigest::new(b"test-faithfulness-binding", wire.len() as u64);
        plain.value(&source).unwrap();
        let mut control = Stop { checks: 0, cause: None };
        let mut checked = ControlledDigest::new(b"test-faithfulness-binding", wire.len() as u64, &mut control);
        checked.value(&source).unwrap();
        assert_eq!(checked.finish(), plain.finish());
        assert!(control.checks >= 3);
        let mut short = PrivateDigest::new(b"test-faithfulness-binding", wire.len() as u64 - 1);
        assert!(matches!(short.value(&source), Err(Int4FaithfulnessMapError::Output)));
        use DecodeCancellationKind::*;
        for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
            ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
            let mut control = Stop { checks: 0, cause: Some(cause) };
            let mut checked = ControlledDigest::new(b"test-faithfulness-binding", u64::MAX, &mut control);
            assert_eq!(checked.value(&source).unwrap_err().cancellation(), Some(cause));
            assert_eq!(checked.inner.remaining, u64::MAX - CHECKPOINT_BYTES as u64);
            drop(checked);
            assert_eq!(control.checks, 2);
        }
    }
}
