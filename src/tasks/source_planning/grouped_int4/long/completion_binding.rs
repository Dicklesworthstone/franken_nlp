//! Private streaming commitments for complete multi-pass source results.
//! Hashes carry association authority inside one process and never enter JSON.
use super::*;

pub(super) struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}

pub(super) fn request<C: DecodeStepControl>(domain: &[u8], nested_request: [u8; 32],
    authority: &(impl Serialize + ?Sized), mapping: Int4SourceMapLimits, control: &mut C)
    -> Result<[u8; 32], Int4SourceMapError> {
    let mut check = || checkpoint(control).map_err(Int4SourceMapError::from);
    let mut writer = Binding::new(domain, u64::MAX, &mut check)?;
    writer.hash.update(nested_request);
    writer.value(authority)?;
    // Bind the original complete ceiling, before discovery reservation or a
    // verification meet can hide a difference between two admitted requests.
    writer.value(&(mapping.chunks, mapping.reduction, mapping.max_model_work,
        (mapping.mask_limits.max_trie_node_visits, mapping.mask_limits.checkpoint_interval_nodes),
        mapping.mask_visits_per_chunk, mapping.max_mask_visits, mapping.verification))?;
    writer.finish()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn result<C: DecodeStepControl>(domain: &[u8], request: [u8; 32],
    discovery_request: [u8; 32], discovery_result: [u8; 32],
    value: &(impl Serialize + ?Sized), cap: u64, control: &mut C)
    -> Result<[u8; 32], Int4SourceMapError> {
    let mut check = || checkpoint(control).map_err(Int4SourceMapError::from);
    let mut writer = Binding::new(domain, cap, &mut check)?;
    writer.hash.update(request);
    // The nested hashes are not serialized. Bind them explicitly so a
    // different discovery owner with identical public output cannot be swapped.
    writer.hash.update(discovery_request);
    writer.hash.update(discovery_result);
    writer.value(value)?;
    writer.finish()
}

/// Preserve the original source-map result commitment byte for byte while
/// making its bounded serialization interruptible by the caller's control.
pub(super) fn source_map_result_with_checkpoints<F>(run: &Int4SourceMapRun,
    cap: u64, check: &mut F) -> Result<[u8; 32], Int4SourceMapError>
    where F: FnMut() -> Result<(), Int4SourceMapError> {
    let mut writer = Binding::new(b"fnlp-int4-source-map-result-v1", cap, check)?;
    writer.value(run)?;
    writer.finish()
}

const CHECKPOINT_BYTES: usize = 64 * 1024;
struct Binding<'a, F> {
    hash: Sha256,
    remaining: u64,
    until_checkpoint: usize,
    check: &'a mut F,
    failure: Option<Int4SourceMapError>,
}
impl<'a, F: FnMut() -> Result<(), Int4SourceMapError>> Binding<'a, F> {
    fn new(domain: &[u8], cap: u64, check: &'a mut F) -> Result<Self, Int4SourceMapError> {
        check()?;
        let mut hash = Sha256::new(); hash.update(domain); hash.update([0]);
        Ok(Self { hash, remaining: cap, until_checkpoint: CHECKPOINT_BYTES, check, failure: None })
    }
    fn poll(&mut self) -> Result<(), Int4SourceMapError> { (self.check)() }
    fn value(&mut self, value: &(impl Serialize + ?Sized)) -> Result<(), Int4SourceMapError> {
        let encoded = serde_json::to_writer(&mut *self, value);
        if let Some(error) = self.failure.take() { return Err(error); }
        encoded.map_err(|_| Int4SourceMapError::Admission)?;
        self.hash.update([0]);
        Ok(())
    }
    fn finish(mut self) -> Result<[u8; 32], Int4SourceMapError> {
        self.poll()?;
        Ok(self.hash.finalize().into())
    }
}
impl<F: FnMut() -> Result<(), Int4SourceMapError>> std::io::Write for Binding<'_, F> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let Some(remaining) = self.remaining.checked_sub(bytes.len() as u64) else {
            self.failure = Some(Int4SourceError::from(SourcePlanningError::OutputBudget).into());
            return Err(std::io::Error::other("INT4 completion binding limit"));
        };
        self.remaining = remaining;
        let mut offset = 0;
        while offset < bytes.len() {
            if self.until_checkpoint == 0 {
                if let Err(error) = self.poll() {
                    self.failure = Some(error);
                    return Err(std::io::Error::other("INT4 completion binding stopped"));
                }
                self.until_checkpoint = CHECKPOINT_BYTES;
            }
            let count = self.until_checkpoint.min(bytes.len() - offset);
            self.hash.update(&bytes[offset..offset + count]);
            self.until_checkpoint -= count;
            offset += count;
        }
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Control { calls: usize, stop: usize, cause: DecodeCancellationKind }
    impl DecodeStepControl for Control {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.calls += 1;
            (self.calls == self.stop).then_some(self.cause)
        }
    }

    #[test]
    fn source_map_result_commitment_remains_byte_for_byte_compatible() {
        use super::super::tests as maps;
        let planner = maps::planner();
        let task = SourceMapTask::Summarize(SummaryOptions::default());
        let prepared = maps::prepare_task(&planner, "éééééééé", &task, maps::limits(), 32);
        let admitted = maps::admitted(&prepared);
        let driver = maps::script(&prepared);
        let run = prepared.execute_with_driver(&admitted, driver).unwrap();
        assert_eq!(run.result_binding,
            private_digest(b"fnlp-int4-source-map-result-v1", &run, u64::MAX).unwrap());
    }

    #[test]
    fn exact_public_byte_cap_succeeds_and_one_byte_short_refuses() {
        let value = "é上海".repeat(64);
        let bytes = serde_json::to_vec(&value).unwrap();
        let digest = result(b"test", [1; 32], [2; 32], [3; 32], &value,
            bytes.len() as u64, &mut Continue).unwrap();
        assert_ne!(digest, [0; 32]);
        let error = result(b"test", [1; 32], [2; 32], [3; 32], &value,
            bytes.len() as u64 - 1, &mut Continue).unwrap_err();
        assert!(matches!(error, Int4SourceMapError::Source(_)));
    }

    #[test]
    fn one_large_serialization_write_preserves_every_terminal_cancellation_cause() {
        use DecodeCancellationKind::*;
        let value = "x".repeat(3 * CHECKPOINT_BYTES);
        for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
            ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
            let mut control = Control { calls: 0, stop: 2, cause };
            let error = result(b"test", [1; 32], [2; 32], [3; 32], &value,
                u64::MAX, &mut control).unwrap_err();
            assert_eq!(error.cancellation(), Some(cause));
            assert_eq!(control.calls, 2);
        }
    }

    #[test]
    fn hidden_discovery_authority_and_domains_change_the_same_public_result() {
        let seal = |domain: &[u8], request, discovery_request, discovery_result|
            result(domain, request, discovery_request, discovery_result, &"same public result",
                1024, &mut Continue).unwrap();
        let expected = seal(b"first", [1; 32], [2; 32], [3; 32]);
        assert_ne!(expected, seal(b"second", [1; 32], [2; 32], [3; 32]));
        assert_ne!(expected, seal(b"first", [4; 32], [2; 32], [3; 32]));
        assert_ne!(expected, seal(b"first", [1; 32], [4; 32], [3; 32]));
        assert_ne!(expected, seal(b"first", [1; 32], [2; 32], [4; 32]));
    }
}
