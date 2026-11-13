//! Process-private associations for complete redaction requests and results.
//! Streaming hashes never expose source equality or allocate another text copy.
use sha2::{Digest, Sha256};
use super::*;

pub(super) struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}

pub(super) fn source<F>(recipe: &Int4DocumentRedactor<'_>, source: &str, check: &mut F)
    -> Result<[u8; 32], Int4LongRedactionError>
where F: FnMut() -> Result<(), Int4LongRedactionError> {
    let mut writer = Binding::new(b"fnlp-int4-document-redaction-source-v1", u64::MAX, check)?;
    let config = &recipe.config;
    let planning = config.planning;
    let compiler = planning.compiler;
    let mapping = config.mapping;
    let identity = recipe.identity.receipt_identity().map_err(|_| Int4RedactionError::Identity)?;
    writer.value(&(source, identity, recipe.group_width, &config.ner, config.per_chunk))?;
    writer.value(&((planning.max_input_bytes, planning.max_context_tokens, planning.max_passages),
        (compiler.max_schema_bytes, compiler.max_string_bytes, compiler.max_array_items, compiler.max_output_bytes,
            compiler.max_states, compiler.max_transitions, compiler.max_mask_bytes), planning.source))?;
    writer.value(&(mapping.chunks, mapping.reduction, mapping.max_model_work,
        (mapping.mask_limits.max_trie_node_visits, mapping.mask_limits.checkpoint_interval_nodes),
        mapping.mask_visits_per_chunk, mapping.max_mask_visits, mapping.verification,
        config.max_result_bytes, recipe.planner.document_encoding(), recipe.planner.map_document_encoding_limits()))?;
    writer.finish()
}

pub(super) fn request<F>(source: [u8; 32], request: &RedactionRequest,
    pseudonyms: Option<&PseudonymIdentity>, check: &mut F) -> Result<[u8; 32], Int4LongRedactionError>
where F: FnMut() -> Result<(), Int4LongRedactionError> {
    let mut writer = Binding::new(b"fnlp-int4-document-redaction-request-v1", u64::MAX, check)?;
    writer.seal(source);
    writer.value(&(request, pseudonyms))?;
    writer.finish()
}

pub(super) fn result<F>(run: &Int4LongRedactionRun, cap: u64, check: &mut F)
    -> Result<[u8; 32], Int4LongRedactionError>
where F: FnMut() -> Result<(), Int4LongRedactionError> {
    let mut writer = Binding::new(b"fnlp-int4-document-redaction-result-v1", cap, check)?;
    writer.seal(run.request_binding);
    writer.seal(run.original.preflight.source_binding);
    if let Some(stage) = &run.verification { writer.seal(stage.preflight.source_binding); }
    writer.value(run)?;
    writer.finish()
}

pub(super) fn cohort<F>(run: &Int4LongRedactionCohortRun, cap: u64, check: &mut F)
    -> Result<[u8; 32], Int4LongRedactionError>
where F: FnMut() -> Result<(), Int4LongRedactionError> {
    let mut writer = Binding::new(b"fnlp-int4-document-redaction-cohort-result-v1", cap, check)?;
    // These owners can differ even when all public JSON is identical. The
    // public width is sealed together with the actual completed-stage stats.
    writer.seal(run.output.request_binding);
    writer.seal(run.output.result_binding);
    writer.value(run)?;
    writer.finish()
}

const CHECKPOINT_BYTES: usize = 64 * 1024;
struct Binding<'a, F> {
    hash: Sha256,
    remaining: u64,
    until_checkpoint: usize,
    check: &'a mut F,
    failure: Option<Int4LongRedactionError>,
}
impl<'a, F: FnMut() -> Result<(), Int4LongRedactionError>> Binding<'a, F> {
    fn new(domain: &[u8], cap: u64, check: &'a mut F) -> Result<Self, Int4LongRedactionError> {
        check()?;
        let mut hash = Sha256::new(); hash.update(domain); hash.update([0]);
        Ok(Self { hash, remaining: cap, until_checkpoint: CHECKPOINT_BYTES, check, failure: None })
    }
    fn seal(&mut self, binding: [u8; 32]) {
        // Fixed-size private framing is outside the public result byte cap.
        self.hash.update(binding); self.hash.update([0]);
    }
    fn value(&mut self, value: &(impl Serialize + ?Sized)) -> Result<(), Int4LongRedactionError> {
        let encoded = serde_json::to_writer(&mut *self, value);
        if let Some(error) = self.failure.take() { return Err(error); }
        encoded.map_err(|_| RedactError::Serialization)?;
        self.hash.update([0]);
        Ok(())
    }
    fn finish(self) -> Result<[u8; 32], Int4LongRedactionError> {
        (self.check)()?;
        Ok(self.hash.finalize().into())
    }
}
impl<F: FnMut() -> Result<(), Int4LongRedactionError>> std::io::Write for Binding<'_, F> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if self.failure.is_some() { return Err(std::io::Error::other("INT4 redaction binding stopped")); }
        let Some(remaining) = self.remaining.checked_sub(bytes.len() as u64) else {
            self.failure = Some(RedactError::OutputBudget.into());
            return Err(std::io::Error::other("INT4 redaction binding limit"));
        };
        self.remaining = remaining;
        let mut offset = 0;
        while offset < bytes.len() {
            if self.until_checkpoint == 0 {
                if let Err(error) = (self.check)() {
                    self.failure = Some(error);
                    return Err(std::io::Error::other("INT4 redaction binding stopped"));
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
pub(crate) fn seal_fixture(run: &mut Int4LongRedactionRun, request: &RedactionRequest,
    pseudonyms: Option<&PseudonymIdentity>) {
    run.request_binding = self::request(run.original.preflight.source_binding, request, pseudonyms,
        &mut || Ok(())).unwrap();
    run.result_binding = result(run, run.original.preflight.max_result_bytes, &mut || Ok(())).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_large_write_is_cancellable_and_preserves_all_original_causes() {
        use DecodeCancellationKind::*;
        let value = "x".repeat(3 * CHECKPOINT_BYTES);
        for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
            ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
            let mut calls = 0;
            let mut check = || {
                calls += 1;
                if calls == 2 { Err(Int4RedactionError::Cancelled(cause).into()) } else { Ok(()) }
            };
            let mut writer = Binding::new(b"test", u64::MAX, &mut check).unwrap();
            assert_eq!(writer.value(&value).unwrap_err().cancellation(), Some(cause));
        }
    }

    #[test]
    fn serialization_cap_excludes_private_framing_and_rejects_one_byte_short() {
        let value = "é上海".repeat(64);
        let bytes = serde_json::to_vec(&value).unwrap();
        let mut check = || Ok(());
        let mut writer = Binding::new(b"test", bytes.len() as u64, &mut check).unwrap();
        writer.seal([3; 32]); writer.value(&value).unwrap();
        assert_ne!(writer.finish().unwrap(), [0; 32]);
        let mut writer = Binding::new(b"test", bytes.len() as u64 - 1, &mut check).unwrap();
        assert!(matches!(writer.value(&value),
            Err(Int4LongRedactionError::Redaction(Int4RedactionError::Redaction(RedactError::OutputBudget)))));
    }
}
