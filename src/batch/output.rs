//! Complete bounded canonical records; failed writes permanently poison output.
use super::*;
/// Kept unused by ordinary records so resource failure can emit run_error.
pub(super) const TERMINAL_RESERVE: usize = 2048;

/// Carry the embedding host's output/admission reservation through delivery,
/// not just through inference. Wire serialization is exactly the inner result;
/// guard types require no Serialize/Debug implementation and never enter JSON.
/// Fields drop in declaration order: result storage is freed before its guard.
pub struct GuardedOutput<T, G> { result: T, _guard: G }
impl<T, G> GuardedOutput<T, G> {
    pub(crate) fn new(result: T, guard: G) -> Self { Self { result, _guard: guard } }
    pub fn result(&self) -> &T { &self.result }
}
impl<T: Serialize, G> Serialize for GuardedOutput<T, G> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.result.serialize(serializer)
    }
}

#[derive(Serialize)]
pub(super) struct Event<'a, T: Serialize> {
    protocol: &'static str,
    schema_version: u32,
    execution: &'static str,
    event: &'static str,
    epoch: u64,
    request_seq: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub caller_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_line: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub byte_offset: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<&'a T>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reserved_work: Option<BatchWork>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<BatchFault>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<&'a BatchSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eof: Option<bool>,
}
impl<'a, T: Serialize> Event<'a, T> {
    pub fn new(event: &'static str, epoch: u64, request_seq: u64) -> Self {
        Self { protocol: BATCH_PROTOCOL, schema_version: 1, execution: BATCH_EXECUTION,
            event, epoch, request_seq, caller_id: None, input_line: None, byte_offset: None,
            result: None, reserved_work: None, error: None, summary: None, eof: None }
    }
    /// Library execution variants share framing/schema, not execution claims.
    /// The legacy constructor keeps the serial label byte-for-byte unchanged.
    pub fn with_execution(mut self, execution: &'static str) -> Self {
        self.execution = execution; self
    }
}

pub(super) struct Sink<'a, W> { writer: &'a mut W, limits: BatchLimits, emitted: u64, poisoned: bool }
impl<'a, W: Write> Sink<'a, W> {
    pub fn new(writer: &'a mut W, limits: BatchLimits) -> Self { Self { writer, limits, emitted: 0, poisoned: false } }
    pub fn poisoned(&self) -> bool { self.poisoned }
    pub fn emit<T: Serialize>(&mut self, event: &Event<'_, T>, terminal: bool) -> Result<(), BatchFault> {
        if self.poisoned { return Err(BatchCode::OutputIo.into()); }
        let line_cap = if terminal { TERMINAL_RESERVE } else { self.limits.max_output_line_bytes };
        // A no-allocation size pass rejects large internal task results BEFORE
        // the canonical serializer builds its tree/byte vector. Output types
        // are trusted Rust Serialize implementations, never executable input.
        let mut count = Counter { remaining: line_cap - 1, overflow: false };
        if serde_json::to_writer(&mut count, event).is_err() {
            return Err(if count.overflow { BatchCode::OutputLineLimit } else { BatchCode::Serialization }.into());
        }
        // Serialize the ORIGINAL value, not the sizing pass's JSON: canonical
        // nonfinite-number rejection must not be bypassed by serde's nulls.
        let mut bytes = canonjson::canonical_bytes(event).map_err(|_| BatchCode::Serialization)?;
        if bytes.len() >= line_cap { return Err(BatchCode::OutputLineLimit.into()); }
        bytes.try_reserve_exact(1).map_err(|_| BatchCode::Allocation)?; bytes.push(b'\n');
        let total = self.emitted.checked_add(bytes.len() as u64).ok_or(BatchCode::OutputLimit)?;
        let ceiling = if terminal { self.limits.max_output_bytes }
            else { self.limits.max_output_bytes - TERMINAL_RESERVE as u64 };
        if total > ceiling { return Err(BatchCode::OutputLimit.into()); }
        // Treat even a flush-only error as possibly delivered. Never retry and
        // never append a terminal record to an unknown/truncated output prefix.
        if self.writer.write_all(&bytes).and_then(|_| self.writer.flush()).is_err() {
            self.poisoned = true; return Err(BatchCode::OutputIo.into());
        }
        self.emitted = total; Ok(())
    }
}
struct Counter { remaining: usize, overflow: bool }
impl Write for Counter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > self.remaining {
            self.overflow = true; return Err(std::io::Error::other("bounded output"));
        }
        self.remaining -= bytes.len(); Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
}

#[cfg(test)]
mod tests;
