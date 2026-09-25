//! Framing/transport fault injection only; no native task success fixtures.
use super::*;
fn metadata() -> serde_json::Value {
    serde_json::json!({"protocol":"fnlp-candidate-batch-v1","evidence":"non_authoritative"})
}
#[test]
fn fragmented_input_stays_private_until_complete_flush_and_keeps_exact_numbers() {
    let mut out = Vec::new();
    let raw = b"{\"n\":12345678901234567890123456789012345678,\"text\":\"a\\nb\"}\n";
    {
        let mut writer = CandidateWriter::new(&mut out, &metadata(), 256, 4096).unwrap();
        for fragment in raw.chunks(3) { writer.write_all(fragment).unwrap(); }
        assert!(writer.inner.is_empty());
        writer.flush().unwrap();
        writer.flush().unwrap(); // must not duplicate a delivered record
    }
    let mut expected = serde_json::to_vec(&metadata()).unwrap(); expected.pop();
    expected.extend_from_slice(b",\"record\":"); expected.extend_from_slice(&raw[..raw.len() - 1]);
    expected.extend_from_slice(b"}\n");
    assert_eq!(out, expected);
    assert_eq!(out.iter().filter(|&&b| b == b'\n').count(), 1);
}
#[test]
fn incomplete_or_multiple_native_frames_never_publish_a_prefix() {
    for raw in [b"{\"secret\":1}".as_slice(), b"{}\n{}\n", b"\n", b"[]\n"] {
        let mut out = Vec::new();
        let mut writer = CandidateWriter::new(&mut out, &metadata(), 256, 4096).unwrap();
        let written = writer.write_all(raw);
        if written.is_ok() { assert!(writer.flush().is_err()); }
        assert!(writer.poisoned);
        assert!(writer.inner.is_empty());
        assert!(writer.write_all(b"{}\n").is_err());
    }
}
#[test]
fn line_budget_counts_the_inner_newline_at_the_exact_boundary() {
    let mut exact = CandidateWriter::new(Vec::new(), &metadata(), 3, 4096).unwrap();
    exact.write_all(b"{}\n").unwrap(); exact.flush().unwrap();
    let mut short = CandidateWriter::new(Vec::new(), &metadata(), 2, 4096).unwrap();
    assert!(short.write_all(b"{}\n").is_err());
    assert!(short.inner.is_empty()); assert!(short.flush().is_err());
}
#[test]
fn aggregate_output_budget_includes_every_provenance_wrapper() {
    let prefix_len = serde_json::to_vec(&metadata()).unwrap().len() - 1 + b",\"record\":".len();
    let record_bytes = prefix_len as u64 + 1 + 3;
    let mut exact = CandidateWriter::new(Vec::new(), &metadata(), 3, record_bytes).unwrap();
    exact.write_all(b"{}\n").unwrap(); exact.flush().unwrap();
    assert_eq!(exact.inner.len() as u64, record_bytes);
    assert_eq!(exact.remaining, 0);
    exact.write_all(b"{}\n").unwrap();
    assert!(exact.flush().is_err());
    assert_eq!(exact.inner.len() as u64, record_bytes);
    let mut short = CandidateWriter::new(Vec::new(), &metadata(), 3, record_bytes - 1).unwrap();
    short.write_all(b"{}\n").unwrap();
    assert!(short.flush().is_err()); assert!(short.inner.is_empty());
}
#[test]
fn oversized_provenance_is_refused_before_any_io() {
    assert!(CandidateWriter::new(Vec::new(), &serde_json::json!({}), 256, 65536).is_err());
    let huge = serde_json::json!({"x":"x".repeat(FRAME_ALLOWANCE as usize)});
    assert!(CandidateWriter::new(Vec::new(), &huge, 256, 65536).is_err());
}
struct PartialFailure { bytes: Vec<u8>, remaining: usize, calls: usize }
impl Write for PartialFailure {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.calls += 1;
        if self.remaining == 0 { return Err(io::Error::other("private transport detail")); }
        let n = self.remaining.min(bytes.len()); self.bytes.extend_from_slice(&bytes[..n]); self.remaining -= n; Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> { self.calls += 1; Ok(()) }
}
#[test]
fn partial_write_permanently_poisons_output_without_appending_an_error_frame() {
    let inner = PartialFailure { bytes: Vec::new(), remaining: 5, calls: 0 };
    let mut writer = CandidateWriter::new(inner, &metadata(), 256, 4096).unwrap();
    writer.write_all(b"{}\n").unwrap(); assert!(writer.flush().is_err());
    let before = writer.inner.calls; let bytes = writer.inner.bytes.clone();
    assert!(writer.write_all(b"{\"event\":\"run_error\"}\n").is_err());
    assert!(writer.flush().is_err());
    assert_eq!(writer.inner.calls, before); assert_eq!(writer.inner.bytes, bytes);
}
struct FlushFailure { bytes: Vec<u8>, flushes: usize }
impl Write for FlushFailure {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> { self.bytes.extend_from_slice(bytes); Ok(bytes.len()) }
    fn flush(&mut self) -> io::Result<()> { self.flushes += 1; Err(io::Error::other("private flush detail")) }
}
#[test]
fn failed_flush_never_retries_a_possibly_delivered_record() {
    let mut writer = CandidateWriter::new(FlushFailure { bytes: Vec::new(), flushes: 0 }, &metadata(), 256, 4096).unwrap();
    writer.write_all(b"{}\n").unwrap(); assert!(writer.flush().is_err());
    let bytes = writer.inner.bytes.clone();
    assert!(writer.flush().is_err()); assert_eq!(writer.inner.flushes, 1); assert_eq!(writer.inner.bytes, bytes);
}
#[test]
fn dropping_an_unflushed_record_never_publishes_private_staging() {
    let mut out = Vec::new();
    {
        let mut writer = CandidateWriter::new(&mut out, &metadata(), 256, 4096).unwrap();
        writer.write_all(b"{\"private\":1}\n").unwrap();
    }
    assert!(out.is_empty());
}
