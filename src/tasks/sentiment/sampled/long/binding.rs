//! Private source/chunk/request association and streaming native-result seals.
use super::*;
use sha2::{Digest, Sha256};

pub(super) fn document<C: DecodeStepControl>(request: &SentimentRequest,
    planner: &SampledSentimentPlanner, limits: SampledSentimentMapLimits, control: &mut C)
    -> Result<[u8; 32], SampledSentimentMapError> {
    let mut writer = Writer::new(b"sampled-sentiment-complete-document-address-v1", u64::MAX, control);
    let mut axes = request.axes.clone(); axes.sort_unstable();
    writer.value(&(SAMPLED_SENTIMENT_MAP_EXECUTION, SAMPLED_SENTIMENT_MAP_AGGREGATION,
        request.document.as_str(), axes, request.budget))?;
    writer.value(&(limits.chunks, limits.max_model_work, limits.max_retained_result_bytes, limits.max_result_bytes))?;
    let p = planner.limits; let c = p.compile;
    writer.value(&(&planner.options, planner.ceiling, planner.input_encoding(), planner.group_width,
        p.max_input_bytes, p.max_total_prompt_tokens, p.max_sample_json_bytes, p.max_result_bytes))?;
    writer.value(&(c.max_schema_bytes, c.max_string_bytes, c.max_array_items, c.max_output_bytes,
        c.max_states, c.max_transitions, c.max_mask_bytes))?;
    // Physical selected-row admission is bound by each prepared chunk below;
    // it must not change the original document/chunk sampling addresses.
    writer.value(&planner.identity.receipt_identity().map_err(|_| SampledSentimentMapError::Admission)?)?;
    writer.checkpoint()?; Ok(writer.finish().0)
}
pub(super) fn chunk_address<C: DecodeStepControl>(document: [u8; 32], id: usize,
    span: VerifiedSourceSpan, tokens: usize, control: &mut C) -> Result<String, SampledSentimentMapError> {
    let mut writer = Writer::new(b"sampled-sentiment-original-chunk-address-v1", u64::MAX, control);
    writer.value(&(document, id, span, tokens))?;
    writer.checkpoint()?;
    Ok(Sha256Digest::of_bytes(&writer.finish().0).to_hex())
}
pub(super) fn item_document<C: DecodeStepControl>(document: [u8; 32], item_id: &str,
    control: &mut C) -> Result<[u8; 32], SampledSentimentMapError> {
    let mut writer = Writer::new(SAMPLED_SENTIMENT_MAP_ITEM_ADDRESS_VERSION.as_bytes(), u64::MAX, control);
    // A JSON tuple frames the caller ID separately from the private complete
    // request commitment. Neither the raw ID nor the commitment is emitted.
    writer.value(&(document, item_id))?;
    writer.checkpoint()?; Ok(writer.finish().0)
}
pub(super) fn geometry<C: DecodeStepControl>(chunks: &[PreparedChunk], control: &mut C)
    -> Result<[u8; 32], SampledSentimentMapError> {
    geometry_values(chunks.iter().map(|chunk| (chunk.id, chunk.span, chunk.tokens)), control)
}
pub(super) fn geometry_values<C: DecodeStepControl>(chunks: impl Iterator<Item = (usize, VerifiedSourceSpan, usize)>,
    control: &mut C) -> Result<[u8; 32], SampledSentimentMapError> {
    let mut writer = Writer::new(b"sampled-sentiment-complete-source-geometry-v1", u64::MAX, control);
    for chunk in chunks { writer.value(&chunk)?; }
    writer.checkpoint()?; Ok(writer.finish().0)
}
pub(super) fn request<C: DecodeStepControl>(document: [u8; 32], chunks: &[PreparedChunk],
    work: SafeBpeWork, control: &mut C) -> Result<[u8; 32], SampledSentimentMapError> {
    let mut writer = Writer::new(b"sampled-sentiment-complete-map-request-v1", u64::MAX, control);
    writer.value(&(document, work, chunks.len()))?;
    for chunk in chunks {
        writer.checkpoint()?;
        writer.value(&(chunk.id, chunk.span, chunk.tokens,
            chunk.plan.execution_identity().receipt_identity().map_err(|_| SampledSentimentMapError::Admission)?))?;
    }
    writer.checkpoint()?; Ok(writer.finish().0)
}
pub(super) fn result<C: DecodeStepControl>(run: &SampledSentimentMapRun, cap: u64, control: &mut C)
    -> Result<[u8; 32], SampledSentimentMapError> {
    let mut writer = Writer::new(b"sampled-sentiment-complete-map-result-v1", cap, control);
    writer.hash.update(run.request_binding); writer.hash.update([0]);
    writer.value(run)?; writer.checkpoint()?; Ok(writer.finish().0)
}
pub(crate) fn value_bytes<C: DecodeStepControl>(value: &impl Serialize, cap: u64, control: &mut C)
    -> Result<u64, SampledSentimentMapError> {
    let mut writer = Writer::new(b"sampled-sentiment-bounded-output-v1", cap, control);
    writer.value(value)?; writer.checkpoint()?; Ok(writer.finish().1)
}

const CHECKPOINT_BYTES: usize = 64 * 1024;
pub(crate) struct Writer<'a, C> {
    hash: Sha256,
    bytes: u64,
    cap: u64,
    exceeded: bool,
    control: &'a mut C,
    until_checkpoint: usize,
    cancelled: Option<DecodeCancellationKind>,
}
impl<'a, C: DecodeStepControl> Writer<'a, C> {
    pub(crate) fn new(domain: &[u8], cap: u64, control: &'a mut C) -> Self {
        let mut hash = Sha256::new(); hash.update(domain); hash.update([0]);
        Self { hash, bytes: 0, cap, exceeded: false, control, until_checkpoint: 0, cancelled: None }
    }
    pub(crate) fn checkpoint(&mut self) -> Result<(), SampledSentimentMapError> {
        if self.cancelled.is_none() { self.cancelled = self.control.prefill_checkpoint(0); }
        if let Some(cause) = self.cancelled { return Err(SampledSentimentMapError::Cancelled(cause)); }
        self.until_checkpoint = CHECKPOINT_BYTES; Ok(())
    }
    pub(crate) fn value(&mut self, value: &(impl Serialize + ?Sized)) -> Result<(), SampledSentimentMapError> {
        if serde_json::to_writer(&mut *self, value).is_err() {
            if let Some(cause) = self.cancelled { return Err(SampledSentimentMapError::Cancelled(cause)); }
            return Err(if self.exceeded { SampledSentimentMapError::Output } else { SampledSentimentMapError::Serialization });
        }
        self.hash.update([0]); Ok(())
    }
    /// Private request authority prefixes the seal without spending output bytes.
    pub(crate) fn bind_private(&mut self, bytes: &[u8]) { self.hash.update(bytes); self.hash.update([0]); }
    pub(crate) fn finish(self) -> ([u8; 32], u64) { (self.hash.finalize().into(), self.bytes) }
}
impl<C: DecodeStepControl> std::io::Write for Writer<'_, C> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let Some(total) = self.bytes.checked_add(bytes.len() as u64).filter(|&n| n <= self.cap) else {
            self.exceeded = true; return Err(std::io::Error::other("sampled sentiment map output bound"));
        };
        for part in bytes.chunks(CHECKPOINT_BYTES) {
            if self.until_checkpoint < part.len() || self.until_checkpoint == 0 {
                self.checkpoint().map_err(|_| std::io::Error::other("sampled sentiment map cancelled"))?;
            }
            self.hash.update(part); self.until_checkpoint -= part.len();
        }
        self.bytes = total; Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
}
