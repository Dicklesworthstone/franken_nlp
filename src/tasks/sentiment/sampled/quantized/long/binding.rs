//! Private streamed bindings over the full source, immutable plan and result.
use super::*;
use super::super::super::long::binding::Writer;

pub(super) fn document<C: DecodeStepControl>(request: &SentimentRequest,
    planner: &Int8SampledSentimentPlanner, limits: Int8SampledSentimentMapLimits,
    item_id: Option<&str>, control: &mut C) -> Result<[u8; 32], Int8SampledSentimentMapError> {
    let mut writer = Writer::new(b"int8-sampled-sentiment-complete-document-address-v1", u64::MAX, control);
    let mut axes = request.axes.clone(); axes.sort_unstable();
    writer.value(&(INT8_SAMPLED_SENTIMENT_MAP_EXECUTION, SAMPLED_SENTIMENT_MAP_AGGREGATION,
        request.document.as_str(), axes, request.budget, item_id))?;
    writer.value(&(limits.chunks, limits.max_model_work, limits.max_retained_result_bytes, limits.max_result_bytes))?;
    let p = planner.limits; let c = p.compile;
    writer.value(&(&planner.options, planner.ceiling, planner.input_encoding(),
        p.max_input_bytes, p.max_total_prompt_tokens, p.max_sample_json_bytes, p.max_result_bytes))?;
    writer.value(&(c.max_schema_bytes, c.max_string_bytes, c.max_array_items, c.max_output_bytes,
        c.max_states, c.max_transitions, c.max_mask_bytes))?;
    writer.value(&planner.identity.receipt_identity().map_err(|_| Int8SampledSentimentMapError::Admission)?)?;
    writer.checkpoint()?; Ok(writer.finish().0)
}
pub(super) fn chunk_address<C: DecodeStepControl>(document: [u8; 32], id: usize,
    span: VerifiedSourceSpan, tokens: usize, control: &mut C) -> Result<String, Int8SampledSentimentMapError> {
    let mut writer = Writer::new(b"int8-sampled-sentiment-original-chunk-address-v1", u64::MAX, control);
    writer.value(&(document, id, span, tokens))?; writer.checkpoint()?;
    Ok(Sha256Digest::of_bytes(&writer.finish().0).to_hex())
}
pub(super) fn geometry<C: DecodeStepControl>(chunks: impl Iterator<Item = (usize, VerifiedSourceSpan, usize)>,
    control: &mut C) -> Result<[u8; 32], Int8SampledSentimentMapError> {
    let mut writer = Writer::new(b"int8-sampled-sentiment-complete-source-geometry-v1", u64::MAX, control);
    for chunk in chunks { writer.value(&chunk)?; }
    writer.checkpoint()?; Ok(writer.finish().0)
}
pub(super) fn request<C: DecodeStepControl>(document: [u8; 32], chunks: &[PreparedChunk],
    work: SafeBpeWork, control: &mut C) -> Result<[u8; 32], Int8SampledSentimentMapError> {
    let mut writer = Writer::new(b"int8-sampled-sentiment-complete-map-request-v1", u64::MAX, control);
    writer.value(&(document, work, chunks.len()))?;
    for chunk in chunks {
        writer.checkpoint()?;
        writer.value(&(chunk.id, chunk.span, chunk.tokens,
            chunk.plan.execution_identity().receipt_identity().map_err(|_| Int8SampledSentimentMapError::Admission)?))?;
    }
    writer.checkpoint()?; Ok(writer.finish().0)
}
pub(super) fn result<C: DecodeStepControl>(run: &Int8SampledSentimentMapRun, cap: u64, control: &mut C)
    -> Result<[u8; 32], Int8SampledSentimentMapError> {
    let mut writer = Writer::new(b"int8-sampled-sentiment-complete-map-result-v1", cap, control);
    writer.bind_private(&run.request_binding);
    writer.value(run)?; writer.checkpoint()?; Ok(writer.finish().0)
}
pub(super) fn value_bytes<C: DecodeStepControl>(value: &impl Serialize, cap: u64, control: &mut C)
    -> Result<u64, Int8SampledSentimentMapError> {
    Ok(super::super::super::long::binding::value_bytes(value, cap, control)?)
}
