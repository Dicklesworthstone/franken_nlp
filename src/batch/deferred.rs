//! Final success staged inside execution and published only after host drain.
use super::*;

/// Not cloneable: publication consumes the sole already-sized terminal frame.
/// The host must retain its stream/output reservation for this object's whole
/// lifetime, including cancellation or panic between preparation and delivery.
pub(crate) struct DeferredCompletion { frame: Vec<u8>, summary: BatchSummary }
impl DeferredCompletion {
    pub(crate) fn publish<W: Write, C: DecodeStepControl>(self, writer: &mut W, control: &mut C)
        -> Result<BatchSummary, BatchRunError> {
        // This is the SAME run control after all native workers joined. No
        // fresh sink, output allowance, deadline or checkpoint budget is made.
        checkpoint(control).map_err(|fault| BatchRunError { fault, summary: self.summary })?;
        writer.write_all(&self.frame).and_then(|_| writer.flush())
            .map_err(|_| BatchRunError { fault: BatchCode::OutputIo.into(), summary: self.summary })?;
        // A failed/partial write or flush cannot be retried or followed by an
        // error record. Consuming self keeps that invariant outside Sink too.
        Ok(self.summary)
    }
}

pub(crate) fn run_ndjson_deferred<R: BufRead, W: Write, P: BatchProcessor, C: DecodeStepControl>(
    reader: &mut R, writer: &mut W, processor: &mut P, limits: BatchLimits, control: &mut C,
) -> Result<DeferredCompletion, BatchRunError> {
    run_ndjson_with_completion(reader, writer, processor, limits, control,
        |sink, summary, epoch, control| {
            let mut event = Event::<()>::new("run_complete", epoch, summary.requests);
            event.summary = Some(summary);
            let prepared = sink.prepare_with_checkpoint(&event, true, || checkpoint(control))?;
            Ok(DeferredCompletion { frame: prepared.bytes, summary: *summary })
        })
}

#[cfg(test)]
mod tests;
