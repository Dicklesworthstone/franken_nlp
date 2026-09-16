//! Bounded physical-line framing. No read_line/read_until growth to EOF.
use super::*;
pub(super) struct Frame { pub bytes: Vec<u8>, pub oversized: bool, pub line: u64, pub offset: u64 }

pub(super) fn read_frame<R: BufRead, C: DecodeStepControl>(reader: &mut R, limits: BatchLimits,
    summary: &mut BatchSummary, control: &mut C) -> Result<Option<Frame>, BatchFault> {
    let mut frame = Frame { bytes: Vec::new(), oversized: false, line: 0, offset: summary.input_bytes };
    let mut started = false; let mut assigned = false;
    let mut payload_seen = 0_u64; let mut first_is_cr = false;
    loop {
        checkpoint(control)?;
        let available = match reader.fill_buf() {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => return Err(BatchCode::InputIo.into()),
        };
        if available.is_empty() {
            if !started { return Ok(None); }
            // A bare CR at EOF is nonempty input, not an empty CRLF record.
            if payload_seen > 0 && !assigned { assign(summary, limits)?; }
            return Ok(Some(frame));
        }
        let remaining = limits.max_input_bytes.checked_sub(summary.input_bytes).ok_or(BatchCode::InputLimit)?;
        if remaining == 0 { return Err(BatchCode::InputLimit.into()); }
        let available = &available[..available.len().min(usize::try_from(remaining).unwrap_or(usize::MAX))];
        if !started {
            summary.input_lines = summary.input_lines.checked_add(1).ok_or(BatchCode::SequenceOverflow)?;
            frame.line = summary.input_lines; started = true;
        }
        let newline = available.iter().position(|&b| b == b'\n');
        let payload = newline.unwrap_or(available.len());
        let consumed = payload + usize::from(newline.is_some());
        if payload_seen == 0 && payload > 0 { first_is_cr = available[0] == b'\r'; }
        payload_seen = payload_seen.checked_add(payload as u64).ok_or(BatchCode::InputLimit)?;
        // Defer a single leading CR until it is known not to be empty CRLF.
        // Every other nonempty record gets a sequence even if later I/O fails.
        if !assigned && (payload_seen > 1 || (payload_seen == 1 && !first_is_cr)) {
            assign(summary, limits)?; assigned = true;
        }
        if !frame.oversized {
            let size = frame.bytes.len().checked_add(payload).ok_or(BatchCode::LineLimit)?;
            if size > limits.max_line_bytes {
                frame.oversized = true; frame.bytes.clear();
            } else {
                frame.bytes.try_reserve_exact(payload).map_err(|_| BatchCode::Allocation)?;
                frame.bytes.extend_from_slice(&available[..payload]);
            }
        }
        summary.input_bytes += consumed as u64;
        reader.consume(consumed);
        if newline.is_some() {
            if frame.bytes.last() == Some(&b'\r') { frame.bytes.pop(); }
            return Ok(Some(frame));
        }
    }
}
fn assign(summary: &mut BatchSummary, limits: BatchLimits) -> Result<(), BatchFault> {
    summary.requests = summary.requests.checked_add(1).ok_or(BatchCode::SequenceOverflow)?;
    if summary.requests > limits.max_requests { return Err(BatchCode::RequestLimit.into()); }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufReader, Cursor};
    struct Continue;
    impl DecodeStepControl for Continue {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
    }
    #[test]
    fn interrupted_record_accounting_is_independent_of_buffer_capacity() {
        let limits = BatchLimits { max_line_bytes: 8, max_document_bytes: 8, max_input_bytes: 64, ..BatchLimits::default() };
        for capacity in [1, 3, 64, 4096] {
            let mut reader = BufReader::with_capacity(capacity, Cursor::new([b'x'; 1000]));
            let mut summary = BatchSummary::default();
            let error = match read_frame(&mut reader, limits, &mut summary, &mut Continue) { Err(e) => e, Ok(_) => panic!() };
            assert_eq!(error.code, BatchCode::InputLimit); assert_eq!(summary.input_bytes, 64);
            assert_eq!(summary.requests, 1); assert_eq!(summary.input_lines, 1);
        }
    }
    #[test]
    fn only_empty_crlf_is_unsequenced_not_a_bare_carriage_return() {
        let mut reader = BufReader::with_capacity(1, Cursor::new(b"\r\n\r")); let mut summary = BatchSummary::default();
        assert!(read_frame(&mut reader, BatchLimits::default(), &mut summary, &mut Continue).unwrap().unwrap().bytes.is_empty());
        assert_eq!(summary.requests, 0);
        assert_eq!(read_frame(&mut reader, BatchLimits::default(), &mut summary, &mut Continue).unwrap().unwrap().bytes, b"\r");
        assert_eq!(summary.requests, 1);
    }
    #[test]
    fn sequence_overflow_refuses_without_wrapping_to_an_existing_id() {
        let mut summary = BatchSummary { requests: u64::MAX, ..BatchSummary::default() };
        assert_eq!(assign(&mut summary, BatchLimits::default()).unwrap_err().code, BatchCode::SequenceOverflow);
        assert_eq!(summary.requests, u64::MAX);
    }
}
