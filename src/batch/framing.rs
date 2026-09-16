//! Bounded physical-line framing. No read_line/read_until growth to EOF.
use super::*;
pub(super) struct Frame { pub bytes: Vec<u8>, pub oversized: bool, pub line: u64, pub offset: u64 }

pub(super) fn read_frame<R: BufRead, C: DecodeStepControl>(reader: &mut R, limits: BatchLimits,
    summary: &mut BatchSummary, control: &mut C) -> Result<Option<Frame>, BatchFault> {
    let mut frame = Frame { bytes: Vec::new(), oversized: false, line: 0, offset: summary.input_bytes };
    let mut started = false;
    loop {
        checkpoint(control)?;
        let available = match reader.fill_buf() {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => return Err(BatchCode::InputIo.into()),
        };
        if available.is_empty() {
            if !started { return Ok(None); }
            // A bare CR is input data; strip CR only as part of a real CRLF.
            return Ok(Some(frame));
        }
        if !started {
            summary.input_lines = summary.input_lines.checked_add(1).ok_or(BatchCode::SequenceOverflow)?;
            frame.line = summary.input_lines; started = true;
        }
        let newline = available.iter().position(|&b| b == b'\n');
        let payload = newline.unwrap_or(available.len());
        let consumed = payload + usize::from(newline.is_some());
        let remaining = limits.max_input_bytes.checked_sub(summary.input_bytes).ok_or(BatchCode::InputLimit)?;
        if consumed as u64 > remaining { return Err(BatchCode::InputLimit.into()); }
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
