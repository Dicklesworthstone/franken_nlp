//! One trusted native event per flush, wrapped without parsing JSON numbers.
//! The source batch sink already emits canonical JSON. This private adapter
//! does not deserialize it, reinterpret precise numbers or invent task results.
use super::*;

pub(in crate::candidate_cli) struct CandidateWriter<W> {
    inner: W,
    prefix: Vec<u8>,
    bytes: Vec<u8>,
    line_cap: usize,
    remaining: u64,
    poisoned: bool,
}
impl<W: Write> CandidateWriter<W> {
    pub(in crate::candidate_cli) fn new(inner: W, provenance: &impl Serialize, line_cap: usize, output_cap: u64)
        -> Result<Self, CandidateError> {
        // Provenance is a closed, code-owned type with bounded model metadata.
        // Private construction never accepts an arbitrary raw JSON prefix.
        let mut prefix = serde_json::to_vec(provenance).map_err(|_| CandidateError::Output)?;
        if prefix.first() != Some(&b'{') || prefix.pop() != Some(b'}') || prefix.len() == 1 || line_cap == 0 {
            return Err(CandidateError::Output);
        }
        prefix.extend_from_slice(b",\"record\":");
        if prefix.len() as u64 + 1 > FRAME_ALLOWANCE || output_cap <= prefix.len() as u64 + 1 {
            return Err(CandidateError::Output);
        }
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(line_cap).map_err(|_| CandidateError::Memory)?;
        Ok(Self { inner, prefix, bytes, line_cap, remaining: output_cap, poisoned: false })
    }
    fn refused(&mut self) -> io::Error {
        self.poisoned = true;
        io::Error::other("candidate batch output refused")
    }
}
impl<W: Write> Write for CandidateWriter<W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.poisoned { return Err(self.refused()); }
        if bytes.is_empty() { return Ok(0); }
        if self.bytes.last() == Some(&b'\n')
            || self.bytes.len().checked_add(bytes.len()).is_none_or(|n| n > self.line_cap)
            || bytes.iter().position(|&b| b == b'\n').is_some_and(|index| index != bytes.len() - 1) {
            return Err(self.refused());
        }
        // Capacity was admitted in new(); no further allocation or publication
        // occurs while the existing native sink is still staging this record.
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        if self.poisoned { return Err(self.refused()); }
        if self.bytes.is_empty() {
            if self.inner.flush().is_err() { return Err(self.refused()); }
            return Ok(());
        }
        if self.bytes.first() != Some(&b'{') || !self.bytes.ends_with(b"}\n") {
            return Err(self.refused());
        }
        // Raw inner LF is replaced by }\n, adding exactly prefix.len()+1 bytes.
        let size = (self.bytes.len() as u64).checked_add(self.prefix.len() as u64 + 1)
            .filter(|&n| n <= self.remaining).ok_or_else(|| self.refused())?;
        let body = &self.bytes[..self.bytes.len() - 1];
        if self.inner.write_all(&self.prefix).and_then(|()| self.inner.write_all(body))
            .and_then(|()| self.inner.write_all(b"}\n")).and_then(|()| self.inner.flush()).is_err() {
            return Err(self.refused());
        }
        self.remaining -= size;
        self.bytes.clear();
        Ok(())
    }
}
// No Drop flush: cancellation/unwind cannot publish an incomplete result.

#[cfg(test)] mod tests;
