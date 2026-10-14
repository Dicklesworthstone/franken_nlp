//! Bounded persistence of the complete source-candidate model, not FNLPQ.
//!
//! An externally retained content pin is mandatory. Header checksums are NOT
//! authenticity. The reader rebuilds the existing role/shape/source/recipe-bound
//! model digest from every restored value before returning executable weights.
//! No source lookup, conversion, native packing, fallback, or activation occurs.
use super::*;
use std::io::{self, Read, Write};

pub const INT4_SNAPSHOT_VERSION: &str = "fnlp-int4-local-snapshot-v1";
pub const SNAPSHOT_HEADER_BYTES: usize = 128;
pub const SNAPSHOT_SCRATCH_BYTES: usize = 64 * 1024;
const MAGIC: &[u8; 8] = b"FNLPI4S\0";

/// Retain this separately from the untrusted snapshot. A caller can supply a
/// pin, not a model identity: the latter is reconstructed and checked in full.
/// Pinning verifies integrity, NOT publisher authentication or source licensing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4SnapshotPin { pub group_width: usize, pub model_content_sha256: [u8; 32] }
impl Int4ModelIdentity {
    pub fn snapshot_pin(&self) -> Int4SnapshotPin {
        Int4SnapshotPin { group_width: self.group_width(), model_content_sha256: self.content_sha256() }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4SnapshotRequirements {
    pub weight_payload_bytes: u64,
    pub scratch_bytes: u64,
    pub file_bytes: u64,
}
impl Int4SnapshotRequirements {
    pub fn for_group_width(width: usize) -> Result<Self, Int4SnapshotError> {
        let model = Int4ModelLoadRequirement::for_group_width(width).map_err(model_error)?;
        Ok(Self { weight_payload_bytes: model.weight_payload_bytes, scratch_bytes: SNAPSHOT_SCRATCH_BYTES as u64,
            file_bytes: model.weight_payload_bytes.checked_add(SNAPSHOT_HEADER_BYTES as u64).ok_or(Int4SnapshotError::Budget)? })
    }
    /// Includes the one-byte trailing-data probe. Valid files supply zero
    /// bytes for that read; an invalid stream cannot cause an unbounded drain.
    pub fn read_io_bytes(self) -> Result<u64, Int4SnapshotError> {
        self.file_bytes.checked_add(1).ok_or(Int4SnapshotError::Budget)
    }
    pub fn check(self, budget: Int4SnapshotBudget, reading: bool) -> Result<(), Int4SnapshotError> {
        let io = if reading { self.read_io_bytes()? } else { self.file_bytes };
        if budget.max_weight_bytes < self.weight_payload_bytes || budget.max_scratch_bytes < self.scratch_bytes
            || budget.max_io_bytes < io { return Err(Int4SnapshotError::Budget); }
        Ok(())
    }
}
/// Whole-model payload and finite IO bounds, not allocator or process RSS caps.
/// The host separately prices metadata, runtime and allocator slack. The same
/// owned weights are priced on save, not allocated/charged a second time.
#[derive(Clone, Copy, Debug)]
pub struct Int4SnapshotBudget { pub max_weight_bytes: u64, pub max_scratch_bytes: u64, pub max_io_bytes: u64 }
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Int4SnapshotError { Format, Integrity, Values, Budget, Allocation, Io, Cancelled(DecodeCancellationKind) }
impl fmt::Display for Int4SnapshotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Format => "INT4 local snapshot format or geometry refused",
            Self::Integrity => "INT4 local snapshot differs from its external model-content pin",
            Self::Values => "INT4 local snapshot contains values outside its recipe",
            Self::Budget => "INT4 local snapshot payload or IO budget exceeded",
            Self::Allocation => "INT4 local snapshot allocation refused",
            Self::Io => "INT4 local snapshot IO failed",
            Self::Cancelled(_) => "INT4 local snapshot cancelled",
        })
    }
}
impl Error for Int4SnapshotError {}
fn model_error(error: Int4ModelError) -> Int4SnapshotError {
    match error {
        Int4ModelError::Cancelled(cause) => Int4SnapshotError::Cancelled(cause),
        Int4ModelError::Budget => Int4SnapshotError::Budget,
        Int4ModelError::Allocation => Int4SnapshotError::Allocation,
        Int4ModelError::NonFinite => Int4SnapshotError::Values,
        _ => Int4SnapshotError::Format,
    }
}
fn poll<C: DecodeStepControl>(control: &mut C) -> Result<(), Int4SnapshotError> {
    checkpoint(control).map_err(model_error)
}

impl Int4ModelWeights {
    /// Writes the exact compact model once, with one reusable 64-KiB buffer.
    /// The caller owns an exclusive destination, filesystem durability and
    /// publication. An error can leave partial bytes; never publish them as a
    /// completed snapshot. flush() is NOT fsync, rename or release activation.
    pub fn write_snapshot<W: Write, C: DecodeStepControl>(&self, target: &mut W,
        budget: Int4SnapshotBudget, control: &mut C) -> Result<Int4SnapshotPin, Int4SnapshotError> {
        let need = Int4SnapshotRequirements::for_group_width(self.group_width())?;
        need.check(budget, false)?;
        poll(control)?;
        let mut scratch = buffer()?;
        let pin = self.identity().snapshot_pin();
        let header = encode_header(pin, self.identity().source_root_sha256(), need)?;
        let mut target = WireWriter { target, remaining: need.file_bytes };
        write_bytes(&mut target, &header, control)?;
        for row in 0..V {
            write_bf16(&mut target, self.embeddings.row(row).map_err(|_| Int4SnapshotError::Format)?, &mut scratch, control)?;
        }
        write_bf16(&mut target, &self.final_norm, &mut scratch, control)?;
        write_matrix(&mut target, &self.head, &mut scratch, control)?;
        for layer in &self.layers {
            write_bf16(&mut target, &layer.norm1, &mut scratch, control)?;
            write_bf16(&mut target, &layer.norm2, &mut scratch, control)?;
            for matrix in [&layer.q, &layer.k, &layer.v, &layer.o, &layer.gate, &layer.up, &layer.down] {
                write_matrix(&mut target, matrix, &mut scratch, control)?;
            }
        }
        if target.remaining != 0 { return Err(Int4SnapshotError::Format); }
        poll(control)?;
        target.flush().map_err(|_| Int4SnapshotError::Io)?;
        poll(control)?;
        Ok(pin)
    }

    /// All budgets and pin geometry are checked BEFORE the first read or model
    /// allocation. File-controlled lengths never set an allocation size. The
    /// body is read directly into its final compact vectors; no BF16 source
    /// model, dequantized matrix, whole-file buffer or second weight copy exists.
    /// The restored source root is provenance only, not independently verified
    /// source possession. Exact EOF and the existing full model digest are both
    /// required. Cancellation/failure drops the entire incomplete model.
    pub fn read_snapshot<R: Read, C: DecodeStepControl>(source: &mut R, pin: Int4SnapshotPin,
        budget: Int4SnapshotBudget, control: &mut C) -> Result<Self, Int4SnapshotError> {
        let need = Int4SnapshotRequirements::for_group_width(pin.group_width)?;
        need.check(budget, true)?;
        poll(control)?;
        let mut input = source.take(need.file_bytes);
        let mut header = [0_u8; SNAPSHOT_HEADER_BYTES];
        read_bytes(&mut input, &mut header, control)?;
        let source_root_sha256 = decode_header(&header, pin, need)?;
        let mut scratch = buffer()?;
        let embeddings = Bf16Matrix::new(V, H, read_bf16_values(&mut input, V * H, &mut scratch, control)?)
            .map_err(|_| Int4SnapshotError::Format)?;
        let final_norm = read_bf16_values(&mut input, H, &mut scratch, control)?;
        let head = read_matrix(&mut input, V, H, pin.group_width, &mut scratch, control)?;
        let mut layers = Vec::new();
        layers.try_reserve_exact(PHYSICAL_LAYER_COUNT).map_err(|_| Int4SnapshotError::Allocation)?;
        for _ in 0..PHYSICAL_LAYER_COUNT {
            layers.push(Int4Layer {
                norm1: read_bf16_values(&mut input, H, &mut scratch, control)?,
                norm2: read_bf16_values(&mut input, H, &mut scratch, control)?,
                q: read_matrix(&mut input, Q, H, pin.group_width, &mut scratch, control)?,
                k: read_matrix(&mut input, K, H, pin.group_width, &mut scratch, control)?,
                v: read_matrix(&mut input, K, H, pin.group_width, &mut scratch, control)?,
                o: read_matrix(&mut input, H, Q, pin.group_width, &mut scratch, control)?,
                gate: read_matrix(&mut input, I, H, pin.group_width, &mut scratch, control)?,
                up: read_matrix(&mut input, I, H, pin.group_width, &mut scratch, control)?,
                down: read_matrix(&mut input, H, I, pin.group_width, &mut scratch, control)?,
            });
        }
        if input.limit() != 0 { return Err(Int4SnapshotError::Format); }
        let source = input.into_inner();
        require_eof(source, control)?;
        drop(scratch);
        let mut model = Self { embeddings, final_norm, head,
            layers: layers.try_into().map_err(|_| Int4SnapshotError::Format)?,
            identity: Int4ModelIdentity { source_root_sha256, content_sha256: [0; 32], group_width: pin.group_width } };
        let actual = model.digest(control).map_err(model_error)?;
        verify_content(actual, pin)?;
        model.identity.content_sha256 = actual;
        poll(control)?;
        Ok(model)
    }
}
fn verify_content(actual: [u8; 32], pin: Int4SnapshotPin) -> Result<(), Int4SnapshotError> {
    if actual == pin.model_content_sha256 { Ok(()) } else { Err(Int4SnapshotError::Integrity) }
}
fn buffer() -> Result<Vec<u8>, Int4SnapshotError> {
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(SNAPSHOT_SCRATCH_BYTES).map_err(|_| Int4SnapshotError::Allocation)?;
    bytes.resize(SNAPSHOT_SCRATCH_BYTES, 0); Ok(bytes)
}
fn encode_header(pin: Int4SnapshotPin, root: &str, need: Int4SnapshotRequirements)
    -> Result<[u8; SNAPSHOT_HEADER_BYTES], Int4SnapshotError> {
    let mut bytes = [0; SNAPSHOT_HEADER_BYTES];
    bytes[..8].copy_from_slice(MAGIC);
    bytes[8..12].copy_from_slice(&1_u32.to_le_bytes());
    bytes[16..24].copy_from_slice(&(pin.group_width as u64).to_le_bytes());
    bytes[24..32].copy_from_slice(&need.weight_payload_bytes.to_le_bytes());
    if root.len() != 64 || !root.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        return Err(Int4SnapshotError::Format);
    }
    for (i, out) in bytes[32..64].iter_mut().enumerate() {
        *out = u8::from_str_radix(&root[i * 2..i * 2 + 2], 16).map_err(|_| Int4SnapshotError::Format)?;
    }
    bytes[64..96].copy_from_slice(&pin.model_content_sha256);
    Ok(bytes)
}
fn decode_header(bytes: &[u8; SNAPSHOT_HEADER_BYTES], pin: Int4SnapshotPin, need: Int4SnapshotRequirements)
    -> Result<String, Int4SnapshotError> {
    if &bytes[..8] != MAGIC || bytes[8..12] != 1_u32.to_le_bytes()
        || bytes[12..16].iter().chain(&bytes[96..]).any(|&b| b != 0)
        || bytes[16..24] != (pin.group_width as u64).to_le_bytes()
        || bytes[24..32] != need.weight_payload_bytes.to_le_bytes() {
        return Err(Int4SnapshotError::Format);
    }
    if bytes[64..96] != pin.model_content_sha256 { return Err(Int4SnapshotError::Integrity); }
    let mut root = String::new();
    root.try_reserve_exact(64).map_err(|_| Int4SnapshotError::Allocation)?;
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for &byte in &bytes[32..64] {
        root.push(char::from(HEX[usize::from(byte >> 4)])); root.push(char::from(HEX[usize::from(byte & 15)]));
    }
    Ok(root)
}
fn read_bytes<R: Read, C: DecodeStepControl>(r: &mut R, bytes: &mut [u8], c: &mut C) -> Result<(), Int4SnapshotError> {
    for chunk in bytes.chunks_mut(SNAPSHOT_SCRATCH_BYTES) {
        let mut offset = 0;
        while offset < chunk.len() {
            poll(c)?;
            match r.read(&mut chunk[offset..]) {
                Ok(0) => return Err(Int4SnapshotError::Io),
                Ok(n) if n <= chunk.len() - offset => offset += n,
                Ok(_) => return Err(Int4SnapshotError::Io),
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {},
                Err(_) => return Err(Int4SnapshotError::Io),
            }
        }
        poll(c)?;
    }
    Ok(())
}
fn write_bytes<W: Write, C: DecodeStepControl>(w: &mut W, bytes: &[u8], c: &mut C) -> Result<(), Int4SnapshotError> {
    for chunk in bytes.chunks(SNAPSHOT_SCRATCH_BYTES) {
        let mut offset = 0;
        while offset < chunk.len() {
            poll(c)?;
            match w.write(&chunk[offset..]) {
                Ok(0) => return Err(Int4SnapshotError::Io),
                Ok(n) if n <= chunk.len() - offset => offset += n,
                Ok(_) => return Err(Int4SnapshotError::Io),
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {},
                Err(_) => return Err(Int4SnapshotError::Io),
            }
        }
        poll(c)?;
    }
    Ok(())
}
fn require_eof<R: Read, C: DecodeStepControl>(r: &mut R, c: &mut C) -> Result<(), Int4SnapshotError> {
    loop {
        poll(c)?;
        match r.read(&mut [0_u8]) {
            Ok(0) => { poll(c)?; return Ok(()); }
            Ok(_) => return Err(Int4SnapshotError::Format),
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {},
            Err(_) => return Err(Int4SnapshotError::Io),
        }
    }
}
fn write_bf16<W: Write, C: DecodeStepControl>(w: &mut W, values: &[Bf16], scratch: &mut [u8], c: &mut C)
    -> Result<(), Int4SnapshotError> {
    for chunk in values.chunks(SNAPSHOT_SCRATCH_BYTES / 2) {
        poll(c)?;
        for (v, bytes) in chunk.iter().zip(scratch.chunks_exact_mut(2)) { bytes.copy_from_slice(&v.to_bits().to_le_bytes()); }
        write_bytes(w, &scratch[..chunk.len() * 2], c)?;
    }
    Ok(())
}
fn read_bf16_values<R: Read, C: DecodeStepControl>(r: &mut R, count: usize, scratch: &mut [u8], c: &mut C)
    -> Result<Vec<Bf16>, Int4SnapshotError> {
    poll(c)?;
    let mut values = Vec::new();
    values.try_reserve_exact(count).map_err(|_| Int4SnapshotError::Allocation)?;
    while values.len() < count {
        let n = (count - values.len()).min(SNAPSHOT_SCRATCH_BYTES / 2);
        read_bytes(r, &mut scratch[..n * 2], c)?;
        for bytes in scratch[..n * 2].chunks_exact(2) {
            let value = Bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]]));
            if !value.to_f32().is_finite() { return Err(Int4SnapshotError::Values); }
            values.push(value);
        }
    }
    Ok(values)
}
fn write_matrix<W: Write, C: DecodeStepControl>(w: &mut W, matrix: &GroupedInt4Matrix,
    scratch: &mut [u8], c: &mut C) -> Result<(), Int4SnapshotError> {
    write_bytes(w, matrix.packed_bytes(), c)?;
    for chunk in matrix.group_scales().chunks(SNAPSHOT_SCRATCH_BYTES / 4) {
        poll(c)?;
        for (scale, bytes) in chunk.iter().zip(scratch.chunks_exact_mut(4)) { bytes.copy_from_slice(&scale.to_bits().to_le_bytes()); }
        write_bytes(w, &scratch[..chunk.len() * 4], c)?;
    }
    Ok(())
}
fn read_matrix<R: Read, C: DecodeStepControl>(r: &mut R, rows: usize, columns: usize, width: usize,
    scratch: &mut [u8], c: &mut C) -> Result<GroupedInt4Matrix, Int4SnapshotError> {
    let need = Int4ConversionRequirement::for_shape(rows, columns, width).map_err(|_| Int4SnapshotError::Format)?;
    let bytes = usize::try_from(need.packed_bytes).map_err(|_| Int4SnapshotError::Budget)?;
    let groups = usize::try_from(need.groups).map_err(|_| Int4SnapshotError::Budget)?;
    poll(c)?;
    let mut packed = Vec::new();
    packed.try_reserve_exact(bytes).map_err(|_| Int4SnapshotError::Allocation)?;
    while packed.len() < bytes {
        poll(c)?;
        let start = packed.len(); let n = (bytes - start).min(SNAPSHOT_SCRATCH_BYTES);
        packed.resize(start + n, 0);
        read_bytes(r, &mut packed[start..], c)?;
    }
    let mut scales = Vec::new();
    scales.try_reserve_exact(groups).map_err(|_| Int4SnapshotError::Allocation)?;
    while scales.len() < groups {
        let n = (groups - scales.len()).min(SNAPSHOT_SCRATCH_BYTES / 4);
        read_bytes(r, &mut scratch[..n * 4], c)?;
        for bytes in scratch[..n * 4].chunks_exact(4) {
            scales.push(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));
        }
    }
    GroupedInt4Matrix::restore_packed(rows, columns, width, packed, scales, c).map_err(|error| match error {
        Int4ConversionError::Cancelled(cause) => Int4SnapshotError::Cancelled(cause),
        _ => Int4SnapshotError::Values,
    })
}
struct WireWriter<'a, W> { target: &'a mut W, remaining: u64 }
impl<W: Write> Write for WireWriter<'_, W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() as u64 > self.remaining { return Err(io::Error::other("snapshot byte budget")); }
        let n = self.target.write(bytes)?;
        if n > bytes.len() { return Err(io::Error::other("invalid snapshot writer length")); }
        self.remaining -= n as u64; Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> { self.target.flush() }
}

#[cfg(test)] mod tests;
