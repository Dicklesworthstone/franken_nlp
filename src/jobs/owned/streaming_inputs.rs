//! Explicit retained creation with one original envelope in memory. The fresh
//! manifest binds the original stream before any staged bytes are reread; no
//! unreceipted frame can become a substitute input or authorize native work.
use super::*;
use crate::jobs::{JobLimits, MismatchField, manifest::ManifestBuilder,
    population::{PopulationReadLimits, PopulationReader, envelope_id},
    retained::RetainedInputCursor};
use std::{fs::File, io::{BufRead, Read, Seek, SeekFrom, Write}};

impl OwnedJob {
    /// The real processor's recipe is already frozen into `fresh`. Its typed
    /// validator sees every original record before the initial journal header
    /// exists. Native execution remains unavailable until the complete original
    /// population, final v1 spool and receipt have all authenticated.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::jobs) fn create_with_retained_ndjson<R, C, E>(root: &Path, key: JobSecret,
        limits: JobLimits, retention: InputRetentionLimits, reader: R,
        transport: PopulationReadLimits, fresh: ManifestBuilder, control: &mut C,
        validate: impl FnMut(&JobInput<'_>, &mut C) -> Result<(), E>)
        -> Result<(Self, RetainedInputCursor<File>), E>
    where R: BufRead, C: DecodeStepControl, E: From<JobError> {
        Self::create_with_retained_ndjson_inner(root, key, limits, retention, reader,
            transport, fresh, control, validate, None)
    }

    #[allow(clippy::too_many_arguments)]
    fn create_with_retained_ndjson_inner<R, C, E>(root: &Path, key: JobSecret,
        limits: JobLimits, retention: InputRetentionLimits, reader: R,
        transport: PopulationReadLimits, mut fresh: ManifestBuilder, control: &mut C,
        mut validate: impl FnMut(&JobInput<'_>, &mut C) -> Result<(), E>,
        fail_at: Option<CreateFault>) -> Result<(Self, RetainedInputCursor<File>), E>
    where R: BufRead, C: DecodeStepControl, E: From<JobError> {
        checkpoint(control)?;
        retention.validate()?;
        let mut source = PopulationReader::new(reader, limits, transport, control)?;
        // Reserved names are create_new under one descriptor-rooted lock.
        // Failed staging is an orphan; it is never overwritten or adopted.
        let files = JobFiles::open(root, true, limits.max_journal_bytes, limits.max_spool_bytes)?;
        let mut spool = files.create_inputs(retention.max_spool_bytes)?;
        let mut items = 0_u64;
        let mut length = 0_u64;
        while let Some(record) = source.next_record(control)? {
            let input = record.input();
            let end = staged_end(length, input.original.len(), retention)?;
            validate(&input, control)?;
            fresh.push(&key, &input, control)?;
            stage_input(&mut spool, items, &input, control)?;
            length = end;
            items = items.checked_add(1).ok_or(JobError::Limit)?;
        }
        let stats = source.stats();
        drop(source);
        let manifest = fresh.finish(&key)?;
        check_secret(&key, &manifest)?;
        if manifest.limits != limits { return Err(JobError::Mismatch(MismatchField::Limits).into()); }
        if items != stats.items || items != manifest.item_count() { return Err(JobError::Corrupt.into()); }
        // The v1 start MAC includes the FINAL population commitment. Only now
        // can placeholder headers be replaced, after checking every reread
        // payload against the original stream's keyed item binding.
        let receipt = seal_inputs(&mut spool, &key, &manifest, retention, length, control)?;
        create_fault(fail_at, CreateFault::InputSpoolWritten)?;
        if files.sync_inputs(&spool, retention.max_spool_bytes)? != receipt.spool_bytes {
            return Err(JobError::Corrupt.into());
        }
        create_fault(fail_at, CreateFault::InputSpoolSynced)?;
        checkpoint(control)?;
        drop(spool);

        // Read back the complete durable v1 stream before publishing the
        // receipt, then reuse that read-only cursor for execution from zero.
        // Declaration order keeps it ahead of JobFiles during error cleanup.
        let file = files.retained_inputs(retention.max_spool_bytes)?.ok_or(JobError::Corrupt)?;
        let mut cursor = RetainedInputCursor::new(file, &key, &manifest, &receipt, control)?;
        let beginning = cursor.bookmark()?;
        while cursor.read_next(&key, &manifest, control)?.is_some() {}
        cursor.replay_from(beginning, control)?;
        checkpoint(control)?;
        let journal = Journal::open(&files.database_path(), limits.max_journal_bytes)?;
        let header = Self::initialize(&journal, &files, &key, &manifest, Some(receipt), control, fail_at)?;
        let job = Self { journal, files, key, manifest, header, poisoned: false, #[cfg(test)] fail_at: None };
        Ok((job, cursor))
    }
}

fn staged_end(offset: u64, bytes: usize, retention: InputRetentionLimits) -> Result<u64, JobError> {
    if bytes == 0 { return Err(JobError::InvalidInput); }
    offset.checked_add(retained::HEADER_BYTES as u64)
        .and_then(|n| n.checked_add(bytes as u64))
        .filter(|&n| n <= retention.max_spool_bytes).ok_or(JobError::Limit)
}

/// No chain value in this temporary header is authority. The eventual seal
/// must reread and compare the original manifest before replacing these zeros.
fn stage_input<W: Write, C: DecodeStepControl>(writer: &mut W, ordinal: u64,
    input: &JobInput<'_>, control: &mut C) -> Result<(), JobError> {
    checkpoint(control)?;
    let mut header = [0_u8; retained::HEADER_BYTES];
    header[..8].copy_from_slice(retained::MAGIC);
    header[8..16].copy_from_slice(&ordinal.to_le_bytes());
    header[16..24].copy_from_slice(&(input.original.len() as u64).to_le_bytes());
    writer.write_all(&header).map_err(|_| JobError::Io)?;
    for chunk in input.original.chunks(8192) {
        checkpoint(control)?;
        writer.write_all(chunk).map_err(|_| JobError::Io)?;
    }
    Ok(())
}

fn seal_inputs<W: Read + Write + Seek, C: DecodeStepControl>(writer: &mut W,
    key: &JobSecret, manifest: &FrozenManifest, retention: InputRetentionLimits,
    expected_bytes: u64, control: &mut C) -> Result<InputReceipt, JobError> {
    checkpoint(control)?;
    retention.validate()?;
    if expected_bytes > retention.max_spool_bytes
        || writer.seek(SeekFrom::End(0)).map_err(|_| JobError::Io)? != expected_bytes {
        return Err(JobError::Corrupt);
    }
    let mut offset = 0_u64;
    let mut chain = retained::start(key, &manifest.binding, retention.max_spool_bytes);
    let mut bytes = Vec::new();
    for ordinal in 0..manifest.item_count() {
        checkpoint(control)?;
        writer.seek(SeekFrom::Start(offset)).map_err(|_| JobError::Io)?;
        let mut header = [0_u8; retained::HEADER_BYTES];
        writer.read_exact(&mut header).map_err(|_| JobError::Corrupt)?;
        let length = u64::from_le_bytes(header[16..24].try_into().map_err(|_| JobError::Corrupt)?);
        let end = offset.checked_add(retained::HEADER_BYTES as u64).and_then(|n| n.checked_add(length))
            .filter(|&n| n <= expected_bytes).ok_or(JobError::Corrupt)?;
        if &header[..8] != retained::MAGIC || header[8..16] != ordinal.to_le_bytes()
            || header[24..] != [0; 32] || length == 0 || length > manifest.limits.max_input_bytes_per_item as u64 {
            return Err(JobError::Corrupt);
        }
        let length = usize::try_from(length).map_err(|_| JobError::Limit)?;
        bytes.try_reserve_exact(length.saturating_sub(bytes.len())).map_err(|_| JobError::Allocation)?;
        bytes.resize(length, 0);
        for chunk in bytes.chunks_mut(8192) {
            checkpoint(control)?;
            writer.read_exact(chunk).map_err(|_| JobError::Corrupt)?;
        }
        let id = envelope_id(&bytes, manifest.limits)?;
        manifest.verify_input(key, ordinal, &JobInput { id: &id, original: &bytes, normalized: &bytes })?;
        chain = retained::next(key, chain, ordinal, offset, &bytes);
        checkpoint(control)?;
        writer.seek(SeekFrom::Start(offset.checked_add(24).ok_or(JobError::Limit)?)).map_err(|_| JobError::Io)?;
        writer.write_all(&chain.0).map_err(|_| JobError::Io)?;
        offset = end;
    }
    if offset != expected_bytes || writer.seek(SeekFrom::End(0)).map_err(|_| JobError::Io)? != expected_bytes {
        return Err(JobError::Corrupt);
    }
    checkpoint(control)?;
    let receipt = InputReceipt { schema_version: 1, items: manifest.item_count(), spool_bytes: offset,
        max_spool_bytes: retention.max_spool_bytes,
        commitment: retained::finish(key, chain, manifest.item_count(), offset) };
    receipt.validate(manifest.limits, manifest.item_count())?;
    Ok(receipt)
}

#[cfg(test)] mod tests;
