//! Deterministic release-package construction and verification.
//!
//! This module deliberately handles every large payload as a bounded stream.
//! It never loads a part, much less a complete model artifact, into memory.

use std::{
    collections::BTreeSet,
    error::Error,
    fmt,
    fs::{self, File, OpenOptions},
    io::{self, BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Exact v1 release part size.  Only the final part may be shorter.
pub const RELEASE_PART_BYTES: u64 = 1_957_046_720;
/// A v1 package uses two-digit part suffixes and therefore permits 64 parts.
pub const MAX_RELEASE_PARTS: usize = 64;

const COPY_BUFFER_BYTES: usize = 1024 * 1024;
const RECEIPT_FILE: &str = "MODEL_ASSET_RECEIPT.json";
const SUMS_FILE: &str = "SHA256SUMS";
const RECONSTRUCTION_FILE: &str = "RECONSTRUCTION.txt";
const CONVERSION_RECEIPT_FILE: &str = "CONVERSION_RECEIPT.json";
const RECEIPT_SCHEMA: &str = "fnlp-model-asset-receipt-v1";
const LICENSE_FILES: [&str; 3] = [
    "APACHE-2.0.txt",
    "ATTRIBUTION.txt",
    "MODIFICATION_NOTICE.txt",
];

/// Input paths and authority names for one immutable package staging run.
#[derive(Clone, Debug)]
pub struct PackageRequest {
    /// Finished canonical Generic `.fnlpq` file to split.
    pub artifact: PathBuf,
    /// New package directory.  It must not already exist.
    pub staging_dir: PathBuf,
    /// Versioned logical artifact name, without a directory component.
    pub logical_artifact_name: String,
    /// Converter receipt copied into the release closure.
    pub conversion_receipt: PathBuf,
    /// Directory containing the three approved release-license files.
    pub license_bundle_dir: PathBuf,
}

/// A successful package write, suitable for maintainer logs and a later upload.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackageReport {
    /// Staging directory created with `create_dir`, never reused or overwritten.
    pub staging_dir: PathBuf,
    /// Digest of the exact original `.fnlpq` byte stream.
    pub artifact_sha256: String,
    /// Exact original `.fnlpq` length.
    pub artifact_bytes: u64,
    /// Ordered part records in manifest order.
    pub parts: Vec<PackageFile>,
}

/// One name/length/digest member of the release closure.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PackageFile {
    /// Basename inside the staging directory.
    pub name: String,
    /// Exact byte length.
    pub bytes: u64,
    /// Lowercase SHA-256 of the exact stored bytes.
    pub sha256: String,
}

/// The fixed, self-contained model release receipt.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelAssetReceipt {
    /// Closed receipt schema identifier.
    pub receipt_schema: String,
    /// Versioned logical name reconstructed by part concatenation.
    pub logical_artifact_name: String,
    /// Exact original artifact length.
    pub artifact_bytes: u64,
    /// SHA-256 of the exact original artifact byte stream.
    pub fnlpq_file_sha256: String,
    /// Exact fixed v1 part size.
    pub part_bytes: u64,
    /// Ordered part records; their listed order is the reconstruction order.
    pub parts: Vec<PackageFile>,
    /// Converter receipt bound into the release closure.
    pub conversion_receipt: PackageFile,
    /// Exact Apache license, attribution, and modification-notice records.
    pub license_bundle: Vec<PackageFile>,
    /// Deterministic reconstruction instructions.
    pub reconstruction: PackageFile,
    /// Version of the splitting protocol, not a release timestamp.
    pub split_tool: String,
}

/// Typed package failures.  All rejection text includes the affected member.
#[derive(Debug)]
pub enum PackageError {
    /// A caller supplied an invalid authority name or path arrangement.
    InvalidInput { field: &'static str, detail: String },
    /// Creating an immutable staging directory would overwrite an existing path.
    StagingExists(PathBuf),
    /// Streaming I/O failed at a named path.
    Io {
        operation: &'static str,
        path: PathBuf,
        source: io::Error,
    },
    /// A source changed while its digest-bound bytes were being copied.
    SourceChanged { path: PathBuf },
    /// A receipt, digest, part order, or package inventory invariant failed.
    Integrity { member: String, detail: String },
    /// JSON serialization/parsing of the small bounded receipt failed.
    ReceiptJson(String),
    /// Checked package arithmetic failed.
    Arithmetic(&'static str),
}

impl fmt::Display for PackageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput { field, detail } => write!(formatter, "invalid {field}: {detail}"),
            Self::StagingExists(path) => write!(
                formatter,
                "staging directory already exists: {}",
                path.display()
            ),
            Self::Io {
                operation,
                path,
                source,
            } => write!(formatter, "{operation} {}: {source}", path.display()),
            Self::SourceChanged { path } => write!(
                formatter,
                "source changed while digest-bound bytes were copied: {}",
                path.display()
            ),
            Self::Integrity { member, detail } => {
                write!(
                    formatter,
                    "release package integrity failure for {member}: {detail}"
                )
            }
            Self::ReceiptJson(detail) => write!(formatter, "release receipt JSON: {detail}"),
            Self::Arithmetic(detail) => {
                write!(formatter, "release package checked arithmetic: {detail}")
            }
        }
    }
}

impl Error for PackageError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Split a canonical artifact into fixed v1 parts and write its immutable closure.
///
/// `staging_dir` is created with `create_dir`, so this operation never replaces a
/// previous package.  Errors leave the incomplete directory available for human
/// inspection rather than deleting evidence.
pub fn package_model(request: &PackageRequest) -> Result<PackageReport, PackageError> {
    package_model_with_part_bytes(request, RELEASE_PART_BYTES)
}

/// Verify every retained package member and stream-concatenate parts into a
/// whole-file hash.  The reconstructed hash must equal the receipt's original
/// converter-output digest.
pub fn verify_model_package(staging_dir: &Path) -> Result<PackageReport, PackageError> {
    let receipt_path = staging_dir.join(RECEIPT_FILE);
    let receipt: ModelAssetReceipt = parse_receipt(&receipt_path)?;
    validate_receipt_shape(&receipt)?;

    let mut expected_names = BTreeSet::new();
    let mut ordered_records = Vec::new();
    for record in &receipt.parts {
        insert_unique_name(&mut expected_names, record)?;
        ordered_records.push(record);
    }
    for record in std::iter::once(&receipt.conversion_receipt)
        .chain(receipt.license_bundle.iter())
        .chain(std::iter::once(&receipt.reconstruction))
    {
        insert_unique_name(&mut expected_names, record)?;
        ordered_records.push(record);
    }
    expected_names.insert(RECEIPT_FILE.to_owned());
    expected_names.insert(SUMS_FILE.to_owned());
    ensure_exact_inventory(staging_dir, &expected_names)?;

    for record in &ordered_records {
        verify_file_record(staging_dir, record)?;
    }
    verify_sha256sums(staging_dir, &ordered_records)?;

    let mut whole_hasher = Sha256::new();
    let mut reconstructed_bytes = 0_u64;
    for record in &receipt.parts {
        stream_hash_into(
            &staging_dir.join(&record.name),
            &mut whole_hasher,
            &mut reconstructed_bytes,
        )?;
    }
    if reconstructed_bytes != receipt.artifact_bytes {
        return Err(PackageError::Integrity {
            member: "reconstruction".to_owned(),
            detail: format!(
                "byte count mismatch expected={} observed={reconstructed_bytes}",
                receipt.artifact_bytes
            ),
        });
    }
    let observed_whole = hex_digest(whole_hasher.finalize());
    if observed_whole != receipt.fnlpq_file_sha256 {
        return Err(PackageError::Integrity {
            member: "reconstruction".to_owned(),
            detail: format!(
                "whole digest mismatch expected={} observed={observed_whole}",
                receipt.fnlpq_file_sha256
            ),
        });
    }

    Ok(PackageReport {
        staging_dir: staging_dir.to_owned(),
        artifact_sha256: receipt.fnlpq_file_sha256,
        artifact_bytes: receipt.artifact_bytes,
        parts: receipt.parts,
    })
}

fn package_model_with_part_bytes(
    request: &PackageRequest,
    part_bytes: u64,
) -> Result<PackageReport, PackageError> {
    validate_logical_name(&request.logical_artifact_name)?;
    if part_bytes == 0 || part_bytes > RELEASE_PART_BYTES {
        return Err(PackageError::InvalidInput {
            field: "part_bytes",
            detail: format!("must be in 1..={RELEASE_PART_BYTES}, found {part_bytes}"),
        });
    }
    require_regular_file(&request.artifact)?;
    require_regular_file(&request.conversion_receipt)?;
    require_license_bundle(&request.license_bundle_dir)?;
    create_staging_dir(&request.staging_dir)?;

    let (artifact_bytes, parts, artifact_sha256) = split_artifact(
        &request.artifact,
        &request.staging_dir,
        &request.logical_artifact_name,
        part_bytes,
    )?;
    let conversion_receipt = copy_digest_bound_file(
        &request.conversion_receipt,
        &request.staging_dir.join(CONVERSION_RECEIPT_FILE),
    )?;
    let mut license_bundle = Vec::with_capacity(LICENSE_FILES.len());
    for name in LICENSE_FILES {
        license_bundle.push(copy_digest_bound_file(
            &request.license_bundle_dir.join(name),
            &request.staging_dir.join(name),
        )?);
    }
    let reconstruction = write_new_file(
        &request.staging_dir.join(RECONSTRUCTION_FILE),
        reconstruction_text(
            &request.logical_artifact_name,
            &parts,
            artifact_bytes,
            &artifact_sha256,
        )
        .as_bytes(),
    )?;
    let receipt = ModelAssetReceipt {
        receipt_schema: RECEIPT_SCHEMA.to_owned(),
        logical_artifact_name: request.logical_artifact_name.clone(),
        artifact_bytes,
        fnlpq_file_sha256: artifact_sha256.clone(),
        part_bytes,
        parts: parts.clone(),
        conversion_receipt,
        license_bundle,
        reconstruction,
        split_tool: format!("franken_nlp-{}", env!("CARGO_PKG_VERSION")),
    };
    let receipt_bytes = serde_json::to_vec_pretty(&receipt)
        .map_err(|error| PackageError::ReceiptJson(error.to_string()))?;
    let receipt_record = write_new_file(&request.staging_dir.join(RECEIPT_FILE), &receipt_bytes)?;
    let mut sum_records = receipt.parts.clone();
    sum_records.push(receipt.conversion_receipt.clone());
    sum_records.extend(receipt.license_bundle.clone());
    sum_records.push(receipt.reconstruction.clone());
    sum_records.push(receipt_record);
    write_sha256sums(&request.staging_dir.join(SUMS_FILE), &sum_records)?;

    verify_model_package(&request.staging_dir)
}

fn split_artifact(
    source_path: &Path,
    staging_dir: &Path,
    logical_name: &str,
    part_bytes: u64,
) -> Result<(u64, Vec<PackageFile>, String), PackageError> {
    let metadata = metadata_regular_file(source_path)?;
    let artifact_bytes = metadata.len();
    let expected_parts = expected_part_count(artifact_bytes, part_bytes)?;
    if expected_parts > MAX_RELEASE_PARTS {
        return Err(PackageError::Integrity {
            member: source_path.display().to_string(),
            detail: format!("part count {expected_parts} exceeds v1 cap {MAX_RELEASE_PARTS}"),
        });
    }

    let source = File::open(source_path).map_err(|source| PackageError::Io {
        operation: "open source artifact",
        path: source_path.to_owned(),
        source,
    })?;
    let mut reader = BufReader::with_capacity(COPY_BUFFER_BYTES, source);
    let mut whole_hasher = Sha256::new();
    let mut buffer = [0_u8; COPY_BUFFER_BYTES];
    let mut parts = Vec::with_capacity(expected_parts);
    let mut remaining_file = artifact_bytes;

    for index in 0..expected_parts {
        let part_len = remaining_file.min(part_bytes);
        let name = part_name(logical_name, index)?;
        let path = staging_dir.join(&name);
        let destination = create_new_file(&path)?;
        let mut writer = BufWriter::with_capacity(COPY_BUFFER_BYTES, destination);
        let mut part_hasher = Sha256::new();
        let mut remaining_part = part_len;
        while remaining_part > 0 {
            let next = remaining_part.min(COPY_BUFFER_BYTES as u64);
            let next =
                usize::try_from(next).map_err(|_| PackageError::Arithmetic("buffer read"))?;
            reader
                .read_exact(&mut buffer[..next])
                .map_err(|source| PackageError::Io {
                    operation: "read source artifact",
                    path: source_path.to_owned(),
                    source,
                })?;
            writer
                .write_all(&buffer[..next])
                .map_err(|source| PackageError::Io {
                    operation: "write package part",
                    path: path.clone(),
                    source,
                })?;
            whole_hasher.update(&buffer[..next]);
            part_hasher.update(&buffer[..next]);
            remaining_part = remaining_part
                .checked_sub(
                    u64::try_from(next).map_err(|_| PackageError::Arithmetic("buffer length"))?,
                )
                .ok_or(PackageError::Arithmetic("part remaining"))?;
        }
        writer.flush().map_err(|source| PackageError::Io {
            operation: "flush package part",
            path: path.clone(),
            source,
        })?;
        parts.push(PackageFile {
            name,
            bytes: part_len,
            sha256: hex_digest(part_hasher.finalize()),
        });
        remaining_file = remaining_file
            .checked_sub(part_len)
            .ok_or(PackageError::Arithmetic("artifact remaining"))?;
    }
    let mut trailing = [0_u8; 1];
    if reader
        .read(&mut trailing)
        .map_err(|source| PackageError::Io {
            operation: "check source artifact length",
            path: source_path.to_owned(),
            source,
        })?
        != 0
        || metadata_regular_file(source_path)?.len() != artifact_bytes
    {
        return Err(PackageError::SourceChanged {
            path: source_path.to_owned(),
        });
    }
    Ok((artifact_bytes, parts, hex_digest(whole_hasher.finalize())))
}

fn copy_digest_bound_file(
    source_path: &Path,
    destination_path: &Path,
) -> Result<PackageFile, PackageError> {
    let expected = hash_regular_file(source_path)?;
    let copied = copy_regular_file(source_path, destination_path)?;
    if copied.bytes != expected.bytes || copied.sha256 != expected.sha256 {
        return Err(PackageError::SourceChanged {
            path: source_path.to_owned(),
        });
    }
    Ok(copied)
}

fn copy_regular_file(
    source_path: &Path,
    destination_path: &Path,
) -> Result<PackageFile, PackageError> {
    let source_bytes = metadata_regular_file(source_path)?.len();
    let source = File::open(source_path).map_err(|source| PackageError::Io {
        operation: "open release closure source",
        path: source_path.to_owned(),
        source,
    })?;
    let destination = create_new_file(destination_path)?;
    let mut reader = BufReader::with_capacity(COPY_BUFFER_BYTES, source);
    let mut writer = BufWriter::with_capacity(COPY_BUFFER_BYTES, destination);
    let mut hasher = Sha256::new();
    let mut total = 0_u64;
    let mut buffer = [0_u8; COPY_BUFFER_BYTES];
    while total < source_bytes {
        let wanted = (source_bytes - total).min(COPY_BUFFER_BYTES as u64);
        let wanted =
            usize::try_from(wanted).map_err(|_| PackageError::Arithmetic("copy buffer"))?;
        reader
            .read_exact(&mut buffer[..wanted])
            .map_err(|source| PackageError::Io {
                operation: "read release closure source",
                path: source_path.to_owned(),
                source,
            })?;
        writer
            .write_all(&buffer[..wanted])
            .map_err(|source| PackageError::Io {
                operation: "write release closure member",
                path: destination_path.to_owned(),
                source,
            })?;
        hasher.update(&buffer[..wanted]);
        total = total
            .checked_add(
                u64::try_from(wanted).map_err(|_| PackageError::Arithmetic("copied bytes"))?,
            )
            .ok_or(PackageError::Arithmetic("copied byte count"))?;
    }
    writer.flush().map_err(|source| PackageError::Io {
        operation: "flush release closure member",
        path: destination_path.to_owned(),
        source,
    })?;
    let mut trailing = [0_u8; 1];
    if reader
        .read(&mut trailing)
        .map_err(|source| PackageError::Io {
            operation: "check release closure source length",
            path: source_path.to_owned(),
            source,
        })?
        != 0
        || metadata_regular_file(source_path)?.len() != source_bytes
    {
        return Err(PackageError::SourceChanged {
            path: source_path.to_owned(),
        });
    }
    Ok(PackageFile {
        name: file_name(destination_path)?,
        bytes: total,
        sha256: hex_digest(hasher.finalize()),
    })
}

fn hash_regular_file(path: &Path) -> Result<PackageFile, PackageError> {
    let source_bytes = metadata_regular_file(path)?.len();
    let source = File::open(path).map_err(|source| PackageError::Io {
        operation: "open digest source",
        path: path.to_owned(),
        source,
    })?;
    let mut reader = BufReader::with_capacity(COPY_BUFFER_BYTES, source);
    let mut hasher = Sha256::new();
    let mut bytes = 0_u64;
    let mut buffer = [0_u8; COPY_BUFFER_BYTES];
    while bytes < source_bytes {
        let wanted = (source_bytes - bytes).min(COPY_BUFFER_BYTES as u64);
        let wanted =
            usize::try_from(wanted).map_err(|_| PackageError::Arithmetic("hash buffer"))?;
        reader
            .read_exact(&mut buffer[..wanted])
            .map_err(|source| PackageError::Io {
                operation: "read digest source",
                path: path.to_owned(),
                source,
            })?;
        hasher.update(&buffer[..wanted]);
        bytes = bytes
            .checked_add(u64::try_from(wanted).map_err(|_| PackageError::Arithmetic("hash bytes"))?)
            .ok_or(PackageError::Arithmetic("hash byte count"))?;
    }
    let mut trailing = [0_u8; 1];
    if reader
        .read(&mut trailing)
        .map_err(|source| PackageError::Io {
            operation: "check digest source length",
            path: path.to_owned(),
            source,
        })?
        != 0
        || metadata_regular_file(path)?.len() != source_bytes
    {
        return Err(PackageError::SourceChanged {
            path: path.to_owned(),
        });
    }
    Ok(PackageFile {
        name: file_name(path)?,
        bytes,
        sha256: hex_digest(hasher.finalize()),
    })
}

fn write_new_file(path: &Path, bytes: &[u8]) -> Result<PackageFile, PackageError> {
    let file = create_new_file(path)?;
    let mut writer = BufWriter::with_capacity(COPY_BUFFER_BYTES, file);
    writer.write_all(bytes).map_err(|source| PackageError::Io {
        operation: "write package metadata",
        path: path.to_owned(),
        source,
    })?;
    writer.flush().map_err(|source| PackageError::Io {
        operation: "flush package metadata",
        path: path.to_owned(),
        source,
    })?;
    Ok(PackageFile {
        name: file_name(path)?,
        bytes: u64::try_from(bytes.len())
            .map_err(|_| PackageError::Arithmetic("metadata bytes"))?,
        sha256: hex_digest(Sha256::digest(bytes)),
    })
}

fn write_sha256sums(path: &Path, records: &[PackageFile]) -> Result<(), PackageError> {
    let mut content = String::new();
    for record in records {
        validate_record(record)?;
        content.push_str(&record.sha256);
        content.push_str("  ");
        content.push_str(&record.name);
        content.push('\n');
    }
    write_new_file(path, content.as_bytes()).map(|_| ())
}

fn verify_sha256sums(staging_dir: &Path, records: &[&PackageFile]) -> Result<(), PackageError> {
    let path = staging_dir.join(SUMS_FILE);
    let content = read_small_file(&path)?;
    let actual = std::str::from_utf8(&content).map_err(|error| PackageError::Integrity {
        member: SUMS_FILE.to_owned(),
        detail: format!("not UTF-8: {error}"),
    })?;
    let mut expected = String::new();
    for record in records {
        expected.push_str(&record.sha256);
        expected.push_str("  ");
        expected.push_str(&record.name);
        expected.push('\n');
    }
    if actual != expected {
        return Err(PackageError::Integrity {
            member: SUMS_FILE.to_owned(),
            detail: "records are missing, renamed, reordered, or digest-drifted".to_owned(),
        });
    }
    Ok(())
}

fn parse_receipt(path: &Path) -> Result<ModelAssetReceipt, PackageError> {
    serde_json::from_slice(&read_small_file(path)?)
        .map_err(|error| PackageError::ReceiptJson(error.to_string()))
}

fn validate_receipt_shape(receipt: &ModelAssetReceipt) -> Result<(), PackageError> {
    if receipt.receipt_schema != RECEIPT_SCHEMA {
        return Err(PackageError::Integrity {
            member: RECEIPT_FILE.to_owned(),
            detail: format!("unknown receipt schema {:?}", receipt.receipt_schema),
        });
    }
    validate_logical_name(&receipt.logical_artifact_name)?;
    if receipt.part_bytes != RELEASE_PART_BYTES {
        return Err(PackageError::Integrity {
            member: RECEIPT_FILE.to_owned(),
            detail: format!(
                "part_bytes must equal {RELEASE_PART_BYTES}, found {}",
                receipt.part_bytes
            ),
        });
    }
    validate_digest(&receipt.fnlpq_file_sha256, RECEIPT_FILE)?;
    let expected_parts = expected_part_count(receipt.artifact_bytes, receipt.part_bytes)?;
    if receipt.parts.len() != expected_parts || receipt.parts.len() > MAX_RELEASE_PARTS {
        return Err(PackageError::Integrity {
            member: RECEIPT_FILE.to_owned(),
            detail: format!(
                "part count expected={expected_parts} observed={}",
                receipt.parts.len()
            ),
        });
    }
    for (index, part) in receipt.parts.iter().enumerate() {
        validate_record(part)?;
        let expected_name = part_name(&receipt.logical_artifact_name, index)?;
        if part.name != expected_name {
            return Err(PackageError::Integrity {
                member: part.name.clone(),
                detail: format!("expected ordered part name {expected_name}"),
            });
        }
        let expected_len = if index + 1 == receipt.parts.len() {
            receipt.artifact_bytes.checked_sub(
                receipt
                    .part_bytes
                    .checked_mul(
                        u64::try_from(index).map_err(|_| PackageError::Arithmetic("part index"))?,
                    )
                    .ok_or(PackageError::Arithmetic("part prefix bytes"))?,
            )
        } else {
            Some(receipt.part_bytes)
        }
        .ok_or(PackageError::Arithmetic("part final bytes"))?;
        if part.bytes != expected_len || (expected_len == 0 && !receipt.parts.is_empty()) {
            return Err(PackageError::Integrity {
                member: part.name.clone(),
                detail: format!(
                    "part length expected={expected_len} observed={}",
                    part.bytes
                ),
            });
        }
    }
    validate_record(&receipt.conversion_receipt)?;
    if receipt.conversion_receipt.name != CONVERSION_RECEIPT_FILE {
        return Err(PackageError::Integrity {
            member: RECEIPT_FILE.to_owned(),
            detail: "conversion receipt name is not canonical".to_owned(),
        });
    }
    if receipt.license_bundle.len() != LICENSE_FILES.len() {
        return Err(PackageError::Integrity {
            member: RECEIPT_FILE.to_owned(),
            detail: "license bundle count is not v1 exact".to_owned(),
        });
    }
    for (record, expected_name) in receipt.license_bundle.iter().zip(LICENSE_FILES) {
        validate_record(record)?;
        if record.name != expected_name {
            return Err(PackageError::Integrity {
                member: record.name.clone(),
                detail: format!("expected exact license member {expected_name}"),
            });
        }
    }
    validate_record(&receipt.reconstruction)?;
    if receipt.reconstruction.name != RECONSTRUCTION_FILE {
        return Err(PackageError::Integrity {
            member: RECEIPT_FILE.to_owned(),
            detail: "reconstruction filename is not canonical".to_owned(),
        });
    }
    Ok(())
}

fn verify_file_record(staging_dir: &Path, record: &PackageFile) -> Result<(), PackageError> {
    let observed = hash_regular_file(&staging_dir.join(&record.name))?;
    if observed.bytes != record.bytes || observed.sha256 != record.sha256 {
        return Err(PackageError::Integrity {
            member: record.name.clone(),
            detail: format!(
                "digest/length expected={}:{} observed={}:{}",
                record.bytes, record.sha256, observed.bytes, observed.sha256
            ),
        });
    }
    Ok(())
}

fn stream_hash_into(path: &Path, hasher: &mut Sha256, total: &mut u64) -> Result<(), PackageError> {
    let file = File::open(path).map_err(|source| PackageError::Io {
        operation: "open reconstruction part",
        path: path.to_owned(),
        source,
    })?;
    let mut reader = BufReader::with_capacity(COPY_BUFFER_BYTES, file);
    let mut buffer = [0_u8; COPY_BUFFER_BYTES];
    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|source| PackageError::Io {
                operation: "read reconstruction part",
                path: path.to_owned(),
                source,
            })?;
        if read == 0 {
            return Ok(());
        }
        hasher.update(&buffer[..read]);
        *total = total
            .checked_add(
                u64::try_from(read)
                    .map_err(|_| PackageError::Arithmetic("reconstruction bytes"))?,
            )
            .ok_or(PackageError::Arithmetic("reconstruction byte count"))?;
    }
}

fn ensure_exact_inventory(
    staging_dir: &Path,
    expected: &BTreeSet<String>,
) -> Result<(), PackageError> {
    let entries = fs::read_dir(staging_dir).map_err(|source| PackageError::Io {
        operation: "read package directory",
        path: staging_dir.to_owned(),
        source,
    })?;
    let mut observed = BTreeSet::new();
    for entry in entries {
        let entry = entry.map_err(|source| PackageError::Io {
            operation: "read package directory entry",
            path: staging_dir.to_owned(),
            source,
        })?;
        let path = entry.path();
        require_regular_file(&path)?;
        observed.insert(file_name(&path)?);
    }
    if &observed != expected {
        return Err(PackageError::Integrity {
            member: "package inventory".to_owned(),
            detail: format!("expected={expected:?} observed={observed:?}"),
        });
    }
    Ok(())
}

fn require_license_bundle(directory: &Path) -> Result<(), PackageError> {
    let metadata = fs::symlink_metadata(directory).map_err(|source| PackageError::Io {
        operation: "inspect license bundle directory",
        path: directory.to_owned(),
        source,
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(PackageError::InvalidInput {
            field: "license_bundle_dir",
            detail: format!("must be a non-symlink directory: {}", directory.display()),
        });
    }
    for name in LICENSE_FILES {
        require_regular_file(&directory.join(name))?;
    }
    let expected = LICENSE_FILES
        .into_iter()
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
    let entries = fs::read_dir(directory).map_err(|source| PackageError::Io {
        operation: "read license bundle directory",
        path: directory.to_owned(),
        source,
    })?;
    let mut observed = BTreeSet::new();
    for entry in entries {
        let entry = entry.map_err(|source| PackageError::Io {
            operation: "read license bundle entry",
            path: directory.to_owned(),
            source,
        })?;
        let path = entry.path();
        require_regular_file(&path)?;
        observed.insert(file_name(&path)?);
    }
    if observed != expected {
        return Err(PackageError::InvalidInput {
            field: "license_bundle_dir",
            detail: format!("must contain exactly {expected:?}, observed {observed:?}"),
        });
    }
    Ok(())
}

fn require_regular_file(path: &Path) -> Result<(), PackageError> {
    metadata_regular_file(path).map(|_| ())
}

fn metadata_regular_file(path: &Path) -> Result<fs::Metadata, PackageError> {
    let metadata = fs::symlink_metadata(path).map_err(|source| PackageError::Io {
        operation: "inspect regular file",
        path: path.to_owned(),
        source,
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(PackageError::InvalidInput {
            field: "file",
            detail: format!("must be a non-symlink regular file: {}", path.display()),
        });
    }
    Ok(metadata)
}

fn create_staging_dir(path: &Path) -> Result<(), PackageError> {
    if path.exists() {
        return Err(PackageError::StagingExists(path.to_owned()));
    }
    fs::create_dir(path).map_err(|source| {
        if source.kind() == io::ErrorKind::AlreadyExists {
            PackageError::StagingExists(path.to_owned())
        } else {
            PackageError::Io {
                operation: "create immutable staging directory",
                path: path.to_owned(),
                source,
            }
        }
    })
}

fn create_new_file(path: &Path) -> Result<File, PackageError> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|source| PackageError::Io {
            operation: "create immutable package file",
            path: path.to_owned(),
            source,
        })
}

fn read_small_file(path: &Path) -> Result<Vec<u8>, PackageError> {
    const MAX_METADATA_BYTES: u64 = 1024 * 1024;
    let metadata = metadata_regular_file(path)?;
    if metadata.len() > MAX_METADATA_BYTES {
        return Err(PackageError::Integrity {
            member: file_name(path)?,
            detail: format!("metadata file exceeds {MAX_METADATA_BYTES} byte cap"),
        });
    }
    let file = File::open(path).map_err(|source| PackageError::Io {
        operation: "open bounded metadata",
        path: path.to_owned(),
        source,
    })?;
    let mut file = file.take(
        MAX_METADATA_BYTES
            .checked_add(1)
            .ok_or(PackageError::Arithmetic("metadata cap"))?,
    );
    let mut bytes = Vec::with_capacity(
        usize::try_from(metadata.len())
            .map_err(|_| PackageError::Arithmetic("metadata capacity"))?,
    );
    file.read_to_end(&mut bytes)
        .map_err(|source| PackageError::Io {
            operation: "read bounded metadata",
            path: path.to_owned(),
            source,
        })?;
    let observed_len =
        u64::try_from(bytes.len()).map_err(|_| PackageError::Arithmetic("metadata length"))?;
    if observed_len > MAX_METADATA_BYTES || observed_len != metadata.len() {
        return Err(PackageError::SourceChanged {
            path: path.to_owned(),
        });
    }
    Ok(bytes)
}

fn expected_part_count(bytes: u64, part_bytes: u64) -> Result<usize, PackageError> {
    if bytes == 0 {
        return Ok(0);
    }
    let count = bytes
        .checked_sub(1)
        .ok_or(PackageError::Arithmetic("part count predecessor"))?
        .checked_div(part_bytes)
        .ok_or(PackageError::Arithmetic("part count divisor"))?
        .checked_add(1)
        .ok_or(PackageError::Arithmetic("part count"))?;
    usize::try_from(count).map_err(|_| PackageError::Arithmetic("part count conversion"))
}

fn part_name(logical_name: &str, index: usize) -> Result<String, PackageError> {
    if index >= MAX_RELEASE_PARTS {
        return Err(PackageError::Integrity {
            member: logical_name.to_owned(),
            detail: format!("part index {index} exceeds v1 two-digit cap"),
        });
    }
    Ok(format!("{logical_name}.part{index:02}"))
}

fn reconstruction_text(
    logical_name: &str,
    parts: &[PackageFile],
    bytes: u64,
    sha256: &str,
) -> String {
    let part_names = parts
        .iter()
        .map(|part| part.name.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    format!(
        "fnlp release reconstruction v1\nlogical_artifact={logical_name}\nartifact_bytes={bytes}\nfnlpq_file_sha256={sha256}\npart_count={}\npart_order={part_names}\n\nConcatenate exactly the listed part_order entries, in that order, into {logical_name}.\nVerify the resulting byte count and SHA-256 against MODEL_ASSET_RECEIPT.json before use.\n",
        parts.len()
    )
}

fn validate_logical_name(name: &str) -> Result<(), PackageError> {
    if name.is_empty()
        || !name.ends_with(".fnlpq")
        || name == ".fnlpq"
        || name.len() > 240
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        return Err(PackageError::InvalidInput {
            field: "logical_artifact_name",
            detail: format!("must be an ASCII filename ending in .fnlpq: {name:?}"),
        });
    }
    Ok(())
}

fn validate_record(record: &PackageFile) -> Result<(), PackageError> {
    if record.name.is_empty()
        || record.name.len() > 255
        || !record
            .name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        return Err(PackageError::Integrity {
            member: record.name.clone(),
            detail: "member name is not a safe basename".to_owned(),
        });
    }
    validate_digest(&record.sha256, &record.name)
}

fn validate_digest(digest: &str, member: &str) -> Result<(), PackageError> {
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return Err(PackageError::Integrity {
            member: member.to_owned(),
            detail: "digest is not 64 lowercase hexadecimal characters".to_owned(),
        });
    }
    Ok(())
}

fn insert_unique_name(
    names: &mut BTreeSet<String>,
    record: &PackageFile,
) -> Result<(), PackageError> {
    if !names.insert(record.name.clone()) {
        return Err(PackageError::Integrity {
            member: record.name.clone(),
            detail: "duplicate release member name".to_owned(),
        });
    }
    Ok(())
}

fn file_name(path: &Path) -> Result<String, PackageError> {
    path.file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| PackageError::InvalidInput {
            field: "path",
            detail: format!("has no UTF-8 basename: {}", path.display()),
        })
}

fn hex_digest(digest: impl AsRef<[u8]>) -> String {
    let mut result = String::with_capacity(64);
    for byte in digest.as_ref() {
        use std::fmt::Write as _;
        let _ = write!(result, "{byte:02x}");
    }
    result
}

#[cfg(test)]
mod tests {
    use super::{
        expected_part_count, part_name, reconstruction_text, validate_logical_name, PackageError,
        PackageFile, RELEASE_PART_BYTES,
    };

    #[test]
    fn exact_release_boundaries_and_two_digit_cap_are_checked() {
        assert_eq!(expected_part_count(0, RELEASE_PART_BYTES).unwrap(), 0);
        assert_eq!(expected_part_count(1, RELEASE_PART_BYTES).unwrap(), 1);
        assert_eq!(
            expected_part_count(RELEASE_PART_BYTES, RELEASE_PART_BYTES).unwrap(),
            1
        );
        assert_eq!(
            expected_part_count(RELEASE_PART_BYTES + 1, RELEASE_PART_BYTES).unwrap(),
            2
        );
        assert_eq!(part_name("model.fnlpq", 0).unwrap(), "model.fnlpq.part00");
        assert_eq!(part_name("model.fnlpq", 63).unwrap(), "model.fnlpq.part63");
        assert!(matches!(
            part_name("model.fnlpq", 64),
            Err(PackageError::Integrity { .. })
        ));
    }

    #[test]
    fn logical_names_are_versioned_basenames_not_paths() {
        validate_logical_name("nanbeige4.2-3b.fnlpq-v1.int8.generic.fnlpq").unwrap();
        for invalid in [
            "",
            "../model.fnlpq",
            "folder/model.fnlpq",
            "model.bin",
            "x.fnlpq\n",
        ] {
            assert!(
                validate_logical_name(invalid).is_err(),
                "accepted {invalid:?}"
            );
        }
    }

    #[test]
    fn reconstruction_instructions_preserve_receipt_order() {
        let parts = vec![
            PackageFile {
                name: "model.fnlpq.part00".to_owned(),
                bytes: 3,
                sha256: "0".repeat(64),
            },
            PackageFile {
                name: "model.fnlpq.part01".to_owned(),
                bytes: 1,
                sha256: "1".repeat(64),
            },
        ];
        let instructions = reconstruction_text("model.fnlpq", &parts, 4, &"2".repeat(64));
        assert!(instructions.contains("part_order=model.fnlpq.part00 model.fnlpq.part01"));
        assert!(instructions.contains("artifact_bytes=4"));
    }
}
