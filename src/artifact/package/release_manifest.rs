//! Build the release manifest consumed by the existing pull verifier.
use super::{ArtifactHashers, PackageError, hex_digest, metadata_regular_file, verify_package_records};
use crate::artifact::{
    converter::{ConversionReceipt, ConversionSourceManifest, extract_chat_template, source_root_digest},
    format::{ArchTarget, SectionKind},
    manifest::{
        LOCAL_DERIVATION_PACKING_POLICY_V1, NANBEIGE42_INT8_V1_COMPATIBILITY, PULL_API_V1,
        RELEASE_MANIFEST_SCHEMA_V1, ReleaseArtifact, ReleaseCompatibility, ReleaseDigests,
        ReleaseLifecycle, ReleaseManifest, ReleasePart, ReleaseSourceClosure, ReleaseSourceFile,
        validate_release_coordinates,
    },
    pull::{LocalReconstructPhase, LocalReconstructProgress, verify_container_identity},
    reader::FnlpqRangeReader,
};
use crate::tokenizer::embedded::PINNED_TOKENIZER_CONFIG_BYTES;
use sha2::{Digest, Sha256};
use std::{fmt::Display, fs::File, io::Read, path::Path};

fn mismatch(member: &str, detail: impl Display) -> PackageError {
    PackageError::Integrity { member: member.to_owned(), detail: detail.to_string() }
}

/// Derive a pull manifest for the named tag in the FrankenNLP release repository.
/// Local input files must remain stable during the invocation. This operation
/// produces metadata for review before publication.
pub fn build_release_manifest(
    staging_dir: &Path,
    artifact: &Path,
    source_manifest: &Path,
    artifact_id: &str,
    release_tag: &str,
) -> Result<ReleaseManifest, PackageError> {
    validate_release_coordinates(artifact_id, release_tag).map_err(|e| mismatch("release", e))?;
    let source = ConversionSourceManifest::load_pinned(source_manifest)
        .map_err(|e| mismatch("source manifest", e))?;
    let source_root = source_root_digest(&source).map_err(|e| mismatch("source root", e))?;
    let (receipt, conversion) = verify_package_records(staging_dir)?;
    if source_root != conversion.source_root_sha256 {
        return Err(mismatch("source root", "pinned conversion closure differs from receipt"));
    }
    let reader = open_bound_artifact(artifact, &conversion)?;
    reader.select_packing(ArchTarget::Generic).map_err(|e| mismatch("Generic packing", e))?;
    let source_sha = |name: &str| source.files.iter().find(|file| file.name == name)
        .map(|file| file.sha256.clone()).ok_or_else(|| mismatch(name, "missing from pinned closure"));
    for (kind, name) in [(SectionKind::ModelConfig, "config.json"), (SectionKind::TokenizerConfig, "tokenizer_config.json")] {
        let section = reader.sections().iter().find(|section| section.kind == kind)
            .ok_or_else(|| mismatch(name, "missing artifact section"))?;
        let observed = reader.raw_section_sha256(section.ordinal).map_err(|e| mismatch(name, e))?;
        if hex_digest(observed) != source_sha(name)? {
            return Err(mismatch(name, "materialized artifact section differs from pinned source"));
        }
    }
    let tokenizer_sha256 = source_sha("tokenizer.model")?;
    if hex_digest(Sha256::digest(PINNED_TOKENIZER_CONFIG_BYTES)) != source_sha("tokenizer_config.json")? {
        return Err(mismatch("tokenizer config", "embedded config differs from pinned source"));
    }
    let template = extract_chat_template(PINNED_TOKENIZER_CONFIG_BYTES)
        .map_err(|e| mismatch("chat template", e))?;
    let parts = receipt.parts.into_iter().enumerate().map(|(index, part)| ReleasePart {
        // Package verification enforces canonical ordering and the 64-part cap.
        id: index as u32,
        mirrors: vec![format!("https://github.com/Dicklesworthstone/franken_nlp/releases/download/{release_tag}/{}", part.name)],
        name: part.name, bytes: part.bytes, sha256: part.sha256,
    }).collect();
    let manifest = ReleaseManifest {
        manifest_schema: RELEASE_MANIFEST_SCHEMA_V1.to_owned(),
        model_id: conversion.model_id, source_revision: conversion.model_revision,
        artifact_id: artifact_id.to_owned(), release_tag: release_tag.to_owned(),
        artifact: ReleaseArtifact { logical_name: receipt.logical_artifact_name,
            bytes: receipt.artifact_bytes, sha256: receipt.fnlpq_file_sha256 },
        parts,
        source: ReleaseSourceClosure { source_root_sha256: source_root,
            files: source.files.into_iter().map(|file| ReleaseSourceFile {
                name: file.name, bytes: file.bytes, sha256: file.sha256,
            }).collect() },
        compatibility: ReleaseCompatibility { recipe_id: conversion.recipe_id,
            converter_id: conversion.converter_commit, fnlpq_format: conversion.artifact_format,
            pull_api: PULL_API_V1.to_owned() },
        digests: ReleaseDigests { tokenizer_sha256, template_sha256: hex_digest(Sha256::digest(template)),
            census_sha256: conversion.census_sha256, logical_model_sha256: conversion.logical_model_sha256,
            packing_set_sha256: reader.packing_set_sha256().to_owned() },
        license_bundle_sha256: conversion.license_bundle_sha256,
        packing_policy: LOCAL_DERIVATION_PACKING_POLICY_V1.to_owned(),
        lifecycle: ReleaseLifecycle { state: "active".to_owned(), superseded_by: None, revocation_reason: None },
    };
    manifest.validate_for(NANBEIGE42_INT8_V1_COMPATIBILITY).map_err(|e| mismatch("manifest", e))?;
    let progress = LocalReconstructProgress { phase: LocalReconstructPhase::VerifyArtifact,
        part_index: None, verified_prefix_bytes: 0, copied_bytes: 0, artifact_bytes: manifest.artifact.bytes };
    verify_container_identity(&reader, &manifest, progress, &mut |_| None)
        .map_err(|e| mismatch("artifact", e))?;
    Ok(manifest)
}

// The checked reader retains the same descriptor that matched both file identities.
fn open_bound_artifact(path: &Path, receipt: &ConversionReceipt) -> Result<FnlpqRangeReader, PackageError> {
    if metadata_regular_file(path)?.len() != receipt.output_len {
        return Err(mismatch("artifact", "length differs from packaged artifact"));
    }
    let mut file = File::open(path).map_err(|e| mismatch("artifact", e))?;
    let opened = file.metadata().map_err(|e| mismatch("artifact", e))?;
    if !opened.is_file() || opened.len() != receipt.output_len {
        return Err(mismatch("artifact", "opened file differs from expected regular file"));
    }
    let mut hashes = ArtifactHashers::new(receipt.output_len);
    let mut buffer = vec![0u8; super::COPY_BUFFER_BYTES];
    let mut left = receipt.output_len;
    while left > 0 {
        let take = left.min(buffer.len() as u64) as usize;
        file.read_exact(&mut buffer[..take]).map_err(|e| mismatch("artifact", e))?;
        hashes.update(&buffer[..take]);
        left -= take as u64;
    }
    if file.read(&mut buffer[..1]).map_err(|e| mismatch("artifact", e))? != 0 {
        return Err(mismatch("artifact", "trailing bytes after expected length"));
    }
    hashes.verify(receipt)?;
    FnlpqRangeReader::from_file_with_checkpoint(file, || None).map_err(|e| mismatch("artifact", e))
}
