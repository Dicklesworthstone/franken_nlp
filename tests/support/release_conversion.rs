use franken_nlp::artifact::{
    converter::{
        CONVERSION_ARTIFACT_FORMAT, CONVERSION_RECEIPT_SCHEMA, ConversionReceipt, GENERIC_PACKING_V1,
        PINNED_CONVERSION_RECIPE, PINNED_REVISION, PINNED_SOURCE_MANIFEST_SHA256, PORTABLE_QUANT_V1,
    },
    format::{digest_domain, framed_sha256_hex},
};
use sha2::{Digest, Sha256};

/// Synthetic converter metadata with both file identities computed over the test payload.
pub fn canonical_receipt(artifact: &[u8]) -> String {
    let bytes = artifact.len() as u64;
    ConversionReceipt {
        receipt_schema: CONVERSION_RECEIPT_SCHEMA.to_owned(),
        model_id: "Nanbeige4.2-3B".to_owned(),
        model_revision: PINNED_REVISION.to_owned(),
        artifact_format: CONVERSION_ARTIFACT_FORMAT.to_owned(),
        source_manifest_sha256: PINNED_SOURCE_MANIFEST_SHA256.to_owned(),
        target_arch: "generic".to_owned(),
        source_root_sha256: "a".repeat(64),
        census_sha256: "b".repeat(64),
        logical_model_sha256: "c".repeat(64),
        converter_commit: "0".repeat(40),
        recipe_id: PINNED_CONVERSION_RECIPE.to_owned(),
        rounding_id: PORTABLE_QUANT_V1.to_owned(),
        packing_id: GENERIC_PACKING_V1.to_owned(),
        measured_peak_rss_bytes: 0,
        measured_scratch_bytes: 0,
        peak_rss_cap_bytes: 0,
        final_disk_bytes: bytes,
        measured_disk_bytes: bytes,
        output_len: bytes,
        fnlpq_file_sha256: framed_sha256_hex(digest_domain::FILE, &[artifact]).unwrap(),
        artifact_raw_sha256: format!("{:x}", Sha256::digest(artifact)),
        license_bundle_sha256: "d".repeat(64),
    }
    .canonical_json()
    .expect("canonical synthetic conversion receipt")
}
