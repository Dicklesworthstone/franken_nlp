//! Integration coverage for the immutable model-release package closure.

// The package code is included directly, matching the established canonjson
// contract-target pattern.  This ensures Cargo's integration target executes
// this module even though the two CLI binaries intentionally have `test = false`.
#[path = "../src/artifact/package.rs"]
mod package;

use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use package::{
    MAX_RELEASE_PARTS, ModelAssetReceipt, PackageError, PackageFile, PackageRequest,
    RELEASE_PART_BYTES, package_model, release_part_count, release_part_len, release_part_name,
    verify_model_package,
};

const LOGICAL_NAME: &str = "tiny-release.fnlpq";

#[test]
fn fixed_part_layout_covers_zero_exact_boundary_one_over_tail_and_64_cap() {
    assert_eq!(release_part_count(0).unwrap(), 0);
    assert_eq!(release_part_count(7).unwrap(), 1);
    assert_eq!(release_part_len(7, 0).unwrap(), 7);

    assert_eq!(release_part_count(RELEASE_PART_BYTES).unwrap(), 1);
    assert_eq!(
        release_part_len(RELEASE_PART_BYTES, 0).unwrap(),
        RELEASE_PART_BYTES
    );
    assert_eq!(release_part_count(RELEASE_PART_BYTES + 1).unwrap(), 2);
    assert_eq!(
        release_part_len(RELEASE_PART_BYTES + 1, 0).unwrap(),
        RELEASE_PART_BYTES
    );
    assert_eq!(release_part_len(RELEASE_PART_BYTES + 1, 1).unwrap(), 1);
    assert_eq!(
        release_part_name(LOGICAL_NAME, MAX_RELEASE_PARTS - 1).unwrap(),
        "tiny-release.fnlpq.part63"
    );

    let at_cap = RELEASE_PART_BYTES * u64::try_from(MAX_RELEASE_PARTS).unwrap();
    assert_eq!(release_part_count(at_cap).unwrap(), MAX_RELEASE_PARTS);
    let cap_error = release_part_count(at_cap + 1).expect_err("65th part rejects");
    assert!(cap_error.to_string().contains("exceeds v1 cap 64"));
}

#[test]
fn zero_and_tail_artifacts_split_and_reassemble_byte_identically() {
    for (label, source) in [("zero", Vec::new()), ("tail", vec![0, 1, 2, 3, 255])] {
        let fixture = package_fixture(label, &source);
        let report = package_model(&fixture.request()).expect("package synthetic artifact");
        let verified = verify_model_package(&fixture.staging_dir).expect("verify package closure");
        assert_eq!(verified, report);

        let reassembled = report.parts.iter().try_fold(Vec::new(), |mut bytes, part| {
            bytes.extend(fs::read(fixture.staging_dir.join(&part.name))?);
            Ok::<_, std::io::Error>(bytes)
        });
        assert_eq!(reassembled.expect("read ordered parts"), source, "{label}");
    }
}

#[test]
fn sha256sums_is_one_strict_record_per_hashed_release_member() {
    let fixture = package_fixture("sha256sums", &[1, 2, 3]);
    let report = package_model(&fixture.request()).expect("package synthetic artifact");
    let sums = fs::read_to_string(fixture.staging_dir.join("SHA256SUMS")).expect("read sums");
    let lines = sums.lines().collect::<Vec<_>>();
    assert_eq!(lines.len(), report.parts.len() + 1 + 3 + 1 + 1);
    for line in lines {
        let (digest, name) = line.split_once("  ").expect("two-space sum delimiter");
        assert_eq!(digest.len(), 64, "digest width");
        assert!(
            digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
        );
        assert!(!name.is_empty());
        assert!(
            name.bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
        );
    }
}

#[test]
fn tampered_missing_and_renamed_part_rejections_name_the_member() {
    let tampered = package_fixture("tampered", &[7, 8, 9]);
    let report = package_model(&tampered.request()).expect("package synthetic artifact");
    let part = &report.parts[0].name;
    OpenOptions::new()
        .append(true)
        .open(tampered.staging_dir.join(part))
        .expect("open test package part")
        .write_all(&[0xff])
        .expect("tamper test package part");
    assert_named_rejection(verify_model_package(&tampered.staging_dir), part);

    for label in ["missing", "renamed"] {
        let fixture = package_fixture(label, &[3, 2, 1]);
        let report = package_model(&fixture.request()).expect("package synthetic artifact");
        let part = &report.parts[0].name;
        fs::rename(
            fixture.staging_dir.join(part),
            fixture.staging_dir.join(format!("{label}-{part}")),
        )
        .expect("rename test package part");
        assert_named_rejection(verify_model_package(&fixture.staging_dir), part);
    }
}

#[test]
fn wrong_order_and_edited_receipt_fields_reject_before_exposure() {
    let wrong_order = package_fixture("wrong-order", &[5]);
    package_model(&wrong_order.request()).expect("package synthetic artifact");
    let mut receipt = read_receipt(&wrong_order.staging_dir);
    receipt.artifact_bytes = RELEASE_PART_BYTES + 1;
    receipt.parts = vec![
        PackageFile {
            name: release_part_name(LOGICAL_NAME, 1).unwrap(),
            bytes: 1,
            sha256: "1".repeat(64),
        },
        PackageFile {
            name: release_part_name(LOGICAL_NAME, 0).unwrap(),
            bytes: RELEASE_PART_BYTES,
            sha256: "2".repeat(64),
        },
    ];
    write_receipt(&wrong_order.staging_dir, &receipt);
    assert_named_rejection(
        verify_model_package(&wrong_order.staging_dir),
        "tiny-release.fnlpq.part01",
    );

    let edited = package_fixture("edited-receipt", &[6]);
    package_model(&edited.request()).expect("package synthetic artifact");
    let mut receipt = read_receipt(&edited.staging_dir);
    receipt.part_bytes = 1;
    write_receipt(&edited.staging_dir, &receipt);
    assert_named_rejection(
        verify_model_package(&edited.staging_dir),
        "MODEL_ASSET_RECEIPT.json",
    );
}

fn assert_named_rejection(result: Result<package::PackageReport, PackageError>, name: &str) {
    let error = result.expect_err("hostile release package must reject");
    assert!(
        error.to_string().contains(name),
        "error must name {name:?}: {error}"
    );
}

fn read_receipt(staging_dir: &Path) -> ModelAssetReceipt {
    serde_json::from_slice(
        &fs::read(staging_dir.join("MODEL_ASSET_RECEIPT.json")).expect("read test receipt"),
    )
    .expect("parse test receipt")
}

fn write_receipt(staging_dir: &Path, receipt: &ModelAssetReceipt) {
    fs::write(
        staging_dir.join("MODEL_ASSET_RECEIPT.json"),
        serde_json::to_vec_pretty(receipt).expect("serialize test receipt"),
    )
    .expect("edit test receipt");
}

struct Fixture {
    artifact: PathBuf,
    conversion_receipt: PathBuf,
    license_bundle_dir: PathBuf,
    staging_dir: PathBuf,
}

impl Fixture {
    fn request(&self) -> PackageRequest {
        PackageRequest {
            artifact: self.artifact.clone(),
            staging_dir: self.staging_dir.clone(),
            logical_artifact_name: LOGICAL_NAME.to_owned(),
            conversion_receipt: self.conversion_receipt.clone(),
            license_bundle_dir: self.license_bundle_dir.clone(),
        }
    }
}

fn package_fixture(label: &str, artifact_bytes: &[u8]) -> Fixture {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock after epoch")
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "franken-nlp-release-package-{label}-{}-{nonce}",
        std::process::id()
    ));
    fs::create_dir(&root).expect("create isolated package fixture directory");
    let artifact = root.join(LOGICAL_NAME);
    fs::write(&artifact, artifact_bytes).expect("write synthetic artifact");
    let conversion_receipt = root.join("conversion-source.json");
    fs::write(&conversion_receipt, b"{\"fixture\":true}\n").expect("write conversion receipt");
    let license_bundle_dir = root.join("license-bundle");
    fs::create_dir(&license_bundle_dir).expect("create license bundle directory");
    for (name, bytes) in [
        ("APACHE-2.0.txt", b"Apache-2.0\n".as_slice()),
        ("ATTRIBUTION.txt", b"Nanbeige Team\n".as_slice()),
        (
            "MODIFICATION_NOTICE.txt",
            b"fixture modification\n".as_slice(),
        ),
    ] {
        fs::write(license_bundle_dir.join(name), bytes).expect("write required license member");
    }
    Fixture {
        artifact,
        conversion_receipt,
        license_bundle_dir,
        staging_dir: root.join("release-staging"),
    }
}
