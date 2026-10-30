//! Real filesystem reconstruction of a small, current-format release package.
//!
//! The public writer, package builder, manifest parser, range reader and
//! reconstruction API are exercised together. The 201 tensor declarations use
//! the frozen census shapes and tiny structural payloads, following the existing
//! fnlpq_census_on_load fixture. They are not runnable model weights or format
//! qualification. Fixed release chunking remains unchanged: this small artifact
//! is one legitimate tail part.

use std::{
    fs::{self, OpenOptions},
    io::{Seek, SeekFrom, Write},
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
use franken_nlp::{
    artifact::{
        converter::expected_nanbeige42_census,
        format::{
            ArchTarget, CanonicalDtype, FnlpqWriterInput, PackingSetInput, SectionKind,
            SectionPayload, SectionRange, TensorInput, framed_sha256_hex,
            logical_model_sha256, logical_tensor_sha256, write,
        },
        manifest::{
            FNLPQ_FORMAT_V1, LOCAL_DERIVATION_PACKING_POLICY_V1, PULL_API_V1,
            RELEASE_MANIFEST_SCHEMA_V1, ReleaseArtifact, ReleaseCompatibility,
            ReleaseDigests, ReleaseLifecycle, ReleaseManifest, ReleasePart,
            ReleaseSourceClosure, ReleaseSourceFile,
        },
        package::{PackageRequest, package_model, verify_model_package},
        pull::{
            LocalReconstructOptions, LocalReconstructPhase, LocalReconstructRequest,
            PullError, reconstruct_local_release, reconstruct_local_release_with_options,
        },
        reader::FnlpqRangeReader,
    },
    native_engine::{
        artifact_bridge::{ArtifactBridgeError, ArtifactLoadBudget, CheckedArtifactSource,
            CurrentCandidateArtifactSource, current_candidate_load_checkpoints},
        decode::DecodeCancellationKind,
        strict_int8::{CurrentCandidateInt8Error, CurrentCandidateInt8Model},
    },
};
use sha2::{Digest, Sha256};

const MODEL_ID: &str = "Nanbeige4.2-3B";
const REVISION: &str = "f56ec5a9650268aa098496734743c25ea778bd2d";
const RECIPE: &str = "nanbeige42-int8-v1";
const STRUCTURAL_STORAGE: &str = "census-structure-v1";
const LOGICAL_NAME: &str = "nanbeige4.2-3b.fnlpq-v1.int8.generic.fnlpq";
const RELEASE_TAG: &str = "models-reconstruction-fixture-v1";
const CONFIG: &[u8] = b"{}";
const TOKENIZER: &[u8] = &[0x0a];
const TEMPLATE: &[u8] = b"structural reconstruction fixture";
static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

#[test]
fn checked_candidate_startup_exceeds_the_old_64_poll_limit_and_fits_its_byte_quote() {
    let fixture = Fixture::new("candidate-startup-budget");
    let path = fixture.root.join(LOGICAL_NAME);
    let mut old_budget = 64_u64;
    let error = CurrentCandidateArtifactSource::open_with_checkpoint(&path, || {
        if old_budget == 0 { return Some(DecodeCancellationKind::PollQuota); }
        old_budget -= 1; None
    }).err().expect("even this structural fixture exceeds the old fixed allowance");
    assert_eq!(error, ArtifactBridgeError::Cancelled(DecodeCancellationKind::PollQuota));
    let quoted = current_candidate_load_checkpoints(fixture.artifact.len() as u64).unwrap();
    let mut polls = 0;
    let source = CurrentCandidateArtifactSource::open_with_checkpoint(&path, || {
        polls += 1;
        (polls > quoted).then_some(DecodeCancellationKind::PollQuota)
    }).expect("complete stored and semantic authentication fits its finite quote");
    assert!(polls > 64 && polls < quoted);
    assert_eq!(source.file_bytes(), fixture.artifact.len() as u64);
    assert_eq!(source.identity().logical_model_sha256, fixture.manifest.digests.logical_model_sha256);
    let error = CurrentCandidateInt8Model::from_checked_source_with_checkpoint(source,
        ArtifactLoadBudget::streaming_only(1, 64 * 1024),
        |_| panic!("cancelled handoff must not materialize any weights"),
        || Some(DecodeCancellationKind::Shutdown),
    ).err().expect("cancellation survives the source-owning model boundary");
    assert!(matches!(error, CurrentCandidateInt8Error::Artifact(
        ArtifactBridgeError::Cancelled(DecodeCancellationKind::Shutdown))));
}

#[cfg(unix)]
#[test]
fn checked_source_handoff_keeps_its_authenticated_descriptor_after_path_replacement() {
    let fixture = Fixture::new("candidate-held-descriptor");
    let path = fixture.root.join(LOGICAL_NAME);
    let source = CurrentCandidateArtifactSource::open(&path).unwrap();
    let identity = source.identity().clone();
    let retained = fixture.root.join("retained-original.fnlpq");
    fs::rename(&path, &retained).unwrap();
    fs::write(&path, b"replacement must never become the inspected model").unwrap();
    assert_eq!(source.identity(), &identity);
    let tokenizer_digest: [u8; 32] = Sha256::digest(TOKENIZER).into();
    assert_eq!(source.tokenizer_model_sha256().unwrap(), tokenizer_digest);
    // This intentionally non-runnable fixture must fail tokenizer binding on
    // the held original. Reopening the replacement would fail artifact parsing
    // instead; no weakened model or synthetic success is accepted.
    let error = CurrentCandidateInt8Model::from_checked_source_with_checkpoint(source,
        ArtifactLoadBudget::streaming_only(1, 64 * 1024), |_| {}, || None,
    ).err().expect("structural tokenizer is not the pinned runnable tokenizer");
    assert!(matches!(error, CurrentCandidateInt8Error::TokenizerMismatch("tokenizer.model")));
    assert_eq!(fs::read(&path).unwrap(), b"replacement must never become the inspected model");
}

#[test]
fn fresh_package_reconstruction_publishes_exact_checked_bytes() {
    let fixture = Fixture::new("fresh");
    let request = fixture.request("fresh");
    let report = reconstruct_local_release(&request).expect("reconstruct real package files");
    assert_eq!(fs::read(&request.output).unwrap(), fixture.artifact);
    assert_eq!(fs::read(&request.staging_output).unwrap(), fixture.artifact);
    assert_eq!(report.artifact_bytes, fixture.artifact.len() as u64);
    assert_eq!(report.artifact_sha256, digest(&fixture.artifact));
    assert_eq!(report.release_manifest_sha256, request.expected_release_manifest_sha256);
    assert_eq!(report.part_count, 1);
    assert_eq!(report.resumed_bytes, 0);
    assert_eq!(report.copied_bytes, fixture.artifact.len() as u64);
    assert!(!report.directory_durability_confirmed);
    let checked = FnlpqRangeReader::open(&request.output).expect("published envelope reopens");
    assert_eq!(checked.tensors().len(), 201);
    assert_eq!(checked.logical_model_sha256(), fixture.manifest.digests.logical_model_sha256);
    assert_eq!(fs::read(fixture.part()).unwrap(), fixture.artifact);
}

#[test]
fn byte_budget_retains_exact_prefix_and_resume_matches_fresh_output() {
    let fixture = Fixture::new("budget-resume");
    let request = fixture.request("resumed");
    let error = reconstruct_local_release_with_options(
        &request,
        &LocalReconstructOptions { resume: false, max_new_bytes: Some(137) },
        |_| None,
    ).expect_err("bounded first invocation cannot finish");
    assert!(matches!(error, PullError::CopyBudgetExceeded {
        limit: 137, copied_bytes: 137, retained_bytes: 137, ..
    }));
    assert_eq!(fs::read(&request.staging_output).unwrap(), fixture.artifact[..137]);
    assert!(!request.output.exists());
    let resumed = reconstruct_local_release_with_options(
        &request, &resume_options(None), |_| None,
    ).expect("continue only after prefix authentication");
    let fresh_request = fixture.request("uninterrupted");
    let fresh = reconstruct_local_release(&fresh_request).expect("independent fresh reconstruction");
    assert_eq!(resumed.resumed_bytes, 137);
    assert_eq!(resumed.copied_bytes, fixture.artifact.len() as u64 - 137);
    assert_eq!(resumed.artifact_sha256, fresh.artifact_sha256);
    assert_eq!(fs::read(&request.output).unwrap(), fs::read(&fresh_request.output).unwrap());
}

#[test]
fn manually_retained_mid_part_prefix_is_authenticated_without_truncation() {
    let fixture = Fixture::new("mid-part");
    let request = fixture.request("resume");
    let retained = fixture.artifact.len() / 3 + 17;
    fs::write(&request.staging_output, &fixture.artifact[..retained]).unwrap();
    let report = reconstruct_local_release_with_options(
        &request, &resume_options(None), |_| None,
    ).expect("valid prefix in the middle of a part resumes");
    assert_eq!(report.resumed_bytes, retained as u64);
    assert_eq!(report.copied_bytes, (fixture.artifact.len() - retained) as u64);
    assert_eq!(fs::read(&request.output).unwrap(), fixture.artifact);
}

#[test]
fn complete_retained_artifact_finishes_with_zero_new_byte_budget() {
    let fixture = Fixture::new("complete");
    let request = fixture.request("resume");
    fs::write(&request.staging_output, &fixture.artifact).unwrap();
    let report = reconstruct_local_release_with_options(
        &request, &resume_options(Some(0)), |_| None,
    ).expect("verified complete staging needs no copy budget");
    assert_eq!(report.resumed_bytes, fixture.artifact.len() as u64);
    assert_eq!(report.copied_bytes, 0);
    assert_eq!(fs::read(&request.output).unwrap(), fixture.artifact);
}

#[test]
fn mismatching_retained_prefix_is_preserved_and_never_extended() {
    let fixture = Fixture::new("prefix-mismatch");
    let request = fixture.request("resume");
    let mut retained = fixture.artifact[..137].to_vec();
    retained[61] ^= 0x80;
    fs::write(&request.staging_output, &retained).unwrap();
    let error = reconstruct_local_release_with_options(
        &request, &resume_options(None), |_| None,
    ).expect_err("same-length damaged prefix must reject");
    assert!(matches!(error, PullError::ResumePrefixMismatch { part: 0, .. }));
    assert_eq!(fs::read(&request.staging_output).unwrap(), retained);
    assert!(!request.output.exists());
}

#[test]
fn source_part_mutation_after_retained_prefix_rejects_before_append() {
    let fixture = Fixture::new("part-mismatch");
    let request = fixture.request("resume");
    let retained = fixture.artifact[..137].to_vec();
    fs::write(&request.staging_output, &retained).unwrap();
    let mut part = OpenOptions::new().write(true).open(fixture.part()).unwrap();
    part.seek(SeekFrom::End(-1)).unwrap();
    part.write_all(&[fixture.artifact[fixture.artifact.len() - 1] ^ 1]).unwrap();
    part.sync_all().unwrap();
    let error = reconstruct_local_release_with_options(
        &request, &resume_options(None), |_| None,
    ).expect_err("authenticate entire touched source part, including unseen suffix");
    assert!(matches!(error, PullError::PartDigest { part: 0, .. }));
    assert_eq!(fs::read(&request.staging_output).unwrap(), retained);
    assert!(!request.output.exists());
}

#[test]
fn manifest_pin_mismatch_cannot_touch_retained_staging() {
    let fixture = Fixture::new("manifest-pin");
    let mut request = fixture.request("resume");
    let retained = fixture.artifact[..137].to_vec();
    fs::write(&request.staging_output, &retained).unwrap();
    request.expected_release_manifest_sha256 = "00".repeat(32);
    let error = reconstruct_local_release_with_options(
        &request, &resume_options(None), |_| None,
    ).expect_err("local manifest cannot choose its own authority pin");
    assert!(matches!(error, PullError::ManifestDigestMismatch { .. }));
    assert_eq!(fs::read(&request.staging_output).unwrap(), retained);
    assert!(!request.output.exists());
}

#[test]
fn retained_trailing_bytes_are_refused_without_truncation() {
    let fixture = Fixture::new("overlong");
    let request = fixture.request("resume");
    let mut retained = fixture.artifact.clone();
    retained.push(0x5a);
    fs::write(&request.staging_output, &retained).unwrap();
    let error = reconstruct_local_release_with_options(
        &request, &resume_options(None), |_| None,
    ).expect_err("resume may never silently truncate an unexpected tail");
    assert!(matches!(error, PullError::ResumeLength { .. }));
    assert_eq!(fs::read(&request.staging_output).unwrap(), retained);
    assert!(!request.output.exists());
}

#[test]
fn fresh_mode_never_adopts_or_overwrites_existing_staging() {
    let fixture = Fixture::new("fresh-existing-stage");
    let request = fixture.request("fresh");
    fs::write(&request.staging_output, b"existing caller bytes").unwrap();
    assert!(reconstruct_local_release(&request).is_err());
    assert_eq!(fs::read(&request.staging_output).unwrap(), b"existing caller bytes");
    assert!(!request.output.exists());
}

#[test]
fn existing_final_output_is_never_replaced_even_after_valid_resume() {
    let fixture = Fixture::new("existing-output");
    let request = fixture.request("resume");
    fs::write(&request.output, b"previous artifact remains visible").unwrap();
    let retained = fixture.artifact[..137].to_vec();
    fs::write(&request.staging_output, &retained).unwrap();
    assert!(reconstruct_local_release_with_options(
        &request, &resume_options(None), |_| None,
    ).is_err());
    assert_eq!(fs::read(&request.output).unwrap(), b"previous artifact remains visible");
    assert!(fs::read(&request.staging_output).unwrap().starts_with(&retained));
}

#[test]
fn prefix_cancellation_preserves_cause_and_all_existing_bytes() {
    let fixture = Fixture::new("cancel-prefix");
    let request = fixture.request("resume");
    let retained = fixture.artifact[..137].to_vec();
    fs::write(&request.staging_output, &retained).unwrap();
    let error = reconstruct_local_release_with_options(
        &request, &resume_options(None),
        |progress| (progress.phase == LocalReconstructPhase::VerifyPrefix)
            .then_some(DecodeCancellationKind::Deadline),
    ).expect_err("prefix cancellation stops before append");
    assert!(matches!(error, PullError::Cancelled {
        cause: DecodeCancellationKind::Deadline, ..
    }));
    assert_eq!(fs::read(&request.staging_output).unwrap(), retained);
    assert!(!request.output.exists());
}

#[test]
fn copy_cancellation_retains_resumable_prefix_and_actual_progress() {
    let fixture = Fixture::new("cancel-copy");
    let request = fixture.request("fresh");
    let error = reconstruct_local_release_with_options(
        &request, &LocalReconstructOptions::default(),
        |progress| (progress.phase == LocalReconstructPhase::Copy && progress.copied_bytes > 0)
            .then_some(DecodeCancellationKind::Shutdown),
    ).expect_err("a multi-chunk copy exposes cancellation between chunks");
    let PullError::Cancelled { cause, progress } = error else {
        panic!("copy cancellation must retain typed cause and progress");
    };
    assert_eq!(cause, DecodeCancellationKind::Shutdown);
    let retained = fs::read(&request.staging_output).unwrap();
    assert!(!retained.is_empty());
    assert!(retained.len() < fixture.artifact.len());
    assert_eq!(progress.copied_bytes, retained.len() as u64);
    assert_eq!(retained, fixture.artifact[..retained.len()]);
    assert!(!request.output.exists());
    let report = reconstruct_local_release_with_options(
        &request, &resume_options(None), |_| None,
    ).expect("cancelled staging resumes through actual verification");
    assert_eq!(report.resumed_bytes, retained.len() as u64);
    assert_eq!(fs::read(&request.output).unwrap(), fixture.artifact);
}

#[test]
fn final_verification_and_publish_cancellation_leave_complete_unpublished_staging() {
    for (phase, cause, label) in [
        (LocalReconstructPhase::VerifyArtifact, DecodeCancellationKind::CostBudget, "verify"),
        (LocalReconstructPhase::Publish, DecodeCancellationKind::LinkedExit, "publish"),
    ] {
        let fixture = Fixture::new(label);
        let request = fixture.request("cancel-final");
        let error = reconstruct_local_release_with_options(
            &request, &LocalReconstructOptions::default(),
            |progress| (progress.phase == phase).then_some(cause),
        ).expect_err("completion may not bypass final cancellation checkpoints");
        assert!(matches!(error, PullError::Cancelled { cause: actual, .. } if actual == cause));
        assert!(!request.output.exists());
        assert_eq!(fs::read(&request.staging_output).unwrap(), fixture.artifact);
    }
}

#[test]
fn authenticated_manifest_with_wrong_whole_digest_cannot_publish() {
    let fixture = Fixture::new("whole-digest");
    let mut request = fixture.request("fresh");
    let mut manifest = fixture.manifest.clone();
    manifest.artifact.sha256 = "00".repeat(32);
    let path = fixture.root.join("wrong-whole.manifest.json");
    fs::write(&path, manifest.canonical_bytes().unwrap()).unwrap();
    request.manifest = path;
    request.expected_release_manifest_sha256 = manifest.release_manifest_sha256().unwrap();
    let error = reconstruct_local_release(&request).expect_err("per-part integrity is insufficient");
    assert!(matches!(error, PullError::ArtifactDigest { .. }));
    assert!(!request.output.exists());
    assert_eq!(fs::read(&request.staging_output).unwrap(), fixture.artifact);
}

#[cfg(unix)]
#[test]
fn resume_does_not_follow_a_staging_symlink() {
    use std::os::unix::fs::symlink;
    let fixture = Fixture::new("stage-symlink");
    let request = fixture.request("resume");
    let target = fixture.root.join("caller-owned-file");
    fs::write(&target, b"unrelated retained content").unwrap();
    symlink(&target, &request.staging_output).unwrap();
    assert!(reconstruct_local_release_with_options(
        &request, &resume_options(None), |_| None,
    ).is_err());
    assert_eq!(fs::read(target).unwrap(), b"unrelated retained content");
    assert!(fs::symlink_metadata(&request.staging_output).unwrap().file_type().is_symlink());
    assert!(!request.output.exists());
}

fn resume_options(max_new_bytes: Option<u64>) -> LocalReconstructOptions {
    LocalReconstructOptions { resume: true, max_new_bytes }
}

struct Fixture {
    root: PathBuf,
    artifact: Vec<u8>,
    manifest: ReleaseManifest,
    manifest_path: PathBuf,
    parts_dir: PathBuf,
}

impl Fixture {
    fn new(label: &str) -> Self {
        let sequence = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "fnlp-local-reconstruct-{label}-{}-{sequence}", std::process::id(),
        ));
        fs::create_dir(&root).expect("unique fixture directory; never adopt an existing root");
        let artifact = structural_artifact();
        let artifact_path = root.join(LOGICAL_NAME);
        fs::write(&artifact_path, &artifact).unwrap();
        let reader = FnlpqRangeReader::open(&artifact_path).expect("fixture must pass real checked reader");
        assert_eq!(reader.tensors().len(), 201);
        let receipt = root.join("conversion.json");
        fs::write(&receipt, b"{\"scope\":\"synthetic-transport-only\"}\n").unwrap();
        let license_dir = root.join("licenses");
        fs::create_dir(&license_dir).unwrap();
        for (name, bytes) in [
            ("APACHE-2.0.txt", b"Apache-2.0\n".as_slice()),
            ("ATTRIBUTION.txt", b"Nanbeige Team\n".as_slice()),
            ("MODIFICATION_NOTICE.txt", b"Synthetic transport fixture only\n".as_slice()),
        ] {
            fs::write(license_dir.join(name), bytes).unwrap();
        }
        let parts_dir = root.join("parts");
        let package = package_model(&PackageRequest {
            artifact: artifact_path, staging_dir: parts_dir.clone(),
            logical_artifact_name: LOGICAL_NAME.to_owned(), conversion_receipt: receipt,
            license_bundle_dir: license_dir,
        }).expect("real immutable package builder");
        assert_eq!(verify_model_package(&parts_dir).unwrap(), package);
        assert_eq!(package.parts.len(), 1, "fixed production chunk policy remains unchanged");
        let manifest = ReleaseManifest {
            manifest_schema: RELEASE_MANIFEST_SCHEMA_V1.to_owned(),
            model_id: MODEL_ID.to_owned(), source_revision: REVISION.to_owned(),
            artifact_id: "nanbeige42-reconstruction-fixture-v1".to_owned(),
            release_tag: RELEASE_TAG.to_owned(),
            artifact: ReleaseArtifact {
                logical_name: LOGICAL_NAME.to_owned(), bytes: package.artifact_bytes,
                sha256: package.artifact_sha256,
            },
            parts: package.parts.into_iter().enumerate().map(|(index, part)| ReleasePart {
                id: index as u32,
                mirrors: vec![format!(
                    "https://github.com/Dicklesworthstone/franken_nlp/releases/download/{RELEASE_TAG}/{}",
                    part.name,
                )],
                name: part.name, bytes: part.bytes, sha256: part.sha256,
            }).collect(),
            source: ReleaseSourceClosure {
                files: vec![ReleaseSourceFile {
                    name: "config.json".to_owned(), bytes: CONFIG.len() as u64, sha256: digest(CONFIG),
                }],
                source_root_sha256: reader.source_root_sha256().to_owned(),
            },
            compatibility: ReleaseCompatibility {
                recipe_id: RECIPE.to_owned(), converter_id: "fnlp-structural-fixture-v1".to_owned(),
                fnlpq_format: FNLPQ_FORMAT_V1.to_owned(), pull_api: PULL_API_V1.to_owned(),
            },
            digests: ReleaseDigests {
                tokenizer_sha256: digest(TOKENIZER), template_sha256: digest(TEMPLATE),
                census_sha256: census_digest(),
                logical_model_sha256: reader.logical_model_sha256().to_owned(),
                packing_set_sha256: reader.packing_set_sha256().to_owned(),
            },
            license_bundle_sha256: reader.license_bundle_sha256().to_owned(),
            packing_policy: LOCAL_DERIVATION_PACKING_POLICY_V1.to_owned(),
            lifecycle: ReleaseLifecycle {
                state: "active".to_owned(), superseded_by: None, revocation_reason: None,
            },
        };
        let manifest_path = root.join("release.manifest.json");
        fs::write(&manifest_path, manifest.canonical_bytes().unwrap()).unwrap();
        Self { root, artifact, manifest, manifest_path, parts_dir }
    }

    fn request(&self, label: &str) -> LocalReconstructRequest {
        LocalReconstructRequest {
            manifest: self.manifest_path.clone(),
            expected_release_manifest_sha256: self.manifest.release_manifest_sha256().unwrap(),
            parts_dir: self.parts_dir.clone(),
            staging_output: self.root.join(format!("{label}.stage")),
            output: self.root.join(format!("{label}.fnlpq")),
        }
    }

    fn part(&self) -> PathBuf { self.parts_dir.join(&self.manifest.parts[0].name) }
}

fn structural_artifact() -> Vec<u8> {
    // More than one streaming chunk tests cancellation during actual copying.
    const BYTES_PER_TENSOR: usize = 8192;
    let mut payload = Vec::new();
    let mut tensors = Vec::new();
    let mut digests = Vec::new();
    for (ordinal, expected) in expected_nanbeige42_census().into_iter().enumerate() {
        let shape: Vec<_> = expected.shape.iter().map(|&n| u32::try_from(n).unwrap()).collect();
        let bytes: Vec<_> = (0..BYTES_PER_TENSOR).map(|n| ((ordinal + n) % 251) as u8).collect();
        let offset = payload.len() as u64;
        payload.extend_from_slice(&bytes);
        let tensor_digest = logical_tensor_sha256(
            &expected.name, "bf16", &shape, STRUCTURAL_STORAGE, &bytes, &[], &[],
        ).unwrap();
        digests.push((expected.name.clone(), tensor_digest));
        tensors.push(TensorInput {
            name: expected.name, canonical_dtype: CanonicalDtype::Bf16, shape,
            canonical_logical_sha256: hex(&tensor_digest), quantization: STRUCTURAL_STORAGE.to_owned(),
            data: SectionRange::new("generic-payload", offset, bytes.len() as u64),
            scale: SectionRange::new("generic-scales", 0, 0),
            row_sum: SectionRange::new("generic-row-sums", 0, 0),
        });
    }
    digests.sort_by(|(left, _), (right, _)| left.as_bytes().cmp(right.as_bytes()));
    let ordered: Vec<_> = digests.into_iter().map(|(_, digest)| digest).collect();
    let logical = logical_model_sha256(MODEL_ID, REVISION, &ordered, &[
        ("model_config", CONFIG), ("tokenizer_model", TOKENIZER),
        ("tokenizer_config", CONFIG), ("chat_template", TEMPLATE),
    ]).unwrap();
    write(&FnlpqWriterInput {
        model_id: MODEL_ID.to_owned(), revision: REVISION.to_owned(), recipe_id: RECIPE.to_owned(),
        source_root_sha256: framed_sha256_hex("fnlpq-source-root-v1", &[b"transport fixture"]).unwrap(),
        logical_model_sha256: hex(&logical), tensors,
        sections: vec![
            SectionPayload::new("generic-payload", SectionKind::GenericTensorPayload, payload, 64),
            SectionPayload::new("generic-scales", SectionKind::GenericTensorScales, [], 8),
            SectionPayload::new("generic-row-sums", SectionKind::GenericTensorRowSums, [], 8),
            SectionPayload::new("model-config", SectionKind::ModelConfig, CONFIG, 8),
            SectionPayload::new("tokenizer-model", SectionKind::TokenizerModel, TOKENIZER, 8),
            SectionPayload::new("tokenizer-config", SectionKind::TokenizerConfig, CONFIG, 8),
            SectionPayload::new("chat-template", SectionKind::ChatTemplate, TEMPLATE, 8),
            SectionPayload::new("license-bundle", SectionKind::LicenseBundle, b"Apache-2.0\n", 8),
        ],
        packing_sets: vec![PackingSetInput {
            id: "generic".to_owned(), target: ArchTarget::Generic,
            section_names: vec!["generic-payload".to_owned(), "generic-scales".to_owned(), "generic-row-sums".to_owned()],
        }],
    }).expect("structural current-format fixture writes").bytes
}

fn census_digest() -> String {
    let census = expected_nanbeige42_census();
    let mut hasher = Sha256::new();
    hasher.update(b"fnlpq-census-v1\0");
    hasher.update((census.len() as u64).to_le_bytes());
    for tensor in census {
        hasher.update((tensor.name.len() as u64).to_le_bytes());
        hasher.update(tensor.name.as_bytes());
        hasher.update(b"BF16");
        hasher.update((tensor.shape.len() as u64).to_le_bytes());
        let mut bytes = 2_u64;
        for dimension in tensor.shape {
            hasher.update(dimension.to_le_bytes());
            bytes = bytes.checked_mul(dimension).unwrap();
        }
        hasher.update(bytes.to_le_bytes());
    }
    hex(&hasher.finalize())
}

fn digest(bytes: &[u8]) -> String { hex(&Sha256::digest(bytes)) }

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes { write!(&mut output, "{byte:02x}").unwrap(); }
    output
}
