//! Real filesystem reconstruction of a small, current-format release package.
//!
//! The public writer, package builder, manifest parser, range reader and
//! reconstruction API are exercised together. The 201 tensor declarations use
//! the frozen census shapes and tiny structural payloads, following the existing
//! fnlpq_census_on_load fixture. They are not runnable model weights or format
//! qualification. Fixed release chunking remains unchanged: this small artifact
//! is one legitimate tail part.

#[path = "support/release_conversion.rs"]
mod release_conversion;

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
            verify_local_release_cache,
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
fn verified_final_cache_hit_needs_neither_release_parts_nor_retained_staging() {
    let fixture = Fixture::new("final-cache-hit");
    let request = fixture.request("installed");
    let reconstructed = reconstruct_local_release(&request).unwrap();
    fs::rename(&fixture.parts_dir, fixture.root.join("retained-parts")).unwrap();
    fs::rename(&request.staging_output, fixture.root.join("retained-staging")).unwrap();
    let report = verify_local_release_cache(
        &request.manifest, &request.expected_release_manifest_sha256, &request.output, |_| None,
    ).unwrap().expect("authenticated final artifact is reusable independently of its source parts");
    assert_eq!(report.artifact_sha256, reconstructed.artifact_sha256);
    assert_eq!(report.release_manifest_sha256, reconstructed.release_manifest_sha256);
    assert_eq!(report.logical_model_sha256, reconstructed.logical_model_sha256);
    assert_eq!(report.artifact_bytes, fixture.artifact.len() as u64);
    assert_eq!(report.part_count, 1);
    assert!(!report.activation_authority);
    assert_eq!(fs::read(&request.output).unwrap(), fixture.artifact);
    assert!(!fixture.parts_dir.exists());
    assert!(!request.staging_output.exists());
}

#[test]
fn published_staging_cannot_be_resumed_as_mutable_but_final_cache_is_reusable() {
    let fixture = Fixture::new("published-stage-alias");
    let request = fixture.request("published");
    reconstruct_local_release(&request).unwrap();
    let mut retry = request.clone();
    retry.output = fixture.root.join("different-final.fnlpq");
    let error = reconstruct_local_release_with_options(
        &retry, &resume_options(None), |_| None,
    ).expect_err("publication leaves a retained hard link, never a reusable mutable inode");
    assert!(matches!(error, PullError::StagingChanged(_)));
    assert_eq!(fs::read(&request.output).unwrap(), fixture.artifact);
    assert_eq!(fs::read(&request.staging_output).unwrap(), fixture.artifact);
    assert!(!retry.output.exists());
    let cached = verify_local_release_cache(&request.manifest,
        &request.expected_release_manifest_sha256, &request.output, |_| None,
    ).unwrap().expect("readonly final verification permits its retained publication link");
    assert_eq!(cached.artifact_sha256, fixture.manifest.artifact.sha256);
}

#[test]
fn linked_partial_staging_refuses_before_copy_and_preserves_both_names() {
    let fixture = Fixture::new("linked-resume-prefix");
    let request = fixture.request("resume");
    let prior = fixture.root.join("prior-retained-file");
    let prefix = &fixture.artifact[..137];
    fs::write(&prior, prefix).unwrap();
    fs::hard_link(&prior, &request.staging_output).unwrap();
    assert!(matches!(reconstruct_local_release_with_options(
        &request, &resume_options(None), |_| None,
    ), Err(PullError::StagingChanged(_))));
    assert_eq!(fs::read(&prior).unwrap(), prefix);
    assert_eq!(fs::read(&request.staging_output).unwrap(), prefix);
    assert!(!request.output.exists());
}

#[test]
fn copy_rechecks_mutable_ownership_after_resume_authentication() {
    let fixture = Fixture::new("late-resume-alias");
    let request = fixture.request("resume");
    let alias = fixture.root.join("retained-prefix-alias");
    let prefix = &fixture.artifact[..137];
    fs::write(&request.staging_output, prefix).unwrap();
    let mut linked = false;
    let result = reconstruct_local_release_with_options(
        &request, &resume_options(None), |progress| {
            if progress.phase == LocalReconstructPhase::Copy && !linked {
                fs::hard_link(&request.staging_output, &alias).unwrap();
                linked = true;
            }
            None
        },
    );
    assert!(linked, "introduce the alias only after the retained prefix is authenticated");
    assert!(matches!(result, Err(PullError::StagingChanged(_))));
    assert_eq!(fs::read(&request.staging_output).unwrap(), prefix);
    assert_eq!(fs::read(&alias).unwrap(), prefix);
    assert!(!request.output.exists());
}

#[test]
fn absent_final_is_a_cache_miss_only_after_manifest_authentication() {
    let fixture = Fixture::new("cache-miss");
    let request = fixture.request("absent");
    assert_eq!(verify_local_release_cache(
        &request.manifest, &request.expected_release_manifest_sha256, &request.output, |_| None,
    ).unwrap(), None);
    let error = verify_local_release_cache(
        &request.manifest, &"00".repeat(32), &request.output, |_| None,
    ).expect_err("missing output must not bypass the independent authority pin");
    assert!(matches!(error, PullError::ManifestDigestMismatch { .. }));
    assert!(!request.output.exists());
    assert!(!request.staging_output.exists());
}

#[test]
fn pinned_manifest_cannot_misidentify_checked_resources_on_publication_or_cache_hit() {
    let fixture = Fixture::new("release-resource-identities");
    let existing = fixture.root.join(LOGICAL_NAME);
    for field in ["digests.tokenizer_sha256", "digests.template_sha256", "digests.census_sha256"] {
        let mut manifest = fixture.manifest.clone();
        let identity = match field {
            "digests.tokenizer_sha256" => &mut manifest.digests.tokenizer_sha256,
            "digests.template_sha256" => &mut manifest.digests.template_sha256,
            _ => &mut manifest.digests.census_sha256,
        };
        *identity = "00".repeat(32);
        let mut request = fixture.request(field);
        request.manifest = fixture.root.join(format!("{field}.json"));
        fs::write(&request.manifest, manifest.canonical_bytes().unwrap()).unwrap();
        request.expected_release_manifest_sha256 = manifest.release_manifest_sha256().unwrap();

        let cache = verify_local_release_cache(
            &request.manifest, &request.expected_release_manifest_sha256, &existing, |_| None,
        );
        assert!(matches!(cache, Err(PullError::ContainerIdentity(actual)) if actual == field));
        let publication = reconstruct_local_release(&request);
        assert!(matches!(publication, Err(PullError::ContainerIdentity(actual)) if actual == field));
        assert!(!request.output.exists());
        assert_eq!(fs::read(request.staging_output).unwrap(), fixture.artifact);
    }
    assert_eq!(fs::read(existing).unwrap(), fixture.artifact);
}

#[test]
fn existing_final_cache_rehashes_bytes_and_refuses_damage_without_mutation() {
    let fixture = Fixture::new("cache-corruption");
    let request = fixture.request("damaged");
    let mut damaged = fixture.artifact.clone();
    let last = damaged.len() - 1;
    damaged[last] ^= 1;
    fs::write(&request.output, &damaged).unwrap();
    let error = verify_local_release_cache(
        &request.manifest, &request.expected_release_manifest_sha256, &request.output, |_| None,
    ).expect_err("matching file length is never cache evidence");
    assert!(matches!(error, PullError::ArtifactDigest { .. }));
    assert_eq!(fs::read(&request.output).unwrap(), damaged);
    let truncated = fixture.root.join("truncated.fnlpq");
    fs::write(&truncated, &fixture.artifact[..137]).unwrap();
    let error = verify_local_release_cache(
        &request.manifest, &request.expected_release_manifest_sha256, &truncated, |_| None,
    ).expect_err("partial artifact is a refusal, never a miss that authorizes replacement");
    assert!(matches!(error, PullError::ArtifactLength { observed: 137, .. }));
    assert_eq!(fs::read(truncated).unwrap(), fixture.artifact[..137]);
    assert!(!request.staging_output.exists());
}

#[test]
fn final_cache_rechecks_manifest_bound_semantic_identity_after_whole_file_digest() {
    let fixture = Fixture::new("cache-semantic-identity");
    let mut manifest = fixture.manifest.clone();
    manifest.digests.logical_model_sha256 = "00".repeat(32);
    let manifest_path = fixture.root.join("different-identity.manifest.json");
    fs::write(&manifest_path, manifest.canonical_bytes().unwrap()).unwrap();
    let error = verify_local_release_cache(
        &manifest_path, &manifest.release_manifest_sha256().unwrap(),
        &fixture.root.join(LOGICAL_NAME), |_| None,
    ).expect_err("whole-file bytes alone do not satisfy the authenticated semantic identity");
    assert!(matches!(error, PullError::ContainerIdentity("digests.logical_model_sha256")));
}

#[test]
fn final_cache_verification_preserves_typed_cancellation_before_io_and_during_rehash() {
    let fixture = Fixture::new("cache-cancellation");
    let request = fixture.request("cancelled");
    let error = verify_local_release_cache(
        &fixture.root.join("unopened-manifest.json"), &request.expected_release_manifest_sha256,
        &request.output, |_| Some(DecodeCancellationKind::Deadline),
    ).expect_err("initial cancellation wins without reading a manifest");
    assert!(matches!(error, PullError::Cancelled { cause: DecodeCancellationKind::Deadline, .. }));
    let mut checkpoints = 0;
    let path = fixture.root.join(LOGICAL_NAME);
    let error = verify_local_release_cache(
        &request.manifest, &request.expected_release_manifest_sha256, &path,
        |progress| {
            assert_eq!(progress.phase, LocalReconstructPhase::VerifyArtifact);
            assert_eq!(progress.copied_bytes, 0);
            checkpoints += 1;
            (checkpoints == 5).then_some(DecodeCancellationKind::Shutdown)
        },
    ).expect_err("whole-file hashing remains cooperatively cancellable");
    assert!(matches!(error, PullError::Cancelled { cause: DecodeCancellationKind::Shutdown, .. }));
    assert_eq!(fs::read(path).unwrap(), fixture.artifact);
    assert!(!request.staging_output.exists());
}

#[cfg(unix)]
#[test]
fn final_cache_refuses_symlinks_and_observed_path_replacement() {
    use std::os::unix::fs::symlink;
    let fixture = Fixture::new("cache-path-identity");
    let request = fixture.request("symlink");
    let source = fixture.root.join(LOGICAL_NAME);
    symlink(&source, &request.output).unwrap();
    assert!(matches!(verify_local_release_cache(
        &request.manifest, &request.expected_release_manifest_sha256, &request.output, |_| None,
    ), Err(PullError::ExistingArtifact { .. })));
    let dangling = fixture.root.join("dangling.fnlpq");
    symlink(fixture.root.join("never-created"), &dangling).unwrap();
    assert!(matches!(verify_local_release_cache(
        &request.manifest, &request.expected_release_manifest_sha256, &dangling, |_| None,
    ), Err(PullError::ExistingArtifact { .. })));
    let retained = fixture.root.join("retained-original.fnlpq");
    let mut checkpoints = 0;
    let error = verify_local_release_cache(
        &request.manifest, &request.expected_release_manifest_sha256, &source,
        |_| {
            checkpoints += 1;
            if checkpoints == 5 {
                fs::rename(&source, &retained).unwrap();
                fs::write(&source, &fixture.artifact).unwrap();
            }
            None
        },
    ).expect_err("same-length, same-byte pathname replacement cannot receive the held-file receipt");
    assert!(matches!(error, PullError::StagingChanged(_)));
    assert_eq!(fs::read(retained).unwrap(), fixture.artifact);
    assert_eq!(fs::read(source).unwrap(), fixture.artifact);
}

#[cfg(feature = "asupersync-runtime")]
#[test]
fn models_acquire_reuses_verified_final_after_local_admission_without_parts_or_copy_work() {
    let fixture = Fixture::new("acquire-cache-cli");
    let parts = fixture.root.join("unopened-parts");
    let staging = fixture.root.join("uncreated-stage");
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_fnlp"))
        .args(["models", "acquire", "--manifest"])
        .arg(&fixture.manifest_path)
        .arg("--manifest-sha256").arg(fixture.manifest.release_manifest_sha256().unwrap())
        .arg("--parts-dir").arg(&parts)
        .arg("--staging-output").arg(&staging)
        .arg("--output").arg(fixture.root.join(LOGICAL_NAME))
        .args(["--memory-mib", "128", "--max-download-bytes", "0", "--max-copy-bytes", "0"])
        .output().expect("invoke the real acquire route with no available parts or work budget");
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(output.status.success(), "{stderr}");
    assert!(stderr.contains("MODELS_ACQUIRE RESULT=PASS"), "{stderr}");
    assert!(stderr.contains("cache_hit=true"), "{stderr}");
    assert!(stderr.contains("network_downloaded_bytes=0"), "{stderr}");
    assert!(stderr.contains("reconstruct_copied_bytes=0"), "{stderr}");
    assert!(stderr.contains("local_workspace_bytes=134217728"), "{stderr}");
    assert!(stderr.contains("activation_authority=false"), "{stderr}");
    assert!(!parts.exists());
    assert!(!staging.exists());
}

#[cfg(feature = "asupersync-runtime")]
#[test]
fn models_acquire_refuses_insufficient_local_memory_before_manifest_or_artifact_io() {
    let fixture = Fixture::new("acquire-local-memory-cli");
    let manifest = fixture.root.join("must-not-be-opened.manifest.json");
    let parts = fixture.root.join("unopened-parts");
    let staging = fixture.root.join("uncreated-stage");
    let artifact = fixture.root.join(LOGICAL_NAME);
    for memory_mib in ["1", "127"] {
        let output = std::process::Command::new(env!("CARGO_BIN_EXE_fnlp"))
            .args(["models", "acquire", "--manifest"])
            .arg(&manifest)
            .arg("--manifest-sha256").arg(fixture.manifest.release_manifest_sha256().unwrap())
            .arg("--parts-dir").arg(&parts)
            .arg("--staging-output").arg(&staging)
            .arg("--output").arg(&artifact)
            .args(["--memory-mib", memory_mib])
            .output().expect("invoke real acquire admission before the missing manifest is opened");
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert_eq!(output.status.code(), Some(9), "{stderr}");
        assert!(stderr.contains("MODELS_ACQUIRE RESULT=REFUSED stage=local-memory"), "{stderr}");
        assert!(!stderr.contains("stage=cache-verify"), "{stderr}");
        assert!(!manifest.exists());
        assert!(!parts.exists());
        assert!(!staging.exists());
        assert_eq!(fs::read(&artifact).unwrap(), fixture.artifact);
    }
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
fn old_domain_pin_and_changed_manifest_refuse_before_parts_or_staging_access() {
    let fixture = Fixture::new("standard-manifest-pin");
    let mut request = fixture.request("retained");
    let manifest_bytes = fs::read(&request.manifest).unwrap();
    let raw_pin = digest(&manifest_bytes);
    let old_domain_pin = former_manifest_domain_digest(&manifest_bytes);
    assert_eq!(request.expected_release_manifest_sha256, raw_pin);
    assert_ne!(raw_pin, old_domain_pin);

    let retained = fixture.artifact[..137].to_vec();
    fs::write(&request.staging_output, &retained).unwrap();
    request.parts_dir = fixture.root.join("must-not-open-parts");
    request.expected_release_manifest_sha256 = old_domain_pin.clone();
    let error = reconstruct_local_release_with_options(
        &request, &resume_options(None), |_| None,
    ).expect_err("the old domain-prefixed pin cannot authorize reconstruction");
    assert!(matches!(error, PullError::ManifestDigestMismatch { expected, observed }
        if expected == old_domain_pin && observed == raw_pin));
    let existing = fixture.root.join(LOGICAL_NAME);
    let error = verify_local_release_cache(
        &request.manifest, &old_domain_pin, &existing, |_| None,
    ).expect_err("cache reuse has the same exact raw-checksum authority");
    assert!(matches!(error, PullError::ManifestDigestMismatch { expected, observed }
        if expected == old_domain_pin && observed == raw_pin));

    let mut changed_bytes = manifest_bytes;
    changed_bytes.push(b'\n');
    request.manifest = fixture.root.join("changed.manifest.json");
    fs::write(&request.manifest, &changed_bytes).unwrap();
    request.expected_release_manifest_sha256 = raw_pin.clone();
    let error = reconstruct_local_release_with_options(
        &request, &resume_options(None), |_| None,
    ).expect_err("equivalent JSON bytes cannot inherit an unchanged authority pin");
    assert!(matches!(error, PullError::ManifestDigestMismatch { expected, observed }
        if expected == raw_pin && observed == digest(&changed_bytes)));
    assert_eq!(fs::read(&request.staging_output).unwrap(), retained);
    assert_eq!(fs::read(&existing).unwrap(), fixture.artifact);
    assert!(!request.parts_dir.exists());
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
fn reconstruction_checks_cancellation_before_manifest_io_or_staging_changes() {
    let fixture = Fixture::new("reconstruct-pre-io");
    for resume in [false, true] {
        let mut request = fixture.request(if resume { "resume" } else { "fresh" });
        request.manifest = fixture.root.join("manifest-must-not-be-opened.json");
        let mut checks = 0;
        let error = reconstruct_local_release_with_options(&request,
            &LocalReconstructOptions { resume, max_new_bytes: None }, |progress| {
                checks += 1;
                assert_eq!(progress.phase, if resume {
                    LocalReconstructPhase::VerifyPrefix
                } else { LocalReconstructPhase::Copy });
                assert_eq!(progress.artifact_bytes, 0, "manifest size is not known before IO");
                assert_eq!(progress.copied_bytes, 0);
                assert_eq!(progress.verified_prefix_bytes, 0);
                Some(DecodeCancellationKind::Deadline)
            }).expect_err("the original control wins before an absent manifest is opened");
        assert!(matches!(error, PullError::Cancelled { cause: DecodeCancellationKind::Deadline, .. }));
        assert_eq!(checks, 1);
        assert!(!request.manifest.exists());
        assert!(!request.staging_output.exists());
        assert!(!request.output.exists());
    }
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
fn cancellation_during_local_publication_preserves_verified_file_without_success() {
    let fixture = Fixture::new("local-publish-stop");
    let request = fixture.request("final");
    let error = reconstruct_local_release_with_options(&request,
        &LocalReconstructOptions::default(), |progress| {
            (progress.phase == LocalReconstructPhase::Publish && request.output.exists())
                .then_some(DecodeCancellationKind::LinkedExit)
        }).expect_err("publication does not erase a terminal invocation stop");
    assert!(matches!(error, PullError::Cancelled {
        cause: DecodeCancellationKind::LinkedExit, progress,
    } if progress.phase == LocalReconstructPhase::Publish
        && progress.copied_bytes == fixture.artifact.len() as u64));
    assert_eq!(fs::read(&request.output).unwrap(), fixture.artifact);
    assert_eq!(fs::read(&request.staging_output).unwrap(), fixture.artifact);
    let cache = verify_local_release_cache(&request.manifest,
        &request.expected_release_manifest_sha256, &request.output, |_| None)
        .unwrap().expect("a retry may authenticate the already verified publication");
    assert_eq!(cache.artifact_sha256, fixture.manifest.artifact.sha256);
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
        fs::write(&receipt, release_conversion::canonical_receipt(&artifact)).unwrap();
        let license_dir = root.join("licenses");
        fs::create_dir(&license_dir).unwrap();
        for (name, bytes) in release_conversion::LICENSES {
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
            // The caller's checksum is independent of the manifest hashing API.
            expected_release_manifest_sha256: digest(&fs::read(&self.manifest_path).unwrap()),
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
            SectionPayload::new("license-bundle", SectionKind::LicenseBundle, release_conversion::license_payload(), 8),
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

fn former_manifest_domain_digest(bytes: &[u8]) -> String {
    let mut hash = Sha256::new();
    hash.update(b"franken_nlp/release-manifest/v1\0");
    hash.update(bytes);
    hex(&hash.finalize())
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes { write!(&mut output, "{byte:02x}").unwrap(); }
    output
}

#[cfg(feature = "asupersync-runtime")]
mod direct_acquisition {
    use super::*;
    use franken_nlp::native_engine::decode::DecodeStepControl;
    use franken_nlp::artifact::pull::http::{
        NetworkPartsOptions, NetworkPartsRequest, NetworkPullError, download_release_parts,
        direct::{NetworkArtifactReport, NetworkArtifactRequest, acquire_release_artifact},
    };

    struct NeverStopped;
    impl DecodeStepControl for NeverStopped {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
    }

    fn request(fixture: &Fixture, label: &str) -> NetworkArtifactRequest {
        let local = fixture.request(label);
        NetworkArtifactRequest {
            manifest: local.manifest,
            expected_release_manifest_sha256: local.expected_release_manifest_sha256,
            staging_output: local.staging_output, output: local.output,
        }
    }

    fn acquire(request: &NetworkArtifactRequest, resume: bool)
        -> Result<NetworkArtifactReport, NetworkPullError>
    {
        let runtime = asupersync::runtime::RuntimeBuilder::current_thread().build().unwrap();
        runtime.block_on(async {
            let cx = asupersync::cx::Cx::current().unwrap();
            acquire_release_artifact(&cx, request, NetworkPartsOptions {
                resume, max_new_bytes: Some(0), ..NetworkPartsOptions::default()
            }, &mut NeverStopped).await
        })
    }

    #[test]
    fn network_and_direct_acquisition_use_the_standard_manifest_checksum() {
        let fixture = Fixture::new("network-standard-manifest-pin");
        let manifest_bytes = fs::read(&fixture.manifest_path).unwrap();
        let raw_pin = digest(&manifest_bytes);
        let old_domain_pin = former_manifest_domain_digest(&manifest_bytes);
        let mut parts = NetworkPartsRequest {
            manifest: fixture.manifest_path.clone(),
            expected_release_manifest_sha256: raw_pin.clone(),
            parts_dir: fixture.parts_dir.clone(),
        };
        let runtime = asupersync::runtime::RuntimeBuilder::current_thread().build().unwrap();
        let report = runtime.block_on(async {
            let cx = asupersync::cx::Cx::current().unwrap();
            download_release_parts(&cx, &parts, NetworkPartsOptions {
                max_new_bytes: Some(0), ..NetworkPartsOptions::default()
            }, &mut NeverStopped).await
        }).expect("standard checksum permits reusing fully verified parts without network");
        assert_eq!(report.release_manifest_sha256, raw_pin);
        assert_eq!(report.downloaded_bytes, 0);
        assert_eq!(report.reused_bytes, fixture.artifact.len() as u64);
        assert_eq!(report.complete_bytes, fixture.artifact.len() as u64);
        assert!(report.parts.iter().all(|part| part.mirror_index.is_none()));

        let mut direct = request(&fixture, "unused-staging");
        direct.output = fixture.root.join(LOGICAL_NAME);
        let cached = acquire(&direct, false).expect("direct cache uses the same raw pin");
        assert!(cached.cache_hit);
        assert_eq!(cached.release_manifest_sha256, raw_pin);
        assert_eq!(cached.downloaded_bytes, 0);

        parts.expected_release_manifest_sha256 = old_domain_pin.clone();
        parts.parts_dir = fixture.root.join("must-not-open-parts");
        let error = runtime.block_on(async {
            let cx = asupersync::cx::Cx::current().unwrap();
            download_release_parts(&cx, &parts, NetworkPartsOptions {
                max_new_bytes: Some(0), ..NetworkPartsOptions::default()
            }, &mut NeverStopped).await
        }).expect_err("old domain pin refuses before inspecting parts or constructing a request");
        assert!(matches!(error, NetworkPullError::Manifest(
            PullError::ManifestDigestMismatch { expected, observed }
        ) if expected == old_domain_pin && observed == raw_pin));
        direct.expected_release_manifest_sha256 = old_domain_pin.clone();
        assert!(matches!(acquire(&direct, false), Err(NetworkPullError::Manifest(
            PullError::ManifestDigestMismatch { expected, observed }
        )) if expected == old_domain_pin && observed == raw_pin));
        assert!(!parts.parts_dir.exists());
        assert!(!direct.staging_output.exists());
        assert_eq!(fs::read(&direct.output).unwrap(), fixture.artifact);
        assert_eq!(fs::read(fixture.part()).unwrap(), fixture.artifact);
    }

    #[test]
    fn complete_direct_staging_publishes_checked_artifact_without_parts_or_network() {
        let fixture = Fixture::new("direct-complete");
        let request = request(&fixture, "direct");
        fs::write(&request.staging_output, &fixture.artifact).unwrap();
        fs::rename(&fixture.parts_dir, fixture.root.join("unavailable-parts")).unwrap();
        let report = acquire(&request, true).expect("complete staged artifact needs no network bytes");
        assert!(!report.cache_hit);
        assert_eq!(report.resumed_bytes, fixture.artifact.len() as u64);
        assert_eq!(report.downloaded_bytes, 0);
        assert_eq!(report.retained_new_bytes, 0);
        assert_eq!(report.artifact_sha256, fixture.manifest.artifact.sha256);
        assert!(!report.activation_authority);
        assert!(!report.directory_durability_confirmed);
        assert_eq!(fs::read(&request.output).unwrap(), fixture.artifact);
        assert_eq!(fs::read(&request.staging_output).unwrap(), fixture.artifact);
        assert!(!fixture.parts_dir.exists());
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            assert_eq!(fs::metadata(&request.output).unwrap().ino(),
                fs::metadata(&request.staging_output).unwrap().ino(),
                "publication must not copy a second artifact");
        }
    }

    #[test]
    fn direct_cache_hit_ignores_staging_with_resume_disabled_and_zero_network_budget() {
        let fixture = Fixture::new("direct-cache");
        let mut request = request(&fixture, "unused");
        request.output = fixture.root.join(LOGICAL_NAME);
        let report = acquire(&request, false).unwrap();
        assert!(report.cache_hit);
        assert_eq!(report.resumed_bytes, 0);
        assert_eq!(report.downloaded_bytes, 0);
        assert_eq!(report.retained_new_bytes, 0);
        assert!(!request.staging_output.exists());
    }

    #[test]
    fn direct_published_stage_refuses_mutable_resume_while_original_cache_hit_survives() {
        let fixture = Fixture::new("direct-published-stage-alias");
        let request = request(&fixture, "published");
        fs::write(&request.staging_output, &fixture.artifact).unwrap();
        acquire(&request, true).unwrap();
        let mut retry = request.clone();
        retry.output = fixture.root.join("different-direct-final.fnlpq");
        assert!(matches!(acquire(&retry, true), Err(NetworkPullError::Manifest(
            PullError::StagingChanged(_)
        ))));
        assert_eq!(fs::read(&request.output).unwrap(), fixture.artifact);
        assert_eq!(fs::read(&request.staging_output).unwrap(), fixture.artifact);
        assert!(!retry.output.exists());
        let cache = acquire(&request, true).unwrap();
        assert!(cache.cache_hit);
        assert_eq!(cache.downloaded_bytes, 0);
        assert_eq!(cache.artifact_sha256, fixture.manifest.artifact.sha256);
    }

    #[test]
    fn direct_cancellation_after_publication_returns_no_success_and_retry_verifies_cache() {
        struct StopAfterPublication(PathBuf);
        impl DecodeStepControl for StopAfterPublication {
            fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
                self.0.exists().then_some(DecodeCancellationKind::Shutdown)
            }
        }
        let fixture = Fixture::new("direct-late-cancel");
        let request = request(&fixture, "published");
        fs::write(&request.staging_output, &fixture.artifact).unwrap();
        let mut control = StopAfterPublication(request.output.clone());
        let runtime = asupersync::runtime::RuntimeBuilder::current_thread().build().unwrap();
        let result = runtime.block_on(async {
            let cx = asupersync::cx::Cx::current().unwrap();
            let result = acquire_release_artifact(&cx, &request, NetworkPartsOptions {
                resume: true, max_new_bytes: Some(0), ..NetworkPartsOptions::default()
            }, &mut control).await;
            assert!(cx.cancel_reason().is_none(), "the invocation stop is independent of runtime cancellation");
            result
        });
        assert!(matches!(result, Err(NetworkPullError::Cancelled {
            cause: Some(asupersync::CancelKind::Shutdown), part: None,
        })));
        assert_eq!(fs::read(&request.output).unwrap(), fixture.artifact);
        assert_eq!(fs::read(&request.staging_output).unwrap(), fixture.artifact);
        let report = acquire(&request, false).unwrap();
        assert!(report.cache_hit, "a late stop never destroys already verified publication");
        assert_eq!(report.downloaded_bytes, 0);
    }

    #[test]
    fn direct_local_verification_observes_caller_control_and_preserves_stored_bytes() {
        struct StopAt { checks: usize, stop_at: usize }
        impl DecodeStepControl for StopAt {
            fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
                self.checks += 1;
                (self.checks >= self.stop_at).then_some(DecodeCancellationKind::CostBudget)
            }
        }
        let fixture = Fixture::new("direct-local-control");
        let runtime = asupersync::runtime::RuntimeBuilder::current_thread().build().unwrap();
        for cached in [false, true] {
            let mut request = request(&fixture, if cached { "cache" } else { "staging" });
            if cached { request.output = fixture.root.join(LOGICAL_NAME); }
            else { fs::write(&request.staging_output, &fixture.artifact).unwrap(); }
            let mut control = StopAt { checks: 0, stop_at: 8 };
            let result = runtime.block_on(async {
                let cx = asupersync::cx::Cx::current().unwrap();
                acquire_release_artifact(&cx, &request, NetworkPartsOptions {
                    resume: true, max_new_bytes: Some(0), ..NetworkPartsOptions::default()
                }, &mut control).await
            });
            assert!(matches!(result, Err(NetworkPullError::Cancelled {
                cause: Some(asupersync::CancelKind::CostBudget), ..
            })));
            assert_eq!(control.checks, 8, "a stopped hook is not renewed or repeatedly charged");
            assert_eq!(request.output.exists(), cached);
            if cached { assert_eq!(fs::read(&request.output).unwrap(), fixture.artifact); }
            else { assert_eq!(fs::read(&request.staging_output).unwrap(), fixture.artifact); }
        }
    }

    #[test]
    fn direct_partial_zero_budget_retains_exact_prefix_without_publication() {
        let fixture = Fixture::new("direct-budget");
        let request = request(&fixture, "partial");
        fs::write(&request.staging_output, &fixture.artifact[..173]).unwrap();
        assert!(matches!(acquire(&request, true),
            Err(NetworkPullError::DownloadBudgetExceeded {
                limit: 0, downloaded_bytes: 0, retained_bytes: 173, part: 0, ..
            })));
        assert_eq!(fs::read(&request.staging_output).unwrap(), fixture.artifact[..173]);
        assert!(!request.output.exists());
    }

    #[test]
    fn direct_bad_pin_and_no_resume_leave_retained_staging_untouched() {
        let fixture = Fixture::new("direct-pin");
        let mut request = request(&fixture, "retained");
        fs::write(&request.staging_output, &fixture.artifact).unwrap();
        let pin = request.expected_release_manifest_sha256.clone();
        request.expected_release_manifest_sha256 = "0".repeat(64);
        assert!(matches!(acquire(&request, true),
            Err(NetworkPullError::Manifest(PullError::ManifestDigestMismatch { .. }))));
        request.expected_release_manifest_sha256 = pin;
        assert!(matches!(acquire(&request, false),
            Err(NetworkPullError::Manifest(PullError::Publish { .. }))));
        assert_eq!(fs::read(&request.staging_output).unwrap(), fixture.artifact);
        assert!(!request.output.exists());
    }

    #[test]
    fn direct_corrupt_complete_part_and_existing_final_are_never_repaired_or_replaced() {
        let fixture = Fixture::new("direct-corrupt");
        let request = request(&fixture, "corrupt");
        let mut corrupt = fixture.artifact.clone();
        corrupt[100] ^= 1;
        fs::write(&request.staging_output, &corrupt).unwrap();
        assert!(matches!(acquire(&request, true), Err(NetworkPullError::Digest { part: 0, .. })));
        assert_eq!(fs::read(&request.staging_output).unwrap(), corrupt);
        assert!(!request.output.exists());
        fs::write(&request.output, b"existing-output-must-survive").unwrap();
        assert!(matches!(acquire(&request, true),
            Err(NetworkPullError::Manifest(PullError::ArtifactLength { .. }))));
        assert_eq!(fs::read(&request.output).unwrap(), b"existing-output-must-survive");
        assert_eq!(fs::read(&request.staging_output).unwrap(), corrupt);
    }

    #[test]
    fn direct_whole_artifact_and_container_identity_gates_remain_mandatory() {
        for whole_digest in [true, false] {
            let fixture = Fixture::new(if whole_digest { "direct-whole" } else { "direct-identity" });
            let mut request = request(&fixture, "checked");
            fs::write(&request.staging_output, &fixture.artifact).unwrap();
            let mut manifest = fixture.manifest.clone();
            if whole_digest { manifest.artifact.sha256 = "0".repeat(64); }
            else { manifest.digests.logical_model_sha256 = "0".repeat(64); }
            request.manifest = fixture.root.join("different.manifest.json");
            fs::write(&request.manifest, manifest.canonical_bytes().unwrap()).unwrap();
            request.expected_release_manifest_sha256 = manifest.release_manifest_sha256().unwrap();
            let error = acquire(&request, true).unwrap_err();
            if whole_digest {
                assert!(matches!(error, NetworkPullError::Manifest(PullError::ArtifactDigest { .. })));
            } else {
                assert!(matches!(error, NetworkPullError::Manifest(PullError::ContainerIdentity(
                    "digests.logical_model_sha256"))));
            }
            assert_eq!(fs::read(&request.staging_output).unwrap(), fixture.artifact);
            assert!(!request.output.exists());
        }
    }

    #[cfg(unix)]
    #[test]
    fn direct_resume_refuses_a_staging_symlink() {
        let fixture = Fixture::new("direct-symlink");
        let request = request(&fixture, "link");
        std::os::unix::fs::symlink(fixture.root.join(LOGICAL_NAME), &request.staging_output).unwrap();
        assert!(matches!(acquire(&request, true),
            Err(NetworkPullError::Manifest(PullError::InvalidPath(_)))));
        assert_eq!(fs::read(fixture.root.join(LOGICAL_NAME)).unwrap(), fixture.artifact);
        assert!(!request.output.exists());
    }

    #[test]
    fn cli_direct_cache_and_complete_resume_succeed_without_a_parts_option() {
        for cached in [false, true] {
            let fixture = Fixture::new(if cached { "direct-cli-cache" } else { "direct-cli-complete" });
            let mut request = request(&fixture, "cli");
            if cached { request.output = fixture.root.join(LOGICAL_NAME); }
            else { fs::write(&request.staging_output, &fixture.artifact).unwrap(); }
            let output = std::process::Command::new(env!("CARGO_BIN_EXE_fnlp"))
                .args(["models", "acquire", "--direct", "--manifest"])
                .arg(&request.manifest).arg("--manifest-sha256")
                .arg(&request.expected_release_manifest_sha256)
                .arg("--staging-output").arg(&request.staging_output)
                .arg("--output").arg(&request.output)
                .args(["--memory-mib", "256", "--max-download-bytes", "0"])
                .output().unwrap();
            let stderr = String::from_utf8(output.stderr).unwrap();
            assert!(output.status.success(), "{stderr}");
            assert!(stderr.contains("MODELS_ACQUIRE RESULT=PASS mode=direct"), "{stderr}");
            assert!(stderr.contains(&format!("cache_hit={cached}")), "{stderr}");
            assert!(stderr.contains("network_downloaded_bytes=0"), "{stderr}");
            assert!(stderr.contains("retained_new_bytes=0"), "{stderr}");
            assert!(stderr.contains("activation_authority=false"), "{stderr}");
            assert_eq!(fs::read(&request.output).unwrap(), fixture.artifact);
            assert_eq!(request.staging_output.exists(), !cached);
        }
    }

    #[test]
    fn cli_direct_admits_both_workspaces_before_opening_a_manifest() {
        let fixture = Fixture::new("direct-cli-memory");
        let manifest = fixture.root.join("absent.manifest.json");
        let request = request(&fixture, "unopened");
        for memory in ["1", "127", "128", "143"] {
            let output = std::process::Command::new(env!("CARGO_BIN_EXE_fnlp"))
                .args(["models", "acquire", "--direct", "--manifest"])
                .arg(&manifest).arg("--manifest-sha256")
                .arg(&request.expected_release_manifest_sha256)
                .arg("--staging-output").arg(&request.staging_output)
                .arg("--output").arg(&request.output)
                .args(["--memory-mib", memory])
                .output().unwrap();
            let stderr = String::from_utf8(output.stderr).unwrap();
            assert_eq!(output.status.code(), Some(9), "{stderr}");
            assert!(!request.staging_output.exists());
            assert!(!request.output.exists());
            assert!(!manifest.exists());
        }
    }
}
