//! Pre-network rejection matrix for the release-bound manifest contract.

use franken_nlp::{
    artifact::manifest::{
        EmbeddedReleaseManifest, FNLPQ_FORMAT_V1, LOCAL_DERIVATION_PACKING_POLICY_V1,
        NANBEIGE42_INT8_V1_COMPATIBILITY, PULL_API_V1, RELEASE_MANIFEST_SCHEMA_V1, ReleaseArtifact,
        ReleaseCompatibility, ReleaseDigests, ReleaseLifecycle, ReleaseManifest, ReleasePart,
        ReleaseSourceClosure, ReleaseSourceFile,
    },
    canonjson,
};

const SHA: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const TAG: &str = "models-nanbeige42-fnlpq-v1";
const LOGICAL_NAME: &str = "nanbeige4.2-3b.fnlpq-v1.int8.generic.fnlpq";

#[test]
fn release_manifest_rejection_matrix() {
    let valid = valid_manifest();
    let canonical = valid.canonical_bytes().expect("valid canonical manifest");
    let parsed = ReleaseManifest::parse(&canonical).expect("valid manifest parses");
    assert_eq!(parsed, valid);
    parsed
        .validate_for(NANBEIGE42_INT8_V1_COMPATIBILITY)
        .expect("current binary compatibility accepts the pinned int8 recipe");
    let digest = valid.release_manifest_sha256().expect("manifest digest");
    let embedded = EmbeddedReleaseManifest::new(
        Box::leak(canonical.clone().into_boxed_slice()),
        Box::leak(digest.clone().into_boxed_str()),
    )
    .expect("embedded bytes and digest agree");
    assert_eq!(embedded.parse().expect("embedded manifest parses"), valid);
    embedded
        .verify_release_attachment(&canonical)
        .expect("release attachment exactly matches embedded bytes");
    let mut stale_attachment = canonical.clone();
    let removed = stale_attachment.pop().expect("non-empty manifest fixture");
    stale_attachment.push(removed ^ 1);
    let attachment_error = embedded
        .verify_release_attachment(&stale_attachment)
        .expect_err("one changed attachment byte rejects");
    assert_eq!(
        attachment_error.invariant,
        "embedded-attached-manifest-identity"
    );

    let mut cases = Vec::new();
    let mut bad_tag = valid.clone();
    bad_tag.release_tag = "latest".to_owned();
    cases.push(("mutable-tag", bad_tag, "immutable-release-tag"));

    let mut bad_name = valid.clone();
    bad_name.artifact.logical_name = "../model.fnlpq".to_owned();
    cases.push(("path-name", bad_name, "logical-artifact-name"));

    let mut bad_part_id = valid.clone();
    bad_part_id.parts[0].id = 1;
    cases.push(("part-id", bad_part_id, "canonical-part-order"));

    let mut bad_sum = valid.clone();
    bad_sum.artifact.bytes = 10;
    cases.push(("part-sum", bad_sum, "part-size-sum"));

    let mut bad_url = valid.clone();
    bad_url.parts[0].mirrors = vec![format!(
        "http://github.com/Dicklesworthstone/franken_nlp/releases/download/{TAG}/{LOGICAL_NAME}.part00"
    )];
    cases.push(("https", bad_url, "https-only-url"));

    let mut bad_url_tag = valid.clone();
    bad_url_tag.parts[0].mirrors = vec![format!(
        "https://github.com/Dicklesworthstone/franken_nlp/releases/download/main/{LOGICAL_NAME}.part00"
    )];
    cases.push(("url-tag", bad_url_tag, "immutable-release-url"));

    let mut duplicate_source = valid.clone();
    duplicate_source.source.files.push(ReleaseSourceFile {
        name: "CONFIG.JSON".to_owned(),
        bytes: 1,
        sha256: SHA.to_owned(),
    });
    cases.push((
        "casefold-source",
        duplicate_source,
        "unique-casefolded-source-name",
    ));

    let mut missing_census = valid.clone();
    missing_census.digests.census_sha256.clear();
    cases.push((
        "required-census",
        missing_census,
        "required-identity-digest",
    ));

    let mut bad_format = valid.clone();
    bad_format.compatibility.fnlpq_format = "fnlpq-v0".to_owned();
    cases.push(("format", bad_format, "fnlpq-format-compatibility"));

    let mut incompatible_recipe = valid.clone();
    incompatible_recipe.compatibility.recipe_id = "nanbeige42-int4-v1".to_owned();
    let incompatibility = incompatible_recipe
        .validate_for(NANBEIGE42_INT8_V1_COMPATIBILITY)
        .expect_err("unlisted future recipe rejects before I/O");
    assert_eq!(incompatibility.invariant, "recipe-compatibility");

    let mut missing_revocation = valid.clone();
    missing_revocation.lifecycle = ReleaseLifecycle {
        state: "revoked".to_owned(),
        superseded_by: None,
        revocation_reason: None,
    };
    cases.push(("revocation", missing_revocation, "revocation-reason"));

    for (name, manifest, invariant) in &cases {
        let error = manifest.validate().expect_err("mutated manifest rejects");
        assert_eq!(error.invariant, *invariant, "case={name} error={error}");
    }

    let pretty = serde_json::to_vec_pretty(&valid).expect("pretty JSON");
    let canonical_error = ReleaseManifest::parse(&pretty).expect_err("noncanonical bytes reject");
    assert_eq!(canonical_error.invariant, "canonical-json");

    let mut unknown = serde_json::to_value(&valid).expect("value");
    unknown.as_object_mut().expect("manifest object").insert(
        "unknown_required_field".to_owned(),
        serde_json::Value::Bool(true),
    );
    let unknown_bytes = canonjson::canonical_bytes(&unknown).expect("canonical injected JSON");
    let unknown_error = ReleaseManifest::parse(&unknown_bytes).expect_err("unknown field rejects");
    assert_eq!(unknown_error.invariant, "manifest-schema");

    eprintln!(
        "MANIFEST_REJECTION_MATRIX RESULT=PASS fixtures_run={} rejected_as_designed={}",
        cases.len() + 3,
        cases.len() + 2,
    );
}

fn valid_manifest() -> ReleaseManifest {
    ReleaseManifest {
        manifest_schema: RELEASE_MANIFEST_SCHEMA_V1.to_owned(),
        model_id: "Nanbeige4.2-3B".to_owned(),
        source_revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
        artifact_id: "nanbeige42-fnlpq-v1-int8-generic".to_owned(),
        release_tag: TAG.to_owned(),
        artifact: ReleaseArtifact {
            logical_name: LOGICAL_NAME.to_owned(),
            bytes: 9,
            sha256: SHA.to_owned(),
        },
        parts: vec![ReleasePart {
            id: 0,
            name: format!("{LOGICAL_NAME}.part00"),
            bytes: 9,
            sha256: SHA.to_owned(),
            mirrors: vec![format!(
                "https://github.com/Dicklesworthstone/franken_nlp/releases/download/{TAG}/{LOGICAL_NAME}.part00"
            )],
        }],
        source: ReleaseSourceClosure {
            files: vec![ReleaseSourceFile {
                name: "config.json".to_owned(),
                bytes: 1,
                sha256: SHA.to_owned(),
            }],
            source_root_sha256: SHA.to_owned(),
        },
        compatibility: ReleaseCompatibility {
            recipe_id: "nanbeige42-int8-v1".to_owned(),
            converter_id: "fnlp-convert-v1".to_owned(),
            fnlpq_format: FNLPQ_FORMAT_V1.to_owned(),
            pull_api: PULL_API_V1.to_owned(),
        },
        digests: ReleaseDigests {
            tokenizer_sha256: SHA.to_owned(),
            template_sha256: SHA.to_owned(),
            census_sha256: SHA.to_owned(),
            logical_model_sha256: SHA.to_owned(),
            packing_set_sha256: SHA.to_owned(),
        },
        license_bundle_sha256: SHA.to_owned(),
        packing_policy: LOCAL_DERIVATION_PACKING_POLICY_V1.to_owned(),
        lifecycle: ReleaseLifecycle {
            state: "active".to_owned(),
            superseded_by: None,
            revocation_reason: None,
        },
    }
}
