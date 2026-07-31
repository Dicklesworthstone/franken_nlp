use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use franken_nlp::execution_identity::{
    ExecutionIdentity, IdentityError, IdentityField, IdentityProjection, NumericsProfile,
    ProvenanceIdentity, PublisherAttestationStatus, Sha256Digest, ThinkingMode, ToolMode,
};

fn digest(label: &str) -> Sha256Digest {
    Sha256Digest::of_bytes(label.as_bytes())
}

fn identity() -> ExecutionIdentity {
    ExecutionIdentity::new(ExecutionIdentity {
        schema_version: 1,
        source_revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
        logical_model_digest: digest("logical-model"),
        artifact_format: "fnlpq-v1".to_owned(),
        quant_recipe: "int8-symmetric-v1".to_owned(),
        packing_set_digest: digest("packing-set"),
        tokenizer_digest: digest("tokenizer"),
        template_digest: digest("template"),
        task_spec: "extract-v1".to_owned(),
        taskir_digest: digest("taskir"),
        prompt_digest: digest("prompt"),
        grammar_compiler_version: "grammar-v1".to_owned(),
        schema_digest: digest("schema"),
        numerics_profile: NumericsProfile::Fast { version: 1 },
        kv_dtype: "bf16".to_owned(),
        sampler_version: "greedy-v1".to_owned(),
        thinking_mode: ThinkingMode::Enabled,
        tool_mode: ToolMode::Json,
        calibration_digest: digest("calibration"),
        decision_policy_digest: digest("decision-policy"),
        backend_semantic_version: "cpu-kernel-v1".to_owned(),
        host_class: Some("zen3-avx2".to_owned()),
        compiler_identity: Some("rustc-nightly-2026-07-31".to_owned()),
    })
    .expect("fully populated fast profile is valid")
}

fn mutate(identity: &ExecutionIdentity, field: IdentityField) -> ExecutionIdentity {
    let mut mutated = identity.clone();
    match field {
        IdentityField::SchemaVersion => mutated.schema_version += 1,
        IdentityField::SourceRevision => mutated.source_revision.push_str("-changed"),
        IdentityField::LogicalModelDigest => {
            mutated.logical_model_digest = digest("logical-model-changed")
        }
        IdentityField::ArtifactFormat => mutated.artifact_format.push_str("-changed"),
        IdentityField::QuantRecipe => mutated.quant_recipe.push_str("-changed"),
        IdentityField::PackingSetDigest => {
            mutated.packing_set_digest = digest("packing-set-changed")
        }
        IdentityField::TokenizerDigest => mutated.tokenizer_digest = digest("tokenizer-changed"),
        IdentityField::TemplateDigest => mutated.template_digest = digest("template-changed"),
        IdentityField::TaskSpec => mutated.task_spec.push_str("-changed"),
        IdentityField::TaskirDigest => mutated.taskir_digest = digest("taskir-changed"),
        IdentityField::PromptDigest => mutated.prompt_digest = digest("prompt-changed"),
        IdentityField::GrammarCompilerVersion => {
            mutated.grammar_compiler_version.push_str("-changed")
        }
        IdentityField::SchemaDigest => mutated.schema_digest = digest("schema-changed"),
        IdentityField::NumericsProfile => {
            mutated.numerics_profile = NumericsProfile::Fast { version: 2 }
        }
        IdentityField::KvDtype => mutated.kv_dtype = "fp16".to_owned(),
        IdentityField::SamplerVersion => mutated.sampler_version.push_str("-changed"),
        IdentityField::ThinkingMode => mutated.thinking_mode = ThinkingMode::Disabled,
        IdentityField::ToolMode => mutated.tool_mode = ToolMode::Xml,
        IdentityField::CalibrationDigest => {
            mutated.calibration_digest = digest("calibration-changed")
        }
        IdentityField::DecisionPolicyDigest => {
            mutated.decision_policy_digest = digest("decision-policy-changed");
        }
        IdentityField::BackendSemanticVersion => {
            mutated.backend_semantic_version.push_str("-changed")
        }
        IdentityField::HostClass => mutated.host_class = Some("m4-neon".to_owned()),
        IdentityField::CompilerIdentity => {
            mutated.compiler_identity = Some("rustc-other".to_owned())
        }
    }
    mutated
}

fn compatibility_matrix() -> BTreeMap<String, [bool; 5]> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("docs/COMPATIBILITY.md");
    let document = fs::read_to_string(&path).expect("read docs/COMPATIBILITY.md at test time");
    let mut matrix = BTreeMap::new();
    for line in document.lines() {
        let cells = line
            .trim()
            .trim_matches('|')
            .split('|')
            .map(str::trim)
            .collect::<Vec<_>>();
        if cells.len() != 7
            || !IdentityField::ALL
                .iter()
                .any(|field| field.name() == cells[0])
        {
            continue;
        }
        let mut invalidates = [false; 5];
        for (index, cell) in cells[1..6].iter().enumerate() {
            invalidates[index] = match *cell {
                "YES" => true,
                "NO" => false,
                other => panic!(
                    "COMPATIBILITY.md field={} has invalid value {other:?}",
                    cells[0]
                ),
            };
        }
        assert!(
            matrix.insert(cells[0].to_owned(), invalidates).is_none(),
            "duplicate field row {}",
            cells[0]
        );
    }
    assert_eq!(
        matrix.len(),
        IdentityField::ALL.len(),
        "COMPATIBILITY.md must document every identity field"
    );
    matrix
}

#[test]
fn every_documented_field_mutation_matches_each_projection() {
    let original = identity();
    let matrix = compatibility_matrix();
    let mut mismatches = Vec::new();
    for field in IdentityField::ALL {
        let changed = mutate(&original, field);
        if field == IdentityField::SchemaVersion {
            assert_eq!(
                changed.validate(),
                Err(IdentityError::InvalidSchemaVersion(2)),
                "schema-version mutations must reject at the versioned identity boundary"
            );
            continue;
        }
        changed
            .validate()
            .expect("non-version single-field mutation remains valid");
        let expected = matrix
            .get(field.name())
            .expect("every code field has a documentation row");
        for (index, projection) in IdentityProjection::ALL.into_iter().enumerate() {
            let before = original.projection_key(projection).expect("original key");
            let after = changed.projection_key(projection).expect("changed key");
            let actual = before != after;
            if actual != expected[index] {
                mismatches.push(format!(
                    "field={} projection={} expected={} actual={} before={} after={}",
                    field.name(),
                    projection.name(),
                    expected[index],
                    actual,
                    &before.to_hex()[..12],
                    &after.to_hex()[..12],
                ));
            }
        }
    }
    if !mismatches.is_empty() {
        for mismatch in &mismatches {
            eprintln!("EXEC_IDENTITY {mismatch}");
        }
        eprintln!("EXEC_IDENTITY RESULT=FAIL mismatches={}", mismatches.len());
        panic!("execution identity compatibility matrix mismatch");
    }
    eprintln!("EXEC_IDENTITY RESULT=PASS mismatches=0");
}

#[test]
fn projections_are_domain_separated() {
    let identity = identity();
    let mut keys = BTreeMap::new();
    for projection in IdentityProjection::ALL {
        let key = identity.projection_key(projection).expect("projection key");
        assert!(
            keys.insert(key, projection.name()).is_none(),
            "projection domains collided"
        );
    }
}

#[test]
fn canonical_identity_serialization_is_frozen() {
    let mut identity = identity();
    let frozen = "a".repeat(64);
    let digest = Sha256Digest::from_hex(&frozen).expect("fixture digest");
    identity.logical_model_digest = digest;
    identity.packing_set_digest = digest;
    identity.tokenizer_digest = digest;
    identity.template_digest = digest;
    identity.taskir_digest = digest;
    identity.prompt_digest = digest;
    identity.schema_digest = digest;
    identity.calibration_digest = digest;
    identity.decision_policy_digest = digest;
    let expected = format!(
        "{{\"artifact_format\":\"fnlpq-v1\",\"backend_semantic_version\":\"cpu-kernel-v1\",\"calibration_digest\":\"{frozen}\",\"compiler_identity\":\"rustc-nightly-2026-07-31\",\"decision_policy_digest\":\"{frozen}\",\"grammar_compiler_version\":\"grammar-v1\",\"host_class\":\"zen3-avx2\",\"kv_dtype\":\"bf16\",\"logical_model_digest\":\"{frozen}\",\"numerics_profile\":\"fast-v1\",\"packing_set_digest\":\"{frozen}\",\"prompt_digest\":\"{frozen}\",\"quant_recipe\":\"int8-symmetric-v1\",\"sampler_version\":\"greedy-v1\",\"schema_digest\":\"{frozen}\",\"schema_version\":1,\"source_revision\":\"f56ec5a9650268aa098496734743c25ea778bd2d\",\"task_spec\":\"extract-v1\",\"taskir_digest\":\"{frozen}\",\"template_digest\":\"{frozen}\",\"thinking_mode\":\"enabled\",\"tokenizer_digest\":\"{frozen}\",\"tool_mode\":\"json\"}}"
    );
    let actual = String::from_utf8(identity.canonical_json_bytes().expect("canonical bytes"))
        .expect("canonical JSON is UTF-8");
    assert_eq!(
        actual, expected,
        "schema field addition or rename changes the frozen bytes"
    );
}

#[test]
fn identity_version_and_fast_context_are_fail_closed() {
    let mut unsupported_version = identity();
    unsupported_version.schema_version += 1;
    assert!(matches!(
        unsupported_version.validate(),
        Err(franken_nlp::execution_identity::IdentityError::InvalidSchemaVersion(2))
    ));

    let mut non_fast = identity();
    non_fast.numerics_profile = NumericsProfile::HfBf16Eager;
    assert!(matches!(
        non_fast.validate(),
        Err(franken_nlp::execution_identity::IdentityError::UnexpectedHostContext)
    ));
    non_fast.host_class = None;
    non_fast.compiler_identity = None;
    non_fast
        .validate()
        .expect("non-fast profiles must omit host/compiler context");
}

#[test]
fn notice_only_provenance_change_never_changes_any_execution_projection() {
    let execution = identity();
    let before = IdentityProjection::ALL.map(|projection| {
        execution
            .projection_key(projection)
            .expect("execution semantic projection")
    });
    let mut provenance = ProvenanceIdentity {
        source_root_sha256: digest("source-root"),
        fnlpq_file_sha256: digest("fnlpq-file"),
        release_manifest_sha256: digest("release-manifest"),
        license_bundle_sha256: digest("license-bundle-before"),
        converter_provenance: "converter-v1".to_owned(),
        build_provenance: "build-v1".to_owned(),
        publisher_attestation_status: PublisherAttestationStatus::Verified,
    };
    let provenance_before = provenance.receipt_digest().expect("provenance receipt");
    provenance.license_bundle_sha256 = digest("license-bundle-corrected-notice");
    let provenance_after = provenance
        .receipt_digest()
        .expect("changed provenance receipt");
    assert_ne!(
        provenance_before, provenance_after,
        "license bundle correction is provenance-visible"
    );
    for (index, projection) in IdentityProjection::ALL.into_iter().enumerate() {
        assert_eq!(
            before[index],
            execution
                .projection_key(projection)
                .expect("unchanged semantic projection"),
            "notice-only provenance change must not alter {}",
            projection.name()
        );
    }
}
