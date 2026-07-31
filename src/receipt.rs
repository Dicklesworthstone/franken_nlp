//! Typed, privacy-preserving `.fnlpr` evidence receipts.
//!
//! A receipt is an honest metadata record, never a boolean claim that a run
//! was "verified". Its completeness grade states exactly what can be replayed,
//! while private content is represented only by domain-separated HMAC-SHA-256
//! commitments. The HMAC key deliberately has no serialization or debug path.

use std::error::Error;
use std::fmt;

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::{
    canonjson,
    execution_identity::{ExecutionIdentity, IdentityError, ProvenanceIdentity, Sha256Digest},
};

/// The fixed schema identifier emitted by this module.
pub const RECEIPT_SCHEMA: &str = "fnlpr-v1";

/// Completeness is a scope statement, never a substitute for a named check.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum ReceiptGrade {
    /// Authorized inputs, artifacts, and any commitment secret are resolvable.
    Replayable,
    /// Structure, cost, and provenance replay, while private content is absent.
    StructuralReplay,
    /// Identity is retained but callers must provide bytes and secrets.
    VerifiableIfArtifactsSupplied,
    /// Historical metadata only; no replay representation is asserted.
    AuditOnly,
}

/// Public retention availability used to validate a receipt grade.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum RetentionState {
    /// The owner-authorized retention policy can resolve this input.
    Resolvable,
    /// A future caller must provide this input through an authorized channel.
    CallerSupplies,
    /// The input is intentionally unavailable to the receipt replay path.
    NotRetained,
}

/// Public retention facts. No path, secret, or content byte appears here.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct ReceiptRetention {
    pub private_inputs: RetentionState,
    pub artifacts: RetentionState,
    pub commitment_secret: RetentionState,
}

/// The domain used to prevent input, output, and configuration commitments
/// from being interchangeable.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum CommitmentDomain {
    Input,
    Output,
    Configuration,
}

impl CommitmentDomain {
    const fn label(self) -> &'static str {
        match self {
            Self::Input => "input",
            Self::Output => "output",
            Self::Configuration => "configuration",
        }
    }
}

/// A public, domain-separated HMAC commitment to private bytes.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ContentCommitment {
    pub domain: CommitmentDomain,
    pub key_id: String,
    pub hmac_sha256: Sha256Digest,
}

/// A private per-receipt or per-job HMAC key.
///
/// The secret is intentionally neither serializable nor `Debug`; exports may
/// carry only [`ContentCommitment::key_id`].
pub struct CommitmentKey {
    key_id: String,
    secret: [u8; 32],
}

impl CommitmentKey {
    /// Construct a key with a non-secret identifier safe for receipt exports.
    pub fn new(key_id: impl Into<String>, secret: [u8; 32]) -> Result<Self, ReceiptError> {
        let key_id = key_id.into();
        validate_authority("commitment_key_id", &key_id)?;
        Ok(Self { key_id, secret })
    }

    /// Commit to private bytes without retaining a raw SHA-256 content digest.
    #[must_use]
    pub fn commit(&self, domain: CommitmentDomain, bytes: &[u8]) -> ContentCommitment {
        let mut message = Vec::with_capacity(RECIPT_HMAC_PREFIX.len() + bytes.len() + 16);
        message.extend_from_slice(RECIPT_HMAC_PREFIX);
        message.extend_from_slice(domain.label().as_bytes());
        message.push(0);
        message.extend_from_slice(bytes);
        ContentCommitment {
            domain,
            key_id: self.key_id.clone(),
            hmac_sha256: Sha256Digest::of_bytes(&hmac_sha256(&self.secret, &message)),
        }
    }
}

const RECIPT_HMAC_PREFIX: &[u8] = b"fnlpr-hmac-v1\0";
const HMAC_BLOCK_BYTES: usize = 64;

/// RFC 2104 HMAC-SHA-256, retained in-tree to keep the dependency allowlist.
#[must_use]
pub fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; 32] {
    let mut key_block = [0_u8; HMAC_BLOCK_BYTES];
    if key.len() > HMAC_BLOCK_BYTES {
        key_block[..32].copy_from_slice(&Sha256::digest(key));
    } else {
        key_block[..key.len()].copy_from_slice(key);
    }
    let mut inner = Sha256::new();
    for byte in key_block {
        inner.update([byte ^ 0x36]);
    }
    inner.update(message);
    let inner_digest = inner.finalize();

    let mut outer = Sha256::new();
    for byte in key_block {
        outer.update([byte ^ 0x5c]);
    }
    outer.update(inner_digest);
    outer.finalize().into()
}

/// One named check performed by the emitting surface.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ReceiptCheck {
    pub name: String,
    pub verdict: CheckVerdict,
    pub evidence_digest: Sha256Digest,
}

/// Typed check outcome retained in a receipt.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum CheckVerdict {
    Pass,
    Fail,
    Skipped,
}

/// Complete artifact identity taxonomy needed by a receipt.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ReceiptArtifacts {
    pub source_root_sha256: Sha256Digest,
    pub logical_model_sha256: Sha256Digest,
    pub packing_set_sha256: Sha256Digest,
    pub license_bundle_sha256: Sha256Digest,
    pub fnlpq_file_sha256: Sha256Digest,
}

/// Code provenance retained separately from semantic identity.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ReceiptCodeIdentity {
    pub binary_commit: String,
    pub converter_commit: String,
}

/// Canonical, typed public projection of one `.fnlpr` receipt.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Receipt {
    pub receipt_schema: String,
    pub grade: ReceiptGrade,
    pub retention: ReceiptRetention,
    pub execution_identity: Sha256Digest,
    pub provenance_identity: Sha256Digest,
    pub source_revision: String,
    pub artifacts: ReceiptArtifacts,
    pub code: ReceiptCodeIdentity,
    pub numerics_profile: String,
    pub recipe_id: String,
    pub prompt_template_sha256: Sha256Digest,
    pub fixture_digests: Vec<Sha256Digest>,
    pub commitments: Vec<ContentCommitment>,
    pub checks: Vec<ReceiptCheck>,
    pub evidence_links: Vec<String>,
}

impl Receipt {
    /// Construct a receipt strictly from the canonical identity pair and typed
    /// public metadata. No caller can supply an ad-hoc semantic identity tuple.
    pub fn from_identities(
        grade: ReceiptGrade,
        retention: ReceiptRetention,
        execution: &ExecutionIdentity,
        provenance: &ProvenanceIdentity,
        code: ReceiptCodeIdentity,
        fixture_digests: Vec<Sha256Digest>,
        commitments: Vec<ContentCommitment>,
        checks: Vec<ReceiptCheck>,
        evidence_links: Vec<String>,
    ) -> Result<Self, ReceiptError> {
        execution.validate().map_err(ReceiptError::Identity)?;
        let receipt = Self {
            receipt_schema: RECEIPT_SCHEMA.to_owned(),
            grade,
            retention,
            execution_identity: execution
                .receipt_identity()
                .map_err(ReceiptError::Identity)?,
            provenance_identity: provenance
                .receipt_digest()
                .map_err(ReceiptError::Identity)?,
            source_revision: execution.source_revision.clone(),
            artifacts: ReceiptArtifacts {
                source_root_sha256: provenance.source_root_sha256,
                logical_model_sha256: execution.logical_model_digest,
                packing_set_sha256: execution.packing_set_digest,
                license_bundle_sha256: provenance.license_bundle_sha256,
                fnlpq_file_sha256: provenance.fnlpq_file_sha256,
            },
            code,
            numerics_profile: execution.numerics_profile.label(),
            recipe_id: execution.quant_recipe.clone(),
            prompt_template_sha256: execution.template_digest,
            fixture_digests,
            commitments,
            checks,
            evidence_links,
        };
        receipt.validate()?;
        Ok(receipt)
    }

    /// Canonical JSON bytes suitable for a `.fnlpr` file or robot NDJSON field.
    pub fn canonical_json(&self) -> Result<String, ReceiptError> {
        self.validate()?;
        canonjson::canonical_string(self)
            .map_err(|error| ReceiptError::Canonical(error.to_string()))
    }

    /// Validate the grade/retention matrix and every public authority string.
    pub fn validate(&self) -> Result<(), ReceiptError> {
        if self.receipt_schema != RECEIPT_SCHEMA {
            return Err(ReceiptError::Schema(self.receipt_schema.clone()));
        }
        validate_grade(self.grade, self.retention)?;
        validate_commit("binary_commit", &self.code.binary_commit)?;
        validate_commit("converter_commit", &self.code.converter_commit)?;
        validate_authority("source_revision", &self.source_revision)?;
        validate_authority("numerics_profile", &self.numerics_profile)?;
        validate_authority("recipe_id", &self.recipe_id)?;
        if self.checks.is_empty() {
            return Err(ReceiptError::MissingChecks);
        }
        for check in &self.checks {
            validate_authority("check.name", &check.name)?;
        }
        for commitment in &self.commitments {
            validate_authority("commitment.key_id", &commitment.key_id)?;
        }
        for link in &self.evidence_links {
            validate_authority("evidence_link", link)?;
        }
        Ok(())
    }
}

/// Typed schema/emitter failures.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReceiptError {
    Identity(IdentityError),
    Canonical(String),
    Schema(String),
    InvalidField {
        field: &'static str,
        reason: &'static str,
    },
    GradeRetention {
        grade: ReceiptGrade,
        retention: ReceiptRetention,
    },
    MissingChecks,
}

impl fmt::Display for ReceiptError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Identity(error) => write!(formatter, "receipt identity: {error}"),
            Self::Canonical(error) => write!(formatter, "receipt canonical JSON: {error}"),
            Self::Schema(schema) => write!(formatter, "unknown receipt schema {schema:?}"),
            Self::InvalidField { field, reason } => write!(formatter, "receipt {field}: {reason}"),
            Self::GradeRetention { grade, retention } => write!(
                formatter,
                "receipt grade {grade:?} is incompatible with public retention {retention:?}"
            ),
            Self::MissingChecks => formatter.write_str("receipt requires at least one named check"),
        }
    }
}

impl Error for ReceiptError {}

fn validate_grade(grade: ReceiptGrade, retention: ReceiptRetention) -> Result<(), ReceiptError> {
    let valid = match grade {
        ReceiptGrade::Replayable => {
            retention
                == ReceiptRetention {
                    private_inputs: RetentionState::Resolvable,
                    artifacts: RetentionState::Resolvable,
                    commitment_secret: RetentionState::Resolvable,
                }
        }
        ReceiptGrade::StructuralReplay => {
            retention
                == ReceiptRetention {
                    private_inputs: RetentionState::NotRetained,
                    artifacts: RetentionState::Resolvable,
                    commitment_secret: RetentionState::NotRetained,
                }
        }
        ReceiptGrade::VerifiableIfArtifactsSupplied => {
            retention
                == ReceiptRetention {
                    private_inputs: RetentionState::CallerSupplies,
                    artifacts: RetentionState::CallerSupplies,
                    commitment_secret: RetentionState::CallerSupplies,
                }
        }
        ReceiptGrade::AuditOnly => {
            retention
                == ReceiptRetention {
                    private_inputs: RetentionState::NotRetained,
                    artifacts: RetentionState::NotRetained,
                    commitment_secret: RetentionState::NotRetained,
                }
        }
    };
    if valid {
        Ok(())
    } else {
        Err(ReceiptError::GradeRetention { grade, retention })
    }
}

fn validate_authority(field: &'static str, value: &str) -> Result<(), ReceiptError> {
    if value.is_empty() || !value.is_ascii() || value.bytes().any(|byte| byte.is_ascii_control()) {
        return Err(ReceiptError::InvalidField {
            field,
            reason: "must be non-empty printable ASCII",
        });
    }
    Ok(())
}

fn validate_commit(field: &'static str, value: &str) -> Result<(), ReceiptError> {
    if value.len() != 40
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(ReceiptError::InvalidField {
            field,
            reason: "must be a lowercase 40-hex Git commit",
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution_identity::{
        NumericsProfile, PublisherAttestationStatus, ThinkingMode, ToolMode,
    };

    fn digest(byte: u8) -> Sha256Digest {
        Sha256Digest::of_bytes(&[byte])
    }

    fn execution() -> ExecutionIdentity {
        ExecutionIdentity::new(ExecutionIdentity {
            schema_version: 1,
            source_revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
            logical_model_digest: digest(1),
            artifact_format: "fnlpq-v1".to_owned(),
            quant_recipe: "nanbeige42-int8-v1".to_owned(),
            packing_set_digest: digest(2),
            tokenizer_digest: digest(3),
            template_digest: digest(4),
            task_spec: "receipt-fixture".to_owned(),
            taskir_digest: digest(5),
            prompt_digest: digest(6),
            grammar_compiler_version: "v1".to_owned(),
            schema_digest: digest(7),
            numerics_profile: NumericsProfile::HfBf16Eager,
            kv_dtype: "bf16".to_owned(),
            sampler_version: "greedy-v1".to_owned(),
            thinking_mode: ThinkingMode::Disabled,
            tool_mode: ToolMode::None,
            calibration_digest: digest(8),
            decision_policy_digest: digest(9),
            backend_semantic_version: "native-v1".to_owned(),
            host_class: None,
            compiler_identity: None,
        })
        .expect("valid receipt execution fixture")
    }

    fn provenance() -> ProvenanceIdentity {
        ProvenanceIdentity {
            source_root_sha256: digest(10),
            fnlpq_file_sha256: digest(11),
            release_manifest_sha256: digest(12),
            license_bundle_sha256: digest(13),
            converter_provenance: "converter-fixture".to_owned(),
            build_provenance: "build-fixture".to_owned(),
            publisher_attestation_status: PublisherAttestationStatus::Pending,
        }
    }

    fn code() -> ReceiptCodeIdentity {
        ReceiptCodeIdentity {
            binary_commit: "a".repeat(40),
            converter_commit: "b".repeat(40),
        }
    }

    fn checks() -> Vec<ReceiptCheck> {
        vec![ReceiptCheck {
            name: "fixture-structural-check".to_owned(),
            verdict: CheckVerdict::Pass,
            evidence_digest: digest(14),
        }]
    }

    #[test]
    fn hmac_sha256_matches_rfc_4231_case_one() {
        let output = hmac_sha256(&[0x0b; 20], b"Hi There");
        assert_eq!(
            hex(&output),
            "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
        );
    }

    #[test]
    fn commitment_domains_are_distinct_and_export_never_contains_raw_digest() {
        let private = b"yes";
        let key = CommitmentKey::new("owner-key-2026", [7; 32]).expect("key id");
        let input = key.commit(CommitmentDomain::Input, private);
        let output = key.commit(CommitmentDomain::Output, private);
        assert_ne!(input.hmac_sha256, output.hmac_sha256);

        let receipt = Receipt::from_identities(
            ReceiptGrade::VerifiableIfArtifactsSupplied,
            ReceiptRetention {
                private_inputs: RetentionState::CallerSupplies,
                artifacts: RetentionState::CallerSupplies,
                commitment_secret: RetentionState::CallerSupplies,
            },
            &execution(),
            &provenance(),
            code(),
            vec![digest(15)],
            vec![input, output],
            checks(),
            vec!["fixture://receipt-privacy".to_owned()],
        )
        .expect("typed receipt");
        let rendered = receipt.canonical_json().expect("canonical receipt");
        assert!(rendered.contains("owner-key-2026"));
        assert!(!rendered.contains(&digest_private(private)));
    }

    #[test]
    fn grade_retention_matrix_rejects_a_replayable_claim_without_secrets() {
        let error = Receipt::from_identities(
            ReceiptGrade::Replayable,
            ReceiptRetention {
                private_inputs: RetentionState::Resolvable,
                artifacts: RetentionState::Resolvable,
                commitment_secret: RetentionState::NotRetained,
            },
            &execution(),
            &provenance(),
            code(),
            Vec::new(),
            Vec::new(),
            checks(),
            Vec::new(),
        )
        .expect_err("a replayable receipt needs its commitment secret");
        assert!(matches!(error, ReceiptError::GradeRetention { .. }));
    }

    fn digest_private(bytes: &[u8]) -> String {
        hex(&Sha256::digest(bytes))
    }

    fn hex(bytes: &[u8]) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut output = String::with_capacity(bytes.len() * 2);
        for byte in bytes {
            output.push(HEX[usize::from(*byte >> 4)] as char);
            output.push(HEX[usize::from(*byte & 0x0f)] as char);
        }
        output
    }
}
