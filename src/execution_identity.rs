//! Canonical execution and provenance identities.
//!
//! Every semantic cache, job, receipt, golden, and calibration artifact derives
//! from this one versioned value. Projection construction is centralized here;
//! callers must never concatenate a partial lookalike key.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::canonjson;

/// The frozen schema version for [`ExecutionIdentity`].
pub const EXECUTION_IDENTITY_SCHEMA_VERSION: u32 = 1;

const HEX: &[u8; 16] = b"0123456789abcdef";

/// A validated SHA-256 digest represented as 32 bytes.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Sha256Digest([u8; 32]);

impl Sha256Digest {
    /// Hash arbitrary bytes with SHA-256.
    #[must_use]
    pub fn of_bytes(bytes: &[u8]) -> Self {
        Self(Sha256::digest(bytes).into())
    }

    /// Parse a lowercase hexadecimal SHA-256 digest.
    pub fn from_hex(value: &str) -> Result<Self, IdentityError> {
        if value.len() != 64 {
            return Err(IdentityError::InvalidDigest(value.to_owned()));
        }
        let mut bytes = [0_u8; 32];
        for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
            let high = hex_nibble(pair[0])
                .ok_or_else(|| IdentityError::InvalidDigest(value.to_owned()))?;
            let low = hex_nibble(pair[1])
                .ok_or_else(|| IdentityError::InvalidDigest(value.to_owned()))?;
            bytes[index] = (high << 4) | low;
        }
        Ok(Self(bytes))
    }

    /// Return lowercase hexadecimal bytes suitable for canonical JSON.
    #[must_use]
    pub fn to_hex(self) -> String {
        let mut result = String::with_capacity(64);
        for byte in self.0 {
            result.push(char::from(HEX[usize::from(byte >> 4)]));
            result.push(char::from(HEX[usize::from(byte & 0x0f)]));
        }
        result
    }
}

impl Serialize for Sha256Digest {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_hex())
    }
}

impl<'de> Deserialize<'de> for Sha256Digest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::from_hex(&value).map_err(serde::de::Error::custom)
    }
}

/// The numeric behavior contract for an execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NumericsProfile {
    /// The primary HF parity authority.
    HfBf16Eager,
    /// The structural bisect oracle, not an HF-token identity claim.
    DiagnosticF32,
    /// A preregistered strict quantized profile.
    StrictQuantized { version: u32 },
    /// A performance-oriented profile whose host/compiler semantics are scoped.
    Fast { version: u32 },
}

impl NumericsProfile {
    #[must_use]
    pub fn label(&self) -> String {
        match self {
            Self::HfBf16Eager => "hf-bf16-eager".to_owned(),
            Self::DiagnosticF32 => "diagnostic-f32".to_owned(),
            Self::StrictQuantized { version } => format!("strict-quantized-v{version}"),
            Self::Fast { version } => format!("fast-v{version}"),
        }
    }

    #[must_use]
    pub fn requires_host_context(&self) -> bool {
        matches!(self, Self::Fast { .. })
    }
}

/// Whether trusted template code enabled model thinking markers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ThinkingMode {
    Disabled,
    Enabled,
}

impl ThinkingMode {
    const fn label(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Enabled => "enabled",
        }
    }
}

/// The template tool branch that contributed to the execution semantics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ToolMode {
    None,
    Xml,
    Json,
}

impl ToolMode {
    const fn label(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Xml => "xml",
            Self::Json => "json",
        }
    }
}

/// The single semantic identity constructed before request admission.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionIdentity {
    pub schema_version: u32,
    pub source_revision: String,
    pub logical_model_digest: Sha256Digest,
    pub artifact_format: String,
    pub quant_recipe: String,
    pub packing_set_digest: Sha256Digest,
    pub tokenizer_digest: Sha256Digest,
    pub template_digest: Sha256Digest,
    pub task_spec: String,
    pub taskir_digest: Sha256Digest,
    pub prompt_digest: Sha256Digest,
    pub grammar_compiler_version: String,
    pub schema_digest: Sha256Digest,
    pub numerics_profile: NumericsProfile,
    pub kv_dtype: String,
    pub sampler_version: String,
    pub thinking_mode: ThinkingMode,
    pub tool_mode: ToolMode,
    pub calibration_digest: Sha256Digest,
    pub decision_policy_digest: Sha256Digest,
    pub backend_semantic_version: String,
    pub host_class: Option<String>,
    pub compiler_identity: Option<String>,
}

impl ExecutionIdentity {
    /// Build the canonical execution identity after checking its validity domain.
    pub fn new(identity: Self) -> Result<Self, IdentityError> {
        identity.validate()?;
        Ok(identity)
    }

    /// Validate the version and the fast-profile host/compiler boundary.
    pub fn validate(&self) -> Result<(), IdentityError> {
        if self.schema_version != EXECUTION_IDENTITY_SCHEMA_VERSION {
            return Err(IdentityError::InvalidSchemaVersion(self.schema_version));
        }
        for (field, value) in [
            ("source_revision", &self.source_revision),
            ("artifact_format", &self.artifact_format),
            ("quant_recipe", &self.quant_recipe),
            ("task_spec", &self.task_spec),
            ("grammar_compiler_version", &self.grammar_compiler_version),
            ("kv_dtype", &self.kv_dtype),
            ("sampler_version", &self.sampler_version),
            ("backend_semantic_version", &self.backend_semantic_version),
        ] {
            if value.is_empty() {
                return Err(IdentityError::EmptyField(field));
            }
        }
        if self.numerics_profile.requires_host_context() {
            if self.host_class.as_deref().is_none_or(str::is_empty) {
                return Err(IdentityError::MissingFastContext("host_class"));
            }
            if self.compiler_identity.as_deref().is_none_or(str::is_empty) {
                return Err(IdentityError::MissingFastContext("compiler_identity"));
            }
        } else if self.host_class.is_some() || self.compiler_identity.is_some() {
            return Err(IdentityError::UnexpectedHostContext);
        }
        Ok(())
    }

    /// Canonical JSON bytes for the complete versioned semantic identity.
    pub fn canonical_json_bytes(&self) -> Result<Vec<u8>, IdentityError> {
        self.validate()?;
        canonical_json(&self.canonical_fields())
    }

    /// Key for the exact prefix/KV cache authority domain.
    pub fn prefix_cache_key(&self) -> Result<Sha256Digest, IdentityError> {
        self.projection_key(IdentityProjection::PrefixCache)
    }

    /// Key for durable semantic job reuse.
    pub fn semantic_job_key(&self) -> Result<Sha256Digest, IdentityError> {
        self.projection_key(IdentityProjection::SemanticJob)
    }

    /// Key written into a receipt to identify one execution's semantics.
    pub fn receipt_identity(&self) -> Result<Sha256Digest, IdentityError> {
        self.projection_key(IdentityProjection::ReceiptIdentity)
    }

    /// Key for frozen comparison/golden artifacts.
    pub fn golden_fixture_key(&self) -> Result<Sha256Digest, IdentityError> {
        self.projection_key(IdentityProjection::GoldenFixture)
    }

    /// Key for calibration artifacts and their validity domain.
    pub fn calibration_artifact_key(&self) -> Result<Sha256Digest, IdentityError> {
        self.projection_key(IdentityProjection::CalibrationArtifact)
    }

    /// Construct a key in a named, domain-separated projection.
    pub fn projection_key(
        &self,
        projection: IdentityProjection,
    ) -> Result<Sha256Digest, IdentityError> {
        self.validate()?;
        let selected = self
            .canonical_fields()
            .into_iter()
            .filter(|(name, _)| {
                projection.includes(
                    field_from_name(name),
                    self.numerics_profile.requires_host_context(),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let bytes = canonical_json(&selected)?;
        Ok(domain_digest(projection.domain_tag(), &bytes))
    }

    fn canonical_fields(&self) -> BTreeMap<&'static str, Value> {
        BTreeMap::from([
            (
                "artifact_format",
                Value::String(self.artifact_format.clone()),
            ),
            (
                "backend_semantic_version",
                Value::String(self.backend_semantic_version.clone()),
            ),
            ("calibration_digest", digest_value(self.calibration_digest)),
            (
                "compiler_identity",
                optional_string_value(&self.compiler_identity),
            ),
            (
                "decision_policy_digest",
                digest_value(self.decision_policy_digest),
            ),
            (
                "grammar_compiler_version",
                Value::String(self.grammar_compiler_version.clone()),
            ),
            ("host_class", optional_string_value(&self.host_class)),
            ("kv_dtype", Value::String(self.kv_dtype.clone())),
            (
                "logical_model_digest",
                digest_value(self.logical_model_digest),
            ),
            (
                "numerics_profile",
                Value::String(self.numerics_profile.label()),
            ),
            ("packing_set_digest", digest_value(self.packing_set_digest)),
            ("prompt_digest", digest_value(self.prompt_digest)),
            ("quant_recipe", Value::String(self.quant_recipe.clone())),
            (
                "sampler_version",
                Value::String(self.sampler_version.clone()),
            ),
            ("schema_digest", digest_value(self.schema_digest)),
            (
                "schema_version",
                Value::Number(serde_json::Number::from(self.schema_version)),
            ),
            (
                "source_revision",
                Value::String(self.source_revision.clone()),
            ),
            ("task_spec", Value::String(self.task_spec.clone())),
            ("taskir_digest", digest_value(self.taskir_digest)),
            ("template_digest", digest_value(self.template_digest)),
            (
                "thinking_mode",
                Value::String(self.thinking_mode.label().to_owned()),
            ),
            ("tokenizer_digest", digest_value(self.tokenizer_digest)),
            (
                "tool_mode",
                Value::String(self.tool_mode.label().to_owned()),
            ),
        ])
    }
}

/// Receipt-only provenance; legal bytes do not alter [`ExecutionIdentity`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProvenanceIdentity {
    pub source_root_sha256: Sha256Digest,
    pub fnlpq_file_sha256: Sha256Digest,
    pub release_manifest_sha256: Sha256Digest,
    pub license_bundle_sha256: Sha256Digest,
    pub converter_provenance: String,
    pub build_provenance: String,
    pub publisher_attestation_status: PublisherAttestationStatus,
}

impl ProvenanceIdentity {
    /// Domain-separated receipt digest for artifact/release provenance.
    pub fn receipt_digest(&self) -> Result<Sha256Digest, IdentityError> {
        if self.converter_provenance.is_empty() {
            return Err(IdentityError::EmptyField("converter_provenance"));
        }
        if self.build_provenance.is_empty() {
            return Err(IdentityError::EmptyField("build_provenance"));
        }
        let fields = BTreeMap::from([
            (
                "build_provenance",
                Value::String(self.build_provenance.clone()),
            ),
            (
                "converter_provenance",
                Value::String(self.converter_provenance.clone()),
            ),
            ("fnlpq_file_sha256", digest_value(self.fnlpq_file_sha256)),
            (
                "license_bundle_sha256",
                digest_value(self.license_bundle_sha256),
            ),
            (
                "publisher_attestation_status",
                Value::String(self.publisher_attestation_status.label().to_owned()),
            ),
            (
                "release_manifest_sha256",
                digest_value(self.release_manifest_sha256),
            ),
            ("source_root_sha256", digest_value(self.source_root_sha256)),
        ]);
        Ok(domain_digest(
            "fnlp-provenance-receipt-v1",
            &canonical_json(&fields)?,
        ))
    }
}

/// Attestation state belongs to provenance, not logical model semantics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PublisherAttestationStatus {
    Verified,
    Pending,
    Absent,
}

impl PublisherAttestationStatus {
    const fn label(self) -> &'static str {
        match self {
            Self::Verified => "verified",
            Self::Pending => "pending",
            Self::Absent => "absent",
        }
    }
}

/// A field in the v1 identity schema, used to mechanically check compatibility.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum IdentityField {
    SchemaVersion,
    SourceRevision,
    LogicalModelDigest,
    ArtifactFormat,
    QuantRecipe,
    PackingSetDigest,
    TokenizerDigest,
    TemplateDigest,
    TaskSpec,
    TaskirDigest,
    PromptDigest,
    GrammarCompilerVersion,
    SchemaDigest,
    NumericsProfile,
    KvDtype,
    SamplerVersion,
    ThinkingMode,
    ToolMode,
    CalibrationDigest,
    DecisionPolicyDigest,
    BackendSemanticVersion,
    HostClass,
    CompilerIdentity,
}

impl IdentityField {
    pub const ALL: [Self; 23] = [
        Self::SchemaVersion,
        Self::SourceRevision,
        Self::LogicalModelDigest,
        Self::ArtifactFormat,
        Self::QuantRecipe,
        Self::PackingSetDigest,
        Self::TokenizerDigest,
        Self::TemplateDigest,
        Self::TaskSpec,
        Self::TaskirDigest,
        Self::PromptDigest,
        Self::GrammarCompilerVersion,
        Self::SchemaDigest,
        Self::NumericsProfile,
        Self::KvDtype,
        Self::SamplerVersion,
        Self::ThinkingMode,
        Self::ToolMode,
        Self::CalibrationDigest,
        Self::DecisionPolicyDigest,
        Self::BackendSemanticVersion,
        Self::HostClass,
        Self::CompilerIdentity,
    ];

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::SchemaVersion => "schema_version",
            Self::SourceRevision => "source_revision",
            Self::LogicalModelDigest => "logical_model_digest",
            Self::ArtifactFormat => "artifact_format",
            Self::QuantRecipe => "quant_recipe",
            Self::PackingSetDigest => "packing_set_digest",
            Self::TokenizerDigest => "tokenizer_digest",
            Self::TemplateDigest => "template_digest",
            Self::TaskSpec => "task_spec",
            Self::TaskirDigest => "taskir_digest",
            Self::PromptDigest => "prompt_digest",
            Self::GrammarCompilerVersion => "grammar_compiler_version",
            Self::SchemaDigest => "schema_digest",
            Self::NumericsProfile => "numerics_profile",
            Self::KvDtype => "kv_dtype",
            Self::SamplerVersion => "sampler_version",
            Self::ThinkingMode => "thinking_mode",
            Self::ToolMode => "tool_mode",
            Self::CalibrationDigest => "calibration_digest",
            Self::DecisionPolicyDigest => "decision_policy_digest",
            Self::BackendSemanticVersion => "backend_semantic_version",
            Self::HostClass => "host_class",
            Self::CompilerIdentity => "compiler_identity",
        }
    }
}

/// Named projection constructors; their keys are domain separated.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IdentityProjection {
    PrefixCache,
    SemanticJob,
    ReceiptIdentity,
    GoldenFixture,
    CalibrationArtifact,
}

impl IdentityProjection {
    pub const ALL: [Self; 5] = [
        Self::PrefixCache,
        Self::SemanticJob,
        Self::ReceiptIdentity,
        Self::GoldenFixture,
        Self::CalibrationArtifact,
    ];

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::PrefixCache => "prefix_cache",
            Self::SemanticJob => "semantic_job",
            Self::ReceiptIdentity => "receipt_identity",
            Self::GoldenFixture => "golden_fixture",
            Self::CalibrationArtifact => "calibration_artifact",
        }
    }

    const fn domain_tag(self) -> &'static str {
        match self {
            Self::PrefixCache => "fnlp-prefix-cache-v1",
            Self::SemanticJob => "fnlp-semantic-job-v1",
            Self::ReceiptIdentity => "fnlp-receipt-identity-v1",
            Self::GoldenFixture => "fnlp-golden-fixture-v1",
            Self::CalibrationArtifact => "fnlp-calibration-artifact-v1",
        }
    }

    const fn includes(self, field: IdentityField, fast_profile: bool) -> bool {
        use IdentityField::{
            ArtifactFormat, BackendSemanticVersion, CalibrationDigest, CompilerIdentity,
            DecisionPolicyDigest, GrammarCompilerVersion, HostClass, KvDtype, LogicalModelDigest,
            NumericsProfile, PackingSetDigest, PromptDigest, QuantRecipe, SamplerVersion,
            SchemaDigest, SchemaVersion, SourceRevision, TaskSpec, TaskirDigest, TemplateDigest,
            ThinkingMode, TokenizerDigest, ToolMode,
        };
        match self {
            Self::PrefixCache => {
                matches!(
                    field,
                    SchemaVersion
                        | SourceRevision
                        | LogicalModelDigest
                        | ArtifactFormat
                        | QuantRecipe
                        | PackingSetDigest
                        | TokenizerDigest
                        | TemplateDigest
                        | PromptDigest
                        | GrammarCompilerVersion
                        | SchemaDigest
                        | NumericsProfile
                        | KvDtype
                        | ThinkingMode
                        | ToolMode
                        | BackendSemanticVersion
                ) || (fast_profile && matches!(field, HostClass | CompilerIdentity))
            }
            Self::SemanticJob | Self::ReceiptIdentity => true,
            // A golden is only authoritative for this complete semantic
            // identity.  In particular, a quantized, differently packed, or
            // differently calibrated execution must never reuse an oracle
            // golden merely because its source model digest matches.
            Self::GoldenFixture => true,
            Self::CalibrationArtifact => {
                matches!(
                    field,
                    SchemaVersion
                        | SourceRevision
                        | LogicalModelDigest
                        | ArtifactFormat
                        | QuantRecipe
                        | PackingSetDigest
                        | TokenizerDigest
                        | TemplateDigest
                        | TaskSpec
                        | TaskirDigest
                        | PromptDigest
                        | GrammarCompilerVersion
                        | SchemaDigest
                        | NumericsProfile
                        | KvDtype
                        | SamplerVersion
                        | ThinkingMode
                        | ToolMode
                        | CalibrationDigest
                        | DecisionPolicyDigest
                        | BackendSemanticVersion
                ) || (fast_profile && matches!(field, HostClass | CompilerIdentity))
            }
        }
    }
}

/// Rejections from the versioned identity boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum IdentityError {
    InvalidDigest(String),
    InvalidSchemaVersion(u32),
    EmptyField(&'static str),
    MissingFastContext(&'static str),
    UnexpectedHostContext,
    Serialization(String),
}

impl fmt::Display for IdentityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDigest(value) => {
                write!(formatter, "invalid lowercase SHA-256 digest: {value}")
            }
            Self::InvalidSchemaVersion(observed) => {
                write!(
                    formatter,
                    "execution identity schema version must equal v{EXECUTION_IDENTITY_SCHEMA_VERSION}, observed {observed}"
                )
            }
            Self::EmptyField(field) => write!(
                formatter,
                "execution identity field {field} must not be empty"
            ),
            Self::MissingFastContext(field) => {
                write!(formatter, "fast numerics profile requires {field}")
            }
            Self::UnexpectedHostContext => formatter.write_str(
                "host_class and compiler_identity are only valid for fast numerics profiles",
            ),
            Self::Serialization(message) => write!(
                formatter,
                "canonical identity serialization failed: {message}"
            ),
        }
    }
}

impl Error for IdentityError {}

fn hex_nibble(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        _ => None,
    }
}

fn field_from_name(name: &str) -> IdentityField {
    match name {
        "schema_version" => IdentityField::SchemaVersion,
        "source_revision" => IdentityField::SourceRevision,
        "logical_model_digest" => IdentityField::LogicalModelDigest,
        "artifact_format" => IdentityField::ArtifactFormat,
        "quant_recipe" => IdentityField::QuantRecipe,
        "packing_set_digest" => IdentityField::PackingSetDigest,
        "tokenizer_digest" => IdentityField::TokenizerDigest,
        "template_digest" => IdentityField::TemplateDigest,
        "task_spec" => IdentityField::TaskSpec,
        "taskir_digest" => IdentityField::TaskirDigest,
        "prompt_digest" => IdentityField::PromptDigest,
        "grammar_compiler_version" => IdentityField::GrammarCompilerVersion,
        "schema_digest" => IdentityField::SchemaDigest,
        "numerics_profile" => IdentityField::NumericsProfile,
        "kv_dtype" => IdentityField::KvDtype,
        "sampler_version" => IdentityField::SamplerVersion,
        "thinking_mode" => IdentityField::ThinkingMode,
        "tool_mode" => IdentityField::ToolMode,
        "calibration_digest" => IdentityField::CalibrationDigest,
        "decision_policy_digest" => IdentityField::DecisionPolicyDigest,
        "backend_semantic_version" => IdentityField::BackendSemanticVersion,
        "host_class" => IdentityField::HostClass,
        "compiler_identity" => IdentityField::CompilerIdentity,
        _ => unreachable!("canonical field name is fixed by ExecutionIdentity"),
    }
}

fn digest_value(digest: Sha256Digest) -> Value {
    Value::String(digest.to_hex())
}

fn optional_string_value(value: &Option<String>) -> Value {
    value.clone().map_or(Value::Null, Value::String)
}

fn canonical_json(fields: &BTreeMap<&'static str, Value>) -> Result<Vec<u8>, IdentityError> {
    canonjson::canonical_bytes(fields)
        .map_err(|error| IdentityError::Serialization(error.to_string()))
}

fn domain_digest(domain_tag: &str, canonical_bytes: &[u8]) -> Sha256Digest {
    let mut digest = Sha256::new();
    digest.update(domain_tag.as_bytes());
    digest.update([0]);
    digest.update(canonical_bytes);
    Sha256Digest(digest.finalize().into())
}
