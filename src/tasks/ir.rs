//! Bounded task intermediate representation and response envelopes.
//!
//! `TaskIR` is deliberately data, not an interpreter for user supplied code.
//! A task compiles its exact prompt-token segments and finite decode contract
//! before the model is loaded.  The execution compiler and independent
//! validator are separate dependency surfaces; this module records their
//! immutable references without importing either one's mutable state.

use std::{collections::BTreeMap, error::Error, fmt, str};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    canonjson,
    error::{FnlpError, StructuredTaskStatus},
    execution_identity::{ExecutionIdentity, IdentityError, Sha256Digest},
    grammar::SchemaNode,
    validation::{self, GroundedValue},
};

/// The frozen schema version for bounded task plans.
pub const TASK_IR_SCHEMA_VERSION: u32 = 1;
/// The frozen schema version for the semantic result envelope.
pub const SEMANTIC_ENVELOPE_SCHEMA_VERSION: u32 = 1;
/// The frozen schema version for the separately emitted telemetry envelope.
pub const TELEMETRY_ENVELOPE_SCHEMA_VERSION: u32 = 1;

/// A typed prompt segment.  `token_ids` are the exact post-tokenization
/// sequence, never a request for the runtime to re-tokenize an opaque string.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PromptSegment {
    kind: PromptSegmentKind,
    token_ids: Vec<u32>,
}

impl PromptSegment {
    /// Construct one exact prompt segment.  Empty document and scaffold
    /// segments are meaningful and remain represented rather than dropped.
    #[must_use]
    pub fn new(kind: PromptSegmentKind, token_ids: Vec<u32>) -> Self {
        Self { kind, token_ids }
    }

    /// The trusted role of this segment in the prompt ABI.
    #[must_use]
    pub const fn kind(&self) -> PromptSegmentKind {
        self.kind
    }

    /// The exact already-tokenized sequence for this segment.
    #[must_use]
    pub fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }
}

/// The closed prompt-segment ABI.  No untyped string segment may bypass it.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PromptSegmentKind {
    GlobalPolicy,
    TaskInstruction,
    Document,
    AnswerScaffold,
}

/// One finite token continuation used by candidates and stop sequences.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TokenSequence {
    token_ids: Vec<u32>,
}

impl TokenSequence {
    /// Construct a token sequence.  The containing `TaskIR` rejects empty
    /// sequences where a finite continuation is required.
    #[must_use]
    pub fn new(token_ids: Vec<u32>) -> Self {
        Self { token_ids }
    }

    #[must_use]
    pub fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }
}

/// One named finite candidate for logit-sliced or trie-scored prefill.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Candidate {
    id: String,
    continuation: TokenSequence,
}

impl Candidate {
    #[must_use]
    pub fn new(id: impl Into<String>, continuation: TokenSequence) -> Self {
        Self {
            id: id.into(),
            continuation,
        }
    }

    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    #[must_use]
    pub fn continuation(&self) -> &TokenSequence {
        &self.continuation
    }
}

/// A checked, immutable trie summary.  The execution compiler owns the actual
/// trie; `TaskIR` binds the exact compiled product by digest and finite counts.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ContinuationTrieReference {
    digest: Sha256Digest,
    node_count: u32,
    terminal_count: u32,
}

impl ContinuationTrieReference {
    #[must_use]
    pub const fn new(digest: Sha256Digest, node_count: u32, terminal_count: u32) -> Self {
        Self {
            digest,
            node_count,
            terminal_count,
        }
    }
}

/// A bounded reference to grammar/execution-program data.  The runtime never
/// receives a general-purpose evaluator through this type.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields, tag = "kind", rename_all = "snake_case")]
pub enum GrammarReference {
    None,
    JsonSchema {
        digest: Sha256Digest,
        compiler_version: String,
    },
    Pattern {
        digest: Sha256Digest,
        compiler_version: String,
    },
}

impl GrammarReference {
    #[must_use]
    pub const fn none() -> Self {
        Self::None
    }

    #[must_use]
    pub fn json_schema(digest: Sha256Digest, compiler_version: impl Into<String>) -> Self {
        Self::JsonSchema {
            digest,
            compiler_version: compiler_version.into(),
        }
    }

    #[must_use]
    pub fn pattern(digest: Sha256Digest, compiler_version: impl Into<String>) -> Self {
        Self::Pattern {
            digest,
            compiler_version: compiler_version.into(),
        }
    }

    const fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }
}

/// Bounds that are checked before task execution and allocation.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TaskBudget {
    pub max_input_tokens: u32,
    pub max_output_tokens: u32,
    pub max_output_bytes: u64,
    pub max_grammar_states: u32,
    pub max_kv_bytes: u64,
}

impl TaskBudget {
    /// Check that every resource axis is explicitly bounded.
    pub fn validate(&self) -> Result<(), TaskIrError> {
        for (field, value) in [
            ("max_input_tokens", u64::from(self.max_input_tokens)),
            ("max_output_tokens", u64::from(self.max_output_tokens)),
            ("max_output_bytes", self.max_output_bytes),
            ("max_grammar_states", u64::from(self.max_grammar_states)),
            ("max_kv_bytes", self.max_kv_bytes),
        ] {
            if value == 0 {
                return Err(TaskIrError::ZeroBudget(field));
            }
        }
        Ok(())
    }

    const fn fits_within(&self, ceiling: &Self) -> bool {
        self.max_input_tokens <= ceiling.max_input_tokens
            && self.max_output_tokens <= ceiling.max_output_tokens
            && self.max_output_bytes <= ceiling.max_output_bytes
            && self.max_grammar_states <= ceiling.max_grammar_states
            && self.max_kv_bytes <= ceiling.max_kv_bytes
    }
}

/// A strategy-specific output cap.  It may only tighten the task-wide bounds.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecodeBudget {
    pub max_tokens: u32,
    pub max_bytes: u64,
}

impl DecodeBudget {
    fn validate_within(&self, task_budget: &TaskBudget) -> Result<(), TaskIrError> {
        if self.max_tokens == 0 {
            return Err(TaskIrError::ZeroBudget("decode.max_tokens"));
        }
        if self.max_bytes == 0 {
            return Err(TaskIrError::ZeroBudget("decode.max_bytes"));
        }
        if self.max_tokens > task_budget.max_output_tokens
            || self.max_bytes > task_budget.max_output_bytes
        {
            return Err(TaskIrError::DecodeBudgetExceedsTaskBudget);
        }
        Ok(())
    }
}

/// The fixed scale used by a finite distribution task.  Integer values avoid
/// silently introducing floating-point semantics into a task plan.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DistributionScale {
    pub minimum: i32,
    pub maximum: i32,
    pub step: u32,
}

impl DistributionScale {
    fn validate(&self) -> Result<(), TaskIrError> {
        if self.minimum >= self.maximum || self.step == 0 {
            return Err(TaskIrError::InvalidDistributionScale);
        }
        let span = i64::from(self.maximum) - i64::from(self.minimum);
        if span % i64::from(self.step) != 0 {
            return Err(TaskIrError::InvalidDistributionScale);
        }
        Ok(())
    }
}

/// The only decode contracts admitted to the task layer.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields, tag = "kind", rename_all = "snake_case")]
pub enum DecodeStrategy {
    PrefillOnly {
        candidates: Vec<Candidate>,
    },
    ConstrainedJson,
    ConstrainedPattern,
    FreeText {
        stops: Vec<TokenSequence>,
        budget: DecodeBudget,
    },
    Distribution {
        scale: DistributionScale,
    },
}

/// A finite postcondition evaluated by a named, bounded product surface.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FinitePostcondition {
    JsonValid,
    MatchesGrammar,
    CandidateSetComplete,
    SourceSpansVerified,
    OutputWithinBudget,
}

/// Cache/reuse authority for the stage represented by this plan.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DependencyScope {
    ItemLocal,
    PartitionReduce,
    CorpusGlobal,
}

impl DependencyScope {
    /// Whether an unchanged item's exact stage result can survive adding a
    /// sibling item.  Reduce and corpus-global products must recompute against
    /// the changed child set/snapshot.
    #[must_use]
    pub const fn preserves_after_child_set_change(self) -> bool {
        matches!(self, Self::ItemLocal)
    }
}

/// The bounded intermediate representation consumed by the task runtime.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TaskIR {
    schema_version: u32,
    prompt_segments: Vec<PromptSegment>,
    decode_strategy: DecodeStrategy,
    grammar: GrammarReference,
    continuation_trie: Option<ContinuationTrieReference>,
    postconditions: Vec<FinitePostcondition>,
    budget: TaskBudget,
    dependency_scope: DependencyScope,
}

impl TaskIR {
    /// Construct and validate a complete, non-executable task plan.
    pub fn new(
        prompt_segments: Vec<PromptSegment>,
        decode_strategy: DecodeStrategy,
        grammar: GrammarReference,
        continuation_trie: Option<ContinuationTrieReference>,
        postconditions: Vec<FinitePostcondition>,
        budget: TaskBudget,
        dependency_scope: DependencyScope,
    ) -> Result<Self, TaskIrError> {
        let ir = Self {
            schema_version: TASK_IR_SCHEMA_VERSION,
            prompt_segments,
            decode_strategy,
            grammar,
            continuation_trie,
            postconditions,
            budget,
            dependency_scope,
        };
        ir.validate()?;
        Ok(ir)
    }

    /// Recheck this value after deserialization at an input boundary.
    pub fn validate(&self) -> Result<(), TaskIrError> {
        if self.schema_version != TASK_IR_SCHEMA_VERSION {
            return Err(TaskIrError::UnsupportedSchemaVersion(self.schema_version));
        }
        self.budget.validate()?;
        self.validate_prompt_segments()?;
        self.validate_postconditions()?;
        self.validate_grammar()?;
        self.validate_decode_strategy()?;
        self.validate_trie()?;
        Ok(())
    }

    /// Canonical bytes form the TaskIR input to `ExecutionIdentity`.
    pub fn canonical_json_bytes(&self) -> Result<Vec<u8>, TaskIrError> {
        self.validate()?;
        canonjson::canonical_bytes(self)
            .map_err(|error| TaskIrError::CanonicalJson(error.to_string()))
    }

    /// Domain-safe digest of the fully validated canonical TaskIR bytes.
    pub fn digest(&self) -> Result<Sha256Digest, TaskIrError> {
        Ok(Sha256Digest::of_bytes(&self.canonical_json_bytes()?))
    }

    #[must_use]
    pub fn prompt_segments(&self) -> &[PromptSegment] {
        &self.prompt_segments
    }

    #[must_use]
    pub const fn decode_strategy(&self) -> &DecodeStrategy {
        &self.decode_strategy
    }

    #[must_use]
    pub const fn grammar(&self) -> &GrammarReference {
        &self.grammar
    }

    #[must_use]
    pub const fn budget(&self) -> &TaskBudget {
        &self.budget
    }

    #[must_use]
    pub const fn dependency_scope(&self) -> DependencyScope {
        self.dependency_scope
    }

    fn validate_prompt_segments(&self) -> Result<(), TaskIrError> {
        if self.prompt_segments.is_empty() {
            return Err(TaskIrError::MissingPromptSegments);
        }
        if self
            .prompt_segments
            .last()
            .is_none_or(|segment| segment.kind != PromptSegmentKind::AnswerScaffold)
        {
            return Err(TaskIrError::PromptMustEndWithAnswerScaffold);
        }
        let mut prompt_tokens = 0_u32;
        for segment in &self.prompt_segments {
            let segment_len = u32::try_from(segment.token_ids.len())
                .map_err(|_| TaskIrError::PromptTokenCountOverflow)?;
            prompt_tokens = prompt_tokens
                .checked_add(segment_len)
                .ok_or(TaskIrError::PromptTokenCountOverflow)?;
        }
        if prompt_tokens == 0 {
            return Err(TaskIrError::EmptyPrompt);
        }
        if prompt_tokens > self.budget.max_input_tokens {
            return Err(TaskIrError::PromptExceedsInputBudget {
                prompt_tokens,
                max_input_tokens: self.budget.max_input_tokens,
            });
        }
        Ok(())
    }

    fn validate_postconditions(&self) -> Result<(), TaskIrError> {
        if self.postconditions.is_empty() {
            return Err(TaskIrError::MissingPostconditions);
        }
        let mut seen = std::collections::BTreeSet::new();
        for condition in &self.postconditions {
            if !seen.insert(*condition) {
                return Err(TaskIrError::DuplicatePostcondition(*condition));
            }
        }
        if !seen.contains(&FinitePostcondition::OutputWithinBudget) {
            return Err(TaskIrError::MissingOutputBudgetPostcondition);
        }
        Ok(())
    }

    fn validate_grammar(&self) -> Result<(), TaskIrError> {
        match &self.grammar {
            GrammarReference::None => Ok(()),
            GrammarReference::JsonSchema {
                compiler_version, ..
            }
            | GrammarReference::Pattern {
                compiler_version, ..
            } if compiler_version.is_empty() => Err(TaskIrError::EmptyGrammarCompilerVersion),
            GrammarReference::JsonSchema { .. } | GrammarReference::Pattern { .. } => Ok(()),
        }
    }

    fn validate_decode_strategy(&self) -> Result<(), TaskIrError> {
        let postconditions = &self.postconditions;
        match &self.decode_strategy {
            DecodeStrategy::PrefillOnly { candidates } => {
                if !self.grammar.is_none() {
                    return Err(TaskIrError::UnexpectedGrammar("prefill_only"));
                }
                if candidates.is_empty() {
                    return Err(TaskIrError::MissingCandidates);
                }
                let mut candidate_ids = std::collections::BTreeSet::new();
                for candidate in candidates {
                    if !valid_identifier(&candidate.id) {
                        return Err(TaskIrError::InvalidCandidateId(candidate.id.clone()));
                    }
                    if !candidate_ids.insert(&candidate.id) {
                        return Err(TaskIrError::DuplicateCandidateId(candidate.id.clone()));
                    }
                    if candidate.continuation.token_ids.is_empty() {
                        return Err(TaskIrError::EmptyCandidateContinuation(
                            candidate.id.clone(),
                        ));
                    }
                }
                require_postcondition(
                    postconditions,
                    FinitePostcondition::CandidateSetComplete,
                    "prefill_only",
                )
            }
            DecodeStrategy::ConstrainedJson => {
                if !matches!(self.grammar, GrammarReference::JsonSchema { .. }) {
                    return Err(TaskIrError::MissingRequiredGrammar("constrained_json"));
                }
                require_postcondition(
                    postconditions,
                    FinitePostcondition::JsonValid,
                    "constrained_json",
                )
            }
            DecodeStrategy::ConstrainedPattern => {
                if !matches!(self.grammar, GrammarReference::Pattern { .. }) {
                    return Err(TaskIrError::MissingRequiredGrammar("constrained_pattern"));
                }
                require_postcondition(
                    postconditions,
                    FinitePostcondition::MatchesGrammar,
                    "constrained_pattern",
                )
            }
            DecodeStrategy::FreeText { stops, budget } => {
                if !self.grammar.is_none() {
                    return Err(TaskIrError::UnexpectedGrammar("free_text"));
                }
                budget.validate_within(&self.budget)?;
                if stops.iter().any(|stop| stop.token_ids.is_empty()) {
                    return Err(TaskIrError::EmptyStopSequence);
                }
                Ok(())
            }
            DecodeStrategy::Distribution { scale } => {
                if !self.grammar.is_none() {
                    return Err(TaskIrError::UnexpectedGrammar("distribution"));
                }
                scale.validate()
            }
        }
    }

    fn validate_trie(&self) -> Result<(), TaskIrError> {
        match (&self.decode_strategy, &self.continuation_trie) {
            (DecodeStrategy::PrefillOnly { .. }, Some(trie)) => {
                if trie.node_count == 0
                    || trie.terminal_count == 0
                    || trie.terminal_count > trie.node_count
                {
                    return Err(TaskIrError::InvalidContinuationTrie);
                }
                Ok(())
            }
            (DecodeStrategy::PrefillOnly { .. }, None) => Ok(()),
            (_, Some(_)) => Err(TaskIrError::UnexpectedContinuationTrie),
            (_, None) => Ok(()),
        }
    }
}

fn require_postcondition(
    postconditions: &[FinitePostcondition],
    required: FinitePostcondition,
    strategy: &'static str,
) -> Result<(), TaskIrError> {
    if postconditions.contains(&required) {
        Ok(())
    } else {
        Err(TaskIrError::MissingRequiredPostcondition { required, strategy })
    }
}

fn valid_identifier(value: &str) -> bool {
    !value.is_empty()
        && value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'-' | b'_')
        })
}

/// Safe-to-log construction errors for bounded task plans.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TaskIrError {
    UnsupportedSchemaVersion(u32),
    MissingPromptSegments,
    PromptMustEndWithAnswerScaffold,
    EmptyPrompt,
    PromptTokenCountOverflow,
    PromptExceedsInputBudget {
        prompt_tokens: u32,
        max_input_tokens: u32,
    },
    ZeroBudget(&'static str),
    DecodeBudgetExceedsTaskBudget,
    InvalidDistributionScale,
    MissingPostconditions,
    DuplicatePostcondition(FinitePostcondition),
    MissingOutputBudgetPostcondition,
    EmptyGrammarCompilerVersion,
    UnexpectedGrammar(&'static str),
    MissingRequiredGrammar(&'static str),
    MissingCandidates,
    InvalidCandidateId(String),
    DuplicateCandidateId(String),
    EmptyCandidateContinuation(String),
    EmptyStopSequence,
    InvalidContinuationTrie,
    UnexpectedContinuationTrie,
    MissingRequiredPostcondition {
        required: FinitePostcondition,
        strategy: &'static str,
    },
    CanonicalJson(String),
    Identity(String),
    TaskSpec(String),
    TaskBudgetExceedsPlanContext,
}

impl fmt::Display for TaskIrError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion(version) => {
                write!(formatter, "unsupported TaskIR schema version {version}")
            }
            Self::MissingPromptSegments => {
                formatter.write_str("TaskIR requires exact prompt segments")
            }
            Self::PromptMustEndWithAnswerScaffold => {
                formatter.write_str("TaskIR prompt must end with answer_scaffold")
            }
            Self::EmptyPrompt => formatter.write_str("TaskIR prompt token sequence is empty"),
            Self::PromptTokenCountOverflow => {
                formatter.write_str("TaskIR prompt token count overflow")
            }
            Self::PromptExceedsInputBudget {
                prompt_tokens,
                max_input_tokens,
            } => write!(
                formatter,
                "TaskIR prompt tokens {prompt_tokens} exceed input budget {max_input_tokens}"
            ),
            Self::ZeroBudget(field) => write!(formatter, "TaskIR requires nonzero budget {field}"),
            Self::DecodeBudgetExceedsTaskBudget => {
                formatter.write_str("decode budget exceeds task budget")
            }
            Self::InvalidDistributionScale => {
                formatter.write_str("distribution scale is not finite and integral")
            }
            Self::MissingPostconditions => {
                formatter.write_str("TaskIR requires finite postconditions")
            }
            Self::DuplicatePostcondition(condition) => {
                write!(formatter, "duplicate TaskIR postcondition {condition:?}")
            }
            Self::MissingOutputBudgetPostcondition => {
                formatter.write_str("TaskIR requires output_within_budget postcondition")
            }
            Self::EmptyGrammarCompilerVersion => {
                formatter.write_str("grammar compiler version is empty")
            }
            Self::UnexpectedGrammar(strategy) => write!(
                formatter,
                "{strategy} decode cannot carry a grammar reference"
            ),
            Self::MissingRequiredGrammar(strategy) => write!(
                formatter,
                "{strategy} decode requires its matching grammar reference"
            ),
            Self::MissingCandidates => {
                formatter.write_str("prefill_only decode requires candidates")
            }
            Self::InvalidCandidateId(id) => write!(formatter, "invalid candidate id {id}"),
            Self::DuplicateCandidateId(id) => write!(formatter, "duplicate candidate id {id}"),
            Self::EmptyCandidateContinuation(id) => {
                write!(formatter, "candidate {id} has an empty continuation")
            }
            Self::EmptyStopSequence => formatter.write_str("free_text stop sequence is empty"),
            Self::InvalidContinuationTrie => {
                formatter.write_str("continuation trie counts are invalid")
            }
            Self::UnexpectedContinuationTrie => {
                formatter.write_str("only prefill_only decode may carry a continuation trie")
            }
            Self::MissingRequiredPostcondition { required, strategy } => write!(
                formatter,
                "{strategy} decode requires postcondition {required:?}"
            ),
            Self::CanonicalJson(error) => {
                write!(formatter, "TaskIR canonical JSON failed: {error}")
            }
            Self::Identity(error) => write!(formatter, "Task plan identity invalid: {error}"),
            Self::TaskSpec(error) => write!(formatter, "Task specification invalid: {error}"),
            Self::TaskBudgetExceedsPlanContext => {
                formatter.write_str("TaskIR budget exceeds plan-context ceiling")
            }
        }
    }
}

impl Error for TaskIrError {}

impl From<IdentityError> for TaskIrError {
    fn from(value: IdentityError) -> Self {
        Self::Identity(value.to_string())
    }
}

impl TaskIrError {
    /// The task/recipe compile failure category at the public error boundary.
    #[must_use]
    pub fn into_fnlp_error(self) -> FnlpError {
        FnlpError::SchemaOrRecipeCompile {
            category: "task_ir_validation",
        }
    }
}

/// Static task metadata.  Schemas remain immutable references; independent
/// schema compilation belongs to the validator/compiler dependency surfaces.
#[derive(Debug, Eq, PartialEq)]
pub struct TaskSpec {
    name: &'static str,
    version: &'static str,
    request_schema: &'static str,
    response_schema: &'static str,
    presets: &'static [&'static str],
}

impl TaskSpec {
    /// Define a static built-in task specification.
    pub const fn new(
        name: &'static str,
        version: &'static str,
        request_schema: &'static str,
        response_schema: &'static str,
        presets: &'static [&'static str],
    ) -> Self {
        Self {
            name,
            version,
            request_schema,
            response_schema,
            presets,
        }
    }

    pub fn validate(&self) -> Result<(), TaskIrError> {
        for (field, value) in [
            ("name", self.name),
            ("version", self.version),
            ("request_schema", self.request_schema),
            ("response_schema", self.response_schema),
        ] {
            if value.is_empty() {
                return Err(TaskIrError::TaskSpec(format!("{field} is empty")));
            }
        }
        if !valid_identifier(self.name) {
            return Err(TaskIrError::TaskSpec(
                "name is not a stable identifier".to_owned(),
            ));
        }
        if !self.version.starts_with('v') || self.version.len() == 1 {
            return Err(TaskIrError::TaskSpec(
                "version must be a vN identifier".to_owned(),
            ));
        }
        if self.presets.iter().any(|preset| !valid_identifier(preset)) {
            return Err(TaskIrError::TaskSpec(
                "preset id is not a stable identifier".to_owned(),
            ));
        }
        Ok(())
    }

    /// Stable task-spec identity used by `ExecutionIdentity::task_spec`.
    #[must_use]
    pub fn identity(&self) -> String {
        format!("{}-{}", self.name, self.version)
    }

    #[must_use]
    pub const fn name(&self) -> &'static str {
        self.name
    }

    #[must_use]
    pub const fn version(&self) -> &'static str {
        self.version
    }

    #[must_use]
    pub const fn request_schema(&self) -> &'static str {
        self.request_schema
    }

    #[must_use]
    pub const fn response_schema(&self) -> &'static str {
        self.response_schema
    }

    #[must_use]
    pub const fn presets(&self) -> &'static [&'static str] {
        self.presets
    }
}

/// Immutable context provided to a task planner before model admission.
#[derive(Debug)]
pub struct PlanContext<'a> {
    execution_identity: &'a ExecutionIdentity,
    budget_ceiling: TaskBudget,
}

impl<'a> PlanContext<'a> {
    /// Construct a planning context whose identity and resource ceiling are
    /// already checked, so task compilation cannot widen either one.
    pub fn new(
        execution_identity: &'a ExecutionIdentity,
        budget_ceiling: TaskBudget,
    ) -> Result<Self, TaskIrError> {
        execution_identity.validate()?;
        budget_ceiling.validate()?;
        Ok(Self {
            execution_identity,
            budget_ceiling,
        })
    }

    #[must_use]
    pub const fn execution_identity(&self) -> &'a ExecutionIdentity {
        self.execution_identity
    }

    #[must_use]
    pub const fn budget_ceiling(&self) -> &TaskBudget {
        &self.budget_ceiling
    }
}

/// A validated task plan bound to a static task specification and plan context.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TaskPlan {
    task_spec_identity: String,
    ir: TaskIR,
}

impl TaskPlan {
    /// Bind an IR to the exact task and execution identity before model load.
    pub fn new(spec: &TaskSpec, context: &PlanContext<'_>, ir: TaskIR) -> Result<Self, FnlpError> {
        spec.validate().map_err(TaskIrError::into_fnlp_error)?;
        ir.validate().map_err(TaskIrError::into_fnlp_error)?;
        let identity = spec.identity();
        if context.execution_identity.task_spec != identity {
            return Err(FnlpError::SchemaOrRecipeCompile {
                category: "task_spec_execution_identity_mismatch",
            });
        }
        if !ir.budget.fits_within(&context.budget_ceiling) {
            return Err(FnlpError::SchemaOrRecipeCompile {
                category: "task_budget_exceeds_plan_context",
            });
        }
        Ok(Self {
            task_spec_identity: identity,
            ir,
        })
    }

    #[must_use]
    pub fn task_spec_identity(&self) -> &str {
        &self.task_spec_identity
    }

    #[must_use]
    pub const fn ir(&self) -> &TaskIR {
        &self.ir
    }
}

/// Raw decoded bytes plus their exact token sequence, retained only until task
/// finalization.  This is not a result envelope and never carries telemetry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecodeOutput {
    token_ids: Vec<u32>,
    bytes: Vec<u8>,
}

impl DecodeOutput {
    #[must_use]
    pub fn new(token_ids: Vec<u32>, bytes: Vec<u8>) -> Self {
        Self { token_ids, bytes }
    }

    #[must_use]
    pub fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }

    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    fn utf8(&self) -> Result<&str, FnlpError> {
        str::from_utf8(&self.bytes).map_err(|_| FnlpError::StructuredTaskNoResult {
            category: "task_decode_output_not_utf8",
        })
    }
}

/// Thin, concrete handle for the separately implemented independent validator.
/// It owns only immutable schema data and delegates to `validation/`; it never
/// accepts grammar automata, masks, transitions, or execution state.
#[derive(Clone, Debug)]
pub struct IndependentValidator {
    schema: SchemaNode,
}

impl IndependentValidator {
    #[must_use]
    pub fn new(schema: SchemaNode) -> Self {
        Self { schema }
    }

    /// Independently validate a decoded JSON response.
    pub fn validate(&self, raw: &DecodeOutput) -> Result<(), FnlpError> {
        validation::validate_json(&self.schema, raw.utf8()?).map_err(|_| {
            FnlpError::StructuredTaskNoResult {
                category: "independent_task_validation_failed",
            }
        })
    }

    /// Independently validate JSON plus the source-span evidence required by
    /// `x-fnlp-source: verbatim` fields.
    pub fn validate_with_grounding(
        &self,
        raw: &DecodeOutput,
        groundings: &[GroundedValue<'_>],
    ) -> Result<(), FnlpError> {
        validation::validate_with_grounding(&self.schema, raw.utf8()?, groundings).map_err(|_| {
            FnlpError::StructuredTaskNoResult {
                category: "independent_task_grounding_validation_failed",
            }
        })
    }
}

/// The declared probability or score space of a task result.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ScoreSpace {
    NotComputed,
    FullVocabSequenceLogprob,
    TrieLocalConditionalProbability,
    SingleStepCandidateConditional,
    SequenceScoreSoftmax,
}

impl ScoreSpace {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::NotComputed => "not_computed",
            Self::FullVocabSequenceLogprob => "full_vocab_sequence_logprob",
            Self::TrieLocalConditionalProbability => "trie_local_conditional_probability",
            Self::SingleStepCandidateConditional => "single_step_candidate_conditional",
            Self::SequenceScoreSoftmax => "sequence_score_softmax",
        }
    }
}

/// Deterministic semantic token counts.  Wall time is deliberately absent.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DeterministicTokenCounts {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub thinking_tokens: u32,
}

impl DeterministicTokenCounts {
    fn validate(&self) -> Result<(), TaskEnvelopeError> {
        self.completion_tokens
            .checked_add(self.thinking_tokens)
            .ok_or(TaskEnvelopeError::TokenCountOverflow)?;
        Ok(())
    }
}

/// Semantic result metadata.  Redundant-looking prompt, recipe, and profile
/// fields are projections of the one `ExecutionIdentity`, checked on every
/// construction so envelopes cannot drift from cache/receipt authority.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticEnvelope {
    schema_version: u32,
    task_spec_version: String,
    execution_identity: ExecutionIdentity,
    prompt_template_sha256: Sha256Digest,
    recipe_id: String,
    numerics_profile: String,
    thinking_mode: String,
    token_counts: DeterministicTokenCounts,
    calibration_digest: Sha256Digest,
    decision_policy_digest: Sha256Digest,
    score_space: ScoreSpace,
    status: StructuredTaskStatus,
}

impl SemanticEnvelope {
    /// Build a complete semantic envelope from the canonical identity.
    pub fn new(
        spec: &TaskSpec,
        execution_identity: ExecutionIdentity,
        token_counts: DeterministicTokenCounts,
        score_space: ScoreSpace,
        status: StructuredTaskStatus,
    ) -> Result<Self, TaskEnvelopeError> {
        spec.validate()
            .map_err(|error| TaskEnvelopeError::TaskSpec(error.to_string()))?;
        execution_identity
            .validate()
            .map_err(|error| TaskEnvelopeError::Identity(error.to_string()))?;
        token_counts.validate()?;
        if execution_identity.task_spec != spec.identity() {
            return Err(TaskEnvelopeError::TaskSpecIdentityMismatch);
        }
        let identity_value = execution_identity_value(&execution_identity)?;
        let thinking_mode = identity_string(&identity_value, "thinking_mode")?;
        let numerics_profile = identity_string(&identity_value, "numerics_profile")?;
        Ok(Self {
            schema_version: SEMANTIC_ENVELOPE_SCHEMA_VERSION,
            task_spec_version: spec.version.to_owned(),
            prompt_template_sha256: execution_identity.template_digest,
            recipe_id: execution_identity.quant_recipe.clone(),
            calibration_digest: execution_identity.calibration_digest,
            decision_policy_digest: execution_identity.decision_policy_digest,
            execution_identity,
            numerics_profile,
            thinking_mode,
            token_counts,
            score_space,
            status,
        })
    }

    /// Recheck envelope completeness and identity-derived projections.
    pub fn validate(&self) -> Result<(), TaskEnvelopeError> {
        if self.schema_version != SEMANTIC_ENVELOPE_SCHEMA_VERSION {
            return Err(TaskEnvelopeError::UnsupportedSemanticSchemaVersion(
                self.schema_version,
            ));
        }
        if self.task_spec_version.is_empty() {
            return Err(TaskEnvelopeError::EmptyField("task_spec_version"));
        }
        self.execution_identity
            .validate()
            .map_err(|error| TaskEnvelopeError::Identity(error.to_string()))?;
        self.token_counts.validate()?;
        if self.prompt_template_sha256 != self.execution_identity.template_digest {
            return Err(TaskEnvelopeError::IdentityProjectionMismatch(
                "prompt_template_sha256",
            ));
        }
        if self.recipe_id != self.execution_identity.quant_recipe {
            return Err(TaskEnvelopeError::IdentityProjectionMismatch("recipe_id"));
        }
        if self.calibration_digest != self.execution_identity.calibration_digest {
            return Err(TaskEnvelopeError::IdentityProjectionMismatch(
                "calibration_digest",
            ));
        }
        if self.decision_policy_digest != self.execution_identity.decision_policy_digest {
            return Err(TaskEnvelopeError::IdentityProjectionMismatch(
                "decision_policy_digest",
            ));
        }
        let identity_value = execution_identity_value(&self.execution_identity)?;
        if self.numerics_profile != identity_string(&identity_value, "numerics_profile")? {
            return Err(TaskEnvelopeError::IdentityProjectionMismatch(
                "numerics_profile",
            ));
        }
        if self.thinking_mode != identity_string(&identity_value, "thinking_mode")? {
            return Err(TaskEnvelopeError::IdentityProjectionMismatch(
                "thinking_mode",
            ));
        }
        Ok(())
    }

    #[must_use]
    pub const fn prompt_template_sha256(&self) -> Sha256Digest {
        self.prompt_template_sha256
    }

    #[must_use]
    pub fn recipe_id(&self) -> &str {
        &self.recipe_id
    }

    #[must_use]
    pub fn numerics_profile(&self) -> &str {
        &self.numerics_profile
    }

    #[must_use]
    pub const fn token_counts(&self) -> DeterministicTokenCounts {
        self.token_counts
    }

    #[must_use]
    pub const fn score_space(&self) -> ScoreSpace {
        self.score_space
    }

    #[must_use]
    pub const fn status(&self) -> StructuredTaskStatus {
        self.status
    }

    fn value(&self) -> Result<Value, TaskEnvelopeError> {
        self.validate()?;
        let mut fields = BTreeMap::new();
        fields.insert(
            "calibration_digest",
            Value::String(self.calibration_digest.to_hex()),
        );
        fields.insert(
            "decision_policy_digest",
            Value::String(self.decision_policy_digest.to_hex()),
        );
        fields.insert(
            "deterministic_token_counts",
            serde_json::to_value(self.token_counts)
                .map_err(|error| TaskEnvelopeError::Serialization(error.to_string()))?,
        );
        fields.insert(
            "execution_identity",
            execution_identity_value(&self.execution_identity)?,
        );
        fields.insert(
            "numerics_profile",
            Value::String(self.numerics_profile.clone()),
        );
        fields.insert(
            "prompt_template_sha256",
            Value::String(self.prompt_template_sha256.to_hex()),
        );
        fields.insert("recipe_id", Value::String(self.recipe_id.clone()));
        fields.insert(
            "result_status",
            Value::String(self.status.status().to_owned()),
        );
        fields.insert(
            "schema_version",
            Value::Number(serde_json::Number::from(self.schema_version)),
        );
        fields.insert(
            "score_space",
            Value::String(self.score_space.label().to_owned()),
        );
        fields.insert(
            "task_spec_version",
            Value::String(self.task_spec_version.clone()),
        );
        fields.insert("thinking_mode", Value::String(self.thinking_mode.clone()));
        Ok(serde_json::to_value(fields)
            .map_err(|error| TaskEnvelopeError::Serialization(error.to_string()))?)
    }
}

/// Telemetry is observed operational data, never semantic replay authority.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TelemetryEnvelope {
    schema_version: u32,
    queue_wait_ns: u64,
    stage_timings_ns: Vec<StageTiming>,
    selected_tuning_row: Option<String>,
    transient_run_id: Option<String>,
    page_faults: Option<PageFaultCounts>,
    host_load_milli: Option<u32>,
    energy_microjoules: Option<u64>,
}

/// One named operational timing measurement.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StageTiming {
    stage: String,
    elapsed_ns: u64,
}

impl StageTiming {
    #[must_use]
    pub fn new(stage: impl Into<String>, elapsed_ns: u64) -> Self {
        Self {
            stage: stage.into(),
            elapsed_ns,
        }
    }
}

/// Volatile page-fault observations, kept outside semantic bytes.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PageFaultCounts {
    pub minor: u64,
    pub major: u64,
}

impl TelemetryEnvelope {
    /// Construct telemetry after checking identifiers and unique stage rows.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        queue_wait_ns: u64,
        stage_timings_ns: Vec<StageTiming>,
        selected_tuning_row: Option<String>,
        transient_run_id: Option<String>,
        page_faults: Option<PageFaultCounts>,
        host_load_milli: Option<u32>,
        energy_microjoules: Option<u64>,
    ) -> Result<Self, TaskEnvelopeError> {
        let envelope = Self {
            schema_version: TELEMETRY_ENVELOPE_SCHEMA_VERSION,
            queue_wait_ns,
            stage_timings_ns,
            selected_tuning_row,
            transient_run_id,
            page_faults,
            host_load_milli,
            energy_microjoules,
        };
        envelope.validate()?;
        Ok(envelope)
    }

    pub fn validate(&self) -> Result<(), TaskEnvelopeError> {
        if self.schema_version != TELEMETRY_ENVELOPE_SCHEMA_VERSION {
            return Err(TaskEnvelopeError::UnsupportedTelemetrySchemaVersion(
                self.schema_version,
            ));
        }
        let mut stages = std::collections::BTreeSet::new();
        for timing in &self.stage_timings_ns {
            if !valid_identifier(&timing.stage) {
                return Err(TaskEnvelopeError::InvalidTelemetryStage(
                    timing.stage.clone(),
                ));
            }
            if !stages.insert(&timing.stage) {
                return Err(TaskEnvelopeError::DuplicateTelemetryStage(
                    timing.stage.clone(),
                ));
            }
        }
        for (field, value) in [
            ("selected_tuning_row", self.selected_tuning_row.as_deref()),
            ("transient_run_id", self.transient_run_id.as_deref()),
        ] {
            if value.is_some_and(str::is_empty) {
                return Err(TaskEnvelopeError::EmptyField(field));
            }
        }
        Ok(())
    }
}

/// One typed response with semantic and optional operational metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TaskResponseEnvelope<T> {
    semantic: SemanticEnvelope,
    result: T,
    telemetry: Option<TelemetryEnvelope>,
}

impl<T> TaskResponseEnvelope<T> {
    #[must_use]
    pub fn new(
        semantic: SemanticEnvelope,
        result: T,
        telemetry: Option<TelemetryEnvelope>,
    ) -> Self {
        Self {
            semantic,
            result,
            telemetry,
        }
    }

    #[must_use]
    pub const fn semantic(&self) -> &SemanticEnvelope {
        &self.semantic
    }

    #[must_use]
    pub const fn result(&self) -> &T {
        &self.result
    }

    #[must_use]
    pub const fn telemetry(&self) -> Option<&TelemetryEnvelope> {
        self.telemetry.as_ref()
    }
}

impl<T> TaskResponseEnvelope<T>
where
    T: Serialize,
{
    /// Canonical replay bytes deliberately omit every telemetry field.
    pub fn canonical_json_bytes(&self) -> Result<Vec<u8>, TaskEnvelopeError> {
        self.json_bytes(false)
    }

    /// Operational response bytes include telemetry when it was observed.
    pub fn observed_json_bytes(&self) -> Result<Vec<u8>, TaskEnvelopeError> {
        self.json_bytes(true)
    }

    fn json_bytes(&self, include_telemetry: bool) -> Result<Vec<u8>, TaskEnvelopeError> {
        let mut fields = BTreeMap::new();
        fields.insert(
            "result",
            serde_json::to_value(&self.result).map_err(serialize_error)?,
        );
        fields.insert("semantic", self.semantic.value()?);
        if include_telemetry {
            if let Some(telemetry) = &self.telemetry {
                telemetry.validate()?;
                fields.insert(
                    "telemetry",
                    serde_json::to_value(telemetry).map_err(serialize_error)?,
                );
            }
        }
        canonjson::canonical_bytes(&fields)
            .map_err(|error| TaskEnvelopeError::CanonicalJson(error.to_string()))
    }
}

fn serialize_error(error: serde_json::Error) -> TaskEnvelopeError {
    TaskEnvelopeError::Serialization(error.to_string())
}

fn execution_identity_value(identity: &ExecutionIdentity) -> Result<Value, TaskEnvelopeError> {
    let bytes = identity
        .canonical_json_bytes()
        .map_err(|error| TaskEnvelopeError::Identity(error.to_string()))?;
    let text =
        str::from_utf8(&bytes).map_err(|error| TaskEnvelopeError::Identity(error.to_string()))?;
    canonjson::parse_str(text).map_err(|error| TaskEnvelopeError::CanonicalJson(error.to_string()))
}

fn identity_string(identity: &Value, field: &'static str) -> Result<String, TaskEnvelopeError> {
    identity
        .get(field)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .ok_or(TaskEnvelopeError::MissingIdentityProjection(field))
}

/// Safe-to-log envelope construction errors.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TaskEnvelopeError {
    UnsupportedSemanticSchemaVersion(u32),
    UnsupportedTelemetrySchemaVersion(u32),
    EmptyField(&'static str),
    TaskSpec(String),
    TaskSpecIdentityMismatch,
    Identity(String),
    MissingIdentityProjection(&'static str),
    IdentityProjectionMismatch(&'static str),
    TokenCountOverflow,
    InvalidTelemetryStage(String),
    DuplicateTelemetryStage(String),
    Serialization(String),
    CanonicalJson(String),
}

impl fmt::Display for TaskEnvelopeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSemanticSchemaVersion(version) => write!(
                formatter,
                "unsupported semantic envelope schema version {version}"
            ),
            Self::UnsupportedTelemetrySchemaVersion(version) => write!(
                formatter,
                "unsupported telemetry envelope schema version {version}"
            ),
            Self::EmptyField(field) => write!(formatter, "envelope field {field} is empty"),
            Self::TaskSpec(error) => write!(formatter, "invalid task spec: {error}"),
            Self::TaskSpecIdentityMismatch => {
                formatter.write_str("task spec does not match execution identity")
            }
            Self::Identity(error) => write!(formatter, "invalid execution identity: {error}"),
            Self::MissingIdentityProjection(field) => {
                write!(formatter, "execution identity lacks projection {field}")
            }
            Self::IdentityProjectionMismatch(field) => write!(
                formatter,
                "envelope projection {field} diverges from execution identity"
            ),
            Self::TokenCountOverflow => formatter.write_str("deterministic token count overflow"),
            Self::InvalidTelemetryStage(stage) => {
                write!(formatter, "invalid telemetry stage {stage}")
            }
            Self::DuplicateTelemetryStage(stage) => {
                write!(formatter, "duplicate telemetry stage {stage}")
            }
            Self::Serialization(error) => {
                write!(formatter, "envelope serialization failed: {error}")
            }
            Self::CanonicalJson(error) => {
                write!(formatter, "envelope canonical JSON failed: {error}")
            }
        }
    }
}

impl Error for TaskEnvelopeError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution_identity::{NumericsProfile, ThinkingMode, ToolMode};

    fn digest(byte: u8) -> Sha256Digest {
        Sha256Digest::of_bytes(&[byte])
    }

    fn budget() -> TaskBudget {
        TaskBudget {
            max_input_tokens: 32,
            max_output_tokens: 16,
            max_output_bytes: 128,
            max_grammar_states: 64,
            max_kv_bytes: 1_024,
        }
    }

    fn constrained_json_ir() -> TaskIR {
        TaskIR::new(
            vec![
                PromptSegment::new(PromptSegmentKind::GlobalPolicy, vec![1]),
                PromptSegment::new(PromptSegmentKind::TaskInstruction, vec![2, 3]),
                PromptSegment::new(PromptSegmentKind::Document, vec![4]),
                PromptSegment::new(PromptSegmentKind::AnswerScaffold, vec![5]),
            ],
            DecodeStrategy::ConstrainedJson,
            GrammarReference::json_schema(digest(1), "grammar-v1"),
            None,
            vec![
                FinitePostcondition::JsonValid,
                FinitePostcondition::OutputWithinBudget,
            ],
            budget(),
            DependencyScope::ItemLocal,
        )
        .expect("valid bounded TaskIR")
    }

    fn execution(task_spec: &str) -> ExecutionIdentity {
        ExecutionIdentity::new(ExecutionIdentity {
            schema_version: 1,
            source_revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
            logical_model_digest: digest(1),
            artifact_format: "fnlpq-v1".to_owned(),
            quant_recipe: "nanbeige42-int8-v1".to_owned(),
            packing_set_digest: digest(2),
            tokenizer_digest: digest(3),
            template_digest: digest(4),
            task_spec: task_spec.to_owned(),
            taskir_digest: digest(5),
            prompt_digest: digest(6),
            grammar_compiler_version: "grammar-v1".to_owned(),
            schema_digest: digest(7),
            numerics_profile: NumericsProfile::HfBf16Eager,
            kv_dtype: "bf16".to_owned(),
            sampler_version: "greedy-v1".to_owned(),
            thinking_mode: ThinkingMode::Disabled,
            tool_mode: ToolMode::None,
            calibration_digest: digest(8),
            decision_policy_digest: digest(9),
            backend_semantic_version: "cpu-v1".to_owned(),
            host_class: None,
            compiler_identity: None,
        })
        .expect("valid identity")
    }

    #[test]
    fn task_ir_construction_is_deterministic_and_dependency_scoped() {
        let first = constrained_json_ir();
        let second = constrained_json_ir();
        assert_eq!(
            first.canonical_json_bytes().expect("canonical IR"),
            second.canonical_json_bytes().expect("canonical IR")
        );
        assert_eq!(
            first.digest().expect("IR digest"),
            second.digest().expect("IR digest")
        );
        assert!(DependencyScope::ItemLocal.preserves_after_child_set_change());
        assert!(!DependencyScope::PartitionReduce.preserves_after_child_set_change());
        assert!(!DependencyScope::CorpusGlobal.preserves_after_child_set_change());
    }

    #[test]
    fn task_ir_rejects_an_unbounded_or_noncanonical_plan() {
        let error = TaskIR::new(
            vec![PromptSegment::new(
                PromptSegmentKind::TaskInstruction,
                vec![1],
            )],
            DecodeStrategy::FreeText {
                stops: vec![],
                budget: DecodeBudget {
                    max_tokens: 17,
                    max_bytes: 64,
                },
            },
            GrammarReference::none(),
            None,
            vec![FinitePostcondition::OutputWithinBudget],
            budget(),
            DependencyScope::ItemLocal,
        )
        .expect_err("answer scaffold and bounded decode are required");
        assert_eq!(error, TaskIrError::PromptMustEndWithAnswerScaffold);
    }

    #[test]
    fn task_plan_cannot_widen_context_budget_or_identity() {
        let spec = TaskSpec::new(
            "extract",
            "v1",
            "extract-request-v1",
            "extract-response-v1",
            &[],
        );
        let identity = execution("extract-v1");
        let context = PlanContext::new(&identity, budget()).expect("valid context");
        let plan = TaskPlan::new(&spec, &context, constrained_json_ir()).expect("bounded plan");
        assert_eq!(plan.task_spec_identity(), "extract-v1");
        let wrong_identity = execution("classify-v1");
        let wrong_context = PlanContext::new(&wrong_identity, budget()).expect("valid context");
        assert!(matches!(
            TaskPlan::new(&spec, &wrong_context, constrained_json_ir()),
            Err(FnlpError::SchemaOrRecipeCompile {
                category: "task_spec_execution_identity_mismatch"
            })
        ));
    }

    #[test]
    fn canonical_response_omits_telemetry_but_observed_response_retains_it() {
        let spec = TaskSpec::new(
            "extract",
            "v1",
            "extract-request-v1",
            "extract-response-v1",
            &[],
        );
        let semantic = SemanticEnvelope::new(
            &spec,
            execution("extract-v1"),
            DeterministicTokenCounts {
                prompt_tokens: 5,
                completion_tokens: 3,
                thinking_tokens: 0,
            },
            ScoreSpace::NotComputed,
            StructuredTaskStatus::Completed,
        )
        .expect("complete semantic envelope");
        let telemetry = TelemetryEnvelope::new(
            12,
            vec![StageTiming::new("decode", 34)],
            Some("scalar-v1".to_owned()),
            Some("transient-run-7".to_owned()),
            Some(PageFaultCounts { minor: 1, major: 0 }),
            Some(500),
            Some(99),
        )
        .expect("valid telemetry");
        let response =
            TaskResponseEnvelope::new(semantic, serde_json::json!({"items": []}), Some(telemetry));
        let canonical =
            String::from_utf8(response.canonical_json_bytes().expect("canonical response"))
                .expect("UTF-8");
        let observed =
            String::from_utf8(response.observed_json_bytes().expect("observed response"))
                .expect("UTF-8");
        assert!(canonical.contains("prompt_template_sha256"));
        assert!(canonical.contains("recipe_id"));
        assert!(canonical.contains("numerics_profile"));
        assert!(!canonical.contains("telemetry"));
        assert!(observed.contains("telemetry"));
        assert!(observed.contains("transient_run_id"));
    }
}
