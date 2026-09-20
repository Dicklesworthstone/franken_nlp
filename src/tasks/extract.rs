//! Schema-bound extraction over exact TaskIR tokens and the native decoder.
//!
//! Source-bound plans enforce exact source membership, not semantic truth or
//! extraction accuracy. Unbound source annotations remain a typed refusal.
//! Trusted orchestration supplies TaskIR segments and the archived control
//! registry; no flatten-and-retokenize path or parse-failure retry is added.

use std::{collections::BTreeSet, error::Error, fmt};
use serde::{Deserialize, Serialize};
use crate::{
    canonjson,
    execution_identity::{ExecutionIdentity, NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    grammar::{
        CompileLimits, SchemaError,
        mask::{VocabMaskOracle, VocabTrie},
        runtime::{JSON_RUNTIME_VERSION, SOURCE_JSON_RUNTIME_VERSION, JsonProgram, SourceRuntimeLimits},
    },
    native_engine::{
        constrained::{JsonDecodeError, JsonDecodeOptions, JsonDecodeOutput, JsonWorkBudget, decode_json_eager},
        decode::DecodeStepControl,
        hf_bf16_eager::{HF_BF16_EAGER_PROFILE, HfBf16EagerEngine},
        lmhead::NANBEIGE_VOCAB_SIZE,
    },
    validation::grounded_fields::SourceFieldEvidence,
    tokenizer::{
        embedded::{EmbeddedTokenizer, PINNED_TOKENIZER_MODEL_BYTES, PINNED_ADDED_TOKENS_BYTES,
            PINNED_TOKENIZER_CONFIG_BYTES, PINNED_SPECIAL_TOKENS_MAP_BYTES},
        specials::TemplateControlIds,
    },
};
use super::ir::{DecodeStrategy, DependencyScope, FinitePostcondition, GrammarReference, PromptSegmentKind, ScoreSpace, TaskIR, TaskPlan};

pub mod grounded;
pub mod quantized;
pub mod semantic;
pub use grounded::{SourceDocument, SourceDocumentEncoder};

/// Explicit sampler policy; this path does not accept seeded or thinking modes.
pub const EXTRACT_SAMPLER_VERSION: &str = "constrained-greedy-eos-v1";

/// Reusable model-free vocabulary. Build once alongside the engine, not once
/// per corpus item. The registry comes from the caller's pinned truth pack;
/// it cannot be replaced with tokenizer-special ids alone.
pub struct ExtractionVocabulary {
    oracle: VocabMaskOracle,
    controls: BTreeSet<u32>,
}

impl ExtractionVocabulary {
    pub fn pinned(controls: &TemplateControlIds) -> Result<Self, ExtractError> {
        if controls.ids().iter().any(|&id| id as usize >= NANBEIGE_VOCAB_SIZE) {
            return Err(ExtractError::Contract("control id outside model vocabulary"));
        }
        let embedded = EmbeddedTokenizer::pinned().map_err(|_| ExtractError::Tokenizer)?;
        let trie = VocabTrie::from_tokenizer(embedded.tokenizer(), NANBEIGE_VOCAB_SIZE)
            .map_err(|_| ExtractError::Tokenizer)?;
        Ok(Self { oracle: VocabMaskOracle::new(trie), controls: controls.ids().clone() })
    }
}

/// An immutable executable extraction plan. It deliberately has no Debug or
/// Serialize implementation: its exact prompt tokens are private content.
pub struct ExtractPlan {
    task_identity: &'static str,
    program: JsonProgram,
    prompt: Vec<u32>,
    options: JsonDecodeOptions,
    controls: BTreeSet<u32>,
    taskir_digest: Sha256Digest,
    prompt_digest: Sha256Digest,
    schema_digest: Sha256Digest,
    policy_digest: Sha256Digest,
    max_kv_bytes: u64,
    max_result_bytes: u64,
}

#[derive(Debug)]
pub enum ExtractError {
    Contract(&'static str),
    Schema(SchemaError),
    Tokenizer,
    AllocationRefused,
    Serialization,
    Decode(JsonDecodeError),
    InvalidResult,
    OutputBudgetExceeded,
}
impl fmt::Display for ExtractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Contract(reason) => write!(f, "extraction contract refused: {reason}"),
            Self::Schema(error) => write!(f, "extraction schema refused: {error}"),
            Self::Tokenizer => f.write_str("extraction pinned vocabulary construction failed"),
            Self::AllocationRefused => f.write_str("extraction allocation refused"),
            Self::Serialization => f.write_str("extraction canonical serialization failed"),
            Self::Decode(error) => write!(f, "extraction has no result: {error}"),
            Self::InvalidResult => f.write_str("extraction failed independent result validation"),
            Self::OutputBudgetExceeded => f.write_str("extraction result envelope exceeds its byte budget"),
        }
    }
}
impl Error for ExtractError {}

/// A successful extraction carries exact JSON text to preserve the runtime's
/// 38-digit decimal domain rather than coercing it through f64/serde Value.
/// This result contains no document/prompt digests or fabricated confidence.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExtractResult {
    pub schema_version: u32,
    pub task_spec_version: String,
    pub score_space: ScoreSpace,
    pub grounding: ExtractionGrounding,
    pub output: JsonDecodeOutput,
    /// Present on source-bound schema-v2 results; omitted for the legacy path.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub source_fields: Vec<SourceFieldEvidence>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtractionGrounding {
    /// Only schema validity is guaranteed by this task version.
    NotRequested,
    /// All present verbatim fields occur in the exact supplied document.
    SourceMembership,
}

#[derive(Serialize)]
struct PolicyWitness<'a> {
    version: &'static str,
    options: &'a JsonDecodeOptions,
    template_control_ids: &'a BTreeSet<u32>,
    tokenizer_assets: [Sha256Digest; 4],
    // All compiler caps affect the admitted output language. Array position
    // is frozen in this witness: schema,string,array,output,states,edges,masks.
    compiler_caps: [u64; 7],
}

impl ExtractPlan {
    /// Compile before model admission. The TaskPlan must name this exact
    /// schema and versioned runtime, not the legacy descriptive graph.
    pub fn from_task_plan(
        task: &TaskPlan,
        schema_source: &str,
        options: JsonDecodeOptions,
        limits: CompileLimits,
        controls: &TemplateControlIds,
    ) -> Result<Self, ExtractError> {
        Self::from_builtin(task, schema_source, options, limits, controls, None, "extract-v1")
    }

    pub(crate) fn from_builtin(
        task: &TaskPlan, schema_source: &str, mut options: JsonDecodeOptions,
        mut limits: CompileLimits, controls: &TemplateControlIds,
        source: Option<(&SourceDocument, SourceRuntimeLimits)>, task_identity: &'static str,
    ) -> Result<Self, ExtractError> {
        let ir = task.ir();
        ir.validate().map_err(|_| ExtractError::Contract("invalid TaskIR"))?;
        if !matches!(task_identity, "extract-v1" | "ner-v1" | "keyphrases-v1" | "summarize-v1" | "answer-v1")
            || task.task_spec_identity() != task_identity || !matches!(ir.decode_strategy(), DecodeStrategy::ConstrainedJson)
        {
            return Err(ExtractError::Contract("source task requires matching constrained_json TaskIR"));
        }
        if task_identity != "extract-v1" && source.is_none() {
            return Err(ExtractError::Contract("source task requires an exact bound document"));
        }
        if source.is_some() { grounded::check_source_postconditions(ir)?; } else { check_postconditions(ir)?; }
        if ir.dependency_scope() != DependencyScope::ItemLocal {
            return Err(ExtractError::Contract("extraction must remain item-local"));
        }
        let schema_digest = Sha256Digest::of_bytes(schema_source.as_bytes());
        let runtime_version = if source.is_some() { SOURCE_JSON_RUNTIME_VERSION } else { JSON_RUNTIME_VERSION };
        match ir.grammar() {
            GrammarReference::JsonSchema { digest, compiler_version }
                if *digest == schema_digest && compiler_version == runtime_version => {}
            _ => return Err(ExtractError::Contract("schema or runtime differs from TaskIR")),
        }
        if options.max_new_tokens == 0 || options.max_new_tokens as u64 > u64::from(ir.budget().max_output_tokens) {
            return Err(ExtractError::Contract("decode token budget exceeds TaskIR"));
        }
        if !controls.entry(options.eos_token_id).is_some_and(|entry| entry.special) {
            return Err(ExtractError::Contract("EOS must be an archived special control"));
        }
        // Caller exclusions may strengthen, never weaken, the control alphabet.
        options.excluded_token_ids.extend(controls.ids().iter().copied());
        if options.excluded_token_ids.iter().any(|&id| id as usize >= NANBEIGE_VOCAB_SIZE) {
            return Err(ExtractError::Contract("excluded id outside vocabulary"));
        }
        let mut documents = 0;
        let mut prompt_len = 0_usize;
        for segment in ir.prompt_segments() {
            prompt_len = prompt_len.checked_add(segment.token_ids().len())
                .ok_or(ExtractError::AllocationRefused)?;
            if segment.token_ids().iter().any(|&id| id as usize >= NANBEIGE_VOCAB_SIZE) {
                return Err(ExtractError::Contract("prompt id outside vocabulary"));
            }
            if segment.kind() == PromptSegmentKind::Document {
                documents += 1;
                if segment.token_ids().iter().any(|&id| controls.contains(id)) {
                    return Err(ExtractError::Contract("document contains privileged control id"));
                }
            }
        }
        if documents == 0 { return Err(ExtractError::Contract("missing document segment")); }
        limits.max_output_bytes = limits.max_output_bytes.min(usize::try_from(ir.budget().max_output_bytes).unwrap_or(usize::MAX));
        limits.max_states = limits.max_states.min(ir.budget().max_grammar_states as usize);
        let program = match source {
            Some((document, source_limits)) => {
                document.verify_task(ir, controls)?;
                JsonProgram::compile_with_source(schema_source, document.text(), limits, source_limits)
            }
            None => JsonProgram::compile(schema_source, limits),
        }.map_err(ExtractError::Schema)?;
        if source.is_some() && !program.requires_source() {
            return Err(ExtractError::Contract("source plan requires a verbatim field"));
        }
        let mut policy_digest = policy_digest(&options, limits, controls.ids())?;
        if let Some(source_limits) = program.source_limits() {
            policy_digest = grounded::source_policy_digest(policy_digest, source_limits)?;
        }
        let taskir_digest = ir.digest().map_err(|_| ExtractError::Serialization)?;
        let prompt_digest = Sha256Digest::of_bytes(&canonjson::canonical_bytes(ir.prompt_segments()).map_err(|_| ExtractError::Serialization)?);
        let mut prompt = Vec::new();
        prompt.try_reserve_exact(prompt_len).map_err(|_| ExtractError::AllocationRefused)?;
        prompt.extend(ir.prompt_segments().iter().flat_map(|segment| segment.token_ids().iter().copied()));
        Ok(Self {
            task_identity, program, prompt, options, controls: controls.ids().clone(), taskir_digest,
            prompt_digest, schema_digest, policy_digest,
            max_kv_bytes: ir.budget().max_kv_bytes, max_result_bytes: ir.budget().max_output_bytes,
        })
    }

    /// Fill only task-owned identity fields. Artifact/model/template authority
    /// remains with the caller. Keep the returned identity in private request
    /// state; exported content commitments require the job's keyed projection.
    pub fn bind_identity(&self, mut identity: ExecutionIdentity) -> Result<ExecutionIdentity, ExtractError> {
        check_profile(&identity, self.task_identity)?;
        identity.taskir_digest = self.taskir_digest;
        identity.prompt_digest = self.prompt_digest;
        identity.schema_digest = self.schema_digest;
        identity.decision_policy_digest = self.policy_digest;
        identity.grammar_compiler_version = self.program.version().to_owned();
        identity.sampler_version = EXTRACT_SAMPLER_VERSION.to_owned();
        identity.tokenizer_digest = Sha256Digest::of_bytes(PINNED_TOKENIZER_MODEL_BYTES);
        identity.validate().map_err(|_| ExtractError::Contract("invalid execution identity"))?;
        Ok(identity)
    }

    /// Verify rather than silently repair identities at execution time.
    pub fn verify_identity(&self, identity: &ExecutionIdentity) -> Result<(), ExtractError> {
        check_profile(identity, self.task_identity)?;
        identity.validate().map_err(|_| ExtractError::Contract("invalid execution identity"))?;
        if identity.taskir_digest != self.taskir_digest || identity.prompt_digest != self.prompt_digest
            || identity.schema_digest != self.schema_digest || identity.decision_policy_digest != self.policy_digest
            || identity.grammar_compiler_version != self.program.version() || identity.sampler_version != EXTRACT_SAMPLER_VERSION
            || identity.tokenizer_digest != Sha256Digest::of_bytes(PINNED_TOKENIZER_MODEL_BYTES)
        {
            return Err(ExtractError::Contract("execution identity does not bind this extraction"));
        }
        Ok(())
    }

    /// Execute an already-admitted model with an immutable pinned vocabulary.
    /// All failures, including cancellation and envelope overflow, are no-result.
    pub fn execute_eager<C: DecodeStepControl>(
        &self,
        engine: &mut HfBf16EagerEngine,
        identity: &ExecutionIdentity,
        vocabulary: &ExtractionVocabulary,
        mut work: JsonWorkBudget,
        control: &mut C,
    ) -> Result<ExtractResult, ExtractError> {
        self.verify_identity(identity)?;
        if vocabulary.controls != self.controls {
            return Err(ExtractError::Contract("vocabulary control registry differs from plan"));
        }
        work.max_kv_bytes = work.max_kv_bytes.min(self.max_kv_bytes);
        let output = decode_json_eager(engine, &self.prompt, &self.program, &vocabulary.oracle, &self.options, work, control)
            .map_err(ExtractError::Decode)?;
        self.finalize(output)
    }

    fn finalize(&self, output: JsonDecodeOutput) -> Result<ExtractResult, ExtractError> {
        self.finalize_profile(output, HF_BF16_EAGER_PROFILE)
    }

    // Only closed native task drivers choose this profile. Public eager
    // finalization retains its exact original profile refusal.
    fn finalize_profile(&self, output: JsonDecodeOutput, profile: &str) -> Result<ExtractResult, ExtractError> {
        if output.schema_version != 1 || output.numerics_profile != profile
            || output.token_ids.is_empty() || output.token_ids.len() > self.options.max_new_tokens
            || output.token_ids.last() != Some(&self.options.eos_token_id)
        { return Err(ExtractError::InvalidResult); }
        if output.token_ids[..output.token_ids.len() - 1].iter().any(|id| {
            *id == self.options.eos_token_id || *id as usize >= NANBEIGE_VOCAB_SIZE || self.options.excluded_token_ids.contains(id)
        }) { return Err(ExtractError::InvalidResult); }
        let source_fields = self.program.source_fields(&output.json).map_err(|_| ExtractError::InvalidResult)?;
        let source_bound = self.program.requires_source();
        let result = ExtractResult {
            schema_version: if source_bound { 2 } else { 1 }, task_spec_version: self.task_identity.to_owned(),
            score_space: ScoreSpace::NotComputed,
            grounding: if source_bound { ExtractionGrounding::SourceMembership } else { ExtractionGrounding::NotRequested },
            output, source_fields,
        };
        let bytes = canonjson::canonical_bytes(&result).map_err(|_| ExtractError::Serialization)?;
        if bytes.len() as u64 > self.max_result_bytes { return Err(ExtractError::OutputBudgetExceeded); }
        Ok(result)
    }
}

// A narrow view of the frozen, typed TaskIR wire shape. This is not an
// external JSON boundary: to_value receives an already validated Rust value.
// Keep this projection until TaskIR exposes a borrowed postcondition accessor.
#[derive(Deserialize)]
struct PostconditionView { postconditions: Vec<FinitePostcondition> }

fn check_postconditions(ir: &TaskIR) -> Result<(), ExtractError> {
    let value = serde_json::to_value(ir).map_err(|_| ExtractError::Serialization)?;
    let view: PostconditionView = serde_json::from_value(value).map_err(|_| ExtractError::Serialization)?;
    if view.postconditions.iter().any(|condition| !matches!(condition,
        FinitePostcondition::JsonValid | FinitePostcondition::MatchesGrammar | FinitePostcondition::OutputWithinBudget
    )) {
        return Err(ExtractError::Contract("unsupported extraction postcondition"));
    }
    Ok(())
}

fn check_profile(identity: &ExecutionIdentity, task_identity: &str) -> Result<(), ExtractError> {
    if identity.task_spec != task_identity || identity.numerics_profile != NumericsProfile::HfBf16Eager
        || identity.thinking_mode != ThinkingMode::Disabled || identity.tool_mode != ToolMode::None || identity.kv_dtype != "bf16"
    { return Err(ExtractError::Contract("requires matching task, eager bf16, thinking off, no tools")); }
    Ok(())
}

fn policy_digest(options: &JsonDecodeOptions, limits: CompileLimits, controls: &BTreeSet<u32>) -> Result<Sha256Digest, ExtractError> {
    let witness = PolicyWitness {
        version: EXTRACT_SAMPLER_VERSION, options, template_control_ids: controls,
        tokenizer_assets: [PINNED_TOKENIZER_MODEL_BYTES, PINNED_ADDED_TOKENS_BYTES,
            PINNED_TOKENIZER_CONFIG_BYTES, PINNED_SPECIAL_TOKENS_MAP_BYTES].map(Sha256Digest::of_bytes),
        compiler_caps: [limits.max_schema_bytes, limits.max_string_bytes, limits.max_array_items,
            limits.max_output_bytes, limits.max_states, limits.max_transitions, limits.max_mask_bytes].map(|n| n as u64),
    };
    Ok(Sha256Digest::of_bytes(&canonjson::canonical_bytes(&witness).map_err(|_| ExtractError::Serialization)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{tasks::{BuiltInTask, ir::{TaskBudget, PlanContext, PromptSegment}},
        tokenizer::specials::ArchivedControlRegistries};

    const SCHEMA: &str = r#"{"type":"boolean"}"#;
    pub(super) fn registry() -> ArchivedControlRegistries {
        ArchivedControlRegistries::from_archived_json(
            r#"{"schema_version":1,"registry":"TokenizerSpecialIds","entries":[{"id":0,"special":true,"surface":"<eos>"}]}"#,
            r#"{"schema_version":1,"registry":"TemplateControlIds","entries":[{"id":0,"special":true,"surface":"<eos>"},{"id":3,"special":false,"surface":"<think>"}]}"#,
        ).unwrap()
    }
    pub(super) fn identity() -> ExecutionIdentity {
        let d = Sha256Digest::of_bytes(b"fixture");
        ExecutionIdentity {
            schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
            artifact_format: "fixture".to_owned(), quant_recipe: "bf16-verbatim".to_owned(), packing_set_digest: d,
            tokenizer_digest: d, template_digest: d, task_spec: "extract-v1".to_owned(), taskir_digest: d,
            prompt_digest: d, grammar_compiler_version: JSON_RUNTIME_VERSION.to_owned(), schema_digest: d,
            numerics_profile: NumericsProfile::HfBf16Eager, kv_dtype: "bf16".to_owned(), sampler_version: "fixture".to_owned(),
            thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None, calibration_digest: d,
            decision_policy_digest: d, backend_semantic_version: "fixture".to_owned(), host_class: None, compiler_identity: None,
        }
    }
    fn task(schema: &str, document_id: u32, output_bytes: u64) -> TaskPlan {
        let id = identity();
        let budget = TaskBudget { max_input_tokens: 64, max_output_tokens: 8, max_output_bytes: output_bytes, max_grammar_states: 4096, max_kv_bytes: 1 << 30 };
        let context = PlanContext::new(&id, budget).unwrap();
        let ir = TaskIR::new(
            vec![PromptSegment::new(PromptSegmentKind::TaskInstruction, vec![1]),
                PromptSegment::new(PromptSegmentKind::Document, vec![document_id]),
                PromptSegment::new(PromptSegmentKind::AnswerScaffold, vec![2])],
            DecodeStrategy::ConstrainedJson, GrammarReference::json_schema(Sha256Digest::of_bytes(schema.as_bytes()), JSON_RUNTIME_VERSION),
            None, vec![FinitePostcondition::JsonValid, FinitePostcondition::OutputWithinBudget], budget, DependencyScope::ItemLocal,
        ).unwrap();
        TaskPlan::new(BuiltInTask::Extract.spec(), &context, ir).unwrap()
    }
    pub(super) fn options() -> JsonDecodeOptions { JsonDecodeOptions { max_new_tokens: 8, eos_token_id: 0, excluded_token_ids: BTreeSet::new() } }
    fn plan(task: &TaskPlan, schema: &str) -> ExtractPlan {
        ExtractPlan::from_task_plan(task, schema, options(), CompileLimits::default(), registry().template_controls()).unwrap()
    }
    pub(super) fn output(json: &str) -> JsonDecodeOutput {
        JsonDecodeOutput { schema_version: 1, numerics_profile: HF_BF16_EAGER_PROFILE.to_owned(), token_ids: vec![1, 0], json: json.to_owned(),
            forward_positions: 4, projected_logits: 4 * NANBEIGE_VOCAB_SIZE as u64, mask_node_visit_charge: 20 }
    }

    #[test]
    fn schema_digest_must_match_the_task_not_just_its_shape() {
        let task = task(SCHEMA, 7, 4096);
        assert!(ExtractPlan::from_task_plan(&task, r#"{"type":"null"}"#, options(), CompileLimits::default(), registry().template_controls()).is_err());
    }
    #[test]
    fn untrusted_document_cannot_supply_nonspecial_thinking_control() {
        let task = task(SCHEMA, 3, 4096);
        assert!(ExtractPlan::from_task_plan(&task, SCHEMA, options(), CompileLimits::default(), registry().template_controls()).is_err());
    }
    #[test]
    fn caller_cannot_remove_archived_controls_from_the_output_alphabet() {
        let p = plan(&task(SCHEMA, 7, 4096), SCHEMA);
        assert!(p.options.excluded_token_ids.contains(&0)); assert!(p.options.excluded_token_ids.contains(&3));
    }
    #[test]
    fn valid_result_is_structural_and_scores_are_not_computed() {
        let p = plan(&task(SCHEMA, 7, 4096), SCHEMA);
        let out = p.finalize(output("true")).unwrap();
        assert_eq!(out.grounding, ExtractionGrounding::NotRequested);
        assert_eq!(out.score_space, ScoreSpace::NotComputed);
        let serialized = canonjson::canonical_string(&out).unwrap();
        assert!(!serialized.contains("prompt_digest")); assert!(!serialized.contains("confidence"));
    }
    #[test]
    fn partial_invalid_and_eosless_outputs_never_finalize() {
        let p = plan(&task(SCHEMA, 7, 4096), SCHEMA);
        assert!(p.finalize(output("tru")).is_err()); assert!(p.finalize(output("null")).is_err());
        let mut out = output("true"); out.token_ids.pop(); assert!(p.finalize(out).is_err());
    }
    #[test]
    fn full_result_envelope_not_only_json_is_byte_bounded() {
        let p = plan(&task(SCHEMA, 7, 8), SCHEMA);
        assert!(matches!(p.finalize(output("true")), Err(ExtractError::OutputBudgetExceeded)));
    }
    #[test]
    fn identity_binding_detects_task_prompt_schema_options_and_tokenizer_drift() {
        let p = plan(&task(SCHEMA, 7, 4096), SCHEMA);
        let id = p.bind_identity(identity()).unwrap(); p.verify_identity(&id).unwrap();
        for field in 0..5 {
            let mut changed = id.clone(); let d = Sha256Digest::of_bytes(b"changed");
            match field { 0 => changed.taskir_digest = d, 1 => changed.prompt_digest = d,
                2 => changed.schema_digest = d, 3 => changed.decision_policy_digest = d, _ => changed.tokenizer_digest = d }
            assert!(p.verify_identity(&changed).is_err());
        }
    }
    #[test]
    fn profile_thinking_and_tool_modes_are_not_silently_downgraded() {
        let p = plan(&task(SCHEMA, 7, 4096), SCHEMA);
        let mut id = identity(); id.thinking_mode = ThinkingMode::Enabled; assert!(p.bind_identity(id).is_err());
        let mut id = identity(); id.tool_mode = ToolMode::Json; assert!(p.bind_identity(id).is_err());
        let mut id = identity(); id.numerics_profile = NumericsProfile::DiagnosticF32; assert!(p.bind_identity(id).is_err());
    }
    #[test]
    fn output_language_caps_and_extra_bans_change_the_bound_policy() {
        let task = task(SCHEMA, 7, 4096); let reg = registry();
        let a = plan(&task, SCHEMA);
        let b = ExtractPlan::from_task_plan(&task, SCHEMA, options(), CompileLimits { max_array_items: 1, ..CompileLimits::default() }, reg.template_controls()).unwrap();
        assert_ne!(a.policy_digest, b.policy_digest);
        let mut opts = options(); opts.excluded_token_ids.insert(8);
        let c = ExtractPlan::from_task_plan(&task, SCHEMA, opts, CompileLimits::default(), reg.template_controls()).unwrap();
        assert_ne!(a.policy_digest, c.policy_digest);
    }
    #[test]
    fn a_declared_source_verification_postcondition_cannot_be_ignored() {
        let task = task(SCHEMA, 7, 4096);
        let mut wire = serde_json::to_value(task.ir()).unwrap();
        wire["postconditions"].as_array_mut().unwrap().push(serde_json::to_value(FinitePostcondition::SourceSpansVerified).unwrap());
        let ir: TaskIR = serde_json::from_value(wire).unwrap();
        ir.validate().unwrap();
        assert!(matches!(check_postconditions(&ir), Err(ExtractError::Contract("unsupported extraction postcondition"))));
    }
    #[test]
    fn source_grounding_and_overbudget_decode_requests_are_refused() {
        let source_schema = r#"{"type":"string","x-fnlp-source":"verbatim"}"#;
        let t = task(source_schema, 7, 4096);
        assert!(ExtractPlan::from_task_plan(&t, source_schema, options(), CompileLimits::default(), registry().template_controls()).is_err());
        let t = task(SCHEMA, 7, 4096); let mut opts = options(); opts.max_new_tokens = 9;
        assert!(ExtractPlan::from_task_plan(&t, SCHEMA, opts, CompileLimits::default(), registry().template_controls()).is_err());
    }
}
