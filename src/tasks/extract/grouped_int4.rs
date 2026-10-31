//! Schema/source extraction bound to the actual grouped-INT4 model.
//!
//! The existing extraction compiler and independent validator remain the
//! authority for JSON, exact decimals and source coordinates. This adapter
//! seals the distinct native profile and complete group-aware work; it never
//! converts an INT8/BF16 executable or accepts a caller-supplied native result.

use super::*;
use crate::native_engine::{
    constrained_int4::{self, INT4_JSON_EXECUTION, Int4JsonBudget, Int4JsonError, Int4JsonRun},
    strict_int4::{Int4Work, StrictInt4Engine, STRICT_INT4_PROFILE},
};

pub const INT4_EXTRACT_VERSION: &str = "grouped-int4-schema-source-extraction-v1";
pub mod cohort;

#[derive(Debug)]
pub enum Int4ExtractError { Extraction(ExtractError), Native(Int4JsonError), Identity }
impl fmt::Display for Int4ExtractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Extraction(e) => write!(f, "INT4 extraction refused: {e}"),
            Self::Native(e) => write!(f, "INT4 extraction execution refused: {e}"),
            Self::Identity => f.write_str("INT4 extraction model, partition or admitted identity differs"),
        }
    }
}
impl Error for Int4ExtractError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Extraction(e) => Some(e), Self::Native(e) => Some(e), Self::Identity => None }
    }
}
impl From<ExtractError> for Int4ExtractError { fn from(e: ExtractError) -> Self { Self::Extraction(e) } }
impl From<Int4JsonError> for Int4ExtractError { fn from(e: Int4JsonError) -> Self { Self::Native(e) } }
impl Int4ExtractError {
    pub fn cancellation(&self) -> Option<crate::native_engine::decode::DecodeCancellationKind> {
        match self {
            Self::Native(e) => e.cancellation(),
            Self::Extraction(ExtractError::Decode(JsonDecodeError::Cancelled(cause))) => Some(*cause),
            _ => None,
        }
    }
}

/// A completed independently checked result; source membership is a byte/span
/// guarantee, not extraction accuracy. Private prompt/source digests remain
/// in the sealed plan and are never exported as public content fingerprints.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Int4ExtractRun {
    pub schema_version: u32,
    pub execution: String,
    pub group_width: usize,
    pub result: ExtractResult,
    pub model_work: Int4Work,
}

/// Private immutable tokens, schema, source automaton and complete identity.
/// No wire constructor, Debug, Deref or coercion into another native profile.
pub struct Int4ExtractPlan {
    extraction: ExtractPlan,
    identity: ExecutionIdentity,
    comparison_policy_digest: Sha256Digest,
    group_width: usize,
    work: Int4Work,
}
impl Int4ExtractPlan {
    #[allow(clippy::too_many_arguments)]
    pub fn from_task_plan(task: &TaskPlan, schema: &str, options: JsonDecodeOptions,
        limits: CompileLimits, controls: &TemplateControlIds, identity: ExecutionIdentity,
        group_width: usize) -> Result<Self, Int4ExtractError> {
        Self::from_builtin(task, schema, options, limits, controls, None, "extract-v1", identity, group_width)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_task_plan_with_source(task: &TaskPlan, schema: &str, options: JsonDecodeOptions,
        limits: CompileLimits, controls: &TemplateControlIds, source: &SourceDocument,
        source_limits: SourceRuntimeLimits, identity: ExecutionIdentity, group_width: usize)
        -> Result<Self, Int4ExtractError> {
        Self::from_builtin(task, schema, options, limits, controls, Some((source, source_limits)), "extract-v1", identity, group_width)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_builtin(task: &TaskPlan, schema: &str, options: JsonDecodeOptions, limits: CompileLimits,
        controls: &TemplateControlIds, source: Option<(&SourceDocument, SourceRuntimeLimits)>,
        task_identity: &'static str, mut identity: ExecutionIdentity, group_width: usize) -> Result<Self, Int4ExtractError> {
        // Refuse profile, candidate namespace and group mismatch BEFORE any
        // source copies or grammar construction. Model identity is rechecked
        // against actual resident weights by the native execution boundary.
        constrained_int4::check_profile(&identity, group_width)?;
        if identity.task_spec != task_identity { return Err(Int4ExtractError::Identity); }
        let extraction = ExtractPlan::from_builtin(task, schema, options, limits, controls, source, task_identity)?;
        let work = constrained_int4::planned_work(group_width, extraction.prompt.len(), extraction.options.max_new_tokens)?;
        identity.taskir_digest = extraction.taskir_digest;
        identity.prompt_digest = extraction.prompt_digest;
        identity.schema_digest = extraction.schema_digest;
        identity.grammar_compiler_version = extraction.program.version().to_owned();
        identity.sampler_version = EXTRACT_SAMPLER_VERSION.to_owned();
        identity.tokenizer_digest = Sha256Digest::of_bytes(PINNED_TOKENIZER_MODEL_BYTES);
        identity.decision_policy_digest = Sha256Digest::of_bytes(&canonjson::canonical_bytes(&(
            INT4_EXTRACT_VERSION, INT4_JSON_EXECUTION, group_width, extraction.policy_digest,
        )).map_err(|_| ExtractError::Serialization)?);
        let comparison_policy_digest = Sha256Digest::of_bytes(&canonjson::canonical_bytes(&(
            INT4_EXTRACT_VERSION, INT4_JSON_EXECUTION, "comparison-subject-group-width-v1", extraction.policy_digest,
        )).map_err(|_| ExtractError::Serialization)?);
        identity.validate().map_err(|_| Int4ExtractError::Identity)?;
        Ok(Self { extraction, identity, comparison_policy_digest, group_width, work })
    }

    // Built-in task finalization is part of the admitted policy, not a
    // post-execution relabeling of a generic extraction result.
    pub(crate) fn with_finalizer_version(mut self, version: &'static str) -> Result<Self, Int4ExtractError> {
        self.identity.decision_policy_digest = Sha256Digest::of_bytes(&canonjson::canonical_bytes(&(
            "int4-built-in-finalizer-v1", version, self.identity.decision_policy_digest,
        )).map_err(|_| ExtractError::Serialization)?);
        self.comparison_policy_digest = Sha256Digest::of_bytes(&canonjson::canonical_bytes(&(
            "int4-built-in-finalizer-v1", version, self.comparison_policy_digest,
        )).map_err(|_| ExtractError::Serialization)?);
        self.identity.validate().map_err(|_| Int4ExtractError::Identity)?;
        Ok(self)
    }

    pub fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    pub fn prompt_tokens(&self) -> usize { self.extraction.prompt.len() }
    pub fn options(&self) -> &JsonDecodeOptions { &self.extraction.options }
    pub fn group_width(&self) -> usize { self.group_width }
    pub fn planned_work(&self) -> Int4Work { self.work }
    pub fn max_result_bytes(&self) -> u64 { self.extraction.max_result_bytes }
    pub fn max_kv_bytes(&self) -> u64 { self.extraction.max_kv_bytes }
    // Paired audit compatibility may ignore only the comparison subject's
    // group width. Complete execution admission still uses the full identity.
    pub(crate) fn comparison_policy_digest(&self) -> Sha256Digest { self.comparison_policy_digest }

    /// All artifact, layout, task, policy and calibration fields are checked.
    /// Execution never repairs a caller's partially matching admission key.
    pub fn verify_identity(&self, admitted: &ExecutionIdentity) -> Result<(), Int4ExtractError> {
        constrained_int4::check_profile(admitted, self.group_width)?;
        let actual = canonjson::canonical_bytes(admitted).map_err(|_| Int4ExtractError::Identity)?;
        let expected = canonjson::canonical_bytes(&self.identity).map_err(|_| Int4ExtractError::Identity)?;
        if actual != expected { return Err(Int4ExtractError::Identity); }
        Ok(())
    }

    /// Recheck a composed result against this sealed schema/source before
    /// lifting coordinates or reducing it. This internal verifier does not
    /// authenticate a serialized receipt or replace native execution.
    pub(crate) fn verify_completed(&self, run: &Int4ExtractRun) -> Result<(), Int4ExtractError> {
        // Bound the original envelope before the temporary validation copy.
        super::quantized::check_size(run, self.max_result_bytes())?;
        if run.schema_version != 1 || run.execution != INT4_EXTRACT_VERSION || run.group_width != self.group_width {
            return Err(ExtractError::InvalidResult.into());
        }
        let output = &run.result.output;
        let expected = constrained_int4::planned_work(self.group_width, self.prompt_tokens(), output.token_ids.len())?;
        if run.model_work != expected || output.forward_positions != expected.forward_positions
            || output.projected_logits != expected.projected_logits {
            return Err(Int4JsonError::WorkMismatch.into());
        }
        // Independently recheck exact decimal JSON, schema, EOS/control IDs,
        // profile, source membership and every local occurrence coordinate.
        let checked = self.extraction.finalize_profile(output.clone(), STRICT_INT4_PROFILE)?;
        if checked != run.result { return Err(ExtractError::InvalidResult.into()); }
        Ok(())
    }

    /// Internal semantic composition may inspect the compiled schema only
    /// while the original task and complete byte-token source remain attached.
    /// This does not expose an eager executable or accept a replacement schema.
    pub(crate) fn semantic_schema(&self, task: &TaskPlan, source: &SourceDocument)
        -> Result<&crate::grammar::SchemaNode, Int4ExtractError> {
        if self.extraction.task_identity != "extract-v1" || task.task_spec_identity() != "extract-v1"
            || task.ir().digest().map_err(|_| ExtractError::Serialization)? != self.extraction.taskir_digest {
            return Err(Int4ExtractError::Identity);
        }
        let mut documents = task.ir().prompt_segments().iter()
            .filter(|segment| segment.kind() == PromptSegmentKind::Document);
        if documents.next().is_none_or(|segment| segment.token_ids() != source.token_ids())
            || documents.next().is_some() || source.text().is_empty()
            || source.text().len() != source.token_ids().len() {
            return Err(Int4ExtractError::Identity);
        }
        Ok(self.extraction.program.declarative_schema())
    }

    pub fn execute<C: DecodeStepControl>(&self, engine: &mut StrictInt4Engine<'_>,
        admitted: &ExecutionIdentity, vocabulary: &ExtractionVocabulary, budget: Int4JsonBudget,
        control: &mut C) -> Result<Int4ExtractRun, Int4ExtractError> {
        self.execute_with(engine, admitted, vocabulary, budget, control, Ok)
    }

    /// Compose only shipped task finalizers inside the exclusive native
    /// session. A semantic/output refusal and late cancellation still abort
    /// and drain every KV slot before host reservations can be released.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_with<C, T, E, F>(&self, engine: &mut StrictInt4Engine<'_>,
        admitted: &ExecutionIdentity, vocabulary: &ExtractionVocabulary, mut budget: Int4JsonBudget,
        control: &mut C, finalize: F) -> Result<T, E>
    where C: DecodeStepControl, E: From<Int4ExtractError> + From<Int4JsonError>,
        F: FnOnce(Int4ExtractRun) -> Result<T, E> {
        self.verify_identity(admitted).map_err(E::from)?;
        if vocabulary.controls != self.extraction.controls {
            return Err(E::from(Int4ExtractError::from(ExtractError::Contract(
                "vocabulary control registry differs from INT4 plan"))));
        }
        budget.json.max_kv_bytes = budget.json.max_kv_bytes.min(self.extraction.max_kv_bytes);
        // The native session owns cleanup and remains live throughout source
        // validation, complete-result accounting and the final checkpoint.
        constrained_int4::decode_json_int4_with(engine, admitted, &self.extraction.prompt,
            &self.extraction.program, &vocabulary.oracle, &self.extraction.options, budget,
            control, |run| finalize(self.finalize(run).map_err(E::from)?))
    }

    fn finalize(&self, run: Int4JsonRun) -> Result<Int4ExtractRun, Int4ExtractError> {
        if run.schema_version != 1 || run.execution != INT4_JSON_EXECUTION || run.group_width != self.group_width {
            return Err(ExtractError::InvalidResult.into());
        }
        let expected = constrained_int4::planned_work(self.group_width, self.prompt_tokens(), run.output.token_ids.len())?;
        if run.model_work != expected || run.output.forward_positions != expected.forward_positions
            || run.output.projected_logits != expected.projected_logits {
            return Err(Int4JsonError::WorkMismatch.into());
        }
        let result = self.extraction.finalize_profile(run.output, STRICT_INT4_PROFILE)?;
        let output = Int4ExtractRun { schema_version: 1, execution: INT4_EXTRACT_VERSION.to_owned(),
            group_width: self.group_width, result, model_work: run.model_work };
        // This existing bounded serializer accepts only trusted typed values;
        // it counts the OUTER envelope before allocating canonical JSON.
        super::quantized::check_size(&output, self.max_result_bytes())?;
        Ok(output)
    }
}

#[cfg(test)] mod tests;
