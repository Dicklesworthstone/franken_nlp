//! Identity-sealed schema/source extraction using actual strict-int8 forwards.
//!
//! Compilation reuses ExtractPlan's exact TaskIR, schema, untrusted-source and
//! control-alphabet checks. This distinct type cannot enter the eager driver.
//! Source occurrences and the complete result envelope are validated before
//! the native session closes; no parse retry or implicit BF16 fallback exists.

use std::io::{self, Write};
use super::*;
use crate::native_engine::{
    constrained_int8::{self, INT8_JSON_EXECUTION, Int8JsonBudget, Int8JsonError, Int8JsonRun},
    strict_int8::{Int8Work, StrictInt8Engine, STRICT_INT8_PROFILE},
};

pub const INT8_EXTRACT_VERSION: &str = "strict-int8-schema-source-extraction-v1";

#[derive(Debug)]
pub enum Int8ExtractError { Extraction(ExtractError), Native(Int8JsonError), Identity }
impl fmt::Display for Int8ExtractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Extraction(e) => write!(f, "int8 extraction refused: {e}"),
            Self::Native(e) => write!(f, "int8 extraction execution refused: {e}"),
            Self::Identity => f.write_str("int8 extraction requires its exact admitted execution identity"),
        }
    }
}
impl Error for Int8ExtractError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Extraction(e) => Some(e), Self::Native(e) => Some(e), Self::Identity => None }
    }
}
impl From<ExtractError> for Int8ExtractError { fn from(e: ExtractError) -> Self { Self::Extraction(e) } }
impl From<Int8JsonError> for Int8ExtractError { fn from(e: Int8JsonError) -> Self { Self::Native(e) } }
impl Int8ExtractError {
    pub fn cancellation(&self) -> Option<crate::native_engine::decode::DecodeCancellationKind> {
        match self {
            Self::Native(e) => e.cancellation(),
            Self::Extraction(ExtractError::Decode(JsonDecodeError::Cancelled(cause))) => Some(*cause),
            _ => None,
        }
    }
}

/// The result keeps exact JSON/decimal text, source occurrences and the named
/// quantized profile. No score or source digest is invented. All integer and
/// attention work is retained alongside the structured task result.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Int8ExtractRun {
    pub schema_version: u32,
    pub execution: String,
    pub result: ExtractResult,
    pub model_work: Int8Work,
}

/// Private prompt/program/identity ownership; no wire constructor, Debug,
/// Deref or conversion to the BF16 executable plan.
pub struct Int8ExtractPlan {
    extraction: ExtractPlan,
    identity: ExecutionIdentity,
    work: Int8Work,
}
impl Int8ExtractPlan {
    pub fn from_task_plan(task: &TaskPlan, schema: &str, options: JsonDecodeOptions,
        limits: CompileLimits, controls: &TemplateControlIds, identity: ExecutionIdentity)
        -> Result<Self, Int8ExtractError> {
        Self::from_builtin(task, schema, options, limits, controls, None, "extract-v1", identity)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_task_plan_with_source(task: &TaskPlan, schema: &str, options: JsonDecodeOptions,
        limits: CompileLimits, controls: &TemplateControlIds, source: &SourceDocument,
        source_limits: SourceRuntimeLimits, identity: ExecutionIdentity) -> Result<Self, Int8ExtractError> {
        Self::from_builtin(task, schema, options, limits, controls, Some((source, source_limits)), "extract-v1", identity)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_builtin(task: &TaskPlan, schema: &str, options: JsonDecodeOptions,
        limits: CompileLimits, controls: &TemplateControlIds, source: Option<(&SourceDocument, SourceRuntimeLimits)>,
        task_identity: &'static str, mut identity: ExecutionIdentity) -> Result<Self, Int8ExtractError> {
        // Refuse wrong profiles before source copies or grammar allocation.
        constrained_int8::check_profile(&identity)?;
        if identity.task_spec != task_identity { return Err(Int8ExtractError::Identity); }
        let extraction = ExtractPlan::from_builtin(task, schema, options, limits, controls, source, task_identity)?;
        let work = constrained_int8::planned_work(extraction.prompt.len(), extraction.options.max_new_tokens)?;
        identity.taskir_digest = extraction.taskir_digest;
        identity.prompt_digest = extraction.prompt_digest;
        identity.schema_digest = extraction.schema_digest;
        identity.grammar_compiler_version = extraction.program.version().to_owned();
        identity.sampler_version = EXTRACT_SAMPLER_VERSION.to_owned();
        identity.tokenizer_digest = Sha256Digest::of_bytes(PINNED_TOKENIZER_MODEL_BYTES);
        identity.decision_policy_digest = Sha256Digest::of_bytes(&canonjson::canonical_bytes(&(
            INT8_EXTRACT_VERSION, INT8_JSON_EXECUTION, extraction.policy_digest,
        )).map_err(|_| ExtractError::Serialization)?);
        identity.validate().map_err(|_| Int8ExtractError::Identity)?;
        Ok(Self { extraction, identity, work })
    }

    pub fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    pub fn prompt_tokens(&self) -> usize { self.extraction.prompt.len() }
    pub fn options(&self) -> &JsonDecodeOptions { &self.extraction.options }
    pub fn planned_work(&self) -> Int8Work { self.work }
    pub fn max_result_bytes(&self) -> u64 { self.extraction.max_result_bytes }

    /// Check ALL fields, including model, packing, template, calibration and
    /// backend facts. Execution cannot silently repair a partly matching key.
    pub fn verify_identity(&self, admitted: &ExecutionIdentity) -> Result<(), Int8ExtractError> {
        constrained_int8::check_profile(admitted)?;
        let actual = canonjson::canonical_bytes(admitted).map_err(|_| Int8ExtractError::Identity)?;
        let expected = canonjson::canonical_bytes(&self.identity).map_err(|_| Int8ExtractError::Identity)?;
        if actual != expected { return Err(Int8ExtractError::Identity); }
        Ok(())
    }

    pub fn execute<C: DecodeStepControl>(&self, engine: &mut StrictInt8Engine<'_>, admitted: &ExecutionIdentity,
        vocabulary: &ExtractionVocabulary, mut budget: Int8JsonBudget, control: &mut C)
        -> Result<Int8ExtractRun, Int8ExtractError> {
        self.verify_identity(admitted)?;
        if vocabulary.controls != self.extraction.controls {
            return Err(ExtractError::Contract("vocabulary control registry differs from int8 plan").into());
        }
        budget.json.max_kv_bytes = budget.json.max_kv_bytes.min(self.extraction.max_kv_bytes);
        constrained_int8::decode_json_int8_with(engine, admitted, &self.extraction.prompt,
            &self.extraction.program, &vocabulary.oracle, &self.extraction.options, budget, control,
            |run| self.finalize(run))
    }

    fn finalize(&self, run: Int8JsonRun) -> Result<Int8ExtractRun, Int8ExtractError> {
        if run.schema_version != 1 || run.execution != INT8_JSON_EXECUTION {
            return Err(ExtractError::InvalidResult.into());
        }
        let count = run.output.token_ids.len();
        if count == 0 || count > self.options().max_new_tokens { return Err(ExtractError::InvalidResult.into()); }
        let expected = constrained_int8::planned_work(self.prompt_tokens(), count)?;
        if run.model_work != expected || run.output.forward_positions != expected.forward_positions
            || run.output.projected_logits != expected.projected_logits {
            return Err(Int8JsonError::WorkMismatch.into());
        }
        let result = self.extraction.finalize_profile(run.output, STRICT_INT8_PROFILE)?;
        let output = Int8ExtractRun { schema_version: 1, execution: INT8_EXTRACT_VERSION.to_owned(),
            result, model_work: run.model_work };
        check_size(&output, self.max_result_bytes())?;
        Ok(output)
    }
}

// Check the OUTER envelope, including all source evidence and model work. The
// bounded counting pass precedes canonical tree/byte allocation. Only trusted
// statically typed result serializers reach this internal boundary.
fn check_size(value: &impl Serialize, cap: u64) -> Result<(), ExtractError> {
    struct Counter { remaining: u64, overflow: bool }
    impl Write for Counter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if bytes.len() as u64 > self.remaining {
                self.overflow = true; return Err(io::Error::other("int8 extraction output bound"));
            }
            self.remaining -= bytes.len() as u64; Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> { Ok(()) }
    }
    let mut counter = Counter { remaining: cap, overflow: false };
    if serde_json::to_writer(&mut counter, value).is_err() {
        return Err(if counter.overflow { ExtractError::OutputBudgetExceeded } else { ExtractError::Serialization });
    }
    if canonjson::canonical_bytes(value).map_err(|_| ExtractError::Serialization)?.len() as u64 > cap {
        return Err(ExtractError::OutputBudgetExceeded);
    }
    Ok(())
}

#[cfg(test)] mod tests;
