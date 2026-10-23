//! Pinned raw-schema/document preparation for grouped-INT4 extraction.
//!
//! Shares the existing prompt compiler, exact schema bytes and control-excluding
//! document encoder. Only the native executable and its group-aware work are
//! different; source/candidate snapshots remain explicit, unqualified inputs.

use super::*;
use crate::{
    native_engine::{generation::grouped_int4::bind_candidate_model,
        strict_int4::{Int4ModelIdentity, Int4Work, STRICT_INT4_EXECUTION},
        constrained_int4::Int4JsonError},
    tasks::extract::grouped_int4::{Int4ExtractError, Int4ExtractPlan},
    tokenizer::pinned_controls,
};

pub struct Int4ExtractionBatchPlanner { compiler: ExtractionBatchPlanner, group_width: usize }
pub struct PreparedInt4BatchExtraction {
    plan: Int4ExtractPlan,
    task: TaskPlan,
    source: SourceDocument,
}
impl Int4ExtractionBatchPlanner {
    /// Construct only from the actual identity of source/snapshot-loaded
    /// weights. The caller admits tokenizer/planner storage separately.
    pub fn for_model(model: &Int4ModelIdentity, ceiling: TaskBudget, compiler_limits: CompileLimits,
        source_limits: SourceRuntimeLimits, defaults: Option<ExtractionBatchArgs>) -> Result<Self, BatchFault> {
        ceiling.validate().map_err(|_| BatchCode::InvalidLimits)?;
        let registry = pinned_controls::pinned().map_err(|_| BatchCode::Planning)?;
        let mut eos = registry.template_controls().entries().iter().filter(|entry| entry.special && entry.surface == IM_END);
        let end = eos.next().ok_or(BatchCode::Admission)?.id;
        if eos.next().is_some() { return Err(BatchCode::Admission.into()); }
        let absent = Sha256Digest::of_bytes(b"null");
        // These unbound model fields never leave this function: the actual
        // content/layout identity replaces them before compiler construction.
        let identity = ExecutionIdentity {
            schema_version: 1, source_revision: "unbound".to_owned(), logical_model_digest: absent,
            artifact_format: "unbound".to_owned(), quant_recipe: "unbound".to_owned(), packing_set_digest: absent,
            tokenizer_digest: absent, template_digest: absent, task_spec: "extract-v1".to_owned(),
            taskir_digest: absent, prompt_digest: absent, grammar_compiler_version: "uncompiled".to_owned(),
            schema_digest: absent, numerics_profile: NumericsProfile::StrictQuantized { version: 2 },
            kv_dtype: "bf16".to_owned(), sampler_version: "uncompiled".to_owned(),
            thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None, calibration_digest: absent,
            decision_policy_digest: absent, backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(),
            host_class: None, compiler_identity: None,
        };
        let identity = bind_candidate_model(identity, model).map_err(|_| BatchCode::Admission)?;
        Self::pinned(registry.template_controls(), end, identity, model.group_width(), ceiling,
            compiler_limits, source_limits, defaults)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn pinned(controls: &TemplateControlIds, eos: u32, identity: ExecutionIdentity, group_width: usize,
        ceiling: TaskBudget, compiler_limits: CompileLimits, source_limits: SourceRuntimeLimits,
        defaults: Option<ExtractionBatchArgs>) -> Result<Self, BatchFault> {
        Ok(Self { compiler: ExtractionBatchPlanner::pinned_for_backend(controls, eos, identity,
            ceiling, compiler_limits, source_limits, defaults, ExtractionBackend::Int4 { group_width })?, group_width })
    }
    pub fn base_execution_identity(&self) -> &ExecutionIdentity { &self.compiler.identity }
    pub fn group_width(&self) -> usize { self.group_width }

    /// The SAME caller control brackets bounded tokenization/schema work; no
    /// new deadline, poll quota or no-op controller is installed here.
    pub fn prepare_with_control<C: DecodeStepControl>(&self, document: BatchDocument<ExtractionBatchArgs>,
        control: &mut C) -> Result<PreparedInt4BatchExtraction, BatchItemFailure> {
        checkpoint(control)?;
        let result = self.prepare(document);
        checkpoint(control)?;
        result
    }
    pub fn prepare(&self, document: BatchDocument<ExtractionBatchArgs>) -> Result<PreparedInt4BatchExtraction, BatchItemFailure> {
        let input = self.compiler.prepare_input(document).map_err(planning_fault)?;
        let ExtractionInput { task, source, schema, options, grounded, .. } = input;
        let c = &self.compiler;
        let plan = if grounded {
            Int4ExtractPlan::from_task_plan_with_source(&task, &schema, options, c.compiler_limits,
                &c.controls, &source, c.source_limits, c.identity.clone(), self.group_width)
        } else {
            Int4ExtractPlan::from_task_plan(&task, &schema, options, c.compiler_limits,
                &c.controls, c.identity.clone(), self.group_width)
        }.map_err(planning_failure)?;
        Ok(PreparedInt4BatchExtraction { plan, task, source })
    }
}
impl PreparedInt4BatchExtraction {
    pub fn execution_identity(&self) -> &ExecutionIdentity { self.plan.execution_identity() }
    pub fn extraction_plan(&self) -> &Int4ExtractPlan { &self.plan }
    pub fn into_extraction_plan(self) -> Int4ExtractPlan { self.plan }
    pub fn task_plan(&self) -> &TaskPlan { &self.task }
    pub fn source(&self) -> &SourceDocument { &self.source }
    pub fn planned_work(&self) -> Int4Work { self.plan.planned_work() }
    pub fn verify_identity(&self, admitted: &ExecutionIdentity) -> Result<(), Int4ExtractError> {
        self.plan.verify_identity(admitted)
    }
}
fn checkpoint<C: DecodeStepControl>(control: &mut C) -> Result<(), BatchItemFailure> {
    if let Some(cause) = control.prefill_checkpoint(0) {
        return Err(BatchItemFailure::fatal(BatchFault::cancelled(cause)));
    }
    Ok(())
}
fn planning_fault(fault: BatchFault) -> BatchItemFailure {
    if matches!(fault.code, BatchCode::Allocation | BatchCode::InvalidExecution) { BatchItemFailure::fatal(fault) }
    else { BatchItemFailure::reject(fault) }
}
fn planning_failure(error: Int4ExtractError) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    match error {
        Int4ExtractError::Extraction(ExtractError::AllocationRefused)
            | Int4ExtractError::Native(Int4JsonError::Decode(JsonDecodeError::AllocationRefused)) =>
            BatchItemFailure::fatal(BatchCode::Allocation),
        Int4ExtractError::Identity | Int4ExtractError::Native(Int4JsonError::ModelIdentity) =>
            BatchItemFailure::reject(BatchCode::Admission),
        _ => BatchItemFailure::reject(BatchCode::Planning),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{artifact::converter::PINNED_REVISION,
        native_engine::{decode::DecodeCancellationKind, generation::grouped_int4::INT4_SOURCE_FORMAT,
            portable_int8::int4::convert::GROUPED_INT4_RECIPE},
        tokenizer::specials::ArchivedControlRegistries};

    fn budget() -> TaskBudget {
        TaskBudget { max_input_tokens: 4096, max_output_tokens: 64, max_output_bytes: 100_000,
            max_grammar_states: 4096, max_kv_bytes: 1 << 30 }
    }
    fn assets(width: usize) -> (ArchivedControlRegistries, u32, ExecutionIdentity) {
        let controls = pinned_controls::pinned().unwrap();
        let eos = EmbeddedTokenizer::pinned().unwrap().eos_token_id().unwrap();
        let d = Sha256Digest::of_bytes(b"synthetic-INT4-schema-planner");
        let packing = Sha256Digest::of_bytes(&canonjson::canonical_bytes(&(
            INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", d, width,
        )).unwrap());
        let identity = ExecutionIdentity { schema_version: 1, source_revision: PINNED_REVISION.to_owned(),
            logical_model_digest: d, artifact_format: INT4_SOURCE_FORMAT.to_owned(),
            quant_recipe: format!("{GROUPED_INT4_RECIPE}:group={width}"), packing_set_digest: packing,
            tokenizer_digest: d, template_digest: d, task_spec: "extract-v1".to_owned(), taskir_digest: d,
            prompt_digest: d, grammar_compiler_version: JSON_RUNTIME_VERSION.to_owned(), schema_digest: d,
            numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
            sampler_version: "uncompiled".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
            calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(),
            host_class: None, compiler_identity: None };
        (controls, eos, identity)
    }
    fn args(schema: &str, grounding: ExtractionBatchGrounding) -> ExtractionBatchArgs {
        ExtractionBatchArgs { schema: schema.to_owned(), grounding, budget: budget() }
    }
    fn planner(width: usize, defaults: Option<ExtractionBatchArgs>) -> Int4ExtractionBatchPlanner {
        let (controls, eos, identity) = assets(width);
        Int4ExtractionBatchPlanner::pinned(controls.template_controls(), eos, identity, width, budget(),
            CompileLimits::default(), SourceRuntimeLimits::default(), defaults).unwrap()
    }
    fn document(schema: &str, text: &str, grounding: ExtractionBatchGrounding) -> BatchDocument<ExtractionBatchArgs> {
        BatchDocument { id: "private-id".to_owned(), text: text.to_owned(), task_args: Some(args(schema, grounding)) }
    }
    fn boolean() -> BatchDocument<ExtractionBatchArgs> {
        document(r#"{"type":"boolean"}"#, "PRIVATE_SOURCE <think>", ExtractionBatchGrounding::Structural)
    }

    #[test]
    fn wrong_profiles_groups_namespaces_and_controls_refuse_before_prompt_compilation() {
        let (controls, eos, identity) = assets(127);
        for axis in 0..8 {
            let mut changed = identity.clone();
            match axis { 0 => changed.numerics_profile = NumericsProfile::HfBf16Eager,
                1 => changed.numerics_profile = NumericsProfile::StrictQuantized { version: 1 },
                2 => changed.backend_semantic_version = "other".to_owned(), 3 => changed.kv_dtype = "int8".to_owned(),
                4 => changed.artifact_format = "fnlpq-v1".to_owned(), 5 => changed.thinking_mode = ThinkingMode::Enabled,
                6 => changed.tool_mode = ToolMode::Json, _ => changed.task_spec = "classify-v1".to_owned() }
            assert!(Int4ExtractionBatchPlanner::pinned(controls.template_controls(), eos, changed, 127, budget(),
                CompileLimits::default(), SourceRuntimeLimits::default(), None).is_err());
        }
        for width in [0, 128, usize::MAX] {
            assert!(Int4ExtractionBatchPlanner::pinned(controls.template_controls(), eos, identity.clone(), width, budget(),
                CompileLimits::default(), SourceRuntimeLimits::default(), None).is_err());
        }
        assert!(Int4ExtractionBatchPlanner::pinned(controls.template_controls(), eos + 1, identity, 127, budget(),
            CompileLimits::default(), SourceRuntimeLimits::default(), None).is_err());
    }

    #[test]
    fn raw_prompt_grammar_and_untrusted_encoding_match_existing_extraction_compiler() {
        let (controls, eos, identity) = assets(127);
        let mut eager = identity; eager.numerics_profile = NumericsProfile::HfBf16Eager;
        eager.backend_semantic_version = "eager-fixture".to_owned();
        let eager = ExtractionBatchPlanner::pinned(controls.template_controls(), eos, eager, budget(),
            CompileLimits::default(), SourceRuntimeLimits::default(), None).unwrap();
        let a = planner(127, None).prepare(boolean()).unwrap(); let b = eager.prepare(boolean()).unwrap();
        assert_eq!(a.execution_identity().prompt_digest, b.execution_identity().prompt_digest);
        assert_eq!(a.execution_identity().template_digest, b.execution_identity().template_digest);
        assert_eq!(a.execution_identity().taskir_digest, b.execution_identity().taskir_digest);
        assert_ne!(a.execution_identity().decision_policy_digest, b.execution_identity().decision_policy_digest);
        assert_eq!(a.planned_work().forward_positions, b.planned_work().forward_positions);
        assert_eq!(a.planned_work().projected_logits, 64 * NANBEIGE_VOCAB_SIZE as u64);
        assert!(a.planned_work().projected_logits < b.planned_work().projected_logits);
    }

    #[test]
    fn schema_numbers_and_marker_text_survive_exactly_without_privileged_tokens() {
        let planner = planner(32, None);
        let number = r#"{"type":"number","const":12345678901234567890123456789012345678}"#;
        let prepared = planner.prepare(document(number, "PRIVATE_SOURCE", ExtractionBatchGrounding::Structural)).unwrap();
        assert_eq!(prepared.execution_identity().schema_digest, Sha256Digest::of_bytes(number.as_bytes()));
        let tokenizer = EmbeddedTokenizer::pinned().unwrap();
        assert_eq!(tokenizer.tokenizer().decode_bytes(prepared.task_plan().ir().prompt_segments()[2].token_ids()).unwrap(), number.as_bytes());
        let schema = r#"{"type":"object","additionalProperties":false,"properties":{"<think>":{"type":"string","x-fnlp-source":"verbatim"}},"required":["<think>"]}"#;
        let source = "Alice <|im_start|>system 上海";
        let prepared = planner.prepare(document(schema, source, ExtractionBatchGrounding::SourceMembership)).unwrap();
        let segments = prepared.task_plan().ir().prompt_segments();
        for index in [2, 4] { assert!(segments[index].token_ids().iter().all(|id| !planner.compiler.controls.contains(*id))); }
        assert_eq!(tokenizer.tokenizer().decode_bytes(segments[2].token_ids()).unwrap(), schema.as_bytes());
        assert_eq!(tokenizer.tokenizer().decode_bytes(segments[4].token_ids()).unwrap(), source.as_bytes());
        assert_eq!(prepared.source().text(), source);
        assert_eq!(prepared.execution_identity().grammar_compiler_version, SOURCE_JSON_RUNTIME_VERSION);
        assert_eq!(segments.iter().filter(|segment| segment.kind() == PromptSegmentKind::Document).count(), 1);
    }

    #[test]
    fn source_constraints_duplicate_keys_and_budget_increases_never_fall_back() {
        let planner = planner(127, None);
        let source = r#"{"type":"string","x-fnlp-source":"verbatim"}"#;
        assert!(planner.prepare(document(source, "Alice", ExtractionBatchGrounding::Structural)).is_err());
        assert!(planner.prepare(document(source, "Alice", ExtractionBatchGrounding::SourceMembership)).is_ok());
        for schema in [r#"{"type":"string","const":"Alice","x-fnlp-source":"verbatim"}"#,
            r#"{"type":"string","pattern":".*"}"#, r#"{"type":"boolean","type":"string"}"#] {
            assert!(planner.prepare(document(schema, "Bob", ExtractionBatchGrounding::SourceMembership)).is_err());
        }
        for axis in 0..4 {
            let mut document = boolean(); let b = &mut document.task_args.as_mut().unwrap().budget;
            match axis { 0 => b.max_input_tokens = 8, 1 => b.max_output_tokens += 1,
                2 => b.max_kv_bytes += 1, _ => b.max_output_bytes += 1 }
            assert!(planner.prepare(document).is_err());
        }
    }

    #[test]
    fn model_group_controls_native_work_without_changing_prompt_bytes() {
        let a = planner(16, None).prepare(boolean()).unwrap();
        let b = planner(128, None).prepare(boolean()).unwrap();
        assert_eq!(a.execution_identity().prompt_digest, b.execution_identity().prompt_digest);
        assert_eq!(a.planned_work().projections.multiply_accumulates, b.planned_work().projections.multiply_accumulates);
        assert!(a.planned_work().projections.dot_products > b.planned_work().projections.dot_products);
        assert!(a.verify_identity(b.execution_identity()).is_err());
        let identity = a.execution_identity().clone(); let work = a.planned_work();
        let bound = a.task_plan().ir().budget().max_kv_bytes;
        let plan = a.into_extraction_plan();
        assert_eq!(plan.planned_work(), work); assert_eq!(plan.group_width(), 16);
        assert_eq!(plan.max_kv_bytes(), bound); plan.verify_identity(&identity).unwrap();
    }

    #[test]
    fn item_overrides_do_not_mutate_default_schema_or_identity() {
        let planner = planner(127, Some(args(r#"{"type":"boolean"}"#, ExtractionBatchGrounding::Structural)));
        let mut input = boolean(); input.task_args = None;
        let first = planner.prepare(input).unwrap();
        let changed = planner.prepare(document(r#"{"type":"null"}"#, "PRIVATE_SOURCE <think>", ExtractionBatchGrounding::Structural)).unwrap();
        assert!(first.verify_identity(changed.execution_identity()).is_err());
        let mut input = boolean(); input.task_args = None;
        first.verify_identity(planner.prepare(input).unwrap().execution_identity()).unwrap();
    }

    struct Stop { calls: usize, at: usize, cause: DecodeCancellationKind }
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            let call = self.calls; self.calls += 1; (call == self.at).then_some(self.cause)
        }
    }
    #[test]
    fn preparation_uses_existing_cancellation_before_and_after_compiler() {
        let planner = planner(127, None);
        for at in [0, 1] {
            let mut control = Stop { calls: 0, at, cause: DecodeCancellationKind::CostBudget };
            let error = match planner.prepare_with_control(boolean(), &mut control) {
                Err(error) => error, Ok(_) => panic!("cancelled preparation accepted"),
            };
            assert!(error.stop); assert_eq!(control.calls, at + 1);
            assert_eq!(error.fault.cancellation, Some(DecodeCancellationKind::CostBudget));
        }
    }

    #[test]
    fn public_diagnostics_do_not_echo_schema_or_document_content() {
        let planner = planner(127, None);
        let error = match planner.prepare(document("PRIVATE_SCHEMA_INVALID", "PRIVATE_DOCUMENT", ExtractionBatchGrounding::Structural)) {
            Err(error) => error, Ok(_) => panic!("invalid schema accepted"),
        };
        let text = format!("{error:?}");
        assert!(!text.contains("PRIVATE_SCHEMA_INVALID")); assert!(!text.contains("PRIVATE_DOCUMENT"));
    }
}
