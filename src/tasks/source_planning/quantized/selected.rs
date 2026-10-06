//! Explicit selected-row plans for the existing four source-backed tasks.
use super::*;
use crate::native_engine::constrained_int8::sparse::Int8JsonSparseLimits;

pub const INT8_SPARSE_SOURCE_EXECUTION: &str = "portable-int8-selected-row-source-portfolio-v1";

impl PreparedInt8SourceTask {
    /// Choose the complete legal-row head BEFORE resource admission. Source
    /// bytes, passage partitions, task options and semantic finalizers remain
    /// unchanged. The resulting policy identity cannot reuse a full-head key.
    pub fn with_selected_rows(mut self, limits: Int8JsonSparseLimits) -> Result<Self, Int8SourceError> {
        self.extraction = self.extraction.with_selected_rows(limits)?
            .with_finalizer_version(INT8_SPARSE_SOURCE_EXECUTION)?;
        Ok(self)
    }
    pub fn selected_rows(&self) -> Option<Int8JsonSparseLimits> { self.extraction.selected_rows() }
    pub fn execution_version(&self) -> &'static str {
        if self.selected_rows().is_some() { INT8_SPARSE_SOURCE_EXECUTION } else { INT8_SOURCE_EXECUTION }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{grammar::runtime::JsonProgram,
        native_engine::{constrained::JsonDecodeOutput, strict_int8::STRICT_INT8_EXECUTION},
        tasks::{extract::ExtractionGrounding, ir::ScoreSpace},
        tokenizer::specials::ArchivedControlRegistries};
    struct Continue;
    impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
    fn planner() -> SourceTaskPlanner {
        let t = EmbeddedTokenizer::pinned().unwrap();
        let entries: Vec<_> = [IM_START, IM_END, THINK_START, THINK_END].iter().map(|&surface| {
            let ids = t.tokenizer().encode_ids_with_options(surface, EncodeOptions { add_bos: false, add_eos: false }).unwrap();
            assert_eq!(ids.len(), 1);
            serde_json::json!({"id":ids[0],"special":surface == IM_START || surface == IM_END,"surface":surface})
        }).collect();
        let specials: Vec<_> = entries.iter().filter(|e| e["special"] == true).cloned().collect();
        let registry = ArchivedControlRegistries::from_archived_json(
            &serde_json::json!({"schema_version":1,"registry":"TokenizerSpecialIds","entries":specials}).to_string(),
            &serde_json::json!({"schema_version":1,"registry":"TemplateControlIds","entries":entries}).to_string()).unwrap();
        SourceTaskPlanner::pinned(registry.template_controls(), t.eos_token_id().unwrap()).unwrap()
    }
    fn budget() -> TaskBudget {
        TaskBudget { max_input_tokens: 8192, max_output_tokens: 512, max_output_bytes: 1 << 20,
            max_grammar_states: 4096, max_kv_bytes: 1 << 31 }
    }
    fn prepare(p: &SourceTaskPlanner, request: &SourceTaskRequest) -> PreparedInt8SourceTask {
        let d = Sha256Digest::of_bytes(b"selected-row-source-unit-fixture");
        let id = ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
            artifact_format: "fixture".to_owned(), quant_recipe: "int8-fixture".to_owned(), packing_set_digest: d,
            tokenizer_digest: p.tokenizer_digest(), template_digest: *p.template_digest(),
            task_spec: request.task().spec().identity(), taskir_digest: d, prompt_digest: d,
            grammar_compiler_version: "none".to_owned(), schema_digest: d,
            numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
            sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
            calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(),
            host_class: None, compiler_identity: None };
        p.plan_int8_with_control(request, &PlanContext::new(&id, budget()).unwrap(),
            SourcePlanningLimits::default(), &mut Continue).unwrap()
    }
    fn fixtures() -> Vec<(SourceTaskRequest, &'static str, &'static str)> {
        vec![
            (SourceTaskRequest::Ner { document: "é上海 and 上海".to_owned(), options: NerOptions::default(), budget: budget() },
                "é上海 and 上海", r#"[{"text":"上海","type":"location"}]"#),
            (SourceTaskRequest::Keyphrases { document: "Rust compiler Rust".to_owned(), options: KeyphraseOptions::default(), budget: budget() },
                "Rust compiler Rust", r#"["compiler","Rust","Rust"]"#),
            (SourceTaskRequest::Summarize { document: "Alice".to_owned(), options: SummaryOptions::default(), budget: budget() },
                "Alice", r#"[{"citations":["Alice"],"text":"An unverified assertion."}]"#),
            (SourceTaskRequest::Answer { question: "Where?".to_owned(), passages: vec![
                AnswerPassage { id: "one".to_owned(), text: "é上海".to_owned() },
                AnswerPassage { id: "two".to_owned(), text: "上海".to_owned() }], options: AnswerOptions::default(), budget: budget() },
                "é上海\n\n上海", r#"{"answer":"A location.","answerable":true,"citations":["上海"]}"#),
        ]
    }
    fn schema(r: &SourceTaskRequest) -> String {
        match r { SourceTaskRequest::Ner { options, .. } => options.schema_source().unwrap(),
            SourceTaskRequest::Keyphrases { options, .. } => options.schema_source().unwrap(),
            SourceTaskRequest::Summarize { options, .. } => options.schema_source().unwrap(),
            SourceTaskRequest::Answer { options, .. } => options.schema_source().unwrap() }
    }
    // Private semantic fixtures only. The public producer still requires the
    // real native decoder; these token IDs and work do not claim model parity.
    fn raw(p: &PreparedInt8SourceTask, r: &SourceTaskRequest, source: &str, json: &str) -> Int8ExtractRun {
        let program = JsonProgram::compile_with_source(&schema(r), source,
            CompileLimits::default(), SourceRuntimeLimits::default()).unwrap();
        let work = if p.selected_rows().is_some() { Int8Work::for_sequence(0, p.prompt_tokens() + 1, 3).unwrap() }
            else { constrained_int8::planned_work(p.prompt_tokens(), 2).unwrap() };
        Int8ExtractRun { schema_version: 1, execution: p.extraction.execution_version().to_owned(), model_work: work,
            result: ExtractResult { schema_version: 2, task_spec_version: r.task().spec().identity(),
                score_space: ScoreSpace::NotComputed, grounding: ExtractionGrounding::SourceMembership,
                source_fields: program.source_fields(json).unwrap(), output: JsonDecodeOutput {
                    schema_version: 1, numerics_profile: STRICT_INT8_PROFILE.to_owned(), token_ids: vec![1, 0],
                    json: json.to_owned(), forward_positions: work.forward_positions,
                    projected_logits: work.projected_logits, mask_node_visit_charge: 20,
                } } }
    }
    fn limits(rows: usize) -> Int8JsonSparseLimits { Int8JsonSparseLimits { max_rows_per_step: rows } }

    #[test]
    fn all_four_sealed_source_plans_keep_exact_semantic_inputs_and_reject_old_admissions() {
        let planner = planner();
        for (request, _, _) in fixtures() {
            let dense = prepare(&planner, &request); let id = dense.execution_identity().clone();
            let work = dense.planned_work();
            let sparse = dense.with_selected_rows(limits(7)).unwrap();
            sparse.verify_identity(sparse.execution_identity()).unwrap();
            assert!(sparse.verify_identity(&id).is_err());
            assert_eq!(sparse.planned_work().forward_positions, work.forward_positions);
            assert_eq!(sparse.planned_work().attention_pairs, work.attention_pairs);
            assert_eq!(sparse.planned_work().projected_logits, 512 * 7);
            let mut before = serde_json::to_value(id).unwrap();
            let mut after = serde_json::to_value(sparse.execution_identity()).unwrap();
            before.as_object_mut().unwrap().remove("decision_policy_digest");
            after.as_object_mut().unwrap().remove("decision_policy_digest");
            assert_eq!(before, after);
            assert_ne!(sparse.execution_identity().decision_policy_digest,
                prepare(&planner, &request).with_selected_rows(limits(8)).unwrap().execution_identity().decision_policy_digest);
            assert!(sparse.with_selected_rows(limits(7)).is_err());
        }
    }
    #[test]
    fn all_four_existing_source_finalizers_keep_identical_semantics_under_new_execution_labels() {
        let planner = planner();
        for (request, source, json) in fixtures() {
            let dense = prepare(&planner, &request);
            let sparse = prepare(&planner, &request).with_selected_rows(limits(7)).unwrap();
            let a = dense.finish(raw(&dense, &request, source, json)).unwrap();
            let b = sparse.finish(raw(&sparse, &request, source, json)).unwrap();
            assert_eq!(a.execution, INT8_SOURCE_EXECUTION); assert_eq!(b.execution, INT8_SPARSE_SOURCE_EXECUTION);
            assert_eq!(canonjson::canonical_bytes(&a.result).unwrap(), canonjson::canonical_bytes(&b.result).unwrap());
            assert!(b.model_work.projected_logits < a.model_work.projected_logits);
            assert!(dense.finish(raw(&sparse, &request, source, json)).is_err());
            assert!(sparse.finish(raw(&dense, &request, source, json)).is_err());
        }
    }
    #[test]
    fn semantic_refusals_and_passage_boundaries_are_not_weakened_by_sparse_selection() {
        let planner = planner();
        let request = SourceTaskRequest::Summarize { document: "Alice".to_owned(), options: SummaryOptions::default(), budget: budget() };
        let p = prepare(&planner, &request).with_selected_rows(limits(7)).unwrap();
        assert!(p.finish(raw(&p, &request, "Alice", r#"[{"citations":[],"text":"A claim."}]"#)).is_err());
        let request = SourceTaskRequest::Answer { question: "Who?".to_owned(), passages: vec![
            AnswerPassage { id: "a".to_owned(), text: "ab".to_owned() },
            AnswerPassage { id: "b".to_owned(), text: "cd".to_owned() }], options: AnswerOptions::default(), budget: budget() };
        let p = prepare(&planner, &request).with_selected_rows(limits(7)).unwrap();
        assert!(p.finish(raw(&p, &request, "ab\n\ncd",
            r#"{"answer":"A claim.","answerable":true,"citations":["b\n\nc"]}"#)).is_err());
    }
    #[test]
    fn bad_head_geometry_is_refused_and_default_source_execution_stays_full() {
        let planner = planner(); let request = fixtures().remove(0).0;
        for cap in [0, crate::native_engine::lmhead::NANBEIGE_VOCAB_SIZE + 1, usize::MAX] {
            assert!(prepare(&planner, &request).with_selected_rows(limits(cap)).is_err());
        }
        let p = prepare(&planner, &request);
        assert!(p.selected_rows().is_none()); assert_eq!(p.execution_version(), INT8_SOURCE_EXECUTION);
        assert_eq!(p.planned_work(), constrained_int8::planned_work(p.prompt_tokens(), 512).unwrap());
    }
}
