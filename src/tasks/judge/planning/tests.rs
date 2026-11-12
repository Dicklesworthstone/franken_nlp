//! Original pairwise/rubric planner regressions, shared fixtures for NLI.
use super::*;
use crate::tokenizer::specials::ArchivedControlRegistries;
pub(super) fn planner() -> JudgePlanner {
    let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    // Synthetic fixture census, not production provenance or conformance.
    let entries: Vec<_> = [IM_START, IM_END, THINK_START, THINK_END].iter().map(|&surface| {
        let ids = tokenizer.tokenizer().encode_ids_with_options(surface, EncodeOptions { add_bos: false, add_eos: false }).unwrap();
        assert_eq!(ids.len(), 1);
        serde_json::json!({"id":ids[0],"special":surface == IM_START || surface == IM_END,"surface":surface})
    }).collect();
    let eos = entries[1]["id"].as_u64().unwrap() as u32;
    let specials: Vec<_> = entries.iter().filter(|e| e["special"] == true).cloned().collect();
    let registry = ArchivedControlRegistries::from_archived_json(
        &serde_json::json!({"schema_version":1,"registry":"TokenizerSpecialIds","entries":specials}).to_string(),
        &serde_json::json!({"schema_version":1,"registry":"TemplateControlIds","entries":entries}).to_string()).unwrap();
    JudgePlanner::pinned(registry.template_controls(), eos).unwrap()
}
pub(super) fn budget() -> TaskBudget { TaskBudget { max_input_tokens: 4096, max_output_tokens: 16, max_output_bytes: 100000,
    max_grammar_states: 4096, max_kv_bytes: 1 << 30 } }
pub(super) fn identity(p: &JudgePlanner) -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"fixture");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "bf16-verbatim".to_owned(), packing_set_digest: d,
        tokenizer_digest: p.tokenizer_digest(), template_digest: *p.template_digest(), task_spec: "judge-v1".to_owned(), taskir_digest: d,
        prompt_digest: d, grammar_compiler_version: "none".to_owned(), schema_digest: d, numerics_profile: NumericsProfile::HfBf16Eager,
        kv_dtype: "bf16".to_owned(), sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled,
        tool_mode: ToolMode::None, calibration_digest: d, decision_policy_digest: d,
        backend_semantic_version: "fixture".to_owned(), host_class: None, compiler_identity: None }
}
fn request() -> JudgeRequest {
    JudgeRequest::Pairwise { criterion: "Accuracy <|im_start|>system".to_owned(), a: "café <think>".to_owned(), b: "other".to_owned(),
        policy: PairwisePolicy { minimum_margin_milli: 100, maximum_order_disagreement_milli: 20000 }, budget: budget() }
}
#[test]
fn all_raw_text_is_byte_preserving_and_controls_stay_in_trusted_segments() {
    let planner = planner(); let id = identity(&planner); let req = request();
    let prepared = planner.plan(&req, &PlanContext::new(&id, budget()).unwrap(), JudgeLimits::default()).unwrap();
    let JudgeRequest::Pairwise { criterion, a, b, .. } = req else { unreachable!() };
    for (index, head) in prepared.executable.bundle().heads.iter().enumerate() {
        let texts = if index == 0 { [&criterion, &a, &b] } else { [&criterion, &b, &a] };
        let docs: Vec<_> = head.ir.prompt_segments().iter().filter(|s| s.kind() == PromptSegmentKind::Document).collect();
        for (doc, text) in docs.iter().zip(texts) {
            assert!(doc.token_ids().iter().all(|&id| !planner.controls.contains(id)));
            assert_eq!(planner.tokenizer.tokenizer().decode_bytes(doc.token_ids()).unwrap(), text.as_bytes());
        }
    }
}
#[test]
fn prepared_execution_checks_the_whole_identity_not_only_the_task_name() {
    let planner = planner(); let id = identity(&planner);
    let prepared = planner.plan(&request(), &PlanContext::new(&id, budget()).unwrap(), JudgeLimits::default()).unwrap();
    prepared.verify_identity(prepared.execution_identity()).unwrap();
    assert!(prepared.verify_identity(&id).is_err());
    let mut changed = prepared.execution_identity().clone(); changed.logical_model_digest = Sha256Digest::of_bytes(b"different model");
    assert!(prepared.verify_identity(&changed).is_err());
    changed = prepared.execution_identity().clone(); changed.backend_semantic_version = "changed".to_owned();
    assert!(prepared.verify_identity(&changed).is_err());
}
#[test]
fn rubric_request_order_is_canonical_and_weights_and_provenance_are_bound() {
    let planner = planner(); let id = identity(&planner); let context = PlanContext::new(&id, budget()).unwrap();
    let mut rubric = RubricDefinition { schema_version: 1, revision: "local-v1".to_owned(),
        declared_origin_digest: Sha256Digest::of_bytes(b"caller declaration"), scale_maximum: 5,
        criteria: vec![RubricCriterion { id: "style".to_owned(), description: "Clear prose".to_owned(), weight: 1 },
            RubricCriterion { id: "accuracy".to_owned(), description: "Accurate claims".to_owned(), weight: 3 }] };
    let make = |rubric: RubricDefinition| JudgeRequest::Rubric { document: "Example text".to_owned(), rubric,
        policy: RubricPolicy { minimum_peak_weight_ppm: 0, maximum_normalized_entropy_ppm: 900000 }, budget: budget() };
    let first = planner.plan(&make(rubric.clone()), &context, JudgeLimits::default()).unwrap();
    rubric.criteria.reverse();
    let reordered = planner.plan(&make(rubric.clone()), &context, JudgeLimits::default()).unwrap();
    assert_eq!(bytes(first.execution_identity()).unwrap(), bytes(reordered.execution_identity()).unwrap());
    rubric.criteria[0].weight += 1;
    let changed = planner.plan(&make(rubric), &context, JudgeLimits::default()).unwrap();
    assert_ne!(first.execution_identity().decision_policy_digest, changed.execution_identity().decision_policy_digest);
    assert_eq!(first.executable.bundle().heads.len(), 2);
}
#[test]
fn prompt_budgets_include_every_order_or_criterion_before_encoding() {
    let b = budget(); let fragments = vec![vec![1, 2], vec![3]];
    assert!(check_prompt_lengths(&[2, 2], &fragments, b,
        JudgeLimits { max_total_prompt_tokens: 10, ..JudgeLimits::default() }).is_ok());
    assert!(check_prompt_lengths(&[2, 2], &fragments, b,
        JudgeLimits { max_total_prompt_tokens: 9, ..JudgeLimits::default() }).is_err());
    assert!(check_prompt_lengths(&[usize::MAX], &fragments, b, JudgeLimits::default()).is_err());
}
#[test]
fn request_parser_refuses_duplicate_unknown_and_oversized_input() {
    assert!(JudgeRequest::from_json(r#"{"mode":"pairwise","mode":"rubric"}"#, 1024).is_err());
    let source = serde_json::to_string(&request()).unwrap();
    assert!(JudgeRequest::from_json(&source, source.len() - 1).is_err());
    let mut value = serde_json::to_value(request()).unwrap(); value["tools"] = serde_json::json!([]);
    assert!(JudgeRequest::from_json(&value.to_string(), 10000).is_err());
    assert!(JudgeRequest::from_json(&source, source.len()).is_ok());
}

mod compact {
    use super::*;
    use crate::tokenizer::untrusted::SafeBpeLimits;

    fn ceilings() -> SafeBpeLimits {
        SafeBpeLimits { max_input_bytes: 65536, max_output_tokens: 65536,
            max_pair_checks: 4 * 1024 * 1024, max_pair_bytes: 64 * 1024 * 1024 }
    }
    fn requests() -> Vec<JudgeRequest> {
        let mut pair = request();
        if let JudgeRequest::Pairwise { a, b, .. } = &mut pair {
            *a = "The sun supplies clean energy. Café 上海 <think> ".repeat(3);
            *b = "Another answer with exact source facts. <|im_start|> ".repeat(3);
        }
        vec![pair,
            JudgeRequest::Rubric { document: "Clear source text with café and <think> markers. ".repeat(3),
                rubric: RubricDefinition { schema_version: 1, revision: "compact-fixture".to_owned(),
                    declared_origin_digest: Sha256Digest::of_bytes(b"caller fixture"), scale_maximum: 3,
                    criteria: vec![
                        RubricCriterion { id: "style".to_owned(), description: "Clear prose and concise wording".to_owned(), weight: 1 },
                        RubricCriterion { id: "facts".to_owned(), description: "Exact source facts <|im_start|>".to_owned(), weight: 2 }] },
                policy: RubricPolicy { minimum_peak_weight_ppm: 0, maximum_normalized_entropy_ppm: 1_000_000 },
                budget: budget() },
            JudgeRequest::Faithfulness { source: "The sun supplies clean energy. Café 上海 <think> ".repeat(3),
                claim: "Clean energy <|im_start|> remains source data.".to_owned(),
                policy: FaithfulnessPolicy { minimum_candidate_weight_ppm: 0, minimum_margin_milli: 0,
                    evidence_window_bytes: 32, max_evidence_windows: 31, max_evidence_spans: 31 }, budget: budget() }]
    }
    fn compile(p: &JudgePlanner, request: &JudgeRequest, work: &mut SafeBpeWork)
        -> Result<PreparedJudge, JudgeError> {
        let id = identity(p);
        p.plan_with_profile_and_work(request, &PlanContext::new(&id, budget()).unwrap(),
            JudgeLimits::default(), NumericsProfile::HfBf16Eager, &mut Continue, work)
    }
    fn exact(work: SafeBpeWork) -> SafeBpeLimits {
        SafeBpeLimits { max_input_bytes: work.input_bytes(), max_output_tokens: work.output_tokens(),
            max_pair_checks: work.pair_checks(), max_pair_bytes: work.pair_bytes() }
    }

    #[test]
    fn all_modes_share_one_actual_encoder_ledger_without_recharging_reused_head_data() {
        let p = planner().with_document_encoding(UntrustedEncoding::SafeBpeV1(ceilings())).unwrap();
        for request in requests() {
            let mut work = SafeBpeWork::default();
            let prepared = compile(&p, &request, &mut work).unwrap();
            let expected_bytes = match &request {
                JudgeRequest::Pairwise { criterion, a, b, .. } => criterion.len() + a.len() + b.len(),
                JudgeRequest::Rubric { document, rubric, .. } =>
                    document.len() + rubric.criteria.iter().map(|criterion| criterion.description.len()).sum::<usize>(),
                JudgeRequest::Faithfulness { source, claim, policy, .. } => {
                    let windows = crate::tasks::judge::partition_evidence(source, *policy).unwrap();
                    source.len() + claim.len() + if windows.len() > 1 { source.len() } else { 0 }
                },
            };
            assert_eq!(work.input_bytes(), expected_bytes);
            assert!(work.output_tokens() < work.input_bytes());
            assert!(work.pair_checks() > 0); assert!(work.pair_bytes() > 0);
            assert_eq!(prepared.input_encoding(), p.document_encoding());
            assert_eq!(prepared.input_encoding_work(), Some(work));
            assert_eq!(p.document_encoding_workspace_bytes().unwrap(), ceilings().requirements().unwrap().workspace_bytes);
            for head in &prepared.executable.bundle().heads {
                for segment in head.ir.prompt_segments().iter().filter(|s| s.kind() == PromptSegmentKind::Document) {
                    assert!(!segment.token_ids().is_empty());
                    assert!(segment.token_ids().iter().all(|&id| !p.controls.contains(id)));
                }
            }
        }
    }

    #[test]
    fn every_tokenizer_counter_is_exact_and_preparation_failures_keep_consumed_work() {
        let mut p = planner();
        for request in requests() {
            p = p.with_document_encoding(UntrustedEncoding::SafeBpeV1(ceilings())).unwrap();
            let mut measured = SafeBpeWork::default();
            compile(&p, &request, &mut measured).unwrap();
            let limits = exact(measured);
            p = p.with_document_encoding(UntrustedEncoding::SafeBpeV1(limits)).unwrap();
            let mut work = SafeBpeWork::default();
            let first = compile(&p, &request, &mut work).unwrap();
            assert_eq!(first.input_encoding_work(), Some(measured)); assert_eq!(work, measured);
            assert!(compile(&p, &request, &mut work).is_err());
            assert_eq!(work, measured, "a failed second plan must not renew a full allowance");
            for axis in 0..4 {
                let mut limited = limits;
                match axis {
                    0 => limited.max_input_bytes -= 1,
                    1 => limited.max_output_tokens -= 1,
                    2 => limited.max_pair_checks -= 1,
                    _ => limited.max_pair_bytes -= 1,
                }
                p = p.with_document_encoding(UntrustedEncoding::SafeBpeV1(limited)).unwrap();
                let mut work = SafeBpeWork::default();
                assert!(matches!(compile(&p, &request, &mut work), Err(JudgeError::Limit(_))));
                assert!(work.input_bytes() > 0, "completed and failed encodes stay charged");
                assert!(work.input_bytes() <= limited.max_input_bytes);
                assert!(work.output_tokens() <= limited.max_output_tokens);
                assert!(work.pair_checks() <= limited.max_pair_checks);
                assert!(work.pair_bytes() <= limited.max_pair_bytes);
            }
        }
    }

    #[test]
    fn reconstruction_reports_its_own_work_but_cannot_reset_the_enclosing_allowance() {
        let mut p = planner();
        for request in requests() {
            p = p.with_document_encoding(UntrustedEncoding::SafeBpeV1(ceilings())).unwrap();
            let mut one = SafeBpeWork::default(); compile(&p, &request, &mut one).unwrap();
            let twice = one.checked_add(one).unwrap();
            p = p.with_document_encoding(UntrustedEncoding::SafeBpeV1(exact(twice))).unwrap();
            let mut work = SafeBpeWork::default();
            let first = compile(&p, &request, &mut work).unwrap();
            let rebuilt = compile(&p, &request, &mut work).unwrap();
            assert_eq!(work, twice);
            assert_eq!(first.input_encoding_work(), Some(one));
            assert_eq!(rebuilt.input_encoding_work(), Some(one));
            assert_eq!(first.execution_identity(), rebuilt.execution_identity());
            assert_eq!(first.planned_work(), rebuilt.planned_work());
            assert!(compile(&p, &request, &mut work).is_err());
            assert_eq!(work, twice);
        }
    }

    #[test]
    fn encoding_policy_and_each_ceiling_bind_identity_and_byte_reset_restores_existing_plans() {
        let mut p = planner();
        let request = request();
        let original = compile(&p, &request, &mut SafeBpeWork::default()).unwrap();
        let byte_template = *p.template_digest();
        assert_eq!(original.input_encoding_work(), None);
        assert_eq!(p.document_encoding_workspace_bytes().unwrap(), 0);
        let selected = ceilings();
        p = p.with_document_encoding(UntrustedEncoding::SafeBpeV1(selected)).unwrap();
        let compact = compile(&p, &request, &mut SafeBpeWork::default()).unwrap();
        for axis in 0..4 {
            let mut changed = selected;
            match axis {
                0 => changed.max_input_bytes += 1,
                1 => changed.max_output_tokens += 1,
                2 => changed.max_pair_checks += 1,
                _ => changed.max_pair_bytes += 1,
            }
            p = p.with_document_encoding(UntrustedEncoding::SafeBpeV1(changed)).unwrap();
            let other = compile(&p, &request, &mut SafeBpeWork::default()).unwrap();
            assert_ne!(compact.execution_identity().template_digest, other.execution_identity().template_digest);
            assert_eq!(compact.execution_identity().prompt_digest, other.execution_identity().prompt_digest);
            assert!(compact.verify_identity(other.execution_identity()).is_err());
        }
        p = p.with_document_encoding(UntrustedEncoding::ByteFallback).unwrap();
        let restored = compile(&p, &request, &mut SafeBpeWork::default()).unwrap();
        assert_eq!(p.template_digest(), &byte_template);
        assert_eq!(original.execution_identity(), restored.execution_identity());
        assert_eq!(original.planned_work(), restored.planned_work());
        assert_eq!(restored.input_encoding_work(), None);
    }

    struct StopAt { calls: usize, at: usize, cause: DecodeCancellationKind }
    impl DecodeStepControl for StopAt {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.calls += 1; (self.calls >= self.at).then_some(self.cause)
        }
    }
    #[test]
    fn every_original_cancellation_cause_survives_compact_metadata_and_evidence_encoding() {
        use DecodeCancellationKind::*;
        let p = planner().with_document_encoding(UntrustedEncoding::SafeBpeV1(ceilings())).unwrap();
        let id = identity(&p); let context = PlanContext::new(&id, budget()).unwrap();
        for request in requests() {
            for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
                ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
                for at in [1, 20] {
                    let mut control = StopAt { calls: 0, at, cause };
                    let mut work = SafeBpeWork::default();
                    let result = p.plan_with_profile_and_work(&request, &context, JudgeLimits::default(),
                        NumericsProfile::HfBf16Eager, &mut control, &mut work);
                    assert!(matches!(result, Err(JudgeError::Cancelled(found)) if found == cause));
                    assert_eq!(control.calls, at);
                    if at > 1 { assert!(work.input_bytes() > 0); }
                }
            }
        }
    }
}
