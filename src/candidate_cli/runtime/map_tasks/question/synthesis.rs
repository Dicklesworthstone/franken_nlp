//! Explicit document QA synthesis, one runtime/model, one completed response.
use super::*;
use crate::tasks::source_planning::quantized::long::question::synthesis::{
    SourceQuestionSynthesis, Int8QuestionSynthesisPreflight,
    hierarchy::Int8QuestionHierarchyPreflight};

enum Expected { Single(Int8QuestionSynthesisPreflight), Hierarchical(Int8QuestionHierarchyPreflight) }
#[allow(clippy::too_many_arguments)]
fn prepare_mode(source: &str, planner: &SourceTaskPlanner, task: &SourceQuestionSynthesis,
    budget: TaskBudget, context: &PlanContext<'_>, command: &MapCommand, mapping: Int8SourceMapLimits,
    control: &mut impl DecodeStepControl) -> Result<Expected, CandidateError> {
    let planning = command.question.synthesis.planning(command.host.planning());
    if let Some(hierarchy) = command.question.synthesis.hierarchy.limits(true)? {
        let expected = planner.preflight_int8_question_hierarchy_with_control(source, task, hierarchy,
            budget, context, planning, mapping, control).map_err(question_error)?;
        command.admit_work(expected.reserved_model_work(), expected.reserved_mask_visits())?;
        Ok(Expected::Hierarchical(expected))
    } else {
        let expected = planner.preflight_int8_question_synthesis_with_control(source, task,
            budget, context, planning, mapping, control).map_err(question_error)?;
        command.admit_work(expected.reserved_model_work().map_err(question_error)?,
            expected.reserved_mask_visits().map_err(question_error)?)?;
        Ok(Expected::Single(expected))
    }
}

pub(super) fn execute(command: MapCommand, args: CandidateArgs, limits: Limits,
    input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
    let (session, prefill) = scheduled_session(&command, &args, limits)?;
    let text = session.read(&command.input, input, args.max_input_bytes)?;
    let question = session.read(command.question.path()?, input, QUESTION_BYTES)?;
    let option_text = command.options.as_ref()
        .map(|path| session.read(path, input, crate::candidate_cli::source::OPTIONS_BYTES)).transpose()?;
    let question = command.question.task(question, option_text.as_deref())?;
    let task = command.question.synthesis.request(question, &command.host)?;
    let mapping = command.question_mapping()?;
    let budget = command.host.task_budget(limits);
    let facts = session.facts(&args)?;
    let (planner, vocabulary) = planner()?;
    let planner = match command.encoding.limits(&command)? {
        Some(encoding) => planner.with_map_document_encoding(encoding.policy(), encoding.complete)
            .map_err(|_| CandidateError::Planning)?,
        None => planner,
    };
    let identity = source_identity(&facts, &planner, BuiltInTask::Answer)?;
    let context = PlanContext::new(&identity, budget).map_err(|_| CandidateError::Planning)?;
    // Reserve every allowed synthesis pass before weights. Discovery uses the
    // remainder, without refunds or renewed budgets between hierarchy levels.
    let expected = prepare_mode(&text, &planner, &task, budget, &context, &command, mapping, &mut session.control())?;
    session.remaining()?;
    let cancellation = CancellationToken::default();
    let model = session.load(&args, limits, &facts, cancellation.clone())?;
    let config = SourceMapConfig { identity, task, budget,
        planning: command.question.synthesis.planning(command.host.planning()), mapping,
        native: session.native(&args)?, preparation_reserve_bytes: limits.preparation_bytes,
        reduction_reserve_bytes: command.reduction_reserve_mib.checked_mul(MIB).ok_or(CandidateError::Arguments)? };
    let planner = Arc::new(planner);
    let vocabulary = Arc::new(vocabulary);
    let width = limits.compute.threads;
    let kernel = crate::native_engine::simd::config::process_kernel();
    match expected {
        Expected::Single(expected) => {
            let result = if width == 1 {
                compute::Completion::Serial(match prefill {
                    None => session.engine.synthesize_int8_document_answer(&model, text,
                        Arc::clone(&planner), vocabulary, config, cancellation),
                    Some(prefill) => session.engine.synthesize_int8_document_answer_layer_major(&model, text,
                        Arc::clone(&planner), vocabulary, config, prefill, cancellation),
                }.map_err(host_error)?)
            } else {
                compute::Completion::Team(match prefill {
                    None => session.engine.synthesize_int8_document_answer_with_threads(&model, text,
                        Arc::clone(&planner), vocabulary, config, kernel, width, cancellation),
                    Some(prefill) => session.engine.synthesize_int8_document_answer_layer_major_with_threads(&model, text,
                        Arc::clone(&planner), vocabulary, config, kernel, width, prefill, cancellation),
                }.map_err(host_error)?)
            };
            session.remaining()?;
            expected.verify_completed_with_control(result.result(), &mut session.control())
                .map_err(|error| error.cancellation().map(cancelled).unwrap_or(CandidateError::Execution))?;
            deliver_scheduled(&session, &facts, result.result(), encoding::report(&planner)?.as_ref(),
                result.compute(), command.max_map_result_bytes + 4096, output)
        }
        Expected::Hierarchical(expected) => {
            let result = if width == 1 {
                compute::Completion::Serial(match prefill {
                    None => session.engine.synthesize_int8_document_answer_hierarchical(&model, text,
                        Arc::clone(&planner), vocabulary, config, expected.limits(), cancellation),
                    Some(prefill) => session.engine.synthesize_int8_document_answer_hierarchical_layer_major(&model, text,
                        Arc::clone(&planner), vocabulary, config, expected.limits(), prefill, cancellation),
                }.map_err(host_error)?)
            } else {
                compute::Completion::Team(match prefill {
                    None => session.engine.synthesize_int8_document_answer_hierarchical_with_threads(&model, text,
                        Arc::clone(&planner), vocabulary, config, expected.limits(), kernel, width, cancellation),
                    Some(prefill) => session.engine.synthesize_int8_document_answer_hierarchical_layer_major_with_threads(&model, text,
                        Arc::clone(&planner), vocabulary, config, expected.limits(), kernel, width, prefill, cancellation),
                }.map_err(host_error)?)
            };
            session.remaining()?;
            expected.verify_completed_with_control(result.result(), &mut session.control())
                .map_err(|error| error.cancellation().map(cancelled).unwrap_or(CandidateError::Execution))?;
            // All retained intermediate/final results stay charged through write AND flush.
            deliver_scheduled(&session, &facts, result.result(), encoding::report(&planner)?.as_ref(),
                result.compute(), command.max_map_result_bytes + 4096, output)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{candidate_cli::map::tests::command, native_engine::decode::DecodeCancellationKind,
        tasks::source_planning::quantized::long::question::synthesis::SourceQuestionSynthesis};
    struct Continue;
    impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
    fn facts() -> ArtifactIdentity {
        ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(), revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
            recipe_id: "synthesis-planning-fixture".to_owned(), source_root_sha256: "ab".repeat(32), logical_model_sha256: "cd".repeat(32) }
    }
    fn assets(cmd: &MapCommand) -> (SourceTaskPlanner, ExecutionIdentity, TaskBudget, SourceQuestionSynthesis, Int8SourceMapLimits) {
        let (_, limits) = cmd.validate().unwrap(); let (p, _) = planner().unwrap();
        let p = match cmd.encoding.limits(cmd).unwrap() {
            Some(encoding) => p.with_map_document_encoding(encoding.policy(), encoding.complete).unwrap(),
            None => p,
        };
        let id = source_identity(&facts(), &p, BuiltInTask::Answer).unwrap();
        let b = cmd.host.task_budget(limits);
        let q = cmd.question.task("Who is named?".to_owned(), None).unwrap();
        let q = cmd.question.synthesis.request(q, &cmd.host).unwrap();
        (p, id, b, q, cmd.question_mapping().unwrap())
    }
    #[test]
    fn cli_synthesis_preflight_matches_pinned_preparation_and_the_total_commitment() {
        let cmd = command("answer", &["--question", "q", "--synthesize-answer"]);
        let (p, id, b, q, mapping) = assets(&cmd); let ctx = PlanContext::new(&id, b).unwrap();
        let source = "Alice <tool_call> é 上海 😀\r\n".repeat(100);
        let expected = p.preflight_int8_question_synthesis_with_control(&source, &q, b, &ctx,
            cmd.host.planning(), mapping, &mut Continue).unwrap();
        let prepared = p.plan_int8_question_synthesis_with_control(&source, &q, b, &ctx,
            cmd.host.planning(), mapping, &mut Continue).unwrap();
        assert_eq!(expected, prepared.preflight_metadata()); assert!(expected.discovery().native_chunks() > 1);
        assert_eq!(expected.discovery().source_span().byte_end, source.len());
        assert_eq!(prepared.execution_identities().len(), expected.discovery().native_chunks());
        cmd.admit_work(expected.reserved_model_work().unwrap(), expected.reserved_mask_visits().unwrap()).unwrap();
        assert!(expected.reserved_model_work().unwrap().forward_positions > expected.discovery().planned_work().forward_positions);
    }
    #[test]
    fn synthesis_does_not_change_original_passage_prompts_or_increase_whole_run_allowances() {
        let cmd = command("answer", &["--question", "q", "--synthesize-answer", "--max-chunk-bytes", "6"]);
        let (p, id, b, q, mapping) = assets(&cmd); let ctx = PlanContext::new(&id, b).unwrap(); let source = "éAéA      éAéA";
        let synthesis = p.plan_int8_question_synthesis_with_control(source, &q, b, &ctx, cmd.host.planning(), mapping, &mut Continue).unwrap();
        let plain = p.plan_int8_question_with_control(source, &q.question, b, &ctx, cmd.host.planning(), mapping, &mut Continue).unwrap();
        let discovery = synthesis.preflight_metadata().discovery(); let plain_expected = plain.preflight_metadata();
        assert_eq!(discovery.source_span(), plain_expected.source_span());
        assert_eq!(discovery.chunk_count(), plain_expected.chunk_count());
        assert_eq!(discovery.native_chunks(), plain_expected.native_chunks());
        assert_eq!(discovery.planned_work(), plain_expected.planned_work());
        assert_eq!(discovery.reserved_mask_visits(), plain_expected.reserved_mask_visits());
        // Synthesis subtracts its reservation from the original discovery
        // allowance, so exact private request authority must differ.
        assert_ne!(discovery, plain_expected);
        for (a, b) in synthesis.execution_identities().zip(plain.execution_identities()) { assert_eq!(a, b); }
        assert_eq!(synthesis.preflight_metadata().discovery().whitespace_chunks(), 1);
    }
    #[test]
    fn all_six_exact_combined_allowances_fail_closed_one_unit_below_the_required_reservation() {
        let cmd = command("answer", &["--question", "q", "--synthesize-answer"]);
        let (p, id, b, q, mapping) = assets(&cmd); let ctx = PlanContext::new(&id, b).unwrap();
        let expected = p.preflight_int8_question_synthesis_with_control("Alice", &q, b, &ctx,
            cmd.host.planning(), mapping, &mut Continue).unwrap();
        let mut exact = mapping; exact.max_model_work = expected.reserved_model_work().unwrap();
        exact.max_mask_visits = expected.reserved_mask_visits().unwrap();
        p.preflight_int8_question_synthesis_with_control("Alice", &q, b, &ctx, cmd.host.planning(), exact, &mut Continue).unwrap();
        for axis in 0..6 {
            let mut bad = exact;
            match axis { 0 => bad.max_model_work.forward_positions -= 1, 1 => bad.max_model_work.projected_logits -= 1,
                2 => bad.max_model_work.attention_pairs -= 1, 3 => bad.max_model_work.projections.dot_products -= 1,
                4 => bad.max_model_work.projections.multiply_accumulates -= 1, _ => bad.max_mask_visits -= 1 }
            assert!(p.preflight_int8_question_synthesis_with_control("Alice", &q, b, &ctx,
                cmd.host.planning(), bad, &mut Continue).is_err(), "{axis}");
        }
    }
    #[test]
    fn cancellation_wrong_task_and_empty_source_fail_before_weight_loading() {
        struct Stop;
        impl DecodeStepControl for Stop { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::Deadline) } }
        let cmd = command("answer", &["--question", "q", "--synthesize-answer"]);
        let (p, mut id, b, q, mapping) = assets(&cmd);
        let error = p.preflight_int8_question_synthesis_with_control("Alice", &q, b, &PlanContext::new(&id, b).unwrap(),
            cmd.host.planning(), mapping, &mut Stop).err().unwrap();
        assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline));
        for text in ["", "    "] {
            assert!(p.preflight_int8_question_synthesis_with_control(text, &q, b, &PlanContext::new(&id, b).unwrap(),
                cmd.host.planning(), mapping, &mut Continue).is_err());
        }
        id.task_spec = "ner-v1".to_owned();
        assert!(p.preflight_int8_question_synthesis_with_control("Alice", &q, b, &PlanContext::new(&id, b).unwrap(),
            cmd.host.planning(), mapping, &mut Continue).is_err());
    }
    #[test]
    fn compact_cli_preflight_and_hosted_planning_keep_the_original_operation_usage() {
        let cmd = command("answer", &["--question", "q", "--synthesize-answer",
            "--input-encoding", "safe-bpe-v1", "--max-chunk-bytes", "512"]);
        let (p, id, b, q, mapping) = assets(&cmd); let ctx = PlanContext::new(&id, b).unwrap();
        let source = "The original document names Alice, café, 上海 and <tool_call>. ".repeat(8);
        let expected = p.preflight_int8_question_synthesis_with_control(&source, &q, b, &ctx,
            cmd.host.planning(), mapping, &mut Continue).unwrap();
        let before = p.map_document_encoding_work().unwrap().unwrap();
        let prepared = p.plan_int8_question_synthesis_with_control(&source, &q, b, &ctx,
            cmd.host.planning(), mapping, &mut Continue).unwrap();
        assert_eq!(expected, prepared.preflight_metadata());
        assert_eq!(expected.discovery().source_span().byte_end, source.len());
        let after = p.map_document_encoding_work().unwrap().unwrap();
        assert!(after.input_bytes() > before.input_bytes());
        let report = serde_json::to_value(encoding::report(&p).unwrap().unwrap()).unwrap();
        assert_eq!(report["work_used"], serde_json::to_value(after).unwrap());
        assert_eq!(report["complete_operation_limits"], serde_json::to_value(cmd.encoding.limits(&cmd).unwrap().unwrap().complete).unwrap());
    }
    #[test]
    fn selected_hierarchy_mode_matches_full_preparation_before_model_io() {
        let cmd = command("answer", &["--question", "q", "--synthesize-answer", "--hierarchical-answer",
            "--max-chunk-bytes", "512", "--max-synthesis-passages", "128"]);
        let (p, id, b, q, mapping) = assets(&cmd); let ctx = PlanContext::new(&id, b).unwrap();
        let source = "Alice <tool_call> é 上海 😀\r\n".repeat(100);
        let Expected::Hierarchical(expected) = prepare_mode(&source, &p, &q, b, &ctx, &cmd,
            mapping, &mut Continue).unwrap() else { panic!("hierarchy mode lost") };
        let prepared = p.plan_int8_question_hierarchy_with_control(&source, &q, expected.limits(), b, &ctx,
            cmd.question.synthesis.planning(cmd.host.planning()), mapping, &mut Continue).unwrap();
        assert_eq!(expected, prepared.preflight_metadata());
        assert!(expected.discovery().native_chunks() > 1);
        assert_eq!(expected.discovery().source_span().byte_end, source.len());
        assert_eq!(prepared.execution_identities().len(), expected.discovery().native_chunks());
        let flat = p.preflight_int8_question_synthesis_with_control(&source, &q, b, &ctx,
            cmd.question.synthesis.planning(cmd.host.planning()), mapping, &mut Continue).unwrap();
        assert!(expected.reserved_model_work().forward_positions > flat.reserved_model_work().unwrap().forward_positions);
        assert!(expected.reserved_mask_visits() > flat.reserved_mask_visits().unwrap());
    }
    #[test]
    fn hierarchy_cli_admits_the_entire_reservation_on_all_six_axes() {
        let mut cmd = command("answer", &["--question", "q", "--synthesize-answer", "--hierarchical-answer"]);
        let (p, id, b, q, mapping) = assets(&cmd); let ctx = PlanContext::new(&id, b).unwrap();
        let Expected::Hierarchical(expected) = prepare_mode("Alice", &p, &q, b, &ctx, &cmd,
            mapping, &mut Continue).unwrap() else { panic!("hierarchy mode lost") };
        let work = expected.reserved_model_work(); let masks = expected.reserved_mask_visits();
        cmd.max_forward_positions = work.forward_positions;
        cmd.max_projected_logits = work.projected_logits;
        cmd.max_attention_pairs = work.attention_pairs;
        cmd.max_dot_products = work.projections.dot_products;
        cmd.max_multiply_accumulates = work.projections.multiply_accumulates;
        cmd.max_total_mask_node_visits = masks;
        let exact = cmd.question_mapping().unwrap();
        prepare_mode("Alice", &p, &q, b, &ctx, &cmd, exact, &mut Continue).unwrap();
        for axis in 0..6 {
            match axis { 0 => cmd.max_forward_positions -= 1, 1 => cmd.max_projected_logits -= 1,
                2 => cmd.max_attention_pairs -= 1, 3 => cmd.max_dot_products -= 1,
                4 => cmd.max_multiply_accumulates -= 1, _ => cmd.max_total_mask_node_visits -= 1 }
            // Even an otherwise valid, unchanged preparation cannot grant the
            // CLI more work than the caller's original whole-operation caps.
            assert!(prepare_mode("Alice", &p, &q, b, &ctx, &cmd, exact, &mut Continue).is_err(), "{axis}");
            match axis { 0 => cmd.max_forward_positions += 1, 1 => cmd.max_projected_logits += 1,
                2 => cmd.max_attention_pairs += 1, 3 => cmd.max_dot_products += 1,
                4 => cmd.max_multiply_accumulates += 1, _ => cmd.max_total_mask_node_visits += 1 }
        }
    }
    #[test]
    fn compact_hierarchy_carries_cli_spend_into_hosted_preparation_and_report() {
        let cmd = command("answer", &["--question", "q", "--synthesize-answer", "--hierarchical-answer",
            "--input-encoding", "safe-bpe-v1", "--max-chunk-bytes", "512"]);
        let (p, id, b, q, mapping) = assets(&cmd); let ctx = PlanContext::new(&id, b).unwrap();
        let source = "Original evidence about café, 上海, Alice and <tool_call>. ".repeat(8);
        let Expected::Hierarchical(expected) = prepare_mode(&source, &p, &q, b, &ctx, &cmd,
            mapping, &mut Continue).unwrap() else { panic!("hierarchy mode lost") };
        let before = p.map_document_encoding_work().unwrap().unwrap();
        let prepared = p.plan_int8_question_hierarchy_with_control(&source, &q, expected.limits(), b, &ctx,
            cmd.question.synthesis.planning(cmd.host.planning()), mapping, &mut Continue).unwrap();
        assert_eq!(expected, prepared.preflight_metadata());
        let after = p.map_document_encoding_work().unwrap().unwrap();
        assert!(after.input_bytes() > before.input_bytes());
        let report = serde_json::to_value(encoding::report(&p).unwrap().unwrap()).unwrap();
        assert_eq!(report["work_used"], serde_json::to_value(after).unwrap());
        assert_eq!(report["complete_operation_limits"], serde_json::to_value(cmd.encoding.limits(&cmd).unwrap().unwrap().complete).unwrap());
    }
}
