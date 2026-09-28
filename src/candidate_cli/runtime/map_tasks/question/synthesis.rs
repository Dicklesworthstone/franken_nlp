//! Explicit two-stage document QA, one runtime/model, one completed response.
use super::*;

pub(super) fn execute(command: MapCommand, args: CandidateArgs, limits: Limits,
    input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
    let session = Session::new(&args, limits)?;
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
    let identity = source_identity(&facts, &planner, BuiltInTask::Answer)?;
    let context = PlanContext::new(&identity, budget).map_err(|_| CandidateError::Planning)?;
    // Fund a maximum-context final pass BEFORE weights. All actual map prompts
    // still preflight against the remainder, without early-EOS refunds.
    let expected = planner.preflight_int8_question_synthesis_with_control(&text, &task, budget, &context,
        command.host.planning(), mapping, &mut session.control()).map_err(|_| CandidateError::Planning)?;
    command.admit_work(expected.reserved_model_work().map_err(|_| CandidateError::Planning)?,
        expected.reserved_mask_visits().map_err(|_| CandidateError::Planning)?)?;
    session.remaining()?;
    let cancellation = CancellationToken::default();
    let model = session.load(&args, limits, &facts, cancellation.clone())?;
    let config = SourceMapConfig { identity, task, budget, planning: command.host.planning(), mapping,
        native: session.native(&args)?, preparation_reserve_bytes: limits.preparation_bytes,
        reduction_reserve_bytes: command.reduction_reserve_mib.checked_mul(MIB).ok_or(CandidateError::Arguments)? };
    let result = session.engine.synthesize_int8_document_answer(&model, text, Arc::new(planner), Arc::new(vocabulary),
        config, cancellation).map_err(|_| CandidateError::Execution)?;
    session.remaining()?;
    expected.verify_completed(result.result()).map_err(|_| CandidateError::Execution)?;
    // Retain output authority through final serialization, write AND flush.
    deliver(&session, &facts, &result, command.max_map_result_bytes + 4096, output)
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
        assert_eq!(synthesis.preflight_metadata().discovery(), plain.preflight_metadata());
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
}
