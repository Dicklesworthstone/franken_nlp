//! Real pinned plans and private lifecycle fixtures; native parity is separate.
use super::*;
use crate::tasks::source_planning::grouped_int4::{INT4_SOURCE_EXECUTION, INT4_SPARSE_SOURCE_EXECUTION};

fn rows(max_rows_per_step: usize) -> Int4JsonSparseLimits { Int4JsonSparseLimits { max_rows_per_step } }

#[test]
fn all_four_serial_tasks_seal_the_same_selected_policy_as_direct_source_plans() {
    let p = planner();
    for kind in [BuiltInTask::Ner, BuiltInTask::Keyphrases, BuiltInTask::Summarize, BuiltInTask::Answer] {
        let dense = compiler(&p, kind).prepare_with_control(document(), &mut Continue).unwrap();
        let selected = compiler(&p, kind).with_selected_rows(rows(7)).unwrap();
        let plan = selected.prepare_with_control(document(), &mut Continue).unwrap();
        assert_eq!(selected.selected_rows(), Some(rows(7)));
        assert_eq!(plan.execution_version(), INT4_SPARSE_SOURCE_EXECUTION);
        assert_eq!(plan.selected_rows(), Some(rows(7)));
        assert_eq!(plan.task_budget(), dense.task_budget());
        assert_eq!(plan.prompt_tokens(), dense.prompt_tokens());
        assert_eq!(plan.max_result_bytes(), dense.max_result_bytes());
        assert_eq!(plan.max_kv_bytes(), dense.max_kv_bytes());
        assert_eq!(plan.request_digest(), dense.request_digest());
        assert_eq!(plan.execution_identity().prompt_digest, dense.execution_identity().prompt_digest);
        assert_eq!(plan.execution_identity().schema_digest, dense.execution_identity().schema_digest);
        assert_ne!(plan.execution_identity().decision_policy_digest, dense.execution_identity().decision_policy_digest);
        assert!(plan.verify_identity(dense.execution_identity()).is_err());
        let direct = dense.with_selected_rows(rows(7)).unwrap();
        plan.verify_identity(direct.execution_identity()).unwrap();
        assert_eq!(plan.planned_work(), direct.planned_work());
        assert_eq!(plan.planned_work().projected_logits, 7 * u64::from(budget().max_output_tokens));
        assert!(plan.require_full_projection().is_err());
        assert_eq!(selected.selected_rows(), Some(rows(7)));
    }
}

#[test]
fn cap_validation_and_full_width_identity_cannot_disguise_the_stream_policy() {
    let p = planner();
    for cap in [0, NANBEIGE_VOCAB_SIZE + 1, usize::MAX] {
        assert_eq!(compiler(&p, BuiltInTask::Keyphrases).with_selected_rows(rows(cap)).err().unwrap().code,
            BatchCode::InvalidLimits);
    }
    assert!(compiler(&p, BuiltInTask::Keyphrases).with_selected_rows(rows(7)).unwrap()
        .with_selected_rows(rows(7)).is_err());
    let dense = compiler(&p, BuiltInTask::Keyphrases);
    assert!(dense.selected_rows().is_none());
    let full = compiler(&p, BuiltInTask::Keyphrases).with_selected_rows(rows(NANBEIGE_VOCAB_SIZE)).unwrap();
    assert_eq!(full.selected_rows(), Some(rows(NANBEIGE_VOCAB_SIZE)));
    let dense = dense.prepare_with_control(document(), &mut Continue).unwrap();
    let full = full.prepare_with_control(document(), &mut Continue).unwrap();
    assert_eq!(dense.planned_work(), full.planned_work());
    assert_eq!(dense.execution_version(), INT4_SOURCE_EXECUTION);
    assert!(full.verify_identity(dense.execution_identity()).is_err());
}

#[test]
fn per_record_tighter_options_do_not_override_the_frozen_projection_policy() {
    let p = planner(); let c = compiler(&p, BuiltInTask::Keyphrases).with_selected_rows(rows(7)).unwrap();
    let mut item = document(); let mut task = args(BuiltInTask::Keyphrases);
    if let SourceBatchArgs::Keyphrases { options, budget } = &mut task {
        options.max_phrases = 1; budget.max_output_tokens = 64;
    }
    item.task_args = Some(task);
    let plan = c.prepare_with_control(item, &mut Continue).unwrap();
    assert_eq!(plan.selected_rows(), Some(rows(7)));
    assert_eq!(plan.planned_work().projected_logits, 7 * 64);
    let mut wire = serde_json::to_value(args(BuiltInTask::Keyphrases)).unwrap();
    wire.as_object_mut().unwrap().insert("max_selected_rows".to_owned(), serde_json::json!(1));
    assert!(serde_json::from_value::<SourceBatchArgs>(wire).is_err());
}

#[test]
fn dense_or_different_cap_injection_fails_before_work_admission_and_native_calls() {
    let p = planner();
    for (expected, supplied) in [(Some(rows(7)), None), (Some(rows(7)), Some(rows(8))),
        (Some(rows(NANBEIGE_VOCAB_SIZE)), None), (None, Some(rows(NANBEIGE_VOCAB_SIZE)))] {
        let c = compiler(&p, BuiltInTask::Keyphrases);
        let c = match supplied { Some(cap) => c.with_selected_rows(cap).unwrap(), None => c };
        let plan = c.prepare_with_control(document(), &mut Continue).unwrap();
        let state = Rc::new(State::default()); let (mut driver, mut admission, _) = host(&state, Mode::Success);
        let mut run = RunState::new(limits(), expected);
        let error = execute_admitted(plan, &mut driver, &mut admission, &mut run,
            context(1, 1), &mut Continue).err().unwrap();
        assert!(error.stop && run.failed); assert_eq!(error.fault.code, BatchCode::Admission);
        assert_eq!(run.reserved, Int4Work::default()); assert_eq!(run.mask_visits, 0);
        assert_eq!(state.admissions.get(), 0); assert_eq!(state.calls.get(), 0);
    }
}

fn set_work(result: &mut Int4SourceTaskRun, work: Int4Work) {
    result.model_work = work;
    let SourceTaskResult::Keyphrases(inner) = &mut result.result else { panic!("keyphrase fixture required") };
    inner.forward_positions = work.forward_positions; inner.projected_logits = work.projected_logits;
}

#[test]
fn variable_rows_require_scored_eos_and_exact_every_axis_completion() {
    let p = planner(); let plan = compiler(&p, BuiltInTask::Keyphrases).with_selected_rows(rows(7)).unwrap()
        .prepare_with_control(document(), &mut Continue).unwrap();
    let mut output = completion(&plan);
    let count = observed(&output.result).tokens;
    let actual = Int4Work::for_sequence(32, 0, plan.prompt_tokens() + count - 1, count * 3).unwrap();
    set_work(&mut output, actual);
    assert!(valid_result(&plan, &output, 5));
    for axis in 0..5 {
        let mut changed = output.clone();
        match axis { 0 => changed.model_work.forward_positions += 1,
            1 => changed.model_work.projected_logits += 1, 2 => changed.model_work.attention_pairs += 1,
            3 => changed.model_work.projections.dot_products += 1,
            _ => changed.model_work.projections.multiply_accumulates += 1 }
        assert!(!valid_result(&plan, &changed, 5), "axis {axis}");
    }
    for projected in [count - 1, count * 7 + 1] {
        let mut changed = output.clone();
        set_work(&mut changed, Int4Work::for_sequence(32, 0, plan.prompt_tokens() + count - 1, projected).unwrap());
        assert!(!valid_result(&plan, &changed, 5));
    }
    let SourceTaskResult::Keyphrases(inner) = &output.result else { panic!("wrong fixture") };
    let eos = *inner.generated_token_ids.last().unwrap();
    for axis in 0..4 {
        let mut ids = inner.generated_token_ids.clone();
        match axis { 0 => { ids.pop(); }, 1 => ids[0] = eos,
            2 => ids.push(eos), _ => ids[0] = NANBEIGE_VOCAB_SIZE as u32 }
        assert!(plan.verify_completion_work(&ids, actual).is_err());
    }
    let mut changed = output.clone(); changed.execution = INT4_SOURCE_EXECUTION.to_owned();
    assert!(!valid_result(&plan, &changed, 5));
    assert!(!valid_result(&plan, &output, 4));
}

#[test]
fn selected_streams_keep_delivery_guards_and_nonrenewable_work_across_epochs() {
    let p = planner();
    for kind in [BuiltInTask::Keyphrases, BuiltInTask::Answer] {
        for failure in 0..3 {
            let state = Rc::new(State::default()); let (driver, admission, _) = host(&state, Mode::Success);
            let mut h = Harness { compiler: compiler(&p, kind).with_selected_rows(rows(7)).unwrap(),
                driver, admission, run: RunState::new(limits(), Some(rows(7))), planning: Rc::new(Cell::new(false)) };
            let mut writer = Writer { state: Rc::clone(&state), failure, broken: false, document: false, flushed: 0 };
            let bytes = b"{\"id\":\"a\",\"text\":\"one\"}\n{\"flush\":true}\n{\"id\":\"a\",\"text\":\"two\"}\n";
            let mut input = Cursor::new(bytes);
            let result = run_ndjson(&mut input, &mut writer, &mut h, BatchLimits::default(), &mut Continue);
            if failure == 0 {
                let summary = result.unwrap(); assert_eq!(summary.succeeded, 2); assert_eq!(summary.failed, 0);
                assert_eq!(summary.reserved_work.projected_logits, 2 * 7 * u64::from(budget().max_output_tokens));
                assert_eq!(h.run.mask_visits, 2 * limits().masks.max_visits_per_item);
                assert_eq!(writer.flushed, 2);
            } else {
                assert_eq!(result.unwrap_err().fault.code, BatchCode::OutputIo);
                assert_eq!(state.calls.get(), 1); assert!(input.position() < bytes.len() as u64);
            }
            assert_eq!(state.drops.get(), state.calls.get()); assert!(!state.live.get());
        }
    }
}

#[test]
fn selected_native_failure_remains_fatal_and_cannot_become_answer_abstention() {
    let p = planner();
    for mode in [Mode::Cancel, Mode::Dirty, Mode::Oversize, Mode::WrongWork] {
        let c = compiler(&p, BuiltInTask::Answer).with_selected_rows(rows(7)).unwrap();
        let state = Rc::new(State::default()); let (mut driver, mut admission, _) = host(&state, mode);
        let mut run = RunState::new(limits(), Some(rows(7)));
        let plan = c.prepare_with_control(document(), &mut Continue).unwrap(); let work = plan.planned_work();
        let error = execute_admitted(plan, &mut driver, &mut admission, &mut run, context(1, 1), &mut Continue).err().unwrap();
        assert!(error.stop && run.failed); assert_eq!(run.reserved, work);
        assert_eq!(run.mask_visits, limits().masks.max_visits_per_item);
        if matches!(mode, Mode::Cancel) { assert_eq!(error.fault.cancellation, Some(DecodeCancellationKind::Deadline)); }
        assert!(!state.live.get()); assert!(!state.running.get()); assert_eq!(state.drops.get(), 1);
        driver.clean = true; driver.mode = Mode::Success;
        assert!(execute_admitted(c.prepare_with_control(document(), &mut Continue).unwrap(),
            &mut driver, &mut admission, &mut run, context(2, 2), &mut Continue).is_err());
        assert_eq!(state.calls.get(), 1);
    }
}
