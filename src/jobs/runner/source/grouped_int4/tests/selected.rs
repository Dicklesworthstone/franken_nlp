//! Exact source planning and authenticated storage, without native inference.
use super::*;

fn rows(max_rows_per_step: usize) -> Int4JsonSparseLimits { Int4JsonSparseLimits { max_rows_per_step } }
fn selected(task: BuiltInTask, cap: usize) -> Int4SourceJobPlanner {
    planner(task).with_selected_rows(rows(cap)).unwrap()
}

#[test]
fn selected_source_recipe_binds_task_native_and_cap_while_dense_bytes_stay_unchanged() {
    for task in [BuiltInTask::Ner, BuiltInTask::Keyphrases, BuiltInTask::Summarize, BuiltInTask::Answer] {
        let dense = planner(task);
        let baseline = serde_json::to_value(&dense.recipe).unwrap();
        assert!(baseline.get("selected_rows").is_none());
        for cap in [1, 7, NANBEIGE_VOCAB_SIZE] {
            let p = selected(task, cap);
            p.check_job_limits(limits(p.native_limits())).unwrap();
            assert_eq!(p.selected_rows(), Some(rows(cap)));
            assert_eq!(p.compiler().unwrap().selected_rows(), Some(rows(cap)));
            let mut value = serde_json::to_value(&p.recipe).unwrap();
            assert_eq!(value["execution"], INT4_SPARSE_SOURCE_EXECUTION);
            assert_eq!(value["selected_rows"]["execution"], INT4_SPARSE_SOURCE_EXECUTION);
            assert_eq!(value["selected_rows"]["native_execution"], INT4_SPARSE_JSON_EXECUTION);
            assert_eq!(value["selected_rows"]["max_rows_per_step"], cap);
            value.as_object_mut().unwrap().remove("selected_rows");
            value["execution"] = baseline["execution"].clone();
            assert_eq!(value, baseline);
            assert_eq!(freeze(&dense).binding.compare(&freeze(&p).binding),
                Err(JobError::Mismatch(MismatchField::Recipe)));
            assert!(p.with_selected_rows(rows(cap)).is_err());
        }
    }
    for cap in [0, NANBEIGE_VOCAB_SIZE + 1, usize::MAX] {
        assert!(planner(BuiltInTask::Keyphrases).with_selected_rows(rows(cap)).is_err());
    }
    for field in 0..5 {
        let mut changed = selected(BuiltInTask::Keyphrases, 7);
        match field {
            0 => changed.recipe.execution = INT4_SOURCE_EXECUTION,
            1 => changed.recipe.selected_rows.as_mut().unwrap().execution = "different-source-task",
            2 => changed.recipe.selected_rows.as_mut().unwrap().native_execution = "different-native",
            3 => changed.recipe.selected_rows.as_mut().unwrap().max_rows_per_step = 8,
            _ => changed.recipe.selected_rows = None,
        }
        assert_eq!(changed.check_job_limits(limits(changed.native_limits())), Err(BatchCode::Admission.into()));
        assert!(changed.compiler().is_err());
    }
}

#[test]
fn foreign_selected_cap_task_and_encoding_refuse_before_durable_work() {
    let policies = [planner(BuiltInTask::Keyphrases), selected(BuiltInTask::Keyphrases, 7),
        selected(BuiltInTask::Keyphrases, 8), selected(BuiltInTask::Keyphrases, NANBEIGE_VOCAB_SIZE)];
    for (index, p) in policies.iter().enumerate() {
        let prepared = prepare(p, document());
        for (expected, owner) in policies.iter().enumerate() {
            let result = durable_plan_work(&prepared, &owner.recipe);
            if index == expected {
                assert_eq!(result.unwrap(), JobWork::from_int4(prepared.planned_work(),
                    owner.native_limits().masks.max_visits_per_item));
            } else {
                let failure = result.unwrap_err();
                assert!(failure.stop); assert_eq!(failure.fault.code, BatchCode::Admission);
            }
        }
    }
    let dense = prepare(&policies[0], document());
    let full = prepare(&policies[3], document());
    assert_eq!(dense.planned_work(), full.planned_work());
    assert_ne!(dense.execution_identity(), full.execution_identity());
    let other_task = selected(BuiltInTask::Ner, 7);
    assert!(durable_plan_work(&prepare(&other_task, document()), &policies[1].recipe).is_err());
    let other_group = configured(BuiltInTask::Keyphrases, 16,
        Some(args(BuiltInTask::Keyphrases)), native(16)).with_selected_rows(rows(7)).unwrap();
    assert!(durable_plan_work(&prepare(&other_group, document()), &policies[1].recipe).is_err());
    let (call, total) = compact_limits();
    let compact = compact_planner(BuiltInTask::Keyphrases, call, total).with_selected_rows(rows(7)).unwrap();
    assert!(durable_plan_work(&prepare(&compact, document()), &policies[1].recipe).is_err());
    let prepared = prepare(&policies[1], document());
    for axis in 0..5 {
        let mut owner = selected(BuiltInTask::Keyphrases, 7);
        match axis {
            0 => owner.recipe.task_ceiling.max_input_tokens = prepared.task_budget().max_input_tokens - 1,
            1 => owner.recipe.task_ceiling.max_output_tokens = prepared.task_budget().max_output_tokens - 1,
            2 => owner.recipe.task_ceiling.max_output_bytes = prepared.task_budget().max_output_bytes - 1,
            3 => owner.recipe.task_ceiling.max_grammar_states = prepared.task_budget().max_grammar_states - 1,
            _ => owner.recipe.task_ceiling.max_kv_bytes -= 1,
        }
        assert!(durable_plan_work(&prepared, &owner.recipe).is_err());
    }
}

#[test]
fn selected_source_quotes_reduce_only_head_work_and_fit_selected_only_lifetime_caps() {
    for task in [BuiltInTask::Ner, BuiltInTask::Keyphrases, BuiltInTask::Summarize, BuiltInTask::Answer] {
        let dense = prepare(&planner(task), document());
        let p = selected(task, 7);
        let prepared = prepare(&p, document());
        let quote = durable_plan_work(&prepared, &p.recipe).unwrap();
        assert_eq!(prepared.execution_version(), INT4_SPARSE_SOURCE_EXECUTION);
        assert_eq!(quote.int4_model_work(), prepared.planned_work());
        assert_eq!(quote.model.projected_logits, 7 * u64::from(budget().max_output_tokens));
        assert_eq!(quote.model.forward_positions, dense.planned_work().forward_positions);
        assert_eq!(quote.model.attention_pairs, dense.planned_work().attention_pairs);
        assert!(quote.model.projections.dot_products < dense.planned_work().projections.dot_products);
        assert!(quote.model.projections.multiply_accumulates < dense.planned_work().projections.multiply_accumulates);
        assert_eq!(quote.mask_node_visits, native(WIDTH).masks.max_visits_per_item);
        assert_eq!(prepared.max_kv_bytes(), dense.max_kv_bytes());
        let mut cap = native(WIDTH); cap.max_model_work = prepared.planned_work();
        cap.masks.max_visits_per_run = cap.masks.max_visits_per_item;
        let bounded = configured(task, WIDTH, Some(args(task)), cap).with_selected_rows(rows(7)).unwrap();
        bounded.check_job_limits(limits(cap)).unwrap();
        assert!(quote.fits(limits(cap).max_work));
        assert!(!JobWork::from_int4(dense.planned_work(), quote.mask_node_visits).fits(limits(cap).max_work));
    }
}

#[test]
fn selecting_rows_preserves_the_existing_compact_source_owner_and_spent_work() {
    let (call, total) = compact_limits();
    for task in [BuiltInTask::Ner, BuiltInTask::Keyphrases, BuiltInTask::Summarize, BuiltInTask::Answer] {
        let p = compact_planner(task, call, total);
        let dense = prepare(&p, document());
        let owner = Arc::clone(&p.planner);
        let spent = owner.map_document_encoding_work().unwrap().unwrap();
        let bytes = p.encoding_workspace_bytes();
        let selected = p.with_selected_rows(rows(7)).unwrap();
        assert!(Arc::ptr_eq(&owner, &selected.planner));
        assert_eq!(selected.encoding_workspace_bytes(), bytes);
        assert_eq!(selected.encoding_work_limits(), Some(total));
        assert_eq!(owner.map_document_encoding_work().unwrap().unwrap(), spent);
        let prepared = prepare(&selected, document());
        assert_eq!(prepared.input_encoding(), &UntrustedEncoding::SafeBpeV1(call));
        assert!(durable_plan_work(&prepared, &selected.recipe).is_ok());
        assert!(durable_plan_work(&dense, &selected.recipe).is_err());
        assert_eq!(owner.map_document_encoding_work().unwrap().unwrap(), spent.checked_add(spent).unwrap());
    }
}

#[test]
fn selected_source_resume_authenticates_before_tail_repair_and_skips_completed_items() {
    let root = Root::new(); let input = inputs(); let events = Rc::new(Events::default());
    let job = limits(native(WIDTH));
    let make = || Processor::new(selected(BuiltInTask::Keyphrases, 7), Rc::clone(&events));
    let mut run = JobRunner::create(&root.0, key(), JobId([9; 16]), job, &input,
        make(), &mut Continue).unwrap();
    assert_eq!(error(run.step(&mut CancelAfterSync(Rc::clone(&events)))),
        JobRunError::Storage(JobError::Cancelled(DecodeCancellationKind::PollQuota)));
    let spent = run.progress().reserved_work;
    assert_eq!(spent.model.projected_logits, 7 * u64::from(budget().max_output_tokens));
    assert_eq!((run.progress().attempts, run.progress().committed), (1, 0));
    drop(run);
    for wrong in [planner(BuiltInTask::Keyphrases), selected(BuiltInTask::Keyphrases, 8),
        selected(BuiltInTask::Keyphrases, NANBEIGE_VOCAB_SIZE)] {
        assert_eq!(error(JobRunner::resume(&root.0, key(), JobId([9; 16]), job, &input,
            Processor::new(wrong, Rc::clone(&events)), TailPolicy::DiscardUncommitted, &mut Continue)),
            JobRunError::Storage(JobError::Mismatch(MismatchField::Recipe)));
        assert_eq!(events.executions.get(), 1);
        assert_eq!(error(JobRunner::resume(&root.0, key(), JobId([9; 16]), job, &input,
            make(), TailPolicy::Refuse, &mut Continue)), JobRunError::Storage(JobError::UncommittedTail));
    }
    let mut resumed = JobRunner::resume(&root.0, key(), JobId([9; 16]), job, &input,
        make(), TailPolicy::DiscardUncommitted, &mut Continue).unwrap();
    assert_eq!(resumed.progress().reserved_work, spent);
    assert_eq!(resumed.step(&mut Continue).unwrap().unwrap().reserved_work, spent.checked_add(spent).unwrap());
    let complete = resumed.run(&mut Continue).unwrap();
    assert_eq!((complete.committed, complete.attempts), (2, 3));
    resumed.materialize_ordered(&mut Continue).unwrap(); drop(resumed);
    let bytes = fs::read(root.0.join("materialized.ndjson")).unwrap();
    let executions = events.executions.get();
    let mut completed = JobRunner::resume(&root.0, key(), JobId([9; 16]), job, &input,
        make(), TailPolicy::Refuse, &mut Continue).unwrap();
    let replay = completed.run(&mut Continue).unwrap();
    assert_eq!(replay.reserved_work, complete.reserved_work);
    assert_eq!((replay.committed, replay.attempts), (2, 3));
    assert_eq!(events.executions.get(), executions);
    assert_eq!(fs::read(root.0.join("materialized.ndjson")).unwrap(), bytes);
}

#[test]
fn each_selected_source_attempt_counter_stays_spent_after_resume() {
    for axis in 0..6 {
        let p = selected(BuiltInTask::Keyphrases, 7);
        let input = inputs();
        let parsed = serde_json::from_slice(input[0].original).unwrap();
        let quote = durable_plan_work(&prepare(&p, parsed), &p.recipe).unwrap();
        let mut cap = native(WIDTH);
        let mut frozen = JobWork::from_int4(cap.max_model_work, cap.masks.max_visits_per_run);
        let mut quoted = quote; *axis_mut(&mut frozen, axis) = *axis_mut(&mut quoted, axis);
        cap.max_model_work = frozen.int4_model_work(); cap.masks.max_visits_per_run = frozen.mask_node_visits;
        let events = Rc::new(Events::default()); let root = Root::new(); let job = limits(cap);
        let make = || configured(BuiltInTask::Keyphrases, WIDTH, Some(args(BuiltInTask::Keyphrases)), cap)
            .with_selected_rows(rows(7)).unwrap();
        let mut processor = Processor::new(make(), Rc::clone(&events)); processor.fail = true;
        let mut run = JobRunner::create(&root.0, key(), JobId([9; 16]), job, &input, processor, &mut Continue).unwrap();
        assert_eq!(error(run.step(&mut Continue)), JobRunError::Processor(BatchCode::Execution.into()));
        assert_eq!(run.progress().reserved_work, quote); drop(run);
        let mut resumed = JobRunner::resume(&root.0, key(), JobId([9; 16]), job, &input,
            Processor::new(make(), Rc::clone(&events)), TailPolicy::Refuse, &mut Continue).unwrap();
        assert_eq!(error(resumed.step(&mut Continue)), JobRunError::Storage(JobError::WorkLimit));
        assert_eq!(events.executions.get(), 1);
        assert_eq!(resumed.progress().reserved_work, quote);
        assert_eq!(resumed.progress().attempts, 1);
    }
}
