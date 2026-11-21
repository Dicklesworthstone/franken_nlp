//! Projection-policy regressions use the real authenticated storage protocol
//! with the private synthetic driver. They make no native model-quality claim.
use super::*;

fn selected(rows: usize) -> Int4JsonSparseLimits { Int4JsonSparseLimits { max_rows_per_step: rows } }

#[test]
fn selected_recipe_preserves_dense_bytes_and_binds_both_execution_versions_and_cap() {
    let dense = recipe(native());
    let dense_json = serde_json::to_value(&dense).unwrap();
    assert!(dense_json.get("selected_rows").is_none());
    assert_eq!(canonjson::canonical_bytes(&dense).unwrap(),
        canonjson::canonical_bytes(&recipe(native()).with_selected_rows(None).unwrap()).unwrap());
    for reuse in [false, true] {
        let make = |rows| {
            let recipe = recipe(native()).with_selected_rows(Some(selected(rows))).unwrap();
            if reuse { recipe.with_reused_prefill().unwrap() } else { recipe }
        };
        let sparse = make(4096);
        sparse.check_job_limits(sparse.job).unwrap();
        assert_eq!(sparse.checked_selected_rows().unwrap(), Some(selected(4096)));
        assert_eq!(sparse.reuses_prefill(), reuse);
        assert_eq!(sparse.sampling.effective_seed, dense.sampling.effective_seed);
        assert_eq!(sparse.job.max_work, dense.job.max_work);
        assert_eq!(canonjson::canonical_bytes(&sparse.native).unwrap(),
            canonjson::canonical_bytes(&dense.native).unwrap());
        assert_eq!(freeze(&sparse).binding.compare(&freeze(&make(4095)).binding),
            Err(JobError::Mismatch(MismatchField::Recipe)));
        distinct(&sparse);
        let mut unselected = serde_json::to_value(&sparse).unwrap();
        unselected.as_object_mut().unwrap().remove("selected_rows");
        unselected["execution"] = serde_json::json!(SAMPLED_SENTIMENT_EXECUTION);
        unselected["native_execution"] = serde_json::json!(INT4_JSON_SAMPLED_EXECUTION);
        assert_eq!(unselected, dense_json);
        let prefill = Int4PrefillLimits { max_batch_rows: 4,
            max_extra_scratch_bytes: Int4PrefillLimits::required_extra_scratch_bytes(4).unwrap() };
        let a = recipe(native()).with_selected_rows(Some(selected(4096))).unwrap()
            .with_prefill(prefill).unwrap().with_reused_prefill().unwrap();
        let b = recipe(native()).with_reused_prefill().unwrap()
            .with_selected_rows(Some(selected(4096))).unwrap().with_prefill(prefill).unwrap();
        assert_eq!(canonjson::canonical_bytes(&a).unwrap(), canonjson::canonical_bytes(&b).unwrap());
    }
}

#[test]
fn mixed_projection_caps_and_execution_contracts_refuse_factory_authority() {
    for reuse in [false, true] { for axis in 0..7 {
        let mut recipe = recipe(native()).with_selected_rows(Some(selected(4096))).unwrap();
        if reuse { recipe = recipe.with_reused_prefill().unwrap(); }
        match axis {
            0 => recipe.execution = SAMPLED_SENTIMENT_EXECUTION,
            1 => recipe.native_execution = INT4_JSON_SAMPLED_EXECUTION,
            2 => recipe.selected_rows.as_mut().unwrap().execution = SAMPLED_SENTIMENT_EXECUTION,
            3 => recipe.selected_rows.as_mut().unwrap().native_execution = INT4_JSON_SAMPLED_EXECUTION,
            4 => recipe.selected_rows.as_mut().unwrap().max_rows_per_step = 0,
            5 => recipe.selected_rows.as_mut().unwrap().max_rows_per_step = usize::MAX,
            _ => recipe.selected_rows = None,
        }
        assert!(recipe.check_job_limits(recipe.job).is_err(), "policy axis {axis}");
    } }
    for rows in [0, crate::native_engine::lmhead::NANBEIGE_VOCAB_SIZE + 1] {
        assert!(recipe(native()).with_selected_rows(Some(selected(rows))).is_err());
    }
}

#[test]
fn projection_policy_mismatch_precedes_tail_repair_for_both_encoding_and_prefill_modes() {
    for compact_input in [false, true] { for reuse in [false, true] {
        let root = Root::new(); let cap = native(); let input = inputs();
        let retention = InputRetentionLimits { max_spool_bytes: 65536 };
        let events = Rc::new(Events::default());
        let make = |projection| {
            let mut processor = Processor::new(cap, Rc::clone(&events));
            processor.recipe = processor.recipe.with_selected_rows(projection).unwrap();
            if compact_input { processor.recipe.input_encoding = compact(); }
            if reuse { processor = processor.with_reused_prefill(); }
            processor
        };
        let policy = Some(selected(4096));
        let mut control = CancelAfterSync(Rc::clone(&events));
        let mut run = JobRunner::create_with_retained_inputs(&root.0, key(), JobId([9; 16]), limits(cap),
            retention, &input, make(policy), &mut control).unwrap();
        assert_eq!(error(run.step(&mut control)),
            JobRunError::Storage(JobError::Cancelled(DecodeCancellationKind::PollQuota)));
        assert_eq!(run.progress().reserved_work, JobWork::from_int4(work(1), 5)); drop(run);
        for changed in [None, Some(selected(4095))] {
            assert_eq!(error(JobRunner::resume_with_retained_inputs(&root.0, key(), JobId([9; 16]), limits(cap),
                retention, make(changed), TailPolicy::DiscardUncommitted, &mut Continue)),
                JobRunError::Storage(JobError::Mismatch(MismatchField::Recipe)));
            assert_eq!(events.executions.get(), 1);
        }
        assert_eq!(error(JobRunner::resume_with_retained_inputs(&root.0, key(), JobId([9; 16]), limits(cap),
            retention, make(policy), TailPolicy::Refuse, &mut Continue)),
            JobRunError::Storage(JobError::UncommittedTail));
        let mut resumed = JobRunner::resume_with_retained_inputs(&root.0, key(), JobId([9; 16]), limits(cap),
            retention, make(policy), TailPolicy::DiscardUncommitted, &mut Continue).unwrap();
        let done = resumed.run(&mut Continue).unwrap();
        assert_eq!((done.committed, done.attempts), (2, 3));
        assert_eq!(done.reserved_work, JobWork::from_int4(work(3), 15));
        resumed.verify(&mut Continue).unwrap(); drop(resumed);
        let mut complete = JobRunner::resume_with_retained_inputs(&root.0, key(), JobId([9; 16]), limits(cap),
            retention, make(policy), TailPolicy::Refuse, &mut Continue).unwrap();
        assert_eq!(complete.run(&mut Continue).unwrap().reserved_work, done.reserved_work);
        assert_eq!(events.executions.get(), 3);
    } }
}

#[test]
fn selected_failures_never_renew_any_of_the_six_durable_attempt_counters() {
    for reuse in [false, true] { for axis in 0..6 {
        let mut cap = native(); cap.max_item_work = work(1); cap.masks.max_visits_per_item = 5;
        if axis < 5 { *work_axis(&mut cap.max_model_work, axis) = *work_axis(&mut work(1), axis); }
        else { cap.masks.max_visits_per_run = 5; }
        let root = Root::new(); let input = inputs(); let events = Rc::new(Events::default());
        let make = || {
            let mut processor = Processor::new(cap, Rc::clone(&events));
            processor.recipe = processor.recipe.with_selected_rows(Some(selected(4096))).unwrap();
            if reuse { processor.with_reused_prefill() } else { processor }
        };
        let expected = if axis % 2 == 0 { BatchCode::Execution.into() }
            else { BatchFault::cancelled(DecodeCancellationKind::CostBudget) };
        let mut processor = make(); processor.fail = Some(expected);
        let mut run = JobRunner::create(&root.0, key(), JobId([9; 16]), limits(cap), &input,
            processor, &mut Continue).unwrap();
        assert_eq!(error(run.step(&mut Continue)), JobRunError::Processor(expected));
        assert_eq!(run.progress().reserved_work, JobWork::from_int4(work(1), 5)); drop(run);
        let mut resumed = JobRunner::resume(&root.0, key(), JobId([9; 16]), limits(cap), &input,
            make(), TailPolicy::Refuse, &mut Continue).unwrap();
        assert_eq!(error(resumed.step(&mut Continue)), JobRunError::Storage(JobError::WorkLimit));
        assert_eq!(events.executions.get(), 1);
        assert_eq!(resumed.progress().reserved_work, JobWork::from_int4(work(1), 5));
    } }
}

#[test]
fn actual_selected_factory_preserves_private_policy_across_compact_and_prompt_schedules() {
    use crate::native_engine::strict_int4::weights::projection_fixture;
    let (model, _) = projection_fixture(32);
    let vocabulary = Arc::new(SampledSentimentVocabulary::pinned().unwrap());
    for encoding in [UntrustedEncoding::ByteFallback, compact()] {
        let make = |projection: Option<Int4JsonSparseLimits>| {
            let planner = SampledSentimentPlanner::for_model_with_encoding(model.identity(),
                options(), task(), planning(), encoding.clone()).unwrap();
            let planner = match projection { Some(cap) => planner.with_selected_rows(cap).unwrap(), None => planner };
            SampledSentimentJobPlanner::new(planner, Arc::clone(&vocabulary), native(), limits(native())).unwrap()
        };
        let dense = make(None);
        let sparse = make(Some(selected(4096)));
        assert_eq!(sparse.selected_rows(), Some(selected(4096)));
        assert_eq!(sparse.preparation_requirement_bytes(), dense.preparation_requirement_bytes());
        assert_eq!(sparse.document_encoding(), dense.document_encoding());
        assert_eq!(sparse.native_limits().max_model_work, dense.native_limits().max_model_work);
        assert_ne!(sparse.base_execution_identity(), dense.base_execution_identity());
        let identity = sparse.base_execution_identity().clone();
        let sparse = sparse.with_layer_major_prefill(Int4PrefillLimits { max_batch_rows: 4,
            max_extra_scratch_bytes: Int4PrefillLimits::required_extra_scratch_bytes(4).unwrap() }).unwrap()
            .with_reused_prefill().unwrap();
        sparse.check_job_limits(limits(native())).unwrap();
        assert_eq!(sparse.base_execution_identity(), &identity);
        assert_eq!(sparse.selected_rows(), Some(selected(4096)));
        assert_eq!(sparse.job_recipe().execution, SAMPLED_SENTIMENT_SPARSE_REUSED_PREFILL_EXECUTION);
    }
}
