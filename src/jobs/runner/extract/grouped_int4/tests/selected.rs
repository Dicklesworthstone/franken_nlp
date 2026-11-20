//! Pinned selected-row planning and authenticated storage; no native inference.
use super::*;

fn rows(max_rows_per_step: usize) -> Int4JsonSparseLimits { Int4JsonSparseLimits { max_rows_per_step } }
fn selected(cap: usize) -> Int4ExtractionJobPlanner { planner().with_selected_rows(rows(cap)).unwrap() }

#[test]
fn selected_recipe_authenticates_both_contracts_and_cap_without_changing_dense_shape() {
    let dense = planner();
    let baseline = serde_json::to_value(&dense.recipe).unwrap();
    assert!(baseline.get("selected_rows").is_none());
    for cap in [1, 7, NANBEIGE_VOCAB_SIZE] {
        let p = selected(cap);
        p.check_job_limits(job_limits(p.native_limits())).unwrap();
        assert_eq!(p.selected_rows(), Some(rows(cap)));
        let mut value = serde_json::to_value(&p.recipe).unwrap();
        assert_eq!(value["execution"], INT4_SPARSE_EXTRACT_VERSION);
        assert_eq!(value["selected_rows"]["execution"], INT4_SPARSE_EXTRACT_VERSION);
        assert_eq!(value["selected_rows"]["native_execution"], INT4_SPARSE_JSON_EXECUTION);
        assert_eq!(value["selected_rows"]["max_rows_per_step"], cap);
        value.as_object_mut().unwrap().remove("selected_rows");
        value["execution"] = baseline["execution"].clone();
        assert_eq!(value, baseline);
        assert_eq!(freeze(&dense).binding.compare(&freeze(&p).binding),
            Err(JobError::Mismatch(MismatchField::Recipe)));
        assert!(p.with_selected_rows(rows(cap)).is_err());
    }
    for cap in [0, NANBEIGE_VOCAB_SIZE + 1, usize::MAX] {
        assert!(planner().with_selected_rows(rows(cap)).is_err());
    }
    let baseline = freeze(&selected(7));
    for field in 0..4 {
        let mut changed = selected(7);
        match field {
            0 => changed.recipe.execution = INT4_EXTRACT_VERSION,
            1 => changed.recipe.selected_rows.as_mut().unwrap().execution = "different-task",
            2 => changed.recipe.selected_rows.as_mut().unwrap().native_execution = "different-native",
            _ => changed.recipe.selected_rows.as_mut().unwrap().max_rows_per_step = 8,
        }
        assert_eq!(baseline.binding.compare(&freeze(&changed).binding),
            Err(JobError::Mismatch(MismatchField::Recipe)));
        assert_eq!(changed.check_job_limits(job_limits(changed.native_limits())),
            Err(BatchCode::Admission.into()));
    }
}

#[test]
fn foreign_strategy_and_exact_cap_refuse_before_any_durable_debit() {
    let policies = [planner(), selected(7), selected(8), selected(NANBEIGE_VOCAB_SIZE)];
    for (index, p) in policies.iter().enumerate() {
        let prepared = p.compiler.prepare(shared_document(document("Alice")).unwrap()).unwrap();
        for (expected, owner) in policies.iter().enumerate() {
            let result = durable_plan_work(&prepared, &owner.recipe);
            if expected == index {
                assert_eq!(result.unwrap(),
                    JobWork::from_int4(prepared.planned_work(), owner.native_limits().masks.max_visits_per_item));
            } else {
                let failure = result.unwrap_err();
                assert!(failure.stop); assert_eq!(failure.fault.code, BatchCode::Admission);
            }
        }
    }
    // Even a vocabulary-wide selected cap stays a distinct authenticated policy.
    let dense = policies[0].compiler.prepare(shared_document(document("Alice")).unwrap()).unwrap();
    let full = policies[3].compiler.prepare(shared_document(document("Alice")).unwrap()).unwrap();
    assert_eq!(dense.planned_work(), full.planned_work());
    assert_ne!(dense.execution_identity(), full.execution_identity());
}

#[test]
fn selected_only_native_ceiling_admits_the_full_quote_without_discounting_masks_or_kv() {
    let dense = planner().compiler.prepare(shared_document(document("Alice")).unwrap()).unwrap();
    let p = selected(7);
    let prepared = p.compiler.prepare(shared_document(document("Alice")).unwrap()).unwrap();
    let quote = durable_plan_work(&prepared, &p.recipe).unwrap();
    assert_eq!(quote.int4_model_work(), prepared.planned_work());
    assert_eq!(quote.model.projected_logits, 7 * u64::from(budget().max_output_tokens));
    assert_eq!(quote.model.forward_positions, dense.planned_work().forward_positions);
    assert_eq!(quote.model.attention_pairs, dense.planned_work().attention_pairs);
    assert!(quote.model.projections.dot_products < dense.planned_work().projections.dot_products);
    assert!(quote.model.projections.multiply_accumulates < dense.planned_work().projections.multiply_accumulates);
    assert_eq!(quote.mask_node_visits, native(WIDTH).masks.max_visits_per_item);
    assert_eq!(prepared.extraction_plan().max_kv_bytes(), budget().max_kv_bytes);
    let mut cap = native(WIDTH);
    cap.max_model_work = prepared.planned_work();
    cap.masks.max_visits_per_run = cap.masks.max_visits_per_item;
    let bounded = configured(WIDTH, CompileLimits::default(), SourceRuntimeLimits::default(), defaults(), cap)
        .unwrap().with_selected_rows(rows(7)).unwrap();
    bounded.check_job_limits(job_limits(cap)).unwrap();
    assert!(quote.fits(job_limits(cap).max_work));
    assert!(!JobWork::from_int4(dense.planned_work(), quote.mask_node_visits).fits(job_limits(cap).max_work));
}

#[test]
fn compact_encoding_and_selected_rows_compose_without_renewing_preparation_work() {
    use crate::tokenizer::untrusted::{SafeBpeLimits, SafeBpeWorkLimits};
    let policy = UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 8192,
        max_output_tokens: 4096, max_pair_checks: 1 << 22, max_pair_bytes: 1 << 26 });
    let total = Some(SafeBpeWorkLimits { max_input_bytes: 65536, max_output_tokens: 32768,
        max_pair_checks: 1 << 24, max_pair_bytes: 1 << 28 });
    let mut first = selected(7).with_document_encoding(policy.clone(), total).unwrap();
    let second = planner().with_document_encoding(policy.clone(), total).unwrap().with_selected_rows(rows(7)).unwrap();
    freeze(&first).binding.compare(&freeze(&second).binding).unwrap();
    assert_eq!(first.encoding_workspace_bytes().unwrap(), second.encoding_workspace_bytes().unwrap());
    let text = "é 上海 上海 <|im_start|> Alice";
    let prepared = first.compiler.prepare_with_invocation_encoding_work(shared_document(document(text)).unwrap(),
        &mut first.encoding_work, first.recipe.encoding_work_limits, &mut Continue).unwrap();
    assert_eq!(prepared.source().text(), text);
    assert_eq!(prepared.source().document_encoding(), &policy);
    assert_eq!(first.encoding_work.input_bytes(), defaults().schema.len() + text.len());
    assert!(durable_plan_work(&prepared, &first.recipe).is_ok());
    assert!(durable_plan_work(&prepared, &selected(7).recipe).is_err());
    let dense = planner().with_document_encoding(policy, total).unwrap();
    assert!(durable_plan_work(&prepared, &dense.recipe).is_err());
    let spent = first.encoding_work;
    let moved = Processor::new(first, Rc::new(Events::default()));
    assert_eq!(moved.planner.encoding_work, spent);
}

#[test]
fn selected_resume_authenticates_policy_before_repair_and_completed_resume_does_no_work() {
    let root = Root::new(); let inputs = inputs(); let events = Rc::new(Events::default());
    let limits = job_limits(native(WIDTH));
    let mut control = CancelAfterSync(Rc::clone(&events));
    let mut run = JobRunner::create(&root.0, key(), JobId([9; 16]), limits, &inputs,
        Processor::new(selected(7), Rc::clone(&events)), &mut control).unwrap();
    assert_eq!(error(run.step(&mut control)),
        JobRunError::Storage(JobError::Cancelled(DecodeCancellationKind::PollQuota)));
    let spent = run.progress().reserved_work;
    assert_eq!(run.progress().attempts, 1); assert_eq!(run.progress().committed, 0);
    assert_eq!(spent.model.projected_logits, 7 * u64::from(budget().max_output_tokens));
    drop(run);
    let before = events.prepared.borrow().len();
    for wrong in [planner(), selected(8)] {
        assert_eq!(error(JobRunner::resume(&root.0, key(), JobId([9; 16]), limits, &inputs,
            Processor::new(wrong, Rc::clone(&events)), TailPolicy::DiscardUncommitted, &mut Continue)),
            JobRunError::Storage(JobError::Mismatch(MismatchField::Recipe)));
        assert_eq!(events.prepared.borrow().len(), before);
        assert_eq!(*events.executed.borrow(), vec![1]);
        // Mismatched policy cannot discard the uncommitted original result.
        assert_eq!(error(JobRunner::resume(&root.0, key(), JobId([9; 16]), limits, &inputs,
            Processor::new(selected(7), Rc::clone(&events)), TailPolicy::Refuse, &mut Continue)),
            JobRunError::Storage(JobError::UncommittedTail));
    }
    let mut resumed = JobRunner::resume(&root.0, key(), JobId([9; 16]), limits, &inputs,
        Processor::new(selected(7), Rc::clone(&events)), TailPolicy::DiscardUncommitted, &mut Continue).unwrap();
    assert_eq!(resumed.progress().reserved_work, spent);
    assert_eq!(resumed.step(&mut Continue).unwrap().unwrap().reserved_work, spent.checked_add(spent).unwrap());
    let finished = resumed.run(&mut Continue).unwrap();
    assert_eq!(finished.committed, 2); assert_eq!(finished.attempts, 3);
    assert_eq!(*events.executed.borrow(), vec![1, 1, 2]);
    drop(resumed);
    let before = events.prepared.borrow().len();
    let mut complete = JobRunner::resume(&root.0, key(), JobId([9; 16]), limits, &inputs,
        Processor::new(selected(7), Rc::clone(&events)), TailPolicy::Refuse, &mut Continue).unwrap();
    let replay = complete.run(&mut Continue).unwrap();
    assert_eq!(replay.reserved_work, finished.reserved_work); assert_eq!(replay.attempts, finished.attempts);
    assert_eq!(events.prepared.borrow().len(), before); assert_eq!(*events.executed.borrow(), vec![1, 1, 2]);
    complete.verify(&mut Continue).unwrap();
    complete.materialize_ordered(&mut Continue).unwrap();
    assert_eq!(fs::read(root.0.join("materialized.ndjson")).unwrap().iter().filter(|&&b| b == b'\n').count(), 2);
}
