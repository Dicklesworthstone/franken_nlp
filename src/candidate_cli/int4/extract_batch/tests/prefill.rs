//! Pure CLI and ownership admission; no model or input is opened.
use super::*;

#[test]
fn packed_cohorts_keep_each_task_and_work_budget_and_quote_extra_scratch_once() {
    let (base, mut base_args) = arguments(&["--cohort-documents", "3"]);
    base_args.memory_mib = 262_144;
    let ordinary = base.validate(&base_args).unwrap();
    for rows in [1, 2, 64] {
        let cap = rows.to_string();
        let (batch, mut args) = arguments(&["--cohort-documents", "3", "--prefill-rows", &cap]);
        args.memory_mib = 262_144;
        let admitted = batch.validate(&args).unwrap();
        let prefill = admitted.common.prefill.unwrap();
        assert_eq!(prefill.max_batch_rows, rows);
        assert_eq!(admitted.common.native, ordinary.common.native);
        assert_eq!(admitted.native.max_model_work, ordinary.native.max_model_work);
        assert_eq!(admitted.native.masks.per_mask, ordinary.native.masks.per_mask);
        assert_eq!(admitted.native.masks.max_visits_per_item, ordinary.native.masks.max_visits_per_item);
        assert_eq!(admitted.native.masks.max_visits_per_run, ordinary.native.masks.max_visits_per_run);
        assert_eq!(admitted.transport, ordinary.transport);
        assert_eq!(batch.extraction.budget(&args, admitted.common), base.extraction.budget(&base_args, ordinary.common));
        let memory = Int4ExtractionCohortMemoryLimits { max_documents: 3, context_tokens: args.context_tokens,
            preparation_reserve_bytes: admitted.preparation_reserve_bytes, io_reserve_bytes: admitted.io_bytes,
            allocator_reserve_bytes: ALLOCATOR_BYTES };
        let reference = memory.requirements(args.group_width, admitted.native, admitted.transport).unwrap();
        let packed = memory.requirements_packed(args.group_width, admitted.native, admitted.transport, prefill).unwrap();
        assert_eq!(packed.workspace_bytes, reference.workspace_bytes + prefill.validate().unwrap());
        let execution = packed.native.kv_bytes + packed.workspace_bytes + packed.input_bytes + packed.output_reservation_bytes;
        assert_eq!(admitted.common.minimum_memory_bytes, grouped_memory_floor(&args, admitted.common, execution).unwrap());
        args.memory_mib = admitted.common.minimum_memory_bytes.div_ceil(MIB);
        batch.validate(&args).unwrap();
        args.memory_mib -= 1;
        assert_eq!(batch.execute_owned(args, NoRead, NoWrite), Err(CandidateError::Arguments));
    }
}

#[test]
fn packed_caps_and_unsupported_combinations_refuse_before_private_io() {
    assert!(definition().try_get_matches_from(["extract-batch", "--source-dir", "source",
        "--source-manifest", "manifest", "--memory-mib", "8192", "--schema", "schema.json",
        "--prefill-rows", "1"]).is_err());
    for rows in [0, 65, usize::MAX] {
        let (batch, args) = arguments(&["--cohort-documents", "3", "--prefill-rows", &rows.to_string()]);
        assert_eq!(batch.execute_owned(args, NoRead, NoWrite), Err(CandidateError::Arguments));
    }
    for selected in [None, Some(7)] {
        let (mut batch, args) = arguments(&["--cohort-documents", "3", "--prefill-rows", "2"]);
        batch.stream.max_selected_rows = selected;
        batch.stream.cohort_documents = None;
        assert_eq!(batch.execute_owned(args, NoRead, NoWrite), Err(CandidateError::Arguments));
    }
}
