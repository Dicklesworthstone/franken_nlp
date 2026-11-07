//! Explicit serial capability and pure admission; no native execution.
use super::*;
use crate::native_engine::{constrained_int4::sparse, lmhead::NANBEIGE_VOCAB_SIZE};

const COMMANDS: [&str; 4] = ["ner-batch", "keyphrases-batch", "summarize-batch", "answer-batch"];

#[test]
fn all_four_selected_batches_preserve_source_authority_and_exact_whole_run_work() {
    for name in COMMANDS {
        let (dense, args) = arguments(name, &["--max-requests", "8", "--group-width", "127"]);
        let reference = dense.validate(&args).unwrap();
        assert!(reference.selected_rows.is_none());
        for cap in [1, 7, NANBEIGE_VOCAB_SIZE] {
            let count = cap.to_string();
            let (selected, args) = arguments(name, &["--max-requests", "8", "--group-width", "127",
                "--max-selected-rows", &count]);
            let admitted = selected.validate(&args).unwrap();
            let policy = admitted.selected_rows.unwrap();
            assert_eq!(policy.max_rows_per_step, cap);
            assert_eq!(admitted.cohort_documents, None);
            assert_eq!(selected.source, dense.source);
            assert_eq!(selected.source.budget(&args, admitted.common), dense.source.budget(&args, reference.common));
            assert_eq!(admitted.common.minimum_memory_bytes, reference.common.minimum_memory_bytes);
            assert_eq!(admitted.common.native, reference.common.native);
            assert_eq!(admitted.native.max_model_work,
                scale_work(sparse::planned_work(127, admitted.common.max_prompt_tokens, args.max_new_tokens, policy).unwrap(), 8).unwrap());
            assert_eq!(admitted.native.max_model_work.forward_positions, reference.native.max_model_work.forward_positions);
            assert_eq!(admitted.native.max_model_work.attention_pairs, reference.native.max_model_work.attention_pairs);
            assert_eq!(admitted.native.max_model_work.projected_logits, cap as u64 * args.max_new_tokens as u64 * 8);
            assert_eq!(admitted.native.masks.max_visits_per_item, reference.native.masks.max_visits_per_item);
            assert_eq!(admitted.native.masks.max_visits_per_run, reference.native.masks.max_visits_per_run);
            assert_eq!(admitted.native.masks.per_mask.max_trie_node_visits,
                reference.native.masks.per_mask.max_trie_node_visits);
            let mut transport = reference.transport;
            transport.max_work.projected_logits = admitted.native.max_model_work.projected_logits;
            assert_eq!(admitted.transport, transport);
            if cap == NANBEIGE_VOCAB_SIZE { assert_eq!(admitted.native.max_model_work, reference.native.max_model_work); }
            else { assert!(admitted.native.max_model_work.projections.dot_products < reference.native.max_model_work.projections.dot_products); }
        }
    }
}

#[test]
fn invalid_caps_and_cohort_composition_refuse_before_private_io() {
    for name in COMMANDS {
        for cap in [0, NANBEIGE_VOCAB_SIZE + 1, usize::MAX] {
            let count = cap.to_string(); let (batch, args) = arguments(name, &["--max-selected-rows", &count]);
            assert_eq!(batch.execute_owned(args, NoRead, NoWrite), Err(CandidateError::Arguments));
        }
        let definition = definitions().into_iter().find(|c| c.get_name() == name).unwrap();
        assert!(definition.try_get_matches_from([name, "--source-dir", "explicit-source",
            "--source-manifest", "explicit-manifest", "--max-selected-rows", "7",
            "--cohort-documents", "1"]).is_err());
        let (mut batch, args) = arguments(name, &["--max-selected-rows", "7"]);
        batch.stream.cohort_documents = Some(1);
        assert_eq!(batch.execute_owned(args, NoRead, NoWrite), Err(CandidateError::Arguments));
    }
}

#[test]
fn selected_source_and_pinned_snapshot_streams_quote_the_same_task_and_work() {
    let digest = "ab".repeat(32);
    for name in COMMANDS {
        let (source, args) = arguments(name, &["--max-selected-rows", "7"]);
        let source_limits = source.validate(&args).unwrap();
        let matches = crate::candidate_cli::definition().try_get_matches_from([
            "candidate", "source-int4", name, "--snapshot", "local.snapshot",
            "--expected-model-sha256", &digest, "--memory-mib", "8192", "--max-selected-rows", "7",
        ]).unwrap();
        let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
            else { panic!("wrong source snapshot route") };
        let Int4Task::SourceBatch(snapshot) = command.task else { panic!("wrong source task") };
        let snapshot_limits = snapshot.validate(&command.args).unwrap();
        assert_eq!(source, snapshot);
        assert!(snapshot_limits.common.snapshot_pin.is_some());
        assert_eq!(source_limits.selected_rows, snapshot_limits.selected_rows);
        assert_eq!(source_limits.native.max_model_work, snapshot_limits.native.max_model_work);
        assert_eq!(source_limits.transport, snapshot_limits.transport);
    }
}

#[test]
fn fixed_stream_selection_is_never_an_accepted_per_item_argument() {
    for name in COMMANDS {
        let (batch, args) = arguments(name, &["--max-selected-rows", "7"]);
        let limits = batch.validate(&args).unwrap(); let budget = batch.source.budget(&args, limits.common);
        let defaults = if batch.kind == Kind::Answer {
            SourceBatchArgs::Answer { passages: vec![AnswerPassage { id: "p".to_owned(), text: "PRIVATE_SOURCE".to_owned() }],
                options: AnswerOptions::default(), budget }
        } else { batch.load_defaults(None, budget, args.max_input_bytes).unwrap().unwrap() };
        let json = serde_json::to_string(&defaults).unwrap();
        batch.load_defaults(Some(&json), budget, args.max_input_bytes).unwrap();
        for key in ["max_selected_rows", "selected_rows", "execution_identity"] {
            let mut wire = serde_json::to_value(&defaults).unwrap();
            wire.as_object_mut().unwrap().insert(key.to_owned(), serde_json::json!(7));
            let json = serde_json::to_string(&wire).unwrap();
            assert!(matches!(batch.load_defaults(Some(&json), budget, args.max_input_bytes), Err(CandidateError::Input)));
        }
    }
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn selected_source_streams_still_require_the_owned_runtime_before_input() {
    for name in COMMANDS {
        let (batch, args) = arguments(name, &["--max-selected-rows", "7"]);
        assert_eq!(batch.execute_owned(args, NoRead, NoWrite), Err(CandidateError::Unavailable));
    }
}
