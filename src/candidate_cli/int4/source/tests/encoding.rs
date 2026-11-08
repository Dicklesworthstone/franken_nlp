//! Explicit source-input policy admission and one-request planner wiring.
use super::*;
use crate::tokenizer::untrusted::UntrustedEncoding;

#[test]
fn compact_input_composes_with_selected_rows_and_prefill_for_every_source_task() {
    for name in ["ner", "keyphrases", "summarize", "answer"] {
        let (_, reference, original_args) = arguments(name, &[]);
        let original = reference.validate(&original_args).unwrap();
        let (kind, options, args) = arguments(name, &["--input-encoding", "safe-bpe-v1",
            "--max-tokenizer-pair-checks", "8388608", "--max-tokenizer-pair-bytes", "134217728",
            "--max-selected-rows", "256", "--prefill-rows", "4"]);
        let limits = options.validate_with_prefill(&args).unwrap();
        assert_eq!(kind.name(), name);
        assert_eq!(options.budget(&args, limits), reference.budget(&original_args, original));
        assert_eq!(options.mask_visits(), reference.mask_visits());
        assert_eq!(options.planning(&args).max_context_tokens, original_args.context_tokens);
        assert_eq!(options.selected_rows().unwrap().unwrap().max_rows_per_step, 256);
        assert_eq!(limits.prefill.unwrap().max_batch_rows, 4);
        let UntrustedEncoding::SafeBpeV1(policy) = args.input_encoding
            .policy(args.max_input_bytes, limits.max_prompt_tokens).unwrap() else { panic!("BPE policy required"); };
        assert_eq!(policy.max_input_bytes, args.max_input_bytes);
        assert_eq!(policy.max_output_tokens, args.context_tokens - args.max_new_tokens);
        assert_eq!(policy.max_pair_checks, 8_388_608);
        assert_eq!(policy.max_pair_bytes, 134_217_728);
        assert_eq!(options.validate(&args).unwrap_err(), CandidateError::Arguments);
        assert_eq!(options.validate_full_projection_with_prefill(&args).unwrap_err(), CandidateError::Arguments);
    }
}

#[test]
fn source_and_pinned_snapshot_admit_the_same_compact_input_contract() {
    let digest = "ab".repeat(32);
    for name in ["ner", "keyphrases", "summarize", "answer"] {
        let (_, source, args) = arguments(name, &["--input-encoding", "safe-bpe-v1"]);
        let source_limits = source.validate_with_prefill(&args).unwrap();
        let parsed = crate::candidate_cli::definition().try_get_matches_from([
            "candidate", "source-int4", name, "--snapshot", "local.snapshot",
            "--expected-model-sha256", &digest, "--memory-mib", "8192", "--input-encoding", "safe-bpe-v1",
        ]).unwrap();
        let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&parsed).unwrap()
            else { panic!("INT4 required"); };
        let Int4Task::Source(_, snapshot) = command.task else { panic!("source task required"); };
        let snapshot_limits = snapshot.validate_with_prefill(&command.args).unwrap();
        assert!(snapshot_limits.snapshot_pin.is_some());
        assert_eq!(source, snapshot);
        assert_eq!(args.input_encoding, command.args.input_encoding);
        assert_eq!(source.budget(&args, source_limits), snapshot.budget(&command.args, snapshot_limits));
        // Shared composition must refuse the policy even without optional
        // selected rows or prefill, rather than silently reverting to bytes.
        assert_eq!(source.validate(&args).unwrap_err(), CandidateError::Arguments);
        assert_eq!(source.validate_full_projection_with_prefill(&args).unwrap_err(), CandidateError::Arguments);
    }
}

#[test]
fn compact_input_prices_complete_workspace_and_refuses_invalid_caps_before_private_io() {
    for name in ["ner", "keyphrases", "summarize", "answer"] {
        let (_, byte_options, byte_args) = arguments(name, &["--max-input-bytes", "1048576"]);
        assert!(byte_options.validate(&byte_args).is_ok());
        for flags in [vec!["--max-input-bytes", "1048576"],
            vec!["--max-tokenizer-pair-checks", "0"], vec!["--max-tokenizer-pair-checks", "67108865"],
            vec!["--max-tokenizer-pair-bytes", "0"], vec!["--max-tokenizer-pair-bytes", "536870913"],
            vec!["--max-tokenizer-pair-bytes", "18446744073709551615"]] {
            let mut extra = vec!["--input-encoding", "safe-bpe-v1", "--options", "private-options.json"];
            extra.extend(flags); let mut output = Vec::new();
            assert_eq!(command(name, &extra).execute(&mut NoRead, &mut output), Err(CandidateError::Arguments));
            assert!(output.is_empty());
        }
        let (_, options, mut args) = arguments(name, &["--input-encoding", "safe-bpe-v1",
            "--max-input-bytes", "1048576", "--preparation-mib", "768"]);
        let admitted = options.validate_with_prefill(&args).unwrap();
        assert!(args.input_encoding.workspace_bytes(args.max_input_bytes, admitted.max_prompt_tokens).unwrap() > 256 * MIB);
        args.memory_mib = admitted.minimum_memory_bytes.saturating_sub(1) / MIB;
        assert_eq!(options.validate_with_prefill(&args).unwrap_err(), CandidateError::Arguments);
        args.memory_mib = admitted.minimum_memory_bytes.div_ceil(MIB);
        assert!(options.validate_with_prefill(&args).is_ok());
        assert!(CandidateCommand::from_matches(&matches(name, &["--max-tokenizer-pair-checks", "1"])).is_err());
    }
}

#[test]
fn cli_policy_reaches_source_execution_identity_and_preserves_task_contracts() {
    for name in ["ner", "keyphrases", "summarize", "answer"] {
        let (kind, options, args) = arguments(name, &["--max-new-tokens", "64"]);
        let default = selected::plan(kind, &options, &args);
        let (kind, options, args) = arguments(name, &["--max-new-tokens", "64", "--input-encoding", "safe-bpe-v1"]);
        let compact = selected::plan(kind, &options, &args);
        assert_eq!(compact.task_budget(), default.task_budget());
        assert_eq!(compact.max_result_bytes(), default.max_result_bytes());
        assert_eq!(compact.max_kv_bytes(), default.max_kv_bytes());
        assert_eq!(compact.execution_identity().schema_digest, default.execution_identity().schema_digest);
        assert_ne!(compact.execution_identity().template_digest, default.execution_identity().template_digest);
        assert!(compact.verify_identity(default.execution_identity()).is_err());
        let (kind, options, args) = arguments(name, &["--max-new-tokens", "64", "--input-encoding", "safe-bpe-v1",
            "--max-tokenizer-pair-checks", "8388608", "--max-selected-rows", "7", "--prefill-rows", "4"]);
        let other = options.configure_plan(selected::plan(kind, &options, &args)).unwrap();
        assert_ne!(other.execution_identity().template_digest, compact.execution_identity().template_digest);
        assert_eq!(other.selected_rows().unwrap().max_rows_per_step, 7);
        assert_eq!(other.task_budget(), compact.task_budget());
    }
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn compact_source_requires_owned_runtime_before_reading_input_or_options() {
    for name in ["ner", "keyphrases", "summarize", "answer"] {
        let mut output = Vec::new();
        assert_eq!(command(name, &["--input-encoding", "safe-bpe-v1", "--options", "private-options.json",
            "--max-selected-rows", "7", "--prefill-rows", "4"]).execute(&mut NoRead, &mut output),
            Err(CandidateError::Unavailable));
        assert!(output.is_empty());
    }
}
