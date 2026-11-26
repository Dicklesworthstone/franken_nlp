use super::*;
use crate::candidate_cli::jobs::scored::tests::args;

pub(in crate::candidate_cli) const SETTINGS: &str = r#"{
    "schema_version":1,"max_chunks":4,"max_chunk_bytes":1024,
    "max_tokenizer_calls":8192,"max_retained_result_bytes":8388608,
    "max_result_bytes":4194304,"minimum_coverage_ppm":1000000
}"#;

pub(in crate::candidate_cli) fn document_args(task: &str, extra: &[&str]) -> ScoreJobArgs {
    let mut flags = vec!["--document-config", "document.json", "--max-input-bytes", "65536",
        "--max-result-bytes", "65536", "--preparation-mib", "1024"];
    if task == "judge" { flags.extend_from_slice(&["--faithfulness-policy", "faithfulness.json"]); }
    flags.extend_from_slice(extra);
    let mut result = args(task, &flags);
    result.host.memory_mib = 32768;
    result
}

pub(in crate::candidate_cli) fn lifetime(args: &ScoreJobArgs) -> JobLimits {
    JobLimits { max_items: 8, max_id_bytes: 128, max_input_bytes_per_item: 65536,
        max_snapshot_bytes: 1 << 20, max_result_bytes: 8 << 20, max_spool_bytes: 64 << 20,
        max_materialized_bytes: 64 << 20, max_journal_bytes: 1 << 20, max_attempts: 16,
        max_work: JobWork { model: args.host.work_ceiling(), mask_node_visits: 0 } }
}

#[test]
fn whole_document_settings_preserve_context_output_and_lifetime_work_scopes() {
    for task in ["classify", "sentiment", "judge"] {
        let a = document_args(task, &[]);
        let (_, limits) = a.validate().unwrap();
        let job = lifetime(&a);
        let mapping = load(SETTINGS, &a, limits, job).unwrap();
        let work = mapping.work();
        assert_eq!(work, a.host.work_ceiling());
        let (chunks, output, retained, coverage) = match mapping {
            Mapping::Classify { mapping, .. } => (mapping.chunks, mapping.max_result_bytes,
                mapping.max_retained_result_bytes, mapping.aggregation.minimum_classified_coverage_ppm),
            Mapping::Sentiment { mapping, .. } => (mapping.chunks, mapping.max_result_bytes,
                mapping.max_retained_result_bytes, mapping.aggregation.minimum_estimated_coverage_ppm),
            Mapping::Faithfulness { mapping, .. } => (mapping.chunks, mapping.max_result_bytes,
                mapping.max_retained_result_bytes, mapping.aggregation.minimum_classified_coverage_ppm),
        };
        assert_eq!((chunks.max_chunks, chunks.max_chunk_bytes), (4, 1024));
        assert_eq!(chunks.context_tokens, a.host.context_tokens);
        assert_eq!(chunks.reserved_tokens, a.host.max_candidate_tokens);
        assert_eq!((output, retained, coverage), (4 << 20, 8 << 20, 1_000_000));
        let mut larger: serde_json::Value = serde_json::from_str(SETTINGS).unwrap();
        larger["max_chunks"] = 128.into();
        assert_eq!(load(&larger.to_string(), &a, limits, job).unwrap().work(), work);
    }
}

#[test]
fn config_rejects_duplicate_unknown_unbounded_and_wrong_version_fields() {
    let a = document_args("classify", &[]);
    let (_, limits) = a.validate().unwrap();
    let job = lifetime(&a);
    for invalid in ["", "{}", "null",
        r#"{"schema_version":1,"\u0073chema_version":1}"#] {
        assert!(load(invalid, &a, limits, job).is_err());
    }
    for (field, value) in [
        ("schema_version", 2_u64), ("max_chunks", 0), ("max_chunks", 257),
        ("max_chunk_bytes", 3), ("max_chunk_bytes", 65537),
        ("max_tokenizer_calls", 0), ("max_result_bytes", 65535),
        ("max_result_bytes", 9 << 20), ("max_retained_result_bytes", u64::MAX),
        ("minimum_coverage_ppm", 1_000_001), ("unknown", 1),
    ] {
        let mut wire: serde_json::Value = serde_json::from_str(SETTINGS).unwrap();
        wire[field] = value.into();
        assert!(load(&wire.to_string(), &a, limits, job).is_err(), "{field}={value}");
    }
    assert!(load(&" ".repeat(CONFIG_BYTES + 1), &a, limits, job).is_err());
}

#[test]
fn more_chunks_cannot_increase_any_immutable_native_work_axis() {
    for task in ["classify", "sentiment", "judge"] {
        let a = document_args(task, &[]);
        let (_, limits) = a.validate().unwrap();
        for axis in 0..5 {
            let mut job = lifetime(&a);
            match axis {
                0 => job.max_work.model.forward_positions -= 1,
                1 => job.max_work.model.projected_logits -= 1,
                2 => job.max_work.model.attention_pairs -= 1,
                3 => job.max_work.model.projections.dot_products -= 1,
                _ => job.max_work.model.projections.multiply_accumulates -= 1,
            }
            assert!(matches!(load(SETTINGS, &a, limits, job),
                Err(Failure { code: "score_document_job_lifetime_work", .. })), "{task}/{axis}");
        }
    }
}

#[test]
fn complete_sentiment_expands_semantic_kv_authority_without_changing_physical_context() {
    let document = document_args("sentiment", &[]);
    let (common, limits) = document.validate().unwrap();
    let task = document.task_budget(limits).unwrap();
    assert_eq!(task.max_kv_bytes, limits.kv_bytes * 4);
    assert_eq!(common.context_tokens, document.host.context_tokens);
    let single = args("sentiment", &[]);
    let (_, ordinary) = single.validate().unwrap();
    assert_eq!(single.task_budget(ordinary).unwrap().max_kv_bytes, ordinary.kv_bytes);
}

#[test]
fn document_encoding_has_one_cumulative_allowance_independent_of_native_context() {
    let a = document_args("classify", &["--input-encoding", "safe-bpe-v1",
        "--max-tokenizer-input-bytes", "131072", "--max-tokenizer-output-tokens", "65536"]);
    let (common, limits) = a.validate().unwrap();
    let UntrustedEncoding::SafeBpeV1(policy) = a.encoding_policy().unwrap()
        else { panic!("compact document policy") };
    assert_eq!((policy.max_input_bytes, policy.max_output_tokens), (131072, 65536));
    assert!(policy.max_output_tokens > common.context_tokens);
    assert_eq!(policy.max_pair_checks, crate::tokenizer::untrusted::SafeBpeLimits::MAX_PAIR_CHECKS);
    assert!(a.encoding_work_limits().unwrap().is_some());
    load(SETTINGS, &a, limits, lifetime(&a)).unwrap();
}

#[test]
fn invalid_document_modes_and_insufficient_preparation_refuse_before_private_io() {
    for task in ["classify", "sentiment"] {
        let mut a = document_args(task, &[]);
        a.document_config = Some("-".into());
        assert!(a.validate().is_err());
    }
    let mut judge = document_args("judge", &[]);
    judge.faithfulness_policy = None;
    assert!(judge.validate().is_err());
    let a = document_args("classify", &[]);
    let (_, mut limits) = a.validate().unwrap();
    limits.preparation_bytes = 1;
    assert!(matches!(load(SETTINGS, &a, limits, lifetime(&a)),
        Err(Failure { code: "score_document_job_preparation", .. })));
}

#[test]
fn impossible_classification_encoding_allowances_fail_before_private_io() {
    for option in ["--max-total-tokenizer-input-bytes", "--max-total-tokenizer-output-tokens",
        "--max-total-tokenizer-pair-checks", "--max-total-tokenizer-pair-bytes"] {
        let flags = ["--input-encoding", "safe-bpe-v1", option, "1"];
        let a = document_args("classify", &flags);
        assert_eq!(a.validate().err().unwrap().code, "score_document_job_tokenizer_work");
        assert!(document_args("sentiment", &flags).encoding_work_limits().is_ok());
    }
}

#[test]
fn complete_faithfulness_requires_one_fixed_local_policy_and_rejects_other_modes() {
    let mut judge = document_args("judge", &[]);
    judge.validate().unwrap();
    for path in ["", "-"] {
        judge.faithfulness_policy = Some(path.into());
        assert_eq!(judge.validate().err().unwrap().code, "score_document_job_faithfulness_policy");
    }
    judge.faithfulness_policy = Some("policy.json".into());
    judge.defaults = Some("defaults.json".into());
    assert_eq!(judge.validate().err().unwrap().code, "score_document_job_faithfulness_policy");
    judge.defaults = None;
    judge.document_config = None;
    assert_eq!(judge.validate().err().unwrap().code, "score_document_job_faithfulness_policy");
    for task in ["classify", "sentiment"] {
        let mut ordinary = document_args(task, &[]);
        ordinary.faithfulness_policy = Some("policy.json".into());
        assert_eq!(ordinary.validate().err().unwrap().code, "score_document_job_faithfulness_policy");
    }
}

#[test]
fn complete_faithfulness_prices_all_evidence_heads_but_reuses_one_physical_context() {
    let document = document_args("judge", &[]);
    let (common, limits) = document.validate().unwrap();
    let task = document.task_budget(limits).unwrap();
    assert_eq!(task.max_kv_bytes, limits.kv_bytes
        * crate::tasks::judge::quantized::faithfulness_map::MAX_FAITHFULNESS_MAP_HEADS as u64);
    assert_eq!(common.context_tokens, document.host.context_tokens);
    let ordinary = args("judge", &[]);
    let (_, native) = ordinary.validate().unwrap();
    assert_eq!(ordinary.task_budget(native).unwrap().max_kv_bytes, native.kv_bytes);
    let Mapping::Faithfulness { planning, mapping } =
        load(SETTINGS, &document, limits, lifetime(&document)).unwrap()
        else { panic!("whole-source judge route") };
    assert_eq!(planning.max_output_bytes, document.host.max_result_bytes as u64);
    assert_eq!(mapping.max_result_bytes, 4 << 20);
    assert_eq!(mapping.max_model_work, document.host.work_ceiling());
}

#[test]
fn complete_faithfulness_compact_work_never_renews_at_a_chunk_or_evidence_head() {
    let flags = ["--input-encoding", "safe-bpe-v1", "--max-tokenizer-input-bytes", "131072",
        "--max-tokenizer-output-tokens", "65536"];
    let a = document_args("judge", &flags);
    let (common, limits) = a.validate().unwrap();
    let UntrustedEncoding::SafeBpeV1(policy) = a.encoding_policy().unwrap()
        else { panic!("compact whole-source policy") };
    assert_eq!((policy.max_input_bytes, policy.max_output_tokens), (131072, 65536));
    assert!(policy.max_output_tokens > common.context_tokens);
    load(SETTINGS, &a, limits, lifetime(&a)).unwrap();
    for option in ["--max-total-tokenizer-input-bytes", "--max-total-tokenizer-output-tokens",
        "--max-total-tokenizer-pair-checks", "--max-total-tokenizer-pair-bytes"] {
        let a = document_args("judge", &["--input-encoding", "safe-bpe-v1", option, "1"]);
        assert_eq!(a.validate().err().unwrap().code, "score_document_job_tokenizer_work");
    }
}
