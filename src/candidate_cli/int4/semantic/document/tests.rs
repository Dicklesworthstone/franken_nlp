//! UNRUN parser/admission regressions; no model quality or native parity claim.
use super::*;
use crate::{native_engine::kv::KV_BYTES_PER_TOKEN,
    tokenizer::untrusted::{SafeBpeLimits, UntrustedEncoding}};

pub(super) fn argv(extra: &[&str]) -> Vec<String> {
    ["candidate", "source-int4", "extract-semantic-map", "--source-dir", "explicit-source",
        "--source-manifest", "explicit-manifest.json", "--schema", "schema.json",
        "--semantic-options", "semantic.json", "--memory-mib", "32768",
        "--max-input-bytes", "4096", "--max-chunks", "4", "--max-chunk-bytes", "1024",
        "--max-result-bytes", "65536", "--max-map-result-bytes", "262144",
        "--max-live-value-bytes", "1048576", "--max-total-value-bytes", "1048576",
        "--max-object-json-bytes", "65536", "--max-object-result-bytes", "262144",
        "--max-grammar-states", "512", "--max-new-tokens", "64",
        "--review-max-chunks", "4", "--review-max-chunk-bytes", "2048",
        "--review-max-result-bytes", "262144", "--review-max-retained-result-bytes", "524288"]
        .into_iter().chain(extra.iter().copied()).map(str::to_owned).collect()
}
fn command(extra: &[&str]) -> Int4Command {
    let matches = crate::candidate_cli::definition().try_get_matches_from(argv(extra)).unwrap();
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("wrong candidate route"); };
    command
}
pub(super) fn fixture() -> (SemanticObjectArgs, Int4Args,
    extract_map::object::ObjectMapLimits, Int4SemanticOptions) {
    let command = command(&[]);
    let Int4Task::SemanticObjectMap(options) = command.task else { panic!("wrong semantic document route"); };
    let args = command.args;
    let limits = options.validate(&args).unwrap();
    let (_, _, _, mut semantic) = super::super::tests::fixture();
    semantic.specification.judge_budget.max_kv_bytes = limits.map.common.native.kv_bytes;
    (options, args, limits, semantic)
}
struct NoRead;
impl Read for NoRead {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused configuration touched private input"); }
}

#[test]
fn dedicated_route_preserves_original_map_and_complete_source_review_bounds() {
    let (command, args, limits, options) = fixture();
    let review = command.review_limits(&args, limits.map.common, &options).unwrap();
    assert_eq!(limits.mapping().mapping.mapping.chunks.max_input_bytes, 4096);
    assert_eq!(limits.mapping().mapping.mapping.chunks.max_chunks, 4);
    assert_eq!(limits.object.max_json_bytes, 65536);
    assert_eq!(review.chunks.max_input_bytes, 4096 + options.limits.max_claim_bytes);
    assert_eq!(review.chunks.max_chunks, 4);
    assert_eq!(review.max_model_work, options.max_model_work);
    assert_eq!(review.aggregation.minimum_classified_coverage_ppm, 1_000_000);
    assert_eq!(limits.map.common.compute.threads, 1);
    assert!(format!("{command:?}").find("semantic.json").is_none());
    let matches = crate::candidate_cli::definition().try_get_matches_from([
        "candidate", "source-int4", "extract-object-map", "--source-dir", "source",
        "--source-manifest", "manifest", "--memory-mib", "8192", "--schema", "schema",
        "--semantic-options", "semantic.json",
    ]);
    assert!(matches.is_err(), "ordinary object extraction must not implicitly review");
}

#[test]
fn source_and_snapshot_use_the_same_semantic_route_and_requested_team() {
    for threads in ["1", "2", "4"] {
        let mut parsed = command(&["--compute-threads", threads]);
        let Int4Task::SemanticObjectMap(command) = &parsed.task else { panic!("wrong route"); };
        let source = command.validate(&parsed.args).unwrap();
        assert_eq!(source.map.common.compute.threads, threads.parse::<usize>().unwrap());
        parsed.args.source_dir = None;
        parsed.args.source_manifest = None;
        parsed.args.snapshot = Some(PathBuf::from("private.snapshot"));
        parsed.args.expected_model_sha256 = Some("ab".repeat(32));
        let snapshot = command.validate(&parsed.args).unwrap();
        assert!(snapshot.map.common.snapshot_pin.is_some());
        assert_eq!(snapshot.map.common.native, source.map.common.native);
        assert_eq!(snapshot.map.common.compute, source.map.common.compute);
    }
}

#[test]
fn unsupported_schedules_and_invalid_review_geometry_refuse_before_input() {
    for extra in [vec!["--max-selected-rows", "16"], vec!["--cohort-chunks", "2"],
        vec!["--compute-threads", "0"], vec!["--compute-threads", "65"],
        vec!["--review-max-tokenizer-calls", "0"],
        vec!["--review-minimum-classified-coverage-ppm", "1000001"]] {
        let matches = crate::candidate_cli::definition().try_get_matches_from(argv(&extra)).unwrap();
        if let Ok(CandidateCommand::SourceInt4(command)) = CandidateCommand::from_matches(&matches) {
            let Int4Task::SemanticObjectMap(options) = command.task else { panic!("wrong route"); };
            let mut output = Vec::new();
            assert!(options.execute(command.args, &mut NoRead, &mut output).is_err());
            assert!(output.is_empty());
        }
    }
    for extra in [vec!["--prefill-rows", "2"], vec!["--temperature", "0.7"], vec!["--thinking", "on"]] {
        match crate::candidate_cli::definition().try_get_matches_from(argv(&extra)) {
            Err(_) => (),
            Ok(matches) => assert!(CandidateCommand::from_matches(&matches).is_err()),
        }
    }
}

#[test]
fn compact_policy_is_document_wide_and_never_reuses_text_request_limits() {
    let parsed = command(&["--input-encoding", "safe-bpe-v1", "--max-tokenizer-input-bytes", "65536",
        "--max-tokenizer-output-tokens", "32768", "--max-tokenizer-pair-checks", "100000",
        "--max-tokenizer-pair-bytes", "1000000"]);
    assert!(parsed.args.input_encoding.is_default());
    let Int4Task::SemanticObjectMap(command) = parsed.task else { panic!("wrong route"); };
    assert_eq!(command.object.map.encoding.policy().unwrap(), UntrustedEncoding::SafeBpeV1(SafeBpeLimits {
        max_input_bytes: 65536, max_output_tokens: 32768, max_pair_checks: 100000, max_pair_bytes: 1000000,
    }));
    assert!(command.validate(&parsed.args).is_ok());
    let matches = crate::candidate_cli::definition().try_get_matches_from(argv(&[
        "--max-tokenizer-input-bytes", "65536"])).unwrap();
    assert!(CandidateCommand::from_matches(&matches).is_err());
}

#[test]
fn all_original_claims_share_explicit_semantic_and_full_cache_bounds() {
    let (command, args, limits, options) = fixture();
    let mut short_cache = options.clone();
    short_cache.specification.judge_budget.max_kv_bytes =
        limits.map.common.native.kv_bytes - KV_BYTES_PER_TOKEN as u64;
    assert!(command.review_limits(&args, limits.map.common, &short_cache).is_err());
    let mut oversized_prompt = options.clone();
    oversized_prompt.specification.judge_budget.max_input_tokens = args.context_tokens as u32;
    assert!(command.review_limits(&args, limits.map.common, &oversized_prompt).is_err());
    let mut no_opt_in = options;
    no_opt_in.specification.experimental_opt_in = false;
    assert!(command.review_limits(&args, limits.map.common, &no_opt_in).is_err());
}

#[test]
fn borrowed_dispatch_cannot_bypass_the_typed_failure_boundary() {
    let command = command(&[]);
    let mut output = Vec::new();
    assert_eq!(command.execute(&mut NoRead, &mut output), Err(CandidateError::Arguments));
    assert!(output.is_empty());
}
