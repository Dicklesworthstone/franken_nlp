use super::*;
use crate::tokenizer::untrusted::{SafeBpeLimits, UntrustedEncoding};

#[test]
fn compact_extraction_prices_schema_and_document_as_one_encoding_request() {
    let (plain, byte_args) = arguments(&[]);
    let ordinary = plain.validate_with_prefill(&byte_args).unwrap();
    let same = plain.validate_one_off_with_prefill(&byte_args).unwrap();
    assert_eq!(ordinary.minimum_memory_bytes, same.minimum_memory_bytes);
    assert_eq!(plain.encoding_policy(&byte_args, ordinary.max_prompt_tokens).unwrap(),
        UntrustedEncoding::ByteFallback);
    assert_eq!(plain.encoding_workspace_extra(&byte_args, ordinary.max_prompt_tokens).unwrap(), 0);

    let (extract, args) = arguments(&["--input-encoding", "safe-bpe-v1"]);
    let limits = extract.validate_one_off_with_prefill(&args).unwrap();
    let UntrustedEncoding::SafeBpeV1(policy) = extract.encoding_policy(&args, limits.max_prompt_tokens).unwrap()
        else { panic!("compact extraction policy was lost"); };
    assert_eq!(policy.max_input_bytes, args.max_input_bytes + crate::candidate_cli::extract::SCHEMA_BYTES);
    assert_eq!(policy.max_output_tokens, limits.max_prompt_tokens);
    let document_quote = args.input_encoding.workspace_bytes(args.max_input_bytes, limits.max_prompt_tokens).unwrap();
    assert_eq!(document_quote + extract.encoding_workspace_extra(&args, limits.max_prompt_tokens).unwrap(),
        policy.requirements().unwrap().workspace_bytes);
    // The shared compound-workflow entrypoint must remain byte-only.
    assert!(extract.validate_with_prefill(&args).is_err());
    assert!(extract.validate_with_projection(&args).is_err());
    assert!(extract.validate(&args).is_err());
}

#[test]
fn compact_extraction_combines_with_source_membership_selected_rows_and_prefill() {
    for flags in [
        &["--input-encoding", "safe-bpe-v1"][..],
        &["--input-encoding", "safe-bpe-v1", "--source-membership"][..],
        &["--input-encoding", "safe-bpe-v1", "--max-selected-rows", "16"][..],
        &["--input-encoding", "safe-bpe-v1", "--source-membership", "--max-selected-rows", "16",
            "--prefill-rows", "4"][..],
    ] {
        let (extract, args) = arguments(flags);
        let limits = extract.validate_one_off_with_prefill(&args).unwrap();
        assert!(!args.input_encoding.is_default());
        assert_eq!(limits.max_prompt_tokens + args.max_new_tokens - 1, args.context_tokens);
    }
}

#[test]
fn extraction_aggregate_cap_and_large_workspace_are_explicit_before_io() {
    let (extract, args) = arguments(&["--input-encoding", "safe-bpe-v1", "--max-input-bytes", "1048576",
        "--preparation-mib", "1024"]);
    let limits = extract.validate_one_off_with_prefill(&args).unwrap();
    let UntrustedEncoding::SafeBpeV1(policy) = extract.encoding_policy(&args, limits.max_prompt_tokens).unwrap()
        else { panic!("missing aggregate policy"); };
    assert_eq!(policy.max_input_bytes, SafeBpeLimits::MAX_INPUT_BYTES);
    for flags in [
        &["--input-encoding", "safe-bpe-v1", "--max-input-bytes", "1048576"][..],
        &["--input-encoding", "safe-bpe-v1", "--max-tokenizer-pair-checks", "0"][..],
        &["--input-encoding", "safe-bpe-v1", "--max-tokenizer-pair-bytes", "0"][..],
        &["--input-encoding", "safe-bpe-v1", "--max-tokenizer-pair-bytes", "18446744073709551615"][..],
    ] {
        let mut output = Vec::new();
        assert_eq!(command(flags).execute(&mut NoRead, &mut output), Err(CandidateError::Arguments));
        assert!(output.is_empty());
    }
}

#[test]
fn encoding_is_not_exposed_on_byte_capacity_document_and_durable_workflows() {
    for route in [vec!["extract-batch"], vec!["extract-map"], vec!["extract-corpus"],
        vec!["extract-semantic"], vec!["extract-job", "start"], vec!["extract-document-job", "start"]] {
        let mut argv = vec!["candidate", "source-int4"];
        argv.extend(route);
        argv.extend(["--source-dir", "explicit-source", "--source-manifest", "explicit-manifest.json",
            "--memory-mib", "8192", "--schema", "schema.json", "--input-encoding", "safe-bpe-v1"]);
        assert!(crate::candidate_cli::definition().try_get_matches_from(argv).is_err());
    }
}
