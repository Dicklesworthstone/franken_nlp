//! Native work admission and real rules-only publication fixtures.
//! Rules-only fixtures exercise serialization, never claim neural execution.
use super::*;
use crate::tasks::redact::{actions::RedactionResult, pseudonym::PseudonymKey, redact_rules};
use super::super::tests::{command, options};

#[test]
fn verification_work_sums_independent_contexts_and_preserves_actual_group_cost() {
    let mut previous = None;
    for width in [3, 128, 10752] {
        let first = constrained_int4::planned_work(width, 7, 8).unwrap();
        let second = constrained_int4::planned_work(width, 64 - 8, 8).unwrap();
        let whole = work_ceiling(first, true, width, 64, 8).unwrap();
        assert_eq!(whole, first.checked_add(second).unwrap());
        assert_eq!(work_ceiling(first, false, width, 64, 8).unwrap(), first);
        assert_eq!(second.forward_positions, 63);
        assert_eq!(whole.forward_positions, first.forward_positions + 63);
        assert_eq!(whole.attention_pairs, first.attention_pairs + second.attention_pairs);
        let concatenated = Int4Work::for_sequence(width, 0, whole.forward_positions as usize, 0).unwrap();
        assert!(whole.attention_pairs < concatenated.attention_pairs);
        assert!(within(first, whole));
        assert!(!within(whole, first));
        if let Some(prior) = previous {
            let prior: Int4Work = prior;
            assert_eq!(whole.forward_positions, prior.forward_positions);
            assert_eq!(whole.projected_logits, prior.projected_logits);
            assert_eq!(whole.attention_pairs, prior.attention_pairs);
            assert_eq!(whole.projections.multiply_accumulates, prior.projections.multiply_accumulates);
            assert!(whole.projections.dot_products < prior.projections.dot_products);
        }
        previous = Some(whole);
    }
}

#[test]
fn every_whole_operation_axis_is_enforced_without_wrapping() {
    let first = constrained_int4::planned_work(128, 7, 8).unwrap();
    let whole = work_ceiling(first, true, 128, 64, 8).unwrap();
    for axis in 0..5 {
        let mut short = whole;
        match axis {
            0 => short.forward_positions -= 1,
            1 => short.projected_logits -= 1,
            2 => short.attention_pairs -= 1,
            3 => short.projections.dot_products -= 1,
            _ => short.projections.multiply_accumulates -= 1,
        }
        assert!(!within(whole, short), "axis {axis} must refuse");
        let mut overflow = first;
        match axis {
            0 => overflow.forward_positions = u64::MAX,
            1 => overflow.projected_logits = u64::MAX,
            2 => overflow.attention_pairs = u64::MAX,
            3 => overflow.projections.dot_products = u64::MAX,
            _ => overflow.projections.multiply_accumulates = u64::MAX,
        }
        assert_eq!(work_ceiling(overflow, true, 128, 64, 8), Err(CandidateError::Planning));
    }
    for (context, output) in [(0, 0), (64, 0), (64, 64), (64, 65), (14, 8)] {
        assert_eq!(work_ceiling(first, true, 128, context, output), Err(CandidateError::Planning));
    }
    assert_eq!(work_ceiling(first, true, 0, 64, 8), Err(CandidateError::Planning));
    assert_eq!(work_ceiling(Int4Work::default(), false, 128, 64, 8), Err(CandidateError::Planning));
}

#[test]
fn typed_cancellation_and_invalid_native_results_never_become_redacted_output() {
    for cause in [DecodeCancellationKind::Deadline, DecodeCancellationKind::Timeout] {
        assert_eq!(source_error(Int4SourceError::Cancelled(cause)), CandidateError::Timeout);
        assert_eq!(planning_error(Int4RedactionError::Source(Int4SourceError::Cancelled(cause))),
            CandidateError::Timeout);
        assert_eq!(execution_error(HostedError::Int4(Int4HostError::Redaction(
            Int4RedactionError::Cancelled(cause)))), CandidateError::Timeout);
    }
    assert_eq!(planning_error(Int4RedactionError::Cancelled(DecodeCancellationKind::PollQuota)), CandidateError::Execution);
    assert_eq!(planning_error(Int4RedactionError::WorkBudget), CandidateError::Planning);
    assert_eq!(planning_error(Int4RedactionError::Identity), CandidateError::Identity);
    assert_eq!(execution_error(HostedError::Int4(Int4HostError::Redaction(Int4RedactionError::InvalidResult))),
        CandidateError::Execution);
    assert_eq!(execution_error(HostedError::ModelIdentity), CandidateError::Identity);
    assert_eq!(execution_error(HostedError::ResourceDomain), CandidateError::Identity);
    let mut expired = DeadlineControl { started: Instant::now(), deadline: Duration::ZERO };
    assert_eq!(expired.checkpoint(0), Some(DecodeCancellationKind::Deadline));
}

#[test]
fn complete_publication_preserves_only_edited_text_explicit_maps_and_public_policy() {
    #[derive(Serialize)]
    struct RulesOnlyFixture<'a> { fixture_kind: &'static str, result: &'a RedactionResult }
    let source = "é a@example.org 上海";
    let key = PseudonymKey::from_bytes(b"0123456789abcdef0123456789abcdef", "public-key-id").unwrap();
    let context = Pseudonyms::full256(&key, "private-publication-namespace", None).unwrap();
    for action in ["mask", "placeholder", "pseudonymize"] {
        let selected = if action == "pseudonymize" {
            command(&["private-document.txt", "--rules", "email", "--action", action, "--include-map",
                "--key-stdin", "--key-id", "public-key-id", "--namespace", "private-publication-namespace"])
        } else { command(&["--rules", "email", "--action", action, "--include-map"]) };
        let options = options(&selected);
        let limits = options.validate(&selected.args).unwrap();
        let request = options.request(&selected.args, limits);
        let result = redact_rules(source, &request, (action == "pseudonymize").then_some(&context)).unwrap();
        assert_eq!(result.verification(), VerificationStatus::CleanDeclaredUnion);
        assert!(result.model_types().is_empty(), "this fixture never simulates a native NER pass");
        assert!(result.text().starts_with("é "));
        assert!(result.text().ends_with(" 上海"));
        assert!(!result.text().contains("a@example.org"));
        let edit = &result.edits()[0];
        assert_eq!(edit.original.span.byte_start, 3);
        assert_eq!(edit.original.span.scalar_start, 2);
        assert_eq!(&source[edit.original.span.byte_start..edit.original.span.byte_end], "a@example.org");
        let output = RulesOnlyFixture { fixture_kind: "rules-only-publication", result: &result };
        let provenance = Provenance { protocol: "fnlp-candidate-int4-redact-v1", schema_version: 1,
            scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "redact",
            model_id: "Nanbeige4.2-3B", source_revision: "fixture", source_root_sha256: "fixture-source",
            model_content_sha256: "fixture-model", candidate_format: "fixture-format", quant_recipe: "fixture:group=128",
            group_width: 128, integer_kernel: "s_scalar_i32", snapshot: None };
        let response = Response { provenance: &provenance, output: &output };
        let mut bytes = Vec::new(); publish(&response, 8192, &mut bytes).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["output"]["fixture_kind"], "rules-only-publication");
        assert_eq!(value["output"]["result"]["text"], result.text());
        assert_eq!(value["output"]["result"]["edits"][0]["original"]["span"]["byte_start"], 3);
        assert_eq!(value["protocol"], "fnlp-candidate-int4-redact-v1");
        if action == "pseudonymize" {
            assert_eq!(value["output"]["result"]["pseudonym_identity"]["encoding"], "full256");
        } else { assert!(value["output"]["result"]["pseudonym_identity"].is_null()); }
        for absent in ["document", "source", "prompt", "prompt_digest", "ner_transcript", "key", "namespace",
            "source_dir", "source_manifest", "snapshot_path"] { assert!(value.get(absent).is_none()); }
        let serialized = std::str::from_utf8(&bytes).unwrap();
        for private in ["a@example.org", "private-document.txt", "private-publication-namespace",
            "0123456789abcdef0123456789abcdef"] { assert!(!serialized.contains(private)); }
        let mut refused = Vec::new();
        assert_eq!(publish(&response, 1, &mut refused), Err(CandidateError::Output));
        assert!(refused.is_empty(), "size refusal must publish no prefix");
    }
}

#[test]
fn writer_flush_failure_is_reported_after_complete_redaction_staging() {
    struct FlushFailure(Vec<u8>);
    impl Write for FlushFailure {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> { self.0.extend_from_slice(bytes); Ok(bytes.len()) }
        fn flush(&mut self) -> io::Result<()> { Err(io::Error::other("fixture flush failure")) }
    }
    let complete = serde_json::json!({"fixture_only":true,"text":"edited content","verification":"clean_declared_union"});
    let mut writer = FlushFailure(Vec::new());
    assert_eq!(publish(&complete, 4096, &mut writer), Err(CandidateError::Output));
    assert_eq!(serde_json::from_slice::<serde_json::Value>(&writer.0).unwrap(), complete);
}
