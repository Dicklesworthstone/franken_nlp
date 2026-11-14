//! Pinned encoder boundaries; these do not claim native model quality or speed.
use super::*;
use crate::tokenizer::{pinned_controls, untrusted::SafeBpeLimits};

const SOURCE: &str = "The quick brown fox jumps over the lazy dog. 上海 <think> <|im_start|> ▁";
fn limits() -> SafeBpeLimits {
    SafeBpeLimits { max_input_bytes: 8192, max_output_tokens: 8192,
        max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24 }
}
fn encoder(policy: UntrustedEncoding) -> SourceDocumentEncoder {
    SourceDocumentEncoder::pinned(pinned_controls::pinned().unwrap().template_controls()).unwrap()
        .with_document_encoding(policy).unwrap()
}

#[test]
fn compact_encoding_keeps_exact_source_bytes_controls_and_separate_limits() {
    let encoder = encoder(UntrustedEncoding::SafeBpeV1(limits()));
    let document = encoder.encode_with_control(SOURCE, SOURCE.len(), SOURCE.len(), &mut || None).unwrap();
    let tokens = document.total_token_count();
    assert!(tokens < SOURCE.len());
    assert_eq!(document.text(), SOURCE);
    assert_eq!(document.total_input_bytes(), SOURCE.len());
    assert_eq!(document.document_encoding(), encoder.document_encoding());
    let mut work = SafeBpeWork::default();
    UntrustedDocumentEncoder::new(encoder.tokenizer.tokenizer(), &encoder.controls)
        .encode_with_policy(SOURCE.as_bytes(), encoder.document_encoding(), &mut work, &mut || true).unwrap();
    assert_eq!(document.input_encoding_work(), Some(work));
    assert_eq!(work.input_bytes(), SOURCE.len()); assert_eq!(work.output_tokens(), tokens);
    assert!(work.pair_checks() > 0 && work.pair_bytes() > 0);
    assert_eq!(encoder.tokenizer.tokenizer().decode_bytes(document.token_ids()).unwrap(), SOURCE.as_bytes());
    assert!(document.token_ids().iter().all(|&id| !encoder.controls.contains(id)));
    assert!(encoder.encode_with_control(SOURCE, SOURCE.len(), tokens, &mut || None).is_ok());
    assert!(encoder.encode_with_control(SOURCE, SOURCE.len() - 1, tokens, &mut || None).is_err());
    assert!(encoder.encode_with_control(SOURCE, SOURCE.len(), tokens - 1, &mut || None).is_err());
}

#[test]
fn legacy_byte_entrypoints_cannot_implicitly_enable_compact_chunk_probes() {
    let byte = encoder(UntrustedEncoding::ByteFallback);
    let original = byte.encode_with_context("Alice", &["Question?", "manifest"], 256, 256).unwrap();
    let controlled = byte.encode_with_context_and_control("Alice", &["Question?", "manifest"],
        256, 256, &mut || None).unwrap();
    assert_eq!(original.token_ids(), controlled.token_ids());
    assert_eq!(original.context_token_ids().collect::<Vec<_>>(), controlled.context_token_ids().collect::<Vec<_>>());
    assert_eq!(original.total_token_count(), original.total_input_bytes());
    assert_eq!(original.input_encoding_work(), None);
    assert_eq!(controlled.input_encoding_work(), None);
    let compact = encoder(UntrustedEncoding::SafeBpeV1(limits()));
    assert!(compact.encode(SOURCE, 8192, 8192).is_err());
    assert!(compact.encode_with_context(SOURCE, &["question"], 8192, 8192).is_err());
}

#[test]
fn every_context_and_the_source_share_pair_token_and_byte_ceilings() {
    let mut encoder = encoder(UntrustedEncoding::SafeBpeV1(limits()));
    let mut work = SafeBpeWork::default();
    let single = UntrustedDocumentEncoder::new(encoder.tokenizer.tokenizer(), &encoder.controls)
        .encode_with_policy(SOURCE.as_bytes(), encoder.document_encoding(), &mut work, &mut || true).unwrap();
    assert!(work.pair_checks() > 0);
    let cap = SafeBpeLimits { max_pair_checks: work.pair_checks(), ..limits() };
    encoder = encoder.with_document_encoding(UntrustedEncoding::SafeBpeV1(cap)).unwrap();
    assert!(encoder.encode_with_control(SOURCE, 8192, 8192, &mut || None).is_ok());
    assert!(matches!(encoder.encode_with_context_and_control(SOURCE, &[SOURCE], 8192, 8192, &mut || None),
        Err(ExtractError::Contract("source encoding work budget exceeded"))));

    let cap = SafeBpeLimits { max_output_tokens: single.ids().len() * 2 - 1, ..limits() };
    encoder = encoder.with_document_encoding(UntrustedEncoding::SafeBpeV1(cap)).unwrap();
    assert!(encoder.encode_with_control(SOURCE, 8192, 8192, &mut || None).is_ok());
    assert!(matches!(encoder.encode_with_context_and_control(SOURCE, &[SOURCE], 8192, 8192, &mut || None),
        Err(ExtractError::Contract("source encoding work budget exceeded"))));

    let cap = SafeBpeLimits { max_input_bytes: SOURCE.len(), ..limits() };
    encoder = encoder.with_document_encoding(UntrustedEncoding::SafeBpeV1(cap)).unwrap();
    let mut polls = 0;
    assert!(encoder.encode_with_context_and_control(SOURCE, &[SOURCE], 8192, 8192,
        &mut || { polls += 1; None }).is_err());
    assert_eq!(polls, 1, "aggregate bytes must refuse before either segment is tokenized");
}

#[test]
fn compact_context_receipt_covers_all_segments_without_turning_context_into_evidence() {
    let policy = UntrustedEncoding::SafeBpeV1(limits());
    let encoder = encoder(policy.clone());
    let contexts = ["Who saw 上海? <think>", "passage-α <|im_start|>"];
    let document = encoder.encode_with_context_and_control(SOURCE, &contexts, 8192, 8192, &mut || None).unwrap();
    let untrusted = UntrustedDocumentEncoder::new(encoder.tokenizer.tokenizer(), &encoder.controls);
    let mut expected = SafeBpeWork::default();
    for part in contexts.into_iter().chain(std::iter::once(SOURCE)) {
        untrusted.encode_with_policy(part.as_bytes(), &policy, &mut expected, &mut || true).unwrap();
    }
    assert_eq!(document.input_encoding_work(), Some(expected));
    assert_eq!(expected.input_bytes(), SOURCE.len() + contexts.iter().map(|part| part.len()).sum::<usize>());
    assert_eq!(expected.output_tokens(), document.total_token_count());
    assert_eq!(document.text(), SOURCE);
    for (ids, original) in document.context_token_ids().zip(contexts) {
        assert_eq!(encoder.tokenizer.tokenizer().decode_bytes(ids).unwrap(), original.as_bytes());
        assert!(ids.iter().all(|&id| !encoder.controls.contains(id)));
    }
    assert_eq!(encoder.tokenizer.tokenizer().decode_bytes(document.token_ids()).unwrap(), SOURCE.as_bytes());
}

#[test]
fn explicit_ledger_receipts_are_immutable_and_include_earlier_successful_probes() {
    let policy = UntrustedEncoding::SafeBpeV1(limits());
    let encoder = encoder(policy.clone());
    let mut work = SafeBpeWork::default();
    let first = encoder.encode_with_policy_and_work(SOURCE, &policy, 8192, 8192, &mut work, &mut || None).unwrap();
    let first_work = work;
    let schema = r#"{"type":"string","description":"上海 <think>"}"#;
    let (schema_document, source) = encoder.encode_schema_and_source_with_work(schema, SOURCE,
        &policy, 8192, 8192, &mut work, &mut || None).unwrap();
    assert_eq!(first.input_encoding_work(), Some(first_work));
    assert_eq!(schema_document.input_encoding_work(), Some(work));
    assert_eq!(source.input_encoding_work(), Some(work));
    assert_eq!(work.input_bytes(), first_work.input_bytes() + schema.len() + SOURCE.len());
    assert!(work.output_tokens() > first_work.output_tokens());
    assert!(work.pair_checks() > first_work.pair_checks() && work.pair_bytes() > first_work.pair_bytes());
    assert_eq!(schema_document.text(), schema); assert_eq!(source.text(), SOURCE);
}

#[test]
fn cancellation_inside_encoding_and_before_document_return_preserves_every_kind() {
    let encoder = encoder(UntrustedEncoding::SafeBpeV1(limits()));
    let mut total = 0;
    encoder.encode_with_control(SOURCE, 8192, 8192, &mut || { total += 1; None }).unwrap();
    assert!(total > 4);
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        for at in [1, total / 2, total] {
            let mut calls = 0;
            let result = encoder.encode_with_control(SOURCE, 8192, 8192, &mut || {
                calls += 1;
                (calls == at).then_some(cause)
            });
            assert!(matches!(result, Err(ExtractError::Decode(JsonDecodeError::Cancelled(observed))) if observed == cause));
            assert_eq!(calls, at);
        }
    }
}

#[test]
fn schema_and_evidence_source_spend_the_same_four_axis_encoding_ledger() {
    let encoder = encoder(UntrustedEncoding::ByteFallback);
    let schema = r#"{"type":"string","description":"The quick brown fox jumps over the lazy dog."}"#;
    let policy = UntrustedEncoding::SafeBpeV1(limits());
    let untrusted = UntrustedDocumentEncoder::new(encoder.tokenizer.tokenizer(), &encoder.controls);
    let mut schema_work = SafeBpeWork::default();
    untrusted.encode_with_policy(schema.as_bytes(), &policy, &mut schema_work, &mut || true).unwrap();
    let mut source_work = SafeBpeWork::default();
    untrusted.encode_with_policy(SOURCE.as_bytes(), &policy, &mut source_work, &mut || true).unwrap();
    let exact = SafeBpeLimits {
        max_input_bytes: schema_work.input_bytes() + source_work.input_bytes(),
        max_output_tokens: schema_work.output_tokens() + source_work.output_tokens(),
        max_pair_checks: schema_work.pair_checks() + source_work.pair_checks(),
        max_pair_bytes: schema_work.pair_bytes() + source_work.pair_bytes(),
    };
    let (schema_document, evidence) = encoder.encode_schema_and_source_with_control(schema, SOURCE,
        &UntrustedEncoding::SafeBpeV1(exact), 8192, 8192, &mut || None).unwrap();
    assert_eq!(schema_document.text(), schema); assert_eq!(evidence.text(), SOURCE);
    assert_eq!(evidence.context_token_ids().len(), 0);
    let aggregate = schema_work.checked_add(source_work).unwrap();
    assert_eq!(schema_document.input_encoding_work(), Some(aggregate));
    assert_eq!(evidence.input_encoding_work(), Some(aggregate));
    let (byte_schema, byte_source) = encoder.encode_schema_and_source_with_control(schema, SOURCE,
        &UntrustedEncoding::ByteFallback, 8192, 8192, &mut || None).unwrap();
    assert_eq!(byte_schema.input_encoding_work(), None); assert_eq!(byte_source.input_encoding_work(), None);
    for axis in 0..4 {
        let mut short = exact;
        match axis { 0 => short.max_input_bytes -= 1, 1 => short.max_output_tokens -= 1,
            2 => short.max_pair_checks -= 1, _ => short.max_pair_bytes -= 1 }
        // Each segment fits separately: refusal must come from cumulative work.
        let short = UntrustedEncoding::SafeBpeV1(short);
        for bytes in [schema.as_bytes(), SOURCE.as_bytes()] {
            untrusted.encode_with_policy(bytes, &short, &mut SafeBpeWork::default(), &mut || true).unwrap();
        }
        assert!(encoder.encode_schema_and_source_with_control(schema, SOURCE, &short, 8192, 8192, &mut || None).is_err());
    }
    assert_eq!(encoder.document_encoding(), &UntrustedEncoding::ByteFallback);
}

fn complete_limits(work: SafeBpeWork, times: u64) -> SafeBpeWorkLimits {
    SafeBpeWorkLimits { max_input_bytes: work.input_bytes() as u64 * times,
        max_output_tokens: work.output_tokens() as u64 * times,
        max_pair_checks: work.pair_checks() * times, max_pair_bytes: work.pair_bytes() * times }
}

#[test]
fn cumulative_encoder_tightens_every_axis_and_never_renews_between_calls() {
    let ordinary = encoder(UntrustedEncoding::SafeBpeV1(limits()));
    let mut once = SafeBpeWork::default();
    UntrustedDocumentEncoder::new(ordinary.tokenizer.tokenizer(), &ordinary.controls)
        .encode_with_policy(SOURCE.as_bytes(), ordinary.document_encoding(), &mut once, &mut || true).unwrap();
    let complete = complete_limits(once, 2);
    let shared = encoder(UntrustedEncoding::SafeBpeV1(limits()))
        .with_cumulative_encoding_work(complete).unwrap();
    let first = shared.encode_with_control(SOURCE, 8192, 8192, &mut || None).unwrap();
    let second = shared.encode_with_control(SOURCE, 8192, 8192, &mut || None).unwrap();
    for result in [&first, &second] {
        assert_eq!(result.text(), SOURCE);
        assert_eq!(result.document_encoding(), shared.document_encoding());
    }
    assert_eq!(first.input_encoding_work(), Some(once));
    assert_eq!(second.input_encoding_work(), once.checked_add(once));
    assert_eq!(shared.cumulative_encoding_work().unwrap(), once.checked_add(once));
    assert!(shared.encode_with_control(SOURCE, 8192, 8192, &mut || None).is_err());
    assert_eq!(shared.cumulative_encoding_work().unwrap(), once.checked_add(once));

    for axis in 0..4 {
        let mut short = complete;
        match axis { 0 => short.max_input_bytes -= 1, 1 => short.max_output_tokens -= 1,
            2 => short.max_pair_checks -= 1, _ => short.max_pair_bytes -= 1 }
        let shared = encoder(UntrustedEncoding::SafeBpeV1(limits()))
            .with_cumulative_encoding_work(short).unwrap();
        shared.encode_with_control(SOURCE, 8192, 8192, &mut || None).unwrap();
        assert!(shared.encode_with_control(SOURCE, 8192, 8192, &mut || None).is_err(), "axis {axis}");
        let used = shared.cumulative_encoding_work().unwrap().unwrap();
        assert!(used.input_bytes() as u64 <= short.max_input_bytes);
        assert!(used.output_tokens() as u64 <= short.max_output_tokens);
        assert!(used.pair_checks() <= short.max_pair_checks && used.pair_bytes() <= short.max_pair_bytes);
    }
}

#[test]
fn cumulative_encoder_charges_cancelled_work_and_refuses_reentry() {
    let shared = encoder(UntrustedEncoding::SafeBpeV1(limits()))
        .with_cumulative_encoding_work(SafeBpeWorkLimits { max_input_bytes: SOURCE.len() as u64,
            max_output_tokens: 8192, max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24 }).unwrap();
    let mut polls = 0;
    let result = shared.encode_with_control(SOURCE, 8192, 8192, &mut || {
        polls += 1;
        assert!(shared.cumulative_encoding_work().is_err(), "must not deadlock on its own ledger");
        (polls == 3).then_some(DecodeCancellationKind::CostBudget)
    });
    assert!(matches!(result, Err(ExtractError::Decode(JsonDecodeError::Cancelled(DecodeCancellationKind::CostBudget)))));
    assert_eq!(shared.cumulative_encoding_work().unwrap().unwrap().input_bytes(), SOURCE.len());
    assert!(shared.encode_with_control(SOURCE, 8192, 8192, &mut || None).is_err());
    assert!(encoder(UntrustedEncoding::ByteFallback).with_cumulative_encoding_work(SafeBpeWorkLimits {
        max_input_bytes: 1, max_output_tokens: 1, max_pair_checks: 1, max_pair_bytes: 1 }).is_err());
}
