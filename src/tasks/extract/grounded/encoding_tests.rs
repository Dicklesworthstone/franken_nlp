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
    for _ in 0..2 {
        let result = shared.encode_with_control(SOURCE, 8192, 8192, &mut || None).unwrap();
        assert_eq!(result.text(), SOURCE);
        assert_eq!(result.document_encoding(), shared.document_encoding());
    }
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
