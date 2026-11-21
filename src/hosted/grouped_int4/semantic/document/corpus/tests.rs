//! Pure declared-memory and authority regressions; no fabricated native success.
use super::*;
use super::super::tests::fixture;
use crate::{batch::BatchWork, tokenizer::untrusted::SafeBpeLimits};

const WIDTH: usize = 32;
fn config() -> Int4SemanticObjectDocumentCorpusConfig {
    let (extraction, options, host) = fixture();
    let work = Int4SemanticObjectDocumentBatchLimits {
        max_model_work: options.max_model_work,
        max_mask_node_visits: host.extraction.mapping.mapping.max_mask_visits,
        max_verification: host.extraction.mapping.verification,
    };
    let transport = BatchLimits {
        max_document_bytes: host.extraction.mapping.mapping.chunks.max_input_bytes,
        max_line_bytes: 8192, max_requests: 32, max_input_bytes: 64 * 1024 * 1024,
        max_output_line_bytes: options.limits.max_output_bytes as usize + 4096,
        max_output_bytes: 64 * 1024 * 1024,
        max_work: BatchWork { forward_positions: work.max_model_work.forward_positions,
            projected_logits: work.max_model_work.projected_logits },
        ..BatchLimits::default()
    };
    Int4SemanticObjectDocumentCorpusConfig { extraction, options, host, work, transport,
        encoding: UntrustedEncoding::ByteFallback, io_reserve_bytes: 16_384 }
}

#[test]
fn arbitrary_stream_length_does_not_multiply_resident_document_memory() {
    let mut c = config();
    let small = c.requirements(WIDTH).unwrap();
    c.transport.max_requests = 100_000;
    c.transport.max_input_bytes *= 2;
    c.transport.max_output_bytes *= 2;
    let larger = c.requirements(WIDTH).unwrap();
    assert_eq!(small, larger);
    let document = int4_semantic_object_document_requirements(WIDTH, c.extraction.budget,
        &c.options, &c.encoding, c.host, None).unwrap();
    assert_eq!(small.native, document.native);
    assert_eq!(small.workspace_bytes, document.workspace_bytes);
    assert_eq!(small.output_reservation_bytes, document.output_reservation_bytes);
    assert!(small.preparation_bytes > document.preparation_bytes,
        "the original fixed request and immutable factory coexist");
    assert!(small.input_bytes > small.preparation_bytes, "transport/IO staging is additional");
}

#[test]
fn every_whole_document_work_axis_must_fit_before_any_stream_can_start() {
    for axis in 0..8 {
        let mut c = config();
        match axis {
            0 => { c.work.max_model_work.forward_positions -= 1;
                c.transport.max_work.forward_positions = c.work.max_model_work.forward_positions; }
            1 => { c.work.max_model_work.projected_logits -= 1;
                c.transport.max_work.projected_logits = c.work.max_model_work.projected_logits; }
            2 => c.work.max_model_work.attention_pairs -= 1,
            3 => c.work.max_model_work.projections.dot_products -= 1,
            4 => c.work.max_model_work.projections.multiply_accumulates -= 1,
            5 => c.work.max_mask_node_visits -= 1,
            6 => c.work.max_verification.max_fields -= 1,
            _ => c.work.max_verification.max_scan_steps -= 1,
        }
        assert!(c.requirements(WIDTH).is_err(), "axis {axis} must refuse before private records");
    }
    let mut c = config();
    c.work.max_verification.max_matches -= 1;
    assert!(c.requirements(WIDTH).is_err());
}

#[test]
fn transport_cannot_substitute_work_or_document_authority() {
    for axis in 0..6 {
        let mut c = config();
        match axis {
            0 => c.transport.max_work.forward_positions -= 1,
            1 => c.transport.max_work.projected_logits -= 1,
            2 => c.transport.max_document_bytes -= 1,
            3 => c.transport.max_output_line_bytes = c.options.limits.max_output_bytes as usize - 1,
            4 => c.io_reserve_bytes = 0,
            _ => c.host.review.chunks.max_input_bytes -= 1,
        }
        assert!(c.requirements(WIDTH).is_err(), "transport axis {axis}");
    }
}

#[test]
fn actual_moved_schema_capacity_cannot_hide_behind_a_short_value() {
    let mut c = config();
    let baseline = c.requirements(WIDTH).unwrap();
    let declared = c.host.compiler.max_schema_bytes;
    let mut bounded = String::with_capacity(declared + 1);
    bounded.push_str(&c.extraction.schema);
    c.extraction.schema = bounded;
    assert_eq!(c.requirements(WIDTH).unwrap(), baseline,
        "bounded reader sentinel fits the pre-IO quote");
    let mut large = String::with_capacity(declared + 65_536);
    large.push_str(&c.extraction.schema);
    let excess = large.capacity() - declared - 1;
    c.extraction.schema = large;
    let priced = c.requirements(WIDTH).unwrap();
    assert_eq!(priced.input_bytes - baseline.input_bytes, excess as u64);
    assert_eq!(priced.preparation_bytes - baseline.preparation_bytes, excess as u64);
    assert_eq!(priced.native, baseline.native);
    assert!(!format!("{c:?}").contains(&c.extraction.schema));
}

#[test]
fn compact_encoder_and_numerical_team_have_one_invocation_charge() {
    let mut c = config();
    let bytes = c.requirements(WIDTH).unwrap();
    let caps = SafeBpeLimits { max_input_bytes: 16384, max_output_tokens: 8192,
        max_pair_checks: 1_000_000, max_pair_bytes: 4_000_000 };
    c.encoding = UntrustedEncoding::SafeBpeV1(caps);
    let compact = c.requirements(WIDTH).unwrap();
    assert_eq!(compact.input_bytes - bytes.input_bytes, caps.requirements().unwrap().workspace_bytes);
    for width in [2, 4] {
        let team = c.requirements_with_threads(WIDTH, width).unwrap();
        assert_eq!(team.input_bytes, compact.input_bytes);
        assert_eq!(team.total_bytes().unwrap() - compact.total_bytes().unwrap(),
            int4_task_team_workspace_bytes(width, WIDTH).unwrap());
    }
}

#[test]
fn identical_memory_shapes_cannot_substitute_reconciliation_or_review_policy() {
    let c = config();
    let extraction = c.host.extraction;
    let review = c.host.review;
    assert!(same_extraction_limits(extraction, extraction));
    assert!(same_review_limits(review, review));
    let mut other = extraction;
    other.reconciliation.max_origins -= 1;
    assert!(!same_extraction_limits(extraction, other));
    let mut other = extraction;
    other.mapping.mapping.chunks.max_tokenizer_calls -= 1;
    assert!(!same_extraction_limits(extraction, other));
    let mut other = review;
    other.aggregation.minimum_classified_coverage_ppm =
        if review.aggregation.minimum_classified_coverage_ppm == 0 { 1 } else { 0 };
    assert!(!same_review_limits(review, other));
    let mut other = review;
    other.max_model_work.attention_pairs -= 1;
    assert!(!same_review_limits(review, other));
}

#[test]
fn stream_owners_and_fixed_compilers_cross_one_owned_blocking_boundary() {
    fn send<T: Send + 'static>() {}
    type Reader = std::io::Cursor<Vec<u8>>;
    send::<StreamInput<Reader, Vec<u8>>>();
    send::<StreamExecution<Reader, Vec<u8>>>();
    send::<DeferredStream<Reader, Vec<u8>>>();
    let _serial = NlpEngine::batch_int4_semantic_object_documents::<Reader, Vec<u8>>;
    let _team = NlpEngine::batch_int4_semantic_object_documents_with_threads::<Reader, Vec<u8>>;
}
