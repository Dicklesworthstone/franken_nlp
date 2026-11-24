//! Host contract tests without model construction, a runtime, or a fake result.
use super::*;
use crate::jobs::runner::source::document::tests::factory;

#[test]
fn every_identity_in_the_document_must_keep_the_fixed_source_task() {
    let p = factory(); let id = p.execution_identity().clone();
    check_task_identities("ner-v1", &[id.clone(), id.clone()]).unwrap();
    for other in ["keyphrases-v1", "summarize-v1", "answer-v1", "extract-v1"] {
        let mut last = id.clone(); last.task_spec = other.into();
        assert!(check_task_identities("ner-v1", &[id.clone(), last]).is_err());
    }
    assert!(check_task_identities("ner-v1", &[]).is_err());
    assert!(check_task_identities("ner-v1", &vec![id.clone(); 257]).is_err());
    assert!(check_task_identities("extract-v1", &[id.clone()]).is_err());
    let mut last = id.clone(); last.schema_version = 0;
    assert!(check_task_identities("ner-v1", &[id, last]).is_err());
}
#[test]
fn workspace_prices_the_maximum_frontier_not_a_single_chunks_output() {
    let p = factory(); let headroom = 1 << 20;
    let b = p.task_budget(); let m = p.document_limits(); let chunks = m.chunks.max_chunks as u64;
    let expected = (b.max_output_bytes * chunks + m.reduction.max_live_value_bytes as u64) * 4
        + u64::from(b.max_output_tokens) * chunks * 8 + headroom;
    assert_eq!(source_document_job_workspace(&p, headroom).unwrap(), expected);
    assert!(expected > b.max_output_bytes * chunks);
    assert!(source_document_job_workspace(&p, 0).is_err());
    assert!(source_document_job_workspace(&p, u64::MAX).is_err());
}
#[test]
fn compact_workspace_is_added_to_the_complete_frontier_and_not_renewed_per_document() {
    use crate::tokenizer::untrusted::{SafeBpeLimits, SafeBpeWorkLimits, UntrustedEncoding};
    let byte = factory(); let reserve = 1 << 20;
    let original = source_document_job_workspace(&byte, reserve).unwrap();
    let call = SafeBpeLimits { max_input_bytes: 4096, max_output_tokens: 4096,
        max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24 };
    let total = SafeBpeWorkLimits { max_input_bytes: 1 << 20, max_output_tokens: 1 << 20,
        max_pair_checks: 1 << 28, max_pair_bytes: 1 << 32 };
    let compact = byte.with_document_encoding(UntrustedEncoding::SafeBpeV1(call), Some(total)).unwrap();
    let workspace = compact.encoding_workspace_bytes().unwrap();
    assert_eq!(workspace, call.requirements().unwrap().workspace_bytes + 1024);
    assert_eq!(source_document_job_workspace(&compact, reserve).unwrap(), original + workspace);
    assert!(source_document_job_workspace(&compact, u64::MAX - (original - reserve)).is_err());
    assert_eq!(compact.encoding_work_limits(), Some(total));
    assert_eq!(compact.encoding_work().unwrap(), Some(crate::tokenizer::untrusted::SafeBpeWork::default()));
}
#[test]
fn whole_owned_input_is_send_without_releasing_its_charge_early() {
    fn send<T: Send + 'static>() {}
    send::<StreamInput<(Int8SourceDocumentJobPlanner, Arc<ExtractionVocabulary>, SourceJobRequest),
        std::io::Cursor<Vec<u8>>, ()>>();
}
