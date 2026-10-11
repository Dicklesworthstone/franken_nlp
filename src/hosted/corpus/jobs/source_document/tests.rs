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
    assert_eq!(document_workspace(&p, headroom).unwrap(), expected);
    assert!(expected > b.max_output_bytes * chunks);
    assert!(document_workspace(&p, 0).is_err());
    assert!(document_workspace(&p, u64::MAX).is_err());
}
#[test]
fn whole_owned_input_is_send_without_releasing_its_charge_early() {
    fn send<T: Send + 'static>() {}
    send::<StreamInput<(Int8SourceDocumentJobPlanner, Arc<ExtractionVocabulary>, SourceJobRequest),
        std::io::Cursor<Vec<u8>>, ()>>();
}
