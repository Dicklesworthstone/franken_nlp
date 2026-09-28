//! Long-document raw discovery is one snapshot, never independent chunk jobs.
use super::*;
use crate::{corpus::entities_int8::long::{prepare_int8_document_entities, PreparedInt8DocumentEntityCorpus,
    Int8DocumentEntityRun, INT8_DOCUMENT_ENTITY_EXECUTION}, tasks::mapreduce::CHUNK_PROFILE};

#[derive(Clone, Copy)]
struct DocumentExpected { base: Expected, chunks: usize }
impl DocumentExpected {
    fn of(prepared: &PreparedInt8DocumentEntityCorpus) -> Self {
        Self { base: Expected { documents: prepared.document_count(), ner_work: prepared.ner_reserved_work(),
            mask_visits: prepared.reserved_mask_visits() }, chunks: prepared.chunk_count() }
    }
}
fn check_document_result(result: &Int8DocumentEntityRun, expected: DocumentExpected) -> Result<(), CandidateError> {
    super::check_result(&result.output, expected.base)?;
    if result.schema_version != 1 || result.execution != INT8_DOCUMENT_ENTITY_EXECUTION
        || result.chunk_profile != CHUNK_PROFILE || result.document_chunks.len() != expected.base.documents {
        return Err(CandidateError::Execution);
    }
    let mut chunks = 0_usize;
    for (geometry, document) in result.document_chunks.iter().zip(&result.output.documents) {
        if geometry.ner_chunks == 0 || geometry.document_id != document.document_id { return Err(CandidateError::Execution); }
        chunks = chunks.checked_add(geometry.ner_chunks).ok_or(CandidateError::Execution)?;
    }
    if chunks != expected.chunks { return Err(CandidateError::Execution); }
    Ok(())
}
pub(super) fn execute(command: ResolveCommand, common: CandidateArgs, limits: Limits,
    input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
    let session = Session::new(&common, limits)?;
    let lease = session.engine.resources().acquire_lease();
    // Conservative preflight staging/witness charge stays alive until the host
    // consumes and drains the prepared snapshot. Overlap with host reservations
    // is intentional; no uncharged handoff or all-chunk grammar collection.
    let preflight = lease.reserve(MemoryClass::JobBuffers, command.graph_reserve_mib * MIB)
        .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
    let text = session.read(&common.input, input, common.max_input_bytes)?;
    let raw = command.discovery.input(&command, &text)?;
    session.remaining()?;
    let facts = session.facts(&common)?;
    let (source, vocabulary) = super::super::super::source_tasks::planner()?;
    let source_identity = super::super::super::source_tasks::source_identity(&facts, &source, BuiltInTask::Ner)?;
    let (resolver, resolution_identity) = super::super::planner(&facts)?;
    let base = command.discovery.config(&command, limits, raw.ner, raw.options);
    let config = command.discovery.long.configuration(&command, base)?;
    let prepared = prepare_int8_document_entities(raw.documents, Arc::new(source), source_identity,
        Arc::new(resolver), resolution_identity, config, &mut session.control()).map_err(|_| CandidateError::Planning)?;
    let expected = DocumentExpected::of(&prepared);
    if expected.base.documents == 0 {
        let result = prepared.finalize_without_model(&mut session.control()).map_err(|_| CandidateError::Execution)?;
        check_document_result(&result, expected)?;
        return super::super::deliver(&session, &facts, &result, command.host.max_result_bytes + 4096, output);
    }
    command.host.admit_plan(prepared.required_ner_context_tokens(), prepared.ner_reserved_work())?;
    session.remaining()?;
    let cancellation = CancellationToken::default();
    let model = session.load(&common, limits, &facts, cancellation.clone())?;
    let result = session.engine.execute_int8_document_entities(&model, prepared, Arc::new(vocabulary), session.native(&common)?,
        limits.preparation_bytes, command.graph_reserve_mib * MIB, cancellation).map_err(|_| CandidateError::Execution)?;
    drop(preflight); drop(lease);
    check_document_result(result.result(), expected)?;
    super::super::deliver(&session, &facts, &result, command.host.max_result_bytes + 4096, output)
}

#[cfg(test)] mod tests;
