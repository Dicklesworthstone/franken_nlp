//! Automatic NER discovery and complete resolution on the existing candidate host.
use super::*;
use crate::{corpus::entities_int8::{self, PreparedInt8EntityCorpus, Int8EntityRun, INT8_ENTITY_EXECUTION},
    native_engine::strict_int8::STRICT_INT8_PROFILE, tasks::BuiltInTask};
mod long;

#[derive(Clone, Copy)]
struct Expected { documents: usize, ner_work: Int8Work, mask_visits: u64 }
impl Expected {
    fn of(prepared: &PreparedInt8EntityCorpus) -> Self {
        Self { documents: prepared.document_count(), ner_work: prepared.ner_reserved_work(),
            mask_visits: prepared.reserved_mask_visits() }
    }
}
fn check_result(result: &Int8EntityRun, expected: Expected) -> Result<(), CandidateError> {
    if result.schema_version != 1 || result.execution != INT8_ENTITY_EXECUTION || result.numerics_profile != STRICT_INT8_PROFILE
        || result.documents.len() != expected.documents || result.resolution.result.document_count != expected.documents
        || result.ner_reserved_work != expected.ner_work || result.reserved_mask_node_visits != expected.mask_visits {
        return Err(CandidateError::Execution);
    }
    Ok(())
}
pub(super) fn execute(command: ResolveCommand, common: CandidateArgs, limits: Limits,
    input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
    if command.discovery.long.chunked { return long::execute(command, common, limits, input, output); }
    let session = Session::new(&common, limits)?;
    let text = session.read(&common.input, input, common.max_input_bytes)?;
    let raw = command.discovery.input(&command, &text)?;
    session.remaining()?;
    let facts = session.facts(&common)?;
    let (source, vocabulary) = super::super::source_tasks::planner()?;
    let source_identity = super::super::source_tasks::source_identity(&facts, &source, BuiltInTask::Ner)?;
    let (resolver, resolution_identity) = super::planner(&facts)?;
    let config = command.discovery.config(&command, limits, raw.ner, raw.options);
    // Exact source plans are checked before weights. The owned plan preserves
    // these witnesses through the host handoff; no borrowed source can drift.
    let prepared = entities_int8::prepare_int8_entities(raw.documents, Arc::new(source), source_identity,
        Arc::new(resolver), resolution_identity, config, &mut session.control()).map_err(|_| CandidateError::Planning)?;
    let expected = Expected::of(&prepared);
    if expected.documents == 0 {
        let result = prepared.finalize_without_model(&mut session.control()).map_err(|_| CandidateError::Execution)?;
        check_result(&result, expected)?;
        return super::deliver(&session, &facts, &result, command.host.max_result_bytes + 4096, output);
    }
    command.host.admit_plan(prepared.required_ner_context_tokens(), prepared.ner_reserved_work())?;
    session.remaining()?;
    let cancellation = CancellationToken::default();
    let model = session.load(&common, limits, &facts, cancellation.clone())?;
    let result = session.engine.execute_int8_entities(&model, prepared, Arc::new(vocabulary), session.native(&common)?,
        limits.preparation_bytes, command.graph_reserve_mib * MIB, cancellation).map_err(|_| CandidateError::Execution)?;
    check_result(result.result(), expected)?;
    super::deliver(&session, &facts, &result, command.host.max_result_bytes + 4096, output)
}

#[cfg(test)] mod tests;
