//! Complete source/graph/scoring preflight before weights, one native snapshot.
use super::*;
use std::sync::Arc;
use super::source_tasks::Session;
use crate::{candidate_cli::resolve::ResolveCommand,
    corpus::{resolve::{ResolutionPlan, RESOLVE_VERSION},
        native_resolve::{ResolutionPlanner, quantized::Int8ResolutionRun}},
    hosted::ResolveConfig, native_engine::strict_int8::Int8Work,
};

fn planner(facts: &ArtifactIdentity) -> Result<(ResolutionPlanner, ExecutionIdentity), CandidateError> {
    let registry = pinned_controls::pinned().map_err(|_| CandidateError::Identity)?;
    let controls = registry.template_controls();
    let eos = controls.entries().iter().find(|e| e.special && e.surface == crate::template::IM_END)
        .map(|e| e.id).ok_or(CandidateError::Identity)?;
    let planner = ResolutionPlanner::pinned(controls, eos).map_err(|_| CandidateError::Planning)?;
    let mut identity = candidate_identity(facts)?;
    identity.task_spec = RESOLVE_VERSION.to_owned();
    identity.template_digest = planner.template_digest();
    identity.tokenizer_digest = planner.tokenizer_digest();
    Ok((planner, identity))
}
#[derive(Clone, Copy)]
struct Expected { documents: usize, mentions: usize, pairs: usize, work: Int8Work }
fn check_result(result: &Int8ResolutionRun, expected: Expected) -> Result<(), CandidateError> {
    if result.planned_model_work != expected.work || result.model_work != expected.work
        || result.head_count != expected.pairs.checked_mul(2).ok_or(CandidateError::Execution)?
        || result.model_evaluated != (expected.pairs != 0)
        || result.result.document_count != expected.documents || result.result.mentions.len() != expected.mentions
        || result.result.judgments.len() != expected.pairs {
        return Err(CandidateError::Execution);
    }
    Ok(())
}

pub(in crate::candidate_cli) fn execute(command: ResolveCommand, common: CandidateArgs, limits: Limits,
    input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
    let session = Session::new(&common, limits)?;
    let lease = session.engine.resources().acquire_lease();
    // A separate modeled commitment covers the temporary preflight graph. It
    // precedes its allocations and is released before the hosted graph is built.
    let graph_charge = lease.reserve(MemoryClass::JobBuffers, command.graph_reserve_mib * MIB)
        .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
    let text = session.read(&common.input, input, common.max_input_bytes)?;
    let population = command.input(&text)?;
    let plan = ResolutionPlan::prepare(&population.documents, population.options, command.graph(), &mut session.control())
        .map_err(|_| CandidateError::Planning)?;
    session.remaining()?;
    let facts = session.facts(&common)?;
    let (planner, identity) = planner(&facts)?;
    let scoring = command.scoring(limits);
    let prepared = planner.prepare_int8(&plan, &identity, scoring, &mut session.control())
        .map_err(|_| CandidateError::Planning)?;
    let expected = Expected { documents: population.documents.len(), mentions: plan.mentions().len(),
        pairs: prepared.pair_count(), work: prepared.planned_work() };
    if expected.pairs == 0 {
        // Genuine source-validated singleton/empty result, not simulated neural
        // scores. No weights/KV/engine are loaded. Shared preparation already
        // prices this bounded result and staging through delivery.
        let result = prepared.finalize_without_model(&mut session.control()).map_err(|_| CandidateError::Execution)?;
        check_result(&result, expected)?;
        return deliver(&session, &facts, &result, command.host.max_result_bytes + 4096, output);
    }
    command.host.admit_plan(prepared.required_context_tokens(), expected.work)?;
    drop(prepared);
    drop(plan);
    drop(graph_charge);
    drop(lease);
    session.remaining()?;
    let cancellation = CancellationToken::default();
    let model = session.load(&common, limits, &facts, cancellation.clone())?;
    // Owned source and the same immutable planner enter the existing host. It
    // independently revalidates all anchors and admits every exact pair before
    // the first forward; no per-pair load, partial response or renewed quota.
    let result = session.engine.resolve_int8(&model, population.documents, Arc::new(planner), ResolveConfig {
        identity, options: population.options, graph: command.graph(), scoring,
        native: session.native(&common)?, preparation_reserve_bytes: limits.preparation_bytes,
        graph_reserve_bytes: command.graph_reserve_mib * MIB,
    }, cancellation).map_err(|_| CandidateError::Execution)?;
    check_result(result.result(), expected)?;
    deliver(&session, &facts, &result, command.host.max_result_bytes + 4096, output)
}
fn deliver<T: Serialize>(session: &Session, facts: &ArtifactIdentity, result: &T,
    cap: usize, output: &mut impl Write) -> Result<(), CandidateError> {
    session.remaining()?;
    let response = CandidateResponse { schema_version: 1, scope: "real-artifact-current-candidate",
        evidence: "non_authoritative", model_id: &facts.model_id, source_revision: &facts.revision,
        source_root_sha256: &facts.source_root_sha256, logical_model_sha256: &facts.logical_model_sha256,
        quant_recipe: &facts.recipe_id, output: result };
    publish(&response, cap, output)
}

#[cfg(test)] mod tests;
