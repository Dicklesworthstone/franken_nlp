//! Native source discovery and one complete resolution under one consumed plan.
use super::*;
use crate::{candidate_cli::int4::source as source_cli,
    corpus::entities_int4::{prepare_int4_entities, PreparedInt4EntityCorpus, Int4EntityRun,
        Int4EntityError, INT4_ENTITY_EXECUTION},
    tasks::{BuiltInTask, mapreduce::CHUNK_PROFILE, ner::NER_TASK_VERSION},
    validation::grounded_fields::GroundingBudget};

#[derive(Clone, Copy)]
struct DiscoveryExpected {
    documents: usize, chunks: usize, chunk_profile: Option<&'static str>, group_width: usize,
    ner_work: Int4Work, whole_work: Int4Work, masks: u64, verification: GroundingBudget,
}
impl DiscoveryExpected {
    fn of(prepared: &PreparedInt4EntityCorpus) -> Self {
        Self { documents: prepared.document_count(), chunks: prepared.chunk_count(),
            chunk_profile: prepared.config().chunks.map(|_| CHUNK_PROFILE), group_width: prepared.group_width(),
            ner_work: prepared.ner_reserved_work(), whole_work: prepared.config().max_model_work,
            masks: prepared.reserved_mask_visits(), verification: prepared.config().verification }
    }
}
fn within(work: Int4Work, ceiling: Int4Work) -> bool {
    work.forward_positions <= ceiling.forward_positions && work.projected_logits <= ceiling.projected_logits
        && work.attention_pairs <= ceiling.attention_pairs && work.projections.fits(ceiling.projections)
}
fn add_work(left: Int4Work, right: Int4Work) -> Result<Int4Work, CandidateError> {
    left.checked_add(right).map_err(|_| CandidateError::Execution)
}
fn check_result(result: &Int4EntityRun, expected: DiscoveryExpected) -> Result<(), CandidateError> {
    let pairs = &result.resolution;
    let reserved = add_work(expected.ner_work, pairs.planned_model_work)?;
    let actual = add_work(result.ner_model_work, pairs.model_work)?;
    if result.schema_version != 1 || result.execution != INT4_ENTITY_EXECUTION
        || result.numerics_profile != STRICT_INT4_PROFILE || result.extraction_task != NER_TASK_VERSION
        || result.group_width != expected.group_width || result.chunk_profile != expected.chunk_profile
        || result.chunk_count != expected.chunks || result.documents.len() != expected.documents
        || pairs.result.document_count != expected.documents || result.ner_reserved_work != expected.ner_work
        || !within(result.ner_model_work, expected.ner_work) || result.reserved_model_work != reserved
        || result.model_work != actual || !within(reserved, expected.whole_work)
        || (expected.documents != 0 && (result.ner_model_work.forward_positions == 0
            || result.ner_model_work.projected_logits == 0))
        || !within(actual, reserved) || result.reserved_mask_node_visits != expected.masks
        || result.mask_node_visit_charge > expected.masks
        || result.verification_used.fields > expected.verification.max_fields
        || result.verification_used.matches > expected.verification.max_matches
        || result.verification_used.scan_steps > expected.verification.max_scan_steps {
        return Err(CandidateError::Execution);
    }
    let (mut chunk_count, mut mention_count, mut masks) = (0_usize, 0_usize, 0_u64);
    let mut work = Int4Work::default();
    for document in &result.documents {
        if document.chunks.is_empty() || (expected.chunk_profile.is_none() && document.chunks.len() != 1) {
            return Err(CandidateError::Execution);
        }
        let (mut chunk_mentions, mut proposed, mut chunk_masks) = (0_usize, 0_usize, 0_u64);
        let mut chunk_work = Int4Work::default();
        for (index, chunk) in document.chunks.iter().enumerate() {
            if chunk.chunk_id != index { return Err(CandidateError::Execution); }
            chunk_mentions = chunk_mentions.checked_add(chunk.anchored_mentions).ok_or(CandidateError::Execution)?;
            proposed = proposed.checked_add(chunk.proposed_entities).ok_or(CandidateError::Execution)?;
            chunk_masks = chunk_masks.checked_add(chunk.mask_node_visits).ok_or(CandidateError::Execution)?;
            chunk_work = add_work(chunk_work, chunk.model_work)?;
        }
        if chunk_work != document.model_work || chunk_masks != document.mask_node_visits
            || proposed != document.proposed_entities || chunk_mentions != document.anchored_mentions {
            return Err(CandidateError::Execution);
        }
        chunk_count = chunk_count.checked_add(document.chunks.len()).ok_or(CandidateError::Execution)?;
        mention_count = mention_count.checked_add(document.anchored_mentions).ok_or(CandidateError::Execution)?;
        masks = masks.checked_add(document.mask_node_visits).ok_or(CandidateError::Execution)?;
        work = add_work(work, document.model_work)?;
    }
    if chunk_count != expected.chunks || mention_count != pairs.result.mentions.len()
        || work != result.ner_model_work || masks != result.mask_node_visit_charge {
        return Err(CandidateError::Execution);
    }
    super::check_result(pairs, Expected { documents: expected.documents, mentions: mention_count,
        pairs: pairs.result.judgments.len(), group_width: expected.group_width, work: pairs.planned_model_work })
}
fn planning_error(error: Int4EntityError) -> CandidateError {
    if let Some(cause) = error.cancellation() { return cancelled(cause); }
    match error { Int4EntityError::Identity => CandidateError::Identity,
        Int4EntityError::InvalidInput => CandidateError::Input, _ => CandidateError::Planning }
}

pub(super) fn execute(args: Int4Args, command: ResolveArgs,
    input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
    let started = Instant::now();
    let limits = command.validate(&args)?;
    let deadline = Duration::from_secs(args.timeout_seconds);
    let remaining = || deadline.checked_sub(started.elapsed()).filter(|time| !time.is_zero())
        .ok_or(CandidateError::Timeout);
    let engine = NlpEngine::builder().resource_config(compute::resource_config(limits))
        .build().map_err(|_| CandidateError::Runtime)?;
    let lease = engine.resources().acquire_lease();
    let preparation_bytes = sum(&[limits.preparation_bytes, command.encoding_workspace_bytes(&args)?])?;
    let _preparation = lease.reserve(MemoryClass::JobBuffers, preparation_bytes)
        .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
    remaining()?;
    let text = if args.input.as_os_str() == "-" { read_input(input, args.max_input_bytes)? }
        else { read_input(&mut crate::local_io::open_document(&args.input)
            .map_err(|_| CandidateError::Input)?, args.max_input_bytes)? };
    remaining()?;
    let raw = command.discovery.input(command, &args, &text)?;
    drop(text);
    let config = command.discovery.configuration(command, &args, limits, raw.ner, raw.options)?;
    if command.cohorts.enabled()
        && entity_cohort_memory_floor(command, &args, limits, &config)? > limits.memory_bytes {
        return Err(CandidateError::Memory);
    }
    remaining()?;
    let cancellation = CancellationToken::default();
    let model = crate::candidate_cli::int4::runtime::load_model(&engine, &args, limits,
        remaining()?, cancellation.clone())?;
    remaining()?;
    let encoding = command.encoding_limits(&args)?;
    let (source, vocabulary) = source_cli::planner()?;
    let source = match encoding {
        Some(limits) => source.with_map_document_encoding(limits.policy(), limits.complete)
            .map_err(|_| CandidateError::Planning)?,
        None => source,
    };
    let source_identity = source_cli::identity(model.model_identity(), &source, BuiltInTask::Ner)?;
    let (resolver, resolution_identity) = super::planner(model.model_identity(), encoding)?;
    let prepared = prepare_int4_entities(raw.documents, Arc::new(source), source_identity,
        Arc::new(resolver), resolution_identity.clone(), config, model.model_identity(),
        &mut DeadlineControl { started, deadline }).map_err(planning_error)?;
    if prepared.retained_input_bytes().map_err(planning_error)? > command.host_input_bytes(&args)? {
        return Err(CandidateError::Memory);
    }
    let expected = DiscoveryExpected::of(&prepared);
    if expected.documents != 0 {
        command.scoring.admit_plan(prepared.required_ner_context_tokens(), args.context_tokens, expected.ner_work)?;
    }
    remaining()?;
    let mut snapshot = limits.snapshot_pin.map(|_| SnapshotRecord::new("loaded", limits.snapshot_need));
    if let Some(path) = &args.save_snapshot {
        // Every NER chunk is preflighted now. Pair graph/work become known only
        // after discovery; later graph/work refusal preserves this explicit file.
        let pin = engine.save_int4_snapshot(&model, path.clone(), limits.snapshot_budget(false)?,
            RunLimits { max_elapsed: remaining()?, max_checkpoints: args.snapshot_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES }, cancellation.clone()).map_err(|_| CandidateError::Model)?;
        if pin != model.model_identity().snapshot_pin() { return Err(CandidateError::Identity); }
        snapshot = Some(SnapshotRecord::new("saved", limits.snapshot_need));
    }
    let content = resolution_identity.logical_model_digest.to_hex();
    let provenance = Provenance { protocol: "fnlp-candidate-int4-entities-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "resolve",
        model_id: "Nanbeige4.2-3B", source_revision: &resolution_identity.source_revision,
        source_root_sha256: model.model_identity().source_root_sha256(), model_content_sha256: &content,
        candidate_format: &resolution_identity.artifact_format, quant_recipe: &resolution_identity.quant_recipe,
        group_width: model.model_identity().group_width(), integer_kernel: limits.kernel.kind().id(), snapshot,
        cohort_pairs: command.cohorts.cohort_pairs, cohort_chunks: command.cohorts.cohort_chunks,
        input_encoding: command.encoding_report(&args)? };
    let native = native_limits(&args, remaining()?);
    let cap = command.cohorts.output_cap(limits.result_bytes)?;
    if command.cohorts.enabled() {
        let result = engine.execute_int4_entities_cohorts(&model, prepared, Arc::new(vocabulary), native,
            limits.preparation_bytes, command.graph_bytes()?, command.cohorts.schedule(),
            limits.kernel, cancellation).map_err(execution_error)?;
        check_result(&result.result().output, expected)?;
        remaining()?;
        publish_complete(&Response { provenance: &provenance, output: result.result() }, cap, output, &remaining)
    } else if limits.compute.threads > 1 {
        let result = engine.execute_int4_entities_with_threads(&model, prepared, Arc::new(vocabulary), native,
            limits.preparation_bytes, command.graph_bytes()?, limits.kernel, limits.compute.threads,
            cancellation).map_err(execution_error)?;
        check_result(result.result(), expected)?;
        remaining()?;
        let response = Response { provenance: &provenance, output: result.result() };
        publish_complete(&compute::Response { response: &response, compute: Some(result.compute()) },
            cap, output, &remaining)
    } else {
        let result = engine.execute_int4_entities(&model, prepared, Arc::new(vocabulary), native,
            limits.preparation_bytes, command.graph_bytes()?, limits.kernel, cancellation).map_err(execution_error)?;
        check_result(result.result(), expected)?;
        remaining()?;
        publish_complete(&Response { provenance: &provenance, output: result.result() }, cap, output, &remaining)
    }
}

#[cfg(test)]
mod tests;
