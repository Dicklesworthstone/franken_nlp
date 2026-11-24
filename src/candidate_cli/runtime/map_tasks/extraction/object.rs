//! Explicit root-object extraction from every chunk, under one complete quote.
use super::*;
use crate::{
    batch::extract::quantized::long::object::{Int8ObjectExtractionPlanner,
        Int8ObjectExtractionMapLimits, Int8ObjectExtractionMapPreflight,
        Int8ObjectExtractionMapError, ROOT_OBJECT_MAP_PROMPT},
    hosted::Int8ObjectExtractionMapConfig,
    tasks::extract::object_reconcile::{RootObjectLimits, ROOT_OBJECT_DERIVATION, ROOT_OBJECT_RECONCILIATION},
};

fn object_planning_error(error: Int8ObjectExtractionMapError) -> CandidateError {
    error.cancellation().map(cancelled).unwrap_or(CandidateError::Planning)
}
fn object_execution_error(error: HostedError) -> CandidateError {
    match error {
        HostedError::Int8ObjectExtractionMap(error) =>
            error.cancellation().map(cancelled).unwrap_or(CandidateError::Execution),
        HostedError::Join { physical_error: Some(error), .. }
            | HostedError::Scope { physical_error: Some(error), .. } => object_execution_error(*error),
        HostedError::ModelIdentity | HostedError::ResourceDomain => CandidateError::Identity,
        other => host_error(other),
    }
}
fn preflight_object(source: &str, request: &ExtractionBatchArgs, planner: &Int8ObjectExtractionPlanner,
    mapping: Int8ObjectExtractionMapLimits, control: &mut impl DecodeStepControl)
    -> Result<Int8ObjectExtractionMapPreflight, CandidateError> {
    if source.is_empty() { return Err(CandidateError::Input); }
    let prepared = planner.plan_document_with_control(source, request, mapping, control)
        .map_err(object_planning_error)?;
    Ok(prepared.preflight_metadata())
}

pub(super) fn execute(command: MapCommand, args: CandidateArgs, limits: Limits,
    input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
    let reconciliation = command.extraction.object_limits(&command)?.ok_or(CandidateError::Arguments)?;
    let object_bytes = command.extraction.object_workspace_bytes(&command)?;
    let prefill = command.extraction.prefill()?;
    if limits.compute != command.compute.admit_prefill(prefill)? { return Err(CandidateError::Arguments); }
    command.extraction.check_schedule_memory(&command, &args, limits)?;
    let session = Session::new_with_prefill(&args, limits, prefill)?;
    // This separate charge precedes private schema/source/model IO. Keep it
    // through final verification, serialization and flush, while the host owns
    // an independent whole-object workspace through physical native drain.
    let lease = session.engine.resources().acquire_lease();
    let _object_preparation = lease.reserve(MemoryClass::JobBuffers, object_bytes)
        .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
    let schema_text = session.read(command.extraction.path()?, input, schema::SCHEMA_BYTES)?;
    let text = session.read(&command.input, input, args.max_input_bytes)?;
    if text.is_empty() { return Err(CandidateError::Input); }
    let request = schema::arguments(schema_text, command.extraction.source_membership,
        command.host.task_budget(limits))?;
    let mapping = Int8ObjectExtractionMapLimits { mapping: command.extraction_mapping()?, reconciliation };
    session.remaining()?;
    let facts = session.facts(&args)?;
    let base = compiler::planner(&facts, &command.host, limits, None)?;
    let base = match command.encoding.limits(&command)? {
        Some(encoding) => base.with_map_document_encoding(encoding.policy(), encoding.complete)
            .map_err(|_| CandidateError::Planning)?,
        None => base,
    };
    let planner = Int8ObjectExtractionPlanner::new(base).map_err(object_planning_error)?;
    // Every real partial schema/prompt and the ORIGINAL schema are admitted
    // before weights. The host rebuilds these sealed plans using the same
    // nonrenewable compact-tokenizer ledger, never a fresh preflight allowance.
    let expected = preflight_object(&text, &request, &planner, mapping, &mut session.control())?;
    command.admit_work(expected.planned_work(), expected.reserved_mask_visits())?;
    let vocabulary = compiler::vocabulary()?;
    session.remaining()?;
    let cancellation = CancellationToken::default();
    let model = session.load(&args, limits, &facts, cancellation.clone())?;
    let config = Int8ObjectExtractionMapConfig { mapping: ExtractionMapConfig {
        request, limits: mapping.mapping, native: session.native(&args)?,
        preparation_reserve_bytes: limits.preparation_bytes,
        reduction_reserve_bytes: command.reduction_reserve_mib.checked_mul(MIB).ok_or(CandidateError::Arguments)?,
    }, reconciliation };
    let planner = Arc::new(planner);
    let result = if limits.compute.threads == 1 {
        compute::Completion::Serial(match prefill {
            None => session.engine.extract_int8_object_document(&model, text, Arc::clone(&planner),
                vocabulary, config, cancellation),
            Some(prefill) => session.engine.extract_int8_object_document_layer_major(&model, text,
                Arc::clone(&planner), vocabulary, config, prefill, cancellation),
        }.map_err(object_execution_error)?)
    } else {
        let kernel = crate::native_engine::simd::config::process_kernel();
        compute::Completion::Team(match prefill {
            None => session.engine.extract_int8_object_document_with_threads(&model, text, Arc::clone(&planner),
                vocabulary, config, kernel, limits.compute.threads, cancellation),
            Some(prefill) => session.engine.extract_int8_object_document_layer_major_with_threads(&model, text,
                Arc::clone(&planner), vocabulary, config, kernel, limits.compute.threads, prefill, cancellation),
        }.map_err(object_execution_error)?)
    };
    expected.verify_completed_with_control(result.result(), &mut session.control())
        .map_err(|error| error.cancellation().map(cancelled).unwrap_or(CandidateError::Execution))?;
    let report = encoding::extraction_report(planner.base_planner())?;
    session.remaining()?;
    let response = CandidateResponse { schema_version: 1, scope: "real-artifact-current-candidate",
        evidence: "non_authoritative", model_id: &facts.model_id, source_revision: &facts.revision,
        source_root_sha256: &facts.source_root_sha256, logical_model_sha256: &facts.logical_model_sha256,
        quant_recipe: &facts.recipe_id, output: result.result() };
    let response = ObjectResponse { response, root_object: ObjectSelection::new(reconciliation) };
    let response = encoding::EncodedResponse { response, input_encoding: report.as_ref() };
    let response = compute::Response { response: &response, compute: result.compute() };
    // Keep the guarded partial map AND final object beside the caller schema
    // reserve through the last checked external write.
    publish_checked(&response, command.max_map_result_bytes + 4096, output,
        &|| session.remaining().map(|_| ()))
}

#[derive(Serialize)]
struct ObjectResponse<T: Serialize> {
    #[serde(flatten)]
    response: T,
    root_object: ObjectSelection,
}
#[derive(Serialize)]
struct ObjectSelection {
    policy: &'static str,
    schema_derivation: &'static str,
    map_prompt: &'static str,
    limits: RootObjectLimits,
}
impl ObjectSelection {
    fn new(limits: RootObjectLimits) -> Self {
        Self { policy: ROOT_OBJECT_RECONCILIATION, schema_derivation: ROOT_OBJECT_DERIVATION,
            map_prompt: ROOT_OBJECT_MAP_PROMPT, limits }
    }
}

#[cfg(test)] mod tests;
