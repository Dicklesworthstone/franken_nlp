//! Owned complete-source INT8 faithfulness populations and exact process admission.
use super::super::*;
use super::host_error;
use super::super::source_tasks::Session;
use std::{io::{BufRead, BufReader}, sync::Arc};
use crate::{
    candidate_cli::{faithfulness_map::corpus::{self as command, Admitted, FaithfulnessCorpusCommand},
        batch::{CandidateWriter, IO_BUFFER_BYTES, completed}},
    hosted::corpus::{CorpusLimits, FaithfulnessDocumentCorpusConfig,
        int8_faithfulness_document_corpus_requirements},
    tasks::judge::{JudgePlanner, FaithfulnessPolicy},
    tokenizer::untrusted::{SafeBpeWorkLimits, UntrustedEncoding},
};

fn corpus_limits(admitted: &Admitted, elapsed: Duration) -> CorpusLimits {
    let document = &admitted.document;
    CorpusLimits { native: NativeLimits { context_tokens: document.common.context_tokens,
        allocator_reserve_bytes: 64 * MIB, run: RunLimits { max_elapsed: elapsed,
            max_checkpoints: document.common.max_checkpoints, cleanup_reserve_bytes: 65_536 } },
        transport: admitted.transport, preparation_reserve_bytes: document.limits.preparation_bytes,
        io_reserve_bytes: admitted.io_bytes }
}

fn sum(values: &[u64]) -> Result<u64, CandidateError> {
    values.iter().try_fold(0_u64, |total, &bytes|
        total.checked_add(bytes).ok_or(CandidateError::Arguments))
}

pub(in crate::candidate_cli) fn admit(admitted: &Admitted) -> Result<u64, CandidateError> {
    let document = &admitted.document;
    if document.limits.compute != document.common.compute.admit_prefill(document.prefill)? {
        return Err(CandidateError::Arguments);
    }
    let host = corpus_limits(admitted, Duration::from_secs(document.common.timeout_seconds));
    let quote = int8_faithfulness_document_corpus_requirements(document.task, document.planning,
        document.mapping, &document.encoding, host).map_err(|_| CandidateError::Arguments)?;
    let quote = match document.prefill {
        Some(prefill) => quote.with_prefill(prefill).map_err(|_| CandidateError::Arguments)?,
        None => quote,
    };
    let loading = sum(&[document.limits.weight_bytes, 192 * MIB, 64 * 1024, 65_536])?;
    let execution = sum(&[document.limits.weight_bytes, 192 * MIB, quote.input_bytes,
        quote.native.kv_bytes, quote.workspace_bytes, quote.output_reservation_bytes,
        document.limits.compute.workspace_bytes, 65_536])?;
    let peak = sum(&[document.limits.preparation_bytes, loading.max(execution)])?;
    if peak > document.limits.memory_bytes { return Err(CandidateError::Arguments); }
    Ok(peak)
}

fn configure(admitted: &Admitted, facts: &ArtifactIdentity, policy: FaithfulnessPolicy)
    -> Result<(Arc<JudgePlanner>, FaithfulnessDocumentCorpusConfig), CandidateError> {
    let document = &admitted.document;
    let mut identity = candidate_identity(facts)?;
    let registry = pinned_controls::pinned().map_err(|_| CandidateError::Identity)?;
    let controls = registry.template_controls();
    let eos = controls.entries().iter().find(|entry|
        entry.special && entry.surface == crate::template::IM_END)
        .map(|entry| entry.id).ok_or(CandidateError::Identity)?;
    let planner = Arc::new(JudgePlanner::pinned(controls, eos)
        .and_then(|planner| planner.with_document_encoding(document.encoding.clone()))
        .map_err(|_| CandidateError::Planning)?);
    identity.task_spec = "judge-v1".to_owned();
    identity.template_digest = *planner.template_digest();
    identity.tokenizer_digest = planner.tokenizer_digest();
    let config = FaithfulnessDocumentCorpusConfig { identity, task_ceiling: document.task,
        planning: document.planning, mapping: document.mapping, policy,
        max_model_work: document.mapping.max_model_work, max_tokenizer_work: admitted.max_tokenizer_work };
    config.validate(&planner).map_err(|_| CandidateError::Planning)?;
    Ok((planner, config))
}

#[derive(Serialize)]
struct Provenance<'a> {
    protocol: &'static str,
    schema_version: u32,
    scope: &'static str,
    evidence: &'static str,
    model_id: &'a str,
    source_revision: &'a str,
    source_root_sha256: &'a str,
    logical_model_sha256: &'a str,
    quant_recipe: &'a str,
    task: &'static str,
    judgment_mode: &'static str,
    complete_document: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_encoding: Option<&'a UntrustedEncoding>,
    #[serde(skip_serializing_if = "Option::is_none")]
    invocation_encoding_work_limits: Option<SafeBpeWorkLimits>,
    #[serde(skip_serializing_if = "Option::is_none")]
    compute_threads: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill_rows: Option<usize>,
}

pub(in crate::candidate_cli) fn execute<R: Read + Send + 'static, W: Write + Send + 'static>(
    command: FaithfulnessCorpusCommand, admitted: Admitted, mut input: R, output: W,
) -> Result<(), CandidateError> {
    admit(&admitted)?;
    let document = &admitted.document;
    let session = Session::new_with_prefill(&document.common, document.limits, document.prefill)?;
    let raw_policy = session.read(&command.transport.policy, &mut input, command::POLICY_BYTES)?;
    let policy = command::parse_policy(&raw_policy)?;
    drop(raw_policy);
    let facts = session.facts(&document.common)?;
    let (planner, config) = configure(&admitted, &facts, policy)?;
    session.remaining()?;
    let provenance = Provenance { protocol: "fnlp-candidate-batch-v1", schema_version: 1,
        scope: "real-artifact-current-candidate", evidence: "non_authoritative",
        model_id: &facts.model_id, source_revision: &facts.revision,
        source_root_sha256: &facts.source_root_sha256, logical_model_sha256: &facts.logical_model_sha256,
        quant_recipe: &facts.recipe_id, task: "judge", judgment_mode: "faithfulness", complete_document: true,
        input_encoding: input_encoding::reported_policy(&document.encoding),
        invocation_encoding_work_limits: admitted.max_tokenizer_work,
        compute_threads: (document.limits.compute.threads > 1).then_some(document.limits.compute.threads),
        prefill_rows: document.prefill.map(|limits| limits.max_batch_rows) };
    let writer = CandidateWriter::new(output, &provenance,
        admitted.transport.max_output_line_bytes, admitted.output_bytes)?;
    let reader: Box<dyn BufRead + Send> = if document.common.input.as_os_str() == "-" {
        Box::new(BufReader::with_capacity(IO_BUFFER_BYTES, input))
    } else {
        Box::new(BufReader::with_capacity(IO_BUFFER_BYTES,
            File::open(&document.common.input).map_err(|_| CandidateError::Input)?))
    };
    session.remaining()?;
    let cancellation = CancellationToken::default();
    let model = session.load(&document.common, document.limits, &facts, cancellation.clone())?;
    let host = corpus_limits(&admitted, session.remaining()?);
    let team = (document.limits.compute.threads > 1).then(||
        (crate::native_engine::simd::config::process_kernel(), document.limits.compute.threads));
    let summary = session.engine.batch_int8_faithfulness_documents_scheduled(&model, planner, config,
        host, document.prefill, team, reader, writer, cancellation).map_err(host_error)?;
    // The host publishes the terminal frame only after all workers join.
    // Per-record errors make the whole invocation fail without a second footer.
    completed(summary)
}

#[cfg(test)]
mod tests;
