//! One resident model, one complete sampled policy, one owned stream lifetime.
use super::*;
use std::{io::{BufRead, BufReader}, sync::Arc, time::{Duration, Instant}};
use crate::{
    NlpEngine, MemoryClass,
    batch::{BatchCode, BatchFault, sentiment::sampled::SampledSentimentBatchPlanner},
    candidate_cli::batch::CandidateWriter,
    hosted::{CancellationToken, HostedError, NativeLimits, RunLimits, corpus::CorpusLimits},
    native_engine::{decode::DecodeCancellationKind,
        strict_int4::weights::snapshot::INT4_SNAPSHOT_VERSION},
    tasks::sentiment::sampled::{SampledSentimentPlanner, SampledSentimentVocabulary},
};
use super::super::runtime::{cancelled, host_failure, task_failure};

fn batch_failure(fault: BatchFault) -> Failure {
    if let Some(cause) = fault.cancellation { return cancelled(cause); }
    use BatchCode::*;
    let (exit, code) = match fault.code {
        InvalidLimits => (ErrorCode::Usage, "sampled_sentiment_batch_limits"),
        InvalidUtf8 | InvalidJson | InvalidEnvelope | InvalidId | DuplicateId | Planning | InputIo =>
            (ErrorCode::InputDecodeOrParse, "sampled_sentiment_batch_input"),
        LineLimit | DocumentLimit | EpochIdLimit | InputLimit | RequestLimit
            | OutputLineLimit | OutputLimit | WorkLimit =>
            (ErrorCode::BudgetOrTimeout, "sampled_sentiment_batch_budget"),
        Admission | Allocation => (ErrorCode::AdmissionOrResourceLimit, "sampled_sentiment_batch_admission"),
        Execution => (ErrorCode::StructuredTaskNoResult, "sampled_sentiment_batch_no_result"),
        Cancelled => (ErrorCode::Generic, "sampled_sentiment_batch_missing_cancellation"),
        SequenceOverflow | InvalidExecution | Serialization | OutputIo =>
            (ErrorCode::Generic, "sampled_sentiment_batch_runtime_or_delivery"),
    };
    Failure { exit, code, cancellation: None }
}

fn execution_failure(error: HostedError) -> Failure {
    match error {
        HostedError::BatchSetup(fault) => batch_failure(fault),
        HostedError::Batch(error) => batch_failure(error.fault),
        HostedError::Stopped { stop, task_error } => {
            if matches!(stop.kind, DecodeCancellationKind::FailFast | DecodeCancellationKind::LinkedExit) {
                if let Some(error) = task_error {
                    let mut failure = execution_failure(*error);
                    failure.cancellation = Some(stop.kind);
                    return failure;
                }
            }
            cancelled(stop.kind)
        }
        error => host_failure(error),
    }
}

pub(super) fn execute<R: Read + Send + 'static, W: Write + Send + 'static>(
    args: Int4Args, batch: BatchArgs, _admitted: BatchLimits, input: R, output: W)
    -> Result<(), Failure> {
    let started = Instant::now();
    let limits = batch.validate(&args)?;
    let common = limits.common;
    let deadline = Duration::from_secs(args.timeout_seconds);
    let remaining = || deadline.checked_sub(started.elapsed()).filter(|time| !time.is_zero())
        .ok_or_else(|| cancelled(DecodeCancellationKind::Deadline));
    let encoding = args.input_encoding.policy(args.max_input_bytes, common.max_prompt_tokens)?;
    let options = batch.sampled.options(&args)?;
    let budget = batch.sampled.budget(&args, common, SentimentAxis::ALL.len())?;
    let engine = NlpEngine::builder().resource_config(compute::resource_config(common))
        .build().map_err(|_| CandidateError::Runtime)?;
    let lease = engine.resources().acquire_lease();
    let _preparation = lease.reserve(MemoryClass::JobBuffers, common.preparation_bytes)
        .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
    remaining()?;
    let reader: Box<dyn BufRead + Send> = if args.input.as_os_str() == "-" {
        Box::new(BufReader::with_capacity(IO_BUFFER_BYTES, input))
    } else {
        Box::new(BufReader::with_capacity(IO_BUFFER_BYTES,
            crate::local_io::open_document(&args.input).map_err(|_| CandidateError::Input)?))
    };
    remaining()?;
    let cancellation = CancellationToken::default();
    let model = crate::candidate_cli::int4::runtime::load_model_with_failure(&engine, &args, common,
        remaining()?, cancellation.clone(), execution_failure)?;
    remaining()?;
    let planner = SampledSentimentPlanner::for_model(model.model_identity(), options, budget,
        batch.sampled.task_limits(&args)).map_err(task_failure)?
        .with_input_encoding(encoding.clone()).map_err(task_failure)?;
    remaining()?;
    let vocabulary = Arc::new(SampledSentimentVocabulary::pinned().map_err(task_failure)?);
    remaining()?;
    let planner = SampledSentimentBatchPlanner::new(planner, vocabulary, budget, limits.native)
        .map_err(batch_failure)?;
    if planner.group_width() != args.group_width || planner.document_encoding() != &encoding
        || planner.preparation_requirement_bytes() > limits.preparation_reserve_bytes {
        return Err(CandidateError::Identity.into());
    }
    let identity = planner.base_execution_identity().clone();
    let mut snapshot = common.snapshot_pin.map(|_| SnapshotRecord::new("loaded", common.snapshot_need));
    if let Some(path) = &args.save_snapshot {
        let pin = engine.save_int4_snapshot(&model, path.clone(), common.snapshot_budget(false)?,
            RunLimits { max_elapsed: remaining()?, max_checkpoints: args.snapshot_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES }, cancellation.clone()).map_err(execution_failure)?;
        if pin != model.model_identity().snapshot_pin() { return Err(CandidateError::Identity.into()); }
        snapshot = Some(SnapshotRecord::new("saved", common.snapshot_need));
    }
    remaining()?;
    let content = identity.logical_model_digest.to_hex();
    let provenance = Provenance {
        protocol: "fnlp-candidate-int4-sentiment-sampled-batch-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative",
        task: "sentiment-sampled", model_id: "Nanbeige4.2-3B",
        source_revision: &identity.source_revision, source_root_sha256: model.model_identity().source_root_sha256(),
        model_content_sha256: &content, candidate_format: &identity.artifact_format,
        quant_recipe: &identity.quant_recipe, group_width: args.group_width,
        integer_kernel: common.kernel.kind().id(), snapshot,
        compute_schedule: compute::Schedule::for_threads(common.compute.threads),
        input_encoding: input_encoding::reported_policy(&encoding),
    };
    // Only this framing sink owns stdout. The host's terminal event is final;
    // rejected records change process status without appending another frame.
    let writer = CandidateWriter::new(output, &provenance, limits.transport.max_output_line_bytes,
        limits.output_bytes)?;
    let corpus = CorpusLimits {
        native: NativeLimits { context_tokens: args.context_tokens, allocator_reserve_bytes: ALLOCATOR_BYTES,
            run: RunLimits { max_elapsed: remaining()?, max_checkpoints: args.max_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES } },
        transport: limits.transport, preparation_reserve_bytes: limits.preparation_reserve_bytes,
        io_reserve_bytes: limits.io_bytes,
    };
    let summary = if common.compute.threads > 1 {
        engine.batch_int4_sampled_sentiment_with_threads(&model, planner, corpus, common.kernel,
            common.compute.threads, reader, writer, cancellation)
    } else {
        engine.batch_int4_sampled_sentiment(&model, planner, corpus, common.kernel, reader, writer, cancellation)
    }.map_err(execution_failure)?;
    completed(summary)
}

#[derive(Serialize)]
struct SnapshotRecord { operation: &'static str, format: &'static str, file_bytes: u64 }
impl SnapshotRecord {
    fn new(operation: &'static str, need: Int4SnapshotRequirements) -> Self {
        Self { operation, format: INT4_SNAPSHOT_VERSION, file_bytes: need.file_bytes }
    }
}
#[derive(Serialize)]
struct Provenance<'a> {
    protocol: &'static str, schema_version: u32, scope: &'static str, evidence: &'static str,
    task: &'static str, model_id: &'static str, source_revision: &'a str, source_root_sha256: &'a str,
    model_content_sha256: &'a str, candidate_format: &'a str, quant_recipe: &'a str,
    group_width: usize, integer_kernel: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    snapshot: Option<SnapshotRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    compute_schedule: Option<compute::Schedule>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_encoding: Option<&'a crate::tokenizer::untrusted::UntrustedEncoding>,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn batch_cancellation_keeps_every_typed_cause() {
        use DecodeCancellationKind::*;
        for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
            ParentCancelled, ResourceUnavailable, LinkedExit, Shutdown] {
            let expected = cancelled(cause);
            for error in [HostedError::BatchSetup(BatchFault::cancelled(cause)),
                HostedError::Batch(crate::batch::BatchRunError {
                    fault: BatchFault::cancelled(cause), summary: BatchSummary::default() })] {
                let actual = execution_failure(error);
                assert_eq!(actual.exit, expected.exit);
                assert_eq!(actual.cancellation, Some(cause));
            }
        }
        for code in [BatchCode::WorkLimit, BatchCode::InputLimit, BatchCode::OutputLimit] {
            assert_eq!(batch_failure(code.into()).exit, ErrorCode::BudgetOrTimeout);
        }
        assert_eq!(batch_failure(BatchCode::Cancelled.into()).exit, ErrorCode::Generic);
    }

    #[test]
    fn every_event_keeps_candidate_provenance_and_exact_inner_bytes() {
        let encoding = crate::tokenizer::untrusted::UntrustedEncoding::ByteFallback;
        let provenance = Provenance { protocol: "fnlp-candidate-int4-sentiment-sampled-batch-v1",
            schema_version: 1, scope: "real-source-current-candidate-int4", evidence: "non_authoritative",
            task: "sentiment-sampled", model_id: "Nanbeige4.2-3B", source_revision: "public-revision",
            source_root_sha256: "public-root", model_content_sha256: "public-content",
            candidate_format: "candidate", quant_recipe: "recipe", group_width: 128,
            integer_kernel: "s_scalar_i32", snapshot: None,
            compute_schedule: compute::Schedule::for_threads(4),
            input_encoding: input_encoding::reported_policy(&encoding) };
        let mut output = Vec::new();
        {
            let mut writer = CandidateWriter::new(&mut output, &provenance, 4096, 16384).unwrap();
            for record in [b"{\"event\":\"run_start\"}\n".as_slice(),
                b"{\"event\":\"result\",\"precise\":0.123456789012345678901}\n",
                b"{\"event\":\"run_complete\"}\n"] {
                writer.write_all(record).unwrap(); writer.flush().unwrap();
            }
        }
        let text = String::from_utf8(output).unwrap();
        assert_eq!(text.lines().count(), 3);
        assert!(text.contains("0.123456789012345678901"));
        for line in text.lines() {
            let value: serde_json::Value = serde_json::from_str(line).unwrap();
            assert_eq!(value["evidence"], "non_authoritative");
            assert_eq!(value["compute_schedule"]["declared_compute_threads"], 4);
            assert!(value.get("input_encoding").is_none());
            assert!(value.get("effective_seed").is_none());
        }
    }
}
