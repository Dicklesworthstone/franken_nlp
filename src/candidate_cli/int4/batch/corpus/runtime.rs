//! Lazy single-resident-model execution behind the bounded corpus stream.
//! No new native scheduler, runtime domain, retry policy or persistence store.
use super::*;
use super::super::runtime as batch_runtime;
use std::{fs::File, io::BufReader, time::{Duration, Instant}};
use crate::{NlpEngine, ResourceHostConfig, RuntimePreset, LeakResponsePolicy, MemoryClass,
    execution_identity::ExecutionIdentity,
    hosted::{CancellationToken, HostedOutput, ResidentInt4,
        grouped_int4::batch::{Int4BatchHostLimits, Int4ChatBatchMember}},
    native_engine::strict_int4::weights::snapshot::INT4_SNAPSHOT_VERSION,
    tasks::chat::grouped_int4::{Int4ChatPlanner,
        batch::{Int4ChatBatchLimits, Int4ChatBatchResult}}};

pub(super) fn execute(args: Int4Args, options: CorpusArgs, quoted: CorpusLimits,
    input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
    let limits = options.validate(&args)?;
    if limits.window.common.minimum_memory_bytes != quoted.window.common.minimum_memory_bytes
        || limits.frame_bytes != quoted.frame_bytes
        || !crate::candidate_cli::int4::prefill::matches(&args, quoted.window.common)?
        || quoted.window.common.kernel.kind() != args.kernel { return Err(CandidateError::Arguments); }
    let started = Instant::now(); let deadline = Duration::from_secs(args.timeout_seconds);
    let engine = NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 0,
        helper_threads: 0, thread_ceiling: 2, memory_ceiling_bytes: limits.window.common.memory_bytes,
        leak_response_policy: LeakResponsePolicy::RecordAndEscalate,
    }).build().map_err(|_| CandidateError::Runtime)?;
    let lease = engine.resources().acquire_lease();
    // The finite whole-run ID set, current input/plans and publication staging
    // remain charged through the footer's external write and flush.
    let _preparation = lease.reserve(MemoryClass::JobBuffers, limits.window.common.preparation_bytes)
        .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
    let mut processor = NativeWindows { engine: &engine, args: &args, limits: limits.window,
        started, deadline, cancellation: CancellationToken::default(), model: None, planner: None,
        first_execution: true };
    processor.remaining()?;
    if args.input.as_os_str() == "-" {
        stream::drive(&mut processor, &mut BufReader::with_capacity(8192, input), output,
            options, limits, args.max_input_bytes, args.max_checkpoints)
    } else {
        let file = File::open(&args.input).map_err(|_| CandidateError::Input)?;
        stream::drive(&mut processor, &mut BufReader::with_capacity(8192, file), output,
            options, limits, args.max_input_bytes, args.max_checkpoints)
    }
}

struct NativeWindows<'a> {
    engine: &'a NlpEngine, args: &'a Int4Args, limits: BatchLimits,
    started: Instant, deadline: Duration, cancellation: CancellationToken,
    model: Option<ResidentInt4>, planner: Option<Int4ChatPlanner>, first_execution: bool,
}
impl NativeWindows<'_> {
    fn remaining(&self) -> Result<Duration, CandidateError> {
        self.deadline.checked_sub(self.started.elapsed()).filter(|time| !time.is_zero()).ok_or(CandidateError::Timeout)
    }
}
struct NativePrepared {
    members: Vec<Int4ChatBatchMember>, items: Vec<batch_runtime::Item>,
    identity: ExecutionIdentity, limits: Int4BatchHostLimits,
}
struct NativeCompleted {
    result: HostedOutput<Int4ChatBatchResult>, items: Vec<batch_runtime::Item>, identity: ExecutionIdentity,
    source_root: String, group_width: usize, kernel: &'static str, snapshot: Option<batch_runtime::Snapshot>,
}
impl Serialize for NativeCompleted {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let digest = self.identity.logical_model_digest.to_hex();
        // The existing fixed-batch response is nested unchanged in the corpus
        // window frame. Its hosted result guard is not unwrapped or discarded.
        batch_runtime::Response {
            schema_version: 1, protocol: PROTOCOL, scope: "real-source-current-candidate-int4",
            evidence: "non_authoritative", model_id: "Nanbeige4.2-3B",
            source_revision: &self.identity.source_revision, source_root_sha256: &self.source_root,
            model_content_sha256: &digest, candidate_format: &self.identity.artifact_format,
            quant_recipe: &self.identity.quant_recipe, group_width: self.group_width, integer_kernel: self.kernel,
            items: &self.items, output: self.result.result(), snapshot: self.snapshot,
        }.serialize(serializer)
    }
}
impl stream::Processor for NativeWindows<'_> {
    type Prepared = NativePrepared;
    type Output = NativeCompleted;
    fn checkpoint(&mut self) -> Result<(), CandidateError> { self.remaining().map(|_| ()) }
    fn prepare(&mut self, window: InputWindow) -> Result<stream::Prepared<NativePrepared>, CandidateError> {
        self.remaining()?;
        // Called only after the current nonempty window is parsed, duplicate
        // checked and transport-admitted. Empty input never touches model IO.
        if self.model.is_none() {
            let model = batch_runtime::load(self.engine, self.args, self.limits.common,
                self.remaining()?, self.cancellation.clone())?;
            let planner = batch_runtime::planner(&model, self.args, self.limits.common)?;
            self.model = Some(model); self.planner = Some(planner);
        }
        let planner = self.planner.as_ref().ok_or(CandidateError::Identity)?;
        let (members, items) = batch_runtime::prepare(window.records, planner, self.args, self.limits.common,
            window.first_sequence, || self.remaining().map(|_| ()))?;
        let identity = members.first().ok_or(CandidateError::Input)?.prepared.execution_identity().clone();
        let limits = Int4BatchHostLimits {
            task: Int4ChatBatchLimits { generation: self.limits.generation, max_result_bytes: self.limits.result_bytes },
            allocator_reserve_bytes: ALLOCATOR_BYTES,
            run: batch_runtime::stage_run(self.remaining()?, self.args.max_checkpoints),
        };
        let quote = match self.limits.common.prefill {
            None => limits.requirements(&members),
            Some(prefill) => limits.requirements_packed(&members, prefill),
        }.map_err(|_| CandidateError::Planning)?;
        Ok(stream::Prepared { value: NativePrepared { members, items, identity, limits },
            planned: quote.generation.planned_work })
    }
    fn execute(&mut self, mut prepared: NativePrepared) -> Result<stream::Completed<NativeCompleted>, CandidateError> {
        self.remaining()?;
        let model = self.model.as_ref().ok_or(CandidateError::Identity)?;
        let mut snapshot = None;
        if self.first_execution {
            if self.limits.common.snapshot_pin.is_some() {
                snapshot = Some(batch_runtime::Snapshot { operation: "loaded", format: INT4_SNAPSHOT_VERSION,
                    file_bytes: self.limits.common.snapshot_need.file_bytes });
            }
            if let Some(path) = &self.args.save_snapshot {
                // Only once, after the FIRST window passes full native and
                // cumulative admission. Later corpus failure does not delete it.
                let pin = self.engine.save_int4_snapshot(model, path.clone(), self.limits.common.snapshot_budget(false)?,
                    batch_runtime::stage_run(self.remaining()?, self.args.snapshot_checkpoints), self.cancellation.clone())
                    .map_err(|_| CandidateError::Model)?;
                if pin != model.model_identity().snapshot_pin() { return Err(CandidateError::Identity); }
                snapshot = Some(batch_runtime::Snapshot { operation: "saved", format: INT4_SNAPSHOT_VERSION,
                    file_bytes: self.limits.common.snapshot_need.file_bytes });
            }
            self.first_execution = false;
        }
        prepared.limits.run.max_elapsed = self.remaining()?;
        let result = match self.limits.common.prefill {
            None => self.engine.execute_int4_chat_batch(model, prepared.members, prepared.limits,
                self.limits.common.kernel, self.cancellation.clone()),
            Some(prefill) => self.engine.execute_int4_chat_batch_packed(model, prepared.members, prepared.limits,
                self.limits.common.kernel, prefill, self.cancellation.clone()),
        }.map_err(|_| CandidateError::Execution)?;
        self.remaining()?;
        let planned = result.result().planned_work; let actual = result.result().model_work;
        let records = result.result().results.len();
        Ok(stream::Completed { value: NativeCompleted { result, items: prepared.items, identity: prepared.identity,
            source_root: model.model_identity().source_root_sha256().to_owned(), group_width: self.args.group_width,
            kernel: self.limits.common.kernel.kind().id(), snapshot }, planned, actual, records })
    }
}
