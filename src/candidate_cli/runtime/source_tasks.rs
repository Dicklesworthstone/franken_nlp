//! Source tasks on the existing hosted INT8 path, with one end-to-end deadline.
use super::*;
use std::{path::PathBuf, sync::{Arc, Mutex}};
use crate::{CommittedMemory,
    hosted::{HostedError, ResidentInt8, SourceLimits},
    native_engine::{artifact_bridge::{ArtifactBridgeError, current_candidate_load_checkpoints},
        strict_int8::{CurrentCandidateInt8Error, prefill::Int8PrefillLimits},
        decode::{DecodeCancellationKind, DecodeStepControl}},
    tasks::{BuiltInTask, extract::ExtractionVocabulary, ir::PlanContext,
        source_planning::{SourceTaskPlanner, SourceTaskRequest, quantized::PreparedInt8SourceTask}},
};
use crate::candidate_cli::source::{self as command, SourceCommand, SourceHostArgs};

/// Must be declared before request/planner/vocabulary/output locals. Those
/// allocations then drop before the preparation charge, including on errors.
/// No uncharged resident model or independent runtime is created here.
pub(super) struct Session {
    // Metadata/file ownership drops before its preparation reservation. The
    // source is moved out before entering any hosted blocking invocation.
    checked_source: Mutex<Option<(PathBuf, CurrentCandidateArtifactSource)>>,
    _preparation: CommittedMemory,
    pub engine: NlpEngine,
    started: Instant,
    elapsed_limit: Duration,
}
impl Session {
    pub(super) fn new(args: &CandidateArgs, limits: Limits) -> Result<Self, CandidateError> {
        Self::new_with_prefill(args, limits, args.policy.prefill()?)
    }
    /// Source and scoring routes carry a separate physical prompt schedule.
    /// They must not inherit generation policy merely to configure a team.
    pub(super) fn new_with_prefill(args: &CandidateArgs, limits: Limits, prefill: Option<Int8PrefillLimits>)
        -> Result<Self, CandidateError> {
        if limits.compute != args.compute.admit_prefill(prefill)? {
            return Err(CandidateError::Arguments);
        }
        let started = Instant::now();
        let engine = NlpEngine::builder().resource_config(compute::resource_config(limits))
            .build().map_err(|_| CandidateError::Runtime)?;
        let lease = engine.resources().acquire_lease();
        let preparation = lease.reserve(MemoryClass::JobBuffers, limits.preparation_bytes)
            .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
        Ok(Self { checked_source: Mutex::new(None), _preparation: preparation, engine, started,
            elapsed_limit: Duration::from_secs(args.timeout_seconds) })
    }
    pub(super) fn remaining(&self) -> Result<Duration, CandidateError> {
        self.elapsed_limit.checked_sub(self.started.elapsed()).filter(|d| !d.is_zero())
            .ok_or(CandidateError::Timeout)
    }
    pub(super) fn read(&self, path: &std::path::Path, input: &mut impl Read, cap: usize)
        -> Result<String, CandidateError> {
        checked_session_stage(|| {
            if path.as_os_str() == "-" { read_input(input, cap) }
            else { read_input(&mut File::open(path).map_err(|_| CandidateError::Input)?, cap) }
        }, || self.remaining().map(|_| ()))
    }
    pub(super) fn facts(&self, args: &CandidateArgs) -> Result<ArtifactIdentity, CandidateError> {
        checked_session_stage(|| {
            let mut held = self.checked_source.lock().map_err(|_| CandidateError::Model)?;
            if let Some((path, source)) = held.as_ref() {
                if path != &args.model { return Err(CandidateError::Identity); }
                return Ok(source.identity().clone());
            }
            let source = inspect(&args.model, &mut self.control())?;
            let facts = source.identity().clone();
            candidate_identity(&facts)?;
            *held = Some((args.model.clone(), source));
            Ok(facts)
        }, || self.remaining().map(|_| ()))
    }
    pub(super) fn native(&self, args: &CandidateArgs) -> Result<NativeLimits, CandidateError> {
        Ok(NativeLimits { context_tokens: args.context_tokens, allocator_reserve_bytes: 64 * MIB,
            run: RunLimits { max_elapsed: self.remaining()?, max_checkpoints: args.max_checkpoints,
                cleanup_reserve_bytes: 65_536 } })
    }
    pub(super) fn load(&self, args: &CandidateArgs, limits: Limits, facts: &ArtifactIdentity,
        cancellation: CancellationToken) -> Result<ResidentInt8, CandidateError> {
        checked_session_stage(|| {
            let (path, source) = self.checked_source.lock().map_err(|_| CandidateError::Model)?
                .take().ok_or(CandidateError::Identity)?;
            if path != args.model || source.identity() != facts { return Err(CandidateError::Identity); }
            let max_checkpoints = current_candidate_load_checkpoints(source.file_bytes()).map_err(artifact_error)?;
            let model = self.engine.load_checked_current_candidate_int8(source, LoadLimits {
                artifact: ArtifactLoadBudget::streaming_only(limits.weight_bytes, 64 * 1024),
                tokenizer_and_metadata_bytes: 128 * MIB, allocator_reserve_bytes: 64 * MIB,
                run: RunLimits { max_elapsed: self.remaining()?, max_checkpoints, cleanup_reserve_bytes: 65_536 },
            }, cancellation).map_err(load_error)?;
            // Plans and weights bind the same held file, including source root and
            // recipe. The host independently checks the plan against resident data.
            if model.artifact_identity() != facts { return Err(CandidateError::Identity); }
            Ok(model)
        }, || self.remaining().map(|_| ()))
    }
    pub(super) fn control(&self) -> PreparationControl {
        PreparationControl { started: self.started, elapsed_limit: self.elapsed_limit }
    }
}

/// Only the session's read/inspection/load stages use this boundary. Within
/// those stages, Execution means an already observed native cancellation;
/// incidental IO, integrity and reservation failures have separate variants.
/// Always poll after the stage, but do not replace an earlier terminal cause
/// when physical cleanup crosses the original invocation deadline.
fn checked_session_stage<T>(stage: impl FnOnce() -> Result<T, CandidateError>,
    mut check: impl FnMut() -> Result<(), CandidateError>) -> Result<T, CandidateError> {
    check()?;
    let result = stage();
    let terminal = check();
    if matches!(&result, Err(CandidateError::Timeout | CandidateError::Execution)) {
        return result;
    }
    terminal?;
    result
}

fn cancelled(cause: DecodeCancellationKind) -> CandidateError {
    match cause {
        DecodeCancellationKind::Deadline | DecodeCancellationKind::Timeout => CandidateError::Timeout,
        _ => CandidateError::Execution,
    }
}
fn artifact_error(error: ArtifactBridgeError) -> CandidateError {
    match error { ArtifactBridgeError::Cancelled(cause) => cancelled(cause), _ => CandidateError::Model }
}
fn load_error(error: HostedError) -> CandidateError {
    match error {
        HostedError::Stopped { stop, .. } => cancelled(stop.kind),
        HostedError::Join { physical_error: Some(error), .. }
            | HostedError::Scope { physical_error: Some(error), .. } => load_error(*error),
        HostedError::Model(CurrentCandidateInt8Error::Artifact(error)) => artifact_error(error),
        HostedError::Reservation(_) => CandidateError::Memory,
        _ => CandidateError::Model,
    }
}

fn inspect(path: &std::path::Path, control: &mut impl DecodeStepControl)
    -> Result<CurrentCandidateArtifactSource, CandidateError> {
    if let Some(cause) = control.checkpoint(0) { return Err(cancelled(cause)); }
    let metadata = std::fs::symlink_metadata(path).map_err(|_| CandidateError::Model)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() { return Err(CandidateError::Model); }
    let mut remaining = current_candidate_load_checkpoints(metadata.len()).map_err(artifact_error)?;
    CurrentCandidateArtifactSource::open_with_checkpoint(path, || {
        if let Some(cause) = control.checkpoint(0) { return Some(cause); }
        if remaining == 0 { return Some(DecodeCancellationKind::PollQuota); }
        remaining -= 1;
        None
    }).map_err(artifact_error)
}

pub(super) struct PreparationControl { started: Instant, elapsed_limit: Duration }
impl DecodeStepControl for PreparationControl {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        (self.started.elapsed() >= self.elapsed_limit).then_some(DecodeCancellationKind::Deadline)
    }
}

pub(super) fn planner() -> Result<(SourceTaskPlanner, ExtractionVocabulary), CandidateError> {
    let controls = pinned_controls::pinned().map_err(|_| CandidateError::Identity)?;
    let eos = controls.template_controls().entries().iter()
        .find(|entry| entry.special && entry.surface == crate::template::IM_END)
        .map(|entry| entry.id).ok_or(CandidateError::Identity)?;
    let planner = SourceTaskPlanner::pinned(controls.template_controls(), eos).map_err(|_| CandidateError::Planning)?;
    let vocabulary = ExtractionVocabulary::pinned(controls.template_controls()).map_err(|_| CandidateError::Planning)?;
    Ok((planner, vocabulary))
}

pub(super) fn source_identity(facts: &ArtifactIdentity, planner: &SourceTaskPlanner, task: BuiltInTask)
    -> Result<ExecutionIdentity, CandidateError> {
    let mut identity = candidate_identity(facts)?;
    identity.task_spec = task.spec().identity();
    identity.template_digest = *planner.template_digest();
    identity.tokenizer_digest = planner.tokenizer_digest();
    Ok(identity)
}

fn prepare(planner: &SourceTaskPlanner, facts: &ArtifactIdentity, request: &SourceTaskRequest,
    host: &SourceHostArgs, control: &mut impl DecodeStepControl) -> Result<PreparedInt8SourceTask, CandidateError> {
    let identity = source_identity(facts, planner, request.task())?;
    let context = PlanContext::new(&identity, request.budget()).map_err(|_| CandidateError::Planning)?;
    let prepared = planner.plan_int8_with_control(request, &context, host.planning(), control)
        .map_err(|error| match error.cancellation() {
            Some(cause) => cancelled(cause), None => CandidateError::Planning,
        })?;
    if prepared.planned_work().forward_positions > host.context_tokens as u64 {
        return Err(CandidateError::Planning);
    }
    Ok(prepared)
}

pub(in crate::candidate_cli) fn execute(command: SourceCommand, args: CandidateArgs, limits: Limits,
    input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
    let prefill = command.args.prefill.limits()?;
    if limits.compute != command.args.compute.admit_prefill(prefill)? {
        return Err(CandidateError::Arguments);
    }
    compute::check_source_memory(&args, limits, command.args.host.max_result_bytes, prefill)?;
    let encoding = args.input_encoding_policy(limits)?;
    let session = Session::new_with_prefill(&args, limits, prefill)?;
    let text = session.read(&command.args.input, input, args.max_input_bytes)?;
    let options = command.args.options.as_ref()
        .map(|path| session.read(path, input, command::OPTIONS_BYTES)).transpose()?;
    let budget = command.args.host.task_budget(limits);
    let request = command::request(command.kind, text, options.as_deref(), budget, args.max_input_bytes)?;
    session.remaining()?;
    let facts = session.facts(&args)?;
    let (planner, vocabulary) = planner()?;
    let planner = planner.with_document_encoding(encoding.clone()).map_err(|_| CandidateError::Planning)?;
    session.remaining()?;
    let prepared = prepare(&planner, &facts, &request, &command.args.host, &mut session.control())?;
    // The mode and row ceiling enter the sealed policy before native admission.
    let prepared = command.args.head.source(prepared)?;
    session.remaining()?;
    let cancellation = CancellationToken::default();
    let model = session.load(&args, limits, &facts, cancellation.clone())?;
    let native_limits = SourceLimits {
        native: session.native(&args)?, preparation_reserve_bytes: limits.preparation_bytes,
        mask_limits: command.args.host.masks(), max_mask_node_visits: command.args.host.max_mask_node_visits,
    };
    let vocabulary = Arc::new(vocabulary);
    let result = if limits.compute.threads == 1 {
        compute::Completion::Serial(match prefill {
            Some(prefill) => session.engine.execute_int8_source_layer_major(&model, prepared,
                vocabulary, native_limits, prefill, cancellation),
            None => session.engine.execute_int8_source(&model, prepared, vocabulary, native_limits, cancellation),
        }.map_err(|_| CandidateError::Execution)?)
    } else {
        let kernel = crate::native_engine::simd::config::process_kernel();
        compute::Completion::Team(match prefill {
            Some(prefill) => session.engine.execute_int8_source_layer_major_with_threads(&model, prepared,
                vocabulary, native_limits, kernel, limits.compute.threads, prefill, cancellation),
            None => session.engine.execute_int8_source_with_threads(&model, prepared,
                vocabulary, native_limits, kernel, limits.compute.threads, cancellation),
        }.map_err(|_| CandidateError::Execution)?)
    };
    session.remaining()?;
    let response = CandidateResponse { schema_version: 1, scope: "real-artifact-current-candidate",
        evidence: "non_authoritative", model_id: &facts.model_id, source_revision: &facts.revision,
        source_root_sha256: &facts.source_root_sha256, logical_model_sha256: &facts.logical_model_sha256,
        quant_recipe: &facts.recipe_id, output: result.result() };
    // No result prefix is written until native/source finalization completes.
    // Both the output guard and preparation owner survive write AND flush.
    let response = compute::Response { response: &response, compute: result.compute() };
    let response = input_encoding::Response { response: &response,
        input_encoding: input_encoding::reported_policy(&encoding) };
    publish_checked(&response, command.args.host.max_result_bytes + 4096, output,
        &|| session.remaining().map(|_| ()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate_cli::source::Kind;

    #[test]
    fn bounded_input_always_checks_deadline_after_success_and_read_or_decode_failure() {
        use std::cell::Cell;
        struct BoundaryReader<'a> {
            bytes: &'static [u8], fail: bool, expire: bool,
            expired: &'a Cell<bool>, calls: usize,
        }
        impl Read for BoundaryReader<'_> {
            fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
                self.calls += 1;
                self.expired.set(self.expire);
                if self.fail { return Err(io::Error::other("private read fixture")); }
                let count = output.len().min(self.bytes.len());
                output[..count].copy_from_slice(&self.bytes[..count]);
                self.bytes = &self.bytes[count..];
                Ok(count)
            }
        }
        let cases = [
            (&b""[..], false, 4, Ok("")),
            (&b"text"[..], false, 4, Ok("text")),
            (&b""[..], true, 4, Err(CandidateError::Input)),
            (&b"\xff"[..], false, 4, Err(CandidateError::Input)),
            (&b"12345"[..], false, 4, Err(CandidateError::Input)),
        ];
        for (bytes, fail, cap, expected) in cases {
            for expire in [false, true] {
                let expired = Cell::new(false);
                let mut input = BoundaryReader { bytes, fail, expire, expired: &expired, calls: 0 };
                let mut polls = 0;
                let result = checked_session_stage(|| read_input(&mut input, cap), || {
                    polls += 1;
                    if expired.get() { Err(CandidateError::Timeout) } else { Ok(()) }
                });
                assert!(input.calls > 0);
                assert_eq!(polls, 2);
                assert_eq!(result, if expire { Err(CandidateError::Timeout) }
                    else { expected.map(str::to_owned) });
            }
        }
    }

    #[test]
    fn incidental_session_failures_yield_to_deadline_and_keep_their_live_category() {
        for error in [CandidateError::Input, CandidateError::Model,
            CandidateError::Identity, CandidateError::Memory] {
            for expire in [false, true] {
                let mut polls = 0;
                let result: Result<(), _> = checked_session_stage(|| Err(error), || {
                    polls += 1;
                    if expire && polls == 2 { Err(CandidateError::Timeout) } else { Ok(()) }
                });
                assert_eq!(polls, 2);
                assert_eq!(result, Err(if expire { CandidateError::Timeout } else { error }));
            }
        }
    }

    #[test]
    fn original_artifact_and_host_cancellation_survives_a_later_session_deadline() {
        use DecodeCancellationKind::*;
        for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast,
            RaceLost, ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
            let failures = [artifact_error(ArtifactBridgeError::Cancelled(cause)),
                load_error(HostedError::Model(CurrentCandidateInt8Error::Artifact(
                    ArtifactBridgeError::Cancelled(cause)))),
                load_error(HostedError::Stopped { stop: crate::hosted::RunStop {
                    kind: cause, runtime_reason: None, runtime_checkpoint_error: None }, task_error: None })];
            for error in failures {
                let mut polls = 0;
                let result: Result<(), _> = checked_session_stage(|| Err(error), || {
                    polls += 1;
                    if polls == 2 { Err(CandidateError::Timeout) } else { Ok(()) }
                });
                assert_eq!(polls, 2, "terminal stage must still poll after cleanup");
                assert_eq!(result, Err(cancelled(cause)), "{cause:?}");
            }
        }
    }

    #[test]
    fn refused_session_stages_skip_io_or_drop_completed_payload_before_return() {
        use std::cell::Cell;
        let calls = Cell::new(0);
        let result = checked_session_stage(|| { calls.set(calls.get() + 1); Ok(()) },
            || Err(CandidateError::Timeout));
        assert_eq!(result, Err(CandidateError::Timeout));
        assert_eq!(calls.get(), 0);

        struct Payload<'a>(&'a Cell<usize>);
        impl Drop for Payload<'_> {
            fn drop(&mut self) { self.0.set(self.0.get() + 1); }
        }
        for expire in [false, true] {
            let drops = Cell::new(0);
            let mut polls = 0;
            let result = checked_session_stage(|| Ok(Payload(&drops)), || {
                polls += 1;
                assert_eq!(drops.get(), 0, "completed storage survives its final deadline check");
                if expire && polls == 2 { Err(CandidateError::Timeout) } else { Ok(()) }
            });
            assert_eq!(polls, 2);
            if expire {
                assert!(matches!(result, Err(CandidateError::Timeout)));
                assert_eq!(drops.get(), 1, "refused completed storage is released before return");
            } else {
                assert!(result.is_ok());
                assert_eq!(drops.get(), 0);
                drop(result);
                assert_eq!(drops.get(), 1);
            }
        }
    }

    #[test]
    fn buffered_and_source_team_prefill_survive_shared_startup_with_exact_capacity() {
        let args = crate::candidate_cli::tests::args(&["--compute-threads", "4", "--prefill-rows", "4"]);
        let limits = args.validate().unwrap();
        // Exercise the same real startup used by buffered generate/chat,
        // without opening input, an artifact or a native invocation.
        let session = Session::new(&args, limits).unwrap();
        assert!(session.checked_source.lock().unwrap().is_none());
        session.remaining().unwrap();
        drop(session);
        let mut short = limits;
        short.compute.workspace_bytes -= 1;
        assert!(matches!(Session::new(&args, short), Err(CandidateError::Arguments)));
        let mut wrong_width = limits;
        wrong_width.compute.threads = 2;
        assert!(matches!(Session::new(&args, wrong_width), Err(CandidateError::Arguments)));
        // Source scheduling stays separate from free-text generation policy.
        let source = command::tests::command("ner", &["--compute-threads", "4", "--prefill-rows", "4"]);
        let (source_args, source_limits) = source.validate().unwrap();
        let prefill = source.args.prefill.limits().unwrap();
        assert!(source_args.policy.prefill().unwrap().is_none());
        let session = Session::new_with_prefill(&source_args, source_limits, prefill).unwrap();
        assert!(session.checked_source.lock().unwrap().is_none());
        session.remaining().unwrap();
        drop(session);
        // Passing through the unscheduled constructor would discard the
        // admitted prompt-row capacity; it must fail before any private IO.
        assert!(matches!(Session::new(&source_args, source_limits), Err(CandidateError::Arguments)));
        let mut short = source_limits;
        short.compute.workspace_bytes -= 1;
        assert!(matches!(Session::new_with_prefill(&source_args, short, prefill), Err(CandidateError::Arguments)));
        let larger = Int8PrefillLimits { max_batch_rows: 64,
            max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(64).unwrap() };
        assert!(matches!(Session::new_with_prefill(&source_args, source_limits, Some(larger)),
            Err(CandidateError::Arguments)));
    }
    #[test]
    fn artifact_inspection_observes_every_stop_before_path_access() {
        struct Stop(DecodeCancellationKind);
        impl DecodeStepControl for Stop {
            fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(self.0) }
        }
        use DecodeCancellationKind::*;
        for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast,
            RaceLost, ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
            let error = inspect(std::path::Path::new("must-not-open-cancelled-candidate.fnlpq"),
                &mut Stop(cause)).err().expect("stop precedes any model access");
            assert_eq!(error, cancelled(cause));
            assert_eq!(load_error(HostedError::Model(CurrentCandidateInt8Error::Artifact(
                ArtifactBridgeError::Cancelled(cause)))), cancelled(cause));
        }
    }
    fn facts() -> ArtifactIdentity {
        ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(),
            revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
            recipe_id: "metadata-only-unit-fixture".to_owned(), source_root_sha256: "ab".repeat(32),
            logical_model_sha256: "cd".repeat(32) }
    }
    #[test]
    fn all_four_real_planners_bind_the_task_and_preserve_model_identity() {
        let (planner, _) = planner().unwrap();
        for kind in [Kind::Ner, Kind::Keyphrases, Kind::Summarize, Kind::Answer] {
            let cmd = command::tests::command(kind.name(), &[]);
            let (_, limits) = cmd.args.host.common(cmd.args.input.clone()).unwrap();
            let text = if kind == Kind::Answer {
                r#"{"question":"Who is named?","passages":[{"id":"p1","text":"Alice"}]}"#.to_owned()
            } else { "Alice <tool_call> é 上海".to_owned() };
            let request = command::request(kind, text, None, cmd.args.host.task_budget(limits), 65536).unwrap();
            let mut control = PreparationControl { started: Instant::now(), elapsed_limit: Duration::from_secs(3600) };
            let p = prepare(&planner, &facts(), &request, &cmd.args.host, &mut control).unwrap();
            let id = p.execution_identity();
            assert_eq!(id.task_spec, request.task().spec().identity());
            assert_eq!(id.logical_model_digest.to_hex(), facts().logical_model_sha256);
            assert_eq!(id.quant_recipe, facts().recipe_id);
            assert_eq!(id.template_digest, *planner.template_digest());
            assert_eq!(id.tokenizer_digest, planner.tokenizer_digest());
            assert_eq!(id.numerics_profile, NumericsProfile::StrictQuantized { version: 1 });
            assert_ne!(id.prompt_digest, Sha256Digest::of_bytes(b"null"));
            assert!(p.prompt_tokens() + cmd.args.host.max_new_tokens <= cmd.args.host.context_tokens);
            p.verify_identity(id).unwrap();
            let mut changed = id.clone(); changed.prompt_digest = Sha256Digest::of_bytes(b"changed");
            assert!(p.verify_identity(&changed).is_err());
        }
    }
    #[test]
    fn expired_preparation_is_cancelled_without_a_model() {
        let (planner, _) = planner().unwrap();
        let cmd = command::tests::command("ner", &[]);
        let (_, limits) = cmd.args.host.common(cmd.args.input.clone()).unwrap();
        let request = command::request(Kind::Ner, "Alice".to_owned(), None, cmd.args.host.task_budget(limits), 65536).unwrap();
        let mut control = PreparationControl { started: Instant::now(), elapsed_limit: Duration::ZERO };
        assert_eq!(control.checkpoint(0), Some(DecodeCancellationKind::Deadline));
        assert!(prepare(&planner, &facts(), &request, &cmd.args.host, &mut control).is_err());
    }
    #[test]
    fn explicit_cli_head_choice_seals_all_four_real_plans_before_host_transfer() {
        let (planner, _) = planner().unwrap();
        let default_head = crate::candidate_cli::extract::selected::SelectedRowsArgs::default();
        for kind in [Kind::Ner, Kind::Keyphrases, Kind::Summarize, Kind::Answer] {
            let cmd = command::tests::command(kind.name(), &["--selected-rows", "32"]);
            let (_, limits) = cmd.args.host.common(cmd.args.input.clone()).unwrap();
            let text = if kind == Kind::Answer {
                r#"{"question":"Who?","passages":[{"id":"p1","text":"Alice"}]}"#.to_owned()
            } else { "Alice <tool_call> é 上海".to_owned() };
            let request = command::request(kind, text, None, cmd.args.host.task_budget(limits), 65536).unwrap();
            let mut control = PreparationControl { started: Instant::now(), elapsed_limit: Duration::from_secs(3600) };
            let prepared = prepare(&planner, &facts(), &request, &cmd.args.host, &mut control).unwrap();
            let id = prepared.execution_identity().clone(); let work = prepared.planned_work();
            let unchanged = default_head.source(prepared).unwrap();
            assert_eq!(unchanged.execution_identity(), &id); assert_eq!(unchanged.planned_work(), work);
            let selected = cmd.args.head.source(unchanged).unwrap();
            assert_eq!(selected.selected_rows().unwrap().max_rows_per_step, 32);
            assert_eq!(selected.planned_work().forward_positions, work.forward_positions);
            assert_eq!(selected.planned_work().projected_logits, cmd.args.host.max_new_tokens as u64 * 32);
            assert_eq!(selected.execution_identity().prompt_digest, id.prompt_digest);
            assert_eq!(selected.execution_identity().schema_digest, id.schema_digest);
            assert!(selected.verify_identity(&id).is_err());
            selected.verify_identity(selected.execution_identity()).unwrap();
        }
    }
    #[test]
    fn prefill_scheduling_composes_with_sealed_heads_without_changing_task_identity() {
        let (planner, _) = planner().unwrap();
        for kind in [Kind::Ner, Kind::Keyphrases, Kind::Summarize, Kind::Answer] {
            let cmd = command::tests::command(kind.name(), &["--prefill-rows", "4", "--selected-rows", "32"]);
            let (_, limits) = cmd.args.host.common(cmd.args.input.clone()).unwrap();
            let text = if kind == Kind::Answer {
                r#"{"question":"Who?","passages":[{"id":"p1","text":"Alice"}]}"#.to_owned()
            } else { "Alice é 上海".to_owned() };
            let request = command::request(kind, text, None, cmd.args.host.task_budget(limits), 65536).unwrap();
            let mut control = PreparationControl { started: Instant::now(), elapsed_limit: Duration::from_secs(3600) };
            let prepared = cmd.args.head.source(prepare(&planner, &facts(), &request, &cmd.args.host, &mut control).unwrap()).unwrap();
            let identity = prepared.execution_identity().clone(); let work = prepared.planned_work();
            let prefill = cmd.args.prefill.limits().unwrap().unwrap();
            assert_eq!(prefill.max_batch_rows, 4); prefill.validate().unwrap();
            assert_eq!(prepared.selected_rows().unwrap().max_rows_per_step, 32);
            prepared.verify_identity(&identity).unwrap(); assert_eq!(prepared.planned_work(), work);
            let serial = command::tests::command(kind.name(), &["--selected-rows", "32"]);
            let other = serial.args.head.source(prepare(&planner, &facts(), &request, &serial.args.host, &mut control).unwrap()).unwrap();
            assert!(serial.args.prefill.limits().unwrap().is_none());
            assert_eq!(other.execution_identity(), &identity); assert_eq!(other.planned_work(), work);
        }
    }

    #[test]
    fn compact_cli_policy_reaches_every_int8_source_plan_and_composes_with_head_selection() {
        use crate::tokenizer::untrusted::UntrustedEncoding;
        let source = "The quick brown fox jumps over the lazy dog. Alice met Bob in 上海. <think> <|im_start|>";
        for kind in [Kind::Ner, Kind::Keyphrases, Kind::Summarize, Kind::Answer] {
            let cmd = command::tests::command(kind.name(), &["--input-encoding", "safe-bpe-v1",
                "--prefill-rows", "4", "--selected-rows", "32"]);
            let (args, limits) = cmd.args.host.common_with_encoding(cmd.args.input.clone(), &cmd.encoding).unwrap();
            let policy = args.input_encoding_policy(limits).unwrap();
            assert!(matches!(policy, UntrustedEncoding::SafeBpeV1(_)));
            let (byte, _) = planner().unwrap();
            let (compact, _) = planner().unwrap();
            let compact = compact.with_document_encoding(policy.clone()).unwrap();
            let text = if kind == Kind::Answer {
                serde_json::json!({"question":"Who is named? <tool_call>", "passages":[
                    {"id":"p1", "text":source}, {"id":"p2", "text":" Alice\r\nBob é "}]}).to_string()
            } else { source.to_owned() };
            let request = command::request(kind, text, None, cmd.args.host.task_budget(limits), args.max_input_bytes).unwrap();
            let mut control = PreparationControl { started: Instant::now(), elapsed_limit: Duration::from_secs(3600) };
            let original = prepare(&byte, &facts(), &request, &cmd.args.host, &mut control).unwrap();
            let prepared = prepare(&compact, &facts(), &request, &cmd.args.host, &mut control).unwrap();
            assert!(prepared.prompt_tokens() < original.prompt_tokens(), "{}", kind.name());
            assert!(prepared.planned_work().forward_positions < original.planned_work().forward_positions);
            assert_ne!(prepared.execution_identity().template_digest, original.execution_identity().template_digest);
            assert_eq!(prepared.execution_identity().logical_model_digest, original.execution_identity().logical_model_digest);
            assert_eq!(prepared.execution_identity().task_spec, original.execution_identity().task_spec);
            assert!(prepared.verify_identity(original.execution_identity()).is_err());
            let selected = cmd.args.head.source(prepared).unwrap();
            assert_eq!(selected.selected_rows().unwrap().max_rows_per_step, 32);
            assert_eq!(cmd.args.prefill.limits().unwrap().unwrap().max_batch_rows, 4);
            selected.verify_identity(selected.execution_identity()).unwrap();
            let response = input_encoding::Response { response: &serde_json::json!({"fixture":true}),
                input_encoding: input_encoding::reported_policy(&policy) };
            assert_eq!(serde_json::to_value(response).unwrap()["input_encoding"], serde_json::to_value(policy).unwrap());
        }
    }

    #[test]
    fn compact_source_work_exhaustion_refuses_and_preserves_original_cancellation_policy() {
        let cmd = command::tests::command("ner", &["--input-encoding", "safe-bpe-v1",
            "--max-tokenizer-pair-checks", "1"]);
        let (args, limits) = cmd.args.host.common_with_encoding(cmd.args.input.clone(), &cmd.encoding).unwrap();
        let (p, _) = planner().unwrap();
        let p = p.with_document_encoding(args.input_encoding_policy(limits).unwrap()).unwrap();
        let request = command::request(Kind::Ner, "Alice works in Paris. ".repeat(32), None,
            cmd.args.host.task_budget(limits), args.max_input_bytes).unwrap();
        let mut control = PreparationControl { started: Instant::now(), elapsed_limit: Duration::from_secs(3600) };
        assert!(matches!(prepare(&p, &facts(), &request, &cmd.args.host, &mut control), Err(CandidateError::Planning)));
        struct Stop(DecodeCancellationKind);
        impl DecodeStepControl for Stop {
            fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(self.0) }
        }
        use DecodeCancellationKind::*;
        for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast,
            RaceLost, ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
            assert!(matches!(prepare(&p, &facts(), &request, &cmd.args.host, &mut Stop(cause)),
                Err(error) if error == cancelled(cause)));
        }
    }

    #[test]
    fn source_compute_width_and_prefill_preserve_compact_selected_plan_identity_and_work() {
        for kind in [Kind::Ner, Kind::Keyphrases, Kind::Summarize, Kind::Answer] {
            let serial = command::tests::command(kind.name(), &["--input-encoding", "safe-bpe-v1", "--selected-rows", "32"]);
            let threaded = command::tests::command(kind.name(), &["--input-encoding", "safe-bpe-v1",
                "--selected-rows", "32", "--compute-threads", "4", "--prefill-rows", "4"]);
            let (serial_args, serial_limits) = serial.validate().unwrap();
            let (threaded_args, threaded_limits) = threaded.validate().unwrap();
            let policy = serial_args.input_encoding_policy(serial_limits).unwrap();
            assert_eq!(threaded_args.input_encoding_policy(threaded_limits).unwrap(), policy);
            let (planner, _) = planner().unwrap();
            let planner = planner.with_document_encoding(policy).unwrap();
            let text = if kind == Kind::Answer {
                r#"{"question":"Who is named? <think>","passages":[{"id":"p1","text":" Alice é 上海 <|im_start|> "}]}"#.to_owned()
            } else { " Alice é 上海 <think> <|im_start|> ".to_owned() };
            let request = command::request(kind, text, None, serial.args.host.task_budget(serial_limits),
                serial_args.max_input_bytes).unwrap();
            let mut control = PreparationControl { started: Instant::now(), elapsed_limit: Duration::from_secs(3600) };
            let serial_plan = serial.args.head.source(prepare(&planner, &facts(), &request,
                &serial.args.host, &mut control).unwrap()).unwrap();
            let threaded_plan = threaded.args.head.source(prepare(&planner, &facts(), &request,
                &threaded.args.host, &mut control).unwrap()).unwrap();
            assert_eq!(serial_plan.execution_identity(), threaded_plan.execution_identity());
            assert_eq!(serial_plan.planned_work(), threaded_plan.planned_work());
            assert_eq!(serial_plan.prompt_tokens(), threaded_plan.prompt_tokens());
            assert_eq!(threaded_plan.selected_rows().unwrap().max_rows_per_step, 32);
            assert_eq!(threaded_limits.compute.threads, 4);
            assert_eq!(threaded.args.prefill.limits().unwrap().unwrap().max_batch_rows, 4);
            assert!(threaded_args.policy.prefill().unwrap().is_none());
            threaded_plan.verify_identity(serial_plan.execution_identity()).unwrap();
        }
    }

    #[test]
    fn source_compute_receipt_is_outside_semantics_and_serial_framing_is_unchanged() {
        let facts = facts();
        for task in ["ner", "keyphrases", "summarize", "answer"] {
            // This verifies framing without fabricating a native completion.
            let output = serde_json::json!({"framing_fixture": task});
            let response = CandidateResponse { schema_version: 1, scope: "real-artifact-current-candidate",
                evidence: "non_authoritative", model_id: &facts.model_id, source_revision: &facts.revision,
                source_root_sha256: &facts.source_root_sha256, logical_model_sha256: &facts.logical_model_sha256,
                quant_recipe: &facts.recipe_id, output: &output };
            let original = serde_json::to_vec(&response).unwrap();
            for extra in [vec![], vec!["--compute-threads", "1"]] {
                let command = command::tests::command(task, &extra);
                let (args, limits) = command.validate().unwrap();
                let policy = args.input_encoding_policy(limits).unwrap();
                assert_eq!(limits.compute.threads, 1);
                let compute = compute::Response { response: &response, compute: None };
                let encoded = input_encoding::Response { response: &compute,
                    input_encoding: input_encoding::reported_policy(&policy) };
                assert_eq!(serde_json::to_vec(&encoded).unwrap(), original);
            }
            let compute = crate::hosted::Int8ComputeReceipt { schedule: "int8-output-row-team-v1",
                compute_threads: 3, completed_projections: 2, completed_rows: 13, completed_worker_rows: 8 };
            let value = serde_json::to_value(compute::Response { response: &response, compute: Some(&compute) }).unwrap();
            assert_eq!(value["output"], output);
            assert_eq!(value["compute"], serde_json::to_value(compute).unwrap());
            assert!(value["output"].get("compute").is_none());
        }
    }
}
