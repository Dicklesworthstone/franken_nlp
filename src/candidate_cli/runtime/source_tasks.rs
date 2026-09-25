//! Source tasks on the existing hosted INT8 path, with one end-to-end deadline.
use super::*;
use std::sync::Arc;
use crate::{CommittedMemory,
    hosted::{ResidentInt8, SourceLimits},
    native_engine::decode::{DecodeCancellationKind, DecodeStepControl},
    tasks::{BuiltInTask, extract::ExtractionVocabulary, ir::PlanContext,
        source_planning::{SourceTaskPlanner, SourceTaskRequest, quantized::PreparedInt8SourceTask}},
};
use crate::candidate_cli::source::{self as command, SourceCommand, SourceHostArgs};

/// Must be declared before request/planner/vocabulary/output locals. Those
/// allocations then drop before the preparation charge, including on errors.
/// No uncharged resident model or independent runtime is created here.
pub(super) struct Session {
    _preparation: CommittedMemory,
    pub engine: NlpEngine,
    started: Instant,
    elapsed_limit: Duration,
}
impl Session {
    pub(super) fn new(args: &CandidateArgs, limits: Limits) -> Result<Self, CandidateError> {
        let started = Instant::now();
        let engine = NlpEngine::builder().resource_config(ResourceHostConfig {
            runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
            max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 0,
            helper_threads: 0, thread_ceiling: 2, memory_ceiling_bytes: limits.memory_bytes,
            leak_response_policy: LeakResponsePolicy::RecordAndEscalate,
        }).build().map_err(|_| CandidateError::Runtime)?;
        let lease = engine.resources().acquire_lease();
        let preparation = lease.reserve(MemoryClass::JobBuffers, limits.preparation_bytes)
            .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
        Ok(Self { _preparation: preparation, engine, started,
            elapsed_limit: Duration::from_secs(args.timeout_seconds) })
    }
    pub(super) fn remaining(&self) -> Result<Duration, CandidateError> {
        self.elapsed_limit.checked_sub(self.started.elapsed()).filter(|d| !d.is_zero())
            .ok_or(CandidateError::Timeout)
    }
    pub(super) fn read(&self, path: &std::path::Path, input: &mut impl Read, cap: usize)
        -> Result<String, CandidateError> {
        self.remaining()?;
        let text = if path.as_os_str() == "-" { read_input(input, cap)? }
            else { read_input(&mut File::open(path).map_err(|_| CandidateError::Input)?, cap)? };
        self.remaining()?;
        Ok(text)
    }
    pub(super) fn facts(&self, args: &CandidateArgs) -> Result<ArtifactIdentity, CandidateError> {
        self.remaining()?;
        let facts = {
            let source = CurrentCandidateArtifactSource::open(&args.model).map_err(|_| CandidateError::Model)?;
            source.identity().clone()
        };
        self.remaining()?;
        candidate_identity(&facts)?;
        Ok(facts)
    }
    pub(super) fn native(&self, args: &CandidateArgs) -> Result<NativeLimits, CandidateError> {
        Ok(NativeLimits { context_tokens: args.context_tokens, allocator_reserve_bytes: 64 * MIB,
            run: RunLimits { max_elapsed: self.remaining()?, max_checkpoints: args.max_checkpoints,
                cleanup_reserve_bytes: 65_536 } })
    }
    pub(super) fn load(&self, args: &CandidateArgs, limits: Limits, facts: &ArtifactIdentity,
        cancellation: CancellationToken) -> Result<ResidentInt8, CandidateError> {
        let model = self.engine.load_current_candidate_int8(args.model.clone(), LoadLimits {
            artifact: ArtifactLoadBudget::streaming_only(limits.weight_bytes, 64 * 1024),
            tokenizer_and_metadata_bytes: 128 * MIB, allocator_reserve_bytes: 64 * MIB,
            run: RunLimits { max_elapsed: self.remaining()?, max_checkpoints: 64, cleanup_reserve_bytes: 65_536 },
        }, cancellation).map_err(|_| CandidateError::Model)?;
        self.remaining()?;
        // The two independent opens may see different files. Include source
        // root and recipe, not just logical weights, in the comparison.
        if model.artifact_identity() != facts { return Err(CandidateError::Identity); }
        Ok(model)
    }
    pub(super) fn control(&self) -> PreparationControl {
        PreparationControl { started: self.started, elapsed_limit: self.elapsed_limit }
    }
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
        .map_err(|_| CandidateError::Planning)?;
    if prepared.planned_work().forward_positions > host.context_tokens as u64 {
        return Err(CandidateError::Planning);
    }
    Ok(prepared)
}

pub(in crate::candidate_cli) fn execute(command: SourceCommand, args: CandidateArgs, limits: Limits,
    input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
    let session = Session::new(&args, limits)?;
    let text = session.read(&command.args.input, input, args.max_input_bytes)?;
    let options = command.args.options.as_ref()
        .map(|path| session.read(path, input, command::OPTIONS_BYTES)).transpose()?;
    let budget = command.args.host.task_budget(limits);
    let request = command::request(command.kind, text, options.as_deref(), budget, args.max_input_bytes)?;
    session.remaining()?;
    let facts = session.facts(&args)?;
    let (planner, vocabulary) = planner()?;
    session.remaining()?;
    let prepared = prepare(&planner, &facts, &request, &command.args.host, &mut session.control())?;
    session.remaining()?;
    let cancellation = CancellationToken::default();
    let model = session.load(&args, limits, &facts, cancellation.clone())?;
    let result = session.engine.execute_int8_source(&model, prepared, Arc::new(vocabulary), SourceLimits {
        native: session.native(&args)?, preparation_reserve_bytes: limits.preparation_bytes,
        mask_limits: command.args.host.masks(), max_mask_node_visits: command.args.host.max_mask_node_visits,
    }, cancellation).map_err(|_| CandidateError::Execution)?;
    session.remaining()?;
    let response = CandidateResponse { schema_version: 1, scope: "real-artifact-current-candidate",
        evidence: "non_authoritative", model_id: &facts.model_id, source_revision: &facts.revision,
        source_root_sha256: &facts.source_root_sha256, logical_model_sha256: &facts.logical_model_sha256,
        quant_recipe: &facts.recipe_id, output: &result };
    // No result prefix is written until native/source finalization completes.
    // Both the output guard and preparation owner survive write AND flush.
    publish(&response, command.args.host.max_result_bytes + 4096, output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate_cli::source::Kind;
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
}
