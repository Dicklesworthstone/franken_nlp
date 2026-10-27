//! Bounded raw-document epochs on one resident INT4 model and native engine.
//! The shared parser owns framing; this adapter owns real-model admission,
//! lifetime work allowances, complete snapshot delivery and physical drain.

use super::*;
use std::{io::{BufRead, Write}, time::Instant};
use serde::Serialize;
use crate::{
    canonjson,
    corpus::{
        entities::{EntityDocument, EntityVerificationWork},
        entities_int4::{Int4EntityConfig, Int4EntityError, Int4EntityRun, prepare_int4_entities},
        entity_stream::{self, EntitySnapshotProcessor, EntityStreamError, EntityStreamLimits,
            EntityStreamSummary, ENTITY_STREAM_PROTOCOL},
    },
    native_engine::{decode::DecodeCancellationKind, portable_int8::ProjectionWork},
    tasks::source_planning::SourceTaskPlanner,
    validation::grounded_fields::GroundingBudget,
};

// This capacity is unavailable to epoch records, including the last one.
const FOOTER_BYTES: u64 = 16_384;

/// Native work (including the scoring ceiling), masks and grounding apply to
/// the entire stream. Graph/input/chunk/result capacities apply to each epoch;
/// transport counts, bytes, checkpoints and time also bound the entire run.
/// Preparation prices pinned planners/vocabulary, policy and compiler storage;
/// graph prices expanded mentions/pair plans; IO prices the caller's buffers.
pub struct Int4EntityCorpusHostConfig {
    pub source_identity: ExecutionIdentity,
    pub resolution_identity: ExecutionIdentity,
    pub task: Int4EntityConfig,
    pub transport: EntityStreamLimits,
    pub native: NativeLimits,
    pub preparation_reserve_bytes: u64,
    pub graph_reserve_bytes: u64,
    pub io_reserve_bytes: u64,
}
impl Int4EntityCorpusHostConfig {
    pub fn requirements(&self, group_width: usize) -> Result<Int4EntityCorpusHostRequirements, HostedError> {
        Int4EntityCorpusHostRequirements::for_config(&self.task, self.transport, self.native,
            self.preparation_reserve_bytes, self.graph_reserve_bytes, self.io_reserve_bytes, group_width)
    }
}

/// Exact modeled host reservations, excluding the already resident weights
/// and any caller-owned preparation. These are commitments, not measured RSS.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4EntityCorpusHostRequirements {
    pub retained_bytes: u64,
    pub temporary_bytes: u64,
    pub output_bytes: u64,
    pub kv_bytes: u64,
    pub scratch_bytes: u64,
    pub cleanup_bytes: u64,
    native: Int4MemoryRequirement,
    input_payload_bytes: u64,
}
impl Int4EntityCorpusHostRequirements {
    /// Identity-independent admission for use before opening source/model or
    /// input files. Reuses the actual parser and task-policy validators.
    #[allow(clippy::too_many_arguments)]
    pub fn for_config(task: &Int4EntityConfig, transport: EntityStreamLimits, native: NativeLimits,
        preparation: u64, graph: u64, io: u64, width: usize) -> Result<Self, HostedError> {
        entity_stream::validate::<HostedError>(transport)
            .map_err(|_| HostedError::Limits("INT4 entity corpus transport"))?;
        task.graph.validate(task.resolution).map_err(|error| entities::execution_error(error.into()))?;
        task.source_planning.validate().map_err(|_| HostedError::Limits("INT4 entity source planning"))?;
        let (required, scratch_bytes) = entities::requirements(task, 0, native, preparation, graph, width)?;
        let document_cap = task.chunks.map_or(task.source_planning.max_input_bytes, |chunks| chunks.max_input_bytes);
        if io == 0 || transport.max_output_bytes <= FOOTER_BYTES
            || transport.max_documents > task.graph.max_documents
            || transport.max_corpus_bytes > task.graph.max_input_bytes
            || transport.max_document_bytes > document_cap {
            return Err(HostedError::Limits("INT4 entity corpus input, IO or terminal capacity"));
        }
        // Original sources, duplicated ID index, Vec capacity and compact NER
        // witnesses are separate from the caller's compiler/planner reserve.
        // The parser allowance also prices hostile bounded JSON trees that
        // will be rejected before any document enters native preparation.
        let input_payload_bytes = sum(&[
            product(transport.max_corpus_bytes as u64, 4)?,
            product(transport.max_documents as u64, 512)?,
            product(task.max_snapshot_chunks as u64, 256)?,
        ])?;
        let retained_bytes = sum(&[input_payload_bytes, product(transport.max_line_bytes as u64, 32)?,
            preparation, io])?;
        let output_bytes = sum(&[output_reserve(transport.max_output_line_bytes.max(task.max_result_bytes) as u64)?,
            output_reserve(FOOTER_BYTES)?])?;
        let result = Self { retained_bytes, temporary_bytes: entities::temporary_bytes(task, graph)?,
            output_bytes, kv_bytes: required.kv_bytes, scratch_bytes,
            cleanup_bytes: native.run.cleanup_reserve_bytes, native: required, input_payload_bytes };
        result.total_bytes()?;
        Ok(result)
    }
    pub fn total_bytes(self) -> Result<u64, HostedError> {
        sum(&[self.retained_bytes, self.temporary_bytes, self.output_bytes,
            self.kv_bytes, self.scratch_bytes, self.cleanup_bytes])
    }
}
fn product(a: u64, b: u64) -> Result<u64, HostedError> {
    a.checked_mul(b).ok_or(HostedError::Limits("INT4 entity corpus memory arithmetic"))
}
fn output_reserve(bytes: u64) -> Result<u64, HostedError> { sum(&[product(bytes, 4)?, 4096]) }

/// Transport output_bytes counts delivered epoch records, excluding the final
/// footer. Counts are acknowledged only after each complete write AND flush.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct Int4EntityCorpusSummary {
    pub transport: EntityStreamSummary,
    pub reserved_model_work: Int4Work,
    pub model_work: Int4Work,
    pub reserved_mask_node_visits: u64,
    pub mask_node_visits: u64,
    pub verification_used: EntityVerificationWork,
}

/// No private source or writer diagnostic is included in Display/Debug.
pub enum Int4EntityCorpusError {
    Host(HostedError),
    Transport(EntityStreamError<HostedError>),
    OutputIo,
    Completion,
}
impl fmt::Display for Int4EntityCorpusError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Host(_) => "INT4 entity corpus host or native execution refused",
            Self::Transport(_) => "INT4 entity corpus input, epoch or delivery refused",
            Self::OutputIo => "INT4 entity corpus terminal delivery is unknown",
            Self::Completion => "INT4 entity corpus terminal completion refused",
        })
    }
}
impl fmt::Debug for Int4EntityCorpusError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Display::fmt(self, f) }
}
impl Error for Int4EntityCorpusError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Host(e) => Some(e), Self::Transport(e) => Some(e), _ => None }
    }
}
impl From<HostedError> for Int4EntityCorpusError { fn from(error: HostedError) -> Self { Self::Host(error) } }
impl Int4EntityCorpusError {
    pub fn cancellation(&self) -> Option<DecodeCancellationKind> {
        match self {
            Self::Host(error) | Self::Transport(EntityStreamError::Handler(error)) => host_cancellation(error),
            Self::Transport(EntityStreamError::Resolution(crate::corpus::resolve::ResolveError::Cancelled(cause))) => Some(*cause),
            _ => None,
        }
    }
}
fn host_cancellation(error: &HostedError) -> Option<DecodeCancellationKind> {
    match error {
        HostedError::Stopped { stop, .. } => Some(stop.kind),
        HostedError::Join { physical_error, .. } | HostedError::Scope { physical_error, .. } =>
            physical_error.as_deref().and_then(host_cancellation),
        HostedError::Int4(Int4HostError::Entities(error)) => error.cancellation(),
        HostedError::Int4(Int4HostError::Native(StrictInt4Error::Cancelled(cause))) => Some(*cause),
        _ => None,
    }
}

#[derive(Clone, Serialize)]
struct Provenance {
    scope: &'static str,
    evidence: &'static str,
    task: &'static str,
    model_id: &'static str,
    source_revision: &'static str,
    source_root_sha256: String,
    model_content_sha256: String,
    candidate_format: String,
    quant_recipe: String,
    group_width: usize,
    integer_kernel: &'static str,
    cluster_scope: &'static str,
}
impl Provenance {
    fn new(model: &Int4ModelIdentity, identity: &ExecutionIdentity, kernel: PackedInt4Kernel) -> Self {
        Self { scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "resolve-corpus",
            model_id: "Nanbeige4.2-3B", source_revision: model.revision(),
            source_root_sha256: model.source_root_sha256().to_owned(),
            model_content_sha256: identity.logical_model_digest.to_hex(),
            candidate_format: identity.artifact_format.clone(), quant_recipe: identity.quant_recipe.clone(),
            group_width: model.group_width(), integer_kernel: kernel.kind().id(), cluster_scope: "epoch_local" }
    }
}
#[derive(Serialize)]
struct EpochResult {
    #[serde(flatten)]
    provenance: Provenance,
    output: Int4EntityRun,
}

struct Input<R, W> {
    reader: Option<R>, writer: Option<W>,
    source: Arc<SourceTaskPlanner>, resolver: Arc<ResolutionPlanner>, vocabulary: Arc<ExtractionVocabulary>,
    config: Int4EntityCorpusHostConfig,
}
struct Completed<W> {
    writer: W,
    summary: Int4EntityCorpusSummary,
    provenance: Provenance,
    max_output_bytes: u64,
    deadline: Instant,
    cancellation: CancellationToken,
}

/// Returned only after the hosted invocation physically drains. The original
/// writer and its reservations stay owned until terminal write/flush completes.
/// Dropping this object intentionally leaves the run without a success footer.
pub struct HostedInt4EntityCorpus<W> {
    completed: Charged<Completed<W>>,
    _output: Arc<CommittedMemory>,
}
impl<W: Write> HostedInt4EntityCorpus<W> {
    pub fn summary(&self) -> &Int4EntityCorpusSummary { &self.completed.value.summary }

    /// Publish the required same-protocol run_complete event exactly once.
    /// Clusters in earlier corpus records remain scoped to their own epoch;
    /// this footer acknowledges the finite run, never a cross-epoch graph.
    /// Failure consumes the writer and cannot trigger retry or another record.
    pub fn finish(mut self) -> Result<Int4EntityCorpusSummary, Int4EntityCorpusError> {
        let completed = &mut self.completed.value;
        terminal_checkpoint(completed.deadline, &completed.cancellation)?;
        let bytes = footer_bytes(&completed.provenance, &completed.summary, completed.max_output_bytes)?;
        terminal_checkpoint(completed.deadline, &completed.cancellation)?;
        completed.writer.write_all(&bytes).and_then(|_| completed.writer.flush())
            .map_err(|_| Int4EntityCorpusError::OutputIo)?;
        Ok(completed.summary)
    }
}
fn terminal_checkpoint(deadline: Instant, cancellation: &CancellationToken) -> Result<(), Int4EntityCorpusError> {
    let cause = cancellation.cause().or_else(|| (Instant::now() >= deadline).then_some(DecodeCancellationKind::Deadline));
    match cause {
        Some(kind) => Err(Int4EntityCorpusError::Host(HostedError::Stopped { stop: RunStop {
            kind, runtime_reason: None, runtime_checkpoint_error: None }, task_error: None })),
        None => Ok(()),
    }
}
fn footer_bytes(provenance: &Provenance, summary: &Int4EntityCorpusSummary, cap: u64)
    -> Result<Vec<u8>, Int4EntityCorpusError> {
    #[derive(Serialize)]
    struct Footer<'a> {
        protocol: &'static str, schema_version: u32, event: &'static str,
        epoch: u64, through_request_seq: u64, eof: bool,
        provenance: &'a Provenance, summary: &'a Int4EntityCorpusSummary,
    }
    let footer = Footer { protocol: ENTITY_STREAM_PROTOCOL, schema_version: 1, event: "run_complete",
        epoch: summary.transport.snapshots, through_request_seq: summary.transport.records, eof: true,
        provenance, summary };
    // Provenance has fixed bounded model facts and summaries contain counters
    // only. Still size-check before canonical tree/byte staging and first IO.
    let mut counter = ByteCounter { remaining: FOOTER_BYTES - 1 };
    serde_json::to_writer(&mut counter, &footer).map_err(|_| Int4EntityCorpusError::Completion)?;
    let mut bytes = canonjson::canonical_bytes(&footer).map_err(|_| Int4EntityCorpusError::Completion)?;
    if bytes.len() as u64 >= FOOTER_BYTES { return Err(Int4EntityCorpusError::Completion); }
    bytes.try_reserve_exact(1).map_err(|_| Int4EntityCorpusError::Completion)?;
    bytes.push(b'\n');
    if summary.transport.output_bytes.checked_add(bytes.len() as u64).is_none_or(|total| total > cap) {
        return Err(Int4EntityCorpusError::Completion);
    }
    Ok(bytes)
}
struct ByteCounter { remaining: u64 }
impl Write for ByteCounter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.remaining = self.remaining.checked_sub(bytes.len() as u64)
            .ok_or_else(|| std::io::Error::other("INT4 entity corpus terminal capacity"))?;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
}

impl NlpEngine {
    /// Execute exact {id,text} and {flush:true} NDJSON records on one loaded
    /// model. All parsing, NER, graph scoring and writes share one hosted run;
    /// the native engine is allocated once, lazily for the first nonempty epoch.
    /// An empty flush is a real empty snapshot; EOF emits only pending docs.
    /// Every nonempty document executes NER even when it yields zero mentions.
    ///
    /// Results appear only after an entire epoch succeeds, and must be treated
    /// as epoch-local. A failed later record or delivery prevents run_complete.
    /// Arbitrary blocking readers/writers provide backpressure but cannot be
    /// preempted inside fill_buf/write/flush; no detached IO worker is created.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int4_entity_corpus<R, W>(&self, model: &ResidentInt4,
        source: Arc<SourceTaskPlanner>, resolver: Arc<ResolutionPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int4EntityCorpusHostConfig, kernel: PackedInt4Kernel, reader: R, writer: W,
        cancellation: CancellationToken) -> Result<HostedInt4EntityCorpus<W>, Int4EntityCorpusError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        dispatch::preflight(self, config.native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain.into()); }
        constrained_int4::check_model(&config.source_identity, model.model_identity())
            .map_err(|_| HostedError::ModelIdentity)?;
        constrained_int4::check_model(&config.resolution_identity, model.model_identity())
            .map_err(|_| HostedError::ModelIdentity)?;
        let required = config.requirements(model.model_identity().group_width())?;
        let run = config.native.run;
        let deadline = Instant::now().checked_add(run.max_elapsed).ok_or(HostedError::Limits("deadline arithmetic"))?;
        let terminal_cancellation = cancellation.clone();
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.retained_bytes)?,
            || Ok(Input { reader: Some(reader), writer: Some(writer), source, resolver, vocabulary, config }))?;
        let temporary = Pending::reserve(&lease, MemoryClass::JobBuffers, required.temporary_bytes)?;
        let output = Pending::reserve(&lease, MemoryClass::JobBuffers, required.output_bytes)?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.scratch_bytes)?;
        let model = model.clone();
        let completed = dispatch::run(self, run, cancellation, move |control| {
            // Capture the WHOLE charged input through queued cancellation,
            // native execution, writer callbacks and the physical return.
            let mut input = input;
            let config = &input.value.config;
            // Authenticate both planners/policies even for an empty stream,
            // before a single input byte is consumed.
            let empty = prepare_int4_entities(Vec::new(), input.value.source.clone(), config.source_identity.clone(),
                input.value.resolver.clone(), config.resolution_identity.clone(), config.task.clone(),
                model.model_identity(), control).map_err(entities::execution_error)?;
            drop(empty);
            let output = Arc::new(output.commit()?);
            let provenance = Provenance::new(model.model_identity(), &config.resolution_identity, kernel);
            let mut processor = Processor { model: &model.inner.loaded.value, engine: None,
                source: input.value.source.clone(), resolver: input.value.resolver.clone(),
                vocabulary: input.value.vocabulary.clone(), config, required, kernel,
                kv: Some(kv), scratch: Some(scratch), output: Arc::clone(&output),
                provenance: provenance.clone(), ledger: Ledger::new(&config.task) };
            let mut transport = config.transport;
            transport.max_output_bytes -= FOOTER_BYTES;
            let result = entity_stream::run_ndjson(
                input.value.reader.as_mut().ok_or(HostedError::CompletionMissing)?,
                input.value.writer.as_mut().ok_or(HostedError::CompletionMissing)?,
                &mut processor, transport, control);
            let summary = processor.ledger.summary;
            // Engine, its pending/committed KV and workspace, and all private
            // snapshot state drop before the physical scope can complete.
            drop(processor);
            drop(temporary);
            match result {
                Err(EntityStreamError::Handler(error)) => Err(error),
                Err(EntityStreamError::Resolution(crate::corpus::resolve::ResolveError::Cancelled(cause))) =>
                    Err(entities::execution_error(Int4EntityError::Graph(crate::corpus::resolve::ResolveError::Cancelled(cause)))),
                Err(error) => Ok(Err(error)),
                Ok(transport) => {
                    let completed = Completed { writer: input.value.writer.take().ok_or(HostedError::CompletionMissing)?,
                        summary: Int4EntityCorpusSummary { transport, ..summary }, provenance,
                        max_output_bytes: input.value.config.transport.max_output_bytes, deadline,
                        cancellation: terminal_cancellation };
                    // Dispose input/planner storage before transferring its
                    // still-live guard to the writer/terminal completion.
                    drop(input.value);
                    drop(lease);
                    Ok(Ok(HostedInt4EntityCorpus { completed: Charged { value: completed, _memory: input._memory }, _output: output }))
                }
            }
        })?;
        completed.map_err(Int4EntityCorpusError::Transport)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum State { Ready, Running, AwaitingDelivery, Failed }
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Allowance { work: Int4Work, scoring: Int4Work, masks: u64, verification: GroundingBudget }
#[derive(Clone, Copy)]
struct Charge {
    reserved: Int4Work, actual: Int4Work, scoring: Int4Work,
    masks_reserved: u64, masks_actual: u64, verification: EntityVerificationWork,
}
struct Ledger { remaining: Allowance, state: State, summary: Int4EntityCorpusSummary }
impl Ledger {
    fn new(task: &Int4EntityConfig) -> Self {
        Self { remaining: Allowance { work: task.max_model_work, scoring: task.scoring.max_model_work,
            masks: task.masks.max_visits_per_run, verification: task.verification },
            state: State::Ready, summary: Int4EntityCorpusSummary::default() }
    }
    fn begin(&mut self) -> Result<Allowance, Int4EntityError> {
        if self.state != State::Ready { self.state = State::Failed; return Err(Int4EntityError::Accounting); }
        self.state = State::Running;
        Ok(self.remaining)
    }
    fn complete(&mut self, charge: Charge) -> Result<(), Int4EntityError> {
        let previous = std::mem::replace(&mut self.state, State::Failed);
        if previous != State::Running { return Err(Int4EntityError::Accounting); }
        subtract_work(charge.reserved, charge.actual)?;
        if charge.masks_actual > charge.masks_reserved { return Err(Int4EntityError::Accounting); }
        let sub = |a: u64, b: u64| a.checked_sub(b).ok_or(Int4EntityError::WorkBudget);
        let v = self.remaining.verification;
        let next = Allowance { work: subtract_work(self.remaining.work, charge.reserved)?,
            scoring: subtract_work(self.remaining.scoring, charge.scoring)?,
            masks: sub(self.remaining.masks, charge.masks_reserved)?,
            verification: GroundingBudget {
                max_fields: v.max_fields.checked_sub(charge.verification.fields).ok_or(Int4EntityError::WorkBudget)?,
                max_matches: v.max_matches.checked_sub(charge.verification.matches).ok_or(Int4EntityError::WorkBudget)?,
                max_scan_steps: sub(v.max_scan_steps, charge.verification.scan_steps)?,
            } };
        let previous = self.summary;
        let add = |a: u64, b: u64| a.checked_add(b).ok_or(Int4EntityError::Accounting);
        let summary = Int4EntityCorpusSummary { transport: previous.transport,
            reserved_model_work: previous.reserved_model_work.checked_add(charge.reserved)?,
            model_work: previous.model_work.checked_add(charge.actual)?,
            reserved_mask_node_visits: add(previous.reserved_mask_node_visits, charge.masks_reserved)?,
            mask_node_visits: add(previous.mask_node_visits, charge.masks_actual)?,
            verification_used: EntityVerificationWork {
                fields: previous.verification_used.fields.checked_add(charge.verification.fields).ok_or(Int4EntityError::Accounting)?,
                matches: previous.verification_used.matches.checked_add(charge.verification.matches).ok_or(Int4EntityError::Accounting)?,
                scan_steps: add(previous.verification_used.scan_steps, charge.verification.scan_steps)?,
            } };
        // Stage every counter before mutation. Native/mask reservations are
        // never refunded, including unused early-EOS capacity after delivery.
        self.remaining = next; self.summary = summary; self.state = State::AwaitingDelivery;
        Ok(())
    }
    fn acknowledge(&mut self) -> Result<(), Int4EntityError> {
        let previous = std::mem::replace(&mut self.state, State::Failed);
        if previous != State::AwaitingDelivery { return Err(Int4EntityError::Accounting); }
        self.state = State::Ready;
        Ok(())
    }
    fn abandon(&mut self) { self.state = State::Failed; }
}
fn subtract_work(a: Int4Work, b: Int4Work) -> Result<Int4Work, Int4EntityError> {
    let sub = |a: u64, b: u64| a.checked_sub(b).ok_or(Int4EntityError::WorkBudget);
    Ok(Int4Work { forward_positions: sub(a.forward_positions, b.forward_positions)?,
        projected_logits: sub(a.projected_logits, b.projected_logits)?,
        attention_pairs: sub(a.attention_pairs, b.attention_pairs)?,
        projections: ProjectionWork { dot_products: sub(a.projections.dot_products, b.projections.dot_products)?,
            multiply_accumulates: sub(a.projections.multiply_accumulates, b.projections.multiply_accumulates)? } })
}

struct Processor<'p, 'm> {
    model: &'m Int4ModelWeights,
    engine: Option<ChargedNative<StrictInt4Engine<'m>>>,
    source: Arc<SourceTaskPlanner>, resolver: Arc<ResolutionPlanner>, vocabulary: Arc<ExtractionVocabulary>,
    config: &'p Int4EntityCorpusHostConfig,
    required: Int4EntityCorpusHostRequirements,
    kernel: PackedInt4Kernel,
    kv: Option<Pending>, scratch: Option<Pending>, output: Arc<CommittedMemory>,
    provenance: Provenance, ledger: Ledger,
}
impl EntitySnapshotProcessor for Processor<'_, '_> {
    type Output = GuardedOutput<EpochResult, Arc<CommittedMemory>>;
    type Error = HostedError;
    fn execute_snapshot<C: DecodeStepControl>(&mut self, documents: Vec<EntityDocument>, control: &mut C)
        -> Result<Self::Output, Self::Error> {
        let remaining = self.ledger.begin().map_err(entities::execution_error)?;
        let empty = documents.is_empty();
        let task = narrowed_task(&self.config.task, remaining, empty).map_err(entities::execution_error)?;
        let prepared = prepare_int4_entities(documents, self.source.clone(), self.config.source_identity.clone(),
            self.resolver.clone(), self.config.resolution_identity.clone(), task, self.model.identity(), control)
            .map_err(entities::execution_error)?;
        let expected = entities::Expected { width: self.model.identity().group_width(), documents: prepared.document_count(),
            chunks: prepared.chunk_count(), ner_work: prepared.ner_reserved_work(), masks: prepared.reserved_mask_visits(),
            max_model_work: remaining.work, chunked: self.config.task.chunks.is_some() };
        if prepared.required_ner_context_tokens() > self.config.native.context_tokens
            || prepared.retained_input_bytes().map_err(entities::execution_error)? > self.required.input_payload_bytes {
            return Err(HostedError::Limits("INT4 entity corpus retained input or context"));
        }
        let result = if empty {
            prepared.finalize_without_model(control).map_err(entities::execution_error)?
        } else {
            if self.engine.is_none() {
                let kv = self.kv.take().ok_or(HostedError::CompletionMissing)?;
                let scratch = self.scratch.take().ok_or(HostedError::CompletionMissing)?;
                let required = self.required.native;
                self.engine = Some(allocate_native(kv, scratch, || {
                    StrictInt4Engine::new(self.model, self.config.native.context_tokens, Int4MemoryBudget {
                        max_kv_bytes: required.kv_bytes, max_rope_bytes: required.rope_bytes,
                        max_scratch_payload_bytes: required.scratch_payload_bound,
                    }).map(|engine| engine.with_kernel(self.kernel)).map_err(native_error)
                })?);
            }
            let engine = self.engine.as_mut().ok_or(HostedError::CompletionMissing)?;
            let result = prepared.execute_with_control(&mut engine.value, &self.vocabulary, control)
                .map_err(entities::execution_error)?;
            if engine.value.is_poisoned() || !engine.value.kv_cache().all_slots_have_len(0) {
                return Err(entities::execution_error(Int4EntityError::Accounting));
            }
            result
        };
        entities::check_result(&result, expected)?;
        self.ledger.complete(Charge { reserved: result.reserved_model_work, actual: result.model_work,
            scoring: result.resolution.planned_model_work, masks_reserved: result.reserved_mask_node_visits,
            masks_actual: result.mask_node_visit_charge, verification: result.verification_used })
            .map_err(entities::execution_error)?;
        Ok(GuardedOutput::new(EpochResult { provenance: self.provenance.clone(), output: result }, Arc::clone(&self.output)))
    }
    fn acknowledge_snapshot(&mut self) -> Result<(), Self::Error> {
        self.ledger.acknowledge().map_err(entities::execution_error)
    }
    fn abandon_snapshot(&mut self) { self.ledger.abandon(); }
}
fn narrowed_task(base: &Int4EntityConfig, remaining: Allowance, empty: bool) -> Result<Int4EntityConfig, Int4EntityError> {
    let mut task = base.clone();
    task.max_model_work = remaining.work;
    task.scoring.max_model_work = remaining.scoring;
    task.masks.max_visits_per_run = remaining.masks;
    task.verification = remaining.verification;
    // The underlying fixed-policy validator requires positive grounding caps.
    // An empty snapshot does no grounding; permit its zero charge even after
    // an earlier epoch exhausted an axis. A nonempty epoch cannot borrow one.
    if empty {
        task.verification.max_fields = task.verification.max_fields.max(1);
        task.verification.max_matches = task.verification.max_matches.max(1);
        task.verification.max_scan_steps = task.verification.max_scan_steps.max(1);
    } else if task.verification.max_fields == 0 || task.verification.max_matches == 0
        || task.verification.max_scan_steps == 0 {
        return Err(Int4EntityError::WorkBudget);
    }
    Ok(task)
}

#[cfg(test)]
mod tests;
