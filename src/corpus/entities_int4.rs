//! Raw original documents -> bounded native INT4 NER -> one complete graph.
//!
//! Every source chunk is compiled before inference; only compact private
//! witnesses survive preparation. One grammar/result is rebuilt at a time.
//! All NER reservations, including unused early-EOS capacity, are subtracted
//! before pair scoring receives the remainder of the same corpus allowance.
//! Intermediate mentions or partial clusters are never successful output.

use std::{error::Error, fmt, mem::size_of, sync::Arc};
use serde::Serialize;
use crate::{
    batch::source::SourceMaskBudget,
    canonjson,
    execution_identity::{ExecutionIdentity, Sha256Digest},
    native_engine::{constrained::JsonWorkBudget, constrained_int4::{self, Int4JsonBudget},
        decode::{DecodeCancellationKind, DecodeStepControl}, kv::KV_BYTES_PER_TOKEN,
        portable_int8::ProjectionWork,
        strict_int4::{Int4ModelIdentity, Int4RunBudget, Int4Work, StrictInt4Engine, StrictInt4Error,
            STRICT_INT4_PROFILE, scoring::Int4ScoringBudget}},
    tasks::{extract::ExtractionVocabulary, ir::{PlanContext, TaskBudget},
        mapreduce::{ChunkLimits, ChunkPlan, MapReduceError, CHUNK_PROFILE},
        ner::{NerOptions, NerResult, NER_TASK_VERSION},
        source_planning::{SourcePlanningLimits, SourceTaskPlanner, SourceTaskRequest, SourceTaskResult,
            grouped_int4::{Int4SourceError, Int4SourceTaskRun, PreparedInt4SourceTask, INT4_SOURCE_EXECUTION,
                long::SourceMapTask}}},
    validation::grounded_fields::{GroundingBudget, VerifiedSourceSpan},
};
use super::{
    entities::{EntityDocument, EntityVerificationWork},
    native_resolve::{ResolutionPlanner, grouped_int4::{self as resolution,
        Int4ResolveError, Int4ResolveLimits, Int4ResolutionRun}},
    resolve::{self, ResolutionDocument, ResolutionPlan, ResolveError, ResolveLimits, ResolveOptions},
};
mod grounding;

pub const INT4_ENTITY_EXECUTION: &str = "grouped-int4-original-document-ner-to-complete-resolution-v1";

/// NER task/output and per-item masks apply to each declared source chunk.
/// Native work, total masks, grounding and graph limits cover the whole corpus.
/// Without chunks, each complete document must fit the single-context planner.
#[derive(Clone, Debug)]
pub struct Int4EntityConfig {
    pub ner: NerOptions,
    pub ner_budget: TaskBudget,
    pub source_planning: SourcePlanningLimits,
    pub masks: SourceMaskBudget,
    pub resolution: ResolveOptions,
    pub graph: ResolveLimits,
    pub scoring: Int4ResolveLimits,
    pub verification: GroundingBudget,
    pub max_model_work: Int4Work,
    pub max_result_bytes: usize,
    pub chunks: Option<ChunkLimits>,
    pub max_snapshot_chunks: usize,
}
impl Int4EntityConfig {
    pub fn validate(&self) -> Result<(), Int4EntityError> {
        self.ner.validate().map_err(|_| Int4EntityError::InvalidLimits)?;
        self.ner_budget.validate().map_err(|_| Int4EntityError::InvalidLimits)?;
        self.scoring.validate()?;
        let v = self.verification;
        if self.masks.max_visits_per_item == 0 || self.masks.per_mask.max_trie_node_visits == 0
            || self.masks.per_mask.checkpoint_interval_nodes == 0
            || !(1..=64 * 1024 * 1024).contains(&self.max_result_bytes)
            || !(1..=16_384).contains(&self.max_snapshot_chunks)
            || !(1..=1_000_000).contains(&v.max_fields) || !(1..=1_000_000).contains(&v.max_matches)
            || v.max_scan_steps == 0 {
            return Err(Int4EntityError::InvalidLimits);
        }
        if let Some(chunks) = self.chunks {
            chunks.effective_token_limit().map_err(partition_error)?;
            if !(1..=256).contains(&chunks.max_chunks) || chunks.max_input_bytes == 0
                || chunks.max_chunk_bytes > self.source_planning.max_input_bytes
                || chunks.context_tokens > self.source_planning.max_context_tokens {
                return Err(Int4EntityError::InvalidLimits);
            }
        }
        Ok(())
    }
}

/// No source text, document ID, generated JSON or private witness in defaults.
pub enum Int4EntityError {
    InvalidLimits, InvalidInput, Identity, WorkBudget, Accounting, Allocation,
    Source(Int4SourceError), Graph(ResolveError), Resolution(Int4ResolveError), Native(StrictInt4Error),
}
impl fmt::Display for Int4EntityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidLimits => "invalid int4 entity snapshot limits",
            Self::InvalidInput => "int4 entity snapshot requires bounded unique nonempty original documents",
            Self::Identity => "int4 entity stage or actual model identity differs",
            Self::WorkBudget => "int4 entity whole-snapshot allowance exceeded",
            Self::Accounting => "int4 entity complete execution diverged from its plan",
            Self::Allocation => "int4 entity bounded allocation refused",
            Self::Source(_) => "int4 entity source NER failed",
            Self::Graph(_) => "int4 entity occurrence verification or graph refused",
            Self::Resolution(_) => "int4 entity complete pair scoring failed",
            Self::Native(_) => "int4 entity native engine refused",
        })
    }
}
impl fmt::Debug for Int4EntityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Display::fmt(self, f) }
}
impl Error for Int4EntityError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Source(e) => Some(e), Self::Graph(e) => Some(e),
            Self::Resolution(e) => Some(e), Self::Native(e) => Some(e), _ => None }
    }
}
impl From<Int4SourceError> for Int4EntityError { fn from(e: Int4SourceError) -> Self { Self::Source(e) } }
impl From<ResolveError> for Int4EntityError { fn from(e: ResolveError) -> Self { Self::Graph(e) } }
impl From<Int4ResolveError> for Int4EntityError { fn from(e: Int4ResolveError) -> Self { Self::Resolution(e) } }
impl From<StrictInt4Error> for Int4EntityError { fn from(e: StrictInt4Error) -> Self { Self::Native(e) } }
impl Int4EntityError {
    pub fn cancellation(&self) -> Option<DecodeCancellationKind> {
        match self {
            Self::Graph(ResolveError::Cancelled(c)) | Self::Native(StrictInt4Error::Cancelled(c)) => Some(*c),
            Self::Source(e) => e.cancellation(), Self::Resolution(e) => e.cancellation(), _ => None,
        }
    }
}

struct ChunkWitness { span: VerifiedSourceSpan, identity: Sha256Digest, prompt_tokens: usize, work: Int4Work }
struct Input { document: EntityDocument, chunks: Vec<ChunkWitness> }
struct CompiledInputs {
    inputs: Vec<Input>, ner_work: Int4Work, mask_visits: u64,
    chunks: usize, maximum_positions: usize, input_bytes: usize,
}
/// Consumed original source ownership with exact private chunk witnesses.
/// The model identity is cloned from actual immutable weights; no caller can
/// construct or deserialize an unchecked plan or a model-free nonempty result.
pub struct PreparedInt4EntityCorpus {
    compiled: CompiledInputs,
    source: Arc<SourceTaskPlanner>, resolver: Arc<ResolutionPlanner>,
    source_identity: ExecutionIdentity, resolution_identity: ExecutionIdentity,
    config: Int4EntityConfig, model: Int4ModelIdentity,
}

#[allow(clippy::too_many_arguments)]
pub fn prepare_int4_entities<C: DecodeStepControl>(documents: Vec<EntityDocument>,
    source: Arc<SourceTaskPlanner>, source_identity: ExecutionIdentity,
    resolver: Arc<ResolutionPlanner>, resolution_identity: ExecutionIdentity,
    config: Int4EntityConfig, model: &Int4ModelIdentity, control: &mut C)
    -> Result<PreparedInt4EntityCorpus, Int4EntityError> {
    resolve::checkpoint(control)?;
    validate(&config, &source, &source_identity, &resolution_identity, model)?;
    // Validate the complete fixed graph/scorer policy even for an empty corpus.
    let empty = ResolutionPlan::prepare(&[], config.resolution, config.graph, control)?;
    resolver.prepare_int4(&empty, &resolution_identity, config.scoring, model, control)?;
    let compiled = compile_inputs(documents, &source, &source_identity, &config, model.group_width(), control)?;
    resolve::checkpoint(control)?;
    Ok(PreparedInt4EntityCorpus { compiled, source, resolver, source_identity,
        resolution_identity, config, model: model.clone() })
}

fn compile_inputs<C: DecodeStepControl>(mut documents: Vec<EntityDocument>, source: &SourceTaskPlanner,
    identity: &ExecutionIdentity, config: &Int4EntityConfig, width: usize, control: &mut C)
    -> Result<CompiledInputs, Int4EntityError> {
    config.validate()?;
    let input_bytes = validate_documents(&mut documents, config)?;
    let context = PlanContext::new(identity, config.ner_budget).map_err(|_| Int4EntityError::Identity)?;
    // The same pinned template/encoder prices the partition and actual NER.
    // This also checks the source policy for a genuine zero-document corpus.
    let capacity = source.int4_map_capacity_with_control(&SourceMapTask::Ner(config.ner.clone()),
        config.ner_budget, &context, config.source_planning, width, control)?;
    let chunk_limits = config.chunks.map(|chunks| capacity.constrain_chunks(chunks)).transpose()?;
    let mut inputs = reserved(documents.len())?;
    let (mut ner_work, mut masks, mut count, mut maximum_positions) = (Int4Work::default(), 0_u64, 0_usize, 0_usize);
    for document in documents {
        resolve::checkpoint(control)?;
        let extents = if let Some(limits) = chunk_limits {
            let partition = partition(&document.text, source, limits, control)?;
            let mut spans = reserved(partition.chunks().len())?;
            spans.extend(partition.chunks().iter().map(|chunk| chunk.span()));
            spans
        } else {
            vec![VerifiedSourceSpan { byte_start: 0, byte_end: document.text.len(),
                scalar_start: 0, scalar_end: document.text.chars().count() }]
        };
        count = count.checked_add(extents.len()).filter(|&n| n <= config.max_snapshot_chunks)
            .ok_or(Int4EntityError::WorkBudget)?;
        let mut chunks = reserved(extents.len())?;
        for span in extents {
            resolve::checkpoint(control)?;
            let text = document.text.get(span.byte_start..span.byte_end).ok_or(Int4EntityError::Accounting)?;
            let plan = source_plan(text, source, identity, config, width, control)?;
            let work = plan.planned_work();
            ner_work = plus(ner_work, work)?;
            if !within(ner_work, config.max_model_work) { return Err(Int4EntityError::WorkBudget); }
            masks = masks.checked_add(config.masks.max_visits_per_item)
                .filter(|&n| n <= config.masks.max_visits_per_run).ok_or(Int4EntityError::WorkBudget)?;
            maximum_positions = maximum_positions.max(usize::try_from(work.forward_positions)
                .map_err(|_| Int4EntityError::WorkBudget)?);
            chunks.push(ChunkWitness { span, identity: witness(plan.execution_identity())?,
                prompt_tokens: plan.prompt_tokens(), work });
            // Drop this grammar/source index before compiling the next chunk.
        }
        inputs.push(Input { document, chunks });
    }
    resolve::checkpoint(control)?;
    Ok(CompiledInputs { inputs, ner_work, mask_visits: masks, chunks: count, maximum_positions, input_bytes })
}

impl PreparedInt4EntityCorpus {
    pub fn document_count(&self) -> usize { self.compiled.inputs.len() }
    pub fn chunk_count(&self) -> usize { self.compiled.chunks }
    pub fn input_bytes(&self) -> usize { self.compiled.input_bytes }
    pub fn ner_reserved_work(&self) -> Int4Work { self.compiled.ner_work }
    pub fn reserved_mask_visits(&self) -> u64 { self.compiled.mask_visits }
    pub fn required_ner_context_tokens(&self) -> usize { self.compiled.maximum_positions }
    pub fn source_identity(&self) -> &ExecutionIdentity { &self.source_identity }
    pub fn resolution_identity(&self) -> &ExecutionIdentity { &self.resolution_identity }
    pub fn config(&self) -> &Int4EntityConfig { &self.config }
    pub fn group_width(&self) -> usize { self.model.group_width() }
    /// Actual retained Vec/String capacities. Planners/configuration, temporary
    /// grammars, expanded mentions and the graph are separate host reservations.
    pub fn retained_input_bytes(&self) -> Result<u64, Int4EntityError> {
        retained_input_bytes(&self.compiled)
    }
    /// Reuse ONE engine through every NER chunk and both orders of all pairs.
    /// This consumes the snapshot, so a failed invocation cannot renew work by
    /// retrying a prepared value. The host retains guards through physical drain.
    pub fn execute_with_control<C: DecodeStepControl>(self, engine: &mut StrictInt4Engine<'_>,
        vocabulary: &ExtractionVocabulary, control: &mut C) -> Result<Int4EntityRun, Int4EntityError> {
        resolve::checkpoint(control)?;
        self.check_engine(engine)?;
        let Self { compiled, source, resolver, source_identity, resolution_identity, config, model } = self;
        let CompiledInputs { inputs, ner_work, mask_visits, chunks, .. } = compiled;
        let width = model.group_width();
        let maps = collect(inputs, &config, width, control, |text, chunk, control| {
            let plan = source_plan(text, &source, &source_identity, &config, width, control)?;
            check_rebuilt(chunk, &plan)?;
            let result = plan.execute_with_control(plan.execution_identity(), engine, vocabulary,
                Int4JsonBudget { native: Int4RunBudget::exact(chunk.work), json: JsonWorkBudget {
                    max_forward_positions: chunk.work.forward_positions, max_projected_logits: chunk.work.projected_logits,
                    max_kv_bytes: config.ner_budget.max_kv_bytes,
                    max_total_mask_node_visits: config.masks.max_visits_per_item, mask_limits: config.masks.per_mask,
                } }, control)?;
            if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) { return Err(Int4EntityError::Accounting); }
            Ok(result)
        })?;
        resolve::checkpoint(control)?;
        let graph = ResolutionPlan::prepare(&maps.documents, config.resolution, config.graph, control)?;
        // Early EOS never refunds unused NER capacity into pair scoring.
        let prepared = resolver.prepare_int4(&graph, &resolution_identity,
            remaining_scoring(&config, ner_work)?, &model, control)?;
        let pair_work = prepared.planned_work();
        let mut admitted = reserved(prepared.pair_count())?;
        for identity in prepared.execution_identities() {
            resolve::checkpoint(control)?;
            admitted.push(identity.clone());
        }
        let resolution = if prepared.pair_count() == 0 {
            prepared.finalize_without_model(control)?
        } else {
            prepared.execute_with_control(&admitted, engine, Int4ScoringBudget {
                native: Int4RunBudget::exact(pair_work), max_kv_bytes: config.scoring.planning.per_head.max_kv_bytes,
            }, control)?
        };
        drop(admitted); drop(graph);
        finish(maps, resolution, ner_work, mask_visits, chunks, &config, width, control)
    }
    /// Only a zero-document snapshot may omit native NER. Nonempty raw text
    /// without supplied mentions is never treated as evidence of no entities.
    pub fn finalize_without_model<C: DecodeStepControl>(self, control: &mut C) -> Result<Int4EntityRun, Int4EntityError> {
        resolve::checkpoint(control)?;
        if !self.compiled.inputs.is_empty() || self.compiled.chunks != 0
            || self.compiled.ner_work != Int4Work::default() || self.compiled.mask_visits != 0 {
            return Err(Int4EntityError::Accounting);
        }
        let graph = ResolutionPlan::prepare(&[], self.config.resolution, self.config.graph, control)?;
        let resolution = self.resolver.prepare_int4(&graph, &self.resolution_identity,
            self.config.scoring, &self.model, control)?.finalize_without_model(control)?;
        let maps = Extracted { documents: Vec::new(), receipts: Vec::new(), work: Int4Work::default(),
            mask_visits: 0, chunks: 0, verification_used: EntityVerificationWork::default() };
        finish(maps, resolution, self.compiled.ner_work, self.compiled.mask_visits, 0,
            &self.config, self.group_width(), control)
    }
    fn check_engine(&self, engine: &StrictInt4Engine<'_>) -> Result<(), Int4EntityError> {
        constrained_int4::check_model(&self.source_identity, engine.model_identity()).map_err(|_| Int4EntityError::Identity)?;
        resolution::check_model(&self.resolution_identity, engine.model_identity())?;
        if engine.model_identity() != &self.model { return Err(Int4EntityError::Identity); }
        if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) { return Err(Int4EntityError::Accounting); }
        let positions = engine.kv_cache().capacity_positions();
        let cap = self.config.ner_budget.max_kv_bytes.min(self.config.scoring.planning.per_head.max_kv_bytes);
        if positions < self.compiled.maximum_positions || (positions as u64).checked_mul(KV_BYTES_PER_TOKEN as u64)
            .is_none_or(|n| n > cap) { return Err(Int4EntityError::WorkBudget); }
        Ok(())
    }
}

#[derive(Serialize)]
pub struct Int4EntityChunkReceipt {
    pub chunk_id: usize,
    pub source_span: VerifiedSourceSpan,
    pub proposed_entities: usize,
    pub anchored_mentions: usize,
    pub model_work: Int4Work,
    pub mask_node_visits: u64,
}
#[derive(Serialize)]
pub struct Int4EntityDocumentReceipt {
    pub document_id: String,
    pub proposed_entities: usize,
    pub anchored_mentions: usize,
    pub model_work: Int4Work,
    pub mask_node_visits: u64,
    pub chunks: Vec<Int4EntityChunkReceipt>,
}
/// Sensitive complete result. Original mention surfaces/offsets are retained;
/// private source hashes and NER token transcripts never enter this envelope.
#[derive(Serialize)]
pub struct Int4EntityRun {
    pub schema_version: u32,
    pub execution: &'static str,
    pub numerics_profile: &'static str,
    pub group_width: usize,
    pub extraction_task: &'static str,
    pub chunk_profile: Option<&'static str>,
    pub chunk_count: usize,
    pub documents: Vec<Int4EntityDocumentReceipt>,
    pub ner_reserved_work: Int4Work,
    pub ner_model_work: Int4Work,
    pub reserved_model_work: Int4Work,
    pub model_work: Int4Work,
    pub reserved_mask_node_visits: u64,
    pub mask_node_visit_charge: u64,
    pub verification_used: EntityVerificationWork,
    pub resolution: Int4ResolutionRun,
    pub warnings: [&'static str; 2],
}
struct Extracted {
    documents: Vec<ResolutionDocument>, receipts: Vec<Int4EntityDocumentReceipt>, work: Int4Work,
    mask_visits: u64, chunks: usize, verification_used: EntityVerificationWork,
}

// Private numerical fault-injection seam only. Production reconstructs pinned
// source plans and calls the concrete native decoder in execute_with_control.
fn collect<C: DecodeStepControl, F>(inputs: Vec<Input>, config: &Int4EntityConfig, width: usize,
    control: &mut C, mut execute: F) -> Result<Extracted, Int4EntityError>
where F: FnMut(&str, &ChunkWitness, &mut C) -> Result<Int4SourceTaskRun, Int4EntityError> {
    let mut documents = reserved(inputs.len())?; let mut receipts = reserved(inputs.len())?;
    let mut verification = config.verification;
    let (mut bytes, mut mentions, mut masks, mut chunks) = (0_usize, 0_usize, 0_u64, 0_usize);
    let mut work = Int4Work::default();
    for input in inputs {
        resolve::checkpoint(control)?;
        if input.chunks.is_empty() { return Err(Int4EntityError::Accounting); }
        let mut document = ResolutionDocument { id: input.document.id, text: input.document.text, mentions: Vec::new() };
        bytes = bytes.checked_add(document.id.len()).and_then(|n| n.checked_add(document.text.len()))
            .filter(|&n| n <= config.graph.max_input_bytes).ok_or(Int4EntityError::WorkBudget)?;
        let mut chunk_receipts = reserved(input.chunks.len())?;
        let (mut cursor, mut scalar, mut proposed, mut document_masks) = (0_usize, 0_usize, 0_usize, 0_u64);
        let mut document_work = Int4Work::default();
        for (index, chunk) in input.chunks.iter().enumerate() {
            resolve::checkpoint(control)?;
            let text = chunk_text(&document.text, chunk.span, cursor, scalar)?;
            let run = execute(text, chunk, control)?;
            resolve::checkpoint(control)?;
            let (ner, actual) = check_run(chunk, run, config, width)?;
            let proposed_entities = ner.entities.len(); let mask_node_visits = ner.mask_node_visit_charge;
            let local = grounding::expand(text, ner, config, &mut verification, control)?;
            let anchored_mentions = local.len();
            grounding::append(&mut document, local, chunk.span, &mut bytes, &mut mentions, config.graph, control)?;
            proposed = proposed.checked_add(proposed_entities).ok_or(Int4EntityError::WorkBudget)?;
            document_work = plus(document_work, actual)?;
            document_masks = document_masks.checked_add(mask_node_visits).ok_or(Int4EntityError::WorkBudget)?;
            chunks = chunks.checked_add(1).filter(|&n| n <= config.max_snapshot_chunks).ok_or(Int4EntityError::WorkBudget)?;
            chunk_receipts.push(Int4EntityChunkReceipt { chunk_id: index, source_span: chunk.span,
                proposed_entities, anchored_mentions, model_work: actual, mask_node_visits });
            cursor = chunk.span.byte_end; scalar = chunk.span.scalar_end;
        }
        if cursor != document.text.len() { return Err(Int4EntityError::Accounting); }
        work = plus(work, document_work)?;
        masks = masks.checked_add(document_masks).filter(|&n| n <= config.masks.max_visits_per_run)
            .ok_or(Int4EntityError::WorkBudget)?;
        receipts.push(Int4EntityDocumentReceipt { document_id: copy(&document.id)?, proposed_entities: proposed,
            anchored_mentions: document.mentions.len(), model_work: document_work,
            mask_node_visits: document_masks, chunks: chunk_receipts });
        documents.push(document);
    }
    resolve::checkpoint(control)?;
    let verification_used = EntityVerificationWork {
        fields: config.verification.max_fields.checked_sub(verification.max_fields).ok_or(Int4EntityError::Accounting)?,
        matches: config.verification.max_matches.checked_sub(verification.max_matches).ok_or(Int4EntityError::Accounting)?,
        scan_steps: config.verification.max_scan_steps.checked_sub(verification.max_scan_steps).ok_or(Int4EntityError::Accounting)?,
    };
    Ok(Extracted { documents, receipts, work, mask_visits: masks, chunks, verification_used })
}
#[allow(clippy::too_many_arguments)]
fn finish<C: DecodeStepControl>(maps: Extracted, resolution: Int4ResolutionRun, ner_work: Int4Work,
    mask_visits: u64, chunks: usize, config: &Int4EntityConfig, width: usize, control: &mut C)
    -> Result<Int4EntityRun, Int4EntityError> {
    resolve::checkpoint(control)?;
    let reserved_model_work = plus(ner_work, resolution.planned_model_work)?;
    let model_work = plus(maps.work, resolution.model_work)?;
    if !within(maps.work, ner_work) || !within(model_work, reserved_model_work)
        || !within(reserved_model_work, config.max_model_work) || maps.mask_visits > mask_visits
        || maps.chunks != chunks || resolution.group_width != width
        || resolution.result.document_count != maps.receipts.len() { return Err(Int4EntityError::Accounting); }
    let Extracted { documents, receipts, work, mask_visits: actual_masks, verification_used, .. } = maps;
    drop(documents);
    let output = Int4EntityRun { schema_version: 1, execution: INT4_ENTITY_EXECUTION,
        numerics_profile: STRICT_INT4_PROFILE, group_width: width, extraction_task: NER_TASK_VERSION,
        chunk_profile: config.chunks.map(|_| CHUNK_PROFILE), chunk_count: chunks,
        documents: receipts, ner_reserved_work: ner_work, ner_model_work: work,
        reserved_model_work, model_work, reserved_mask_node_visits: mask_visits,
        mask_node_visit_charge: actual_masks, verification_used, resolution,
        warnings: ["ner_recall_and_resolution_quality_not_established", "ner_partition_boundaries_may_split_entities"] };
    resolve::check_output(&output, config.max_result_bytes)?;
    resolve::checkpoint(control)?;
    Ok(output)
}

fn validate(config: &Int4EntityConfig, source: &SourceTaskPlanner, a: &ExecutionIdentity,
    b: &ExecutionIdentity, model: &Int4ModelIdentity) -> Result<(), Int4EntityError> {
    config.validate()?;
    constrained_int4::check_model(a, model).map_err(|_| Int4EntityError::Identity)?;
    resolution::check_model(b, model)?;
    resolve::check_output(a, 16_384)?; resolve::check_output(b, 16_384)?;
    if a.task_spec != NER_TASK_VERSION || a.tokenizer_digest != source.tokenizer_digest()
        || a.template_digest != *source.template_digest() { return Err(Int4EntityError::Identity); }
    same_model(a, b)
}
fn same_model(a: &ExecutionIdentity, b: &ExecutionIdentity) -> Result<(), Int4EntityError> {
    // Exclude task-owned fields only; every present/future model or host fact
    // remains included by full structural equality, not a short allowlist.
    let mut normalized = a.clone();
    normalized.template_digest = b.template_digest; normalized.task_spec = b.task_spec.clone();
    normalized.taskir_digest = b.taskir_digest; normalized.prompt_digest = b.prompt_digest;
    normalized.grammar_compiler_version = b.grammar_compiler_version.clone(); normalized.schema_digest = b.schema_digest;
    normalized.sampler_version = b.sampler_version.clone(); normalized.calibration_digest = b.calibration_digest;
    normalized.decision_policy_digest = b.decision_policy_digest;
    if &normalized != b { return Err(Int4EntityError::Identity); }
    Ok(())
}
fn validate_documents(documents: &mut [EntityDocument], config: &Int4EntityConfig) -> Result<usize, Int4EntityError> {
    if documents.len() > config.graph.max_documents { return Err(Int4EntityError::WorkBudget); }
    documents.sort_unstable_by(|a, b| a.id.cmp(&b.id));
    let maximum = config.chunks.map_or(config.source_planning.max_input_bytes, |chunks| chunks.max_input_bytes);
    let mut bytes = 0_usize;
    for (index, document) in documents.iter().enumerate() {
        if document.id.is_empty() || document.id.len() > 256 || document.id.chars().any(char::is_control)
            || document.text.is_empty() || (index > 0 && documents[index - 1].id == document.id) {
            return Err(Int4EntityError::InvalidInput);
        }
        if document.text.len() > maximum { return Err(Int4EntityError::WorkBudget); }
        bytes = bytes.checked_add(document.id.len()).and_then(|n| n.checked_add(document.text.len()))
            .filter(|&n| n <= config.graph.max_input_bytes).ok_or(Int4EntityError::WorkBudget)?;
    }
    Ok(bytes)
}
fn source_plan<C: DecodeStepControl>(text: &str, source: &SourceTaskPlanner, identity: &ExecutionIdentity,
    config: &Int4EntityConfig, width: usize, control: &mut C) -> Result<PreparedInt4SourceTask, Int4EntityError> {
    resolve::checkpoint(control)?;
    let request = SourceTaskRequest::Ner { document: copy(text)?, options: config.ner.clone(), budget: config.ner_budget };
    let context = PlanContext::new(identity, config.ner_budget).map_err(|_| Int4EntityError::Identity)?;
    Ok(source.plan_int4_with_control(&request, &context, config.source_planning, width, control)?)
}
fn check_rebuilt(chunk: &ChunkWitness, plan: &PreparedInt4SourceTask) -> Result<(), Int4EntityError> {
    if chunk.identity != witness(plan.execution_identity())? || chunk.work != plan.planned_work()
        || chunk.prompt_tokens != plan.prompt_tokens() { return Err(Int4EntityError::Accounting); }
    Ok(())
}
fn check_run(chunk: &ChunkWitness, run: Int4SourceTaskRun, config: &Int4EntityConfig, width: usize)
    -> Result<(NerResult, Int4Work), Int4EntityError> {
    if run.schema_version != 1 || run.execution != INT4_SOURCE_EXECUTION || run.group_width != width {
        return Err(Int4EntityError::Accounting);
    }
    let SourceTaskResult::Ner(ner) = run.result else { return Err(Int4EntityError::Accounting); };
    let expected = constrained_int4::planned_work(width, chunk.prompt_tokens, ner.generated_token_ids.len())
        .map_err(|_| Int4EntityError::Accounting)?;
    if ner.generated_token_ids.len() > config.ner_budget.max_output_tokens as usize
        || run.model_work != expected || !within(expected, chunk.work)
        || ner.forward_positions != expected.forward_positions || ner.projected_logits != expected.projected_logits
        || ner.mask_node_visit_charge > config.masks.max_visits_per_item { return Err(Int4EntityError::Accounting); }
    Ok((ner, expected))
}
fn remaining_scoring(config: &Int4EntityConfig, reserved_ner: Int4Work) -> Result<Int4ResolveLimits, Int4EntityError> {
    let mut scoring = config.scoring;
    scoring.planning.max_pairs = scoring.planning.max_pairs.min(config.graph.max_candidate_pairs);
    scoring.max_model_work = meet(scoring.max_model_work, subtract(config.max_model_work, reserved_ner)?);
    Ok(scoring)
}
fn retained_input_bytes(compiled: &CompiledInputs) -> Result<u64, Int4EntityError> {
    let mut bytes = (compiled.inputs.capacity() as u64).checked_mul(size_of::<Input>() as u64)
        .ok_or(Int4EntityError::WorkBudget)?;
    for input in &compiled.inputs {
        bytes = bytes.checked_add(input.document.id.capacity() as u64)
            .and_then(|n| n.checked_add(input.document.text.capacity() as u64))
            .and_then(|n| n.checked_add((input.chunks.capacity() as u64).checked_mul(size_of::<ChunkWitness>() as u64)?))
            .ok_or(Int4EntityError::WorkBudget)?;
    }
    Ok(bytes)
}
fn chunk_text(source: &str, span: VerifiedSourceSpan, byte_start: usize, scalar_start: usize)
    -> Result<&str, Int4EntityError> {
    if span.byte_start != byte_start || span.scalar_start != scalar_start || span.byte_end <= span.byte_start {
        return Err(Int4EntityError::Accounting);
    }
    let text = source.get(span.byte_start..span.byte_end).ok_or(Int4EntityError::Accounting)?;
    if span.scalar_start.checked_add(text.chars().count()) != Some(span.scalar_end) { return Err(Int4EntityError::Accounting); }
    Ok(text)
}
fn partition<'s, C: DecodeStepControl>(text: &'s str, source: &SourceTaskPlanner, limits: ChunkLimits, control: &mut C)
    -> Result<ChunkPlan<'s>, Int4EntityError> {
    let mut cancelled = None;
    let result = ChunkPlan::build_with_checkpoints(text, limits, |part| {
        source.source_encoder().encode(part, limits.max_chunk_bytes, limits.max_chunk_bytes)
            .map(|document| document.total_token_count()).map_err(|_| MapReduceError::Tokenizer)
    }, || {
        if let Some(cause) = control.prefill_checkpoint(0) {
            cancelled = Some(cause); Err(MapReduceError::Cancelled)
        } else { Ok(()) }
    });
    if let Some(cause) = cancelled { return Err(ResolveError::Cancelled(cause).into()); }
    result.map_err(partition_error)
}
fn partition_error(error: MapReduceError) -> Int4EntityError {
    match error {
        MapReduceError::InvalidLimits => Int4EntityError::InvalidLimits,
        MapReduceError::InputBudget | MapReduceError::ChunkBudget | MapReduceError::TokenBudget
            | MapReduceError::TokenizerBudget => Int4EntityError::WorkBudget,
        MapReduceError::AllocationRefused => Int4EntityError::Allocation,
        _ => Int4EntityError::Accounting,
    }
}
fn witness(identity: &ExecutionIdentity) -> Result<Sha256Digest, Int4EntityError> {
    Ok(Sha256Digest::of_bytes(&canonjson::canonical_bytes(identity).map_err(|_| Int4EntityError::Accounting)?))
}
fn copy(text: &str) -> Result<String, Int4EntityError> {
    let mut value = String::new(); value.try_reserve_exact(text.len()).map_err(|_| Int4EntityError::Allocation)?;
    value.push_str(text); Ok(value)
}
fn reserved<T>(count: usize) -> Result<Vec<T>, Int4EntityError> {
    let mut value = Vec::new(); value.try_reserve_exact(count).map_err(|_| Int4EntityError::Allocation)?; Ok(value)
}
fn plus(a: Int4Work, b: Int4Work) -> Result<Int4Work, Int4EntityError> {
    a.checked_add(b).map_err(|_| Int4EntityError::WorkBudget)
}
fn within(a: Int4Work, b: Int4Work) -> bool {
    a.forward_positions <= b.forward_positions && a.projected_logits <= b.projected_logits
        && a.attention_pairs <= b.attention_pairs && a.projections.fits(b.projections)
}
fn meet(a: Int4Work, b: Int4Work) -> Int4Work {
    Int4Work { forward_positions: a.forward_positions.min(b.forward_positions),
        projected_logits: a.projected_logits.min(b.projected_logits), attention_pairs: a.attention_pairs.min(b.attention_pairs),
        projections: ProjectionWork { dot_products: a.projections.dot_products.min(b.projections.dot_products),
            multiply_accumulates: a.projections.multiply_accumulates.min(b.projections.multiply_accumulates) } }
}
fn subtract(a: Int4Work, b: Int4Work) -> Result<Int4Work, Int4EntityError> {
    let sub = |a: u64, b: u64| a.checked_sub(b).ok_or(Int4EntityError::WorkBudget);
    Ok(Int4Work { forward_positions: sub(a.forward_positions, b.forward_positions)?,
        projected_logits: sub(a.projected_logits, b.projected_logits)?, attention_pairs: sub(a.attention_pairs, b.attention_pairs)?,
        projections: ProjectionWork { dot_products: sub(a.projections.dot_products, b.projections.dot_products)?,
            multiply_accumulates: sub(a.projections.multiply_accumulates, b.projections.multiply_accumulates)? } })
}

#[cfg(test)]
mod tests;
