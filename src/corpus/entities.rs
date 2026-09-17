//! Raw documents -> native source-constrained NER -> complete corpus resolution.
//!
//! One existing engine, one host admission authority, no loader or runtime.
//! Every NER plan is preflighted before inference, then rebuilt one at a time
//! and checked against its private witness. Pair work is known only after NER;
//! it must fit the NONRENEWABLE remainder of the same whole-corpus allowance.
//! No intermediate mentions/clusters are published as a completed corpus.

use std::{error::Error, fmt, mem::size_of};
use serde::{Deserialize, Serialize};
use crate::{
    batch::{BatchDocument, BatchFault, BatchItemFailure, BatchProcessor, BatchWork,
        source::{GuardedOutput, NativeSourceBatch, PreparedBatchSource, SourceBatchAdmission,
            SourceBatchArgs, SourceBatchPlanner, SourceMaskBudget}},
    canonjson,
    execution_identity::{ExecutionIdentity, Sha256Digest},
    native_engine::{decode::DecodeStepControl,
        hf_bf16_eager::HfBf16EagerEngine, kv::KV_BYTES_PER_TOKEN, lmhead::NANBEIGE_VOCAB_SIZE},
    tasks::{extract::ExtractionVocabulary, ir::TaskBudget, ner::{NerOptions, NerResult, NER_TASK_VERSION},
        source_planning::{SourcePlanningLimits, SourceTaskPlanner, SourceTaskResult}},
    validation::grounded_fields::GroundingBudget,
};
use super::{
    native_resolve::{NativeResolutionResult, NativeResolveError, NativeResolveLimits, ResolutionPlanner},
    resolution_stream::document_from_ner,
    resolve::{self, ResolutionDocument, ResolutionPlan, ResolveError, ResolveLimits, ResolveOptions},
};

pub const ENTITY_CORPUS_EXECUTION: &str = "native-ner-to-complete-corpus-resolution-v1";

/// Original UTF-8; no caller-supplied mentions, model identity, or task override.
/// No Debug: a diagnostic must not accidentally expose a document.
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EntityDocument { pub id: String, pub text: String }

#[derive(Clone, Debug)]
pub struct EntityCorpusConfig {
    pub ner: NerOptions,
    pub ner_budget: TaskBudget,
    pub source_planning: SourcePlanningLimits,
    pub masks: SourceMaskBudget,
    pub resolution: ResolveOptions,
    pub corpus: ResolveLimits,
    pub native: NativeResolveLimits,
    /// Independent NER occurrence verification, shared across ALL documents.
    pub verification: GroundingBudget,
    /// Both stages together, not an allowance renewed at their boundary.
    pub max_work: BatchWork,
    pub max_result_bytes: usize,
    /// Inline retained NER+pair guards; the host separately prices guard heaps.
    pub max_retained_guard_bytes: usize,
}

#[derive(Debug)]
pub enum EntityCorpusError {
    InvalidLimits, InvalidInput, Identity, WorkBudget, Accounting, AllocationRefused,
    Serialization, Batch(BatchItemFailure), Resolution(ResolveError), Native(NativeResolveError),
}
impl fmt::Display for EntityCorpusError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidLimits => "invalid raw entity corpus limits",
            Self::InvalidInput => "raw entity corpus requires unique bounded document ids",
            Self::Identity => "entity stages disagree on admitted model or backend identity",
            Self::WorkBudget => "raw entity corpus aggregate allowance exceeded",
            Self::Accounting => "raw entity corpus execution diverged from its plan",
            Self::AllocationRefused => "raw entity corpus allocation refused",
            Self::Serialization => "raw entity corpus serialization failed",
            Self::Batch(_) => "raw entity corpus NER planning or execution failed",
            Self::Resolution(_) => "raw entity corpus source verification or graph failed",
            Self::Native(_) => "raw entity corpus native pair scoring failed",
        })
    }
}
impl Error for EntityCorpusError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Batch(e) => Some(&e.fault), Self::Resolution(e) => Some(e),
            Self::Native(e) => Some(e), _ => None }
    }
}
impl From<ResolveError> for EntityCorpusError { fn from(e: ResolveError) -> Self { Self::Resolution(e) } }
impl From<NativeResolveError> for EntityCorpusError { fn from(e: NativeResolveError) -> Self { Self::Native(e) } }
impl From<BatchItemFailure> for EntityCorpusError { fn from(e: BatchItemFailure) -> Self { Self::Batch(e) } }
impl From<BatchFault> for EntityCorpusError {
    fn from(e: BatchFault) -> Self { Self::Batch(BatchItemFailure::fatal(e)) }
}

struct PreparedDocument { document: EntityDocument, identity: Sha256Digest, work: BatchWork }

/// Owns the frozen original documents, but only small per-document witnesses,
/// not a corpus-sized collection of grammar/source indexes. Consumed on run.
/// A prepared value has no deserializer, clone, or public content fingerprint.
pub struct PreparedEntityCorpus<'p> {
    documents: Vec<PreparedDocument>,
    compiler: SourceBatchPlanner<'p>,
    resolver: &'p ResolutionPlanner,
    resolution_identity: ExecutionIdentity,
    config: EntityCorpusConfig,
    ner_work: BatchWork,
    maximum_ner_positions: u64,
    mask_visits: u64,
    input_bytes: usize,
}
impl PreparedEntityCorpus<'_> {
    pub fn document_count(&self) -> usize { self.documents.len() }
    pub fn input_bytes(&self) -> usize { self.input_bytes }
    pub fn ner_reserved_work(&self) -> BatchWork { self.ner_work }
    pub fn reserved_mask_visits(&self) -> u64 { self.mask_visits }
    pub fn maximum_ner_positions(&self) -> u64 { self.maximum_ner_positions }
    /// Sound bound before the candidate graph exists; pair work may use only
    /// the portion remaining after all conservative NER charges.
    pub fn whole_run_work_ceiling(&self) -> BatchWork { self.config.max_work }

    /// `run_guard` is the host's actual reservation for source storage, one
    /// live NER compiler/index, graph/pair plans and complete output workspace.
    /// Per-item guards AND that run guard remain live through final delivery.
    /// No model or scheduler authority is manufactured by this composition.
    pub fn execute_native<A: SourceBatchAdmission, C: DecodeStepControl, G>(self,
        engine: &mut HfBf16EagerEngine, vocabulary: &ExtractionVocabulary,
        mut admission: A, run_guard: G, control: &mut C,
    ) -> Result<GuardedOutput<EntityCorpusResult, (Vec<A::Guard>, Vec<A::Guard>, G)>, EntityCorpusError> {
        resolve::checkpoint(control)?;
        if !engine.kv_cache().all_slots_have_len(0) { return Err(EntityCorpusError::Accounting); }
        let capacity = engine.kv_cache().capacity_positions() as u64;
        let kv = capacity.checked_mul(KV_BYTES_PER_TOKEN as u64).ok_or(EntityCorpusError::WorkBudget)?;
        if capacity < self.maximum_ner_positions || kv > self.config.ner_budget.max_kv_bytes
            || kv > self.config.native.per_head.max_kv_bytes { return Err(EntityCorpusError::WorkBudget); }
        let ner_guard_bytes = guard_bytes::<A::Guard>(self.documents.len(), self.config.max_retained_guard_bytes)?;
        let Self { documents, compiler, resolver, resolution_identity, config, ner_work, mask_visits, .. } = self;
        let maps = {
            let mut processor = NativeSourceBatch::new(compiler, engine, vocabulary,
                BorrowAdmission(&mut admission), config.masks)?;
            let maps = collect_extractions(documents, &config, control, |input, control| {
                let prepared = processor.prepare(batch_document(&input.document, &config)?)?;
                check_prepared(input, &prepared)?;
                processor.execute(prepared, control).map_err(EntityCorpusError::from)
            })?;
            if processor.reserved_mask_visits() != mask_visits { return Err(EntityCorpusError::Accounting); }
            maps
        };
        resolve::checkpoint(control)?;
        // Fail the WHOLE snapshot, never discard expensive or unscored pairs.
        let plan = ResolutionPlan::prepare(&maps.documents, config.resolution, config.corpus, control)?;
        let mut native = config.native;
        native.max_pairs = native.max_pairs.min(config.corpus.max_candidate_pairs);
        native.max_work = meet(native.max_work, subtract(config.max_work, ner_work)?);
        native.max_retained_guard_bytes = native.max_retained_guard_bytes
            .min(config.max_retained_guard_bytes - ner_guard_bytes);
        let prepared = resolver.prepare(&plan, &resolution_identity, native, control)?;
        let reserved_work = add(ner_work, prepared.planned_work())?;
        // Borrow the REAL run guard rather than moving it into the child: on
        // child failure it must still outlive the parent's source/map storage.
        let scored = prepared.execute_native(engine, admission, &run_guard, control)?;
        drop(plan);
        let actual_work = add(maps.actual_work, scored.result().actual_work)?;
        if !fits(actual_work, reserved_work) || !fits(reserved_work, config.max_work) {
            return Err(EntityCorpusError::Accounting);
        }
        let (resolution, (pair_guards, _)) = scored.into_parts();
        let Extracted { documents, receipts, actual_work: ner_actual_work, mask_visits: actual_masks,
            verification_used, guards } = maps;
        drop(documents);
        let output = GuardedOutput::new(EntityCorpusResult {
            schema_version: 1, execution: ENTITY_CORPUS_EXECUTION,
            extraction_task: NER_TASK_VERSION, documents: receipts,
            ner_reserved_work: ner_work, ner_actual_work, reserved_work, actual_work,
            reserved_mask_visits: mask_visits, actual_mask_visits: actual_masks,
            verification_used, resolution,
        }, (guards, pair_guards, run_guard));
        resolve::check_output(output.result(), config.max_result_bytes)?;
        resolve::checkpoint(control)?;
        Ok(output)
    }
}

/// Validate/configure both stages before ANY model call, canonicalize document
/// order, and preflight exact NER work. Source plans are rebuilt one at a time
/// at execution and must reproduce their complete private identity and charge.
#[allow(clippy::too_many_arguments)]
pub fn prepare_entity_corpus<'p, C: DecodeStepControl>(mut documents: Vec<EntityDocument>,
    source: &'p SourceTaskPlanner, source_identity: &ExecutionIdentity,
    resolver: &'p ResolutionPlanner, resolution_identity: &ExecutionIdentity,
    config: EntityCorpusConfig, control: &mut C,
) -> Result<PreparedEntityCorpus<'p>, EntityCorpusError> {
    resolve::checkpoint(control)?;
    config.ner.validate().map_err(|_| EntityCorpusError::InvalidLimits)?;
    config.ner_budget.validate().map_err(|_| EntityCorpusError::InvalidLimits)?;
    if source_identity.task_spec != NER_TASK_VERSION || config.masks.max_visits_per_item == 0
        || config.masks.per_mask.max_trie_node_visits == 0 || config.masks.per_mask.checkpoint_interval_nodes == 0
        || !(1..=64 * 1024 * 1024).contains(&config.max_result_bytes)
        || config.max_retained_guard_bytes > 64 * 1024 * 1024 {
        return Err(EntityCorpusError::InvalidLimits);
    }
    same_engine_identity(source_identity, resolution_identity)?;
    let empty = ResolutionPlan::prepare(&[], config.resolution, config.corpus, control)?;
    resolver.prepare(&empty, resolution_identity, config.native, control)?;
    let input_bytes = validate_documents(&mut documents, &config)?;
    let mask_visits = (documents.len() as u64).checked_mul(config.masks.max_visits_per_item)
        .filter(|&n| n <= config.masks.max_visits_per_run).ok_or(EntityCorpusError::WorkBudget)?;
    let compiler = SourceBatchPlanner::new(source, source_identity.clone(), config.ner_budget,
        config.source_planning, None)?;
    let mut inputs = reserved(documents.len())?;
    let mut ner_work = BatchWork::default(); let mut maximum_ner_positions = 0;
    for document in documents {
        resolve::checkpoint(control)?;
        let prepared = compiler.prepare(batch_document(&document, &config)?)?;
        let work = prepared.planned_work();
        ner_work = add(ner_work, work)?;
        if !fits(ner_work, config.max_work) { return Err(EntityCorpusError::WorkBudget); }
        maximum_ner_positions = maximum_ner_positions.max(work.forward_positions);
        inputs.push(PreparedDocument { document, identity: identity_digest(prepared.execution_identity())?, work });
    }
    resolve::checkpoint(control)?;
    Ok(PreparedEntityCorpus { documents: inputs, compiler, resolver,
        resolution_identity: resolution_identity.clone(), config, ner_work,
        maximum_ner_positions, mask_visits, input_bytes })
}

#[derive(Serialize)]
pub struct EntityDocumentReceipt {
    pub document_id: String,
    pub proposed_entities: usize,
    pub anchored_mentions: usize,
    pub actual_work: BatchWork,
    pub mask_visits: u64,
}
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct EntityVerificationWork { pub fields: usize, pub matches: usize, pub scan_steps: u64 }
#[derive(Serialize)]
pub struct EntityCorpusResult {
    pub schema_version: u32,
    pub execution: &'static str,
    pub extraction_task: &'static str,
    /// Canonically ordered, including documents that yielded zero mentions.
    pub documents: Vec<EntityDocumentReceipt>,
    pub ner_reserved_work: BatchWork,
    pub ner_actual_work: BatchWork,
    pub reserved_work: BatchWork,
    pub actual_work: BatchWork,
    pub reserved_mask_visits: u64,
    pub actual_mask_visits: u64,
    pub verification_used: EntityVerificationWork,
    pub resolution: NativeResolutionResult,
}

// Storage fields precede guards so failed intermediate work frees its private
// results BEFORE relinquishing admission. The host's run guard lives outside.
struct Extracted<G> {
    documents: Vec<ResolutionDocument>, receipts: Vec<EntityDocumentReceipt>,
    actual_work: BatchWork, mask_visits: u64, verification_used: EntityVerificationWork,
    guards: Vec<G>,
}
fn collect_extractions<G, C: DecodeStepControl, F>(inputs: Vec<PreparedDocument>, config: &EntityCorpusConfig,
    control: &mut C, mut execute: F) -> Result<Extracted<G>, EntityCorpusError>
where F: FnMut(&PreparedDocument, &mut C) -> Result<GuardedOutput<SourceTaskResult, G>, EntityCorpusError> {
    let mut guards = reserved(inputs.len())?;
    let mut documents = reserved(inputs.len())?; let mut receipts = reserved(inputs.len())?;
    let mut verification = config.verification;
    let mut actual_work = BatchWork::default(); let mut mask_visits = 0_u64;
    let (mut bytes, mut mentions) = (0_usize, 0_usize);
    for input in inputs {
        resolve::checkpoint(control)?;
        let (result, guard) = execute(&input, control)?.into_parts();
        guards.push(guard);
        let SourceTaskResult::Ner(ner) = result else { return Err(EntityCorpusError::Accounting); };
        let work = observed_ner(&ner, input.work, config)?;
        actual_work = add(actual_work, work)?;
        mask_visits = mask_visits.checked_add(ner.mask_node_visit_charge).ok_or(EntityCorpusError::WorkBudget)?;
        let proposed_entities = ner.entities.len(); let visits = ner.mask_node_visit_charge;
        let document = document_from_ner(input.document.id, input.document.text, ner,
            config.corpus, &mut verification, control)?;
        // Bound the complete EXPANDED corpus, not only each document in isolation.
        charge_document(&document, &mut bytes, &mut mentions, config.corpus)?;
        receipts.push(EntityDocumentReceipt { document_id: copy_text(&document.id)?, proposed_entities,
            anchored_mentions: document.mentions.len(), actual_work: work, mask_visits: visits });
        documents.push(document);
    }
    resolve::checkpoint(control)?;
    let verification_used = EntityVerificationWork {
        fields: config.verification.max_fields - verification.max_fields,
        matches: config.verification.max_matches - verification.max_matches,
        scan_steps: config.verification.max_scan_steps - verification.max_scan_steps,
    };
    Ok(Extracted { documents, receipts, actual_work, mask_visits, verification_used, guards })
}
fn observed_ner(ner: &NerResult, reserved: BatchWork, config: &EntityCorpusConfig) -> Result<BatchWork, EntityCorpusError> {
    let prompt = reserved.forward_positions.checked_add(1)
        .and_then(|n| n.checked_sub(u64::from(config.ner_budget.max_output_tokens))).ok_or(EntityCorpusError::Accounting)?;
    let positions = prompt.checked_add(ner.generated_token_ids.len() as u64).and_then(|n| n.checked_sub(1));
    let work = BatchWork { forward_positions: ner.forward_positions, projected_logits: ner.projected_logits };
    if prompt == 0 || ner.generated_token_ids.is_empty() || ner.generated_token_ids.len() > config.ner_budget.max_output_tokens as usize
        || positions != Some(work.forward_positions) || !fits(work, reserved)
        || work.forward_positions.checked_mul(NANBEIGE_VOCAB_SIZE as u64) != Some(work.projected_logits)
        || ner.mask_node_visit_charge > config.masks.max_visits_per_item {
        return Err(EntityCorpusError::Accounting);
    }
    Ok(work)
}
fn charge_document(d: &ResolutionDocument, bytes: &mut usize, mentions: &mut usize, limits: ResolveLimits)
    -> Result<(), EntityCorpusError> {
    let next_mentions = mentions.checked_add(d.mentions.len()).filter(|&n| n <= limits.max_mentions)
        .ok_or(EntityCorpusError::WorkBudget)?;
    let mut next_bytes = bytes.checked_add(d.id.len()).and_then(|n| n.checked_add(d.text.len()))
        .ok_or(EntityCorpusError::WorkBudget)?;
    for mention in &d.mentions {
        next_bytes = next_bytes.checked_add(mention.entity_type.len()).and_then(|n| n.checked_add(mention.surface.len()))
            .ok_or(EntityCorpusError::WorkBudget)?;
    }
    if next_bytes > limits.max_input_bytes { return Err(EntityCorpusError::WorkBudget); }
    *bytes = next_bytes; *mentions = next_mentions; Ok(())
}
fn validate_documents(documents: &mut [EntityDocument], config: &EntityCorpusConfig) -> Result<usize, EntityCorpusError> {
    if documents.len() > config.corpus.max_documents { return Err(EntityCorpusError::WorkBudget); }
    documents.sort_unstable_by(|a, b| a.id.cmp(&b.id));
    let mut bytes = 0_usize;
    for (index, d) in documents.iter().enumerate() {
        if d.id.is_empty() || d.id.len() > 256 || d.id.chars().any(char::is_control)
            || (index > 0 && documents[index - 1].id == d.id) { return Err(EntityCorpusError::InvalidInput); }
        if d.text.len() > config.source_planning.max_input_bytes { return Err(EntityCorpusError::WorkBudget); }
        bytes = bytes.checked_add(d.id.len()).and_then(|n| n.checked_add(d.text.len()))
            .filter(|&n| n <= config.corpus.max_input_bytes).ok_or(EntityCorpusError::WorkBudget)?;
    }
    Ok(bytes)
}
fn same_engine_identity(source: &ExecutionIdentity, resolution: &ExecutionIdentity) -> Result<(), EntityCorpusError> {
    source.validate().map_err(|_| EntityCorpusError::Identity)?;
    resolution.validate().map_err(|_| EntityCorpusError::Identity)?;
    resolve::check_output(source, 16_384)?;
    resolve::check_output(resolution, 16_384)?;
    let mut normalized = source.clone();
    // Enumerate only task-owned differences; any new host/model identity field
    // automatically remains covered by ExecutionIdentity's full equality.
    normalized.template_digest = resolution.template_digest;
    normalized.task_spec = resolution.task_spec.clone();
    normalized.taskir_digest = resolution.taskir_digest;
    normalized.prompt_digest = resolution.prompt_digest;
    normalized.grammar_compiler_version = resolution.grammar_compiler_version.clone();
    normalized.schema_digest = resolution.schema_digest;
    normalized.sampler_version = resolution.sampler_version.clone();
    normalized.calibration_digest = resolution.calibration_digest;
    normalized.decision_policy_digest = resolution.decision_policy_digest;
    if &normalized != resolution { return Err(EntityCorpusError::Identity); }
    Ok(())
}
fn batch_document(d: &EntityDocument, config: &EntityCorpusConfig) -> Result<BatchDocument<SourceBatchArgs>, EntityCorpusError> {
    Ok(BatchDocument { id: copy_text(&d.id)?, text: copy_text(&d.text)?, task_args: Some(SourceBatchArgs::Ner {
        options: config.ner.clone(), budget: config.ner_budget }) })
}
fn identity_digest(id: &ExecutionIdentity) -> Result<Sha256Digest, EntityCorpusError> {
    Ok(Sha256Digest::of_bytes(&canonjson::canonical_bytes(id).map_err(|_| EntityCorpusError::Serialization)?))
}
fn check_prepared(expected: &PreparedDocument, actual: &PreparedBatchSource) -> Result<(), EntityCorpusError> {
    if expected.work != actual.planned_work() || expected.identity != identity_digest(actual.execution_identity())? {
        return Err(EntityCorpusError::Accounting);
    }
    Ok(())
}
pub(super) struct BorrowAdmission<'a, A>(pub &'a mut A);
impl<A: SourceBatchAdmission> SourceBatchAdmission for BorrowAdmission<'_, A> {
    type Guard = A::Guard;
    fn admit(&mut self, proposed: &ExecutionIdentity, work: BatchWork) -> Result<(ExecutionIdentity, Self::Guard), BatchItemFailure> {
        self.0.admit(proposed, work)
    }
}
fn copy_text(s: &str) -> Result<String, EntityCorpusError> {
    let mut out = String::new(); out.try_reserve_exact(s.len()).map_err(|_| EntityCorpusError::AllocationRefused)?;
    out.push_str(s); Ok(out)
}
fn reserved<T>(count: usize) -> Result<Vec<T>, EntityCorpusError> {
    let mut out = Vec::new(); out.try_reserve_exact(count).map_err(|_| EntityCorpusError::AllocationRefused)?; Ok(out)
}
fn guard_bytes<G>(count: usize, cap: usize) -> Result<usize, EntityCorpusError> {
    count.checked_mul(size_of::<G>()).filter(|&n| n <= cap).ok_or(EntityCorpusError::WorkBudget)
}
pub(super) fn add(a: BatchWork, b: BatchWork) -> Result<BatchWork, EntityCorpusError> {
    Ok(BatchWork { forward_positions: a.forward_positions.checked_add(b.forward_positions).ok_or(EntityCorpusError::WorkBudget)?,
        projected_logits: a.projected_logits.checked_add(b.projected_logits).ok_or(EntityCorpusError::WorkBudget)? })
}
pub(super) fn subtract(a: BatchWork, b: BatchWork) -> Result<BatchWork, EntityCorpusError> {
    Ok(BatchWork { forward_positions: a.forward_positions.checked_sub(b.forward_positions).ok_or(EntityCorpusError::WorkBudget)?,
        projected_logits: a.projected_logits.checked_sub(b.projected_logits).ok_or(EntityCorpusError::WorkBudget)? })
}
fn fits(a: BatchWork, b: BatchWork) -> bool { a.forward_positions <= b.forward_positions && a.projected_logits <= b.projected_logits }
fn meet(a: BatchWork, b: BatchWork) -> BatchWork {
    BatchWork { forward_positions: a.forward_positions.min(b.forward_positions), projected_logits: a.projected_logits.min(b.projected_logits) }
}

#[cfg(test)]
mod tests;
