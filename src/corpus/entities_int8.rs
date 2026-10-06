//! Raw documents -> strict INT8 NER -> verified mentions -> complete resolution.
//!
//! A single consumed snapshot owns its exact sources and compact NER witnesses.
//! Only one source grammar is resident at a time. Pair planning uses the
//! nonrenewable remainder AFTER reserving all NER work, never a fresh allowance.
//! The caller owns model/runtime/preparation/output admission through delivery.

use std::{error::Error, fmt, mem::size_of, sync::Arc};
use serde::Serialize;
use crate::{
    batch::source::SourceMaskBudget,
    canonjson,
    execution_identity::{ExecutionIdentity, Sha256Digest},
    native_engine::{
        constrained::JsonWorkBudget,
        constrained_int8::{self, Int8JsonBudget},
        decode::{DecodeCancellationKind, DecodeStepControl},
        kv::KV_BYTES_PER_TOKEN,
        portable_int8::ProjectionWork,
        strict_int8::{Int8RunBudget, Int8Work, StrictInt8Engine, StrictInt8Error,
            STRICT_INT8_PROFILE, scoring::Int8ScoringBudget, prefill::Int8PrefillLimits},
    },
    tasks::{extract::ExtractionVocabulary, ir::{PlanContext, TaskBudget},
        ner::{NerOptions, NerResult, NER_TASK_VERSION},
        source_planning::{SourcePlanningLimits, SourceTaskPlanner, SourceTaskRequest, SourceTaskResult,
            quantized::{Int8SourceError, Int8SourceTaskRun, PreparedInt8SourceTask, INT8_SOURCE_EXECUTION}}},
    validation::grounded_fields::GroundingBudget,
};
use super::{
    entities::{EntityDocument, EntityVerificationWork},
    native_resolve::{ResolutionPlanner,
        quantized::{Int8ResolveError, Int8ResolveLimits, Int8ResolutionRun}},
    resolve::{self, ResolutionDocument, ResolutionPlan, ResolveError, ResolveLimits, ResolveOptions},
};
mod grounding;
mod prefill;
pub mod long;

pub const INT8_ENTITY_EXECUTION: &str = "portable-int8-ner-to-complete-corpus-resolution-v1";

/// Explicit whole-snapshot ceilings. The scoring limit is an additional
/// stage cap, NOT authority to renew max_model_work after extraction.
#[derive(Clone, Debug)]
pub struct Int8EntityConfig {
    pub ner: NerOptions,
    pub ner_budget: TaskBudget,
    pub source_planning: SourcePlanningLimits,
    pub masks: SourceMaskBudget,
    pub resolution: ResolveOptions,
    pub graph: ResolveLimits,
    pub scoring: Int8ResolveLimits,
    /// Occurrence-recovery work is shared by every document in the snapshot.
    pub verification: GroundingBudget,
    pub max_model_work: Int8Work,
    pub max_result_bytes: usize,
}

/// No source text, document id, generated JSON or private witness in diagnostics.
pub enum Int8EntityError {
    InvalidLimits, InvalidInput, Identity, WorkBudget, Accounting, Allocation,
    Source(Int8SourceError), Graph(ResolveError), Resolution(Int8ResolveError), Native(StrictInt8Error),
}
impl fmt::Display for Int8EntityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidLimits => "invalid int8 entity snapshot limits",
            Self::InvalidInput => "int8 entity snapshot requires bounded unique original documents",
            Self::Identity => "int8 entity stage or model identity differs",
            Self::WorkBudget => "int8 entity whole-snapshot allowance exceeded",
            Self::Accounting => "int8 entity execution diverged from its complete plan",
            Self::Allocation => "int8 entity allocation refused",
            Self::Source(_) => "int8 entity source NER failed",
            Self::Graph(_) => "int8 entity occurrence verification or graph refused",
            Self::Resolution(_) => "int8 entity complete pair scoring failed",
            Self::Native(_) => "int8 entity native engine refused",
        })
    }
}
impl fmt::Debug for Int8EntityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Display::fmt(self, f) }
}
impl Error for Int8EntityError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Source(e) => Some(e), Self::Graph(e) => Some(e),
            Self::Resolution(e) => Some(e), Self::Native(e) => Some(e), _ => None }
    }
}
impl From<Int8SourceError> for Int8EntityError { fn from(e: Int8SourceError) -> Self { Self::Source(e) } }
impl From<ResolveError> for Int8EntityError { fn from(e: ResolveError) -> Self { Self::Graph(e) } }
impl From<Int8ResolveError> for Int8EntityError { fn from(e: Int8ResolveError) -> Self { Self::Resolution(e) } }
impl From<StrictInt8Error> for Int8EntityError { fn from(e: StrictInt8Error) -> Self { Self::Native(e) } }
impl Int8EntityError {
    pub fn cancellation(&self) -> Option<DecodeCancellationKind> {
        match self {
            Self::Graph(ResolveError::Cancelled(c)) | Self::Native(StrictInt8Error::Cancelled(c)) => Some(*c),
            Self::Source(e) => e.cancellation(), Self::Resolution(e) => e.cancellation(), _ => None,
        }
    }
}

struct Input {
    document: EntityDocument,
    witness: Sha256Digest,
    prompt_tokens: usize,
    work: Int8Work,
}

/// No public constructor, Clone or Deserialize: preflight cannot be skipped or
/// a caller-supplied mention/score list substituted for original documents.
/// Arcs share pinned immutable planners, never a second runtime or model.
pub struct PreparedInt8EntityCorpus {
    inputs: Vec<Input>,
    source: Arc<SourceTaskPlanner>,
    resolver: Arc<ResolutionPlanner>,
    source_identity: ExecutionIdentity,
    resolution_identity: ExecutionIdentity,
    config: Int8EntityConfig,
    ner_work: Int8Work,
    mask_visits: u64,
    maximum_ner_positions: usize,
    input_bytes: usize,
    prefill: Option<Int8PrefillLimits>,
}

/// Plan EVERY document before any neural call. The graph cannot be known until
/// NER completes, so only its fixed policy/compiler is checked at this stage.
#[allow(clippy::too_many_arguments)]
pub fn prepare_int8_entities<C: DecodeStepControl>(mut documents: Vec<EntityDocument>,
    source: Arc<SourceTaskPlanner>, source_identity: ExecutionIdentity,
    resolver: Arc<ResolutionPlanner>, resolution_identity: ExecutionIdentity,
    config: Int8EntityConfig, control: &mut C) -> Result<PreparedInt8EntityCorpus, Int8EntityError> {
    resolve::checkpoint(control)?;
    validate(&config, &source, &source_identity, &resolution_identity)?;
    let empty = ResolutionPlan::prepare(&[], config.resolution, config.graph, control)?;
    resolver.prepare_int8(&empty, &resolution_identity, config.scoring, control)?;
    let input_bytes = validate_documents(&mut documents, &config)?;
    let mask_visits = (documents.len() as u64).checked_mul(config.masks.max_visits_per_item)
        .filter(|&n| n <= config.masks.max_visits_per_run).ok_or(Int8EntityError::WorkBudget)?;
    let mut inputs = reserved(documents.len())?;
    let mut ner_work = Int8Work::default(); let mut maximum_ner_positions = 0;
    for document in documents {
        resolve::checkpoint(control)?;
        let plan = source_plan(&document.text, &source, &source_identity, &config, control)?;
        let work = plan.planned_work();
        ner_work = plus(ner_work, work)?;
        if !within(ner_work, config.max_model_work) { return Err(Int8EntityError::WorkBudget); }
        maximum_ner_positions = maximum_ner_positions.max(usize::try_from(work.forward_positions)
            .map_err(|_| Int8EntityError::WorkBudget)?);
        inputs.push(Input { document, witness: witness(plan.execution_identity())?,
            prompt_tokens: plan.prompt_tokens(), work });
        // The plan/grammar/index is dropped before preparing the next source.
    }
    resolve::checkpoint(control)?;
    Ok(PreparedInt8EntityCorpus { inputs, source, resolver, source_identity, resolution_identity,
        config, ner_work, mask_visits, maximum_ner_positions, input_bytes, prefill: None })
}
impl PreparedInt8EntityCorpus {
    /// Select physical prompt scheduling before transferring this consumed
    /// snapshot to its admission owner. Applies to BOTH NER and pair scoring;
    /// exact identities, source witnesses and work budgets are unchanged.
    pub fn with_layer_major_prefill(mut self, limits: Int8PrefillLimits) -> Result<Self, Int8EntityError> {
        prefill::configure(&mut self.prefill, limits)?;
        Ok(self)
    }
    /// The host must price and retain this extra workspace through all stages.
    pub fn prefill_limits(&self) -> Option<Int8PrefillLimits> { self.prefill }
    pub fn document_count(&self) -> usize { self.inputs.len() }
    pub fn input_bytes(&self) -> usize { self.input_bytes }
    pub fn ner_reserved_work(&self) -> Int8Work { self.ner_work }
    pub fn reserved_mask_visits(&self) -> u64 { self.mask_visits }
    pub fn required_ner_context_tokens(&self) -> usize { self.maximum_ner_positions }
    pub fn source_identity(&self) -> &ExecutionIdentity { &self.source_identity }
    pub fn resolution_identity(&self) -> &ExecutionIdentity { &self.resolution_identity }
    pub fn config(&self) -> &Int8EntityConfig { &self.config }
    /// Actual retained raw String/Vec capacities. Pinned planner/configuration,
    /// temporary grammars, expanded mentions and allocator overhead are separate
    /// host preparation/graph commitments, not falsely included in this count.
    pub fn retained_input_bytes(&self) -> Result<u64, Int8EntityError> {
        let mut bytes = (self.inputs.capacity() as u64).checked_mul(size_of::<Input>() as u64)
            .ok_or(Int8EntityError::WorkBudget)?;
        for input in &self.inputs {
            bytes = bytes.checked_add(input.document.id.capacity() as u64)
                .and_then(|n| n.checked_add(input.document.text.capacity() as u64)).ok_or(Int8EntityError::WorkBudget)?;
        }
        Ok(bytes)
    }
    /// Reuse ONE actual strict-INT8 engine for every NER pass and pair head.
    /// Consumes the snapshot; failed or unwound operations cannot be retried
    /// through this value with renewed allowances. Caller discards a failed
    /// engine and retains its memory authority until all storage has drained.
    pub fn execute_with_control<C: DecodeStepControl>(self, engine: &mut StrictInt8Engine<'_>,
        vocabulary: &ExtractionVocabulary, control: &mut C) -> Result<Int8EntityRun, Int8EntityError> {
        resolve::checkpoint(control)?;
        check_engine(&self, engine)?;
        let Self { inputs, source, resolver, source_identity, resolution_identity,
            config, ner_work, mask_visits, prefill, .. } = self;
        let maps = collect(inputs, &config, control, |input, control| {
            let plan = source_plan(&input.document.text, &source, &source_identity, &config, control)?;
            check_rebuilt(input, &plan)?;
            let work = input.work;
            let result = prefill::source(&plan, engine, vocabulary,
                Int8JsonBudget { native: Int8RunBudget::exact(work), json: JsonWorkBudget {
                    max_forward_positions: work.forward_positions, max_projected_logits: work.projected_logits,
                    max_kv_bytes: config.ner_budget.max_kv_bytes,
                    max_total_mask_node_visits: config.masks.max_visits_per_item, mask_limits: config.masks.per_mask,
                } }, prefill, control)?;
            if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) { return Err(Int8EntityError::Accounting); }
            Ok(result)
        })?;
        resolve::checkpoint(control)?;
        let plan = ResolutionPlan::prepare(&maps.documents, config.resolution, config.graph, control)?;
        let scoring = remaining_scoring(&config, ner_work)?;
        let prepared = resolver.prepare_int8(&plan, &resolution_identity, scoring, control)?;
        let pair_work = prepared.planned_work();
        let mut admitted = reserved(prepared.pair_count())?;
        for identity in prepared.execution_identities() {
            resolve::checkpoint(control)?;
            admitted.push(identity.clone());
        }
        // The concrete resolver independently validates all identities, native
        // schedules and real model bindings before its first pair forward.
        let resolution = if prepared.pair_count() == 0 {
            prepared.finalize_without_model(control)?
        } else {
            prefill::resolution(prepared, &admitted, engine, Int8ScoringBudget {
                native: Int8RunBudget::exact(pair_work), max_kv_bytes: config.scoring.planning.per_head.max_kv_bytes,
            }, prefill, control)?
        };
        drop(plan);
        finish(maps, resolution, ner_work, mask_visits, &config, control)
    }

    /// Only an EMPTY raw snapshot can omit NER. A nonempty document list with
    /// no caller-supplied mentions is not evidence that there are no entities.
    pub fn finalize_without_model<C: DecodeStepControl>(self, control: &mut C) -> Result<Int8EntityRun, Int8EntityError> {
        resolve::checkpoint(control)?;
        if !self.inputs.is_empty() || self.ner_work != Int8Work::default() || self.mask_visits != 0 {
            return Err(Int8EntityError::Accounting);
        }
        let plan = ResolutionPlan::prepare(&[], self.config.resolution, self.config.graph, control)?;
        let prepared = self.resolver.prepare_int8(&plan, &self.resolution_identity, self.config.scoring, control)?;
        let resolution = prepared.finalize_without_model(control)?;
        let maps = Extracted { documents: Vec::new(), receipts: Vec::new(), work: Int8Work::default(),
            mask_visits: 0, verification_used: EntityVerificationWork::default() };
        finish(maps, resolution, self.ner_work, self.mask_visits, &self.config, control)
    }
}

#[derive(Serialize)]
pub struct Int8EntityDocumentReceipt {
    pub document_id: String,
    pub proposed_entities: usize,
    pub anchored_mentions: usize,
    pub model_work: Int8Work,
    pub mask_node_visits: u64,
}
/// Completed sensitive snapshot, not telemetry. Exact mention surfaces remain
/// in resolution; private source/prompt hashes and NER token transcripts do not.
#[derive(Serialize)]
pub struct Int8EntityRun {
    pub schema_version: u32,
    pub execution: &'static str,
    pub numerics_profile: &'static str,
    pub extraction_task: &'static str,
    pub documents: Vec<Int8EntityDocumentReceipt>,
    pub ner_reserved_work: Int8Work,
    pub ner_model_work: Int8Work,
    pub reserved_model_work: Int8Work,
    pub model_work: Int8Work,
    pub reserved_mask_node_visits: u64,
    pub mask_node_visit_charge: u64,
    pub verification_used: EntityVerificationWork,
    pub resolution: Int8ResolutionRun,
}
struct Extracted {
    documents: Vec<ResolutionDocument>, receipts: Vec<Int8EntityDocumentReceipt>, work: Int8Work,
    mask_visits: u64, verification_used: EntityVerificationWork,
}
// Private fault-injection seam. Public execution always uses the concrete
// source planner/engine above, never a plug-in or fabricated inference receipt.
fn collect<C: DecodeStepControl, F>(inputs: Vec<Input>, config: &Int8EntityConfig, control: &mut C,
    mut execute: F) -> Result<Extracted, Int8EntityError>
where F: FnMut(&Input, &mut C) -> Result<Int8SourceTaskRun, Int8EntityError> {
    let mut documents = reserved(inputs.len())?; let mut receipts = reserved(inputs.len())?;
    let mut verification = config.verification;
    let mut work = Int8Work::default(); let mut masks = 0_u64;
    let (mut bytes, mut mentions) = (0_usize, 0_usize);
    for input in inputs {
        resolve::checkpoint(control)?;
        let run = execute(&input, control)?;
        let (ner, actual) = check_run(&input, run, config)?;
        work = plus(work, actual)?;
        masks = masks.checked_add(ner.mask_node_visit_charge)
            .filter(|&n| n <= config.masks.max_visits_per_run).ok_or(Int8EntityError::WorkBudget)?;
        let proposed_entities = ner.entities.len(); let mask_node_visits = ner.mask_node_visit_charge;
        let document = grounding::expand(input.document, ner, config, &mut verification, control)?;
        grounding::charge(&document, &mut bytes, &mut mentions, config.graph)?;
        receipts.push(Int8EntityDocumentReceipt { document_id: copy(&document.id)?, proposed_entities,
            anchored_mentions: document.mentions.len(), model_work: actual, mask_node_visits });
        documents.push(document);
    }
    resolve::checkpoint(control)?;
    let verification_used = EntityVerificationWork {
        fields: config.verification.max_fields - verification.max_fields,
        matches: config.verification.max_matches - verification.max_matches,
        scan_steps: config.verification.max_scan_steps - verification.max_scan_steps,
    };
    Ok(Extracted { documents, receipts, work, mask_visits: masks, verification_used })
}
fn finish<C: DecodeStepControl>(maps: Extracted, resolution: Int8ResolutionRun, ner_work: Int8Work,
    mask_visits: u64, config: &Int8EntityConfig, control: &mut C) -> Result<Int8EntityRun, Int8EntityError> {
    resolve::checkpoint(control)?;
    let reserved_model_work = plus(ner_work, resolution.planned_model_work)?;
    let model_work = plus(maps.work, resolution.model_work)?;
    if !within(maps.work, ner_work) || !within(model_work, reserved_model_work)
        || !within(reserved_model_work, config.max_model_work) || maps.mask_visits > mask_visits
        || resolution.result.document_count != maps.receipts.len() { return Err(Int8EntityError::Accounting); }
    let Extracted { documents, receipts, work, mask_visits: actual_masks, verification_used } = maps;
    drop(documents);
    let output = Int8EntityRun { schema_version: 1, execution: INT8_ENTITY_EXECUTION,
        numerics_profile: STRICT_INT8_PROFILE, extraction_task: NER_TASK_VERSION,
        documents: receipts, ner_reserved_work: ner_work, ner_model_work: work,
        reserved_model_work, model_work, reserved_mask_node_visits: mask_visits,
        mask_node_visit_charge: actual_masks, verification_used, resolution };
    resolve::check_output(&output, config.max_result_bytes)?;
    resolve::checkpoint(control)?;
    Ok(output)
}
fn source_plan<C: DecodeStepControl>(text: &str, planner: &SourceTaskPlanner, identity: &ExecutionIdentity,
    config: &Int8EntityConfig, control: &mut C) -> Result<PreparedInt8SourceTask, Int8EntityError> {
    resolve::checkpoint(control)?;
    let request = SourceTaskRequest::Ner { document: copy(text)?, options: config.ner.clone(), budget: config.ner_budget };
    let context = PlanContext::new(identity, config.ner_budget).map_err(|_| Int8EntityError::Identity)?;
    Ok(planner.plan_int8_with_control(&request, &context, config.source_planning, control)?)
}
fn check_rebuilt(input: &Input, plan: &PreparedInt8SourceTask) -> Result<(), Int8EntityError> {
    if input.witness != witness(plan.execution_identity())? || input.work != plan.planned_work()
        || input.prompt_tokens != plan.prompt_tokens() { return Err(Int8EntityError::Accounting); }
    Ok(())
}
fn check_run(input: &Input, run: Int8SourceTaskRun, config: &Int8EntityConfig) -> Result<(NerResult, Int8Work), Int8EntityError> {
    if run.schema_version != 1 || run.execution != INT8_SOURCE_EXECUTION { return Err(Int8EntityError::Accounting); }
    let SourceTaskResult::Ner(ner) = run.result else { return Err(Int8EntityError::Accounting); };
    let expected = constrained_int8::planned_work(input.prompt_tokens, ner.generated_token_ids.len())
        .map_err(|_| Int8EntityError::Accounting)?;
    if ner.generated_token_ids.len() > config.ner_budget.max_output_tokens as usize
        || run.model_work != expected || !within(expected, input.work)
        || ner.forward_positions != expected.forward_positions || ner.projected_logits != expected.projected_logits
        || ner.mask_node_visit_charge > config.masks.max_visits_per_item { return Err(Int8EntityError::Accounting); }
    Ok((ner, expected))
}
fn remaining_scoring(config: &Int8EntityConfig, reserved_ner: Int8Work) -> Result<Int8ResolveLimits, Int8EntityError> {
    let mut scoring = config.scoring;
    scoring.planning.max_pairs = scoring.planning.max_pairs.min(config.graph.max_candidate_pairs);
    scoring.max_model_work = meet(scoring.max_model_work, subtract(config.max_model_work, reserved_ner)?);
    Ok(scoring)
}
fn validate(config: &Int8EntityConfig, source: &SourceTaskPlanner, a: &ExecutionIdentity, b: &ExecutionIdentity)
    -> Result<(), Int8EntityError> {
    config.ner.validate().map_err(|_| Int8EntityError::InvalidLimits)?;
    config.ner_budget.validate().map_err(|_| Int8EntityError::InvalidLimits)?;
    a.validate().map_err(|_| Int8EntityError::Identity)?;
    b.validate().map_err(|_| Int8EntityError::Identity)?;
    constrained_int8::check_profile(a).map_err(|_| Int8EntityError::Identity)?;
    constrained_int8::check_profile(b).map_err(|_| Int8EntityError::Identity)?;
    resolve::check_output(a, 16_384)?; resolve::check_output(b, 16_384)?;
    if a.task_spec != NER_TASK_VERSION || a.tokenizer_digest != source.tokenizer_digest()
        || a.template_digest != *source.template_digest() { return Err(Int8EntityError::Identity); }
    same_model(a, b)?;
    if config.masks.max_visits_per_item == 0 || config.masks.per_mask.max_trie_node_visits == 0
        || config.masks.per_mask.checkpoint_interval_nodes == 0
        || !(1..=64 * 1024 * 1024).contains(&config.max_result_bytes) {
        return Err(Int8EntityError::InvalidLimits);
    }
    Ok(())
}
fn same_model(a: &ExecutionIdentity, b: &ExecutionIdentity) -> Result<(), Int8EntityError> {
    // Enumerate TASK-owned differences only. All present and future model/host
    // fields remain included by full structural equality, not a short allowlist.
    let mut normalized = a.clone();
    normalized.template_digest = b.template_digest; normalized.task_spec = b.task_spec.clone();
    normalized.taskir_digest = b.taskir_digest; normalized.prompt_digest = b.prompt_digest;
    normalized.grammar_compiler_version = b.grammar_compiler_version.clone(); normalized.schema_digest = b.schema_digest;
    normalized.sampler_version = b.sampler_version.clone(); normalized.calibration_digest = b.calibration_digest;
    normalized.decision_policy_digest = b.decision_policy_digest;
    if &normalized != b { return Err(Int8EntityError::Identity); }
    Ok(())
}
fn validate_documents(documents: &mut [EntityDocument], config: &Int8EntityConfig) -> Result<usize, Int8EntityError> {
    if documents.len() > config.graph.max_documents { return Err(Int8EntityError::WorkBudget); }
    documents.sort_unstable_by(|a, b| a.id.cmp(&b.id));
    let mut bytes = 0_usize;
    for (index, d) in documents.iter().enumerate() {
        if d.id.is_empty() || d.id.len() > 256 || d.id.chars().any(char::is_control)
            || (index > 0 && documents[index - 1].id == d.id) { return Err(Int8EntityError::InvalidInput); }
        if d.text.len() > config.source_planning.max_input_bytes { return Err(Int8EntityError::WorkBudget); }
        bytes = bytes.checked_add(d.id.len()).and_then(|n| n.checked_add(d.text.len()))
            .filter(|&n| n <= config.graph.max_input_bytes).ok_or(Int8EntityError::WorkBudget)?;
    }
    Ok(bytes)
}
fn check_engine(prepared: &PreparedInt8EntityCorpus, engine: &StrictInt8Engine<'_>) -> Result<(), Int8EntityError> {
    let identity = &prepared.source_identity; let model = engine.artifact_identity();
    if model.model_id != "Nanbeige4.2-3B" || model.revision != identity.source_revision || model.recipe_id != identity.quant_recipe
        || Sha256Digest::from_hex(&model.logical_model_sha256).ok() != Some(identity.logical_model_digest) {
        return Err(Int8EntityError::Identity);
    }
    if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) { return Err(Int8EntityError::Accounting); }
    let positions = engine.kv_cache().capacity_positions();
    let cap = prepared.config.ner_budget.max_kv_bytes.min(prepared.config.scoring.planning.per_head.max_kv_bytes);
    if positions < prepared.maximum_ner_positions || (positions as u64).checked_mul(KV_BYTES_PER_TOKEN as u64)
        .is_none_or(|n| n > cap) { return Err(Int8EntityError::WorkBudget); }
    Ok(())
}
fn witness(id: &ExecutionIdentity) -> Result<Sha256Digest, Int8EntityError> {
    Ok(Sha256Digest::of_bytes(&canonjson::canonical_bytes(id).map_err(|_| Int8EntityError::Accounting)?))
}
fn copy(text: &str) -> Result<String, Int8EntityError> {
    let mut value = String::new(); value.try_reserve_exact(text.len()).map_err(|_| Int8EntityError::Allocation)?;
    value.push_str(text); Ok(value)
}
fn reserved<T>(count: usize) -> Result<Vec<T>, Int8EntityError> {
    let mut value = Vec::new(); value.try_reserve_exact(count).map_err(|_| Int8EntityError::Allocation)?; Ok(value)
}
fn plus(a: Int8Work, b: Int8Work) -> Result<Int8Work, Int8EntityError> {
    a.checked_add(b).map_err(|_| Int8EntityError::WorkBudget)
}
fn within(a: Int8Work, b: Int8Work) -> bool {
    a.forward_positions <= b.forward_positions && a.projected_logits <= b.projected_logits
        && a.attention_pairs <= b.attention_pairs && a.projections.fits(b.projections)
}
fn meet(a: Int8Work, b: Int8Work) -> Int8Work {
    Int8Work { forward_positions: a.forward_positions.min(b.forward_positions),
        projected_logits: a.projected_logits.min(b.projected_logits), attention_pairs: a.attention_pairs.min(b.attention_pairs),
        projections: ProjectionWork { dot_products: a.projections.dot_products.min(b.projections.dot_products),
            multiply_accumulates: a.projections.multiply_accumulates.min(b.projections.multiply_accumulates) } }
}
fn subtract(a: Int8Work, b: Int8Work) -> Result<Int8Work, Int8EntityError> {
    let sub = |a: u64, b: u64| a.checked_sub(b).ok_or(Int8EntityError::WorkBudget);
    Ok(Int8Work { forward_positions: sub(a.forward_positions, b.forward_positions)?,
        projected_logits: sub(a.projected_logits, b.projected_logits)?, attention_pairs: sub(a.attention_pairs, b.attention_pairs)?,
        projections: ProjectionWork { dot_products: sub(a.projections.dot_products, b.projections.dot_products)?,
            multiply_accumulates: sub(a.projections.multiply_accumulates, b.projections.multiply_accumulates)? } })
}

#[cfg(test)] mod tests;
