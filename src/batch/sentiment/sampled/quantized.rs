//! Nonrenewable native work and admission for resident sampled INT8 populations.
//! Complete document adapters share the existing transport arguments, while
//! all five native counters and mask work remain one invocation-wide ledger.
use std::sync::Arc;
use super::{SampledSentimentBatchArgs, item_budget};
use crate::{
    batch::{BatchCode, BatchDocument, BatchFault, BatchItemFailure, BatchProcessor,
        BatchRequestContext, BatchWork, extract::ExtractionMaskBudget, output::GuardedOutput},
    execution_identity::{ExecutionIdentity, Sha256Digest},
    native_engine::{artifact_bridge::ArtifactIdentity, constrained::JsonWorkBudget,
        constrained_int8::{self, Int8JsonBudget, JSON_SAMPLER_VERSION, json_sampling_workspace_bytes},
        decode::DecodeStepControl, portable_int8::ProjectionWork,
        strict_int8::{Int8MemoryRequirement, Int8RunBudget, Int8Work, StrictInt8Engine,
            STRICT_INT8_PROFILE, prefill::Int8PrefillLimits}},
    tasks::{ir::TaskBudget, sentiment::{SentimentAxis, SentimentRequest, sampled::{
        SampledSentimentLimits, SampledSentimentOptions, SampledSentimentVocabulary,
        quantized::{Int8SampledSentimentError, Int8SampledSentimentPlanner}}}},
    tokenizer::untrusted::{SafeBpeWork, UntrustedEncoding},
};

pub mod long;

/// The item cap covers a whole document; the run cap spans all documents and
/// failed attempts. Neither flush, early EOS nor rejected admission refunds it.
#[derive(Clone, Copy, Debug)]
pub struct Int8SampledSentimentBatchLimits {
    pub max_item_work: Int8Work,
    pub max_model_work: Int8Work,
    pub masks: ExtractionMaskBudget,
    pub max_sampler_bytes: u64,
}
impl Int8SampledSentimentBatchLimits {
    pub fn validate(self) -> Result<(), BatchFault> {
        let masks = self.masks;
        if !nonzero(self.max_item_work) || !nonzero(self.max_model_work)
            || !fits(self.max_item_work, self.max_model_work)
            || masks.per_mask.max_trie_node_visits == 0 || masks.per_mask.checkpoint_interval_nodes == 0
            || masks.max_visits_per_item == 0 || masks.max_visits_per_run < masks.max_visits_per_item
            || self.max_sampler_bytes < json_sampling_workspace_bytes().map_err(|_| BatchCode::InvalidLimits)? {
            return Err(BatchCode::InvalidLimits.into());
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum State { Ready, Running, Failed }
struct RunState {
    limits: Int8SampledSentimentBatchLimits,
    remaining: Int8Work,
    remaining_masks: u64,
    last_sequence: u64,
    state: State,
}
impl RunState {
    fn new(limits: Int8SampledSentimentBatchLimits) -> Self {
        Self { limits, remaining: limits.max_model_work, remaining_masks: limits.masks.max_visits_per_run,
            last_sequence: 0, state: State::Ready }
    }
    fn enter(&mut self) -> Result<(), BatchItemFailure> {
        if self.state != State::Ready { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
        self.state = State::Running;
        Ok(())
    }
    fn check_available(&self, work: Int8Work, masks: u64) -> Result<(), BatchItemFailure> {
        subtract(self.remaining, work)?;
        if masks > self.remaining_masks { return Err(BatchItemFailure::reject(BatchCode::WorkLimit)); }
        Ok(())
    }
    fn charge(&mut self, work: Int8Work, masks: u64, context: BatchRequestContext) -> Result<(), BatchItemFailure> {
        if self.state != State::Running || context.request_seq <= self.last_sequence
            || context.epoch == 0 || context.input_line == 0 {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        let remaining = subtract(self.remaining, work).map_err(|e| BatchItemFailure::fatal(e.fault))?;
        let remaining_masks = self.remaining_masks.checked_sub(masks)
            .ok_or_else(|| BatchItemFailure::fatal(BatchCode::WorkLimit))?;
        self.remaining = remaining;
        self.remaining_masks = remaining_masks;
        self.last_sequence = context.request_seq;
        Ok(())
    }
}
fn settle<T>(result: Result<T, BatchItemFailure>, clean: bool, run: &mut RunState) -> Result<T, BatchItemFailure> {
    match result {
        Ok(value) if clean && run.state == State::Running => { run.state = State::Ready; Ok(value) }
        Ok(_) => { run.state = State::Failed; Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)) }
        Err(mut error) => {
            if !clean || run.state != State::Running || error.must_stop() { error.stop = true; }
            run.state = if error.stop { State::Failed } else { State::Ready };
            Err(error)
        }
    }
}
fn nonzero(work: Int8Work) -> bool {
    work.forward_positions != 0 && work.projected_logits != 0 && work.attention_pairs != 0
        && work.projections.dot_products != 0 && work.projections.multiply_accumulates != 0
}
fn fits(work: Int8Work, cap: Int8Work) -> bool {
    work.forward_positions <= cap.forward_positions && work.projected_logits <= cap.projected_logits
        && work.attention_pairs <= cap.attention_pairs && work.projections.fits(cap.projections)
}
fn subtract(a: Int8Work, b: Int8Work) -> Result<Int8Work, BatchItemFailure> {
    let sub = |a: u64, b: u64| a.checked_sub(b).ok_or_else(|| BatchItemFailure::reject(BatchCode::WorkLimit));
    Ok(Int8Work { forward_positions: sub(a.forward_positions, b.forward_positions)?,
        projected_logits: sub(a.projected_logits, b.projected_logits)?,
        attention_pairs: sub(a.attention_pairs, b.attention_pairs)?,
        projections: ProjectionWork { dot_products: sub(a.projections.dot_products, b.projections.dot_products)?,
            multiply_accumulates: sub(a.projections.multiply_accumulates, b.projections.multiply_accumulates)? } })
}
fn native_budget(work: Int8Work, kv_bytes: u64, masks: u64, limits: Int8SampledSentimentBatchLimits)
    -> Int8JsonBudget {
    Int8JsonBudget { native: Int8RunBudget::exact(work), json: JsonWorkBudget {
        max_forward_positions: work.forward_positions, max_projected_logits: work.projected_logits,
        max_kv_bytes: kv_bytes, max_total_mask_node_visits: masks, mask_limits: limits.masks.per_mask } }
}
fn transport_work(work: Int8Work) -> BatchWork {
    BatchWork { forward_positions: work.forward_positions, projected_logits: work.projected_logits }
}
fn clean(engine: &StrictInt8Engine<'_>) -> bool {
    !engine.is_poisoned() && engine.profile() == STRICT_INT8_PROFILE && engine.kv_cache().all_slots_have_len(0)
}
fn check_model(identity: &ExecutionIdentity, model: &ArtifactIdentity) -> Result<(), BatchFault> {
    constrained_int8::check_profile(identity).map_err(|_| BatchCode::Admission)?;
    if model.model_id != "Nanbeige4.2-3B" || model.revision != identity.source_revision
        || model.recipe_id != identity.quant_recipe
        || Sha256Digest::from_hex(&model.logical_model_sha256).ok() != Some(identity.logical_model_digest) {
        return Err(BatchCode::Admission.into());
    }
    Ok(())
}
fn validate_prefill(prefill: Option<Int8PrefillLimits>) -> Result<(), BatchFault> {
    if let Some(limits) = prefill { limits.validate().map_err(|_| BatchCode::InvalidLimits)?; }
    Ok(())
}
fn task_failure(error: Int8SampledSentimentError) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    match error {
        Int8SampledSentimentError::Shared(error) => super::task_failure(error),
        Int8SampledSentimentError::Native(_) => BatchItemFailure::fatal(BatchCode::Execution),
    }
}
fn checkpoint<C: DecodeStepControl>(control: &mut C) -> Result<(), BatchItemFailure> {
    match control.prefill_checkpoint(0) {
        Some(cause) => Err(BatchItemFailure::fatal(BatchFault::cancelled(cause))), None => Ok(()),
    }
}
