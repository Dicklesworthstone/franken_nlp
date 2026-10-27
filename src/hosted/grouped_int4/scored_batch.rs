//! Complete ordered scoring streams inside one physical native INT4 invocation.
use super::*;
use std::io::{BufRead, Write};
use crate::{
    batch::{self as stream, BatchCode, BatchItemFailure, BatchSummary,
        scored_int4::{Int4ScoredAdmission, Int4ScoredBatchAdmission, Int4ScoredBatchLimits,
            Int4ScoredBatchPlanner, Int4ScoredTask, NativeInt4ScoredBatch}},
    hosted::corpus::CorpusLimits,
    native_engine::{constrained_int4, strict_int4::Int4Work},
};

struct StreamInput<P: Int4ScoredTask, R, W> {
    planner: Option<Int4ScoredBatchPlanner<P>>, reader: R, writer: W,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Requirements { native: Int4MemoryRequirement, workspace_bytes: u64, input_bytes: u64 }

impl NlpEngine {
    /// Run one fixed scoring task over the existing bounded NDJSON transport.
    /// P is sealed to the three actual model-bound classification, sentiment
    /// and judge planners; task_args retain their existing per-task schemas.
    /// One resident model and one native context serve the entire serial stream.
    ///
    /// The caller prices the owned planner/defaults and reader/writer buffers
    /// in CorpusLimits. This host retains those charges, native workspace and
    /// real output guards through physical execution and writer delivery.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int4_scored<P, R, W>(&self, model: &ResidentInt4,
        planner: Int4ScoredBatchPlanner<P>, limits: CorpusLimits, kernel: PackedInt4Kernel,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where P: Int4ScoredTask, R: BufRead + Send + 'static, W: Write + Send + 'static {
        dispatch::preflight(self, limits.native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        constrained_int4::check_model(planner.base_execution_identity(), model.model_identity())
            .map_err(|_| HostedError::ModelIdentity)?;
        if planner.group_width() != model.model_identity().group_width()
            || planner.base_execution_identity().task_spec != P::TASK {
            return Err(HostedError::ModelIdentity);
        }
        let work = planner.limits();
        let required = requirements(model.model_identity().group_width(), work, limits)?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes)?,
            || Ok(StreamInput { planner: Some(planner), reader, writer }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let model = model.clone();
        dispatch::run(self, limits.native.run, cancellation, move |control| {
            let mut input = input;
            checkpoint(control)?;
            let planner = input.value.planner.take().ok_or(HostedError::CompletionMissing)?;
            let mut engine = allocate_native(kv, scratch, || {
                StrictInt4Engine::new(&model.inner.loaded.value, limits.native.context_tokens,
                    Int4MemoryBudget { max_kv_bytes: required.native.kv_bytes,
                        max_rope_bytes: required.native.rope_bytes,
                        max_scratch_payload_bytes: required.native.scratch_payload_bound })
                    .map(|engine| engine.with_kernel(kernel)).map_err(native_error)
            })?;
            let admission = ScoredAdmission { lease: &lease, model: model.model_identity(),
                context: limits.native.context_tokens, kv_bytes: required.native.kv_bytes,
                max_item_work: work.max_item_work, task_spec: P::TASK,
                output_bytes: limits.transport.max_output_line_bytes as u64 };
            let result = {
                let mut processor = NativeInt4ScoredBatch::new(planner, &mut engine.value, admission)
                    .map_err(HostedError::BatchSetup)?;
                stream::run_ndjson(&mut input.value.reader, &mut input.value.writer,
                    &mut processor, limits.transport, control).map_err(HostedError::Batch)
            };
            drop(engine);
            drop(input);
            drop(lease);
            result
        })
    }
}

fn requirements(width: usize, work: Int4ScoredBatchLimits, limits: CorpusLimits) -> Result<Requirements, HostedError> {
    work.validate().map_err(HostedError::BatchSetup)?;
    let input_bytes = limits.reservation_bytes()?;
    if limits.native.allocator_reserve_bytes == 0 { return Err(HostedError::Limits("INT4 scoring stream allocator reserve")); }
    let native = Int4MemoryRequirement::for_context(limits.native.context_tokens, width).map_err(native_error)?;
    let workspace_bytes = sum(&[native.rope_bytes, native.scratch_payload_bound, limits.native.allocator_reserve_bytes])?;
    Ok(Requirements { native, workspace_bytes, input_bytes })
}

struct ScoredAdmission<'a> {
    lease: &'a EngineLease,
    model: &'a Int4ModelIdentity,
    context: usize,
    kv_bytes: u64,
    max_item_work: Int4Work,
    task_spec: &'static str,
    output_bytes: u64,
}
impl Int4ScoredBatchAdmission for ScoredAdmission<'_> {
    type Guard = Pending;
    fn admit(&mut self, request: Int4ScoredAdmission<'_>) -> Result<(ExecutionIdentity, Pending), BatchItemFailure> {
        constrained_int4::check_model(request.facts.identity, self.model)
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        check_request(self.model.group_width(), self.context, self.kv_bytes, self.max_item_work,
            self.task_spec, self.output_bytes, &request)?;
        // Score outputs retain candidate strings/distributions, not generated
        // token arrays. Reserve their complete result bytes through sink flush.
        let guard = output_claim(self.lease, request.facts.max_result_bytes, 0).map_err(|error| match error {
            HostedError::Reservation(_) => BatchItemFailure::fatal(BatchCode::Admission),
            _ => BatchItemFailure::fatal(BatchCode::InvalidExecution),
        })?;
        Ok((request.facts.identity.clone(), guard))
    }
}

#[allow(clippy::too_many_arguments)]
fn check_request(width: usize, context: usize, kv_bytes: u64, ceiling: Int4Work,
    task_spec: &str, output_bytes: u64, request: &Int4ScoredAdmission<'_>) -> Result<(), BatchItemFailure> {
    let facts = &request.facts;
    constrained_int4::check_profile(facts.identity, width).map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
    if !matches!(task_spec, "classify-v1" | "sentiment-v1" | "judge-v1") || facts.identity.task_spec != task_spec
        || facts.group_width != width || request.kv_reservation_bytes != kv_bytes
        || facts.task_budget.validate().is_err() {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    let required = Int4MemoryRequirement::for_context(context, width)
        .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
    if required.kv_bytes != kv_bytes || facts.required_context == 0 || facts.required_context > context
        || required.kv_bytes > facts.task_budget.max_kv_bytes {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    let work = facts.model_work;
    if facts.head_count == 0 || work.forward_positions == 0 || work.projected_logits == 0
        || work.attention_pairs == 0 || work.projections.dot_products == 0 || work.projections.multiply_accumulates == 0
        || work.forward_positions > ceiling.forward_positions || work.projected_logits > ceiling.projected_logits
        || work.attention_pairs > ceiling.attention_pairs || !work.projections.fits(ceiling.projections) {
        return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
    }
    // A bundle may perform more forwards than context slots and a conditional
    // sentiment head may project only its finite candidate rows. Neither is a
    // single dense-projection causal sequence; native plans verify each head.
    if facts.max_result_bytes == 0 || facts.max_result_bytes > facts.task_budget.max_output_bytes
        || facts.max_result_bytes > output_bytes { return Err(BatchItemFailure::reject(BatchCode::OutputLineLimit)); }
    Ok(())
}

#[cfg(test)]
mod tests;
