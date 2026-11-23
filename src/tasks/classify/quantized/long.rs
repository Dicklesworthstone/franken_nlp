//! Complete-document classification on the existing native INT8 scorer.
//!
//! Every original-source chunk and independent label head is planned before
//! native execution. One work allowance spans the whole document; one cache
//! is reused serially. A source-scalar mixture is an explicit descriptive
//! reduction, never a full-context model probability or calibrated confidence.
use super::*;
use std::{cell::RefCell, io};
use sha2::{Digest, Sha256};
use crate::{
    tasks::{
        classify::{ClassificationMode, ClassificationPolicy, ClassificationAggregate,
            ClassificationAggregationPolicy, ClassificationAggregationError,
            aggregation::{ClassificationChunkView, aggregate_with_control}},
        ir::PromptSegmentKind,
        mapreduce::{CHUNK_PROFILE, ChunkLimits, ChunkPlan, MapReduceError},
    },
    tokenizer::untrusted::{SafeBpeLimits, SafeBpeWork, UntrustedEncoding},
    validation::grounded_fields::VerifiedSourceSpan,
};

pub const INT8_CLASSIFICATION_MAP_EXECUTION: &str = "portable-int8-complete-classification-map-v1";
pub const MAX_CLASSIFICATION_MAP_CHUNKS: usize = 256;
const MAX_BYTES: u64 = 64 * 1024 * 1024;
const SEMANTICS: &str = "independent-chunks-flat-source-scalar-mixture-v1";
const CHECKPOINT_BYTES: usize = 64 * 1024;

#[derive(Clone, Copy, Debug)]
pub struct Int8ClassificationMapLimits {
    pub chunks: ChunkLimits,
    /// Complete-document allowance, never renewed by another chunk or label.
    pub max_model_work: Int8Work,
    pub max_retained_result_bytes: u64,
    pub max_result_bytes: u64,
    pub aggregation: ClassificationAggregationPolicy,
}
impl Int8ClassificationMapLimits {
    pub(crate) fn validate(self, task: TaskBudget) -> Result<(), Int8ClassificationMapError> {
        self.chunks.effective_token_limit().map_err(Int8ClassificationMapError::Chunk)?;
        task.validate().map_err(|_| Int8ClassificationMapError::Limits)?;
        self.aggregation.validate()?;
        if !(1..=MAX_CLASSIFICATION_MAP_CHUNKS).contains(&self.chunks.max_chunks)
            || !(1..=MAX_BYTES).contains(&self.max_retained_result_bytes)
            || !(1..=MAX_BYTES).contains(&self.max_result_bytes)
            || self.max_model_work.forward_positions == 0 || self.max_model_work.projected_logits == 0
            || self.max_model_work.attention_pairs == 0 || self.max_model_work.projections.dot_products == 0
            || self.max_model_work.projections.multiply_accumulates == 0 {
            return Err(Int8ClassificationMapError::Limits);
        }
        Ok(())
    }
}

pub enum Int8ClassificationMapError {
    Limits, EmptySource, Work, Admission, Allocation, Output, Serialization,
    Chunk(MapReduceError), Classification(Int8ClassificationError), Aggregation(ClassificationAggregationError),
}
impl fmt::Display for Int8ClassificationMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Limits => "invalid INT8 classification document limits",
            Self::EmptySource => "INT8 classification document requires nonempty source",
            Self::Work => "INT8 classification whole-document work allowance exceeded",
            Self::Admission => "INT8 classification document identity or capacity differs",
            Self::Allocation => "INT8 classification document allocation refused",
            Self::Output => "complete INT8 classification document exceeds its output allowance",
            Self::Serialization => "INT8 classification document serialization refused",
            Self::Chunk(_) => "INT8 classification document partition refused",
            Self::Classification(_) => "INT8 classification document task refused",
            Self::Aggregation(_) => "INT8 classification complete-document aggregation refused",
        })
    }
}
impl fmt::Debug for Int8ClassificationMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Display::fmt(self, f) }
}
impl Error for Int8ClassificationMapError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Chunk(error) => Some(error), Self::Classification(error) => Some(error),
            Self::Aggregation(error) => Some(error), _ => None }
    }
}
impl From<Int8ClassificationError> for Int8ClassificationMapError {
    fn from(error: Int8ClassificationError) -> Self { Self::Classification(error) }
}
impl From<ClassificationPlanningError> for Int8ClassificationMapError {
    fn from(error: ClassificationPlanningError) -> Self { Self::Classification(error.into()) }
}
impl From<StrictInt8Error> for Int8ClassificationMapError {
    fn from(error: StrictInt8Error) -> Self { Self::Classification(error.into()) }
}
impl From<ClassificationAggregationError> for Int8ClassificationMapError {
    fn from(error: ClassificationAggregationError) -> Self { Self::Aggregation(error) }
}
impl Int8ClassificationMapError {
    pub fn cancellation(&self) -> Option<DecodeCancellationKind> {
        match self { Self::Classification(error) => error.cancellation(),
            Self::Aggregation(ClassificationAggregationError::Cancelled(cause)) => Some(*cause), _ => None }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Int8ClassificationMapChunk {
    pub chunk_id: usize,
    pub source_span: VerifiedSourceSpan,
    /// Source segment only, excluding trusted scaffolds and label metadata.
    pub source_tokens: usize,
    pub run: Int8ClassificationRun,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Int8ClassificationMapEncoding {
    pub policy: UntrustedEncoding,
    /// Includes capacity metadata, every source probe and every final plan.
    pub work: SafeBpeWork,
}

/// The source and payload seals are process-private, skipped by serialization
/// and Debug, and cannot be reconstructed from external task-result JSON.
#[derive(Clone, PartialEq, Serialize)]
pub struct Int8ClassificationMapRun {
    pub schema_version: u32,
    pub execution: String,
    pub numerics_profile: String,
    pub chunk_profile: String,
    pub semantics: String,
    pub single_context_equivalence: bool,
    pub source_span: VerifiedSourceSpan,
    pub chunk_limits: ChunkLimits,
    pub tokenizer_checks: usize,
    pub planned_model_work: Int8Work,
    pub model_work: Int8Work,
    pub retained_result_bytes: u64,
    pub chunks: Vec<Int8ClassificationMapChunk>,
    pub aggregate: ClassificationAggregate,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_encoding: Option<Int8ClassificationMapEncoding>,
    #[serde(skip)]
    request_binding: [u8; 32],
    #[serde(skip)]
    result_binding: [u8; 32],
}
impl fmt::Debug for Int8ClassificationMapRun {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Int8ClassificationMapRun")
            .field("schema_version", &self.schema_version).field("execution", &self.execution)
            .field("numerics_profile", &self.numerics_profile).field("source_span", &self.source_span)
            .field("chunks", &self.chunks.len()).field("model_work", &self.model_work)
            .field("retained_result_bytes", &self.retained_result_bytes).finish_non_exhaustive()
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub struct Int8ClassificationMapPreflight {
    source: VerifiedSourceSpan,
    chunks: usize,
    heads: usize,
    mode: ClassificationMode,
    policy: ClassificationPolicy,
    work: Int8Work,
    retained_bytes: u64,
    result_bytes: u64,
    aggregation: ClassificationAggregationPolicy,
    chunk_limits: ChunkLimits,
    tokenizer_checks: usize,
    source_geometry: [u8; 32],
    encoding: Option<SafeBpeLimits>,
    encoding_work: SafeBpeWork,
    request_binding: [u8; 32],
}
impl fmt::Debug for Int8ClassificationMapPreflight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Int8ClassificationMapPreflight")
            .field("source", &self.source).field("chunks", &self.chunks).field("heads", &self.heads)
            .field("mode", &self.mode).field("work", &self.work)
            .field("retained_bytes", &self.retained_bytes).field("result_bytes", &self.result_bytes)
            .field("chunk_limits", &self.chunk_limits).field("tokenizer_checks", &self.tokenizer_checks)
            .finish_non_exhaustive()
    }
}
impl Int8ClassificationMapPreflight {
    pub fn source_span(&self) -> VerifiedSourceSpan { self.source }
    pub fn chunk_count(&self) -> usize { self.chunks }
    pub fn planned_work(&self) -> Int8Work { self.work }
    pub fn max_result_bytes(&self) -> u64 { self.result_bytes }

    fn encoding_record(&self) -> Option<Int8ClassificationMapEncoding> {
        self.encoding.map(|limits| Int8ClassificationMapEncoding {
            policy: UntrustedEncoding::SafeBpeV1(limits), work: self.encoding_work })
    }
    pub fn verify_completed(&self, run: &Int8ClassificationMapRun) -> Result<(), Int8ClassificationMapError> {
        self.verify_completed_with_control(run, &mut Continue)
    }
    /// Verify the exact admitted request and every complete public payload
    /// field under the caller's original cancellation/deadline allowance.
    pub fn verify_completed_with_control<C: DecodeStepControl>(&self, run: &Int8ClassificationMapRun,
        control: &mut C) -> Result<(), Int8ClassificationMapError> {
        planning::checkpoint(control)?;
        if run.request_binding != self.request_binding
            || run.result_binding != result_binding(run, self.result_bytes, control)?.0 {
            return Err(Int8ClassificationMapError::Admission);
        }
        if run.schema_version != 1 || run.execution != INT8_CLASSIFICATION_MAP_EXECUTION
            || run.numerics_profile != STRICT_INT8_PROFILE || run.chunk_profile != CHUNK_PROFILE
            || run.semantics != SEMANTICS || run.single_context_equivalence || run.source_span != self.source
            || run.chunks.len() != self.chunks || run.chunk_limits != self.chunk_limits
            || run.tokenizer_checks != self.tokenizer_checks
            || run.planned_model_work != self.work || run.model_work != self.work
            || run.retained_result_bytes > self.retained_bytes
            || run.input_encoding != self.encoding_record()
            || geometry(run.chunks.iter().map(|chunk| (chunk.chunk_id, chunk.source_span, chunk.source_tokens))) != self.source_geometry {
            return Err(Int8ClassificationMapError::Admission);
        }
        let mut byte = 0; let mut scalar = 0;
        let mut work = Int8Work::default(); let mut retained = 0_u64;
        for (index, chunk) in run.chunks.iter().enumerate() {
            planning::checkpoint(control)?;
            let span = chunk.source_span;
            if chunk.chunk_id != index || span.byte_start != byte || span.scalar_start != scalar
                || span.byte_end <= byte || span.scalar_end <= scalar || chunk.source_tokens == 0
                || (self.encoding.is_none() && chunk.source_tokens != span.byte_end - byte)
                || chunk.run.schema_version != 1 || chunk.run.execution != INT8_CLASSIFICATION_EXECUTION
                || chunk.run.numerics_profile != STRICT_INT8_PROFILE || chunk.run.head_count != self.heads {
                return Err(Int8ClassificationMapError::Admission);
            }
            byte = span.byte_end; scalar = span.scalar_end;
            work = work.checked_add(chunk.run.model_work)?;
            let room = self.retained_bytes.checked_sub(retained).ok_or(Int8ClassificationMapError::Output)?;
            retained = retained.checked_add(value_bytes(chunk, room, control)?)
                .ok_or(Int8ClassificationMapError::Output)?;
        }
        if byte != self.source.byte_end || scalar != self.source.scalar_end || work != self.work
            || retained != run.retained_result_bytes { return Err(Int8ClassificationMapError::Admission); }
        // Recheck finite scores and exact aggregate semantics as typed data.
        // serde's null spelling for nonfinite floats never supplies authority.
        let aggregate = aggregate_with_control(run.chunks.iter().map(|chunk| ClassificationChunkView {
            chunk_id: chunk.chunk_id, span: chunk.source_span, result: &chunk.run.result }), self.chunks,
            self.source.byte_end, self.source.scalar_end, self.mode, self.policy, self.aggregation, control)?;
        if aggregate != run.aggregate { return Err(Int8ClassificationMapError::Admission); }
        planning::checkpoint(control)?; Ok(())
    }
}

/// Borrows the exact original source and retains all compiled chunk plans.
/// There is no public result-injection driver or Deserialize constructor.
pub struct PreparedInt8ClassificationMap<'a> {
    chunks: ChunkPlan<'a>,
    plans: Vec<PreparedInt8Classification>,
    limits: Int8ClassificationMapLimits,
    expected: Int8ClassificationMapPreflight,
    task: TaskBudget,
}
impl ClassificationPlanner {
    pub fn plan_int8_document_with_control<'a, C: DecodeStepControl>(&self,
        request: &'a ClassificationRequest, context: &PlanContext<'_>, planning: ClassificationLimits,
        limits: Int8ClassificationMapLimits, control: &mut C)
        -> Result<PreparedInt8ClassificationMap<'a>, Int8ClassificationMapError> {
        planning::checkpoint(control)?;
        limits.validate(request.budget)?; planning.validate()?;
        self.document_encoding().validate().map_err(|_| Int8ClassificationMapError::Limits)?;
        if request.document.is_empty() { return Err(Int8ClassificationMapError::EmptySource); }
        if request.document.len() > limits.chunks.max_input_bytes {
            return Err(Int8ClassificationMapError::Chunk(MapReduceError::InputBudget));
        }
        let mut work = SafeBpeWork::default();
        let capacity = self.source_token_capacity_with_work(request, *context.budget_ceiling(), planning,
            limits.chunks.context_tokens, &mut work, control)?;
        let reserved = limits.chunks.context_tokens.checked_sub(capacity)
            .ok_or(Int8ClassificationMapError::Limits)?;
        let chunk_limits = ChunkLimits {
            max_chunk_tokens: limits.chunks.max_chunk_tokens.min(capacity),
            reserved_tokens: limits.chunks.reserved_tokens.max(reserved), ..limits.chunks };
        let error = RefCell::new(None);
        let chunks = {
            let control = RefCell::new(&mut *control);
            ChunkPlan::build_with_checkpoints(&request.document, chunk_limits,
                |text| self.source_tokens_with_work(text, &mut work, &mut **control.borrow_mut())
                    .map_err(|cause| { *error.borrow_mut() = Some(cause); MapReduceError::Tokenizer }),
                || match control.borrow_mut().prefill_checkpoint(0) {
                    None => Ok(()),
                    Some(cause) => {
                        *error.borrow_mut() = Some(ClassificationPlanningError::Cancelled(cause));
                        Err(MapReduceError::Cancelled)
                    }
                })
        };
        if let Some(error) = error.into_inner() { return Err(error.into()); }
        let chunks = chunks.map_err(Int8ClassificationMapError::Chunk)?;
        let last = chunks.chunks().last().ok_or(Int8ClassificationMapError::EmptySource)?.span();
        let heads = if request.mode == ClassificationMode::Exclusive { 1 } else { request.labels.len() };
        let mut plans = planning::reserved(chunks.chunks().len())?;
        let mut total = Int8Work::default();
        for chunk in chunks.chunks() {
            planning::checkpoint(control)?;
            let mut document = String::new();
            document.try_reserve_exact(chunk.text().len()).map_err(|_| Int8ClassificationMapError::Allocation)?;
            document.push_str(chunk.text());
            let row = ClassificationRequest { document, labels: request.labels.clone(), mode: request.mode,
                policy: request.policy, budget: request.budget };
            let plan = self.plan_int8_with_encoding_work(&row, context, planning, control, &mut work)?;
            if plan.head_count() != heads || plan.task_budget() != request.budget
                || source_token_count(&plan)? != chunk.tokens()
                || plan.required_context() > chunk_limits.context_tokens {
                return Err(Int8ClassificationMapError::Admission);
            }
            total = total.checked_add(plan.planned_work())?;
            if !fits(total, limits.max_model_work) { return Err(Int8ClassificationMapError::Work); }
            plans.push(plan);
        }
        let request_binding = request_binding(request, context, planning, limits,
            self.document_encoding(), work, &chunks, &plans, control)?;
        let expected = Int8ClassificationMapPreflight {
            source: VerifiedSourceSpan { byte_start: 0, byte_end: last.byte_end,
                scalar_start: 0, scalar_end: last.scalar_end },
            chunks: chunks.chunks().len(), heads, mode: request.mode, policy: request.policy, work: total,
            retained_bytes: limits.max_retained_result_bytes, result_bytes: limits.max_result_bytes,
            aggregation: limits.aggregation, chunk_limits, tokenizer_checks: chunks.tokenizer_calls(),
            source_geometry: geometry(chunks.chunks().iter().map(|chunk| (chunk.id(), chunk.span(), chunk.tokens()))),
            encoding: match self.document_encoding() { UntrustedEncoding::ByteFallback => None,
                UntrustedEncoding::SafeBpeV1(limits) => Some(*limits) },
            encoding_work: work, request_binding,
        };
        planning::checkpoint(control)?;
        Ok(PreparedInt8ClassificationMap { chunks, plans, limits, expected, task: request.budget })
    }
}
impl PreparedInt8ClassificationMap<'_> {
    pub fn preflight_metadata(&self) -> Int8ClassificationMapPreflight { self.expected }
    pub fn chunk_count(&self) -> usize { self.plans.len() }
    pub fn head_count(&self) -> usize { self.expected.heads }
    pub fn planned_work(&self) -> Int8Work { self.expected.work }
    pub fn task_budget(&self) -> TaskBudget { self.task }
    pub fn required_context(&self) -> usize {
        self.plans.iter().map(PreparedInt8Classification::required_context).max().unwrap_or(0)
    }
    pub fn max_result_bytes(&self) -> u64 { self.limits.max_result_bytes }
    pub fn identities(&self) -> Result<Vec<ExecutionIdentity>, Int8ClassificationMapError> {
        let mut identities = planning::reserved(self.plans.len())?;
        identities.extend(self.plans.iter().map(|plan| plan.execution_identity().clone()));
        Ok(identities)
    }
    pub fn preflight_with_control<C: DecodeStepControl>(&self, admitted: &[ExecutionIdentity],
        engine: &StrictInt8Engine<'_>, budget: Int8ScoringBudget, control: &mut C)
        -> Result<(), Int8ClassificationMapError> {
        planning::checkpoint(control)?;
        self.check_identities(admitted, budget.native, control)?;
        // Preflight the last chunk as well as the first before any forward.
        for (plan, identity) in self.plans.iter().zip(admitted) {
            planning::checkpoint(control)?;
            plan.preflight(identity, engine, Int8ScoringBudget {
                native: Int8RunBudget::exact(plan.planned_work()), max_kv_bytes: budget.max_kv_bytes })?;
        }
        planning::checkpoint(control)?; Ok(())
    }
    pub fn execute_with_control<C: DecodeStepControl>(&self, admitted: &[ExecutionIdentity],
        engine: &mut StrictInt8Engine<'_>, budget: Int8ScoringBudget, control: &mut C)
        -> Result<Int8ClassificationMapRun, Int8ClassificationMapError> {
        self.preflight_with_control(admitted, engine, budget, control)?;
        self.collect_with_control(control, |index, plan, control| {
            let run = plan.execute_with_control(&admitted[index], engine, Int8ScoringBudget {
                native: Int8RunBudget::exact(plan.planned_work()), max_kv_bytes: budget.max_kv_bytes }, control)?;
            if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) {
                return Err(Int8ClassificationMapError::Admission);
            }
            Ok(run)
        })
    }
    fn check_identities<C: DecodeStepControl>(&self, admitted: &[ExecutionIdentity],
        budget: Int8RunBudget, control: &mut C) -> Result<(), Int8ClassificationMapError> {
        if self.plans.is_empty() || self.plans.len() != self.chunks.chunks().len()
            || admitted.len() != self.plans.len() || self.plans.len() != self.expected.chunks {
            return Err(Int8ClassificationMapError::Admission);
        }
        if !fits(self.planned_work(), self.limits.max_model_work) { return Err(Int8ClassificationMapError::Work); }
        check_work_ceiling(self.planned_work(), budget)?;
        let mut total = Int8Work::default();
        for (plan, identity) in self.plans.iter().zip(admitted) {
            planning::checkpoint(control)?;
            plan.verify_identity(identity)?;
            if plan.task_budget() != self.task || plan.head_count() != self.head_count() {
                return Err(Int8ClassificationMapError::Admission);
            }
            total = total.checked_add(plan.planned_work())?;
        }
        if total != self.planned_work() { return Err(Int8ClassificationMapError::Admission); }
        planning::checkpoint(control)?; Ok(())
    }
    /// Private seam for synthetic semantic tests; production always uses the
    /// checked native engine above and cannot accept caller-provided scores.
    fn collect_with_control<C: DecodeStepControl, F>(&self, control: &mut C, mut execute: F)
        -> Result<Int8ClassificationMapRun, Int8ClassificationMapError>
    where F: FnMut(usize, &PreparedInt8Classification, &mut C) -> Result<Int8ClassificationRun, Int8ClassificationMapError> {
        let mut chunks = planning::reserved(self.plans.len())?;
        let mut work = Int8Work::default(); let mut bytes = 0_u64;
        for (index, (plan, source)) in self.plans.iter().zip(self.chunks.chunks()).enumerate() {
            planning::checkpoint(control)?;
            let run = execute(index, plan, control)?;
            planning::checkpoint(control)?;
            if run.schema_version != 1 || run.execution != INT8_CLASSIFICATION_EXECUTION
                || run.numerics_profile != STRICT_INT8_PROFILE || run.head_count != self.head_count()
                || run.model_work != plan.planned_work() { return Err(Int8ClassificationMapError::Admission); }
            let chunk = Int8ClassificationMapChunk {
                chunk_id: index, source_span: source.span(), source_tokens: source.tokens(), run };
            let room = self.limits.max_retained_result_bytes.checked_sub(bytes)
                .ok_or(Int8ClassificationMapError::Output)?;
            bytes = bytes.checked_add(value_bytes(&chunk, room, control)?)
                .ok_or(Int8ClassificationMapError::Output)?;
            work = work.checked_add(chunk.run.model_work)?;
            chunks.push(chunk);
        }
        if chunks.len() != self.expected.chunks || work != self.planned_work() {
            return Err(Int8ClassificationMapError::Admission);
        }
        let span = self.expected.source;
        let aggregate = aggregate_with_control(chunks.iter().map(|chunk| ClassificationChunkView {
            chunk_id: chunk.chunk_id, span: chunk.source_span, result: &chunk.run.result }), chunks.len(),
            span.byte_end, span.scalar_end, self.expected.mode, self.expected.policy, self.limits.aggregation, control)?;
        let mut run = Int8ClassificationMapRun {
            schema_version: 1, execution: INT8_CLASSIFICATION_MAP_EXECUTION.to_owned(),
            numerics_profile: STRICT_INT8_PROFILE.to_owned(), chunk_profile: CHUNK_PROFILE.to_owned(),
            semantics: SEMANTICS.to_owned(), single_context_equivalence: false, source_span: span,
            chunk_limits: self.chunks.limits(), tokenizer_checks: self.chunks.tokenizer_calls(),
            planned_model_work: self.planned_work(), model_work: work, retained_result_bytes: bytes,
            chunks, aggregate, input_encoding: self.expected.encoding_record(),
            request_binding: self.expected.request_binding, result_binding: [0; 32],
        };
        run.result_binding = result_binding(&run, self.limits.max_result_bytes, control)?.0;
        planning::checkpoint(control)?; Ok(run)
    }
}

fn source_token_count(plan: &PreparedInt8Classification) -> Result<usize, Int8ClassificationMapError> {
    let source = |head: &ClassificationHead| head.task.ir().prompt_segments().get(4)
        .filter(|segment| segment.kind() == PromptSegmentKind::Document)
        .map(|segment| segment.token_ids().len());
    let count = plan.inner.heads.first().and_then(source).filter(|&count| count != 0)
        .ok_or(Int8ClassificationMapError::Admission)?;
    if plan.inner.heads.iter().any(|head| source(head) != Some(count)) {
        return Err(Int8ClassificationMapError::Admission);
    }
    Ok(count)
}
fn fits(work: Int8Work, cap: Int8Work) -> bool {
    work.forward_positions <= cap.forward_positions && work.projected_logits <= cap.projected_logits
        && work.attention_pairs <= cap.attention_pairs && work.projections.fits(cap.projections)
}
fn geometry(rows: impl Iterator<Item = (usize, VerifiedSourceSpan, usize)>) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"fnlp-int8-classification-map-source-geometry-v1");
    for (id, span, tokens) in rows {
        for value in [id, span.byte_start, span.byte_end, span.scalar_start, span.scalar_end, tokens] {
            hash.update((value as u64).to_le_bytes());
        }
    }
    hash.finalize().into()
}

#[allow(clippy::too_many_arguments)]
fn request_binding<C: DecodeStepControl>(request: &ClassificationRequest, context: &PlanContext<'_>,
    planning: ClassificationLimits, limits: Int8ClassificationMapLimits, encoding: &UntrustedEncoding,
    work: SafeBpeWork, chunks: &ChunkPlan<'_>, plans: &[PreparedInt8Classification], control: &mut C)
    -> Result<[u8; 32], Int8ClassificationMapError> {
    let mut writer = BoundedDigest::new(b"fnlp-int8-classification-map-request-v1", u64::MAX, control);
    writer.value(request)?;
    writer.value(&(context.execution_identity(), context.budget_ceiling()))?;
    writer.value(&(limits.chunks, limits.max_model_work, limits.max_retained_result_bytes,
        limits.max_result_bytes, limits.aggregation, encoding, work))?;
    let scoring = planning.scoring;
    writer.value(&(planning.max_labels, planning.max_input_bytes, planning.max_label_id_bytes,
        planning.max_label_description_bytes, planning.max_total_label_bytes, planning.max_context_tokens,
        planning.max_total_prompt_tokens, planning.max_work))?;
    writer.value(&(scoring.max_candidates, scoring.max_total_tokens, scoring.max_nodes,
        scoring.max_depth, scoring.max_candidate_id_bytes, scoring.max_projected_logits))?;
    writer.value(&plans.len())?;
    for (chunk, plan) in chunks.chunks().iter().zip(plans) {
        writer.checkpoint()?;
        writer.value(&(chunk.id(), chunk.span(), chunk.tokens(), plan.execution_identity(), plan.planned_work()))?;
    }
    writer.checkpoint()?; Ok(writer.finish().0)
}
fn result_binding<C: DecodeStepControl>(run: &Int8ClassificationMapRun, cap: u64, control: &mut C)
    -> Result<([u8; 32], u64), Int8ClassificationMapError> {
    let mut writer = BoundedDigest::new(b"fnlp-int8-classification-map-result-v1", cap, control);
    writer.hash.update(run.request_binding); writer.hash.update([0]);
    writer.value(run)?;
    writer.checkpoint()?; Ok(writer.finish())
}
fn value_bytes<C: DecodeStepControl>(value: &impl Serialize, cap: u64, control: &mut C)
    -> Result<u64, Int8ClassificationMapError> {
    let mut writer = BoundedDigest::new(b"fnlp-int8-classification-map-byte-count-v1", cap, control);
    writer.value(value)?;
    writer.checkpoint()?; Ok(writer.finish().1)
}

/// Stream finite repository-owned values without allocating a JSON tree or
/// second serialized payload. Field ordering changes only the private hash;
/// JSON punctuation/escaping/numbers have the canonical writer's byte length.
struct BoundedDigest<'a, C> {
    hash: Sha256,
    bytes: u64,
    cap: u64,
    exceeded: bool,
    control: &'a mut C,
    until_checkpoint: usize,
    cancelled: Option<DecodeCancellationKind>,
}
impl<'a, C: DecodeStepControl> BoundedDigest<'a, C> {
    fn new(domain: &[u8], cap: u64, control: &'a mut C) -> Self {
        let mut hash = Sha256::new(); hash.update(domain); hash.update([0]);
        Self { hash, bytes: 0, cap, exceeded: false, control, until_checkpoint: 0, cancelled: None }
    }
    fn checkpoint(&mut self) -> Result<(), Int8ClassificationMapError> {
        if self.cancelled.is_none() { self.cancelled = self.control.prefill_checkpoint(0); }
        if let Some(cause) = self.cancelled { return Err(ClassificationPlanningError::Cancelled(cause).into()); }
        self.until_checkpoint = CHECKPOINT_BYTES;
        Ok(())
    }
    fn value(&mut self, value: &(impl Serialize + ?Sized)) -> Result<(), Int8ClassificationMapError> {
        if serde_json::to_writer(&mut *self, value).is_err() {
            if let Some(cause) = self.cancelled { return Err(ClassificationPlanningError::Cancelled(cause).into()); }
            return Err(if self.exceeded { Int8ClassificationMapError::Output }
                else { Int8ClassificationMapError::Serialization });
        }
        // Zero cannot occur unescaped in JSON, so adjacent values are framed.
        self.hash.update([0]); Ok(())
    }
    fn finish(self) -> ([u8; 32], u64) { (self.hash.finalize().into(), self.bytes) }
}
impl<C: DecodeStepControl> io::Write for BoundedDigest<'_, C> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let mut written = 0;
        while written < bytes.len() {
            if self.until_checkpoint == 0 {
                self.checkpoint().map_err(|_| io::Error::other("classification map serialization cancelled"))?;
            }
            let count = self.until_checkpoint.min(bytes.len() - written);
            let next = self.bytes.checked_add(count as u64).filter(|&next| next <= self.cap);
            let Some(next) = next else {
                self.exceeded = true; return Err(io::Error::other("classification map serialization limit"));
            };
            self.hash.update(&bytes[written..written + count]);
            self.bytes = next; self.until_checkpoint -= count; written += count;
        }
        Ok(written)
    }
    fn flush(&mut self) -> io::Result<()> { Ok(()) }
}
struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}

#[cfg(test)]
mod tests;
