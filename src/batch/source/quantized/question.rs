//! Complete-document INT8 QA with one exact private question per NDJSON item.
//! Every nonblank passage retains the original question; the sealed task owns
//! complete source coverage, independent answers and original-coordinate cites.
//! Attempts and flush epochs never renew native, mask or verification authority.
use std::{fmt, sync::Arc};
use serde::Deserialize;
use crate::{
    batch::{BatchCode, BatchDocument, BatchFault, BatchItemFailure, BatchProcessor,
        BatchRequestContext, BatchWork, generation::GuardedOutput},
    execution_identity::{ExecutionIdentity, Sha256Digest},
    grammar::mask::MaskWorkLimits,
    native_engine::{artifact_bridge::ArtifactIdentity, constrained_int8, decode::DecodeStepControl,
        strict_int8::{Int8MemoryRequirement, Int8Work, StrictInt8Engine, STRICT_INT8_PROFILE,
            prefill::Int8PrefillLimits}},
    tasks::{answer::{AnswerOptions, ANSWER_TASK_VERSION}, extract::ExtractionVocabulary,
        ir::{PlanContext, TaskBudget}, mapreduce::MapReduceError,
        source_planning::{SourcePlanningError, SourcePlanningLimits, SourceTaskPlanner,
            quantized::{Int8SourceError, long::{Int8SourceMapError, Int8SourceMapLimits,
                question::{Int8QuestionPreflight, Int8QuestionRun, PreparedInt8Question, SourceQuestion}}}}},
    tokenizer::untrusted::{SafeBpeWork, SafeBpeWorkLimits, UntrustedEncoding},
    validation::grounded_fields::GroundingBudget,
};
pub(crate) mod encoding;
pub mod memory;

/// Frozen, reconstructed and admitted identities coexist before native work.
pub const QUESTION_DOCUMENT_CHUNK_METADATA_BYTES: u64 = 32_768;
const MAX_QUESTION_IDENTITY_BYTES: usize = 8192;

/// A strict object supplies only the original question. Fixed options, source,
/// verification policy, budget and scheduling cannot be replaced by a record.
pub struct QuestionDocumentArgs { pub question: String }
impl<'de> Deserialize<'de> for QuestionDocumentArgs {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Arguments;
        impl<'de> serde::de::Visitor<'de> for Arguments {
            type Value = QuestionDocumentArgs;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str("an object containing one question") }
            fn visit_map<M: serde::de::MapAccess<'de>>(self, mut map: M) -> Result<Self::Value, M::Error> {
                let mut question = None;
                while let Some(key) = map.next_key::<String>()? {
                    if key != "question" { return Err(serde::de::Error::unknown_field(&key, &["question"])); }
                    if question.is_some() { return Err(serde::de::Error::duplicate_field("question")); }
                    question = Some(map.next_value::<String>()?);
                }
                Ok(QuestionDocumentArgs { question: question.ok_or_else(|| serde::de::Error::missing_field("question"))? })
            }
        }
        deserializer.deserialize_map(Arguments)
    }
}

/// One immutable answer policy and finite per-document and complete-run work.
/// Independent passage answers retain no generated-token arrays in the result.
#[derive(Clone, Copy)]
pub struct Int8QuestionDocumentBatchConfig {
    pub options: AnswerOptions,
    pub budget: TaskBudget,
    pub planning: SourcePlanningLimits,
    pub mapping: Int8SourceMapLimits,
    pub verification: GroundingBudget,
    pub max_question_bytes: usize,
    pub max_model_work: Int8Work,
    pub max_mask_visits: u64,
    pub max_verification: GroundingBudget,
}
impl Int8QuestionDocumentBatchConfig {
    pub(crate) fn validate(&self) -> Result<(), BatchFault> {
        self.budget.validate().map_err(|_| BatchCode::InvalidLimits)?;
        self.planning.validate().map_err(|_| BatchCode::InvalidLimits)?;
        self.options.validate().map_err(|_| BatchCode::InvalidLimits)?;
        let schema = self.options.schema_source().map_err(|_| BatchCode::InvalidLimits)?;
        self.mapping.validate(self.budget, self.planning).map_err(|_| BatchCode::InvalidLimits)?;
        if schema.len() > self.planning.compiler.max_schema_bytes || !self.mapping.reduction.validate()
            || self.mapping.chunks.context_tokens != self.planning.max_context_tokens
            || !(1..=64 * 1024 * 1024).contains(&self.mapping.chunks.max_input_bytes)
            || !(1..=64 * 1024 * 1024).contains(&self.max_question_bytes)
            || self.max_question_bytes > self.planning.max_input_bytes
            || !positive(self.mapping.max_model_work) || !fits(self.mapping.max_model_work, self.max_model_work)
            || self.mapping.max_mask_visits > self.max_mask_visits
            || !valid_verification(self.verification)
            || !verification_fits(self.verification, self.max_verification) {
            return Err(BatchCode::InvalidLimits.into());
        }
        let native = Int8MemoryRequirement::for_context(self.planning.max_context_tokens)
            .map_err(|_| BatchCode::InvalidLimits)?;
        if native.kv_bytes != self.budget.max_kv_bytes { return Err(BatchCode::Admission.into()); }
        Ok(())
    }
}

/// Actual-model-bound compiler. No default source/question is retained, and a
/// private owner seals every prepared item's exact original input and policy.
pub struct Int8QuestionDocumentBatchPlanner {
    planner: Arc<SourceTaskPlanner>, identity: ExecutionIdentity,
    config: Int8QuestionDocumentBatchConfig, owner: Arc<()>,
}
impl Int8QuestionDocumentBatchPlanner {
    pub fn for_model(model: &ArtifactIdentity, planner: Arc<SourceTaskPlanner>, identity: ExecutionIdentity,
        config: Int8QuestionDocumentBatchConfig) -> Result<Self, BatchFault> {
        check_model(&identity, model)?;
        Self::configured(planner, identity, config)
    }
    fn configured(planner: Arc<SourceTaskPlanner>, identity: ExecutionIdentity,
        config: Int8QuestionDocumentBatchConfig) -> Result<Self, BatchFault> {
        config.validate()?; check_identity_size(&identity)?;
        super::check_configuration(&planner, &identity, config.budget, config.planning, None)?;
        planner.map_document_encoding_workspace_bytes().map_err(|_| BatchCode::InvalidLimits)?;
        if identity.task_spec != ANSWER_TASK_VERSION { return Err(BatchCode::Admission.into()); }
        Ok(Self { planner, identity, config, owner: Arc::new(()) })
    }
    /// Preparation and exact reconstruction share one operation allowance for
    /// the original source/question pair, including rejected tokenizer probes.
    pub fn with_map_document_encoding(mut self, policy: UntrustedEncoding, complete: SafeBpeWorkLimits)
        -> Result<Self, BatchFault> {
        let planner = encoding::configure(&self.planner, policy, complete)?;
        let mut identity = self.identity.clone(); identity.template_digest = *planner.template_digest();
        super::check_configuration(&planner, &identity, self.config.budget, self.config.planning, None)?;
        check_identity_size(&identity)?;
        self.planner = planner; self.identity = identity; self.reset_preparation_owner();
        Ok(self)
    }
    pub fn config(&self) -> &Int8QuestionDocumentBatchConfig { &self.config }
    pub fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    pub fn base_execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    pub fn retained_owners(&self) -> Arc<SourceTaskPlanner> { Arc::clone(&self.planner) }
    pub fn document_encoding(&self) -> &UntrustedEncoding { self.planner.document_encoding() }
    pub fn map_document_encoding_limits(&self) -> Option<SafeBpeWorkLimits> { self.planner.map_document_encoding_limits() }
    pub fn encoding_workspace_bytes(&self) -> Result<u64, BatchFault> {
        self.planner.map_document_encoding_workspace_bytes().map_err(|_| BatchCode::InvalidLimits.into())
    }
    pub(crate) fn encoding_recipe(&self) -> Result<Option<encoding::EncodingRecipe>, BatchFault> {
        encoding::recipe(&self.planner)
    }
    pub(crate) fn reset_preparation_owner(&mut self) { self.owner = Arc::new(()); }
    pub fn max_result_bytes(&self) -> u64 { self.config.mapping.reduction.max_result_bytes as u64 }
    pub fn max_output_tokens(&self) -> u64 { 0 }
    pub(crate) fn owns(&self, prepared: &PreparedInt8QuestionDocument) -> Result<(), BatchItemFailure> {
        if !Arc::ptr_eq(&self.owner, &prepared.owner) { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
        encoding::verify_operation(&self.planner, &prepared.operation)
    }
    pub fn prepare_with_control<C: DecodeStepControl>(&self, source: String,
        args: QuestionDocumentArgs, control: &mut C) -> Result<PreparedInt8QuestionDocument, BatchItemFailure> {
        let operation = encoding::operation(&self.planner)?;
        let plan = self.plan(&source, &args.question, &operation, control, false)?;
        let (expected, identities) = self.snapshot(&plan, control)?;
        drop(plan); poll(control)?;
        Ok(PreparedInt8QuestionDocument { source, question: args.question, expected, identities,
            owner: Arc::clone(&self.owner), operation })
    }
    fn plan<'s, C: DecodeStepControl>(&self, source: &'s str, question: &str,
        operation: &SourceTaskPlanner, control: &mut C, fatal: bool)
        -> Result<PreparedInt8Question<'s>, BatchItemFailure> {
        encoding::verify_operation(&self.planner, operation)?; poll(control)?;
        let reject = |code| if fatal { BatchItemFailure::fatal(code) } else { BatchItemFailure::reject(code) };
        let c = &self.config;
        if source.is_empty() || source.len() > c.mapping.chunks.max_input_bytes
            || question.is_empty() || question.len() > c.max_question_bytes {
            return Err(reject(BatchCode::DocumentLimit));
        }
        if !question.chars().any(|ch| !ch.is_whitespace()) { return Err(reject(BatchCode::Planning)); }
        let context = PlanContext::new(&self.identity, c.budget)
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        let mut copied = String::new();
        copied.try_reserve_exact(question.len()).map_err(|_| BatchItemFailure::fatal(BatchCode::Allocation))?;
        copied.push_str(question);
        let question = SourceQuestion { question: copied, options: c.options, verification: c.verification };
        let plan = operation.plan_int8_question_with_control(source, &question, c.budget,
            &context, c.planning, c.mapping, control).map_err(|e| map_failure(e, fatal))?;
        let proof = plan.preflight_metadata();
        if proof.chunk_count() == 0 || proof.chunk_count() > c.mapping.chunks.max_chunks
            || proof.native_chunks() == 0 || proof.native_chunks() > proof.chunk_count()
            || proof.source_span().byte_start != 0 || proof.source_span().byte_end != source.len()
            || plan.max_result_bytes() != self.max_result_bytes()
            || !positive(proof.planned_work()) || !fits(proof.planned_work(), c.mapping.max_model_work)
            || proof.reserved_mask_visits() == 0 || proof.reserved_mask_visits() > c.mapping.max_mask_visits {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        poll(control)?; Ok(plan)
    }
    fn snapshot<C: DecodeStepControl>(&self, plan: &PreparedInt8Question<'_>, control: &mut C)
        -> Result<(Int8QuestionPreflight, Vec<ExecutionIdentity>), BatchItemFailure> {
        let proof = plan.preflight_metadata(); let rows = plan.execution_identities();
        if rows.len() != proof.native_chunks() || rows.len() == 0 {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        let mut identities = Vec::new();
        identities.try_reserve_exact(rows.len()).map_err(|_| BatchItemFailure::fatal(BatchCode::Allocation))?;
        for identity in rows {
            poll(control)?; check_identity_size(identity).map_err(BatchItemFailure::fatal)?;
            identities.push(identity.clone());
        }
        Ok((proof, identities))
    }
    fn compare<C: DecodeStepControl>(&self, plan: &PreparedInt8Question<'_>,
        prepared: &PreparedInt8QuestionDocument, control: &mut C) -> Result<(), BatchItemFailure> {
        self.owns(prepared)?;
        let (proof, identities) = self.snapshot(plan, control)?;
        if proof != prepared.expected || identities != prepared.identities {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        poll(control)
    }
}

/// Owns original bytes and opaque task commitments, never a deserializable
/// native result or caller-provided factory. Reconstruction uses the SAME
/// cumulative tokenizer operation and must match every prepared identity.
pub struct PreparedInt8QuestionDocument {
    source: String, question: String, expected: Int8QuestionPreflight,
    identities: Vec<ExecutionIdentity>, owner: Arc<()>, operation: Arc<SourceTaskPlanner>,
}
impl PreparedInt8QuestionDocument {
    pub fn chunk_count(&self) -> usize { self.expected.chunk_count() }
    pub fn native_chunks(&self) -> usize { self.expected.native_chunks() }
    pub fn planned_work(&self) -> Int8Work { self.expected.planned_work() }
    pub fn reserved_mask_visits(&self) -> u64 { self.expected.reserved_mask_visits() }
    pub fn execution_identities(&self) -> &[ExecutionIdentity] { &self.identities }
    pub fn map_document_encoding_work(&self) -> Result<Option<SafeBpeWork>, BatchFault> {
        self.operation.map_document_encoding_work().map_err(|_| BatchCode::Admission.into())
    }
}
pub type Int8QuestionDocumentRun = Int8QuestionRun;
pub struct Int8QuestionDocumentAdmission<'a> {
    pub identities: &'a [ExecutionIdentity], pub model_work: Int8Work, pub mask_node_visits: u64,
    pub mask_limits: MaskWorkLimits, pub kv_reservation_bytes: u64, pub max_result_bytes: u64,
    pub max_output_tokens: u64, pub chunk_count: usize, pub native_chunks: usize,
}
pub trait Int8QuestionDocumentBatchAdmission {
    type Guard;
    fn admit(&mut self, request: Int8QuestionDocumentAdmission<'_>)
        -> Result<(Vec<ExecutionIdentity>, Self::Guard), BatchItemFailure>;
}

pub struct NativeInt8QuestionDocumentBatch<'p, 'e, 'w, 'v, A: Int8QuestionDocumentBatchAdmission> {
    planner: &'p Int8QuestionDocumentBatchPlanner, engine: &'e mut StrictInt8Engine<'w>,
    vocabulary: &'v ExtractionVocabulary, admission: A, ledger: Ledger, prefill: Option<Int8PrefillLimits>,
}
impl<'p, 'e, 'w, 'v, A: Int8QuestionDocumentBatchAdmission> NativeInt8QuestionDocumentBatch<'p, 'e, 'w, 'v, A> {
    pub fn new(planner: &'p Int8QuestionDocumentBatchPlanner, engine: &'e mut StrictInt8Engine<'w>,
        vocabulary: &'v ExtractionVocabulary, admission: A) -> Result<Self, BatchFault> {
        capacity(planner, engine).map_err(|e| e.fault)?;
        Ok(Self { planner, engine, vocabulary, admission, ledger: Ledger::default(), prefill: None })
    }
    pub fn new_layer_major(planner: &'p Int8QuestionDocumentBatchPlanner, engine: &'e mut StrictInt8Engine<'w>,
        vocabulary: &'v ExtractionVocabulary, admission: A, prefill: Int8PrefillLimits) -> Result<Self, BatchFault> {
        prefill.validate().map_err(|_| BatchCode::InvalidLimits)?;
        let mut native = Self::new(planner, engine, vocabulary, admission)?;
        native.prefill = Some(prefill); Ok(native)
    }
    pub fn reserved_model_work(&self) -> Int8Work { self.ledger.work }
    pub fn reserved_mask_visits(&self) -> u64 { self.ledger.masks }
    pub fn reserved_verification(&self) -> GroundingBudget { self.ledger.verification }
    pub fn is_poisoned(&self) -> bool { self.ledger.failed || !clean(self.engine) }
}
impl<A: Int8QuestionDocumentBatchAdmission> BatchProcessor for NativeInt8QuestionDocumentBatch<'_, '_, '_, '_, A> {
    type Args = QuestionDocumentArgs;
    type Prepared = PreparedInt8QuestionDocument;
    type Output = GuardedOutput<Int8QuestionDocumentRun, A::Guard>;
    fn prepare(&mut self, _: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        self.ledger.failed = true; Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>, control: &mut C)
        -> Result<Self::Prepared, BatchItemFailure> {
        self.ledger.ready()?; self.ledger.failed = true;
        let result = (|| {
            poll(control)?; capacity(self.planner, self.engine)?;
            let args = document.task_args.ok_or_else(|| BatchItemFailure::reject(BatchCode::Planning))?;
            let operation = encoding::operation(&self.planner.planner)?;
            let plan = self.planner.plan(&document.text, &args.question, &operation, control, false)?;
            let (expected, identities) = self.planner.snapshot(&plan, control)?;
            plan.preflight(&identities, self.engine).map_err(|e| map_failure(e, true))?;
            drop(plan); poll(control)?;
            Ok(PreparedInt8QuestionDocument { source: document.text, question: args.question,
                expected, identities, owner: Arc::clone(&self.planner.owner), operation })
        })();
        if !result.as_ref().is_err_and(|e: &BatchItemFailure| e.stop) && clean(self.engine) { self.ledger.failed = false; }
        result
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork {
        let work = prepared.planned_work(); BatchWork { forward_positions: work.forward_positions, projected_logits: work.projected_logits }
    }
    fn execute<C: DecodeStepControl>(&mut self, _: Self::Prepared, _: &mut C) -> Result<Self::Output, BatchItemFailure> {
        self.ledger.failed = true; Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        if let Err(error) = self.planner.owns(&prepared) { self.ledger.failed = true; return Err(error); }
        self.ledger.begin(&self.planner.config, prepared.expected, context)?;
        let result = (|| {
            poll(control)?;
            let kv = capacity(self.planner, self.engine)?;
            let plan = self.planner.plan(&prepared.source, &prepared.question, &prepared.operation, control, true)?;
            self.planner.compare(&plan, &prepared, control)?;
            plan.preflight(&prepared.identities, self.engine).map_err(|e| map_failure(e, true))?;
            let (admitted, guard) = self.admission.admit(Int8QuestionDocumentAdmission {
                identities: &prepared.identities, model_work: prepared.planned_work(),
                mask_node_visits: prepared.reserved_mask_visits(), mask_limits: self.planner.config.mapping.mask_limits,
                kv_reservation_bytes: kv, max_result_bytes: self.planner.max_result_bytes(), max_output_tokens: 0,
                chunk_count: prepared.chunk_count(), native_chunks: prepared.native_chunks(),
            })?;
            poll(control)?;
            if admitted != prepared.identities { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
            let output = match self.prefill {
                Some(prefill) => plan.execute_layer_major_with_control(&admitted, self.engine, self.vocabulary, prefill, control),
                None => plan.execute_with_control(&admitted, self.engine, self.vocabulary, control),
            }.map_err(|e| map_failure(e, true))?;
            prepared.expected.verify_completed_with_control(&output, control).map_err(|e| map_failure(e, true))?;
            capacity(self.planner, self.engine)?; poll(control)?;
            Ok(GuardedOutput::new(output, guard))
        })();
        match result { Ok(output) => { self.ledger.failed = false; Ok(output) },
            Err(mut failure) => { failure.stop = true; Err(failure) } }
    }
}

fn clean(engine: &StrictInt8Engine<'_>) -> bool { !engine.is_poisoned() && engine.kv_cache().all_slots_have_len(0) }
fn capacity(planner: &Int8QuestionDocumentBatchPlanner, engine: &StrictInt8Engine<'_>) -> Result<u64, BatchItemFailure> {
    check_model(planner.execution_identity(), engine.artifact_identity()).map_err(BatchItemFailure::fatal)?;
    let c = planner.config;
    if !clean(engine) || engine.profile() != STRICT_INT8_PROFILE
        || engine.kv_cache().capacity_positions() != c.planning.max_context_tokens {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    Ok(c.budget.max_kv_bytes)
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
fn positive(w: Int8Work) -> bool {
    w.forward_positions > 0 && w.projected_logits > 0 && w.attention_pairs > 0
        && w.projections.dot_products > 0 && w.projections.multiply_accumulates > 0
}
fn fits(a: Int8Work, b: Int8Work) -> bool {
    a.forward_positions <= b.forward_positions && a.projected_logits <= b.projected_logits
        && a.attention_pairs <= b.attention_pairs && a.projections.fits(b.projections)
}
fn check_identity_size(identity: &ExecutionIdentity) -> Result<(), BatchFault> {
    struct Counter(usize);
    impl std::io::Write for Counter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0 = self.0.checked_sub(bytes.len()).ok_or_else(|| std::io::Error::other("question identity bound"))?;
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
    }
    serde_json::to_writer(Counter(MAX_QUESTION_IDENTITY_BYTES), identity).map_err(|_| BatchCode::InvalidLimits.into())
}
fn valid_verification(v: GroundingBudget) -> bool {
    (1..=1_000_000).contains(&v.max_fields) && (1..=1_000_000).contains(&v.max_matches) && v.max_scan_steps > 0
}
fn verification_fits(a: GroundingBudget, b: GroundingBudget) -> bool {
    a.max_fields <= b.max_fields && a.max_matches <= b.max_matches && a.max_scan_steps <= b.max_scan_steps
}
fn add_verification(a: GroundingBudget, b: GroundingBudget) -> Option<GroundingBudget> {
    Some(GroundingBudget { max_fields: a.max_fields.checked_add(b.max_fields)?,
        max_matches: a.max_matches.checked_add(b.max_matches)?, max_scan_steps: a.max_scan_steps.checked_add(b.max_scan_steps)? })
}
struct Ledger { work: Int8Work, masks: u64, verification: GroundingBudget,
    sequence: u64, epoch: u64, line: u64, failed: bool }
impl Default for Ledger {
    fn default() -> Self {
        Self { work: Int8Work::default(), masks: 0,
            verification: GroundingBudget { max_fields: 0, max_matches: 0, max_scan_steps: 0 },
            sequence: 0, epoch: 0, line: 0, failed: false }
    }
}
impl Ledger {
    fn ready(&self) -> Result<(), BatchItemFailure> {
        if self.failed { Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)) } else { Ok(()) }
    }
    fn begin(&mut self, c: &Int8QuestionDocumentBatchConfig, proof: Int8QuestionPreflight, context: BatchRequestContext)
        -> Result<(), BatchItemFailure> {
        self.ready()?; self.failed = true;
        if context.request_seq <= self.sequence || context.epoch == 0 || context.epoch < self.epoch || context.input_line <= self.line {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        let exhausted = || BatchItemFailure::fatal(BatchCode::WorkLimit);
        let work = self.work.checked_add(proof.planned_work()).ok().filter(|&w| fits(w, c.max_model_work)).ok_or_else(exhausted)?;
        let masks = self.masks.checked_add(proof.reserved_mask_visits()).filter(|&n| n <= c.max_mask_visits).ok_or_else(exhausted)?;
        let verification = add_verification(self.verification, c.verification)
            .filter(|&v| verification_fits(v, c.max_verification)).ok_or_else(exhausted)?;
        self.work = work; self.masks = masks; self.verification = verification;
        self.sequence = context.request_seq; self.epoch = context.epoch; self.line = context.input_line;
        Ok(())
    }
}
fn poll(control: &mut impl DecodeStepControl) -> Result<(), BatchItemFailure> {
    match control.prefill_checkpoint(0) { Some(cause) => Err(BatchItemFailure::fatal(BatchFault::cancelled(cause))), None => Ok(()) }
}
fn map_failure(error: Int8SourceMapError, fatal: bool) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    let (code, stop) = match error {
        Int8SourceMapError::InvalidLimits => (BatchCode::InvalidLimits, true),
        Int8SourceMapError::EmptySource => (BatchCode::DocumentLimit, fatal),
        Int8SourceMapError::WorkLimit => (BatchCode::WorkLimit, fatal),
        Int8SourceMapError::Admission => (BatchCode::Admission, true),
        Int8SourceMapError::Allocation | Int8SourceMapError::Chunk(MapReduceError::AllocationRefused) => (BatchCode::Allocation, true),
        Int8SourceMapError::Chunk(MapReduceError::InvalidLimits) => (BatchCode::InvalidLimits, true),
        Int8SourceMapError::Chunk(_) => (BatchCode::Planning, fatal),
        Int8SourceMapError::Source(Int8SourceError::Planning(SourcePlanningError::InputBudget)) => (BatchCode::DocumentLimit, fatal),
        Int8SourceMapError::Source(Int8SourceError::Planning(SourcePlanningError::ContextBudget)) => (BatchCode::WorkLimit, fatal),
        Int8SourceMapError::Source(_) | Int8SourceMapError::Reduction(_) => (BatchCode::InvalidExecution, true),
    };
    if stop { BatchItemFailure::fatal(code) } else { BatchItemFailure::reject(code) }
}

#[cfg(test)] mod tests;
