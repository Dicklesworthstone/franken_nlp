//! Reconciled root objects followed by complete-original-source field reviews.
//!
//! A successful result retains the exact native object map. Every review uses
//! lossless source chunks and the original faithfulness reducer; its aggregate
//! is explicitly different from a single-context entailment judgment. Nothing
//! here repairs a value, invents absent properties or authorizes acceptance.

use super::*;
use sha2::{Digest, Sha256};
use crate::{
    batch::extract::grouped_int4::long::object::{Int4ObjectExtractionMapError,
        Int4ObjectExtractionMapLimits, Int4ObjectExtractionMapPreflight, Int4ObjectExtractionMapRun,
        PreparedInt4ObjectExtractionMap},
    tasks::{extract::object_reconcile::RootObjectSchema, ir::TaskBudget,
        judge::grouped_int4::faithfulness_map::{Int4FaithfulnessMapError, Int4FaithfulnessMapLimits,
            Int4FaithfulnessMapPreflight, Int4FaithfulnessMapRun, PreparedInt4FaithfulnessMap,
            INT4_FAITHFULNESS_MAP_EXECUTION}},
};

pub const INT4_SEMANTIC_OBJECT_DOCUMENT_EXECUTION: &str =
    "grouped-int4-reconciled-object-complete-source-reviews-v1";
pub const SEMANTIC_OBJECT_DOCUMENT_SCOPE: &str =
    "complete-original-source-independent-chunks-not-single-context-equivalent-v1";
const CALIBRATION: &str = "experimental_uncalibrated_diagnostic_only";

pub enum Int4SemanticObjectDocumentError {
    Contract, WorkBudget, Accounting, OutputBudget, Serialization,
    Cancelled(DecodeCancellationKind),
    Extraction(Int4ObjectExtractionMapError),
    Semantic(Int4SemanticError),
    JudgeMap(Int4FaithfulnessMapError),
}
impl fmt::Display for Int4SemanticObjectDocumentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Contract => "semantic object document contract refused",
            Self::WorkBudget => "semantic object document complete allowance exceeded",
            Self::Accounting => "semantic object document completion mismatch",
            Self::OutputBudget => "semantic object document complete output exceeds its limit",
            Self::Serialization => "semantic object document binding serialization failed",
            Self::Cancelled(_) => "semantic object document cancelled",
            Self::Extraction(_) => "semantic object document extraction or reconciliation failed",
            Self::Semantic(_) => "semantic object document claims refused",
            Self::JudgeMap(_) => "semantic object document complete-source review failed",
        })
    }
}
impl fmt::Debug for Int4SemanticObjectDocumentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Display::fmt(self, f) }
}
impl Error for Int4SemanticObjectDocumentError {}
impl From<Int4ObjectExtractionMapError> for Int4SemanticObjectDocumentError {
    fn from(e: Int4ObjectExtractionMapError) -> Self { Self::Extraction(e) }
}
impl From<Int4SemanticError> for Int4SemanticObjectDocumentError {
    fn from(e: Int4SemanticError) -> Self { Self::Semantic(e) }
}
impl From<SemanticError> for Int4SemanticObjectDocumentError {
    fn from(e: SemanticError) -> Self { Self::Semantic(e.into()) }
}
impl From<Int4FaithfulnessMapError> for Int4SemanticObjectDocumentError {
    fn from(e: Int4FaithfulnessMapError) -> Self { Self::JudgeMap(e) }
}
impl Int4SemanticObjectDocumentError {
    pub fn cancellation(&self) -> Option<DecodeCancellationKind> {
        match self {
            Self::Cancelled(cause) => Some(*cause),
            Self::Extraction(error) => error.cancellation(),
            Self::Semantic(error) => error.cancellation(),
            Self::JudgeMap(error) => error.cancellation(),
            _ => None,
        }
    }
}

/// The only public constructor consumes a genuine exact-source object map.
/// No caller-provided result, replacement schema or review claim is executable.
pub struct PreparedInt4SemanticObjectDocument<'s> {
    extraction: PreparedInt4ObjectExtractionMap<'s>,
    source: &'s str,
    judge: Arc<Int4JudgePlanner>,
    options: Int4SemanticOptions,
    judge_mapping: Int4FaithfulnessMapLimits,
    extraction_budget: TaskBudget,
    extraction_limits: Int4ObjectExtractionMapLimits,
    expected: Int4SemanticObjectDocumentPreflight,
}
impl<'s> PreparedInt4SemanticObjectDocument<'s> {
    pub fn new<C: DecodeStepControl>(extraction: PreparedInt4ObjectExtractionMap<'s>,
        judge: Arc<Int4JudgePlanner>, options: Int4SemanticOptions,
        judge_mapping: Int4FaithfulnessMapLimits, control: &mut C)
        -> Result<Self, Int4SemanticObjectDocumentError> {
        poll_document(control)?;
        options.validate_schema(extraction.original_schema())?;
        judge_mapping.validate(options.specification.judge_budget)?;
        let original = extraction.preflight_metadata();
        let source = extraction.original_source();
        let extraction_budget = extraction.task_budget().ok_or(Int4SemanticObjectDocumentError::Contract)?;
        let extraction_limits = extraction.limits();
        let encoding = original.document_encoding();
        let initial_encoding_work = original.encoding_work().unwrap_or_default();
        Int4SemanticEncoding::from_work(&encoding, initial_encoding_work)?;
        if source.is_empty() || source.len() > judge_mapping.chunks.max_input_bytes
            || original.chunk_count() == 0 || original.map_batch_chunks() != 1
            || original.selected_rows().is_some()
            || original.group_width() != judge.group_width()
            || judge.document_encoding() != &encoding
            || matches!(encoding, UntrustedEncoding::SafeBpeV1(_)) != original.encoding_work().is_some()
            || judge.execution_identity().task_spec != "judge-v1"
            || extraction.max_result_bytes() > options.limits.max_output_bytes
            || judge_mapping.max_result_bytes > options.limits.max_output_bytes {
            return Err(Int4SemanticObjectDocumentError::Contract);
        }
        check_judge_configuration(&judge, options.specification.judge_budget)?;
        for identity in extraction.execution_identities() {
            poll_document(control)?;
            if identity.task_spec != "extract-v1" {
                return Err(Int4SemanticObjectDocumentError::Contract);
            }
            constrained_int4::check_profile(identity, original.group_width())
                .map_err(|_| Int4SemanticObjectDocumentError::Contract)?;
            same_model(identity, judge.execution_identity())?;
        }
        reserve_combined(original.planned_work(), Int4Work::default(), options.max_model_work)?;
        let request_binding = request_binding(source, &judge, &options, judge_mapping, control)?;
        let expected = Int4SemanticObjectDocumentPreflight {
            extraction: original, request_binding, encoding, initial_encoding_work,
            limits: options.limits, max_model_work: options.max_model_work,
        };
        poll_document(control)?;
        Ok(Self { extraction, source, judge, options, judge_mapping,
            extraction_budget, extraction_limits, expected })
    }

    pub fn execution_identities(&self) -> impl ExactSizeIterator<Item = &ExecutionIdentity> {
        self.extraction.execution_identities()
    }
    pub fn judge_identity(&self) -> &ExecutionIdentity { self.judge.execution_identity() }
    pub fn group_width(&self) -> usize { self.expected.extraction.group_width() }
    pub fn source_bytes(&self) -> usize { self.source.len() }
    pub fn extraction_budget(&self) -> TaskBudget { self.extraction_budget }
    pub fn judge_budget(&self) -> TaskBudget { self.options.specification.judge_budget }
    pub fn judge_planning_limits(&self) -> JudgeLimits { self.judge.semantic_configuration().1 }
    pub fn extraction_limits(&self) -> Int4ObjectExtractionMapLimits { self.extraction_limits }
    pub fn judge_mapping_limits(&self) -> Int4FaithfulnessMapLimits { self.judge_mapping }
    pub fn semantic_limits(&self) -> SemanticLimits { self.options.limits }
    pub fn max_model_work(&self) -> Int4Work { self.options.max_model_work }
    pub fn max_result_bytes(&self) -> u64 { self.options.limits.max_output_bytes }
    pub fn input_encoding(&self) -> &UntrustedEncoding { &self.expected.encoding }
    pub fn input_encoding_work(&self) -> Option<SafeBpeWork> { self.expected.extraction.encoding_work() }
    pub fn encoding_workspace_bytes(&self) -> Result<u64, Int4SemanticObjectDocumentError> {
        self.judge.document_encoding_workspace_bytes()
            .map_err(|e| Int4SemanticObjectDocumentError::Semantic(e.into()))
    }
    pub fn extraction_preflight(&self) -> Int4ObjectExtractionMapPreflight { self.expected.extraction }
    pub fn preflight_metadata(&self) -> Int4SemanticObjectDocumentPreflight { self.expected.clone() }

    /// Initial model/identity and physical cache preflight. Dynamic field maps
    /// are necessarily prepared after extraction, then ALL preflight before
    /// any reviewer forward. A failure in either phase returns no combined run.
    pub fn preflight_with_control<C: DecodeStepControl>(&self, admitted: &[ExecutionIdentity],
        engine: &StrictInt4Engine<'_>, control: &mut C)
        -> Result<(), Int4SemanticObjectDocumentError> {
        poll_document(control)?;
        if admitted.len() != self.extraction.execution_identities().len()
            || admitted.iter().zip(self.extraction.execution_identities()).any(|(a, b)| a != b) {
            return Err(Int4SemanticObjectDocumentError::Contract);
        }
        check_engine(engine, self.group_width(), self.extraction_budget.max_kv_bytes.min(self.judge_budget().max_kv_bytes))?;
        for identity in self.execution_identities().chain(std::iter::once(self.judge_identity())) {
            poll_document(control)?;
            constrained_int4::check_model(identity, engine.model_identity())
                .map_err(|_| Int4SemanticObjectDocumentError::Contract)?;
        }
        let positions = engine.kv_cache().capacity_positions();
        if self.extraction_limits.mapping.mapping.chunks.context_tokens > positions
            || self.judge_mapping.chunks.context_tokens > positions {
            return Err(Int4SemanticObjectDocumentError::Contract);
        }
        poll_document(control)
    }

    pub fn execute_with_control<C: DecodeStepControl>(self, admitted: &[ExecutionIdentity],
        engine: &mut StrictInt4Engine<'_>, vocabulary: &ExtractionVocabulary, control: &mut C)
        -> Result<Int4SemanticObjectDocumentRun, Int4SemanticObjectDocumentError> {
        self.preflight_with_control(admitted, engine, control)?;
        let Self { extraction, source, judge, options, judge_mapping, expected, .. } = self;
        let (extraction, schema) = extraction.execute_with_original_schema(admitted, engine, vocabulary, control)?;
        let fields = prepare_fields(&schema, &extraction, &expected, &options, control)?;
        let requests = review_requests(source, &fields, &options, control)?;
        let reviews = prepare_reviews(&judge, &requests, &expected, &options, judge_mapping, control)?;
        // Every real field/chunk/head is checked before any review executes.
        reviews.preflight(engine, options.specification.judge_budget.max_kv_bytes, control)?;
        let mut outputs = vector(reviews.maps.len())?;
        for plan in &reviews.maps {
            poll_document(control)?;
            let identities = plan.identities()?;
            let output = plan.execute_with_control(&identities, engine, Int4ScoringBudget {
                native: Int4RunBudget::exact(plan.planned_work()),
                max_kv_bytes: options.specification.judge_budget.max_kv_bytes,
            }, control)?;
            plan.preflight_metadata().verify_completed_with_control(&output, control)?;
            outputs.push(output);
        }
        check_engine(engine, expected.extraction.group_width(), options.specification.judge_budget.max_kv_bytes)?;
        finish_combined(extraction, fields, &reviews, outputs, expected, &options, control)
    }
}

fn check_judge_configuration(judge: &Int4JudgePlanner, task: TaskBudget)
    -> Result<(), Int4SemanticObjectDocumentError> {
    let (ceiling, limits) = judge.semantic_configuration();
    let p = limits.per_head;
    if task.max_input_tokens > ceiling.max_input_tokens || task.max_output_tokens > ceiling.max_output_tokens
        || task.max_output_bytes > ceiling.max_output_bytes || task.max_grammar_states > ceiling.max_grammar_states
        || task.max_kv_bytes > ceiling.max_kv_bytes || limits.max_total_prompt_tokens == 0
        || limits.max_total_candidate_tokens == 0 || limits.max_total_projected_logits == 0
        || limits.max_output_bytes == 0 || p.max_candidates == 0 || p.max_total_tokens == 0
        || p.max_nodes == 0 || p.max_depth == 0 || p.max_candidate_id_bytes == 0 || p.max_projected_logits == 0 {
        return Err(Int4SemanticObjectDocumentError::Contract);
    }
    Ok(())
}
fn check_engine(engine: &StrictInt4Engine<'_>, width: usize, kv_cap: u64)
    -> Result<(), Int4SemanticObjectDocumentError> {
    let bytes = (engine.kv_cache().capacity_positions() as u64).checked_mul(KV_BYTES_PER_TOKEN as u64)
        .ok_or(Int4SemanticObjectDocumentError::Contract)?;
    if engine.profile() != STRICT_INT4_PROFILE || engine.model_identity().group_width() != width
        || engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) || bytes == 0 || bytes > kv_cap {
        return Err(Int4SemanticObjectDocumentError::Contract);
    }
    Ok(())
}
fn prepare_fields<C: DecodeStepControl>(schema: &RootObjectSchema, extraction: &Int4ObjectExtractionMapRun,
    expected: &Int4SemanticObjectDocumentPreflight, options: &Int4SemanticOptions, control: &mut C)
    -> Result<Vec<RenderedField>, Int4SemanticObjectDocumentError> {
    poll_document(control)?;
    expected.extraction.verify_completed(extraction)?;
    poll_document(control)?;
    let limits = options.limits;
    let value = parse_json_with_limits(extraction.object.json(), JsonLimits {
        max_input_bytes: limits.max_input_json_bytes, max_string_lexeme_bytes: limits.max_input_json_bytes,
        max_container_entries: limits.max_walk_nodes, ..JsonLimits::default()
    }).map_err(|_| Int4SemanticObjectDocumentError::Contract)?;
    let fields = claims::render_fields(schema.original_schema(), &value, &options.specification.claims, limits)?;
    poll_document(control)?; Ok(fields)
}
fn review_requests<C: DecodeStepControl>(source: &str, fields: &[RenderedField],
    options: &Int4SemanticOptions, control: &mut C)
    -> Result<Vec<JudgeRequest>, Int4SemanticObjectDocumentError> {
    let mut requests = vector(fields.iter().filter(|field| field.claim.is_some()).count())?;
    for field in fields {
        poll_document(control)?;
        if let Some(claim) = &field.claim {
            requests.push(JudgeRequest::Faithfulness { source: copy(source)?, claim: copy(claim)?,
                policy: options.specification.judge_policy, budget: options.specification.judge_budget });
        }
    }
    Ok(requests)
}
struct PreparedReviews<'a> {
    maps: Vec<PreparedInt4FaithfulnessMap<'a>>,
    work: Int4Work,
    prompt_tokens: usize,
    encoding_work: SafeBpeWork,
}
fn prepare_reviews<'a, C: DecodeStepControl>(judge: &Int4JudgePlanner, requests: &'a [JudgeRequest],
    expected: &Int4SemanticObjectDocumentPreflight, options: &Int4SemanticOptions,
    mapping: Int4FaithfulnessMapLimits, control: &mut C)
    -> Result<PreparedReviews<'a>, Int4SemanticObjectDocumentError> {
    let mut review = PreparedReviews { maps: vector(requests.len())?, work: Int4Work::default(),
        prompt_tokens: 0, encoding_work: expected.initial_encoding_work };
    for request in requests {
        poll_document(control)?;
        let plan = judge.plan_faithfulness_document_with_encoding_work(request, mapping, control, &mut review.encoding_work)?;
        if plan.group_width() != expected.extraction.group_width()
            || plan.preflight_metadata().source_span() != expected.extraction.mapped().source_span() {
            return Err(Int4SemanticObjectDocumentError::Accounting);
        }
        review.work = plus(review.work, plan.planned_work())?;
        reserve_combined(expected.extraction.planned_work(), review.work, options.max_model_work)?;
        review.prompt_tokens = review.prompt_tokens.checked_add(plan.total_prompt_tokens()?)
            .ok_or(Int4SemanticObjectDocumentError::WorkBudget)?;
        if review.prompt_tokens > options.limits.max_total_prompt_tokens
            || review.work.projected_logits > options.limits.max_total_projected_logits {
            return Err(Int4SemanticObjectDocumentError::WorkBudget);
        }
        Int4SemanticEncoding::from_work(&expected.encoding, review.encoding_work)?;
        review.maps.push(plan);
    }
    poll_document(control)?; Ok(review)
}
impl PreparedReviews<'_> {
    fn preflight<C: DecodeStepControl>(&self, engine: &StrictInt4Engine<'_>, kv: u64, control: &mut C)
        -> Result<(), Int4SemanticObjectDocumentError> {
        for plan in &self.maps {
            poll_document(control)?;
            plan.preflight_with_control(&plan.identities()?, engine, Int4ScoringBudget {
                native: Int4RunBudget::exact(plan.planned_work()), max_kv_bytes: kv,
            }, control)?;
        }
        poll_document(control)
    }
}

/// This names the review operation, not an entailment/correctness verdict.
/// Actual chunk aggregation, including conflicts and abstention, stays nested.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticObjectFieldStatus { Reviewed, NotChecked }
#[derive(Serialize)]
pub struct SemanticObjectFieldResult {
    pub json_pointer: String,
    pub status: SemanticObjectFieldStatus,
    pub not_checked: Option<SemanticNotChecked>,
    pub claim_rule_id: Option<String>,
    pub rendered_claim: Option<String>,
    pub same_model_correlated: Option<bool>,
    pub prompt_tokens: usize,
    pub review: Option<Int4FaithfulnessMapRun>,
    #[serde(skip)]
    expected: Option<Int4FaithfulnessMapPreflight>,
}
#[derive(Serialize)]
pub struct SemanticObjectDocumentReceipt {
    pub schema_version: u32,
    pub algorithm: &'static str,
    pub claim_rendering: &'static str,
    pub judge_algorithm: &'static str,
    pub scope: &'static str,
    pub specification_revision: String,
    pub calibration: &'static str,
    pub same_model_correlated: bool,
    pub single_context_equivalence: bool,
    pub attempted_fields: usize,
    pub checked_fields: usize,
    pub not_checked_fields: usize,
    pub fields: Vec<SemanticObjectFieldResult>,
}
#[derive(Serialize)]
pub struct Int4SemanticObjectDocumentRun {
    pub schema_version: u32,
    pub execution: &'static str,
    pub numerics_profile: &'static str,
    pub group_width: usize,
    pub extraction_reserved_work: Int4Work,
    pub semantic_model_work: Int4Work,
    pub reserved_model_work: Int4Work,
    pub model_work: Int4Work,
    pub reserved_mask_node_visits: u64,
    pub mask_node_visit_charge: u64,
    pub review_prompt_tokens: usize,
    pub judge_head_count: usize,
    pub rewound_positions: u64,
    /// Exact original map, object JSON, provenance, tokens and evidence.
    pub extraction: Int4ObjectExtractionMapRun,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_encoding: Option<Int4SemanticEncoding>,
    pub semantic: SemanticObjectDocumentReceipt,
    #[serde(skip)]
    request_binding: [u8; 32],
    #[serde(skip)]
    result_binding: [u8; 32],
}

/// Sealed original-map and complete-request authority. This is not wire data.
#[derive(Clone)]
pub struct Int4SemanticObjectDocumentPreflight {
    extraction: Int4ObjectExtractionMapPreflight,
    request_binding: [u8; 32],
    encoding: UntrustedEncoding,
    initial_encoding_work: SafeBpeWork,
    limits: SemanticLimits,
    max_model_work: Int4Work,
}
impl Int4SemanticObjectDocumentPreflight {
    pub fn verify_completed_with_control<C: DecodeStepControl>(&self, run: &Int4SemanticObjectDocumentRun,
        control: &mut C) -> Result<(), Int4SemanticObjectDocumentError> {
        poll_document(control)?;
        if run.request_binding != self.request_binding
            || run.result_binding != result_binding(run, self.limits.max_output_bytes, control)? {
            return Err(Int4SemanticObjectDocumentError::Accounting);
        }
        self.extraction.verify_completed(&run.extraction)?;
        poll_document(control)?;
        let mut work = Int4Work::default(); let mut prompts = 0_usize; let mut heads = 0_usize;
        let mut rewound = 0_u64; let mut attempted = 0_usize; let mut checked = 0_usize;
        let mut encoding_work = self.initial_encoding_work;
        for field in &run.semantic.fields {
            poll_document(control)?;
            match (&field.review, field.expected) {
                (Some(review), Some(expected)) => {
                    expected.verify_completed_with_control(review, control)?;
                    if expected.source_span() != self.extraction.mapped().source_span()
                        || review.group_width != self.extraction.group_width() || field.prompt_tokens == 0
                        || field.claim_rule_id.is_none() || field.rendered_claim.is_none()
                        || field.same_model_correlated != Some(true) {
                        return Err(Int4SemanticObjectDocumentError::Accounting);
                    }
                    let (status, reason) = review_status(review);
                    if field.status != status || field.not_checked != reason {
                        return Err(Int4SemanticObjectDocumentError::Accounting);
                    }
                    work = plus(work, review.model_work)?;
                    prompts = prompts.checked_add(field.prompt_tokens).ok_or(Int4SemanticObjectDocumentError::Accounting)?;
                    heads = heads.checked_add(expected.head_count()).ok_or(Int4SemanticObjectDocumentError::Accounting)?;
                    for chunk in &review.chunks {
                        rewound = rewound.checked_add(chunk.run.rewound_positions).ok_or(Int4SemanticObjectDocumentError::Accounting)?;
                    }
                    encoding_work = encoding_work.checked_add(review.input_encoding.as_ref().map(|e| e.work).unwrap_or_default())
                        .ok_or(Int4SemanticObjectDocumentError::Accounting)?;
                    attempted += 1;
                    checked += usize::from(status == SemanticObjectFieldStatus::Reviewed);
                }
                (None, None) => {
                    if field.status != SemanticObjectFieldStatus::NotChecked || field.not_checked.is_none()
                        || field.not_checked == Some(SemanticNotChecked::JudgeAbstained)
                        || field.claim_rule_id.is_some() || field.rendered_claim.is_some()
                        || field.same_model_correlated.is_some() || field.prompt_tokens != 0 {
                        return Err(Int4SemanticObjectDocumentError::Accounting);
                    }
                }
                _ => return Err(Int4SemanticObjectDocumentError::Accounting),
            }
        }
        let reserved = reserve_combined(self.extraction.planned_work(), work, self.max_model_work)?;
        let actual = plus(run.extraction.partial_map.model_work, work)?;
        let encoding = Int4SemanticEncoding::from_work(&self.encoding, encoding_work)?;
        let s = &run.semantic;
        if run.schema_version != 1 || run.execution != INT4_SEMANTIC_OBJECT_DOCUMENT_EXECUTION
            || run.numerics_profile != STRICT_INT4_PROFILE || run.group_width != self.extraction.group_width()
            || run.extraction_reserved_work != self.extraction.planned_work()
            || run.semantic_model_work != work || run.reserved_model_work != reserved || run.model_work != actual
            || !fits(actual, reserved) || run.input_encoding != encoding
            || run.reserved_mask_node_visits != self.extraction.mapped().reserved_mask_visits()
            || run.mask_node_visit_charge != run.extraction.partial_map.mask_node_visit_charge
            || run.mask_node_visit_charge > run.reserved_mask_node_visits
            || run.review_prompt_tokens != prompts || prompts > self.limits.max_total_prompt_tokens
            || work.projected_logits > self.limits.max_total_projected_logits
            || run.judge_head_count != heads || run.rewound_positions != rewound
            || s.schema_version != 1 || s.algorithm != SEMANTIC_VERIFICATION_VERSION
            || s.claim_rendering != CLAIM_RENDER_VERSION || s.judge_algorithm != INT4_FAITHFULNESS_MAP_EXECUTION
            || s.scope != SEMANTIC_OBJECT_DOCUMENT_SCOPE || s.calibration != CALIBRATION
            || !s.same_model_correlated || s.single_context_equivalence || s.fields.len() > self.limits.max_fields
            || s.attempted_fields != attempted || s.checked_fields != checked
            || s.not_checked_fields != s.fields.len().checked_sub(checked).ok_or(Int4SemanticObjectDocumentError::Accounting)? {
            return Err(Int4SemanticObjectDocumentError::Accounting);
        }
        poll_document(control)
    }
}
fn review_status(run: &Int4FaithfulnessMapRun) -> (SemanticObjectFieldStatus, Option<SemanticNotChecked>) {
    if run.aggregate.relation.is_some() { (SemanticObjectFieldStatus::Reviewed, None) }
    else { (SemanticObjectFieldStatus::NotChecked, Some(SemanticNotChecked::JudgeAbstained)) }
}
#[allow(clippy::too_many_arguments)]
fn finish_combined<C: DecodeStepControl>(extraction: Int4ObjectExtractionMapRun, fields: Vec<RenderedField>,
    reviews: &PreparedReviews<'_>, results: Vec<Int4FaithfulnessMapRun>,
    expected: Int4SemanticObjectDocumentPreflight, options: &Int4SemanticOptions, control: &mut C)
    -> Result<Int4SemanticObjectDocumentRun, Int4SemanticObjectDocumentError> {
    poll_document(control)?;
    expected.extraction.verify_completed(&extraction)?;
    poll_document(control)?;
    if results.len() != reviews.maps.len() || fields.iter().filter(|f| f.claim.is_some()).count() != results.len() {
        return Err(Int4SemanticObjectDocumentError::Accounting);
    }
    let mut results = reviews.maps.iter().zip(results);
    let mut output = vector(fields.len())?;
    let mut checked = 0_usize; let mut heads = 0_usize; let mut rewound = 0_u64;
    for field in fields {
        poll_document(control)?;
        let (review, authority, prompts, status, not_checked) = if field.claim.is_some() {
            let (plan, review) = results.next().ok_or(Int4SemanticObjectDocumentError::Accounting)?;
            let authority = plan.preflight_metadata();
            authority.verify_completed_with_control(&review, control)?;
            let (status, reason) = review_status(&review);
            checked += usize::from(status == SemanticObjectFieldStatus::Reviewed);
            heads = heads.checked_add(plan.head_count()).ok_or(Int4SemanticObjectDocumentError::Accounting)?;
            for chunk in &review.chunks {
                rewound = rewound.checked_add(chunk.run.rewound_positions).ok_or(Int4SemanticObjectDocumentError::Accounting)?;
            }
            (Some(review), Some(authority), plan.total_prompt_tokens()?, status, reason)
        } else { (None, None, 0, SemanticObjectFieldStatus::NotChecked, field.not_checked) };
        output.push(SemanticObjectFieldResult { json_pointer: field.pointer, status, not_checked,
            claim_rule_id: field.rule_id, rendered_claim: field.claim, same_model_correlated: review.as_ref().map(|_| true),
            prompt_tokens: prompts, review, expected: authority });
    }
    if results.next().is_some() { return Err(Int4SemanticObjectDocumentError::Accounting); }
    let not_checked_fields = output.len() - checked;
    let mut result = Int4SemanticObjectDocumentRun {
        schema_version: 1, execution: INT4_SEMANTIC_OBJECT_DOCUMENT_EXECUTION, numerics_profile: STRICT_INT4_PROFILE,
        group_width: expected.extraction.group_width(), extraction_reserved_work: expected.extraction.planned_work(),
        semantic_model_work: reviews.work,
        reserved_model_work: reserve_combined(expected.extraction.planned_work(), reviews.work, options.max_model_work)?,
        model_work: plus(extraction.partial_map.model_work, reviews.work)?,
        reserved_mask_node_visits: expected.extraction.mapped().reserved_mask_visits(),
        mask_node_visit_charge: extraction.partial_map.mask_node_visit_charge,
        review_prompt_tokens: reviews.prompt_tokens, judge_head_count: heads, rewound_positions: rewound,
        input_encoding: Int4SemanticEncoding::from_work(&expected.encoding, reviews.encoding_work)?, extraction,
        semantic: SemanticObjectDocumentReceipt { schema_version: 1, algorithm: SEMANTIC_VERIFICATION_VERSION,
            claim_rendering: CLAIM_RENDER_VERSION, judge_algorithm: INT4_FAITHFULNESS_MAP_EXECUTION,
            scope: SEMANTIC_OBJECT_DOCUMENT_SCOPE, specification_revision: copy(&options.specification.revision)?,
            calibration: CALIBRATION, same_model_correlated: true, single_context_equivalence: false,
            attempted_fields: reviews.maps.len(), checked_fields: checked, not_checked_fields, fields: output },
        request_binding: expected.request_binding, result_binding: [0; 32],
    };
    result.result_binding = result_binding(&result, options.limits.max_output_bytes, control)?;
    expected.verify_completed_with_control(&result, control)?;
    Ok(result)
}

fn request_binding<C: DecodeStepControl>(source: &str, judge: &Int4JudgePlanner, options: &Int4SemanticOptions,
    mapping: Int4FaithfulnessMapLimits, control: &mut C) -> Result<[u8; 32], Int4SemanticObjectDocumentError> {
    // Every input is already bounded. Streaming avoids another source/schema
    // allocation and preserves cancellation through private association.
    let mut hash = ControlledDigest::new(b"fnlp-semantic-object-document-request-v1", u64::MAX, control);
    hash.value(&(INT4_SEMANTIC_OBJECT_DOCUMENT_EXECUTION, SEMANTIC_OBJECT_DOCUMENT_SCOPE,
        source, judge.execution_identity(), options))?;
    hash.value(&(mapping.chunks, mapping.max_model_work, mapping.max_retained_result_bytes,
        mapping.max_result_bytes, mapping.aggregation))?;
    let (ceiling, p) = judge.semantic_configuration(); let h = p.per_head;
    hash.value(&(ceiling, p.max_total_prompt_tokens, p.max_total_candidate_tokens,
        p.max_total_projected_logits, p.max_output_bytes))?;
    hash.value(&(h.max_candidates, h.max_total_tokens, h.max_nodes, h.max_depth,
        h.max_candidate_id_bytes, h.max_projected_logits, judge.document_encoding()))?;
    hash.checkpoint()?; Ok(hash.finish())
}
fn result_binding<C: DecodeStepControl>(run: &Int4SemanticObjectDocumentRun, cap: u64, control: &mut C)
    -> Result<[u8; 32], Int4SemanticObjectDocumentError> {
    let mut hash = ControlledDigest::new(b"fnlp-semantic-object-document-result-v1", cap, control);
    hash.seal(&run.request_binding);
    // Bind hidden source/claim ownership as well as public result bytes.
    // Swapping a whole valid same-looking map from another source must fail.
    for field in &run.semantic.fields {
        if let Some(review) = &field.review {
            let (request, result) = review.completion_binding();
            hash.seal(&request); hash.seal(&result);
        }
    }
    hash.value(run)?; hash.checkpoint()?; Ok(hash.finish())
}
fn poll_document(control: &mut impl DecodeStepControl) -> Result<(), Int4SemanticObjectDocumentError> {
    match control.prefill_checkpoint(0) {
        Some(cause) => Err(Int4SemanticObjectDocumentError::Cancelled(cause)), None => Ok(()),
    }
}
struct ControlledDigest<'a, C> {
    hash: Sha256, remaining: u64, until_checkpoint: usize, control: &'a mut C,
    exceeded: bool, cancelled: Option<DecodeCancellationKind>,
}
impl<'a, C: DecodeStepControl> ControlledDigest<'a, C> {
    fn new(domain: &[u8], cap: u64, control: &'a mut C) -> Self {
        let mut hash = Sha256::new(); hash.update(domain); hash.update([0]);
        Self { hash, remaining: cap, until_checkpoint: 0, control, exceeded: false, cancelled: None }
    }
    fn seal(&mut self, bytes: &[u8; 32]) { self.hash.update(bytes); self.hash.update([0]); }
    fn checkpoint(&mut self) -> Result<(), Int4SemanticObjectDocumentError> {
        if self.cancelled.is_none() { self.cancelled = self.control.prefill_checkpoint(0); }
        if let Some(cause) = self.cancelled { return Err(Int4SemanticObjectDocumentError::Cancelled(cause)); }
        self.until_checkpoint = 64 * 1024; Ok(())
    }
    fn value(&mut self, value: &(impl Serialize + ?Sized)) -> Result<(), Int4SemanticObjectDocumentError> {
        if serde_json::to_writer(&mut *self, value).is_err() {
            if let Some(cause) = self.cancelled { return Err(Int4SemanticObjectDocumentError::Cancelled(cause)); }
            return Err(if self.exceeded { Int4SemanticObjectDocumentError::OutputBudget }
                else { Int4SemanticObjectDocumentError::Serialization });
        }
        self.hash.update([0]); Ok(())
    }
    fn finish(self) -> [u8; 32] { self.hash.finalize().into() }
}
impl<C: DecodeStepControl> std::io::Write for ControlledDigest<'_, C> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let mut offset = 0;
        while offset < bytes.len() {
            if self.until_checkpoint == 0 {
                self.checkpoint().map_err(|_| std::io::Error::other("semantic object binding cancelled"))?;
            }
            let count = self.until_checkpoint.min(bytes.len() - offset);
            let Some(remaining) = self.remaining.checked_sub(count as u64) else {
                self.exceeded = true; return Err(std::io::Error::other("semantic object binding limit"));
            };
            self.remaining = remaining; self.hash.update(&bytes[offset..offset + count]);
            self.until_checkpoint -= count; offset += count;
        }
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
}

#[cfg(test)]
mod tests;
