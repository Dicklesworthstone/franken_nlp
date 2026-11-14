//! Explicit same-model review of complete summary bullets or passage answers.
//! The existing source result is retained unchanged. Review sees the COMPLETE
//! original evidence, not just its displayed citations, and never certifies
//! citation entailment, atomic-claim coverage, correctness or acceptance.
use super::*;
use std::collections::BTreeSet;
use crate::{
    grammar::mask::MaskWorkLimits,
    native_engine::{kv::KV_BYTES_PER_TOKEN, lmhead::scoring::ScoringWork,
        strict_int4::{Int4RunBudget, scoring::Int4ScoringBudget}},
    tasks::{answer::{AnswerStatus, ANSWER_PASSAGE_LAYOUT_VERSION}, ir::ScoreSpace,
        extract::semantic::grouped_int4::Int4SemanticEncoding,
        judge::{partition_evidence, FaithfulnessPolicy, FaithfulnessRelation, FaithfulnessResult,
            JudgeLimits, JudgeRequest, JudgeResult, FAITHFULNESS_VERSION,
            grouped_int4::{Int4JudgeError, Int4JudgePlanner, Int4JudgeRun, PreparedInt4Judge,
                INT4_JUDGE_EXECUTION}},
        summarize::{CitationGuarantee, SummarySemanticSupport}},
    validation::grounded_fields::VerifiedSourceSpan,
};

pub const INT4_SOURCE_SEMANTIC_EXECUTION: &str = "grouped-int4-source-correlated-second-reader-v1";
pub const SOURCE_SEMANTIC_VERSION: &str = "source-explicit-complete-statement-faithfulness-v1";
pub const SOURCE_SEMANTIC_CLAIM_VERSION: &str = "exact-bullet-or-canonical-question-answer-v1";
pub const SOURCE_SEMANTIC_SCOPE: &str = "complete_original_source_not_citation_only_or_atomic_claim_coverage";
pub const SOURCE_SEMANTIC_CALIBRATION: &str = "experimental_uncalibrated_diagnostic_only";
const MAX_BYTES: usize = 64 * 1024 * 1024;
const PASSAGE_JOIN: &str = "\n\n";

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SourceSemanticLimits {
    pub max_statements: usize,
    pub max_source_bytes: usize,
    pub max_claim_bytes: usize,
    pub max_total_claim_bytes: usize,
    /// Across every retained review head, not renewable per statement.
    pub max_total_prompt_tokens: usize,
    pub max_total_projected_logits: u64,
    /// Complete original source result plus every diagnostic and native count.
    pub max_output_bytes: u64,
}
impl Default for SourceSemanticLimits {
    fn default() -> Self {
        Self { max_statements: 32, max_source_bytes: 1024 * 1024, max_claim_bytes: 16 * 1024,
            max_total_claim_bytes: 1024 * 1024, max_total_prompt_tokens: 1_000_000,
            max_total_projected_logits: 100_000_000, max_output_bytes: 8 * 1024 * 1024 }
    }
}
impl SourceSemanticLimits {
    pub fn validate(self) -> Result<(), Int4SourceSemanticError> {
        if !(1..=1024).contains(&self.max_statements)
            || !(1..=MAX_BYTES).contains(&self.max_source_bytes)
            || !(1..=1024 * 1024).contains(&self.max_claim_bytes)
            || !(1..=MAX_BYTES).contains(&self.max_total_claim_bytes)
            || !(1..=16_777_216).contains(&self.max_total_prompt_tokens)
            || self.max_total_projected_logits == 0
            || !(1..=MAX_BYTES as u64).contains(&self.max_output_bytes) {
            return Err(Int4SourceSemanticError::Contract);
        }
        Ok(())
    }
}

/// No automatic opt-in or default policy. All five native work ceilings cover
/// the maximum source-generation reservation plus EVERY complete review.
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Int4SourceSemanticOptions {
    pub schema_version: u32,
    pub experimental_opt_in: bool,
    pub judge_policy: FaithfulnessPolicy,
    pub judge_budget: TaskBudget,
    pub limits: SourceSemanticLimits,
    pub max_model_work: Int4Work,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SourceSemanticInputFacts {
    pub source_bytes: usize,
    pub question_bytes: usize,
    pub passage_count: usize,
}
impl Int4SourceSemanticOptions {
    pub fn validate(&self) -> Result<(), Int4SourceSemanticError> {
        self.limits.validate()?;
        self.judge_policy.validate().map_err(Int4JudgeError::from)?;
        self.judge_budget.validate().map_err(|_| Int4SourceSemanticError::Contract)?;
        let w = self.max_model_work;
        if self.schema_version != 1 || !self.experimental_opt_in
            || self.judge_budget.max_output_bytes > self.limits.max_output_bytes
            || w.forward_positions == 0 || w.projected_logits == 0 || w.attention_pairs == 0
            || w.projections.dot_products == 0 || w.projections.multiply_accumulates == 0 {
            return Err(Int4SourceSemanticError::Contract);
        }
        Ok(())
    }
    /// Model-free preflight. The sealed constructor also binds these exact
    /// source bytes, question, passage IDs/order, options and source task budget
    /// to the original source plan; these size facts grant no execution rights.
    pub fn input_facts(&self, request: &SourceTaskRequest) -> Result<SourceSemanticInputFacts, Int4SourceSemanticError> {
        self.validate()?;
        request.budget().validate().map_err(|_| Int4SourceSemanticError::Contract)?;
        if request.budget().max_output_bytes > self.limits.max_output_bytes {
            return Err(Int4SourceSemanticError::Contract);
        }
        let facts = match request {
            SourceTaskRequest::Summarize { document, options, .. } => {
                options.validate().map_err(Int4SourceError::from)?;
                if options.max_bullets > self.limits.max_statements { return Err(Int4SourceSemanticError::Contract); }
                SourceSemanticInputFacts { source_bytes: document.len(), question_bytes: 0, passage_count: 0 }
            }
            SourceTaskRequest::Answer { question, passages, options, .. } => {
                options.validate().map_err(Int4SourceError::from)?;
                if question.trim().is_empty() || question.len() > self.limits.max_claim_bytes
                    || passages.is_empty() || passages.len() > 1024 {
                    return Err(Int4SourceSemanticError::Contract);
                }
                let mut ids = BTreeSet::new();
                let mut source_bytes = PASSAGE_JOIN.len().checked_mul(passages.len() - 1)
                    .ok_or(Int4SourceSemanticError::Contract)?;
                for passage in passages {
                    source_bytes = source_bytes.checked_add(passage.text.len()).ok_or(Int4SourceSemanticError::Contract)?;
                    if passage.id.is_empty() || passage.id.len() > 128 || !ids.insert(passage.id.as_str())
                        || passage.text.trim().is_empty() || source_bytes > self.limits.max_source_bytes {
                        return Err(Int4SourceSemanticError::Contract);
                    }
                }
                SourceSemanticInputFacts { source_bytes, question_bytes: question.len(), passage_count: passages.len() }
            }
            _ => return Err(Int4SourceSemanticError::Contract),
        };
        if facts.source_bytes == 0 || facts.source_bytes > self.limits.max_source_bytes {
            return Err(Int4SourceSemanticError::Contract);
        }
        Ok(facts)
    }
    pub fn policy_digest(&self) -> Result<Sha256Digest, Int4SourceSemanticError> {
        self.validate()?;
        let bytes = canonjson::canonical_bytes(&(SOURCE_SEMANTIC_VERSION, SOURCE_SEMANTIC_CLAIM_VERSION,
            SOURCE_SEMANTIC_SCOPE, self)).map_err(|_| Int4SourceSemanticError::Serialization)?;
        Ok(Sha256Digest::of_bytes(&bytes))
    }
    /// Check complete original evidence coverage before loading a model. This
    /// uses the same passage layout as the sealed constructor, without making
    /// question or manifest bytes part of the judge's evidence.
    pub fn validate_evidence_with_control<C: DecodeStepControl>(&self, request: &SourceTaskRequest,
        control: &mut C) -> Result<SourceSemanticInputFacts, Int4SourceSemanticError> {
        poll(control)?;
        let facts = self.input_facts(request)?;
        match request {
            SourceTaskRequest::Summarize { document, .. } => check_evidence(document, self.judge_policy, control)?,
            SourceTaskRequest::Answer { passages, .. } => {
                let (source, _) = join_passages(passages, facts.source_bytes, control)?;
                check_evidence(&source, self.judge_policy, control)?;
            }
            _ => return Err(Int4SourceSemanticError::Contract),
        }
        Ok(facts)
    }
}

pub enum Int4SourceSemanticError {
    Contract, InputMismatch, WorkBudget, ClaimBudget, Accounting, Allocation, Serialization,
    Cancelled(DecodeCancellationKind), Source(Int4SourceError), Judge(Int4JudgeError),
}
impl fmt::Display for Int4SourceSemanticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Contract => "INT4 source review configuration refused",
            Self::InputMismatch => "INT4 source review original request or evidence differs",
            Self::WorkBudget => "INT4 source review whole-request native work exceeded",
            Self::ClaimBudget => "INT4 source review complete statement allowance exceeded",
            Self::Accounting => "INT4 source review complete-result accounting diverged",
            Self::Allocation => "INT4 source review allocation refused",
            Self::Serialization => "INT4 source review serialization refused",
            Self::Cancelled(_) => "INT4 source review cancelled",
            Self::Source(_) => "INT4 source review generation or source validation failed",
            Self::Judge(_) => "INT4 source review complete faithfulness judgment failed",
        })
    }
}
impl fmt::Debug for Int4SourceSemanticError { fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Display::fmt(self, f) } }
impl Error for Int4SourceSemanticError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Source(e) => Some(e), Self::Judge(e) => Some(e), _ => None }
    }
}
impl From<Int4SourceError> for Int4SourceSemanticError { fn from(e: Int4SourceError) -> Self { Self::Source(e) } }
impl From<ExtractError> for Int4SourceSemanticError { fn from(e: ExtractError) -> Self { Self::Source(e.into()) } }
impl From<Int4JudgeError> for Int4SourceSemanticError { fn from(e: Int4JudgeError) -> Self { Self::Judge(e) } }
impl Int4SourceSemanticError {
    pub fn cancellation(&self) -> Option<DecodeCancellationKind> {
        match self { Self::Cancelled(c) => Some(*c), Self::Source(e) => e.cancellation(),
            Self::Judge(e) => e.cancellation(), _ => None }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SourceSemanticPassage {
    pub passage_id: String,
    /// Original passage range in the review's exact joined source. Judge
    /// evidence windows use these global coordinates; joins are not passages.
    pub span: VerifiedSourceSpan,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SourceSemanticEvidenceLayout {
    Document { span: VerifiedSourceSpan },
    Passages { profile: &'static str, passages: Vec<SourceSemanticPassage> },
}
struct EvidenceContext {
    source: String,
    question: Option<String>,
    layout: SourceSemanticEvidenceLayout,
    facts: SourceSemanticInputFacts,
}
impl EvidenceContext {
    fn new<C: DecodeStepControl>(request: SourceTaskRequest, options: &Int4SourceSemanticOptions,
        control: &mut C) -> Result<Self, Int4SourceSemanticError> {
        poll(control)?;
        let facts = options.input_facts(&request)?;
        let (source, question, layout) = match request {
            SourceTaskRequest::Summarize { document, .. } => {
                let span = VerifiedSourceSpan { byte_start: 0, byte_end: document.len(),
                    scalar_start: 0, scalar_end: document.chars().count() };
                (document, None, SourceSemanticEvidenceLayout::Document { span })
            }
            SourceTaskRequest::Answer { question, passages, .. } => {
                let (source, layout) = join_passages(&passages, facts.source_bytes, control)?;
                (source, Some(question), SourceSemanticEvidenceLayout::Passages {
                    profile: ANSWER_PASSAGE_LAYOUT_VERSION, passages: layout })
            }
            _ => return Err(Int4SourceSemanticError::Contract),
        };
        // A whole-source judge must cover every original byte, even if source
        // generation later abstains. No tail is silently dropped from review.
        check_evidence(&source, options.judge_policy, control)?;
        if source.len() != facts.source_bytes { return Err(Int4SourceSemanticError::InputMismatch); }
        Ok(Self { source, question, layout, facts })
    }
}
fn join_passages<C: DecodeStepControl>(passages: &[AnswerPassage], bytes: usize, control: &mut C)
    -> Result<(String, Vec<SourceSemanticPassage>), Int4SourceSemanticError> {
    let mut source = String::new();
    source.try_reserve_exact(bytes).map_err(|_| Int4SourceSemanticError::Allocation)?;
    let mut layout = vector(passages.len())?; let mut scalar = 0;
    for (index, passage) in passages.iter().enumerate() {
        poll(control)?;
        if index != 0 { source.push_str(PASSAGE_JOIN); scalar += PASSAGE_JOIN.len(); }
        let byte_start = source.len(); let scalar_start = scalar;
        source.push_str(&passage.text); scalar += passage.text.chars().count();
        layout.push(SourceSemanticPassage { passage_id: copy(&passage.id)?, span: VerifiedSourceSpan {
            byte_start, byte_end: source.len(), scalar_start, scalar_end: scalar } });
    }
    if source.len() != bytes { return Err(Int4SourceSemanticError::InputMismatch); }
    Ok((source, layout))
}
fn check_evidence<C: DecodeStepControl>(source: &str, policy: FaithfulnessPolicy, control: &mut C)
    -> Result<(), Int4SourceSemanticError> {
    poll(control)?;
    partition_evidence(source, policy).map_err(Int4JudgeError::from)?;
    poll(control)
}

/// Sealed native composition. No wire result or caller-supplied source can be
/// substituted for the prepared source task's exact request. Both stages run
/// on one native engine; every dynamic judge preflights before review begins.
pub struct PreparedInt4SourceSemantic {
    source_task: PreparedInt4SourceTask,
    evidence: EvidenceContext,
    judge: Int4JudgePlanner,
    options: Int4SourceSemanticOptions,
    policy_digest: Sha256Digest,
}
impl PreparedInt4SourceSemantic {
    pub fn new<C: DecodeStepControl>(source_task: PreparedInt4SourceTask, request: SourceTaskRequest,
        judge: Int4JudgePlanner, options: Int4SourceSemanticOptions, control: &mut C)
        -> Result<Self, Int4SourceSemanticError> {
        poll(control)?;
        if source_task.input_encoding() != judge.document_encoding() {
            return Err(Int4SourceSemanticError::Contract);
        }
        let encoding = Int4SemanticEncoding::from_work(source_task.input_encoding(),
            source_task.input_encoding_work().unwrap_or_default())
            .map_err(|_| Int4SourceSemanticError::Contract)?;
        if encoding.is_some() != source_task.input_encoding_work().is_some() {
            return Err(Int4SourceSemanticError::Contract);
        }
        let (ceiling, limits) = judge.semantic_configuration();
        check_configuration(&source_task, &request, judge.execution_identity(), judge.group_width(), ceiling, limits, &options)?;
        poll(control)?;
        let evidence = EvidenceContext::new(request, &options, control)?;
        let policy_digest = options.policy_digest()?;
        poll(control)?;
        Ok(Self { source_task, evidence, judge, options, policy_digest })
    }
    pub fn execution_identity(&self) -> &ExecutionIdentity { self.source_task.execution_identity() }
    pub fn judge_identity(&self) -> &ExecutionIdentity { self.judge.execution_identity() }
    pub fn group_width(&self) -> usize { self.source_task.group_width() }
    pub fn source_reserved_work(&self) -> Int4Work { self.source_task.planned_work() }
    pub fn max_model_work(&self) -> Int4Work { self.options.max_model_work }
    pub fn source_budget(&self) -> TaskBudget { self.source_task.task_budget() }
    pub fn judge_budget(&self) -> TaskBudget { self.options.judge_budget }
    pub fn judge_planning_limits(&self) -> JudgeLimits { self.judge.semantic_configuration().1 }
    pub fn input_facts(&self) -> SourceSemanticInputFacts { self.evidence.facts }
    pub fn semantic_limits(&self) -> SourceSemanticLimits { self.options.limits }
    pub fn policy_digest(&self) -> Sha256Digest { self.policy_digest }
    pub fn max_result_bytes(&self) -> u64 { self.options.limits.max_output_bytes }
    /// One immutable input policy bounds the original source and ALL dependent
    /// review heads, including every generated claim and evidence window.
    pub fn input_encoding(&self) -> &UntrustedEncoding { self.source_task.input_encoding() }
    pub fn input_encoding_work(&self) -> Option<SafeBpeWork> { self.source_task.input_encoding_work() }
    /// The same transient encoder workspace is reused sequentially. Hosts
    /// admit it before any native generation or review allocation.
    pub fn encoding_workspace_bytes(&self) -> Result<u64, Int4SourceSemanticError> {
        Ok(self.judge.document_encoding_workspace_bytes()?)
    }

    pub fn execute_with_control<C: DecodeStepControl>(self, engine: &mut StrictInt4Engine<'_>,
        vocabulary: &ExtractionVocabulary, masks: MaskWorkLimits, mask_visits: u64, control: &mut C)
        -> Result<Int4SourceSemanticRun, Int4SourceSemanticError> {
        poll(control)?;
        if masks.max_trie_node_visits == 0 || masks.checkpoint_interval_nodes == 0 || mask_visits == 0 {
            return Err(Int4SourceSemanticError::Contract);
        }
        let kv = self.check_capacity(engine)?;
        let reserved = self.source_reserved_work();
        let source_task = self.source_task.execute_with_control(self.execution_identity(), engine, vocabulary,
            Int4JsonBudget { native: Int4RunBudget::exact(reserved), json: JsonWorkBudget {
                max_forward_positions: reserved.forward_positions, max_projected_logits: reserved.projected_logits,
                max_kv_bytes: kv, max_total_mask_node_visits: mask_visits, mask_limits: masks } }, control)?;
        self.check_capacity(engine)?;
        let review = self.prepare_review(source_task, control)?;
        for plan in &review.judges {
            poll(control)?;
            plan.preflight(plan.execution_identity(), engine, Int4ScoringBudget {
                native: Int4RunBudget::exact(plan.planned_work()), max_kv_bytes: kv })?;
        }
        let mut results = vector(review.judges.len())?;
        for plan in &review.judges {
            poll(control)?;
            results.push(plan.execute_with_control(plan.execution_identity(), engine, Int4ScoringBudget {
                native: Int4RunBudget::exact(plan.planned_work()), max_kv_bytes: kv }, control)?);
            self.check_capacity(engine)?;
        }
        self.finish(review, results, control)
    }
    fn check_capacity(&self, engine: &StrictInt4Engine<'_>) -> Result<u64, Int4SourceSemanticError> {
        for identity in [self.execution_identity(), self.judge_identity()] {
            constrained_int4::check_model(identity, engine.model_identity()).map_err(|_| Int4SourceSemanticError::Contract)?;
        }
        let positions = engine.kv_cache().capacity_positions();
        let cap = self.source_budget().max_kv_bytes.min(self.judge_budget().max_kv_bytes);
        if engine.profile() != STRICT_INT4_PROFILE || engine.model_identity().group_width() != self.group_width()
            || engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0)
            || self.source_reserved_work().forward_positions > positions as u64 {
            return Err(Int4SourceSemanticError::Contract);
        }
        (positions as u64).checked_mul(KV_BYTES_PER_TOKEN as u64).filter(|&n| n != 0 && n <= cap)
            .ok_or(Int4SourceSemanticError::Contract)
    }
    fn prepare_review<C: DecodeStepControl>(&self, source_task: Int4SourceTaskRun, control: &mut C)
        -> Result<Review, Int4SourceSemanticError> {
        let statements = prepare_statements(&self.source_task, &self.evidence, &source_task, self.options.limits, control)?;
        let count = statements.iter().filter(|s| s.claim.is_some()).count();
        let mut judges = vector(count)?; let mut work = Int4Work::default();
        // Seed from the sealed original question/manifest/source encoding.
        // The public consuming execution owns this ledger until completion;
        // no statement, evidence head or failed review renews its allowance.
        let mut encoding_work = self.input_encoding_work().unwrap_or_default();
        let mut logits_left = self.options.limits.max_total_projected_logits;
        // Fixed shares cap the sum of retained complete-head prompts. This
        // also makes statement order irrelevant to prompt admission authority.
        let prompt_share = self.options.limits.max_total_prompt_tokens / count.max(1);
        for statement in &statements {
            let Some(claim) = &statement.claim else { continue; };
            poll(control)?;
            let request = JudgeRequest::Faithfulness { source: copy(&self.evidence.source)?, claim: copy(claim)?,
                policy: self.options.judge_policy, budget: self.options.judge_budget };
            let plan = self.judge.plan_semantic_with_encoding_work(&request, prompt_share,
                logits_left, control, &mut encoding_work)?;
            if plan.group_width() != self.group_width() { return Err(Int4SourceSemanticError::Accounting); }
            same_model(self.execution_identity(), plan.execution_identity())?;
            logits_left = logits_left.checked_sub(plan.planned_work().projected_logits).ok_or(Int4SourceSemanticError::WorkBudget)?;
            work = plus(work, plan.planned_work())?;
            reserve_combined(self.source_reserved_work(), work, self.options.max_model_work)?;
            judges.push(plan);
        }
        poll(control)?;
        Int4SemanticEncoding::from_work(self.input_encoding(), encoding_work)
            .map_err(|_| Int4SourceSemanticError::Accounting)?;
        Ok(Review { source_task, statements, judges, work, encoding_work })
    }
    fn finish<C: DecodeStepControl>(self, review: Review, results: Vec<Int4JudgeRun>, control: &mut C)
        -> Result<Int4SourceSemanticRun, Int4SourceSemanticError> {
        let source_reserved_work = self.source_reserved_work(); let group_width = self.group_width();
        let cap = self.max_result_bytes();
        let (statements, semantic_model_work, scoring, judge_head_count, rewound_positions) =
            finish_statements(&review.statements, &review.judges, results, self.options.judge_policy, group_width, control)?;
        let mut encoding_work = self.input_encoding_work().unwrap_or_default();
        for plan in &review.judges {
            if plan.input_encoding() != self.input_encoding()
                || plan.input_encoding_work().is_some() != self.input_encoding_work().is_some() {
                return Err(Int4SourceSemanticError::Accounting);
            }
            encoding_work = encoding_work.checked_add(plan.input_encoding_work().unwrap_or_default())
                .ok_or(Int4SourceSemanticError::Accounting)?;
        }
        if semantic_model_work != review.work || encoding_work != review.encoding_work {
            return Err(Int4SourceSemanticError::Accounting);
        }
        let reserved_model_work = reserve_combined(source_reserved_work, semantic_model_work, self.max_model_work())?;
        let model_work = plus(review.source_task.model_work, semantic_model_work)?;
        if !fits(model_work, reserved_model_work) { return Err(Int4SourceSemanticError::Accounting); }
        let attempted_statements = review.judges.len();
        let decided_statements = statements.iter().filter(|s| matches!(s.status, SourceSemanticStatus::Entailed
            | SourceSemanticStatus::Contradicted | SourceSemanticStatus::Unsupported)).count();
        let abstained_statements = statements.iter().filter(|s| s.status == SourceSemanticStatus::JudgeAbstained).count();
        let not_checked_statements = statements.iter().filter(|s| s.status == SourceSemanticStatus::NotChecked).count();
        if attempted_statements != decided_statements + abstained_statements
            || statements.len() != attempted_statements + not_checked_statements {
            return Err(Int4SourceSemanticError::Accounting);
        }
        let input_encoding = Int4SemanticEncoding::from_work(self.input_encoding(), review.encoding_work)
            .map_err(|_| Int4SourceSemanticError::Accounting)?;
        let run = Int4SourceSemanticRun { schema_version: 1, execution: INT4_SOURCE_SEMANTIC_EXECUTION,
            numerics_profile: STRICT_INT4_PROFILE, group_width, source_reserved_work, semantic_model_work,
            reserved_model_work, model_work, judge_head_count, rewound_positions, source_task: review.source_task,
            semantic: SourceSemanticReceipt { schema_version: 1, algorithm: SOURCE_SEMANTIC_VERSION,
                claim_rendering: SOURCE_SEMANTIC_CLAIM_VERSION, evidence_scope: SOURCE_SEMANTIC_SCOPE,
                judge_algorithm: FAITHFULNESS_VERSION, calibration: SOURCE_SEMANTIC_CALIBRATION,
                same_model_correlated: true, policy_digest: self.policy_digest, policy: self.options.judge_policy,
                evidence_layout: self.evidence.layout, attempted_statements, decided_statements,
                abstained_statements, not_checked_statements, statements, work: scoring },
            untrusted_fields: ["source_task.result", "semantic.evidence_layout", "semantic.statements"],
            input_encoding };
        poll(control)?;
        check_size(&run, cap)?;
        poll(control)?;
        Ok(run)
    }
}

#[allow(clippy::too_many_arguments)]
fn check_configuration(source: &PreparedInt4SourceTask, request: &SourceTaskRequest,
    judge_identity: &ExecutionIdentity, width: usize, ceiling: TaskBudget, limits: JudgeLimits,
    options: &Int4SourceSemanticOptions) -> Result<(), Int4SourceSemanticError> {
    options.input_facts(request)?;
    source.require_full_projection()?;
    let digest = Sha256Digest::of_bytes(&canonjson::canonical_bytes(request).map_err(|_| Int4SourceSemanticError::Serialization)?);
    if digest != source.request_digest() || request.budget() != source.task_budget()
        || request.task().spec().identity() != source.execution_identity().task_spec {
        return Err(Int4SourceSemanticError::InputMismatch);
    }
    for identity in [source.execution_identity(), judge_identity] {
        constrained_int4::check_profile(identity, width).map_err(|_| Int4SourceSemanticError::Contract)?;
    }
    if source.group_width() != width || judge_identity.task_spec != "judge-v1" {
        return Err(Int4SourceSemanticError::Contract);
    }
    same_model(source.execution_identity(), judge_identity)?;
    let b = options.judge_budget;
    if b.max_input_tokens > ceiling.max_input_tokens || b.max_output_tokens > ceiling.max_output_tokens
        || b.max_output_bytes > ceiling.max_output_bytes || b.max_grammar_states > ceiling.max_grammar_states
        || b.max_kv_bytes > ceiling.max_kv_bytes || limits.max_total_prompt_tokens == 0
        || limits.max_total_candidate_tokens == 0 || limits.max_total_projected_logits == 0
        || limits.max_output_bytes == 0 || limits.per_head.max_candidates == 0
        || limits.per_head.max_total_tokens == 0 || limits.per_head.max_nodes == 0
        || limits.per_head.max_depth == 0 || limits.per_head.max_candidate_id_bytes == 0
        || limits.per_head.max_projected_logits == 0 {
        return Err(Int4SourceSemanticError::Contract);
    }
    reserve_combined(source.planned_work(), Int4Work::default(), options.max_model_work)?;
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SourceSemanticTarget { SummaryBullet { index: usize }, Answer, EmptySummary }
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceSemanticStatus { Entailed, Contradicted, Unsupported, JudgeAbstained, NotChecked }
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceSemanticNoCheck { AnswerAbstained, EmptySummary }
#[derive(Serialize)]
pub struct SourceSemanticStatement {
    pub target: SourceSemanticTarget,
    pub claim: Option<String>,
    pub status: SourceSemanticStatus,
    pub not_checked: Option<SourceSemanticNoCheck>,
    pub judge: Option<FaithfulnessResult>,
}
#[derive(Serialize)]
pub struct SourceSemanticReceipt {
    pub schema_version: u32,
    pub algorithm: &'static str,
    pub claim_rendering: &'static str,
    pub evidence_scope: &'static str,
    pub judge_algorithm: &'static str,
    pub calibration: &'static str,
    pub same_model_correlated: bool,
    /// Binds rendering/scope, explicit options and all finite ceilings; it is
    /// not a fingerprint of the private document or question.
    pub policy_digest: Sha256Digest,
    pub policy: FaithfulnessPolicy,
    pub evidence_layout: SourceSemanticEvidenceLayout,
    pub attempted_statements: usize,
    pub decided_statements: usize,
    pub abstained_statements: usize,
    pub not_checked_statements: usize,
    pub statements: Vec<SourceSemanticStatement>,
    pub work: ScoringWork,
}
#[derive(Serialize)]
pub struct Int4SourceSemanticRun {
    pub schema_version: u32,
    pub execution: &'static str,
    pub numerics_profile: &'static str,
    pub group_width: usize,
    pub source_reserved_work: Int4Work,
    pub semantic_model_work: Int4Work,
    pub reserved_model_work: Int4Work,
    pub model_work: Int4Work,
    pub judge_head_count: usize,
    pub rewound_positions: u64,
    /// Original generated tokens, text, source citations and not_assessed
    /// semantic_support are preserved; review never repairs or promotes them.
    pub source_task: Int4SourceTaskRun,
    pub semantic: SourceSemanticReceipt,
    pub untrusted_fields: [&'static str; 3],
    /// Compact policy and complete original-plus-review encoding work.
    /// Omitted for byte input to preserve the existing result shape.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_encoding: Option<Int4SemanticEncoding>,
}
struct Statement { target: SourceSemanticTarget, claim: Option<String>, not_checked: Option<SourceSemanticNoCheck> }
struct Review {
    source_task: Int4SourceTaskRun, statements: Vec<Statement>, judges: Vec<PreparedInt4Judge>,
    work: Int4Work, encoding_work: SafeBpeWork,
}

fn prepare_statements<C: DecodeStepControl>(prepared: &PreparedInt4SourceTask, evidence: &EvidenceContext,
    run: &Int4SourceTaskRun, limits: SourceSemanticLimits, control: &mut C)
    -> Result<Vec<Statement>, Int4SourceSemanticError> {
    poll(control)?;
    if run.schema_version != 1 || run.execution != prepared.execution_version() || run.group_width != prepared.group_width()
        || !fits(run.model_work, prepared.planned_work()) {
        return Err(Int4SourceSemanticError::Accounting);
    }
    let mut statements = vector(limits.max_statements)?;
    match &run.result {
        SourceTaskResult::Summarize(result) => {
            if prepared.execution_identity().task_spec != SUMMARIZE_TASK_VERSION || evidence.question.is_some()
                || result.schema_version != 1 || result.task_spec_version != SUMMARIZE_TASK_VERSION
                || result.numerics_profile != STRICT_INT4_PROFILE || result.score_space != ScoreSpace::NotComputed
                || result.semantic_support != SummarySemanticSupport::NotAssessed
                || result.citation_guarantee != CitationGuarantee::StructuralSourceMembership
                || result.bullets.len() > limits.max_statements
                || result.forward_positions != run.model_work.forward_positions
                || result.projected_logits != run.model_work.projected_logits {
                return Err(Int4SourceSemanticError::Accounting);
            }
            prepared.verify_completion_work(&result.generated_token_ids, run.model_work)?;
            for (index, bullet) in result.bullets.iter().enumerate() {
                poll(control)?;
                if bullet.text.trim().is_empty() || bullet.citations.is_empty() { return Err(Int4SourceSemanticError::Accounting); }
                statements.push(Statement { target: SourceSemanticTarget::SummaryBullet { index },
                    claim: Some(copy(&bullet.text)?), not_checked: None });
            }
            if statements.is_empty() {
                statements.push(Statement { target: SourceSemanticTarget::EmptySummary,
                    claim: None, not_checked: Some(SourceSemanticNoCheck::EmptySummary) });
            }
        }
        SourceTaskResult::Answer(result) => {
            let question = evidence.question.as_deref().ok_or(Int4SourceSemanticError::Accounting)?;
            if prepared.execution_identity().task_spec != ANSWER_TASK_VERSION
                || result.schema_version != 1 || result.task_spec_version != ANSWER_TASK_VERSION
                || result.numerics_profile != STRICT_INT4_PROFILE || result.score_space != ScoreSpace::NotComputed
                || result.semantic_support != SummarySemanticSupport::NotAssessed
                || result.citation_guarantee != CitationGuarantee::StructuralSourceMembership
                || result.forward_positions != run.model_work.forward_positions
                || result.projected_logits != run.model_work.projected_logits {
                return Err(Int4SourceSemanticError::Accounting);
            }
            prepared.verify_completion_work(&result.generated_token_ids, run.model_work)?;
            let (claim, not_checked) = if result.answerable {
                let answer = result.answer.as_deref().filter(|a| !a.trim().is_empty()).ok_or(Int4SourceSemanticError::Accounting)?;
                if result.status != AnswerStatus::Answered || result.citations.is_empty() { return Err(Int4SourceSemanticError::Accounting); }
                (Some(answer_claim(question, answer, limits.max_claim_bytes)?), None)
            } else {
                if result.status != AnswerStatus::Abstained || result.answer.is_some() || !result.citations.is_empty() {
                    return Err(Int4SourceSemanticError::Accounting);
                }
                (None, Some(SourceSemanticNoCheck::AnswerAbstained))
            };
            statements.push(Statement { target: SourceSemanticTarget::Answer, claim, not_checked });
        }
        _ => return Err(Int4SourceSemanticError::Accounting),
    }
    let mut bytes = 0_usize;
    for statement in &statements {
        poll(control)?;
        if let Some(claim) = &statement.claim {
            bytes = bytes.checked_add(claim.len()).ok_or(Int4SourceSemanticError::ClaimBudget)?;
            if claim.len() > limits.max_claim_bytes || bytes > limits.max_total_claim_bytes {
                return Err(Int4SourceSemanticError::ClaimBudget);
            }
        }
    }
    Ok(statements)
}

fn answer_claim(question: &str, answer: &str, cap: usize) -> Result<String, Int4SourceSemanticError> {
    // Canonical escaping keeps question and answer distinct even when their
    // text contains delimiters. Both remain untrusted CLAIM data; only the
    // original passages enter the separate evidence segment of the judge.
    #[derive(Serialize)]
    struct Claim<'a> { question: &'a str, answer: &'a str }
    const PREFIX: &str = "The proposed answer to this question is supported by the source: ";
    let body = Claim { question, answer };
    check_size(&body, cap.checked_sub(PREFIX.len()).ok_or(Int4SourceSemanticError::ClaimBudget)? as u64)?;
    let json = canonjson::canonical_string(&body).map_err(|_| Int4SourceSemanticError::Serialization)?;
    let bytes = PREFIX.len().checked_add(json.len()).filter(|&n| n <= cap).ok_or(Int4SourceSemanticError::ClaimBudget)?;
    let mut claim = String::new(); claim.try_reserve_exact(bytes).map_err(|_| Int4SourceSemanticError::Allocation)?;
    claim.push_str(PREFIX); claim.push_str(&json); Ok(claim)
}

type FinishedStatements = (Vec<SourceSemanticStatement>, Int4Work, ScoringWork, usize, u64);
fn finish_statements<C: DecodeStepControl>(statements: &[Statement], plans: &[PreparedInt4Judge],
    runs: Vec<Int4JudgeRun>, policy: FaithfulnessPolicy, width: usize, control: &mut C)
    -> Result<FinishedStatements, Int4SourceSemanticError> {
    poll(control)?;
    let count = statements.iter().filter(|s| s.claim.is_some()).count();
    if runs.len() != plans.len() || plans.len() != count { return Err(Int4SourceSemanticError::Accounting); }
    let mut work = Int4Work::default(); let mut scoring = ScoringWork { prefix_evaluations: 0, scored_edges: 0, projected_logits: 0 };
    let (mut heads, mut rewound) = (0_usize, 0_u64);
    let mut judged = vector(count)?;
    for (plan, run) in plans.iter().zip(runs) {
        poll(control)?;
        if run.schema_version != 1 || run.execution != INT4_JUDGE_EXECUTION || run.numerics_profile != STRICT_INT4_PROFILE
            || run.group_width != width || run.model_work != plan.planned_work() || run.head_count != plan.head_count() {
            return Err(Int4SourceSemanticError::Accounting);
        }
        let JudgeResult::Faithfulness(result) = run.result else { return Err(Int4SourceSemanticError::Accounting); };
        if result.schema_version != 1 || result.task_spec != "judge-v1" || result.algorithm != FAITHFULNESS_VERSION
            || result.policy != policy || result.work.projected_logits != run.model_work.projected_logits
            || result.relation.is_some() == result.abstention.is_some() {
            return Err(Int4SourceSemanticError::Accounting);
        }
        work = plus(work, run.model_work)?;
        scoring.prefix_evaluations = scoring.prefix_evaluations.checked_add(result.work.prefix_evaluations)
            .ok_or(Int4SourceSemanticError::Accounting)?;
        scoring.scored_edges = scoring.scored_edges.checked_add(result.work.scored_edges).ok_or(Int4SourceSemanticError::Accounting)?;
        scoring.projected_logits = scoring.projected_logits.checked_add(result.work.projected_logits).ok_or(Int4SourceSemanticError::Accounting)?;
        heads = heads.checked_add(run.head_count).ok_or(Int4SourceSemanticError::Accounting)?;
        rewound = rewound.checked_add(run.rewound_positions).ok_or(Int4SourceSemanticError::Accounting)?;
        judged.push(result);
    }
    let mut judged = judged.into_iter(); let mut results = vector(statements.len())?;
    for statement in statements {
        poll(control)?;
        let judge = if statement.claim.is_some() { Some(judged.next().ok_or(Int4SourceSemanticError::Accounting)?) } else { None };
        let status = match judge.as_ref().and_then(|r| r.relation) {
            Some(FaithfulnessRelation::Entailed) => SourceSemanticStatus::Entailed,
            Some(FaithfulnessRelation::Contradicted) => SourceSemanticStatus::Contradicted,
            Some(FaithfulnessRelation::Unsupported) => SourceSemanticStatus::Unsupported,
            None if judge.is_some() => SourceSemanticStatus::JudgeAbstained,
            None => SourceSemanticStatus::NotChecked,
        };
        if (statement.not_checked.is_some()) != (judge.is_none()) { return Err(Int4SourceSemanticError::Accounting); }
        results.push(SourceSemanticStatement { target: statement.target.clone(), claim: statement.claim.as_deref().map(copy).transpose()?,
            status, not_checked: statement.not_checked, judge });
    }
    if judged.next().is_some() { return Err(Int4SourceSemanticError::Accounting); }
    poll(control)?;
    Ok((results, work, scoring, heads, rewound))
}
fn same_model(a: &ExecutionIdentity, b: &ExecutionIdentity) -> Result<(), Int4SourceSemanticError> {
    let mut normalized = a.clone();
    normalized.task_spec = b.task_spec.clone(); normalized.template_digest = b.template_digest;
    normalized.taskir_digest = b.taskir_digest; normalized.prompt_digest = b.prompt_digest;
    normalized.schema_digest = b.schema_digest; normalized.grammar_compiler_version = b.grammar_compiler_version.clone();
    normalized.sampler_version = b.sampler_version.clone(); normalized.calibration_digest = b.calibration_digest;
    normalized.decision_policy_digest = b.decision_policy_digest;
    if &normalized != b { return Err(Int4SourceSemanticError::Contract); }
    Ok(())
}
fn poll(control: &mut impl DecodeStepControl) -> Result<(), Int4SourceSemanticError> {
    match control.prefill_checkpoint(0) { Some(c) => Err(Int4SourceSemanticError::Cancelled(c)), None => Ok(()) }
}
fn fits(a: Int4Work, b: Int4Work) -> bool {
    a.forward_positions <= b.forward_positions && a.projected_logits <= b.projected_logits
        && a.attention_pairs <= b.attention_pairs && a.projections.fits(b.projections)
}
fn plus(a: Int4Work, b: Int4Work) -> Result<Int4Work, Int4SourceSemanticError> {
    a.checked_add(b).map_err(|_| Int4SourceSemanticError::WorkBudget)
}
fn reserve_combined(source: Int4Work, review: Int4Work, cap: Int4Work) -> Result<Int4Work, Int4SourceSemanticError> {
    let combined = plus(source, review)?;
    if !fits(combined, cap) { return Err(Int4SourceSemanticError::WorkBudget); }
    Ok(combined)
}
fn vector<T>(size: usize) -> Result<Vec<T>, Int4SourceSemanticError> {
    let mut out = Vec::new(); out.try_reserve_exact(size).map_err(|_| Int4SourceSemanticError::Allocation)?; Ok(out)
}
fn copy(text: &str) -> Result<String, Int4SourceSemanticError> {
    let mut out = String::new(); out.try_reserve_exact(text.len()).map_err(|_| Int4SourceSemanticError::Allocation)?;
    out.push_str(text); Ok(out)
}

#[cfg(test)] mod tests;
