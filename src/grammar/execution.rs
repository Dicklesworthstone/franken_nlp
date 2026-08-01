//! Exact constrained-execution planning over compiled grammar states.
//!
//! The grammar compiler owns the typed JSON state machine and the vocabulary
//! oracle owns tokenizer-derived legality.  This module joins those two
//! products only after every sampler processor has produced the final legal
//! mask.  It deliberately chooses between equivalent work plans; it never
//! approximates a legal set or turns diagnostic signals into an acceptance
//! decision.

use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

use super::{compiler::CompiledSchema, mask::DenseTokenMask};

/// The measured crossover used to choose sparse projection.
///
/// `exclusive_legal_row_count` is deliberately a strict upper bound: a state
/// with exactly that many legal rows remains on the universal full-projection
/// path.  The measurement identity is carried into the plan log so a future
/// tuning result cannot silently change execution semantics.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SparseProjectionThreshold {
    exclusive_legal_row_count: usize,
    measurement_id: String,
}

impl SparseProjectionThreshold {
    /// Construct one named, measured sparse-projection crossover.
    pub fn new(
        exclusive_legal_row_count: usize,
        measurement_id: impl Into<String>,
    ) -> Result<Self, ExecutionCompileError> {
        let measurement_id = measurement_id.into();
        if measurement_id.trim().is_empty() {
            return Err(ExecutionCompileError::EmptyMeasurementIdentity);
        }
        Ok(Self {
            exclusive_legal_row_count,
            measurement_id,
        })
    }

    /// The strict number of legal rows below which sparse projection is used.
    #[must_use]
    pub const fn exclusive_legal_row_count(&self) -> usize {
        self.exclusive_legal_row_count
    }

    /// Immutable identity of the measurement that supplied this threshold.
    #[must_use]
    pub fn measurement_id(&self) -> &str {
        &self.measurement_id
    }

    fn selects_sparse(&self, legal_row_count: usize) -> bool {
        legal_row_count < self.exclusive_legal_row_count
    }
}

/// Whether a configured EOS id is legal at this product state.
///
/// The `ExplicitAccepting` variant is intentionally the only way an EOS id
/// may enter a payload mask.  Template controls are separate and never become
/// payload ids merely because an EOS policy is configured.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EosTransition {
    /// The tokenizer/model has no configured EOS for this task surface.
    NotConfigured,
    /// A configured EOS is forbidden at this non-accepting state.
    Forbidden { token_id: u32 },
    /// The task explicitly permits this EOS as an accepting transition.
    ExplicitAccepting { token_id: u32 },
}

/// The token-alphabet proof carried by each product state.
///
/// `template_control_ids` are all role/think/tool markers that trusted
/// template code may emit.  They must be absent from every untrusted payload
/// mask.  EOS is represented by [`EosTransition`] rather than being folded
/// into this set, because an accepting EOS transition is not payload text.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PayloadTokenAlphabet {
    template_control_ids: BTreeSet<u32>,
    eos_transition: EosTransition,
}

impl PayloadTokenAlphabet {
    /// Build an alphabet proof for one product state.
    #[must_use]
    pub fn new(
        template_control_ids: impl IntoIterator<Item = u32>,
        eos_transition: EosTransition,
    ) -> Self {
        Self {
            template_control_ids: template_control_ids.into_iter().collect(),
            eos_transition,
        }
    }

    /// The trusted-template controls excluded from payload legality.
    #[must_use]
    pub fn template_control_ids(&self) -> &BTreeSet<u32> {
        &self.template_control_ids
    }

    /// The EOS policy for this exact product state.
    #[must_use]
    pub const fn eos_transition(&self) -> EosTransition {
        self.eos_transition
    }

    fn validate(
        &self,
        state_id: usize,
        legal_tokens: &DenseTokenMask,
    ) -> Result<(), ExecutionCompileError> {
        for &token_id in &self.template_control_ids {
            if legal_tokens.contains(token_id) {
                return Err(ExecutionCompileError::TemplateControlRepresentable {
                    state_id,
                    token_id,
                });
            }
        }
        match self.eos_transition {
            EosTransition::NotConfigured => Ok(()),
            EosTransition::Forbidden { token_id } if legal_tokens.contains(token_id) => {
                Err(ExecutionCompileError::ForbiddenEosLegal { state_id, token_id })
            }
            EosTransition::ExplicitAccepting { token_id } if !legal_tokens.contains(token_id) => {
                Err(ExecutionCompileError::MissingExplicitEosTransition { state_id, token_id })
            }
            EosTransition::Forbidden { .. } | EosTransition::ExplicitAccepting { .. } => Ok(()),
        }
    }
}

/// Telemetry that needs full-vocabulary projection rather than a forced skip.
///
/// These fields are diagnostics only.  They do not carry an acceptance method
/// and must never be used as a fabricated-confidence or escalation signal.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ProjectionTelemetry {
    pub logprobs: bool,
    pub pre_mask_argmax: bool,
    pub legal_mass: bool,
    pub legal_illegal_margin: bool,
}

impl ProjectionTelemetry {
    fn requires_projection(self) -> bool {
        self.logprobs || self.pre_mask_argmax || self.legal_mass || self.legal_illegal_margin
    }
}

/// A non-core sampler processor supplied by the task/sampler boundary.
///
/// An unsupported processor never disappears from a forced-token proof.  It
/// disables the optimization and leaves the universal projection path intact.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AdditionalProcessor {
    /// A processor whose resulting legal set is known exactly.
    Supported {
        name: String,
        legal_tokens: DenseTokenMask,
    },
    /// A processor the execution compiler cannot model for forcing.
    Unsupported { name: String },
}

impl AdditionalProcessor {
    fn name(&self) -> &str {
        match self {
            Self::Supported { name, .. } | Self::Unsupported { name } => name,
        }
    }
}

/// The complete processor intersection required before a projection skip.
///
/// All named masks are legality after that processor has run.  Their
/// intersection therefore remains independent of the model's unknown raw
/// logits.  The caller's universal mask must be identical to the resulting
/// witness mask before a forced primitive can be selected.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ForcedWitnessRequest {
    pub grammar_tokenizer_legal: DenseTokenMask,
    pub eos_min_length_stop_legal: DenseTokenMask,
    pub banned_token_legal: DenseTokenMask,
    pub repetition_presence_frequency_legal: DenseTokenMask,
    pub logit_bias_legal: DenseTokenMask,
    pub tool_thinking_phase_legal: DenseTokenMask,
    pub other_processors: Vec<AdditionalProcessor>,
    pub telemetry: ProjectionTelemetry,
    pub payload_alphabet: PayloadTokenAlphabet,
}

/// One named processor fact retained in a forced-token witness.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProcessorStateFact {
    processor: String,
    legal_token_count: usize,
}

impl ProcessorStateFact {
    /// Stable processor-state name included in the proof witness.
    #[must_use]
    pub fn processor(&self) -> &str {
        &self.processor
    }

    /// Count of legal ids after this processor's legality operation.
    #[must_use]
    pub const fn legal_token_count(&self) -> usize {
        self.legal_token_count
    }
}

/// A proof that one token id is the full processor intersection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ForcedTokenWitness {
    token_id: u32,
    effective_legal_tokens: DenseTokenMask,
    processor_facts: Vec<ProcessorStateFact>,
}

impl ForcedTokenWitness {
    /// The only legal token id independently of model logits.
    #[must_use]
    pub const fn token_id(&self) -> u32 {
        self.token_id
    }

    /// The final legal mask proven by the named processor intersection.
    #[must_use]
    pub const fn effective_legal_tokens(&self) -> &DenseTokenMask {
        &self.effective_legal_tokens
    }

    /// Named processor-state facts retained with this witness.
    #[must_use]
    pub fn processor_facts(&self) -> &[ProcessorStateFact] {
        &self.processor_facts
    }

    /// Build a complete forced-token witness or state why forcing is disabled.
    pub fn build(
        request: &ForcedWitnessRequest,
    ) -> Result<ForcedWitnessOutcome, ForcedWitnessError> {
        let vocab_size = request.grammar_tokenizer_legal.vocab_size();
        validate_mask_width(
            "eos/min-length/stop",
            vocab_size,
            &request.eos_min_length_stop_legal,
        )?;
        validate_mask_width("banned-token", vocab_size, &request.banned_token_legal)?;
        validate_mask_width(
            "repetition/presence/frequency",
            vocab_size,
            &request.repetition_presence_frequency_legal,
        )?;
        validate_mask_width("logit-bias", vocab_size, &request.logit_bias_legal)?;
        validate_mask_width(
            "tool/thinking-phase",
            vocab_size,
            &request.tool_thinking_phase_legal,
        )?;

        if request.telemetry.requires_projection() {
            return Ok(ForcedWitnessOutcome::Disabled(
                ForcedDisableReason::ProjectionTelemetryRequested,
            ));
        }

        let mut additional_names = BTreeSet::new();
        for processor in &request.other_processors {
            if processor.name().trim().is_empty() {
                return Err(ForcedWitnessError::EmptyAdditionalProcessorName);
            }
            if !additional_names.insert(processor.name().to_owned()) {
                return Err(ForcedWitnessError::DuplicateAdditionalProcessor {
                    name: processor.name().to_owned(),
                });
            }
            match processor {
                AdditionalProcessor::Supported { legal_tokens, .. } => {
                    validate_mask_width(processor.name(), vocab_size, legal_tokens)?;
                }
                AdditionalProcessor::Unsupported { name } => {
                    return Ok(ForcedWitnessOutcome::Disabled(
                        ForcedDisableReason::UnsupportedProcessor { name: name.clone() },
                    ));
                }
            }
        }

        request
            .payload_alphabet
            .validate(0, &request.grammar_tokenizer_legal)
            .map_err(ForcedWitnessError::PayloadAlphabet)?;

        let mut masks = vec![
            ("grammar/tokenizer", &request.grammar_tokenizer_legal),
            ("eos/min-length/stop", &request.eos_min_length_stop_legal),
            ("banned-token", &request.banned_token_legal),
            (
                "repetition/presence/frequency",
                &request.repetition_presence_frequency_legal,
            ),
            ("logit-bias", &request.logit_bias_legal),
            ("tool/thinking-phase", &request.tool_thinking_phase_legal),
        ];
        for processor in &request.other_processors {
            let AdditionalProcessor::Supported { name, legal_tokens } = processor else {
                unreachable!("unsupported processors returned before mask intersection")
            };
            masks.push((name, legal_tokens));
        }

        let mut effective_legal_tokens = DenseTokenMask::empty(vocab_size);
        for token_id in request.grammar_tokenizer_legal.legal_ids() {
            if masks.iter().all(|(_, mask)| mask.contains(token_id)) {
                effective_legal_tokens
                    .set_legal(token_id)
                    .expect("token id originated from a checked vocabulary mask");
            }
        }
        let legal_ids: Vec<_> = effective_legal_tokens.legal_ids().collect();
        if legal_ids.len() != 1 {
            return Ok(ForcedWitnessOutcome::Disabled(
                ForcedDisableReason::NotExactlyOneLegalToken {
                    legal_token_count: legal_ids.len(),
                },
            ));
        }
        let processor_facts = masks
            .iter()
            .map(|(processor, mask)| ProcessorStateFact {
                processor: (*processor).to_owned(),
                legal_token_count: mask.legal_ids().count(),
            })
            .collect();
        Ok(ForcedWitnessOutcome::Eligible(Self {
            token_id: legal_ids[0],
            effective_legal_tokens,
            processor_facts,
        }))
    }
}

fn validate_mask_width(
    processor: &str,
    expected_vocab_size: usize,
    mask: &DenseTokenMask,
) -> Result<(), ForcedWitnessError> {
    if mask.vocab_size() != expected_vocab_size {
        return Err(ForcedWitnessError::VocabSizeMismatch {
            processor: processor.to_owned(),
            expected_vocab_size,
            actual_vocab_size: mask.vocab_size(),
        });
    }
    Ok(())
}

/// Result of attempting to construct a forced-token witness.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ForcedWitnessOutcome {
    /// Projection can be skipped for exactly this token id.
    Eligible(ForcedTokenWitness),
    /// Projection stays enabled and this explains why.
    Disabled(ForcedDisableReason),
}

/// A fail-closed reason why `FeedForced` was not selected.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ForcedDisableReason {
    ProjectionTelemetryRequested,
    UnsupportedProcessor { name: String },
    NotExactlyOneLegalToken { legal_token_count: usize },
    WitnessDoesNotMatchUniversalMask,
    MicroPrefillNotVerified,
}

impl ForcedDisableReason {
    fn requires_full_projection(&self) -> bool {
        matches!(self, Self::ProjectionTelemetryRequested)
    }
}

/// Errors in the witness request itself, before any forced optimization.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ForcedWitnessError {
    VocabSizeMismatch {
        processor: String,
        expected_vocab_size: usize,
        actual_vocab_size: usize,
    },
    EmptyAdditionalProcessorName,
    DuplicateAdditionalProcessor {
        name: String,
    },
    PayloadAlphabet(ExecutionCompileError),
}

impl fmt::Display for ForcedWitnessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::VocabSizeMismatch {
                processor,
                expected_vocab_size,
                actual_vocab_size,
            } => write!(
                f,
                "{processor} mask vocabulary width {actual_vocab_size} does not match {expected_vocab_size}"
            ),
            Self::EmptyAdditionalProcessorName => {
                f.write_str("additional processor name must not be empty")
            }
            Self::DuplicateAdditionalProcessor { name } => {
                write!(
                    f,
                    "additional processor {name:?} was supplied more than once"
                )
            }
            Self::PayloadAlphabet(error) => error.fmt(f),
        }
    }
}

impl Error for ForcedWitnessError {}

/// Model-gated evidence required before a multi-token micro-prefill is used.
///
/// This value only transports the exact comparison receipt.  Constructing it
/// does not promote the receipt or make a parity claim; central evidence
/// review owns that authority.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvPointEqualityEvidence {
    profile: String,
    receipt_id: String,
    compared_token_count: usize,
}

impl KvPointEqualityEvidence {
    /// Required Nanbeige cache depth: 22 layers across two decoder loops.
    pub const REQUIRED_KV_SLOT_COUNT: usize = 44;

    /// Retain a named model-gated receipt only when it reports byte-exact
    /// equality at every required KV slot for at least one teacher-fed token.
    pub fn exact_44_slot(
        profile: impl Into<String>,
        receipt_id: impl Into<String>,
        compared_token_count: usize,
        compared_kv_slot_count: usize,
        every_kv_point_byte_exact: bool,
    ) -> Result<Self, ExecutionCompileError> {
        let profile = profile.into();
        let receipt_id = receipt_id.into();
        if profile.trim().is_empty() || receipt_id.trim().is_empty() || compared_token_count == 0 {
            return Err(ExecutionCompileError::EmptyKvEqualityEvidence);
        }
        if compared_kv_slot_count != Self::REQUIRED_KV_SLOT_COUNT || !every_kv_point_byte_exact {
            return Err(ExecutionCompileError::IncompleteKvEqualityEvidence {
                compared_kv_slot_count,
                every_kv_point_byte_exact,
            });
        }
        Ok(Self {
            profile,
            receipt_id,
            compared_token_count,
        })
    }

    /// Numerics profile under which every KV point was compared.
    #[must_use]
    pub fn profile(&self) -> &str {
        &self.profile
    }

    /// Immutable receipt identifier retained with the execution plan.
    #[must_use]
    pub fn receipt_id(&self) -> &str {
        &self.receipt_id
    }

    /// Number of teacher-fed token positions covered by the retained receipt.
    #[must_use]
    pub const fn compared_token_count(&self) -> usize {
        self.compared_token_count
    }
}

/// The legal execution strategy for a forced token run.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ForcedFeedStrategy {
    /// Feed each token in order, retaining the universal exact fallback.
    Sequential,
    /// Teacher-feed one bounded run only after a 44-slot KV equality receipt.
    MicroPrefill { evidence: KvPointEqualityEvidence },
}

/// A bounded run whose every token has an independently complete witness.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ForcedRun {
    tokens: Vec<u32>,
    witnesses: Vec<ForcedTokenWitness>,
    requested_micro_prefill: Option<KvPointEqualityEvidence>,
}

/// A forcing request presented to the execution compiler.
///
/// The disabled form is retained rather than discarded so the compiler's
/// stage-line record identifies why it selected the universal fallback.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ForcedPath {
    /// Every token has a complete witness and may be fed exactly.
    Proven(ForcedRun),
    /// Witness construction declined the optimization; projection remains on.
    Disabled(ForcedDisableReason),
}

impl ForcedRun {
    /// Conservative default cap for a single teacher-fed run.
    ///
    /// Callers with a narrower request budget should use [`Self::new_bounded`]
    /// and retain that cap in their request receipt.  This bound never permits
    /// skipped transformer or KV work.
    pub const DEFAULT_MAX_TOKENS: usize = 64;

    /// Build one forced run.  A unique byte suffix is intentionally not an
    /// input to this constructor: only exact token ids can enter the path.
    pub fn new(
        tokens: Vec<u32>,
        witnesses: Vec<ForcedTokenWitness>,
        requested_micro_prefill: Option<KvPointEqualityEvidence>,
    ) -> Result<Self, ExecutionCompileError> {
        Self::new_bounded(
            tokens,
            witnesses,
            requested_micro_prefill,
            Self::DEFAULT_MAX_TOKENS,
        )
    }

    /// Build one forced run under the caller's explicit token cap.
    pub fn new_bounded(
        tokens: Vec<u32>,
        witnesses: Vec<ForcedTokenWitness>,
        requested_micro_prefill: Option<KvPointEqualityEvidence>,
        max_tokens: usize,
    ) -> Result<Self, ExecutionCompileError> {
        if max_tokens == 0 {
            return Err(ExecutionCompileError::ZeroForcedRunLimit);
        }
        if tokens.is_empty() {
            return Err(ExecutionCompileError::EmptyForcedRun);
        }
        if tokens.len() > max_tokens {
            return Err(ExecutionCompileError::ForcedRunExceedsLimit {
                token_count: tokens.len(),
                max_tokens,
            });
        }
        if tokens.len() != witnesses.len() {
            return Err(ExecutionCompileError::ForcedRunWitnessCountMismatch {
                token_count: tokens.len(),
                witness_count: witnesses.len(),
            });
        }
        for (index, (token_id, witness)) in tokens.iter().zip(&witnesses).enumerate() {
            if *token_id != witness.token_id() {
                return Err(ExecutionCompileError::ForcedRunTokenWitnessMismatch {
                    index,
                    token_id: *token_id,
                    witness_token_id: witness.token_id(),
                });
            }
        }
        Ok(Self {
            tokens,
            witnesses,
            requested_micro_prefill,
        })
    }

    /// Exact token ids that are still fed through all transformer layers.
    #[must_use]
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// One independently complete witness per token id.
    #[must_use]
    pub fn witnesses(&self) -> &[ForcedTokenWitness] {
        &self.witnesses
    }

    /// Sequentially visit forced token ids with mandatory cancellation
    /// checkpoints.  The visitor is where the caller performs the ordinary
    /// one-token transformer/KV update; this helper never treats a forced id
    /// as permission to skip model work.
    pub fn visit_sequentially<F, C>(
        &self,
        checkpoint_interval_tokens: usize,
        mut checkpoint: C,
        mut visit_token: F,
    ) -> Result<(), ForcedRunVisitError>
    where
        F: FnMut(u32),
        C: FnMut(ForcedRunProgress) -> bool,
    {
        if checkpoint_interval_tokens == 0 {
            return Err(ForcedRunVisitError::InvalidCheckpointInterval);
        }
        for (next_token_index, &token_id) in self.tokens.iter().enumerate() {
            if next_token_index % checkpoint_interval_tokens == 0
                && !checkpoint(ForcedRunProgress {
                    next_token_index,
                    total_token_count: self.tokens.len(),
                })
            {
                return Err(ForcedRunVisitError::CancelledBeforeToken { next_token_index });
            }
            visit_token(token_id);
        }
        Ok(())
    }

    fn strategy(&self) -> (ForcedFeedStrategy, Option<ForcedDisableReason>) {
        match &self.requested_micro_prefill {
            Some(evidence) if self.tokens.len() > 1 => (
                ForcedFeedStrategy::MicroPrefill {
                    evidence: evidence.clone(),
                },
                None,
            ),
            Some(_) | None if self.tokens.len() > 1 => (
                ForcedFeedStrategy::Sequential,
                Some(ForcedDisableReason::MicroPrefillNotVerified),
            ),
            Some(_) | None => (ForcedFeedStrategy::Sequential, None),
        }
    }
}

/// Source-free progress provided to a forced-run cancellation checkpoint.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ForcedRunProgress {
    /// Index of the next token whose 44-layer/KV update has not started.
    pub next_token_index: usize,
    /// Total bounded token count in this exact run.
    pub total_token_count: usize,
}

/// Forced-run traversal refuses partial successful execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ForcedRunVisitError {
    InvalidCheckpointInterval,
    CancelledBeforeToken { next_token_index: usize },
}

impl fmt::Display for ForcedRunVisitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCheckpointInterval => {
                f.write_str("forced-run checkpoint interval must be nonzero")
            }
            Self::CancelledBeforeToken { next_token_index } => write!(
                f,
                "forced run cancelled before token index {next_token_index}; no partial success was returned"
            ),
        }
    }
}

impl Error for ForcedRunVisitError {}

/// The source-product state supplied by the separately gated grounding path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SourceProductState {
    pub source_state_id: u64,
}

/// Input for exactly one compiled grammar/product state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProductExecutionState {
    pub state_id: usize,
    /// Final legal ids after every enabled sampler processor.  This is the
    /// mask consumed by both universal and sparse projection.
    pub legal_tokens: DenseTokenMask,
    pub payload_alphabet: PayloadTokenAlphabet,
    pub source_product: Option<SourceProductState>,
    pub forced_path: Option<ForcedPath>,
}

/// Full-vocabulary diagnostics returned only by a full projection scan.
#[derive(Clone, Debug, PartialEq)]
pub struct FullProjectionAudit {
    pub pre_mask_argmax: DiagnosticValue<u32>,
    pub pre_mask_argmax_is_legal: DiagnosticValue<bool>,
    pub legal_probability_mass: DiagnosticValue<f64>,
    pub best_legal_minus_best_illegal: DiagnosticValue<f32>,
}

impl FullProjectionAudit {
    fn not_requested() -> Self {
        Self {
            pre_mask_argmax: DiagnosticValue::NotRequested,
            pre_mask_argmax_is_legal: DiagnosticValue::NotRequested,
            legal_probability_mass: DiagnosticValue::NotRequested,
            best_legal_minus_best_illegal: DiagnosticValue::NotRequested,
        }
    }

    fn not_computed() -> Self {
        Self {
            pre_mask_argmax: DiagnosticValue::NotComputed,
            pre_mask_argmax_is_legal: DiagnosticValue::NotComputed,
            legal_probability_mass: DiagnosticValue::NotComputed,
            best_legal_minus_best_illegal: DiagnosticValue::NotComputed,
        }
    }
}

/// Distinguishes disabled diagnostics from sparse-path unknown values.
#[derive(Clone, Debug, PartialEq)]
pub enum DiagnosticValue<T> {
    NotRequested,
    NotComputed,
    Value(T),
}

/// Universal masked full-vocabulary projection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FullProjection {
    legal_tokens: DenseTokenMask,
    collect_audit: bool,
}

impl FullProjection {
    /// Build the universal projection primitive over an exact legal mask.
    #[must_use]
    pub fn new(legal_tokens: DenseTokenMask, collect_audit: bool) -> Self {
        Self {
            legal_tokens,
            collect_audit,
        }
    }

    /// The dense legal mask ANDed into the full sampler scan.
    #[must_use]
    pub const fn legal_tokens(&self) -> &DenseTokenMask {
        &self.legal_tokens
    }

    /// Return the legal rows observed during a full scan, in token-id order.
    ///
    /// This is the structural sparse==full differential surface.  It is not
    /// a probability, quality score, or acceptance signal.
    pub fn legal_row_logits(&self, logits: &[f32]) -> Result<Vec<LegalRowLogit>, ProjectionError> {
        validate_logits(logits, self.legal_tokens.vocab_size())?;
        Ok(self
            .legal_tokens
            .legal_ids()
            .map(|token_id| LegalRowLogit {
                token_id,
                logit: logits[usize::try_from(token_id)
                    .expect("token id originated from a checked vocabulary mask")],
            })
            .collect())
    }

    /// Evaluate all vocabulary rows while admitting only legal ids.
    pub fn select(&self, logits: &[f32]) -> Result<ProjectionSelection, ProjectionError> {
        validate_logits(logits, self.legal_tokens.vocab_size())?;
        let best_legal = select_best(
            (0..self.legal_tokens.vocab_size()).filter(|index| {
                self.legal_tokens
                    .contains(u32::try_from(*index).expect("vocabulary index fits u32"))
            }),
            logits,
        )?;
        let audit = if self.collect_audit {
            full_audit(&self.legal_tokens, logits, best_legal)?
        } else {
            FullProjectionAudit::not_requested()
        };
        Ok(ProjectionSelection {
            token_id: best_legal.0,
            logit: best_legal.1,
            audit,
        })
    }
}

/// Sparse projection over every legal lm-head row and no illegal rows.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProjectLegal {
    legal_rows: Vec<u32>,
    vocab_size: usize,
}

impl ProjectLegal {
    /// Derive the complete ordered sparse row list from the final legal mask.
    #[must_use]
    pub fn from_mask(legal_tokens: &DenseTokenMask) -> Self {
        Self {
            legal_rows: legal_tokens.legal_ids().collect(),
            vocab_size: legal_tokens.vocab_size(),
        }
    }

    /// Every evaluated lm-head row, in deterministic token-id order.
    #[must_use]
    pub fn legal_rows(&self) -> &[u32] {
        &self.legal_rows
    }

    /// Return every sparse legal row, in the same token-id order as the
    /// universal masked full scan.
    pub fn legal_row_logits(&self, logits: &[f32]) -> Result<Vec<LegalRowLogit>, ProjectionError> {
        validate_logits(logits, self.vocab_size)?;
        Ok(self
            .legal_rows
            .iter()
            .copied()
            .map(|token_id| LegalRowLogit {
                token_id,
                logit: logits[usize::try_from(token_id)
                    .expect("token id originated from a checked vocabulary mask")],
            })
            .collect())
    }

    /// Evaluate every legal row.  Illegal-logit diagnostics remain unknown.
    pub fn select(&self, logits: &[f32]) -> Result<ProjectionSelection, ProjectionError> {
        validate_logits(logits, self.vocab_size)?;
        let best_legal = select_best(
            self.legal_rows.iter().map(|token_id| {
                usize::try_from(*token_id).expect("token id originated from vocabulary width")
            }),
            logits,
        )?;
        Ok(ProjectionSelection {
            token_id: best_legal.0,
            logit: best_legal.1,
            audit: FullProjectionAudit::not_computed(),
        })
    }
}

/// A selected legal token and optional diagnostics, never an acceptance grade.
#[derive(Clone, Debug, PartialEq)]
pub struct ProjectionSelection {
    pub token_id: u32,
    pub logit: f32,
    pub audit: FullProjectionAudit,
}

/// One legal lm-head row observed through either equivalent projection path.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LegalRowLogit {
    pub token_id: u32,
    pub logit: f32,
}

fn validate_logits(logits: &[f32], expected_vocab_size: usize) -> Result<(), ProjectionError> {
    if logits.len() != expected_vocab_size {
        return Err(ProjectionError::VocabSizeMismatch {
            expected_vocab_size,
            actual_vocab_size: logits.len(),
        });
    }
    if let Some((index, _)) = logits.iter().enumerate().find(|(_, logit)| logit.is_nan()) {
        return Err(ProjectionError::NanLogit { index });
    }
    Ok(())
}

fn select_best(
    rows: impl IntoIterator<Item = usize>,
    logits: &[f32],
) -> Result<(u32, f32), ProjectionError> {
    let mut best: Option<(usize, f32)> = None;
    for index in rows {
        let logit = logits[index];
        if best.is_none_or(|(_, best_logit)| logit > best_logit) {
            best = Some((index, logit));
        }
    }
    let Some((index, logit)) = best else {
        return Err(ProjectionError::NoLegalToken);
    };
    Ok((
        u32::try_from(index).expect("configured model vocabulary fits u32"),
        logit,
    ))
}

fn full_audit(
    legal_tokens: &DenseTokenMask,
    logits: &[f32],
    best_legal: (u32, f32),
) -> Result<FullProjectionAudit, ProjectionError> {
    let best_any = select_best(0..logits.len(), logits)?;
    let best_illegal = select_best(
        (0..legal_tokens.vocab_size()).filter(|index| {
            !legal_tokens.contains(u32::try_from(*index).expect("vocabulary index fits u32"))
        }),
        logits,
    )
    .ok();
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut total = 0.0_f64;
    let mut legal = 0.0_f64;
    for (index, logit) in logits.iter().enumerate() {
        let probability_term = (f64::from(*logit) - maximum).exp();
        total += probability_term;
        if legal_tokens.contains(u32::try_from(index).expect("vocabulary index fits u32")) {
            legal += probability_term;
        }
    }
    let margin = match best_illegal {
        Some((_, illegal_logit)) => DiagnosticValue::Value(best_legal.1 - illegal_logit),
        None => DiagnosticValue::NotComputed,
    };
    Ok(FullProjectionAudit {
        pre_mask_argmax: DiagnosticValue::Value(best_any.0),
        pre_mask_argmax_is_legal: DiagnosticValue::Value(legal_tokens.contains(best_any.0)),
        legal_probability_mass: DiagnosticValue::Value(legal / total),
        best_legal_minus_best_illegal: margin,
    })
}

/// One exact primitive selected for a grammar/source product state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExecutionPrimitive {
    FullProjection(FullProjection),
    ProjectLegal(ProjectLegal),
    FeedForced {
        tokens: Vec<u32>,
        strategy: ForcedFeedStrategy,
        witnesses: Vec<ForcedTokenWitness>,
    },
    CopyFromSource(SourceProductState),
}

impl ExecutionPrimitive {
    /// Stable primitive name for stage-line and receipt logging.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::FullProjection(_) => "FullProjection",
            Self::ProjectLegal(_) => "ProjectLegal",
            Self::FeedForced { .. } => "FeedForced",
            Self::CopyFromSource(_) => "CopyFromSource",
        }
    }
}

/// Structured stage-line data for one compiled state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionPlanLog {
    pub state_id: usize,
    pub primitive: &'static str,
    pub legal_row_count: usize,
    pub sparse_threshold: SparseProjectionThreshold,
    pub forced_witnesses: Vec<ForcedTokenWitness>,
    pub fallback_reason: Option<ForcedDisableReason>,
}

/// A selected primitive together with its non-authoritative stage-line data.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledExecutionState {
    primitive: ExecutionPrimitive,
    log: ExecutionPlanLog,
}

impl CompiledExecutionState {
    /// The one primitive chosen for this product state.
    #[must_use]
    pub const fn primitive(&self) -> &ExecutionPrimitive {
        &self.primitive
    }

    /// Receipt-ready path-selection data; callers perform the actual logging.
    #[must_use]
    pub const fn log(&self) -> &ExecutionPlanLog {
        &self.log
    }
}

/// The complete execution plan for one [`CompiledSchema`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledExecutionPlan {
    states: BTreeMap<usize, CompiledExecutionState>,
}

impl CompiledExecutionPlan {
    /// Lookup the primitive for one logical compiler state.
    #[must_use]
    pub fn state(&self, state_id: usize) -> Option<&CompiledExecutionState> {
        self.states.get(&state_id)
    }

    /// Iterate every planned state in deterministic logical-state order.
    pub fn states(&self) -> impl Iterator<Item = (&usize, &CompiledExecutionState)> {
        self.states.iter()
    }
}

/// Exact primitive chooser over one bounded compiled schema.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionCompiler {
    sparse_threshold: SparseProjectionThreshold,
    collect_full_projection_audit: bool,
}

impl ExecutionCompiler {
    /// Construct a compiler that chooses only between universal and sparse
    /// paths proven equivalent by their differential gate.
    #[must_use]
    pub fn new(
        sparse_threshold: SparseProjectionThreshold,
        collect_full_projection_audit: bool,
    ) -> Self {
        Self {
            sparse_threshold,
            collect_full_projection_audit,
        }
    }

    /// Compile one primitive for every logical state in `schema`.
    ///
    /// The state vector must cover exactly `0..schema.automaton().state_count()`.
    /// This prevents an execution caller from silently omitting a grammar state
    /// and relying on an unplanned path.
    pub fn compile_schema(
        &self,
        schema: &CompiledSchema,
        states: Vec<ProductExecutionState>,
    ) -> Result<CompiledExecutionPlan, ExecutionCompileError> {
        let expected_state_count = schema.automaton().state_count();
        if states.len() != expected_state_count {
            return Err(ExecutionCompileError::StateCountMismatch {
                expected_state_count,
                actual_state_count: states.len(),
            });
        }
        let mut planned = BTreeMap::new();
        for state in states {
            if state.state_id >= expected_state_count {
                return Err(ExecutionCompileError::StateIdOutOfRange {
                    state_id: state.state_id,
                    state_count: expected_state_count,
                });
            }
            if planned.contains_key(&state.state_id) {
                return Err(ExecutionCompileError::DuplicateStateId {
                    state_id: state.state_id,
                });
            }
            let state_id = state.state_id;
            planned.insert(state_id, self.compile_state(state)?);
        }
        for state_id in 0..expected_state_count {
            if !planned.contains_key(&state_id) {
                return Err(ExecutionCompileError::MissingStateId { state_id });
            }
        }
        Ok(CompiledExecutionPlan { states: planned })
    }

    /// Compile a single state for focused tests and incremental callers.
    pub fn compile_state(
        &self,
        state: ProductExecutionState,
    ) -> Result<CompiledExecutionState, ExecutionCompileError> {
        state
            .payload_alphabet
            .validate(state.state_id, &state.legal_tokens)?;
        let legal_row_count = state.legal_tokens.legal_ids().count();
        if legal_row_count == 0 {
            return Err(ExecutionCompileError::Unsatisfiable {
                state_id: state.state_id,
            });
        }
        if state.source_product.is_some() && state.forced_path.is_some() {
            return Err(ExecutionCompileError::SourceAndForcedConflict {
                state_id: state.state_id,
            });
        }

        if let Some(source_state) = state.source_product {
            return Ok(self.compiled(
                state.state_id,
                legal_row_count,
                ExecutionPrimitive::CopyFromSource(source_state),
                Vec::new(),
                None,
            ));
        }

        if let Some(forced_path) = state.forced_path {
            match forced_path {
                ForcedPath::Disabled(reason) => {
                    return Ok(self.projection_fallback(
                        state.state_id,
                        state.legal_tokens,
                        Some(reason),
                    ));
                }
                ForcedPath::Proven(forced_run) => {
                    let first_witness = forced_run
                        .witnesses()
                        .first()
                        .expect("ForcedRun rejects an empty witness vector");
                    if first_witness.effective_legal_tokens() == &state.legal_tokens {
                        let (strategy, fallback_reason) = forced_run.strategy();
                        return Ok(self.compiled(
                            state.state_id,
                            legal_row_count,
                            ExecutionPrimitive::FeedForced {
                                tokens: forced_run.tokens().to_vec(),
                                strategy,
                                witnesses: forced_run.witnesses().to_vec(),
                            },
                            forced_run.witnesses().to_vec(),
                            fallback_reason,
                        ));
                    }
                    return Ok(self.projection_fallback(
                        state.state_id,
                        state.legal_tokens,
                        Some(ForcedDisableReason::WitnessDoesNotMatchUniversalMask),
                    ));
                }
            }
        }

        Ok(self.projection_fallback(state.state_id, state.legal_tokens, None))
    }

    fn projection_fallback(
        &self,
        state_id: usize,
        legal_tokens: DenseTokenMask,
        fallback_reason: Option<ForcedDisableReason>,
    ) -> CompiledExecutionState {
        let legal_row_count = legal_tokens.legal_ids().count();
        let requires_full_projection = fallback_reason
            .as_ref()
            .is_some_and(ForcedDisableReason::requires_full_projection);
        let primitive =
            if !requires_full_projection && self.sparse_threshold.selects_sparse(legal_row_count) {
                ExecutionPrimitive::ProjectLegal(ProjectLegal::from_mask(&legal_tokens))
            } else {
                ExecutionPrimitive::FullProjection(FullProjection::new(
                    legal_tokens,
                    self.collect_full_projection_audit,
                ))
            };
        self.compiled(
            state_id,
            legal_row_count,
            primitive,
            Vec::new(),
            fallback_reason,
        )
    }

    fn compiled(
        &self,
        state_id: usize,
        legal_row_count: usize,
        primitive: ExecutionPrimitive,
        forced_witnesses: Vec<ForcedTokenWitness>,
        fallback_reason: Option<ForcedDisableReason>,
    ) -> CompiledExecutionState {
        let log = ExecutionPlanLog {
            state_id,
            primitive: primitive.name(),
            legal_row_count,
            sparse_threshold: self.sparse_threshold.clone(),
            forced_witnesses,
            fallback_reason,
        };
        CompiledExecutionState { primitive, log }
    }
}

/// Projection-only failures.  The task layer maps these to typed no-result
/// outcomes; this compiler never emits a partial constrained selection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProjectionError {
    VocabSizeMismatch {
        expected_vocab_size: usize,
        actual_vocab_size: usize,
    },
    NanLogit {
        index: usize,
    },
    NoLegalToken,
}

impl fmt::Display for ProjectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::VocabSizeMismatch {
                expected_vocab_size,
                actual_vocab_size,
            } => write!(
                f,
                "logit width {actual_vocab_size} does not match vocabulary width {expected_vocab_size}"
            ),
            Self::NanLogit { index } => write!(f, "logit row {index} is NaN"),
            Self::NoLegalToken => f.write_str("constrained state has no legal token"),
        }
    }
}

impl Error for ProjectionError {}

/// Compile-time failures that preserve typed no-result rather than guessing a
/// weaker grammar/sampler route.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExecutionCompileError {
    EmptyMeasurementIdentity,
    EmptyKvEqualityEvidence,
    IncompleteKvEqualityEvidence {
        compared_kv_slot_count: usize,
        every_kv_point_byte_exact: bool,
    },
    StateCountMismatch {
        expected_state_count: usize,
        actual_state_count: usize,
    },
    StateIdOutOfRange {
        state_id: usize,
        state_count: usize,
    },
    DuplicateStateId {
        state_id: usize,
    },
    MissingStateId {
        state_id: usize,
    },
    Unsatisfiable {
        state_id: usize,
    },
    TemplateControlRepresentable {
        state_id: usize,
        token_id: u32,
    },
    ForbiddenEosLegal {
        state_id: usize,
        token_id: u32,
    },
    MissingExplicitEosTransition {
        state_id: usize,
        token_id: u32,
    },
    SourceAndForcedConflict {
        state_id: usize,
    },
    EmptyForcedRun,
    ZeroForcedRunLimit,
    ForcedRunExceedsLimit {
        token_count: usize,
        max_tokens: usize,
    },
    ForcedRunWitnessCountMismatch {
        token_count: usize,
        witness_count: usize,
    },
    ForcedRunTokenWitnessMismatch {
        index: usize,
        token_id: u32,
        witness_token_id: u32,
    },
}

impl fmt::Display for ExecutionCompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyMeasurementIdentity => {
                f.write_str("sparse projection threshold requires a measurement identity")
            }
            Self::EmptyKvEqualityEvidence => {
                f.write_str("micro-prefill evidence requires both profile and receipt id")
            }
            Self::IncompleteKvEqualityEvidence {
                compared_kv_slot_count,
                every_kv_point_byte_exact,
            } => write!(
                f,
                "micro-prefill evidence must byte-compare all {} KV slots (received {compared_kv_slot_count}, byte_exact={every_kv_point_byte_exact})",
                KvPointEqualityEvidence::REQUIRED_KV_SLOT_COUNT
            ),
            Self::StateCountMismatch {
                expected_state_count,
                actual_state_count,
            } => write!(
                f,
                "execution input has {actual_state_count} states but compiled schema has {expected_state_count}"
            ),
            Self::StateIdOutOfRange {
                state_id,
                state_count,
            } => write!(f, "state id {state_id} is outside 0..{state_count}"),
            Self::DuplicateStateId { state_id } => {
                write!(f, "state id {state_id} was supplied more than once")
            }
            Self::MissingStateId { state_id } => write!(f, "state id {state_id} was not planned"),
            Self::Unsatisfiable { state_id } => {
                write!(
                    f,
                    "state id {state_id} has no legal token and is a typed no-result"
                )
            }
            Self::TemplateControlRepresentable { state_id, token_id } => write!(
                f,
                "state id {state_id} exposes trusted template control token {token_id} as payload"
            ),
            Self::ForbiddenEosLegal { state_id, token_id } => write!(
                f,
                "non-accepting state id {state_id} exposes forbidden EOS token {token_id}"
            ),
            Self::MissingExplicitEosTransition { state_id, token_id } => write!(
                f,
                "accepting state id {state_id} lacks explicit EOS token {token_id}"
            ),
            Self::SourceAndForcedConflict { state_id } => write!(
                f,
                "state id {state_id} cannot be both CopyFromSource and FeedForced"
            ),
            Self::EmptyForcedRun => {
                f.write_str("forced run must contain at least one exact token id")
            }
            Self::ZeroForcedRunLimit => f.write_str("forced run token limit must be nonzero"),
            Self::ForcedRunExceedsLimit {
                token_count,
                max_tokens,
            } => write!(
                f,
                "forced run has {token_count} tokens and exceeds its explicit cap of {max_tokens}"
            ),
            Self::ForcedRunWitnessCountMismatch {
                token_count,
                witness_count,
            } => write!(
                f,
                "forced run has {token_count} tokens but {witness_count} witnesses"
            ),
            Self::ForcedRunTokenWitnessMismatch {
                index,
                token_id,
                witness_token_id,
            } => write!(
                f,
                "forced run token {index} is {token_id}, but its witness proves {witness_token_id}"
            ),
        }
    }
}

impl Error for ExecutionCompileError {}
