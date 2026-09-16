//! Concrete native execution of independent sentiment heads. Each exact axis
//! prompt prefills once; its candidate branches reuse the prefix session. Axis
//! prompts are NOT interchangeable, so KV is cleared between dimensions while
//! retaining the one caller-admitted engine's buffers. No model weight clone,
//! second KV reservation, loader, worker or runtime is introduced here.

use std::{error::Error, fmt};
use serde::{Deserialize, Serialize};
use crate::{
    canonjson,
    native_engine::{
        decode::{DecodeCancellationKind, DecodeStepControl},
        hf_bf16_eager::{HfBf16EagerEngine, HF_BF16_EAGER_PROFILE,
            candidate_scoring::{EagerPrefixSession, PrefixBudget, PrefixScoringError, PrefixWork}},
        kv::KV_BYTES_PER_TOKEN,
    },
};
use super::{SentimentError, SentimentPlan, SentimentResult};

pub const SENTIMENT_NATIVE_EXECUTION: &str = "eager-independent-affect-prefix-heads-v1";

/// The whole response is bounded, including native work and strategy metadata.
/// These counters describe completed algorithmic work, not measured throughput.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EagerSentimentRun {
    pub schema_version: u32,
    pub execution: String,
    pub numerics_profile: String,
    pub result: SentimentResult,
    pub native_work: PrefixWork,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SentimentNativeError {
    Task(SentimentError),
    Native(PrefixScoringError),
    EngineAlreadyPrimed,
    ContextBudget,
    KvBudget,
    WorkBudget,
    ArithmeticOverflow,
    AllocationRefused,
    ExecutionDiverged,
    Cancelled(DecodeCancellationKind),
    OutputBudget,
    Serialization,
}
impl fmt::Display for SentimentNativeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Task(error) => write!(f, "native sentiment refused: {error}"),
            Self::Native(error) => write!(f, "native sentiment failed: {error}"),
            Self::EngineAlreadyPrimed => f.write_str("native sentiment requires an empty admitted engine"),
            Self::ContextBudget => f.write_str("a sentiment axis exceeds admitted context"),
            Self::KvBudget => f.write_str("engine KV reservation exceeds a sentiment task budget"),
            Self::WorkBudget => f.write_str("sentiment exceeds aggregate native work budget"),
            Self::ArithmeticOverflow => f.write_str("sentiment native work arithmetic overflowed"),
            Self::AllocationRefused => f.write_str("sentiment native prompt allocation refused"),
            Self::ExecutionDiverged => f.write_str("sentiment native work disagrees with the complete plan"),
            Self::Cancelled(_) => f.write_str("native sentiment cancelled"),
            Self::OutputBudget => f.write_str("complete native sentiment response exceeds output budget"),
            Self::Serialization => f.write_str("native sentiment serialization failed"),
        }
    }
}
impl Error for SentimentNativeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Task(e) => Some(e), Self::Native(e) => Some(e), _ => None }
    }
}
impl From<SentimentError> for SentimentNativeError {
    fn from(error: SentimentError) -> Self { Self::Task(error) }
}
impl From<PrefixScoringError> for SentimentNativeError {
    fn from(error: PrefixScoringError) -> Self {
        match error {
            PrefixScoringError::Cancelled(cause) => Self::Cancelled(cause),
            other => Self::Native(other),
        }
    }
}

impl SentimentPlan {
    /// Execute every dimension on the existing shape-checked engine. The caller
    /// owns artifact, execution-identity and process admission; this method
    /// never grants them. A nonempty caller cache is refused without mutation.
    pub fn execute_eager(
        &self, engine: &mut HfBf16EagerEngine, budget: PrefixBudget,
    ) -> Result<EagerSentimentRun, SentimentNativeError> {
        self.execute_eager_with_control(engine, budget, &mut Continue)
    }

    /// As above, with the caller's single cancellation/deadline controller.
    /// Candidate scoring commits no generated tokens: all native polling uses
    /// the prefill channel. A failed head aborts the whole bundle, and session
    /// drop clears logical KV on normal errors, cancellation and unwinding.
    pub fn execute_eager_with_control<C: DecodeStepControl>(
        &self, engine: &mut HfBf16EagerEngine, budget: PrefixBudget, control: &mut C,
    ) -> Result<EagerSentimentRun, SentimentNativeError> {
        if !engine.kv_cache().all_slots_have_len(0) {
            return Err(SentimentNativeError::EngineAlreadyPrimed);
        }
        let capacity = engine.kv_cache().capacity_positions();
        let kv_bytes = (capacity as u64).checked_mul(KV_BYTES_PER_TOKEN as u64)
            .ok_or(SentimentNativeError::ArithmeticOverflow)?;
        let mut expected = PrefixWork::default();
        // Preflight ALL heads before any forward, not merely the first one.
        for head in &self.heads {
            let positions = head.prompt_len.checked_add(head.max_prefix)
                .ok_or(SentimentNativeError::ArithmeticOverflow)?;
            if positions > capacity { return Err(SentimentNativeError::ContextBudget); }
            if kv_bytes > head.ir.budget().max_kv_bytes { return Err(SentimentNativeError::KvBudget); }
            add_work(&mut expected, head_bound(head.prompt_len, head.prefix_count,
                head.work.projected_logits)?)?;
        }
        let mut ledger = Ledger::new(expected, budget)?;
        let mut completed = PrefixWork::default();
        let result = self.execute_heads(|head, mode| {
            if let Some(cause) = control.prefill_checkpoint(0) {
                return Err(SentimentNativeError::Cancelled(cause));
            }
            let bound = head_bound(head.prompt_len, head.prefix_count, head.work.projected_logits)?;
            // A head receives exactly its declared slice of the aggregate
            // budget, never a renewable copy of the whole bundle allowance.
            let head_budget = ledger.charge(bound)?;
            let mut prompt = Vec::new();
            prompt.try_reserve_exact(head.prompt_len).map_err(|_| SentimentNativeError::AllocationRefused)?;
            prompt.extend(head.ir.prompt_segments().iter().flat_map(|s| s.token_ids().iter().copied()));
            let mut session = EagerPrefixSession::new(
                engine, &prompt, head.max_prefix, head.ir.budget().max_kv_bytes, head_budget, control,
            )?;
            let scores = match head.scorer.score(&mut session, mode) {
                Ok(scores) => scores,
                Err(error) => return Err(match session.last_error() {
                    Some(native) => native.clone().into(),
                    None => SentimentError::Scoring(error).into(),
                }),
            };
            let actual = session.work();
            check_work(actual, bound)?;
            drop(session); // No previous dimension's prompt survives into the next head.
            add_work(&mut completed, actual)?;
            Ok(scores)
        })?;
        check_work(completed, expected)?;
        let run = EagerSentimentRun {
            schema_version: 1, execution: SENTIMENT_NATIVE_EXECUTION.to_owned(),
            numerics_profile: HF_BF16_EAGER_PROFILE.to_owned(), result, native_work: completed,
        };
        let bytes = canonjson::canonical_bytes(&run).map_err(|_| SentimentNativeError::Serialization)?;
        if bytes.len() as u64 > self.max_output_bytes { return Err(SentimentNativeError::OutputBudget); }
        Ok(run)
    }
}

struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}

fn head_bound(prompt: usize, prefixes: usize, rows: u64) -> Result<PrefixWork, SentimentNativeError> {
    if prompt == 0 || prefixes == 0 { return Err(SentimentNativeError::ExecutionDiverged); }
    let continuations = prefixes - 1;
    let positions = prompt.checked_add(continuations).ok_or(SentimentNativeError::ArithmeticOverflow)?;
    Ok(PrefixWork {
        prefix_evaluations: prefixes as u64, forward_positions: positions as u64,
        prompt_positions: prompt as u64, continuation_positions: continuations as u64,
        projected_logits: rows, rewound_positions: 0,
    })
}

fn check_work(actual: PrefixWork, expected: PrefixWork) -> Result<(), SentimentNativeError> {
    // Rewound positions depend on candidate branch shape. All actual forwards,
    // projected rows and evaluations have exact precomputed counts. EOS is
    // included in projection work but must never enter the KV-forward count.
    if actual.prefix_evaluations != expected.prefix_evaluations
        || actual.forward_positions != expected.forward_positions
        || actual.prompt_positions != expected.prompt_positions
        || actual.continuation_positions != expected.continuation_positions
        || actual.projected_logits != expected.projected_logits {
        return Err(SentimentNativeError::ExecutionDiverged);
    }
    Ok(())
}

fn add_work(total: &mut PrefixWork, next: PrefixWork) -> Result<(), SentimentNativeError> {
    let add = |a: u64, b: u64| a.checked_add(b).ok_or(SentimentNativeError::ArithmeticOverflow);
    // Validate every axis before publishing a new aggregate counter value.
    let updated = PrefixWork {
        prefix_evaluations: add(total.prefix_evaluations, next.prefix_evaluations)?,
        forward_positions: add(total.forward_positions, next.forward_positions)?,
        prompt_positions: add(total.prompt_positions, next.prompt_positions)?,
        continuation_positions: add(total.continuation_positions, next.continuation_positions)?,
        projected_logits: add(total.projected_logits, next.projected_logits)?,
        rewound_positions: add(total.rewound_positions, next.rewound_positions)?,
    };
    *total = updated;
    Ok(())
}

struct Ledger { positions: u64, rows: u64 }
impl Ledger {
    fn new(expected: PrefixWork, budget: PrefixBudget) -> Result<Self, SentimentNativeError> {
        if expected.forward_positions > budget.max_forward_positions
            || expected.projected_logits > budget.max_projected_logits {
            return Err(SentimentNativeError::WorkBudget);
        }
        Ok(Self { positions: budget.max_forward_positions, rows: budget.max_projected_logits })
    }
    fn charge(&mut self, head: PrefixWork) -> Result<PrefixBudget, SentimentNativeError> {
        let positions = self.positions.checked_sub(head.forward_positions)
            .ok_or(SentimentNativeError::WorkBudget)?;
        let rows = self.rows.checked_sub(head.projected_logits)
            .ok_or(SentimentNativeError::WorkBudget)?;
        self.positions = positions;
        self.rows = rows;
        Ok(PrefixBudget { max_forward_positions: head.forward_positions,
            max_projected_logits: head.projected_logits })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_axis_prompts_and_unique_continuation_prefixes_are_charged() {
        let first = head_bound(100, 4, 6).unwrap();
        let second = head_bound(120, 7, 9).unwrap();
        let mut all = first; add_work(&mut all, second).unwrap();
        assert_eq!(all.forward_positions, 229);
        assert_eq!(all.prompt_positions, 220);
        assert_eq!(all.continuation_positions, 9);
        assert_eq!(all.projected_logits, 15);
        assert!(Ledger::new(all, PrefixBudget { max_forward_positions: 228, max_projected_logits: 15 }).is_err());
        assert!(Ledger::new(all, PrefixBudget { max_forward_positions: 229, max_projected_logits: 14 }).is_err());
    }
    #[test]
    fn each_head_gets_only_its_slice_and_failed_charge_is_atomic() {
        let head = head_bound(5, 4, 6).unwrap();
        let mut all = head; add_work(&mut all, head).unwrap();
        let mut ledger = Ledger::new(all, PrefixBudget { max_forward_positions: 16, max_projected_logits: 12 }).unwrap();
        let first = ledger.charge(head).unwrap();
        assert_eq!(first.max_forward_positions, 8); assert_eq!(first.max_projected_logits, 6);
        let too_large = head_bound(5, 4, 7).unwrap();
        assert!(ledger.charge(too_large).is_err());
        assert_eq!((ledger.positions, ledger.rows), (8, 6));
        ledger.charge(head).unwrap(); assert_eq!((ledger.positions, ledger.rows), (0, 0));
        assert!(ledger.charge(head).is_err());
    }
    #[test]
    fn incorrect_forward_or_projection_counts_cannot_be_success() {
        let expected = head_bound(5, 4, 6).unwrap();
        let mut wrong = expected; wrong.continuation_positions += 1;
        assert_eq!(check_work(wrong, expected), Err(SentimentNativeError::ExecutionDiverged));
        let mut wrong = expected; wrong.projected_logits += 166144;
        assert_eq!(check_work(wrong, expected), Err(SentimentNativeError::ExecutionDiverged));
        let mut valid = expected; valid.rewound_positions = 2;
        assert!(check_work(valid, expected).is_ok());
    }
    #[test]
    fn work_overflow_and_cancellation_cause_remain_typed() {
        assert!(head_bound(usize::MAX, 2, 1).is_err());
        let mut total = PrefixWork { projected_logits: u64::MAX, ..PrefixWork::default() };
        let before = total;
        assert!(add_work(&mut total, head_bound(1, 1, 1).unwrap()).is_err());
        assert_eq!(total, before);
        assert_eq!(SentimentNativeError::from(PrefixScoringError::Cancelled(DecodeCancellationKind::Deadline)),
            SentimentNativeError::Cancelled(DecodeCancellationKind::Deadline));
    }
}
