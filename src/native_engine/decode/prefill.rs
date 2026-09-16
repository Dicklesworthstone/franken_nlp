//! Generation-only prefill retention. The trace-collecting engine API remains
//! available to conformance callers; generation needs only the final forward.

use super::{DecodeCancellationKind, DecodeError, DecodeStepControl};

pub(super) enum PrefillOutcome<T> {
    Complete(T),
    Cancelled(DecodeCancellationKind),
}

/// Drop each intermediate forward before beginning the next one. Even a
/// replace-in-place `Option<T>` can briefly retain two forwards during RHS
/// evaluation; splitting out the last token keeps at most one live here.
pub(super) fn prefill_last<T, C, F>(
    tokens: &[u32],
    control: &mut C,
    mut forward: F,
) -> Result<PrefillOutcome<T>, DecodeError>
where
    C: DecodeStepControl,
    F: FnMut(u32) -> Result<T, DecodeError>,
{
    let (&last, prefix) = tokens.split_last().ok_or(DecodeError::EmptyPrompt)?;
    for (index, &token) in prefix.iter().enumerate() {
        if let Some(kind) = control.prefill_checkpoint(index) {
            return Ok(PrefillOutcome::Cancelled(kind));
        }
        drop(forward(token)?);
    }
    if let Some(kind) = control.prefill_checkpoint(prefix.len()) {
        return Ok(PrefillOutcome::Cancelled(kind));
    }
    Ok(PrefillOutcome::Complete(forward(last)?))
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, rc::Rc};

    use super::*;

    struct Continue;
    impl DecodeStepControl for Continue {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            None
        }
    }

    struct TrackedForward {
        token: u32,
        live: Rc<Cell<usize>>,
    }
    impl Drop for TrackedForward {
        fn drop(&mut self) {
            self.live.set(self.live.get() - 1);
        }
    }

    #[test]
    fn intermediate_forwards_drop_before_the_next_model_call() {
        let live = Rc::new(Cell::new(0));
        let mut observed = Vec::new();
        let result = prefill_last(&[7, 8, 9], &mut Continue, |token| {
            assert_eq!(live.get(), 0, "previous trace/logits must already be released");
            observed.push(token);
            live.set(1);
            Ok(TrackedForward { token, live: Rc::clone(&live) })
        }).unwrap();
        assert_eq!(observed, vec![7, 8, 9]);
        let PrefillOutcome::Complete(last) = result else { panic!("unexpected cancellation") };
        assert_eq!(last.token, 9);
        assert_eq!(live.get(), 1);
        drop(last);
        assert_eq!(live.get(), 0);
    }

    #[test]
    fn legacy_control_cancels_before_any_forward() {
        struct Cancel;
        impl DecodeStepControl for Cancel {
            fn checkpoint(&mut self, next: usize) -> Option<DecodeCancellationKind> {
                assert_eq!(next, 0);
                Some(DecodeCancellationKind::Deadline)
            }
        }
        let result = prefill_last::<(), _, _>(&[7, 8], &mut Cancel, |_| {
            panic!("cancelled request must not execute the model")
        }).unwrap();
        assert!(matches!(result, PrefillOutcome::Cancelled(DecodeCancellationKind::Deadline)));
    }

    #[test]
    fn cancellation_between_prompt_tokens_stops_remaining_work() {
        struct CancelAfterOne;
        impl DecodeStepControl for CancelAfterOne {
            fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
                panic!("dedicated prefill hook must be used")
            }
            fn prefill_checkpoint(&mut self, index: usize) -> Option<DecodeCancellationKind> {
                (index == 1).then_some(DecodeCancellationKind::User)
            }
        }
        let mut observed = Vec::new();
        let result = prefill_last(&[7, 8, 9], &mut CancelAfterOne, |token| {
            observed.push(token);
            Ok(())
        }).unwrap();
        assert_eq!(observed, vec![7]);
        assert!(matches!(result, PrefillOutcome::Cancelled(DecodeCancellationKind::User)));
    }

    #[test]
    fn failed_forward_never_runs_the_suffix() {
        let mut observed = Vec::new();
        let result = prefill_last(&[7, 8, 9], &mut Continue, |token| {
            observed.push(token);
            if token == 8 { Err(DecodeError::ContextBudgetOverflow) } else { Ok(()) }
        });
        assert!(matches!(result, Err(DecodeError::ContextBudgetOverflow)));
        assert_eq!(observed, vec![7, 8]);
    }

    #[test]
    fn empty_prompt_refuses_without_model_work() {
        let result = prefill_last::<(), _, _>(&[], &mut Continue, |_| panic!("empty prompt"));
        assert!(matches!(result, Err(DecodeError::EmptyPrompt)));
    }

    #[test]
    fn one_token_prompt_preserves_its_only_forward() {
        let result = prefill_last(&[42], &mut Continue, Ok).unwrap();
        assert!(matches!(result, PrefillOutcome::Complete(42)));
    }

    #[test]
    fn logprob_refuses_nonfinite_unselected_rows() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(super::super::full_vocabulary_logprob(&[0.0, value], 0).is_err());
        }
    }

    #[test]
    fn logprob_keeps_extreme_finite_arithmetic_in_f64() {
        let logprob = super::super::full_vocabulary_logprob(&[f32::MAX, f32::MAX], 0).unwrap();
        assert!((logprob + std::f32::consts::LN_2).abs() < 1.0e-6);
        assert_eq!(super::super::full_vocabulary_logprob(&[f32::MAX, -f32::MAX], 0).unwrap(), 0.0);
        assert!(super::super::full_vocabulary_logprob(&[f32::MAX, -f32::MAX], 1).is_err());
    }
}
