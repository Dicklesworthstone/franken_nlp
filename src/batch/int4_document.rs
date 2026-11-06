//! Shared lifetime accounting for complete-document native INT4 streams.
//!
//! Preparation performs no model work. An execution attempt charges all five
//! counters atomically before replanning, admission or native execution. Flush
//! has no entry point here and cannot replenish the invocation allowance.
use super::{BatchCode, BatchItemFailure, BatchRequestContext};
use crate::native_engine::{portable_int8::ProjectionWork, strict_int4::Int4Work};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum State { Ready, Running, Failed }

pub(crate) struct RunState {
    remaining: Int4Work,
    last_sequence: u64,
    state: State,
}

impl RunState {
    pub(crate) fn new(remaining: Int4Work) -> Self {
        Self { remaining, last_sequence: 0, state: State::Ready }
    }

    pub(crate) fn remaining(&self) -> Int4Work { self.remaining }
    pub(crate) fn is_unavailable(&self) -> bool { self.state != State::Ready }
    pub(crate) fn fail(&mut self) { self.state = State::Failed; }

    pub(crate) fn enter(&mut self) -> Result<(), BatchItemFailure> {
        if self.state != State::Ready {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        self.state = State::Running;
        Ok(())
    }

    pub(crate) fn charge(&mut self, work: Int4Work, context: BatchRequestContext)
        -> Result<(), BatchItemFailure> {
        if self.state != State::Running || context.request_seq <= self.last_sequence
            || context.epoch == 0 || context.input_line == 0 {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        let remaining = subtract(self.remaining, work)
            .map_err(|error| BatchItemFailure::fatal(error.fault))?;
        self.remaining = remaining;
        self.last_sequence = context.request_seq;
        Ok(())
    }

    pub(crate) fn finish_preparation<T>(&mut self, result: Result<T, BatchItemFailure>, clean: bool)
        -> Result<T, BatchItemFailure> {
        match result {
            Ok(value) if clean && self.state == State::Running => {
                self.state = State::Ready;
                Ok(value)
            }
            Ok(_) => {
                self.state = State::Failed;
                Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
            }
            Err(mut error) => {
                // Native abort may deliberately poison its engine. Keep the
                // original typed failure while permanently refusing reuse.
                if !clean || self.state != State::Running || error.fault.cancellation.is_some() {
                    error.stop = true;
                }
                self.state = if error.stop { State::Failed } else { State::Ready };
                Err(error)
            }
        }
    }

    pub(crate) fn finish_execution<T>(&mut self, result: Result<T, BatchItemFailure>, clean: bool)
        -> Result<T, BatchItemFailure> {
        // Every execution error follows an attempt debit, including an
        // admission refusal. Neither error recovery nor unwind refunds it.
        self.finish_preparation(result.map_err(|mut error| { error.stop = true; error }), clean)
    }
}

pub(crate) fn nonzero(work: Int4Work) -> bool {
    work.forward_positions != 0 && work.projected_logits != 0 && work.attention_pairs != 0
        && work.projections.dot_products != 0 && work.projections.multiply_accumulates != 0
}

pub(crate) fn fits(work: Int4Work, ceiling: Int4Work) -> bool {
    work.forward_positions <= ceiling.forward_positions && work.projected_logits <= ceiling.projected_logits
        && work.attention_pairs <= ceiling.attention_pairs && work.projections.fits(ceiling.projections)
}

pub(crate) fn subtract(remaining: Int4Work, work: Int4Work) -> Result<Int4Work, BatchItemFailure> {
    let sub = |available: u64, requested: u64| available.checked_sub(requested)
        .ok_or_else(|| BatchItemFailure::reject(BatchCode::WorkLimit));
    Ok(Int4Work {
        forward_positions: sub(remaining.forward_positions, work.forward_positions)?,
        projected_logits: sub(remaining.projected_logits, work.projected_logits)?,
        attention_pairs: sub(remaining.attention_pairs, work.attention_pairs)?,
        projections: ProjectionWork {
            dot_products: sub(remaining.projections.dot_products, work.projections.dot_products)?,
            multiply_accumulates: sub(remaining.projections.multiply_accumulates, work.projections.multiply_accumulates)?,
        },
    })
}
