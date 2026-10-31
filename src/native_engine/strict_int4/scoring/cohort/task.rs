//! Closed task composition keeps semantic decisions in their existing finalizer.
use super::*;

pub(crate) fn contexts(schedules: &[CandidateSchedule]) -> Result<Vec<usize>, Int4ScoringError> {
    check_count(schedules.len())?;
    let width = schedules[0].group_width; check_group(width)?;
    let mut contexts = reserve(schedules.len())?;
    for schedule in schedules {
        if schedule.group_width != width { return Err(Int4ScoringError::Identity); }
        if schedule.context == 0 || schedule.context > DEFAULT_ADMITTED_CONTEXT_CAP { return Err(Int4ScoringError::Input); }
        contexts.push(schedule.context);
    }
    Ok(contexts)
}

/// Only the three sealed task implementations implement this adapter. It is
/// deliberately crate-private: callers cannot inject native scores, scorers,
/// head boundaries or semantic finalizers into the public document API.
pub(crate) trait DocumentPlan {
    type Output: Serialize;
    type Error: From<Int4ScoringError>;
    fn schedules(&self) -> &[CandidateSchedule];
    fn work(&self) -> Int4Work;
    fn kv_cap(&self) -> u64;
    fn output_cap(&self) -> u64;
    fn verify_document(&self, admitted: &ExecutionIdentity, model: &Int4ModelIdentity)
        -> Result<(), Self::Error>;
    fn prompts<C: DecodeStepControl>(&self, control: &mut C) -> Result<Vec<Vec<u32>>, Self::Error>;
    fn append_requests<'a>(&'a self, admitted: &'a ExecutionIdentity, prompts: &'a [Vec<u32>],
        budget: Int4ScoringCohortBudget, requests: &mut Vec<CompiledRequest<'a>>) -> Result<(), Self::Error>;
    fn finish_document<C: DecodeStepControl>(&self, heads: &mut std::vec::IntoIter<Int4CandidateRun>,
        control: &mut C) -> Result<Self::Output, Self::Error>;
}

/// Preparation and admission stay with the host. This composition invokes the
/// real native driver ONCE over all document heads, then keeps every semantic
/// finalizer inside that driver's exclusive, failure-poisoning lifetime.
pub(crate) fn execute_documents<P: DocumentPlan, C: DecodeStepControl>(
    documents: &[(&P, &ExecutionIdentity)], engine: &mut Int4CohortEngine<'_>,
    budget: Int4ScoringCohortBudget, control: &mut C,
) -> Result<Int4ScoredCohort<Vec<P::Output>>, P::Error> {
    checkpoint(control).map_err(P::Error::from)?;
    check_count(documents.len()).map_err(P::Error::from)?;
    engine.check_idle().map_err(Int4ScoringError::from).map_err(P::Error::from)?;
    for &(plan, identity) in documents {
        checkpoint(control).map_err(P::Error::from)?;
        plan.verify_document(identity, engine.model_identity())?;
    }
    let mut capacities = reserve(engine.sequence_count()).map_err(P::Error::from)?;
    for slot in 0..engine.sequence_count() {
        capacities.push(engine.capacity(slot).map_err(Int4ScoringError::from).map_err(P::Error::from)?);
    }
    let (schedules, work) = document_layout(documents, &capacities, budget).map_err(P::Error::from)?;
    let mut prompts = reserve(documents.len()).map_err(P::Error::from)?;
    for &(plan, _) in documents { prompts.push(plan.prompts(control)?); }
    let mut requests = reserve(schedules.len()).map_err(P::Error::from)?;
    for (&(plan, identity), prompts) in documents.iter().zip(&prompts) {
        let start = requests.len();
        plan.append_requests(identity, prompts, budget, &mut requests)?;
        if requests.len() - start != plan.schedules().len() {
            return Err(P::Error::from(Int4ScoringError::Accounting));
        }
    }
    execute_compiled_with(engine, &requests, budget, control, |run, control|
        finish_documents(documents, run, &schedules, work, budget.max_output_bytes, control))
}

/// Price actual resident capacities per document. A formerly serial task cap
/// must cover ALL of that document's simultaneous heads; excess capacity in a
/// supplied engine is resident memory too and receives no hidden discount.
pub(crate) fn document_layout<P: DocumentPlan>(documents: &[(&P, &ExecutionIdentity)],
    capacities: &[usize], budget: Int4ScoringCohortBudget,
) -> Result<(Vec<CandidateSchedule>, Int4Work), Int4ScoringError> {
    check_count(documents.len())?; check_count(capacities.len())?;
    check_output_limit(budget.max_output_bytes)?;
    let mut schedules = reserve(capacities.len())?;
    let mut work = Int4Work::default(); let mut kv = 0_u64;
    for &(plan, identity) in documents {
        check_count(plan.schedules().len())?; check_output_limit(plan.output_cap())?;
        same_model(documents[0].1, identity)?;
        let mut document_work = Int4Work::default(); let mut document_kv = 0_u64;
        for &schedule in plan.schedules() {
            let capacity = *capacities.get(schedules.len()).ok_or(Int4ScoringError::Input)?;
            if capacity == 0 || capacity > DEFAULT_ADMITTED_CONTEXT_CAP {
                return Err(StrictInt4Error::Context.into());
            }
            if let Some(first) = schedules.first() {
                let first: &CandidateSchedule = first;
                if first.group_width != schedule.group_width { return Err(Int4ScoringError::Identity); }
            }
            check_capacity(schedule, capacity, Int4ScoringBudget {
                native: Int4RunBudget::exact(schedule.model), max_kv_bytes: plan.kv_cap().min(budget.max_kv_bytes) })?;
            document_kv = add(document_kv, mul(capacity as u64, KV_BYTES_PER_TOKEN as u64)?)?;
            document_work = document_work.checked_add(schedule.model)?;
            schedules.push(schedule);
        }
        if document_work != plan.work() { return Err(Int4ScoringError::Accounting); }
        if document_kv > plan.kv_cap() { return Err(StrictInt4Error::Memory.into()); }
        work = work.checked_add(document_work)?; kv = add(kv, document_kv)?;
    }
    if schedules.len() != capacities.len() { return Err(Int4ScoringError::Input); }
    if kv > budget.max_kv_bytes { return Err(StrictInt4Error::Memory.into()); }
    if work.forward_positions > budget.native.max_forward_positions || work.attention_pairs > budget.native.max_attention_pairs
        || !work.projections.fits(budget.native.max_projection_work) { return Err(StrictInt4Error::Work.into()); }
    Ok((schedules, work))
}

pub(crate) fn finish_documents<P: DocumentPlan, C: DecodeStepControl>(documents: &[(&P, &ExecutionIdentity)],
    run: Int4CandidateCohortRun, schedules: &[CandidateSchedule], work: Int4Work, cap: u64, control: &mut C,
) -> Result<Int4ScoredCohort<Vec<P::Output>>, P::Error> {
    finish(run, schedules, work, cap, |heads| {
        let mut output = reserve(documents.len()).map_err(P::Error::from)?;
        for &(plan, _) in documents {
            checkpoint(control).map_err(P::Error::from)?;
            let remaining = heads.len();
            let result = plan.finish_document(heads, control)?;
            if remaining.checked_sub(heads.len()) != Some(plan.schedules().len()) {
                return Err(P::Error::from(Int4ScoringError::Accounting));
            }
            check_output(&result, plan.output_cap()).map_err(P::Error::from)?;
            output.push(result);
        }
        checkpoint(control).map_err(P::Error::from)?;
        Ok(output)
    })
}

pub(crate) fn finish<T, E, F>(run: Int4CandidateCohortRun, schedules: &[CandidateSchedule], expected: Int4Work,
    cap: u64, finalize: F) -> Result<Int4ScoredCohort<T>, E>
where T: Serialize, E: From<Int4ScoringError>, F: FnOnce(&mut std::vec::IntoIter<Int4CandidateRun>) -> Result<T, E> {
    let validate = (|| -> Result<(), Int4ScoringError> {
        check_count(schedules.len())?;
        let width = schedules[0].group_width; check_group(width)?;
        if run.schema_version != 1 || run.execution != INT4_SCORING_COHORT_EXECUTION
            || run.numerics_profile != STRICT_INT4_PROFILE || run.group_width != width || run.model_work != expected
            || run.heads.len() != schedules.len() || run.group_steps == 0 || run.group_steps > expected.forward_positions {
            return Err(Int4ScoringError::Accounting);
        }
        let mut work = Int4Work::default(); let mut projections = 0_u64;
        for (head, schedule) in run.heads.iter().zip(schedules) {
            if schedule.group_width != width || head.group_width != width
                || head.schema_version != 1 || head.execution != INT4_SCORING_EXECUTION
                || head.numerics_profile != STRICT_INT4_PROFILE || head.model_work != schedule.model
                || head.scores.work != schedule.scoring || head.rewound_positions != schedule.rewound {
                return Err(Int4ScoringError::Accounting);
            }
            work = work.checked_add(head.model_work)?;
            projections = add(projections, schedule.scoring.prefix_evaluations as u64)?;
        }
        if work != expected || run.projection_groups == 0 || run.projection_groups > projections {
            return Err(Int4ScoringError::Accounting);
        }
        Ok(())
    })();
    validate.map_err(E::from)?;
    let mut heads = run.heads.into_iter();
    let output = finalize(&mut heads)?;
    if heads.next().is_some() { return Err(E::from(Int4ScoringError::Accounting)); }
    let result = Int4ScoredCohort { schema_version: 1, execution: INT4_SCORING_COHORT_EXECUTION.to_owned(),
        group_steps: run.group_steps, projection_groups: run.projection_groups, output };
    check_output(&result, cap).map_err(E::from)?;
    Ok(result)
}

#[cfg(test)] pub(crate) mod fixtures;
