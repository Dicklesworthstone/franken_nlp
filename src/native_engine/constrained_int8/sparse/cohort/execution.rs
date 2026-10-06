//! Ragged structured cursors and all-or-nothing finalization.
use super::*;
struct Cursor<'a> {
    grammar: JsonState<'a>, bytes: Vec<u8>, tokens: Vec<u32>,
    positions: usize, projected: usize, mask_charge: u64, done: bool,
}
impl<'a> Cursor<'a> {
    fn new(request: &Int8JsonCohortRequest<'a>) -> Result<Self, Int8JsonError> {
        Ok(Self { grammar: request.program.initial_state(),
            bytes: reserved(request.program.max_output_bytes())?,
            tokens: reserved(request.options.max_new_tokens)?,
            positions: 0, projected: 0, mask_charge: 0, done: false })
    }
    fn next(&self, prompt: &[u32]) -> Result<u32, Int8JsonError> {
        prompt.get(self.positions).copied().or_else(|| self.tokens.last().copied())
            .ok_or_else(|| JsonDecodeError::IllegalTransition.into())
    }
}
pub(super) fn drive<V, D, T, E, F>(requests: &[Int8JsonCohortRequest<'_>], vocabulary: &V,
    budget: Int8JsonCohortBudget, driver: &mut D, finalize: F) -> Result<T, E>
where V: Vocabulary, D: GroupDriver, E: From<Int8JsonError>, F: FnOnce(Int8JsonCohortRun) -> Result<T, E> {
    let result = run(requests, vocabulary, budget, driver).map_err(E::from).and_then(|run| {
        let next = run.sequences.iter().map(|row| row.output.token_ids.len()).max().unwrap_or(0);
        let result = finalize(run)?;
        poll(driver.control(), next).map_err(E::from)?;
        Ok(result)
    });
    if result.is_err() { driver.abort(); }
    result
}
fn run<V: Vocabulary, D: GroupDriver>(requests: &[Int8JsonCohortRequest<'_>], vocabulary: &V,
    budget: Int8JsonCohortBudget, driver: &mut D) -> Result<Int8JsonCohortRun, Int8JsonError> {
    let planned_work = requests_preflight(requests, vocabulary, budget)?;
    let mut cursors = reserved(requests.len())?;
    for request in requests { cursors.push(Cursor::new(request)?); }
    let mut steps = reserved(requests.len())?; let mut group_steps = 0_u64;
    while cursors.iter().any(|row| !row.done) {
        steps.clear();
        for (slot, row) in cursors.iter().enumerate() {
            if row.done { continue; }
            if row.positions < requests[slot].prompt.len() {
                if let Some(cause) = driver.control().prefill_checkpoint(row.positions) {
                    return Err(JsonDecodeError::Cancelled(cause).into());
                }
            } else { poll(driver.control(), row.tokens.len())?; }
            steps.push(CohortToken { sequence: slot, token: row.next(requests[slot].prompt)? });
        }
        // One actual shared-layer call, not a loop over single-row engines.
        // Prompt and generated rows coexist with independent causal positions.
        driver.append_group(&steps)?;
        group_steps = group_steps.checked_add(1).ok_or(StrictInt8Error::Work)?;
        for step in &steps {
            let slot = step.sequence; let request = &requests[slot]; let row = &mut cursors[slot];
            row.positions = row.positions.checked_add(1).ok_or(StrictInt8Error::Work)?;
            if row.positions < request.prompt.len() { continue; }
            let index = row.tokens.len();
            poll(driver.control(), index)?;
            row.mask_charge = row.mask_charge.checked_add(vocabulary.mask_charge(request.budget.json.mask_limits) as u64)
                .filter(|&n| n <= request.budget.json.max_total_mask_node_visits)
                .ok_or(JsonDecodeError::BudgetExceeded("cohort row mask work"))?;
            let mask = vocabulary.mask(&row.grammar, request.budget.json.mask_limits, driver.control(), index)?;
            if mask.vocab_size() != vocabulary.width() { return Err(JsonDecodeError::IllegalTransition.into()); }
            let remaining = request.budget.json.max_projected_logits.checked_sub(row.projected as u64)
                .ok_or(Int8JsonError::WorkMismatch)?;
            let (selected, count) = selection::project(&mask, row.grammar.is_accepting(), request.options,
                request.limits.max_rows_per_step, remaining, &mut RowHead { driver, slot }, index)?;
            row.projected = row.projected.checked_add(count).ok_or(StrictInt8Error::Work)?;
            drop(mask); poll(driver.control(), index)?;
            row.tokens.push(selected);
            if selected == request.options.eos_token_id { row.done = true; continue; }
            let emitted = vocabulary.bytes(selected).filter(|b| !b.is_empty()).ok_or(JsonDecodeError::IllegalTransition)?;
            if row.bytes.len().checked_add(emitted.len()).is_none_or(|n| n > request.program.max_output_bytes()) {
                return Err(JsonDecodeError::BudgetExceeded("cohort row JSON bytes").into());
            }
            if !row.grammar.consume_bytes(emitted) { return Err(JsonDecodeError::IllegalTransition.into()); }
            row.bytes.extend_from_slice(emitted);
            if row.tokens.len() >= request.options.max_new_tokens {
                return Err(JsonDecodeError::BudgetExceeded("cohort output before EOS").into());
            }
        }
    }
    let mut sequences = reserved(requests.len())?; let mut model_work = Int8Work::default();
    let mut longest = 0; let mut masks = 0_u64;
    for (slot, row) in cursors.into_iter().enumerate() {
        let json = String::from_utf8(row.bytes).map_err(|_| JsonDecodeError::IndependentValidation)?;
        requests[slot].program.validate_json(&json).map_err(|_| JsonDecodeError::IndependentValidation)?;
        let work = driver.work(slot)?;
        let positions = requests[slot].prompt.len().checked_add(row.tokens.len() - 1).ok_or(StrictInt8Error::Work)?;
        if positions != row.positions || work != Int8Work::for_sequence(0, positions, row.projected)? {
            return Err(Int8JsonError::WorkMismatch);
        }
        longest = longest.max(work.forward_positions); model_work = model_work.checked_add(work)?;
        masks = masks.checked_add(row.mask_charge).ok_or(StrictInt8Error::Work)?;
        sequences.push(Int8JsonRun { schema_version: 1, execution: INT8_SPARSE_JSON_EXECUTION.to_owned(), model_work: work,
            output: JsonDecodeOutput { schema_version: 1, numerics_profile: STRICT_INT8_PROFILE.to_owned(),
                token_ids: row.tokens, json, forward_positions: work.forward_positions,
                projected_logits: work.projected_logits, mask_node_visit_charge: row.mask_charge } });
    }
    if group_steps != longest || masks > budget.max_mask_node_visits { return Err(Int8JsonError::WorkMismatch); }
    let result = Int8JsonCohortRun { schema_version: 1, execution: INT8_JSON_COHORT_EXECUTION.to_owned(),
        sequences, group_steps, planned_work, model_work };
    check_size(&result, budget.max_result_bytes)?;
    Ok(result)
}
fn check_size(value: &impl Serialize, cap: u64) -> Result<(), Int8JsonError> {
    struct Counter { remaining: u64 }
    impl std::io::Write for Counter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.remaining = self.remaining.checked_sub(bytes.len() as u64)
                .ok_or_else(|| std::io::Error::other("cohort envelope byte bound"))?;
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
    }
    serde_json::to_writer(Counter { remaining: cap }, value)
        .map_err(|_| JsonDecodeError::BudgetExceeded("complete cohort envelope").into())
}
