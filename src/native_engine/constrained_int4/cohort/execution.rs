//! Ragged native forward/head groups with independent constrained cursors.
use super::*;

struct Cursor<'a> {
    grammar: JsonState<'a>, bytes: Vec<u8>, tokens: Vec<u32>,
    positions: usize, projected: usize, mask_charge: u64, done: bool,
}
impl<'a> Cursor<'a> {
    fn new(request: &Int4JsonCohortRequest<'a>) -> Result<Self, Int4JsonError> {
        Ok(Self { grammar: request.program.initial_state(),
            bytes: reserved(request.program.max_output_bytes())?, tokens: reserved(request.options.max_new_tokens)?,
            positions: 0, projected: 0, mask_charge: 0, done: false })
    }
    fn next(&self, prompt: &[u32]) -> Result<u32, Int4JsonError> {
        prompt.get(self.positions).copied().or_else(|| self.tokens.last().copied())
            .ok_or_else(|| JsonDecodeError::IllegalTransition.into())
    }
}

pub(super) fn drive<V, D, T, E, F>(group_width: usize, requests: &[Int4JsonCohortRequest<'_>], vocabulary: &V,
    budget: Int4JsonCohortBudget, driver: &mut D, finalize: F) -> Result<T, E>
where V: Vocabulary, D: GroupDriver, E: From<Int4JsonError>, F: FnOnce(Int4JsonCohortRun) -> Result<T, E> {
    let result = run(group_width, requests, vocabulary, budget, driver).map_err(E::from).and_then(|run| {
        let step = run.sequences.iter().map(|row| row.output.token_ids.len()).max().unwrap_or(0);
        let result = finalize(run)?;
        poll(driver.control(), step).map_err(E::from)?;
        Ok(result)
    });
    if result.is_err() { driver.abort(); }
    result
}
fn run<V: Vocabulary, D: GroupDriver>(group_width: usize, requests: &[Int4JsonCohortRequest<'_>], vocabulary: &V,
    budget: Int4JsonCohortBudget, driver: &mut D) -> Result<Int4JsonCohortRun, Int4JsonError> {
    let planned_work = requests_preflight(group_width, requests, vocabulary, budget)?;
    let mut cursors = reserved(requests.len())?;
    for request in requests { cursors.push(Cursor::new(request)?); }
    let mut steps = reserved(requests.len())?; let mut ready = reserved(requests.len())?;
    let mut group_steps = 0_u64; let mut projection_groups = 0_u64;
    while cursors.iter().any(|row| !row.done) {
        steps.clear(); ready.clear();
        for (slot, row) in cursors.iter().enumerate() {
            if row.done { continue; }
            if row.positions < requests[slot].prompt.len() {
                if let Some(cause) = driver.control().prefill_checkpoint(row.positions) {
                    return Err(JsonDecodeError::Cancelled(cause).into());
                }
            } else { poll(driver.control(), row.tokens.len())?; }
            steps.push(Int4CohortToken { sequence: slot, token: row.next(requests[slot].prompt)? });
        }
        driver.append_group(&steps)?;
        group_steps = group_steps.checked_add(1).ok_or(StrictInt4Error::Work)?;
        for step in &steps {
            let row = &mut cursors[step.sequence];
            row.positions = row.positions.checked_add(1).ok_or(StrictInt4Error::Work)?;
            if row.positions >= requests[step.sequence].prompt.len() { ready.push(step.sequence); }
        }
        if ready.is_empty() { continue; }
        // One real shared head call over only ready sequences. Selection-major
        // rows remain full-vocabulary and must be finite even for illegal IDs.
        let logits = driver.logits_group(&ready)?;
        let expected = vocabulary.width().checked_mul(ready.len()).ok_or(StrictInt4Error::Memory)?;
        checked_logits(&logits, expected)?;
        projection_groups = projection_groups.checked_add(1).ok_or(StrictInt4Error::Work)?;
        for (&slot, logits) in ready.iter().zip(logits.chunks_exact(vocabulary.width())) {
            let request = &requests[slot]; let row = &mut cursors[slot]; let step = row.tokens.len();
            poll(driver.control(), step)?;
            row.projected = row.projected.checked_add(vocabulary.width()).ok_or(StrictInt4Error::Work)?;
            if row.projected as u64 > request.budget.json.max_projected_logits {
                return Err(JsonDecodeError::BudgetExceeded("cohort row projected logits").into());
            }
            row.mask_charge = row.mask_charge.checked_add(vocabulary.mask_charge(request.budget.json.mask_limits) as u64)
                .filter(|&n| n <= request.budget.json.max_total_mask_node_visits)
                .ok_or(JsonDecodeError::BudgetExceeded("cohort row mask work"))?;
            let mask = vocabulary.mask(&row.grammar, request.budget.json.mask_limits, driver.control(), step)?;
            if mask.vocab_size() != vocabulary.width() { return Err(JsonDecodeError::IllegalTransition.into()); }
            let selected = select(logits, &mask, row.grammar.is_accepting(), request.options)?;
            drop(mask); poll(driver.control(), step)?;
            row.tokens.push(selected);
            if selected == request.options.eos_token_id { row.done = true; continue; }
            let emitted = vocabulary.bytes(selected).filter(|bytes| !bytes.is_empty())
                .ok_or(JsonDecodeError::IllegalTransition)?;
            if row.bytes.len().checked_add(emitted.len()).is_none_or(|n| n > request.program.max_output_bytes()) {
                return Err(JsonDecodeError::BudgetExceeded("cohort row JSON bytes").into());
            }
            if !row.grammar.consume_bytes(emitted) { return Err(JsonDecodeError::IllegalTransition.into()); }
            row.bytes.extend_from_slice(emitted);
            if row.tokens.len() >= request.options.max_new_tokens {
                return Err(JsonDecodeError::BudgetExceeded("cohort output tokens before EOS").into());
            }
        }
    }
    let mut sequences = reserved(requests.len())?; let mut model_work = Int4Work::default();
    let mut longest = 0_u64; let mut masks = 0_u64;
    for (slot, row) in cursors.into_iter().enumerate() {
        let json = String::from_utf8(row.bytes).map_err(|_| JsonDecodeError::IndependentValidation)?;
        requests[slot].program.validate_json(&json).map_err(|_| JsonDecodeError::IndependentValidation)?;
        let work = driver.work(slot)?;
        let positions = requests[slot].prompt.len().checked_add(row.tokens.len() - 1).ok_or(StrictInt4Error::Work)?;
        if positions != row.positions || work != Int4Work::for_sequence(group_width, 0, positions, row.projected)? {
            return Err(Int4JsonError::WorkMismatch);
        }
        longest = longest.max(work.forward_positions); model_work = model_work.checked_add(work)?;
        masks = masks.checked_add(row.mask_charge).ok_or(StrictInt4Error::Work)?;
        sequences.push(Int4JsonRun { schema_version: 1, execution: INT4_JSON_EXECUTION.to_owned(), group_width,
            model_work: work, output: JsonDecodeOutput { schema_version: 1, numerics_profile: STRICT_INT4_PROFILE.to_owned(),
                token_ids: row.tokens, json, forward_positions: work.forward_positions,
                projected_logits: work.projected_logits, mask_node_visit_charge: row.mask_charge } });
    }
    if group_steps != longest || projection_groups == 0 || projection_groups > group_steps
        || masks > budget.max_mask_node_visits { return Err(Int4JsonError::WorkMismatch); }
    let result = Int4JsonCohortRun { schema_version: 1, execution: INT4_JSON_COHORT_EXECUTION.to_owned(), sequences,
        group_steps, projection_groups, planned_work, model_work };
    check_size(&result, budget.max_result_bytes)?;
    Ok(result)
}
fn check_size(value: &impl Serialize, cap: u64) -> Result<(), Int4JsonError> {
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
