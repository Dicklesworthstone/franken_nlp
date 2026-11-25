//! One controlled, model-free compiler for live and retained judgment records.
use super::*;
use std::io;

const MAX_DEFAULT_BYTES: usize = 1024 * 1024;

impl JudgeCorpusConfig {
    /// Check the actual pinned compiler and quote the simultaneously live
    /// encoder plus invocation ledger. Plain byte fallback has no policy
    /// extension; compact execution requires all four whole-invocation caps.
    pub fn encoding_requirements(&self, planner: &JudgePlanner, limits: Option<SafeBpeWorkLimits>)
        -> Result<(UntrustedEncoding, Option<SafeBpeWorkLimits>, u64), HostedError> {
        self.validate(planner)?;
        let policy = planner.document_encoding().clone();
        let bytes = EncodingWorkScope::workspace(&policy, limits).map_err(HostedError::BatchSetup)?;
        Ok((policy, limits, bytes))
    }

    /// Fixed configuration only. Never invent a document to validate a future
    /// stream, and never rewrite the supplied model/task identity.
    pub fn validate(&self, planner: &JudgePlanner) -> Result<(), HostedError> {
        self.identity.validate().map_err(|_| HostedError::ModelIdentity)?;
        check_binding(&self.identity, planner.template_digest(), planner.tokenizer_digest())?;
        planner.document_encoding().validate()
            .map_err(|_| HostedError::Limits("judge corpus input encoding"))?;
        self.task_ceiling.validate().map_err(|_| HostedError::Limits("judge corpus task ceiling"))?;
        validate_work(self.max_model_work)?;
        let limits = self.planning;
        let head = limits.per_head;
        if head.max_candidates == 0 || head.max_total_tokens == 0 || head.max_nodes == 0
            || head.max_depth == 0 || head.max_candidate_id_bytes == 0 || head.max_projected_logits == 0
            || limits.max_total_prompt_tokens == 0 || limits.max_total_candidate_tokens == 0
            || limits.max_total_projected_logits == 0 || !(1..=64 * 1024 * 1024).contains(&limits.max_output_bytes) {
            return Err(HostedError::Limits("judge corpus planning limits"));
        }
        if let Some(args) = &self.defaults {
            // Count borrowed serialization before any defaults clone. No JSON
            // tree or unbounded staging buffer is allocated just to check size.
            check_serialized_size(args, MAX_DEFAULT_BYTES)?;
            check_args(args, self.task_ceiling)
                .map_err(|_| HostedError::Limits("judge corpus defaults"))?;
        }
        Ok(())
    }
}

#[cfg(test)]
pub(super) fn prepare<C: DecodeStepControl>(planner: &JudgePlanner, config: &JudgeCorpusConfig,
    document: BatchDocument<JudgeBatchArgs>, control: &mut C) -> Result<PreparedInt8Judge, BatchItemFailure> {
    let encoding = EncodingWorkScope::new(planner.document_encoding(), None)
        .map_err(BatchItemFailure::fatal)?;
    prepare_with_encoding_scope(planner, config, &encoding, document, control)
}

pub(super) fn prepare_with_encoding_scope<C: DecodeStepControl>(planner: &JudgePlanner,
    config: &JudgeCorpusConfig, encoding: &EncodingWorkScope,
    document: BatchDocument<JudgeBatchArgs>, control: &mut C) -> Result<PreparedInt8Judge, BatchItemFailure> {
    checkpoint(control)?;
    if planner.document_encoding() != encoding.policy() {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    let args = document.task_args.or_else(|| config.defaults.clone())
        .ok_or_else(|| BatchItemFailure::reject(BatchCode::Planning))?;
    check_args(&args, config.task_ceiling)?;
    let context = PlanContext::new(&config.identity, config.task_ceiling)
        .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
    let request = args.into_request(document.text);
    let plan = encoding.prepare(control, |remaining, work, control| {
        let effective = remaining.map(UntrustedEncoding::SafeBpeV1)
            .unwrap_or(UntrustedEncoding::ByteFallback);
        planner.plan_int8_with_encoding_budget(&request, &context,
            config.planning, control, &effective, work).map_err(planning_failure)
    })?;
    checkpoint(control)?;
    Ok(plan)
}

fn check_args(args: &JudgeBatchArgs, ceiling: TaskBudget) -> Result<(), BatchItemFailure> {
    let reject = || BatchItemFailure::reject(BatchCode::Planning);
    let budget = match args {
        JudgeBatchArgs::Pairwise { criterion, b, budget, .. } => {
            if criterion.is_empty() || b.is_empty() { return Err(reject()); }
            // Pairwise margins are u32 milli-log-odds, not ppm probabilities.
            *budget
        }
        JudgeBatchArgs::Rubric { rubric, policy, budget } => {
            rubric.validate().map_err(|_| reject())?;
            if policy.minimum_peak_weight_ppm > 1_000_000 || policy.maximum_normalized_entropy_ppm > 1_000_000 {
                return Err(reject());
            }
            *budget
        }
        JudgeBatchArgs::Faithfulness { claim, policy, budget } => {
            if claim.is_empty() { return Err(reject()); }
            policy.validate().map_err(|_| reject())?;
            *budget
        }
    };
    budget.validate().map_err(|_| reject())?;
    if budget.max_input_tokens > ceiling.max_input_tokens || budget.max_output_tokens > ceiling.max_output_tokens
        || budget.max_output_bytes > ceiling.max_output_bytes || budget.max_grammar_states > ceiling.max_grammar_states
        || budget.max_kv_bytes > ceiling.max_kv_bytes { return Err(reject()); }
    Ok(())
}

pub(super) fn check_serialized_size<T: Serialize + ?Sized>(value: &T, cap: usize) -> Result<(), HostedError> {
    struct Counter { bytes: usize, cap: usize }
    impl io::Write for Counter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.bytes = self.bytes.checked_add(bytes.len()).filter(|&n| n <= self.cap)
                .ok_or_else(|| io::Error::other("judge settings size limit"))?;
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> { Ok(()) }
    }
    serde_json::to_writer(Counter { bytes: 0, cap }, value)
        .map_err(|_| HostedError::Limits("judge settings serialization bound"))
}

/// Count complete compact-result serialization without allocating a second
/// output. Preserve terminal cancellation even when the same write also
/// discovers that the provenance extension exceeds the original byte cap.
pub(super) fn check_output_bytes<T: Serialize + ?Sized, C: DecodeStepControl>(
    value: &T, cap: u64, control: &mut C) -> Result<(), BatchItemFailure> {
    struct Counter<'a, C> {
        bytes: u64, cap: u64, control: &'a mut C,
        cancellation: Option<crate::native_engine::decode::DecodeCancellationKind>,
        limited: bool,
    }
    impl<C: DecodeStepControl> io::Write for Counter<'_, C> {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if let Some(cause) = self.control.prefill_checkpoint(0) {
                self.cancellation = Some(cause);
                return Err(io::Error::other("judge output cancelled"));
            }
            let next = self.bytes.checked_add(bytes.len() as u64).filter(|&n| n <= self.cap);
            match next {
                Some(next) => { self.bytes = next; Ok(bytes.len()) }
                None => { self.limited = true; Err(io::Error::other("judge result byte limit")) }
            }
        }
        fn flush(&mut self) -> io::Result<()> { Ok(()) }
    }
    checkpoint(control)?;
    let mut writer = Counter { bytes: 0, cap, control, cancellation: None, limited: false };
    let result = serde_json::to_writer(&mut writer, value);
    if let Some(cause) = writer.cancellation.or_else(|| writer.control.prefill_checkpoint(0)) {
        return Err(BatchItemFailure::fatal(BatchFault::cancelled(cause)));
    }
    match result {
        Ok(()) => Ok(()),
        Err(_) if writer.limited => Err(BatchItemFailure::reject(BatchCode::OutputLineLimit)),
        Err(_) => Err(BatchItemFailure::fatal(BatchCode::Serialization)),
    }
}

#[cfg(test)] pub(super) mod tests;
