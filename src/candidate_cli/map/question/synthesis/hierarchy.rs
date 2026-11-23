//! Explicit question-aware quotation reduction; never an automatic retry.
use super::*;
use crate::tasks::source_planning::quantized::long::question::synthesis::hierarchy::{
    QuestionHierarchyLimits, memory::workspace_bytes};

#[derive(Args)]
pub(in crate::candidate_cli) struct HierarchyArgs {
    /// Reduce collected quotes through bounded context-sized levels before
    /// the final answer. Unselected evidence can be lost; all passes remain visible.
    #[arg(long, requires = "synthesize_answer")]
    hierarchical_answer: bool,
    /// Maximum levels INCLUDING the final answer; default 8.
    #[arg(long, requires = "hierarchical_answer")]
    answer_max_levels: Option<usize>,
    /// ALL additional native passes across ALL levels; default 16.
    /// Reserve every permitted pass at maximum context before discovery.
    #[arg(long, requires = "hierarchical_answer")]
    answer_max_passes: Option<usize>,
    /// Shared exact grouping attempts including question and manifest; default 8192.
    #[arg(long, requires = "hierarchical_answer")]
    answer_max_tokenizer_calls: Option<usize>,
    /// Shared grouping bytes including repeated question and manifests; default 67108864.
    #[arg(long, requires = "hierarchical_answer")]
    answer_max_tokenizer_bytes: Option<u64>,
}
impl HierarchyArgs {
    pub(in crate::candidate_cli) fn limits(&self, synthesize: bool)
        -> Result<Option<QuestionHierarchyLimits>, CandidateError> {
        let supplied = self.answer_max_levels.is_some() || self.answer_max_passes.is_some()
            || self.answer_max_tokenizer_calls.is_some() || self.answer_max_tokenizer_bytes.is_some();
        if !self.hierarchical_answer {
            return if supplied { Err(CandidateError::Arguments) } else { Ok(None) };
        }
        if !synthesize { return Err(CandidateError::Arguments); }
        let defaults = QuestionHierarchyLimits::default();
        let limits = QuestionHierarchyLimits {
            max_levels: self.answer_max_levels.unwrap_or(defaults.max_levels),
            max_passes: self.answer_max_passes.unwrap_or(defaults.max_passes),
            max_tokenizer_calls: self.answer_max_tokenizer_calls.unwrap_or(defaults.max_tokenizer_calls),
            max_tokenizer_bytes: self.answer_max_tokenizer_bytes.unwrap_or(defaults.max_tokenizer_bytes),
        };
        limits.validate().map_err(|_| CandidateError::Arguments)?;
        Ok(Some(limits))
    }
}

impl SynthesisArgs {
    /// The complete hierarchy's non-weight owners must fit even before private
    /// input/model IO. The actual resident model remains charged by the shared
    /// process broker and can only reduce the available allowance further.
    pub(in crate::candidate_cli) fn check_memory(&self, command: &MapCommand, limits: Limits)
        -> Result<(), CandidateError> {
        let Some(hierarchy) = self.hierarchy.limits(self.synthesize_answer)? else { return Ok(()); };
        let budget = command.host.task_budget(limits);
        let synthesis = QuestionSynthesisLimits {
            max_evidence_passages: self.max_synthesis_passages.unwrap_or(source::MAX_PASSAGES),
            max_evidence_bytes: self.max_synthesis_evidence_bytes.unwrap_or(16384).min(command.host.max_input_bytes),
        };
        let workspace = workspace_bytes(budget, command.question_mapping()?, command.max_chunks,
            command.reduction_reserve_mib.checked_mul(MIB).ok_or(CandidateError::Arguments)?,
            synthesis, command.question.verification()?, hierarchy).map_err(|_| CandidateError::Arguments)?;
        let native = Int8MemoryRequirement::for_context(command.host.context_tokens).map_err(|_| CandidateError::Arguments)?;
        let encoding = command.encoding.limits(command)?.map(|e| e.workspace_bytes()).transpose()?.unwrap_or(0);
        let tokens = (command.max_chunks as u64).checked_add(hierarchy.max_passes as u64)
            .and_then(|n| n.checked_mul(u64::from(budget.max_output_tokens)))
            .and_then(|n| n.checked_mul(8)).ok_or(CandidateError::Arguments)?;
        let output = (command.max_map_result_bytes as u64).checked_mul(4)
            .and_then(|n| n.checked_add(tokens)).and_then(|n| n.checked_add(4096))
            .ok_or(CandidateError::Arguments)?;
        let owners = [limits.preparation_bytes, limits.preparation_bytes, workspace, output,
            native.kv_bytes, native.rope_bytes, native.scratch_payload_bound, 64 * MIB,
            command.host.max_input_bytes as u64, super::super::QUESTION_BYTES as u64, encoding];
        let bytes = owners.into_iter().try_fold(0_u64, |total, value| total.checked_add(value).ok_or(CandidateError::Arguments))?;
        if bytes > limits.memory_bytes { return Err(CandidateError::Arguments); }
        Ok(())
    }
}

#[cfg(test)] mod tests;
