//! Exact code-owned scaffold pricing for automatic long-document partitioning.
//!
//! This describes capacity, not a prepared task or model admission. No source
//! placeholder, inference, second template or assumed chars/token ratio is used.
use super::*;
use super::long::SourceMapTask;
use crate::tasks::mapreduce::ChunkLimits;

/// Content-free limits derived from the SAME rendered fragments used by each
/// source task. Private fields prevent callers from inventing a sizing result.
/// Actual chunks still pass source encoding and complete task compilation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int8SourceMapCapacity {
    scaffold_tokens: usize,
    reserved_tokens: usize,
    max_source_tokens: usize,
    context_tokens: usize,
}
impl Int8SourceMapCapacity {
    pub fn scaffold_tokens(self) -> usize { self.scaffold_tokens }
    pub fn reserved_tokens(self) -> usize { self.reserved_tokens }
    pub fn max_source_tokens(self) -> usize { self.max_source_tokens }
    pub fn context_tokens(self) -> usize { self.context_tokens }

    /// Tighten, never enlarge, caller limits. Reserve the exact scaffold AND
    /// all possible output tokens; a caller's larger reserve is retained.
    /// Byte, cardinality and tokenizer-work limits are not altered.
    pub fn constrain_chunks(self, mut chunks: ChunkLimits) -> Result<ChunkLimits, Int8SourceError> {
        chunks.context_tokens = chunks.context_tokens.min(self.context_tokens);
        chunks.reserved_tokens = chunks.reserved_tokens.max(self.reserved_tokens);
        chunks.max_chunk_tokens = chunks.max_chunk_tokens.min(self.max_source_tokens);
        chunks.effective_token_limit().map_err(|_| SourcePlanningError::ContextBudget)?;
        Ok(chunks)
    }
}

impl SourceTaskPlanner {
    /// Compute the available SOURCE-token space before reading/partitioning a
    /// long document. The identity must already match this planner, task and
    /// strict-INT8 backend. Options change the schema and are priced exactly.
    /// No fabricated document is used to obtain an otherwise unbound plan.
    pub fn int8_map_capacity_with_control<C: DecodeStepControl>(&self, task: &SourceMapTask,
        budget: TaskBudget, context: &PlanContext<'_>, limits: SourcePlanningLimits,
        control: &mut C) -> Result<Int8SourceMapCapacity, Int8SourceError> {
        checkpoint(control)?;
        constrained_int8::check_profile(context.execution_identity())?;
        let (kind, schema) = match task {
            SourceMapTask::Ner(options) => (BuiltInTask::Ner, options.schema_source()?),
            SourceMapTask::Keyphrases(options) => (BuiltInTask::Keyphrases, options.schema_source()?),
            SourceMapTask::Summarize(options) => (BuiltInTask::Summarize, options.schema_source()?),
        };
        self.check_context_for_profile(kind, context, budget, limits,
            NumericsProfile::StrictQuantized { version: 1 })?;
        if schema.len() > limits.compiler.max_schema_bytes {
            return Err(SourcePlanningError::InputBudget.into());
        }
        crate::native_engine::strict_int8::Int8MemoryRequirement::for_context(limits.max_context_tokens)
            .map_err(|_| SourcePlanningError::ContextBudget)?;
        let mut scaffold = 0_usize;
        // Deliberately share render_fragments and its segment boundaries with
        // SourceTaskPlanner::task. Flattening/re-tokenizing changes the ABI.
        for fragment in render_fragments(kind, &schema)? {
            checkpoint(control)?;
            let tokens = self.tokenizer.tokenizer().encode_ids_with_options(&fragment,
                EncodeOptions { add_bos: false, add_eos: false })
                .map_err(|_| SourcePlanningError::Contract("source map scaffold encoding"))?;
            scaffold = scaffold.checked_add(tokens.len()).ok_or(SourcePlanningError::ContextBudget)?;
        }
        let capacity = price(scaffold, budget.max_input_tokens as usize,
            budget.max_output_tokens as usize, limits.max_context_tokens)?;
        checkpoint(control)?;
        Ok(capacity)
    }
}

fn price(scaffold: usize, prompt_limit: usize, output: usize, context: usize)
    -> Result<Int8SourceMapCapacity, SourcePlanningError> {
    let reserved = scaffold.checked_add(output).ok_or(SourcePlanningError::ContextBudget)?;
    let prompt_room = prompt_limit.checked_sub(scaffold).ok_or(SourcePlanningError::ContextBudget)?;
    let context_room = context.checked_sub(reserved).ok_or(SourcePlanningError::ContextBudget)?;
    let available = prompt_room.min(context_room);
    if scaffold == 0 || output == 0 || available == 0 { return Err(SourcePlanningError::ContextBudget); }
    Ok(Int8SourceMapCapacity { scaffold_tokens: scaffold, reserved_tokens: reserved,
        max_source_tokens: available, context_tokens: context })
}

#[cfg(test)] mod tests;
