//! Exact trusted-scaffold capacity for grouped-INT4 source document maps.
//! The SAME rendered fragments and segment boundaries price actual task
//! compilation. No placeholder source, assumed token ratio or model run.
use super::*;
use super::long::SourceMapTask;
use crate::{native_engine::strict_int4::Int4MemoryRequirement, tasks::mapreduce::ChunkLimits};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4SourceMapCapacity {
    scaffold_tokens: usize,
    reserved_tokens: usize,
    max_source_tokens: usize,
    context_tokens: usize,
}
impl Int4SourceMapCapacity {
    pub fn scaffold_tokens(self) -> usize { self.scaffold_tokens }
    pub fn reserved_tokens(self) -> usize { self.reserved_tokens }
    pub fn max_source_tokens(self) -> usize { self.max_source_tokens }
    pub fn context_tokens(self) -> usize { self.context_tokens }
    /// Only tighten caller limits. Every actual chunk is compiled afterward.
    pub fn constrain_chunks(self, mut chunks: ChunkLimits) -> Result<ChunkLimits, Int4SourceError> {
        chunks.context_tokens = chunks.context_tokens.min(self.context_tokens);
        chunks.reserved_tokens = chunks.reserved_tokens.max(self.reserved_tokens);
        chunks.max_chunk_tokens = chunks.max_chunk_tokens.min(self.max_source_tokens);
        chunks.effective_token_limit().map_err(|_| SourcePlanningError::ContextBudget)?;
        Ok(chunks)
    }
}
impl SourceTaskPlanner {
    #[allow(clippy::too_many_arguments)]
    pub fn int4_map_capacity_with_control<C: DecodeStepControl>(&self, task: &SourceMapTask,
        budget: TaskBudget, context: &PlanContext<'_>, planning: SourcePlanningLimits,
        group_width: usize, control: &mut C) -> Result<Int4SourceMapCapacity, Int4SourceError> {
        checkpoint(control)?;
        constrained_int4::check_profile(context.execution_identity(), group_width)?;
        let (kind, schema) = match task {
            SourceMapTask::Ner(options) => (BuiltInTask::Ner, options.schema_source()?),
            SourceMapTask::Keyphrases(options) => (BuiltInTask::Keyphrases, options.schema_source()?),
            SourceMapTask::Summarize(options) => (BuiltInTask::Summarize, options.schema_source()?),
        };
        self.check_context_for_profile(kind, context, budget, planning,
            NumericsProfile::StrictQuantized { version: 2 })?;
        if schema.len() > planning.compiler.max_schema_bytes {
            return Err(SourcePlanningError::InputBudget.into());
        }
        Int4MemoryRequirement::for_context(planning.max_context_tokens, group_width)
            .map_err(|_| SourcePlanningError::ContextBudget)?;
        let mut scaffold = 0_usize;
        for fragment in render_fragments(kind, &schema)? {
            checkpoint(control)?;
            let ids = self.tokenizer.tokenizer().encode_ids_with_options(&fragment,
                EncodeOptions { add_bos: false, add_eos: false })
                .map_err(|_| SourcePlanningError::Contract("INT4 source map scaffold encoding"))?;
            scaffold = scaffold.checked_add(ids.len()).ok_or(SourcePlanningError::ContextBudget)?;
        }
        let output = budget.max_output_tokens as usize;
        let reserved = scaffold.checked_add(output).ok_or(SourcePlanningError::ContextBudget)?;
        let prompt_room = (budget.max_input_tokens as usize).checked_sub(scaffold)
            .ok_or(SourcePlanningError::ContextBudget)?;
        let context_room = planning.max_context_tokens.checked_sub(reserved)
            .ok_or(SourcePlanningError::ContextBudget)?;
        let available = prompt_room.min(context_room);
        if scaffold == 0 || output == 0 || available == 0 {
            return Err(SourcePlanningError::ContextBudget.into());
        }
        checkpoint(control)?;
        Ok(Int4SourceMapCapacity { scaffold_tokens: scaffold, reserved_tokens: reserved,
            max_source_tokens: available, context_tokens: planning.max_context_tokens })
    }
}
