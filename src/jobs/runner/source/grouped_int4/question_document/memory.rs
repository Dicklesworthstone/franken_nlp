//! One ownership quotation for the authenticated population and complete QA.
//! This is shared by the CLI and host before population IO. It models retained
//! allocations and declared slack; it is neither a permit nor an RSS measure.
use crate::{
    batch::{BatchCode, BatchFault, BatchLimits, BatchWork,
        source::grouped_int4::question::{Int4QuestionDocumentBatchConfig,
            memory::{Int4QuestionDocumentMemoryLimits, Int4QuestionDocumentMemoryRequirement}}},
    jobs::{JobInputMemory, JobLimits, population::PopulationReadLimits},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4QuestionDocumentJobMemoryLimits {
    pub document: Int4QuestionDocumentMemoryLimits,
    pub transport: PopulationReadLimits,
    pub journal_reserve_bytes: u64,
    pub serialization_reserve_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4QuestionDocumentJobMemoryRequirement {
    /// Complete native, population, document frontier and output ownership.
    /// `input_bytes` includes the authenticated population and job buffers.
    pub document: Int4QuestionDocumentMemoryRequirement,
    /// Exactly the common JobHostLimits population reservation. Preparation
    /// and IO appear once in the complete input quote, not twice.
    pub population_buffer_bytes: u64,
}

impl Int4QuestionDocumentJobMemoryLimits {
    pub fn requirements(self, width: usize, config: Int4QuestionDocumentBatchConfig,
        job: JobLimits, input: JobInputMemory) -> Result<Int4QuestionDocumentJobMemoryRequirement, BatchFault> {
        super::check_limits(config, job)?;
        self.transport.validate().map_err(|_| BatchCode::InvalidLimits)?;
        if self.journal_reserve_bytes == 0 || self.serialization_reserve_bytes == 0
            || config.mapping.chunks.max_input_bytes > job.max_input_bytes_per_item
            || config.max_question_bytes > job.max_input_bytes_per_item
            || config.mapping.reduction.max_result_bytes > job.max_result_bytes {
            return Err(BatchCode::InvalidLimits.into());
        }
        let output_line = job.max_result_bytes.max(4096);
        // The shared complete-document quote prices one bounded record parser
        // and every source/question/identity representation. The job's complete
        // id index and selected input lifetime are accounted independently below;
        // live epoch limits do not restrict durable item ids or populations.
        let transport = BatchLimits { max_line_bytes: job.max_input_bytes_per_item,
            max_document_bytes: config.mapping.chunks.max_input_bytes,
            max_id_bytes: 1, max_epoch_ids: 1, max_epoch_id_bytes: 1, max_json_depth: 64,
            max_input_bytes: self.transport.max_stream_bytes, max_requests: job.max_items,
            max_output_line_bytes: output_line, max_output_bytes: product(output_line as u64, 2)?,
            max_work: BatchWork { forward_positions: job.max_work.model.forward_positions,
                projected_logits: job.max_work.model.projected_logits } };
        let mut document = self.document.requirements(width, config, transport)
            .map_err(|_| BatchCode::InvalidLimits)?;
        let population_buffer_bytes = sum(&[
            input.buffer_bytes(job).map_err(|_| BatchCode::InvalidLimits)?,
            self.document.preparation_reserve_bytes, self.document.io_reserve_bytes,
            self.journal_reserve_bytes, self.serialization_reserve_bytes,
        ])?;
        let shared = sum(&[self.document.preparation_reserve_bytes, self.document.io_reserve_bytes])?;
        let extra = population_buffer_bytes.checked_sub(shared).ok_or(BatchCode::InvalidLimits)?;
        document.input_bytes = sum(&[document.input_bytes, extra])?;
        sum(&[document.input_bytes, document.native.kv_bytes, document.workspace_bytes,
            document.document_workspace_bytes, document.output_reservation_bytes])?;
        Ok(Int4QuestionDocumentJobMemoryRequirement { document, population_buffer_bytes })
    }
}

fn product(value: u64, factor: u64) -> Result<u64, BatchFault> {
    value.checked_mul(factor).ok_or_else(|| BatchCode::InvalidLimits.into())
}
fn sum(values: &[u64]) -> Result<u64, BatchFault> {
    values.iter().try_fold(0_u64, |total, value| total.checked_add(*value)
        .ok_or_else(|| BatchCode::InvalidLimits.into()))
}

#[cfg(test)] mod tests;
