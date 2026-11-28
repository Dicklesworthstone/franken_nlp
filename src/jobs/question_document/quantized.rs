//! Feature-independent memory and lifetime admission for INT8 document QA jobs.
//! Original source/question envelopes remain the durable item. These checked
//! quotes perform no IO and grant no allocation or native execution permit.
use crate::{
    batch::{BatchCode, BatchFault, BatchLimits, BatchWork,
        source::quantized::question::{Int8QuestionDocumentBatchConfig,
            memory::{Int8QuestionDocumentMemoryLimits, Int8QuestionDocumentMemoryRequirement}}},
    jobs::{JobInputMemory, JobLimits, JobWork, population::PopulationReadLimits},
    validation::grounded_fields::GroundingBudget,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int8QuestionDocumentJobMemoryLimits {
    pub document: Int8QuestionDocumentMemoryLimits,
    pub transport: PopulationReadLimits,
    pub journal_reserve_bytes: u64,
    pub serialization_reserve_bytes: u64,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int8QuestionDocumentJobMemoryRequirement {
    pub document: Int8QuestionDocumentMemoryRequirement,
    /// Common population and job-buffer reservation. Preparation/IO appear
    /// exactly once in the complete input quote, not twice.
    pub population_buffer_bytes: u64,
}
impl Int8QuestionDocumentJobMemoryLimits {
    pub fn requirements(self, config: Int8QuestionDocumentBatchConfig, job: JobLimits, input: JobInputMemory)
        -> Result<Int8QuestionDocumentJobMemoryRequirement, BatchFault> {
        check_limits(config, job)?;
        self.transport.validate().map_err(|_| BatchCode::InvalidLimits)?;
        if self.journal_reserve_bytes == 0 || self.serialization_reserve_bytes == 0 {
            return Err(BatchCode::InvalidLimits.into());
        }
        let output_line = job.max_result_bytes.max(4096);
        // One bounded record parser and every complete source/question owner.
        // The durable id index and chosen input lifetime are priced separately.
        let transport = BatchLimits { max_line_bytes: job.max_input_bytes_per_item,
            max_document_bytes: config.mapping.chunks.max_input_bytes,
            max_id_bytes: 1, max_epoch_ids: 1, max_epoch_id_bytes: 1, max_json_depth: 64,
            max_input_bytes: self.transport.max_stream_bytes, max_requests: job.max_items,
            max_output_line_bytes: output_line, max_output_bytes: product(output_line as u64, 2)?,
            max_work: BatchWork { forward_positions: job.max_work.model.forward_positions,
                projected_logits: job.max_work.model.projected_logits } };
        let mut document = self.document.requirements(config, transport).map_err(|_| BatchCode::InvalidLimits)?;
        let population_buffer_bytes = sum(&[input.buffer_bytes(job).map_err(|_| BatchCode::InvalidLimits)?,
            self.document.preparation_reserve_bytes, self.document.io_reserve_bytes,
            self.journal_reserve_bytes, self.serialization_reserve_bytes])?;
        let common = sum(&[self.document.preparation_reserve_bytes, self.document.io_reserve_bytes])?;
        let extra = population_buffer_bytes.checked_sub(common).ok_or(BatchCode::InvalidLimits)?;
        document.input_bytes = sum(&[document.input_bytes, extra])?;
        sum(&[document.input_bytes, document.native.kv_bytes, document.workspace_bytes,
            document.document_workspace_bytes, document.output_reservation_bytes])?;
        Ok(Int8QuestionDocumentJobMemoryRequirement { document, population_buffer_bytes })
    }
}

pub(crate) fn check_limits(config: Int8QuestionDocumentBatchConfig, limits: JobLimits) -> Result<(), BatchFault> {
    limits.validate().map_err(|_| BatchCode::InvalidLimits)?; config.validate()?;
    if limits.max_work != (JobWork { model: config.max_model_work, mask_node_visits: config.max_mask_visits })
        || config.mapping.chunks.max_input_bytes > limits.max_input_bytes_per_item
        || config.max_question_bytes > limits.max_input_bytes_per_item
        || config.mapping.reduction.max_result_bytes > limits.max_result_bytes {
        return Err(BatchCode::InvalidLimits.into());
    }
    // Verification has no journal columns. The immutable total attempt cap
    // therefore bounds all three counters across failures and resumed runs.
    // A new process-local ledger cannot renew this complete lifetime authority.
    verification_lifetime(config.verification, config.max_verification, limits.max_attempts)
}
fn verification_lifetime(per: GroundingBudget, lifetime: GroundingBudget, attempts: u64) -> Result<(), BatchFault> {
    let count = usize::try_from(attempts).map_err(|_| BatchCode::InvalidLimits)?;
    if count == 0 || per.max_fields.checked_mul(count).is_none_or(|n| n > lifetime.max_fields)
        || per.max_matches.checked_mul(count).is_none_or(|n| n > lifetime.max_matches)
        || per.max_scan_steps.checked_mul(attempts).is_none_or(|n| n > lifetime.max_scan_steps) {
        return Err(BatchCode::InvalidLimits.into());
    }
    Ok(())
}
fn product(value: u64, factor: u64) -> Result<u64, BatchFault> {
    value.checked_mul(factor).ok_or_else(|| BatchCode::InvalidLimits.into())
}
fn sum(values: &[u64]) -> Result<u64, BatchFault> {
    values.iter().try_fold(0_u64, |total, value| total.checked_add(*value).ok_or_else(|| BatchCode::InvalidLimits.into()))
}

#[cfg(test)] pub(crate) mod tests;
