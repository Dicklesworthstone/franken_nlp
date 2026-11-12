//! Shared modeled job-buffer floor, independent of a native runtime or model.
//! The selected input lifetime changes payload ownership, never logical limits
//! or execution authority. Hosts add their explicit preparation/IO/database and
//! serialization reserves and acquire the actual process-ledger permits.
use super::{JobError, JobLimits};

/// Keep the modeled diagnostic ancestry and the actual input parsers aligned.
pub(super) const INPUT_JSON_DEPTH: usize = 64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JobInputMemory {
    /// Original NDJSON is retained in memory for the whole invocation.
    CompletePopulation,
    /// Explicit retained creation reads and seals one original record at a
    /// time; authenticated replay then lends one retained record at a time.
    /// The complete bounded keyed manifest index remains resident in both.
    RetainedRecord,
}
impl JobInputMemory {
    /// Common input/index/frame/staging floor. This is an admission model,
    /// not an allocator/RSS measurement or permission to allocate memory.
    pub fn buffer_bytes(self, limits: JobLimits) -> Result<u64, JobError> {
        limits.validate()?;
        let population = match self {
            Self::CompletePopulation => product(limits.max_snapshot_bytes, 2)?,
            Self::RetainedRecord => 0,
        };
        sum(&[population, product(limits.max_items, 1024)?,
            product(limits.max_input_bytes_per_item as u64, 8)?,
            parser_workspace_bytes(limits.max_input_bytes_per_item as u64)?,
            product(limits.max_result_bytes as u64, 4)?, 2 * 1024 * 1024])
    }
}

/// Conservative payload model for the existing strict JSON chokepoint, including
/// malformed input before typed rejection. Counting only raw bytes underprices
/// dense arrays, sparse B-tree objects and copied diagnostic-path ancestry.
///
/// The pinned std B-tree has 11 entries and at most 12 child pointers per node;
/// 16 pointer words include its header/alignment. Price two such nodes per input
/// byte (including split overlap), plus four Value slots per input byte for
/// array capacity and growth. These intentionally loose simultaneous bounds
/// cover both duplicate-key preflight and Value construction without depending
/// on accepted task shapes. Decoded strings and parser scratch retain the
/// separate eight-byte-per-input-byte allowance above.
///
/// JsonPath escapes a key to at most twice its raw size, then copies prefixes
/// through depth-bounded recursion. Four times the record length covers path
/// growth; 42 bytes per level cover slash/index text and growth on 64-bit hosts.
/// Extra live paths cover construction and error propagation. Explicit host
/// serialization/allocator reserves remain mandatory; this is not an RSS cap.
fn parser_workspace_bytes(record: u64) -> Result<u64, JobError> {
    let word = std::mem::size_of::<usize>() as u64;
    let value = std::mem::size_of::<serde_json::Value>() as u64;
    let string = std::mem::size_of::<String>() as u64;
    let node = sum(&[product(word, 16)?, product(sum(&[string, value])?, 11)?])?;
    let objects = product(product(record, 2)?, node)?;
    let arrays = product(product(record, 4)?, value)?;
    let depth = INPUT_JSON_DEPTH as u64;
    let path = sum(&[product(record, 4)?, product(depth, 42)?, 64])?;
    sum(&[objects, arrays, product(path, depth + 4)?])
}
fn product(value: u64, factor: u64) -> Result<u64, JobError> {
    value.checked_mul(factor).ok_or(JobError::Limit)
}
fn sum(values: &[u64]) -> Result<u64, JobError> {
    values.iter().try_fold(0_u64, |total, value| total.checked_add(*value).ok_or(JobError::Limit))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jobs::tests::limits;

    #[test]
    fn streamed_retained_input_removes_only_the_complete_payload_reservation() {
        let job = limits();
        let complete = JobInputMemory::CompletePopulation.buffer_bytes(job).unwrap();
        let streamed = JobInputMemory::RetainedRecord.buffer_bytes(job).unwrap();
        assert_eq!(complete - streamed, 2 * job.max_snapshot_bytes);
        assert!(streamed >= 1024 * job.max_items + 8 * job.max_input_bytes_per_item as u64
            + 4 * job.max_result_bytes as u64 + 2 * 1024 * 1024);
    }

    #[test]
    fn streamed_payload_memory_scales_with_record_and_index_not_population_bytes() {
        let job = limits();
        let baseline = JobInputMemory::RetainedRecord.buffer_bytes(job).unwrap();
        let mut larger = job; larger.max_snapshot_bytes *= 1024;
        assert_eq!(JobInputMemory::RetainedRecord.buffer_bytes(larger).unwrap(), baseline);
        larger = job; larger.max_input_bytes_per_item *= 2;
        assert!(JobInputMemory::RetainedRecord.buffer_bytes(larger).unwrap() - baseline >
            8 * job.max_input_bytes_per_item as u64);
        larger = job; larger.max_items += 1;
        assert_eq!(JobInputMemory::RetainedRecord.buffer_bytes(larger).unwrap(), baseline + 1024);
    }

    #[test]
    fn checked_quotes_reject_invalid_limits_and_complete_payload_overflow() {
        let mut job = limits(); job.max_snapshot_bytes = i64::MAX as u64;
        assert_eq!(JobInputMemory::CompletePopulation.buffer_bytes(job), Err(JobError::Limit));
        assert!(JobInputMemory::RetainedRecord.buffer_bytes(job).is_ok());
        job.max_items = 0;
        for mode in [JobInputMemory::CompletePopulation, JobInputMemory::RetainedRecord] {
            assert_eq!(mode.buffer_bytes(job), Err(JobError::InvalidLimits));
        }
    }

    #[test]
    fn dense_json_arrays_have_a_separate_tree_allowance_before_typed_rejection() {
        let input = format!("[{}]", vec!["0"; 4096].join(","));
        let value = crate::canonjson::parse_str_with_limits(&input, crate::canonjson::ParseLimits {
            max_depth: INPUT_JSON_DEPTH, max_string_bytes: input.len(),
        }).unwrap();
        let serde_json::Value::Array(values) = value else { panic!("array fixture"); };
        let slots = values.capacity() * std::mem::size_of::<serde_json::Value>();
        assert!(slots > 8 * input.len(), "raw record allowance cannot cover this parser shape");
        assert!(parser_workspace_bytes(input.len() as u64).unwrap() > slots as u64);
    }

    #[test]
    fn copied_escaped_ancestry_is_priced_even_when_the_typed_envelope_is_invalid() {
        let key = "~/".repeat(4096);
        let levels = INPUT_JSON_DEPTH - 1;
        let input = format!("{{\"{key}\":{}0{}}}", "[".repeat(levels), "]".repeat(levels));
        crate::canonjson::parse_str_with_limits(&input, crate::canonjson::ParseLimits {
            max_depth: INPUT_JSON_DEPTH, max_string_bytes: input.len(),
        }).unwrap();
        let retained_prefix_bytes = (2 * key.len() + 1) * levels;
        assert!(retained_prefix_bytes > 8 * input.len());
        assert!(parser_workspace_bytes(input.len() as u64).unwrap() > retained_prefix_bytes as u64);
        assert_eq!(parser_workspace_bytes(u64::MAX), Err(JobError::Limit));
    }
}
