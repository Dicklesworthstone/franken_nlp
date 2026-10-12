//! Explicit reduction settings inside the existing source-document config.
//! Available to model-free CLI parsing even when owned-job execution is absent.
use super::*;
use crate::{corpus::summarize::CorpusSummaryLimits,
    tasks::{corpus_keyphrases::CorpusKeyphraseLimits,
        source_planning::quantized::long::{summary::Int8SummaryLimits, keyphrase_reduction::Int8KeyphraseLimits}}};

#[derive(Clone, Copy, serde::Deserialize, serde::Serialize)]
#[serde(tag = "task", rename_all = "snake_case", deny_unknown_fields)]
pub(in crate::candidate_cli) enum ReductionSettings {
    Summarize { schema_version: u32, aggregation: CorpusSummaryLimits, max_bullets: usize,
        max_result_bytes: usize, max_lifetime_scan_work: u64 },
    Keyphrases { schema_version: u32, aggregation: PhraseAggregation, max_phrases: usize,
        max_result_bytes: usize, max_lifetime_scan_work: u64 },
}
#[derive(Clone, Copy, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub(in crate::candidate_cli) struct PhraseAggregation {
    max_unique_phrases: usize, max_evidence_spans: usize, max_value_bytes: usize, max_scan_work: u64,
}
impl PhraseAggregation {
    fn native(self) -> CorpusKeyphraseLimits {
        let Self { max_unique_phrases, max_evidence_spans, max_value_bytes, max_scan_work } = self;
        CorpusKeyphraseLimits { max_unique_phrases, max_evidence_spans, max_value_bytes, max_scan_work }
    }
}
/// Checked data, not an executable plan or native/model admission token.
pub(in crate::candidate_cli) enum PreparedReduction {
    Summary { limits: Int8SummaryLimits, max_lifetime_scan_work: u64 },
    Keyphrases { limits: Int8KeyphraseLimits, max_lifetime_scan_work: u64 },
}
impl ReductionSettings {
    pub(super) fn validate(self, task: &str, mapping: Int8SourceMapLimits, lifetime: JobLimits)
        -> Result<PreparedReduction, Failure> {
        lifetime.validate().map_err(|_| Failure::usage("job_limits"))?;
        let (prepared, scans, maximum, output) = match (self, task) {
            (Self::Summarize { schema_version: 1, aggregation, max_bullets, max_result_bytes, max_lifetime_scan_work }, "summarize") => {
                let limits = Int8SummaryLimits { aggregation, max_bullets, max_result_bytes };
                limits.validate(mapping).map_err(|_| Failure::usage("document_summary_reduction_limits"))?;
                (PreparedReduction::Summary { limits, max_lifetime_scan_work }, aggregation.max_scan_steps,
                    max_lifetime_scan_work, max_result_bytes)
            }
            (Self::Keyphrases { schema_version: 1, aggregation, max_phrases, max_result_bytes, max_lifetime_scan_work }, "keyphrases") => {
                let aggregation = aggregation.native();
                let limits = Int8KeyphraseLimits { aggregation, max_phrases, max_result_bytes };
                limits.validate(mapping).map_err(|_| Failure::usage("document_keyphrase_reduction_limits"))?;
                (PreparedReduction::Keyphrases { limits, max_lifetime_scan_work }, aggregation.max_scan_work,
                    max_lifetime_scan_work, max_result_bytes)
            }
            _ => return Err(Failure::usage("source_document_reduction_task_or_version")),
        };
        if scans == 0 || scans.checked_mul(lifetime.max_attempts).is_none_or(|n| n > maximum)
            || output > lifetime.max_result_bytes {
            return Err(Failure::usage("source_document_reduction_lifetime"));
        }
        Ok(prepared)
    }
}

#[cfg(test)] mod tests;
