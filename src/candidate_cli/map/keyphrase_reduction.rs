//! Explicit document-wide keyphrase policy, distinct from native chunk options.
use super::*;
use crate::tasks::{corpus_keyphrases::CorpusKeyphraseLimits,
    source_planning::quantized::long::keyphrase_reduction::Int8KeyphraseLimits};

#[derive(Args)]
pub(in crate::candidate_cli) struct KeyphraseReductionArgs {
    /// Union exact native keyphrases and evidence, then rank the whole document.
    /// Requires --task keyphrases; no neural reranking or stemming is performed.
    #[arg(long, conflicts_with_all = ["reduce_summary", "synthesize_summary"])]
    pub reduce_keyphrases: bool,
    /// Final document-wide phrase cap (default 16), not each native chunk's cap.
    #[arg(long, requires = "reduce_keyphrases")]
    document_keyphrases: Option<usize>,
    /// Complete pre-selection phrase union (default 4096). Overflow fails.
    #[arg(long, requires = "reduce_keyphrases")]
    max_unique_keyphrases: Option<usize>,
    /// Complete original-coordinate occurrence evidence (default 65536).
    #[arg(long, requires = "reduce_keyphrases")]
    max_keyphrase_evidence_spans: Option<usize>,
    /// One independent verification-work allowance across all chunks (default 536870912).
    #[arg(long, requires = "reduce_keyphrases")]
    max_keyphrase_scan_work: Option<u64>,
}
impl KeyphraseReductionArgs {
    pub(super) fn validate(&self, task: &str, bytes: usize) -> Result<(), CandidateError> {
        if self.reduce_keyphrases && task != "keyphrases" { return Err(CandidateError::Arguments); }
        self.limits(bytes).map(|_| ())
    }
    pub(in crate::candidate_cli) fn limits(&self, bytes: usize) -> Result<Option<Int8KeyphraseLimits>, CandidateError> {
        if !self.reduce_keyphrases {
            if self.document_keyphrases.is_some() || self.max_unique_keyphrases.is_some()
                || self.max_keyphrase_evidence_spans.is_some() || self.max_keyphrase_scan_work.is_some() {
                return Err(CandidateError::Arguments);
            }
            return Ok(None);
        }
        let defaults = CorpusKeyphraseLimits::default();
        let aggregation = CorpusKeyphraseLimits {
            max_unique_phrases: self.max_unique_keyphrases.unwrap_or(defaults.max_unique_phrases),
            max_evidence_spans: self.max_keyphrase_evidence_spans.unwrap_or(defaults.max_evidence_spans),
            max_value_bytes: bytes,
            max_scan_work: self.max_keyphrase_scan_work.unwrap_or(defaults.max_scan_work),
        };
        aggregation.validate().map_err(|_| CandidateError::Arguments)?;
        let max_phrases = self.document_keyphrases.unwrap_or(16);
        if !(1..=4096).contains(&max_phrases) || aggregation.max_scan_work > 1_000_000_000_000 {
            return Err(CandidateError::Arguments);
        }
        Ok(Some(Int8KeyphraseLimits { aggregation, max_phrases, max_result_bytes: bytes }))
    }
}

#[cfg(test)] mod tests;
