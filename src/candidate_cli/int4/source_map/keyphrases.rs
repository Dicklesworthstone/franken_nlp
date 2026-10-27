//! Explicit exact document ranking; native per-chunk options stay independent.
use super::*;
use crate::tasks::corpus_keyphrases::CorpusKeyphraseLimits;

#[derive(Args, Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct KeyphraseArgs {
    /// Union every exact native phrase and its source evidence, then rank once.
    /// Raw independent chunk results remain the default without this flag.
    #[arg(long)]
    reduce_keyphrases: bool,
    /// Final document-wide phrase cap (default 16), separate from --options.
    #[arg(long, requires = "reduce_keyphrases")]
    document_keyphrases: Option<usize>,
    /// Complete candidate union before final selection (default 4096).
    #[arg(long, requires = "reduce_keyphrases")]
    max_unique_keyphrases: Option<usize>,
    /// All retained original-coordinate evidence spans (default 65536).
    #[arg(long, requires = "reduce_keyphrases")]
    max_keyphrase_evidence_spans: Option<usize>,
    /// Complete aggregate value cap before final selection (default 4194304).
    #[arg(long, requires = "reduce_keyphrases")]
    max_keyphrase_value_bytes: Option<usize>,
    /// Independent source reverification work across all chunks (default 536870912).
    #[arg(long, requires = "reduce_keyphrases")]
    max_keyphrase_scan_work: Option<u64>,
    /// Complete ranked-document envelope cap (default 4194304).
    #[arg(long, requires = "reduce_keyphrases")]
    max_keyphrase_result_bytes: Option<usize>,
}
pub(super) fn definition(command: clap::Command) -> clap::Command {
    KeyphraseArgs::augment_args(command
        .about("Extract every chunk's keyphrases or explicitly rank their complete exact union")
        .long_about("Process one complete UTF-8 document using native grouped INT4 keyphrase extraction. The default retains independent chunk results and local ranks. --reduce-keyphrases instead preserves every candidate through intermediate reductions, unions only byte-identical phrases and verified original-coordinate evidence, and applies --document-keyphrases once at the complete final root. Ranking uses distinct supporting-chunk count, sum of local ranks, first original byte offset and exact lexical bytes; it is not a probability or neural reranking. Multiple occurrences within one chunk give one support vote. Case, Unicode spelling and whitespace are not normalized. --options supplies separate per-chunk extraction settings. Complete candidate, evidence, verification, work and result budgets fail closed; no successful prefix or intermediate top-k pruning is published. Source conversion and digest-pinned snapshots use the same checked path. Model selection and chunk boundaries do not establish recall, phrase-boundary completeness or single-context equivalence."))
}
impl KeyphraseArgs {
    pub(super) fn limits(self, mapping: Int4SourceMapLimits)
        -> Result<Option<Int4KeyphraseLimits>, CandidateError> {
        if !self.reduce_keyphrases {
            if self.document_keyphrases.is_some() || self.max_unique_keyphrases.is_some()
                || self.max_keyphrase_evidence_spans.is_some() || self.max_keyphrase_value_bytes.is_some()
                || self.max_keyphrase_scan_work.is_some() || self.max_keyphrase_result_bytes.is_some() {
                return Err(CandidateError::Arguments);
            }
            return Ok(None);
        }
        let defaults = Int4KeyphraseLimits::default();
        let aggregation = CorpusKeyphraseLimits {
            max_unique_phrases: self.max_unique_keyphrases.unwrap_or(defaults.aggregation.max_unique_phrases),
            max_evidence_spans: self.max_keyphrase_evidence_spans.unwrap_or(defaults.aggregation.max_evidence_spans),
            max_value_bytes: self.max_keyphrase_value_bytes.unwrap_or(defaults.aggregation.max_value_bytes),
            max_scan_work: self.max_keyphrase_scan_work.unwrap_or(defaults.aggregation.max_scan_work),
        };
        if aggregation.max_scan_work > 1_000_000_000_000 { return Err(CandidateError::Arguments); }
        let limits = Int4KeyphraseLimits { aggregation,
            max_phrases: self.document_keyphrases.unwrap_or(defaults.max_phrases),
            max_result_bytes: self.max_keyphrase_result_bytes.unwrap_or(defaults.max_result_bytes) };
        limits.validate(mapping).map_err(|_| CandidateError::Arguments)?;
        Ok(Some(limits))
    }
}
pub(super) fn workspace_bytes(budget: TaskBudget, mapping: Int4SourceMapLimits,
    chunks: usize, reserve: u64, limits: Int4KeyphraseLimits) -> Result<u64, CandidateError> {
    limits.validate(mapping).map_err(|_| CandidateError::Arguments)?;
    let aggregation = limits.aggregation;
    sum(&[super::workspace_bytes(budget, mapping, chunks, reserve)?,
        times(aggregation.max_value_bytes as u64, 8)?,
        times(aggregation.max_unique_phrases as u64, 512)?,
        times(times(aggregation.max_evidence_spans as u64,
            std::mem::size_of::<VerifiedSourceSpan>() as u64)?, 2)?])
}

#[cfg(test)]
mod tests;
