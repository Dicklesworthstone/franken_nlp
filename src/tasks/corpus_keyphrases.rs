//! Long-document keyphrases: real native maps, deterministic exact reduction.
//!
//! All candidates survive intermediate reductions or the whole operation fails
//! its budget. Applying top-k at each level would lose globally frequent phrases
//! and make the answer depend on fan-in. Top-k is applied only at publication.
//! No case folding, stemming, fuzzy union, hidden score calibration or new pool.

use std::{collections::{BTreeMap, BTreeSet}, mem::size_of};
use serde::Serialize;
use crate::{
    canonjson,
    execution_identity::{ExecutionIdentity, Sha256Digest},
    grammar::{CompileLimits, runtime::SourceRuntimeLimits},
    native_engine::{constrained::{JsonDecodeOptions, JsonWorkBudget},
        decode::DecodeStepControl, hf_bf16_eager::{HfBf16EagerEngine, HF_BF16_EAGER_PROFILE}},
    tokenizer::specials::TemplateControlIds,
    validation::grounded_fields::{SourceOccurrence, VerifiedSourceSpan},
};
use super::{
    extract::{ExtractError, ExtractionGrounding, ExtractionVocabulary, SourceDocument, SourceDocumentEncoder},
    ir::{PromptSegmentKind, ScoreSpace, TaskPlan},
    keyphrases::{KeyphraseError, KeyphraseOptions, KeyphrasePlan, KeyphraseResult,
        KEYPHRASES_RANKING, KEYPHRASES_TASK_VERSION},
    mapreduce::{MapOutput, MapReduceTask, ReduceInput, ReductionPolicy, SourceChunk},
};

pub const CORPUS_KEYPHRASE_POLICY: &str = "exact-union-support-rank-sum-source-v1";

/// Static composition boundary. NativeKeyphrasePass is the concrete production
/// implementation. Returned phrase membership and complete occurrences are
/// independently rechecked; a deserialized result is not trusted as evidence.
pub trait KeyphrasePass {
    fn options(&self) -> KeyphraseOptions;
    fn run(&mut self, chunk: &SourceChunk<'_>) -> Result<KeyphraseResult, KeyphraseError>;
}

#[derive(Clone, Copy, Debug)]
pub struct CorpusKeyphraseLimits {
    pub max_unique_phrases: usize,
    pub max_evidence_spans: usize,
    /// Bounds canonical aggregate bytes AND a conservative stored-field charge;
    /// neither is advertised as allocator-observed peak heap size.
    pub max_value_bytes: usize,
    /// Aggregate linear matching work across maps, charged before scanning.
    pub max_scan_work: u64,
}
impl Default for CorpusKeyphraseLimits {
    fn default() -> Self {
        Self { max_unique_phrases: 4096, max_evidence_spans: 65_536,
            max_value_bytes: 4 * 1024 * 1024, max_scan_work: 512 * 1024 * 1024 }
    }
}
impl CorpusKeyphraseLimits {
    fn validate(self) -> Result<(), KeyphraseError> {
        if !(1..=65_536).contains(&self.max_unique_phrases)
            || !(1..=1_000_000).contains(&self.max_evidence_spans)
            || !(1..=64 * 1024 * 1024).contains(&self.max_value_bytes) || self.max_scan_work == 0
        { return Err(KeyphraseError::InvalidOptions); }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ChunkPhraseEvidence {
    pub chunk_id: usize,
    pub local_rank: usize,
    /// Exact ORIGINAL-document coordinates, not chunk-relative coordinates.
    pub spans: Vec<VerifiedSourceSpan>,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CorpusPhrase {
    pub text: String,
    pub rank_sum: u64,
    /// Sorted, distinct chunk ids; one vote per chunk, not per occurrence.
    pub evidence: Vec<ChunkPhraseEvidence>,
}

/// Minted only after source verification. Sorted lexically during reduction;
/// it contains all candidates until into_ranked applies the final user limit.
#[derive(Serialize)]
pub struct KeyphraseAggregate {
    phrases: Vec<CorpusPhrase>,
    mapped_chunks: usize,
    forward_positions: u64,
    projected_logits: u64,
    mask_node_visit_charge: u64,
}
impl KeyphraseAggregate {
    pub fn phrases(&self) -> &[CorpusPhrase] { &self.phrases }
    pub fn into_ranked(mut self, max_phrases: usize, max_bytes: usize) -> Result<CorpusKeyphraseResult, KeyphraseError> {
        if !(1..=4096).contains(&max_phrases) || !(1..=64 * 1024 * 1024).contains(&max_bytes) {
            return Err(KeyphraseError::InvalidOptions);
        }
        self.phrases.sort_by(|a, b| b.evidence.len().cmp(&a.evidence.len())
            .then_with(|| a.rank_sum.cmp(&b.rank_sum))
            .then_with(|| first_byte(a).cmp(&first_byte(b)))
            .then_with(|| a.text.as_bytes().cmp(b.text.as_bytes())));
        let omitted_candidates = self.phrases.len().saturating_sub(max_phrases);
        self.phrases.truncate(max_phrases);
        let result = CorpusKeyphraseResult { schema_version: 1,
            task_spec_version: KEYPHRASES_TASK_VERSION.to_owned(), ranking_policy: CORPUS_KEYPHRASE_POLICY.to_owned(),
            score_space: ScoreSpace::NotComputed, grounding: ExtractionGrounding::SourceMembership,
            mapped_chunks: self.mapped_chunks, omitted_candidates, phrases: self.phrases,
            forward_positions: self.forward_positions, projected_logits: self.projected_logits,
            mask_node_visit_charge: self.mask_node_visit_charge,
            warnings: [CorpusKeyphraseWarning::ChunkBoundariesMaySplitPhrases,
                CorpusKeyphraseWarning::SingleContextEquivalenceNotEstablished,
                CorpusKeyphraseWarning::ModelSelectionIsNotRecallGuarantee],
        };
        check_bytes(&result, max_bytes)?;
        Ok(result)
    }
}
fn first_byte(phrase: &CorpusPhrase) -> usize {
    phrase.evidence.first().and_then(|e| e.spans.first()).map_or(usize::MAX, |s| s.byte_start)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CorpusKeyphraseWarning {
    ChunkBoundariesMaySplitPhrases, SingleContextEquivalenceNotEstablished, ModelSelectionIsNotRecallGuarantee,
}
#[derive(Serialize)]
pub struct CorpusKeyphraseResult {
    pub schema_version: u32,
    pub task_spec_version: String,
    pub ranking_policy: String,
    pub score_space: ScoreSpace,
    pub grounding: ExtractionGrounding,
    pub mapped_chunks: usize,
    pub omitted_candidates: usize,
    /// Array order is rank. Supporting-chunk count is evidence.len(), not a
    /// probability. Repeated occurrences in one chunk never add extra votes.
    pub phrases: Vec<CorpusPhrase>,
    pub forward_positions: u64,
    pub projected_logits: u64,
    pub mask_node_visit_charge: u64,
    pub warnings: [CorpusKeyphraseWarning; 3],
}

/// Pass to tasks::mapreduce::execute with an admitted ChunkPlan. The shared
/// executor provides completion ordering, lineage, tree and whole-run byte caps.
/// A failed map/reduce poisons this task, including unwinds during model work.
pub struct CorpusKeyphraseTask<P> {
    pass: P,
    options: KeyphraseOptions,
    limits: CorpusKeyphraseLimits,
    scan_remaining: u64,
    failed: bool,
}
impl<P: KeyphrasePass> CorpusKeyphraseTask<P> {
    pub fn new(pass: P, limits: CorpusKeyphraseLimits) -> Result<Self, KeyphraseError> {
        limits.validate()?;
        let options = pass.options(); options.validate()?;
        Ok(Self { pass, options, limits, scan_remaining: limits.max_scan_work, failed: false })
    }
    pub fn scan_work_remaining(&self) -> u64 { self.scan_remaining }
    pub fn pass(&self) -> &P { &self.pass }
}
impl<P: KeyphrasePass> MapReduceTask for CorpusKeyphraseTask<P> {
    type Value = KeyphraseAggregate;
    type Error = KeyphraseError;
    fn policy(&self) -> ReductionPolicy {
        ReductionPolicy { id: CORPUS_KEYPHRASE_POLICY, may_discard_information: true }
    }
    fn map_batch(&mut self, chunks: &[SourceChunk<'_>]) -> Result<Vec<MapOutput<Self::Value>>, Self::Error> {
        if self.failed || self.pass.options() != self.options { return Err(KeyphraseError::InvalidOptions); }
        self.failed = true;
        let mut outputs = Vec::new();
        outputs.try_reserve_exact(chunks.len()).map_err(|_| KeyphraseError::AllocationRefused)?;
        // Sequential native maps reuse one admitted engine. This adapter makes
        // no batched-GEMM claim; the outer map batch is an orchestration batch.
        for chunk in chunks {
            let raw = self.pass.run(chunk)?;
            let value = map_value(chunk, raw, self.options, self.limits, &mut self.scan_remaining)?;
            outputs.push(MapOutput { chunk_id: chunk.id(), value });
        }
        self.failed = false;
        Ok(outputs)
    }
    fn reduce(&mut self, input: ReduceInput<'_, Self::Value>) -> Result<Self::Value, Self::Error> {
        if self.failed { return Err(KeyphraseError::InvalidOptions); }
        self.failed = true;
        let value = merge_values(input.children.iter().map(|node| node.value()), self.limits)?;
        self.failed = false;
        Ok(value)
    }
}

fn map_value(chunk: &SourceChunk<'_>, raw: KeyphraseResult, options: KeyphraseOptions,
    limits: CorpusKeyphraseLimits, scan_remaining: &mut u64) -> Result<KeyphraseAggregate, KeyphraseError> {
    if raw.schema_version != 1 || raw.task_spec_version != KEYPHRASES_TASK_VERSION
        || raw.numerics_profile != HF_BF16_EAGER_PROFILE || raw.ranking_policy != KEYPHRASES_RANKING
        || raw.score_space != ScoreSpace::NotComputed || raw.grounding != ExtractionGrounding::SourceMembership
        || raw.phrases.len() > options.max_phrases || raw.phrases.len() > limits.max_unique_phrases
    { return Err(KeyphraseError::InvalidResult); }
    let mut phrases = Vec::new(); let mut seen = BTreeSet::new(); let mut span_count = 0_usize;
    phrases.try_reserve_exact(raw.phrases.len()).map_err(|_| KeyphraseError::AllocationRefused)?;
    for (index, phrase) in raw.phrases.iter().enumerate() {
        if phrase.rank != index + 1 || phrase.text.is_empty()
            || phrase.text.chars().count() > options.max_phrase_scalars || !seen.insert(phrase.text.as_str())
        { return Err(KeyphraseError::InvalidResult); }
        let local = occurrences(chunk.text(), &phrase.text, limits.max_evidence_spans, scan_remaining)?;
        if local.is_empty() || local != phrase.spans
            || phrase.occurrence != if local.len() == 1 { SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous }
        { return Err(KeyphraseError::InvalidResult); }
        span_count = span_count.checked_add(local.len()).filter(|&n| n <= limits.max_evidence_spans)
            .ok_or(KeyphraseError::OutputBudgetExceeded)?;
        let mut spans = local;
        let origin = chunk.span();
        for span in &mut spans {
            span.byte_start = span.byte_start.checked_add(origin.byte_start).ok_or(KeyphraseError::InvalidResult)?;
            span.byte_end = span.byte_end.checked_add(origin.byte_start).ok_or(KeyphraseError::InvalidResult)?;
            span.scalar_start = span.scalar_start.checked_add(origin.scalar_start).ok_or(KeyphraseError::InvalidResult)?;
            span.scalar_end = span.scalar_end.checked_add(origin.scalar_start).ok_or(KeyphraseError::InvalidResult)?;
        }
        phrases.push(CorpusPhrase { text: copy_text(&phrase.text)?, rank_sum: phrase.rank as u64,
            evidence: vec![ChunkPhraseEvidence { chunk_id: chunk.id(), local_rank: phrase.rank, spans }] });
    }
    phrases.sort_by(|a, b| a.text.cmp(&b.text));
    let value = KeyphraseAggregate { phrases, mapped_chunks: 1, forward_positions: raw.forward_positions,
        projected_logits: raw.projected_logits, mask_node_visit_charge: raw.mask_node_visit_charge };
    check_shape(&value, limits)?; check_bytes(&value, limits.max_value_bytes)?;
    Ok(value)
}

/// Linear KMP search retains overlapping occurrences. Scalar counting advances
/// monotonically with matches, avoiding a quadratic rescan for repeated text.
fn occurrences(source: &str, text: &str, max_spans: usize, remaining: &mut u64) -> Result<Vec<VerifiedSourceSpan>, KeyphraseError> {
    if text.is_empty() { return Err(KeyphraseError::InvalidResult); }
    let work = (source.len() as u64).checked_add(text.len() as u64).and_then(|n| n.checked_mul(4))
        .ok_or(KeyphraseError::OutputBudgetExceeded)?;
    *remaining = remaining.checked_sub(work).ok_or(KeyphraseError::OutputBudgetExceeded)?;
    let needle = text.as_bytes(); let mut prefix = Vec::new();
    prefix.try_reserve_exact(needle.len()).map_err(|_| KeyphraseError::AllocationRefused)?;
    prefix.resize(needle.len(), 0_usize);
    let mut matched = 0;
    for i in 1..needle.len() {
        while matched > 0 && needle[i] != needle[matched] { matched = prefix[matched - 1]; }
        if needle[i] == needle[matched] { matched += 1; }
        prefix[i] = matched;
    }
    let mut spans = Vec::new(); matched = 0;
    let (mut previous_byte, mut previous_scalar) = (0, 0);
    let scalar_length = text.chars().count();
    for (i, &byte) in source.as_bytes().iter().enumerate() {
        while matched > 0 && byte != needle[matched] { matched = prefix[matched - 1]; }
        if byte == needle[matched] { matched += 1; }
        if matched == needle.len() {
            let end = i + 1; let start = end - needle.len();
            if !source.is_char_boundary(start) || !source.is_char_boundary(end) { return Err(KeyphraseError::InvalidResult); }
            if spans.len() == max_spans { return Err(KeyphraseError::OutputBudgetExceeded); }
            previous_scalar += source[previous_byte..start].chars().count(); previous_byte = start;
            spans.try_reserve(1).map_err(|_| KeyphraseError::AllocationRefused)?;
            spans.push(VerifiedSourceSpan { byte_start: start, byte_end: end,
                scalar_start: previous_scalar, scalar_end: previous_scalar + scalar_length });
            matched = prefix[matched - 1];
        }
    }
    Ok(spans)
}

fn copy_text(text: &str) -> Result<String, KeyphraseError> {
    let mut copy = String::new();
    copy.try_reserve_exact(text.len()).map_err(|_| KeyphraseError::AllocationRefused)?;
    copy.push_str(text); Ok(copy)
}
fn check_bytes(value: &impl Serialize, cap: usize) -> Result<(), KeyphraseError> {
    if canonjson::canonical_bytes(value).map_err(|_| KeyphraseError::Serialization)?.len() > cap {
        return Err(KeyphraseError::OutputBudgetExceeded);
    }
    Ok(())
}
fn check_shape(value: &KeyphraseAggregate, limits: CorpusKeyphraseLimits) -> Result<(), KeyphraseError> {
    let mut spans = 0_usize; let mut bytes = 0_usize;
    if value.phrases.len() > limits.max_unique_phrases { return Err(KeyphraseError::OutputBudgetExceeded); }
    for phrase in &value.phrases {
        bytes = bytes.checked_add(phrase.text.len() + size_of::<CorpusPhrase>()).ok_or(KeyphraseError::OutputBudgetExceeded)?;
        for evidence in &phrase.evidence {
            spans = spans.checked_add(evidence.spans.len()).filter(|&n| n <= limits.max_evidence_spans)
                .ok_or(KeyphraseError::OutputBudgetExceeded)?;
            let added = evidence.spans.len().checked_mul(size_of::<VerifiedSourceSpan>())
                .and_then(|n| n.checked_add(size_of::<ChunkPhraseEvidence>())).ok_or(KeyphraseError::OutputBudgetExceeded)?;
            bytes = bytes.checked_add(added).ok_or(KeyphraseError::OutputBudgetExceeded)?;
        }
        if bytes > limits.max_value_bytes { return Err(KeyphraseError::OutputBudgetExceeded); }
    }
    Ok(())
}

fn merge_values<'a>(values: impl IntoIterator<Item = &'a KeyphraseAggregate>, limits: CorpusKeyphraseLimits)
    -> Result<KeyphraseAggregate, KeyphraseError> {
    let mut merged: BTreeMap<&str, CorpusPhrase> = BTreeMap::new();
    let mut total = KeyphraseAggregate { phrases: Vec::new(), mapped_chunks: 0,
        forward_positions: 0, projected_logits: 0, mask_node_visit_charge: 0 };
    let (mut spans, mut stored_bytes) = (0_usize, 0_usize);
    for value in values {
        total.mapped_chunks = total.mapped_chunks.checked_add(value.mapped_chunks).ok_or(KeyphraseError::InvalidResult)?;
        total.forward_positions = total.forward_positions.checked_add(value.forward_positions).ok_or(KeyphraseError::InvalidResult)?;
        total.projected_logits = total.projected_logits.checked_add(value.projected_logits).ok_or(KeyphraseError::InvalidResult)?;
        total.mask_node_visit_charge = total.mask_node_visit_charge.checked_add(value.mask_node_visit_charge).ok_or(KeyphraseError::InvalidResult)?;
        for phrase in &value.phrases {
            let new = !merged.contains_key(phrase.text.as_str());
            if new {
                if merged.len() == limits.max_unique_phrases { return Err(KeyphraseError::OutputBudgetExceeded); }
                stored_bytes = stored_bytes.checked_add(phrase.text.len() + size_of::<CorpusPhrase>())
                    .ok_or(KeyphraseError::OutputBudgetExceeded)?;
            }
            for item in &phrase.evidence {
                spans = spans.checked_add(item.spans.len()).filter(|&n| n <= limits.max_evidence_spans)
                    .ok_or(KeyphraseError::OutputBudgetExceeded)?;
                let bytes = item.spans.len().checked_mul(size_of::<VerifiedSourceSpan>())
                    .and_then(|n| n.checked_add(size_of::<ChunkPhraseEvidence>())).ok_or(KeyphraseError::OutputBudgetExceeded)?;
                stored_bytes = stored_bytes.checked_add(bytes).ok_or(KeyphraseError::OutputBudgetExceeded)?;
            }
            if stored_bytes > limits.max_value_bytes { return Err(KeyphraseError::OutputBudgetExceeded); }
            if new {
                merged.insert(phrase.text.as_str(), CorpusPhrase { text: copy_text(&phrase.text)?, rank_sum: 0, evidence: Vec::new() });
            }
            let target = merged.get_mut(phrase.text.as_str()).ok_or(KeyphraseError::InvalidResult)?;
            if target.evidence.last().zip(phrase.evidence.first()).is_some_and(|(a, b)| a.chunk_id >= b.chunk_id) {
                return Err(KeyphraseError::InvalidResult);
            }
            target.rank_sum = target.rank_sum.checked_add(phrase.rank_sum).ok_or(KeyphraseError::InvalidResult)?;
            target.evidence.try_reserve(phrase.evidence.len()).map_err(|_| KeyphraseError::AllocationRefused)?;
            for evidence in &phrase.evidence {
                let mut copy = Vec::new();
                copy.try_reserve_exact(evidence.spans.len()).map_err(|_| KeyphraseError::AllocationRefused)?;
                copy.extend_from_slice(&evidence.spans);
                target.evidence.push(ChunkPhraseEvidence { chunk_id: evidence.chunk_id, local_rank: evidence.local_rank, spans: copy });
            }
        }
    }
    total.phrases.try_reserve_exact(merged.len()).map_err(|_| KeyphraseError::AllocationRefused)?;
    total.phrases.extend(merged.into_values());
    check_bytes(&total, limits.max_value_bytes)?;
    Ok(total)
}

pub struct NativeKeyphraseConfig {
    pub options: KeyphraseOptions,
    pub decode: JsonDecodeOptions,
    pub compiler: CompileLimits,
    pub source: SourceRuntimeLimits,
    pub max_source_bytes: usize,
    pub max_source_tokens: usize,
    /// One work budget for ALL chunks. KV is a residency ceiling, not a counter.
    pub total_work: JsonWorkBudget,
}

/// Reuses the caller's admitted native engine and pinned encoder/vocabulary.
/// The factory supplies trusted TaskIR scaffolds, never flattened source prose.
pub struct NativeKeyphrasePass<'a, F, C> {
    engine: &'a mut HfBf16EagerEngine,
    encoder: &'a SourceDocumentEncoder,
    vocabulary: &'a ExtractionVocabulary,
    controls: &'a TemplateControlIds,
    control: &'a mut C,
    factory: F,
    config: NativeKeyphraseConfig,
    remaining: JsonWorkBudget,
    failed: bool,
    contract: Option<Sha256Digest>,
}
impl<'a, F, C> NativeKeyphrasePass<'a, F, C>
where F: FnMut(&SourceDocument, KeyphraseOptions) -> Result<(TaskPlan, ExecutionIdentity), KeyphraseError>,
      C: DecodeStepControl {
    #[allow(clippy::too_many_arguments)]
    pub fn new(engine: &'a mut HfBf16EagerEngine, encoder: &'a SourceDocumentEncoder,
        vocabulary: &'a ExtractionVocabulary, controls: &'a TemplateControlIds, control: &'a mut C,
        config: NativeKeyphraseConfig, factory: F) -> Result<Self, KeyphraseError> {
        config.options.validate()?;
        if config.max_source_bytes == 0 || config.max_source_tokens == 0 { return Err(KeyphraseError::InvalidOptions); }
        let remaining = config.total_work;
        Ok(Self { engine, encoder, vocabulary, controls, control, factory, config,
            remaining, failed: false, contract: None })
    }
    pub fn remaining_work(&self) -> JsonWorkBudget { self.remaining }
}
impl<F, C> KeyphrasePass for NativeKeyphrasePass<'_, F, C>
where F: FnMut(&SourceDocument, KeyphraseOptions) -> Result<(TaskPlan, ExecutionIdentity), KeyphraseError>,
      C: DecodeStepControl {
    fn options(&self) -> KeyphraseOptions { self.config.options }
    fn run(&mut self, chunk: &SourceChunk<'_>) -> Result<KeyphraseResult, KeyphraseError> {
        if self.failed { return Err(KeyphraseError::InvalidOptions); }
        let document = self.encoder.encode(chunk.text(), self.config.max_source_bytes, self.config.max_source_tokens)?;
        if document.token_ids().len() != chunk.tokens() {
            return Err(ExtractError::Contract("chunk token count differs from pinned source encoding").into());
        }
        let (task, identity) = (self.factory)(&document, self.config.options)?;
        let plan = KeyphrasePlan::from_task_plan(&task, &document, self.config.options, self.config.decode.clone(),
            self.config.compiler, self.controls, self.config.source)?;
        let identity = plan.bind_identity(identity)?;
        let observed = pass_contract(&task, &identity)?;
        if self.contract.is_some_and(|saved| saved != observed) {
            self.failed = true;
            return Err(ExtractError::Contract("keyphrase model/scaffold/policy changed between chunks").into());
        }
        self.contract = Some(observed);
        self.failed = true;
        let result = plan.execute_eager(self.engine, &identity, self.vocabulary, self.remaining, self.control)?;
        self.remaining = charge(self.remaining, &result)?;
        self.failed = false;
        Ok(result)
    }
}
fn pass_contract(task: &TaskPlan, identity: &ExecutionIdentity) -> Result<Sha256Digest, KeyphraseError> {
    let mut invariant = identity.clone();
    let erased = Sha256Digest::of_bytes(b"keyphrases-source-erasure-v1");
    invariant.prompt_digest = erased; invariant.taskir_digest = erased;
    let scaffold: Vec<_> = task.ir().prompt_segments().iter().filter(|s| s.kind() != PromptSegmentKind::Document).collect();
    let identity_bytes = invariant.canonical_json_bytes().map_err(|_| KeyphraseError::Serialization)?;
    Ok(Sha256Digest::of_bytes(&canonjson::canonical_bytes(
        &("keyphrases-native-contract-v1", identity_bytes, scaffold, task.ir().budget())
    ).map_err(|_| KeyphraseError::Serialization)?))
}
fn charge(mut left: JsonWorkBudget, result: &KeyphraseResult) -> Result<JsonWorkBudget, KeyphraseError> {
    left.max_forward_positions = left.max_forward_positions.checked_sub(result.forward_positions).ok_or(KeyphraseError::InvalidResult)?;
    left.max_projected_logits = left.max_projected_logits.checked_sub(result.projected_logits).ok_or(KeyphraseError::InvalidResult)?;
    left.max_total_mask_node_visits = left.max_total_mask_node_visits.checked_sub(result.mask_node_visit_charge).ok_or(KeyphraseError::InvalidResult)?;
    Ok(left)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{grammar::mask::MaskWorkLimits, tasks::{keyphrases::RankedKeyphrase,
        mapreduce::{self, ChunkPlan, ChunkLimits, ExecutionLimits}}, validation::{validate_source_span, SourceSpan}};

    struct Fixture { corrupt: bool }
    impl KeyphrasePass for Fixture {
        fn options(&self) -> KeyphraseOptions { KeyphraseOptions::default() }
        fn run(&mut self, chunk: &SourceChunk<'_>) -> Result<KeyphraseResult, KeyphraseError> {
            let mut phrases = Vec::new();
            for text in ["Rust", "rust", "上海", "aba", "abc", "def"] {
                let spans = occurrences(chunk.text(), text, 100, &mut 1_000_000).unwrap();
                if spans.is_empty() { continue; }
                phrases.push(RankedKeyphrase { rank: phrases.len() + 1, text: text.to_owned(),
                    occurrence: if spans.len() == 1 { SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous }, spans });
            }
            if self.corrupt && !phrases.is_empty() { phrases[0].spans[0].byte_start += 1; }
            Ok(KeyphraseResult { schema_version: 1, task_spec_version: KEYPHRASES_TASK_VERSION.to_owned(),
                numerics_profile: HF_BF16_EAGER_PROFILE.to_owned(), ranking_policy: KEYPHRASES_RANKING.to_owned(),
                score_space: ScoreSpace::NotComputed, grounding: ExtractionGrounding::SourceMembership, phrases,
                generated_token_ids: vec![1, 0], forward_positions: 2, projected_logits: 20, mask_node_visit_charge: 200 })
        }
    }
    fn chunks(source: &str, bytes: usize) -> ChunkPlan<'_> {
        ChunkPlan::build(source, ChunkLimits { max_chunk_bytes: bytes, ..ChunkLimits::default() }, |s| Ok(s.len())).unwrap()
    }
    #[test]
    fn overlapping_and_unicode_matches_have_exact_coordinates() {
        let source = "é上海 上海 ababa";
        for text in ["上海", "aba", "é", " "] {
            let spans = occurrences(source, text, 100, &mut 1_000_000).unwrap();
            for s in &spans { validate_source_span(source, text,
                SourceSpan::new(s.byte_start, s.byte_end, s.scalar_start, s.scalar_end)).unwrap(); }
            if text == "aba" || text == "上海" { assert_eq!(spans.len(), 2); }
        }
    }
    #[test]
    fn linear_matcher_agrees_with_independent_boundary_enumeration() {
        for width in 0..7 {
            for bits in 0..(1_usize << width) {
                let source: String = (0..width).map(|i| if bits & (1 << i) == 0 { 'a' } else { 'é' }).collect();
                for pattern in ["a", "é", "aa", "aé", "éa", "éé", "aéa"] {
                    let expected: Vec<_> = source.char_indices().filter_map(|(byte, _)|
                        source[byte..].starts_with(pattern).then_some(byte)).collect();
                    let observed = occurrences(&source, pattern, 100, &mut 1_000_000).unwrap();
                    assert_eq!(observed.iter().map(|s| s.byte_start).collect::<Vec<_>>(), expected);
                }
            }
        }
    }
    #[test]
    fn rank_and_evidence_are_independent_of_tree_and_map_batch_shape() {
        let source = "Rust abc Rust def Rust abc Rust def Rust";
        let plan = chunks(source, 9); let mut reference = None;
        for fan_in in 2..7 { for batch in 1..5 {
            let mut task = CorpusKeyphraseTask::new(Fixture { corrupt: false }, CorpusKeyphraseLimits::default()).unwrap();
            let result = mapreduce::execute(&plan, &mut task, ExecutionLimits {
                reduce_fan_in: fan_in, map_batch_chunks: batch, ..ExecutionLimits::default()
            }, || Ok(())).unwrap().into_value().into_ranked(16, 65536).unwrap();
            assert_eq!(result.phrases[0].text, "Rust");
            assert_eq!(result.mapped_chunks, plan.chunks().len());
            assert_eq!(result.forward_positions, 2 * plan.chunks().len() as u64);
            let bytes = canonjson::canonical_bytes(&result).unwrap();
            if let Some(saved) = &reference { assert_eq!(&bytes, saved); } else { reference = Some(bytes); }
        } }
    }
    #[test]
    fn chunk_occurrences_lift_to_original_unicode_source() {
        let source = "é上海 abc 上海 def 上海"; let plan = chunks(source, 12);
        let mut task = CorpusKeyphraseTask::new(Fixture { corrupt: false }, CorpusKeyphraseLimits::default()).unwrap();
        let result = mapreduce::execute(&plan, &mut task, ExecutionLimits::default(), || Ok(())).unwrap()
            .into_value().into_ranked(16, 65536).unwrap();
        for phrase in result.phrases { for item in phrase.evidence { for s in item.spans {
            validate_source_span(source, &phrase.text, SourceSpan::new(s.byte_start, s.byte_end,
                s.scalar_start, s.scalar_end)).unwrap();
        } } }
    }
    #[test]
    fn repeated_occurrences_add_one_vote_per_chunk_and_top_k_is_final_only() {
        let plan = chunks("Rust Rust Rust abc def", 64);
        let mut task = CorpusKeyphraseTask::new(Fixture { corrupt: false }, CorpusKeyphraseLimits::default()).unwrap();
        let value = mapreduce::execute(&plan, &mut task, ExecutionLimits::default(), || Ok(())).unwrap().into_value();
        assert_eq!(value.phrases().len(), 3);
        let result = value.into_ranked(1, 65536).unwrap();
        assert_eq!(result.omitted_candidates, 2); assert_eq!(result.phrases[0].evidence.len(), 1);
        assert_eq!(result.phrases[0].evidence[0].spans.len(), 3);
    }
    #[test]
    fn incomplete_occurrences_and_duplicate_votes_are_rejected() {
        let plan = chunks("Rust Rust", 64); let chunk = &plan.chunks()[0];
        for mode in 0..3 {
            let mut raw = Fixture { corrupt: false }.run(chunk).unwrap();
            match mode { 0 => { raw.phrases[0].spans.pop(); },
                1 => { raw.phrases.push(raw.phrases[0].clone()); raw.phrases[1].rank = 2; },
                _ => raw.phrases[0].rank = 2 }
            assert!(map_value(chunk, raw, KeyphraseOptions::default(), CorpusKeyphraseLimits::default(), &mut 1_000_000).is_err());
        }
    }
    #[test]
    fn failed_task_is_poisoned_instead_of_reusing_partially_spent_work() {
        let plan = chunks("Rust abc", 64);
        let mut task = CorpusKeyphraseTask::new(Fixture { corrupt: true }, CorpusKeyphraseLimits::default()).unwrap();
        assert!(task.map_batch(plan.chunks()).is_err());
        task.pass.corrupt = false;
        assert!(task.map_batch(plan.chunks()).is_err());
    }
    #[test]
    fn candidate_span_scan_and_serialized_budgets_are_independent() {
        let plan = chunks("Rust abc def", 64);
        for mode in 0..4 {
            let mut limits = CorpusKeyphraseLimits::default();
            match mode { 0 => limits.max_unique_phrases = 1, 1 => limits.max_evidence_spans = 1,
                2 => limits.max_scan_work = 1, _ => limits.max_value_bytes = 1 }
            let mut task = CorpusKeyphraseTask::new(Fixture { corrupt: false }, limits).unwrap();
            assert!(task.map_batch(plan.chunks()).is_err());
        }
        assert!(occurrences("aaaa", "a", 3, &mut 1_000_000).is_err());
        assert!(occurrences("abc", "", 100, &mut 1_000_000).is_err());
    }
    #[test]
    fn native_work_charges_all_chunks_without_replenishing_any_axis() {
        let plan = chunks("Rust", 64); let result = Fixture { corrupt: false }.run(&plan.chunks()[0]).unwrap();
        let initial = JsonWorkBudget { max_forward_positions: 4, max_projected_logits: 40,
            max_total_mask_node_visits: 400, max_kv_bytes: 1234, mask_limits: MaskWorkLimits::default() };
        let exhausted = charge(charge(initial, &result).unwrap(), &result).unwrap();
        assert_eq!(exhausted.max_forward_positions, 0); assert_eq!(exhausted.max_projected_logits, 0);
        assert_eq!(exhausted.max_total_mask_node_visits, 0); assert_eq!(exhausted.max_kv_bytes, 1234);
        assert!(charge(exhausted, &result).is_err());
        for axis in 0..3 {
            let mut bad = initial;
            match axis { 0 => bad.max_forward_positions = 1, 1 => bad.max_projected_logits = 19,
                _ => bad.max_total_mask_node_visits = 199 }
            assert!(charge(bad, &result).is_err());
        }
    }
}
