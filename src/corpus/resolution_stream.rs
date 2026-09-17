//! Flush-scoped entity resolution, distinct from item-local batch execution.
//!
//! Input records are {id,text,mentions} or {id,text,ner:<NerResult>}; the only
//! control is exactly {"flush":true}. A flush finalizes ONE complete snapshot.
//! EOF finalizes a nonempty remaining snapshot. A malformed/overbudget record
//! aborts the current snapshot instead of silently publishing a partial corpus.
//! Earlier acknowledged snapshots remain delivered. No retry or persistence.
//!
//! This named library protocol is not an extension of the frozen robot schema.
//! The host owns blocking IO, execution regions and model/process admission.

use std::{collections::BTreeSet, error::Error, fmt, io::{BufRead, Write}, marker::PhantomData};
use serde::{Deserialize, Serialize};
use crate::{canonjson, batch::BatchWork, execution_identity::ExecutionIdentity,
    native_engine::{decode::DecodeStepControl, hf_bf16_eager::{HfBf16EagerEngine, HF_BF16_EAGER_PROFILE}},
    tasks::{extract::ExtractionGrounding, ir::ScoreSpace, ner::{NerResult, NER_TASK_VERSION}},
    validation::grounded_fields::{GroundingBudget, FieldGroundingError, SourceOccurrence, scan_occurrences}};
use super::{resolve::{self, MentionInput, ResolutionDocument, ResolutionPlan, ResolveError, ResolveLimits, ResolveOptions},
    native_resolve::{GuardedOutput, NativeResolutionResult, NativeResolveError, NativeResolveLimits,
        ResolutionPlanner, ResolveAdmission}};

pub const RESOLUTION_STREAM_PROTOCOL: &str = "fnlp-resolution-corpus-v1";

/// Expand each verified occurrence into a separate contextual mention. Equal
/// spellings are NOT assumed to be the same entity. Duplicate NER proposals
/// are checked before exact-span deduplication, so they cannot inflate votes or
/// hide a corrupt proof. Deserialization never establishes model provenance.
pub fn document_from_ner<C: DecodeStepControl>(id: String, text: String, ner: NerResult,
    limits: ResolveLimits, verification: &mut GroundingBudget, control: &mut C)
    -> Result<ResolutionDocument, ResolveError> {
    if !(1..=64 * 1024 * 1024).contains(&limits.max_input_bytes)
        || !(1..=65_536).contains(&limits.max_mentions) || !(1..=16_384).contains(&limits.max_surface_bytes) {
        return Err(ResolveError::InvalidLimits);
    }
    if id.is_empty() || id.len() > 256 || id.chars().any(char::is_control)
        || ner.schema_version != 1 || ner.task_spec_version != NER_TASK_VERSION
        || ner.numerics_profile != HF_BF16_EAGER_PROFILE || ner.score_space != ScoreSpace::NotComputed
        || ner.grounding != ExtractionGrounding::SourceMembership { return Err(ResolveError::InvalidInput); }
    let mut bytes = id.len().checked_add(text.len()).filter(|&n| n <= limits.max_input_bytes).ok_or(ResolveError::InputBudget)?;
    if ner.entities.len() > limits.max_mentions { return Err(ResolveError::InputBudget); }
    let mut seen = BTreeSet::new(); let mut mentions = Vec::new();
    for entity in &ner.entities {
        resolve::checkpoint(control)?;
        if entity.text.is_empty() || entity.text.len() > limits.max_surface_bytes { return Err(ResolveError::InvalidInput); }
        verification.max_fields = verification.max_fields.checked_sub(1).ok_or(ResolveError::InputBudget)?;
        let spans = scan_occurrences(&text, &entity.text, verification).map_err(|e| match e {
            FieldGroundingError::AllocationRefused => ResolveError::AllocationRefused,
            FieldGroundingError::WorkBudget => ResolveError::ScanBudget,
            FieldGroundingError::MatchBudget | FieldGroundingError::FieldBudget => ResolveError::InputBudget,
            _ => ResolveError::InvalidAnchor,
        })?;
        if spans != entity.spans || entity.occurrence != if spans.len() == 1 { SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous } {
            return Err(ResolveError::InvalidAnchor);
        }
        for span in spans {
            if !seen.insert((entity.entity_type, span.byte_start, span.byte_end)) { continue; }
            if mentions.len() == limits.max_mentions { return Err(ResolveError::InputBudget); }
            let entity_type = entity.entity_type.label();
            bytes = bytes.checked_add(entity_type.len()).and_then(|n| n.checked_add(entity.text.len()))
                .filter(|&n| n <= limits.max_input_bytes).ok_or(ResolveError::InputBudget)?;
            let mut surface = String::new(); surface.try_reserve_exact(entity.text.len()).map_err(|_| ResolveError::AllocationRefused)?;
            surface.push_str(&entity.text);
            mentions.try_reserve(1).map_err(|_| ResolveError::AllocationRefused)?;
            mentions.push(MentionInput { entity_type: entity_type.to_owned(), surface, span });
        }
    }
    resolve::checkpoint(control)?;
    Ok(ResolutionDocument { id, text, mentions })
}

#[derive(Clone, Copy, Debug)]
pub struct ResolutionStreamLimits {
    /// Payload before LF, including CR in CRLF. Oversized input ends the run.
    pub max_line_bytes: usize,
    pub max_input_bytes: u64,
    pub max_records: u64,
    pub max_snapshots: u64,
    /// Complete snapshot envelope INCLUDING final LF.
    pub max_output_line_bytes: usize,
    pub max_output_bytes: u64,
    /// Nonrenewable NER verification allowance across every flush epoch.
    pub ner_verification: GroundingBudget,
}
impl Default for ResolutionStreamLimits {
    fn default() -> Self {
        Self { max_line_bytes: 16 * 1024 * 1024, max_input_bytes: 1024 * 1024 * 1024,
            max_records: 100_000, max_snapshots: 4096, max_output_line_bytes: 32 * 1024 * 1024,
            max_output_bytes: 1024 * 1024 * 1024,
            ner_verification: GroundingBudget { max_fields: 100_000, max_matches: 1_000_000, max_scan_steps: 256 * 1024 * 1024 } }
    }
}
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct ResolutionStreamSummary {
    pub input_bytes: u64, pub records: u64, pub snapshots: u64, pub documents: u64, pub output_bytes: u64,
}
#[derive(Debug)]
pub enum ResolutionStreamError<E> {
    InvalidLimits, InputIo, OutputIo, InputLimit, LineLimit, RecordLimit, SnapshotLimit,
    OutputLimit, InvalidRecord, Resolution(ResolveError), Handler(E),
}
impl<E> fmt::Display for ResolutionStreamError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self { Self::InvalidLimits => "invalid corpus stream limits",
            Self::InputIo => "corpus stream input failed", Self::OutputIo => "corpus stream output failed; delivery is unknown",
            Self::InputLimit => "corpus stream input budget exceeded", Self::LineLimit => "corpus stream record too large",
            Self::RecordLimit => "corpus stream record budget exceeded", Self::SnapshotLimit => "corpus stream snapshot budget exceeded",
            Self::OutputLimit => "complete corpus stream output exceeds budget", Self::InvalidRecord => "invalid corpus stream record",
            Self::Resolution(_) => "corpus stream source or snapshot failed", Self::Handler(_) => "corpus stream resolution failed" })
    }
}
impl<E: Error + 'static> Error for ResolutionStreamError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Resolution(e) => Some(e), Self::Handler(e) => Some(e), _ => None }
    }
}
/// Static host boundary. Output must own its storage (and resource guards) so
/// they survive serialization/write/flush. No per-document provisional IDs.
pub trait SnapshotResolver {
    type Output: Serialize;
    type Error;
    fn resolve<C: DecodeStepControl>(&mut self, plan: &ResolutionPlan<'_>, control: &mut C)
        -> Result<Self::Output, Self::Error>;
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NerDocument { id: String, text: String, ner: NerResult }
#[derive(Deserialize)]
#[serde(untagged)]
enum InputDocument { Mentions(ResolutionDocument), Ner(NerDocument) }

pub fn run_ndjson<R: BufRead, W: Write, P: SnapshotResolver, C: DecodeStepControl>(reader: &mut R,
    writer: &mut W, resolver: &mut P, options: ResolveOptions, corpus_limits: ResolveLimits,
    limits: ResolutionStreamLimits, control: &mut C)
    -> Result<ResolutionStreamSummary, ResolutionStreamError<P::Error>> {
    if !(1..=64 * 1024 * 1024).contains(&limits.max_line_bytes) || limits.max_input_bytes == 0
        || limits.max_records == 0 || limits.max_snapshots == 0
        || !(2..=64 * 1024 * 1024).contains(&limits.max_output_line_bytes) || limits.max_output_bytes == 0 {
        return Err(ResolutionStreamError::InvalidLimits);
    }
    // Validate core configuration even on an empty stream, without inference.
    ResolutionPlan::prepare(&[], options, corpus_limits, control).map_err(ResolutionStreamError::Resolution)?;
    let mut summary = ResolutionStreamSummary::default(); let mut documents = Vec::new();
    let mut ids = BTreeSet::new(); let (mut input_bytes, mut mentions) = (0_usize, 0_usize);
    let mut verification = limits.ner_verification;
    while let Some(line) = read_line(reader, limits, &mut summary, control)? {
        if line.is_empty() { continue; }
        summary.records = summary.records.checked_add(1).filter(|&n| n <= limits.max_records).ok_or(ResolutionStreamError::RecordLimit)?;
        let text = std::str::from_utf8(&line).map_err(|_| ResolutionStreamError::InvalidRecord)?;
        let value = canonjson::parse_str_with_limits(text, canonjson::ParseLimits { max_depth: 64, max_string_bytes: limits.max_line_bytes })
            .map_err(|_| ResolutionStreamError::InvalidRecord)?;
        if value.as_object().is_some_and(|o| o.len() == 1 && o.get("flush").and_then(|v| v.as_bool()) == Some(true)) {
            publish(&documents, writer, resolver, options, corpus_limits, limits, &mut summary, control)?;
            documents.clear(); ids.clear(); input_bytes = 0; mentions = 0; continue;
        }
        let document = match serde_json::from_value(value).map_err(|_| ResolutionStreamError::InvalidRecord)? {
            InputDocument::Mentions(d) => d,
            InputDocument::Ner(d) => document_from_ner(d.id, d.text, d.ner, corpus_limits, &mut verification, control)
                .map_err(ResolutionStreamError::Resolution)?,
        };
        if document.id.is_empty() || document.id.len() > 256 || document.id.chars().any(char::is_control)
            || ids.contains(&document.id) { return Err(ResolutionStreamError::InvalidRecord); }
        mentions = mentions.checked_add(document.mentions.len()).ok_or(ResolutionStreamError::InputLimit)?;
        input_bytes = input_bytes.checked_add(document.id.len()).and_then(|n| n.checked_add(document.text.len())).ok_or(ResolutionStreamError::InputLimit)?;
        for mention in &document.mentions {
            input_bytes = input_bytes.checked_add(mention.entity_type.len()).and_then(|n| n.checked_add(mention.surface.len())).ok_or(ResolutionStreamError::InputLimit)?;
        }
        if documents.len() == corpus_limits.max_documents || mentions > corpus_limits.max_mentions || input_bytes > corpus_limits.max_input_bytes {
            return Err(ResolutionStreamError::InputLimit);
        }
        ids.insert(document.id.clone());
        documents.try_reserve(1).map_err(|_| ResolutionStreamError::Resolution(ResolveError::AllocationRefused))?;
        documents.push(document);
    }
    if !documents.is_empty() { publish(&documents, writer, resolver, options, corpus_limits, limits, &mut summary, control)?; }
    Ok(summary)
}
fn read_line<R: BufRead, C: DecodeStepControl, E>(reader: &mut R, limits: ResolutionStreamLimits,
    summary: &mut ResolutionStreamSummary, control: &mut C) -> Result<Option<Vec<u8>>, ResolutionStreamError<E>> {
    let mut line = Vec::new();
    loop {
        resolve::checkpoint(control).map_err(ResolutionStreamError::Resolution)?;
        let available = reader.fill_buf().map_err(|_| ResolutionStreamError::InputIo)?;
        if available.is_empty() { return Ok((!line.is_empty()).then_some(line)); }
        let delimiter = available.iter().position(|&b| b == b'\n');
        let consumed = delimiter.map_or(available.len(), |i| i + 1);
        let payload = delimiter.unwrap_or(consumed);
        if line.len().checked_add(payload).is_none_or(|n| n > limits.max_line_bytes) { return Err(ResolutionStreamError::LineLimit); }
        let total = summary.input_bytes.checked_add(consumed as u64).filter(|&n| n <= limits.max_input_bytes).ok_or(ResolutionStreamError::InputLimit)?;
        line.try_reserve(payload).map_err(|_| ResolutionStreamError::Resolution(ResolveError::AllocationRefused))?;
        line.extend_from_slice(&available[..payload]); reader.consume(consumed); summary.input_bytes = total;
        if delimiter.is_some() { if line.last() == Some(&b'\r') { line.pop(); } return Ok(Some(line)); }
    }
}
#[allow(clippy::too_many_arguments)]
fn publish<W: Write, P: SnapshotResolver, C: DecodeStepControl>(documents: &[ResolutionDocument],
    writer: &mut W, resolver: &mut P, options: ResolveOptions, corpus_limits: ResolveLimits,
    limits: ResolutionStreamLimits, summary: &mut ResolutionStreamSummary, control: &mut C)
    -> Result<(), ResolutionStreamError<P::Error>> {
    let epoch = summary.snapshots.checked_add(1).filter(|&n| n <= limits.max_snapshots).ok_or(ResolutionStreamError::SnapshotLimit)?;
    let document_count = summary.documents.checked_add(documents.len() as u64).ok_or(ResolutionStreamError::InputLimit)?;
    let plan = ResolutionPlan::prepare(documents, options, corpus_limits, control).map_err(ResolutionStreamError::Resolution)?;
    let result = resolver.resolve(&plan, control).map_err(ResolutionStreamError::Handler)?;
    resolve::checkpoint(control).map_err(ResolutionStreamError::Resolution)?;
    #[derive(Serialize)]
    struct Event<'a, T> { protocol: &'static str, schema_version: u32, event: &'static str, epoch: u64, result: &'a T }
    let event = Event { protocol: RESOLUTION_STREAM_PROTOCOL, schema_version: 1, event: "corpus", epoch, result: &result };
    resolve::check_output(&event, limits.max_output_line_bytes - 1).map_err(ResolutionStreamError::Resolution)?;
    let mut bytes = canonjson::canonical_bytes(&event).map_err(|_| ResolutionStreamError::Resolution(ResolveError::Serialization))?;
    bytes.try_reserve_exact(1).map_err(|_| ResolutionStreamError::Resolution(ResolveError::AllocationRefused))?; bytes.push(b'\n');
    let output_bytes = summary.output_bytes.checked_add(bytes.len() as u64).filter(|&n| n <= limits.max_output_bytes).ok_or(ResolutionStreamError::OutputLimit)?;
    // Blocking IO cannot be preempted by a cooperative token checkpoint. A
    // failed write OR flush has unknown delivery: return immediately, no retry,
    // no appended error and no additional input admission. Result guard lives.
    writer.write_all(&bytes).and_then(|_| writer.flush()).map_err(|_| ResolutionStreamError::OutputIo)?;
    summary.snapshots = epoch; summary.documents = document_count; summary.output_bytes = output_bytes;
    Ok(())
}

/// Concrete native stream adapter. Pair admission is the existing host hook;
/// run_admission supplies the actual whole-snapshot memory/output guard. Work
/// ceilings survive flush epochs, are reserved BEFORE admission and are never
/// refunded after a failure. Failed/unwound native work poisons this adapter.
pub struct NativeResolutionStream<'p, 'e, A, F, G> {
    planner: &'p ResolutionPlanner, engine: &'e mut HfBf16EagerEngine, identity: ExecutionIdentity,
    limits: NativeResolveLimits, remaining: BatchWork, pair_admission: A, run_admission: F,
    failed: bool, guard_type: PhantomData<fn() -> G>,
}
impl<'p, 'e, A, F, G> NativeResolutionStream<'p, 'e, A, F, G>
where A: ResolveAdmission, F: for<'s> FnMut(&ResolutionPlan<'s>, BatchWork) -> Result<G, NativeResolveError> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(planner: &'p ResolutionPlanner, engine: &'e mut HfBf16EagerEngine, identity: ExecutionIdentity,
        limits: NativeResolveLimits, total_work: BatchWork, pair_admission: A, run_admission: F) -> Result<Self, NativeResolveError> {
        identity.validate().map_err(|_| NativeResolveError::Contract)?;
        if !engine.kv_cache().all_slots_have_len(0) { return Err(NativeResolveError::Accounting); }
        Ok(Self { planner, engine, identity, limits, remaining: total_work, pair_admission, run_admission,
            failed: false, guard_type: PhantomData })
    }
    pub fn remaining_work(&self) -> BatchWork { self.remaining }
}
struct BorrowAdmission<'a, A>(&'a mut A);
impl<A: ResolveAdmission> ResolveAdmission for BorrowAdmission<'_, A> {
    type Guard = A::Guard;
    fn admit(&mut self, proposed: &ExecutionIdentity, work: BatchWork)
        -> Result<(ExecutionIdentity, Self::Guard), crate::batch::BatchItemFailure> { self.0.admit(proposed, work) }
}
impl<A, F, G> SnapshotResolver for NativeResolutionStream<'_, '_, A, F, G>
where A: ResolveAdmission, F: for<'s> FnMut(&ResolutionPlan<'s>, BatchWork) -> Result<G, NativeResolveError> {
    type Output = GuardedOutput<NativeResolutionResult, (Vec<A::Guard>, G)>;
    type Error = NativeResolveError;
    fn resolve<C: DecodeStepControl>(&mut self, plan: &ResolutionPlan<'_>, control: &mut C) -> Result<Self::Output, Self::Error> {
        if self.failed { return Err(NativeResolveError::Contract); }
        let mut limits = self.limits;
        limits.max_work.forward_positions = limits.max_work.forward_positions.min(self.remaining.forward_positions);
        limits.max_work.projected_logits = limits.max_work.projected_logits.min(self.remaining.projected_logits);
        let prepared = self.planner.prepare(plan, &self.identity, limits, control)?;
        let work = prepared.planned_work(); reserve(&mut self.remaining, work)?;
        self.failed = true;
        let guard = (self.run_admission)(plan, work)?;
        let result = prepared.execute_native(self.engine, BorrowAdmission(&mut self.pair_admission), guard, control)?;
        self.failed = false; Ok(result)
    }
}
fn reserve(remaining: &mut BatchWork, work: BatchWork) -> Result<(), NativeResolveError> {
    let next = BatchWork {
        forward_positions: remaining.forward_positions.checked_sub(work.forward_positions).ok_or(NativeResolveError::WorkBudget)?,
        projected_logits: remaining.projected_logits.checked_sub(work.projected_logits).ok_or(NativeResolveError::WorkBudget)?,
    };
    *remaining = next; Ok(())
}

#[cfg(test)]
mod tests;
