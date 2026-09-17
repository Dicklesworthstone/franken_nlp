//! Raw-document NDJSON entity extraction AND resolution on one existing engine.
//!
//! Records are exactly {id,text} or {"flush":true}. Successful delivery means
//! one complete snapshot, including zero-mention documents. Malformed input,
//! failed inference, cancellation or unknown delivery never publishes a partial
//! corpus as complete. This named library protocol does not activate a CLI.

use std::{collections::BTreeSet, error::Error, fmt, io::{BufRead, Write}, marker::PhantomData};
use serde::Serialize;
use crate::{batch::{BatchWork, source::{GuardedOutput, SourceBatchAdmission}}, canonjson,
    execution_identity::ExecutionIdentity,
    native_engine::{decode::DecodeStepControl, hf_bf16_eager::HfBf16EagerEngine},
    tasks::{extract::ExtractionVocabulary, source_planning::SourceTaskPlanner},
    validation::grounded_fields::GroundingBudget};
use super::{entities::{self, BorrowAdmission, EntityCorpusConfig, EntityCorpusError, EntityCorpusResult,
    EntityDocument, EntityVerificationWork, PreparedEntityCorpus, prepare_entity_corpus},
    native_resolve::ResolutionPlanner, resolve::{self, ResolveError}};

pub const ENTITY_STREAM_PROTOCOL: &str = "fnlp-raw-entity-corpus-v1";

#[derive(Clone, Copy, Debug)]
pub struct EntityStreamLimits {
    pub max_line_bytes: usize,
    pub max_document_bytes: usize,
    pub max_corpus_bytes: usize,
    pub max_documents: usize,
    pub max_input_bytes: u64,
    pub max_records: u64,
    pub max_snapshots: u64,
    /// Complete canonical envelope including its terminating LF.
    pub max_output_line_bytes: usize,
    pub max_output_bytes: u64,
}
impl Default for EntityStreamLimits {
    fn default() -> Self {
        Self { max_line_bytes: 4 * 1024 * 1024, max_document_bytes: 1024 * 1024,
            max_corpus_bytes: 16 * 1024 * 1024, max_documents: 4096,
            max_input_bytes: 1024 * 1024 * 1024, max_records: 100_000, max_snapshots: 4096,
            max_output_line_bytes: 32 * 1024 * 1024, max_output_bytes: 1024 * 1024 * 1024 }
    }
}
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct EntityStreamSummary {
    pub input_bytes: u64, pub records: u64, pub snapshots: u64, pub documents: u64, pub output_bytes: u64,
}
#[derive(Debug)]
pub enum EntityStreamError<E> {
    InvalidLimits, InputIo, OutputIo, InputLimit, LineLimit, RecordLimit, SnapshotLimit,
    OutputLimit, InvalidRecord, Resolution(ResolveError), Handler(E),
}
impl<E> fmt::Display for EntityStreamError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidLimits => "invalid raw entity stream limits", Self::InputIo => "raw entity input failed",
            Self::OutputIo => "raw entity output failed; delivery is unknown", Self::InputLimit => "raw entity input limit exceeded",
            Self::LineLimit => "raw entity input record exceeds byte limit", Self::RecordLimit => "raw entity record limit exceeded",
            Self::SnapshotLimit => "raw entity snapshot limit exceeded", Self::OutputLimit => "raw entity complete output exceeds limit",
            Self::InvalidRecord => "invalid raw entity record", Self::Resolution(_) => "raw entity stream cancelled or serialization failed",
            Self::Handler(_) => "raw entity snapshot execution failed",
        })
    }
}
impl<E: Error + 'static> Error for EntityStreamError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Resolution(e) => Some(e), Self::Handler(e) => Some(e), _ => None }
    }
}

/// Static embedding boundary, not a deserializable plugin. Output owns its
/// admission guards. Acknowledgement happens ONLY after a complete write/flush;
/// any failure after execution must abandon, never renew, that delivery.
pub trait EntitySnapshotProcessor {
    type Output: Serialize;
    type Error;
    fn execute_snapshot<C: DecodeStepControl>(&mut self, documents: Vec<EntityDocument>, control: &mut C)
        -> Result<Self::Output, Self::Error>;
    fn acknowledge_snapshot(&mut self) -> Result<(), Self::Error>;
    fn abandon_snapshot(&mut self);
}

pub fn run_ndjson<R: BufRead, W: Write, P: EntitySnapshotProcessor, C: DecodeStepControl>(reader: &mut R,
    writer: &mut W, processor: &mut P, limits: EntityStreamLimits, control: &mut C)
    -> Result<EntityStreamSummary, EntityStreamError<P::Error>> {
    validate(limits)?;
    let mut summary = EntityStreamSummary::default(); let mut documents = Vec::new();
    let mut ids = BTreeSet::new(); let mut corpus_bytes = 0_usize;
    while let Some(line) = read_line(reader, limits, &mut summary, control)? {
        if line.is_empty() { continue; }
        summary.records = summary.records.checked_add(1).filter(|&n| n <= limits.max_records).ok_or(EntityStreamError::RecordLimit)?;
        let text = std::str::from_utf8(&line).map_err(|_| EntityStreamError::InvalidRecord)?;
        let value = canonjson::parse_str_with_limits(text, canonjson::ParseLimits { max_depth: 16, max_string_bytes: limits.max_line_bytes })
            .map_err(|_| EntityStreamError::InvalidRecord)?;
        if value.as_object().is_some_and(|o| o.len() == 1 && o.get("flush").and_then(|v| v.as_bool()) == Some(true)) {
            publish(&mut documents, writer, processor, limits, &mut summary, false, control)?;
            ids.clear(); corpus_bytes = 0; continue;
        }
        let d: EntityDocument = serde_json::from_value(value).map_err(|_| EntityStreamError::InvalidRecord)?;
        if d.id.is_empty() || d.id.len() > 256 || d.id.chars().any(char::is_control) || ids.contains(&d.id) {
            return Err(EntityStreamError::InvalidRecord);
        }
        let next = corpus_bytes.checked_add(d.id.len()).and_then(|n| n.checked_add(d.text.len()))
            .filter(|&n| n <= limits.max_corpus_bytes).ok_or(EntityStreamError::InputLimit)?;
        if d.text.len() > limits.max_document_bytes || documents.len() == limits.max_documents {
            return Err(EntityStreamError::InputLimit);
        }
        documents.try_reserve(1).map_err(|_| EntityStreamError::Resolution(ResolveError::AllocationRefused))?;
        ids.insert(d.id.clone()); documents.push(d); corpus_bytes = next;
    }
    if !documents.is_empty() { publish(&mut documents, writer, processor, limits, &mut summary, true, control)?; }
    Ok(summary)
}
fn validate<E>(l: EntityStreamLimits) -> Result<(), EntityStreamError<E>> {
    if !(1..=64 * 1024 * 1024).contains(&l.max_line_bytes) || !(1..=64 * 1024 * 1024).contains(&l.max_document_bytes)
        || !(1..=64 * 1024 * 1024).contains(&l.max_corpus_bytes) || !(1..=65_536).contains(&l.max_documents)
        || l.max_input_bytes == 0 || l.max_records == 0 || l.max_snapshots == 0
        || !(2..=64 * 1024 * 1024).contains(&l.max_output_line_bytes) || l.max_output_bytes == 0 {
        return Err(EntityStreamError::InvalidLimits);
    }
    Ok(())
}
fn read_line<R: BufRead, C: DecodeStepControl, E>(reader: &mut R, limits: EntityStreamLimits,
    summary: &mut EntityStreamSummary, control: &mut C) -> Result<Option<Vec<u8>>, EntityStreamError<E>> {
    let mut line = Vec::new();
    loop {
        resolve::checkpoint(control).map_err(EntityStreamError::Resolution)?;
        let available = reader.fill_buf().map_err(|_| EntityStreamError::InputIo)?;
        if available.is_empty() { return Ok((!line.is_empty()).then_some(line)); }
        // Never scan a caller's arbitrarily large BufRead slice past the byte
        // limit plus one rejecting byte, even before allocating the line.
        let window = available.len().min(limits.max_line_bytes - line.len() + 1);
        let available = &available[..window];
        let delimiter = available.iter().position(|&b| b == b'\n');
        let consumed = delimiter.map_or(available.len(), |i| i + 1);
        let payload = delimiter.unwrap_or(consumed);
        if line.len().checked_add(payload).is_none_or(|n| n > limits.max_line_bytes) { return Err(EntityStreamError::LineLimit); }
        let total = summary.input_bytes.checked_add(consumed as u64).filter(|&n| n <= limits.max_input_bytes).ok_or(EntityStreamError::InputLimit)?;
        line.try_reserve(payload).map_err(|_| EntityStreamError::Resolution(ResolveError::AllocationRefused))?;
        line.extend_from_slice(&available[..payload]); reader.consume(consumed); summary.input_bytes = total;
        if delimiter.is_some() { if line.last() == Some(&b'\r') { line.pop(); } return Ok(Some(line)); }
    }
}
#[allow(clippy::too_many_arguments)]
fn publish<W: Write, P: EntitySnapshotProcessor, C: DecodeStepControl>(documents: &mut Vec<EntityDocument>,
    writer: &mut W, processor: &mut P, limits: EntityStreamLimits,
    summary: &mut EntityStreamSummary, eof: bool, control: &mut C) -> Result<(), EntityStreamError<P::Error>> {
    let epoch = summary.snapshots.checked_add(1).filter(|&n| n <= limits.max_snapshots).ok_or(EntityStreamError::SnapshotLimit)?;
    let count = summary.documents.checked_add(documents.len() as u64).ok_or(EntityStreamError::InputLimit)?;
    resolve::checkpoint(control).map_err(EntityStreamError::Resolution)?;
    let output = match processor.execute_snapshot(std::mem::take(documents), control) {
        Ok(output) => output,
        Err(error) => { processor.abandon_snapshot(); return Err(EntityStreamError::Handler(error)); }
    };
    // The output (including every native/source/run guard) lives outside this
    // fallible delivery scope. Unknown delivery cannot be followed by an error
    // line, an acknowledgement, another model request or an automatic retry.
    let delivered = (|| -> Result<u64, EntityStreamError<P::Error>> {
        resolve::checkpoint(control).map_err(EntityStreamError::Resolution)?;
        #[derive(Serialize)]
        struct Event<'a, T> {
            protocol: &'static str, schema_version: u32, event: &'static str,
            epoch: u64, through_request_seq: u64, eof: bool, result: &'a T,
        }
        let event = Event { protocol: ENTITY_STREAM_PROTOCOL, schema_version: 1, event: "corpus",
            epoch, through_request_seq: summary.records, eof, result: &output };
        resolve::check_output(&event, limits.max_output_line_bytes - 1).map_err(EntityStreamError::Resolution)?;
        let mut bytes = canonjson::canonical_bytes(&event).map_err(|_| EntityStreamError::Resolution(ResolveError::Serialization))?;
        bytes.try_reserve_exact(1).map_err(|_| EntityStreamError::Resolution(ResolveError::AllocationRefused))?; bytes.push(b'\n');
        let total = summary.output_bytes.checked_add(bytes.len() as u64).filter(|&n| n <= limits.max_output_bytes).ok_or(EntityStreamError::OutputLimit)?;
        writer.write_all(&bytes).and_then(|_| writer.flush()).map_err(|_| EntityStreamError::OutputIo)?;
        processor.acknowledge_snapshot().map_err(EntityStreamError::Handler)?;
        Ok(total)
    })();
    match delivered {
        Ok(bytes) => { summary.snapshots = epoch; summary.documents = count; summary.output_bytes = bytes; Ok(()) }
        Err(error) => { processor.abandon_snapshot(); Err(error) }
    }
}

/// Finite allowances shared across every snapshot in a native stream.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct EntityStreamBudget {
    pub work: BatchWork,
    pub mask_visits: u64,
    pub verification: GroundingBudget,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum State { Ready, Running(EntityStreamBudget), AwaitingDelivery(EntityStreamBudget), Failed }
struct Ledger { remaining: EntityStreamBudget, state: State }
impl Ledger {
    fn ready(&self) -> Result<(), EntityCorpusError> {
        if self.state == State::Ready { Ok(()) } else { Err(EntityCorpusError::Accounting) }
    }
    fn begin(&mut self, charge: EntityStreamBudget) -> Result<(), EntityCorpusError> {
        self.ready()?;
        // Stage ALL arithmetic before mutation: a failed second/third axis
        // must not partially debit the first one or renew a previous budget.
        let next = subtract_budget(self.remaining, charge)?;
        self.remaining = next; self.state = State::Running(charge); Ok(())
    }
    fn complete(&mut self, work: BatchWork, verification: EntityVerificationWork)
        -> Result<(), EntityCorpusError> {
        let state = std::mem::replace(&mut self.state, State::Failed);
        let State::Running(charge) = state else { return Err(EntityCorpusError::Accounting); };
        let unused = EntityStreamBudget {
            work: entities::subtract(charge.work, work)?, mask_visits: 0,
            verification: GroundingBudget {
                max_fields: charge.verification.max_fields.checked_sub(verification.fields).ok_or(EntityCorpusError::Accounting)?,
                max_matches: charge.verification.max_matches.checked_sub(verification.matches).ok_or(EntityCorpusError::Accounting)?,
                max_scan_steps: charge.verification.max_scan_steps.checked_sub(verification.scan_steps).ok_or(EntityCorpusError::Accounting)?,
            },
        };
        self.state = State::AwaitingDelivery(unused); Ok(())
    }
    fn acknowledge(&mut self) -> Result<(), EntityCorpusError> {
        let state = std::mem::replace(&mut self.state, State::Failed);
        let State::AwaitingDelivery(unused) = state else { return Err(EntityCorpusError::Accounting); };
        self.remaining = add_budget(self.remaining, unused)?;
        self.state = State::Ready; Ok(())
    }
    fn abandon(&mut self) { self.state = State::Failed; }
}
fn subtract_budget(a: EntityStreamBudget, b: EntityStreamBudget) -> Result<EntityStreamBudget, EntityCorpusError> {
    Ok(EntityStreamBudget { work: entities::subtract(a.work, b.work)?,
        mask_visits: a.mask_visits.checked_sub(b.mask_visits).ok_or(EntityCorpusError::WorkBudget)?,
        verification: GroundingBudget {
            max_fields: a.verification.max_fields.checked_sub(b.verification.max_fields).ok_or(EntityCorpusError::WorkBudget)?,
            max_matches: a.verification.max_matches.checked_sub(b.verification.max_matches).ok_or(EntityCorpusError::WorkBudget)?,
            max_scan_steps: a.verification.max_scan_steps.checked_sub(b.verification.max_scan_steps).ok_or(EntityCorpusError::WorkBudget)?,
        } })
}
fn add_budget(a: EntityStreamBudget, b: EntityStreamBudget) -> Result<EntityStreamBudget, EntityCorpusError> {
    Ok(EntityStreamBudget { work: entities::add(a.work, b.work)?,
        mask_visits: a.mask_visits.checked_add(b.mask_visits).ok_or(EntityCorpusError::Accounting)?,
        verification: GroundingBudget {
            max_fields: a.verification.max_fields.checked_add(b.verification.max_fields).ok_or(EntityCorpusError::Accounting)?,
            max_matches: a.verification.max_matches.checked_add(b.verification.max_matches).ok_or(EntityCorpusError::Accounting)?,
            max_scan_steps: a.verification.max_scan_steps.checked_add(b.verification.max_scan_steps).ok_or(EntityCorpusError::Accounting)?,
        } })
}

pub struct NativeEntityStream<'p, 'e, 'v, A, F, G> {
    source: &'p SourceTaskPlanner, resolver: &'p ResolutionPlanner,
    source_identity: ExecutionIdentity, resolution_identity: ExecutionIdentity,
    engine: &'e mut HfBf16EagerEngine, vocabulary: &'v ExtractionVocabulary,
    config: EntityCorpusConfig, admission: A, run_admission: F, ledger: Ledger,
    guard_type: PhantomData<fn() -> G>,
}
impl<'p, 'e, 'v, A, F, G> NativeEntityStream<'p, 'e, 'v, A, F, G>
where A: SourceBatchAdmission, F: for<'s> FnMut(&PreparedEntityCorpus<'s>) -> Result<G, EntityCorpusError> {
    #[allow(clippy::too_many_arguments)]
    pub fn new<C: DecodeStepControl>(source: &'p SourceTaskPlanner, resolver: &'p ResolutionPlanner,
        source_identity: ExecutionIdentity, resolution_identity: ExecutionIdentity,
        engine: &'e mut HfBf16EagerEngine, vocabulary: &'v ExtractionVocabulary,
        config: EntityCorpusConfig, total_budget: EntityStreamBudget, admission: A, run_admission: F, control: &mut C)
        -> Result<Self, EntityCorpusError> {
        if !engine.kv_cache().all_slots_have_len(0) { return Err(EntityCorpusError::Accounting); }
        // Validate shared model identity and resolver configuration before input.
        prepare_entity_corpus(Vec::new(), source, &source_identity, resolver, &resolution_identity, config.clone(), control)?;
        Ok(Self { source, resolver, source_identity, resolution_identity, engine, vocabulary, config, admission,
            run_admission, ledger: Ledger { remaining: total_budget, state: State::Ready }, guard_type: PhantomData })
    }
    pub fn remaining_budget(&self) -> EntityStreamBudget { self.ledger.remaining }
    pub fn ready_for_snapshot(&self) -> bool { self.ledger.state == State::Ready }
}
impl<A, F, G> EntitySnapshotProcessor for NativeEntityStream<'_, '_, '_, A, F, G>
where A: SourceBatchAdmission, F: for<'s> FnMut(&PreparedEntityCorpus<'s>) -> Result<G, EntityCorpusError> {
    type Output = GuardedOutput<EntityCorpusResult, (Vec<A::Guard>, Vec<A::Guard>, G)>;
    type Error = EntityCorpusError;
    fn execute_snapshot<C: DecodeStepControl>(&mut self, documents: Vec<EntityDocument>, control: &mut C)
        -> Result<Self::Output, Self::Error> {
        self.ledger.ready()?;
        let mut config = self.config.clone(); let remaining = self.ledger.remaining;
        config.max_work.forward_positions = config.max_work.forward_positions.min(remaining.work.forward_positions);
        config.max_work.projected_logits = config.max_work.projected_logits.min(remaining.work.projected_logits);
        config.masks.max_visits_per_run = config.masks.max_visits_per_run.min(remaining.mask_visits);
        config.verification.max_fields = config.verification.max_fields.min(remaining.verification.max_fields);
        config.verification.max_matches = config.verification.max_matches.min(remaining.verification.max_matches);
        config.verification.max_scan_steps = config.verification.max_scan_steps.min(remaining.verification.max_scan_steps);
        let verification = config.verification;
        let prepared = prepare_entity_corpus(documents, self.source, &self.source_identity,
            self.resolver, &self.resolution_identity, config, control)?;
        let charge = EntityStreamBudget { work: prepared.whole_run_work_ceiling(),
            mask_visits: prepared.reserved_mask_visits(), verification };
        // Reserve worst-case unknown pair work BEFORE the first NER call.
        // Failed/panicked/undelivered work stays charged and prevents reuse.
        self.ledger.begin(charge)?;
        let guard = (self.run_admission)(&prepared)?;
        let output = prepared.execute_native(self.engine, self.vocabulary, BorrowAdmission(&mut self.admission), guard, control)?;
        // Only never-used pair/verification allowance can return, and ONLY
        // after the runner acknowledges write+flush. NER maximums and mask
        // charges remain spent even when the model ended early.
        self.ledger.complete(output.result().reserved_work, output.result().verification_used)?;
        Ok(output)
    }
    fn acknowledge_snapshot(&mut self) -> Result<(), Self::Error> { self.ledger.acknowledge() }
    fn abandon_snapshot(&mut self) { self.ledger.abandon(); }
}

#[cfg(test)]
mod tests;
