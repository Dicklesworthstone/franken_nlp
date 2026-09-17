//! Static-dispatch map/reduce orchestration over an admitted ChunkPlan.
//!
//! This owns no scheduler, thread pool, model, or durable cache. The supplied
//! task owns model admission, TaskIR validation, and in-call resource limits.
//! Calls are sequential at this boundary; a map call may batch compatible
//! requests on the caller's already-admitted native compute team.

use std::{error::Error, fmt, io, ops::Range};

use serde::{Deserialize, Serialize};

use super::{CHUNK_PROFILE, ChunkPlan, MapReduceError, SourceChunk};
use crate::{canonjson, validation::grounded_fields::VerifiedSourceSpan};

pub const EXECUTION_PROFILE: &str = "ordered-contiguous-fanin-v1";
const MAX_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Serialize)]
pub struct ReductionPolicy {
    /// Versioned, code-owned conflict/dedup/weighting policy, not user prose.
    pub id: &'static str,
    /// A declaration, not an independently established losslessness proof.
    pub may_discard_information: bool,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionLimits {
    pub map_batch_chunks: usize,
    pub reduce_fan_in: usize,
    pub max_reduction_levels: usize,
    pub max_task_calls: usize,
    pub max_value_bytes: usize,
    /// Sum of serialized sizes of live values, including old AND new values
    /// at a reduction boundary. This is not a Rust heap-size measurement.
    pub max_live_value_bytes: usize,
    /// Cumulative serialized sizes of all accepted map/reduce values.
    pub max_total_value_bytes: usize,
    /// Complete canonical final envelope, not just its value.
    pub max_result_bytes: usize,
}

impl Default for ExecutionLimits {
    fn default() -> Self {
        Self {
            map_batch_chunks: 8,
            reduce_fan_in: 8,
            max_reduction_levels: 16,
            max_task_calls: 65_536,
            max_value_bytes: 1024 * 1024,
            max_live_value_bytes: 16 * 1024 * 1024,
            max_total_value_bytes: MAX_BYTES,
            max_result_bytes: 4 * 1024 * 1024,
        }
    }
}

impl ExecutionLimits {
    fn validate(self) -> bool {
        (1..=1024).contains(&self.map_batch_chunks)
            && (2..=1024).contains(&self.reduce_fan_in)
            && self.max_reduction_levels <= 64
            && (1..=1_000_000).contains(&self.max_task_calls)
            && [self.max_value_bytes, self.max_live_value_bytes,
                self.max_total_value_bytes, self.max_result_bytes]
                .iter().all(|n| (1..=MAX_BYTES).contains(n))
            && self.max_value_bytes <= self.max_live_value_bytes
    }
}

pub struct MapOutput<T> {
    pub chunk_id: usize,
    pub value: T,
}

/// Provenance is minted by the orchestrator, not accepted from model output.
/// The interval includes every contributing source chunk, including evidence
/// that a lossy reducer may omit from its final prose.
#[derive(Serialize)]
pub struct ReductionNode<T> {
    first_chunk: usize,
    end_chunk: usize,
    source_span: VerifiedSourceSpan,
    value: T,
    #[serde(skip)]
    value_bytes: usize,
}

impl<T> ReductionNode<T> {
    pub fn chunk_range(&self) -> Range<usize> { self.first_chunk..self.end_chunk }
    pub fn source_span(&self) -> VerifiedSourceSpan { self.source_span }
    pub fn value(&self) -> &T { &self.value }
}

pub struct ReduceInput<'a, T> {
    /// One-based level; groups are contiguous, left-to-right within a level.
    pub level: usize,
    pub group: usize,
    pub children: &'a [ReductionNode<T>],
}

/// Implementations consume and finalize real task executions. Errors retain
/// their original type. This is a statically dispatched internal composition
/// boundary, not a runtime plugin registry or a new source of model authority.
///
/// Values and their Serialize implementations must be stable data. The task
/// must bound allocations, token/context usage and cancellation within each
/// call; the orchestrator cannot preempt a native call or arbitrary Serialize.
pub trait MapReduceTask {
    type Value: Serialize;
    type Error;

    fn policy(&self) -> ReductionPolicy;
    fn map_batch(&mut self, chunks: &[SourceChunk<'_>]) -> Result<Vec<MapOutput<Self::Value>>, Self::Error>;
    fn reduce(&mut self, input: ReduceInput<'_, Self::Value>) -> Result<Self::Value, Self::Error>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStage {
    Map { first_chunk: usize },
    Reduce { level: usize, group: usize },
}

#[derive(Debug)]
pub enum ExecutionError<E> {
    InvalidLimits,
    InvalidPolicy,
    EmptySource,
    InvalidBatch,
    WorkBudget,
    ValueBudget,
    LiveValueBudget,
    TotalValueBudget,
    ResultBudget,
    Serialization,
    AllocationRefused,
    Checkpoint(MapReduceError),
    Task { stage: TaskStage, source: E },
    Invariant,
}

impl<E> fmt::Display for ExecutionError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidLimits => "invalid map/reduce execution limits",
            Self::InvalidPolicy => "invalid map/reduce policy identity",
            Self::EmptySource => "map/reduce requires at least one source chunk",
            Self::InvalidBatch => "map batch must return every requested chunk exactly once",
            Self::WorkBudget => "map/reduce call or depth budget exceeded",
            Self::ValueBudget => "map/reduce value byte budget exceeded",
            Self::LiveValueBudget => "map/reduce live serialized-value budget exceeded",
            Self::TotalValueBudget => "map/reduce cumulative serialized-value budget exceeded",
            Self::ResultBudget => "map/reduce complete result byte budget exceeded",
            Self::Serialization => "map/reduce value is not canonical JSON data",
            Self::AllocationRefused => "map/reduce allocation refused",
            Self::Checkpoint(_) => "map/reduce checkpoint refused execution",
            Self::Task { .. } => "map/reduce task execution failed",
            Self::Invariant => "map/reduce lineage invariant failed",
        })
    }
}

impl<E: Error + 'static> Error for ExecutionError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Task { source, .. } => Some(source),
            Self::Checkpoint(source) => Some(source),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReductionWarning {
    SingleContextEquivalenceNotEstablished,
    PolicyMayDiscardInformation,
}

#[derive(Serialize)]
pub struct MapReduceResult<T> {
    schema_version: u32,
    chunk_profile: &'static str,
    execution_profile: &'static str,
    policy: ReductionPolicy,
    chunk_count: usize,
    map_batches: usize,
    reduce_calls: usize,
    reduction_levels: usize,
    cumulative_value_bytes: usize,
    warnings: Vec<ReductionWarning>,
    root: ReductionNode<T>,
}

impl<T> MapReduceResult<T> {
    pub fn root(&self) -> &ReductionNode<T> { &self.root }
    pub fn into_value(self) -> T { self.root.value }
    pub fn warnings(&self) -> &[ReductionWarning] { &self.warnings }
    pub fn map_batches(&self) -> usize { self.map_batches }
    pub fn reduce_calls(&self) -> usize { self.reduce_calls }
    pub fn reduction_levels(&self) -> usize { self.reduction_levels }
}

/// Execute a fixed tree. Map completion order cannot affect reduction order.
/// A singleton group is carried unchanged; no synthetic reduce call or value
/// is inserted. Errors/cancellation return no partial success result. Callbacks
/// must not publish externally visible partial results on this API's behalf.
pub fn execute<T, C>(
    plan: &ChunkPlan<'_>,
    task: &mut T,
    limits: ExecutionLimits,
    mut checkpoint: C,
) -> Result<MapReduceResult<T::Value>, ExecutionError<T::Error>>
where
    T: MapReduceTask,
    C: FnMut() -> Result<(), MapReduceError>,
{
    if !limits.validate() { return Err(ExecutionError::InvalidLimits); }
    let policy = task.policy();
    if policy.id.is_empty() || policy.id.len() > 128
        || !policy.id.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
    {
        return Err(ExecutionError::InvalidPolicy);
    }
    let chunks = plan.chunks();
    if chunks.is_empty() { return Err(ExecutionError::EmptySource); }
    let map_batches = chunks.len().div_ceil(limits.map_batch_chunks);
    let (expected_levels, expected_reductions) = tree_cost(chunks.len(), limits.reduce_fan_in);
    if expected_levels > limits.max_reduction_levels
        || map_batches.checked_add(expected_reductions).is_none_or(|n| n > limits.max_task_calls)
    {
        return Err(ExecutionError::WorkBudget);
    }
    checkpoint().map_err(ExecutionError::Checkpoint)?;
    let mut accounting = Accounting { live: 0, total: 0 };
    let mut frontier = Vec::new();
    frontier.try_reserve_exact(chunks.len()).map_err(|_| ExecutionError::AllocationRefused)?;
    for batch in chunks.chunks(limits.map_batch_chunks) {
        checkpoint().map_err(ExecutionError::Checkpoint)?;
        let first = batch.first().ok_or(ExecutionError::Invariant)?;
        let mut outputs = task.map_batch(batch).map_err(|source| ExecutionError::Task {
            stage: TaskStage::Map { first_chunk: first.id() }, source,
        })?;
        checkpoint().map_err(ExecutionError::Checkpoint)?;
        if outputs.len() != batch.len() { return Err(ExecutionError::InvalidBatch); }
        outputs.sort_unstable_by_key(|output| output.chunk_id);
        // Check the complete id set before accepting any member of this batch.
        if outputs.iter().zip(batch).any(|(output, chunk)| output.chunk_id != chunk.id()) {
            return Err(ExecutionError::InvalidBatch);
        }
        for (output, chunk) in outputs.into_iter().zip(batch) {
            let bytes = value_size(&output.value, limits.max_value_bytes).map_err(size_error)?;
            accounting.admit(bytes, limits)?;
            frontier.push(ReductionNode { first_chunk: chunk.id(), end_chunk: chunk.id() + 1,
                source_span: chunk.span(), value: output.value, value_bytes: bytes });
        }
    }
    let (mut level, mut reduce_calls) = (0, 0);
    while frontier.len() > 1 {
        level += 1;
        let mut next = Vec::new();
        next.try_reserve_exact(frontier.len().div_ceil(limits.reduce_fan_in))
            .map_err(|_| ExecutionError::AllocationRefused)?;
        let mut remaining = frontier.into_iter();
        let mut group_number = 0;
        while remaining.len() != 0 {
            checkpoint().map_err(ExecutionError::Checkpoint)?;
            let mut group = Vec::new();
            group.try_reserve_exact(remaining.len().min(limits.reduce_fan_in))
                .map_err(|_| ExecutionError::AllocationRefused)?;
            group.extend(remaining.by_ref().take(limits.reduce_fan_in));
            if group.len() == 1 {
                next.push(group.pop().ok_or(ExecutionError::Invariant)?);
            } else {
                let first = group.first().ok_or(ExecutionError::Invariant)?;
                let last = group.last().ok_or(ExecutionError::Invariant)?;
                let (first_chunk, end_chunk) = (first.first_chunk, last.end_chunk);
                let source_span = VerifiedSourceSpan { byte_start: first.source_span.byte_start,
                    byte_end: last.source_span.byte_end, scalar_start: first.source_span.scalar_start,
                    scalar_end: last.source_span.scalar_end };
                let child_bytes: usize = group.iter().map(|node| node.value_bytes).sum();
                let value = task.reduce(ReduceInput { level, group: group_number, children: &group })
                    .map_err(|source| ExecutionError::Task {
                        stage: TaskStage::Reduce { level, group: group_number }, source,
                    })?;
                checkpoint().map_err(ExecutionError::Checkpoint)?;
                let bytes = value_size(&value, limits.max_value_bytes).map_err(size_error)?;
                // New output and its inputs coexist until the callback returns
                // and its output passes validation. Account that peak explicitly.
                accounting.admit(bytes, limits)?;
                drop(group);
                accounting.live = accounting.live.checked_sub(child_bytes).ok_or(ExecutionError::Invariant)?;
                next.push(ReductionNode { first_chunk, end_chunk, source_span, value, value_bytes: bytes });
                reduce_calls += 1;
            }
            group_number += 1;
        }
        frontier = next;
    }
    if level != expected_levels || reduce_calls != expected_reductions { return Err(ExecutionError::Invariant); }
    let root = frontier.pop().ok_or(ExecutionError::Invariant)?;
    let mut warnings = Vec::new();
    warnings.try_reserve_exact(2).map_err(|_| ExecutionError::AllocationRefused)?;
    warnings.push(ReductionWarning::SingleContextEquivalenceNotEstablished);
    if policy.may_discard_information { warnings.push(ReductionWarning::PolicyMayDiscardInformation); }
    let result = MapReduceResult { schema_version: 1, chunk_profile: CHUNK_PROFILE,
        execution_profile: EXECUTION_PROFILE, policy, chunk_count: chunks.len(), map_batches,
        reduce_calls, reduction_levels: level, cumulative_value_bytes: accounting.total, warnings, root };
    value_size(&result, limits.max_result_bytes).map_err(|error| match error {
        SizeError::Limit => ExecutionError::ResultBudget,
        SizeError::Serialization => ExecutionError::Serialization,
    })?;
    checkpoint().map_err(ExecutionError::Checkpoint)?;
    Ok(result)
}

fn tree_cost(mut width: usize, fan_in: usize) -> (usize, usize) {
    let (mut levels, mut calls) = (0, 0);
    while width > 1 {
        calls += width / fan_in + usize::from(width % fan_in > 1);
        width = width.div_ceil(fan_in);
        levels += 1;
    }
    (levels, calls)
}

struct Accounting { live: usize, total: usize }
impl Accounting {
    fn admit<E>(&mut self, bytes: usize, limits: ExecutionLimits) -> Result<(), ExecutionError<E>> {
        let live = self.live.checked_add(bytes).filter(|&n| n <= limits.max_live_value_bytes)
            .ok_or(ExecutionError::LiveValueBudget)?;
        let total = self.total.checked_add(bytes).filter(|&n| n <= limits.max_total_value_bytes)
            .ok_or(ExecutionError::TotalValueBudget)?;
        self.live = live; self.total = total; Ok(())
    }
}

enum SizeError { Limit, Serialization }
fn size_error<E>(error: SizeError) -> ExecutionError<E> {
    match error { SizeError::Limit => ExecutionError::ValueBudget, SizeError::Serialization => ExecutionError::Serialization }
}

/// Reject oversized serialization using a non-allocating sink before asking
/// the canonical writer to allocate. The second pass also rejects non-finite
/// floats and other noncanonical values. Callers supply stable Serialize data.
fn value_size(value: &impl Serialize, cap: usize) -> Result<usize, SizeError> {
    struct Counter { bytes: usize, cap: usize, exceeded: bool }
    impl io::Write for Counter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            if buf.len() > self.cap - self.bytes {
                self.exceeded = true;
                return Err(io::Error::other("serialized value budget exceeded"));
            }
            self.bytes += buf.len(); Ok(buf.len())
        }
        fn flush(&mut self) -> io::Result<()> { Ok(()) }
    }
    let mut counter = Counter { bytes: 0, cap, exceeded: false };
    if serde_json::to_writer(&mut counter, value).is_err() {
        return Err(if counter.exceeded { SizeError::Limit } else { SizeError::Serialization });
    }
    let bytes = canonjson::canonical_bytes(value).map_err(|_| SizeError::Serialization)?;
    if bytes.len() > cap { return Err(SizeError::Limit); }
    Ok(bytes.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::ChunkLimits;

    #[derive(Default)]
    struct Concat { maps: usize, reductions: usize, invalid_ids: bool, fail_reduce: bool }
    impl MapReduceTask for Concat {
        type Value = String;
        type Error = &'static str;
        fn policy(&self) -> ReductionPolicy {
            ReductionPolicy { id: "concatenate-source-v1", may_discard_information: false }
        }
        fn map_batch(&mut self, chunks: &[SourceChunk<'_>]) -> Result<Vec<MapOutput<String>>, Self::Error> {
            self.maps += 1;
            Ok(chunks.iter().rev().map(|chunk| MapOutput {
                chunk_id: if self.invalid_ids { usize::MAX } else { chunk.id() }, value: chunk.text().to_owned(),
            }).collect())
        }
        fn reduce(&mut self, input: ReduceInput<'_, String>) -> Result<String, Self::Error> {
            self.reductions += 1;
            if self.fail_reduce { return Err("deliberately unavailable"); }
            for pair in input.children.windows(2) {
                assert_eq!(pair[0].chunk_range().end, pair[1].chunk_range().start);
                assert_eq!(pair[0].source_span().byte_end, pair[1].source_span().byte_start);
            }
            Ok(input.children.iter().map(|node| node.value().as_str()).collect())
        }
    }
    fn plan(source: &str) -> ChunkPlan<'_> {
        ChunkPlan::build(source, ChunkLimits { max_chunk_bytes: 4, max_chunk_tokens: 4,
            ..ChunkLimits::default() }, |text| Ok(text.chars().count())).unwrap()
    }
    fn limits() -> ExecutionLimits {
        ExecutionLimits { map_batch_chunks: 2, reduce_fan_in: 2, ..ExecutionLimits::default() }
    }

    #[test]
    fn reversed_map_completion_still_reduces_in_source_order_with_full_lineage() {
        let source = "abcdefghijklmnopqrst";
        let plan = plan(source);
        let mut task = Concat::default();
        let result = execute(&plan, &mut task, limits(), || Ok(())).unwrap();
        assert_eq!(result.root().value(), source);
        assert_eq!(result.root().chunk_range(), 0..5);
        let span = result.root().source_span();
        assert_eq!((span.byte_start, span.byte_end, span.scalar_start, span.scalar_end), (0, 20, 0, 20));
        assert_eq!((result.map_batches(), result.reduce_calls(), result.reduction_levels()), (3, 4, 3));
        assert_eq!(result.warnings(), &[ReductionWarning::SingleContextEquivalenceNotEstablished]);
        let replay = execute(&plan, &mut Concat::default(), limits(), || Ok(())).unwrap();
        assert_eq!(canonjson::canonical_bytes(&result).unwrap(), canonjson::canonical_bytes(&replay).unwrap());
    }

    #[test]
    fn singleton_does_not_invent_reduction_and_empty_source_is_no_result() {
        let result = execute(&plan("é"), &mut Concat::default(), limits(), || Ok(())).unwrap();
        assert_eq!(result.reduce_calls(), 0);
        assert_eq!(result.into_value(), "é");
        let mut task = Concat::default();
        assert!(matches!(execute(&plan(""), &mut task, limits(), || Ok(())), Err(ExecutionError::EmptySource)));
        assert_eq!(task.maps, 0);
    }

    #[test]
    fn impossible_work_budget_rejects_before_model_work() {
        let mut task = Concat::default();
        let options = ExecutionLimits { max_reduction_levels: 2, ..limits() };
        assert!(matches!(execute(&plan("abcdefghijklmnopqrst"), &mut task, options, || Ok(())),
            Err(ExecutionError::WorkBudget)));
        assert_eq!(task.maps, 0);
        let options = ExecutionLimits { max_task_calls: 1, ..limits() };
        assert!(matches!(execute(&plan("abcdefgh"), &mut task, options, || Ok(())), Err(ExecutionError::WorkBudget)));
        assert_eq!(task.maps, 0);
    }

    #[test]
    fn forged_or_duplicate_ids_cannot_be_attached_to_a_different_source() {
        let mut task = Concat { invalid_ids: true, ..Concat::default() };
        assert!(matches!(execute(&plan("abcdefgh"), &mut task, limits(), || Ok(())), Err(ExecutionError::InvalidBatch)));
        assert_eq!(task.reductions, 0);
    }

    #[test]
    fn cancellation_after_map_returns_no_result_and_never_reduces() {
        let mut task = Concat::default();
        let mut checks = 0;
        let result = execute(&plan("abcdefgh"), &mut task, limits(), || {
            checks += 1;
            if checks == 3 { Err(MapReduceError::Cancelled) } else { Ok(()) }
        });
        assert!(matches!(result, Err(ExecutionError::Checkpoint(MapReduceError::Cancelled))));
        assert_eq!((task.maps, task.reductions), (1, 0));
    }

    #[test]
    fn native_task_error_retains_the_stage_and_original_error() {
        let mut task = Concat { fail_reduce: true, ..Concat::default() };
        assert!(matches!(execute(&plan("abcdefgh"), &mut task, limits(), || Ok(())),
            Err(ExecutionError::Task { stage: TaskStage::Reduce { level: 1, group: 0 }, source: "deliberately unavailable" })));
    }

    #[test]
    fn live_budget_includes_old_and_new_values_at_reduce_boundary() {
        // Two four-character JSON strings use 12 bytes, plus the combined
        // eight-character JSON string uses 10: peak 22, not final size 10.
        let options = ExecutionLimits { max_value_bytes: 12, max_live_value_bytes: 21, ..limits() };
        assert!(matches!(execute(&plan("abcdefgh"), &mut Concat::default(), options, || Ok(())),
            Err(ExecutionError::LiveValueBudget)));
        let options = ExecutionLimits { max_live_value_bytes: 22, ..options };
        assert!(execute(&plan("abcdefgh"), &mut Concat::default(), options, || Ok(())).is_ok());
    }

    #[test]
    fn cumulative_and_complete_envelope_budgets_are_separate() {
        let options = ExecutionLimits { max_total_value_bytes: 21, ..limits() };
        assert!(matches!(execute(&plan("abcdefgh"), &mut Concat::default(), options, || Ok(())),
            Err(ExecutionError::TotalValueBudget)));
        let options = ExecutionLimits { max_result_bytes: 8, ..limits() };
        assert!(matches!(execute(&plan("abc"), &mut Concat::default(), options, || Ok(())),
            Err(ExecutionError::ResultBudget)));
    }

    #[test]
    fn serialization_budget_counts_escaping_and_rejects_nonfinite_numbers() {
        assert!(matches!(value_size(&"\\\\", 5), Err(SizeError::Limit)));
        assert!(matches!(value_size(&f64::NAN, 128), Err(SizeError::Serialization)));
        assert!(matches!(value_size(&f64::INFINITY, 128), Err(SizeError::Serialization)));
    }

    #[test]
    fn cost_matches_actual_tree_for_all_small_shapes() {
        for count in 1..=128 {
            for fan_in in 2..=16 {
                let (mut width, mut levels, mut calls) = (count, 0, 0);
                while width > 1 {
                    let mut next = 0;
                    for start in (0..width).step_by(fan_in) {
                        calls += usize::from((width - start).min(fan_in) > 1);
                        next += 1;
                    }
                    width = next; levels += 1;
                }
                assert_eq!(tree_cost(count, fan_in), (levels, calls));
            }
        }
    }
}
