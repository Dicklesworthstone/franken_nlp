//! Process-wide scheduler resources and admission accounting.
//!
//! This module owns the one [`EngineResources`] broker shared by every future
//! [`NlpEngine`].  It deliberately establishes the process-domain accounting
//! before a request scheduler is wired to it: no library code reads ambient
//! environment variables, no caller creates a second runtime host, and every
//! allocation is charged through an owned two-phase reservation guard.
//!
//! The concrete preset worker counts are the pin-scoped OQ-35 census values:
//! `current_thread=1`, `low_latency=4`, and `high_throughput=8`.  They are
//! recorded configuration inputs, not a CPU-team sizing certificate.

use std::{
    cell::Cell,
    collections::BTreeMap,
    fmt,
    marker::PhantomData,
    rc::Rc,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};

thread_local! {
    /// A synchronous facade call may not nest a second runtime entry on one
    /// thread. The guard is intentionally process-local because a production
    /// process has exactly one broker host.
    static ACTIVE_SYNC_RESOURCE: Cell<Option<usize>> = const { Cell::new(None) };
}

#[cfg(feature = "asupersync-runtime")]
thread_local! {
    /// Set only by the runtime builder's worker lifecycle callbacks. Ambient
    /// `Cx::current()` is deliberately not consulted: the census records that
    /// it is not a least-authority boundary at this pin.
    static RUNTIME_WORKER_THREAD: Cell<bool> = const { Cell::new(false) };
}

#[cfg(feature = "asupersync-runtime")]
fn mark_runtime_worker_started() {
    RUNTIME_WORKER_THREAD.with(|worker| worker.set(true));
}

#[cfg(feature = "asupersync-runtime")]
fn mark_runtime_worker_stopped() {
    RUNTIME_WORKER_THREAD.with(|worker| worker.set(false));
}

#[cfg(feature = "asupersync-runtime")]
fn is_runtime_worker_thread() -> bool {
    RUNTIME_WORKER_THREAD.with(Cell::get)
}

#[cfg(not(feature = "asupersync-runtime"))]
const fn is_runtime_worker_thread() -> bool {
    false
}

/// The asupersync runtime-builder profile selected by an entrypoint.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum RuntimePreset {
    /// One-shot CLI and deterministic-first work.
    CurrentThread,
    /// Interactive work with low-latency scheduler tuning.
    LowLatency,
    /// Bounded batch work with a larger ready-lane batch.
    HighThroughput,
}

impl RuntimePreset {
    /// Stable robot/receipt spelling.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CurrentThread => "current_thread",
            Self::LowLatency => "low_latency",
            Self::HighThroughput => "high_throughput",
        }
    }

    /// Concrete worker count observed by the OQ-35 census at the locked pin.
    ///
    /// `low_latency` retains the pin's deterministic default worker count;
    /// its distinct observed values are scheduling knobs, not worker count.
    pub const fn observed_runtime_workers(self) -> usize {
        match self {
            Self::CurrentThread => 1,
            Self::LowLatency => 4,
            Self::HighThroughput => 8,
        }
    }
}

/// The independently counted components of the process runnable-thread bound.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ThreadInventory {
    /// Runtime scheduler workers.
    pub runtime_workers: usize,
    /// Maximum simultaneous `spawn_blocking` coordinators.
    pub blocking_coordinators: usize,
    /// Scoped CPU workers per blocking coordinator.
    pub scoped_cpu_children_per_coordinator: usize,
    /// Long-lived helpers separately admitted by a future scheduler surface.
    pub helper_threads: usize,
    /// The total worst-case runnable threads covered by this inventory.
    pub total_runnable_threads: usize,
}

/// Policy for an obligation dropped before commit or abort.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum LeakResponsePolicy {
    /// Lab/CI: fail immediately after restoring the ledger's balance.
    PanicInLabOrCi,
    /// Production: retain a leak row and write an escalation diagnostic.
    RecordAndEscalate,
}

impl LeakResponsePolicy {
    const fn as_str(self) -> &'static str {
        match self {
            Self::PanicInLabOrCi => "panic_in_lab_or_ci",
            Self::RecordAndEscalate => "record_and_escalate",
        }
    }
}

/// Fields fixed by the first successful process-host installation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResourceConfigField {
    RuntimePreset,
    RuntimeWorkers,
    BlockingCoordinators,
    ScopedCpuChildrenPerCoordinator,
    HelperThreads,
    ThreadCeiling,
    MemoryCeilingBytes,
    LeakResponsePolicy,
}

impl ResourceConfigField {
    const ALL: [Self; 8] = [
        Self::RuntimePreset,
        Self::RuntimeWorkers,
        Self::BlockingCoordinators,
        Self::ScopedCpuChildrenPerCoordinator,
        Self::HelperThreads,
        Self::ThreadCeiling,
        Self::MemoryCeilingBytes,
        Self::LeakResponsePolicy,
    ];

    /// Stable diagnostic spelling for a configuration conflict.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RuntimePreset => "runtime_preset",
            Self::RuntimeWorkers => "runtime_workers",
            Self::BlockingCoordinators => "blocking_coordinators",
            Self::ScopedCpuChildrenPerCoordinator => "scoped_cpu_children_per_coordinator",
            Self::HelperThreads => "helper_threads",
            Self::ThreadCeiling => "thread_ceiling",
            Self::MemoryCeilingBytes => "memory_ceiling_bytes",
            Self::LeakResponsePolicy => "leak_response_policy",
        }
    }
}

/// A typed configuration value carried in a field-level conflict.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResourceConfigValue {
    Preset(RuntimePreset),
    Count(usize),
    Bytes(u64),
    LeakPolicy(LeakResponsePolicy),
}

impl fmt::Display for ResourceConfigValue {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Preset(value) => formatter.write_str(value.as_str()),
            Self::Count(value) => write!(formatter, "{value}"),
            Self::Bytes(value) => write!(formatter, "{value}"),
            Self::LeakPolicy(value) => formatter.write_str(value.as_str()),
        }
    }
}

/// Fixed process-host settings supplied by the first engine builder.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResourceHostConfig {
    pub runtime_preset: RuntimePreset,
    pub runtime_workers: usize,
    pub max_blocking_coordinators: usize,
    pub scoped_cpu_children_per_coordinator: usize,
    pub helper_threads: usize,
    pub thread_ceiling: usize,
    pub memory_ceiling_bytes: u64,
    pub leak_response_policy: LeakResponsePolicy,
}

impl ResourceHostConfig {
    /// Start from a census-observed runtime preset while keeping every host
    /// dependent limit explicit.  In particular, this never guesses physical
    /// cores or host memory.
    pub fn for_preset(
        runtime_preset: RuntimePreset,
        max_blocking_coordinators: usize,
        scoped_cpu_children_per_coordinator: usize,
        helper_threads: usize,
        thread_ceiling: usize,
        memory_ceiling_bytes: u64,
    ) -> Result<Self, ResourceConfigError> {
        Self::new(
            runtime_preset,
            runtime_preset.observed_runtime_workers(),
            max_blocking_coordinators,
            scoped_cpu_children_per_coordinator,
            helper_threads,
            thread_ceiling,
            memory_ceiling_bytes,
            LeakResponsePolicy::RecordAndEscalate,
        )
    }

    /// Construct an explicitly overridden host envelope.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        runtime_preset: RuntimePreset,
        runtime_workers: usize,
        max_blocking_coordinators: usize,
        scoped_cpu_children_per_coordinator: usize,
        helper_threads: usize,
        thread_ceiling: usize,
        memory_ceiling_bytes: u64,
        leak_response_policy: LeakResponsePolicy,
    ) -> Result<Self, ResourceConfigError> {
        let config = Self {
            runtime_preset,
            runtime_workers,
            max_blocking_coordinators,
            scoped_cpu_children_per_coordinator,
            helper_threads,
            thread_ceiling,
            memory_ceiling_bytes,
            leak_response_policy,
        };
        config.validate()?;
        Ok(config)
    }

    /// Recompute the complete process thread inventory using checked math.
    pub fn thread_inventory(self) -> Result<ThreadInventory, ResourceConfigError> {
        self.validate()?;
        let scoped_cpu_children = self
            .max_blocking_coordinators
            .checked_mul(self.scoped_cpu_children_per_coordinator)
            .ok_or(ResourceConfigError::ThreadEnvelopeOverflow)?;
        let total_runnable_threads = self
            .runtime_workers
            .checked_add(self.max_blocking_coordinators)
            .and_then(|total| total.checked_add(scoped_cpu_children))
            .and_then(|total| total.checked_add(self.helper_threads))
            .ok_or(ResourceConfigError::ThreadEnvelopeOverflow)?;
        if total_runnable_threads > self.thread_ceiling {
            return Err(ResourceConfigError::ThreadEnvelopeExceedsCeiling {
                required: total_runnable_threads,
                ceiling: self.thread_ceiling,
            });
        }
        Ok(ThreadInventory {
            runtime_workers: self.runtime_workers,
            blocking_coordinators: self.max_blocking_coordinators,
            scoped_cpu_children_per_coordinator: self.scoped_cpu_children_per_coordinator,
            helper_threads: self.helper_threads,
            total_runnable_threads,
        })
    }

    fn validate(self) -> Result<(), ResourceConfigError> {
        for (field, value) in [
            ("runtime_workers", self.runtime_workers),
            ("max_blocking_coordinators", self.max_blocking_coordinators),
            (
                "scoped_cpu_children_per_coordinator",
                self.scoped_cpu_children_per_coordinator,
            ),
            ("thread_ceiling", self.thread_ceiling),
        ] {
            if value == 0 {
                return Err(ResourceConfigError::ZeroThreadLimit { field });
            }
        }
        if self.memory_ceiling_bytes == 0 {
            return Err(ResourceConfigError::ZeroMemoryCeiling);
        }
        let required = self.thread_inventory_unchecked()?;
        if required > self.thread_ceiling {
            return Err(ResourceConfigError::ThreadEnvelopeExceedsCeiling {
                required,
                ceiling: self.thread_ceiling,
            });
        }
        Ok(())
    }

    fn thread_inventory_unchecked(self) -> Result<usize, ResourceConfigError> {
        self.runtime_workers
            .checked_add(self.max_blocking_coordinators)
            .and_then(|total| {
                self.max_blocking_coordinators
                    .checked_mul(self.scoped_cpu_children_per_coordinator)
                    .and_then(|children| total.checked_add(children))
            })
            .and_then(|total| total.checked_add(self.helper_threads))
            .ok_or(ResourceConfigError::ThreadEnvelopeOverflow)
    }

    fn first_conflict(self, requested: Self) -> Option<ResourceConfigConflict> {
        ResourceConfigField::ALL.into_iter().find_map(|field| {
            let installed = self.value_for(field);
            let requested_value = requested.value_for(field);
            (installed != requested_value).then_some(ResourceConfigConflict {
                field,
                installed,
                requested: requested_value,
            })
        })
    }

    const fn value_for(self, field: ResourceConfigField) -> ResourceConfigValue {
        match field {
            ResourceConfigField::RuntimePreset => ResourceConfigValue::Preset(self.runtime_preset),
            ResourceConfigField::RuntimeWorkers => ResourceConfigValue::Count(self.runtime_workers),
            ResourceConfigField::BlockingCoordinators => {
                ResourceConfigValue::Count(self.max_blocking_coordinators)
            }
            ResourceConfigField::ScopedCpuChildrenPerCoordinator => {
                ResourceConfigValue::Count(self.scoped_cpu_children_per_coordinator)
            }
            ResourceConfigField::HelperThreads => ResourceConfigValue::Count(self.helper_threads),
            ResourceConfigField::ThreadCeiling => ResourceConfigValue::Count(self.thread_ceiling),
            ResourceConfigField::MemoryCeilingBytes => {
                ResourceConfigValue::Bytes(self.memory_ceiling_bytes)
            }
            ResourceConfigField::LeakResponsePolicy => {
                ResourceConfigValue::LeakPolicy(self.leak_response_policy)
            }
        }
    }
}

/// A malformed host envelope was rejected before any process state changed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResourceConfigError {
    ZeroThreadLimit { field: &'static str },
    ZeroMemoryCeiling,
    ThreadEnvelopeOverflow,
    ThreadEnvelopeExceedsCeiling { required: usize, ceiling: usize },
}

impl fmt::Display for ResourceConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroThreadLimit { field } => write!(formatter, "{field} must be non-zero"),
            Self::ZeroMemoryCeiling => formatter.write_str("memory_ceiling_bytes must be non-zero"),
            Self::ThreadEnvelopeOverflow => formatter.write_str("thread envelope arithmetic overflow"),
            Self::ThreadEnvelopeExceedsCeiling { required, ceiling } => write!(
                formatter,
                "thread envelope requires {required} runnable threads but ceiling is {ceiling}"
            ),
        }
    }
}

impl std::error::Error for ResourceConfigError {}

/// A later builder requested a different fixed process-host field.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResourceConfigConflict {
    pub field: ResourceConfigField,
    pub installed: ResourceConfigValue,
    pub requested: ResourceConfigValue,
}

impl fmt::Display for ResourceConfigConflict {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "resource host conflict field={} installed={} requested={}",
            self.field.as_str(),
            self.installed,
            self.requested,
        )
    }
}

impl std::error::Error for ResourceConfigConflict {}

/// Failure to install or reuse the one process resource host.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResourceBrokerError {
    InvalidConfig(ResourceConfigError),
    ConfigConflict(ResourceConfigConflict),
    Runtime(RuntimeHostError),
}

impl fmt::Display for ResourceBrokerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(error) => write!(formatter, "invalid resource host config: {error}"),
            Self::ConfigConflict(error) => error.fmt(formatter),
            Self::Runtime(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for ResourceBrokerError {}

/// The process host could not establish the required production runtime seam.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RuntimeHostError {
    RuntimeBuild { detail: String },
    MissingBlockingPool,
}

impl fmt::Display for RuntimeHostError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RuntimeBuild { detail } => {
                write!(formatter, "asupersync runtime construction failed: {detail}")
            }
            Self::MissingBlockingPool => formatter.write_str(
                "asupersync runtime has no blocking-pool handle; inline fallback is not a production route",
            ),
        }
    }
}

impl std::error::Error for RuntimeHostError {}

/// A category that must be charged to the process-wide memory ledger.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum MemoryClass {
    Weights,
    KvPages,
    ActivationScratch,
    LogitScratch,
    PrefixCache,
    GrammarCache,
    JobBuffers,
    Staging,
}

impl MemoryClass {
    /// Stable receipt/health spelling.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Weights => "weights",
            Self::KvPages => "kv_pages",
            Self::ActivationScratch => "activation_scratch",
            Self::LogitScratch => "logit_scratch",
            Self::PrefixCache => "prefix_cache",
            Self::GrammarCache => "grammar_cache",
            Self::JobBuffers => "job_buffers",
            Self::Staging => "staging",
        }
    }
}

/// A per-class view of reserved and committed bytes.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MemoryClassCharge {
    pub reserved_bytes: u64,
    pub committed_bytes: u64,
}

/// A point-in-time ledger observation for future `robot health` output.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemorySnapshot {
    pub ceiling_bytes: u64,
    pub reserved_bytes: u64,
    pub committed_bytes: u64,
    pub outstanding_obligations: usize,
    pub recorded_leaks: usize,
    pub by_class: BTreeMap<MemoryClass, MemoryClassCharge>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ObligationState {
    Reserved,
    Committed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct OutstandingObligation {
    bytes: u64,
    engine_lease_id: u64,
    memory_class: MemoryClass,
    state: ObligationState,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct LeakRecord {
    obligation_id: u64,
    engine_lease_id: u64,
    bytes: u64,
    memory_class: MemoryClass,
}

#[derive(Debug, Default)]
struct MemoryLedgerState {
    reserved_bytes: u64,
    committed_bytes: u64,
    next_obligation_id: u64,
    obligations: BTreeMap<u64, OutstandingObligation>,
    leaks: Vec<LeakRecord>,
}

#[derive(Debug)]
struct MemoryLedger {
    ceiling_bytes: u64,
    leak_response_policy: LeakResponsePolicy,
    state: Mutex<MemoryLedgerState>,
}

/// Admission denial or an invalid reservation transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReservationError {
    CapacityExceeded {
        requested_bytes: u64,
        available_bytes: u64,
        ceiling_bytes: u64,
    },
    ArithmeticOverflow,
    UnknownObligation { obligation_id: u64 },
    InvalidTransition { obligation_id: u64 },
    LedgerInvariant { obligation_id: u64 },
}

impl fmt::Display for ReservationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CapacityExceeded {
                requested_bytes,
                available_bytes,
                ceiling_bytes,
            } => write!(
                formatter,
                "memory admission refused requested={requested_bytes} available={available_bytes} ceiling={ceiling_bytes}"
            ),
            Self::ArithmeticOverflow => formatter.write_str("memory ledger arithmetic overflow"),
            Self::UnknownObligation { obligation_id } => {
                write!(formatter, "unknown reservation obligation {obligation_id}")
            }
            Self::InvalidTransition { obligation_id } => {
                write!(formatter, "invalid reservation transition for obligation {obligation_id}")
            }
            Self::LedgerInvariant { obligation_id } => {
                write!(formatter, "memory ledger invariant failed for obligation {obligation_id}")
            }
        }
    }
}

impl std::error::Error for ReservationError {}

impl MemoryLedger {
    fn new(ceiling_bytes: u64, leak_response_policy: LeakResponsePolicy) -> Self {
        Self {
            ceiling_bytes,
            leak_response_policy,
            state: Mutex::new(MemoryLedgerState::default()),
        }
    }

    fn reserve(
        self: &Arc<Self>,
        engine_lease_id: u64,
        memory_class: MemoryClass,
        bytes: u64,
    ) -> Result<MemoryReservation, ReservationError> {
        let mut state = lock_unpoisoned(&self.state);
        let occupied = state
            .reserved_bytes
            .checked_add(state.committed_bytes)
            .ok_or(ReservationError::ArithmeticOverflow)?;
        let next_occupied = occupied
            .checked_add(bytes)
            .ok_or(ReservationError::ArithmeticOverflow)?;
        if next_occupied > self.ceiling_bytes {
            return Err(ReservationError::CapacityExceeded {
                requested_bytes: bytes,
                available_bytes: self.ceiling_bytes.saturating_sub(occupied),
                ceiling_bytes: self.ceiling_bytes,
            });
        }
        let obligation_id = state
            .next_obligation_id
            .checked_add(1)
            .ok_or(ReservationError::ArithmeticOverflow)?;
        state.next_obligation_id = obligation_id;
        state.reserved_bytes = state
            .reserved_bytes
            .checked_add(bytes)
            .ok_or(ReservationError::ArithmeticOverflow)?;
        state.obligations.insert(
            obligation_id,
            OutstandingObligation {
                bytes,
                engine_lease_id,
                memory_class,
                state: ObligationState::Reserved,
            },
        );
        Ok(MemoryReservation {
            ledger: Some(Arc::clone(self)),
            obligation_id,
            bytes,
            engine_lease_id,
            memory_class,
        })
    }

    fn commit(&self, obligation_id: u64) -> Result<(), ReservationError> {
        let mut state = lock_unpoisoned(&self.state);
        let Some(obligation) = state.obligations.get(&obligation_id).copied() else {
            return Err(ReservationError::UnknownObligation { obligation_id });
        };
        if obligation.state != ObligationState::Reserved {
            return Err(ReservationError::InvalidTransition { obligation_id });
        }
        state.reserved_bytes = state
            .reserved_bytes
            .checked_sub(obligation.bytes)
            .ok_or(ReservationError::LedgerInvariant { obligation_id })?;
        state.committed_bytes = state
            .committed_bytes
            .checked_add(obligation.bytes)
            .ok_or(ReservationError::ArithmeticOverflow)?;
        let Some(stored) = state.obligations.get_mut(&obligation_id) else {
            return Err(ReservationError::LedgerInvariant { obligation_id });
        };
        stored.state = ObligationState::Committed;
        Ok(())
    }

    fn abort(&self, obligation_id: u64) -> Result<(), ReservationError> {
        let mut state = lock_unpoisoned(&self.state);
        let Some(obligation) = state.obligations.get(&obligation_id).copied() else {
            return Err(ReservationError::UnknownObligation { obligation_id });
        };
        if obligation.state != ObligationState::Reserved {
            return Err(ReservationError::InvalidTransition { obligation_id });
        }
        state.reserved_bytes = state
            .reserved_bytes
            .checked_sub(obligation.bytes)
            .ok_or(ReservationError::LedgerInvariant { obligation_id })?;
        state.obligations.remove(&obligation_id);
        Ok(())
    }

    fn release_committed(&self, obligation_id: u64) {
        let mut state = lock_unpoisoned(&self.state);
        let Some(obligation) = state.obligations.get(&obligation_id).copied() else {
            eprintln!("ENGINE_RESOURCES RELEASE_UNKNOWN obligation_id={obligation_id}");
            return;
        };
        if obligation.state != ObligationState::Committed {
            eprintln!("ENGINE_RESOURCES RELEASE_UNCOMMITTED obligation_id={obligation_id}");
            return;
        }
        let Some(next_committed) = state.committed_bytes.checked_sub(obligation.bytes) else {
            eprintln!("ENGINE_RESOURCES RELEASE_INVARIANT obligation_id={obligation_id}");
            return;
        };
        state.committed_bytes = next_committed;
        state.obligations.remove(&obligation_id);
    }

    fn record_leak(&self, obligation_id: u64) -> LeakResponsePolicy {
        let mut state = lock_unpoisoned(&self.state);
        let Some(obligation) = state.obligations.get(&obligation_id).copied() else {
            return self.leak_response_policy;
        };
        if obligation.state != ObligationState::Reserved {
            return self.leak_response_policy;
        }
        let Some(next_reserved) = state.reserved_bytes.checked_sub(obligation.bytes) else {
            eprintln!("ENGINE_RESOURCES LEAK_INVARIANT obligation_id={obligation_id}");
            return self.leak_response_policy;
        };
        state.reserved_bytes = next_reserved;
        state.obligations.remove(&obligation_id);
        state.leaks.push(LeakRecord {
            obligation_id,
            engine_lease_id: obligation.engine_lease_id,
            bytes: obligation.bytes,
            memory_class: obligation.memory_class,
        });
        eprintln!(
            "ENGINE_RESOURCES OBLIGATION_LEAK obligation_id={} engine_lease_id={} bytes={} class={} policy={}",
            obligation_id,
            obligation.engine_lease_id,
            obligation.bytes,
            obligation.memory_class.as_str(),
            self.leak_response_policy.as_str(),
        );
        self.leak_response_policy
    }

    fn snapshot(&self) -> MemorySnapshot {
        let state = lock_unpoisoned(&self.state);
        let mut by_class = BTreeMap::new();
        for obligation in state.obligations.values() {
            let charge = by_class
                .entry(obligation.memory_class)
                .or_insert_with(MemoryClassCharge::default);
            match obligation.state {
                ObligationState::Reserved => charge.reserved_bytes += obligation.bytes,
                ObligationState::Committed => charge.committed_bytes += obligation.bytes,
            }
        }
        MemorySnapshot {
            ceiling_bytes: self.ceiling_bytes,
            reserved_bytes: state.reserved_bytes,
            committed_bytes: state.committed_bytes,
            outstanding_obligations: state.obligations.len(),
            recorded_leaks: state.leaks.len(),
            by_class,
        }
    }
}

/// An uncommitted allocation admission obligation.
///
/// The guard must be explicitly committed after allocation succeeds or
/// explicitly aborted on error/cancellation. Dropping it is never a silent
/// rollback: lab/CI panics and production records an escalation diagnostic.
#[derive(Debug)]
pub struct MemoryReservation {
    ledger: Option<Arc<MemoryLedger>>,
    obligation_id: u64,
    bytes: u64,
    engine_lease_id: u64,
    memory_class: MemoryClass,
}

impl MemoryReservation {
    pub const fn obligation_id(&self) -> u64 {
        self.obligation_id
    }

    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    pub const fn engine_lease_id(&self) -> u64 {
        self.engine_lease_id
    }

    pub const fn memory_class(&self) -> MemoryClass {
        self.memory_class
    }

    /// Mark the reservation as successfully allocated and retain its process
    /// charge until the returned guard is released or dropped.
    pub fn commit(mut self) -> Result<CommittedMemory, ReservationError> {
        let ledger = self
            .ledger
            .as_ref()
            .ok_or(ReservationError::InvalidTransition {
                obligation_id: self.obligation_id,
            })?;
        ledger.commit(self.obligation_id)?;
        let ledger = self
            .ledger
            .take()
            .expect("reservation ledger exists after a successful commit");
        Ok(CommittedMemory {
            ledger: Some(ledger),
            obligation_id: self.obligation_id,
            bytes: self.bytes,
            engine_lease_id: self.engine_lease_id,
            memory_class: self.memory_class,
        })
    }

    /// Restore the exact reserved balance after failed allocation or cancel.
    pub fn abort(mut self) -> Result<(), ReservationError> {
        let ledger = self
            .ledger
            .as_ref()
            .ok_or(ReservationError::InvalidTransition {
                obligation_id: self.obligation_id,
            })?;
        ledger.abort(self.obligation_id)?;
        self.ledger
            .take()
            .expect("reservation ledger exists after a successful abort");
        Ok(())
    }
}

impl Drop for MemoryReservation {
    fn drop(&mut self) {
        let Some(ledger) = self.ledger.take() else {
            return;
        };
        if ledger.record_leak(self.obligation_id) == LeakResponsePolicy::PanicInLabOrCi {
            panic!(
                "EngineResources reservation leaked obligation_id={} engine_lease_id={} bytes={} class={}",
                self.obligation_id,
                self.engine_lease_id,
                self.bytes,
                self.memory_class.as_str(),
            );
        }
    }
}

/// A committed allocation charge. Its lifetime is the allocation lifetime.
#[derive(Debug)]
pub struct CommittedMemory {
    ledger: Option<Arc<MemoryLedger>>,
    obligation_id: u64,
    bytes: u64,
    engine_lease_id: u64,
    memory_class: MemoryClass,
}

impl CommittedMemory {
    pub const fn obligation_id(&self) -> u64 {
        self.obligation_id
    }

    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    pub const fn engine_lease_id(&self) -> u64 {
        self.engine_lease_id
    }

    pub const fn memory_class(&self) -> MemoryClass {
        self.memory_class
    }

    /// Release the allocation charge before the guard's natural drop.
    pub fn release(mut self) {
        if let Some(ledger) = self.ledger.take() {
            ledger.release_committed(self.obligation_id);
        }
    }
}

impl Drop for CommittedMemory {
    fn drop(&mut self) {
        if let Some(ledger) = self.ledger.take() {
            ledger.release_committed(self.obligation_id);
        }
    }
}

#[derive(Debug, Default)]
struct LeaseState {
    active: BTreeMap<u64, ()>,
}

#[cfg(feature = "asupersync-runtime")]
struct RuntimeHost {
    runtime: asupersync::runtime::Runtime,
    blocking_pool: asupersync::runtime::blocking_pool::BlockingPoolHandle,
}

#[cfg(feature = "asupersync-runtime")]
impl RuntimeHost {
    fn build(config: ResourceHostConfig) -> Result<Self, RuntimeHostError> {
        use asupersync::runtime::RuntimeBuilder;

        let builder = match config.runtime_preset {
            RuntimePreset::CurrentThread => RuntimeBuilder::current_thread(),
            RuntimePreset::LowLatency => RuntimeBuilder::low_latency(),
            RuntimePreset::HighThroughput => RuntimeBuilder::high_throughput(),
        }
        .worker_threads(config.runtime_workers)
        // A nonzero pool is a production precondition. The configured maximum
        // is independently counted in the thread inventory above.
        .blocking_threads(1, config.max_blocking_coordinators)
        .on_thread_start(mark_runtime_worker_started)
        .on_thread_stop(mark_runtime_worker_stopped);
        let runtime = builder
            .build()
            .map_err(|error| RuntimeHostError::RuntimeBuild {
                detail: error.to_string(),
            })?;
        let blocking_pool = runtime
            .blocking_handle()
            .ok_or(RuntimeHostError::MissingBlockingPool)?;
        Ok(Self {
            runtime,
            blocking_pool,
        })
    }
}

#[cfg(feature = "asupersync-runtime")]
impl fmt::Debug for RuntimeHost {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RuntimeHost")
            .field("blocking_pool", &"configured")
            .finish_non_exhaustive()
    }
}

/// The process-owned runtime/admission/memory host shared by engine leases.
#[derive(Debug)]
pub struct EngineResources {
    config: ResourceHostConfig,
    thread_inventory: ThreadInventory,
    memory: Arc<MemoryLedger>,
    next_lease_id: AtomicU64,
    leases: Mutex<LeaseState>,
    #[cfg(feature = "asupersync-runtime")]
    runtime: RuntimeHost,
}

impl EngineResources {
    fn new(
        config: ResourceHostConfig,
        thread_inventory: ThreadInventory,
    ) -> Result<Self, RuntimeHostError> {
        #[cfg(feature = "asupersync-runtime")]
        let runtime = RuntimeHost::build(config)?;
        Ok(Self {
            config,
            thread_inventory,
            memory: Arc::new(MemoryLedger::new(
                config.memory_ceiling_bytes,
                config.leak_response_policy,
            )),
            next_lease_id: AtomicU64::new(0),
            leases: Mutex::new(LeaseState::default()),
            #[cfg(feature = "asupersync-runtime")]
            runtime,
        })
    }

    /// Fixed configuration selected by the first successful installation.
    pub const fn config(&self) -> ResourceHostConfig {
        self.config
    }

    /// Complete runnable-thread accounting used by future health/receipt code.
    pub const fn thread_inventory(&self) -> ThreadInventory {
        self.thread_inventory
    }

    /// Current aggregate memory charge snapshot.
    pub fn memory_snapshot(&self) -> MemorySnapshot {
        self.memory.snapshot()
    }

    /// Acquire one engine lease. An engine cannot own an independent memory or
    /// runtime domain.
    pub fn acquire_lease(self: &Arc<Self>) -> EngineLease {
        let lease_id = self.next_lease_id.fetch_add(1, Ordering::Relaxed) + 1;
        let mut leases = lock_unpoisoned(&self.leases);
        leases.active.insert(lease_id, ());
        EngineLease {
            resources: Arc::clone(self),
            lease_id,
        }
    }

    /// Number of still-live engine leases for diagnostic inventory only.
    pub fn active_lease_count(&self) -> usize {
        lock_unpoisoned(&self.leases).active.len()
    }

    /// Whether this build has retained the required real blocking-pool handle.
    /// The default crate feature intentionally returns false because it cannot
    /// execute production requests without `asupersync-runtime`.
    #[cfg(feature = "asupersync-runtime")]
    pub fn has_real_blocking_pool(&self) -> bool {
        let _ = &self.runtime.blocking_pool;
        true
    }

    /// Whether this build has retained the required real blocking-pool handle.
    #[cfg(not(feature = "asupersync-runtime"))]
    pub fn has_real_blocking_pool(&self) -> bool {
        false
    }

    #[cfg(feature = "asupersync-runtime")]
    pub(crate) fn runtime(&self) -> &asupersync::runtime::Runtime {
        &self.runtime.runtime
    }
}

/// One engine's handle into the process-wide resource domain.
#[derive(Debug)]
pub struct EngineLease {
    resources: Arc<EngineResources>,
    lease_id: u64,
}

impl EngineLease {
    pub const fn id(&self) -> u64 {
        self.lease_id
    }

    /// Reserve aggregate process memory before the caller allocates bytes.
    pub fn reserve(
        &self,
        memory_class: MemoryClass,
        bytes: u64,
    ) -> Result<MemoryReservation, ReservationError> {
        self.resources
            .memory
            .reserve(self.lease_id, memory_class, bytes)
    }

    pub fn resources(&self) -> &Arc<EngineResources> {
        &self.resources
    }
}

impl Drop for EngineLease {
    fn drop(&mut self) {
        lock_unpoisoned(&self.resources.leases)
            .active
            .remove(&self.lease_id);
    }
}

/// The single process broker. Its mutex ensures a racing first install builds
/// one host rather than transiently creating multiple resource domains.
#[derive(Debug, Default)]
pub struct ResourceBroker {
    installed: Mutex<Option<Arc<EngineResources>>>,
}

impl ResourceBroker {
    fn install(
        &self,
        requested: ResourceHostConfig,
    ) -> Result<Arc<EngineResources>, ResourceBrokerError> {
        let thread_inventory = requested
            .thread_inventory()
            .map_err(ResourceBrokerError::InvalidConfig)?;
        let mut installed = lock_unpoisoned(&self.installed);
        if let Some(existing) = installed.as_ref() {
            if let Some(conflict) = existing.config.first_conflict(requested) {
                return Err(ResourceBrokerError::ConfigConflict(conflict));
            }
            return Ok(Arc::clone(existing));
        }
        let resources = Arc::new(
            EngineResources::new(requested, thread_inventory).map_err(ResourceBrokerError::Runtime)?,
        );
        *installed = Some(Arc::clone(&resources));
        Ok(resources)
    }

    fn installed(&self) -> Option<Arc<EngineResources>> {
        lock_unpoisoned(&self.installed).as_ref().map(Arc::clone)
    }

    #[cfg(test)]
    fn isolated_for_test() -> Self {
        Self::default()
    }
}

static PROCESS_BROKER: OnceLock<ResourceBroker> = OnceLock::new();

fn process_broker() -> &'static ResourceBroker {
    PROCESS_BROKER.get_or_init(ResourceBroker::default)
}

/// Install or compatibly reuse the one process resource host.
///
/// Advanced callers may call this before the first engine. There is no public
/// constructor for a second process host.
pub fn install_process_resources(
    config: ResourceHostConfig,
) -> Result<Arc<EngineResources>, ResourceBrokerError> {
    process_broker().install(config)
}

/// Return the installed process host without creating one.
///
/// Health and receipt surfaces use this observation to report an honest
/// `not_installed` state. They must never create an unbounded host merely to
/// make a diagnostic document look populated.
pub fn installed_process_resources() -> Option<Arc<EngineResources>> {
    process_broker().installed()
}

/// The synchronous library facade. Inference methods arrive on later scheduler
/// beads; this type already guarantees that each engine holds a broker lease.
#[derive(Debug)]
pub struct NlpEngine {
    lease: EngineLease,
}

impl NlpEngine {
    /// Begin a library configuration snapshot. This method deliberately does
    /// not read the environment; the CLI snapshots `FNLP_*` once at its own
    /// boundary before calling the builder.
    pub fn builder() -> NlpEngineBuilder {
        NlpEngineBuilder {
            resource_config: None,
        }
    }

    pub fn resources(&self) -> &Arc<EngineResources> {
        self.lease.resources()
    }

    pub const fn lease_id(&self) -> u64 {
        self.lease.id()
    }

    /// Enter a synchronous public operation without nesting a runtime.
    ///
    /// Future inference and task methods must retain the returned guard until
    /// their owned blocking closure has drained. Calling from this runtime's
    /// worker thread or recursively from a synchronous engine call returns a
    /// typed failure instead of attempting `block_on`.
    pub fn enter_sync_call(&self) -> Result<EngineCallGuard, ReentrantCall> {
        if is_runtime_worker_thread() {
            return Err(ReentrantCall::RuntimeWorker);
        }
        let resource_key = Arc::as_ptr(self.resources()) as usize;
        ACTIVE_SYNC_RESOURCE.with(|active| {
            if active.get().is_some() {
                return Err(ReentrantCall::NestedSyncCall);
            }
            active.set(Some(resource_key));
            Ok(EngineCallGuard {
                resource_key,
                // A guard restores a thread-local entry on Drop and therefore
                // must never cross to a different thread.
                _not_send_or_sync: PhantomData,
            })
        })
    }
}

/// Typed refusal for a synchronous call that would nest the owned runtime.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReentrantCall {
    /// The caller is already executing on this process host's runtime worker.
    RuntimeWorker,
    /// The current OS thread already holds a synchronous engine entry guard.
    NestedSyncCall,
}

impl fmt::Display for ReentrantCall {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RuntimeWorker => formatter.write_str(
                "ReentrantCall: synchronous NlpEngine entry from a runtime worker is refused",
            ),
            Self::NestedSyncCall => formatter.write_str(
                "ReentrantCall: nested synchronous NlpEngine entry is refused",
            ),
        }
    }
}

impl std::error::Error for ReentrantCall {}

/// Ownership marker for an admitted synchronous engine operation.
#[derive(Debug)]
pub struct EngineCallGuard {
    resource_key: usize,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl Drop for EngineCallGuard {
    fn drop(&mut self) {
        ACTIVE_SYNC_RESOURCE.with(|active| {
            if active.get() == Some(self.resource_key) {
                active.set(None);
            } else {
                eprintln!(
                    "ENGINE_RESOURCES SYNC_GUARD_INVARIANT expected_resource_key={}",
                    self.resource_key,
                );
            }
        });
    }
}

/// Builder for a synchronous [`NlpEngine`] backed by the process host.
#[derive(Clone, Copy, Debug, Default)]
pub struct NlpEngineBuilder {
    resource_config: Option<ResourceHostConfig>,
}

impl NlpEngineBuilder {
    /// Request installation or compatible reuse of this explicit host config.
    pub const fn resource_config(mut self, resource_config: ResourceHostConfig) -> Self {
        self.resource_config = Some(resource_config);
        self
    }

    /// Acquire the process host and its engine lease.
    pub fn build(self) -> Result<NlpEngine, EngineBuildError> {
        let resources = match self.resource_config {
            Some(config) => install_process_resources(config).map_err(EngineBuildError::Broker)?,
            None => process_broker()
                .installed()
                .ok_or(EngineBuildError::HostNotInstalled)?,
        };
        Ok(NlpEngine {
            lease: resources.acquire_lease(),
        })
    }
}

/// The library builder did not create an unbounded private host.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EngineBuildError {
    HostNotInstalled,
    Broker(ResourceBrokerError),
}

impl fmt::Display for EngineBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HostNotInstalled => formatter.write_str(
                "EngineResources host is not installed; provide an explicit ResourceHostConfig",
            ),
            Self::Broker(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for EngineBuildError {}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use std::{
        panic::{AssertUnwindSafe, catch_unwind},
        sync::{Arc, Barrier},
        thread,
    };

    use super::*;

    fn config() -> ResourceHostConfig {
        ResourceHostConfig::new(
            RuntimePreset::LowLatency,
            4,
            2,
            3,
            1,
            32,
            1_000,
            LeakResponsePolicy::RecordAndEscalate,
        )
        .expect("test config is valid")
    }

    #[test]
    fn inventory_counts_runtime_blocking_scoped_and_helpers() {
        let inventory = config().thread_inventory().expect("inventory fits");
        assert_eq!(inventory.runtime_workers, 4);
        assert_eq!(inventory.blocking_coordinators, 2);
        assert_eq!(inventory.scoped_cpu_children_per_coordinator, 3);
        assert_eq!(inventory.helper_threads, 1);
        assert_eq!(inventory.total_runnable_threads, 13);
    }

    #[test]
    fn first_install_is_shared_and_conflicts_name_each_field() {
        let broker = ResourceBroker::isolated_for_test();
        let installed = broker.install(config()).expect("first install succeeds");
        let compatible = broker.install(config()).expect("compatible reuse succeeds");
        assert!(Arc::ptr_eq(&installed, &compatible));

        for field in ResourceConfigField::ALL {
            let mut requested = config();
            match field {
                ResourceConfigField::RuntimePreset => {
                    requested.runtime_preset = RuntimePreset::HighThroughput
                }
                ResourceConfigField::RuntimeWorkers => requested.runtime_workers = 5,
                ResourceConfigField::BlockingCoordinators => {
                    requested.max_blocking_coordinators = 3
                }
                ResourceConfigField::ScopedCpuChildrenPerCoordinator => {
                    requested.scoped_cpu_children_per_coordinator = 4
                }
                ResourceConfigField::HelperThreads => requested.helper_threads = 2,
                ResourceConfigField::ThreadCeiling => requested.thread_ceiling = 31,
                ResourceConfigField::MemoryCeilingBytes => requested.memory_ceiling_bytes = 1_001,
                ResourceConfigField::LeakResponsePolicy => {
                    requested.leak_response_policy = LeakResponsePolicy::PanicInLabOrCi
                }
            }
            let error = broker
                .install(requested)
                .expect_err("a fixed field mismatch must refuse a second host");
            assert_eq!(
                error,
                ResourceBrokerError::ConfigConflict(ResourceConfigConflict {
                    field,
                    installed: config().value_for(field),
                    requested: requested.value_for(field),
                })
            );
        }
    }

    #[test]
    fn racing_first_install_has_one_arc() {
        let broker = Arc::new(ResourceBroker::isolated_for_test());
        let barrier = Arc::new(Barrier::new(9));
        let mut joins = Vec::new();
        for _ in 0..8 {
            let broker = Arc::clone(&broker);
            let barrier = Arc::clone(&barrier);
            joins.push(thread::spawn(move || {
                barrier.wait();
                broker.install(config()).expect("compatible install")
            }));
        }
        barrier.wait();
        let first = joins
            .pop()
            .expect("eight joins exist")
            .join()
            .expect("worker must not panic");
        for join in joins {
            let contender = join.join().expect("worker must not panic");
            assert!(Arc::ptr_eq(&first, &contender));
        }
    }

    #[test]
    fn reservations_commit_abort_and_release_exactly() {
        let config = config();
        let thread_inventory = config.thread_inventory().expect("inventory is valid");
        let resources = Arc::new(
            EngineResources::new(config, thread_inventory).expect("valid resources"),
        );
        let lease = resources.acquire_lease();
        let committed = lease
            .reserve(MemoryClass::Weights, 600)
            .expect("first reservation fits")
            .commit()
            .expect("commit succeeds");
        assert!(matches!(
            lease.reserve(MemoryClass::KvPages, 401),
            Err(ReservationError::CapacityExceeded { .. })
        ));
        lease
            .reserve(MemoryClass::KvPages, 400)
            .expect("remaining capacity fits")
            .abort()
            .expect("abort restores reservation balance");
        let snapshot = resources.memory_snapshot();
        assert_eq!(snapshot.reserved_bytes, 0);
        assert_eq!(snapshot.committed_bytes, 600);
        assert_eq!(snapshot.by_class[&MemoryClass::Weights].committed_bytes, 600);
        committed.release();
        let snapshot = resources.memory_snapshot();
        assert_eq!(snapshot.committed_bytes, 0);
        assert!(snapshot.by_class.is_empty());
    }

    #[test]
    fn leaked_lab_reservation_panics_after_restoring_capacity() {
        let ledger = Arc::new(MemoryLedger::new(10, LeakResponsePolicy::PanicInLabOrCi));
        let reservation = ledger
            .reserve(7, MemoryClass::Staging, 10)
            .expect("reservation fits");
        let panic = catch_unwind(AssertUnwindSafe(|| drop(reservation)));
        assert!(panic.is_err(), "lab policy must panic on an obligation leak");
        let snapshot = ledger.snapshot();
        assert_eq!(snapshot.reserved_bytes, 0);
        assert_eq!(snapshot.committed_bytes, 0);
        assert_eq!(snapshot.recorded_leaks, 1);
    }

    #[test]
    fn synchronous_entry_refuses_recursion_and_recovers_after_drop() {
        let broker = ResourceBroker::isolated_for_test();
        let resources = broker.install(config()).expect("host installs");
        let engine = NlpEngine {
            lease: resources.acquire_lease(),
        };
        let guard = engine.enter_sync_call().expect("first entry is admitted");
        assert!(
            matches!(engine.enter_sync_call(), Err(ReentrantCall::NestedSyncCall)),
            "nested entry must fail before it can attempt a second runtime"
        );
        drop(guard);
        let recovered = engine
            .enter_sync_call()
            .expect("dropping the outer guard restores entry");
        drop(recovered);
    }
}
