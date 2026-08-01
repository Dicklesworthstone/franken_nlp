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
        mpsc,
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

/// Bounded runtime guardrails applied to the one process host.
///
/// These are concrete asupersync defaults observed at the selected pin, kept
/// explicit so scheduler receipts can report the actual cancellation and
/// checkpoint-monitor envelope rather than a preset name alone.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RuntimeGuardrails {
    pub deadline_check_interval_millis: u64,
    pub checkpoint_timeout_millis: u64,
    pub cancel_attribution_max_depth: usize,
    pub cancel_attribution_max_memory_bytes: usize,
}

impl RuntimeGuardrails {
    /// The pinned defaults: 1s monitor checks, 30s checkpoint stall warning,
    /// and the finite 16-deep / 4096-byte cancellation attribution envelope.
    pub const PINNED_DEFAULTS: Self = Self {
        deadline_check_interval_millis: 1_000,
        checkpoint_timeout_millis: 30_000,
        cancel_attribution_max_depth: 16,
        cancel_attribution_max_memory_bytes: 4_096,
    };
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
    RuntimeFeatureDisabled,
    RuntimeBuild { detail: String },
    MissingBlockingPool,
}

impl fmt::Display for RuntimeHostError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RuntimeFeatureDisabled => formatter.write_str(
                "asupersync-runtime feature is disabled; a production EngineResources host cannot use an inline fallback",
            ),
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
    /// Explicit capacity retained for modeled safety and emergency terms.
    AdmissionReserve,
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
            Self::AdmissionReserve => "admission_reserve",
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

    /// Read available aggregate capacity without constructing a diagnostic
    /// snapshot or allocating a per-class map. Robot planning uses this
    /// observation only; reservation remains authoritative under races.
    fn available_bytes(&self) -> Result<u64, ReservationError> {
        let state = lock_unpoisoned(&self.state);
        let occupied = state
            .reserved_bytes
            .checked_add(state.committed_bytes)
            .ok_or(ReservationError::ArithmeticOverflow)?;
        self.ceiling_bytes
            .checked_sub(occupied)
            .ok_or(ReservationError::LedgerInvariant { obligation_id: 0 })
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

/// The product-default usable context limit. Nanbeige's observed 262,144
/// position limit is not an admission promise.
pub const DEFAULT_CONTEXT_TOKEN_CAP: u64 = 8_192;
/// Logical bf16 K/V bytes across the model's 44 K/V slots for one token in
/// one sequence: `44 * 2 * 8 * 128 * 2`.
pub const BF16_KV_BYTES_PER_TOKEN: u64 = 180_224;
/// Payload bytes for an int8 K/V cache token before scales and allocator
/// metadata.
pub const INT8_KV_PAYLOAD_BYTES_PER_TOKEN: u64 = 90_112;
/// Per-token f32 K/V-scale storage for an int8 cache.
pub const INT8_KV_F32_SCALE_BYTES_PER_TOKEN: u64 = 2_816;
/// Per-token f16 K/V-scale storage for an int8 cache.
pub const INT8_KV_F16_SCALE_BYTES_PER_TOKEN: u64 = 1_408;
/// One dense f32 lm-head row over the fixed 166,144-token vocabulary.
pub const FULL_F32_LOGIT_ROW_BYTES: u64 = 664_576;

/// K/V-cache representation selected by an immutable artifact/profile.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvCacheQuantization {
    Bf16,
    Int8F32Scales,
    Int8F16Scales,
}

impl KvCacheQuantization {
    /// Stable CLI, robot, and receipt spelling.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Int8F32Scales => "int8-f32-scales",
            Self::Int8F16Scales => "int8-f16-scales",
        }
    }

    /// Parse only the closed cache-accounting profiles.
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "bf16" => Some(Self::Bf16),
            "int8" | "int8-f32-scales" => Some(Self::Int8F32Scales),
            "int8-f16-scales" => Some(Self::Int8F16Scales),
            _ => None,
        }
    }

    const fn payload_bytes_per_token(self) -> u64 {
        match self {
            Self::Bf16 => BF16_KV_BYTES_PER_TOKEN,
            Self::Int8F32Scales | Self::Int8F16Scales => INT8_KV_PAYLOAD_BYTES_PER_TOKEN,
        }
    }

    const fn scale_bytes_per_token(self) -> u64 {
        match self {
            Self::Bf16 => 0,
            Self::Int8F32Scales => INT8_KV_F32_SCALE_BYTES_PER_TOKEN,
            Self::Int8F16Scales => INT8_KV_F16_SCALE_BYTES_PER_TOKEN,
        }
    }
}

/// Exact mapped and resident accounting for one file-backed or owned memory
/// region. Mapped virtual bytes and committed resident bytes are intentionally
/// separate; a plan must not substitute one for the other.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidencyAccounting {
    pub mapped_bytes: u64,
    pub resident_bytes: u64,
}

impl ResidencyAccounting {
    /// Construct a region whose resident commitment cannot exceed its mapping.
    pub const fn new(mapped_bytes: u64, resident_bytes: u64) -> Option<Self> {
        if resident_bytes <= mapped_bytes {
            Some(Self {
                mapped_bytes,
                resident_bytes,
            })
        } else {
            None
        }
    }
}

/// Inputs to the checked, allocation-free admission certificate calculation.
///
/// The caller must attach exact fixed residency and page metadata before this
/// plan can admit a request. The defaults intentionally leave those two facts
/// unconfigured so an estimator cannot turn a model-free command into a false
/// memory promise.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdmissionRequest {
    pub context_tokens: u64,
    pub batch_rows: u64,
    pub kv_quantization: KvCacheQuantization,
    pub local_memory_budget_bytes: Option<u64>,
    pub os_reserve_bytes: u64,
    pub fixed_residency: Option<ResidencyAccounting>,
    pub elastic_cache_bytes: u64,
    pub replicated_weight_residency: ResidencyAccounting,
    pub kv_page_metadata_bytes_per_token: Option<u64>,
    pub activation_bytes_per_row: u64,
    pub grammar_state_bytes_per_row: u64,
    pub source_state_bytes_per_row: u64,
    pub queue_bytes_per_row: u64,
    pub output_buffer_bytes_per_row: u64,
    pub unmodeled_emergency_reserve_bytes: u64,
    pub safety_margin_bytes: u64,
}

impl AdmissionRequest {
    /// Start a decode admission plan with the stable context cap and an
    /// explicit 64 MiB safety/emergency envelope. Fixed residency, page-table,
    /// and allocator-padding facts remain deliberately unconfigured.
    pub const fn decode(
        context_tokens: u64,
        batch_rows: u64,
        kv_quantization: KvCacheQuantization,
    ) -> Self {
        Self {
            context_tokens,
            batch_rows,
            kv_quantization,
            local_memory_budget_bytes: None,
            os_reserve_bytes: 0,
            fixed_residency: None,
            elastic_cache_bytes: 0,
            replicated_weight_residency: ResidencyAccounting {
                mapped_bytes: 0,
                resident_bytes: 0,
            },
            kv_page_metadata_bytes_per_token: None,
            activation_bytes_per_row: 0,
            grammar_state_bytes_per_row: 0,
            source_state_bytes_per_row: 0,
            queue_bytes_per_row: 0,
            output_buffer_bytes_per_row: 0,
            unmodeled_emergency_reserve_bytes: 64 * 1024 * 1024,
            safety_margin_bytes: 64 * 1024 * 1024,
        }
    }

    pub const fn with_local_memory_budget(mut self, bytes: u64) -> Self {
        self.local_memory_budget_bytes = Some(bytes);
        self
    }

    pub const fn with_os_reserve(mut self, bytes: u64) -> Self {
        self.os_reserve_bytes = bytes;
        self
    }

    pub const fn with_fixed_residency(mut self, residency: ResidencyAccounting) -> Self {
        self.fixed_residency = Some(residency);
        self
    }

    pub const fn with_elastic_cache(mut self, bytes: u64) -> Self {
        self.elastic_cache_bytes = bytes;
        self
    }

    pub const fn with_replicated_weight_residency(
        mut self,
        residency: ResidencyAccounting,
    ) -> Self {
        self.replicated_weight_residency = residency;
        self
    }

    pub const fn with_kv_page_metadata_per_token(mut self, bytes: u64) -> Self {
        self.kv_page_metadata_bytes_per_token = Some(bytes);
        self
    }

    #[allow(clippy::too_many_arguments)]
    pub const fn with_elastic_rows(
        mut self,
        activation_bytes_per_row: u64,
        grammar_state_bytes_per_row: u64,
        source_state_bytes_per_row: u64,
        queue_bytes_per_row: u64,
        output_buffer_bytes_per_row: u64,
    ) -> Self {
        self.activation_bytes_per_row = activation_bytes_per_row;
        self.grammar_state_bytes_per_row = grammar_state_bytes_per_row;
        self.source_state_bytes_per_row = source_state_bytes_per_row;
        self.queue_bytes_per_row = queue_bytes_per_row;
        self.output_buffer_bytes_per_row = output_buffer_bytes_per_row;
        self
    }

    pub const fn with_reserves(
        mut self,
        unmodeled_emergency_reserve_bytes: u64,
        safety_margin_bytes: u64,
    ) -> Self {
        self.unmodeled_emergency_reserve_bytes = unmodeled_emergency_reserve_bytes;
        self.safety_margin_bytes = safety_margin_bytes;
        self
    }
}

/// A named certificate term. The order is the deterministic first-violation
/// order used for a refusal explanation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdmissionTerm {
    OsReserve,
    FixedResidency,
    ElasticCache,
    ReplicatedWeightResidency,
    KvPayload,
    KvScales,
    KvPageMetadata,
    ActivationRows,
    FullLogitRows,
    GrammarState,
    SourceState,
    QueueBuffers,
    OutputBuffers,
    UnmodeledEmergencyReserve,
    SafetyMargin,
}

impl AdmissionTerm {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OsReserve => "os_reserve",
            Self::FixedResidency => "fixed_residency",
            Self::ElasticCache => "elastic_cache",
            Self::ReplicatedWeightResidency => "replicated_weight_residency",
            Self::KvPayload => "kv_payload",
            Self::KvScales => "kv_scales",
            Self::KvPageMetadata => "kv_page_metadata",
            Self::ActivationRows => "activation_rows",
            Self::FullLogitRows => "full_logit_rows",
            Self::GrammarState => "grammar_state",
            Self::SourceState => "source_state",
            Self::QueueBuffers => "queue_buffers",
            Self::OutputBuffers => "output_buffers",
            Self::UnmodeledEmergencyReserve => "unmodeled_emergency_reserve",
            Self::SafetyMargin => "safety_margin",
        }
    }
}

/// Computed bytes for every named admission term. `None` means a required
/// physical fact was not supplied, never that the term was silently ignored.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdmissionTerms {
    pub os_reserve_bytes: u64,
    pub fixed_mapped_bytes: Option<u64>,
    pub fixed_resident_bytes: Option<u64>,
    pub elastic_cache_bytes: u64,
    pub replicated_weight_mapped_bytes: u64,
    pub replicated_weight_resident_bytes: u64,
    pub kv_payload_bytes: u64,
    pub kv_scale_bytes: u64,
    pub kv_page_metadata_bytes: Option<u64>,
    pub activation_bytes: u64,
    pub full_logit_bytes: u64,
    pub grammar_state_bytes: u64,
    pub source_state_bytes: u64,
    pub queue_bytes: u64,
    pub output_buffer_bytes: u64,
    pub unmodeled_emergency_reserve_bytes: u64,
    pub safety_margin_bytes: u64,
    /// Allocatable process-ledger commitment; excludes the OS reserve.
    pub committed_bytes: Option<u64>,
    /// Full process peak, including the non-allocatable OS reserve.
    pub peak_bytes: Option<u64>,
}

/// Checked-arithmetic failures never wrap an admission term into a smaller
/// value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdmissionBuildError {
    ZeroBatchRows,
    ArithmeticOverflow { term: AdmissionTerm },
}

impl fmt::Display for AdmissionBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroBatchRows => formatter.write_str("admission batch_rows must be non-zero"),
            Self::ArithmeticOverflow { term } => write!(
                formatter,
                "admission arithmetic overflow while calculating {}",
                term.as_str()
            ),
        }
    }
}

impl std::error::Error for AdmissionBuildError {}

/// A concrete refusal emitted before any request allocation happens.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdmissionRejection {
    ContextCapExceeded {
        requested_tokens: u64,
        cap_tokens: u64,
    },
    FixedResidencyUnconfigured,
    KvPageMetadataUnconfigured,
    MemoryBudgetUnconfigured,
    LocalBudgetExceeded {
        first_violated_term: AdmissionTerm,
        required_peak_bytes: u64,
        budget_bytes: u64,
    },
    AggregateCapacityExceeded {
        first_violated_term: AdmissionTerm,
        requested_ledger_bytes: u64,
        available_ledger_bytes: u64,
        ledger_ceiling_bytes: u64,
    },
}

impl AdmissionRejection {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ContextCapExceeded { .. } => "context_cap_exceeded",
            Self::FixedResidencyUnconfigured => "fixed_residency_unconfigured",
            Self::KvPageMetadataUnconfigured => "kv_page_metadata_unconfigured",
            Self::MemoryBudgetUnconfigured => "memory_budget_unconfigured",
            Self::LocalBudgetExceeded { .. } => "local_budget_exceeded",
            Self::AggregateCapacityExceeded { .. } => "aggregate_capacity_exceeded",
        }
    }

    pub const fn first_violated_term(self) -> Option<AdmissionTerm> {
        match self {
            Self::LocalBudgetExceeded {
                first_violated_term,
                ..
            }
            | Self::AggregateCapacityExceeded {
                first_violated_term,
                ..
            } => Some(first_violated_term),
            Self::ContextCapExceeded { .. }
            | Self::FixedResidencyUnconfigured
            | Self::KvPageMetadataUnconfigured
            | Self::MemoryBudgetUnconfigured => None,
        }
    }
}

impl fmt::Display for AdmissionRejection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContextCapExceeded {
                requested_tokens,
                cap_tokens,
            } => write!(
                formatter,
                "context request {requested_tokens} exceeds default admitted cap {cap_tokens}"
            ),
            Self::FixedResidencyUnconfigured => formatter
                .write_str("fixed mapped/resident model and packing accounting is not configured"),
            Self::KvPageMetadataUnconfigured => formatter.write_str(
                "KV allocator padding and page-table bytes per token are not configured",
            ),
            Self::MemoryBudgetUnconfigured => {
                formatter.write_str("local memory budget is not configured")
            }
            Self::LocalBudgetExceeded {
                first_violated_term,
                required_peak_bytes,
                budget_bytes,
            } => write!(
                formatter,
                "local memory budget first exceeds at {} required_peak={} budget={}",
                first_violated_term.as_str(),
                required_peak_bytes,
                budget_bytes,
            ),
            Self::AggregateCapacityExceeded {
                first_violated_term,
                requested_ledger_bytes,
                available_ledger_bytes,
                ledger_ceiling_bytes,
            } => write!(
                formatter,
                "aggregate memory ledger first exceeds at {} requested={} available={} ceiling={}",
                first_violated_term.as_str(),
                requested_ledger_bytes,
                available_ledger_bytes,
                ledger_ceiling_bytes,
            ),
        }
    }
}

impl std::error::Error for AdmissionRejection {}

/// Result of a local or process-aggregate preflight.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdmissionDecision {
    Admitted,
    Refused(AdmissionRejection),
}

impl AdmissionDecision {
    pub const fn status(self) -> &'static str {
        match self {
            Self::Admitted => "admitted",
            Self::Refused(_) => "refused",
        }
    }

    pub const fn rejection(self) -> Option<AdmissionRejection> {
        match self {
            Self::Admitted => None,
            Self::Refused(rejection) => Some(rejection),
        }
    }
}

/// An allocation-free, replayable statement of one request or bounded batch's
/// complete memory and thread envelope.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdmissionCertificate {
    request: AdmissionRequest,
    thread_inventory: ThreadInventory,
    terms: AdmissionTerms,
}

impl AdmissionCertificate {
    /// Calculate all known terms with checked arithmetic. This performs no
    /// memory reservation and no model, allocator, or filesystem operation.
    pub fn build(
        request: AdmissionRequest,
        thread_inventory: ThreadInventory,
    ) -> Result<Self, AdmissionBuildError> {
        if request.batch_rows == 0 {
            return Err(AdmissionBuildError::ZeroBatchRows);
        }
        let token_rows = request
            .context_tokens
            .checked_mul(request.batch_rows)
            .ok_or(AdmissionBuildError::ArithmeticOverflow {
                term: AdmissionTerm::KvPayload,
            })?;
        let kv_payload_bytes = token_rows
            .checked_mul(request.kv_quantization.payload_bytes_per_token())
            .ok_or(AdmissionBuildError::ArithmeticOverflow {
                term: AdmissionTerm::KvPayload,
            })?;
        let kv_scale_bytes = token_rows
            .checked_mul(request.kv_quantization.scale_bytes_per_token())
            .ok_or(AdmissionBuildError::ArithmeticOverflow {
                term: AdmissionTerm::KvScales,
            })?;
        let kv_page_metadata_bytes = request
            .kv_page_metadata_bytes_per_token
            .map(|bytes| {
                token_rows
                    .checked_mul(bytes)
                    .ok_or(AdmissionBuildError::ArithmeticOverflow {
                        term: AdmissionTerm::KvPageMetadata,
                    })
            })
            .transpose()?;
        let activation_bytes = checked_row_bytes(
            request.batch_rows,
            request.activation_bytes_per_row,
            AdmissionTerm::ActivationRows,
        )?;
        let full_logit_bytes = checked_row_bytes(
            request.batch_rows,
            FULL_F32_LOGIT_ROW_BYTES,
            AdmissionTerm::FullLogitRows,
        )?;
        let grammar_state_bytes = checked_row_bytes(
            request.batch_rows,
            request.grammar_state_bytes_per_row,
            AdmissionTerm::GrammarState,
        )?;
        let source_state_bytes = checked_row_bytes(
            request.batch_rows,
            request.source_state_bytes_per_row,
            AdmissionTerm::SourceState,
        )?;
        let queue_bytes = checked_row_bytes(
            request.batch_rows,
            request.queue_bytes_per_row,
            AdmissionTerm::QueueBuffers,
        )?;
        let output_buffer_bytes = checked_row_bytes(
            request.batch_rows,
            request.output_buffer_bytes_per_row,
            AdmissionTerm::OutputBuffers,
        )?;
        let terms = AdmissionTerms {
            os_reserve_bytes: request.os_reserve_bytes,
            fixed_mapped_bytes: request.fixed_residency.map(|value| value.mapped_bytes),
            fixed_resident_bytes: request.fixed_residency.map(|value| value.resident_bytes),
            elastic_cache_bytes: request.elastic_cache_bytes,
            replicated_weight_mapped_bytes: request.replicated_weight_residency.mapped_bytes,
            replicated_weight_resident_bytes: request.replicated_weight_residency.resident_bytes,
            kv_payload_bytes,
            kv_scale_bytes,
            kv_page_metadata_bytes,
            activation_bytes,
            full_logit_bytes,
            grammar_state_bytes,
            source_state_bytes,
            queue_bytes,
            output_buffer_bytes,
            unmodeled_emergency_reserve_bytes: request.unmodeled_emergency_reserve_bytes,
            safety_margin_bytes: request.safety_margin_bytes,
            committed_bytes: None,
            peak_bytes: None,
        };
        let (committed_bytes, peak_bytes) = complete_totals(terms)?;
        Ok(Self {
            request,
            thread_inventory,
            terms: AdmissionTerms {
                committed_bytes,
                peak_bytes,
                ..terms
            },
        })
    }

    pub const fn request(&self) -> AdmissionRequest {
        self.request
    }

    pub const fn thread_inventory(&self) -> ThreadInventory {
        self.thread_inventory
    }

    pub const fn terms(&self) -> AdmissionTerms {
        self.terms
    }

    /// Apply the request-local cap and explicit local budget.
    pub fn local_decision(&self) -> AdmissionDecision {
        if self.request.context_tokens > DEFAULT_CONTEXT_TOKEN_CAP {
            return AdmissionDecision::Refused(AdmissionRejection::ContextCapExceeded {
                requested_tokens: self.request.context_tokens,
                cap_tokens: DEFAULT_CONTEXT_TOKEN_CAP,
            });
        }
        if self.terms.fixed_resident_bytes.is_none() {
            return AdmissionDecision::Refused(AdmissionRejection::FixedResidencyUnconfigured);
        }
        if self.terms.kv_page_metadata_bytes.is_none() {
            return AdmissionDecision::Refused(AdmissionRejection::KvPageMetadataUnconfigured);
        }
        let Some(budget_bytes) = self.request.local_memory_budget_bytes else {
            return AdmissionDecision::Refused(AdmissionRejection::MemoryBudgetUnconfigured);
        };
        let required_peak_bytes = self
            .terms
            .peak_bytes
            .expect("complete terms have a peak after residency and metadata checks");
        if required_peak_bytes > budget_bytes {
            return AdmissionDecision::Refused(AdmissionRejection::LocalBudgetExceeded {
                first_violated_term: self.first_peak_term_over(budget_bytes),
                required_peak_bytes,
                budget_bytes,
            });
        }
        AdmissionDecision::Admitted
    }

    /// Apply an observed aggregate-ledger availability after the local
    /// certificate has passed. This remains a preflight; the real reservation
    /// below is still authoritative under racing callers.
    pub fn aggregate_decision(
        &self,
        available_ledger_bytes: u64,
        ledger_ceiling_bytes: u64,
    ) -> AdmissionDecision {
        if let decision @ AdmissionDecision::Refused(_) = self.local_decision() {
            return decision;
        }
        let requested_ledger_bytes = self
            .terms
            .committed_bytes
            .expect("locally admitted certificate has complete ledger bytes");
        if requested_ledger_bytes > available_ledger_bytes {
            let occupied_ledger_bytes = ledger_ceiling_bytes.saturating_sub(available_ledger_bytes);
            return AdmissionDecision::Refused(AdmissionRejection::AggregateCapacityExceeded {
                first_violated_term: self
                    .first_ledger_term_over(occupied_ledger_bytes, ledger_ceiling_bytes),
                requested_ledger_bytes,
                available_ledger_bytes,
                ledger_ceiling_bytes,
            });
        }
        AdmissionDecision::Admitted
    }

    fn first_peak_term_over(&self, budget_bytes: u64) -> AdmissionTerm {
        first_term_over_from(0, budget_bytes, self.peak_terms())
            .expect("known peak must exceed the supplied local budget")
    }

    fn first_ledger_term_over(
        &self,
        occupied_ledger_bytes: u64,
        ledger_ceiling_bytes: u64,
    ) -> AdmissionTerm {
        first_term_over_from(
            occupied_ledger_bytes,
            ledger_ceiling_bytes,
            self.ledger_terms(),
        )
        .expect("aggregate capacity failure has a violating ledger term")
    }

    fn peak_terms(&self) -> [(AdmissionTerm, u64); 15] {
        let terms = self.terms;
        [
            (AdmissionTerm::OsReserve, terms.os_reserve_bytes),
            (
                AdmissionTerm::FixedResidency,
                terms.fixed_resident_bytes.unwrap_or(0),
            ),
            (AdmissionTerm::ElasticCache, terms.elastic_cache_bytes),
            (
                AdmissionTerm::ReplicatedWeightResidency,
                terms.replicated_weight_resident_bytes,
            ),
            (AdmissionTerm::KvPayload, terms.kv_payload_bytes),
            (AdmissionTerm::KvScales, terms.kv_scale_bytes),
            (
                AdmissionTerm::KvPageMetadata,
                terms.kv_page_metadata_bytes.unwrap_or(0),
            ),
            (AdmissionTerm::ActivationRows, terms.activation_bytes),
            (AdmissionTerm::FullLogitRows, terms.full_logit_bytes),
            (AdmissionTerm::GrammarState, terms.grammar_state_bytes),
            (AdmissionTerm::SourceState, terms.source_state_bytes),
            (AdmissionTerm::QueueBuffers, terms.queue_bytes),
            (AdmissionTerm::OutputBuffers, terms.output_buffer_bytes),
            (
                AdmissionTerm::UnmodeledEmergencyReserve,
                terms.unmodeled_emergency_reserve_bytes,
            ),
            (AdmissionTerm::SafetyMargin, terms.safety_margin_bytes),
        ]
    }

    fn ledger_terms(&self) -> [(AdmissionTerm, u64); 14] {
        let terms = self.terms;
        [
            (
                AdmissionTerm::FixedResidency,
                terms.fixed_resident_bytes.unwrap_or(0),
            ),
            (AdmissionTerm::ElasticCache, terms.elastic_cache_bytes),
            (
                AdmissionTerm::ReplicatedWeightResidency,
                terms.replicated_weight_resident_bytes,
            ),
            (AdmissionTerm::KvPayload, terms.kv_payload_bytes),
            (AdmissionTerm::KvScales, terms.kv_scale_bytes),
            (
                AdmissionTerm::KvPageMetadata,
                terms.kv_page_metadata_bytes.unwrap_or(0),
            ),
            (AdmissionTerm::ActivationRows, terms.activation_bytes),
            (AdmissionTerm::FullLogitRows, terms.full_logit_bytes),
            (AdmissionTerm::GrammarState, terms.grammar_state_bytes),
            (AdmissionTerm::SourceState, terms.source_state_bytes),
            (AdmissionTerm::QueueBuffers, terms.queue_bytes),
            (AdmissionTerm::OutputBuffers, terms.output_buffer_bytes),
            (
                AdmissionTerm::UnmodeledEmergencyReserve,
                terms.unmodeled_emergency_reserve_bytes,
            ),
            (AdmissionTerm::SafetyMargin, terms.safety_margin_bytes),
        ]
    }

    fn charges(&self) -> AdmissionCharges {
        let terms = self.terms;
        AdmissionCharges {
            weights: terms
                .fixed_resident_bytes
                .unwrap_or(0)
                .checked_add(terms.replicated_weight_resident_bytes)
                .expect("certificate totals prevent weight charge overflow"),
            prefix_cache: terms.elastic_cache_bytes,
            kv_pages: terms
                .kv_payload_bytes
                .checked_add(terms.kv_scale_bytes)
                .and_then(|total| total.checked_add(terms.kv_page_metadata_bytes.unwrap_or(0)))
                .expect("certificate totals prevent KV charge overflow"),
            activation_scratch: terms.activation_bytes,
            logit_scratch: terms.full_logit_bytes,
            grammar_cache: terms.grammar_state_bytes,
            job_buffers: terms
                .source_state_bytes
                .checked_add(terms.queue_bytes)
                .and_then(|total| total.checked_add(terms.output_buffer_bytes))
                .expect("certificate totals prevent job-buffer charge overflow"),
            admission_reserve: terms
                .unmodeled_emergency_reserve_bytes
                .checked_add(terms.safety_margin_bytes)
                .expect("certificate totals prevent reserve charge overflow"),
        }
    }
}

fn checked_row_bytes(
    rows: u64,
    bytes_per_row: u64,
    term: AdmissionTerm,
) -> Result<u64, AdmissionBuildError> {
    rows.checked_mul(bytes_per_row)
        .ok_or(AdmissionBuildError::ArithmeticOverflow { term })
}

fn complete_totals(
    terms: AdmissionTerms,
) -> Result<(Option<u64>, Option<u64>), AdmissionBuildError> {
    let (Some(fixed_resident_bytes), Some(kv_page_metadata_bytes)) =
        (terms.fixed_resident_bytes, terms.kv_page_metadata_bytes)
    else {
        return Ok((None, None));
    };
    let ledger_terms = [
        (AdmissionTerm::FixedResidency, fixed_resident_bytes),
        (AdmissionTerm::ElasticCache, terms.elastic_cache_bytes),
        (
            AdmissionTerm::ReplicatedWeightResidency,
            terms.replicated_weight_resident_bytes,
        ),
        (AdmissionTerm::KvPayload, terms.kv_payload_bytes),
        (AdmissionTerm::KvScales, terms.kv_scale_bytes),
        (AdmissionTerm::KvPageMetadata, kv_page_metadata_bytes),
        (AdmissionTerm::ActivationRows, terms.activation_bytes),
        (AdmissionTerm::FullLogitRows, terms.full_logit_bytes),
        (AdmissionTerm::GrammarState, terms.grammar_state_bytes),
        (AdmissionTerm::SourceState, terms.source_state_bytes),
        (AdmissionTerm::QueueBuffers, terms.queue_bytes),
        (AdmissionTerm::OutputBuffers, terms.output_buffer_bytes),
        (
            AdmissionTerm::UnmodeledEmergencyReserve,
            terms.unmodeled_emergency_reserve_bytes,
        ),
        (AdmissionTerm::SafetyMargin, terms.safety_margin_bytes),
    ];
    let mut committed_bytes = 0_u64;
    for (term, bytes) in ledger_terms {
        committed_bytes = committed_bytes
            .checked_add(bytes)
            .ok_or(AdmissionBuildError::ArithmeticOverflow { term })?;
    }
    let peak_bytes = committed_bytes.checked_add(terms.os_reserve_bytes).ok_or(
        AdmissionBuildError::ArithmeticOverflow {
            term: AdmissionTerm::OsReserve,
        },
    )?;
    Ok((Some(committed_bytes), Some(peak_bytes)))
}

fn first_term_over_from<const N: usize>(
    start_bytes: u64,
    ceiling_bytes: u64,
    terms: [(AdmissionTerm, u64); N],
) -> Option<AdmissionTerm> {
    let mut total = start_bytes;
    for (term, bytes) in terms {
        total = total.checked_add(bytes)?;
        if total > ceiling_bytes {
            return Some(term);
        }
    }
    None
}

#[derive(Clone, Copy, Debug)]
struct AdmissionCharges {
    weights: u64,
    prefix_cache: u64,
    kv_pages: u64,
    activation_scratch: u64,
    logit_scratch: u64,
    grammar_cache: u64,
    job_buffers: u64,
    admission_reserve: u64,
}

/// Failure while turning a preflight certificate into owned ledger
/// obligations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdmissionError {
    Refused(AdmissionRejection),
    Reservation(ReservationError),
}

impl fmt::Display for AdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Refused(error) => error.fmt(formatter),
            Self::Reservation(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for AdmissionError {}

/// All uncommitted ledger obligations for one admission certificate. Call
/// [`Self::commit`] only after the caller has acquired every represented
/// resource; call [`Self::abort`] on cancellation or allocation failure.
#[derive(Debug)]
pub struct AdmissionReservation {
    weights: Option<MemoryReservation>,
    prefix_cache: Option<MemoryReservation>,
    kv_pages: Option<MemoryReservation>,
    activation_scratch: Option<MemoryReservation>,
    logit_scratch: Option<MemoryReservation>,
    grammar_cache: Option<MemoryReservation>,
    job_buffers: Option<MemoryReservation>,
    admission_reserve: Option<MemoryReservation>,
}

impl AdmissionReservation {
    fn empty() -> Self {
        Self {
            weights: None,
            prefix_cache: None,
            kv_pages: None,
            activation_scratch: None,
            logit_scratch: None,
            grammar_cache: None,
            job_buffers: None,
            admission_reserve: None,
        }
    }

    /// Restore every reservation exactly, including when cancellation arrives
    /// after only a prefix of the certificate has been allocated.
    pub fn abort(mut self) -> Result<(), ReservationError> {
        self.abort_all()
    }

    /// Commit all owned obligations. The returned guard holds every ledger
    /// charge until it is released or dropped after request drain.
    pub fn commit(mut self) -> Result<CommittedAdmission, AdmissionError> {
        let mut committed = CommittedAdmission::empty();
        macro_rules! commit_slot {
            ($slot:ident) => {
                match commit_admission_slot(&mut self.$slot) {
                    Ok(memory) => committed.$slot = memory,
                    Err(error) => {
                        committed.release();
                        let _ = self.abort_all();
                        return Err(AdmissionError::Reservation(error));
                    }
                }
            };
        }
        commit_slot!(weights);
        commit_slot!(prefix_cache);
        commit_slot!(kv_pages);
        commit_slot!(activation_scratch);
        commit_slot!(logit_scratch);
        commit_slot!(grammar_cache);
        commit_slot!(job_buffers);
        commit_slot!(admission_reserve);
        Ok(committed)
    }

    fn abort_all(&mut self) -> Result<(), ReservationError> {
        let mut first_error = None;
        abort_admission_slot(&mut self.weights, &mut first_error);
        abort_admission_slot(&mut self.prefix_cache, &mut first_error);
        abort_admission_slot(&mut self.kv_pages, &mut first_error);
        abort_admission_slot(&mut self.activation_scratch, &mut first_error);
        abort_admission_slot(&mut self.logit_scratch, &mut first_error);
        abort_admission_slot(&mut self.grammar_cache, &mut first_error);
        abort_admission_slot(&mut self.job_buffers, &mut first_error);
        abort_admission_slot(&mut self.admission_reserve, &mut first_error);
        first_error.map_or(Ok(()), Err)
    }
}

fn commit_admission_slot(
    slot: &mut Option<MemoryReservation>,
) -> Result<Option<CommittedMemory>, ReservationError> {
    slot.take().map(MemoryReservation::commit).transpose()
}

fn abort_admission_slot(
    slot: &mut Option<MemoryReservation>,
    first_error: &mut Option<ReservationError>,
) {
    if let Some(reservation) = slot.take()
        && let Err(error) = reservation.abort()
    {
        first_error.get_or_insert(error);
    }
}

/// The committed counterpart to [`AdmissionReservation`]. Its fields are
/// private so a request cannot release one certificate term while retaining
/// another and silently falsify its capacity claim.
#[derive(Debug)]
pub struct CommittedAdmission {
    weights: Option<CommittedMemory>,
    prefix_cache: Option<CommittedMemory>,
    kv_pages: Option<CommittedMemory>,
    activation_scratch: Option<CommittedMemory>,
    logit_scratch: Option<CommittedMemory>,
    grammar_cache: Option<CommittedMemory>,
    job_buffers: Option<CommittedMemory>,
    admission_reserve: Option<CommittedMemory>,
}

impl CommittedAdmission {
    fn empty() -> Self {
        Self {
            weights: None,
            prefix_cache: None,
            kv_pages: None,
            activation_scratch: None,
            logit_scratch: None,
            grammar_cache: None,
            job_buffers: None,
            admission_reserve: None,
        }
    }

    /// Release every committed certificate term before normal request drop.
    pub fn release(mut self) {
        for memory in [
            &mut self.weights,
            &mut self.prefix_cache,
            &mut self.kv_pages,
            &mut self.activation_scratch,
            &mut self.logit_scratch,
            &mut self.grammar_cache,
            &mut self.job_buffers,
            &mut self.admission_reserve,
        ] {
            if let Some(memory) = memory.take() {
                memory.release();
            }
        }
    }
}

#[derive(Debug, Default)]
struct LeaseState {
    active: BTreeMap<u64, ()>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct OutstandingClosure {
    engine_lease_id: u64,
    wrapper_cancelled: bool,
}

#[derive(Debug, Default)]
struct CompletionState {
    active: BTreeMap<u64, OutstandingClosure>,
}

/// Snapshot of pool closures that remain live after their async wrappers have
/// resolved or cancelled.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OutstandingClosureSnapshot {
    pub active_closures: usize,
    pub wrapper_cancelled_closures: usize,
}

/// The lifecycle of one fixed scoped-CPU team.
///
/// The pinned `ScopedCpu::spawn` surface caps only the number of children. It
/// intentionally does not decide whether formation is still legal after the
/// scope has begun draining. This protocol owns that missing state transition:
/// workers are formed first, the coordinator seals formation, then workers may
/// pass the entry gate into fallible work. Once sealed, no API can form another
/// worker, including while cancellation or panic drain is in progress.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SealedTeamPhase {
    Forming,
    Sealed,
    Running,
    Draining,
    Joined,
}

impl SealedTeamPhase {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Forming => "forming",
            Self::Sealed => "sealed",
            Self::Running => "running",
            Self::Draining => "draining",
            Self::Joined => "joined",
        }
    }
}

/// Why the coordinator entered the terminal drain phase.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TeamDrainReason {
    Completed,
    Cancelled,
    WorkerPanicked,
    CoordinatorPanicked,
}

impl TeamDrainReason {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Cancelled => "cancelled",
            Self::WorkerPanicked => "worker_panicked",
            Self::CoordinatorPanicked => "coordinator_panicked",
        }
    }
}

/// A fixed child ordinal allocated during the one legal spawn phase.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct SealedTeamWorker {
    ordinal: usize,
}

impl SealedTeamWorker {
    pub const fn ordinal(self) -> usize {
        self.ordinal
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SealedTeamWorkerState {
    Formed,
    Running,
    Exited,
}

#[derive(Debug)]
struct SealedTeamState {
    phase: SealedTeamPhase,
    expected_children: usize,
    workers: BTreeMap<usize, SealedTeamWorkerState>,
    drain_reason: Option<TeamDrainReason>,
}

/// A stable protocol snapshot suitable for stage-line telemetry and tests.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SealedTeamSnapshot {
    pub phase: SealedTeamPhase,
    pub expected_children: usize,
    pub formed_children: usize,
    pub running_children: usize,
    pub exited_children: usize,
    pub drain_reason: Option<TeamDrainReason>,
}

/// Typed refusals for the sealed team protocol.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SealedTeamError {
    FormationFull {
        expected_children: usize,
    },
    SpawnAfterSeal {
        phase: SealedTeamPhase,
    },
    SealBeforeCompleteFormation {
        expected_children: usize,
        formed_children: usize,
    },
    ReleaseBeforeSeal {
        phase: SealedTeamPhase,
    },
    WorkerStartBeforeRelease {
        phase: SealedTeamPhase,
    },
    UnknownWorker {
        ordinal: usize,
    },
    WorkerAlreadyRunning {
        ordinal: usize,
    },
    WorkerNotRunning {
        ordinal: usize,
    },
    JoinBeforeDrain {
        phase: SealedTeamPhase,
    },
    JoinBeforeWorkersExit {
        expected_children: usize,
        exited_children: usize,
    },
}

impl fmt::Display for SealedTeamError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FormationFull { expected_children } => {
                write!(formatter, "sealed team formation already has {expected_children} children")
            }
            Self::SpawnAfterSeal { phase } => write!(
                formatter,
                "sealed team refuses worker formation after {}",
                phase.as_str()
            ),
            Self::SealBeforeCompleteFormation {
                expected_children,
                formed_children,
            } => write!(
                formatter,
                "sealed team cannot seal {formed_children}/{expected_children} children"
            ),
            Self::ReleaseBeforeSeal { phase } => write!(
                formatter,
                "sealed team cannot release workers from {}",
                phase.as_str()
            ),
            Self::WorkerStartBeforeRelease { phase } => write!(
                formatter,
                "sealed team worker cannot start from {}",
                phase.as_str()
            ),
            Self::UnknownWorker { ordinal } => {
                write!(formatter, "sealed team has no worker ordinal {ordinal}")
            }
            Self::WorkerAlreadyRunning { ordinal } => {
                write!(formatter, "sealed team worker {ordinal} already started")
            }
            Self::WorkerNotRunning { ordinal } => {
                write!(formatter, "sealed team worker {ordinal} is not running")
            }
            Self::JoinBeforeDrain { phase } => write!(
                formatter,
                "sealed team cannot join from {}",
                phase.as_str()
            ),
            Self::JoinBeforeWorkersExit {
                expected_children,
                exited_children,
            } => write!(
                formatter,
                "sealed team cannot join {exited_children}/{expected_children} exited children"
            ),
        }
    }
}

impl std::error::Error for SealedTeamError {}

/// The coordinator-owned protocol for one bounded scoped-CPU team.
///
/// This type is deliberately independent of the foundation's scheduler types
/// so the repository-owned bounded state model can exercise the native-thread
/// protocol under every formation/seal/cancel/panic/join order. The production
/// adapter holds one of these values for the exact lifetime of its one
/// `spawn_blocking` closure and one `scoped_cpu` region.
#[derive(Debug)]
pub struct SealedCpuTeam {
    state: Mutex<SealedTeamState>,
}

impl SealedCpuTeam {
    /// Begin formation for exactly `expected_children` scoped children. The
    /// coordinator itself is intentionally not counted here.
    pub fn new(expected_children: usize) -> Self {
        Self {
            state: Mutex::new(SealedTeamState {
                phase: SealedTeamPhase::Forming,
                expected_children,
                workers: BTreeMap::new(),
                drain_reason: None,
            }),
        }
    }

    /// Allocate one child ordinal during the sole legal formation phase.
    pub fn form_worker(&self) -> Result<SealedTeamWorker, SealedTeamError> {
        let mut state = lock_unpoisoned(&self.state);
        if state.phase != SealedTeamPhase::Forming {
            return Err(SealedTeamError::SpawnAfterSeal { phase: state.phase });
        }
        if state.workers.len() == state.expected_children {
            return Err(SealedTeamError::FormationFull {
                expected_children: state.expected_children,
            });
        }
        let ordinal = state.workers.len();
        state.workers.insert(ordinal, SealedTeamWorkerState::Formed);
        eprintln!(
            "SEALED_TEAM STAGE=FORMATION RESULT=FORMED worker_ordinal={ordinal} formed_children={} expected_children={}",
            state.workers.len(),
            state.expected_children,
        );
        Ok(SealedTeamWorker { ordinal })
    }

    /// Irrevocably end the only worker-formation phase.
    pub fn seal(&self) -> Result<(), SealedTeamError> {
        let mut state = lock_unpoisoned(&self.state);
        if state.phase != SealedTeamPhase::Forming {
            return Err(SealedTeamError::SpawnAfterSeal { phase: state.phase });
        }
        if state.workers.len() != state.expected_children {
            return Err(SealedTeamError::SealBeforeCompleteFormation {
                expected_children: state.expected_children,
                formed_children: state.workers.len(),
            });
        }
        state.phase = SealedTeamPhase::Sealed;
        eprintln!(
            "SEALED_TEAM STAGE=FORMATION RESULT=SEALED expected_children={}",
            state.expected_children,
        );
        Ok(())
    }

    /// Open the entry gate after formation has been sealed.
    pub fn release_workers(&self) -> Result<(), SealedTeamError> {
        let mut state = lock_unpoisoned(&self.state);
        if state.phase != SealedTeamPhase::Sealed {
            return Err(SealedTeamError::ReleaseBeforeSeal { phase: state.phase });
        }
        state.phase = SealedTeamPhase::Running;
        eprintln!(
            "SEALED_TEAM STAGE=ENTRY_GATE RESULT=RELEASED expected_children={}",
            state.expected_children,
        );
        Ok(())
    }

    /// Record the first checkpoint boundary before a child begins fallible
    /// work. The worker can reach this only after the coordinator opened the
    /// entry gate.
    pub fn worker_started(&self, worker: SealedTeamWorker) -> Result<(), SealedTeamError> {
        let mut state = lock_unpoisoned(&self.state);
        if state.phase != SealedTeamPhase::Running {
            return Err(SealedTeamError::WorkerStartBeforeRelease { phase: state.phase });
        }
        let Some(worker_state) = state.workers.get_mut(&worker.ordinal) else {
            return Err(SealedTeamError::UnknownWorker {
                ordinal: worker.ordinal,
            });
        };
        if *worker_state != SealedTeamWorkerState::Formed {
            return Err(SealedTeamError::WorkerAlreadyRunning {
                ordinal: worker.ordinal,
            });
        }
        *worker_state = SealedTeamWorkerState::Running;
        eprintln!(
            "SEALED_TEAM STAGE=WORKER RESULT=STARTED worker_ordinal={}",
            worker.ordinal,
        );
        Ok(())
    }

    /// Start cooperative drain. This closes the protocol to new formation and
    /// lets each existing worker leave at its next checkpoint.
    pub fn begin_drain(&self, reason: TeamDrainReason) {
        let mut state = lock_unpoisoned(&self.state);
        if state.phase == SealedTeamPhase::Joined {
            return;
        }
        state.phase = SealedTeamPhase::Draining;
        if state.drain_reason.is_none() {
            state.drain_reason = Some(reason);
            eprintln!(
                "SEALED_TEAM STAGE=DRAIN RESULT=STARTED reason={}",
                reason.as_str(),
            );
        }
    }

    /// Record child exit after its final checkpoint or a contained panic.
    pub fn worker_exited(&self, worker: SealedTeamWorker) -> Result<(), SealedTeamError> {
        let mut state = lock_unpoisoned(&self.state);
        let Some(worker_state) = state.workers.get_mut(&worker.ordinal) else {
            return Err(SealedTeamError::UnknownWorker {
                ordinal: worker.ordinal,
            });
        };
        if *worker_state != SealedTeamWorkerState::Running {
            return Err(SealedTeamError::WorkerNotRunning {
                ordinal: worker.ordinal,
            });
        }
        *worker_state = SealedTeamWorkerState::Exited;
        eprintln!(
            "SEALED_TEAM STAGE=WORKER RESULT=EXITED worker_ordinal={}",
            worker.ordinal,
        );
        Ok(())
    }

    /// Record a child panic before drain. The child is terminal immediately;
    /// sibling workers still need to observe drain and exit before join.
    pub fn worker_panicked(&self, worker: SealedTeamWorker) -> Result<(), SealedTeamError> {
        self.begin_drain(TeamDrainReason::WorkerPanicked);
        self.worker_exited(worker)
    }

    /// Fire only after the scoped region has joined every fixed child.
    pub fn join(&self) -> Result<(), SealedTeamError> {
        let mut state = lock_unpoisoned(&self.state);
        if state.phase == SealedTeamPhase::Running {
            state.phase = SealedTeamPhase::Draining;
            state.drain_reason = Some(TeamDrainReason::Completed);
        }
        if state.phase != SealedTeamPhase::Draining {
            return Err(SealedTeamError::JoinBeforeDrain { phase: state.phase });
        }
        let exited_children = state
            .workers
            .values()
            .filter(|worker| **worker == SealedTeamWorkerState::Exited)
            .count();
        if exited_children != state.expected_children {
            return Err(SealedTeamError::JoinBeforeWorkersExit {
                expected_children: state.expected_children,
                exited_children,
            });
        }
        let reason = state
            .drain_reason
            .ok_or(SealedTeamError::JoinBeforeDrain { phase: state.phase })?;
        state.phase = SealedTeamPhase::Joined;
        eprintln!(
            "SEALED_TEAM STAGE=LATCH RESULT=FIRED joined_children={} reason={}",
            exited_children,
            reason.as_str(),
        );
        Ok(())
    }

    pub fn snapshot(&self) -> SealedTeamSnapshot {
        let state = lock_unpoisoned(&self.state);
        let formed_children = state.workers.len();
        let running_children = state
            .workers
            .values()
            .filter(|worker| **worker == SealedTeamWorkerState::Running)
            .count();
        let exited_children = state
            .workers
            .values()
            .filter(|worker| **worker == SealedTeamWorkerState::Exited)
            .count();
        SealedTeamSnapshot {
            phase: state.phase,
            expected_children: state.expected_children,
            formed_children,
            running_children,
            exited_children,
            drain_reason: state.drain_reason,
        }
    }
}

/// A capacity-one command or reply lane. The sender deliberately uses
/// `try_send`: a second in-flight command is a typed backpressure refusal, not
/// a hidden queue or a blocking coordinator wait.
pub struct CapacityOneSender<T> {
    sender: mpsc::SyncSender<T>,
}

pub struct CapacityOneReceiver<T> {
    receiver: mpsc::Receiver<T>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CapacityOneLaneError {
    Full,
    Disconnected,
}

impl fmt::Display for CapacityOneLaneError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Full => formatter.write_str("capacity-one lane already has an in-flight message"),
            Self::Disconnected => formatter.write_str("capacity-one lane is disconnected"),
        }
    }
}

impl std::error::Error for CapacityOneLaneError {}

pub fn capacity_one_lane<T>() -> (CapacityOneSender<T>, CapacityOneReceiver<T>) {
    let (sender, receiver) = mpsc::sync_channel(1);
    (CapacityOneSender { sender }, CapacityOneReceiver { receiver })
}

/// Translate the process' effective compute-team width into the child cap for
/// one lending `Cx::scoped_cpu` region.
///
/// The blocking coordinator owns one deterministic shard itself. The pinned
/// foundation counts only calls to `ScopedCpu::spawn`, so a width of one is a
/// valid coordinator-only run and every wider team may create at most
/// `width - 1` children.
pub const fn scoped_cpu_child_cap(effective_compute_team_width: usize) -> Option<usize> {
    effective_compute_team_width.checked_sub(1)
}

impl<T> CapacityOneSender<T> {
    pub fn try_send(&self, value: T) -> Result<(), CapacityOneLaneError> {
        self.sender.try_send(value).map_err(|error| match error {
            mpsc::TrySendError::Full(_) => CapacityOneLaneError::Full,
            mpsc::TrySendError::Disconnected(_) => CapacityOneLaneError::Disconnected,
        })
    }
}

impl<T> CapacityOneReceiver<T> {
    pub fn recv(&self) -> Result<T, CapacityOneLaneError> {
        self.receiver
            .recv()
            .map_err(|_| CapacityOneLaneError::Disconnected)
    }
}

#[cfg(feature = "asupersync-runtime")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EntryGateState {
    Waiting,
    Released,
    Aborted,
}

/// A count-independent entry gate: if a foundation spawn refusal occurs while
/// formation is being created, the coordinator can still wake every already
/// formed worker into the non-fallible abort path before the scoped join.
#[cfg(feature = "asupersync-runtime")]
#[derive(Debug)]
struct EntryGate {
    state: Mutex<EntryGateState>,
    changed: std::sync::Condvar,
}

#[cfg(feature = "asupersync-runtime")]
impl EntryGate {
    fn new() -> Self {
        Self {
            state: Mutex::new(EntryGateState::Waiting),
            changed: std::sync::Condvar::new(),
        }
    }

    fn wait(&self) -> bool {
        let mut state = lock_unpoisoned(&self.state);
        while *state == EntryGateState::Waiting {
            state = self
                .changed
                .wait(state)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
        *state == EntryGateState::Released
    }

    fn release(&self) {
        let mut state = lock_unpoisoned(&self.state);
        if *state == EntryGateState::Waiting {
            *state = EntryGateState::Released;
            self.changed.notify_all();
        }
    }

    fn abort(&self) {
        let mut state = lock_unpoisoned(&self.state);
        if *state == EntryGateState::Waiting {
            *state = EntryGateState::Aborted;
            self.changed.notify_all();
        }
    }
}

#[cfg(feature = "asupersync-runtime")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SealedTeamCommand {
    Checkpoint,
    Stop,
}

#[cfg(feature = "asupersync-runtime")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SealedTeamReply {
    Checkpointed { worker_ordinal: usize },
    Cancelled { worker_ordinal: usize },
    Stopped { worker_ordinal: usize },
}

#[cfg(feature = "asupersync-runtime")]
fn drain_sealed_team_workers(command_senders: &mut Vec<CapacityOneSender<SealedTeamCommand>>) {
    for command_sender in command_senders.iter() {
        let _ = command_sender.try_send(SealedTeamCommand::Stop);
    }
    // A lane containing an already-issued checkpoint command cannot accept a
    // second stop message. Dropping all command senders is the bounded
    // fail-closed stop signal for that case: after a worker drains its one
    // in-flight command, `recv` observes disconnection and exits.
    command_senders.clear();
}

/// Typed execution result for the feature-gated native CPU team seam.
///
/// The outer task handle still preserves asupersync's cancellation/panic join
/// outcome. This inner result records the scoped CPU outcome without flattening
/// it into a generic application error.
#[cfg(feature = "asupersync-runtime")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SealedTeamRunError {
    Protocol(SealedTeamError),
    ScopeCancelled { detail: String },
    ScopePanicked { worker_ordinal: usize, detail: String },
    ScopeWorkerCapExceeded { cap: usize },
    CommandLane {
        worker_ordinal: usize,
        error: CapacityOneLaneError,
    },
    ReplyLaneDisconnected { worker_ordinal: usize },
    UnexpectedReply {
        expected_worker_ordinal: usize,
        actual_worker_ordinal: usize,
    },
}

#[cfg(feature = "asupersync-runtime")]
impl fmt::Display for SealedTeamRunError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Protocol(error) => error.fmt(formatter),
            Self::ScopeCancelled { detail } => {
                write!(formatter, "sealed scoped-CPU team cancelled: {detail}")
            }
            Self::ScopePanicked {
                worker_ordinal,
                detail,
            } => write!(
                formatter,
                "sealed scoped-CPU worker {worker_ordinal} panicked: {detail}"
            ),
            Self::ScopeWorkerCapExceeded { cap } => {
                write!(formatter, "sealed scoped-CPU team exceeded child cap {cap}")
            }
            Self::CommandLane {
                worker_ordinal,
                error,
            } => write!(
                formatter,
                "sealed scoped-CPU worker {worker_ordinal} command lane failed: {error}"
            ),
            Self::ReplyLaneDisconnected { worker_ordinal } => write!(
                formatter,
                "sealed scoped-CPU worker {worker_ordinal} reply lane disconnected"
            ),
            Self::UnexpectedReply {
                expected_worker_ordinal,
                actual_worker_ordinal,
            } => write!(
                formatter,
                "sealed scoped-CPU expected reply from worker {expected_worker_ordinal}, got worker {actual_worker_ordinal}"
            ),
        }
    }
}

#[cfg(feature = "asupersync-runtime")]
impl std::error::Error for SealedTeamRunError {}

#[cfg(feature = "asupersync-runtime")]
fn drain_reason_for_run_error(error: &SealedTeamRunError) -> TeamDrainReason {
    match error {
        SealedTeamRunError::ScopeCancelled { .. } => TeamDrainReason::Cancelled,
        SealedTeamRunError::ScopePanicked { .. }
        | SealedTeamRunError::ReplyLaneDisconnected { .. } => TeamDrainReason::WorkerPanicked,
        SealedTeamRunError::Protocol(_)
        | SealedTeamRunError::ScopeWorkerCapExceeded { .. }
        | SealedTeamRunError::CommandLane { .. }
        | SealedTeamRunError::UnexpectedReply { .. } => {
            TeamDrainReason::CoordinatorPanicked
        }
    }
}

/// Refusal before the one permitted `spawn_blocking` crossing can be admitted.
#[cfg(feature = "asupersync-runtime")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SealedTeamLaunchError {
    InvalidTeamWidth,
    MissingRequestContext,
    SpawnBlocking { detail: String },
}

#[cfg(feature = "asupersync-runtime")]
impl fmt::Display for SealedTeamLaunchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTeamWidth => {
                formatter.write_str("sealed scoped-CPU team width must include its coordinator")
            }
            Self::MissingRequestContext => formatter.write_str(
                "sealed scoped-CPU team requires an asupersync request context",
            ),
            Self::SpawnBlocking { detail } => {
                write!(formatter, "sealed scoped-CPU team spawn_blocking refused: {detail}")
            }
        }
    }
}

#[cfg(feature = "asupersync-runtime")]
impl std::error::Error for SealedTeamLaunchError {}

#[cfg(feature = "asupersync-runtime")]
impl SealedCpuTeam {
    /// Reconcile the model only after `Cx::scoped_cpu` has returned, which is
    /// the foundation's join guarantee for every child, including panicking or
    /// cancellation-observing children that could not report their own exit.
    fn reconcile_after_scoped_join(
        &self,
        default_reason: TeamDrainReason,
    ) -> Result<SealedTeamSnapshot, SealedTeamError> {
        {
            let mut state = lock_unpoisoned(&self.state);
            if state.phase == SealedTeamPhase::Joined {
                return Ok(snapshot_from_sealed_team_state(&state));
            }
            state.phase = SealedTeamPhase::Draining;
            state.drain_reason.get_or_insert(default_reason);
            for worker_state in state.workers.values_mut() {
                *worker_state = SealedTeamWorkerState::Exited;
            }
        }
        self.join()?;
        Ok(self.snapshot())
    }
}

#[cfg(feature = "asupersync-runtime")]
fn snapshot_from_sealed_team_state(state: &SealedTeamState) -> SealedTeamSnapshot {
    let formed_children = state.workers.len();
    let running_children = state
        .workers
        .values()
        .filter(|worker| **worker == SealedTeamWorkerState::Running)
        .count();
    let exited_children = state
        .workers
        .values()
        .filter(|worker| **worker == SealedTeamWorkerState::Exited)
        .count();
    SealedTeamSnapshot {
        phase: state.phase,
        expected_children: state.expected_children,
        formed_children,
        running_children,
        exited_children,
        drain_reason: state.drain_reason,
    }
}

#[cfg(feature = "asupersync-runtime")]
fn run_sealed_cpu_checkpoint_team(
    blocking_cx: &asupersync::cx::Cx,
    child_count: usize,
) -> Result<SealedTeamSnapshot, SealedTeamRunError> {
    use asupersync::cx::ScopedCpuError;

    let team = Arc::new(SealedCpuTeam::new(child_count));
    let gate = Arc::new(EntryGate::new());
    let scope_result = blocking_cx.scoped_cpu(child_count, |scope| {
        let mut workers = Vec::with_capacity(child_count);
        for _ in 0..child_count {
            workers.push(team.form_worker().map_err(SealedTeamRunError::Protocol)?);
        }
        let mut command_senders = Vec::with_capacity(child_count);
        let mut reply_receivers = Vec::with_capacity(child_count);
        for worker in workers {
            let worker_team = Arc::clone(&team);
            let worker_gate = Arc::clone(&gate);
            let (command_sender, command_receiver) = capacity_one_lane();
            let (reply_sender, reply_receiver) = capacity_one_lane();
            if let Err(error) = scope.spawn(move |cpu_cx| {
                if !worker_gate.wait() {
                    return;
                }
                if worker_team.worker_started(worker).is_err() {
                    return;
                }
                if cpu_cx.checkpoint().is_err() {
                    worker_team.begin_drain(TeamDrainReason::Cancelled);
                    let _ = reply_sender.try_send(SealedTeamReply::Cancelled {
                        worker_ordinal: worker.ordinal(),
                    });
                    let _ = worker_team.worker_exited(worker);
                    return;
                }
                loop {
                    let command = match command_receiver.recv() {
                        Ok(command) => command,
                        Err(_) => {
                            worker_team.begin_drain(TeamDrainReason::Cancelled);
                            let _ = worker_team.worker_exited(worker);
                            return;
                        }
                    };
                    match command {
                        SealedTeamCommand::Checkpoint => {
                            let reply = if cpu_cx.checkpoint().is_ok() {
                                SealedTeamReply::Checkpointed {
                                    worker_ordinal: worker.ordinal(),
                                }
                            } else {
                                worker_team.begin_drain(TeamDrainReason::Cancelled);
                                SealedTeamReply::Cancelled {
                                    worker_ordinal: worker.ordinal(),
                                }
                            };
                            let terminal = matches!(reply, SealedTeamReply::Cancelled { .. });
                            if reply_sender.try_send(reply).is_err() {
                                worker_team.begin_drain(TeamDrainReason::CoordinatorPanicked);
                                let _ = worker_team.worker_exited(worker);
                                return;
                            }
                            if terminal {
                                let _ = worker_team.worker_exited(worker);
                                return;
                            }
                        }
                        SealedTeamCommand::Stop => {
                            let _ = reply_sender.try_send(SealedTeamReply::Stopped {
                                worker_ordinal: worker.ordinal(),
                            });
                            let _ = worker_team.worker_exited(worker);
                            return;
                        }
                    }
                }
            }) {
                team.begin_drain(TeamDrainReason::CoordinatorPanicked);
                gate.abort();
                drain_sealed_team_workers(&mut command_senders);
                return Err(match error {
                    ScopedCpuError::WorkerCapExceeded { cap } => {
                        SealedTeamRunError::ScopeWorkerCapExceeded { cap }
                    }
                    ScopedCpuError::Cancelled(error) => SealedTeamRunError::ScopeCancelled {
                        detail: error.to_string(),
                    },
                    ScopedCpuError::ChildPanicked { child, message } => {
                        SealedTeamRunError::ScopePanicked {
                            worker_ordinal: child,
                            detail: message,
                        }
                    }
                });
            }
            command_senders.push(command_sender);
            reply_receivers.push(reply_receiver);
        }
        if let Err(error) = team.seal() {
            team.begin_drain(TeamDrainReason::CoordinatorPanicked);
            gate.abort();
            drain_sealed_team_workers(&mut command_senders);
            return Err(SealedTeamRunError::Protocol(error));
        }
        if let Err(error) = team.release_workers() {
            team.begin_drain(TeamDrainReason::CoordinatorPanicked);
            gate.abort();
            drain_sealed_team_workers(&mut command_senders);
            return Err(SealedTeamRunError::Protocol(error));
        }
        gate.release();
        if let Err(error) = blocking_cx.checkpoint() {
            team.begin_drain(TeamDrainReason::Cancelled);
            drain_sealed_team_workers(&mut command_senders);
            return Err(SealedTeamRunError::ScopeCancelled {
                detail: error.to_string(),
            });
        }
        eprintln!(
            "SEALED_TEAM STAGE=COORDINATOR RESULT=CHECKPOINTED child_count={child_count}",
        );
        for worker_ordinal in 0..command_senders.len() {
            if let Err(error) = command_senders[worker_ordinal]
                .try_send(SealedTeamCommand::Checkpoint)
            {
                let reason = match error {
                    CapacityOneLaneError::Full => TeamDrainReason::CoordinatorPanicked,
                    CapacityOneLaneError::Disconnected => TeamDrainReason::WorkerPanicked,
                };
                team.begin_drain(reason);
                drain_sealed_team_workers(&mut command_senders);
                return Err(SealedTeamRunError::CommandLane {
                    worker_ordinal,
                    error,
                });
            }
        }
        for worker_ordinal in 0..reply_receivers.len() {
            let reply = match reply_receivers[worker_ordinal].recv() {
                Ok(reply) => reply,
                Err(CapacityOneLaneError::Disconnected) => {
                    team.begin_drain(TeamDrainReason::WorkerPanicked);
                    drain_sealed_team_workers(&mut command_senders);
                    return Err(SealedTeamRunError::ReplyLaneDisconnected { worker_ordinal });
                }
                Err(CapacityOneLaneError::Full) => unreachable!("receivers cannot report full"),
            };
            let actual_worker_ordinal = match reply {
                SealedTeamReply::Checkpointed { worker_ordinal }
                | SealedTeamReply::Cancelled { worker_ordinal }
                | SealedTeamReply::Stopped { worker_ordinal } => worker_ordinal,
            };
            if actual_worker_ordinal != worker_ordinal {
                team.begin_drain(TeamDrainReason::CoordinatorPanicked);
                drain_sealed_team_workers(&mut command_senders);
                return Err(SealedTeamRunError::UnexpectedReply {
                    expected_worker_ordinal: worker_ordinal,
                    actual_worker_ordinal,
                });
            }
            if matches!(reply, SealedTeamReply::Cancelled { .. }) {
                team.begin_drain(TeamDrainReason::Cancelled);
                drain_sealed_team_workers(&mut command_senders);
                return Err(SealedTeamRunError::ScopeCancelled {
                    detail: format!("worker {worker_ordinal} observed cancellation before reply"),
                });
            }
            if matches!(reply, SealedTeamReply::Stopped { .. }) {
                team.begin_drain(TeamDrainReason::CoordinatorPanicked);
                drain_sealed_team_workers(&mut command_senders);
                return Err(SealedTeamRunError::UnexpectedReply {
                    expected_worker_ordinal: worker_ordinal,
                    actual_worker_ordinal,
                });
            }
        }
        for worker_ordinal in 0..command_senders.len() {
            if let Err(error) = command_senders[worker_ordinal].try_send(SealedTeamCommand::Stop) {
                let reason = match error {
                    CapacityOneLaneError::Full => TeamDrainReason::CoordinatorPanicked,
                    CapacityOneLaneError::Disconnected => TeamDrainReason::WorkerPanicked,
                };
                team.begin_drain(reason);
                drain_sealed_team_workers(&mut command_senders);
                return Err(SealedTeamRunError::CommandLane {
                    worker_ordinal,
                    error,
                });
            }
        }
        for worker_ordinal in 0..reply_receivers.len() {
            let reply = match reply_receivers[worker_ordinal].recv() {
                Ok(reply) => reply,
                Err(CapacityOneLaneError::Disconnected) => {
                    team.begin_drain(TeamDrainReason::WorkerPanicked);
                    drain_sealed_team_workers(&mut command_senders);
                    return Err(SealedTeamRunError::ReplyLaneDisconnected { worker_ordinal });
                }
                Err(CapacityOneLaneError::Full) => unreachable!("receivers cannot report full"),
            };
            let actual_worker_ordinal = match reply {
                SealedTeamReply::Checkpointed { worker_ordinal }
                | SealedTeamReply::Cancelled { worker_ordinal }
                | SealedTeamReply::Stopped { worker_ordinal } => worker_ordinal,
            };
            if actual_worker_ordinal != worker_ordinal {
                team.begin_drain(TeamDrainReason::CoordinatorPanicked);
                drain_sealed_team_workers(&mut command_senders);
                return Err(SealedTeamRunError::UnexpectedReply {
                    expected_worker_ordinal: worker_ordinal,
                    actual_worker_ordinal,
                });
            }
            if matches!(reply, SealedTeamReply::Cancelled { .. }) {
                team.begin_drain(TeamDrainReason::Cancelled);
                drain_sealed_team_workers(&mut command_senders);
                return Err(SealedTeamRunError::ScopeCancelled {
                    detail: format!("worker {worker_ordinal} observed cancellation while stopping"),
                });
            }
            if matches!(reply, SealedTeamReply::Checkpointed { .. }) {
                team.begin_drain(TeamDrainReason::CoordinatorPanicked);
                drain_sealed_team_workers(&mut command_senders);
                return Err(SealedTeamRunError::UnexpectedReply {
                    expected_worker_ordinal: worker_ordinal,
                    actual_worker_ordinal,
                });
            }
        }
        Ok(())
    });

    match scope_result {
        Ok(Ok(())) => team
            .reconcile_after_scoped_join(TeamDrainReason::Completed)
            .map_err(SealedTeamRunError::Protocol),
        Ok(Err(error)) => {
            let _ = team.reconcile_after_scoped_join(drain_reason_for_run_error(&error));
            Err(error)
        }
        Err(ScopedCpuError::Cancelled(error)) => {
            let _ = team.reconcile_after_scoped_join(TeamDrainReason::Cancelled);
            Err(SealedTeamRunError::ScopeCancelled {
                detail: error.to_string(),
            })
        }
        Err(ScopedCpuError::ChildPanicked { child, message }) => {
            let _ = team.reconcile_after_scoped_join(TeamDrainReason::WorkerPanicked);
            Err(SealedTeamRunError::ScopePanicked {
                worker_ordinal: child,
                detail: message,
            })
        }
        Err(ScopedCpuError::WorkerCapExceeded { cap }) => {
            let _ = team.reconcile_after_scoped_join(TeamDrainReason::CoordinatorPanicked);
            Err(SealedTeamRunError::ScopeWorkerCapExceeded { cap })
        }
    }
}

#[cfg(feature = "asupersync-runtime")]
struct RuntimeHost {
    runtime: asupersync::runtime::Runtime,
    blocking_pool: asupersync::runtime::blocking_pool::BlockingPoolHandle,
}

#[cfg(feature = "asupersync-runtime")]
impl RuntimeHost {
    fn build(
        config: ResourceHostConfig,
        guardrails: RuntimeGuardrails,
    ) -> Result<Self, RuntimeHostError> {
        use std::time::Duration;

        use asupersync::{
            runtime::RuntimeBuilder,
            types::CancelAttributionConfig,
        };

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
        .on_thread_stop(mark_runtime_worker_stopped)
        .cancel_attribution_config(CancelAttributionConfig::new(
            guardrails.cancel_attribution_max_depth,
            guardrails.cancel_attribution_max_memory_bytes,
        ))
        .deadline_monitoring(move |monitor| {
            monitor
                .enabled(true)
                .check_interval(Duration::from_millis(
                    guardrails.deadline_check_interval_millis,
                ))
                .checkpoint_timeout(Duration::from_millis(
                    guardrails.checkpoint_timeout_millis,
                ))
                .on_warning(|warning| {
                    eprintln!(
                        "ENGINE_RESOURCES DEADLINE_WARNING reason={:?} last_checkpoint={:?} checkpoint_history_entries={}",
                        warning.reason,
                        warning.last_checkpoint_message,
                        warning.checkpoint_history.len(),
                    );
                })
        });
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
    guardrails: RuntimeGuardrails,
    memory: Arc<MemoryLedger>,
    next_lease_id: AtomicU64,
    leases: Mutex<LeaseState>,
    next_closure_id: AtomicU64,
    completions: Mutex<CompletionState>,
    #[cfg(feature = "asupersync-runtime")]
    runtime: RuntimeHost,
}

impl EngineResources {
    fn new(
        config: ResourceHostConfig,
        thread_inventory: ThreadInventory,
    ) -> Result<Self, RuntimeHostError> {
        #[cfg(feature = "asupersync-runtime")]
        {
            let guardrails = RuntimeGuardrails::PINNED_DEFAULTS;
            let runtime = RuntimeHost::build(config, guardrails)?;
            return Ok(Self {
                config,
                thread_inventory,
                guardrails,
                memory: Arc::new(MemoryLedger::new(
                    config.memory_ceiling_bytes,
                    config.leak_response_policy,
                )),
                next_lease_id: AtomicU64::new(0),
                leases: Mutex::new(LeaseState::default()),
                next_closure_id: AtomicU64::new(0),
                completions: Mutex::new(CompletionState::default()),
                runtime,
            });
        }

        #[cfg(not(feature = "asupersync-runtime"))]
        {
            let _ = (config, thread_inventory);
            Err(RuntimeHostError::RuntimeFeatureDisabled)
        }
    }

    #[cfg(all(test, not(feature = "asupersync-runtime")))]
    fn new_for_test(config: ResourceHostConfig, thread_inventory: ThreadInventory) -> Self {
        let guardrails = RuntimeGuardrails::PINNED_DEFAULTS;
        Self {
            config,
            thread_inventory,
            guardrails,
            memory: Arc::new(MemoryLedger::new(
                config.memory_ceiling_bytes,
                config.leak_response_policy,
            )),
            next_lease_id: AtomicU64::new(0),
            leases: Mutex::new(LeaseState::default()),
            next_closure_id: AtomicU64::new(0),
            completions: Mutex::new(CompletionState::default()),
        }
    }

    #[cfg(all(test, feature = "asupersync-runtime"))]
    fn new_for_test(config: ResourceHostConfig, thread_inventory: ThreadInventory) -> Self {
        Self::new(config, thread_inventory)
            .expect("feature-enabled test host must retain a real blocking pool")
    }

    /// Fixed configuration selected by the first successful installation.
    pub const fn config(&self) -> ResourceHostConfig {
        self.config
    }

    /// Complete runnable-thread accounting used by future health/receipt code.
    pub const fn thread_inventory(&self) -> ThreadInventory {
        self.thread_inventory
    }

    /// The finite deadline-monitor and cancellation-attribution limits bound
    /// into this process host.
    pub const fn runtime_guardrails(&self) -> RuntimeGuardrails {
        self.guardrails
    }

    /// Current aggregate memory charge snapshot.
    pub fn memory_snapshot(&self) -> MemorySnapshot {
        self.memory.snapshot()
    }

    /// Read remaining aggregate allocation capacity without building the
    /// diagnostic snapshot map. This is safe for an allocation-free plan
    /// preview, but callers must still reserve because other engines can race.
    pub fn available_memory_bytes(&self) -> Result<u64, ReservationError> {
        self.memory.available_bytes()
    }

    /// Evaluate a complete certificate against the current process ledger
    /// without reserving or allocating. A subsequent reservation is required
    /// before any request resource becomes visible.
    pub fn preflight_admission(
        &self,
        certificate: &AdmissionCertificate,
    ) -> Result<AdmissionDecision, ReservationError> {
        Ok(certificate.aggregate_decision(
            self.available_memory_bytes()?,
            self.config.memory_ceiling_bytes,
        ))
    }

    fn reserve_admission(
        &self,
        engine_lease_id: u64,
        certificate: &AdmissionCertificate,
    ) -> Result<AdmissionReservation, AdmissionError> {
        match self
            .preflight_admission(certificate)
            .map_err(AdmissionError::Reservation)?
        {
            AdmissionDecision::Admitted => {}
            AdmissionDecision::Refused(rejection) => {
                return Err(AdmissionError::Refused(rejection));
            }
        }
        let charges = certificate.charges();
        let mut reservation = AdmissionReservation::empty();
        macro_rules! reserve_slot {
            ($slot:ident, $memory_class:expr, $bytes:expr) => {
                if let Err(error) = self.reserve_admission_slot(
                    engine_lease_id,
                    $memory_class,
                    $bytes,
                    &mut reservation.$slot,
                ) {
                    let _ = reservation.abort_all();
                    return Err(AdmissionError::Reservation(error));
                }
            };
        }
        reserve_slot!(weights, MemoryClass::Weights, charges.weights);
        reserve_slot!(prefix_cache, MemoryClass::PrefixCache, charges.prefix_cache);
        reserve_slot!(kv_pages, MemoryClass::KvPages, charges.kv_pages);
        reserve_slot!(
            activation_scratch,
            MemoryClass::ActivationScratch,
            charges.activation_scratch
        );
        reserve_slot!(
            logit_scratch,
            MemoryClass::LogitScratch,
            charges.logit_scratch
        );
        reserve_slot!(
            grammar_cache,
            MemoryClass::GrammarCache,
            charges.grammar_cache
        );
        reserve_slot!(job_buffers, MemoryClass::JobBuffers, charges.job_buffers);
        reserve_slot!(
            admission_reserve,
            MemoryClass::AdmissionReserve,
            charges.admission_reserve
        );
        Ok(reservation)
    }

    fn reserve_admission_slot(
        &self,
        engine_lease_id: u64,
        memory_class: MemoryClass,
        bytes: u64,
        slot: &mut Option<MemoryReservation>,
    ) -> Result<(), ReservationError> {
        if bytes != 0 {
            *slot = Some(self.memory.reserve(engine_lease_id, memory_class, bytes)?);
        }
        Ok(())
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

    /// Count pool closures that still own their resources through actual
    /// completion. Scheduler shutdown must treat this as drain work.
    pub fn outstanding_closure_snapshot(&self) -> OutstandingClosureSnapshot {
        let completions = lock_unpoisoned(&self.completions);
        OutstandingClosureSnapshot {
            active_closures: completions.active.len(),
            wrapper_cancelled_closures: completions
                .active
                .values()
                .filter(|closure| closure.wrapper_cancelled)
                .count(),
        }
    }

    fn register_blocking_closure(self: &Arc<Self>, engine_lease_id: u64) -> BlockingClosureGuard {
        let closure_id = self.next_closure_id.fetch_add(1, Ordering::Relaxed) + 1;
        let mut completions = lock_unpoisoned(&self.completions);
        completions.active.insert(
            closure_id,
            OutstandingClosure {
                engine_lease_id,
                wrapper_cancelled: false,
            },
        );
        BlockingClosureGuard {
            resources: Arc::clone(self),
            closure_id,
        }
    }

    fn mark_wrapper_cancelled(&self, closure_id: u64) {
        let mut completions = lock_unpoisoned(&self.completions);
        let Some(closure) = completions.active.get_mut(&closure_id) else {
            eprintln!("ENGINE_RESOURCES CANCELLED_WRAPPER_UNKNOWN closure_id={closure_id}");
            return;
        };
        closure.wrapper_cancelled = true;
    }

    fn complete_blocking_closure(&self, closure_id: u64) {
        let mut completions = lock_unpoisoned(&self.completions);
        let Some(closure) = completions.active.remove(&closure_id) else {
            eprintln!("ENGINE_RESOURCES COMPLETION_UNKNOWN closure_id={closure_id}");
            return;
        };
        eprintln!(
            "ENGINE_RESOURCES CLOSURE_COMPLETE closure_id={} engine_lease_id={} wrapper_cancelled={}",
            closure_id,
            closure.engine_lease_id,
            closure.wrapper_cancelled,
        );
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

    /// Turn a complete local and aggregate admission certificate into owned
    /// two-phase process-ledger obligations. The caller must commit only after
    /// allocation succeeds, or abort on cancellation/error before allocation.
    pub fn reserve_admission(
        &self,
        certificate: &AdmissionCertificate,
    ) -> Result<AdmissionReservation, AdmissionError> {
        self.resources.reserve_admission(self.lease_id, certificate)
    }

    pub fn resources(&self) -> &Arc<EngineResources> {
        &self.resources
    }

    /// Register a pool closure before releasing it into fallible work.
    ///
    /// The closure must retain this guard together with its admission, memory,
    /// and output guards. An async wrapper cancellation calls
    /// [`BlockingClosureGuard::mark_wrapper_cancelled`] but cannot release the
    /// process resources; only the closure's actual completion drops the guard.
    pub fn register_blocking_closure(&self) -> BlockingClosureGuard {
        self.resources.register_blocking_closure(self.lease_id)
    }

    /// Launch the one permitted blocking closure for a bounded sealed CPU
    /// checkpoint team. The returned task remains owned by the request region;
    /// its physical closure retains the completion guard until `scoped_cpu`
    /// has joined every fixed child and the closure returns.
    ///
    /// This is the foundation seam, not a scheduler shortcut: callers supply
    /// the already-admitted effective team width. The coordinator keeps one
    /// deterministic shard, while the scope creates at most `width - 1`
    /// children. Later stage schedulers attach their capacity-one
    /// command/reply loops without creating another team.
    #[cfg(feature = "asupersync-runtime")]
    pub fn spawn_sealed_cpu_checkpoint_team(
        &self,
        effective_compute_team_width: usize,
    ) -> Result<
        asupersync::runtime::TaskHandle<Result<SealedTeamSnapshot, SealedTeamRunError>>,
        SealedTeamLaunchError,
    > {
        use asupersync::cx::Cx;

        let child_count = scoped_cpu_child_cap(effective_compute_team_width)
            .ok_or(SealedTeamLaunchError::InvalidTeamWidth)?;
        let completion_guard = self.register_blocking_closure();
        self.resources.runtime().block_on(async move {
            let request_cx = Cx::current().ok_or(SealedTeamLaunchError::MissingRequestContext)?;
            request_cx
                .spawn_blocking(move |blocking_cx| {
                    let _completion_guard = completion_guard;
                    run_sealed_cpu_checkpoint_team(&blocking_cx, child_count)
                })
                .map_err(|error| SealedTeamLaunchError::SpawnBlocking {
                    detail: error.to_string(),
                })
        })
    }
}

impl Drop for EngineLease {
    fn drop(&mut self) {
        lock_unpoisoned(&self.resources.leases)
            .active
            .remove(&self.lease_id);
    }
}

/// Actual-completion registration for one `spawn_blocking` closure.
///
/// This guard is intentionally `Send`: it is moved into the physical pool
/// closure and drops only after that closure's work, resource guards, and
/// completion latch have reached their terminal cleanup path.
#[derive(Debug)]
pub struct BlockingClosureGuard {
    resources: Arc<EngineResources>,
    closure_id: u64,
}

impl BlockingClosureGuard {
    pub const fn closure_id(&self) -> u64 {
        self.closure_id
    }

    /// Record wrapper cancellation without releasing the physical closure.
    pub fn mark_wrapper_cancelled(&self) {
        self.resources.mark_wrapper_cancelled(self.closure_id);
    }
}

impl Drop for BlockingClosureGuard {
    fn drop(&mut self) {
        self.resources.complete_blocking_closure(self.closure_id);
    }
}

/// The single process broker. Its mutex ensures a racing first install builds
/// one host rather than transiently creating multiple resource domains.
#[derive(Debug, Default)]
pub struct ResourceBroker {
    installed: Mutex<Option<Arc<EngineResources>>>,
    #[cfg(test)]
    allow_test_runtime_model: bool,
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
        #[cfg(test)]
        let resources = if self.allow_test_runtime_model {
            Arc::new(EngineResources::new_for_test(requested, thread_inventory))
        } else {
            Arc::new(
                EngineResources::new(requested, thread_inventory)
                    .map_err(ResourceBrokerError::Runtime)?,
            )
        };
        #[cfg(not(test))]
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
        Self {
            installed: Mutex::new(None),
            allow_test_runtime_model: true,
        }
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
    fn pinned_guardrails_are_finite_and_nonzero() {
        let guardrails = RuntimeGuardrails::PINNED_DEFAULTS;
        assert!(guardrails.deadline_check_interval_millis > 0);
        assert!(guardrails.checkpoint_timeout_millis > 0);
        assert!(guardrails.cancel_attribution_max_depth > 0);
        assert!(guardrails.cancel_attribution_max_memory_bytes > 0);
        assert!(guardrails.cancel_attribution_max_depth < usize::MAX);
        assert!(guardrails.cancel_attribution_max_memory_bytes < usize::MAX);
    }

    #[cfg(not(feature = "asupersync-runtime"))]
    #[test]
    fn production_broker_refuses_to_model_a_missing_runtime_pool() {
        let broker = ResourceBroker::default();
        assert_eq!(
            broker
                .install(config())
                .expect_err("production host must not use an inline fallback"),
            ResourceBrokerError::Runtime(RuntimeHostError::RuntimeFeatureDisabled),
        );
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
        let resources = Arc::new(EngineResources::new_for_test(config, thread_inventory));
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

    #[test]
    fn cancelled_wrapper_remains_outstanding_until_closure_drop() {
        let broker = ResourceBroker::isolated_for_test();
        let resources = broker.install(config()).expect("host installs");
        let lease = resources.acquire_lease();
        let closure = lease.register_blocking_closure();
        assert_eq!(
            resources.outstanding_closure_snapshot(),
            OutstandingClosureSnapshot {
                active_closures: 1,
                wrapper_cancelled_closures: 0,
            }
        );
        closure.mark_wrapper_cancelled();
        assert_eq!(
            resources.outstanding_closure_snapshot(),
            OutstandingClosureSnapshot {
                active_closures: 1,
                wrapper_cancelled_closures: 1,
            },
            "wrapper cancellation must not release a live pool closure"
        );
        drop(closure);
        assert_eq!(
            resources.outstanding_closure_snapshot(),
            OutstandingClosureSnapshot {
                active_closures: 0,
                wrapper_cancelled_closures: 0,
            },
            "only actual completion clears the outstanding drain entry"
        );
    }

    #[cfg(feature = "asupersync-runtime")]
    #[test]
    fn sealed_cpu_team_joins_before_its_completion_latch_releases() {
        let config = config();
        let inventory = config.thread_inventory().expect("inventory is valid");
        let resources = Arc::new(EngineResources::new_for_test(config, inventory));
        let lease = resources.acquire_lease();

        assert_eq!(
            lease
                .spawn_sealed_cpu_checkpoint_team(0)
                .expect_err("zero width cannot omit the coordinator"),
            SealedTeamLaunchError::InvalidTeamWidth
        );
        assert_eq!(
            resources.outstanding_closure_snapshot().active_closures,
            0,
            "a pre-entry refusal must not retain a completion latch"
        );

        let mut handle = lease
            .spawn_sealed_cpu_checkpoint_team(4)
            .expect("one coordinator and three sealed children launch");
        let joined = resources.runtime().block_on(async {
            let request_cx = asupersync::cx::Cx::current()
                .expect("runtime block_on installs the ambient request context");
            handle.join(&request_cx).await
        });
        let snapshot = joined
            .expect("blocking wrapper joins without a task failure")
            .expect("sealed scoped CPU region completes");
        assert_eq!(snapshot.phase, SealedTeamPhase::Joined);
        assert_eq!(snapshot.expected_children, 3);
        assert_eq!(snapshot.formed_children, 3);
        assert_eq!(snapshot.exited_children, 3);
        assert_eq!(snapshot.drain_reason, Some(TeamDrainReason::Completed));
        assert_eq!(
            resources.outstanding_closure_snapshot().active_closures,
            0,
            "the physical closure drops its latch only after the scoped join"
        );
    }

    fn admission_config(memory_ceiling_bytes: u64) -> ResourceHostConfig {
        ResourceHostConfig::new(
            RuntimePreset::LowLatency,
            4,
            2,
            3,
            1,
            32,
            memory_ceiling_bytes,
            LeakResponsePolicy::RecordAndEscalate,
        )
        .expect("admission test config is valid")
    }

    fn complete_admission_request(
        context_tokens: u64,
        batch_rows: u64,
        budget_bytes: u64,
    ) -> AdmissionRequest {
        AdmissionRequest::decode(context_tokens, batch_rows, KvCacheQuantization::Bf16)
            .with_local_memory_budget(budget_bytes)
            .with_fixed_residency(
                ResidencyAccounting::new(100, 100).expect("resident bytes fit mapping"),
            )
            .with_kv_page_metadata_per_token(0)
            .with_reserves(0, 0)
    }

    #[test]
    fn certificate_computes_every_term_and_thread_inventory_independently() {
        let host = admission_config(u64::MAX);
        let request = AdmissionRequest::decode(2, 3, KvCacheQuantization::Bf16)
            .with_local_memory_budget(u64::MAX)
            .with_os_reserve(11)
            .with_fixed_residency(
                ResidencyAccounting::new(1_000, 800).expect("resident bytes fit mapping"),
            )
            .with_elastic_cache(13)
            .with_replicated_weight_residency(
                ResidencyAccounting::new(20, 19).expect("resident bytes fit mapping"),
            )
            .with_kv_page_metadata_per_token(7)
            .with_elastic_rows(23, 29, 31, 37, 41)
            .with_reserves(43, 47);
        let certificate = AdmissionCertificate::build(
            request,
            host.thread_inventory().expect("thread inventory fits"),
        )
        .expect("term arithmetic fits");
        let terms = certificate.terms();
        assert_eq!(terms.fixed_mapped_bytes, Some(1_000));
        assert_eq!(terms.fixed_resident_bytes, Some(800));
        assert_eq!(terms.elastic_cache_bytes, 13);
        assert_eq!(terms.replicated_weight_mapped_bytes, 20);
        assert_eq!(terms.replicated_weight_resident_bytes, 19);
        assert_eq!(terms.kv_payload_bytes, 6 * BF16_KV_BYTES_PER_TOKEN);
        assert_eq!(terms.kv_scale_bytes, 0);
        assert_eq!(terms.kv_page_metadata_bytes, Some(42));
        assert_eq!(terms.activation_bytes, 3 * 23);
        assert_eq!(terms.full_logit_bytes, 3 * FULL_F32_LOGIT_ROW_BYTES);
        assert_eq!(terms.grammar_state_bytes, 3 * 29);
        assert_eq!(terms.source_state_bytes, 3 * 31);
        assert_eq!(terms.queue_bytes, 3 * 37);
        assert_eq!(terms.output_buffer_bytes, 3 * 41);
        assert_eq!(terms.unmodeled_emergency_reserve_bytes, 43);
        assert_eq!(terms.safety_margin_bytes, 47);
        assert_eq!(terms.os_reserve_bytes, 11);
        assert_eq!(certificate.thread_inventory().runtime_workers, 4);
        assert_eq!(certificate.thread_inventory().blocking_coordinators, 2);
        assert_eq!(
            certificate
                .thread_inventory()
                .scoped_cpu_children_per_coordinator,
            3
        );
        assert_eq!(certificate.thread_inventory().helper_threads, 1);
        assert_eq!(certificate.thread_inventory().total_runnable_threads, 13);
        assert_eq!(certificate.local_decision(), AdmissionDecision::Admitted);
    }

    #[test]
    fn certificate_keeps_the_8192_and_64_row_kv_numbers_exact() {
        let inventory = admission_config(u64::MAX)
            .thread_inventory()
            .expect("thread inventory fits");
        let one = AdmissionCertificate::build(
            complete_admission_request(DEFAULT_CONTEXT_TOKEN_CAP, 1, u64::MAX),
            inventory,
        )
        .expect("8192-row certificate fits arithmetic");
        assert_eq!(one.terms().kv_payload_bytes, 1_476_395_008);
        assert_eq!(one.terms().kv_payload_bytes, 1_408 * 1024 * 1024);

        let sixty_four = AdmissionCertificate::build(
            complete_admission_request(DEFAULT_CONTEXT_TOKEN_CAP, 64, u64::MAX),
            inventory,
        )
        .expect("64-row certificate fits arithmetic");
        assert_eq!(sixty_four.terms().kv_payload_bytes, 94_489_280_512);
        assert_eq!(sixty_four.terms().kv_payload_bytes, 88 * 1024 * 1024 * 1024);
    }

    #[test]
    fn certificate_abort_restores_the_exact_process_ledger_balance() {
        let config = admission_config(1_000_000);
        let inventory = config.thread_inventory().expect("thread inventory fits");
        let resources = Arc::new(EngineResources::new_for_test(config, inventory));
        let certificate = AdmissionCertificate::build(
            complete_admission_request(1, 1, 1_000_000),
            resources.thread_inventory(),
        )
        .expect("certificate fits arithmetic");
        let expected = certificate
            .terms()
            .committed_bytes
            .expect("complete certificate has ledger bytes");
        let lease = resources.acquire_lease();
        let reservation = lease
            .reserve_admission(&certificate)
            .expect("complete certificate reserves before allocation");
        assert_eq!(resources.memory_snapshot().reserved_bytes, expected);
        reservation
            .abort()
            .expect("cancellation restores every two-phase obligation");
        let snapshot = resources.memory_snapshot();
        assert_eq!(snapshot.reserved_bytes, 0);
        assert_eq!(snapshot.committed_bytes, 0);
        assert_eq!(snapshot.outstanding_obligations, 0);
    }

    #[test]
    fn aggregate_multi_engine_admission_cannot_double_promise_process_memory() {
        let config = admission_config(1_000_000);
        let inventory = config.thread_inventory().expect("thread inventory fits");
        let resources = Arc::new(EngineResources::new_for_test(config, inventory));
        let certificate = AdmissionCertificate::build(
            complete_admission_request(1, 1, 1_000_000),
            resources.thread_inventory(),
        )
        .expect("certificate fits arithmetic");
        let first_engine = resources.acquire_lease();
        let second_engine = resources.acquire_lease();
        let committed = first_engine
            .reserve_admission(&certificate)
            .expect("first engine reserves the aggregate capacity")
            .commit()
            .expect("first engine commits after allocation");
        assert!(matches!(
            second_engine.reserve_admission(&certificate),
            Err(AdmissionError::Refused(
                AdmissionRejection::AggregateCapacityExceeded { .. }
            ))
        ));
        committed.release();
        second_engine
            .reserve_admission(&certificate)
            .expect("release makes the same process capacity available again")
            .abort()
            .expect("second engine cancellation restores the aggregate ledger");
        let snapshot = resources.memory_snapshot();
        assert_eq!(snapshot.reserved_bytes, 0);
        assert_eq!(snapshot.committed_bytes, 0);
    }

    #[test]
    fn certificate_refuses_missing_authority_and_checked_overflow_without_wrapping() {
        let inventory = admission_config(u64::MAX)
            .thread_inventory()
            .expect("thread inventory fits");
        let incomplete = AdmissionCertificate::build(
            AdmissionRequest::decode(1, 1, KvCacheQuantization::Bf16),
            inventory,
        )
        .expect("known partial terms fit arithmetic");
        assert_eq!(
            incomplete.local_decision(),
            AdmissionDecision::Refused(AdmissionRejection::FixedResidencyUnconfigured)
        );
        assert_eq!(
            AdmissionCertificate::build(
                AdmissionRequest::decode(u64::MAX, 2, KvCacheQuantization::Bf16),
                inventory,
            ),
            Err(AdmissionBuildError::ArithmeticOverflow {
                term: AdmissionTerm::KvPayload
            })
        );
    }
}
