//! Typed library and CLI error boundary.
//!
//! This module is the sole authority for the stable process exit-code
//! contract.  Keep the numeric values and machine names stable: changing one
//! is a versioned breaking change for both human and robot callers.

use std::{fmt, process::ExitCode as ProcessExitCode};

use serde::{Serialize, Serializer};

/// A stable CLI exit code.
///
/// `ErrorCode::Ok` is represented by successful typed results, not by an
/// [`FnlpError`].  The remaining variants classify terminal failures at the
/// library/CLI policy boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum ErrorCode {
    /// The operation completed successfully, including calibrated abstention.
    Ok = 0,
    /// An uncategorized error or a violated internal invariant.
    Generic = 1,
    /// The caller supplied invalid CLI arguments.
    Usage = 2,
    /// The requested model artifact is not installed or cannot be located.
    ModelNotFound = 3,
    /// Input bytes could not be decoded or parsed.
    InputDecodeOrParse = 4,
    /// A deadline, poll, cost, or other execution budget was exhausted.
    BudgetOrTimeout = 5,
    /// The caller or runtime cancelled the operation.
    Cancelled = 6,
    /// An artifact failed integrity, format, or version validation.
    ArtifactIntegrityOrFormatOrVersion = 7,
    /// A schema or data-only recipe could not be compiled.
    SchemaOrRecipeCompile = 8,
    /// Admission or a checked resource limit refused the operation.
    AdmissionOrResourceLimit = 9,
    /// A structured task could not produce a valid result at all.
    StructuredTaskNoResult = 10,
}

impl ErrorCode {
    /// The complete frozen process-code domain in numeric order.
    pub const ALL: [Self; 11] = [
        Self::Ok,
        Self::Generic,
        Self::Usage,
        Self::ModelNotFound,
        Self::InputDecodeOrParse,
        Self::BudgetOrTimeout,
        Self::Cancelled,
        Self::ArtifactIntegrityOrFormatOrVersion,
        Self::SchemaOrRecipeCompile,
        Self::AdmissionOrResourceLimit,
        Self::StructuredTaskNoResult,
    ];

    /// Return the frozen numeric representation used by the OS process status.
    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    /// Convert this stable value to Rust's process-status wrapper.
    pub fn as_process_exit(self) -> ProcessExitCode {
        ProcessExitCode::from(self.as_u8())
    }
}

impl Serialize for ErrorCode {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u8(self.as_u8())
    }
}

/// One machine-readable row of the frozen exit-code table.
///
/// Robot schema generation and documentation tooling consume
/// [`EXIT_CODE_TABLE`] instead of maintaining their own copy.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct ExitCodeSpec {
    /// The process status supplied to the operating system.
    pub code: ErrorCode,
    /// Stable snake_case machine name.
    pub name: &'static str,
    /// Concise human-facing meaning.
    pub description: &'static str,
}

/// The one canonical, machine-readable exit-code table.
pub const EXIT_CODE_TABLE: [ExitCodeSpec; 11] = [
    ExitCodeSpec {
        code: ErrorCode::Ok,
        name: "ok",
        description: "successful result, including calibrated abstention",
    },
    ExitCodeSpec {
        code: ErrorCode::Generic,
        name: "generic",
        description: "uncategorized failure or invariant violation",
    },
    ExitCodeSpec {
        code: ErrorCode::Usage,
        name: "usage",
        description: "invalid command-line usage",
    },
    ExitCodeSpec {
        code: ErrorCode::ModelNotFound,
        name: "model_not_found",
        description: "model artifact is unavailable",
    },
    ExitCodeSpec {
        code: ErrorCode::InputDecodeOrParse,
        name: "input_decode_or_parse",
        description: "input decoding or parsing failed",
    },
    ExitCodeSpec {
        code: ErrorCode::BudgetOrTimeout,
        name: "budget_or_timeout",
        description: "execution budget or timeout exhausted",
    },
    ExitCodeSpec {
        code: ErrorCode::Cancelled,
        name: "cancelled",
        description: "operation cancelled",
    },
    ExitCodeSpec {
        code: ErrorCode::ArtifactIntegrityOrFormatOrVersion,
        name: "artifact_integrity_or_format_or_version",
        description: "artifact integrity, format, or version mismatch",
    },
    ExitCodeSpec {
        code: ErrorCode::SchemaOrRecipeCompile,
        name: "schema_or_recipe_compile",
        description: "schema or recipe compilation failed",
    },
    ExitCodeSpec {
        code: ErrorCode::AdmissionOrResourceLimit,
        name: "admission_or_resource_limit",
        description: "admission or resource limit refused the operation",
    },
    ExitCodeSpec {
        code: ErrorCode::StructuredTaskNoResult,
        name: "structured_task_no_result",
        description: "structured task could not produce a valid result",
    },
];

/// Serialize the canonical table for robot-schema and documentation consumers.
///
/// Callers must not maintain a parallel table: this function is the only
/// source for the stable JSON rows (`code`, `name`, and `description`).
pub fn exit_code_table_json() -> Result<Vec<u8>, serde_json::Error> {
    serde_json::to_vec(&EXIT_CODE_TABLE)
}

/// The pinned asupersync cancellation kinds, retained until this module maps a
/// terminal result for the CLI.
///
/// This mirrors `asupersync::types::CancelKind` without making the public
/// error boundary depend on the optional runtime feature.  The runtime adapter
/// must preserve the original `CancelReason` attribution in its diagnostic or
/// robot envelope and convert its kind losslessly to this type only at the
/// library/CLI policy boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum CancellationKind {
    /// Explicit cancellation requested by the caller.
    User,
    /// A timeout fired.
    Timeout,
    /// A deadline budget was exhausted.
    Deadline,
    /// A poll-quota budget was exhausted.
    PollQuota,
    /// A cost budget was exhausted.
    CostBudget,
    /// A supervised sibling failed under a fail-fast policy.
    FailFast,
    /// A loser-drain cancellation that must never be root-terminal.
    RaceLost,
    /// An owning parent region cancelled normally.
    ParentCancelled,
    /// Checked resources were unavailable.
    ResourceUnavailable,
    /// The runtime is shutting down.
    Shutdown,
    /// A linked task exited abnormally.
    LinkedExit,
}

impl CancellationKind {
    /// All pinned variants.  Tests must remain exhaustive over this array.
    pub const ALL: [Self; 11] = [
        Self::User,
        Self::Timeout,
        Self::Deadline,
        Self::PollQuota,
        Self::CostBudget,
        Self::FailFast,
        Self::RaceLost,
        Self::ParentCancelled,
        Self::ResourceUnavailable,
        Self::Shutdown,
        Self::LinkedExit,
    ];

    /// Stable diagnostic/robot spelling for this exact cancellation cause.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Timeout => "timeout",
            Self::Deadline => "deadline",
            Self::PollQuota => "poll_quota",
            Self::CostBudget => "cost_budget",
            Self::FailFast => "fail_fast",
            Self::RaceLost => "race_lost",
            Self::ParentCancelled => "parent_cancelled",
            Self::ResourceUnavailable => "resource_unavailable",
            Self::Shutdown => "shutdown",
            Self::LinkedExit => "linked_exit",
        }
    }
}

/// A cancellation that has reached the CLI policy boundary.
///
/// `underlying` is meaningful only for supervised `FailFast` and `LinkedExit`
/// paths.  It lets their process status reflect the known original failure
/// rather than misrepresenting them as ordinary user cancellation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Cancellation {
    kind: CancellationKind,
    underlying: Option<Box<FnlpError>>,
}

impl Cancellation {
    /// Construct a cancellation without an underlying supervised failure.
    pub const fn new(kind: CancellationKind) -> Self {
        Self {
            kind,
            underlying: None,
        }
    }

    /// Attach the known root error for a fail-fast or linked-exit cancellation.
    pub fn with_underlying(kind: CancellationKind, underlying: FnlpError) -> Self {
        Self {
            kind,
            underlying: Some(Box::new(underlying)),
        }
    }

    /// Return the exact pinned cancellation kind for diagnostics and robots.
    pub const fn kind(&self) -> CancellationKind {
        self.kind
    }

    /// The attached supervised failure, when the runtime has one.
    pub fn underlying(&self) -> Option<&FnlpError> {
        self.underlying.as_deref()
    }

    /// Map this exact cause at the one allowed policy boundary.
    pub fn exit_code(&self) -> ErrorCode {
        match self.kind {
            CancellationKind::Timeout
            | CancellationKind::Deadline
            | CancellationKind::PollQuota
            | CancellationKind::CostBudget => ErrorCode::BudgetOrTimeout,
            CancellationKind::User
            | CancellationKind::ParentCancelled
            | CancellationKind::Shutdown => ErrorCode::Cancelled,
            CancellationKind::ResourceUnavailable => ErrorCode::AdmissionOrResourceLimit,
            CancellationKind::FailFast | CancellationKind::LinkedExit => self
                .underlying()
                .map_or(ErrorCode::Generic, FnlpError::exit_code),
            // RaceLost is only valid while draining a losing branch.  A root
            // terminal RaceLost proves a supervision invariant was violated.
            CancellationKind::RaceLost => ErrorCode::Generic,
        }
    }

    /// Whether this terminal cancellation requires a retained crashpack.
    pub const fn crashpack_required(&self) -> bool {
        matches!(self.kind, CancellationKind::RaceLost)
    }

    /// A stable, non-content diagnostic marker for the robot envelope.
    pub const fn diagnostic_marker(&self) -> &'static str {
        if self.crashpack_required() {
            "invariant_failure=race_lost_root crashpack_required=true"
        } else {
            "cancellation_policy_boundary"
        }
    }
}

/// Typed errors exposed by the library and mapped once to a stable exit code.
///
/// The payload is deliberately a static category rather than document text:
/// diagnostics must identify the failure without retaining or echoing private
/// prompt/input/output bytes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FnlpError {
    /// An uncategorized failure.
    Generic { category: &'static str },
    /// Command-line syntax or semantic usage was invalid.
    Usage { category: &'static str },
    /// The selected model artifact was unavailable.
    ModelNotFound { category: &'static str },
    /// Input bytes could not be decoded or parsed.
    InputDecodeOrParse { category: &'static str },
    /// A named budget or timeout was exhausted.
    BudgetOrTimeout { category: &'static str },
    /// A cancellation whose exact kind remains available to diagnostics.
    Cancelled(Cancellation),
    /// Artifact bytes failed integrity, format, or version checks.
    ArtifactIntegrityOrFormatOrVersion { category: &'static str },
    /// A schema or bounded data-only recipe could not compile.
    SchemaOrRecipeCompile { category: &'static str },
    /// Admission or a checked resource guard refused the operation.
    AdmissionOrResourceLimit { category: &'static str },
    /// A structured task cannot produce a valid result at all.
    StructuredTaskNoResult { category: &'static str },
    /// An invariant breach that requires stronger evidence than an item error.
    InvariantViolation { category: &'static str },
}

impl FnlpError {
    /// Map this typed error to the frozen process exit-code contract.
    pub fn exit_code(&self) -> ErrorCode {
        match self {
            Self::Generic { .. } | Self::InvariantViolation { .. } => ErrorCode::Generic,
            Self::Usage { .. } => ErrorCode::Usage,
            Self::ModelNotFound { .. } => ErrorCode::ModelNotFound,
            Self::InputDecodeOrParse { .. } => ErrorCode::InputDecodeOrParse,
            Self::BudgetOrTimeout { .. } => ErrorCode::BudgetOrTimeout,
            Self::Cancelled(cancellation) => cancellation.exit_code(),
            Self::ArtifactIntegrityOrFormatOrVersion { .. } => {
                ErrorCode::ArtifactIntegrityOrFormatOrVersion
            }
            Self::SchemaOrRecipeCompile { .. } => ErrorCode::SchemaOrRecipeCompile,
            Self::AdmissionOrResourceLimit { .. } => ErrorCode::AdmissionOrResourceLimit,
            Self::StructuredTaskNoResult { .. } => ErrorCode::StructuredTaskNoResult,
        }
    }

    /// Only this variant may be emitted as a per-document `doc_error`.
    ///
    /// All other variants are run-level failures.  In particular, cancellation
    /// and panic are never relabeled as a document parse failure.
    pub const fn is_per_document_input_error(&self) -> bool {
        matches!(self, Self::InputDecodeOrParse { .. })
    }
}

impl fmt::Display for FnlpError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled(cancellation) => write!(
                formatter,
                "fnlp error [{}]: {}",
                cancellation.exit_code().as_u8(),
                cancellation.kind().as_str()
            ),
            Self::Generic { category }
            | Self::Usage { category }
            | Self::ModelNotFound { category }
            | Self::InputDecodeOrParse { category }
            | Self::BudgetOrTimeout { category }
            | Self::ArtifactIntegrityOrFormatOrVersion { category }
            | Self::SchemaOrRecipeCompile { category }
            | Self::AdmissionOrResourceLimit { category }
            | Self::StructuredTaskNoResult { category }
            | Self::InvariantViolation { category } => {
                write!(
                    formatter,
                    "fnlp error [{}]: {category}",
                    self.exit_code().as_u8()
                )
            }
        }
    }
}

impl std::error::Error for FnlpError {}

/// A successful structured-task result state.
///
/// Abstention is semantically successful and therefore deliberately distinct
/// from [`FnlpError::StructuredTaskNoResult`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StructuredTaskStatus {
    /// A valid task result was produced.
    Completed,
    /// The calibrated task intentionally abstained from a valid response.
    Abstained,
}

impl StructuredTaskStatus {
    /// Stable status text for human and robot result envelopes.
    pub const fn status(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Abstained => "abstained",
        }
    }

    /// Both successful task states exit zero.
    pub const fn exit_code(self) -> ErrorCode {
        ErrorCode::Ok
    }
}

/// A supervised panic report.  Panic is intentionally not an [`FnlpError`]
/// because it must retain its stronger `run_error` handling and crashpack
/// obligation rather than being flattened into a per-document failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PanicFailure {
    category: &'static str,
}

impl PanicFailure {
    /// Construct a non-sensitive panic classification for a supervised result.
    pub const fn new(category: &'static str) -> Self {
        Self { category }
    }

    /// The non-content classification attached to diagnostics.
    pub const fn category(self) -> &'static str {
        self.category
    }

    /// Panics require heavier retained evidence.
    pub const fn crashpack_required(self) -> bool {
        true
    }
}

/// A terminal run state after outcome severity has reached the policy boundary.
///
/// The scheduler must preserve `Outcome::{Ok, Err, Cancelled, Panicked}` up
/// to this point.  This type is deliberately terminal-only: it does not grant
/// adapters permission to flatten cancellation or panic earlier.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RunTerminal {
    /// A successful structured task result.
    Success(StructuredTaskStatus),
    /// A typed, non-panic failure.
    Error(FnlpError),
    /// A supervised panic, always emitted as `run_error`.
    Panicked(PanicFailure),
}

impl RunTerminal {
    /// The stable exit code for this terminal state.
    pub fn exit_code(&self) -> ErrorCode {
        match self {
            Self::Success(status) => status.exit_code(),
            Self::Error(error) => error.exit_code(),
            Self::Panicked(_) => ErrorCode::Generic,
        }
    }

    /// Return the OS process exit wrapper for the terminal state.
    pub fn as_process_exit(&self) -> ProcessExitCode {
        self.exit_code().as_process_exit()
    }

    /// The robot event kind for a terminal run.  Panics can never be `doc_error`.
    pub const fn robot_event_name(&self) -> &'static str {
        match self {
            Self::Success(_) => "run_complete",
            Self::Error(_) | Self::Panicked(_) => "run_error",
        }
    }

    /// Whether diagnostics must retain a crashpack for this terminal state.
    pub const fn crashpack_required(&self) -> bool {
        match self {
            Self::Success(_) => false,
            Self::Error(FnlpError::Cancelled(cancellation)) => cancellation.crashpack_required(),
            Self::Error(FnlpError::InvariantViolation { .. }) | Self::Panicked(_) => true,
            Self::Error(_) => false,
        }
    }
}
