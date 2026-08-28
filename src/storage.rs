//! Optional, metadata-only job history.
//!
//! This module deliberately has no API for document, prompt, output, result,
//! or spool bytes.  Durable content spools are a later, separately-owned
//! surface.  When storage is disabled, [`MetadataStore::open`] returns before
//! inspecting the configured path, so the disabled default has no database or
//! filesystem side effect.

use std::fmt;
#[cfg(feature = "metadata-store")]
use std::path::Path;
use std::path::PathBuf;

/// The file mode required for a database created by the Unix platform path.
///
/// Other platforms fail closed until their owner-only platform surface is
/// explicitly ratified; there is no permissive fallback.
#[cfg(unix)]
pub const OWNER_ONLY_DATABASE_MODE: u32 = 0o600;

/// An opaque identifier assigned by the owning engine or CLI.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MetadataId(u64);

impl MetadataId {
    /// Creates an identifier from an engine-owned monotonic value.
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the engine-owned numeric representation.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// A SHA-256 digest used as a commitment, never as retained content.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Sha256Digest([u8; 32]);

impl Sha256Digest {
    /// Creates a digest from its fixed-width SHA-256 bytes.
    #[must_use]
    pub const fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Encodes the digest for the metadata database.
    #[must_use]
    pub fn to_hex(self) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut encoded = String::with_capacity(64);
        for byte in self.0 {
            encoded.push(char::from(HEX[usize::from(byte >> 4)]));
            encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
        }
        encoded
    }
}

/// Lifecycle state that may be retained for a job or item.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JobState {
    /// Accepted but not yet started.
    Pending,
    /// Currently executing.
    Running,
    /// Completed successfully.
    Succeeded,
    /// Reached a typed failure outcome.
    Failed,
    /// Reached a typed cancellation outcome.
    Cancelled,
}

impl JobState {
    #[cfg(feature = "metadata-store")]
    const fn as_database_value(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Running => "running",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }

    #[cfg(feature = "metadata-store")]
    fn from_database_value(value: &str) -> Option<Self> {
        match value {
            "pending" => Some(Self::Pending),
            "running" => Some(Self::Running),
            "succeeded" => Some(Self::Succeeded),
            "failed" => Some(Self::Failed),
            "cancelled" => Some(Self::Cancelled),
            _ => None,
        }
    }
}

/// A compact error summary.  Error message text is intentionally not retained.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TypedErrorSummary {
    /// Stable machine-readable error category.
    pub code: u16,
    /// Stable machine-readable operation or policy context.
    pub context_code: u16,
}

/// Metadata retained for a job.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct JobMetadata {
    /// Engine-owned job identifier.
    pub job_id: MetadataId,
    /// Commitment to the owner-supplied manifest.
    pub manifest_digest: Sha256Digest,
    /// Current lifecycle state.
    pub state: JobState,
    /// Owner-supplied UTC milliseconds since the Unix epoch.
    pub created_at_ms: u64,
    /// Owner-supplied UTC milliseconds since the Unix epoch.
    pub updated_at_ms: u64,
}

/// Metadata retained for an item within a job.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ItemMetadata {
    /// Parent job identifier.
    pub job_id: MetadataId,
    /// Engine-owned item identifier.
    pub item_id: MetadataId,
    /// Commitment to the input bytes; the bytes themselves are not retained.
    pub input_digest: Sha256Digest,
    /// Current lifecycle state.
    pub state: JobState,
    /// Number of attempted executions.
    pub attempt_count: u32,
    /// Owner-supplied UTC milliseconds since the Unix epoch.
    pub updated_at_ms: u64,
}

/// A state-transition audit record for a job item.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StateTransitionMetadata {
    /// Engine-owned transition identifier.
    pub transition_id: MetadataId,
    /// Parent job identifier.
    pub job_id: MetadataId,
    /// Parent item identifier.
    pub item_id: MetadataId,
    /// State reached at this transition.
    pub state: JobState,
    /// Owner-supplied UTC milliseconds since the Unix epoch.
    pub recorded_at_ms: u64,
}

/// A completed or interrupted attempt's metadata.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AttemptMetadata {
    /// Engine-owned attempt identifier.
    pub attempt_id: MetadataId,
    /// Parent job identifier.
    pub job_id: MetadataId,
    /// Parent item identifier.
    pub item_id: MetadataId,
    /// One-based attempt number.
    pub attempt_number: u32,
    /// Typed terminal or intermediate state.
    pub state: JobState,
    /// Owner-supplied UTC milliseconds since the Unix epoch.
    pub recorded_at_ms: u64,
}

/// A scalar measurement associated with an item attempt.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MetricMetadata {
    /// Engine-owned metric identifier.
    pub metric_id: MetadataId,
    /// Parent job identifier.
    pub job_id: MetadataId,
    /// Parent item identifier.
    pub item_id: MetadataId,
    /// Stable metric kind identifier.
    pub kind_code: u16,
    /// Numeric measurement in the kind's documented unit.
    pub value: f64,
    /// Owner-supplied UTC milliseconds since the Unix epoch.
    pub recorded_at_ms: u64,
}

/// A typed error observation without error-message text.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ErrorMetadata {
    /// Engine-owned error identifier.
    pub error_id: MetadataId,
    /// Parent job identifier.
    pub job_id: MetadataId,
    /// Parent item identifier.
    pub item_id: MetadataId,
    /// Closed error category and context codes.
    pub summary: TypedErrorSummary,
    /// Owner-supplied UTC milliseconds since the Unix epoch.
    pub recorded_at_ms: u64,
}

/// The runtime switch and engine-owned location for optional metadata history.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StoreConfig {
    enabled: bool,
    #[cfg(feature = "metadata-store")]
    database_path: PathBuf,
}

impl StoreConfig {
    /// Disables the store.  The path is never inspected in this configuration.
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            #[cfg(feature = "metadata-store")]
            database_path: PathBuf::new(),
        }
    }

    /// Disables the store while retaining an owner-supplied path solely for
    /// configuration round-trips.  [`MetadataStore::open`] still never
    /// inspects this path while disabled.
    #[must_use]
    pub fn disabled_at_path(database_path: impl Into<PathBuf>) -> Self {
        #[cfg(not(feature = "metadata-store"))]
        let _ = database_path;

        Self {
            enabled: false,
            #[cfg(feature = "metadata-store")]
            database_path: database_path.into(),
        }
    }

    /// Enables metadata history at an engine- or CLI-owned database location.
    #[must_use]
    pub fn metadata_only(database_path: impl Into<PathBuf>) -> Self {
        #[cfg(not(feature = "metadata-store"))]
        let _ = database_path;

        Self {
            enabled: true,
            #[cfg(feature = "metadata-store")]
            database_path: database_path.into(),
        }
    }

    /// Whether runtime configuration selected metadata history.
    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        self.enabled
    }
}

/// Failure modes deliberately omit filesystem paths and database error text.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StorageError {
    /// Runtime requested the optional store without compiling its feature.
    MetadataStoreFeatureDisabled,
    /// The active OS does not have an approved owner-only file surface.
    PlatformSurfaceUnavailable,
    /// The owner supplied an empty, non-UTF-8, or otherwise unusable path.
    InvalidDatabasePath,
    /// The database is not a regular owner-only file.
    OwnerOnlyPermissionRequired,
    /// A database operation failed; the underlying message is intentionally not retained.
    DatabaseOperationFailed { operation: &'static str },
    /// An owner-provided value cannot be represented in the database integer domain.
    IntegerOutOfRange,
    /// Schema policy rejected a table or column definition.
    SchemaPolicyViolation {
        /// Static table name from the compiled schema description.
        table: &'static str,
        /// Static column name from the compiled schema description.
        column: &'static str,
    },
    /// A stored lifecycle state was not one of the compiled state enums.
    InvalidStoredState,
}

impl fmt::Display for StorageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MetadataStoreFeatureDisabled => {
                formatter.write_str("metadata store feature is not compiled")
            }
            Self::PlatformSurfaceUnavailable => {
                formatter.write_str("owner-only database platform surface is unavailable")
            }
            Self::InvalidDatabasePath => formatter.write_str("invalid metadata database path"),
            Self::OwnerOnlyPermissionRequired => {
                formatter.write_str("metadata database must be a regular owner-only file")
            }
            Self::DatabaseOperationFailed { operation } => {
                write!(formatter, "metadata database operation failed: {operation}")
            }
            Self::IntegerOutOfRange => {
                formatter.write_str("metadata value is outside the database integer domain")
            }
            Self::SchemaPolicyViolation { table, column } => {
                write!(
                    formatter,
                    "metadata schema policy rejected {table}.{column}"
                )
            }
            Self::InvalidStoredState => {
                formatter.write_str("metadata store contained an unknown state")
            }
        }
    }
}

impl std::error::Error for StorageError {}

/// Classification for every permitted metadata column.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ColumnClass {
    /// Opaque engine-owned identifiers.
    Identifier,
    /// Cryptographic commitment.
    Digest,
    /// Closed lifecycle enum.
    State,
    /// Attempt sequence or count.
    Attempt,
    /// Owner-supplied timestamp.
    TimestampMillis,
    /// Closed numeric metric kind.
    MetricKind,
    /// Scalar metric value.
    MetricValue,
    /// Closed typed error category.
    ErrorCode,
    /// Closed typed error context.
    ErrorContextCode,
}

/// A column admitted to the metadata-only schema.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ColumnSpec {
    /// Stable column name.
    pub name: &'static str,
    /// Permitted value classification.
    pub class: ColumnClass,
}

/// A table admitted to the metadata-only schema.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TableSpec {
    /// Stable table name.
    pub name: &'static str,
    /// Closed set of permitted columns.
    pub columns: &'static [ColumnSpec],
}

const JOB_COLUMNS: &[ColumnSpec] = &[
    ColumnSpec {
        name: "job_id",
        class: ColumnClass::Identifier,
    },
    ColumnSpec {
        name: "manifest_digest",
        class: ColumnClass::Digest,
    },
    ColumnSpec {
        name: "state",
        class: ColumnClass::State,
    },
    ColumnSpec {
        name: "created_at_ms",
        class: ColumnClass::TimestampMillis,
    },
    ColumnSpec {
        name: "updated_at_ms",
        class: ColumnClass::TimestampMillis,
    },
];
const ITEM_COLUMNS: &[ColumnSpec] = &[
    ColumnSpec {
        name: "item_id",
        class: ColumnClass::Identifier,
    },
    ColumnSpec {
        name: "job_id",
        class: ColumnClass::Identifier,
    },
    ColumnSpec {
        name: "input_digest",
        class: ColumnClass::Digest,
    },
    ColumnSpec {
        name: "state",
        class: ColumnClass::State,
    },
    ColumnSpec {
        name: "attempt_count",
        class: ColumnClass::Attempt,
    },
    ColumnSpec {
        name: "updated_at_ms",
        class: ColumnClass::TimestampMillis,
    },
];
const STATE_COLUMNS: &[ColumnSpec] = &[
    ColumnSpec {
        name: "transition_id",
        class: ColumnClass::Identifier,
    },
    ColumnSpec {
        name: "job_id",
        class: ColumnClass::Identifier,
    },
    ColumnSpec {
        name: "item_id",
        class: ColumnClass::Identifier,
    },
    ColumnSpec {
        name: "state",
        class: ColumnClass::State,
    },
    ColumnSpec {
        name: "recorded_at_ms",
        class: ColumnClass::TimestampMillis,
    },
];
const ATTEMPT_COLUMNS: &[ColumnSpec] = &[
    ColumnSpec {
        name: "attempt_id",
        class: ColumnClass::Identifier,
    },
    ColumnSpec {
        name: "job_id",
        class: ColumnClass::Identifier,
    },
    ColumnSpec {
        name: "item_id",
        class: ColumnClass::Identifier,
    },
    ColumnSpec {
        name: "attempt_number",
        class: ColumnClass::Attempt,
    },
    ColumnSpec {
        name: "state",
        class: ColumnClass::State,
    },
    ColumnSpec {
        name: "recorded_at_ms",
        class: ColumnClass::TimestampMillis,
    },
];
const METRIC_COLUMNS: &[ColumnSpec] = &[
    ColumnSpec {
        name: "metric_id",
        class: ColumnClass::Identifier,
    },
    ColumnSpec {
        name: "job_id",
        class: ColumnClass::Identifier,
    },
    ColumnSpec {
        name: "item_id",
        class: ColumnClass::Identifier,
    },
    ColumnSpec {
        name: "kind_code",
        class: ColumnClass::MetricKind,
    },
    ColumnSpec {
        name: "value",
        class: ColumnClass::MetricValue,
    },
    ColumnSpec {
        name: "recorded_at_ms",
        class: ColumnClass::TimestampMillis,
    },
];
const ERROR_COLUMNS: &[ColumnSpec] = &[
    ColumnSpec {
        name: "error_id",
        class: ColumnClass::Identifier,
    },
    ColumnSpec {
        name: "job_id",
        class: ColumnClass::Identifier,
    },
    ColumnSpec {
        name: "item_id",
        class: ColumnClass::Identifier,
    },
    ColumnSpec {
        name: "error_code",
        class: ColumnClass::ErrorCode,
    },
    ColumnSpec {
        name: "context_code",
        class: ColumnClass::ErrorContextCode,
    },
    ColumnSpec {
        name: "recorded_at_ms",
        class: ColumnClass::TimestampMillis,
    },
];

const TABLES: &[TableSpec] = &[
    TableSpec {
        name: "jobs",
        columns: JOB_COLUMNS,
    },
    TableSpec {
        name: "items",
        columns: ITEM_COLUMNS,
    },
    TableSpec {
        name: "state_transitions",
        columns: STATE_COLUMNS,
    },
    TableSpec {
        name: "attempts",
        columns: ATTEMPT_COLUMNS,
    },
    TableSpec {
        name: "metrics",
        columns: METRIC_COLUMNS,
    },
    TableSpec {
        name: "errors",
        columns: ERROR_COLUMNS,
    },
];

/// Returns the complete, closed metadata-schema allowlist.
#[must_use]
pub const fn schema_tables() -> &'static [TableSpec] {
    TABLES
}

/// Validates one table against the compiled metadata allowlist.
pub fn validate_table_schema(table: &TableSpec) -> Result<usize, StorageError> {
    let expected = TABLES
        .iter()
        .find(|candidate| candidate.name == table.name)
        .ok_or(StorageError::SchemaPolicyViolation {
            table: table.name,
            column: "<table>",
        })?;

    if expected.columns.len() != table.columns.len() {
        return Err(StorageError::SchemaPolicyViolation {
            table: table.name,
            column: "<column-count>",
        });
    }

    for column in table.columns {
        if !expected.columns.iter().any(|allowed| allowed == column) {
            return Err(StorageError::SchemaPolicyViolation {
                table: table.name,
                column: column.name,
            });
        }
    }
    Ok(table.columns.len())
}

/// Validates the complete schema and returns the count of checked columns.
pub fn validate_schema_policy(tables: &[TableSpec]) -> Result<usize, StorageError> {
    if tables.len() != TABLES.len() {
        return Err(StorageError::SchemaPolicyViolation {
            table: "<schema>",
            column: "<table-count>",
        });
    }
    tables.iter().try_fold(0, |count, table| {
        validate_table_schema(table).map(|checked| count + checked)
    })
}

/// Runtime-gated metadata store.  Its enabled variant is feature-gated too.
pub enum MetadataStore {
    /// The default: no database is opened and no path is inspected.
    Disabled,
    /// The optional, metadata-only database connection.
    #[cfg(feature = "metadata-store")]
    Enabled(EnabledMetadataStore),
}

impl MetadataStore {
    /// Opens the optional database only when both feature and runtime config allow it.
    pub fn open(config: StoreConfig) -> Result<Self, StorageError> {
        if !config.enabled {
            return Ok(Self::Disabled);
        }

        #[cfg(not(feature = "metadata-store"))]
        {
            let _ = config;
            Err(StorageError::MetadataStoreFeatureDisabled)
        }

        #[cfg(feature = "metadata-store")]
        {
            EnabledMetadataStore::open(&config.database_path).map(Self::Enabled)
        }
    }

    /// Returns whether this process actually has an open metadata database.
    #[must_use]
    pub const fn is_open(&self) -> bool {
        match self {
            Self::Disabled => false,
            #[cfg(feature = "metadata-store")]
            Self::Enabled(_) => true,
        }
    }

    /// Closes the database under engine or CLI ownership.
    pub fn close(self) {}

    /// Persists a job's metadata when persistence is enabled.
    pub fn record_job(&self, record: JobMetadata) -> Result<(), StorageError> {
        match self {
            Self::Disabled => Ok(()),
            #[cfg(feature = "metadata-store")]
            Self::Enabled(store) => store.record_job(record),
        }
    }

    /// Persists item metadata when persistence is enabled.
    pub fn record_item(&self, record: ItemMetadata) -> Result<(), StorageError> {
        match self {
            Self::Disabled => Ok(()),
            #[cfg(feature = "metadata-store")]
            Self::Enabled(store) => store.record_item(record),
        }
    }

    /// Persists a state transition when persistence is enabled.
    pub fn record_state_transition(
        &self,
        record: StateTransitionMetadata,
    ) -> Result<(), StorageError> {
        match self {
            Self::Disabled => Ok(()),
            #[cfg(feature = "metadata-store")]
            Self::Enabled(store) => store.record_state_transition(record),
        }
    }

    /// Persists an attempt record when persistence is enabled.
    pub fn record_attempt(&self, record: AttemptMetadata) -> Result<(), StorageError> {
        match self {
            Self::Disabled => Ok(()),
            #[cfg(feature = "metadata-store")]
            Self::Enabled(store) => store.record_attempt(record),
        }
    }

    /// Persists a numeric metric when persistence is enabled.
    pub fn record_metric(&self, record: MetricMetadata) -> Result<(), StorageError> {
        match self {
            // The `record` parameter is intentionally consumed only by the
            // feature-gated `Enabled` arm. Acknowledging it here silences the
            // default-build `unused_variable` warning without renaming the
            // public-API parameter.
            Self::Disabled => {
                let _ = record;
                Ok(())
            }
            #[cfg(feature = "metadata-store")]
            Self::Enabled(store) => store.record_metric(record),
        }
    }

    /// Persists a typed error summary when persistence is enabled.
    pub fn record_error(&self, record: ErrorMetadata) -> Result<(), StorageError> {
        match self {
            Self::Disabled => {
                let _ = record;
                Ok(())
            }
            #[cfg(feature = "metadata-store")]
            Self::Enabled(store) => store.record_error(record),
        }
    }

    /// Reads a stored job state without exposing arbitrary database values.
    pub fn job_state(&self, job_id: MetadataId) -> Result<Option<JobState>, StorageError> {
        match self {
            Self::Disabled => {
                let _ = job_id;
                Ok(None)
            }
            #[cfg(feature = "metadata-store")]
            Self::Enabled(store) => store.job_state(job_id),
        }
    }

    /// Reads a stored item state without exposing arbitrary database values.
    pub fn item_state(&self, item_id: MetadataId) -> Result<Option<JobState>, StorageError> {
        match self {
            Self::Disabled => {
                let _ = item_id;
                Ok(None)
            }
            #[cfg(feature = "metadata-store")]
            Self::Enabled(store) => store.item_state(item_id),
        }
    }
}

#[cfg(feature = "metadata-store")]
pub struct EnabledMetadataStore {
    connection: fsqlite::Connection,
}

#[cfg(feature = "metadata-store")]
impl EnabledMetadataStore {
    fn open(database_path: &Path) -> Result<Self, StorageError> {
        prepare_owner_only_database(database_path)?;
        let path = database_path
            .to_str()
            .ok_or(StorageError::InvalidDatabasePath)?;
        let connection = fsqlite::Connection::open(path)
            .map_err(|_| StorageError::DatabaseOperationFailed { operation: "open" })?;
        let store = Self { connection };
        store.initialize_schema()?;
        Ok(store)
    }

    fn initialize_schema(&self) -> Result<(), StorageError> {
        self.connection
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS jobs (\
                    job_id INTEGER PRIMARY KEY,\
                    manifest_digest TEXT NOT NULL,\
                    state TEXT NOT NULL,\
                    created_at_ms INTEGER NOT NULL,\
                    updated_at_ms INTEGER NOT NULL\
                );\
                CREATE TABLE IF NOT EXISTS items (\
                    item_id INTEGER PRIMARY KEY,\
                    job_id INTEGER NOT NULL,\
                    input_digest TEXT NOT NULL,\
                    state TEXT NOT NULL,\
                    attempt_count INTEGER NOT NULL,\
                    updated_at_ms INTEGER NOT NULL\
                );\
                CREATE TABLE IF NOT EXISTS state_transitions (\
                    transition_id INTEGER PRIMARY KEY,\
                    job_id INTEGER NOT NULL,\
                    item_id INTEGER NOT NULL,\
                    state TEXT NOT NULL,\
                    recorded_at_ms INTEGER NOT NULL\
                );\
                CREATE TABLE IF NOT EXISTS attempts (\
                    attempt_id INTEGER PRIMARY KEY,\
                    job_id INTEGER NOT NULL,\
                    item_id INTEGER NOT NULL,\
                    attempt_number INTEGER NOT NULL,\
                    state TEXT NOT NULL,\
                    recorded_at_ms INTEGER NOT NULL\
                );\
                CREATE TABLE IF NOT EXISTS metrics (\
                    metric_id INTEGER PRIMARY KEY,\
                    job_id INTEGER NOT NULL,\
                    item_id INTEGER NOT NULL,\
                    kind_code INTEGER NOT NULL,\
                    value REAL NOT NULL,\
                    recorded_at_ms INTEGER NOT NULL\
                );\
                CREATE TABLE IF NOT EXISTS errors (\
                    error_id INTEGER PRIMARY KEY,\
                    job_id INTEGER NOT NULL,\
                    item_id INTEGER NOT NULL,\
                    error_code INTEGER NOT NULL,\
                    context_code INTEGER NOT NULL,\
                    recorded_at_ms INTEGER NOT NULL\
                );",
            )
            .map_err(|_| StorageError::DatabaseOperationFailed {
                operation: "initialize schema",
            })
    }

    fn record_job(&self, record: JobMetadata) -> Result<(), StorageError> {
        let parameters = [
            sqlite_integer(record.job_id.get())?,
            fsqlite::SqliteValue::Text(record.manifest_digest.to_hex().into()),
            fsqlite::SqliteValue::Text(record.state.as_database_value().into()),
            sqlite_integer(record.created_at_ms)?,
            sqlite_integer(record.updated_at_ms)?,
        ];
        self.connection
            .execute_with_params(
                "INSERT INTO jobs (job_id, manifest_digest, state, created_at_ms, updated_at_ms) \
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                &parameters,
            )
            .map_err(|_| StorageError::DatabaseOperationFailed {
                operation: "record job",
            })?;
        Ok(())
    }

    fn record_item(&self, record: ItemMetadata) -> Result<(), StorageError> {
        let parameters = [
            sqlite_integer(record.item_id.get())?,
            sqlite_integer(record.job_id.get())?,
            fsqlite::SqliteValue::Text(record.input_digest.to_hex().into()),
            fsqlite::SqliteValue::Text(record.state.as_database_value().into()),
            fsqlite::SqliteValue::Integer(i64::from(record.attempt_count)),
            sqlite_integer(record.updated_at_ms)?,
        ];
        self.connection
            .execute_with_params(
                "INSERT INTO items (item_id, job_id, input_digest, state, attempt_count, updated_at_ms) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                &parameters,
            )
            .map_err(|_| StorageError::DatabaseOperationFailed {
                operation: "record item",
            })?;
        Ok(())
    }

    fn record_state_transition(&self, record: StateTransitionMetadata) -> Result<(), StorageError> {
        let parameters = [
            sqlite_integer(record.transition_id.get())?,
            sqlite_integer(record.job_id.get())?,
            sqlite_integer(record.item_id.get())?,
            fsqlite::SqliteValue::Text(record.state.as_database_value().into()),
            sqlite_integer(record.recorded_at_ms)?,
        ];
        self.connection
            .execute_with_params(
                "INSERT INTO state_transitions (transition_id, job_id, item_id, state, recorded_at_ms) \
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                &parameters,
            )
            .map_err(|_| StorageError::DatabaseOperationFailed {
                operation: "record state transition",
            })?;
        let state_parameters = [
            fsqlite::SqliteValue::Text(record.state.as_database_value().into()),
            sqlite_integer(record.recorded_at_ms)?,
            sqlite_integer(record.item_id.get())?,
            sqlite_integer(record.job_id.get())?,
        ];
        let updated = self
            .connection
            .execute_with_params(
                "UPDATE items SET state = ?1, updated_at_ms = ?2 WHERE item_id = ?3 AND job_id = ?4",
                &state_parameters,
            )
            .map_err(|_| StorageError::DatabaseOperationFailed {
                operation: "apply item state transition",
            })?;
        if updated != 1 {
            return Err(StorageError::DatabaseOperationFailed {
                operation: "apply exactly one item state transition",
            });
        }
        Ok(())
    }

    fn record_attempt(&self, record: AttemptMetadata) -> Result<(), StorageError> {
        let parameters = [
            sqlite_integer(record.attempt_id.get())?,
            sqlite_integer(record.job_id.get())?,
            sqlite_integer(record.item_id.get())?,
            fsqlite::SqliteValue::Integer(i64::from(record.attempt_number)),
            fsqlite::SqliteValue::Text(record.state.as_database_value().into()),
            sqlite_integer(record.recorded_at_ms)?,
        ];
        self.connection
            .execute_with_params(
                "INSERT INTO attempts (attempt_id, job_id, item_id, attempt_number, state, recorded_at_ms) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                &parameters,
            )
            .map_err(|_| StorageError::DatabaseOperationFailed {
                operation: "record attempt",
            })?;
        Ok(())
    }

    fn record_metric(&self, record: MetricMetadata) -> Result<(), StorageError> {
        let parameters = [
            sqlite_integer(record.metric_id.get())?,
            sqlite_integer(record.job_id.get())?,
            sqlite_integer(record.item_id.get())?,
            fsqlite::SqliteValue::Integer(i64::from(record.kind_code)),
            fsqlite::SqliteValue::Float(record.value),
            sqlite_integer(record.recorded_at_ms)?,
        ];
        self.connection
            .execute_with_params(
                "INSERT INTO metrics (metric_id, job_id, item_id, kind_code, value, recorded_at_ms) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                &parameters,
            )
            .map_err(|_| StorageError::DatabaseOperationFailed {
                operation: "record metric",
            })?;
        Ok(())
    }

    fn record_error(&self, record: ErrorMetadata) -> Result<(), StorageError> {
        let parameters = [
            sqlite_integer(record.error_id.get())?,
            sqlite_integer(record.job_id.get())?,
            sqlite_integer(record.item_id.get())?,
            fsqlite::SqliteValue::Integer(i64::from(record.summary.code)),
            fsqlite::SqliteValue::Integer(i64::from(record.summary.context_code)),
            sqlite_integer(record.recorded_at_ms)?,
        ];
        self.connection
            .execute_with_params(
                "INSERT INTO errors (error_id, job_id, item_id, error_code, context_code, recorded_at_ms) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                &parameters,
            )
            .map_err(|_| StorageError::DatabaseOperationFailed {
                operation: "record typed error",
            })?;
        Ok(())
    }

    fn job_state(&self, job_id: MetadataId) -> Result<Option<JobState>, StorageError> {
        self.state_for("jobs", "job_id", job_id)
    }

    fn item_state(&self, item_id: MetadataId) -> Result<Option<JobState>, StorageError> {
        self.state_for("items", "item_id", item_id)
    }

    fn state_for(
        &self,
        table: &'static str,
        identifier_column: &'static str,
        identifier: MetadataId,
    ) -> Result<Option<JobState>, StorageError> {
        let parameters = [sqlite_integer(identifier.get())?];
        let query = match (table, identifier_column) {
            ("jobs", "job_id") => "SELECT state FROM jobs WHERE job_id = ?1",
            ("items", "item_id") => "SELECT state FROM items WHERE item_id = ?1",
            _ => {
                return Err(StorageError::DatabaseOperationFailed {
                    operation: "read state query policy",
                });
            }
        };
        let rows = self
            .connection
            .query_with_params(query, &parameters)
            .map_err(|_| StorageError::DatabaseOperationFailed {
                operation: "read stored state",
            })?;
        match rows.as_slice() {
            [] => Ok(None),
            [row] => match row.get(0) {
                Some(fsqlite::SqliteValue::Text(value)) => {
                    JobState::from_database_value(value.as_str())
                        .map(Some)
                        .ok_or(StorageError::InvalidStoredState)
                }
                _ => Err(StorageError::InvalidStoredState),
            },
            _ => Err(StorageError::DatabaseOperationFailed {
                operation: "read unique stored state",
            }),
        }
    }
}

#[cfg(feature = "metadata-store")]
fn sqlite_integer(value: u64) -> Result<fsqlite::SqliteValue, StorageError> {
    i64::try_from(value)
        .map(fsqlite::SqliteValue::Integer)
        .map_err(|_| StorageError::IntegerOutOfRange)
}

#[cfg(all(feature = "metadata-store", unix))]
fn prepare_owner_only_database(database_path: &Path) -> Result<(), StorageError> {
    use std::fs::{self, OpenOptions};
    use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

    if database_path.as_os_str().is_empty() {
        return Err(StorageError::InvalidDatabasePath);
    }
    let parent = database_path
        .parent()
        .ok_or(StorageError::InvalidDatabasePath)?;
    if parent.as_os_str().is_empty() || fs::symlink_metadata(parent).is_err() {
        return Err(StorageError::PlatformSurfaceUnavailable);
    }
    if fs::symlink_metadata(parent)
        .map_err(|_| StorageError::PlatformSurfaceUnavailable)?
        .file_type()
        .is_symlink()
    {
        return Err(StorageError::OwnerOnlyPermissionRequired);
    }

    match fs::symlink_metadata(database_path) {
        Ok(metadata) => {
            if !metadata.file_type().is_file()
                || metadata.file_type().is_symlink()
                || metadata.permissions().mode() & 0o077 != 0
            {
                return Err(StorageError::OwnerOnlyPermissionRequired);
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(OWNER_ONLY_DATABASE_MODE)
                .open(database_path)
                .map_err(|_| StorageError::OwnerOnlyPermissionRequired)?;
            file.set_permissions(fs::Permissions::from_mode(OWNER_ONLY_DATABASE_MODE))
                .map_err(|_| StorageError::OwnerOnlyPermissionRequired)?;
        }
        Err(_) => return Err(StorageError::OwnerOnlyPermissionRequired),
    }

    let metadata = fs::symlink_metadata(database_path)
        .map_err(|_| StorageError::OwnerOnlyPermissionRequired)?;
    if !metadata.file_type().is_file()
        || metadata.file_type().is_symlink()
        || metadata.permissions().mode() & 0o077 != 0
    {
        return Err(StorageError::OwnerOnlyPermissionRequired);
    }
    Ok(())
}

#[cfg(all(feature = "metadata-store", not(unix)))]
fn prepare_owner_only_database(_database_path: &Path) -> Result<(), StorageError> {
    Err(StorageError::PlatformSurfaceUnavailable)
}
