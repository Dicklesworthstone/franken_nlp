//! Versioned, data-only robot-mode NDJSON surface.
//!
//! This is intentionally a typed emission skeleton. Parsing external NDJSON
//! remains blocked on the duplicate-key-rejecting canonical JSON boundary; no
//! robot request parser is allowed to silently use serde_json's last-key-wins
//! behavior.

use std::io::{self, Write};

use serde::Serialize;

use crate::{native_engine::dispatch, orchestrator};

/// Schema v2 adds the additive `convert_stage` event used by `fnlp convert
/// --robot`. Existing event names and fields retain their v1 spelling.
pub const ROBOT_SCHEMA_VERSION: u32 = 2;

/// A stable robot event name. Do not rename an event without a schema-version
/// bump and a reviewed golden update.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RobotEventType {
    RunStart,
    Stage,
    Doc,
    DocError,
    Token,
    Flush,
    RunComplete,
    RunError,
}

impl RobotEventType {
    pub const ALL: [Self; 8] = [
        Self::RunStart,
        Self::Stage,
        Self::Doc,
        Self::DocError,
        Self::Token,
        Self::Flush,
        Self::RunComplete,
        Self::RunError,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RunStart => "run_start",
            Self::Stage => "stage",
            Self::Doc => "doc",
            Self::DocError => "doc_error",
            Self::Token => "token",
            Self::Flush => "flush",
            Self::RunComplete => "run_complete",
            Self::RunError => "run_error",
        }
    }

    pub const fn requires_request_seq(self) -> bool {
        matches!(
            self,
            Self::Doc | Self::DocError | Self::Token | Self::RunComplete
        )
    }
}

/// Location information for errors caused by a line of robot NDJSON input.
/// Raw input bytes never enter a data event.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InputLocation {
    input_line: u64,
    byte_offset: Option<u64>,
    json_path: Option<String>,
}

impl InputLocation {
    pub fn new(input_line: u64, byte_offset: Option<u64>, json_path: Option<String>) -> Self {
        Self {
            input_line,
            byte_offset,
            json_path,
        }
    }
}

/// One NDJSON envelope. The fixed Rust field order is the emission order until
/// franken_nlp-gxn supplies the canonical JSON writer that owns this boundary.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RobotEvent {
    schema_version: u32,
    event: RobotEventType,
    #[serde(skip_serializing_if = "Option::is_none")]
    caller_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    request_seq: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    status: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_line: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    byte_offset: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    json_path: Option<String>,
}

impl RobotEvent {
    pub fn new(event: RobotEventType) -> Self {
        Self {
            schema_version: ROBOT_SCHEMA_VERSION,
            event,
            caller_id: None,
            request_seq: None,
            status: None,
            input_line: None,
            byte_offset: None,
            json_path: None,
        }
    }

    pub fn with_caller_id(mut self, caller_id: impl Into<String>) -> Self {
        self.caller_id = Some(caller_id.into());
        self
    }

    pub fn with_request_seq(mut self, request_seq: u64) -> Self {
        self.request_seq = Some(request_seq);
        self
    }

    pub fn with_status(mut self, status: impl Into<String>) -> Self {
        self.status = Some(status.into());
        self
    }

    pub fn with_input_location(mut self, location: InputLocation) -> Self {
        self.input_line = Some(location.input_line);
        self.byte_offset = location.byte_offset;
        self.json_path = location.json_path;
        self
    }

    /// A complete, non-sensitive representative event for contract tests.
    pub fn skeleton(event: RobotEventType) -> Self {
        let mut result = Self::new(event).with_caller_id("robot-contract-caller");
        if event.requires_request_seq() {
            result = result.with_request_seq(1);
        }
        if event == RobotEventType::DocError {
            result = result.with_input_location(InputLocation::new(
                1,
                Some(0),
                Some("/synthetic".to_owned()),
            ));
        }
        result
    }

    pub fn validate(&self) -> Result<(), RobotContractError> {
        if self.schema_version != ROBOT_SCHEMA_VERSION {
            return Err(RobotContractError::new(
                "/schema_version",
                "unexpected schema version",
            ));
        }
        if self
            .caller_id
            .as_deref()
            .is_some_and(|caller_id| caller_id.is_empty())
        {
            return Err(RobotContractError::new(
                "/caller_id",
                "must not be empty when supplied",
            ));
        }
        if self.event.requires_request_seq() && self.request_seq.is_none() {
            return Err(RobotContractError::new(
                "/request_seq",
                "is required for this per-request event",
            ));
        }
        if self.event == RobotEventType::DocError && self.input_line.is_none() {
            return Err(RobotContractError::new(
                "/input_line",
                "is required for doc_error",
            ));
        }
        if self.input_line == Some(0) {
            return Err(RobotContractError::new("/input_line", "must be one-based"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RobotContractError {
    path: &'static str,
    detail: &'static str,
}

impl RobotContractError {
    const fn new(path: &'static str, detail: &'static str) -> Self {
        Self { path, detail }
    }

    pub const fn path(&self) -> &'static str {
        self.path
    }
}

impl std::fmt::Display for RobotContractError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "robot contract violation at {}: {}",
            self.path, self.detail
        )
    }
}

impl std::error::Error for RobotContractError {}

/// Serialize exactly one data-only NDJSON event. Diagnostics are deliberately
/// the caller's responsibility and must be written to stderr, never here.
pub fn write_event<W: Write>(writer: &mut W, event: &RobotEvent) -> io::Result<()> {
    event
        .validate()
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
    serde_json::to_writer(&mut *writer, event).map_err(io::Error::other)?;
    writer.write_all(b"\n")
}

/// One versioned, identity-carrying stage event from `fnlp convert --robot`.
///
/// The human conversion transcript remains on stderr. This event carries the
/// same machine-relevant fields on stdout without requiring a consumer to
/// parse diagnostic prose.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RobotConvertStageEvent {
    schema_version: u32,
    event: &'static str,
    command: &'static str,
    stage: String,
    result: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_manifest: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    destination: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    staging_artifact: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_root_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    census_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fnlpq_file_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    license_bundle_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tensors: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sections: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    staging_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<String>,
}

impl RobotConvertStageEvent {
    /// Begin a typed conversion-stage record. `stage` and `result` use the
    /// same values as the human `CONVERT STAGE=... RESULT=...` transcript.
    pub fn new(stage: impl Into<String>, result: impl Into<String>) -> Self {
        Self {
            schema_version: ROBOT_SCHEMA_VERSION,
            event: "convert_stage",
            command: "convert",
            stage: stage.into(),
            result: result.into(),
            source: None,
            source_manifest: None,
            destination: None,
            staging_artifact: None,
            source_root_sha256: None,
            census_sha256: None,
            fnlpq_file_sha256: None,
            license_bundle_sha256: None,
            tensors: None,
            sections: None,
            staging_bytes: None,
            reason: None,
        }
    }

    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    pub fn with_source_manifest(mut self, source_manifest: impl Into<String>) -> Self {
        self.source_manifest = Some(source_manifest.into());
        self
    }

    pub fn with_destination(mut self, destination: impl Into<String>) -> Self {
        self.destination = Some(destination.into());
        self
    }

    pub fn with_staging_artifact(mut self, staging_artifact: impl Into<String>) -> Self {
        self.staging_artifact = Some(staging_artifact.into());
        self
    }

    pub fn with_source_root_sha256(mut self, source_root_sha256: impl Into<String>) -> Self {
        self.source_root_sha256 = Some(source_root_sha256.into());
        self
    }

    pub fn with_census_sha256(mut self, census_sha256: impl Into<String>) -> Self {
        self.census_sha256 = Some(census_sha256.into());
        self
    }

    pub fn with_fnlpq_file_sha256(mut self, fnlpq_file_sha256: impl Into<String>) -> Self {
        self.fnlpq_file_sha256 = Some(fnlpq_file_sha256.into());
        self
    }

    pub fn with_license_bundle_sha256(mut self, license_bundle_sha256: impl Into<String>) -> Self {
        self.license_bundle_sha256 = Some(license_bundle_sha256.into());
        self
    }

    pub const fn with_tensors(mut self, tensors: u64) -> Self {
        self.tensors = Some(tensors);
        self
    }

    pub const fn with_sections(mut self, sections: u64) -> Self {
        self.sections = Some(sections);
        self
    }

    pub const fn with_staging_bytes(mut self, staging_bytes: u64) -> Self {
        self.staging_bytes = Some(staging_bytes);
        self
    }

    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }

    pub fn validate(&self) -> Result<(), RobotContractError> {
        for (path, value) in [
            ("/stage", self.stage.as_str()),
            ("/result", self.result.as_str()),
        ] {
            if value.is_empty() || !value.is_ascii() {
                return Err(RobotContractError::new(path, "must be non-empty ASCII"));
            }
        }
        for (path, value) in [
            ("/source_root_sha256", self.source_root_sha256.as_deref()),
            ("/census_sha256", self.census_sha256.as_deref()),
            ("/fnlpq_file_sha256", self.fnlpq_file_sha256.as_deref()),
            (
                "/license_bundle_sha256",
                self.license_bundle_sha256.as_deref(),
            ),
        ] {
            if value.is_some_and(|value| !is_lower_sha256(value)) {
                return Err(RobotContractError::new(path, "must be lowercase SHA-256"));
            }
        }
        Ok(())
    }
}

/// Serialize exactly one typed conversion stage event to the robot data
/// channel. Human diagnostics remain the caller's stderr responsibility.
pub fn write_convert_stage_event<W: Write>(
    writer: &mut W,
    event: &RobotConvertStageEvent,
) -> io::Result<()> {
    event
        .validate()
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
    serde_json::to_writer(&mut *writer, event).map_err(io::Error::other)?;
    writer.write_all(b"\n")
}

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RobotCommand {
    Schema,
    Health,
    Backends,
}

/// The error-map bead owns the executable table. This frozen representation
/// fixes the machine contract now, so that bead must make its table match this
/// schema rather than adding a second public vocabulary.
const ROBOT_SCHEMA_JSON: &str = r#"{"$id":"https://franken-nlp.dev/schema/robot/v2","$schema":"https://json-schema.org/draft/2020-12/schema","additionalProperties":false,"allOf":[{"if":{"properties":{"event":{"const":"doc_error"}}},"then":{"required":["input_line","request_seq"]}},{"if":{"properties":{"event":{"enum":["doc","token","run_complete"]}}},"then":{"required":["request_seq"]}},{"if":{"properties":{"event":{"const":"convert_stage"}}},"then":{"required":["command","result","stage"]}}],"properties":{"byte_offset":{"minimum":0,"type":"integer"},"caller_id":{"minLength":1,"type":"string"},"census_sha256":{"pattern":"^[0-9a-f]{64}$","type":"string"},"command":{"const":"convert","type":"string"},"destination":{"type":"string"},"event":{"enum":["run_start","stage","doc","doc_error","token","flush","run_complete","run_error","convert_stage"],"type":"string"},"fnlpq_file_sha256":{"pattern":"^[0-9a-f]{64}$","type":"string"},"input_line":{"minimum":1,"type":"integer"},"json_path":{"type":"string"},"license_bundle_sha256":{"pattern":"^[0-9a-f]{64}$","type":"string"},"reason":{"type":"string"},"request_seq":{"minimum":0,"type":"integer"},"result":{"minLength":1,"type":"string"},"schema_version":{"const":2,"type":"integer"},"sections":{"minimum":0,"type":"integer"},"source":{"type":"string"},"source_manifest":{"type":"string"},"source_root_sha256":{"pattern":"^[0-9a-f]{64}$","type":"string"},"stage":{"minLength":1,"type":"string"},"staging_artifact":{"type":"string"},"staging_bytes":{"minimum":0,"type":"integer"},"status":{"type":"string"},"tensors":{"minimum":0,"type":"integer"}},"required":["event","schema_version"],"title":"franken_nlp robot NDJSON v2","type":"object","x_fnlp_robot":{"commands":{"backends":{"fields":["architecture","backends","kind","schema_version","status"],"kind":"robot_backends","status":"populated"},"convert":{"fields":["command","stage","result","source","source_manifest","destination","staging_artifact","source_root_sha256","census_sha256","tensors","sections","fnlpq_file_sha256","staging_bytes","license_bundle_sha256","reason"],"kind":"robot_convert_stage","status":"partial"},"health":{"fields":["capabilities","kind","schema_version","thread_inventory"],"kind":"robot_health","status":"conditional"},"schema":{"kind":"robot_schema"}},"exit_code_authority":"src/error.rs","exit_codes":[{"code":0,"name":"ok"},{"code":1,"name":"generic"},{"code":2,"name":"usage"},{"code":3,"name":"model_not_found"},{"code":4,"name":"input_decode_or_parse"},{"code":5,"name":"budget_or_timeout"},{"code":6,"name":"cancelled"},{"code":7,"name":"artifact_integrity_or_format_or_version"},{"code":8,"name":"schema_or_recipe_compile"},{"code":9,"name":"admission_or_resource_limit"},{"code":10,"name":"structured_task_no_result"}],"request_seq_events":["doc","doc_error","token","run_complete"],"stderr":"diagnostics_only","stdout":"data_only","volatile_fields":[]}}"#;

pub fn schema_json_bytes() -> Vec<u8> {
    let mut result = ROBOT_SCHEMA_JSON.as_bytes().to_vec();
    result.push(b'\n');
    result
}

#[derive(Serialize)]
struct Unpopulated {
    status: &'static str,
}

#[derive(Serialize)]
struct ThreadInventoryDocument {
    status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    runtime_preset: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    runtime_workers: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    blocking_coordinators: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    scoped_cpu_children_per_coordinator: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    helper_threads: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    total_runnable_threads: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thread_ceiling: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    runtime_binding: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    active_engine_leases: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    outstanding_pool_closures: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cancelled_wrapper_closures: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    deadline_check_interval_millis: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    checkpoint_timeout_millis: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cancel_attribution_max_depth: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cancel_attribution_max_memory_bytes: Option<usize>,
}

impl ThreadInventoryDocument {
    fn not_installed() -> Self {
        Self {
            status: "not_installed",
            runtime_preset: None,
            runtime_workers: None,
            blocking_coordinators: None,
            scoped_cpu_children_per_coordinator: None,
            helper_threads: None,
            total_runnable_threads: None,
            thread_ceiling: None,
            runtime_binding: None,
            active_engine_leases: None,
            outstanding_pool_closures: None,
            cancelled_wrapper_closures: None,
            deadline_check_interval_millis: None,
            checkpoint_timeout_millis: None,
            cancel_attribution_max_depth: None,
            cancel_attribution_max_memory_bytes: None,
        }
    }

    fn from_resources(resources: &orchestrator::EngineResources) -> Self {
        let config = resources.config();
        let inventory = resources.thread_inventory();
        let outstanding = resources.outstanding_closure_snapshot();
        let guardrails = resources.runtime_guardrails();
        Self {
            status: "configured",
            runtime_preset: Some(config.runtime_preset.as_str()),
            runtime_workers: Some(inventory.runtime_workers),
            blocking_coordinators: Some(inventory.blocking_coordinators),
            scoped_cpu_children_per_coordinator: Some(
                inventory.scoped_cpu_children_per_coordinator,
            ),
            helper_threads: Some(inventory.helper_threads),
            total_runnable_threads: Some(inventory.total_runnable_threads),
            thread_ceiling: Some(config.thread_ceiling),
            runtime_binding: Some(if resources.has_real_blocking_pool() {
                "real_blocking_pool"
            } else {
                "not_compiled"
            }),
            active_engine_leases: Some(resources.active_lease_count()),
            outstanding_pool_closures: Some(outstanding.active_closures),
            cancelled_wrapper_closures: Some(outstanding.wrapper_cancelled_closures),
            deadline_check_interval_millis: Some(guardrails.deadline_check_interval_millis),
            checkpoint_timeout_millis: Some(guardrails.checkpoint_timeout_millis),
            cancel_attribution_max_depth: Some(guardrails.cancel_attribution_max_depth),
            cancel_attribution_max_memory_bytes: Some(
                guardrails.cancel_attribution_max_memory_bytes,
            ),
        }
    }
}

#[derive(Serialize)]
struct HealthDocument {
    capabilities: Unpopulated,
    kind: &'static str,
    schema_version: u32,
    thread_inventory: ThreadInventoryDocument,
}

#[derive(Serialize)]
struct BackendsDocument {
    backends: dispatch::BackendReport,
    kind: &'static str,
    schema_version: u32,
    status: &'static str,
}

fn write_json_document<W: Write, T: Serialize>(writer: &mut W, document: &T) -> io::Result<()> {
    serde_json::to_writer(&mut *writer, document).map_err(io::Error::other)?;
    writer.write_all(b"\n")
}

/// Implement the non-engine robot subcommands. These bytes are data-only and
/// intentionally unaffected by NO_COLOR, CI, or TERM=dumb.
pub fn write_command<W: Write, D: Write>(
    writer: &mut W,
    _diagnostics: &mut D,
    command: RobotCommand,
) -> io::Result<()> {
    match command {
        RobotCommand::Schema => writer.write_all(&schema_json_bytes()),
        RobotCommand::Health => write_json_document(
            writer,
            &HealthDocument {
                capabilities: Unpopulated {
                    status: "unpopulated",
                },
                kind: "robot_health",
                schema_version: ROBOT_SCHEMA_VERSION,
                thread_inventory: orchestrator::installed_process_resources()
                    .as_deref()
                    .map_or_else(ThreadInventoryDocument::not_installed, ThreadInventoryDocument::from_resources),
            },
        ),
        RobotCommand::Backends => write_json_document(
            writer,
            &BackendsDocument {
                backends: dispatch::host_backend_report(),
                kind: "robot_backends",
                schema_version: ROBOT_SCHEMA_VERSION,
                status: "populated",
            },
        ),
    }
}
