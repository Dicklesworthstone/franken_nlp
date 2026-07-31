//! Versioned, data-only robot-mode NDJSON surface.
//!
//! This is intentionally a typed emission skeleton. Parsing external NDJSON
//! remains blocked on the duplicate-key-rejecting canonical JSON boundary; no
//! robot request parser is allowed to silently use serde_json's last-key-wins
//! behavior.

use std::io::{self, Write};

use serde::Serialize;

use crate::native_engine::dispatch;

pub const ROBOT_SCHEMA_VERSION: u32 = 1;

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RobotCommand {
    Schema,
    Health,
    Backends,
}

/// The error-map bead owns the executable table. This frozen representation
/// fixes the machine contract now, so that bead must make its table match this
/// schema rather than adding a second public vocabulary.
const ROBOT_SCHEMA_JSON: &str = r#"{"$id":"https://franken-nlp.dev/schema/robot/v1","$schema":"https://json-schema.org/draft/2020-12/schema","additionalProperties":false,"allOf":[{"if":{"properties":{"event":{"const":"doc_error"}}},"then":{"required":["input_line","request_seq"]}},{"if":{"properties":{"event":{"enum":["doc","token","run_complete"]}}},"then":{"required":["request_seq"]}}],"properties":{"byte_offset":{"minimum":0,"type":"integer"},"caller_id":{"minLength":1,"type":"string"},"event":{"enum":["run_start","stage","doc","doc_error","token","flush","run_complete","run_error"],"type":"string"},"input_line":{"minimum":1,"type":"integer"},"json_path":{"type":"string"},"request_seq":{"minimum":0,"type":"integer"},"schema_version":{"const":1,"type":"integer"},"status":{"type":"string"}},"required":["event","schema_version"],"title":"franken_nlp robot NDJSON v1","type":"object","x_fnlp_robot":{"commands":{"backends":{"fields":["architecture","backends","kind","schema_version","status"],"kind":"robot_backends","status":"populated"},"health":{"kind":"robot_health","status":"unpopulated"},"schema":{"kind":"robot_schema"}},"exit_code_authority":"src/error.rs","exit_codes":[{"code":0,"name":"ok"},{"code":1,"name":"generic"},{"code":2,"name":"usage"},{"code":3,"name":"model_not_found"},{"code":4,"name":"input_decode_or_parse"},{"code":5,"name":"budget_or_timeout"},{"code":6,"name":"cancelled"},{"code":7,"name":"artifact_integrity_or_format_or_version"},{"code":8,"name":"schema_or_recipe_compile"},{"code":9,"name":"admission_or_resource_limit"},{"code":10,"name":"structured_task_no_result"}],"request_seq_events":["doc","doc_error","token","run_complete"],"stderr":"diagnostics_only","stdout":"data_only","volatile_fields":[]}}"#;

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
struct HealthDocument {
    capabilities: Unpopulated,
    kind: &'static str,
    schema_version: u32,
    thread_inventory: Unpopulated,
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
                thread_inventory: Unpopulated {
                    status: "unpopulated",
                },
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
