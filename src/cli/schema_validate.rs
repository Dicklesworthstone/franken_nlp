//! Model-free verification of externally supplied extraction results.
use std::{
    convert::Infallible,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::ExitCode,
};

use crate::{
    error::ErrorCode,
    grammar::{CompileLimits, compile_json_schema, runtime::SourceRuntimeLimits},
    local_io,
    textutil::{TextError, read_utf8},
    validation::{self, JsonLimits, JsonParseErrorKind, grounded_fields::{
        EvidenceMode, FieldGroundingError, GroundingBudget, GroundingScanError,
        verify_source_fields_with_checkpoints,
    }},
};

#[derive(clap::Args)]
pub(super) struct Validate {
    /// Schema in the supported bounded JSON-Schema subset.
    #[arg(value_name = "SCHEMA")]
    schema: PathBuf,
    /// UTF-8 JSON instance. Use '-' for stdin on at most one input.
    #[arg(value_name = "INSTANCE", default_value = "-")]
    instance: PathBuf,
    /// Exact original UTF-8 document, required for schemas declaring verbatim fields.
    #[arg(long, value_name = "DOCUMENT")]
    source: Option<PathBuf>,
}

#[derive(Debug, Eq, PartialEq)]
struct Failure(ErrorCode, &'static str);

fn read(path: &Path, input: &mut impl Read, cap: usize) -> Result<String, Failure> {
    let result = if path == Path::new("-") {
        read_utf8(input, cap)
    } else {
        let mut file = local_io::open_document(path)
            .map_err(|_| Failure(ErrorCode::InputDecodeOrParse, "regular_input_open_failed"))?;
        read_utf8(&mut file, cap)
    };
    result.map_err(|error| match error {
        TextError::InputBudget | TextError::AllocationRefused =>
            Failure(ErrorCode::AdmissionOrResourceLimit, "input_limit"),
        _ => Failure(ErrorCode::InputDecodeOrParse, "invalid_utf8_or_read_failed"),
    })
}

fn validate_instance(schema: &crate::grammar::CompiledSchema, instance: &str, source: Option<&str>)
    -> Result<(), Failure> {
    let value = validation::json::parse_json_with_node_limit(instance, JsonLimits::default(), 65_536)
        .map_err(|error| match error.kind() {
            JsonParseErrorKind::Resource => Failure(ErrorCode::AdmissionOrResourceLimit, "instance_limit"),
            _ => Failure(ErrorCode::InputDecodeOrParse, "invalid_instance_json"),
        })?;
    validation::validate_value(schema.root(), &value)
        .map_err(|_| Failure(ErrorCode::StructuredTaskNoResult, "schema_constraint"))?;
    if schema.requires_verbatim_source() {
        let source = source.ok_or(Failure(ErrorCode::Usage, "verbatim_source_required"))?;
        verify_source_fields_with_checkpoints(schema.root(), &value, source, GroundingBudget::default(),
            EvidenceMode::CheckOnly, || Ok::<(), Infallible>(()))
            .map_err(|err| match err {
                GroundingScanError::Checkpoint(never) => match never {},
                GroundingScanError::Verification(FieldGroundingError::Absent) =>
                    Failure(ErrorCode::StructuredTaskNoResult, "verbatim_value_absent"),
                GroundingScanError::Verification(FieldGroundingError::Shape) =>
                    Failure(ErrorCode::Generic, "validator_shape_disagreement"),
                GroundingScanError::Verification(_) =>
                    Failure(ErrorCode::AdmissionOrResourceLimit, "source_verification_limit"),
            })?;
    }
    Ok(())
}

impl Validate {
    fn check(&self, input: &mut impl Read) -> Result<bool, Failure> {
        let stdin_paths = [Some(self.schema.as_path()), Some(self.instance.as_path()), self.source.as_deref()]
            .into_iter().flatten().filter(|path| *path == Path::new("-")).count();
        if stdin_paths > 1 { return Err(Failure(ErrorCode::Usage, "inputs_share_stdin")); }

        let limits = CompileLimits::default();
        let text = read(&self.schema, input, limits.max_schema_bytes)?;
        let schema = compile_json_schema(&text, limits)
            .map_err(|_| Failure(ErrorCode::SchemaOrRecipeCompile, "schema_compile_failed"))?;
        if schema.requires_verbatim_source() && self.source.is_none() {
            return Err(Failure(ErrorCode::Usage, "verbatim_source_required"));
        }
        let instance = read(&self.instance, input, JsonLimits::default().max_input_bytes)?;
        let source = self.source.as_deref().map(|path|
            read(path, input, SourceRuntimeLimits::default().index.max_source_bytes)).transpose()?;
        validate_instance(&schema, &instance, source.as_deref())?;
        Ok(schema.requires_verbatim_source())
    }

    pub(super) fn run(&self, input: &mut impl Read, out: &mut impl Write, err: &mut impl Write) -> ExitCode {
        match self.check(input) {
            Ok(grounded) => {
                if writeln!(out, "{{\"schema_version\":1,\"valid\":true,\"verbatim_schema_checked\":{grounded}}}")
                    .and_then(|()| out.flush()).is_ok() {
                    ExitCode::SUCCESS
                } else { ErrorCode::Generic.as_process_exit() }
            }
            Err(Failure(code, reason)) => {
                let _ = writeln!(err, "SCHEMA mode=validate RESULT=FAIL reason={reason}");
                code.as_process_exit()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_nested_verbatim_values_against_original_unicode_with_repeated_occurrences() {
        let schema = compile_json_schema(
            r#"{"type":"array","maxItems":4,"items":{"type":"string","x-fnlp-source":"verbatim"}}"#,
            CompileLimits::default()).unwrap();
        let instance = r#"["\u00e9","fox"]"#;
        assert_eq!(validate_instance(&schema, instance, Some("é fox é")), Ok(()));
        assert_eq!(validate_instance(&schema, instance, Some("e fox")),
            Err(Failure(ErrorCode::StructuredTaskNoResult, "verbatim_value_absent")));
        assert_eq!(validate_instance(&schema, instance, None),
            Err(Failure(ErrorCode::Usage, "verbatim_source_required")));
    }

    #[test]
    fn exact_numbers_duplicate_keys_and_schema_constraints_use_independent_validation() {
        let schema = compile_json_schema(r#"{"type":"object","properties":{"n":{"type":"integer","const":9007199254740993}},"required":["n"],"additionalProperties":false}"#,
            CompileLimits::default()).unwrap();
        assert_eq!(validate_instance(&schema, r#"{"n":9007199254740993}"#, None), Ok(()));
        assert_eq!(validate_instance(&schema, r#"{"n":9007199254740992}"#, None),
            Err(Failure(ErrorCode::StructuredTaskNoResult, "schema_constraint")));
        assert_eq!(validate_instance(&schema, r#"{"n":1,"n":9007199254740993}"#, None),
            Err(Failure(ErrorCode::InputDecodeOrParse, "invalid_instance_json")));
    }

    #[test]
    fn conflicting_stdin_roles_emit_no_result_and_consume_no_input() {
        let command = Validate { schema: "-".into(), instance: "-".into(), source: None };
        let mut input = std::io::Cursor::new(b"untouched");
        let (mut out, mut err) = (Vec::new(), Vec::new());
        assert_eq!(command.run(&mut input, &mut out, &mut err), ErrorCode::Usage.as_process_exit());
        assert!(out.is_empty());
        assert_eq!(input.position(), 0);
        assert!(String::from_utf8(err).unwrap().contains("inputs_share_stdin"));
    }
}
