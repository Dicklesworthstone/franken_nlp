//! Exact schema defaults for the existing bounded extraction corpus runner.
use super::*;
use crate::{
    batch::extract::{ExtractionBatchArgs, ExtractionBatchGrounding},
    candidate_cli::extract as schema_cli,
    grammar::runtime::JsonProgram,
    validation::{JsonLimits, JsonValue, parse_json_with_limits},
};

impl BatchCommand {
    pub(super) fn validate_extraction_flags(&self) -> Result<(), CandidateError> {
        if self.task != "extract" && (self.schema.is_some() || self.source_membership) {
            return Err(CandidateError::Arguments);
        }
        if self.schema.is_some() && self.defaults.is_some() || self.source_membership && self.schema.is_none() {
            return Err(CandidateError::Arguments);
        }
        if let Some(path) = &self.schema { schema_cli::check_schema_path(path)?; }
        Ok(())
    }

    /// None means each record MUST supply its own complete task_args. A shared
    /// schema/default never becomes permission for an unbounded request or a
    /// different task, model, template, sampling mode or execution identity.
    pub(in crate::candidate_cli) fn load_extract_defaults(&self, schema: Option<String>, json: Option<&str>,
        ceiling: TaskBudget) -> Result<Option<ExtractionBatchArgs>, CandidateError> {
        self.validate_extraction_flags()?;
        if self.task != "extract" || schema.is_some() != self.schema.is_some()
            || json.is_some() != self.defaults.is_some() { return Err(CandidateError::Arguments); }
        let defaults = match (schema, json) {
            (Some(schema), None) => Some(schema_cli::arguments(schema, self.source_membership, ceiling)?),
            (None, Some(json)) => {
                if json.len() > MAX_SOURCE_ARGUMENT_BYTES { return Err(CandidateError::Input); }
                // Only the OUTER configuration uses Value; its schema is an
                // exact string. Schema numeric literals never pass through f64.
                let value = canonjson::parse_str_with_limits(json, canonjson::ParseLimits {
                    max_depth: 8, max_string_bytes: MAX_SOURCE_ARGUMENT_BYTES,
                }).map_err(|_| CandidateError::Input)?;
                Some(serde_json::from_value::<ExtractionBatchArgs>(value).map_err(|_| CandidateError::Input)?)
            }
            (None, None) => None,
            _ => return Err(CandidateError::Arguments),
        };
        if let Some(defaults) = &defaults { check_defaults(defaults, ceiling, &self.host)?; }
        Ok(defaults)
    }
}

fn check_defaults(args: &ExtractionBatchArgs, ceiling: TaskBudget, host: &SourceHostArgs)
    -> Result<(), CandidateError> {
    let b = args.budget;
    b.validate().map_err(|_| CandidateError::Planning)?;
    if b.max_input_tokens > ceiling.max_input_tokens || b.max_output_tokens > ceiling.max_output_tokens
        || b.max_output_bytes > ceiling.max_output_bytes || b.max_grammar_states > ceiling.max_grammar_states
        || b.max_kv_bytes != ceiling.max_kv_bytes { return Err(CandidateError::Planning); }
    if args.schema.is_empty() || args.schema.len() > schema_cli::SCHEMA_BYTES { return Err(CandidateError::Input); }
    // Exact duplicate-rejecting decimal parser, including for source schemas.
    // Source-bound compilation depends on EACH actual document. Do not invent
    // a placeholder source or claim syntax inspection proved its membership.
    let declaration = parse_json_with_limits(&args.schema, JsonLimits {
        max_input_bytes: schema_cli::SCHEMA_BYTES,
        max_string_lexeme_bytes: schema_cli::SCHEMA_BYTES,
        ..JsonLimits::default()
    }).map_err(|_| CandidateError::Planning)?;
    if !matches!(declaration, JsonValue::Object(_)) { return Err(CandidateError::Planning); }
    if args.grounding == ExtractionBatchGrounding::Structural {
        JsonProgram::compile(&args.schema, schema_cli::compiler(host)).map_err(|_| CandidateError::Planning)?;
    }
    // The native per-item compiler subsequently enforces every keyword,
    // grounding requirement and task ceiling for BOTH defaults and overrides.
    Ok(())
}

#[cfg(test)] mod tests;
