//! Job limits are immutable lifetime authority, not multiplied batch defaults.
use super::*;
use crate::{
    batch::{source::SourceBatchArgs, extract::{ExtractionBatchArgs, ExtractionBatchGrounding, ExtractionMaskBudget,
        quantized::Int8ExtractionBatchLimits}},
    candidate_cli::extract as schema_cli,
    grammar::runtime::JsonProgram,
    tasks::{ir::TaskBudget, ner::NerOptions, keyphrases::KeyphraseOptions, summarize::SummaryOptions},
    validation::{JsonLimits, JsonValue, parse_json_with_limits},
};

pub(in crate::candidate_cli) enum Defaults {
    Source(Option<SourceBatchArgs>),
    Extract(Option<ExtractionBatchArgs>),
}
pub(in crate::candidate_cli) fn parse_limits(text: &str) -> Result<JobLimits, Failure> {
    if text.len() > LIMIT_BYTES { return Err(Failure::usage("job_limits")); }
    let value = canonjson::parse_str_with_limits(text, canonjson::ParseLimits {
        max_depth: 8, max_string_bytes: 4096,
    }).map_err(|_| Failure::usage("job_limits"))?;
    let limits: JobLimits = serde_json::from_value(value).map_err(|_| Failure::usage("job_limits"))?;
    limits.validate().map_err(|_| Failure::usage("job_limits"))?;
    Ok(limits)
}
pub(in crate::candidate_cli) fn native_limits(args: &JobArgs, limits: JobLimits) -> Result<Int8ExtractionBatchLimits, Failure> {
    if args.host.max_result_bytes > limits.max_result_bytes { return Err(Failure::usage("stored_result_ceiling")); }
    let native = Int8ExtractionBatchLimits { max_model_work: limits.max_work.model, masks: ExtractionMaskBudget {
        per_mask: args.host.masks(), max_visits_per_item: args.host.max_mask_node_visits,
        max_visits_per_run: limits.max_work.mask_node_visits,
    } };
    native.validate().map_err(|_| Failure::usage("lifetime_native_or_mask_limits"))?;
    Ok(native)
}
fn json(text: &str) -> Result<serde_json::Value, CandidateError> {
    if text.len() > DEFAULT_BYTES { return Err(CandidateError::Input); }
    canonjson::parse_str_with_limits(text, canonjson::ParseLimits {
        max_depth: 16, max_string_bytes: DEFAULT_BYTES,
    }).map_err(|_| CandidateError::Input)
}
fn budget_fits(b: TaskBudget, ceiling: TaskBudget) -> Result<(), CandidateError> {
    b.validate().map_err(|_| CandidateError::Planning)?;
    if b.max_input_tokens > ceiling.max_input_tokens || b.max_output_tokens > ceiling.max_output_tokens
        || b.max_output_bytes > ceiling.max_output_bytes || b.max_grammar_states > ceiling.max_grammar_states
        || b.max_kv_bytes != ceiling.max_kv_bytes { return Err(CandidateError::Planning); }
    Ok(())
}
pub(in crate::candidate_cli) fn load_defaults(args: &JobArgs, ceiling: TaskBudget, text: Option<&str>, schema: Option<String>)
    -> Result<Defaults, CandidateError> {
    if text.is_some() != args.defaults.is_some() || schema.is_some() != args.schema.is_some() {
        return Err(CandidateError::Arguments);
    }
    if args.task == "extract" {
        let defaults = match (text, schema) {
            (Some(text), None) => Some(serde_json::from_value::<ExtractionBatchArgs>(json(text)?).map_err(|_| CandidateError::Input)?),
            (None, Some(schema)) => Some(schema_cli::arguments(schema, args.source_membership, ceiling)?),
            (None, None) => None,
            _ => return Err(CandidateError::Arguments),
        };
        if let Some(defaults) = &defaults {
            budget_fits(defaults.budget, ceiling)?;
            if defaults.schema.is_empty() || defaults.schema.len() > schema_cli::SCHEMA_BYTES { return Err(CandidateError::Input); }
            let declaration = parse_json_with_limits(&defaults.schema, JsonLimits {
                max_input_bytes: schema_cli::SCHEMA_BYTES, max_string_lexeme_bytes: schema_cli::SCHEMA_BYTES,
                ..JsonLimits::default()
            }).map_err(|_| CandidateError::Planning)?;
            if !matches!(declaration, JsonValue::Object(_)) { return Err(CandidateError::Planning); }
            if defaults.grounding == ExtractionBatchGrounding::Structural {
                JsonProgram::compile(&defaults.schema, schema_cli::compiler(&args.host)).map_err(|_| CandidateError::Planning)?;
            }
            // Verbatim defaults compile with EACH original source, never a
            // fabricated example document. Syntax inspection is not grounding.
        }
        return Ok(Defaults::Extract(defaults));
    }
    if schema.is_some() || args.source_membership { return Err(CandidateError::Arguments); }
    let defaults = match text {
        Some(text) => Some(serde_json::from_value::<SourceBatchArgs>(json(text)?).map_err(|_| CandidateError::Input)?),
        None => match args.task.as_str() {
            "ner" => Some(SourceBatchArgs::Ner { options: NerOptions::default(), budget: ceiling }),
            "keyphrases" => Some(SourceBatchArgs::Keyphrases { options: KeyphraseOptions::default(), budget: ceiling }),
            "summarize" => Some(SourceBatchArgs::Summarize { options: SummaryOptions::default(), budget: ceiling }),
            "answer" => None,
            _ => return Err(CandidateError::Arguments),
        },
    };
    if let Some(defaults) = &defaults {
        let (task, budget) = match defaults {
            SourceBatchArgs::Ner { options, budget } => { options.schema_source().map_err(|_| CandidateError::Planning)?; ("ner", budget) }
            SourceBatchArgs::Keyphrases { options, budget } => { options.schema_source().map_err(|_| CandidateError::Planning)?; ("keyphrases", budget) }
            SourceBatchArgs::Summarize { options, budget } => { options.schema_source().map_err(|_| CandidateError::Planning)?; ("summarize", budget) }
            SourceBatchArgs::Answer { passages, options, budget } => {
                options.schema_source().map_err(|_| CandidateError::Planning)?;
                if passages.is_empty() || passages.len() > source::MAX_PASSAGES { return Err(CandidateError::Input); }
                let mut ids = std::collections::BTreeSet::new(); let mut bytes = 0_usize;
                for p in passages {
                    if p.id.is_empty() || p.id.len() > 128 || p.id.chars().any(char::is_control)
                        || !ids.insert(p.id.as_str()) { return Err(CandidateError::Input); }
                    bytes = bytes.checked_add(p.id.len()).and_then(|n| n.checked_add(p.text.len()))
                        .filter(|&n| n <= args.host.max_input_bytes).ok_or(CandidateError::Input)?;
                }
                ("answer", budget)
            }
        };
        if task != args.task { return Err(CandidateError::Planning); }
        budget_fits(*budget, ceiling)?;
    }
    Ok(Defaults::Source(defaults))
}
