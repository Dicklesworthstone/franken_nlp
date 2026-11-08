//! Explicit whole-document record reduction; map inference budgets stay fixed.
use super::*;
use crate::tasks::extract::array_union::{RootArrayUnionLimits, ROOT_ARRAY_UNION_POLICY};

pub(super) const HELP: &str = "Root-array reduction: --root-array-union requires an array at the schema root. It adds a combined exact record array to the retained chunk results. Records use source order; equal canonical values share one record while all contributing chunk/element origins remain available. Decimal equality is exact, including large integers. The original schema validates the combined array, so its maxItems still applies. Per-chunk schema constraints also remain in force. All --max-union-* caps are for the complete document; exceeding any cap fails the document without truncation. The final chunk-plus-union envelope must fit --max-map-result-bytes. Defaults: 4096 distinct records, 16384 origins, 262144 JSON nodes, input bytes from --max-total-value-bytes, and JSON/envelope bytes from --max-map-result-bytes. These limits add a modeled memory reservation before input or model IO. On extract-document-job resume, repeat the same union policy and limits: the immutable recipe authenticates them. This deterministic reduction adds no model work or semantic verification.";

/// Public policy metadata contains no schema, extracted value or source path.
#[derive(Serialize)]
pub(in crate::candidate_cli::int4) struct ArrayUnionSelection {
    policy: &'static str,
    limits: RootArrayUnionLimits,
}
impl ArrayUnionSelection {
    pub(in crate::candidate_cli::int4) fn new(limits: RootArrayUnionLimits) -> Self {
        Self { policy: ROOT_ARRAY_UNION_POLICY, limits }
    }
}

#[derive(Args, Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct ArrayUnionArgs {
    /// Combine root-array records by exact canonical value, retaining every origin.
    #[arg(long)]
    root_array_union: bool,
    /// Maximum distinct records in the final array; original schema limits also apply.
    #[arg(long, requires = "root_array_union")]
    max_union_records: Option<usize>,
    /// Maximum original array elements, including every duplicate occurrence.
    #[arg(long, requires = "root_array_union")]
    max_union_origins: Option<usize>,
    /// Sum of exact chunk JSON bytes inspected by the reducer.
    #[arg(long, requires = "root_array_union")]
    max_union_input_bytes: Option<usize>,
    /// Maximum canonical final array bytes before result-envelope serialization.
    #[arg(long, requires = "root_array_union")]
    max_union_json_bytes: Option<usize>,
    /// JSON value nodes across all input arrays and the final validation parse.
    #[arg(long, requires = "root_array_union")]
    max_union_nodes: Option<usize>,
    /// Maximum aggregate envelope bytes; the whole document cap still applies.
    #[arg(long, requires = "root_array_union")]
    max_union_result_bytes: Option<usize>,
}
impl ArrayUnionArgs {
    pub(super) fn limits(self, mapping: MappingArgs) -> Result<Option<RootArrayUnionLimits>, CandidateError> {
        let overrides = [self.max_union_records, self.max_union_origins, self.max_union_input_bytes,
            self.max_union_json_bytes, self.max_union_nodes, self.max_union_result_bytes];
        if !self.root_array_union {
            if overrides.iter().any(Option::is_some) { return Err(CandidateError::Arguments); }
            return Ok(None);
        }
        let limits = RootArrayUnionLimits {
            max_records: self.max_union_records.unwrap_or(4096),
            max_origins: self.max_union_origins.unwrap_or(16_384),
            max_input_bytes: self.max_union_input_bytes.unwrap_or(mapping.max_total_value_bytes),
            max_json_bytes: self.max_union_json_bytes.unwrap_or(mapping.max_map_result_bytes),
            max_nodes: self.max_union_nodes.unwrap_or(262_144),
            max_result_bytes: self.max_union_result_bytes.unwrap_or(mapping.max_map_result_bytes),
        };
        limits.requirements().map_err(|_| CandidateError::Arguments)?;
        if limits.max_json_bytes > mapping.max_map_result_bytes
            || limits.max_result_bytes > mapping.max_map_result_bytes {
            return Err(CandidateError::Arguments);
        }
        Ok(Some(limits))
    }
    pub(super) fn check_schema(self, schema: &str, compiler: crate::grammar::CompileLimits, mapping: MappingArgs)
        -> Result<(), CandidateError> {
        if let Some(limits) = self.limits(mapping)? {
            let declaration = crate::grammar::runtime::JsonValidationSchema::compile(schema, compiler)
                .map_err(|_| CandidateError::Planning)?;
            limits.validate_schema(declaration.root()).map_err(|_| CandidateError::Planning)?;
        }
        Ok(())
    }
}
