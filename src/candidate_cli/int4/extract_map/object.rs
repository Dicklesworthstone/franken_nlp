//! Explicit complete-document root-object extraction and exact reconciliation.
use super::*;
use crate::{batch::extract::{ExtractionBatchGrounding, grouped_int4::long::object::Int4ObjectExtractionMapLimits},
    native_engine::decode::DecodeStepControl,
    tasks::extract::object_reconcile::{RootObjectLimits, RootObjectSchema}};
#[cfg(feature = "asupersync-runtime")] mod runtime;
#[cfg(test)] mod tests;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(in crate::candidate_cli::int4) struct ObjectMapArgs { map: MapArgs, object: ObjectArgs }
#[derive(Args, Clone, Copy, Debug, Eq, PartialEq)]
struct ObjectArgs {
    #[arg(long, default_value_t = 4096)]
    max_object_properties: usize,
    #[arg(long, default_value_t = 16384)]
    max_object_origins: usize,
    /// Aggregate JSON bytes inspected across all partial chunk objects.
    #[arg(long)]
    max_object_input_bytes: Option<usize>,
    #[arg(long)]
    max_object_json_bytes: Option<usize>,
    #[arg(long, default_value_t = 262_144)]
    max_object_nodes: usize,
    /// Complete reconciled-object envelope; the map-plus-object cap still applies.
    #[arg(long)]
    max_object_result_bytes: Option<usize>,
}

/// Explicit opt-in for population routes. Every override requires the mode so
/// a typo or a resume invocation cannot silently discard the selected policy.
#[derive(Args, Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(in crate::candidate_cli::int4) struct ReconciliationArgs {
    /// Combine complementary root properties; reject conflicts and missing required fields.
    #[arg(long, conflicts_with = "root_array_union")]
    reconcile_root_object: bool,
    #[arg(long, requires = "reconcile_root_object")]
    max_object_properties: Option<usize>,
    #[arg(long, requires = "reconcile_root_object")]
    max_object_origins: Option<usize>,
    /// Aggregate JSON bytes inspected across all partial chunk objects.
    #[arg(long, requires = "reconcile_root_object")]
    max_object_input_bytes: Option<usize>,
    #[arg(long, requires = "reconcile_root_object")]
    max_object_json_bytes: Option<usize>,
    #[arg(long, requires = "reconcile_root_object")]
    max_object_nodes: Option<usize>,
    /// Reconciled-object envelope within the complete --max-map-result-bytes cap.
    #[arg(long, requires = "reconcile_root_object")]
    max_object_result_bytes: Option<usize>,
}
impl ReconciliationArgs {
    pub(in crate::candidate_cli::int4) fn augment_definition(command: clap::Command) -> clap::Command {
        Self::augment_args(command)
    }
    pub(in crate::candidate_cli::int4) fn limits(self, mapping: MappingArgs)
        -> Result<Option<RootObjectLimits>, CandidateError> {
        let overrides = [self.max_object_properties, self.max_object_origins, self.max_object_input_bytes,
            self.max_object_json_bytes, self.max_object_nodes, self.max_object_result_bytes];
        if !self.reconcile_root_object {
            if overrides.iter().any(Option::is_some) { return Err(CandidateError::Arguments); }
            return Ok(None);
        }
        ObjectArgs { max_object_properties: self.max_object_properties.unwrap_or(4096),
            max_object_origins: self.max_object_origins.unwrap_or(16_384),
            max_object_input_bytes: self.max_object_input_bytes, max_object_json_bytes: self.max_object_json_bytes,
            max_object_nodes: self.max_object_nodes.unwrap_or(262_144),
            max_object_result_bytes: self.max_object_result_bytes }.limits(mapping).map(Some)
    }
    pub(in crate::candidate_cli::int4) fn workspace_bytes(self, map: &MapArgs, limits: MapLimits)
        -> Result<u64, CandidateError> {
        let object = self.limits(map.mapping)?;
        if object.is_some() && map.extraction.schema_path().to_str().is_some_and(|path| path.contains("://")) {
            return Err(CandidateError::Arguments);
        }
        crate::batch::extract::grouped_int4::long::stream::memory::root_object_workspace_bytes(
            limits.mapping, map.extraction.compiler(), object).map_err(|_| CandidateError::Arguments)
    }
    pub(in crate::candidate_cli::int4) fn request<C: DecodeStepControl>(self, map: &MapArgs,
        schema: String, budget: TaskBudget, control: &mut C) -> Result<ExtractionBatchArgs, CandidateError> {
        match self.limits(map.mapping)? {
            None => map.request(schema, budget),
            Some(object) => {
                if map.array_union.limits(map.mapping)?.is_some() { return Err(CandidateError::Arguments); }
                object_request(map, schema, budget, object, control)
            }
        }
    }
}

/// Public selection describes the policy and bounds without private schema,
/// source bytes, paths, extracted values or content-derived commitments.
#[derive(Serialize)]
pub(in crate::candidate_cli::int4) struct RootObjectSelection {
    policy: &'static str,
    schema_derivation: &'static str,
    map_prompt: &'static str,
    limits: RootObjectLimits,
}
impl RootObjectSelection {
    pub(in crate::candidate_cli::int4) fn new(limits: RootObjectLimits) -> Self {
        Self { policy: crate::tasks::extract::object_reconcile::ROOT_OBJECT_RECONCILIATION,
            schema_derivation: crate::tasks::extract::object_reconcile::ROOT_OBJECT_DERIVATION,
            map_prompt: crate::batch::extract::grouped_int4::long::object::ROOT_OBJECT_MAP_PROMPT, limits }
    }
}
impl ObjectArgs {
    fn limits(self, mapping: MappingArgs) -> Result<RootObjectLimits, CandidateError> {
        let limits = RootObjectLimits { max_properties: self.max_object_properties, max_origins: self.max_object_origins,
            max_input_bytes: self.max_object_input_bytes.unwrap_or(mapping.max_total_value_bytes),
            max_json_bytes: self.max_object_json_bytes.unwrap_or(mapping.max_map_result_bytes),
            max_nodes: self.max_object_nodes,
            max_result_bytes: self.max_object_result_bytes.unwrap_or(mapping.max_map_result_bytes) };
        limits.workspace_bytes().map_err(|_| CandidateError::Arguments)?;
        if limits.max_json_bytes > mapping.max_map_result_bytes || limits.max_result_bytes > mapping.max_map_result_bytes {
            return Err(CandidateError::Arguments);
        }
        Ok(limits)
    }
}
pub(in crate::candidate_cli::int4) fn definition() -> clap::Command {
    ObjectArgs::augment_args(MappingArgs::augment_args(extract::definition().name("extract-object-map")
        .about("Extract complementary root-object properties across a complete document")
        .long_about("Use an explicit root-object schema for a complete UTF-8 document. A versioned derivation empties ONLY its root required set for each native chunk; nested objects and arrays retain all original constraints and are atomic values. A separate fixed instruction asks the model to omit root properties whose complete values are absent from a chunk. Every actual chunk plan preflights before the first forward. The deterministic reducer combines complementary properties, coalesces exactly equal values with ALL origins, and refuses any conflicting value or missing final required property. No first/last-wins choice, null substitution, recursive merge or model repair. The final exact JSON is independently validated against the ORIGINAL schema. With --source-membership, every final annotated field is independently matched against the COMPLETE original source, retaining all exact occurrences; this uses the remaining original field/match/scan allowance after chunk verification. Chunk-native outputs and their original-coordinate evidence remain available. Structural validity, lexical membership and conflict-free reconciliation do not establish semantic accuracy or extraction completeness. All work, tokenizer, output, memory and deadline limits cover the entire invocation. The whole map-plus-object envelope must fit --max-map-result-bytes. Execution is serial and full-vocabulary by default. --cohort-chunks N shares native decoder passes across up to N independent chunks, admitting every window before model execution. --max-selected-rows N evaluates ALL legal grammar rows, including accepted EOS; a wider legal set refuses without pruning or fallback. These options compose and retain independent chunk KV, the complete original-schema reducer and one resident model. Ordered populations use extract-corpus --reconcile-root-object; retained jobs use extract-document-job start|resume --reconcile-root-object. Input remains byte-encoded; packed prefill, CPU teams, Safe-BPE, sampling, thinking and tools require separate execution contracts.")
        .mut_arg("input", |arg| arg.help("Complete UTF-8 document; '-' reads stdin; not NDJSON"))
        .mut_arg("max_input_bytes", |arg| arg.default_value("1048576").help("Maximum bytes of the complete original document"))
        .mut_arg("preparation_mib", |arg| arg.default_value("1024"))
        .mut_arg("max_checkpoints", |arg| arg.help("One allowance for hosted preparation, every native chunk and final reconciliation"))))
        .arg(extract::selected_rows_argument())
        .arg(clap::Arg::new("cohort_chunks").long("cohort-chunks")
            .value_name("N").value_parser(clap::value_parser!(usize))
            .help("Share native decoder passes across up to N independent partial-object chunks (1..=64); serial if omitted"))
}
#[derive(Clone, Copy, Debug)]
struct ObjectMapLimits { map: MapLimits, object: RootObjectLimits, object_workspace_bytes: u64 }
impl ObjectMapLimits {
    fn mapping(self) -> Int4ObjectExtractionMapLimits {
        Int4ObjectExtractionMapLimits { mapping: self.map.mapping, reconciliation: self.object }
    }
}
impl ObjectMapArgs {
    pub(in crate::candidate_cli::int4) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        Ok(Self { map: MapArgs { extraction: extract::ExtractionArgs::from_matches_with_projection(matches)?,
            mapping: MappingArgs::from_arg_matches(matches)?,
            cohort_chunks: matches.get_one::<usize>("cohort_chunks").copied(),
            array_union: ArrayUnionArgs::default() }, object: ObjectArgs::from_arg_matches(matches)? })
    }
    fn validate(&self, args: &Int4Args) -> Result<ObjectMapLimits, CandidateError> {
        // Revalidate the owned arguments too: parser absence is not authority.
        if self.map.array_union.limits(self.map.mapping)?.is_some()
            || !args.input_encoding.is_default() || args.compute.threads() != 1 {
            return Err(CandidateError::Arguments);
        }
        let remote = self.map.extraction.schema_path().to_str().is_some_and(|path| path.contains("://"));
        if remote { return Err(CandidateError::Arguments); }
        let mut map = self.map.validate(args)?;
        let object = self.object.limits(self.map.mapping)?;
        let object_workspace_bytes = sum(&[
            Int4ObjectExtractionMapLimits { mapping: map.mapping, reconciliation: object }
                .reconciliation_workspace_bytes().map_err(|_| CandidateError::Arguments)?,
            RootObjectSchema::workspace_bytes(self.map.extraction.compiler()).map_err(|_| CandidateError::Arguments)?,
        ])?;
        map.common.minimum_memory_bytes = sum(&[map.common.minimum_memory_bytes, object_workspace_bytes])?;
        map.workspace_bytes = sum(&[map.workspace_bytes, object_workspace_bytes])?;
        if map.common.minimum_memory_bytes > map.common.memory_bytes { return Err(CandidateError::Arguments); }
        Ok(ObjectMapLimits { map, object, object_workspace_bytes })
    }
    fn request<C: DecodeStepControl>(&self, schema: String, budget: TaskBudget,
        limits: ObjectMapLimits, control: &mut C) -> Result<ExtractionBatchArgs, CandidateError> {
        object_request(&self.map, schema, budget, limits.object, control)
    }
    pub(in crate::candidate_cli::int4) fn execute(self, args: Int4Args, input: &mut impl Read, output: &mut impl Write)
        -> Result<(), CandidateError> {
        let limits = self.validate(&args)?;
        #[cfg(feature = "asupersync-runtime")]
        { runtime::execute(args, self, limits, input, output) }
        #[cfg(not(feature = "asupersync-runtime"))]
        { let _ = (self, args, limits, input, output); Err(CandidateError::Unavailable) }
    }
}
fn object_request<C: DecodeStepControl>(map: &MapArgs, schema: String, budget: TaskBudget,
    limits: RootObjectLimits, control: &mut C) -> Result<ExtractionBatchArgs, CandidateError> {
    let derived = RootObjectSchema::derive(&schema, map.extraction.compiler(), limits, control)
        .map_err(planning_error)?;
    // The complete ORIGINAL object may exceed one chunk's output ceiling;
    // the partial contract, not the full required set, is the native schema.
    let partial = map.extraction.shared_defaults_with_projection(derived.partial_schema().to_owned(), budget)?;
    if derived.requires_source() != (partial.grounding == ExtractionBatchGrounding::SourceMembership) {
        return Err(CandidateError::Planning);
    }
    if let Some(cause) = control.prefill_checkpoint(0) {
        return Err(planning_error(crate::tasks::extract::object_reconcile::RootObjectError::Cancelled(cause)));
    }
    Ok(ExtractionBatchArgs { schema, grounding: partial.grounding, budget })
}
fn planning_error(error: crate::tasks::extract::object_reconcile::RootObjectError) -> CandidateError {
    use crate::{native_engine::decode::DecodeCancellationKind, tasks::extract::object_reconcile::RootObjectError};
    match error {
        RootObjectError::Cancelled(DecodeCancellationKind::Timeout | DecodeCancellationKind::Deadline) => CandidateError::Timeout,
        RootObjectError::Cancelled(_) => CandidateError::Execution,
        _ => CandidateError::Planning,
    }
}
