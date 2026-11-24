//! Whole-document extraction options, separate from fixed source-task options.
use super::*;
use crate::{batch::extract::quantized::long::{Int8ExtractionMapLimits, object::Int8ObjectExtractionMapLimits},
    tasks::extract::object_reconcile::{RootObjectLimits, RootObjectSchema},
    candidate_cli::extract as schema, native_engine::strict_int8::prefill::Int8PrefillLimits,
    validation::grounded_fields::{GroundingBudget, SourceFieldEvidence, VerifiedSourceSpan}};

#[derive(Args)]
pub(in crate::candidate_cli) struct ExtractionArgs {
    /// Exact local JSON schema for --task extract; stdin belongs to the source.
    #[arg(long, value_name = "FILE", required_if_eq("task", "extract"), conflicts_with = "options")]
    schema: Option<PathBuf>,
    /// Enforce verbatim source annotations per chunk, with original coordinates.
    /// This does not establish semantic truth or merge independent JSON objects.
    #[arg(long, requires = "schema")]
    pub source_membership: bool,
    /// Reconcile partial root objects into one original-schema object. Only
    /// root required fields become optional per chunk; nested values stay atomic.
    /// Equal exact values keep all origins. Conflicts or missing fields refuse.
    #[arg(long, requires = "schema")]
    pub reconcile_object: bool,
    /// Distinct final root properties (default 4096), across the whole document.
    #[arg(long, requires = "reconcile_object")]
    max_object_properties: Option<usize>,
    /// All property origins, including exact duplicates (default 16384).
    #[arg(long, requires = "reconcile_object")]
    max_object_origins: Option<usize>,
    /// Sum of all partial-object JSON bytes; defaults to max-total-value-bytes.
    #[arg(long, requires = "reconcile_object")]
    max_object_input_bytes: Option<usize>,
    /// Exact final JSON string bytes; defaults to max-map-result-bytes.
    #[arg(long, requires = "reconcile_object")]
    max_object_json_bytes: Option<usize>,
    /// All partial value nodes plus final independent parse (default 262144).
    #[arg(long, requires = "reconcile_object")]
    max_object_nodes: Option<usize>,
    /// Final object plus all provenance; defaults to max-map-result-bytes.
    #[arg(long, requires = "reconcile_object")]
    max_object_result_bytes: Option<usize>,
    /// Nonrenewable independent verification fields across ALL chunks (default 4096).
    #[arg(long, requires = "schema")]
    max_extraction_fields: Option<usize>,
    /// Original-coordinate occurrence spans across ALL chunks (default 16384).
    #[arg(long, requires = "schema")]
    max_extraction_evidence_spans: Option<usize>,
    /// Nonrenewable independent source verification work (default 67108864).
    #[arg(long, requires = "schema")]
    max_extraction_scan_steps: Option<u64>,
    /// Bounded layer-major prompt morsels for extraction, NER, keyphrases or
    /// cited-summary maps and their exact reducers, 1..=64 rows. Composes with
    /// the same persistent team. QA and neural synthesis are not supported.
    /// Extra coordinator and worker storage is admitted before private IO.
    #[arg(long, value_name = "ROWS")]
    prefill_rows: Option<usize>,
}
impl ExtractionArgs {
    pub(super) fn validate(&self, task: &str) -> Result<(), CandidateError> {
        if task != "extract" {
            if self.schema.is_some() || self.source_membership || self.max_extraction_fields.is_some()
                || self.max_extraction_evidence_spans.is_some() || self.max_extraction_scan_steps.is_some()
                || self.reconcile_object || self.has_object_bounds() {
                return Err(CandidateError::Arguments);
            }
            return Ok(());
        }
        schema::check_schema_path(self.path()?)?;
        self.verification()?;
        self.prefill()?;
        Ok(())
    }
    pub(in crate::candidate_cli) fn path(&self) -> Result<&std::path::Path, CandidateError> {
        self.schema.as_deref().ok_or(CandidateError::Arguments)
    }
    fn verification(&self) -> Result<GroundingBudget, CandidateError> {
        let defaults = GroundingBudget::default();
        let budget = GroundingBudget {
            max_fields: self.max_extraction_fields.unwrap_or(defaults.max_fields),
            max_matches: self.max_extraction_evidence_spans.unwrap_or(defaults.max_matches),
            max_scan_steps: self.max_extraction_scan_steps.unwrap_or(defaults.max_scan_steps),
        };
        if !(1..=1_000_000).contains(&budget.max_fields) || !(1..=1_000_000).contains(&budget.max_matches)
            || !(1..=1_000_000_000_000).contains(&budget.max_scan_steps) { return Err(CandidateError::Arguments); }
        Ok(budget)
    }
    /// Retained original schemas and schema prompt storage for every prepared
    /// chunk are additional to the generic grammar/context staging allowance.
    pub(super) fn schema_reserve_per_chunk(&self) -> u64 {
        if self.schema.is_some() { schema::SCHEMA_BYTES as u64 * 2 } else { 0 }
    }
    pub(in crate::candidate_cli) fn prefill(&self) -> Result<Option<Int8PrefillLimits>, CandidateError> {
        self.prefill_rows.map(|rows| {
            let bytes = Int8PrefillLimits::required_extra_scratch_bytes(rows).map_err(|_| CandidateError::Arguments)?;
            Ok(Int8PrefillLimits { max_batch_rows: rows, max_extra_scratch_bytes: bytes })
        }).transpose()
    }
    fn has_object_bounds(&self) -> bool {
        self.max_object_properties.is_some() || self.max_object_origins.is_some()
            || self.max_object_input_bytes.is_some() || self.max_object_json_bytes.is_some()
            || self.max_object_nodes.is_some() || self.max_object_result_bytes.is_some()
    }
    pub(in crate::candidate_cli) fn object_limits(&self, command: &MapCommand)
        -> Result<Option<RootObjectLimits>, CandidateError> {
        if !self.reconcile_object {
            return if self.has_object_bounds() { Err(CandidateError::Arguments) } else { Ok(None) };
        }
        if command.task != "extract" || self.schema.is_none() || command.options.is_some() {
            return Err(CandidateError::Arguments);
        }
        let limits = RootObjectLimits {
            max_properties: self.max_object_properties.unwrap_or(4096),
            max_origins: self.max_object_origins.unwrap_or(16384),
            max_input_bytes: self.max_object_input_bytes.unwrap_or(command.max_total_value_bytes),
            max_json_bytes: self.max_object_json_bytes.unwrap_or(command.max_map_result_bytes),
            max_nodes: self.max_object_nodes.unwrap_or(262144),
            max_result_bytes: self.max_object_result_bytes.unwrap_or(command.max_map_result_bytes),
        };
        limits.workspace_bytes().map_err(|_| CandidateError::Arguments)?;
        if limits.max_json_bytes > command.max_map_result_bytes
            || limits.max_result_bytes > command.max_map_result_bytes {
            return Err(CandidateError::Arguments);
        }
        Ok(Some(limits))
    }
    /// Both original/partial schema declarations are additional to the exact
    /// reducer quote. A tiny output cap cannot hide a large optional schema.
    pub(in crate::candidate_cli) fn object_workspace_bytes(&self, command: &MapCommand)
        -> Result<u64, CandidateError> {
        let Some(reconciliation) = self.object_limits(command)? else { return Ok(0); };
        let limits = Int8ObjectExtractionMapLimits { mapping: command.extraction_mapping()?, reconciliation };
        limits.reconciliation_workspace_bytes().map_err(|_| CandidateError::Arguments)?
            .checked_add(RootObjectSchema::workspace_bytes(schema::compiler(&command.host))
                .map_err(|_| CandidateError::Arguments)?).ok_or(CandidateError::Arguments)
    }
    pub(in crate::candidate_cli) fn check_schedule_memory(&self, command: &MapCommand, args: &CandidateArgs, limits: Limits)
        -> Result<(), CandidateError> {
        let object_bytes = self.object_workspace_bytes(command)?;
        if limits.compute.threads == 1 && self.prefill()?.is_none() && object_bytes == 0 { return Ok(()); }
        if self.schedule_peak_bytes(command, args, limits)? > limits.memory_bytes {
            return Err(CandidateError::Arguments);
        }
        Ok(())
    }
    pub(super) fn schedule_peak_bytes(&self, command: &MapCommand, args: &CandidateArgs, limits: Limits)
        -> Result<u64, CandidateError> {
        let native = Int8MemoryRequirement::for_context(args.context_tokens).map_err(|_| CandidateError::Arguments)?;
        let tokens = (command.max_chunks as u64).checked_mul(args.max_new_tokens as u64)
            .ok_or(CandidateError::Arguments)?;
        let base = compute::request_peak_bytes(limits, native, command.max_map_result_bytes as u64,
            tokens, 0, limits.preparation_bytes)?;
        let extra = self.prefill()?.map(Int8PrefillLimits::validate).transpose()
            .map_err(|_| CandidateError::Arguments)?.unwrap_or(0);
        // Price the complete retained frontier, original-coordinate scans and
        // a separately transferred input owner, not one chunk multiplied only
        // after allocation. The host independently reserves its actual counts.
        let retained = (command.host.max_result_bytes as u64).checked_mul(command.max_chunks as u64)
            .and_then(|n| n.checked_add(command.max_live_value_bytes as u64))
            .and_then(|n| n.checked_mul(4)).ok_or(CandidateError::Arguments)?;
        let verification = self.verification()?;
        let fields = (verification.max_fields as u64).checked_mul(std::mem::size_of::<SourceFieldEvidence>() as u64)
            .ok_or(CandidateError::Arguments)?;
        let spans = (verification.max_matches as u64).checked_mul(std::mem::size_of::<VerifiedSourceSpan>() as u64)
            .and_then(|n| n.checked_mul(2)).ok_or(CandidateError::Arguments)?;
        let encoding = command.encoding.limits(command)?.map(|limits| limits.workspace_bytes())
            .transpose()?.unwrap_or(0);
        // The caller retains schema/preflight ownership through final write;
        // the host independently reserves its own schema/reducer peak through
        // physical drain. Admit both before either source or schema is read.
        let object = self.object_workspace_bytes(command)?.checked_mul(2).ok_or(CandidateError::Arguments)?;
        [base, extra, object, retained, fields, spans, tokens.checked_mul(8).ok_or(CandidateError::Arguments)?,
            command.reduction_reserve_mib.checked_mul(MIB).ok_or(CandidateError::Arguments)?,
            (args.max_input_bytes as u64).checked_mul(4).ok_or(CandidateError::Arguments)?,
            2 * schema::SCHEMA_BYTES as u64, encoding]
            .into_iter().try_fold(0_u64, |sum, bytes| sum.checked_add(bytes))
            .ok_or(CandidateError::Arguments)
    }
}
impl MapCommand {
    /// Requested ceilings only. The actual extraction compiler tightens context
    /// using the exact schema, pinned scaffold and real source token counts.
    pub(in crate::candidate_cli) fn extraction_mapping(&self) -> Result<Int8ExtractionMapLimits, CandidateError> {
        if self.kind()? != BuiltInTask::Extract || self.options.is_some() { return Err(CandidateError::Arguments); }
        self.extraction.validate(&self.task)?;
        let chunks = ChunkLimits { max_input_bytes: self.host.max_input_bytes,
            max_chunk_bytes: self.max_chunk_bytes.unwrap_or(self.host.context_tokens).min(self.host.max_input_bytes),
            max_chunk_tokens: self.host.context_tokens, context_tokens: self.host.context_tokens,
            reserved_tokens: self.host.max_new_tokens, max_chunks: self.max_chunks,
            max_tokenizer_calls: self.max_tokenizer_calls };
        chunks.effective_token_limit().map_err(|_| CandidateError::Arguments)?;
        Ok(Int8ExtractionMapLimits { mapping: self.mapping_limits(chunks), verification: self.extraction.verification()? })
    }
}

#[cfg(test)] mod tests;
