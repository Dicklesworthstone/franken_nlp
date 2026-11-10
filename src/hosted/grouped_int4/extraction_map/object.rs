//! One charged native map and deterministic original-schema object reduction.
use super::*;
use crate::{batch::extract::grouped_int4::long::{
    object::{Int4ObjectExtractionPlanner, Int4ObjectExtractionMapLimits, Int4ObjectExtractionMapRun,
        Int4ObjectExtractionMapCohortRun, Int4ObjectExtractionMapError},
    cohort::{Int4ExtractionMapCohortBudget, Int4ExtractionMapCohortRequirement, COHORT_ENVELOPE_BYTES}},
    tasks::extract::object_reconcile::{RootObjectLimits, RootObjectSchema}, grammar::CompileLimits};

pub struct Int4ObjectExtractionMapConfig {
    /// The original caller schema remains here; chunk schemas are derived only
    /// inside the separately versioned object planner.
    pub mapping: Int4ExtractionMapConfig,
    pub reconciliation: RootObjectLimits,
}
impl Int4ObjectExtractionMapConfig {
    pub fn limits(&self) -> Int4ObjectExtractionMapLimits {
        Int4ObjectExtractionMapLimits { mapping: self.mapping.limits, reconciliation: self.reconciliation }
    }
    /// Native and decoder peak for one simultaneous chunk window. Complete
    /// document reduction, original/partial schemas and reconciliation remain
    /// separately charged through final original-schema validation.
    pub fn cohort_requirements(&self, group_width: usize)
        -> Result<Int4ExtractionMapCohortRequirement, HostedError> {
        self.limits().reconciliation_workspace_bytes().map_err(object_error)?;
        self.mapping.cohort_requirements(group_width)
    }
    pub fn workspace_bytes(&self, chunks: usize, compiler: CompileLimits) -> Result<u64, HostedError> {
        sum(&[self.mapping.workspace_bytes(chunks)?, self.object_workspace_bytes(compiler)?])
    }
    fn object_workspace_bytes(&self, compiler: CompileLimits) -> Result<u64, HostedError> {
        sum(&[self.limits().reconciliation_workspace_bytes().map_err(object_error)?,
            RootObjectSchema::workspace_bytes(compiler).map_err(|e| object_error(e.into()))?])
    }
}
struct ObjectInput {
    source: String,
    planner: Arc<Int4ObjectExtractionPlanner>,
    vocabulary: Arc<ExtractionVocabulary>,
    config: Int4ObjectExtractionMapConfig,
}
impl NlpEngine {
    /// Extract partial root objects from every original chunk, then reconcile
    /// exact atomic property values and independently validate the original
    /// schema. Conflicts/missing required fields fail the entire invocation.
    /// Original-source evidence, when requested, consumes the same cumulative
    /// verification budget as chunk evidence. Native work is never repeated.
    #[allow(clippy::too_many_arguments)]
    pub fn extract_int4_object_document(&self, model: &ResidentInt4, source: String,
        planner: Arc<Int4ObjectExtractionPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int4ObjectExtractionMapConfig, kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<Int4ObjectExtractionMapRun>, HostedError> {
        let base = &config.mapping;
        dispatch::preflight(self, base.native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        let width = model.model_identity().group_width();
        if planner.group_width() != width { return Err(HostedError::ModelIdentity); }
        let identity = planner.base_execution_identity();
        constrained_int4::check_model(identity, model.model_identity()).map_err(|_| HostedError::ModelIdentity)?;
        let required = Int4MemoryRequirement::for_context(base.native.context_tokens, width).map_err(native_error)?;
        validate(base, identity, width, source.len(), required.kv_bytes)?;
        if base.limits.mapping.reduction.map_batch_chunks != 1 {
            return Err(object_error(Int4ObjectExtractionMapError::Admission));
        }
        let object_bytes = config.object_workspace_bytes(planner.compiler_limits())?;
        let bytes = input_bytes(source.capacity(), base)?;
        let run = base.native.run;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?,
            || Ok(ObjectInput { source, planner, vocabulary, config }))?;
        let model = model.clone();
        dispatch::run(self, run, cancellation, move |control| {
            let input = input;
            let config = &input.value.config;
            let base = &config.mapping;
            poll(control)?;
            // Schema derivation and its independent parses precede map
            // execution, so this additional charge precedes preparation too.
            let object_workspace = Pending::reserve(&lease, MemoryClass::JobBuffers, object_bytes)?;
            let prepared = input.value.planner.plan_document_with_control(&input.value.source,
                &base.request, config.limits(), control).map_err(object_error)?;
            let expected = prepared.preflight_metadata();
            let count = expected.chunk_count();
            if prepared.group_width() != width || expected.group_width() != width
                || expected.selected_rows() != input.value.planner.selected_rows()
                || prepared.execution_identities().len() != count || required.kv_bytes > prepared.max_kv_bytes() {
                return Err(HostedError::ModelIdentity);
            }
            let reduction = Pending::reserve(&lease, MemoryClass::JobBuffers, base.workspace_bytes(count)?)?;
            let tokens = (count as u64).checked_mul(u64::from(base.request.budget.max_output_tokens))
                .ok_or(HostedError::Limits("INT4 object map token arithmetic"))?;
            let output = output_claim(&lease, prepared.max_result_bytes(), tokens)?;
            let mut admitted = Vec::new(); admitted.try_reserve_exact(count)
                .map_err(|_| HostedError::Limits("INT4 object map identity storage"))?;
            for identity in prepared.execution_identities() {
                poll(control)?;
                constrained_int4::check_model(identity, model.model_identity()).map_err(|_| HostedError::ModelIdentity)?;
                check_identity(identity, width)?;
                let base = input.value.planner.base_execution_identity();
                if identity.tokenizer_digest != base.tokenizer_digest || identity.template_digest != base.template_digest {
                    return Err(HostedError::ModelIdentity);
                }
                admitted.push(identity.clone());
            }
            poll(control)?;
            let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
            let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch,
                sum(&[required.rope_bytes, required.scratch_payload_bound, base.native.allocator_reserve_bytes])?)?;
            let mut native = allocate_native(kv, scratch, || {
                StrictInt4Engine::new(&model.inner.loaded.value, base.native.context_tokens, Int4MemoryBudget {
                    max_kv_bytes: required.kv_bytes, max_rope_bytes: required.rope_bytes,
                    max_scratch_payload_bytes: required.scratch_payload_bound,
                }).map(|engine| engine.with_kernel(kernel)).map_err(native_error)
            })?;
            let result = allocate(output, || {
                let result = prepared.execute_with_control(&admitted, &mut native.value, &input.value.vocabulary, control)
                    .map_err(object_error)?;
                expected.verify_completed(&result).map_err(object_error)?;
                if native.value.is_poisoned() || !native.value.kv_cache().all_slots_have_len(0) {
                    return Err(object_error(Int4ObjectExtractionMapError::Admission));
                }
                poll(control)?;
                Ok(result)
            })?;
            drop(native);
            drop(admitted);
            drop(input);
            drop(reduction);
            drop(object_workspace);
            drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }

    /// Extract bounded windows of partial root objects through shared native
    /// decoder layers, then reconcile every chunk against the original schema.
    /// All real chunk identities and exact window peaks are admitted before
    /// native execution. Schema, reduction and peak native reservations remain
    /// live through reconciliation; no successful prefix can be published.
    #[allow(clippy::too_many_arguments)]
    pub fn extract_int4_object_document_cohorts(&self, model: &ResidentInt4, source: String,
        planner: Arc<Int4ObjectExtractionPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int4ObjectExtractionMapConfig, kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<Int4ObjectExtractionMapCohortRun>, HostedError> {
        let base = &config.mapping;
        dispatch::preflight(self, base.native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        let width = model.model_identity().group_width();
        if planner.group_width() != width { return Err(HostedError::ModelIdentity); }
        let identity = planner.base_execution_identity();
        constrained_int4::check_model(identity, model.model_identity()).map_err(|_| HostedError::ModelIdentity)?;
        // Each task authorizes one full context, independent of its neighbors.
        let row = Int4MemoryRequirement::for_context(base.native.context_tokens, width).map_err(native_error)?;
        validate(base, identity, width, source.len(), row.kv_bytes)?;
        let ceiling = config.cohort_requirements(width)?;
        let object_bytes = config.object_workspace_bytes(planner.compiler_limits())?;
        let bytes = input_bytes(source.capacity(), base)?;
        let run = base.native.run;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?,
            || Ok(ObjectInput { source, planner, vocabulary, config }))?;
        let model = model.clone();
        dispatch::run(self, run, cancellation, move |control| {
            let input = input;
            let config = &input.value.config;
            let base = &config.mapping;
            poll(control)?;
            // Original and derived schemas are materialized during preparation.
            let object_workspace = Pending::reserve(&lease, MemoryClass::JobBuffers, object_bytes)?;
            let prepared = input.value.planner.plan_document_with_control(&input.value.source,
                &base.request, config.limits(), control).map_err(object_error)?;
            let expected = prepared.preflight_metadata();
            let count = expected.chunk_count();
            if prepared.group_width() != width || expected.group_width() != width
                || expected.selected_rows() != input.value.planner.selected_rows()
                || prepared.execution_identities().len() != count || row.kv_bytes > prepared.max_kv_bytes() {
                return Err(HostedError::ModelIdentity);
            }
            // Every actual window, including a shorter final window, must fit
            // all component peaks before the first native engine is allocated.
            let required = prepared.cohort_requirements(base.native.context_tokens).map_err(object_error)?;
            check_object_peak(required, ceiling)?;
            let reduction = Pending::reserve(&lease, MemoryClass::JobBuffers, base.workspace_bytes(count)?)?;
            let tokens = (count as u64).checked_mul(u64::from(base.request.budget.max_output_tokens))
                .ok_or(HostedError::Limits("INT4 object map cohort token arithmetic"))?;
            let output = output_claim(&lease,
                sum(&[prepared.max_result_bytes(), COHORT_ENVELOPE_BYTES])?, tokens)?;
            let mut admitted = Vec::new();
            admitted.try_reserve_exact(count)
                .map_err(|_| HostedError::Limits("INT4 object map identity storage"))?;
            for identity in prepared.execution_identities() {
                poll(control)?;
                constrained_int4::check_model(identity, model.model_identity()).map_err(|_| HostedError::ModelIdentity)?;
                check_identity(identity, width)?;
                let base = input.value.planner.base_execution_identity();
                if identity.tokenizer_digest != base.tokenizer_digest || identity.template_digest != base.template_digest {
                    return Err(HostedError::ModelIdentity);
                }
                admitted.push(identity.clone());
            }
            poll(control)?;
            let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
            let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch,
                sum(&[required.native.rope_bytes, required.native.scratch_payload_bound,
                    base.native.allocator_reserve_bytes])?)?;
            let workspace = Pending::reserve(&lease, MemoryClass::JobBuffers, required.workspace_bytes)?;
            let budget = Int4ExtractionMapCohortBudget {
                context_limit: base.native.context_tokens,
                native_memory: Int4MemoryBudget { max_kv_bytes: required.native.kv_bytes,
                    max_rope_bytes: required.native.rope_bytes,
                    max_scratch_payload_bytes: required.native.scratch_payload_bound },
                max_workspace_bytes: required.workspace_bytes,
                max_result_bytes: required.max_result_bytes,
            };
            let result = allocate(output, || {
                let result = prepared.execute_cohorts_with_control(&admitted, &model.inner.loaded.value,
                    &input.value.vocabulary, budget, kernel, control).map_err(object_error)?;
                expected.verify_cohorts_completed(&result).map_err(object_error)?;
                poll(control)?;
                Ok(result)
            })?;
            drop(admitted);
            drop(input);
            drop(workspace);
            drop(scratch);
            drop(kv);
            drop(reduction);
            drop(object_workspace);
            drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }
}
fn check_object_peak(required: Int4ExtractionMapCohortRequirement,
    ceiling: Int4ExtractionMapCohortRequirement) -> Result<(), HostedError> {
    if required.native.kv_bytes > ceiling.native.kv_bytes
        || required.native.rope_bytes > ceiling.native.rope_bytes
        || required.native.scratch_payload_bound > ceiling.native.scratch_payload_bound
        || required.workspace_bytes > ceiling.workspace_bytes || required.max_result_bytes > ceiling.max_result_bytes {
        return Err(HostedError::Limits("INT4 object extraction complete cohort peak exceeds declaration"));
    }
    Ok(())
}
fn object_error(error: Int4ObjectExtractionMapError) -> HostedError {
    HostedError::Int4(Int4HostError::ObjectExtractionMap(error))
}

#[cfg(test)] mod tests;
