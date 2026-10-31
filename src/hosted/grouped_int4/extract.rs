//! Schema/source extraction on the charged resident INT4 model and runtime.
//! Preparation stays caller-admitted; all native and output storage belongs to
//! the actual blocking invocation until finalization and physical cleanup.

use super::*;
use crate::{
    grammar::mask::MaskWorkLimits,
    native_engine::{constrained::JsonWorkBudget,
        constrained_int4::{self, Int4JsonBudget}, strict_int4::{Int4Work, prefill::Int4PrefillLimits}},
    tasks::extract::{ExtractionVocabulary,
        grouped_int4::{Int4ExtractError, Int4ExtractPlan, Int4ExtractRun}},
};

impl NlpEngine {
    /// Execute a sealed schema/source plan using either source-converted or
    /// independently pinned snapshot-loaded weights. No artifact activation,
    /// sampler, parse retry or automatic kernel selection occurs here.
    /// The caller charges the immutable vocabulary and preparation separately;
    /// the returned guard retains the complete result through its delivery.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int4_extract(&self, model: &ResidentInt4, prepared: Int4ExtractPlan,
        vocabulary: Arc<ExtractionVocabulary>, native: NativeLimits,
        mask_limits: MaskWorkLimits, max_mask_node_visits: u64, kernel: PackedInt4Kernel,
        cancellation: CancellationToken) -> Result<HostedOutput<Int4ExtractRun>, HostedError> {
        self.execute_int4_extract_scheduled(model, prepared, vocabulary, native, mask_limits,
            max_mask_node_visits, kernel, None, cancellation)
    }
    /// Use bounded layer-major prompt morsels for the same exact schema/source
    /// plan. Extra prompt storage is admitted with full native/output storage;
    /// grammar masks, sequential decode and independent finalization are unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int4_extract_layer_major(&self, model: &ResidentInt4, prepared: Int4ExtractPlan,
        vocabulary: Arc<ExtractionVocabulary>, native: NativeLimits,
        mask_limits: MaskWorkLimits, max_mask_node_visits: u64, kernel: PackedInt4Kernel,
        prefill: Int4PrefillLimits, cancellation: CancellationToken)
        -> Result<HostedOutput<Int4ExtractRun>, HostedError> {
        self.execute_int4_extract_scheduled(model, prepared, vocabulary, native, mask_limits,
            max_mask_node_visits, kernel, Some(prefill), cancellation)
    }
    #[allow(clippy::too_many_arguments)]
    fn execute_int4_extract_scheduled(&self, model: &ResidentInt4, prepared: Int4ExtractPlan,
        vocabulary: Arc<ExtractionVocabulary>, native: NativeLimits,
        mask_limits: MaskWorkLimits, max_mask_node_visits: u64, kernel: PackedInt4Kernel,
        prefill: Option<Int4PrefillLimits>, cancellation: CancellationToken)
        -> Result<HostedOutput<Int4ExtractRun>, HostedError> {
        dispatch::preflight(self, native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        constrained_int4::check_model(prepared.execution_identity(), model.model_identity())
            .map_err(Int4ExtractError::from).map_err(extraction_error)?;
        if prepared.group_width() != model.model_identity().group_width() { return Err(HostedError::ModelIdentity); }
        let work = prepared.planned_work();
        let (required, scratch_bytes) = scheduled_requirements(work, prepared.group_width(),
            prepared.max_kv_bytes(), native, mask_limits, max_mask_node_visits, prefill)?;
        let lease = self.resources().acquire_lease();
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
        let workspace = Pending::reserve(&lease, MemoryClass::ActivationScratch, scratch_bytes)?;
        let output = output_claim(&lease, prepared.max_result_bytes(), prepared.options().max_new_tokens as u64)?;
        let model = model.clone();
        dispatch::run(self, native.run, cancellation, move |control| {
            let mut engine = allocate_native(kv, workspace, || {
                StrictInt4Engine::new(&model.inner.loaded.value, native.context_tokens, Int4MemoryBudget {
                    max_kv_bytes: required.kv_bytes, max_rope_bytes: required.rope_bytes,
                    max_scratch_payload_bytes: required.scratch_payload_bound,
                }).map(|engine| engine.with_kernel(kernel)).map_err(native_error)
            })?;
            let budget = Int4JsonBudget { native: Int4RunBudget::exact(work), json: JsonWorkBudget {
                max_forward_positions: work.forward_positions, max_projected_logits: work.projected_logits,
                max_kv_bytes: required.kv_bytes, max_total_mask_node_visits: max_mask_node_visits, mask_limits,
            } };
            let result = allocate(output, || {
                let run = match prefill {
                    None => prepared.execute(&mut engine.value, prepared.execution_identity(),
                        &vocabulary, budget, control),
                    Some(limits) => prepared.execute_layer_major(&mut engine.value, prepared.execution_identity(),
                        &vocabulary, budget, limits, control),
                }.map_err(extraction_error)?;
                // The task finalizer checks exact completed work and source
                // evidence. The host also requires the native session to have
                // drained before any guarded completion can leave this call.
                if run.group_width != prepared.group_width() || engine.value.is_poisoned()
                    || !engine.value.kv_cache().all_slots_have_len(0) {
                    return Err(extraction_error(Int4ExtractError::Identity));
                }
                Ok(run)
            })?;
            drop(engine);
            drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }
}

fn extraction_error(error: Int4ExtractError) -> HostedError { HostedError::Int4(Int4HostError::Extraction(error)) }

/// Both structured task families retain the same native and mask admission.
/// The declared prompt window adds storage once, independent of prompt length;
/// its owner follows the native engine through the complete physical call.
pub(super) fn scheduled_requirements(work: Int4Work, width: usize, max_kv_bytes: u64,
    native: NativeLimits, masks: MaskWorkLimits, max_mask_node_visits: u64,
    prefill: Option<Int4PrefillLimits>) -> Result<(Int4MemoryRequirement, u64), HostedError> {
    let required = requirements(work, width, max_kv_bytes, native, masks, max_mask_node_visits)?;
    let base = sum(&[required.rope_bytes, required.scratch_payload_bound, native.allocator_reserve_bytes])?;
    let scratch = super::prefill::with_extra_workspace(base, prefill)?;
    Ok((required, scratch))
}

pub(super) fn requirements(work: Int4Work, width: usize, max_kv_bytes: u64, native: NativeLimits,
    masks: MaskWorkLimits, max_mask_node_visits: u64) -> Result<Int4MemoryRequirement, HostedError> {
    native.run.validate()?;
    if work.forward_positions == 0 || work.forward_positions > native.context_tokens as u64
        || native.allocator_reserve_bytes == 0 || masks.max_trie_node_visits == 0
        || masks.checkpoint_interval_nodes == 0 || max_mask_node_visits == 0 {
        return Err(HostedError::Limits("INT4 extraction work, mask or allocator limits"));
    }
    let required = Int4MemoryRequirement::for_context(native.context_tokens, width).map_err(native_error)?;
    if required.kv_bytes > max_kv_bytes { return Err(HostedError::Limits("INT4 extraction complete KV allocation")); }
    sum(&[required.rope_bytes, required.scratch_payload_bound, native.allocator_reserve_bytes])?;
    Ok(required)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn native() -> NativeLimits {
        NativeLimits { context_tokens: 128, allocator_reserve_bytes: 65_536,
            run: RunLimits { max_elapsed: Duration::from_secs(1), max_checkpoints: 1000,
                cleanup_reserve_bytes: 65_536 } }
    }
    fn masks() -> MaskWorkLimits { MaskWorkLimits { max_trie_node_visits: 1000, checkpoint_interval_nodes: 16 } }
    #[test]
    fn full_native_capacity_is_charged_even_for_a_short_structured_request() {
        for width in [1, 3, 32, 128, 10752] {
            let work = constrained_int4::planned_work(width, 8, 4).unwrap();
            let need = Int4MemoryRequirement::for_context(128, width).unwrap();
            assert_eq!(requirements(work, width, need.kv_bytes, native(), masks(), 4000).unwrap(), need);
            assert!(requirements(work, width, need.kv_bytes - 1, native(), masks(), 4000).is_err());
        }
    }
    #[test]
    fn bad_context_masks_and_overflow_refuse_before_native_allocation() {
        let work = constrained_int4::planned_work(128, 8, 4).unwrap();
        let mut n = native(); n.context_tokens = 10;
        assert!(requirements(work, 128, u64::MAX, n, masks(), 4000).is_err());
        for width in [0, 10753, usize::MAX] {
            assert!(requirements(work, width, u64::MAX, native(), masks(), 4000).is_err());
        }
        let mut n = native(); n.allocator_reserve_bytes = u64::MAX;
        assert!(requirements(work, 128, u64::MAX, n, masks(), 4000).is_err());
        assert!(requirements(work, 128, u64::MAX, native(), masks(), 0).is_err());
        for m in [MaskWorkLimits { max_trie_node_visits: 0, checkpoint_interval_nodes: 1 },
            MaskWorkLimits { max_trie_node_visits: 1, checkpoint_interval_nodes: 0 }] {
            assert!(requirements(work, 128, u64::MAX, native(), m, 4000).is_err());
        }
    }
    #[test]
    fn structured_prompt_windows_price_static_extra_storage_without_changing_task_kv() {
        for width in [1, 3, 128, 10752] {
            let n = native();
            let work = constrained_int4::planned_work(width, 8, 4).unwrap();
            let required = Int4MemoryRequirement::for_context(n.context_tokens, width).unwrap();
            let (serial, base) = scheduled_requirements(work, width, required.kv_bytes,
                n, masks(), 4000, None).unwrap();
            assert_eq!(serial, required);
            assert_eq!(base, required.rope_bytes + required.scratch_payload_bound + n.allocator_reserve_bytes);
            for rows in [1, 3, 64] {
                let extra = Int4PrefillLimits::required_extra_scratch_bytes(rows).unwrap();
                for ceiling in [extra, u64::MAX] {
                    let limits = Int4PrefillLimits { max_batch_rows: rows, max_extra_scratch_bytes: ceiling };
                    let (grouped, scratch) = scheduled_requirements(work, width, required.kv_bytes,
                        n, masks(), 4000, Some(limits)).unwrap();
                    assert_eq!(grouped, serial);
                    // Even an eight-token prompt retains the declared window.
                    // A larger upper bound grants no additional allocation.
                    assert_eq!(scratch, base + extra);
                }
            }
        }
    }
    #[test]
    fn structured_prefill_limits_cannot_bypass_context_kv_masks_or_checked_storage() {
        let width = 128;
        let work = constrained_int4::planned_work(width, 8, 4).unwrap();
        let n = native();
        let required = Int4MemoryRequirement::for_context(n.context_tokens, width).unwrap();
        let extra = Int4PrefillLimits::required_extra_scratch_bytes(4).unwrap();
        let limits = Int4PrefillLimits { max_batch_rows: 4, max_extra_scratch_bytes: extra };
        for rows in [0, 65, usize::MAX] {
            assert!(scheduled_requirements(work, width, required.kv_bytes, n, masks(), 4000,
                Some(Int4PrefillLimits { max_batch_rows: rows, max_extra_scratch_bytes: u64::MAX })).is_err());
        }
        assert!(scheduled_requirements(work, width, required.kv_bytes, n, masks(), 4000,
            Some(Int4PrefillLimits { max_extra_scratch_bytes: extra - 1, ..limits })).is_err());
        assert!(scheduled_requirements(work, width, required.kv_bytes - 1, n, masks(), 4000, Some(limits)).is_err());
        assert!(scheduled_requirements(work, width, required.kv_bytes,
            NativeLimits { context_tokens: 10, ..n }, masks(), 4000, Some(limits)).is_err());
        assert!(scheduled_requirements(work, width, required.kv_bytes, n, masks(), 0, Some(limits)).is_err());
        assert!(scheduled_requirements(work, width, required.kv_bytes, n,
            MaskWorkLimits { checkpoint_interval_nodes: 0, ..masks() }, 4000, Some(limits)).is_err());
        // Base scratch alone fits, but adding the prompt rails must not wrap.
        let overflow = NativeLimits { allocator_reserve_bytes:
            u64::MAX - required.rope_bytes - required.scratch_payload_bound, ..n };
        assert!(scheduled_requirements(work, width, required.kv_bytes,
            overflow, masks(), 4000, None).is_ok());
        assert!(scheduled_requirements(work, width, required.kv_bytes,
            overflow, masks(), 4000, Some(limits)).is_err());
    }
    #[test]
    fn typed_causes_survive_without_private_debug_output() {
        let cause = Int4ExtractError::from(crate::native_engine::constrained_int4::Int4JsonError::from(
            StrictInt4Error::Cancelled(crate::native_engine::decode::DecodeCancellationKind::Deadline)));
        let error = Int4HostError::Extraction(cause);
        assert!(error.source().is_some());
        assert_eq!(error.to_string(), format!("{error:?}"));
        assert_eq!(error.to_string(), "INT4 schema extraction execution or finalization refused");
    }
    #[test]
    fn extraction_owners_can_cross_the_existing_blocking_boundary() {
        fn send<T: Send + 'static>() {}
        send::<Int4ExtractPlan>(); send::<Int4ExtractRun>(); send::<Arc<ExtractionVocabulary>>();
        send::<Int4PrefillLimits>();
        let _grouped = NlpEngine::execute_int4_extract_layer_major;
    }
}
