//! The complete bounded source-task portfolio on the resident INT4 model.
//! Independent task finalization remains inside the native KV session, and
//! the result's memory charge stays live until the caller releases delivery.

use super::*;
use crate::{
    grammar::mask::MaskWorkLimits,
    native_engine::{constrained::JsonWorkBudget, constrained_int4::{self, Int4JsonBudget},
        strict_int4::prefill::Int4PrefillLimits},
    tasks::{extract::ExtractionVocabulary,
        source_planning::grouped_int4::{Int4SourceError, Int4SourceTaskRun, PreparedInt4SourceTask}},
};

impl NlpEngine {
    /// Execute NER, keyphrases, a structurally cited summary, or passage QA.
    /// Preparation and the immutable vocabulary are caller-admitted, as with
    /// schema extraction. Native storage and result ownership belong to the
    /// actual blocking invocation through finalization and physical cleanup.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int4_source_task(&self, model: &ResidentInt4, prepared: PreparedInt4SourceTask,
        vocabulary: Arc<ExtractionVocabulary>, native: NativeLimits,
        mask_limits: MaskWorkLimits, max_mask_node_visits: u64, kernel: PackedInt4Kernel,
        cancellation: CancellationToken) -> Result<HostedOutput<Int4SourceTaskRun>, HostedError> {
        self.execute_int4_source_task_scheduled(model, prepared, vocabulary, native, mask_limits,
            max_mask_node_visits, kernel, None, cancellation)
    }
    /// Opt in to bounded layer-major prompts for NER, keyphrases, summary or
    /// passage QA. The original constrained decode and semantic finalizer run
    /// within the same native session, with one additional prompt memory charge.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int4_source_task_layer_major(&self, model: &ResidentInt4, prepared: PreparedInt4SourceTask,
        vocabulary: Arc<ExtractionVocabulary>, native: NativeLimits,
        mask_limits: MaskWorkLimits, max_mask_node_visits: u64, kernel: PackedInt4Kernel,
        prefill: Int4PrefillLimits, cancellation: CancellationToken)
        -> Result<HostedOutput<Int4SourceTaskRun>, HostedError> {
        self.execute_int4_source_task_scheduled(model, prepared, vocabulary, native, mask_limits,
            max_mask_node_visits, kernel, Some(prefill), cancellation)
    }
    #[allow(clippy::too_many_arguments)]
    fn execute_int4_source_task_scheduled(&self, model: &ResidentInt4, prepared: PreparedInt4SourceTask,
        vocabulary: Arc<ExtractionVocabulary>, native: NativeLimits,
        mask_limits: MaskWorkLimits, max_mask_node_visits: u64, kernel: PackedInt4Kernel,
        prefill: Option<Int4PrefillLimits>, cancellation: CancellationToken)
        -> Result<HostedOutput<Int4SourceTaskRun>, HostedError> {
        dispatch::preflight(self, native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        constrained_int4::check_model(prepared.execution_identity(), model.model_identity())
            .map_err(Int4SourceError::from).map_err(task_error)?;
        if prepared.group_width() != model.model_identity().group_width() {
            return Err(HostedError::ModelIdentity);
        }
        let work = prepared.planned_work();
        let (required, scratch_bytes) = extract::scheduled_requirements(work, prepared.group_width(),
            prepared.max_kv_bytes(), native, mask_limits, max_mask_node_visits, prefill)?;
        let lease = self.resources().acquire_lease();
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
        let workspace = Pending::reserve(&lease, MemoryClass::ActivationScratch, scratch_bytes)?;
        let output = output_claim(&lease, prepared.max_result_bytes(),
            prepared.task_budget().max_output_tokens as u64)?;
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
                    None => prepared.execute_with_control(prepared.execution_identity(), &mut engine.value,
                        &vocabulary, budget, control),
                    Some(limits) => prepared.execute_layer_major_with_control(prepared.execution_identity(),
                        &mut engine.value, &vocabulary, budget, limits, control),
                }.map_err(task_error)?;
                if run.group_width != prepared.group_width() || engine.value.is_poisoned()
                    || !engine.value.kv_cache().all_slots_have_len(0) {
                    return Err(task_error(Int4SourceError::InvalidResult));
                }
                Ok(run)
            })?;
            drop(engine);
            drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }
}

fn task_error(error: Int4SourceError) -> HostedError { HostedError::Int4(Int4HostError::SourceTask(error)) }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_engine::decode::DecodeCancellationKind;

    #[test]
    fn source_task_owners_cross_the_real_blocking_boundary() {
        fn send<T: Send + 'static>() {}
        send::<PreparedInt4SourceTask>();
        send::<Int4SourceTaskRun>();
        send::<Arc<ExtractionVocabulary>>();
        send::<Int4PrefillLimits>();
        let _grouped = NlpEngine::execute_int4_source_task_layer_major;
    }

    #[test]
    fn all_cancellation_causes_survive_the_host_boundary() {
        for cause in [DecodeCancellationKind::User, DecodeCancellationKind::Timeout,
            DecodeCancellationKind::Deadline, DecodeCancellationKind::PollQuota,
            DecodeCancellationKind::CostBudget, DecodeCancellationKind::FailFast,
            DecodeCancellationKind::RaceLost, DecodeCancellationKind::ParentCancelled,
            DecodeCancellationKind::ResourceUnavailable, DecodeCancellationKind::Shutdown,
            DecodeCancellationKind::LinkedExit] {
            let error = task_error(Int4SourceError::Cancelled(cause));
            assert!(matches!(&error, HostedError::Int4(Int4HostError::SourceTask(e))
                if e.cancellation() == Some(cause)));
            assert!(error.source().is_some());
        }
    }

    #[test]
    fn default_diagnostics_do_not_print_private_task_content() {
        let cause = crate::tasks::source_planning::SourcePlanningError::Contract("private passage sentinel");
        let error = Int4HostError::SourceTask(Int4SourceError::Planning(cause));
        assert!(error.source().is_some());
        assert_eq!(error.to_string(), format!("{error:?}"));
        assert_eq!(error.to_string(), "INT4 source task execution or finalization refused");
        assert!(!error.to_string().contains("private passage"));
    }
}
