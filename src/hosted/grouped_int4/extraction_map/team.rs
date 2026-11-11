//! One sealed numerical team across an entire extraction document and reducer.
use super::*;

impl NlpEngine {
    /// Execute every independent schema chunk with one persistent output-row
    /// team. Full-vocabulary and exact selected-row plans, source evidence and
    /// the optional root-array union use their ordinary validators and budgets.
    /// The team joins before the guarded complete document can be returned.
    /// This route does not combine numerical teams with native chunk cohorts.
    #[allow(clippy::too_many_arguments)]
    pub fn extract_int4_document_with_threads(&self, model: &ResidentInt4, source: String,
        planner: Arc<Int4ExtractionBatchPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int4ExtractionMapConfig, kernel: PackedInt4Kernel, compute_threads: usize,
        cancellation: CancellationToken) -> Result<HostedInt4TeamResult<Int4ExtractionMapRun>, HostedError> {
        dispatch::preflight_int4_team(self, config.native.run, compute_threads)?;
        team_requirements(&config, model.model_identity().group_width(), compute_threads)?;
        let run = config.native.run;
        let admitted = admit_document(self, model, source, planner, vocabulary, config)?;
        dispatch::run_int4_team(self, run, cancellation, model.clone(), compute_threads,
            kernel, move |weights, control, team| {
                let output = execute_document(admitted, weights, kernel, control, Some(&mut *team))?;
                // execute_document has dropped its engine before sampling the
                // counters. The dispatcher still owns and joins the same team.
                let compute = team.stats().into();
                Ok(HostedInt4TeamResult::new(output, compute))
            })
    }
}

fn team_requirements(config: &Int4ExtractionMapConfig, group_width: usize, compute_threads: usize)
    -> Result<u64, HostedError> {
    if config.limits.mapping.reduction.map_batch_chunks != 1 {
        return Err(HostedError::Limits("INT4 document teams require serial chunk scheduling"));
    }
    let extra = int4_task_team_workspace_bytes(compute_threads, group_width)?;
    let native = Int4MemoryRequirement::for_context(config.native.context_tokens, group_width).map_err(native_error)?;
    sum(&[native.rope_bytes, native.scratch_payload_bound, config.native.allocator_reserve_bytes, extra])?;
    Ok(extra)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_team_workspace_is_independent_of_document_chunk_and_reducer_counts() {
        let mut config = super::super::tests::config();
        config.limits.mapping.reduction.map_batch_chunks = 1;
        for threads in [2, 3, 4, 64] {
            let expected = int4_task_team_workspace_bytes(threads, 32).unwrap();
            for chunks in [1, 16, 256] {
                config.limits.mapping.chunks.max_chunks = chunks;
                assert_eq!(team_requirements(&config, 32, threads).unwrap(), expected);
            }
        }
        for threads in [0, 1, 65, usize::MAX] {
            assert!(team_requirements(&config, 32, threads).is_err());
        }
        config.limits.mapping.reduction.map_batch_chunks = 2;
        assert!(team_requirements(&config, 32, 2).is_err());
        config.limits.mapping.reduction.map_batch_chunks = 1;
        config.native.allocator_reserve_bytes = u64::MAX;
        assert!(team_requirements(&config, 32, 2).is_err());
    }

    #[test]
    fn complete_document_and_its_compute_receipt_share_the_guarded_result_owner() {
        fn send<T: Send + 'static>() {}
        send::<DocumentAdmission>();
        send::<HostedInt4TeamResult<Int4ExtractionMapRun>>();
        let _entry = NlpEngine::extract_int4_document_with_threads;
    }
}
