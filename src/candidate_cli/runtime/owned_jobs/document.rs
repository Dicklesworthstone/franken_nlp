//! Pinned document-job factory before model loading; no alternate native route.
use super::*;
use crate::{candidate_cli::jobs::document as settings,
    jobs::runner::extract::document::Int8DocumentExtractionJobPlanner};

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare(args: &JobArgs, facts: &ArtifactIdentity, ceiling: TaskBudget,
    defaults: Defaults, text: &str, lifetime: JobLimits, limits: Limits,
    control: &mut impl DecodeStepControl) -> Result<(PreparedJob, Arc<ExtractionVocabulary>), Failure> {
    checkpoint(control)?;
    let Defaults::Extract(Some(request)) = defaults else {
        return Err(Failure::usage("document_job_requires_fixed_schema"));
    };
    let (config, reduction_reserve_bytes) = settings::load(text, args, request, lifetime, limits)?;
    checkpoint(control)?;
    let registry = pinned_controls::pinned().map_err(|_| CandidateError::Identity)?;
    let controls = registry.template_controls();
    let eos = controls.entries().iter().find(|e| e.special && e.surface == crate::template::IM_END)
        .map(|e| e.id).ok_or(CandidateError::Identity)?;
    let mut identity = candidate_identity(facts)?; identity.task_spec = "extract-v1".to_owned();
    let planner = Int8DocumentExtractionJobPlanner::pinned(controls, eos, identity, ceiling,
        schema_cli::compiler(&args.host), args.host.planning().source, config)
        .map_err(|_| CandidateError::Planning)?;
    planner.check_job_limits(lifetime).map_err(|_| Failure::usage("document_job_lifetime_limits"))?;
    checkpoint(control)?;
    let vocabulary = Arc::new(ExtractionVocabulary::pinned(controls).map_err(|_| CandidateError::Planning)?);
    checkpoint(control)?;
    Ok((PreparedJob::Document { planner, reduction_reserve_bytes }, vocabulary))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn facts() -> ArtifactIdentity {
        ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(), revision: "fixture".to_owned(),
            recipe_id: "nanbeige42-int8-v1".to_owned(), source_root_sha256: "0".repeat(64), logical_model_sha256: "1".repeat(64) }
    }
    struct Continue;
    impl DecodeStepControl for Continue {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
    }
    #[test]
    fn missing_document_configuration_cannot_silently_fall_back_to_one_context() {
        let args = command::tests::args("extract", &["--schema", "s", "--document-config", "d", "--preparation-mib", "1024"]);
        let (_, limits) = args.validate().unwrap(); let lifetime = command::tests::lifetime();
        let error = super::super::prepare(&args, &facts(), args.host.task_budget(limits), Defaults::Extract(None),
            command::native_limits(&args, lifetime).unwrap(), &mut Continue).err().unwrap();
        assert_eq!(error.code, "document_job_configuration_missing");
        assert!(prepare(&args, &facts(), args.host.task_budget(limits), Defaults::Extract(None), "{}",
            lifetime, limits, &mut Continue).is_err());
    }
    #[test]
    fn cancellation_wins_before_settings_compilation_or_vocabulary_allocation() {
        struct Stop;
        impl DecodeStepControl for Stop {
            fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::Deadline) }
        }
        let args = command::tests::args("extract", &["--schema", "s", "--document-config", "d"]);
        let (_, limits) = args.validate().unwrap();
        let error = prepare(&args, &facts(), args.host.task_budget(limits), Defaults::Extract(None), "private invalid data",
            command::tests::lifetime(), limits, &mut Stop).err().unwrap();
        assert_eq!(error.code, "job_execution_budget");
    }
}
