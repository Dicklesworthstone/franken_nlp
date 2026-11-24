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
    let reconciliation = args.reconciliation.limits(config.limits.mapping.reduction.max_total_value_bytes,
        config.limits.mapping.reduction.max_result_bytes)?;
    checkpoint(control)?;
    let registry = pinned_controls::pinned().map_err(|_| CandidateError::Identity)?;
    let controls = registry.template_controls();
    let eos = controls.entries().iter().find(|e| e.special && e.surface == crate::template::IM_END)
        .map(|e| e.id).ok_or(CandidateError::Identity)?;
    let mut identity = candidate_identity(facts)?; identity.task_spec = "extract-v1".to_owned();
    let planner = Int8DocumentExtractionJobPlanner::pinned(controls, eos, identity, ceiling,
        schema_cli::compiler(&args.host), args.host.planning().source, config)
        .map_err(|_| CandidateError::Planning)?
        .with_document_encoding(args.encoding_policy(limits)?, args.encoding_work_limits()?)
        .map_err(|_| CandidateError::Planning)?;
    checkpoint(control)?;
    let planner = match reconciliation {
        Some(reconciliation) => planner.with_root_object_reconciliation(reconciliation)
            .map_err(|_| CandidateError::Planning)?,
        None => planner,
    };
    planner.check_job_limits(lifetime).map_err(|_| Failure::usage("document_job_lifetime_limits"))?;
    checkpoint(control)?;
    let vocabulary = Arc::new(ExtractionVocabulary::pinned(controls).map_err(|_| CandidateError::Planning)?);
    checkpoint(control)?;
    Ok((PreparedJob::Document { planner, reduction_reserve_bytes }, vocabulary))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonjson;
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
    fn configured_factory(args: &JobArgs) -> (PreparedJob, Limits) {
        let (_, limits) = args.validate().unwrap();
        let mut lifetime = command::tests::lifetime(); lifetime.max_result_bytes = 4 << 20;
        let settings = serde_json::json!({"schema_version":1,"max_chunks":4,"max_chunk_bytes":16,
            "max_tokenizer_calls":1024,"max_result_bytes":4 << 20,"max_live_value_bytes":8 << 20,
            "max_total_value_bytes":16 << 20,"reduction_reserve_mib":64,
            "verification":{"max_fields":17,"max_matches":19,"max_scan_steps":23}}).to_string();
        let schema = r#" {"type":"number","const":12345678901234567890123456789012345678} "#;
        let request = schema_cli::arguments(schema.to_owned(), false, args.host.task_budget(limits)).unwrap();
        let (prepared, _) = prepare(args, &facts(), args.host.task_budget(limits), Defaults::Extract(Some(request)),
            &settings, lifetime, limits, &mut Continue).unwrap();
        (prepared, limits)
    }
    fn document_options(extra: &[&str]) -> JobArgs {
        let mut flags = vec!["--schema", "s", "--document-config", "d", "--preparation-mib", "1024"];
        flags.extend_from_slice(extra);
        let mut args = command::tests::args("extract", &flags); args.host.memory_mib = 16384; args
    }
    #[test]
    fn compact_document_factory_binds_cli_quotes_caps_and_recipe_across_all_schedules() {
        let mut baseline = None;
        for schedule in [&[][..], &["--compute-threads", "4"][..],
            &["--prefill-rows", "7"][..], &["--compute-threads", "4", "--prefill-rows", "7"][..]] {
            let mut flags = vec!["--input-encoding", "safe-bpe-v1"]; flags.extend_from_slice(schedule);
            let args = document_options(&flags);
            let (prepared, limits) = configured_factory(&args);
            let encoding = args.encoding_policy(limits).unwrap();
            let total = args.encoding_work_limits().unwrap();
            let workspace = args.encoding_workspace_bytes(limits).unwrap();
            check_encoding_quote(&prepared, &encoding, total, workspace).unwrap();
            assert_eq!(check_encoding_quote(&prepared, &encoding, total, workspace - 1).unwrap_err().code,
                "job_encoding_quote_mismatch");
            assert!(check_encoding_quote(&prepared, &UntrustedEncoding::ByteFallback, None, 0).is_err());
            let mut wrong = total.unwrap(); wrong.max_pair_bytes += 1;
            assert!(check_encoding_quote(&prepared, &encoding, Some(wrong), workspace).is_err());
            let PreparedJob::Document { planner, reduction_reserve_bytes } = prepared
                else { panic!("wrong document factory") };
            assert_eq!(reduction_reserve_bytes, 64 * MIB);
            assert_eq!(planner.document_budget(), args.host.task_budget(limits));
            assert_eq!(planner.encoding_work().unwrap(), Some(crate::tokenizer::untrusted::SafeBpeWork::default()));
            let recipe = canonjson::canonical_bytes(planner.job_recipe()).unwrap();
            let actual = (recipe, planner.execution_identity().clone(), planner.document_work());
            if let Some(baseline) = &baseline { assert_eq!(&actual, baseline); }
            else { baseline = Some(actual); }
        }
        let byte = document_options(&[]);
        let (prepared, limits) = configured_factory(&byte);
        check_encoding_quote(&prepared, &UntrustedEncoding::ByteFallback, None, 0).unwrap();
        let PreparedJob::Document { planner, .. } = prepared else { panic!("wrong byte factory") };
        assert_eq!(byte.encoding_workspace_bytes(limits).unwrap(), 0);
        let value = serde_json::to_value(planner.job_recipe()).unwrap();
        assert!(value["extraction"].get("input_encoding").is_none());
        assert!(value["extraction"].get("encoding_work_limits").is_none());
        assert_ne!(canonjson::canonical_bytes(planner.job_recipe()).unwrap(), baseline.unwrap().0);
    }
    #[test]
    fn compact_document_caps_change_the_authenticated_recipe_without_expanding_native_work() {
        let first = document_options(&["--input-encoding", "safe-bpe-v1"]);
        let changed = document_options(&["--input-encoding", "safe-bpe-v1",
            "--max-total-tokenizer-pair-bytes", "4294967297"]);
        let (first, _) = configured_factory(&first); let (changed, _) = configured_factory(&changed);
        let PreparedJob::Document { planner: first, .. } = first else { panic!("document route") };
        let PreparedJob::Document { planner: changed, .. } = changed else { panic!("document route") };
        assert_eq!(first.document_work(), changed.document_work());
        assert_eq!(first.document_budget(), changed.document_budget());
        assert_eq!(first.encoding_workspace_bytes().unwrap(), changed.encoding_workspace_bytes().unwrap());
        assert_ne!(canonjson::canonical_bytes(first.job_recipe()).unwrap(),
            canonjson::canonical_bytes(changed.job_recipe()).unwrap());
        assert_ne!(first.execution_identity(), changed.execution_identity());
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

    const OBJECT_SCHEMA: &str = r#" {"type":"object","additionalProperties":false,"required":["n"],"properties":{"n":{"type":"number","const":9007199254740993}}} "#;
    fn object_limits() -> crate::tasks::extract::object_reconcile::RootObjectLimits {
        crate::tasks::extract::object_reconcile::RootObjectLimits { max_properties: 16, max_origins: 32,
            max_input_bytes: 1 << 20, max_json_bytes: 8192, max_nodes: 1024, max_result_bytes: 32768 }
    }
    fn object_options(bounds: crate::tasks::extract::object_reconcile::RootObjectLimits, extra: &[&str]) -> JobArgs {
        let values = [bounds.max_properties.to_string(), bounds.max_origins.to_string(),
            bounds.max_input_bytes.to_string(), bounds.max_json_bytes.to_string(),
            bounds.max_nodes.to_string(), bounds.max_result_bytes.to_string()];
        let mut flags: Vec<&str> = vec!["--reconcile-object", "--max-object-properties", &values[0],
            "--max-object-origins", &values[1], "--max-object-input-bytes", &values[2],
            "--max-object-json-bytes", &values[3], "--max-object-nodes", &values[4],
            "--max-object-result-bytes", &values[5]];
        flags.extend_from_slice(extra);
        document_options(&flags)
    }
    fn schema_factory(args: &JobArgs, schema: &str, control: &mut impl DecodeStepControl)
        -> Result<(PreparedJob, Limits), Failure> {
        let (_, limits) = args.validate()?;
        let mut lifetime = command::tests::lifetime(); lifetime.max_result_bytes = 4 << 20;
        let settings = serde_json::json!({"schema_version":1,"max_chunks":4,"max_chunk_bytes":16,
            "max_tokenizer_calls":1024,"max_result_bytes":4 << 20,"max_live_value_bytes":8 << 20,
            "max_total_value_bytes":16 << 20,"reduction_reserve_mib":64,
            "verification":{"max_fields":17,"max_matches":19,"max_scan_steps":23}}).to_string();
        let ceiling = args.host.task_budget(limits);
        // Exercise the real CLI default loader too: it must preserve original
        // bytes and defer the full required set to the final object byte cap.
        let defaults = command::load_defaults(args, ceiling, None, Some(schema.to_owned()))?;
        let (prepared, _) = prepare(args, &facts(), ceiling, defaults, &settings, lifetime, limits, control)?;
        Ok((prepared, limits))
    }

    #[test]
    fn object_factory_accepts_a_final_required_object_larger_than_a_native_chunk() {
        use crate::grammar::runtime::JsonProgram;
        use crate::tasks::extract::object_reconcile::RootObjectSchema;
        let args = object_options(object_limits(), &["--max-result-bytes", "2048", "--context-tokens", "4096"]);
        let value = "x".repeat(768);
        let schema = format!(r#" {{"type":"object","additionalProperties":false,"required":["a","b","c"],"properties":{{"a":{{"type":"string","const":"{value}"}},"b":{{"type":"string","const":"{value}"}},"c":{{"type":"string","const":"{value}"}}}}}} "#);
        let compiler = schema_cli::compiler(&args.host);
        assert!(JsonProgram::compile(&schema, compiler).is_err());
        let partial = RootObjectSchema::derive(&schema, compiler, object_limits(), &mut Continue).unwrap();
        assert!(JsonProgram::compile(partial.partial_schema(), compiler).is_ok());
        let (prepared, limits) = schema_factory(&args, &schema, &mut Continue).unwrap();
        let selection = root_object_selection(&prepared).unwrap();
        assert_eq!(serde_json::to_value(selection).unwrap(),
            serde_json::to_value(RootObjectSelection::new(object_limits())).unwrap());
        let PreparedJob::Document { planner, .. } = prepared else { panic!("wrong object factory") };
        assert_eq!(planner.request().schema, schema);
        assert_eq!(planner.document_budget().max_output_bytes, 2048);
        assert_eq!(planner.document_budget(), args.host.task_budget(limits));
        assert_eq!(serde_json::to_value(planner.job_recipe()).unwrap()["request"]["schema"], schema);
        let ordinary = document_options(&["--max-result-bytes", "2048", "--context-tokens", "4096"]);
        assert!(schema_factory(&ordinary, &schema, &mut Continue).is_err());
    }

    #[test]
    fn actual_object_factory_and_metadata_compose_with_all_encoding_and_physical_schedules() {
        for compact in [false, true] {
            let mut baseline = None;
            for schedule in [&[][..], &["--compute-threads", "4"][..],
                &["--prefill-rows", "7"][..], &["--compute-threads", "4", "--prefill-rows", "7"][..]] {
                let mut extra = Vec::new();
                if compact { extra.extend_from_slice(&["--input-encoding", "safe-bpe-v1"]); }
                extra.extend_from_slice(schedule);
                let args = object_options(object_limits(), &extra);
                let (prepared, limits) = schema_factory(&args, OBJECT_SCHEMA, &mut Continue).unwrap();
                let encoding = args.encoding_policy(limits).unwrap();
                check_encoding_quote(&prepared, &encoding, args.encoding_work_limits().unwrap(),
                    args.encoding_workspace_bytes(limits).unwrap()).unwrap();
                let selection = root_object_selection(&prepared).unwrap();
                let PreparedJob::Document { planner, reduction_reserve_bytes } = prepared
                    else { panic!("wrong object factory") };
                let actual_limits = planner.root_object_reconciliation().unwrap();
                assert_eq!(actual_limits, object_limits());
                assert_eq!(serde_json::to_value(selection).unwrap(),
                    serde_json::to_value(RootObjectSelection::new(actual_limits)).unwrap());
                assert_eq!(planner.request().schema, OBJECT_SCHEMA);
                assert_eq!(reduction_reserve_bytes, 64 * MIB);
                assert_eq!(planner.encoding_work().unwrap(),
                    compact.then_some(crate::tokenizer::untrusted::SafeBpeWork::default()));
                let recipe = canonjson::canonical_bytes(planner.job_recipe()).unwrap();
                let actual = (recipe, planner.execution_identity().clone(), planner.document_work());
                if let Some(baseline) = &baseline { assert_eq!(&actual, baseline); }
                else { baseline = Some(actual); }
            }
        }
        let (ordinary, _) = schema_factory(&document_options(&[]), OBJECT_SCHEMA, &mut Continue).unwrap();
        assert!(root_object_selection(&ordinary).is_none());
        let PreparedJob::Document { planner, .. } = ordinary else { panic!("wrong ordinary factory") };
        assert!(serde_json::to_value(planner.job_recipe()).unwrap().get("root_object_reconciliation").is_none());
    }

    #[test]
    fn actual_object_factory_authenticates_each_reducer_cap_without_increasing_native_work() {
        let (first, _) = schema_factory(&object_options(object_limits(), &[]), OBJECT_SCHEMA, &mut Continue).unwrap();
        let PreparedJob::Document { planner: first, .. } = first else { panic!("wrong object factory") };
        let original_recipe = canonjson::canonical_bytes(first.job_recipe()).unwrap();
        for axis in 0..6 {
            let mut changed = object_limits();
            match axis { 0 => changed.max_properties += 1, 1 => changed.max_origins += 1,
                2 => changed.max_input_bytes += 1, 3 => changed.max_json_bytes += 1,
                4 => changed.max_nodes += 1, _ => changed.max_result_bytes += 1 }
            let (prepared, _) = schema_factory(&object_options(changed, &[]), OBJECT_SCHEMA, &mut Continue).unwrap();
            let selection = root_object_selection(&prepared).unwrap();
            assert_eq!(serde_json::to_value(selection).unwrap(),
                serde_json::to_value(RootObjectSelection::new(changed)).unwrap(), "axis {axis}");
            let PreparedJob::Document { planner, .. } = prepared else { panic!("wrong changed factory") };
            assert_ne!(canonjson::canonical_bytes(planner.job_recipe()).unwrap(), original_recipe, "axis {axis}");
            // Reducer limits change the authenticated recipe, not the pinned
            // native prompt, task budget or durable attempt debit.
            assert_eq!(planner.execution_identity(), first.execution_identity(), "axis {axis}");
            assert_eq!(planner.document_budget(), first.document_budget(), "axis {axis}");
            assert_eq!(planner.document_work(), first.document_work(), "axis {axis}");
        }
    }

    #[test]
    fn object_factory_refuses_wrong_shape_or_grounding_and_preserves_config_boundary_cancellation() {
        let args = object_options(object_limits(), &[]);
        for schema in [r#"{"type":"boolean"}"#,
            r#"{"type":"array","items":{"type":"boolean"},"maxItems":2}"#,
            r#"{"type":"object","properties":{},"additionalProperties":true}"#] {
            assert!(schema_factory(&args, schema, &mut Continue).is_err());
        }
        let annotated = r#"{"type":"object","properties":{"name":{"type":"string","maxLength":32,"x-fnlp-source":"verbatim"}},"required":["name"],"additionalProperties":false}"#;
        assert!(schema_factory(&args, annotated, &mut Continue).is_err());
        let grounded = object_options(object_limits(), &["--source-membership"]);
        assert!(schema_factory(&grounded, OBJECT_SCHEMA, &mut Continue).is_err());
        assert!(schema_factory(&grounded, annotated, &mut Continue).is_ok());
        struct StopAt { calls: usize, stop: usize, cause: DecodeCancellationKind }
        impl DecodeStepControl for StopAt {
            fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
                self.calls += 1;
                (self.calls == self.stop).then_some(self.cause)
            }
        }
        for cause in [DecodeCancellationKind::User, DecodeCancellationKind::Timeout,
            DecodeCancellationKind::Deadline, DecodeCancellationKind::PollQuota,
            DecodeCancellationKind::CostBudget, DecodeCancellationKind::FailFast,
            DecodeCancellationKind::RaceLost, DecodeCancellationKind::ParentCancelled,
            DecodeCancellationKind::ResourceUnavailable, DecodeCancellationKind::Shutdown,
            DecodeCancellationKind::LinkedExit] {
            // Before settings, immediately before object derivation, and after
            // it but before vocabulary allocation all retain the original cause.
            for stop in [1, 3, 4] {
                let error = schema_factory(&args, OBJECT_SCHEMA,
                    &mut StopAt { calls: 0, stop, cause }).err().unwrap();
                let expected = cancelled(cause);
                assert_eq!(error.exit, expected.exit, "{cause:?} at checkpoint {stop}");
                assert_eq!(error.code, expected.code, "{cause:?} at checkpoint {stop}");
            }
        }
    }
}
