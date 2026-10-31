//! Whole-source preflight followed by one guarded native map, summary or QA.
use super::*;
use std::{sync::Arc, time::{Duration, Instant}};
use crate::{
    NlpEngine, ResourceHostConfig, RuntimePreset, LeakResponsePolicy, MemoryClass,
    execution_identity::ExecutionIdentity,
    hosted::{CancellationToken, HostedError, Int4HostError, Int4SourceMapConfig, NativeLimits, RunLimits},
    native_engine::{decode::{DecodeCancellationKind, DecodeStepControl},
        strict_int4::weights::snapshot::INT4_SNAPSHOT_VERSION},
    tasks::{ir::PlanContext, source_planning::{SourceTaskPlanner,
        grouped_int4::long::{Int4SourceMapError, Int4SourceMapPreflight,
            cohort::{Int4SourceMapCohortStats, hierarchy::Int4HierarchyCohortStats},
            question::{SourceQuestion, Int4QuestionPreflight,
                synthesis::{SourceQuestionSynthesis, Int4QuestionSynthesisPreflight,
                    hierarchy::{QuestionHierarchyLimits, Int4QuestionHierarchyPreflight}}},
            summary::synthesis::{SourceSummarySynthesis, Int4SummarySynthesisPreflight,
                hierarchy::{SummaryHierarchyLimits, Int4SummaryHierarchyPreflight}}}}},
};

enum Request {
    Map(SourceMapTask), Question(SourceQuestion), SummarySynthesis(SourceSummarySynthesis),
    QuestionSynthesis(SourceQuestionSynthesis),
}
enum Prepared {
    Map(Int4SourceMapConfig, Int4SourceMapPreflight),
    Question(Int4SourceMapConfig<SourceQuestion>, Int4QuestionPreflight),
    SummarySynthesis(Int4SourceMapConfig<SourceSummarySynthesis>, Int4SummarySynthesisPreflight),
    SummaryHierarchy(Int4SourceMapConfig<SourceSummarySynthesis>, Int4SummaryHierarchyPreflight),
    QuestionSynthesis(Int4SourceMapConfig<SourceQuestionSynthesis>, Int4QuestionSynthesisPreflight),
    QuestionHierarchy(Int4SourceMapConfig<SourceQuestionSynthesis>, Int4QuestionHierarchyPreflight),
}
fn configuration<T>(identity: ExecutionIdentity, task: T, args: &Int4Args,
    options: &MapArgs, limits: MapLimits, elapsed: Duration) -> Int4SourceMapConfig<T> {
    Int4SourceMapConfig { identity, task, budget: options.source.budget(args, limits.common),
        planning: options.planning(args), mapping: limits.mapping,
        native: NativeLimits { context_tokens: args.context_tokens, allocator_reserve_bytes: ALLOCATOR_BYTES,
            run: RunLimits { max_elapsed: elapsed, max_checkpoints: args.max_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES } },
        preparation_reserve_bytes: limits.common.preparation_bytes,
        reduction_reserve_bytes: limits.reduction_reserve_bytes }
}
struct DeadlineControl { started: Instant, deadline: Duration }
impl DecodeStepControl for DeadlineControl {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        (self.started.elapsed() >= self.deadline).then_some(DecodeCancellationKind::Deadline)
    }
}
fn preflight(source: &str, planner: &SourceTaskPlanner, config: &Int4SourceMapConfig,
    width: usize, summary: Option<Int4SummaryLimits>, control: &mut impl DecodeStepControl)
    -> Result<Int4SourceMapPreflight, CandidateError> {
    if source.is_empty() { return Err(CandidateError::Input); }
    let context = PlanContext::new(&config.identity, config.budget).map_err(|_| CandidateError::Planning)?;
    let prepared = planner.plan_int4_map_with_control(source, &config.task, config.budget, &context,
        config.planning, config.mapping, width, control).map_err(planning_error)?;
    if let Some(summary) = summary { prepared.check_summary(summary).map_err(|_| CandidateError::Planning)?; }
    // Only bounded geometry/work metadata survives. The host replans the same
    // original document under its own charge before native work can start.
    Ok(prepared.preflight_metadata())
}
fn keyphrase_preflight(source: &str, planner: &SourceTaskPlanner, config: &Int4SourceMapConfig,
    width: usize, aggregation: Int4KeyphraseLimits, control: &mut impl DecodeStepControl)
    -> Result<Int4SourceMapPreflight, CandidateError> {
    if source.is_empty() { return Err(CandidateError::Input); }
    let context = PlanContext::new(&config.identity, config.budget).map_err(|_| CandidateError::Planning)?;
    let prepared = planner.plan_int4_map_with_control(source, &config.task, config.budget, &context,
        config.planning, config.mapping, width, control).map_err(planning_error)?;
    prepared.check_keyphrases(aggregation).map_err(|_| CandidateError::Planning)?;
    // Final top-k changes only publication; every original chunk remains in
    // this preflight and the complete candidate union survives reduction.
    Ok(prepared.preflight_metadata())
}
fn question_preflight(source: &str, planner: &SourceTaskPlanner,
    config: &Int4SourceMapConfig<SourceQuestion>, width: usize, control: &mut impl DecodeStepControl)
    -> Result<Int4QuestionPreflight, CandidateError> {
    if source.is_empty() { return Err(CandidateError::Input); }
    let context = PlanContext::new(&config.identity, config.budget).map_err(|_| CandidateError::Planning)?;
    planner.preflight_int4_question_with_control(source, &config.task, config.budget, &context,
        config.planning, config.mapping, width, control).map_err(planning_error)
}
fn summary_synthesis_preflight(source: &str, planner: &SourceTaskPlanner,
    config: &Int4SourceMapConfig<SourceSummarySynthesis>, width: usize, control: &mut impl DecodeStepControl)
    -> Result<Int4SummarySynthesisPreflight, CandidateError> {
    if source.is_empty() { return Err(CandidateError::Input); }
    let context = PlanContext::new(&config.identity, config.budget).map_err(|_| CandidateError::Planning)?;
    let prepared = planner.plan_int4_summary_synthesis_with_control(source, config.task, config.budget,
        &context, config.planning, config.mapping, width, control).map_err(planning_error)?;
    // Every real map and the maximum-context final pass are reserved together.
    // The generated quote document is checked separately once it exists.
    Ok(prepared.preflight_metadata())
}
fn question_synthesis_preflight(source: &str, planner: &SourceTaskPlanner,
    config: &Int4SourceMapConfig<SourceQuestionSynthesis>, width: usize, control: &mut impl DecodeStepControl)
    -> Result<Int4QuestionSynthesisPreflight, CandidateError> {
    if source.is_empty() { return Err(CandidateError::Input); }
    let context = PlanContext::new(&config.identity, config.budget).map_err(|_| CandidateError::Planning)?;
    planner.preflight_int4_question_synthesis_with_control(source, &config.task, config.budget,
        &context, config.planning, config.mapping, width, control).map_err(planning_error)
}
fn summary_hierarchy_preflight(source: &str, planner: &SourceTaskPlanner,
    config: &Int4SourceMapConfig<SourceSummarySynthesis>, hierarchy: SummaryHierarchyLimits,
    width: usize, control: &mut impl DecodeStepControl)
    -> Result<Int4SummaryHierarchyPreflight, CandidateError> {
    if source.is_empty() { return Err(CandidateError::Input); }
    let context = PlanContext::new(&config.identity, config.budget).map_err(|_| CandidateError::Planning)?;
    let prepared = planner.plan_int4_summary_hierarchy_with_control(source, config.task, hierarchy,
        config.budget, &context, config.planning, config.mapping, width, control).map_err(planning_error)?;
    // Every permitted hierarchy pass is reserved before the first map starts.
    // Complete future levels are planned when their exact quoted evidence exists.
    Ok(prepared.preflight_metadata())
}
fn question_hierarchy_preflight(source: &str, planner: &SourceTaskPlanner,
    config: &Int4SourceMapConfig<SourceQuestionSynthesis>, hierarchy: QuestionHierarchyLimits,
    width: usize, control: &mut impl DecodeStepControl)
    -> Result<Int4QuestionHierarchyPreflight, CandidateError> {
    if source.is_empty() { return Err(CandidateError::Input); }
    let context = PlanContext::new(&config.identity, config.budget).map_err(|_| CandidateError::Planning)?;
    // Complete discovery and EVERY permitted answer pass share one reservation.
    // Future groups also check the exact question and passage manifest.
    planner.preflight_int4_question_hierarchy_with_control(source, &config.task, hierarchy,
        config.budget, &context, config.planning, config.mapping, width, control).map_err(planning_error)
}
fn check_synthesis_schedule<T>(config: &Int4SourceMapConfig<T>, limits: MapLimits, width: usize)
    -> Result<(), CandidateError> {
    let quote = match limits.hierarchy_cohorts {
        Some(schedule) => {
            if schedule.discovery_cohorts != limits.cohort_chunks.is_some() { return Err(CandidateError::Memory); }
            Some(config.hierarchy_cohort_requirements(width, schedule).map_err(|_| CandidateError::Memory)?)
        },
        None => limits.cohort_chunks.map(|_| config.synthesis_cohort_requirements(width))
            .transpose().map_err(|_| CandidateError::Memory)?,
    };
    let preparation = limits.hierarchy_cohorts.map(|schedule|
        schedule.additional_preparation_bytes(config.preparation_reserve_bytes))
        .transpose().map_err(|_| CandidateError::Memory)?.unwrap_or(0);
    if quote != limits.cohort || preparation != limits.hierarchy_preparation_bytes {
        return Err(CandidateError::Memory);
    }
    Ok(())
}
fn cancelled(cause: DecodeCancellationKind) -> CandidateError {
    match cause {
        DecodeCancellationKind::Deadline | DecodeCancellationKind::Timeout => CandidateError::Timeout,
        _ => CandidateError::Execution,
    }
}
fn planning_error(error: Int4SourceMapError) -> CandidateError {
    error.cancellation().map(cancelled).unwrap_or(CandidateError::Planning)
}
fn execution_error(error: HostedError) -> CandidateError {
    match error {
        HostedError::Stopped { stop, .. } => cancelled(stop.kind),
        HostedError::Join { physical_error: Some(error), .. }
            | HostedError::Scope { physical_error: Some(error), .. } => execution_error(*error),
        HostedError::Int4(Int4HostError::SourceMap(error)) =>
            error.cancellation().map(cancelled).unwrap_or(CandidateError::Execution),
        HostedError::Int4(Int4HostError::SourceSummary(error)) =>
            error.cancellation().map(cancelled).unwrap_or(CandidateError::Execution),
        HostedError::Int4(Int4HostError::SourceKeyphrases(error)) =>
            error.cancellation().map(cancelled).unwrap_or(CandidateError::Execution),
        HostedError::Reservation(_) => CandidateError::Memory,
        HostedError::ModelIdentity | HostedError::ResourceDomain => CandidateError::Identity,
        _ => CandidateError::Execution,
    }
}
pub(super) fn execute(args: Int4Args, options: MapArgs, _admitted: MapLimits,
    input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
    let started = Instant::now();
    let limits = options.validate(&args)?;
    let common = limits.common;
    let deadline = Duration::from_secs(args.timeout_seconds);
    let remaining = || deadline.checked_sub(started.elapsed()).filter(|time| !time.is_zero())
        .ok_or(CandidateError::Timeout);
    let engine = NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 0,
        helper_threads: 0, thread_ceiling: 2, memory_ceiling_bytes: common.memory_bytes,
        leak_response_policy: LeakResponsePolicy::RecordAndEscalate,
    }).build().map_err(|_| CandidateError::Runtime)?;
    let lease = engine.resources().acquire_lease();
    let _preparation = lease.reserve(MemoryClass::JobBuffers, common.preparation_bytes)
        .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
    remaining()?;
    let option_json = options.source.options_path().map(|path| {
        read_input(&mut crate::local_io::open_document(path).map_err(|_| CandidateError::Input)?,
            shared::OPTIONS_BYTES)
    }).transpose()?;
    remaining()?;
    let text = if args.input.as_os_str() == "-" { read_input(input, args.max_input_bytes)? }
        else { read_input(&mut crate::local_io::open_document(&args.input)
            .map_err(|_| CandidateError::Input)?, args.max_input_bytes)? };
    remaining()?;
    let budget = options.source.budget(&args, common);
    let (text, task) = if let Some(question) = &options.question {
        if text.trim().is_empty() { return Err(CandidateError::Input); }
        let question_text = read_input(&mut crate::local_io::open_document(question.path())
            .map_err(|_| CandidateError::Input)?, question::QUESTION_BYTES)?;
        remaining()?;
        let task = question.task(question_text, option_json.as_deref())?;
        let task = if limits.question_synthesis.is_some() {
            Request::QuestionSynthesis(question.synthesis_request(task, options.planning(&args))?)
        } else { Request::Question(task) };
        (text, task)
    } else {
        let (text, task) = options.request(text, option_json.as_deref(), budget, args.max_input_bytes)?;
        let task = match (options.summary, task) {
            (Some(summary), SourceMapTask::Summarize(map_options)) if summary.synthesis.synthesize_summary =>
                Request::SummarySynthesis(summary.synthesis.request(map_options, options.planning(&args))?),
            (_, task) => Request::Map(task),
        };
        (text, task)
    };
    drop(option_json);
    remaining()?;
    let cancellation = CancellationToken::default();
    let model = super::super::runtime::load_model(&engine, &args, common, remaining()?, cancellation.clone())?;
    remaining()?;
    let (planner, vocabulary) = source::planner()?;
    let identity = source::identity(model.model_identity(), &planner, source::task(options.kind))?;
    let prepared = match task {
        Request::Map(task) => {
            let config = configuration(identity.clone(), task, &args, &options, limits, remaining()?);
            if let Some(quote) = limits.cohort {
                if config.cohort_requirements(args.group_width).map_err(|_| CandidateError::Memory)? != quote {
                    return Err(CandidateError::Memory);
                }
            }
            let expected = if let Some(aggregation) = limits.keyphrases {
                if config.keyphrase_workspace_bytes(limits.mapping.chunks.max_chunks, aggregation)
                    .map_err(|_| CandidateError::Memory)? != limits.workspace_bytes {
                    return Err(CandidateError::Memory);
                }
                keyphrase_preflight(&text, &planner, &config, model.model_identity().group_width(),
                    aggregation, &mut DeadlineControl { started, deadline })?
            } else {
                if config.workspace_bytes(limits.mapping.chunks.max_chunks).map_err(|_| CandidateError::Memory)?
                    != limits.workspace_bytes { return Err(CandidateError::Memory); }
                preflight(&text, &planner, &config, model.model_identity().group_width(),
                    limits.summary, &mut DeadlineControl { started, deadline })?
            };
            Prepared::Map(config, expected)
        }
        Request::Question(task) => {
            let config = configuration(identity.clone(), task, &args, &options, limits, remaining()?);
            if let Some(quote) = limits.cohort {
                if config.cohort_requirements(args.group_width).map_err(|_| CandidateError::Memory)? != quote {
                    return Err(CandidateError::Memory);
                }
            }
            if config.question_workspace_bytes(limits.mapping.chunks.max_chunks)
                .map_err(|_| CandidateError::Memory)? != limits.workspace_bytes {
                return Err(CandidateError::Memory);
            }
            let expected = question_preflight(&text, &planner, &config,
                model.model_identity().group_width(), &mut DeadlineControl { started, deadline })?;
            Prepared::Question(config, expected)
        }
        Request::SummarySynthesis(task) => {
            let config = configuration(identity.clone(), task, &args, &options, limits, remaining()?);
            check_synthesis_schedule(&config, limits, args.group_width)?;
            if let Some(hierarchy) = limits.hierarchy {
                if config.summary_hierarchy_workspace_bytes(limits.mapping.chunks.max_chunks, hierarchy)
                    .map_err(|_| CandidateError::Memory)? != limits.workspace_bytes {
                    return Err(CandidateError::Memory);
                }
                let expected = summary_hierarchy_preflight(&text, &planner, &config, hierarchy,
                    model.model_identity().group_width(), &mut DeadlineControl { started, deadline })?;
                Prepared::SummaryHierarchy(config, expected)
            } else {
                if config.summary_synthesis_workspace_bytes(limits.mapping.chunks.max_chunks)
                    .map_err(|_| CandidateError::Memory)? != limits.workspace_bytes {
                    return Err(CandidateError::Memory);
                }
                let expected = summary_synthesis_preflight(&text, &planner, &config,
                    model.model_identity().group_width(), &mut DeadlineControl { started, deadline })?;
                Prepared::SummarySynthesis(config, expected)
            }
        }
        Request::QuestionSynthesis(task) => {
            let config = configuration(identity.clone(), task, &args, &options, limits, remaining()?);
            check_synthesis_schedule(&config, limits, args.group_width)?;
            if let Some(hierarchy) = limits.question_hierarchy {
                if config.question_hierarchy_workspace_bytes(limits.mapping.chunks.max_chunks, hierarchy)
                    .map_err(|_| CandidateError::Memory)? != limits.workspace_bytes {
                    return Err(CandidateError::Memory);
                }
                let expected = question_hierarchy_preflight(&text, &planner, &config, hierarchy,
                    model.model_identity().group_width(), &mut DeadlineControl { started, deadline })?;
                Prepared::QuestionHierarchy(config, expected)
            } else {
                if config.question_synthesis_workspace_bytes(limits.mapping.chunks.max_chunks)
                    .map_err(|_| CandidateError::Memory)? != limits.workspace_bytes {
                    return Err(CandidateError::Memory);
                }
                let expected = question_synthesis_preflight(&text, &planner, &config,
                    model.model_identity().group_width(), &mut DeadlineControl { started, deadline })?;
                Prepared::QuestionSynthesis(config, expected)
            }
        }
    };
    remaining()?;
    let mut snapshot = common.snapshot_pin.map(|_| SnapshotRecord::new("loaded", common.snapshot_need));
    if let Some(path) = &args.save_snapshot {
        // Persist only after the COMPLETE original source maps and, if enabled,
        // the maximum-context synthesis work preflight. Generated evidence fit
        // is checked later. This complete model file survives task failure.
        let pin = engine.save_int4_snapshot(&model, path.clone(), common.snapshot_budget(false)?,
            RunLimits { max_elapsed: remaining()?, max_checkpoints: args.snapshot_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES }, cancellation.clone()).map_err(|_| CandidateError::Model)?;
        if pin != model.model_identity().snapshot_pin() { return Err(CandidateError::Identity); }
        snapshot = Some(SnapshotRecord::new("saved", common.snapshot_need));
    }
    let content = identity.logical_model_digest.to_hex();
    let provenance = Provenance {
        protocol: if limits.question_hierarchy.is_some() { "fnlp-candidate-int4-answer-hierarchy-v1" }
            else if limits.question_synthesis.is_some() { "fnlp-candidate-int4-answer-synthesis-v1" }
            else if options.kind == Kind::Answer { "fnlp-candidate-int4-document-qa-v1" }
            else if limits.hierarchy.is_some() { "fnlp-candidate-int4-summary-hierarchy-v1" }
            else if limits.synthesis.is_some() { "fnlp-candidate-int4-summary-synthesis-v1" }
            else if limits.keyphrases.is_some() { "fnlp-candidate-int4-keyphrase-ranking-v1" }
            else { "fnlp-candidate-int4-source-map-v1" }, schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: options.kind.name(),
        model_id: "Nanbeige4.2-3B", source_revision: &identity.source_revision,
        source_root_sha256: model.model_identity().source_root_sha256(), model_content_sha256: &content,
        candidate_format: &identity.artifact_format, quant_recipe: &identity.quant_recipe,
        group_width: model.model_identity().group_width(), integer_kernel: common.kernel.kind().id(), snapshot,
    };
    match prepared {
        Prepared::Map(mut config, expected) => {
            config.native.run.max_elapsed = remaining()?;
            if let Some(aggregation) = limits.keyphrases {
                if let Some(chunks) = limits.cohort_chunks {
                    let result = engine.reduce_int4_keyphrases_document_cohorts(&model, text,
                        Arc::new(planner), Arc::new(vocabulary), config, aggregation, common.kernel, cancellation)
                        .map_err(execution_error)?;
                    expected.verify_keyphrases_completed(&result.result().document).map_err(|_| CandidateError::Execution)?;
                    remaining()?;
                    publish_cohort(&provenance, &result.result().document, &result.result().cohorts,
                        chunks, limits.max_result_bytes + 4096, output)
                } else {
                    let result = engine.reduce_int4_keyphrases_document(&model, text, Arc::new(planner), Arc::new(vocabulary),
                        config, aggregation, common.kernel, cancellation).map_err(execution_error)?;
                    expected.verify_keyphrases_completed(result.result()).map_err(|_| CandidateError::Execution)?;
                    remaining()?;
                    publish(&Response { provenance: &provenance, output: result.result() },
                        limits.max_result_bytes + 4096, output)
                }
            } else if let Some(summary) = limits.summary {
                if let Some(chunks) = limits.cohort_chunks {
                    let result = engine.summarize_int4_source_cohorts(&model, text, Arc::new(planner), Arc::new(vocabulary),
                        config, summary, common.kernel, cancellation).map_err(execution_error)?;
                    expected.verify_summary_completed(&result.result().document).map_err(|_| CandidateError::Execution)?;
                    remaining()?;
                    publish_cohort(&provenance, &result.result().document, &result.result().cohorts,
                        chunks, limits.max_result_bytes + 4096, output)
                } else {
                    let result = engine.summarize_int4_source(&model, text, Arc::new(planner), Arc::new(vocabulary),
                        config, summary, common.kernel, cancellation).map_err(execution_error)?;
                    expected.verify_summary_completed(result.result()).map_err(|_| CandidateError::Execution)?;
                    remaining()?;
                    publish(&Response { provenance: &provenance, output: result.result() },
                        limits.max_result_bytes + 4096, output)
                }
            } else if let Some(chunks) = limits.cohort_chunks {
                let result = engine.map_int4_source_cohorts(&model, text, Arc::new(planner), Arc::new(vocabulary),
                    config, common.kernel, cancellation).map_err(execution_error)?;
                expected.verify_completed(&result.result().document).map_err(|_| CandidateError::Execution)?;
                remaining()?;
                // Preserve the semantic document and keep its host guard live
                // through complete serialization, write and flush. Only this
                // outer scheduling field describes shared native execution.
                publish_cohort(&provenance, &result.result().document, &result.result().cohorts,
                    chunks, limits.max_result_bytes + 4096, output)
            } else {
                let result = engine.map_int4_source(&model, text, Arc::new(planner), Arc::new(vocabulary),
                    config, common.kernel, cancellation).map_err(execution_error)?;
                expected.verify_completed(result.result()).map_err(|_| CandidateError::Execution)?;
                remaining()?;
                publish(&Response { provenance: &provenance, output: result.result() },
                    limits.max_result_bytes + 4096, output)
            }
        }
        Prepared::Question(mut config, expected) => {
            config.native.run.max_elapsed = remaining()?;
            if let Some(chunks) = limits.cohort_chunks {
                let result = engine.answer_int4_document_cohorts(&model, text, Arc::new(planner), Arc::new(vocabulary),
                    config, common.kernel, cancellation).map_err(execution_error)?;
                expected.verify_completed(&result.result().document).map_err(|_| CandidateError::Execution)?;
                remaining()?;
                // The document includes whitespace-only chunks; physical
                // stats count only actual native rows and nonempty windows.
                publish_cohort(&provenance, &result.result().document, &result.result().cohorts,
                    chunks, limits.max_result_bytes + 4096, output)
            } else {
                let result = engine.answer_int4_document(&model, text, Arc::new(planner), Arc::new(vocabulary),
                    config, common.kernel, cancellation).map_err(execution_error)?;
                expected.verify_completed(result.result()).map_err(|_| CandidateError::Execution)?;
                remaining()?;
                // The hosted output guard survives staging, write AND flush.
                // Native failure cannot publish a successful prefix or abstention.
                publish(&Response { provenance: &provenance, output: result.result() },
                    limits.max_result_bytes + 4096, output)
            }
        }
        Prepared::SummarySynthesis(mut config, expected) => {
            config.native.run.max_elapsed = remaining()?;
            if let Some(chunks) = limits.cohort_chunks {
                let result = engine.synthesize_int4_source_summary_cohorts(&model, text,
                    Arc::new(planner), Arc::new(vocabulary), config, common.kernel, cancellation)
                    .map_err(execution_error)?;
                expected.verify_completed(&result.result().document).map_err(|_| CandidateError::Execution)?;
                remaining()?;
                publish_discovery_cohort(&provenance, &result.result().document,
                    &result.result().discovery_cohorts, chunks, limits.max_result_bytes + 4096, output)
            } else {
                let result = engine.synthesize_int4_source_summary(&model, text, Arc::new(planner), Arc::new(vocabulary),
                    config, common.kernel, cancellation).map_err(execution_error)?;
                expected.verify_completed(result.result()).map_err(|_| CandidateError::Execution)?;
                remaining()?;
                publish(&Response { provenance: &provenance, output: result.result() },
                    limits.max_result_bytes + 4096, output)
            }
        }
        Prepared::QuestionSynthesis(mut config, expected) => {
            config.native.run.max_elapsed = remaining()?;
            if let Some(chunks) = limits.cohort_chunks {
                let result = engine.synthesize_int4_document_answer_cohorts(&model, text,
                    Arc::new(planner), Arc::new(vocabulary), config, common.kernel, cancellation)
                    .map_err(execution_error)?;
                expected.verify_completed(&result.result().document).map_err(|_| CandidateError::Execution)?;
                remaining()?;
                publish_discovery_cohort(&provenance, &result.result().document,
                    &result.result().discovery_cohorts, chunks, limits.max_result_bytes + 4096, output)
            } else {
                let result = engine.synthesize_int4_document_answer(&model, text, Arc::new(planner), Arc::new(vocabulary),
                    config, common.kernel, cancellation).map_err(execution_error)?;
                expected.verify_completed(result.result()).map_err(|_| CandidateError::Execution)?;
                remaining()?;
                publish(&Response { provenance: &provenance, output: result.result() },
                    limits.max_result_bytes + 4096, output)
            }
        }
        Prepared::QuestionHierarchy(mut config, expected) => {
            config.native.run.max_elapsed = remaining()?;
            if let Some(schedule) = limits.hierarchy_cohorts {
                let result = engine.synthesize_int4_document_answer_hierarchy_grouped(&model, text,
                    Arc::new(planner), Arc::new(vocabulary), config, expected.limits(), schedule, common.kernel, cancellation)
                    .map_err(execution_error)?;
                expected.verify_completed(&result.result().document).map_err(|_| CandidateError::Execution)?;
                remaining()?;
                publish_hierarchy_cohorts(&provenance, &result.result().document,
                    result.result().discovery_cohorts.as_ref(), &result.result().hierarchy_cohorts,
                    limits.cohort_chunks, schedule, limits.max_result_bytes + 4096, output)
            } else if let Some(chunks) = limits.cohort_chunks {
                let result = engine.synthesize_int4_document_answer_hierarchy_cohorts(&model, text,
                    Arc::new(planner), Arc::new(vocabulary), config, expected.limits(), common.kernel, cancellation)
                    .map_err(execution_error)?;
                expected.verify_completed(&result.result().document).map_err(|_| CandidateError::Execution)?;
                remaining()?;
                publish_discovery_cohort(&provenance, &result.result().document,
                    &result.result().discovery_cohorts, chunks, limits.max_result_bytes + 4096, output)
            } else {
                let result = engine.synthesize_int4_document_answer_hierarchy(&model, text,
                    Arc::new(planner), Arc::new(vocabulary), config, expected.limits(), common.kernel, cancellation)
                    .map_err(execution_error)?;
                expected.verify_completed(result.result()).map_err(|_| CandidateError::Execution)?;
                remaining()?;
                publish(&Response { provenance: &provenance, output: result.result() },
                    limits.max_result_bytes + 4096, output)
            }
        }
        Prepared::SummaryHierarchy(mut config, expected) => {
            config.native.run.max_elapsed = remaining()?;
            if let Some(schedule) = limits.hierarchy_cohorts {
                let result = engine.synthesize_int4_source_summary_hierarchy_grouped(&model, text,
                    Arc::new(planner), Arc::new(vocabulary), config, expected.limits(), schedule, common.kernel, cancellation)
                    .map_err(execution_error)?;
                expected.verify_completed(&result.result().document).map_err(|_| CandidateError::Execution)?;
                remaining()?;
                publish_hierarchy_cohorts(&provenance, &result.result().document,
                    result.result().discovery_cohorts.as_ref(), &result.result().hierarchy_cohorts,
                    limits.cohort_chunks, schedule, limits.max_result_bytes + 4096, output)
            } else if let Some(chunks) = limits.cohort_chunks {
                let result = engine.synthesize_int4_source_summary_hierarchy_cohorts(&model, text,
                    Arc::new(planner), Arc::new(vocabulary), config, expected.limits(), common.kernel, cancellation)
                    .map_err(execution_error)?;
                expected.verify_completed(&result.result().document).map_err(|_| CandidateError::Execution)?;
                remaining()?;
                publish_discovery_cohort(&provenance, &result.result().document,
                    &result.result().discovery_cohorts, chunks, limits.max_result_bytes + 4096, output)
            } else {
                let result = engine.synthesize_int4_source_summary_hierarchy(&model, text,
                    Arc::new(planner), Arc::new(vocabulary), config, expected.limits(), common.kernel, cancellation)
                    .map_err(execution_error)?;
                expected.verify_completed(result.result()).map_err(|_| CandidateError::Execution)?;
                remaining()?;
                publish(&Response { provenance: &provenance, output: result.result() },
                    limits.max_result_bytes + 4096, output)
            }
        }
    }
}
#[derive(Serialize)]
struct SnapshotRecord { operation: &'static str, format: &'static str, file_bytes: u64 }
impl SnapshotRecord {
    fn new(operation: &'static str, need: Int4SnapshotRequirements) -> Self {
        Self { operation, format: INT4_SNAPSHOT_VERSION, file_bytes: need.file_bytes }
    }
}
#[derive(Serialize)]
struct Provenance<'a> {
    protocol: &'static str, schema_version: u32, scope: &'static str, evidence: &'static str, task: &'static str,
    model_id: &'static str, source_revision: &'a str, source_root_sha256: &'a str, model_content_sha256: &'a str,
    candidate_format: &'a str, quant_recipe: &'a str, group_width: usize, integer_kernel: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    snapshot: Option<SnapshotRecord>,
}
#[derive(Serialize)]
struct Response<'a, 'b, T: Serialize> {
    #[serde(flatten)]
    provenance: &'a Provenance<'b>,
    output: &'a T,
}
#[derive(Serialize)]
struct CohortSchedule<'a> {
    execution: &'static str,
    declared_max_chunks: usize,
    physical: &'a Int4SourceMapCohortStats,
}
#[derive(Serialize)]
struct CohortResponse<'a, 'b, T: Serialize> {
    #[serde(flatten)]
    response: Response<'a, 'b, T>,
    cohort_schedule: CohortSchedule<'a>,
}
fn publish_cohort<T: Serialize>(provenance: &Provenance<'_>, document: &T, physical: &Int4SourceMapCohortStats,
    chunks: usize, max_bytes: usize, output: &mut impl Write) -> Result<(), CandidateError> {
    publish(&CohortResponse { response: Response { provenance, output: document },
        cohort_schedule: CohortSchedule { execution: "cross_chunk_constrained_cohorts",
            declared_max_chunks: chunks, physical } }, max_bytes, output)
}
fn publish_discovery_cohort<T: Serialize>(provenance: &Provenance<'_>, document: &T,
    physical: &Int4SourceMapCohortStats, chunks: usize, max_bytes: usize, output: &mut impl Write)
    -> Result<(), CandidateError> {
    // The semantic receipt retains every dependent pass. This separate ledger
    // covers discovery only; it cannot claim hierarchy groups ran in parallel.
    publish(&CohortResponse { response: Response { provenance, output: document },
        cohort_schedule: CohortSchedule { execution: "cross_chunk_constrained_discovery_cohorts",
            declared_max_chunks: chunks, physical } }, max_bytes, output)
}
#[derive(Serialize)]
struct HierarchySchedule<'a> {
    execution: &'static str,
    declared_max_groups: usize,
    physical: &'a Int4HierarchyCohortStats,
}
#[derive(Serialize)]
struct HierarchyResponse<'a, 'b, T: Serialize> {
    #[serde(flatten)]
    response: Response<'a, 'b, T>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cohort_schedule: Option<CohortSchedule<'a>>,
    hierarchy_cohort_schedule: HierarchySchedule<'a>,
}
#[allow(clippy::too_many_arguments)]
fn publish_hierarchy_cohorts<T: Serialize>(provenance: &Provenance<'_>, document: &T,
    discovery: Option<&Int4SourceMapCohortStats>, hierarchy: &Int4HierarchyCohortStats,
    chunks: Option<usize>, schedule: Int4HierarchyCohortSchedule, max_bytes: usize, output: &mut impl Write)
    -> Result<(), CandidateError> {
    if schedule.discovery_cohorts != chunks.is_some() || discovery.is_some() != chunks.is_some() {
        return Err(CandidateError::Execution);
    }
    let cohort_schedule = discovery.zip(chunks).map(|(physical, declared_max_chunks)| CohortSchedule {
        execution: "cross_chunk_constrained_discovery_cohorts", declared_max_chunks, physical });
    publish(&HierarchyResponse { response: Response { provenance, output: document }, cohort_schedule,
        hierarchy_cohort_schedule: HierarchySchedule { execution: "within_level_constrained_hierarchy_cohorts",
            declared_max_groups: schedule.max_groups, physical: hierarchy } }, max_bytes, output)
}

#[cfg(test)]
mod tests;
