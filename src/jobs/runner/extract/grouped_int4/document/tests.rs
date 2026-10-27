//! UNRUN real pinned-planning and private-contract fixtures. These do not run
//! native inference, qualify model accuracy or establish crash recovery proof.
use super::*;
use crate::{
    artifact::converter::PINNED_REVISION,
    batch::extract::ExtractionBatchGrounding,
    canonjson,
    execution_identity::{NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    jobs::{FrozenManifest, JobContract, JobError, JobId, JobInput, JobSecret, MismatchField},
    native_engine::{decode::DecodeCancellationKind, generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::{ProjectionWork, int4::convert::GROUPED_INT4_RECIPE},
        strict_int4::STRICT_INT4_EXECUTION},
    tokenizer::{embedded::EmbeddedTokenizer, pinned_controls},
};

const WIDTH: usize = 127;
const SOURCE: &str = "éabc----βdef";
const SOURCE_SCHEMA: &str = r#"{"type":"string","maxLength":64,"x-fnlp-source":"verbatim"}"#;
struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 8192, max_output_tokens: 64, max_output_bytes: 65_536,
        max_grammar_states: 8192, max_kv_bytes: Int4MemoryRequirement::for_context(8192, WIDTH).unwrap().kv_bytes }
}
fn request(grounded: bool) -> ExtractionBatchArgs {
    ExtractionBatchArgs { schema: if grounded { SOURCE_SCHEMA } else { r#"{"type":"boolean"}"# }.to_owned(),
        grounding: if grounded { ExtractionBatchGrounding::SourceMembership } else { ExtractionBatchGrounding::Structural },
        budget: budget() }
}
fn maximum_work() -> Int4Work {
    let n = u64::MAX / 4;
    Int4Work { forward_positions: n, projected_logits: n, attention_pairs: n,
        projections: ProjectionWork { dot_products: n, multiply_accumulates: n } }
}
fn mapping() -> Int4ExtractionMapLimits {
    Int4ExtractionMapLimits { mapping: Int4ExtractionMappingLimits {
        chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 8, max_chunk_tokens: 8,
            context_tokens: 8192, reserved_tokens: 64, max_chunks: 8, max_tokenizer_calls: 1024 },
        reduction: ExecutionLimits { map_batch_chunks: 1, reduce_fan_in: 2,
            max_value_bytes: 4 << 20, ..ExecutionLimits::default() },
        max_model_work: maximum_work(), mask_limits: MaskWorkLimits::default(),
        mask_visits_per_chunk: 1000, max_mask_visits: 8000 }, verification: GroundingBudget::default() }
}
fn limits() -> JobLimits {
    JobLimits { max_items: 2, max_id_bytes: 128, max_input_bytes_per_item: 4096,
        max_snapshot_bytes: 65_536, max_result_bytes: 4 << 20, max_spool_bytes: 16 << 20,
        max_materialized_bytes: 16 << 20, max_journal_bytes: 1 << 20, max_attempts: 4,
        max_work: JobWork::from_int4(maximum_work(), 32_000) }
}
fn lifetime() -> GroundingBudget {
    GroundingBudget { max_fields: 16_384, max_matches: 65_536, max_scan_steps: 256 * 1024 * 1024 }
}
fn identity(width: usize) -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"private whole-document schema fixture, not model authority");
    let packing = Sha256Digest::of_bytes(&canonjson::canonical_bytes(&(
        INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", d, width)).unwrap());
    ExecutionIdentity { schema_version: 1, source_revision: PINNED_REVISION.to_owned(), logical_model_digest: d,
        artifact_format: INT4_SOURCE_FORMAT.to_owned(), quant_recipe: format!("{GROUPED_INT4_RECIPE}:group={width}"),
        packing_set_digest: packing, tokenizer_digest: d, template_digest: d, task_spec: "extract-v1".to_owned(),
        taskir_digest: d, prompt_digest: d, grammar_compiler_version: "uncompiled".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "uncompiled".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None }
}
#[allow(clippy::too_many_arguments)]
fn build(width: usize, ceiling: TaskBudget, compiler: CompileLimits, source: SourceRuntimeLimits,
    request: ExtractionBatchArgs, mapping: Int4ExtractionMapLimits, job: JobLimits, lifetime: GroundingBudget)
    -> Result<Int4DocumentExtractionJobPlanner, BatchFault> {
    let controls = pinned_controls::pinned().unwrap();
    let eos = EmbeddedTokenizer::pinned().unwrap().eos_token_id().unwrap();
    let base = Int4ExtractionJobPlanner::pinned(controls.template_controls(), eos, identity(width), width, ceiling,
        compiler, source, request, native_limits(mapping, job))?;
    Int4DocumentExtractionJobPlanner::configured(base, mapping, job, lifetime)
}
fn factory(grounded: bool) -> Int4DocumentExtractionJobPlanner {
    build(WIDTH, budget(), CompileLimits::default(), SourceRuntimeLimits::default(),
        request(grounded), mapping(), limits(), lifetime()).unwrap()
}
fn document(text: &str) -> BatchDocument<ExtractionDocumentArgs> {
    BatchDocument { id: "whole-document".to_owned(), text: text.to_owned(), task_args: None }
}
fn prepare(factory: &Int4DocumentExtractionJobPlanner, text: &str) -> PreparedInt4ExtractionDocument {
    factory.prepare_with_control(document(text), &mut Continue).unwrap()
}
fn freeze(factory: &Int4DocumentExtractionJobPlanner) -> FrozenManifest {
    let input = br#"{"id":"whole-document","text":"private complete source"}"#;
    FrozenManifest::freeze(&JobSecret::from_bytes([7; 32]), JobContract { job_id: JobId([9; 16]),
        execution: factory.execution_identity(), recipe: factory.job_recipe(), limits: factory.job_limits() },
        [JobInput { id: "whole-document", original: input, normalized: input }], &mut Continue).unwrap()
}
fn axis(work: &mut Int4Work, axis: usize) -> &mut u64 {
    match axis { 0 => &mut work.forward_positions, 1 => &mut work.projected_logits, 2 => &mut work.attention_pairs,
        3 => &mut work.projections.dot_products, _ => &mut work.projections.multiply_accumulates }
}

#[test]
fn both_grounding_modes_compile_every_real_unicode_chunk_and_reconstruct_exactly() {
    for grounded in [false, true] {
        let factory = factory(grounded); let prepared = prepare(&factory, SOURCE);
        assert_eq!(prepared.source, SOURCE);
        assert_eq!(prepared.chunk_count(), 2);
        assert_eq!(prepared.execution_identities().len(), 2);
        let span = prepared.expected.source_span();
        assert_eq!((span.byte_start, span.byte_end), (0, SOURCE.len()));
        assert_eq!((span.scalar_start, span.scalar_end), (0, SOURCE.chars().count()));
        for identity in prepared.execution_identities() {
            assert_eq!(identity.task_spec, "extract-v1");
            assert_eq!(identity.numerics_profile, NumericsProfile::StrictQuantized { version: 2 });
            assert_eq!(identity.schema_digest, Sha256Digest::of_bytes(factory.request().schema.as_bytes()));
            assert_eq!(identity.grammar_compiler_version, if grounded { SOURCE_JSON_RUNTIME_VERSION } else { JSON_RUNTIME_VERSION });
            assert_ne!(identity.prompt_digest, factory.execution_identity().prompt_digest);
        }
        assert_ne!(prepared.identities[0].prompt_digest, prepared.identities[1].prompt_digest);
        let plan = factory.plan(SOURCE, &mut Continue).unwrap();
        factory.compare(&plan, &prepared, &mut Continue).unwrap();
        assert_eq!(prepared.work, receipt_work(plan.preflight_metadata()));
    }
}

#[test]
fn durable_debit_uses_actual_all_chunk_work_and_masks_not_configured_document_ceiling() {
    let factory = factory(true); let prepared = prepare(&factory, SOURCE);
    let maximum = JobWork::from_int4(mapping().mapping.max_model_work, mapping().mapping.max_mask_visits);
    assert!(prepared.work.fits(maximum));
    assert_ne!(prepared.work, maximum);
    assert_eq!(prepared.work.mask_node_visits, 2000);
    let actual = prepared.work.int4_model_work();
    assert!(actual.forward_positions > 0 && actual.projected_logits > 0 && actual.attention_pairs > 0);
    assert!(actual.projections.dot_products > 0 && actual.projections.multiply_accumulates > 0);
    let one = prepare(&factory, "Alice");
    assert_eq!(one.chunk_count(), 1);
    assert_eq!(one.work.mask_node_visits, 1000);
    assert!(one.work.model.forward_positions < prepared.work.model.forward_positions);
    assert!(one.work.model.projected_logits < prepared.work.model.projected_logits);
}

#[test]
fn final_chunk_must_fit_every_actual_native_axis_and_mask_reservation() {
    let prepared = prepare(&factory(true), SOURCE);
    for field in 0..6 {
        let mut map = mapping();
        if field == 5 { map.mapping.max_mask_visits = prepared.work.mask_node_visits - 1; }
        else {
            map.mapping.max_model_work = prepared.work.int4_model_work();
            *axis(&mut map.mapping.max_model_work, field) -= 1;
        }
        let factory = build(WIDTH, budget(), CompileLimits::default(), SourceRuntimeLimits::default(),
            request(true), map, limits(), lifetime()).unwrap();
        let error = factory.prepare_with_control(document(SOURCE), &mut Continue).err().unwrap();
        assert!(error.stop); assert_eq!(error.fault.code, BatchCode::WorkLimit, "field {field}");
    }
}

#[test]
fn native_group_partition_changes_dot_debit_and_execution_commitment() {
    let a = build(1, budget(), CompileLimits::default(), SourceRuntimeLimits::default(),
        request(false), mapping(), limits(), lifetime()).unwrap();
    let b = build(128, budget(), CompileLimits::default(), SourceRuntimeLimits::default(),
        request(false), mapping(), limits(), lifetime()).unwrap();
    let wa = prepare(&a, SOURCE).work.int4_model_work(); let wb = prepare(&b, SOURCE).work.int4_model_work();
    assert_eq!(wa.forward_positions, wb.forward_positions);
    assert_eq!(wa.projected_logits, wb.projected_logits);
    assert_eq!(wa.attention_pairs, wb.attention_pairs);
    assert_eq!(wa.projections.multiply_accumulates, wb.projections.multiply_accumulates);
    assert!(wa.projections.dot_products > wb.projections.dot_products);
    assert_eq!(freeze(&a).binding.compare(&freeze(&b).binding), Err(JobError::Mismatch(MismatchField::Execution)));
}

#[test]
fn raw_schema_decimal_lexemes_and_whitespace_remain_exact_private_recipe_inputs() {
    let schemas = [
        " {\"type\":\"number\",\"const\":12345678901234567890123456789012345678} \n",
        " {\"type\":\"number\",\"const\":12345678901234567890123456789012345679} \n",
        "{\"type\":\"number\",\"const\":12345678901234567890123456789012345678}",
    ];
    let mut bindings = Vec::new();
    for schema in schemas {
        let mut args = request(false); args.schema = schema.to_owned();
        let factory = build(WIDTH, budget(), CompileLimits::default(), SourceRuntimeLimits::default(),
            args, mapping(), limits(), lifetime()).unwrap();
        assert_eq!(factory.request().schema, schema);
        assert_eq!(serde_json::to_value(factory.job_recipe()).unwrap()["extraction"]["defaults"]["schema"], schema);
        let prepared = prepare(&factory, SOURCE);
        assert!(prepared.identities.iter().all(|id| id.schema_digest == Sha256Digest::of_bytes(schema.as_bytes())));
        bindings.push(freeze(&factory));
    }
    for changed in &bindings[1..] {
        assert_eq!(bindings[0].binding.compare(&changed.binding), Err(JobError::Mismatch(MismatchField::Recipe)));
    }
}

#[test]
fn compiler_source_and_document_policies_change_the_keyed_recipe() {
    let baseline = freeze(&factory(true));
    for field in 0..32 {
        let mut compiler = CompileLimits::default(); let mut source = SourceRuntimeLimits::default();
        let mut map = mapping(); let mut lifetime = lifetime(); let mut request = request(true);
        match field {
            0 => compiler.max_schema_bytes += 1, 1 => compiler.max_string_bytes += 1,
            2 => compiler.max_array_items += 1, 3 => compiler.max_output_bytes += 1,
            4 => compiler.max_states += 1, 5 => compiler.max_transitions += 1, 6 => compiler.max_mask_bytes += 1,
            7 => source.index.max_source_bytes += 1, 8 => source.index.max_index_bytes += 1,
            9 => source.index.max_build_steps += 1, 10 => source.index.max_matches += 1,
            11 => source.verification.max_fields += 1, 12 => source.verification.max_matches += 1,
            13 => source.verification.max_scan_steps += 1,
            14 => map.mapping.chunks.max_chunk_bytes += 1, 15 => map.mapping.chunks.max_tokenizer_calls += 1,
            16 => map.mapping.reduction.reduce_fan_in += 1, 17 => map.mapping.reduction.max_live_value_bytes -= 1,
            18 => map.mapping.mask_limits.max_trie_node_visits += 1,
            19 => map.mapping.mask_limits.checkpoint_interval_nodes += 1,
            20 => map.mapping.mask_visits_per_chunk += 1, 21 => map.mapping.max_mask_visits += 1,
            22..=26 => *axis(&mut map.mapping.max_model_work, field - 22) -= 1,
            27 => map.verification.max_fields -= 1, 28 => map.verification.max_matches -= 1,
            29 => map.verification.max_scan_steps -= 1, 30 => lifetime.max_matches += 1,
            _ => request.budget.max_output_tokens -= 1,
        }
        let changed = build(WIDTH, budget(), compiler, source, request, map, limits(), lifetime).unwrap();
        assert_eq!(baseline.binding.compare(&freeze(&changed).binding), Err(JobError::Mismatch(MismatchField::Recipe)),
            "field {field}");
    }
}

#[test]
fn prepared_document_cannot_cross_factory_or_source_identity_receipt_boundary() {
    let original = factory(true); let other = factory(true);
    let prepared = prepare(&original, SOURCE);
    assert!(other.owns(&prepared).is_err());
    for mutation in 0..5 {
        let mut prepared = prepare(&original, SOURCE);
        match mutation {
            0 => prepared.identities.last_mut().unwrap().prompt_digest = Sha256Digest::of_bytes(b"wrong final chunk"),
            1 => { prepared.source.push('!'); },
            2 => prepared.expected = prepare(&original, "Alice").expected,
            3 => prepared.work.model.projections.dot_products -= 1,
            _ => { prepared.identities.pop(); },
        }
        let plan = original.plan(&prepared.source, &mut Continue).unwrap();
        assert!(original.compare(&plan, &prepared, &mut Continue).is_err(), "mutation {mutation}");
    }
}

#[test]
fn final_result_and_full_resident_kv_are_fixed_before_job_storage() {
    for delta in [-1_i64, 1] {
        let mut ceiling = budget(); ceiling.max_kv_bytes = (ceiling.max_kv_bytes as i64 + delta) as u64;
        let mut request = request(false); request.budget = ceiling;
        assert!(build(WIDTH, ceiling, CompileLimits::default(), SourceRuntimeLimits::default(),
            request, mapping(), limits(), lifetime()).is_err());
    }
    let mut reduced_request = request(false); reduced_request.budget.max_kv_bytes -= 1;
    assert!(build(WIDTH, budget(), CompileLimits::default(), SourceRuntimeLimits::default(),
        reduced_request, mapping(), limits(), lifetime()).is_err());
    let mut map = mapping(); map.mapping.chunks.context_tokens -= 1;
    assert!(build(WIDTH, budget(), CompileLimits::default(), SourceRuntimeLimits::default(),
        request(false), map, limits(), lifetime()).is_err());
    let mut job = limits(); job.max_result_bytes -= 1;
    assert!(build(WIDTH, budget(), CompileLimits::default(), SourceRuntimeLimits::default(),
        request(false), mapping(), job, lifetime()).is_err());
}

#[test]
fn document_work_cannot_exceed_any_persisted_native_or_mask_axis() {
    for field in 0..6 {
        let mut job = limits(); let map = mapping();
        job.max_work = JobWork::from_int4(map.mapping.max_model_work, map.mapping.max_mask_visits);
        if field == 5 { job.max_work.mask_node_visits -= 1; }
        else {
            let mut work = job.max_work.int4_model_work(); *axis(&mut work, field) -= 1;
            job.max_work = JobWork::from_int4(work, job.max_work.mask_node_visits);
        }
        assert!(build(WIDTH, budget(), CompileLimits::default(), SourceRuntimeLimits::default(),
            request(true), map, job, lifetime()).is_err(), "field {field}");
    }
}

#[test]
fn every_job_limit_stays_identical_on_start_and_resume() {
    let factory = factory(false); let exact = factory.job_limits();
    factory.check_job_limits(exact).unwrap();
    for field in 0..16 {
        let mut changed = exact;
        match field {
            0 => changed.max_items += 1, 1 => changed.max_id_bytes += 1,
            2 => changed.max_input_bytes_per_item += 1, 3 => changed.max_snapshot_bytes += 1,
            4 => changed.max_result_bytes += 1, 5 => changed.max_spool_bytes += 1,
            6 => changed.max_materialized_bytes += 1, 7 => changed.max_journal_bytes += 1,
            8 => changed.max_attempts += 1, 9 => changed.max_work.mask_node_visits += 1,
            10..=14 => {
                let mut work = changed.max_work.int4_model_work(); *axis(&mut work, field - 10) += 1;
                changed.max_work = JobWork::from_int4(work, changed.max_work.mask_node_visits);
            }
            _ => changed.max_attempts -= 1,
        }
        assert_eq!(factory.check_job_limits(changed), Err(BatchCode::InvalidLimits.into()), "field {field}");
    }
}

#[test]
fn independent_verification_lifetime_includes_failed_and_uncertain_attempts() {
    for field in 0..3 {
        let mut max = lifetime();
        match field { 0 => max.max_fields -= 1, 1 => max.max_matches -= 1, _ => max.max_scan_steps -= 1 }
        assert!(build(WIDTH, budget(), CompileLimits::default(), SourceRuntimeLimits::default(),
            request(true), mapping(), limits(), max).is_err());
    }
    let mut per = mapping().verification; per.max_scan_steps = u64::MAX;
    let mut max = lifetime(); max.max_scan_steps = u64::MAX;
    assert!(verify_lifetime(per, max, 2).is_err());
    per = mapping().verification; per.max_fields = usize::MAX; max.max_fields = usize::MAX;
    assert!(verify_lifetime(per, max, 2).is_err());
    assert!(verify_lifetime(mapping().verification, lifetime(), 0).is_err());
    assert!(verify_lifetime(mapping().verification, lifetime(), limits().max_attempts + 1).is_err());
}

#[test]
fn original_documents_refuse_empty_oversized_cancelled_or_per_record_policy_overrides() {
    let factory = factory(true);
    assert_eq!(factory.prepare_with_control(document(""), &mut Continue).err().unwrap().fault.code, BatchCode::DocumentLimit);
    assert_eq!(factory.prepare_with_control(document(&"x".repeat(4097)), &mut Continue).err().unwrap().fault.code,
        BatchCode::DocumentLimit);
    struct Cancel;
    impl DecodeStepControl for Cancel {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::Deadline) }
    }
    let error = factory.prepare_with_control(document(SOURCE), &mut Cancel).err().unwrap();
    assert!(error.stop); assert_eq!(error.fault.cancellation, Some(DecodeCancellationKind::Deadline));
    for input in [r#"{"schema":"override"}"#, r#"{"grounding":"structural"}"#,
        r#"{"budget":{"max_output_tokens":1}}"#, r#"{"context_tokens":1}"#] {
        assert!(serde_json::from_str::<ExtractionDocumentArgs>(input).is_err());
    }
    let mut input = document(SOURCE);
    input.task_args = Some(serde_json::from_str("{}").unwrap());
    assert_eq!(factory.prepare_with_control(input, &mut Continue).unwrap().chunk_count(), 2);
}

#[test]
fn empty_schema_oversized_schema_or_nonpositive_map_authority_never_prepare() {
    let factory = factory(false);
    let mut args = request(false); args.schema.clear();
    let changed = build(WIDTH, budget(), CompileLimits::default(), SourceRuntimeLimits::default(),
        args, mapping(), limits(), lifetime()).unwrap();
    assert!(changed.prepare_with_control(document(SOURCE), &mut Continue).is_err());
    let mut args = request(false); args.schema = "x".repeat(1024 * 1024);
    assert!(build(WIDTH, budget(), CompileLimits::default(), SourceRuntimeLimits::default(),
        args, mapping(), limits(), lifetime()).is_err());
    for field in 0..8 {
        let mut map = mapping();
        match field {
            0..=4 => *axis(&mut map.mapping.max_model_work, field) = 0,
            5 => map.mapping.mask_visits_per_chunk = 0, 6 => map.verification.max_fields = 0,
            _ => map.mapping.reduction.max_result_bytes = 0,
        }
        assert!(build(WIDTH, budget(), CompileLimits::default(), SourceRuntimeLimits::default(),
            request(false), map, limits(), lifetime()).is_err());
    }
    let mut override_args = factory.request().clone(); override_args.schema.push(' ');
    assert!(factory.compiler.plan_document_with_control(SOURCE, &override_args, mapping(), &mut Continue).is_err());
}

#[test]
fn host_token_quote_accounts_for_every_retained_native_chunk_transcript() {
    let factory = factory(true); let prepared = prepare(&factory, SOURCE);
    assert_eq!(factory.output_token_ceiling().unwrap(), 8 * 64);
    assert_eq!(factory.output_tokens(prepared.chunk_count()).unwrap(), 2 * 64);
    assert_eq!(factory.max_result_bytes(), mapping().mapping.reduction.max_result_bytes as u64);
}

struct Admission;
impl Int4ExtractionDocumentBatchAdmission for Admission {
    type Guard = ();
    fn admit(&mut self, _: Int4ExtractionDocumentAdmission<'_>) -> Result<(Vec<ExecutionIdentity>, ()), BatchItemFailure> {
        Err(BatchItemFailure::fatal(BatchCode::Admission))
    }
}
#[test]
fn real_native_processor_factory_and_owned_preparation_satisfy_host_contracts() {
    fn durable<P: DurableBatchProcessor<Recipe = Int4DocumentExtractionJobRecipe,
        Args = ExtractionDocumentArgs, Prepared = PreparedInt4ExtractionDocument,
        Output = GuardedOutput<Int4ExtractionMapRun, ()>>>() {}
    fn send<T: Send + 'static>() {}
    durable::<Int4DocumentExtractionJobProcessor<'static, 'static, 'static, 'static, Admission>>();
    send::<Int4DocumentExtractionJobPlanner>(); send::<PreparedInt4ExtractionDocument>();
}
