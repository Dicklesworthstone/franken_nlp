//! CLI/configuration regressions; no runtime, native model or filesystem fixtures.
use super::*;

fn args(task: &str) -> JobArgs { super::super::tests::args(task,
    &["--source-document-config", "document.json", "--preparation-mib", "1024"]) }
fn lifetime() -> JobLimits {
    let mut l = super::super::tests::lifetime(); l.max_result_bytes = 4 << 20; l
}
fn settings() -> SourceDocumentSettings {
    SourceDocumentSettings { schema_version: 1, max_chunks: 8, max_chunk_bytes: 8,
        max_tokenizer_calls: 1024, max_result_bytes: 4 << 20, max_live_value_bytes: 16 << 20,
        max_total_value_bytes: 64 << 20, reduction_reserve_mib: 64 }
}
fn text() -> String { serde_json::to_string(&settings()).unwrap() }
fn load_default(args: &JobArgs, text: &str, lifetime: JobLimits) -> Result<(Int8SourceMapLimits, u64), Failure> {
    let (_, limits) = args.validate()?;
    load(text, args, args.host.task_budget(limits), lifetime, limits)
}

#[test]
fn all_three_start_and_resume_routes_accept_explicit_source_documents_without_a_schema() {
    for task in ["ner", "keyphrases", "summarize"] {
        for mode in ["start", "resume"] {
            let command = super::super::tests::command(mode, task,
                &["--source-document-config", "document.json", "--preparation-mib", "1024"]);
            assert_eq!(command.mode().name(), mode);
            let (_, a) = command.into_parts();
            assert!(a.schema.is_none()); assert!(a.document_config.is_none());
            a.validate().unwrap(); load_default(&a, &text(), lifetime()).unwrap();
        }
    }
    // Existing explicit extraction and ordinary source jobs retain their routes.
    let extraction = super::super::tests::args("extract", &["--schema", "s", "--document-config", "d"]);
    assert!(extraction.source_document_config.is_none()); extraction.validate().unwrap();
    let single = super::super::tests::args("ner", &[]);
    assert!(single.source_document_config.is_none()); single.validate().unwrap();
}
#[test]
fn incompatible_task_schema_and_private_path_options_are_rejected() {
    for task in ["answer", "extract"] { assert!(args(task).validate().is_err()); }
    let mut a = args("ner"); a.schema = Some("schema.json".into()); assert!(a.validate().is_err());
    let mut a = args("ner"); a.document_config = Some("other.json".into()); assert!(a.validate().is_err());
    let mut a = args("ner"); a.source_membership = true; assert!(a.validate().is_err());
    for path in ["", "-"] {
        let mut a = args("ner"); a.source_document_config = Some(path.into()); assert!(a.validate().is_err());
    }
    let mut a = args("ner"); a.store_results = false;
    struct NoRead;
    impl Read for NoRead { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("IO without storage consent") } }
    let c = JobCommand { operation: Operation::Start(a) };
    let mut out = Vec::new(); assert!(c.execute_owned(NoRead, &mut out).is_err()); assert!(out.is_empty());
}
#[test]
fn settings_reject_duplicate_unknown_partial_oversized_and_foreign_policy_fields() {
    parse(&text()).unwrap();
    for field in ["schema_version", "max_chunks", "max_chunk_bytes", "max_tokenizer_calls", "max_result_bytes",
        "max_live_value_bytes", "max_total_value_bytes", "reduction_reserve_mib"] {
        let mut value = serde_json::to_value(settings()).unwrap(); value.as_object_mut().unwrap().remove(field);
        assert!(parse(&value.to_string()).is_err(), "missing {field}");
        let duplicate = text().replacen('{', &format!("{{\"{field}\":1,"), 1);
        assert!(parse(&duplicate).is_err(), "duplicate {field}");
    }
    for field in ["verification", "schema", "model", "max_work", "max_attempts", "seed"] {
        let mut value = serde_json::to_value(settings()).unwrap(); value[field] = serde_json::json!(1);
        assert!(parse(&value.to_string()).is_err(), "foreign {field}");
    }
    for text in ["".to_owned(), "{}".to_owned(), "[]".to_owned(), " ".repeat(CONFIG_BYTES + 1)] {
        assert!(parse(&text).is_err());
    }
}
#[test]
fn native_lifetime_authority_is_never_multiplied_and_masks_stay_bounded() {
    let a = args("ner"); let l = lifetime();
    let (m, _) = load_default(&a, &text(), l).unwrap();
    assert_eq!(m.max_model_work, l.max_work.model);
    assert_eq!(m.mask_visits_per_chunk, a.host.max_mask_node_visits);
    assert_eq!(m.max_mask_visits, 8 * a.host.max_mask_node_visits);
    let mut changed = l; changed.max_items *= 2; changed.max_attempts *= 2;
    assert_eq!(load_default(&a, &text(), changed).unwrap().0.max_model_work, m.max_model_work);
    let mut changed = settings(); changed.max_chunks *= 2;
    assert_eq!(load_default(&a, &serde_json::to_string(&changed).unwrap(), l).unwrap().0.max_model_work, m.max_model_work);
    let mut capped = l; capped.max_work.mask_node_visits = 2 * a.host.max_mask_node_visits;
    assert_eq!(load_default(&a, &text(), capped).unwrap().0.max_mask_visits, capped.max_work.mask_node_visits);
    for axis in 0..6 {
        let mut l = l;
        match axis { 0 => l.max_work.model.forward_positions = 0, 1 => l.max_work.model.projected_logits = 0,
            2 => l.max_work.model.attention_pairs = 0, 3 => l.max_work.model.projections.dot_products = 0,
            4 => l.max_work.model.projections.multiply_accumulates = 0, _ => l.max_work.mask_node_visits = 0 }
        assert!(load_default(&a, &text(), l).is_err(), "axis {axis}");
    }
}
#[test]
fn every_invalid_dimension_and_complete_output_limit_fails_closed() {
    let a = args("ner");
    for (field, value) in [("schema_version", 2_u64), ("max_chunks", 0), ("max_chunks", 257),
        ("max_chunk_bytes", 3), ("max_chunk_bytes", a.host.max_input_bytes as u64 + 1),
        ("max_tokenizer_calls", 0), ("max_tokenizer_calls", 1_000_001),
        ("max_result_bytes", 0), ("max_result_bytes", 64 * 1024 * 1024 + 1),
        ("max_live_value_bytes", 1), ("max_total_value_bytes", 1), ("reduction_reserve_mib", 0),
        ("reduction_reserve_mib", u64::MAX)] {
        let mut changed = serde_json::to_value(settings()).unwrap(); changed[field] = serde_json::json!(value);
        assert!(load_default(&a, &changed.to_string(), lifetime()).is_err(), "{field}={value}");
    }
    let mut smaller = lifetime(); smaller.max_result_bytes = settings().max_result_bytes - 1;
    assert!(load_default(&a, &text(), smaller).is_err());
}
#[test]
fn per_chunk_overrides_cannot_escape_host_budget_or_whole_context_memory() {
    let a = args("ner"); let (_, limits) = a.validate().unwrap(); let budget = a.host.task_budget(limits);
    for axis in 0..5 {
        let mut b = budget;
        match axis { 0 => b.max_input_tokens += 1, 1 => b.max_output_tokens += 1,
            2 => b.max_output_bytes += 1, 3 => b.max_grammar_states += 1, _ => b.max_kv_bytes -= 1 }
        assert!(load(&text(), &a, b, lifetime(), limits).is_err());
    }
    let mut l = limits; l.preparation_bytes = 1;
    assert_eq!(load(&text(), &a, budget, lifetime(), l).err().unwrap().code, "source_document_job_preparation");
    let mut changed = settings(); changed.reduction_reserve_mib += 1;
    let (first, reserve) = load_default(&a, &text(), lifetime()).unwrap();
    let (next, larger) = load_default(&a, &serde_json::to_string(&changed).unwrap(), lifetime()).unwrap();
    assert_eq!(first.chunks, next.chunks); assert_eq!(first.reduction, next.reduction);
    assert_eq!(first.max_model_work, next.max_model_work); assert_eq!(larger, reserve + MIB);
}
#[test]
fn missing_document_mode_is_not_inferred_from_settings_content() {
    let a = super::super::tests::args("ner", &[]); let (_, limits) = a.validate().unwrap();
    let error = load(&text(), &a, a.host.task_budget(limits), lifetime(), limits).err().unwrap();
    assert_eq!(error.code, "source_document_job_configuration_missing");
}
#[test]
#[cfg(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64"))))]
fn unsupported_profile_refuses_before_configuration_key_model_or_population_io() {
    struct NoRead;
    impl Read for NoRead { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("unavailable profile read") } }
    let c = JobCommand { operation: Operation::Start(args("ner")) }; let mut out = Vec::new();
    let error = c.execute_owned(NoRead, &mut out).err().unwrap();
    assert_eq!(error.code, "owned_job_profile_unavailable"); assert!(out.is_empty());
}
