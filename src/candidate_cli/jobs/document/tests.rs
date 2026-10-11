//! Model-free CLI/configuration contracts, not native execution evidence.
use super::*;
use crate::candidate_cli::jobs::tests::{args, command, lifetime};
use crate::batch::extract::ExtractionBatchGrounding;
fn options() -> JobArgs {
    args("extract", &["--schema", "schema.json", "--document-config", "document.json", "--preparation-mib", "1024"])
}
fn settings() -> DocumentSettings {
    DocumentSettings { schema_version: 1, max_chunks: 4, max_chunk_bytes: 16, max_tokenizer_calls: 1024,
        max_result_bytes: 4 << 20, max_live_value_bytes: 8 << 20, max_total_value_bytes: 16 << 20,
        verification: GroundingBudget { max_fields: 17, max_matches: 19, max_scan_steps: 23 }, reduction_reserve_mib: 64 }
}
fn job() -> JobLimits { let mut l = lifetime(); l.max_result_bytes = 4 << 20; l }
fn request(a: &JobArgs, limits: Limits) -> ExtractionBatchArgs {
    ExtractionBatchArgs { schema: r#"{"type":"boolean"}"#.to_owned(),
        grounding: ExtractionBatchGrounding::Structural, budget: a.host.task_budget(limits) }
}
fn load_settings(d: &DocumentSettings, a: &JobArgs, l: JobLimits) -> Result<(ExtractionDocumentBatchConfig, u64), Failure> {
    let (_, limits) = a.validate()?;
    load(&serde_json::to_string(d).unwrap(), a, request(a, limits), l, limits)
}
#[test]
fn both_start_and_resume_require_an_explicit_fixed_schema_for_document_mode() {
    for operation in ["start", "resume"] {
        let (_, a) = command(operation, "extract", &["--schema", "s", "--document-config", "d"]).into_parts();
        a.validate().unwrap(); assert!(a.document_config.is_some());
    }
    for task in ["ner", "keyphrases", "summarize", "answer"] {
        assert!(args(task, &["--schema", "s", "--document-config", "d"]).validate().is_err());
    }
    let mut a = options(); a.schema = None; assert!(a.validate().is_err());
    let mut a = options(); a.defaults = Some("defaults.json".into()); assert!(a.validate().is_err());
    for path in ["", "-"] { let mut a = options(); a.document_config = Some(path.into()); assert!(a.validate().is_err()); }
    assert!(args("extract", &[]).validate().is_ok());
}
#[test]
fn settings_are_bounded_duplicate_rejecting_and_not_an_executable_recipe() {
    let raw = serde_json::to_string(&settings()).unwrap(); parse(&raw).unwrap();
    for text in ["{}".to_owned(), " ".repeat(DOCUMENT_BYTES + 1),
        raw.replacen('{', "{\"schema_version\":1,", 1),
        raw.replacen('{', "{\"\\u0073chema_version\":1,", 1)] {
        assert!(parse(&text).is_err());
    }
    for field in ["schema", "model", "prompt", "max_work", "retry", "command"] {
        let mut value = serde_json::to_value(settings()).unwrap(); value[field] = serde_json::json!("untrusted");
        assert!(parse(&value.to_string()).is_err());
    }
}
#[test]
fn native_work_scales_by_chunks_but_lifetime_authority_is_never_multiplied() {
    let a = options(); let l = job();
    let one = constrained_int8::planned_work(a.host.context_tokens - a.host.max_new_tokens, a.host.max_new_tokens).unwrap();
    for chunks in [1, 4] {
        let mut d = settings(); d.max_chunks = chunks;
        let (c, _) = load_settings(&d, &a, l).unwrap();
        assert_eq!(c.limits.mapping.max_model_work, scale(one, chunks as u64).unwrap());
        assert_eq!(c.limits.mapping.max_mask_visits, a.host.max_mask_node_visits * chunks as u64);
        assert_eq!(c.max_model_work, l.max_work.model); assert_eq!(c.max_mask_visits, l.max_work.mask_node_visits);
        assert_eq!(c.limits.mapping.chunks.max_input_bytes, a.host.max_input_bytes);
    }
    let mut more_attempts = l; more_attempts.max_attempts += 1;
    let (a_config, _) = load_settings(&settings(), &a, l).unwrap();
    let (b_config, _) = load_settings(&settings(), &a, more_attempts).unwrap();
    assert_eq!(a_config.max_model_work, b_config.max_model_work);
    assert_eq!(a_config.max_mask_visits, b_config.max_mask_visits);
    assert_eq!(b_config.max_verification.max_fields - a_config.max_verification.max_fields, 17);
}
#[test]
fn verification_covers_durable_attempts_including_failed_attempts_not_only_inputs() {
    let a = options(); let l = job(); assert!(l.max_attempts > l.max_items);
    let (c, _) = load_settings(&settings(), &a, l).unwrap();
    assert_eq!(c.max_verification.max_fields, 17 * l.max_attempts as usize);
    assert_eq!(c.max_verification.max_matches, 19 * l.max_attempts as usize);
    assert_eq!(c.max_verification.max_scan_steps, 23 * l.max_attempts);
    assert_eq!(c.limits.verification, settings().verification);
}
#[test]
fn exact_schema_and_grounding_are_carried_without_json_number_coercion() {
    let mut a = options(); let (_, limits) = a.validate().unwrap(); let mut r = request(&a, limits);
    let schema = " {\"type\":\"number\",\"const\":12345678901234567890123456789012345678}\n";
    r.schema = schema.to_owned();
    let (c, reserve) = load(&serde_json::to_string(&settings()).unwrap(), &a, r, job(), limits).unwrap();
    assert_eq!(c.request.schema.as_bytes(), schema.as_bytes()); assert_eq!(reserve, 64 * MIB);
    a.source_membership = true;
    let mut r = request(&a, limits); r.grounding = ExtractionBatchGrounding::SourceMembership;
    r.schema = r#"{"type":"string","const":"Alice","x-fnlp-source":"verbatim"}"#.to_owned();
    let (c, _) = load(&serde_json::to_string(&settings()).unwrap(), &a, r, job(), limits).unwrap();
    assert_eq!(c.request.grounding, ExtractionBatchGrounding::SourceMembership);
    // Configuration never fabricated an empty source and rejected this constant.
    assert!(c.request.schema.contains("Alice"));
}
#[test]
fn document_result_and_each_durable_work_axis_must_fit_the_original_limits() {
    let a = options(); let (c, _) = load_settings(&settings(), &a, job()).unwrap();
    for axis in 0..7 {
        let mut l = job(); l.max_work = JobWork { model: c.limits.mapping.max_model_work,
            mask_node_visits: c.limits.mapping.max_mask_visits };
        match axis { 0 => l.max_work.model.forward_positions -= 1,
            1 => l.max_work.model.projected_logits -= 1, 2 => l.max_work.model.attention_pairs -= 1,
            3 => l.max_work.model.projections.dot_products -= 1, 4 => l.max_work.model.projections.multiply_accumulates -= 1,
            5 => l.max_work.mask_node_visits -= 1, _ => l.max_result_bytes -= 1 }
        assert!(load_settings(&settings(), &a, l).is_err(), "axis {axis}");
    }
}
#[test]
fn all_settings_domains_and_preparation_underfunding_fail_before_model_loading() {
    let a = options();
    for axis in 0..13 {
        let mut d = settings();
        match axis { 0 => d.schema_version = 2, 1 => d.max_chunks = 0, 2 => d.max_chunks = 257,
            3 => d.max_chunk_bytes = 3, 4 => d.max_tokenizer_calls = 0, 5 => d.max_result_bytes = 0,
            6 => d.max_live_value_bytes = d.max_result_bytes - 1, 7 => d.max_total_value_bytes = d.max_result_bytes - 1,
            8 => d.verification.max_fields = 0, 9 => d.verification.max_matches = 1_000_001,
            10 => d.verification.max_scan_steps = 1_000_000_000_001, 11 => d.reduction_reserve_mib = 0,
            _ => d.reduction_reserve_mib = u64::MAX }
        assert!(load_settings(&d, &a, job()).is_err(), "axis {axis}");
    }
    let (_, mut limits) = a.validate().unwrap(); let r = request(&a, limits); limits.preparation_bytes = 1;
    assert!(load(&serde_json::to_string(&settings()).unwrap(), &a, r, job(), limits).is_err());
}
#[test]
fn every_native_scaling_axis_uses_checked_arithmetic() {
    let a = options(); let one = constrained_int8::planned_work(a.host.context_tokens - a.host.max_new_tokens,
        a.host.max_new_tokens).unwrap();
    for axis in 0..5 {
        let mut w = one;
        match axis { 0 => w.forward_positions = u64::MAX, 1 => w.projected_logits = u64::MAX,
            2 => w.attention_pairs = u64::MAX, 3 => w.projections.dot_products = u64::MAX,
            _ => w.projections.multiply_accumulates = u64::MAX }
        assert!(scale(w, 2).is_err());
    }
    assert!(scale(one, 0).is_err());
}
#[cfg(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64"))))]
#[test]
fn unavailable_profiles_refuse_both_modes_before_any_key_settings_or_population_io() {
    struct NoRead;
    impl Read for NoRead { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("read") } }
    struct NoWrite;
    impl Write for NoWrite {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("write") }
        fn flush(&mut self) -> io::Result<()> { panic!("flush") }
    }
    for operation in ["start", "resume"] {
        let c = command(operation, "extract", &["--schema", "s", "--document-config", "d"]);
        assert_eq!(c.execute_owned(NoRead, &mut NoWrite).err().unwrap().code, "owned_job_profile_unavailable");
    }
}
