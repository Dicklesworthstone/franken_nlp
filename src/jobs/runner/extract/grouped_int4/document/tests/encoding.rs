//! UNRUN pinned-planning and authenticated-policy fixtures. Preparation work
//! is bounded independently on each call; these do not exercise native recovery.
use super::*;
use crate::tokenizer::untrusted::SafeBpeLimits;

fn caps() -> SafeBpeLimits {
    SafeBpeLimits { max_input_bytes: 65536, max_output_tokens: 65536,
        max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24 }
}
fn compact(grounded: bool) -> Int4DocumentExtractionJobPlanner {
    factory(grounded).with_document_encoding(UntrustedEncoding::SafeBpeV1(caps())).unwrap()
}

#[test]
fn explicit_byte_fallback_preserves_recipe_bytes_and_prepared_ownership() {
    let ordinary = factory(true); let prepared = prepare(&ordinary, SOURCE);
    let bytes = serde_json::to_vec(ordinary.job_recipe()).unwrap(); let frozen = freeze(&ordinary);
    assert!(serde_json::to_value(ordinary.job_recipe()).unwrap()["extraction"].get("input_encoding").is_none());
    assert_eq!(ordinary.encoding_workspace_bytes().unwrap(), 0);
    let explicit = ordinary.with_document_encoding(UntrustedEncoding::ByteFallback).unwrap();
    assert_eq!(serde_json::to_vec(explicit.job_recipe()).unwrap(), bytes);
    frozen.binding.compare(&freeze(&explicit).binding).unwrap();
    explicit.owns(&prepared).unwrap();
    assert_eq!(prepared.expected.mapped().document_encoding(), UntrustedEncoding::ByteFallback);
    assert_eq!(prepared.expected.mapped().encoding_work(), None);
}

#[test]
fn full_compact_policy_is_private_resume_authority_and_does_not_rewrite_defaults() {
    let ordinary = factory(true); let frozen = freeze(&ordinary);
    let baseline = serde_json::to_value(ordinary.job_recipe()).unwrap();
    let compact = compact(true); let compact_frozen = freeze(&compact);
    let mut encoded = serde_json::to_value(compact.job_recipe()).unwrap();
    assert_eq!(encoded["extraction"]["input_encoding"],
        serde_json::to_value(UntrustedEncoding::SafeBpeV1(caps())).unwrap());
    encoded["extraction"].as_object_mut().unwrap().remove("input_encoding");
    assert_eq!(encoded, baseline);
    assert_eq!(compact.request().schema, SOURCE_SCHEMA);
    assert_eq!(compact.encoding_workspace_bytes().unwrap(), caps().requirements().unwrap().workspace_bytes);
    assert_eq!(frozen.binding.compare(&compact_frozen.binding), Err(JobError::Mismatch(MismatchField::Recipe)));
    for axis in 0..4 {
        let mut changed = caps();
        match axis { 0 => changed.max_input_bytes += 1, 1 => changed.max_output_tokens += 1,
            2 => changed.max_pair_checks += 1, _ => changed.max_pair_bytes += 1 }
        let changed = factory(true).with_document_encoding(UntrustedEncoding::SafeBpeV1(changed)).unwrap();
        assert_eq!(compact_frozen.binding.compare(&freeze(&changed).binding),
            Err(JobError::Mismatch(MismatchField::Recipe)), "axis {axis}");
    }
    frozen_equal(&compact, &self::compact(true));
}

#[test]
fn compact_preparation_preserves_original_coordinates_schema_and_reconstruction_identity() {
    for grounded in [false, true] {
        let ordinary = factory(grounded); let old = prepare(&ordinary, SOURCE);
        let compact = ordinary.with_document_encoding(UntrustedEncoding::SafeBpeV1(caps())).unwrap();
        assert_eq!(compact.owns(&old).unwrap_err().fault.code, BatchCode::Admission);
        let prepared = prepare(&compact, SOURCE);
        assert_eq!(prepared.source, SOURCE);
        assert_eq!(prepared.chunk_count(), 2);
        let mapped = prepared.expected.mapped();
        assert_eq!(mapped.document_encoding(), UntrustedEncoding::SafeBpeV1(caps()));
        let span = mapped.source_span();
        assert_eq!((span.byte_start, span.byte_end), (0, SOURCE.len()));
        assert_eq!((span.scalar_start, span.scalar_end), (0, SOURCE.chars().count()));
        let work = mapped.encoding_work().unwrap();
        assert!(work.input_bytes() > compact.request().schema.len() + SOURCE.len());
        assert!(work.output_tokens() > 0 && work.pair_checks() > 0 && work.pair_bytes() > 0);
        for (before, actual) in old.identities.iter().zip(&prepared.identities) {
            assert_eq!(actual.schema_digest, Sha256Digest::of_bytes(compact.request().schema.as_bytes()));
            assert_eq!(actual.template_digest, before.template_digest);
            assert_ne!(actual.decision_policy_digest, before.decision_policy_digest);
        }
        let plan = compact.plan(SOURCE, &mut Continue).unwrap();
        compact.compare(&plan, &prepared, &mut Continue).unwrap();
        assert_eq!(prepared.work, receipt_work(plan.preflight_metadata()));
        drop(plan);
        // Selecting the identical immutable policy does not invalidate a
        // preparation that already authenticates those exact limits.
        let same = compact.with_document_encoding(UntrustedEncoding::SafeBpeV1(caps())).unwrap();
        same.owns(&prepared).unwrap();
        let repeated = prepare(&same, SOURCE);
        assert_eq!(repeated.expected, prepared.expected);
        assert_eq!(repeated.identities, prepared.identities);
        assert_eq!(repeated.work, prepared.work);
    }
}

#[test]
fn exact_per_preparation_caps_cover_all_probes_and_final_chunks_and_refuse_short_authority() {
    let measured = prepare(&compact(true), SOURCE).expected.mapped().encoding_work().unwrap();
    let exact = SafeBpeLimits { max_input_bytes: measured.input_bytes(),
        max_output_tokens: measured.output_tokens(), max_pair_checks: measured.pair_checks(),
        max_pair_bytes: measured.pair_bytes() };
    let factory = self::factory(true).with_document_encoding(UntrustedEncoding::SafeBpeV1(exact)).unwrap();
    let first = prepare(&factory, SOURCE);
    // A reconstruction or resume starts another bounded preparation under
    // the same committed policy. This is not a lifetime tokenizer ledger.
    let second = prepare(&factory, SOURCE);
    assert_eq!(first.expected.mapped().encoding_work(), Some(measured));
    assert_eq!(second.expected, first.expected);
    let reconstructed = factory.plan(SOURCE, &mut Continue).unwrap();
    factory.compare(&reconstructed, &first, &mut Continue).unwrap();
    for axis in 0..4 {
        let mut short = exact;
        match axis { 0 => short.max_input_bytes -= 1, 1 => short.max_output_tokens -= 1,
            2 => short.max_pair_checks -= 1, _ => short.max_pair_bytes -= 1 }
        let short = self::factory(true).with_document_encoding(UntrustedEncoding::SafeBpeV1(short)).unwrap();
        let failure = short.prepare_with_control(document(SOURCE), &mut Continue).err().unwrap();
        assert!(failure.stop, "axis {axis}"); assert_eq!(failure.fault.cancellation, None);
    }
}

#[test]
fn compact_policy_composes_with_selected_rows_and_chunk_cohorts_in_either_order() {
    let rows = Int4JsonSparseLimits { max_rows_per_step: 7 };
    for grounded in [false, true] {
        let mut map = mapping(); map.mapping.reduction.map_batch_chunks = 3;
        let base = build(WIDTH, budget(), CompileLimits::default(), SourceRuntimeLimits::default(),
            request(grounded), map, limits(), lifetime()).unwrap();
        let compact = base.with_document_encoding(UntrustedEncoding::SafeBpeV1(caps())).unwrap();
        let before = prepare(&compact, SOURCE);
        let selected = compact.with_selected_rows(rows).unwrap();
        assert_eq!(selected.owns(&before).unwrap_err().fault.code, BatchCode::Admission);
        let selected_plan = prepare(&selected, SOURCE);
        assert_eq!(selected_plan.expected.mapped().encoding_work(), before.expected.mapped().encoding_work());
        let grouped = selected.with_chunk_cohorts().unwrap();
        assert_eq!(grouped.owns(&selected_plan).unwrap_err().fault.code, BatchCode::Admission);
        grouped.check_job_limits(limits()).unwrap();
        let grouped_plan = prepare(&grouped, SOURCE);
        assert_eq!(grouped_plan.identities, selected_plan.identities);
        assert_eq!(grouped_plan.work, selected_plan.work);
        assert_eq!(grouped_plan.expected, selected_plan.expected);
        let inverse = cohort_factory(3, grounded).with_selected_rows(rows).unwrap()
            .with_document_encoding(UntrustedEncoding::SafeBpeV1(caps())).unwrap();
        frozen_equal(&grouped, &inverse);
        let inverse_plan = prepare(&inverse, SOURCE);
        assert_eq!(inverse_plan.identities, grouped_plan.identities);
        assert_eq!(inverse_plan.work, grouped_plan.work);
        assert_eq!(inverse.encoding_workspace_bytes().unwrap(), caps().requirements().unwrap().workspace_bytes);
    }
}

#[test]
fn compact_object_and_invalid_encoder_limits_refuse_at_factory_configuration() {
    let object = object_factory(OBJECT_SCHEMA, object_limits(), mapping(), lifetime()).unwrap();
    let before = serde_json::to_vec(object.job_recipe()).unwrap();
    let object = object.with_document_encoding(UntrustedEncoding::ByteFallback).unwrap();
    assert_eq!(serde_json::to_vec(object.job_recipe()).unwrap(), before);
    assert_eq!(object.with_document_encoding(UntrustedEncoding::SafeBpeV1(caps())).err().unwrap().code,
        BatchCode::InvalidLimits);
    for axis in 0..8 {
        let mut invalid = caps();
        match axis { 0 => invalid.max_input_bytes = 0, 1 => invalid.max_output_tokens = 0,
            2 => invalid.max_pair_checks = 0, 3 => invalid.max_pair_bytes = 0,
            4 => invalid.max_input_bytes = SafeBpeLimits::MAX_INPUT_BYTES + 1,
            5 => invalid.max_output_tokens = SafeBpeLimits::MAX_OUTPUT_TOKENS + 1,
            6 => invalid.max_pair_checks = SafeBpeLimits::MAX_PAIR_CHECKS + 1,
            _ => invalid.max_pair_bytes = SafeBpeLimits::MAX_PAIR_BYTES + 1 }
        assert_eq!(factory(false).with_document_encoding(UntrustedEncoding::SafeBpeV1(invalid)).err().unwrap().code,
            BatchCode::InvalidLimits, "axis {axis}");
    }
}

#[test]
fn compiler_recipe_policy_drift_refuses_before_recovery_or_preparation() {
    for axis in 0..6 {
        let mut altered = if axis == 0 { factory(false) } else { compact(false) };
        match axis {
            0 => altered.recipe.extraction.input_encoding = Some(UntrustedEncoding::ByteFallback),
            1 => altered.recipe.extraction.input_encoding = None,
            _ => {
                let mut changed = caps();
                match axis { 2 => changed.max_input_bytes += 1, 3 => changed.max_output_tokens += 1,
                    4 => changed.max_pair_checks += 1, _ => changed.max_pair_bytes += 1 }
                altered.recipe.extraction.input_encoding = Some(UntrustedEncoding::SafeBpeV1(changed));
            }
        }
        assert_eq!(altered.check_job_limits(limits()).unwrap_err().code, BatchCode::Admission, "axis {axis}");
        assert_eq!(altered.encoding_workspace_bytes().unwrap_err().code, BatchCode::Admission, "axis {axis}");
        let failure = altered.prepare_with_control(document(SOURCE), &mut Continue).err().unwrap();
        assert!(failure.stop); assert_eq!(failure.fault.code, BatchCode::Admission, "axis {axis}");
    }
}

#[test]
fn compact_preparation_preserves_typed_cancellation() {
    struct Cancel;
    impl DecodeStepControl for Cancel {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::Deadline) }
    }
    let failure = compact(true).prepare_with_control(document(SOURCE), &mut Cancel).err().unwrap();
    assert!(failure.stop); assert_eq!(failure.fault.cancellation, Some(DecodeCancellationKind::Deadline));
}
