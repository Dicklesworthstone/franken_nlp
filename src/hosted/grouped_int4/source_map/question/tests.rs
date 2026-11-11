//! Host admission/ownership fixtures, without native model execution.
use super::*;
use crate::{execution_identity::NumericsProfile, native_engine::decode::DecodeCancellationKind,
    tasks::{answer::AnswerOptions, source_planning::grouped_int4::Int4SourceError},
    tokenizer::untrusted::{SafeBpeLimits, SafeBpeWorkLimits, UntrustedEncoding},
    validation::grounded_fields::GroundingBudget};

pub(super) fn fixture(width: usize) -> (SourceTaskPlanner, Int4SourceMapConfig<SourceQuestion>) {
    let (planner, base) = super::super::tests::fixture(SourceMapTask::Summarize(Default::default()), width);
    let mut identity = base.identity; identity.task_spec = ANSWER_TASK_VERSION.to_owned();
    let task = SourceQuestion { question: "Who founded the company?".to_owned(), options: AnswerOptions::default(),
        verification: GroundingBudget { max_fields: 512, max_matches: 1024, max_scan_steps: 1 << 20 } };
    (planner, Int4SourceMapConfig { identity, task, budget: base.budget, planning: base.planning,
        mapping: base.mapping, native: base.native, preparation_reserve_bytes: base.preparation_reserve_bytes,
        reduction_reserve_bytes: base.reduction_reserve_bytes })
}
#[test]
fn question_requires_actual_group_assets_and_full_resident_kv() {
    for width in [16, 32, 128] {
        let (planner, c) = fixture(width);
        assert_eq!(question_requirements(&c, 32, width, &planner).unwrap().kv_bytes, c.budget.max_kv_bytes);
        assert!(question_requirements(&c, 32, if width == 16 { 32 } else { 16 }, &planner).is_err());
    }
    for axis in 0..8 {
        let (planner, mut c) = fixture(32);
        match axis {
            0 => c.identity.task_spec = "ner-v1".to_owned(),
            1 => c.identity.numerics_profile = NumericsProfile::StrictQuantized { version: 1 },
            2 => c.identity.tokenizer_digest = Sha256Digest::of_bytes(b"changed tokenizer"),
            3 => c.identity.template_digest = Sha256Digest::of_bytes(b"changed template"),
            4 => c.budget.max_kv_bytes -= 1, 5 => c.native.context_tokens += 1,
            6 => c.planning.max_context_tokens += 1, _ => c.preparation_reserve_bytes = 0,
        }
        assert!(question_requirements(&c, 32, 32, &planner).is_err(), "axis {axis}");
    }
}
#[test]
fn invalid_question_and_cumulative_citation_allowances_refuse_before_capture() {
    for axis in 0..8 {
        let (planner, mut c) = fixture(32);
        match axis {
            0 => c.task.question = " \r\n".to_owned(),
            1 => c.task.question = "x".repeat(c.budget.max_input_tokens as usize + 1),
            2 => c.task.options.max_citations = 0, 3 => c.task.verification.max_fields = 0,
            4 => c.task.verification.max_matches = 1_000_001, 5 => c.task.verification.max_scan_steps = 0,
            6 => c.planning.max_input_bytes = 3, _ => c.reduction_reserve_bytes = 0,
        }
        assert!(question_requirements(&c, 32, 32, &planner).is_err(), "axis {axis}");
    }
    let (planner, c) = fixture(32);
    for bytes in [0, c.mapping.chunks.max_input_bytes + 1] { assert!(question_requirements(&c, bytes, 32, &planner).is_err()); }
}

fn compact_policy() -> (UntrustedEncoding, SafeBpeWorkLimits) {
    (UntrustedEncoding::SafeBpeV1(SafeBpeLimits {
        max_input_bytes: 4096, max_output_tokens: 4096,
        max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24,
    }), SafeBpeWorkLimits { max_input_bytes: 65_536, max_output_tokens: 65_536,
        max_pair_checks: 1 << 24, max_pair_bytes: 1 << 28 })
}

#[test]
fn only_explicit_compact_policy_admits_question_bytes_above_the_token_count_cap() {
    for width in [16, 32, 128] {
        let (planner, mut c) = fixture(width);
        c.task.question = "Where did Alice live? ".repeat(80);
        assert!(c.task.question.len() > c.budget.max_input_tokens as usize);
        assert!(question_requirements(&c, 32, width, &planner).is_err());
        let (policy, work) = compact_policy();
        let planner = planner.with_map_document_encoding(policy, work).unwrap();
        c.identity.template_digest = *planner.template_digest();
        c.preparation_reserve_bytes = planner.map_document_encoding_workspace_bytes().unwrap();
        let before = planner.map_document_encoding_work().unwrap();
        let native = question_requirements(&c, 32, width, &planner).unwrap();
        assert_eq!(native, Int4MemoryRequirement::for_context(c.native.context_tokens, width).unwrap());
        // This is admission, not a promise that arbitrary admitted bytes fit
        // the model. The controlled planner owns actual token/prompt checks.
        assert_eq!(planner.map_document_encoding_work().unwrap(), before);
        c.planning.max_input_bytes = c.task.question.len() - 1;
        assert!(question_requirements(&c, 32, width, &planner).is_err());
        c.planning.max_input_bytes = 8192;
        c.task.question = "x".repeat(4097);
        assert!(question_requirements(&c, 32, width, &planner).is_err());
    }
}

#[test]
fn compact_question_maps_require_the_whole_ledger_and_one_reusable_workspace() {
    let (policy, work) = compact_policy();
    let (planner, mut c) = fixture(32);
    let planner = planner.with_document_encoding(policy.clone()).unwrap();
    c.identity.template_digest = *planner.template_digest();
    // A one-request encoder cannot authorize repeated question/manifest and
    // passage encodings. The complete-document ledger is mandatory.
    assert!(question_requirements(&c, 32, 32, &planner).is_err());

    let (planner, mut c) = fixture(32);
    let planner = planner.with_map_document_encoding(policy.clone(), work).unwrap();
    let bytes = planner.map_document_encoding_workspace_bytes().unwrap();
    c.identity.template_digest = *planner.template_digest();
    c.preparation_reserve_bytes = bytes;
    for chunks in [1, 8, 256] {
        c.mapping.chunks.max_chunks = chunks;
        question_requirements(&c, 32, 32, &planner).unwrap();
        assert_eq!(planner.map_document_encoding_workspace_bytes().unwrap(), bytes);
        assert_eq!(input_bytes(1024, &c).unwrap(),
            1024 + c.task.question.capacity() as u64 + bytes);
    }
    c.preparation_reserve_bytes = bytes - 1;
    assert!(question_requirements(&c, 32, 32, &planner).is_err());
    c.preparation_reserve_bytes = bytes;
    let (changed, _) = fixture(32);
    let changed = changed.with_map_document_encoding(policy,
        SafeBpeWorkLimits { max_pair_bytes: work.max_pair_bytes + 1, ..work }).unwrap();
    assert_ne!(planner.template_digest(), changed.template_digest());
    assert!(question_requirements(&c, 32, 32, &changed).is_err());
    c.identity.template_digest = *changed.template_digest();
    question_requirements(&c, 32, 32, &changed).unwrap();
}
#[test]
fn original_question_capacity_and_all_evidence_owners_are_charged() {
    let (_, mut c) = fixture(32);
    let mut question = String::with_capacity(8192); question.push_str("Who?"); c.task.question = question;
    assert_eq!(input_bytes(1024, &c).unwrap(), 1024 + c.task.question.capacity() as u64 + c.preparation_reserve_bytes);
    let fields = c.task.verification.max_fields as u64
        * (std::mem::size_of::<PassageCitation>() + std::mem::size_of::<SourceCitation>()) as u64;
    let spans = c.task.verification.max_matches as u64
        * (std::mem::size_of::<PassageOccurrence>() + 2 * std::mem::size_of::<VerifiedSourceSpan>()) as u64;
    assert_eq!(c.question_workspace_bytes(3).unwrap(), c.workspace_bytes(3).unwrap() + fields + spans);
    let before = c.question_workspace_bytes(3).unwrap(); c.task.verification.max_fields += 1;
    assert_eq!(c.question_workspace_bytes(3).unwrap() - before,
        (std::mem::size_of::<PassageCitation>() + std::mem::size_of::<SourceCitation>()) as u64);
    let before = c.question_workspace_bytes(3).unwrap(); c.task.verification.max_matches += 1;
    assert_eq!(c.question_workspace_bytes(3).unwrap() - before,
        (std::mem::size_of::<PassageOccurrence>() + 2 * std::mem::size_of::<VerifiedSourceSpan>()) as u64);
}
#[test]
fn whole_population_and_checked_arithmetic_prevent_under_reservation() {
    let (_, c) = fixture(128);
    assert_eq!(c.question_workspace_bytes(3).unwrap() - c.question_workspace_bytes(2).unwrap(),
        c.budget.max_output_bytes * 4 + u64::from(c.budget.max_output_tokens) * 8);
    for count in [0, 9, 257, usize::MAX] { assert!(c.question_workspace_bytes(count).is_err()); }
    let (_, mut c) = fixture(128); c.budget.max_output_bytes = u64::MAX;
    assert!(c.question_workspace_bytes(2).is_err());
    let (_, mut c) = fixture(128); c.preparation_reserve_bytes = u64::MAX;
    assert!(input_bytes(32, &c).is_err());
}
#[test]
fn typed_cancellation_is_retained_with_redacted_diagnostics() {
    for cause in [DecodeCancellationKind::Deadline, DecodeCancellationKind::ParentCancelled,
        DecodeCancellationKind::ResourceUnavailable] {
        let error = map_error(Int4SourceMapError::Source(Int4SourceError::Cancelled(cause)));
        let HostedError::Int4(Int4HostError::SourceMap(source)) = &error else { panic!("wrong stage") };
        assert_eq!(source.cancellation(), Some(cause));
        assert!(!format!("{error:?}").contains("Who founded"));
    }
}
#[test]
fn complete_private_capture_is_sendable_to_physical_host() {
    fn send<T: Send + 'static>() {}
    send::<Int4SourceMapConfig<SourceQuestion>>(); send::<Charged<SourceMapInput<SourceQuestion>>>();
    let _host = NlpEngine::answer_int4_document;
}
