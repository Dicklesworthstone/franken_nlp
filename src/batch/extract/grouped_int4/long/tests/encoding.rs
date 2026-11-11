//! Real pinned tokenization and private native fault injection. These are
//! contract regressions, not executed model-quality or speed evidence.
use super::*;

const TEXT: &str = "The quick brown fox jumps over the lazy dog. 上海 <think> <|im_start|> ▁\r\nThe quick brown fox jumps over the lazy dog.";

fn policy() -> UntrustedEncoding {
    UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 1 << 20,
        max_output_tokens: 1 << 20, max_pair_checks: 1 << 26, max_pair_bytes: 1 << 29 })
}
fn compact(policy: UntrustedEncoding) -> Int4ExtractionBatchPlanner {
    planner().with_document_encoding(policy).unwrap()
}
fn mapping() -> Int4ExtractionMapLimits {
    let mut limits = limits();
    limits.mapping.chunks.max_chunk_bytes = 48;
    limits
}
fn measured_policy(work: SafeBpeWork) -> UntrustedEncoding {
    UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: work.input_bytes(),
        max_output_tokens: work.output_tokens(), max_pair_checks: work.pair_checks(), max_pair_bytes: work.pair_bytes() })
}

#[test]
fn compact_maps_preserve_the_exact_partition_schema_and_original_unicode_evidence() {
    let byte = planner(); let compact = compact(policy()); let args = source_args();
    let original = prepare(&byte, TEXT, &args, mapping());
    let prepared = prepare(&compact, TEXT, &args, mapping());
    let expected = prepared.preflight_metadata();
    assert_eq!(prepared.document_encoding(), policy());
    assert_eq!(prepared.encoding_workspace_bytes().unwrap(), memory::encoding_workspace_bytes(&policy()).unwrap());
    assert_eq!(expected.source_span(), original.preflight_metadata().source_span());
    assert_eq!(prepared.chunks.chunks().iter().map(|chunk| chunk.text()).collect::<String>(), TEXT);
    assert_eq!(prepared.chunks.chunks().len(), original.chunks.chunks().len());
    assert!(expected.planned_work().forward_positions < original.preflight_metadata().planned_work().forward_positions);
    let controls = pinned_controls::pinned().unwrap();
    let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    for ((chunk, plan), byte) in prepared.chunks.chunks().iter().zip(&prepared.plans).zip(&original.plans) {
        assert_eq!(chunk.span().byte_end - chunk.span().byte_start, chunk.text().len());
        assert_eq!(plan.source().text(), chunk.text());
        assert_eq!(plan.source().total_token_count(), chunk.tokens());
        assert_eq!(tokenizer.tokenizer().decode_bytes(plan.source().token_ids()).unwrap(), chunk.text().as_bytes());
        assert!(plan.source().token_ids().iter().all(|&id| !controls.template_controls().contains(id)));
        assert_eq!(plan.execution_identity().schema_digest, Sha256Digest::of_bytes(args.schema.as_bytes()));
        assert_eq!(plan.execution_identity().template_digest, byte.execution_identity().template_digest);
        assert_ne!(plan.execution_identity().decision_policy_digest, byte.execution_identity().decision_policy_digest);
        assert!(plan.plan.prompt_tokens() + args.budget.max_output_tokens as usize <= mapping().mapping.chunks.context_tokens);
    }
    let completed = run(prepared, Script::new(&args)).unwrap();
    expected.verify_completed(&completed).unwrap();
    assert_eq!(completed.input_encoding, Some(policy()));
    assert_eq!(completed.input_encoding_work, expected.encoding_work());
    for chunk in completed.mapped.root().value().chunks() {
        for field in &chunk.original_fields {
            for span in &field.spans {
                assert_eq!(&TEXT[span.byte_start..span.byte_end],
                    &TEXT[chunk.source_span.byte_start..chunk.source_span.byte_end]);
                assert_eq!(TEXT[..span.byte_start].chars().count(), span.scalar_start);
                assert_eq!(TEXT[..span.byte_end].chars().count(), span.scalar_end);
            }
        }
    }
}

#[test]
fn sizing_every_probe_and_every_final_schema_source_share_all_four_work_axes() {
    let args = source_args();
    let compiler = compact(policy());
    let prepared = prepare(&compiler, TEXT, &args, mapping());
    let work = prepared.preflight_metadata().encoding_work().unwrap();
    let chunks = prepared.chunks.chunks().len();
    assert_eq!(prepared.chunks.tokenizer_calls(), chunks);
    assert_eq!(work.input_bytes(), (chunks + 1) * args.schema.len() + 2 * TEXT.len());
    let exact = measured_policy(work);
    let exact_plan = prepare(&compact(exact.clone()), TEXT, &args, mapping());
    assert_eq!(exact_plan.preflight_metadata().encoding_work(), Some(work));
    for axis in 0..4 {
        let UntrustedEncoding::SafeBpeV1(mut short) = exact.clone() else { unreachable!() };
        match axis { 0 => short.max_input_bytes -= 1, 1 => short.max_output_tokens -= 1,
            2 => short.max_pair_checks -= 1, _ => short.max_pair_bytes -= 1 }
        let short = compact(UntrustedEncoding::SafeBpeV1(short));
        // Any one actual chunk fits. Refusal must come from whole-document
        // work, never an implicit fresh counter at the next plan or probe.
        short.prepare_with_control(BatchDocument { id: "one".to_owned(),
            text: prepared.chunks.chunks()[0].text().to_owned(), task_args: Some(args.clone()) }, &mut Continue).unwrap();
        assert!(short.plan_document_with_control(TEXT, &args, mapping(), &mut Continue).is_err(), "axis {axis}");
    }
}

#[test]
fn discarded_nonmonotone_chunk_probes_spend_work_and_do_not_consume_final_token_room() {
    let args = source_args(); let compiler = compact(policy());
    let mut limits = mapping();
    limits.mapping.chunks.max_chunk_bytes = 256;
    limits.mapping.chunks.max_chunk_tokens = 8;
    let prepared = prepare(&compiler, TEXT, &args, limits);
    assert!(prepared.chunks.tokenizer_calls() > prepared.chunks.chunks().len());
    assert!(prepared.chunks.chunks().iter().all(|chunk| chunk.tokens() <= 8));
    let minimum = (prepared.chunks.chunks().len() + 1) * args.schema.len() + 2 * TEXT.len();
    assert!(prepared.preflight_metadata().encoding_work().unwrap().input_bytes() > minimum);
    limits.mapping.chunks.max_tokenizer_calls = 1;
    assert!(matches!(compiler.plan_document_with_control(TEXT, &args, limits, &mut Continue),
        Err(Int4ExtractionMapError::Chunk(MapReduceError::TokenizerBudget))));
}

#[test]
fn token_capacity_is_based_on_actual_compact_schema_and_source_not_raw_bytes() {
    let compiler = compact(policy()); let args = source_args();
    let mut limits = mapping(); limits.mapping.chunks.max_chunk_bytes = 256;
    let first = prepare(&compiler, TEXT, &args, limits);
    assert_eq!(first.plans.len(), 1);
    let prompt = first.plans[0].plan.prompt_tokens();
    assert!(prompt < TEXT.len() + args.schema.len()
        + compiler.compiler.fragments.iter().map(Vec::len).sum::<usize>());
    let mut bounded = args.clone(); bounded.budget.max_input_tokens = prompt as u32;
    limits.mapping.chunks.context_tokens = prompt + bounded.budget.max_output_tokens as usize;
    let exact = prepare(&compiler, TEXT, &bounded, limits);
    assert_eq!(exact.plans.len(), 1);
    assert_eq!(exact.plans[0].plan.prompt_tokens(), prompt);
    assert_eq!(exact.chunks.chunks()[0].tokens(), exact.plans[0].source().total_token_count());
    // A byte-only request cannot fit the same complete source in this prompt.
    if let Ok(byte) = planner().plan_document_with_control(TEXT, &bounded, limits, &mut Continue) {
        assert!(byte.plans.len() > 1);
    }
}

#[test]
fn compact_input_fits_one_complete_partition_where_byte_input_needs_many_at_the_same_context() {
    let source = "The quick brown fox jumps over the lazy dog. ".repeat(8);
    let compact = compact(policy()); let byte = planner();
    let mut args = args(r#"{"type":"boolean"}"#, ExtractionBatchGrounding::Structural);
    let mut limits = mapping(); limits.mapping.chunks.max_chunk_bytes = source.len();
    let prepared = prepare(&compact, &source, &args, limits);
    assert_eq!(prepared.plans.len(), 1);
    let prompt = prepared.plans[0].plan.prompt_tokens();
    let byte_fixed = byte.compiler.fragments.iter().map(Vec::len).sum::<usize>() + args.schema.len();
    let input = prompt.max(byte_fixed + 16);
    assert!(input < byte_fixed + source.len());
    args.budget.max_input_tokens = input as u32;
    limits.mapping.chunks.context_tokens = input + args.budget.max_output_tokens as usize;
    let compact = prepare(&compact, &source, &args, limits);
    let byte = prepare(&byte, &source, &args, limits);
    assert_eq!(compact.plans.len(), 1); assert!(byte.plans.len() > 1);
    assert_eq!(compact.chunks.chunks()[0].text(), source);
    assert_eq!(byte.chunks.chunks().iter().map(|chunk| chunk.text()).collect::<String>(), source);
    assert_eq!(compact.chunks.limits().context_tokens, byte.chunks.limits().context_tokens);
    assert!(compact.preflight_metadata().planned_work().forward_positions
        < byte.preflight_metadata().planned_work().forward_positions);
}

#[test]
fn same_tokens_with_a_different_policy_cannot_replace_a_final_chunk_or_completed_binding() {
    let args = source_args(); let compiler = compact(policy());
    let UntrustedEncoding::SafeBpeV1(mut changed) = policy() else { unreachable!() };
    changed.max_pair_checks -= 1;
    let other = compact(UntrustedEncoding::SafeBpeV1(changed));
    let mut prepared = prepare(&compiler, TEXT, &args, mapping());
    let text = prepared.chunks.chunks().last().unwrap().text().to_owned();
    let replacement = other.prepare_with_control(BatchDocument { id: "tail".to_owned(), text,
        task_args: Some(args.clone()) }, &mut Continue).unwrap();
    assert_eq!(prepared.plans.last().unwrap().source().token_ids(), replacement.source().token_ids());
    *prepared.plans.last_mut().unwrap() = replacement;
    let script = Script::new(&args); let calls = Rc::clone(&script.calls);
    assert!(run(prepared, script).is_err()); assert_eq!(calls.get(), 0);

    let prepared = prepare(&compiler, TEXT, &args, mapping()); let expected = prepared.preflight_metadata();
    let mut result = run(prepared, Script::new(&args)).unwrap();
    assert!(prepare(&other, TEXT, &args, mapping()).preflight_metadata().verify_completed(&result).is_err());
    result.input_encoding = None; assert!(expected.verify_completed(&result).is_err());
    result.input_encoding = Some(policy());
    result.input_encoding_work = None; assert!(expected.verify_completed(&result).is_err());
    result.input_encoding_work = expected.encoding_work(); expected.verify_completed(&result).unwrap();
    let byte = run(prepare(&planner(), TEXT, &args, mapping()), Script::new(&args)).unwrap();
    let value = serde_json::to_value(&byte).unwrap();
    assert!(value.get("input_encoding").is_none()); assert!(value.get("input_encoding_work").is_none());
}

#[test]
fn compact_selected_array_union_keeps_original_occurrences_and_every_origin() {
    let args = args(r#"{"type":"array","items":{"type":"string","x-fnlp-source":"verbatim"},"maxItems":8}"#,
        ExtractionBatchGrounding::SourceMembership);
    let source = "éAéAéAéA";
    let mut limits = limits(); limits.root_array_union = Some(union_limits());
    for selected in [None, Some(Int4JsonSparseLimits { max_rows_per_step: 65 })] {
        let compiler = compact(policy());
        let compiler = if let Some(selected) = selected { compiler.with_selected_rows(selected).unwrap() } else { compiler };
        let prepared = prepare(&compiler, source, &args, limits); let expected = prepared.preflight_metadata();
        let mut script = Script::new(&args); script.json = Some(r#"["é"]"#.to_owned());
        let completed = run(prepared, script).unwrap(); expected.verify_completed(&completed).unwrap();
        assert_eq!(completed.max_selected_rows, selected.map(|limits| limits.max_rows_per_step));
        let union = completed.root_array_union.as_ref().unwrap();
        assert_eq!(union.json(), r#"["é"]"#); assert_eq!(union.records()[0].origins.len(), 2);
        assert_eq!(completed.verification_used.matches, 4);
        for chunk in completed.mapped.root().value().chunks() {
            for span in &chunk.original_fields[0].spans { assert_eq!(&source[span.byte_start..span.byte_end], "é"); }
        }
    }
}

#[test]
fn cancellation_inside_schema_probes_and_final_compilation_retains_every_kind() {
    struct Control { calls: usize, stop: Option<usize>, cause: DecodeCancellationKind }
    impl DecodeStepControl for Control {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.calls += 1; (self.stop == Some(self.calls)).then_some(self.cause)
        }
    }
    let compiler = compact(policy()); let args = source_args();
    let mut observed = Control { calls: 0, stop: None, cause: DecodeCancellationKind::Deadline };
    compiler.plan_document_with_control(TEXT, &args, mapping(), &mut observed).unwrap();
    assert!(observed.calls > 16);
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        for stop in [1, 4, observed.calls / 2, observed.calls - 1, observed.calls] {
            let mut control = Control { calls: 0, stop: Some(stop), cause };
            let error = compiler.plan_document_with_control(TEXT, &args, mapping(), &mut control).err().unwrap();
            assert_eq!(error.cancellation(), Some(cause)); assert_eq!(control.calls, stop);
            assert!(!format!("{error:?}").contains(TEXT));
        }
    }
}
