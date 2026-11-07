//! Private completion/lifetime fixtures for the explicit serial head policy.
use super::*;
use crate::{grammar::runtime::JsonProgram,
    tasks::extract::grouped_int4::{INT4_EXTRACT_VERSION, INT4_SPARSE_EXTRACT_VERSION}};

fn selected(cap: usize) -> Int4ExtractionBatchPlanner {
    planner(WIDTH, None).with_selected_rows(Int4JsonSparseLimits { max_rows_per_step: cap }).unwrap()
}

#[test]
fn selected_output_retains_guards_and_all_reserved_work_across_flush_epochs() {
    for cap in [7, NANBEIGE_VOCAB_SIZE] {
        let compiler = selected(cap); let prepared = compiler.prepare(document()).unwrap();
        let work = prepared.planned_work(); let state = Rc::new(State::default());
        let (mut engine, mut admission, mut run) = host(&state, Mode::Success);
        run.limits.max_model_work = work;
        let output = execute_admitted(prepared, &mut engine, &mut admission, &mut run,
            context(1, 1), &mut Continue).unwrap();
        assert!(state.live.get()); assert!(!state.native.get()); assert!(!run.failed);
        assert_eq!(run.reserved, work); assert_eq!(run.mask_visits, limits().masks.max_visits_per_item);
        let bytes = canonjson::canonical_string(&output).unwrap();
        assert!(bytes.contains(INT4_SPARSE_EXTRACT_VERSION));
        for private in ["PRIVATE_SOURCE", "prompt_digest", "taskir_digest", "decision_policy_digest", "guard"] {
            assert!(!bytes.contains(private));
        }
        // The fixture projects fewer rows than the admitted maximum. Neither
        // unused rows nor the completed output guard renew the run allowance.
        assert!(state.live.get()); drop(output); assert_eq!(state.drops.get(), 1);
        let error = execute_admitted(compiler.prepare(document()).unwrap(), &mut engine, &mut admission,
            &mut run, context(3, 2), &mut Continue).err().unwrap();
        assert!(error.stop && run.failed); assert_eq!(error.fault.code, BatchCode::WorkLimit);
        assert_eq!(run.reserved, work); assert_eq!(state.admissions.get(), 1); assert_eq!(state.calls.get(), 1);
    }
}

#[test]
fn selected_cancel_and_invalid_completion_drain_and_permanently_close_reuse() {
    let compiler = selected(7);
    for mode in [Mode::Cancel, Mode::Dirty, Mode::Oversized, Mode::WrongWork, Mode::WrongGroup] {
        let state = Rc::new(State::default()); let (mut engine, mut admission, mut run) = host(&state, mode);
        let error = execute_admitted(compiler.prepare(document()).unwrap(), &mut engine, &mut admission,
            &mut run, context(1, 1), &mut Continue).err().unwrap();
        assert!(error.stop && run.failed); assert!(!state.live.get()); assert!(!state.native.get());
        if matches!(mode, Mode::Cancel) {
            assert_eq!(error.fault.cancellation, Some(DecodeCancellationKind::Deadline));
        }
        engine.clean = true; engine.mode = Mode::Success;
        assert!(execute_admitted(compiler.prepare(document()).unwrap(), &mut engine, &mut admission,
            &mut run, context(2, 1), &mut Continue).is_err());
        assert_eq!(state.calls.get(), 1); assert_eq!(state.drops.get(), 1);
        assert!(run.reserved.forward_positions > 0 && run.mask_visits > 0);
    }
}

#[test]
fn sparse_completion_rechecks_unicode_evidence_eos_exact_work_and_strategy() {
    let schema = r#"{"type":"string","const":"上海","x-fnlp-source":"verbatim"}"#;
    let source = "é 上海 and 上海 <|im_start|>";
    let input = BatchDocument { id: "private-id".to_owned(), text: source.to_owned(),
        task_args: Some(ExtractionBatchArgs { schema: schema.to_owned(),
            grounding: ExtractionBatchGrounding::SourceMembership, budget: budget() }) };
    let prepared = selected(7).prepare(input).unwrap();
    let json = "\"上海\"";
    let program = JsonProgram::compile_with_source(schema, source,
        CompileLimits::default(), SourceRuntimeLimits::default()).unwrap();
    let fields = program.source_fields(json).unwrap();
    assert_eq!(fields.len(), 1); assert_eq!(fields[0].spans.len(), 2);
    for span in &fields[0].spans {
        assert_eq!(&source[span.byte_start..span.byte_end], "上海");
        assert_eq!(source[..span.byte_start].chars().count(), span.scalar_start);
    }
    let mut tokens = EmbeddedTokenizer::pinned().unwrap().tokenizer().encode_byte_fallback_only(json.as_bytes()).unwrap();
    tokens.push(prepared.plan.options().eos_token_id);
    let work = Int4Work::for_sequence(WIDTH, 0, prepared.plan.prompt_tokens() + tokens.len() - 1,
        2 * tokens.len()).unwrap();
    let completed = Int4ExtractRun { schema_version: 1, execution: INT4_SPARSE_EXTRACT_VERSION.to_owned(),
        group_width: WIDTH, model_work: work,
        result: ExtractResult { schema_version: 2, task_spec_version: "extract-v1".to_owned(),
            score_space: ScoreSpace::NotComputed, grounding: ExtractionGrounding::SourceMembership,
            source_fields: fields,
            output: JsonDecodeOutput { schema_version: 1, numerics_profile: STRICT_INT4_PROFILE.to_owned(),
                token_ids: tokens, json: json.to_owned(), forward_positions: work.forward_positions,
                projected_logits: work.projected_logits, mask_node_visit_charge: 20 } } };
    assert!(valid_result(&prepared, &completed, 20));
    for axis in 0..12 {
        let mut changed = completed.clone();
        match axis {
            0 => changed.execution = INT4_EXTRACT_VERSION.to_owned(),
            1 => changed.result.source_fields.clear(),
            2 => changed.result.source_fields[0].spans[0].scalar_end += 1,
            3 => changed.result.source_fields[0].spans.pop().map(|_| ()).unwrap(),
            4 => changed.result.output.json = "\"北京\"".to_owned(),
            5 => { changed.result.output.token_ids.pop(); },
            6 => changed.result.output.token_ids[0] = prepared.plan.options().eos_token_id,
            7 => changed.model_work.attention_pairs += 1,
            8 => changed.model_work.projections.dot_products += 1,
            9 => changed.result.output.projected_logits += 1,
            10 => changed.result.output.mask_node_visit_charge = 21,
            _ => changed.result.grounding = ExtractionGrounding::NotRequested,
        }
        assert!(!valid_result(&prepared, &changed, 20), "corruption {axis}");
    }
}
