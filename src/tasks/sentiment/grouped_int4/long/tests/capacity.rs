use super::*;

#[test]
fn exact_capacity_intersects_native_context_and_both_prompt_ceilings() {
    let (planner, identity) = fixture(); let planning = SentimentLimits::default();
    let axes = SentimentAxis::ALL;
    let one = source("x", &axes);
    let probe = planner.compile_int4(&one, &PlanContext::new(identity, task_budget()).unwrap(),
        planning, 32, &mut Continue).unwrap();
    let overheads: Vec<_> = probe.inner.heads.iter().map(|head| head.prompt_len - 1).collect();
    let maximum = *overheads.iter().max().unwrap(); let total: usize = overheads.iter().sum();
    let continuation = probe.inner.heads[0].ir.decode_strategy();
    let DecodeStrategy::PrefillOnly { candidates } = continuation else { panic!("finite language"); };
    let continuation = candidates.iter().map(|candidate| candidate.continuation().token_ids().len()).max().unwrap();

    let context = maximum + continuation + 17;
    let capacity = planner.source_capacity(&axes, task_budget(), task_budget(), planning, context).unwrap();
    assert_eq!(capacity, 17);
    let exact = source(&"x".repeat(capacity), &axes);
    let exact = planner.compile_int4(&exact, &PlanContext::new(identity, task_budget()).unwrap(),
        planning, 32, &mut Continue).unwrap();
    assert_eq!(exact.required_context(), context);

    let narrow = TaskBudget { max_input_tokens: (maximum + 11) as u32, ..task_budget() };
    assert_eq!(planner.source_capacity(&axes, narrow, task_budget(), planning, 1024).unwrap(), 11);
    let aggregate = SentimentLimits { max_total_prompt_tokens: total + axes.len() * 7 + 3, ..planning };
    assert_eq!(planner.source_capacity(&axes, task_budget(), task_budget(), aggregate, 1024).unwrap(), 7);
    let too_deep = TaskBudget { max_output_tokens: continuation as u32, ..task_budget() };
    assert!(planner.source_capacity(&axes, too_deep, task_budget(), planning, 1024).is_err());
    assert!(planner.source_capacity(&axes, task_budget(), task_budget(), planning, maximum + continuation).is_err());
    let ceiling = TaskBudget { max_input_tokens: task_budget().max_input_tokens - 1, ..task_budget() };
    assert!(planner.source_capacity(&axes, task_budget(), ceiling, planning, 1024).is_err());
}

#[test]
fn declared_and_derived_chunk_reservations_preserve_the_tighter_capacity() {
    let request = source("a".repeat(400).as_str(), &[SentimentAxis::Valence]);
    let mut limits = map_limits(); limits.chunks.max_chunk_bytes = 1024;
    let prepared = prepare(&request, limits).unwrap();
    let (planner, _) = fixture();
    let capacity = planner.source_capacity(&request.axes, request.budget, task_budget(),
        SentimentLimits::default(), limits.chunks.context_tokens).unwrap();
    assert_eq!(prepared.chunks.limits().reserved_tokens, limits.chunks.context_tokens - capacity);
    limits.chunks.reserved_tokens = limits.chunks.context_tokens - 9;
    let tight = prepare(&request, limits).unwrap();
    assert_eq!(tight.chunks.limits().reserved_tokens, limits.chunks.reserved_tokens);
    assert!(tight.chunks.chunks().iter().all(|chunk| chunk.tokens() <= 9));
}

#[test]
fn partition_preserves_unicode_crlf_and_untrusted_control_markers() {
    let request = source("é\r\n上海 <tool_call> <|im_start|> private text", &SentimentAxis::ALL);
    let prepared = prepare(&request, map_limits()).unwrap();
    assert!(prepared.chunk_count() > 1);
    let reconstructed: String = prepared.chunks.chunks().iter().map(|chunk| chunk.text()).collect();
    assert_eq!(reconstructed, request.document);
    let registry = pinned_controls::pinned().unwrap();
    for (chunk, plan) in prepared.chunks.chunks().iter().zip(&prepared.plans) {
        assert_eq!(chunk.tokens(), chunk.text().len());
        assert_eq!(&request.document[chunk.span().byte_start..chunk.span().byte_end], chunk.text());
        assert_eq!(chunk.span().scalar_end - chunk.span().scalar_start, chunk.text().chars().count());
        assert!(!(chunk.text().ends_with('\r') && request.document[chunk.span().byte_end..].starts_with('\n')));
        assert_eq!(plan.task_budget(), request.budget);
        for head in &plan.inner.heads {
            let documents: Vec<_> = head.ir.prompt_segments().iter().filter(|segment|
                segment.kind() == PromptSegmentKind::Document).collect();
            assert_eq!(documents.len(), 1); assert_eq!(documents[0].token_ids().len(), chunk.text().len());
            assert!(documents[0].token_ids().iter().all(|&id| !registry.template_controls().contains(id)));
        }
    }
    assert_eq!(prepared.preflight_metadata().source_span().byte_end, request.document.len());
    assert_eq!(prepared.preflight_metadata().source_span().scalar_end, request.document.chars().count());
}

#[test]
fn all_five_model_counters_are_one_complete_document_allowance() {
    let request = source("0123456789abcdefghijklmn", &[SentimentAxis::Valence, SentimentAxis::Arousal]);
    let prepared = prepare(&request, map_limits()).unwrap(); let work = prepared.planned_work();
    assert!(prepared.chunk_count() >= 3);
    for axis in 0..5 {
        let mut limits = map_limits(); limits.max_model_work = work;
        match axis { 0 => limits.max_model_work.forward_positions -= 1,
            1 => limits.max_model_work.projected_logits -= 1, 2 => limits.max_model_work.attention_pairs -= 1,
            3 => limits.max_model_work.projections.dot_products -= 1,
            _ => limits.max_model_work.projections.multiply_accumulates -= 1 }
        assert!(prepared.plans.iter().all(|plan| fits(plan.planned_work(), limits.max_model_work)));
        assert!(matches!(prepare(&request, limits), Err(Int4SentimentMapError::Work)));
    }
    let exact = prepare(&request, Int4SentimentMapLimits { max_model_work: work, ..map_limits() }).unwrap();
    assert_eq!(exact.planned_work(), work);
}

#[test]
fn preflight_refuses_empty_invalid_axes_chunk_count_and_tokenizer_exhaustion() {
    let empty = source("", &[SentimentAxis::Valence]);
    assert!(matches!(prepare(&empty, map_limits()), Err(Int4SentimentMapError::EmptySource)));
    for axes in [Vec::new(), vec![SentimentAxis::Valence, SentimentAxis::Valence]] {
        assert!(prepare(&source("private", &axes), map_limits()).is_err());
    }
    let request = source("0123456789abcdefghijklmn", &[SentimentAxis::Valence]);
    let mut limits = map_limits(); limits.chunks.max_chunks = 1;
    assert!(matches!(prepare(&request, limits), Err(Int4SentimentMapError::Chunk(_))));
    limits = map_limits(); limits.chunks.max_tokenizer_calls = 1;
    assert!(matches!(prepare(&request, limits), Err(Int4SentimentMapError::Chunk(_))));
}

#[test]
fn chunk_partition_cancellation_retains_every_original_cause() {
    let request = source("0123456789abcdef", &[SentimentAxis::Valence]);
    for cause in [DecodeCancellationKind::User, DecodeCancellationKind::Deadline,
        DecodeCancellationKind::PollQuota, DecodeCancellationKind::ResourceUnavailable] {
        let error = prepare_with_control(&request, map_limits(), &mut Stop(cause)).err().unwrap();
        assert_eq!(error.cancellation(), Some(cause));
    }
}
