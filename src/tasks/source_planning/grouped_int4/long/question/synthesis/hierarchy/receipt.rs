//! Complete ordered frontier, native work and final-answer consistency checks.
use super::*;
use std::collections::BTreeSet;

pub(super) fn verify(expected: &Int4QuestionHierarchyPreflight, run: &Int4QuestionHierarchyRun) -> Result<(), Int4SourceMapError> {
    let base = expected.base; let map = &run.discovery;
    base.discovery.verify_completed(map)?;
    if run.schema_version != 1 || run.execution != INT4_QUESTION_HIERARCHY_EXECUTION
        || run.numerics_profile != STRICT_INT4_PROFILE || run.group_width != expected.group_width()
        || run.semantics != SEMANTICS || run.limits != expected.limits
        || run.calibration != AnswerCalibration::Uncalibrated
        || run.citation_guarantee != CitationGuarantee::StructuralSourceMembership
        || run.semantic_support != SummarySemanticSupport::NotAssessed
        || run.untrusted_fields != ["passes", "discovery.mapped.root.value"]
        || run.reserved_model_work != expected.reserved_work || run.reserved_mask_node_visits != expected.reserved_masks
        || run.passes.len() > expected.limits.max_passes || run.levels.len() > expected.limits.max_levels
        || !run.verification_used.fits(base.discovery.verification)
        || run.verification_used.fields < map.verification_used.fields || run.verification_used.matches < map.verification_used.matches
        || run.verification_used.scan_steps < map.verification_used.scan_steps
        || run.tokenizer_work.calls > expected.limits.max_tokenizer_calls || run.tokenizer_work.bytes > expected.limits.max_tokenizer_bytes
        || run.tokenizer_work.calls < run.passes.len() { return Err(invalid()); }
    let mut frontier = Vec::new();
    for chunk in map.mapped.root().value().chunks() {
        let mut seen = BTreeSet::new();
        for citation in &chunk.citations {
            if seen.insert(citation.quote.as_str()) {
                frontier.try_reserve(1).map_err(|_| Int4SourceMapError::Allocation)?;
                frontier.push(citation.quote.as_str());
            }
        }
    }
    let initial = transport::quote_size(&frontier)?;
    if run.initial_evidence != initial || initial.passages > base.limits.max_evidence_passages
        || initial.bytes > base.limits.max_evidence_bytes { return Err(invalid()); }
    let mut planned = map.planned_model_work; let mut actual = map.model_work; let mut masks = map.mask_node_visit_charge;
    let mut counted_bytes = 0_u64;
    for pass in &run.passes {
        verify_pass(expected, pass)?;
        planned = add_work(planned, pass.planned_model_work).ok_or_else(invalid)?;
        actual = add_work(actual, pass.native.model_work).ok_or_else(invalid)?;
        let SourceTaskResult::Answer(raw) = &pass.native.result else { return Err(invalid()); };
        masks = masks.checked_add(raw.mask_node_visit_charge).ok_or_else(invalid)?;
        counted_bytes = counted_bytes.checked_add(pass.context_bytes as u64).ok_or_else(invalid)?;
    }
    if run.planned_model_work != planned || run.model_work != actual || run.mask_node_visit_charge != masks
        || !fits(planned, expected.reserved_work) || !fits(actual, planned) || masks > expected.reserved_masks
        || run.tokenizer_work.bytes < counted_bytes { return Err(invalid()); }
    if initial.passages == 0 {
        if initial.bytes != 0 || map.answered_chunks != 0 || !run.passes.is_empty() || !run.levels.is_empty() || run.final_pass.is_some()
            || run.status != QuestionHierarchyStatus::NoEvidenceCollected || run.tokenizer_work != HierarchyTokenizerWork::default() {
            return Err(invalid());
        }
        return Ok(());
    }
    if initial.bytes == 0 || map.answered_chunks == 0 || run.levels.is_empty() { return Err(invalid()); }
    let mut next_pass = 0; let mut final_pass = None;
    for (level_index, level) in run.levels.iter().enumerate() {
        let size = transport::quote_size(&frontier)?;
        if level.input != size || level.passes.start != next_pass || level.passes.start >= level.passes.end
            || level.passes.end > run.passes.len() { return Err(invalid()); }
        let passes = &run.passes[level.passes.clone()];
        let mut next_passage = 0;
        for (group, pass) in passes.iter().enumerate() {
            if pass.level != level_index || pass.group != group || pass.input_passages.start != next_passage
                || pass.input_passages.end > frontier.len() { return Err(invalid()); }
            let quotes = &frontier[pass.input_passages.clone()];
            if transport::quote_size(quotes)?.bytes != pass.input_bytes { return Err(invalid()); }
            let mut ids = vector(quotes.len())?;
            for index in pass.input_passages.clone() { ids.push(format!("evidence_{index}")); }
            let bytes = transport::context_bytes(expected.capacity.question_bytes,
                ids.iter().zip(quotes).map(|(id, text)| (id.as_str(), *text)))?;
            if bytes != pass.context_bytes { return Err(invalid()); }
            // Native local citations must still designate an actual quote
            // passage in this group; synthetic joins/question/manifest cannot.
            verify_local_passages(pass, &ids, quotes)?;
            next_passage = pass.input_passages.end;
        }
        if next_passage != frontier.len() { return Err(invalid()); }
        next_pass = level.passes.end;
        match level.next_evidence {
            None => {
                if passes.len() != 1 || level_index + 1 != run.levels.len() { return Err(invalid()); }
                final_pass = Some(level.passes.start);
            }
            Some(next) => {
                let selected = transport::selected_quotes(passes)?;
                if passes.len() < 2 || next != transport::quote_size(&selected)?
                    || next.passages > base.limits.max_evidence_passages || next.bytes > base.limits.max_evidence_bytes
                    || next.bytes >= size.bytes || (next.passages == 0) != (next.bytes == 0) { return Err(invalid()); }
                if next.passages == 0 && level_index + 1 != run.levels.len() { return Err(invalid()); }
                frontier = selected;
            }
        }
    }
    if next_pass != run.passes.len() || run.final_pass != final_pass { return Err(invalid()); }
    let status = match final_pass {
        Some(index) => match run.passes[index].answer.status {
            SynthesisStatus::Answered => QuestionHierarchyStatus::Answered,
            SynthesisStatus::Abstained => QuestionHierarchyStatus::Abstained,
            SynthesisStatus::NoEvidenceCollected => return Err(invalid()),
        },
        None if frontier.is_empty() => QuestionHierarchyStatus::EvidenceExhausted,
        _ => return Err(invalid()),
    };
    if run.status != status { return Err(invalid()); }
    Ok(())
}
pub(super) fn verify_pass(expected: &Int4QuestionHierarchyPreflight, pass: &QuestionHierarchyPass) -> Result<(), Int4SourceMapError> {
    let native = &pass.native;
    let SourceTaskResult::Answer(raw) = &native.result else { return Err(invalid()); };
    let status = evidence::check_answer(raw, expected.base.options)?;
    let ids = &raw.generated_token_ids; let eos = expected.capacity.eos;
    if pass.input_passages.start >= pass.input_passages.end || pass.input_passages.end > expected.base.limits.max_evidence_passages
        || pass.input_bytes == 0 || pass.input_bytes > expected.base.limits.max_evidence_bytes
        || pass.context_bytes > expected.capacity.input_bytes
        || pass.input_bytes.checked_add(expected.capacity.question_bytes).is_none_or(|n| pass.context_bytes <= n)
        || pass.prompt_tokens <= expected.capacity.scaffold || pass.prompt_tokens > expected.capacity.prompt
        || ids.is_empty() || ids.len() > expected.base.max_output_tokens || ids.last() != Some(&eos)
        || ids[..ids.len() - 1].iter().any(|id| *id == eos || *id as usize >= NANBEIGE_VOCAB_SIZE) { return Err(invalid()); }
    let planned = constrained_int4::planned_work(expected.group_width(), pass.prompt_tokens, expected.base.max_output_tokens)
        .map_err(Int4SourceError::from)?;
    let actual = constrained_int4::planned_work(expected.group_width(), pass.prompt_tokens, ids.len()).map_err(Int4SourceError::from)?;
    if native.schema_version != 1 || native.execution != INT4_SOURCE_EXECUTION || native.group_width != expected.group_width()
        || pass.planned_model_work != planned || native.model_work != actual
        || !fits(planned, expected.base.synthesis_reserve) || !fits(actual, planned)
        || raw.forward_positions != actual.forward_positions || raw.projected_logits != actual.projected_logits
        || raw.mask_node_visit_charge > expected.base.synthesis_masks || pass.answer.status != status
        || raw.answer != pass.answer.answer || raw.citations.len() != pass.answer.citations.len() { return Err(invalid()); }
    for (local, original) in raw.citations.iter().zip(&pass.answer.citations) {
        if local.quote != original.quote || local.spans.is_empty()
            || local.occurrence != if local.spans.len() == 1 { SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous } {
            return Err(invalid());
        }
        evidence::check_original_citation(original, expected.base.options, expected.discovery().source_span())?;
    }
    Ok(())
}
fn verify_local_passages(pass: &QuestionHierarchyPass, ids: &[String], quotes: &[&str]) -> Result<(), Int4SourceMapError> {
    let SourceTaskResult::Answer(raw) = &pass.native.result else { return Err(invalid()); };
    // One bounded scalar-boundary index per passage. Recounting the entire
    // prefix for every repeated occurrence would make receipt checks quadratic.
    let mut boundaries = vector(quotes.len())?;
    for text in quotes {
        let mut positions = vector(text.len().checked_add(1).ok_or_else(invalid)?)?;
        positions.extend(text.char_indices().map(|(byte, _)| byte));
        positions.push(text.len());
        boundaries.push(positions);
    }
    for citation in &raw.citations {
        let mut previous = None;
        for occurrence in &citation.spans {
            let index = ids.iter().position(|id| id == &occurrence.passage_id).ok_or_else(invalid)?;
            let text = quotes[index]; let span = occurrence.span;
            let key = (index, span.byte_start);
            if previous.is_some_and(|old| old >= key) || span.byte_start >= span.byte_end
                || text.get(span.byte_start..span.byte_end) != Some(citation.quote.as_str())
                || boundaries[index].binary_search(&span.byte_start).ok() != Some(span.scalar_start)
                || boundaries[index].binary_search(&span.byte_end).ok() != Some(span.scalar_end) { return Err(invalid()); }
            previous = Some(key);
        }
    }
    Ok(())
}
