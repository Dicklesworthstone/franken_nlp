//! Complete lineage and all-axis accounting checks, independent of scheduling.
use super::*;
use std::collections::BTreeSet;

pub(super) fn verify(expected: &Int8SummaryHierarchyPreflight, run: &Int8SummaryHierarchyRun)
    -> Result<(), Int8SourceMapError> {
    let d = expected.discovery;
    let map = &run.discovery;
    let root = map.mapped.root();
    let v = d.limits.verification;
    if run.schema_version != 1 || run.execution != INT8_SUMMARY_HIERARCHY_EXECUTION
        || run.numerics_profile != STRICT_INT8_PROFILE || run.semantics != SEMANTICS || run.limits != expected.limits
        || run.citation_guarantee != CitationGuarantee::StructuralSourceMembership
        || run.semantic_support != SummarySemanticSupport::NotAssessed
        || run.untrusted_fields != ["passes", "discovery.mapped.root.value"]
        || map.schema_version != 1 || map.execution != INT8_SOURCE_MAP_EXECUTION
        || map.semantics != "independent-chunks-no-cross-chunk-reasoning-v1"
        || root.chunk_range() != (0..d.chunks) || root.value().len() != d.chunks || root.source_span() != d.source_span
        || map.planned_model_work != d.discovery_work || map.reserved_mask_node_visits != d.discovery_masks
        || !within(map.model_work, d.discovery_work) || map.mask_node_visit_charge > d.discovery_masks
        || run.reserved_model_work != expected.reserved_work || run.reserved_mask_node_visits != expected.reserved_masks
        || run.passes.len() > expected.limits.max_passes || run.levels.len() > expected.limits.max_levels
        || run.verification_used.fields > v.max_fields || run.verification_used.matches > v.max_matches
        || run.verification_used.scan_steps > v.max_scan_steps
        || run.tokenizer_work.calls > expected.limits.max_tokenizer_calls
        || run.tokenizer_work.bytes > expected.limits.max_tokenizer_bytes
        || run.tokenizer_work.calls < run.passes.len() {
        return Err(invalid());
    }
    let mut initial = EvidenceSize { segments: 0, bytes: 0 };
    for chunk in root.value().chunks() {
        let SourceTaskResult::Summarize(raw) = &chunk.native.result else { return Err(invalid()); };
        let mut seen = BTreeSet::new();
        for bullet in &raw.bullets { for citation in &bullet.citations {
            if seen.insert(citation.quote.as_str()) {
                initial.bytes = initial.bytes.checked_add(if initial.segments == 0 { 0 } else { 2 })
                    .and_then(|n| n.checked_add(citation.quote.len())).ok_or_else(invalid)?;
                initial.segments = initial.segments.checked_add(1).ok_or_else(invalid)?;
            }
        } }
    }
    if run.initial_evidence != initial || initial.segments > d.limits.max_evidence_segments
        || initial.bytes > d.limits.max_evidence_bytes { return Err(invalid()); }
    let mut planned = map.planned_model_work;
    let mut actual = map.model_work;
    let mut masks = map.mask_node_visit_charge;
    let mut counted_bytes = 0_u64;
    for pass in &run.passes {
        verify_pass(expected, pass)?;
        planned = add_work(planned, pass.planned_model_work).ok_or_else(invalid)?;
        actual = add_work(actual, pass.native.model_work).ok_or_else(invalid)?;
        masks = masks.checked_add(mask_charge(&pass.native.result)?).ok_or_else(invalid)?;
        counted_bytes = counted_bytes.checked_add(pass.input_bytes as u64).ok_or_else(invalid)?;
    }
    if run.planned_model_work != planned || run.model_work != actual || run.mask_node_visit_charge != masks
        || !within(planned, expected.reserved_work) || !within(actual, planned) || masks > expected.reserved_masks
        || run.tokenizer_work.bytes < counted_bytes { return Err(invalid()); }
    if initial.segments == 0 {
        if initial.bytes != 0 || !run.passes.is_empty() || !run.levels.is_empty() || run.final_pass.is_some()
            || run.status != SummarySynthesisStatus::NoEvidenceCollected || run.tokenizer_work != HierarchyTokenizerWork::default() {
            return Err(invalid());
        }
        return Ok(());
    }
    if run.levels.is_empty() { return Err(invalid()); }
    let mut frontier = initial;
    let mut next_pass = 0;
    let mut final_pass = None;
    for (level_index, level) in run.levels.iter().enumerate() {
        if level.input != frontier || level.passes.start != next_pass || level.passes.start >= level.passes.end
            || level.passes.end > run.passes.len() { return Err(invalid()); }
        let passes = &run.passes[level.passes.clone()];
        let mut next_segment = 0;
        let mut bytes = 0_usize;
        for (group, pass) in passes.iter().enumerate() {
            if pass.level != level_index || pass.group != group || pass.input_segments.start != next_segment
                || pass.input_segments.end > frontier.segments { return Err(invalid()); }
            next_segment = pass.input_segments.end;
            bytes = bytes.checked_add(if group == 0 { 0 } else { 2 })
                .and_then(|n| n.checked_add(pass.input_bytes)).ok_or_else(invalid)?;
        }
        if next_segment != frontier.segments || bytes != frontier.bytes { return Err(invalid()); }
        next_pass = level.passes.end;
        match level.next_evidence {
            None => {
                if passes.len() != 1 || level_index + 1 != run.levels.len() { return Err(invalid()); }
                final_pass = Some(level.passes.start);
            }
            Some(next) => {
                if passes.len() < 2 || next != transport::selected_size(passes)?
                    || next.segments > d.limits.max_evidence_segments || next.bytes > d.limits.max_evidence_bytes
                    || next.bytes >= frontier.bytes || (next.segments == 0) != (next.bytes == 0) {
                    return Err(invalid());
                }
                if next.segments == 0 && level_index + 1 != run.levels.len() { return Err(invalid()); }
                frontier = next;
            }
        }
    }
    if next_pass != run.passes.len() || run.final_pass != final_pass { return Err(invalid()); }
    let status = match final_pass {
        Some(index) if !run.passes[index].bullets.is_empty() => SummarySynthesisStatus::Synthesized,
        Some(_) => SummarySynthesisStatus::NoBulletsProduced,
        None if frontier.segments == 0 => SummarySynthesisStatus::NoBulletsProduced,
        _ => return Err(invalid()),
    };
    if run.status != status { return Err(invalid()); }
    Ok(())
}

pub(super) fn verify_pass(expected: &Int8SummaryHierarchyPreflight, pass: &SummaryHierarchyPass)
    -> Result<(), Int8SourceMapError> {
    let native = &pass.native;
    let SourceTaskResult::Summarize(raw) = &native.result else { return Err(invalid()); };
    evidence::check_summary(raw, expected.discovery.options)?;
    if pass.input_segments.start >= pass.input_segments.end || pass.input_bytes == 0 || pass.input_tokens == 0
        || pass.input_tokens > expected.capacity.max_source_tokens() || raw.generated_token_ids.is_empty()
        || raw.generated_token_ids.len() > expected.output_tokens { return Err(invalid()); }
    let prompt = expected.capacity.scaffold_tokens().checked_add(pass.input_tokens).ok_or_else(invalid)?;
    let planned = constrained_int8::planned_work(prompt, expected.output_tokens).map_err(|_| invalid())?;
    let actual = constrained_int8::planned_work(prompt, raw.generated_token_ids.len()).map_err(|_| invalid())?;
    if native.schema_version != 1 || native.execution != INT8_SOURCE_EXECUTION
        || pass.planned_model_work != planned || native.model_work != actual
        || !within(planned, expected.discovery.synthesis_reserve) || !within(actual, planned)
        || raw.forward_positions != actual.forward_positions || raw.projected_logits != actual.projected_logits
        || raw.mask_node_visit_charge > expected.discovery.synthesis_masks || raw.bullets.len() != pass.bullets.len() {
        return Err(invalid());
    }
    for (raw, lifted) in raw.bullets.iter().zip(&pass.bullets) {
        if raw.text != lifted.text || raw.citations.len() != lifted.citations.len() { return Err(invalid()); }
        for (local, original) in raw.citations.iter().zip(&lifted.citations) {
            if local.quote != original.quote { return Err(invalid()); }
            evidence::check_citation(original, expected.discovery.options.max_quote_scalars)?;
            if original.spans.iter().any(|s| s.byte_end > expected.source_span().byte_end
                || s.scalar_end > expected.source_span().scalar_end) { return Err(invalid()); }
        }
    }
    Ok(())
}
