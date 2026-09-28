//! Long original documents -> independently grounded NER chunks -> ONE graph.
//! Chunk IDs never become document IDs. Only compact private witnesses survive
//! preparation; one source grammar/result is resident at a time during discovery.
use super::*;
use crate::{tasks::{mapreduce::{ChunkLimits, ChunkPlan, MapReduceError, CHUNK_PROFILE},
    source_planning::quantized::long::SourceMapTask},
    validation::grounded_fields::VerifiedSourceSpan};

pub const INT8_DOCUMENT_ENTITY_EXECUTION: &str = "portable-int8-chunk-ner-original-document-resolution-v1";

/// NER token/output/mask-per-item limits in entities apply to EACH chunk.
/// Native work, occurrence recovery, graph and mask-per-run limits cover the
/// complete snapshot, including all NER chunks AND subsequent pair scoring.
#[derive(Clone, Debug)]
pub struct Int8DocumentEntityConfig {
    pub entities: Int8EntityConfig,
    pub chunks: ChunkLimits,
    pub max_snapshot_chunks: usize,
}
struct ChunkWitness { span: VerifiedSourceSpan, identity: Sha256Digest, prompt_tokens: usize, work: Int8Work }
struct DocumentInput { document: EntityDocument, chunks: Vec<ChunkWitness> }

/// Consumed source ownership and private exact-plan witnesses, not a receipt
/// supplied by a caller. No Clone/Deserialize or model-free nonempty completion.
pub struct PreparedInt8DocumentEntityCorpus {
    inputs: Vec<DocumentInput>, source: Arc<SourceTaskPlanner>, resolver: Arc<ResolutionPlanner>,
    source_identity: ExecutionIdentity, resolution_identity: ExecutionIdentity,
    config: Int8DocumentEntityConfig, ner_work: Int8Work, masks: u64,
    chunks: usize, maximum_positions: usize, input_bytes: usize,
}

/// Partition with the pinned source encoder and exact trusted scaffold pricing.
/// Compile EVERY chunk before the first neural call; keep only its immutable
/// source extent, complete identity witness, prompt length and work commitment.
#[allow(clippy::too_many_arguments)]
pub fn prepare_int8_document_entities<C: DecodeStepControl>(mut documents: Vec<EntityDocument>,
    source: Arc<SourceTaskPlanner>, source_identity: ExecutionIdentity,
    resolver: Arc<ResolutionPlanner>, resolution_identity: ExecutionIdentity,
    config: Int8DocumentEntityConfig, control: &mut C)
    -> Result<PreparedInt8DocumentEntityCorpus, Int8EntityError> {
    resolve::checkpoint(control)?;
    validate(&config.entities, &source, &source_identity, &resolution_identity)?;
    validate_partition(&config)?;
    let input_bytes = validate_documents(&mut documents, &config.entities)?;
    let empty = ResolutionPlan::prepare(&[], config.entities.resolution, config.entities.graph, control)?;
    resolver.prepare_int8(&empty, &resolution_identity, config.entities.scoring, control)?;
    let context = PlanContext::new(&source_identity, config.entities.ner_budget).map_err(|_| Int8EntityError::Identity)?;
    let capacity = source.int8_map_capacity_with_control(&SourceMapTask::Ner(config.entities.ner.clone()),
        config.entities.ner_budget, &context, config.entities.source_planning, control)?;
    let limits = capacity.constrain_chunks(config.chunks)?;
    let mut inputs = reserved(documents.len())?;
    let (mut ner_work, mut masks, mut count, mut maximum_positions) = (Int8Work::default(), 0_u64, 0_usize, 0_usize);
    for document in documents {
        resolve::checkpoint(control)?;
        // An empty snapshot is a genuine zero-work case; an empty document is
        // not a lossless nonempty partition or an implicit NER-success receipt.
        if document.text.is_empty() { return Err(Int8EntityError::InvalidInput); }
        let partition = partition(&document.text, &source, limits, control)?;
        count = count.checked_add(partition.chunks().len()).filter(|&n| n <= config.max_snapshot_chunks)
            .ok_or(Int8EntityError::WorkBudget)?;
        let mut chunks = reserved(partition.chunks().len())?;
        for chunk in partition.chunks() {
            resolve::checkpoint(control)?;
            let plan = source_plan(chunk.text(), &source, &source_identity, &config.entities, control)?;
            let work = plan.planned_work();
            let next = plus(ner_work, work)?;
            let next_masks = masks.checked_add(config.entities.masks.max_visits_per_item)
                .filter(|&n| n <= config.entities.masks.max_visits_per_run).ok_or(Int8EntityError::WorkBudget)?;
            if !within(next, config.entities.max_model_work) { return Err(Int8EntityError::WorkBudget); }
            maximum_positions = maximum_positions.max(usize::try_from(work.forward_positions)
                .map_err(|_| Int8EntityError::WorkBudget)?);
            chunks.push(ChunkWitness { span: chunk.span(), identity: witness(plan.execution_identity())?,
                prompt_tokens: plan.prompt_tokens(), work });
            ner_work = next; masks = next_masks;
        }
        drop(partition);
        inputs.push(DocumentInput { document, chunks });
    }
    resolve::checkpoint(control)?;
    Ok(PreparedInt8DocumentEntityCorpus { inputs, source, resolver, source_identity, resolution_identity,
        config, ner_work, masks, chunks: count, maximum_positions, input_bytes })
}
fn validate_partition(config: &Int8DocumentEntityConfig) -> Result<(), Int8EntityError> {
    config.chunks.effective_token_limit().map_err(partition_error)?;
    if !(1..=256).contains(&config.chunks.max_chunks) || !(1..=16_384).contains(&config.max_snapshot_chunks)
        || config.chunks.max_input_bytes == 0
        || config.chunks.max_chunk_bytes > config.entities.source_planning.max_input_bytes
        || config.chunks.context_tokens > config.entities.source_planning.max_context_tokens {
        return Err(Int8EntityError::InvalidLimits);
    }
    Ok(())
}
fn partition<'s, C: DecodeStepControl>(text: &'s str, source: &SourceTaskPlanner, limits: ChunkLimits, control: &mut C)
    -> Result<ChunkPlan<'s>, Int8EntityError> {
    let mut cancelled = None;
    let result = ChunkPlan::build_with_checkpoints(text, limits, |part| {
        source.source_encoder().encode(part, limits.max_chunk_bytes, limits.max_chunk_bytes)
            .map(|d| d.total_token_count()).map_err(|_| MapReduceError::Tokenizer)
    }, || {
        if let Some(cause) = control.prefill_checkpoint(0) {
            cancelled = Some(cause); Err(MapReduceError::Cancelled)
        } else { Ok(()) }
    });
    if let Some(cause) = cancelled { return Err(ResolveError::Cancelled(cause).into()); }
    result.map_err(partition_error)
}
fn partition_error(error: MapReduceError) -> Int8EntityError {
    match error {
        MapReduceError::InvalidLimits => Int8EntityError::InvalidLimits,
        MapReduceError::InputBudget | MapReduceError::ChunkBudget | MapReduceError::TokenBudget
            | MapReduceError::TokenizerBudget => Int8EntityError::WorkBudget,
        MapReduceError::AllocationRefused => Int8EntityError::Allocation,
        _ => Int8EntityError::Accounting,
    }
}
impl PreparedInt8DocumentEntityCorpus {
    pub fn document_count(&self) -> usize { self.inputs.len() }
    pub fn chunk_count(&self) -> usize { self.chunks }
    pub fn input_bytes(&self) -> usize { self.input_bytes }
    pub fn ner_reserved_work(&self) -> Int8Work { self.ner_work }
    pub fn reserved_mask_visits(&self) -> u64 { self.masks }
    pub fn required_ner_context_tokens(&self) -> usize { self.maximum_positions }
    pub fn source_identity(&self) -> &ExecutionIdentity { &self.source_identity }
    pub fn resolution_identity(&self) -> &ExecutionIdentity { &self.resolution_identity }
    pub fn config(&self) -> &Int8DocumentEntityConfig { &self.config }
    /// Actual retained input/witness capacities only. Planners, compiler scratch,
    /// expanded mentions, graph plans and allocator overhead are host commitments.
    pub fn retained_input_bytes(&self) -> Result<u64, Int8EntityError> {
        let mut bytes = (self.inputs.capacity() as u64).checked_mul(size_of::<DocumentInput>() as u64)
            .ok_or(Int8EntityError::WorkBudget)?;
        for input in &self.inputs {
            bytes = bytes.checked_add(input.document.id.capacity() as u64)
                .and_then(|n| n.checked_add(input.document.text.capacity() as u64))
                .and_then(|n| n.checked_add((input.chunks.capacity() as u64).checked_mul(size_of::<ChunkWitness>() as u64)?))
                .ok_or(Int8EntityError::WorkBudget)?;
        }
        Ok(bytes)
    }
    /// One exclusive native engine executes all source chunks and both orders
    /// of every candidate pair. No chunk-level clustering or source concatenation.
    pub fn execute_with_control<C: DecodeStepControl>(self, engine: &mut StrictInt8Engine<'_>,
        vocabulary: &ExtractionVocabulary, control: &mut C) -> Result<Int8DocumentEntityRun, Int8EntityError> {
        resolve::checkpoint(control)?;
        check_native(&self, engine)?;
        let Self { inputs, source, resolver, source_identity, resolution_identity,
            config, ner_work, masks, chunks, .. } = self;
        let (maps, geometry) = collect_chunks(inputs, &config.entities, control, |input, control| {
            let plan = source_plan(&input.document.text, &source, &source_identity, &config.entities, control)?;
            check_rebuilt(input, &plan)?;
            let work = input.work;
            let result = plan.execute_with_control(plan.execution_identity(), engine, vocabulary,
                Int8JsonBudget { native: Int8RunBudget::exact(work), json: JsonWorkBudget {
                    max_forward_positions: work.forward_positions, max_projected_logits: work.projected_logits,
                    max_kv_bytes: config.entities.ner_budget.max_kv_bytes,
                    max_total_mask_node_visits: config.entities.masks.max_visits_per_item,
                    mask_limits: config.entities.masks.per_mask,
                } }, control)?;
            if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) { return Err(Int8EntityError::Accounting); }
            Ok(result)
        })?;
        resolve::checkpoint(control)?;
        let plan = ResolutionPlan::prepare(&maps.documents, config.entities.resolution, config.entities.graph, control)?;
        // Early EOS NEVER refunds NER's reserved work into the pair stage.
        let scoring = remaining_scoring(&config.entities, ner_work)?;
        let prepared = resolver.prepare_int8(&plan, &resolution_identity, scoring, control)?;
        let pair_work = prepared.planned_work();
        let mut admitted = reserved(prepared.pair_count())?;
        for identity in prepared.execution_identities() {
            resolve::checkpoint(control)?; admitted.push(identity.clone());
        }
        let resolution = if prepared.pair_count() == 0 {
            prepared.finalize_without_model(control)?
        } else {
            prepared.execute_with_control(&admitted, engine, Int8ScoringBudget {
                native: Int8RunBudget::exact(pair_work), max_kv_bytes: config.entities.scoring.planning.per_head.max_kv_bytes,
            }, control)?
        };
        drop(admitted); drop(plan);
        let output = finish(maps, resolution, ner_work, masks, &config.entities, control)?;
        finish_document(output, geometry, chunks, config.entities.max_result_bytes, control)
    }
    pub fn finalize_without_model<C: DecodeStepControl>(self, control: &mut C)
        -> Result<Int8DocumentEntityRun, Int8EntityError> {
        resolve::checkpoint(control)?;
        if !self.inputs.is_empty() || self.chunks != 0 || self.ner_work != Int8Work::default() || self.masks != 0 {
            return Err(Int8EntityError::Accounting);
        }
        let config = &self.config.entities;
        let plan = ResolutionPlan::prepare(&[], config.resolution, config.graph, control)?;
        let prepared = self.resolver.prepare_int8(&plan, &self.resolution_identity, config.scoring, control)?;
        let resolution = prepared.finalize_without_model(control)?;
        let maps = Extracted { documents: Vec::new(), receipts: Vec::new(), work: Int8Work::default(),
            mask_visits: 0, verification_used: EntityVerificationWork::default() };
        let output = finish(maps, resolution, self.ner_work, self.masks, config, control)?;
        finish_document(output, Vec::new(), 0, config.max_result_bytes, control)
    }
}
fn check_native(prepared: &PreparedInt8DocumentEntityCorpus, engine: &StrictInt8Engine<'_>) -> Result<(), Int8EntityError> {
    let identity = &prepared.source_identity; let model = engine.artifact_identity();
    if model.model_id != "Nanbeige4.2-3B" || model.revision != identity.source_revision || model.recipe_id != identity.quant_recipe
        || Sha256Digest::from_hex(&model.logical_model_sha256).ok() != Some(identity.logical_model_digest) {
        return Err(Int8EntityError::Identity);
    }
    if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) { return Err(Int8EntityError::Accounting); }
    let positions = engine.kv_cache().capacity_positions(); let c = &prepared.config.entities;
    let cap = c.ner_budget.max_kv_bytes.min(c.scoring.planning.per_head.max_kv_bytes);
    if positions < prepared.maximum_positions || (positions as u64).checked_mul(KV_BYTES_PER_TOKEN as u64)
        .is_none_or(|n| n > cap) { return Err(Int8EntityError::WorkBudget); }
    Ok(())
}

#[derive(Serialize)]
pub struct EntityDocumentChunks { pub document_id: String, pub ner_chunks: usize }
#[derive(Serialize)]
pub struct Int8DocumentEntityRun {
    pub schema_version: u32,
    pub execution: &'static str,
    pub chunk_profile: &'static str,
    pub document_chunks: Vec<EntityDocumentChunks>,
    /// Original-document identities, offsets, contexts and complete-link graph.
    /// Discovery work/receipts aggregate all chunks of each original document.
    pub output: Int8EntityRun,
    pub warnings: [&'static str; 2],
}
fn finish_document<C: DecodeStepControl>(output: Int8EntityRun, geometry: Vec<EntityDocumentChunks>,
    count: usize, cap: usize, control: &mut C) -> Result<Int8DocumentEntityRun, Int8EntityError> {
    let mut total = 0_usize;
    if geometry.len() != output.documents.len() { return Err(Int8EntityError::Accounting); }
    for (entry, document) in geometry.iter().zip(&output.documents) {
        if entry.document_id != document.document_id || entry.ner_chunks == 0 { return Err(Int8EntityError::Accounting); }
        total = total.checked_add(entry.ner_chunks).ok_or(Int8EntityError::Accounting)?;
    }
    if total != count { return Err(Int8EntityError::Accounting); }
    let result = Int8DocumentEntityRun { schema_version: 1, execution: INT8_DOCUMENT_ENTITY_EXECUTION,
        chunk_profile: CHUNK_PROFILE, document_chunks: geometry, output,
        warnings: ["ner_chunk_boundaries_may_split_entities", "ner_recall_and_resolution_quality_not_established"] };
    resolve::check_output(&result, cap)?;
    resolve::checkpoint(control)?;
    Ok(result)
}

// Private fault-injection seam, never public authority to supply native results.
fn collect_chunks<C: DecodeStepControl, F>(inputs: Vec<DocumentInput>, config: &Int8EntityConfig,
    control: &mut C, mut execute: F) -> Result<(Extracted, Vec<EntityDocumentChunks>), Int8EntityError>
where F: FnMut(&Input, &mut C) -> Result<Int8SourceTaskRun, Int8EntityError> {
    let mut documents = reserved(inputs.len())?; let mut receipts = reserved(inputs.len())?;
    let mut geometry = reserved(inputs.len())?;
    let mut verification = config.verification;
    let (mut bytes, mut mentions, mut masks) = (0_usize, 0_usize, 0_u64);
    let mut work = Int8Work::default();
    for input in inputs {
        resolve::checkpoint(control)?;
        if input.chunks.is_empty() { return Err(Int8EntityError::Accounting); }
        let mut document = ResolutionDocument { id: input.document.id, text: input.document.text, mentions: Vec::new() };
        bytes = bytes.checked_add(document.id.len()).and_then(|n| n.checked_add(document.text.len()))
            .filter(|&n| n <= config.graph.max_input_bytes).ok_or(Int8EntityError::WorkBudget)?;
        let (mut cursor, mut scalar, mut proposed, mut document_masks) = (0_usize, 0_usize, 0_usize, 0_u64);
        let mut document_work = Int8Work::default();
        for chunk in &input.chunks {
            resolve::checkpoint(control)?;
            let local = local_input(&document, chunk, cursor, scalar)?;
            let run = execute(&local, control)?;
            let (ner, actual) = check_run(&local, run, config)?;
            proposed = proposed.checked_add(ner.entities.len()).ok_or(Int8EntityError::WorkBudget)?;
            document_work = plus(document_work, actual)?;
            document_masks = document_masks.checked_add(ner.mask_node_visit_charge).ok_or(Int8EntityError::WorkBudget)?;
            let local = grounding::expand(local.document, ner, config, &mut verification, control)?;
            append_mentions(&mut document, local.mentions, chunk.span, &mut bytes, &mut mentions, config.graph, control)?;
            cursor = chunk.span.byte_end; scalar = chunk.span.scalar_end;
        }
        if cursor != document.text.len() { return Err(Int8EntityError::Accounting); }
        work = plus(work, document_work)?;
        masks = masks.checked_add(document_masks).filter(|&n| n <= config.masks.max_visits_per_run)
            .ok_or(Int8EntityError::WorkBudget)?;
        receipts.push(Int8EntityDocumentReceipt { document_id: copy(&document.id)?, proposed_entities: proposed,
            anchored_mentions: document.mentions.len(), model_work: document_work, mask_node_visits: document_masks });
        geometry.push(EntityDocumentChunks { document_id: copy(&document.id)?, ner_chunks: input.chunks.len() });
        documents.push(document);
    }
    resolve::checkpoint(control)?;
    let verification_used = EntityVerificationWork { fields: config.verification.max_fields - verification.max_fields,
        matches: config.verification.max_matches - verification.max_matches,
        scan_steps: config.verification.max_scan_steps - verification.max_scan_steps };
    Ok((Extracted { documents, receipts, work, mask_visits: masks, verification_used }, geometry))
}
fn local_input(document: &ResolutionDocument, chunk: &ChunkWitness, byte_start: usize, scalar_start: usize)
    -> Result<Input, Int8EntityError> {
    let s = chunk.span;
    if s.byte_start != byte_start || s.scalar_start != scalar_start || s.byte_end <= s.byte_start {
        return Err(Int8EntityError::Accounting);
    }
    let text = document.text.get(s.byte_start..s.byte_end).ok_or(Int8EntityError::Accounting)?;
    if s.scalar_start.checked_add(text.chars().count()) != Some(s.scalar_end) { return Err(Int8EntityError::Accounting); }
    Ok(Input { document: EntityDocument { id: copy(&document.id)?, text: copy(text)? },
        witness: chunk.identity, prompt_tokens: chunk.prompt_tokens, work: chunk.work })
}
#[allow(clippy::too_many_arguments)]
fn append_mentions<C: DecodeStepControl>(document: &mut ResolutionDocument, local: Vec<resolve::MentionInput>,
    extent: VerifiedSourceSpan, bytes: &mut usize, count: &mut usize, limits: ResolveLimits, control: &mut C)
    -> Result<(), Int8EntityError> {
    for mut mention in local {
        resolve::checkpoint(control)?;
        let offset = |n: usize, base: usize| n.checked_add(base).ok_or(Int8EntityError::Accounting);
        let span = VerifiedSourceSpan { byte_start: offset(mention.span.byte_start, extent.byte_start)?,
            byte_end: offset(mention.span.byte_end, extent.byte_start)?,
            scalar_start: offset(mention.span.scalar_start, extent.scalar_start)?,
            scalar_end: offset(mention.span.scalar_end, extent.scalar_start)? };
        if span.byte_end > extent.byte_end || span.scalar_end > extent.scalar_end
            || span.byte_start >= span.byte_end || span.scalar_start >= span.scalar_end
            || document.text.get(span.byte_start..span.byte_end) != Some(mention.surface.as_str()) {
            return Err(Int8EntityError::Accounting);
        }
        let next_count = count.checked_add(1).filter(|&n| n <= limits.max_mentions).ok_or(Int8EntityError::WorkBudget)?;
        let next_bytes = bytes.checked_add(mention.entity_type.len()).and_then(|n| n.checked_add(mention.surface.len()))
            .filter(|&n| n <= limits.max_input_bytes).ok_or(Int8EntityError::WorkBudget)?;
        document.mentions.try_reserve(1).map_err(|_| Int8EntityError::Allocation)?;
        mention.span = span; document.mentions.push(mention); *count = next_count; *bytes = next_bytes;
    }
    Ok(())
}

#[cfg(test)] mod tests;
