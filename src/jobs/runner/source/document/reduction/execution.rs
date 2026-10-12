//! Native document reduction with one durable debit and one whole-output guard.
use super::*;
use crate::validation::grounded_fields::VerifiedSourceSpan;

/// Serializes the native result directly with its own execution and warnings.
/// No wrapper adds bytes outside the admitted native output ceiling.
pub enum Int8ReducedSourceDocumentRun { Summary(Int8CorpusSummaryRun), Keyphrases(Int8CorpusKeyphraseRun) }
impl Serialize for Int8ReducedSourceDocumentRun {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self { Self::Summary(value) => value.serialize(serializer), Self::Keyphrases(value) => value.serialize(serializer) }
    }
}
impl Int8ReducedSourceDocumentRun {
    fn check(&self, policy: SourceDocumentReduction, prepared: &PreparedSourceDocument) -> Result<(), BatchItemFailure> {
        let invalid = || BatchItemFailure::fatal(BatchCode::InvalidExecution);
        let (planned, observed, chunks, span, scans) = match (self, policy) {
            (Self::Summary(r), SourceDocumentReduction::Summary { limits, .. }) => {
                if r.schema_version != 1 || r.execution != policy.execution()
                    || r.summary.reduction_policy != SUMMARY_REDUCTION_POLICY
                    || r.summary.requested_max_bullets != limits.max_bullets || r.summary.bullets.len() > limits.max_bullets {
                    return Err(invalid());
                }
                (JobWork { model: r.planned_model_work, mask_node_visits: r.reserved_mask_node_visits },
                    JobWork { model: r.model_work, mask_node_visits: r.mask_node_visit_charge },
                    r.summary.mapped_chunks, r.source_span, r.verification_scan_steps)
            }
            (Self::Keyphrases(r), SourceDocumentReduction::Keyphrases { limits, .. }) => {
                if r.schema_version != 1 || r.execution != policy.execution()
                    || r.keyphrases.ranking_policy != CORPUS_KEYPHRASE_POLICY
                    || r.requested_max_phrases != limits.max_phrases || r.keyphrases.phrases.len() > limits.max_phrases {
                    return Err(invalid());
                }
                (JobWork { model: r.planned_model_work, mask_node_visits: r.reserved_mask_node_visits },
                    JobWork { model: r.model_work, mask_node_visits: r.mask_node_visit_charge },
                    r.keyphrases.mapped_chunks, r.source_span, r.verification_scan_work)
            }
            _ => return Err(invalid()),
        };
        check_accounting(prepared, planned, observed, chunks, span, scans, policy.scan_work_per_attempt())
    }
}
pub(super) fn check_accounting(prepared: &PreparedSourceDocument, planned: JobWork, observed: JobWork, chunks: usize,
    span: VerifiedSourceSpan, scans: u64, scan_cap: u64) -> Result<(), BatchItemFailure> {
    if planned != prepared.work || !observed.fits(prepared.work) || chunks != prepared.chunk_count()
        || span.byte_start != 0 || span.byte_end != prepared.source.len()
        || span.scalar_start != 0 || span.scalar_end != prepared.source.chars().count() || scans > scan_cap {
        return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
    }
    Ok(())
}

/// Reuses the existing preparation, private owner check and durable work ledger.
/// Public construction requires actual native weights and host admission.
pub struct Int8ReducedSourceDocumentJobProcessor<'p, 'e, 'w, 'v, A: SourceDocumentBatchAdmission> {
    inner: Int8SourceDocumentJobProcessor<'p, 'e, 'w, 'v, A>,
}
impl<'p, 'e, 'w, 'v, A: SourceDocumentBatchAdmission> Int8ReducedSourceDocumentJobProcessor<'p, 'e, 'w, 'v, A> {
    pub fn new(planner: &'p Int8SourceDocumentJobPlanner, engine: &'e mut StrictInt8Engine<'w>,
        vocabulary: &'v ExtractionVocabulary, admission: A) -> Result<Self, BatchFault> {
        let policy = planner.reduction_policy().ok_or(BatchCode::InvalidLimits)?;
        policy.validate(&planner.task, planner.mapping, planner.limits)?;
        check_capacity(planner, engine).map_err(|e| e.fault)?;
        Ok(Self { inner: Int8SourceDocumentJobProcessor { planner, engine, vocabulary, admission, ledger: Ledger::default() } })
    }
}
impl<A: SourceDocumentBatchAdmission> BatchProcessor for Int8ReducedSourceDocumentJobProcessor<'_, '_, '_, '_, A> {
    type Args = SourceDocumentArgs;
    type Prepared = PreparedSourceDocument;
    type Output = GuardedOutput<Int8ReducedSourceDocumentRun, A::Guard>;
    fn prepare(&mut self, document: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> { self.inner.prepare(document) }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>, control: &mut C)
        -> Result<Self::Prepared, BatchItemFailure> { self.inner.prepare_with_control(document, control) }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork { self.inner.planned_work(prepared) }
    fn execute<C: DecodeStepControl>(&mut self, _: Self::Prepared, _: &mut C) -> Result<Self::Output, BatchItemFailure> {
        self.inner.ledger.failed = true; Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        let inner = &mut self.inner;
        inner.ledger.begin(prepared.work, context, inner.planner.limits)?;
        let result = (|| {
            inner.planner.owns(&prepared)?;
            poll(control)?;
            let policy = inner.planner.reduction_policy().ok_or_else(|| BatchItemFailure::fatal(BatchCode::InvalidExecution))?;
            let kv = check_capacity(inner.planner, inner.engine)?;
            let plan = inner.planner.plan(&prepared.source, control)?;
            if !snapshot_matches(&plan, &prepared) { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
            plan.preflight(&prepared.identities, inner.engine).map_err(map_failure)?;
            let (admitted, guard) = inner.admission.admit(SourceDocumentAdmission {
                identities: &prepared.identities, model_work: prepared.work.model,
                mask_node_visits: prepared.work.mask_node_visits, mask_limits: inner.planner.mapping.mask_limits,
                kv_reservation_bytes: kv, max_result_bytes: policy.max_result_bytes(),
            })?;
            poll(control)?;
            // Real reducers recheck every admitted identity and retain every
            // candidate until the root, with one controller throughout.
            let output = policy.execute(plan, &admitted, inner.engine, inner.vocabulary, control)?;
            output.check(policy, &prepared)?;
            check_capacity(inner.planner, inner.engine)?;
            poll(control)?;
            Ok(GuardedOutput::new(output, guard))
        })();
        match result {
            Ok(output) => { inner.ledger.failed = false; Ok(output) }
            Err(mut failure) => { failure.stop = true; Err(failure) }
        }
    }
}
impl<A: SourceDocumentBatchAdmission> DurableBatchProcessor for Int8ReducedSourceDocumentJobProcessor<'_, '_, '_, '_, A> {
    type Recipe = Int8SourceDocumentJobRecipe;
    fn execution_identity(&self) -> &ExecutionIdentity { self.inner.execution_identity() }
    fn job_recipe(&self) -> &Self::Recipe { self.inner.job_recipe() }
    fn validate_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> { self.inner.validate_job_limits(limits) }
    fn durable_work(&self, prepared: &Self::Prepared) -> Result<JobWork, BatchItemFailure> { self.inner.durable_work(prepared) }
    fn max_result_bytes(&self, _: &Self::Prepared) -> u64 {
        self.inner.planner.reduction_policy().map_or(0, SourceDocumentReduction::max_result_bytes)
    }
}
