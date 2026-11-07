//! Durable receipts delegate to the shared storage-neutral native planner.
use super::*;
pub(super) use native_document::planning::{Expected, NativePlan};

impl NativePlan<'_> {
    pub(super) fn compare<C: DecodeStepControl>(&self, prepared: &PreparedInt4SourceDocument,
        control: &mut C) -> Result<(), BatchItemFailure> {
        self.compare_receipt(prepared.expected,
            DocumentWork { model: prepared.work.int4_model_work(), mask_node_visits: prepared.work.mask_node_visits },
            &prepared.identities, control)
    }
}
impl Int4SourceDocumentJobPlanner {
    pub(super) fn plan<'s, C: DecodeStepControl>(&self, source: &'s str, control: &mut C)
        -> Result<NativePlan<'s>, BatchItemFailure> {
        self.configuration().plan(source, control)
    }
    /// Prepare the same complete native discovery used by live document corpora.
    /// The private factory seal and durable JobWork projection remain unchanged.
    pub fn prepare_with_control<C: DecodeStepControl>(&self, document: BatchDocument<SourceDocumentArgs>,
        control: &mut C) -> Result<PreparedInt4SourceDocument, BatchItemFailure> {
        let plan = self.plan(&document.text, control)?;
        let expected = plan.expected();
        let identities = plan.identities(control)?;
        if identities.len() != expected.native_chunks() { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
        let work = expected_job_work(expected)?;
        drop(plan);
        poll(control)?;
        Ok(PreparedInt4SourceDocument { source: document.text, identities, expected, work, owner: Arc::clone(&self.owner) })
    }
}
pub(super) fn expected_job_work(expected: Expected) -> Result<JobWork, BatchItemFailure> {
    let work = expected.work()?;
    Ok(JobWork::from_int4(work.model, work.mask_node_visits))
}
