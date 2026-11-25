//! One bounded tokenizer owner for live streams and optional durable jobs.
//! This module has no runtime, storage, model or platform dependency.
use std::sync::{Arc, Mutex};
use crate::{
    batch::{BatchCode, BatchFault, BatchItemFailure},
    native_engine::decode::DecodeStepControl,
    tokenizer::untrusted::{SafeBpeLimits, SafeBpeWork, SafeBpeWorkLimits, UntrustedEncoding},
};

/// One invocation owns every preparation attempt, including model-free
/// previews, rejected records and the consuming native handoff. Only counters
/// are retained, never source text. The same private owner also seals prepared
/// work so another invocation cannot execute it against a fresh allowance.
pub(crate) struct EncodingWorkScope {
    pub(crate) policy: UntrustedEncoding,
    pub(crate) limits: Option<SafeBpeWorkLimits>,
    used: Mutex<SafeBpeWork>,
}
impl EncodingWorkScope {
    pub(crate) fn new(policy: &UntrustedEncoding, limits: Option<SafeBpeWorkLimits>)
        -> Result<Arc<Self>, BatchFault> {
        Self::workspace(policy, limits)?;
        Ok(Arc::new(Self { policy: policy.clone(), limits, used: Mutex::new(SafeBpeWork::default()) }))
    }
    pub(crate) fn workspace(policy: &UntrustedEncoding, limits: Option<SafeBpeWorkLimits>) -> Result<u64, BatchFault> {
        match (policy, limits) {
            (UntrustedEncoding::ByteFallback, None) => Ok(0),
            (UntrustedEncoding::SafeBpeV1(call), Some(total)) => {
                total.validate().map_err(|_| BatchCode::InvalidLimits)?;
                call.requirements().map_err(|_| BatchCode::InvalidLimits)?.workspace_bytes
                    .checked_add(1024).ok_or_else(|| BatchCode::InvalidLimits.into())
            }
            _ => Err(BatchCode::Admission.into()),
        }
    }
    pub(crate) fn workspace_bytes(&self) -> Result<u64, BatchFault> {
        Self::workspace(&self.policy, self.limits)
    }
    #[cfg(feature = "asupersync-runtime")]
    pub(crate) fn policy(&self) -> &UntrustedEncoding { &self.policy }
    #[cfg(feature = "asupersync-runtime")]
    pub(crate) fn limits(&self) -> Option<SafeBpeWorkLimits> { self.limits }
    #[cfg(any(test, all(feature = "metadata-store", target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64"))))]
    pub(crate) fn work(&self) -> Result<SafeBpeWork, BatchFault> {
        self.used.try_lock().map(|used| *used).map_err(|_| BatchCode::InvalidExecution.into())
    }
    #[cfg(all(feature = "metadata-store", target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub(crate) fn verify(self: &Arc<Self>, owner: &Arc<Self>) -> Result<(), BatchItemFailure> {
        if !Arc::ptr_eq(self, owner) { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
        Ok(())
    }
    pub(crate) fn prepare<T, C: DecodeStepControl>(&self, control: &mut C,
        prepare: impl FnOnce(Option<SafeBpeLimits>, &mut SafeBpeWork, &mut C) -> Result<T, BatchItemFailure>)
        -> Result<T, BatchItemFailure> {
        if self.policy == UntrustedEncoding::ByteFallback {
            return prepare(None, &mut SafeBpeWork::default(), control);
        }
        if let Some(cause) = control.prefill_checkpoint(0) {
            return Err(BatchItemFailure::fatal(BatchFault::cancelled(cause)));
        }
        // Refuse overlapping/reentrant use instead of deadlocking or granting
        // another ledger. A panic poisons this owner and cannot renew work.
        let mut used = self.used.try_lock().map_err(|_| BatchItemFailure::fatal(BatchCode::InvalidExecution))?;
        let (UntrustedEncoding::SafeBpeV1(call), Some(total)) = (&self.policy, self.limits) else {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        };
        let effective = total.remaining_call(*used, *call)
            .map_err(|_| BatchItemFailure::fatal(BatchCode::WorkLimit))?;
        let mut work = SafeBpeWork::default();
        let result = prepare(Some(effective), &mut work, control);
        // Debit before inspecting success, failure or typed cancellation.
        *used = used.checked_add(work).ok_or_else(|| BatchItemFailure::fatal(BatchCode::WorkLimit))?;
        result
    }
}

