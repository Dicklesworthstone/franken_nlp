//! One physical native owner for an entire durable document invocation.
//! Cohort processors allocate exact windows inside each attempt; this owner
//! retains the maximum declaration through native drain and durable commit.
use super::*;

pub(super) enum DocumentNative<'w, B> {
    Serial(ChargedNative<StrictInt4Engine<'w>>),
    Cohorts {
        budget: B,
        _kv: Pending,
        _scratch: Pending,
        _decoder: Pending,
    },
}
impl<'w, B> DocumentNative<'w, B> {
    /// Keep the existing reusable scalar allocation when selected. A cohort
    /// invocation reserves its complete peak without constructing an unused
    /// scalar engine alongside the per-window engines owned by the processor.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(weights: &'w Int4ModelWeights, context: usize,
        required: Int4MemoryRequirement, kernel: PackedInt4Kernel,
        kv: Pending, scratch: Pending, cohort: Option<(B, Pending)>) -> Result<Self, HostedError> {
        if let Some((budget, decoder)) = cohort {
            return Ok(Self::Cohorts { budget, _kv: kv, _scratch: scratch, _decoder: decoder });
        }
        let engine = allocate_native(kv, scratch, || {
            StrictInt4Engine::new(weights, context, Int4MemoryBudget {
                max_kv_bytes: required.kv_bytes, max_rope_bytes: required.rope_bytes,
                max_scratch_payload_bytes: required.scratch_payload_bound,
            }).map(|engine| engine.with_kernel(kernel)).map_err(native_error)
        })?;
        Ok(Self::Serial(engine))
    }
}
