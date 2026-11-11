//! One physical native owner for an entire durable document invocation.
//! Cohort processors allocate exact windows inside each attempt; this owner
//! retains the maximum declaration through native drain and durable commit.
use super::*;
use crate::native_engine::strict_int4::projection::Int4ProjectionExecutor;

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
        Self::with_projection(weights, context, required, kernel, kv, scratch, cohort, None)
    }

    /// A complete invocation may lend one persistent row executor to the
    /// reusable engine. Native chunk/hierarchy cohorts own a separate
    /// scheduler and cannot share this route implicitly.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn with_projection(weights: &'w Int4ModelWeights, context: usize,
        required: Int4MemoryRequirement, kernel: PackedInt4Kernel,
        kv: Pending, scratch: Pending, cohort: Option<(B, Pending)>,
        projection: Option<&'w mut (dyn Int4ProjectionExecutor + 'w)>) -> Result<Self, HostedError> {
        if cohort.is_some() && projection.is_some() {
            return Err(HostedError::Limits("INT4 document team cannot combine with native cohorts"));
        }
        if let Some((budget, decoder)) = cohort {
            return Ok(Self::Cohorts { budget, _kv: kv, _scratch: scratch, _decoder: decoder });
        }
        let engine = allocate_native(kv, scratch, || {
            let engine = StrictInt4Engine::new(weights, context, Int4MemoryBudget {
                max_kv_bytes: required.kv_bytes, max_rope_bytes: required.rope_bytes,
                max_scratch_payload_bytes: required.scratch_payload_bound,
            }).map_err(native_error)?.with_kernel(kernel);
            match projection {
                Some(team) => engine.with_projection_executor(team).map_err(native_error),
                None => Ok(engine),
            }
        })?;
        Ok(Self::Serial(engine))
    }
}
