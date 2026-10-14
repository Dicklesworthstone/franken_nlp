//! Local model persistence through the same process-owned physical invocation.
use super::*;
use std::fs::{File, OpenOptions};
use crate::native_engine::strict_int4::weights::snapshot::{Int4SnapshotBudget, Int4SnapshotPin, Int4SnapshotRequirements};

impl Int4LoadRequirements {
    /// Pure aggregate calculation before any snapshot IO. The caller retains
    /// its external pin; lengths supplied by the file cannot enlarge the plan.
    pub fn for_snapshot(pin: Int4SnapshotPin, budget: Int4SnapshotBudget,
        allocator_reserve_bytes: u64, run: RunLimits) -> Result<Self, HostedError> {
        run.validate()?;
        if allocator_reserve_bytes == 0 { return Err(HostedError::Limits("snapshot allocator reserve required")); }
        let need = Int4SnapshotRequirements::for_group_width(pin.group_width).map_err(snapshot_error)?;
        need.check(budget, true).map_err(snapshot_error)?;
        Ok(Self { resident_bytes: sum(&[need.weight_payload_bytes, allocator_reserve_bytes])?,
            staging_bytes: need.scratch_bytes })
    }
}
impl NlpEngine {
    /// Restore an explicit externally pinned local snapshot without touching
    /// original source files or rerunning quantization. The full model remains
    /// a candidate. The pin must come from a trusted conversion/receipt, not
    /// merely from the same untrusted file. No publisher authenticity is implied.
    #[allow(clippy::too_many_arguments)]
    pub fn load_int4_snapshot(&self, path: PathBuf, pin: Int4SnapshotPin,
        budget: Int4SnapshotBudget, allocator_reserve_bytes: u64, run: RunLimits,
        cancellation: CancellationToken) -> Result<ResidentInt4, HostedError> {
        dispatch::preflight(self, run)?;
        let charges = Int4LoadRequirements::for_snapshot(pin, budget, allocator_reserve_bytes, run)?;
        let need = Int4SnapshotRequirements::for_group_width(pin.group_width).map_err(snapshot_error)?;
        if path.as_os_str().is_empty() { return Err(HostedError::Limits("explicit snapshot path required")); }
        let lease = self.resources().acquire_lease();
        let resident = Pending::reserve(&lease, MemoryClass::Weights, charges.resident_bytes)?;
        let staging = Pending::reserve(&lease, MemoryClass::Staging, charges.staging_bytes)?;
        dispatch::run(self, run, cancellation, move |control| {
            // All captures and charges live until physical completion, including
            // cancellation while queued or inside a blocking file operation.
            let staging = staging.commit()?;
            snapshot_poll(control)?;
            let mut file = File::open(&path).map_err(|_| snapshot_error(Int4SnapshotError::Io))?;
            let metadata = file.metadata().map_err(|_| snapshot_error(Int4SnapshotError::Io))?;
            file_geometry(metadata.is_file(), metadata.len(), need.file_bytes)?;
            let loaded = allocate(resident, || Int4ModelWeights::read_snapshot(&mut file, pin, budget, control)
                .map_err(snapshot_error))?;
            drop(file); drop(staging);
            Ok(ResidentInt4 { inner: Arc::new(ResidentModel { loaded, lease }) })
        })
    }

    /// Save one charged model to a NEW explicit file. create_new refuses every
    /// existing destination (including symlinks); nothing is overwritten,
    /// removed, renamed into an active catalog, or silently retried. The caller
    /// must admit enough disk space. The file and model charge stay alive until
    /// writes, flush and sync_all physically complete on the existing host.
    ///
    /// On any failure/cancellation the path may remain partial OR complete.
    /// It is deliberately preserved, never reported as a completed snapshot.
    /// This is not atomic publication or a release/durability certificate.
    /// Reload always requires the independent pin and a complete digest check.
    pub fn save_int4_snapshot(&self, model: &ResidentInt4, path: PathBuf,
        budget: Int4SnapshotBudget, run: RunLimits, cancellation: CancellationToken)
        -> Result<Int4SnapshotPin, HostedError> {
        dispatch::preflight(self, run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        let need = Int4SnapshotRequirements::for_group_width(model.model_identity().group_width()).map_err(snapshot_error)?;
        need.check(budget, false).map_err(snapshot_error)?;
        if path.as_os_str().is_empty() { return Err(HostedError::Limits("explicit new snapshot path required")); }
        let lease = self.resources().acquire_lease();
        // Weights/allocator headroom remain covered by the shared resident
        // charge. Only codec staging is added; no second model is allocated.
        let staging = Pending::reserve(&lease, MemoryClass::Staging, need.scratch_bytes)?;
        let model = model.clone();
        dispatch::run(self, run, cancellation, move |control| {
            let staging = staging.commit()?;
            snapshot_poll(control)?;
            let mut file = OpenOptions::new().write(true).create_new(true).open(&path)
                .map_err(|_| snapshot_error(Int4SnapshotError::Io))?;
            let pin = model.inner.loaded.value.write_snapshot(&mut file, budget, control).map_err(snapshot_error)?;
            snapshot_poll(control)?;
            file.sync_all().map_err(|_| snapshot_error(Int4SnapshotError::Io))?;
            let metadata = file.metadata().map_err(|_| snapshot_error(Int4SnapshotError::Io))?;
            file_geometry(metadata.is_file(), metadata.len(), need.file_bytes)?;
            snapshot_poll(control)?;
            drop(file); drop(staging); drop(model); drop(lease);
            Ok(pin)
        })
    }
}
fn snapshot_error(error: Int4SnapshotError) -> HostedError { HostedError::Int4(Int4HostError::Snapshot(error)) }
fn snapshot_poll(control: &mut RunControl) -> Result<(), HostedError> {
    match control.prefill_checkpoint(0) {
        Some(cause) => Err(snapshot_error(Int4SnapshotError::Cancelled(cause))), None => Ok(()),
    }
}
fn file_geometry(regular: bool, actual: u64, expected: u64) -> Result<(), HostedError> {
    if regular && actual == expected { Ok(()) } else { Err(snapshot_error(Int4SnapshotError::Format)) }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn run() -> RunLimits {
        RunLimits { max_elapsed: Duration::from_secs(1), max_checkpoints: 100, cleanup_reserve_bytes: 65536 }
    }
    #[test]
    fn all_snapshot_group_partitions_charge_the_complete_model_and_one_codec_buffer() {
        for width in 1..=MAX_MODEL_K {
            let pin = Int4SnapshotPin { group_width: width, model_content_sha256: [1; 32] };
            let need = Int4SnapshotRequirements::for_group_width(width).unwrap();
            let budget = Int4SnapshotBudget { max_weight_bytes: need.weight_payload_bytes,
                max_scratch_bytes: need.scratch_bytes, max_io_bytes: need.read_io_bytes().unwrap() };
            let required = Int4LoadRequirements::for_snapshot(pin, budget, 4096, run()).unwrap();
            assert_eq!(required.resident_bytes, need.weight_payload_bytes + 4096);
            assert_eq!(required.staging_bytes, 65536);
            for axis in 0..3 {
                let mut bad = budget;
                match axis { 0 => bad.max_weight_bytes -= 1, 1 => bad.max_scratch_bytes -= 1, _ => bad.max_io_bytes -= 1 }
                assert!(Int4LoadRequirements::for_snapshot(pin, bad, 4096, run()).is_err());
            }
            assert!(Int4LoadRequirements::for_snapshot(pin, budget, 0, run()).is_err());
            assert!(Int4LoadRequirements::for_snapshot(pin, budget, u64::MAX, run()).is_err());
        }
    }
    #[test]
    fn only_exact_regular_files_reach_the_model_reader() {
        file_geometry(true, 100, 100).unwrap();
        for (regular, length) in [(false, 100), (true, 99), (true, 101), (true, u64::MAX)] {
            assert!(file_geometry(regular, length, 100).is_err());
        }
    }
    #[test]
    fn snapshot_errors_retain_typed_causes_without_paths_in_default_diagnostics() {
        let error = snapshot_error(Int4SnapshotError::Integrity);
        assert!(matches!(error, HostedError::Int4(Int4HostError::Snapshot(Int4SnapshotError::Integrity))));
        assert_eq!(error.to_string(), "hosted source-candidate INT4 loading or execution failed");
        assert!(error.source().is_some());
    }
}
