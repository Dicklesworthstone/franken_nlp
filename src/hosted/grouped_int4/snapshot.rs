//! Local model persistence through the same process-owned physical invocation.
use super::*;
use std::{fs::{self, File, OpenOptions}, path::Path};
use crate::local_io::LocalIoError;
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
    ///
    /// Only an exact-length, non-symlink regular file is opened. The supported
    /// Linux local-IO profile pins directory/file handles before a data-open;
    /// other platforms still require a stable caller-controlled pathname.
    /// Metadata checks do not provide a portable hostile-root transaction,
    /// and synchronous filesystem operations remain cooperatively cancellable.
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
            let mut file = open_snapshot(&path, need.file_bytes, crate::local_io::open_document)?;
            snapshot_poll(control)?;
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
/// Reject known special files before opening them: a FIFO can block in open(),
/// before its descriptor metadata or the codec gets a cancellation checkpoint.
/// The opener is the existing local-IO authority, not a second platform ABI.
/// Recheck its held descriptor, since preflight metadata alone is not authority
/// for either the opened file's type or its eventual allocation geometry.
fn open_snapshot(
    path: &Path, expected_bytes: u64,
    open: impl FnOnce(&Path) -> Result<File, LocalIoError>,
) -> Result<File, HostedError> {
    let metadata = fs::symlink_metadata(path).map_err(|_| snapshot_error(Int4SnapshotError::Io))?;
    file_geometry(!metadata.file_type().is_symlink() && metadata.is_file(), metadata.len(), expected_bytes)?;
    let file = open(path).map_err(|error| snapshot_error(match error {
        LocalIoError::UnsafeFile | LocalIoError::InvalidPath => Int4SnapshotError::Format,
        _ => Int4SnapshotError::Io,
    }))?;
    let metadata = file.metadata().map_err(|_| snapshot_error(Int4SnapshotError::Io))?;
    file_geometry(metadata.is_file(), metadata.len(), expected_bytes)?;
    Ok(file)
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
    use std::{io::Write, sync::atomic::{AtomicU64, Ordering}};
    static NEXT_FILE: AtomicU64 = AtomicU64::new(0);
    fn fixture_path() -> PathBuf {
        let sequence = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("fnlp-snapshot-open-{}-{sequence}", std::process::id()));
        fs::create_dir(&path).unwrap(); path
    }
    fn fixture_file(path: &Path, bytes: &[u8]) -> File {
        let mut file = OpenOptions::new().read(true).write(true).create_new(true).open(path).unwrap();
        file.write_all(bytes).unwrap(); file
    }
    fn format_error(error: HostedError) {
        assert!(matches!(error, HostedError::Int4(Int4HostError::Snapshot(Int4SnapshotError::Format))));
    }
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
    fn wrong_snapshot_geometry_is_rejected_before_the_data_opener() {
        let root = fixture_path(); let short = root.join("short");
        let _file = fixture_file(&short, b"short");
        for path in [&root, &short] {
            format_error(open_snapshot(path, 8, |_| panic!("invalid geometry must not open data")).unwrap_err());
        }
        format_error(open_snapshot(&short, 4, |_| panic!("overlong snapshot must not open data")).unwrap_err());
        #[cfg(unix)]
        format_error(open_snapshot(Path::new("/dev/null"), 0, |_| panic!("device must not open data")).unwrap_err());
        assert!(matches!(open_snapshot(&root.join("absent"), 8, |_| panic!("absent file must not open data")),
            Err(HostedError::Int4(Int4HostError::Snapshot(Int4SnapshotError::Io)))));
    }
    #[test]
    fn snapshot_geometry_is_rechecked_on_the_actual_opened_descriptor() {
        let root = fixture_path(); let path = root.join("snapshot");
        let _file = fixture_file(&path, b"expected");
        let short = fixture_file(&root.join("short"), b"short");
        format_error(open_snapshot(&path, 8, |_| Ok(short)).unwrap_err());
        #[cfg(unix)]
        format_error(open_snapshot(&path, 8, |_| File::open(&root).map_err(|_| LocalIoError::Io)).unwrap_err());
        assert_eq!(open_snapshot(&path, 8, crate::local_io::open_document).unwrap().metadata().unwrap().len(), 8);
    }
    #[cfg(unix)]
    #[test]
    fn symlink_snapshot_is_refused_without_following_its_target() {
        let root = fixture_path(); let target = root.join("target"); let link = root.join("snapshot");
        let _file = fixture_file(&target, b"expected");
        std::os::unix::fs::symlink(&target, &link).unwrap();
        format_error(open_snapshot(&link, 8, |_| panic!("symlink must not reach data opener")).unwrap_err());
    }
    #[test]
    fn snapshot_errors_retain_typed_causes_without_paths_in_default_diagnostics() {
        let error = snapshot_error(Int4SnapshotError::Integrity);
        assert!(matches!(error, HostedError::Int4(Int4HostError::Snapshot(Int4SnapshotError::Integrity))));
        assert_eq!(error.to_string(), "hosted source-candidate INT4 loading or execution failed");
        assert!(error.source().is_some());
    }
}
