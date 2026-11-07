//! Owned-job files under the existing descriptor-rooted local IO profile.
//! A kernel-held advisory lock survives neither process exit nor File drop;
//! no stale lock-file deletion or process-ID guessing is used on recovery.
use super::*;
use crate::jobs::JobError;
use std::fs::TryLockError;

const DATABASE: &str = "journal.fsqlite";
const SPOOL: &str = "results.spool";
const INPUTS: &str = "inputs.spool";
const LOCK: &str = "job.lock";
const MATERIALIZED: &str = "materialized.ndjson";

// OwnedJob declares its database connection BEFORE this aggregate, so the
// connection/sidecars close before the directory handle and exclusive lock.
pub(crate) struct JobFiles {
    pub spool: File,
    database: File,
    parent: File,
    _lock: File,
}
impl JobFiles {
    pub fn open(root: &Path, create: bool, max_database: u64, max_spool: u64) -> Result<Self, JobError> {
        let (parent, _) = super::parent(&root.join(DATABASE), true).map_err(local_error)?;
        let lock = open_owned(&parent, LOCK, create, true, false, 4096)?;
        match lock.try_lock() {
            Ok(()) => {}, Err(TryLockError::WouldBlock) => return Err(JobError::Busy),
            Err(TryLockError::Error(_)) => return Err(JobError::Platform),
        }
        // Never let fsqlite follow a pre-existing sidecar symlink. Same-UID,
        // root and malicious filesystem races remain outside this profile.
        check_sidecars(&parent, max_database, create)?;
        let database = open_owned(&parent, DATABASE, create, true, false, max_database)?;
        let spool = open_owned(&parent, SPOOL, create, true, false, max_spool)?;
        let files = Self { spool, database, parent, _lock: lock };
        if create {
            files.database.sync_all().map_err(|_| JobError::Io)?;
            files.spool.sync_all().map_err(|_| JobError::Io)?;
            files._lock.sync_all().map_err(|_| JobError::Io)?;
            files.sync_directory()?;
        }
        Ok(files)
    }
    pub fn database_path(&self) -> PathBuf { fd_path(&self.parent).join(DATABASE) }
    pub fn check_database(&self, max: u64) -> Result<(), JobError> {
        check_parent(&self.parent)?;
        let held = self.database.metadata().map_err(|_| JobError::Io)?;
        let named = fs::symlink_metadata(self.database_path()).map_err(|_| JobError::Io)?;
        if !named.is_file() || identity(&held) != identity(&named) || held.len() > max {
            return Err(JobError::UnsafeStorage);
        }
        check_file(&held, &self.parent, false, max)?;
        check_sidecars(&self.parent, max, false)
    }
    pub fn sync_database(&self) -> Result<(), JobError> {
        self.database.sync_all().map_err(|_| JobError::Io)?; self.sync_directory()
    }
    pub fn sync_directory(&self) -> Result<(), JobError> {
        check_parent(&self.parent)?; self.parent.sync_all().map_err(|_| JobError::Io)
    }
    /// Explicit input-retention opt-in, under this job's existing lock. The
    /// caller must price and bound the complete framed snapshot BEFORE opening
    /// this file. An orphan from failed initialization is never overwritten.
    /// No receipt may be journaled until `sync_inputs` succeeds.
    pub fn create_inputs(&self, max: u64) -> Result<File, JobError> {
        input_limit(max)?;
        open_owned(&self.parent, INPUTS, true, true, false, max)
    }
    /// Seal the written file before its journal receipt is published. Returns
    /// its checked physical size for comparison with the planned frame total.
    /// Content authentication belongs to the input-framing/journal layer.
    pub fn sync_inputs(&self, file: &File, max: u64) -> Result<u64, JobError> {
        input_limit(max)?;
        let length = check_owned_name(&self.parent, INPUTS, file, max)?;
        file.sync_all().map_err(|_| JobError::Io)?;
        self.sync_directory()?;
        if check_owned_name(&self.parent, INPUTS, file, max)? != length {
            return Err(JobError::UnsafeStorage);
        }
        Ok(length)
    }
    /// Read-only access while this owner retains the exclusive job lock. File
    /// presence alone never authorizes replay: the caller must authenticate a
    /// journal receipt and the exact frames before using retained input bytes.
    pub fn retained_inputs(&self, max: u64) -> Result<Option<File>, JobError> {
        input_limit(max)?;
        check_parent(&self.parent)?;
        match fs::symlink_metadata(fd_path(&self.parent).join(INPUTS)) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(_) => Err(JobError::Io),
            Ok(_) => {
                let file = open_owned(&self.parent, INPUTS, false, false, false, max)?;
                check_owned_name(&self.parent, INPUTS, &file, max)?;
                Ok(Some(file))
            }
        }
    }
    pub fn destination(&self) -> Result<Destination, JobError> {
        check_parent(&self.parent)?;
        // Do not accumulate a new full-size stage on each crashed retry.
        if self.has_stages()? { return Err(JobError::UncommittedTail); }
        Ok(Destination { parent: self.parent.try_clone().map_err(|_| JobError::Io)?, name: OsString::from(MATERIALIZED) })
    }
    pub fn has_stages(&self) -> Result<bool, JobError> {
        scan_stages(&self.parent, |_| Ok(()))
    }
    /// Only called after authenticated prefix recovery AND explicit discard
    /// policy. The job root reserves the existing local-output stage namespace.
    /// A named inode is pinned/rechecked before unlink; published output is
    /// never removed, including an alias left by interrupted hard-link publish.
    pub fn discard_stages(&self, max: u64) -> Result<(), JobError> {
        scan_stages(&self.parent, |name| {
            let held = open_owned(&self.parent, name, false, false, true, max)?;
            let meta = held.metadata().map_err(|_| JobError::Io)?;
            if meta.nlink() != 1 {
                let published = fs::symlink_metadata(fd_path(&self.parent).join(MATERIALIZED)).map_err(|_| JobError::UnsafeStorage)?;
                if !published.is_file() || identity(&meta) != identity(&published) { return Err(JobError::UnsafeStorage); }
            }
            let path = fd_path(&self.parent).join(name);
            let named = fs::symlink_metadata(&path).map_err(|_| JobError::Io)?;
            if !named.is_file() || identity(&meta) != identity(&named) { return Err(JobError::UnsafeStorage); }
            fs::remove_file(path).map_err(|_| JobError::Io)?;
            Ok(())
        })?;
        self.sync_directory()
    }
    pub fn materialized(&self, max: u64) -> Result<Option<File>, JobError> {
        check_parent(&self.parent)?;
        match fs::symlink_metadata(fd_path(&self.parent).join(MATERIALIZED)) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(_) => Err(JobError::Io),
            // Read-only recovery permits an extra hard link left by a crash
            // between publication and staging unlink. It never mutates either
            // name and must independently compare every byte to journal frames.
            Ok(_) => open_owned(&self.parent, MATERIALIZED, false, false, true, max).map(Some),
        }
    }
}
fn input_limit(max: u64) -> Result<(), JobError> {
    if max == 0 || max > i64::MAX as u64 { return Err(JobError::InvalidLimits); }
    Ok(())
}
fn check_owned_name(parent: &File, name: &str, file: &File, max: u64) -> Result<u64, JobError> {
    check_parent(parent)?;
    let held = file.metadata().map_err(|_| JobError::Io)?;
    let named = fs::symlink_metadata(fd_path(parent).join(name)).map_err(|_| JobError::UnsafeStorage)?;
    check_file(&held, parent, false, max)?;
    check_file(&named, parent, false, max)?;
    if identity(&held) != identity(&named) || held.len() != named.len() {
        return Err(JobError::UnsafeStorage);
    }
    Ok(held.len())
}
fn check_parent(parent: &File) -> Result<(), JobError> {
    let meta = parent.metadata().map_err(|_| JobError::Io)?;
    if !meta.is_dir() || meta.uid() != process_uid().map_err(local_error)? || meta.mode() & 0o7777 != 0o700 {
        return Err(JobError::UnsafeStorage);
    }
    Ok(())
}
fn check_file(meta: &Metadata, parent: &File, allow_links: bool, max: u64) -> Result<(), JobError> {
    let owner = parent.metadata().map_err(|_| JobError::Io)?.uid();
    if !meta.is_file() || meta.uid() != owner || meta.mode() & 0o7777 != 0o600
        || meta.nlink() == 0 || (!allow_links && meta.nlink() != 1) || meta.len() > max {
        return Err(JobError::UnsafeStorage);
    }
    Ok(())
}
fn open_owned(parent: &File, name: &str, create: bool, write: bool, allow_links: bool, max: u64) -> Result<File, JobError> {
    check_parent(parent)?;
    let path = fd_path(parent).join(name);
    if create {
        let file = OpenOptions::new().read(true).write(true).create_new(true).mode(0o600)
            .custom_flags(NOFOLLOW | NONBLOCK).open(&path).map_err(|e| {
                if e.kind() == std::io::ErrorKind::AlreadyExists { JobError::AlreadyExists } else { JobError::Io }
            })?;
        file.set_permissions(fs::Permissions::from_mode(0o600)).map_err(|_| JobError::Io)?;
        check_file(&file.metadata().map_err(|_| JobError::Io)?, parent, false, max)?;
        return Ok(file);
    }
    let pin = OpenOptions::new().read(true).custom_flags(PATH_ONLY | NOFOLLOW).open(&path).map_err(|_| JobError::UnsafeStorage)?;
    let before = pin.metadata().map_err(|_| JobError::Io)?;
    check_file(&before, parent, allow_links, max)?;
    let file = OpenOptions::new().read(true).write(write).custom_flags(NONBLOCK).open(fd_path(&pin)).map_err(|_| JobError::Io)?;
    let after = file.metadata().map_err(|_| JobError::Io)?;
    check_file(&after, parent, allow_links, max)?;
    if identity(&before) != identity(&after) || before.len() != after.len() { return Err(JobError::UnsafeStorage); }
    Ok(file)
}
fn check_sidecars(parent: &File, max: u64, creating: bool) -> Result<(), JobError> {
    for name in ["journal.fsqlite-journal", "journal.fsqlite-wal", "journal.fsqlite-shm"] {
        match fs::symlink_metadata(fd_path(parent).join(name)) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {},
            Err(_) => return Err(JobError::Io),
            Ok(meta) => {
                if creating { return Err(JobError::AlreadyExists); }
                // This job version uses DELETE/FULL, not a silent WAL-mode
                // downgrade. A foreign journal mode is a recovery mismatch.
                if name != "journal.fsqlite-journal" { return Err(JobError::UnsafeStorage); }
                let bound = max.checked_mul(2).and_then(|n| n.checked_add(65_536)).ok_or(JobError::Limit)?;
                check_file(&meta, parent, false, bound)?;
            }
        }
    }
    Ok(())
}
pub(crate) fn local_error(error: LocalIoError) -> JobError {
    match error {
        LocalIoError::UnsupportedProfile => JobError::Platform,
        LocalIoError::AlreadyExists => JobError::AlreadyExists,
        LocalIoError::PublicationUncertain => JobError::PublicationUncertain,
        LocalIoError::Io => JobError::Io,
        _ => JobError::UnsafeStorage,
    }
}

fn scan_stages(parent: &File, mut visit: impl FnMut(&str) -> Result<(), JobError>) -> Result<bool, JobError> {
    check_parent(parent)?;
    let mut found = false;
    for (index, entry) in fs::read_dir(fd_path(parent)).map_err(|_| JobError::Io)?.enumerate() {
        if index >= 4096 { return Err(JobError::Limit); }
        let name = entry.map_err(|_| JobError::Io)?.file_name();
        let Some(name) = name.to_str() else { continue; };
        let Some(body) = name.strip_prefix(".fnlp-redact-").and_then(|n| n.strip_suffix(".part")) else { continue; };
        let Some((pid, sequence)) = body.split_once('-') else { return Err(JobError::UnsafeStorage); };
        if pid.is_empty() || sequence.is_empty() || !pid.bytes().all(|b| b.is_ascii_digit())
            || !sequence.bytes().all(|b| b.is_ascii_digit()) || pid.parse::<u32>().is_err() || sequence.parse::<u64>().is_err() {
            return Err(JobError::UnsafeStorage);
        }
        found = true; visit(name)?;
    }
    Ok(found)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{io::{Seek, SeekFrom}, os::unix::fs::{DirBuilderExt, symlink}};

    static NEXT: AtomicU64 = AtomicU64::new(0);
    fn root() -> PathBuf {
        let path = std::env::temp_dir().join(format!("fnlp-job-inputs-{}-{}",
            std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        fs::DirBuilder::new().mode(0o700).create(&path).unwrap();
        path
    }
    fn create(root: &Path) -> JobFiles {
        JobFiles::open(root, true, 1024 * 1024, 4096).unwrap()
    }
    fn write_inputs(files: &JobFiles, bytes: &[u8]) -> File {
        let mut input = files.create_inputs(4096).unwrap();
        input.write_all(bytes).unwrap();
        assert_eq!(files.sync_inputs(&input, 4096).unwrap(), bytes.len() as u64);
        input
    }

    #[test]
    fn inputs_are_explicit_private_and_reopened_read_only_under_the_same_lock() {
        let dir = root();
        let files = create(&dir);
        assert!(!dir.join(INPUTS).exists());
        assert!(files.retained_inputs(4096).unwrap().is_none());
        drop(write_inputs(&files, b"exact private snapshot"));
        assert_eq!(fs::metadata(dir.join(INPUTS)).unwrap().mode() & 0o7777, 0o600);
        let mut retained = files.retained_inputs(4096).unwrap().unwrap();
        let mut bytes = Vec::new();
        retained.read_to_end(&mut bytes).unwrap();
        assert_eq!(bytes, b"exact private snapshot");
        assert!(retained.write_all(b"mutate").is_err());
        assert!(matches!(JobFiles::open(&dir, false, 1024 * 1024, 4096), Err(JobError::Busy)));
        drop(retained); drop(files);
        let reopened = JobFiles::open(&dir, false, 1024 * 1024, 4096).unwrap();
        assert_eq!(reopened.retained_inputs(4096).unwrap().unwrap().metadata().unwrap().len(), 22);
    }

    #[test]
    fn a_prior_input_file_is_never_replaced_even_if_incomplete() {
        let dir = root(); let files = create(&dir);
        let mut unfinished = files.create_inputs(4096).unwrap();
        unfinished.write_all(b"partial").unwrap();
        drop(unfinished);
        assert!(matches!(files.create_inputs(4096), Err(JobError::AlreadyExists)));
        assert_eq!(fs::read(dir.join(INPUTS)).unwrap(), b"partial");
    }

    #[test]
    fn input_caps_are_checked_before_reading_or_acknowledging_a_file() {
        let dir = root(); let files = create(&dir);
        assert!(matches!(files.create_inputs(0), Err(JobError::InvalidLimits)));
        assert!(matches!(files.create_inputs(u64::MAX), Err(JobError::InvalidLimits)));
        assert!(!dir.join(INPUTS).exists());
        let mut input = files.create_inputs(4).unwrap();
        input.write_all(b"oversize").unwrap();
        assert_eq!(files.sync_inputs(&input, 4), Err(JobError::UnsafeStorage));
        assert!(matches!(files.retained_inputs(4), Err(JobError::UnsafeStorage)));
        assert_eq!(fs::read(dir.join(INPUTS)).unwrap(), b"oversize");
    }

    #[test]
    fn unsafe_input_modes_links_and_types_are_refused() {
        let dir = root(); let files = create(&dir);
        let input = write_inputs(&files, b"private");
        input.set_permissions(fs::Permissions::from_mode(0o640)).unwrap();
        assert!(matches!(files.retained_inputs(4096), Err(JobError::UnsafeStorage)));
        assert_eq!(files.sync_inputs(&input, 4096), Err(JobError::UnsafeStorage));

        let dir = root(); let files = create(&dir);
        drop(write_inputs(&files, b"private"));
        fs::hard_link(dir.join(INPUTS), dir.join("alias")).unwrap();
        assert!(matches!(files.retained_inputs(4096), Err(JobError::UnsafeStorage)));

        let dir = root(); let files = create(&dir);
        symlink(SPOOL, dir.join(INPUTS)).unwrap();
        assert!(matches!(files.retained_inputs(4096), Err(JobError::UnsafeStorage)));
        assert!(matches!(files.create_inputs(4096), Err(JobError::AlreadyExists)));

        let dir = root(); let files = create(&dir);
        fs::DirBuilder::new().mode(0o700).create(dir.join(INPUTS)).unwrap();
        assert!(matches!(files.retained_inputs(4096), Err(JobError::UnsafeStorage)));
    }

    #[test]
    fn input_sync_refuses_a_replaced_name_or_a_different_held_inode() {
        let dir = root(); let files = create(&dir);
        let input = write_inputs(&files, b"original");
        fs::rename(dir.join(INPUTS), dir.join("original")).unwrap();
        let replacement = write_inputs(&files, b"replacement");
        assert_eq!(files.sync_inputs(&input, 4096), Err(JobError::UnsafeStorage));
        assert_eq!(files.sync_inputs(&files.spool, 4096), Err(JobError::UnsafeStorage));
        assert_eq!(files.sync_inputs(&replacement, 4096).unwrap(), 11);
        assert_eq!(fs::read(dir.join("original")).unwrap(), b"original");
        assert_eq!(fs::read(dir.join(INPUTS)).unwrap(), b"replacement");
    }

    #[test]
    fn renamed_job_parent_cannot_redirect_input_creation_or_readback() {
        let base = root(); let path = base.join("job");
        fs::DirBuilder::new().mode(0o700).create(&path).unwrap();
        let files = create(&path);
        fs::rename(&path, base.join("held-job")).unwrap();
        fs::DirBuilder::new().mode(0o700).create(&path).unwrap();
        drop(write_inputs(&files, b"held parent"));
        assert!(!path.join(INPUTS).exists());
        assert_eq!(fs::read(base.join("held-job").join(INPUTS)).unwrap(), b"held parent");
        let mut retained = files.retained_inputs(4096).unwrap().unwrap();
        retained.seek(SeekFrom::Start(0)).unwrap();
        let mut bytes = Vec::new(); retained.read_to_end(&mut bytes).unwrap();
        assert_eq!(bytes, b"held parent");
    }

    #[test]
    fn relaxed_parent_blocks_input_read_and_sync() {
        let dir = root(); let files = create(&dir);
        let input = write_inputs(&files, b"private");
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o750)).unwrap();
        assert!(matches!(files.retained_inputs(4096), Err(JobError::UnsafeStorage)));
        assert_eq!(files.sync_inputs(&input, 4096), Err(JobError::UnsafeStorage));
    }
}
