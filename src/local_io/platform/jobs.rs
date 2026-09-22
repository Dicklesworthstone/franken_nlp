//! Owned-job files under the existing descriptor-rooted local IO profile.
//! A kernel-held advisory lock survives neither process exit nor File drop;
//! no stale lock-file deletion or process-ID guessing is used on recovery.
use super::*;
use crate::jobs::JobError;
use std::fs::TryLockError;

const DATABASE: &str = "journal.fsqlite";
const SPOOL: &str = "results.spool";
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
