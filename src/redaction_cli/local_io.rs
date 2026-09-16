//! Explicit local-output profile, separate from model-root/cache authority.
//!
//! Linux x86_64/aarch64 only, requiring procfs and a filesystem implementing
//! exclusive creation, hard links and file/directory sync. No unsafe code or
//! external helper is used. Each path component is opened relative to a live
//! directory handle via our own /proc/self/fd entry; untrusted symlinks are not
//! followed. A private 0700 parent is mandatory for keys and output files.
//! Same-UID/root attackers and malicious mounts are outside this profile.
//! This implementation is not platform ratification or a durability receipt.
use std::{fs::File, path::Path};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LocalIoError {
    UnsupportedProfile, InvalidPath, PrivateParentRequired, UnsafeFile,
    AlreadyExists, Io, StageExhausted, PublicationUncertain,
}
pub(super) const PROFILE_SUPPORTED: bool = cfg!(all(target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64")));

#[cfg(all(target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64")))]
mod platform {
    use super::*;
    use std::{ffi::OsString, fs::{self, OpenOptions, Metadata}, io::{Read, Write},
        os::{fd::AsRawFd, unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt}},
        path::{Component, PathBuf}, sync::atomic::{AtomicU64, Ordering}};
    // Linux UAPI, not portable Unix constants. arm64 overrides the generic
    // directory/no-follow bits for AArch32 compatibility.
    // Sources: include/uapi/asm-generic/fcntl.h and
    // arch/arm64/include/uapi/asm/fcntl.h in torvalds/linux.
    #[cfg(target_arch = "x86_64")]
    const DIRECTORY: i32 = 1 << 16;
    #[cfg(target_arch = "x86_64")]
    const NOFOLLOW: i32 = 1 << 17;
    #[cfg(target_arch = "aarch64")]
    const DIRECTORY: i32 = 1 << 14;
    #[cfg(target_arch = "aarch64")]
    const NOFOLLOW: i32 = 1 << 15;
    const NONBLOCK: i32 = 1 << 11;
    const PATH_ONLY: i32 = 1 << 21;
    static NEXT_STAGE: AtomicU64 = AtomicU64::new(0);

    fn io<T>(value: std::io::Result<T>) -> Result<T, LocalIoError> { value.map_err(|_| LocalIoError::Io) }
    fn fd_path(file: &File) -> PathBuf { PathBuf::from(format!("/proc/self/fd/{}", file.as_raw_fd())) }
    fn identity(meta: &Metadata) -> (u64, u64) { (meta.dev(), meta.ino()) }

    fn process_uid() -> Result<u32, LocalIoError> {
        // Reject setuid/fsuid transitions rather than guessing which owner is
        // authoritative. Status content is bounded and never appears in errors.
        let mut bytes = Vec::new();
        io(File::open("/proc/self/status"))?.take(65_537).read_to_end(&mut bytes).map_err(|_| LocalIoError::Io)?;
        if bytes.len() > 65_536 { return Err(LocalIoError::UnsafeFile); }
        let text = std::str::from_utf8(&bytes).map_err(|_| LocalIoError::UnsafeFile)?;
        let row = text.lines().find_map(|l| l.strip_prefix("Uid:")).ok_or(LocalIoError::UnsafeFile)?;
        let mut fields = row.split_ascii_whitespace();
        let uid = fields.next().and_then(|n| n.parse::<u32>().ok()).ok_or(LocalIoError::UnsafeFile)?;
        for _ in 0..3 {
            if fields.next().and_then(|n| n.parse::<u32>().ok()) != Some(uid) { return Err(LocalIoError::UnsafeFile); }
        }
        if fields.next().is_some() || io(fs::metadata("/proc/self"))?.uid() != uid { return Err(LocalIoError::UnsafeFile); }
        Ok(uid)
    }
    fn directory(path: &Path) -> Result<File, LocalIoError> {
        io(OpenOptions::new().read(true).custom_flags(DIRECTORY | NOFOLLOW | NONBLOCK).open(path))
    }
    fn parent(path: &Path, private: bool) -> Result<(File, OsString), LocalIoError> {
        // Reject traversal rather than lexical-normalizing away a boundary.
        // Component::Normal alone is permitted for each child lookup.
        let mut components: Vec<_> = path.components().collect();
        if components.len() > 256 { return Err(LocalIoError::InvalidPath); }
        let Some(Component::Normal(name)) = components.pop() else { return Err(LocalIoError::InvalidPath); };
        let name = name.to_owned();
        let mut dir = directory(if path.is_absolute() { Path::new("/") } else { Path::new(".") })?;
        for component in components {
            match component {
                Component::RootDir | Component::CurDir => {},
                Component::Normal(child) => { dir = directory(&fd_path(&dir).join(child))?; },
                _ => return Err(LocalIoError::InvalidPath),
            }
        }
        if private {
            let meta = io(dir.metadata())?;
            if meta.uid() != process_uid()? || meta.mode() & 0o7777 != 0o700 {
                return Err(LocalIoError::PrivateParentRequired);
            }
        }
        Ok((dir, name))
    }
    fn regular(path: &Path, private: bool) -> Result<File, LocalIoError> {
        let (dir, name) = parent(path, private)?;
        // O_PATH obtains an identity without opening a device or blocking on a
        // FIFO. O_NOFOLLOW means a symlink itself is inspected and refused.
        let pin = io(OpenOptions::new().read(true).custom_flags(PATH_ONLY | NOFOLLOW).open(fd_path(&dir).join(name)))?;
        let before = io(pin.metadata())?;
        if !before.is_file() { return Err(LocalIoError::UnsafeFile); }
        if private && (before.uid() != process_uid()? || before.mode() & 0o7077 != 0
            || before.nlink() != 1 || !(32..=4096).contains(&before.len())) {
            return Err(LocalIoError::UnsafeFile);
        }
        // This is our own live descriptor's trusted procfs magic link, not an
        // attacker-supplied path. Keep pin alive until the read handle is bound.
        let file = io(OpenOptions::new().read(true).custom_flags(NONBLOCK).open(fd_path(&pin)))?;
        let after = io(file.metadata())?;
        if identity(&before) != identity(&after) || !after.is_file()
            || (private && (after.uid() != before.uid() || after.mode() != before.mode()
                || after.nlink() != 1 || after.len() != before.len())) { return Err(LocalIoError::UnsafeFile); }
        Ok(file)
    }
    pub(crate) fn open_key(path: &Path) -> Result<File, LocalIoError> { regular(path, true) }
    pub(crate) fn open_document(path: &Path) -> Result<File, LocalIoError> { regular(path, false) }
    pub(crate) fn same_file(a: &File, b: &File) -> Result<bool, LocalIoError> {
        Ok(identity(&io(a.metadata())?) == identity(&io(b.metadata())?))
    }

    pub(crate) struct Destination { parent: File, name: OsString }
    impl Destination {
        pub(crate) fn prepare(path: &Path) -> Result<Self, LocalIoError> {
            let (parent, name) = parent(path, true)?;
            if name.as_encoded_bytes().starts_with(b".fnlp-redact-") { return Err(LocalIoError::InvalidPath); }
            let destination = Self { parent, name };
            // Early user feedback only. The later no-replace hard link, not
            // this metadata lookup, is the publication exclusion authority.
            match fs::symlink_metadata(destination.path()) {
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(destination),
                Ok(_) => Err(LocalIoError::AlreadyExists),
                Err(_) => Err(LocalIoError::Io),
            }
        }
        fn path(&self) -> PathBuf { fd_path(&self.parent).join(&self.name) }
        pub(crate) fn same_target(&self, other: &Self) -> Result<bool, LocalIoError> {
            Ok(self.name == other.name && same_file(&self.parent, &other.parent)?)
        }
        pub(crate) fn stage(self, bytes: &[u8]) -> Result<StagedOutput, LocalIoError> {
            for _ in 0..128 {
                let sequence = NEXT_STAGE.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
                    .map_err(|_| LocalIoError::StageExhausted)?;
                let name = OsString::from(format!(".fnlp-redact-{}-{sequence}.part", std::process::id()));
                if name == self.name { continue; }
                let path = fd_path(&self.parent).join(&name);
                let file = match OpenOptions::new().write(true).create_new(true).mode(0o600)
                    .custom_flags(NOFOLLOW | NONBLOCK).open(&path) {
                    Ok(file) => file,
                    Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                    Err(_) => return Err(LocalIoError::Io),
                };
                let mut stage = StagedOutput { destination: self, name, file };
                io(stage.file.set_permissions(fs::Permissions::from_mode(0o600)))?;
                io(stage.file.write_all(bytes))?;
                io(stage.file.sync_all())?;
                stage.check_stage()?;
                return Ok(stage);
            }
            Err(LocalIoError::StageExhausted)
        }
    }
    pub(crate) struct StagedOutput { destination: Destination, name: OsString, file: File }
    impl StagedOutput {
        fn path(&self) -> PathBuf { fd_path(&self.destination.parent).join(&self.name) }
        fn check_stage(&self) -> Result<(), LocalIoError> {
            let held = io(self.file.metadata())?;
            let named = io(fs::symlink_metadata(self.path()))?;
            let parent = io(self.destination.parent.metadata())?;
            if !named.is_file() || identity(&held) != identity(&named) || held.nlink() != 1
                || held.mode() & 0o7777 != 0o600 || held.uid() != parent.uid()
                || parent.mode() & 0o7777 != 0o700 || parent.uid() != process_uid()? {
                return Err(LocalIoError::UnsafeFile);
            }
            Ok(())
        }
        pub(crate) fn publish(self) -> Result<(), LocalIoError> {
            self.check_stage()?;
            // Hard-link publication is same-directory and cannot replace an
            // existing file/symlink/device. Never fall back to replacing rename.
            match fs::hard_link(self.path(), self.destination.path()) {
                Ok(()) => {},
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => return Err(LocalIoError::AlreadyExists),
                Err(_) => return Err(LocalIoError::PublicationUncertain),
            }
            // From this point the complete destination may be visible. Never
            // unlink it on failure: report uncertain publication, no success.
            self.destination.parent.sync_all().map_err(|_| LocalIoError::PublicationUncertain)?;
            fs::remove_file(self.path()).map_err(|_| LocalIoError::PublicationUncertain)?;
            self.destination.parent.sync_all().map_err(|_| LocalIoError::PublicationUncertain)?;
            Ok(())
        }
    }
    impl Drop for StagedOutput {
        fn drop(&mut self) {
            // Remove only our own exact staging inode, never a replacement or
            // the published destination. Interrupted cleanup may leave a 0600
            // stage; startup does not blindly scavenge user directories.
            if let (Ok(held), Ok(named)) = (self.file.metadata(), fs::symlink_metadata(self.path())) {
                if named.is_file() && identity(&held) == identity(&named) { let _ = fs::remove_file(self.path()); }
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::os::unix::fs::{DirBuilderExt, symlink};
        fn root() -> PathBuf {
            let n = NEXT_STAGE.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!("fnlp-private-{}-{n}", std::process::id()));
            fs::DirBuilder::new().mode(0o700).create(&path).unwrap(); path
        }
        #[test]
        fn output_is_complete_private_and_never_replaces() {
            let dir = root(); let out = dir.join("out");
            let stage = Destination::prepare(&out).unwrap().stage(b"complete").unwrap();
            assert!(!out.exists()); stage.publish().unwrap();
            assert_eq!(fs::read(&out).unwrap(), b"complete");
            assert_eq!(fs::metadata(&out).unwrap().mode() & 0o777, 0o600);
            assert!(matches!(Destination::prepare(&out), Err(LocalIoError::AlreadyExists)));
        }
        #[test]
        fn racing_destination_is_not_overwritten() {
            let dir = root(); let out = dir.join("out");
            let stage = Destination::prepare(&out).unwrap().stage(b"new").unwrap();
            fs::write(&out, b"keep").unwrap();
            assert_eq!(stage.publish(), Err(LocalIoError::AlreadyExists));
            assert_eq!(fs::read(out).unwrap(), b"keep");
        }
        #[test]
        fn destination_parent_rename_cannot_redirect_staged_bytes() {
            let root = root(); let dir = root.join("private");
            fs::DirBuilder::new().mode(0o700).create(&dir).unwrap();
            let stage = Destination::prepare(&dir.join("out")).unwrap().stage(b"secret").unwrap();
            fs::rename(&dir, root.join("moved")).unwrap();
            fs::DirBuilder::new().mode(0o700).create(&dir).unwrap();
            stage.publish().unwrap();
            assert!(!dir.join("out").exists()); assert_eq!(fs::read(root.join("moved/out")).unwrap(), b"secret");
        }
        #[test]
        fn symlink_parent_and_final_target_are_refused() {
            let dir = root(); symlink(&dir, dir.join("alias")).unwrap();
            assert!(Destination::prepare(&dir.join("alias/out")).is_err());
            symlink("absent", dir.join("out")).unwrap();
            assert!(matches!(Destination::prepare(&dir.join("out")), Err(LocalIoError::AlreadyExists)));
        }
        #[test]
        fn private_key_checks_handle_mode_links_size_and_identity() {
            let dir = root(); let path = dir.join("key");
            let mut key = OpenOptions::new().write(true).create_new(true).mode(0o600).open(&path).unwrap();
            key.write_all(&[0x42; 32]).unwrap();
            assert!(open_key(&path).is_ok());
            fs::hard_link(&path, dir.join("alias")).unwrap(); assert!(open_key(&path).is_err());
            fs::remove_file(dir.join("alias")).unwrap();
            key.set_permissions(fs::Permissions::from_mode(0o644)).unwrap(); assert!(open_key(&path).is_err());
            key.set_permissions(fs::Permissions::from_mode(0o600)).unwrap();
            symlink(&path, dir.join("sym")).unwrap(); assert!(open_key(&dir.join("sym")).is_err());
            assert!(same_file(&open_key(&path).unwrap(), &open_document(&path).unwrap()).unwrap());
        }
        #[test]
        fn relaxed_parent_is_refused_and_dropped_stage_is_removed() {
            let dir = root(); let stage = Destination::prepare(&dir.join("out")).unwrap().stage(b"private").unwrap();
            let path = stage.path(); drop(stage); assert!(!path.exists());
            fs::set_permissions(&dir, fs::Permissions::from_mode(0o755)).unwrap();
            assert!(matches!(Destination::prepare(&dir.join("out")), Err(LocalIoError::PrivateParentRequired)));
        }
        #[test]
        fn path_traversal_and_alias_destinations_are_detected() {
            let dir = root(); assert!(Destination::prepare(&dir.join("../out")).is_err());
            let a = Destination::prepare(&dir.join("out")).unwrap();
            let b = Destination::prepare(&dir.join("./out")).unwrap();
            assert!(a.same_target(&b).unwrap());
        }
    }
}

#[cfg(not(all(target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64"))))]
mod platform {
    use super::*;
    pub(crate) fn open_key(_: &Path) -> Result<File, LocalIoError> { Err(LocalIoError::UnsupportedProfile) }
    pub(crate) fn open_document(path: &Path) -> Result<File, LocalIoError> {
        if !std::fs::metadata(path).map_err(|_| LocalIoError::Io)?.is_file() { return Err(LocalIoError::UnsafeFile); }
        let file = File::open(path).map_err(|_| LocalIoError::Io)?;
        if !file.metadata().map_err(|_| LocalIoError::Io)?.is_file() { return Err(LocalIoError::UnsafeFile); }
        Ok(file)
    }
    pub(crate) fn same_file(_: &File, _: &File) -> Result<bool, LocalIoError> { Err(LocalIoError::UnsupportedProfile) }
    pub(crate) struct Destination;
    impl Destination {
        pub(crate) fn prepare(_: &Path) -> Result<Self, LocalIoError> { Err(LocalIoError::UnsupportedProfile) }
        pub(crate) fn same_target(&self, _: &Self) -> Result<bool, LocalIoError> { Err(LocalIoError::UnsupportedProfile) }
        pub(crate) fn stage(self, _: &[u8]) -> Result<StagedOutput, LocalIoError> { Err(LocalIoError::UnsupportedProfile) }
    }
    pub(crate) struct StagedOutput;
    impl StagedOutput { pub(crate) fn publish(self) -> Result<(), LocalIoError> { Err(LocalIoError::UnsupportedProfile) } }
}
pub(super) use platform::{Destination, open_document, open_key, same_file};
