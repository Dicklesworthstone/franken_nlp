//! Explicit one-time native CPU policy. No environment lookup or hot-loop lock.
//! The first model bind freezes scalar unless the host opted into a detected
//! kernel beforehand. Repeating the same request is idempotent; a late change
//! cannot silently create a mixture of process defaults in resident models.
use std::{error::Error, fmt, sync::OnceLock};
use super::{CpuKernelError, Int8Kernel, Int8KernelKind};

static PROCESS_KERNEL: OnceLock<Int8Kernel> = OnceLock::new();

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KernelConfigurationError {
    Unavailable(CpuKernelError),
    AlreadyConfigured { active: Int8KernelKind, requested: Int8KernelKind },
}
impl fmt::Display for KernelConfigurationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable(_) => f.write_str("explicit INT8 kernel is unavailable"),
            Self::AlreadyConfigured { .. } => f.write_str("native INT8 kernel policy is already fixed"),
        }
    }
}
impl Error for KernelConfigurationError {}

/// Call before model binding. This is an explicit experimental selection, not
/// a performance-table award or a model-fidelity certificate. Unsupported ISA
/// requests fail even when a different kernel has already been selected.
pub fn configure_process_kernel(kind: Int8KernelKind) -> Result<Int8KernelKind, KernelConfigurationError> {
    let kernel = Int8Kernel::detect(kind).map_err(KernelConfigurationError::Unavailable)?;
    install(&PROCESS_KERNEL, kernel)
}
/// Observation does not itself freeze the default or touch model state.
pub fn configured_process_kernel() -> Option<Int8KernelKind> {
    PROCESS_KERNEL.get().map(|kernel| kernel.kind())
}
pub(crate) fn process_kernel() -> Int8Kernel {
    *PROCESS_KERNEL.get_or_init(Int8Kernel::scalar)
}
fn install(cell: &OnceLock<Int8Kernel>, requested: Int8Kernel) -> Result<Int8KernelKind, KernelConfigurationError> {
    let active = cell.get_or_init(|| requested).kind();
    if active != requested.kind() {
        return Err(KernelConfigurationError::AlreadyConfigured { active, requested: requested.kind() });
    }
    Ok(active)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn repeated_selection_is_idempotent_without_resetting_the_cell() {
        let cell = OnceLock::new();
        assert!(cell.get().is_none());
        for _ in 0..2 { assert_eq!(install(&cell, Int8Kernel::scalar()).unwrap(), Int8KernelKind::Scalar); }
    }
    #[test]
    fn a_late_change_never_replaces_a_bound_default() {
        let cell = OnceLock::new();
        let _ = cell.get_or_init(Int8Kernel::scalar);
        for kind in Int8KernelKind::ALL {
            if kind == Int8KernelKind::Scalar { continue; }
            match Int8Kernel::detect(kind) {
                Ok(kernel) => {
                    assert_eq!(install(&cell, kernel), Err(KernelConfigurationError::AlreadyConfigured {
                        active: Int8KernelKind::Scalar, requested: kind }));
                    assert_eq!(cell.get().unwrap().kind(), Int8KernelKind::Scalar);
                }
                Err(CpuKernelError::UnsupportedIsa) => eprintln!("INT8_POLICY tier={kind} RESULT=SKIPPED_UNSUPPORTED_ISA"),
                Err(e) => panic!("unexpected kernel error: {e}"),
            }
        }
    }
}
