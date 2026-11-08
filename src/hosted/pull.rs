//! Process-owned runtime adapter for candidate release-part acquisition.
//!
//! This deliberately uses the existing EngineResources runtime rather than
//! creating a nested/ambient runtime. The explicit staging reservation models
//! HTTP/TLS/body bookkeeping that the pinned client allocates outside the
//! artifact byte stream; part bytes themselves are streamed to files.

use std::{error::Error, fmt};

use asupersync::cx::Cx;

use super::{HostedError, Pending};
use crate::{
    MemoryClass, NlpEngine,
    artifact::pull::http::{
        NetworkPartsOptions, NetworkPartsReport, NetworkPartsRequest, NetworkPullError,
        download_release_parts,
        direct::{NetworkArtifactReport, NetworkArtifactRequest, acquire_release_artifact},
    },
    artifact::pull::LOCAL_RELEASE_WORKSPACE_RESERVE_BYTES,
};

const MIN_NETWORK_RESERVE_BYTES: u64 = 2 * 1024 * 1024;

#[derive(Clone, Copy, Debug)]
pub struct NetworkPullHostOptions {
    /// Modeled process-ledger reserve for HTTP/TLS/DNS/body bookkeeping.
    /// This is not an RSS cap and does not include on-disk part bytes.
    pub network_reserve_bytes: u64,
}

impl Default for NetworkPullHostOptions {
    fn default() -> Self {
        Self { network_reserve_bytes: 16 * 1024 * 1024 }
    }
}

#[derive(Debug)]
pub enum HostedNetworkPullError {
    Hosted(HostedError),
    Transfer(NetworkPullError),
}

impl fmt::Display for HostedNetworkPullError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hosted(error) => write!(f, "hosted release acquisition refused: {error}"),
            Self::Transfer(error) => error.fmt(f),
        }
    }
}
impl Error for HostedNetworkPullError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Hosted(error) => Some(error),
            Self::Transfer(error) => Some(error),
        }
    }
}
impl From<HostedError> for HostedNetworkPullError {
    fn from(error: HostedError) -> Self { Self::Hosted(error) }
}
impl From<NetworkPullError> for HostedNetworkPullError {
    fn from(error: NetworkPullError) -> Self { Self::Transfer(error) }
}

impl NlpEngine {
    /// Stream an explicit release directly into one candidate artifact.
    ///
    /// Both local verification and transport bookkeeping are admitted before
    /// any manifest, artifact or network IO, including a final cache hit. The
    /// reservations and structured runtime remain owned until all verification
    /// and publication work completes. This does not activate a model root.
    pub fn acquire_release_artifact_candidate(
        &self,
        request: NetworkArtifactRequest,
        transfer: NetworkPartsOptions,
        host: NetworkPullHostOptions,
    ) -> Result<NetworkArtifactReport, HostedNetworkPullError> {
        if host.network_reserve_bytes < MIN_NETWORK_RESERVE_BYTES {
            return Err(HostedError::Limits("release acquisition network reserve").into());
        }
        let _sync_call = self.enter_sync_call().map_err(|_| HostedError::Reentrant)?;
        let lease = self.resources().acquire_lease();
        let local = Pending::reserve(&lease, MemoryClass::Staging,
            LOCAL_RELEASE_WORKSPACE_RESERVE_BYTES)?.commit()?;
        let network = Pending::reserve(&lease, MemoryClass::Staging, host.network_reserve_bytes)?
            .commit()?;
        let result = self.resources().runtime().block_on(async move {
            let cx = Cx::current().ok_or(HostedError::MissingRuntimeContext)?;
            acquire_release_artifact(&cx, &request, transfer)
                .await
                .map_err(HostedNetworkPullError::Transfer)
        });
        drop(network);
        drop(local);
        drop(lease);
        result
    }

    /// Run candidate release-part acquisition on the one process-owned runtime.
    ///
    /// The returned report has no activation authority. The runtime context,
    /// staging-memory obligation and request inputs remain owned until the
    /// network future has physically completed or returned an error.
    pub fn download_release_parts_candidate(
        &self,
        request: NetworkPartsRequest,
        transfer: NetworkPartsOptions,
        host: NetworkPullHostOptions,
    ) -> Result<NetworkPartsReport, HostedNetworkPullError> {
        if host.network_reserve_bytes < MIN_NETWORK_RESERVE_BYTES {
            return Err(HostedError::Limits("release acquisition network reserve").into());
        }
        let _sync_call = self.enter_sync_call().map_err(|_| HostedError::Reentrant)?;
        let lease = self.resources().acquire_lease();
        let network = Pending::reserve(&lease, MemoryClass::Staging, host.network_reserve_bytes)?
            .commit()?;
        let result = self.resources().runtime().block_on(async move {
            let cx = Cx::current().ok_or(HostedError::MissingRuntimeContext)?;
            download_release_parts(&cx, &request, transfer)
                .await
                .map_err(HostedNetworkPullError::Transfer)
        });
        drop(network);
        drop(lease);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn network_reserve_is_explicit_and_nonzero() {
        assert!(NetworkPullHostOptions::default().network_reserve_bytes >= MIN_NETWORK_RESERVE_BYTES);
        assert_eq!(MIN_NETWORK_RESERVE_BYTES, 2 * 1024 * 1024);
    }

    fn send<T: Send + 'static>() {}

    #[test]
    fn transfer_request_and_report_can_cross_the_owned_runtime_boundary() {
        send::<NetworkPartsRequest>();
        send::<NetworkPartsOptions>();
        send::<NetworkPartsReport>();
        send::<NetworkArtifactRequest>();
        send::<NetworkArtifactReport>();
        send::<HostedNetworkPullError>();
    }
}
