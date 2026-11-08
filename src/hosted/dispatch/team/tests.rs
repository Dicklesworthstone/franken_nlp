use super::*;

#[test]
fn workspace_quote_covers_the_actual_reusable_child_payloads_and_protocol_reserves() {
    for width in [2, 3, 7, 64] {
        for group_width in [1, 4, 128, MAX_MODEL_K] {
            let workspace = Workspace::new(width, group_width).unwrap();
            let payload = workspace.children.iter().map(WorkerBuffers::payload_bytes).sum::<usize>()
                + workspace.coordinator_scratch.len();
            assert_eq!(int4_team_workspace_bytes(width, group_width).unwrap(),
                payload as u64 + 4096 * (width as u64 - 1) + 65536);
            assert_eq!(workspace.children.len(), width - 1);
        }
    }
    for width in [0, 1, 65, usize::MAX] {
        assert!(Workspace::new(width, 128).is_err());
    }
}

// The selected dependency intentionally gates Cx::for_testing. These native
// protocol tests use that existing opt-in census feature; no production path
// constructs a context or alternate resource domain out of band.
#[cfg(feature = "asupersync-census")]
mod native;
