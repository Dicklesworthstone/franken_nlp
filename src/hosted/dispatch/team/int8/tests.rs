use super::*;

#[test]
fn fixed_quote_covers_every_child_payload_and_formation_reserve() {
    for width in [2, 3, 7, 64] {
        let workspace = Workspace::new(width).unwrap();
        let payload = workspace.children.iter().map(WorkerBuffers::payload_bytes).sum::<usize>();
        assert_eq!(int8_team_workspace_bytes(width).unwrap(),
            payload as u64 + 4096 * (width as u64 - 1) + 65536);
        assert_eq!(workspace.children.len(), width - 1);
    }
    for width in [0, 1, 65, usize::MAX] { assert!(Workspace::new(width).is_err()); }
}

#[test]
fn balanced_ranges_cover_every_position_once_including_idle_workers() {
    for width in [2, 3, 7, 64] {
        for rows in [1, 2, 3, 6, 10, 1024, NANBEIGE_VOCAB_SIZE] {
            let ranges: Vec<_> = (0..width).map(|ordinal| projection::row_range(rows, width, ordinal)).collect();
            assert_eq!(ranges[0].start, 0);
            assert_eq!(ranges.last().unwrap().end, rows);
            assert!(ranges.windows(2).all(|adjacent| adjacent[0].end == adjacent[1].start));
            assert!(ranges.iter().all(|range| range.len() <= rows.div_ceil(width)));
            assert_eq!(ranges.iter().map(|range| range.len()).sum::<usize>(), rows);
        }
    }
}

// The pinned dependency intentionally gates testing Cx construction. These
// tests use actual scoped workers and lanes, with no alternative runtime in
// production and no synthetic whole-model parity claim.
#[cfg(feature = "asupersync-census")]
mod native;
