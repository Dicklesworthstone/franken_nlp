//! G0-08 checked bounded range-read accounting probe.
//!
//! This deterministic in-memory stand-in locks the panel bound and overflow
//! rules. It intentionally does not report process RSS: the ADR remains
//! BLOCKED until a retained shard-scale host measurement exists.

const PANEL_CAP_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RangeError {
    Overflow,
    ExceedsPanelCap,
    OutsideSource,
}

fn checked_range(
    source_len: usize,
    offset: usize,
    bytes: usize,
) -> Result<std::ops::Range<usize>, RangeError> {
    if bytes > PANEL_CAP_BYTES {
        return Err(RangeError::ExceedsPanelCap);
    }
    let end = offset.checked_add(bytes).ok_or(RangeError::Overflow)?;
    if end > source_len {
        return Err(RangeError::OutsideSource);
    }
    Ok(offset..end)
}

fn bounded_panel<'a>(
    source: &'a [u8],
    offset: usize,
    bytes: usize,
) -> Result<&'a [u8], RangeError> {
    let range = checked_range(source.len(), offset, bytes)?;
    Ok(&source[range])
}

#[test]
fn converter_panel_access_rejects_overflow_and_never_exceeds_the_declared_cap() {
    let source = (0_u8..=255)
        .cycle()
        .take(2 * 1024 * 1024)
        .collect::<Vec<_>>();
    let panel = bounded_panel(&source, 128, 1_024).expect("bounded range must fit");
    assert_eq!(panel.len(), 1_024);
    assert_eq!(panel[0], 128);
    assert_eq!(panel[1_023], 127);
    assert_eq!(
        bounded_panel(&source, source.len() - 4, 8),
        Err(RangeError::OutsideSource)
    );
    assert_eq!(
        checked_range(source.len(), usize::MAX, 1),
        Err(RangeError::Overflow)
    );
    assert_eq!(
        checked_range(source.len(), 0, PANEL_CAP_BYTES + 1),
        Err(RangeError::ExceedsPanelCap)
    );
    let estimated_peak_bytes = PANEL_CAP_BYTES
        .checked_add(PANEL_CAP_BYTES)
        .and_then(|bytes| bytes.checked_add(PANEL_CAP_BYTES))
        .expect("bounded peak model must fit usize");
    println!(
        "G0_PROBE8 case=bounded-range-access RESULT=PASS panel_cap_bytes={PANEL_CAP_BYTES} estimated_panel_scratch_output_peak_bytes={estimated_peak_bytes} authority=range-model-only"
    );
    println!("G0_PROBE8 RESULT=PASS cases=1 authority=range-model-only");
}
