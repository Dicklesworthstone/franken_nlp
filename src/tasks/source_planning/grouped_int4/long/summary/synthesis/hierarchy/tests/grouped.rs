//! Scheduling fixtures exercise the common reducer, not native math or quality.
use super::*;
use super::super::execution::Driver;

struct Windows {
    inner: Script, width: usize, windows: Vec<usize>, stale_witness: Option<usize>,
    short: bool, reversed: bool,
}
impl Driver for Windows {
    type Witness = Sha256Digest;
    fn batch_width(&self) -> usize { self.width }
    fn count<C: DecodeStepControl>(&mut self, text: &str, control: &mut C) -> Result<usize, Int4SourceMapError> {
        self.inner.count(text, control)
    }
    fn prepare<C: DecodeStepControl>(&mut self, text: &str, control: &mut C) -> Result<Self::Witness, Int4SourceMapError> {
        let witness = self.inner.prepare(text, control)?;
        Ok(if self.stale_witness == Some(self.inner.preparations.len()) {
            Sha256Digest::of_bytes(b"changed after complete-level admission")
        } else { witness })
    }
    fn run<C: DecodeStepControl>(&mut self, text: &str, witness: &Self::Witness, control: &mut C)
        -> Result<(Int4SourceTaskRun, Int4Work), Int4SourceMapError> {
        self.inner.run(text, witness, control)
    }
    fn run_batch<C: DecodeStepControl>(&mut self, texts: &[&str], witnesses: &[Self::Witness], control: &mut C)
        -> Result<Vec<(Int4SourceTaskRun, Int4Work)>, Int4SourceMapError> {
        if texts.is_empty() || texts.len() > self.width || texts.len() != witnesses.len() {
            return Err(Int4SourceMapError::Admission);
        }
        // Match the native adapter's all-witnesses-before-first-forward gate.
        for (text, witness) in texts.iter().zip(witnesses) {
            if *witness != Sha256Digest::of_bytes(text.as_bytes()) { return Err(Int4SourceMapError::Admission); }
        }
        self.windows.push(texts.len());
        let mut rows = vector(texts.len())?;
        for (text, witness) in texts.iter().zip(witnesses) { rows.push(self.inner.run(text, witness, control)?); }
        if self.short { rows.pop(); }
        if self.reversed { rows.reverse(); }
        Ok(rows)
    }
}
fn windows(expected: Int4SummaryHierarchyPreflight, width: usize) -> Windows {
    Windows { inner: script(expected), width, windows: vec![], stale_witness: None, short: false, reversed: false }
}
fn grouped_complete(width: usize, no_evidence: bool, empty_outputs: bool) -> (Int4SummaryHierarchyRun, Windows) {
    let (tail, discovery) = discovered(no_evidence);
    let mut driver = windows(tail.expected, width); driver.inner.empty = empty_outputs;
    let run = tail.execute(discovery, |tail, collection, remaining, control|
        execution::reduce(SOURCE, collection, tail.base.request, tail.expected, 4 << 20, remaining,
            &mut driver, control), &mut go()).unwrap();
    (run, driver)
}

#[test]
fn bounded_windows_keep_complete_semantic_bytes_evidence_order_and_global_ledgers() {
    for (no_evidence, empty_outputs) in [(false, false), (false, true), (true, false)] {
        let (serial, _) = grouped_complete(1, no_evidence, empty_outputs);
        let bytes = crate::canonjson::canonical_bytes(&serial).unwrap();
        if !empty_outputs {
            assert_eq!(bytes, crate::canonjson::canonical_bytes(&complete(no_evidence).1).unwrap());
        }
        for width in [1, 2, 4] {
            let (run, driver) = grouped_complete(width, no_evidence, empty_outputs);
            driver.inner.expected.verify_completed(&run).unwrap();
            assert_eq!(crate::canonjson::canonical_bytes(&run).unwrap(), bytes);
            assert!(driver.windows.iter().all(|&rows| rows <= width));
            if no_evidence { assert!(driver.windows.is_empty()); assert!(driver.inner.inputs.is_empty()); }
            else if empty_outputs {
                assert_eq!(driver.windows, if width == 1 { vec![1, 1] } else { vec![2] });
                assert_eq!(run.status, SummarySynthesisStatus::NoBulletsProduced);
                assert_eq!(run.passes.len(), 2);
            } else {
                assert_eq!(driver.windows, if width == 1 { vec![1, 1, 1] } else { vec![2, 1] });
                assert_eq!(driver.inner.inputs, ["éabc", "βdef", "é\n\nβ"]);
                assert_eq!(driver.inner.events, ["prepare", "prepare", "run", "run", "prepare", "run"]);
                assert_eq!(run.passes[1].bullets[0].citations[0].spans[0],
                    VerifiedSourceSpan { byte_start: 9, byte_end: 11, scalar_start: 8, scalar_end: 9 });
                assert_eq!(run.tokenizer_work, HierarchyTokenizerWork { calls: 5, bytes: 30 });
            }
        }
    }
}

#[test]
fn final_group_admission_or_recompiled_witness_failure_prevents_the_whole_window() {
    for stale in [false, true] { for width in [2, 4] {
        let (expected, _, collection, mut remaining) = setup(false);
        let start = remaining; let mut driver = windows(expected, width);
        if stale { driver.stale_witness = Some(2); } else { driver.inner.fail_prepare = Some(2); }
        assert!(matches!(execution::reduce(SOURCE, collection, request(), expected, 4 << 20,
            &mut remaining, &mut driver, &mut go()), Err(Int4SourceMapError::Admission)));
        assert_eq!(driver.inner.preparations.len(), 2);
        assert!(driver.inner.inputs.is_empty()); assert!(driver.windows.is_empty());
        assert_eq!(remaining, start);
    } }
}

#[test]
fn short_reordered_or_failed_native_window_never_reaches_a_later_level() {
    for axis in 0..3 {
        let (expected, _, collection, mut remaining) = setup(false); let start = remaining;
        let mut driver = windows(expected, 2);
        match axis { 0 => driver.short = true, 1 => driver.reversed = true, _ => driver.inner.fail_run = Some(2) }
        let error = execution::reduce(SOURCE, collection, request(), expected, 4 << 20,
            &mut remaining, &mut driver, &mut go()).err().unwrap();
        assert_eq!(driver.windows, [2]); assert_eq!(driver.inner.inputs.len(), 2);
        if axis == 2 { assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline)); }
        if axis != 1 { assert_eq!(remaining, start); }
    }
}

#[test]
fn batching_does_not_renew_tokenizer_verifier_pass_or_publication_budgets() {
    let (expected, _, collection, start) = setup(false); let mut remaining = start;
    execution::reduce(SOURCE, collection, request(), expected, 4 << 20,
        &mut remaining, &mut windows(expected, 2), &mut go()).unwrap();
    let used = SourceMapVerification::used(start, remaining).unwrap();
    for axis in 0..8 {
        let (mut expected, _, collection, mut remaining) = setup(false);
        match axis {
            0 => remaining.max_fields = used.fields - 1,
            1 => remaining.max_matches = used.matches - 1,
            2 => remaining.max_scan_steps = used.scan_steps - 1,
            3 => expected.limits.max_tokenizer_calls = 4,
            4 => expected.limits.max_tokenizer_bytes = 29,
            5 => expected.limits.max_passes = 2,
            6 => expected.limits.max_levels = 1,
            _ => (),
        }
        let mut driver = windows(expected, 2);
        assert!(execution::reduce(SOURCE, collection, request(), expected, if axis == 7 { 1 } else { 4 << 20 },
            &mut remaining, &mut driver, &mut go()).is_err(), "{axis}");
        if axis == 5 || axis == 6 { assert!(driver.windows.is_empty()); }
    }
}

#[test]
fn invalid_width_nonprogress_and_last_checkpoint_fail_closed() {
    for width in [0, 65, usize::MAX] {
        let (expected, _, collection, mut remaining) = setup(false); let mut driver = windows(expected, width);
        assert!(matches!(execution::reduce(SOURCE, collection, request(), expected, 4 << 20,
            &mut remaining, &mut driver, &mut go()), Err(Int4SourceMapError::InvalidLimits)));
        assert!(driver.inner.preparations.is_empty()); assert!(driver.windows.is_empty());
    }
    let (expected, _, collection, mut remaining) = setup(false); let mut driver = windows(expected, 2);
    driver.inner.full = true;
    assert!(execution::reduce(SOURCE, collection, request(), expected, 4 << 20,
        &mut remaining, &mut driver, &mut go()).is_err());
    assert_eq!(driver.windows, [2]);
    let (expected, _, collection, mut remaining) = setup(false); let mut control = go();
    execution::reduce(SOURCE, collection, request(), expected, 4 << 20,
        &mut remaining, &mut windows(expected, 2), &mut control).unwrap();
    let (expected, _, collection, mut remaining) = setup(false); let mut driver = windows(expected, 2);
    let error = execution::reduce(SOURCE, collection, request(), expected, 4 << 20, &mut remaining, &mut driver,
        &mut Control { calls: 0, stop: control.calls }).err().unwrap();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline));
    assert_eq!(driver.windows, [2, 1]); assert_eq!(driver.inner.inputs.len(), 3);
}
