//! UNRUN synthetic-driver contracts for prompt scheduling through the original
//! JSON decoder. These fixtures do not establish real-model parity or speed.
use super::*;
use crate::native_engine::constrained_int4::prefill::{PrefillDriver, drive_layer_major};

fn limits(rows: usize) -> Int4PrefillLimits {
    Int4PrefillLimits { max_batch_rows: rows,
        max_extra_scratch_bytes: Int4PrefillLimits::required_extra_scratch_bytes(rows).unwrap() }
}
impl PrefillDriver for Script {
    fn append_layer_major(&mut self, prompt: &[u32], limits: Int4PrefillLimits) -> Result<(), Int4JsonError> {
        self.prompt_calls += 1;
        for chunk in prompt.chunks(limits.max_batch_rows) {
            self.prompt_morsels.push(chunk.len());
            for &token in chunk {
                if let Some(cause) = self.control.prefill_checkpoint(self.seen.len()) {
                    return Err(StrictInt4Error::Cancelled(cause).into());
                }
                self.append(token)?;
            }
        }
        Ok(())
    }
}

#[test]
fn prompt_morsels_preserve_tokens_masks_scored_eos_and_exact_group_work() {
    let p = program(r#"{"type":"boolean"}"#);
    let v = table(&[b"", b"true", b"malformed"]);
    let prompt = [2; 9]; let options = options(2); let budget = budget(prompt.len(), 2, v.width());
    let mut serial = Script::new(vec![100.0, 1.0, 1000.0]);
    let expected = execute(&prompt, &p, &v, &options, budget, &mut serial).unwrap();
    for rows in [1, 2, 4, 8, 64] {
        let mut grouped = Script::new(serial.row.clone());
        let actual: Int4JsonRun = drive_layer_major(GROUP, &prompt, &p, &v, &options,
            budget, &mut grouped, limits(rows), Ok::<_, Int4JsonError>).unwrap();
        assert_eq!(crate::canonjson::canonical_bytes(&actual).unwrap(),
            crate::canonjson::canonical_bytes(&expected).unwrap());
        assert_eq!(grouped.seen, serial.seen);
        assert_eq!(grouped.prompt_calls, 1);
        assert_eq!(grouped.prompt_morsels, prompt.chunks(rows).map(|chunk| chunk.len()).collect::<Vec<_>>());
        assert_eq!(grouped.heads, 2); // One final-prompt head and one scored EOS.
        assert!(!grouped.aborted);
        assert_eq!(actual.model_work, Int4Work::for_sequence(GROUP, 0, 10, 6).unwrap());
    }
}

#[test]
fn source_membership_numeric_continuation_and_utf8_keep_the_shared_grammar_semantics() {
    let source = JsonProgram::compile_with_source(
        r#"{"type":"string","maxLength":16,"x-fnlp-source":"verbatim"}"#,
        "Alice and Alice", CompileLimits::default(), SourceRuntimeLimits::default()).unwrap();
    let cases = [
        (source, table(&[b"", br#""Mallory""#, br#""Alice""#]), vec![10.0, 100.0, 1.0], 2),
        (program(r#"{"type":"integer","enum":[1,10]}"#),
            table(&[b"", b"1", b"0"]), vec![0.0, 2.0, 1.0], 3),
        (program(r#"{"type":"string","maxLength":1,"enum":["é"]}"#),
            table(&[b"", b"\"", &[0xc3], &[0xa9]]), vec![100.0, 3.0, 2.0, 1.0], 5),
    ];
    for (program, vocabulary, row, output) in cases {
        let prompt = [1; 5]; let options = options(output); let budget = budget(5, output, vocabulary.width());
        let mut serial = Script::new(row.clone());
        let expected = execute(&prompt, &program, &vocabulary, &options, budget, &mut serial).unwrap();
        for rows in [1, 4, 64] {
            let mut grouped = Script::new(row.clone());
            let actual = drive_layer_major(GROUP, &prompt, &program, &vocabulary, &options,
                budget, &mut grouped, limits(rows), Ok::<_, Int4JsonError>).unwrap();
            assert_eq!(actual, expected);
            assert_eq!(grouped.seen, serial.seen); assert_eq!(grouped.heads, serial.heads);
            assert_eq!(grouped.prompt_calls, 1); assert!(!grouped.aborted);
        }
    }
}

#[test]
fn invalid_prefill_declarations_refuse_without_driver_entry_or_abort() {
    let p = program(r#"{"type":"boolean"}"#); let v = table(&[b"", b"true"]);
    let mut short = limits(4); short.max_extra_scratch_bytes -= 1;
    for cap in [short, Int4PrefillLimits { max_batch_rows: 0, max_extra_scratch_bytes: u64::MAX },
        Int4PrefillLimits { max_batch_rows: 65, max_extra_scratch_bytes: u64::MAX },
        Int4PrefillLimits { max_batch_rows: usize::MAX, max_extra_scratch_bytes: u64::MAX }] {
        let mut d = Script::new(vec![0.0, 1.0]);
        let result = drive_layer_major(GROUP, &[1, 1], &p, &v, &options(2), budget(2, 2, 2),
            &mut d, cap, Ok::<_, Int4JsonError>);
        assert!(result.is_err()); assert!(!d.aborted); assert!(d.seen.is_empty());
        assert_eq!(d.prompt_calls, 0); assert_eq!(d.heads, 0);
    }
}

#[test]
fn every_original_work_and_mask_ceiling_still_precedes_layer_major_prompt_entry() {
    let p = program(r#"{"type":"boolean"}"#); let v = table(&[b"", b"true"]);
    for axis in 0..7 {
        let mut b = budget(5, 2, 2);
        match axis {
            0 => b.json.max_forward_positions -= 1, 1 => b.json.max_projected_logits -= 1,
            2 => b.json.max_total_mask_node_visits -= 1, 3 => b.native.max_forward_positions -= 1,
            4 => b.native.max_attention_pairs -= 1, 5 => b.native.max_projection_work.dot_products -= 1,
            _ => b.native.max_projection_work.multiply_accumulates -= 1,
        }
        let mut d = Script::new(vec![0.0, 1.0]);
        assert!(drive_layer_major(GROUP, &[1; 5], &p, &v, &options(2), b,
            &mut d, limits(4), Ok::<_, Int4JsonError>).is_err());
        assert!(d.seen.is_empty()); assert_eq!(d.prompt_calls, 0); assert_eq!(d.heads, 0);
    }
}

#[test]
fn prompt_failure_cancellation_and_invalid_work_never_retry_or_return_partial_json() {
    let p = program(r#"{"type":"boolean"}"#); let v = table(&[b"", b"true"]);
    for axis in 0..5 {
        let mut d = Script::new(vec![0.0, 1.0]);
        match axis { 0 => d.fail_append = true, 1 => d.control.prefill = Some(2),
            2 => d.control.step = Some(1), 3 => d.corrupt_work = true, _ => d.row[0] = f32::NAN }
        let error = drive_layer_major(GROUP, &[1; 5], &p, &v, &options(2), budget(5, 2, 2),
            &mut d, limits(4), Ok::<_, Int4JsonError>).unwrap_err();
        assert!(d.aborted); assert_eq!(d.prompt_calls, 1);
        if axis <= 1 { assert_eq!(d.heads, 0); }
        if axis == 1 { assert_eq!(error.cancellation(), Some(DecodeCancellationKind::CostBudget)); }
        if axis == 2 { assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline)); }
    }
}

#[test]
fn semantic_finalizer_refusal_and_its_final_checkpoint_abort_the_same_driver() {
    let p = program(r#"{"type":"boolean"}"#); let v = table(&[b"", b"true"]);
    for cancelled in [false, true] {
        let mut d = Script::new(vec![0.0, 1.0]);
        if cancelled { d.control.step = Some(2); }
        let mut finalized = false;
        let result: Result<(), Int4JsonError> = drive_layer_major(GROUP, &[1; 5], &p, &v, &options(2),
            budget(5, 2, 2), &mut d, limits(4), |run| {
                finalized = true;
                assert_eq!(run.output.json, "true"); assert_eq!(run.output.token_ids, [1, 0]);
                if cancelled { Ok(()) } else { Err(JsonDecodeError::IndependentValidation.into()) }
            });
        assert!(finalized); assert!(d.aborted); assert_eq!(d.prompt_calls, 1); assert_eq!(d.heads, 2);
        let error = result.unwrap_err();
        if cancelled { assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline)); }
    }
}
