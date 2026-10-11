//! ARM arithmetic fixtures. The safe A3 body is exercised on every test host;
//! ISA tests explicitly skip where the real OS does not advertise support.
use super::*;

fn oracle(input: &[i8], weights: &[i8]) -> i32 {
    let sum: i64 = input.iter().zip(weights)
        .map(|(&x, &w)| i64::from(x) * i64::from(w)).sum();
    i32::try_from(sum).unwrap()
}
fn check_safe(inputs: &[&[i8]], weights: &[i8]) {
    let mut output = vec![999; inputs.len()];
    autovec::dot_rows_into(inputs, weights, &mut output).unwrap();
    for row in 0..inputs.len() { assert_eq!(output[row], oracle(inputs[row], weights)); }
}

#[test]
fn safe_autovec_covers_every_byte_pair_with_independent_row_signs() {
    for x in -128_i16..=127 {
        let a = [x as i8; 16]; let b = [!(x as i8); 16];
        for w in -128_i16..=127 { check_safe(&[&a, &b, &[-128; 16], &[127; 16]], &[w as i8; 16]); }
    }
}
#[test]
fn safe_autovec_covers_all_alignments_cohort_sizes_and_k_tails() {
    let storage: Vec<Vec<i8>> = (0..4).map(|row| (0..MAX_MODEL_K + 16)
        .map(|i| ((i * 53 + row * 73) % 256) as u8 as i8).collect()).collect();
    let w: Vec<i8> = (0..MAX_MODEL_K + 16).map(|i| ((i * 29 + 128) % 256) as u8 as i8).collect();
    for m in 0..=4 {
        for offset in 0..16 {
            for k in (0..48).chain(int8::MODEL_KS.into_iter().flat_map(|base| (0..16).map(move |tail| base - tail))) {
                let inputs: Vec<&[i8]> = storage[..m].iter().enumerate().map(|(row, bytes)| {
                    let start = (offset + row * 3) % 16; &bytes[start..start + k]
                }).collect();
                check_safe(&inputs, &w[15 - offset..15 - offset + k]);
            }
        }
    }
}
#[test]
fn safe_autovec_extreme_lanes_do_not_saturate_or_mix_rows() {
    for k in int8::MODEL_KS {
        let a = vec![-128; k]; let b = vec![127; k];
        check_safe(&[&a, &b, &a, &b], &a);
        for lane in 0..33 {
            let mut impulse = vec![0; k]; impulse[lane] = -128;
            check_safe(&[&impulse, &a], &a);
        }
    }
}
#[test]
fn safe_autovec_rejects_every_bad_row_before_writing_any_output() {
    let good = [1; 32]; let short = [1; 31];
    let mut output = [123; 4];
    assert_eq!(autovec::dot_rows_into(&[&good, &good, &good, &short], &good, &mut output),
        Err(CpuKernelError::LengthMismatch));
    assert_eq!(output, [123; 4]);
    assert_eq!(autovec::dot_rows_into(&[&good; 4], &good, &mut output[..3]), Err(CpuKernelError::Shape));
    assert_eq!(output, [123; 4]);
    let long = vec![0; MAX_MODEL_K + 1];
    assert_eq!(autovec::dot_rows_into(&[&long], &long, &mut output[..1]), Err(CpuKernelError::ReductionTooLong));
    assert_eq!(output, [123; 4]);
    let mut too_many = [123; 5];
    assert_eq!(autovec::dot_rows_into(&[&good; 5], &good, &mut too_many), Err(CpuKernelError::Shape));
    assert_eq!(too_many, [123; 5]);
}
#[test]
fn safe_autovec_allows_readonly_aliases_and_leaves_outside_output_unchanged() {
    let bytes = [-128; 33]; let mut output = [123; 4];
    autovec::dot_rows_into(&[&bytes, &bytes], &bytes, &mut output[1..3]).unwrap();
    assert_eq!(output, [123, 33 * 16384, 33 * 16384, 123]);
    assert_eq!(bytes, [-128; 33]);
    autovec::dot_rows_into(&[], &bytes, &mut []).unwrap();
    autovec::dot_rows_into(&[&[], &[]], &[], &mut output[1..3]).unwrap();
    assert_eq!(output, [123, 0, 0, 123]);
}

fn has_dotprod() -> bool {
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    { return std::arch::is_aarch64_feature_detected!("neon") && std::arch::is_aarch64_feature_detected!("dotprod"); }
    #[cfg(not(all(target_arch = "aarch64", target_endian = "little")))]
    { false }
}
#[test]
fn actual_cpu_and_compilation_target_determine_arm_handle_availability() {
    for (kind, expected) in [(Int8KernelKind::Aarch64Dotprod, has_dotprod()),
        (Int8KernelKind::Aarch64Autovec, cfg!(target_arch = "aarch64"))] {
        match Int8Kernel::detect(kind) {
            Ok(kernel) => { assert!(expected); assert_eq!(kernel.kind(), kind); }
            Err(CpuKernelError::UnsupportedIsa) => assert!(!expected),
            Err(error) => panic!("wrong refusal for {kind}: {error}"),
        }
    }
}
#[test]
fn sdot_lane_impulses_and_all_tail_lengths_match_the_i64_oracle() {
    let kernel = match Int8Kernel::detect(Int8KernelKind::Aarch64Dotprod) {
        Ok(kernel) => kernel,
        Err(CpuKernelError::UnsupportedIsa) => {
            eprintln!("ARM_INT8 tier=a2_dotprod RESULT=SKIPPED_UNSUPPORTED_ISA"); return;
        }
        Err(error) => panic!("unexpected SDOT error: {error}"),
    };
    // Every index includes vector-lane boundaries, the 16-byte boundary and
    // scalar cleanup. Row-dependent signs catch accidental across-row sums.
    for k in 1..=65 {
        for index in 0..k {
            let mut a = vec![0; k]; a[index] = -128;
            let b = vec![127; k]; let w = vec![-128; k];
            let mut output = [123; 2];
            kernel.dot_rows_into(&[&a, &b], &w, &mut output).unwrap();
            assert_eq!(output, [16384, -16256 * k as i32]);
            assert_eq!(kernel.dot(&a, &w).unwrap(), oracle(&a, &w));
        }
    }
}
