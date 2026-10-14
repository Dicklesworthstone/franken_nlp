use super::*;
use std::io::Cursor;

#[derive(Default)]
struct Control { calls: usize, stop_at: Option<usize> }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.calls += 1;
        (Some(self.calls) == self.stop_at).then_some(DecodeCancellationKind::Deadline)
    }
}
fn pin(width: usize) -> Int4SnapshotPin { Int4SnapshotPin { group_width: width, model_content_sha256: [0xcd; 32] } }
fn budget(width: usize) -> Int4SnapshotBudget {
    let need = Int4SnapshotRequirements::for_group_width(width).unwrap();
    Int4SnapshotBudget { max_weight_bytes: need.weight_payload_bytes, max_scratch_bytes: need.scratch_bytes,
        max_io_bytes: need.read_io_bytes().unwrap() }
}
fn matrix(n: usize, k: usize, width: usize) -> GroupedInt4Matrix {
    let need = Int4ConversionRequirement::for_shape(n, k, width).unwrap();
    let values: Vec<_> = (0..n * k).map(|i| Bf16::from_f32((i % 29) as f32 / 4.0 - 3.5)).collect();
    GroupedInt4Matrix::from_bf16(n, k, width, &values, Int4ConversionBudget {
        max_payload_bytes: need.peak_payload_bytes().unwrap(), max_source_values: need.source_values, max_groups: need.groups,
    }, &mut Control::default()).unwrap()
}

#[test]
fn all_group_partitions_have_exact_non_overlapping_payload_and_header_geometry() {
    for width in 1..=MAX_MODEL_K {
        let need = Int4SnapshotRequirements::for_group_width(width).unwrap();
        let scales: u64 = PROJECTION_SHAPES.iter().map(|&(n, k)| (n * k.div_ceil(width) * 4) as u64).sum::<u64>() * 22
            + (V * H.div_ceil(width) * 4) as u64;
        let packed = PROJECTION_SHAPES.iter().map(|&(n, k)| (n * k / 2) as u64).sum::<u64>() * 22 + (V * H / 2) as u64;
        assert_eq!(need.weight_payload_bytes, scales + packed + ((V * H + 45 * H) * 2) as u64);
        assert_eq!(need.file_bytes, need.weight_payload_bytes + 128);
        let header = encode_header(pin(width), &"ab".repeat(32), need).unwrap();
        assert_eq!(decode_header(&header, pin(width), need).unwrap(), "ab".repeat(32));
        assert_eq!(&header[..8], b"FNLPI4S\0");
    }
}

#[test]
fn every_header_byte_is_fixed_pinned_or_bound_by_the_model_digest() {
    let pin = pin(128); let need = Int4SnapshotRequirements::for_group_width(128).unwrap();
    let header = encode_header(pin, &"ab".repeat(32), need).unwrap();
    for offset in 0..SNAPSHOT_HEADER_BYTES {
        let mut changed = header; changed[offset] ^= 1;
        if (32..64).contains(&offset) {
            // Source provenance is bound by the full model digest, not by the
            // header parsing step alone. No executable model is returned here.
            let root = decode_header(&changed, pin, need).unwrap();
            assert_ne!(identity_prefix(&root, 128).finalize(), identity_prefix(&"ab".repeat(32), 128).finalize());
        } else { assert!(decode_header(&changed, pin, need).is_err(), "header byte {offset}"); }
    }
    for root in ["a".repeat(63), "AB".repeat(32), "gh".repeat(32)] {
        assert!(encode_header(pin, &root, need).is_err());
    }
    assert!(decode_header(&header, Int4SnapshotPin { group_width: 32, ..pin }, need).is_err());
    assert_eq!(verify_content([0xff; 32], pin), Err(Int4SnapshotError::Integrity));
    verify_content(pin.model_content_sha256, pin).unwrap();
}

#[test]
fn every_budget_axis_and_invalid_pin_refuse_before_any_source_read() {
    struct NoRead;
    impl Read for NoRead { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("source read before admission"); } }
    for axis in 0..3 {
        let mut cap = budget(128);
        match axis { 0 => cap.max_weight_bytes -= 1, 1 => cap.max_scratch_bytes -= 1, _ => cap.max_io_bytes -= 1 }
        assert!(matches!(Int4ModelWeights::read_snapshot(&mut NoRead, pin(128), cap, &mut Control::default()),
            Err(Int4SnapshotError::Budget)));
    }
    for width in [0, MAX_MODEL_K + 1, usize::MAX] {
        assert!(Int4ModelWeights::read_snapshot(&mut NoRead, pin(width), budget(128), &mut Control::default()).is_err());
    }
    let mut stop = Control { calls: 0, stop_at: Some(1) };
    assert!(matches!(Int4ModelWeights::read_snapshot(&mut NoRead, pin(128), budget(128), &mut stop),
        Err(Int4SnapshotError::Cancelled(DecodeCancellationKind::Deadline))));
}

#[test]
fn malformed_or_truncated_headers_fail_without_body_allocation_or_read() {
    let need = Int4SnapshotRequirements::for_group_width(128).unwrap();
    let header = encode_header(pin(128), &"ab".repeat(32), need).unwrap();
    for end in 0..128 {
        assert!(Int4ModelWeights::read_snapshot(&mut &header[..end], pin(128), budget(128), &mut Control::default()).is_err());
    }
    let mut changed = header; changed[64] ^= 1;
    let mut stream = Cursor::new(changed);
    assert!(matches!(Int4ModelWeights::read_snapshot(&mut stream, pin(128), budget(128), &mut Control::default()),
        Err(Int4SnapshotError::Integrity)));
    assert_eq!(stream.position(), 128);
}

#[test]
fn bf16_wire_roundtrip_keeps_every_finite_bit_pattern_including_signed_zero() {
    let values: Vec<_> = (0..=u16::MAX).map(Bf16::from_bits).filter(|v| v.to_f32().is_finite()).collect();
    let mut scratch = buffer().unwrap(); let mut bytes = Vec::new();
    write_bf16(&mut bytes, &values, &mut scratch, &mut Control::default()).unwrap();
    let restored = read_bf16_values(&mut bytes.as_slice(), values.len(), &mut scratch, &mut Control::default()).unwrap();
    assert_eq!(values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), restored.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    for bits in [0x7f80_u16, 0xff80, 0x7fc1, 0xffc1] {
        assert_eq!(read_bf16_values(&mut &bits.to_le_bytes()[..], 1, &mut scratch, &mut Control::default()).err(), Some(Int4SnapshotError::Values));
    }
}

#[test]
fn matrix_wire_roundtrip_has_no_dense_expansion_and_keeps_exact_digest_and_projection() {
    use crate::native_engine::portable_int8::{ActivationBuffer, LinearRows, ProjectionLedger};
    for k in [1, 3, 33, 3072, MAX_MODEL_K] {
        for width in [1, 3, 32, 128, MAX_MODEL_K] {
            let a = matrix(2, k, width); let mut bytes = Vec::new(); let mut scratch = buffer().unwrap();
            write_matrix(&mut bytes, &a, &mut scratch, &mut Control::default()).unwrap();
            assert_eq!(bytes.len(), a.packed_bytes().len() + 4 * a.group_scales().len());
            let b = read_matrix(&mut bytes.as_slice(), 2, k, width, &mut scratch, &mut Control::default()).unwrap();
            assert_eq!(a.content_sha256(), b.content_sha256());
            let mut activation = ActivationBuffer::try_new(k).unwrap(); activation.encode_f32(&vec![1.0; k]).unwrap();
            let run = |m: &GroupedInt4Matrix| {
                let v = m.as_linear(); let mut out = [0.0; 2];
                v.project_f32_into(&activation, LinearRows::All, &mut out, &mut vec![0; v.scratch_len()],
                    &mut ProjectionLedger::new(v.planned_work(LinearRows::All).unwrap()), &mut Control::default()).unwrap();
                out.map(f32::to_bits)
            };
            assert_eq!(run(&a), run(&b));
        }
    }
}

#[test]
fn matrix_wire_rejects_truncation_negative_eight_and_invalid_scale_encodings() {
    let a = matrix(2, 3, 2); let mut bytes = Vec::new(); let mut scratch = buffer().unwrap();
    write_matrix(&mut bytes, &a, &mut scratch, &mut Control::default()).unwrap();
    for end in 0..bytes.len() {
        assert!(read_matrix(&mut &bytes[..end], 2, 3, 2, &mut scratch, &mut Control::default()).is_err());
    }
    for bad in [8_u8, 0x80, 0x88] {
        let mut changed = bytes.clone(); changed[0] = bad;
        assert!(read_matrix(&mut changed.as_slice(), 2, 3, 2, &mut scratch, &mut Control::default()).is_err());
    }
    for bad in [0.0_f32, -0.0, -1.0, f32::NAN, f32::INFINITY] {
        let mut changed = bytes.clone(); changed[3..7].copy_from_slice(&bad.to_le_bytes());
        assert!(read_matrix(&mut changed.as_slice(), 2, 3, 2, &mut scratch, &mut Control::default()).is_err());
    }
}

#[test]
fn short_and_interrupted_io_preserve_bytes_and_can_be_cancelled() {
    struct Short { inner: Cursor<Vec<u8>>, calls: usize }
    impl Read for Short {
        fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
            self.calls += 1; if self.calls % 3 == 0 { return Err(io::ErrorKind::Interrupted.into()); }
            let n = bytes.len().min(3); self.inner.read(&mut bytes[..n])
        }
    }
    impl Write for Short {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.calls += 1; if self.calls % 3 == 0 { return Err(io::ErrorKind::Interrupted.into()); }
            self.inner.write(&bytes[..bytes.len().min(3)])
        }
        fn flush(&mut self) -> io::Result<()> { Ok(()) }
    }
    let expected: Vec<_> = (0..=255).collect();
    let mut short = Short { inner: Cursor::new(Vec::new()), calls: 0 };
    write_bytes(&mut short, &expected, &mut Control::default()).unwrap();
    short.inner.set_position(0); let mut actual = vec![0; expected.len()];
    read_bytes(&mut short, &mut actual, &mut Control::default()).unwrap();
    assert_eq!(actual, expected);
    let mut stop = Control { calls: 0, stop_at: Some(3) };
    assert_eq!(write_bytes(&mut short, &[1; 64], &mut stop), Err(Int4SnapshotError::Cancelled(DecodeCancellationKind::Deadline)));
}

#[test]
fn exact_eof_and_write_ceiling_never_scan_or_emit_unbounded_extra_bytes() {
    let mut stream = Cursor::new(vec![9; 100]);
    assert_eq!(require_eof(&mut stream, &mut Control::default()), Err(Int4SnapshotError::Format));
    assert_eq!(stream.position(), 1);
    require_eof(&mut &[][..], &mut Control::default()).unwrap();
    let mut out = Vec::new();
    let mut bounded = WireWriter { target: &mut out, remaining: 3 };
    write_bytes(&mut bounded, &[1, 2, 3], &mut Control::default()).unwrap();
    assert_eq!(bounded.remaining, 0);
    assert_eq!(write_bytes(&mut bounded, &[4], &mut Control::default()), Err(Int4SnapshotError::Io));
    assert_eq!(out, [1, 2, 3]);
}

#[test]
#[ignore = "requires controller-admitted source model, snapshot destination and disk space"]
fn full_model_snapshot_reloads_same_identity_and_logits_without_source_access() {
    use crate::{artifact::converter::{ConversionSourceManifest, prepare_nanbeige42_input}, native_engine::{
        strict_int4::{StrictInt4Engine, Int4MemoryRequirement, Int4MemoryBudget, Int4RunBudget, Int4Work},
        portable_int8::LinearRows,
    }};
    let dir = std::env::var("FNLP_INT4_TEST_SOURCE_DIR").expect("explicit source directory");
    let manifest = std::env::var("FNLP_INT4_TEST_SOURCE_MANIFEST").expect("explicit pinned manifest");
    let path = std::env::var("FNLP_INT4_TEST_SNAPSHOT").expect("explicit new snapshot path; preserved after test");
    let source = prepare_nanbeige42_input(dir, ConversionSourceManifest::load_pinned(manifest).unwrap(), false, 65536).unwrap();
    let need = Int4ModelLoadRequirement::for_group_width(128).unwrap();
    let model = Int4ModelWeights::from_prepared(&source, 128, Int4ModelLoadBudget {
        max_payload_bytes: need.peak_payload_bytes().unwrap(), max_source_values: need.source_values,
        max_groups: need.groups, max_shard_revalidation_bytes: need.shard_revalidation_bytes,
    }, &mut Control::default()).unwrap();
    let identity = model.identity().clone();
    let run = |model: &Int4ModelWeights| {
        let memory = Int4MemoryRequirement::for_context(4, 128).unwrap();
        let mut engine = StrictInt4Engine::new(model, 4, Int4MemoryBudget { max_kv_bytes: memory.kv_bytes,
            max_rope_bytes: memory.rope_bytes, max_scratch_payload_bytes: memory.scratch_payload_bound }).unwrap();
        let mut control = Control::default();
        let mut session = engine.session(Int4RunBudget::exact(Int4Work::for_sequence(128, 0, 1, 3).unwrap()), &mut control).unwrap();
        session.append(1).unwrap(); session.logits(LinearRows::Selected(&[1, 2, 3])).unwrap()
    };
    let expected = run(&model);
    let mut file = std::fs::OpenOptions::new().write(true).create_new(true).open(&path).unwrap();
    let actual_pin = model.write_snapshot(&mut file, budget(128), &mut Control::default()).unwrap();
    file.sync_all().unwrap(); drop(file); drop(model); drop(source);
    let mut file = std::fs::File::open(&path).unwrap();
    let restored = Int4ModelWeights::read_snapshot(&mut file, actual_pin, budget(128), &mut Control::default()).unwrap();
    assert_eq!(restored.identity(), &identity);
    assert_eq!(run(&restored).iter().map(|v| v.to_bits()).collect::<Vec<_>>(), expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
}
