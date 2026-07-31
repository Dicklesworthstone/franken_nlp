use std::collections::BTreeSet;
use std::path::PathBuf;

use franken_nlp::artifact::format::{framed_sha256, ArchTarget};
use franken_nlp::artifact::reader::{FnlpqArtifact, FnlpqReadError};

const GOLDEN: &str = include_str!("fixtures/fnlpq/golden/tiny_v1.hex");
const HOSTILE_CASES: &str = include_str!("fixtures/fnlpq/hostile_cases.json");

const CORPUS_IDS: &[&str] = &[
    "FNLPG-H001-TRUNCATED-PRELUDE",
    "FNLPG-H002-HEADER-LEN-OVERFLOW",
    "FNLPG-H003-HEADER-OVER-CAP",
    "FNLPG-H004-WRONG-FILE-LEN",
    "FNLPG-H005-UNKNOWN-REQUIRED-FLAG",
    "FNLPG-H006-HEADER-SHA-MISMATCH",
    "FNLPG-H007-DUPLICATE-JSON-KEY",
    "FNLPG-H008-UNICODE-ESCAPE",
    "FNLPG-H009-SECTION-TABLE-OVERLAP",
    "FNLPG-H010-DUPLICATE-REQUIRED-SECTION-KIND",
    "FNLPG-H011-UNKNOWN-SECTION-KIND",
    "FNLPG-H012-NONZERO-ALIGNMENT-GAP",
    "FNLPG-H013-RANGE-INTO-HEADER",
    "FNLPG-H014-DUPLICATE-TENSOR-NAME",
    "FNLPG-H015-NAN-SCALE",
    "FNLPG-H016-PACKING-ARCH-MISMATCH",
    "FNLPG-H017-DUPLICATE-SECTION-NAME",
    "FNLPG-H018-INFINITE-SCALE",
];

#[test]
fn owned_reader_accepts_writer_golden_and_refuses_wrong_target() {
    let artifact = FnlpqArtifact::from_bytes(golden_bytes()).expect("writer golden must parse");
    assert_eq!(artifact.model_id(), "FnlpqTinyGolden");
    assert_eq!(
        artifact.revision(),
        "f56ec5a9650268aa098496734743c25ea778bd2d"
    );
    assert_eq!(artifact.sections().len(), 9);
    assert_eq!(artifact.tensors().len(), 1);
    assert_eq!(
        artifact.select_packing(ArchTarget::Generic).unwrap().id,
        "generic"
    );
    assert!(matches!(
        artifact.select_packing(ArchTarget::Aarch64Sdot),
        Err(FnlpqReadError::MissingPackingDerivation { .. })
    ));
}

#[test]
fn corpus_index_has_a_dedicated_fixture_for_every_oq31_hostile_id() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/corpus/fnlpq");
    let named: BTreeSet<_> = std::fs::read_dir(&root)
        .expect("committed fnlpq hostile corpus directory")
        .map(|entry| entry.expect("corpus entry").path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "json")
        })
        .map(|path| std::fs::read_to_string(path).expect("corpus fixture text"))
        .collect();
    assert_eq!(CORPUS_IDS.len(), 18);
    for id in CORPUS_IDS {
        assert!(HOSTILE_CASES.contains(id), "OQ-31 index omitted {id}");
        assert!(
            named.iter().any(|fixture| fixture.contains(id)),
            "missing dedicated hostile corpus fixture {id}"
        );
    }
}

#[test]
fn fixed_preheader_directory_and_scale_mutations_reject_without_panicking() {
    let valid = golden_bytes();
    let directory_start = directory_start(&valid);
    let directory_end = directory_start + 9 * 80;
    let first_section_offset = u64_at(&valid, directory_start + 16) as usize;

    let mut cases = Vec::new();
    cases.push(("FNLPG-H001-TRUNCATED-PRELUDE", valid[..79].to_vec()));

    let mut header_overflow = valid.clone();
    header_overflow[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
    cases.push(("FNLPG-H002-HEADER-LEN-OVERFLOW", header_overflow));

    let mut header_cap = valid.clone();
    header_cap[16..24].copy_from_slice(&(1_048_577_u64).to_le_bytes());
    cases.push(("FNLPG-H003-HEADER-OVER-CAP", header_cap));

    let mut wrong_file_len = valid.clone();
    let mismatched_len = wrong_file_len.len() as u64 + 1;
    wrong_file_len[40..48].copy_from_slice(&mismatched_len.to_le_bytes());
    cases.push(("FNLPG-H004-WRONG-FILE-LEN", wrong_file_len));

    let mut unknown_flags = valid.clone();
    unknown_flags[12..16].copy_from_slice(&1_u32.to_le_bytes());
    cases.push(("FNLPG-H005-UNKNOWN-REQUIRED-FLAG", unknown_flags));

    let mut header_flip = valid.clone();
    header_flip[80] ^= 1;
    cases.push(("FNLPG-H006-HEADER-SHA-MISMATCH", header_flip));

    let mut overlap = valid.clone();
    overlap[directory_start + 80 + 16..directory_start + 80 + 24]
        .copy_from_slice(&(first_section_offset as u64).to_le_bytes());
    cases.push(("FNLPG-H009-SECTION-TABLE-OVERLAP", overlap));

    let mut unknown_kind = valid.clone();
    unknown_kind[directory_start..directory_start + 4].copy_from_slice(&99_u32.to_le_bytes());
    cases.push(("FNLPG-H011-UNKNOWN-SECTION-KIND", unknown_kind));

    let mut gap_fraud = valid.clone();
    assert!(
        first_section_offset > directory_end,
        "golden carries an alignment gap"
    );
    gap_fraud[directory_end] = 1;
    cases.push(("FNLPG-H012-NONZERO-ALIGNMENT-GAP", gap_fraud));

    let mut into_header = valid.clone();
    into_header[directory_start + 16..directory_start + 24].copy_from_slice(&80_u64.to_le_bytes());
    cases.push(("FNLPG-H013-RANGE-INTO-HEADER", into_header));

    let mut nan_scale = valid.clone();
    let scale_directory = directory_start + 80;
    let scale_offset = u64_at(&nan_scale, scale_directory + 16) as usize;
    let scale_len = u64_at(&nan_scale, scale_directory + 24) as usize;
    nan_scale[scale_offset..scale_offset + 4].copy_from_slice(&f32::NAN.to_bits().to_le_bytes());
    refresh_section_digest(
        &mut nan_scale,
        scale_directory,
        b"generic-scales",
        scale_offset,
        scale_len,
    );
    cases.push(("FNLPG-H015-NAN-SCALE", nan_scale));

    let mut inf_scale = valid.clone();
    inf_scale[scale_offset..scale_offset + 4]
        .copy_from_slice(&f32::INFINITY.to_bits().to_le_bytes());
    refresh_section_digest(
        &mut inf_scale,
        scale_directory,
        b"generic-scales",
        scale_offset,
        scale_len,
    );
    cases.push(("FNLPG-H018-INFINITE-SCALE", inf_scale));

    for (id, mutation) in &cases {
        let error = FnlpqArtifact::from_bytes(mutation.clone()).expect_err(id);
        eprintln!("FNLPG case={id} RESULT=REJECT detail={error}");
    }
    eprintln!(
        "FUZZ_SUMMARY cases={} rejects={} zero_panics=true",
        cases.len(),
        cases.len()
    );
}

#[test]
fn sampled_single_bit_mutations_are_all_typed_rejections() {
    let valid = golden_bytes();
    let mut cases = 0_usize;
    for offset in (0..valid.len()).step_by(97) {
        let mut mutation = valid.clone();
        mutation[offset] ^= 0x01;
        assert!(
            FnlpqArtifact::from_bytes(mutation).is_err(),
            "bit mutation at offset {offset} unexpectedly parsed"
        );
        cases += 1;
    }
    eprintln!("FUZZ_SUMMARY cases={cases} rejects={cases} zero_panics=true");
}

fn golden_bytes() -> Vec<u8> {
    let digits: Vec<_> = GOLDEN
        .bytes()
        .filter(|byte| !byte.is_ascii_whitespace())
        .collect();
    digits
        .chunks_exact(2)
        .map(|pair| (nibble(pair[0]) << 4) | nibble(pair[1]))
        .collect()
}

fn directory_start(bytes: &[u8]) -> usize {
    80 + u64_at(bytes, 16) as usize
}

fn u64_at(bytes: &[u8], start: usize) -> u64 {
    u64::from_le_bytes(
        bytes[start..start + 8]
            .try_into()
            .expect("fixed fixture range"),
    )
}

fn refresh_section_digest(
    bytes: &mut [u8],
    directory_offset: usize,
    name: &[u8],
    payload_offset: usize,
    payload_len: usize,
) {
    let digest = framed_sha256(
        "fnlpq-section-v1",
        &[name, &bytes[payload_offset..payload_offset + payload_len]],
    )
    .expect("fixed digest tag");
    bytes[directory_offset + 48..directory_offset + 80].copy_from_slice(&digest);
}

fn nibble(byte: u8) -> u8 {
    match byte {
        b'0'..=b'9' => byte - b'0',
        b'a'..=b'f' => byte - b'a' + 10,
        _ => panic!("invalid fixture hex byte {byte:?}"),
    }
}
