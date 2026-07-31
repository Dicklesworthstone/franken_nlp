use std::collections::BTreeMap;

use franken_nlp::artifact::safetensors::{
    CheckedShard, RowPanel, SafetensorDtype, SafetensorsError, SourceDigest, TensorExpectation,
    diff_census_entries, validate_index_mapping, verify_source_bytes,
};

const DUPLICATE_TENSOR_HEADER: &[u8] =
    include_bytes!("corpus/safetensors/duplicate_tensor_header.json");
const UNKNOWN_DTYPE_HEADER: &[u8] = include_bytes!("corpus/safetensors/unknown_dtype_header.json");

#[test]
fn hostile_headers_reject_before_any_range_is_exposed() {
    let cap = (franken_nlp::artifact::safetensors::MAX_HEADER_BYTES + 1).to_le_bytes();
    assert!(matches!(
        CheckedShard::from_bytes("cap.safetensors", &cap),
        Err(SafetensorsError::HeaderTooLarge { .. })
    ));

    assert!(matches!(
        CheckedShard::from_bytes(
            "duplicate.safetensors",
            &synthetic(DUPLICATE_TENSOR_HEADER, &[0, 0]),
        ),
        Err(SafetensorsError::DuplicateJsonKey { path, .. }) if path == "/weight"
    ));
    assert!(matches!(
        CheckedShard::from_bytes(
            "unknown.safetensors",
            &synthetic(UNKNOWN_DTYPE_HEADER, &[0, 0]),
        ),
        Err(SafetensorsError::UnknownDtype { .. })
    ));
    assert!(matches!(
        shard(
            r#"{"overflow":{"dtype":"BF16","shape":[18446744073709551615,2],"data_offsets":[0,2]}}"#,
            &[0, 0],
        ),
        Err(SafetensorsError::ShapeProductOverflow { .. })
    ));
}

#[test]
fn range_invariants_reject_overlap_gap_length_and_bounds() {
    assert!(matches!(
        shard(
            r#"{"a":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"b":{"dtype":"BF16","shape":[1],"data_offsets":[1,3]}}"#,
            &[0, 0, 0],
        ),
        Err(SafetensorsError::RangeOverlap { tensor, offset: 1, prior_end: 2, .. }) if tensor == "b"
    ));
    assert!(matches!(
        shard(
            r#"{"a":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"b":{"dtype":"BF16","shape":[1],"data_offsets":[4,6]}}"#,
            &[0, 0, 0, 0, 0, 0],
        ),
        Err(SafetensorsError::RangeGap { tensor, expected_offset: 2, actual_offset: 4, .. }) if tensor == "b"
    ));
    assert!(matches!(
        shard(
            r#"{"a":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#,
            &[0, 0],
        ),
        Err(SafetensorsError::RangeOutOfBounds {
            end: 4,
            data_len: 2,
            ..
        })
    ));
    assert!(matches!(
        shard(
            r#"{"a":{"dtype":"BF16","shape":[1],"data_offsets":[0,3]}}"#,
            &[0, 0, 0],
        ),
        Err(SafetensorsError::RangeLengthMismatch {
            expected: 2,
            actual: 3,
            ..
        })
    ));
}

#[test]
fn valid_synthetic_ranges_match_the_whole_blob_baseline() {
    let header = r#"{"rows":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]},"tail":{"dtype":"U8","shape":[3],"data_offsets":[8,11]}}"#;
    let payload = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    let blob = synthetic(header.as_bytes(), &payload);
    let checked =
        CheckedShard::from_bytes("tiny.safetensors", &blob).expect("valid synthetic shard");

    let rows = checked
        .range_for(
            "rows",
            RowPanel::Rows {
                start_row: 1,
                row_count: 1,
            },
        )
        .expect("bounded row panel");
    assert_eq!(
        &blob[rows.file_offset as usize..(rows.file_offset + rows.len) as usize],
        &[5, 6, 7, 8]
    );
    let whole_tail = checked
        .range_for("tail", RowPanel::WholeTensor)
        .expect("whole tensor remains one bounded range");
    assert_eq!(
        &blob[whole_tail.file_offset as usize..(whole_tail.file_offset + whole_tail.len) as usize],
        &[9, 10, 11]
    );
    assert!(matches!(
        checked.range_for(
            "rows",
            RowPanel::Rows {
                start_row: 2,
                row_count: 1
            }
        ),
        Err(SafetensorsError::RowPanelOutOfBounds { .. })
    ));
}

#[test]
fn index_mapping_and_digest_refusal_are_typed() {
    let left_header = r#"{"other":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#;
    let right_header = r#"{"weight":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#;
    let left = CheckedShard::from_bytes(
        "left.safetensors",
        &synthetic(left_header.as_bytes(), &[0, 0]),
    )
    .expect("left header");
    let right = CheckedShard::from_bytes(
        "right.safetensors",
        &synthetic(right_header.as_bytes(), &[0, 0]),
    )
    .expect("right header");
    let headers = BTreeMap::from([
        ("left.safetensors".to_owned(), left),
        ("right.safetensors".to_owned(), right),
    ]);
    assert!(matches!(
        validate_index_mapping(
            "index.json",
            br#"{"weight_map":{"weight":"left.safetensors"}}"#,
            &headers,
        ),
        Err(SafetensorsError::IndexShardMismatch { .. })
    ));

    let source = SourceDigest::new(
        "fixture.safetensors",
        2,
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    )
    .expect("valid expected digest grammar");
    assert!(matches!(
        verify_source_bytes(&source, &[0, 1]),
        Err(SafetensorsError::SourceDigest { expected, actual, .. })
            if expected != actual
    ));

    let wrong_length = SourceDigest::new(
        "length-fixture.safetensors",
        3,
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    )
    .expect("valid expected digest grammar");
    assert!(matches!(
        verify_source_bytes(&wrong_length, &[0, 1]),
        Err(SafetensorsError::SourceLength {
            expected: 3,
            actual: 2,
            ..
        })
    ));
}

#[test]
fn mutation_samples_are_total_and_census_diff_is_exact() {
    for seed in 0_u8..=255 {
        let bytes = [seed; 16];
        let _ = CheckedShard::from_bytes("mutation.safetensors", &bytes);
    }

    let header = r#"{"weight":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#;
    let checked =
        CheckedShard::from_bytes("census.safetensors", &synthetic(header.as_bytes(), &[0, 0]))
            .expect("valid fixture");
    let actual = checked.tensor("weight").expect("weight exists");
    assert_eq!(actual.dtype, SafetensorDtype::Bf16);
    assert_eq!(actual.shape, vec![1]);
    assert_eq!(actual.len, 2);

    let expected = TensorExpectation {
        name: "weight".to_owned(),
        dtype: SafetensorDtype::Bf16,
        shape: vec![1],
        len: 2,
    };
    let diff = diff_census_entries(&checked.census(), &[expected]).expect("unique expected census");
    assert!(diff.is_match(), "exact census must not drift: {diff:?}");
}

fn shard(header: &str, payload: &[u8]) -> Result<CheckedShard, SafetensorsError> {
    CheckedShard::from_bytes(
        "hostile.safetensors",
        &synthetic(header.as_bytes(), payload),
    )
}

fn synthetic(header: &[u8], payload: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(8 + header.len() + payload.len());
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header);
    bytes.extend_from_slice(payload);
    bytes
}
