use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use franken_nlp::artifact::converter::expected_nanbeige42_census;
use franken_nlp::artifact::format::{
    ArchTarget, CanonicalDtype, FnlpqWriteError, FnlpqWriterInput, PackingSetInput, SectionKind,
    SectionPayload, SectionRange, TensorInput, framed_sha256_hex, logical_model_sha256,
    logical_tensor_sha256, write,
};
use franken_nlp::artifact::reader::{FnlpqArtifact, FnlpqReadError};
use sha2::{Digest, Sha256};

const PINNED_REVISION: &str = "f56ec5a9650268aa098496734743c25ea778bd2d";
const BF16_RECIPE: &str = "bf16-verbatim-v1";
const STRUCTURAL_CENSUS_RECIPE: &str = "census-structure-v1";
const REAL_INT8_GENERIC_BYTES: u64 = 4_690_873_282;
const REAL_INT8_GENERIC_RAW_SHA256: &str =
    "15c57d4db1b635b84b84c73ec12e805fb4c383e99d17070dbbbbe8a8d741a1d7";
const REAL_INT8_GENERIC_SOURCE_ROOT_SHA256: &str =
    "71686a7075712c2b38e80fe387b9f23e3befac6e367bb4cb0d0ba5a90374c7fd";
const REAL_INT8_GENERIC_LOGICAL_MODEL_SHA256: &str =
    "1d24f68a8eecc89fe68450e19ec2c9d2959054f3fac28cf4f12d3fc3dd7daae8";
const REAL_INT8_GENERIC_TOKENIZER_STORED_SHA256: &str =
    "2e2501d35af4ee7434cb82c93f599f86201f6823d2afae9aaf1c6ae63f08dc3d";
const REAL_INT8_GENERIC_LICENSE_BUNDLE_SHA256: &str =
    "123d4a23b8edc328ba8fd52b802993408037891f38484e1a48a4d4022200713b";

/// Executes only when an orchestrator provides the immutable converted
/// artifact through `FNLP_REAL_FNLPQ`.  This is deliberately a reader gate,
/// not a conversion-qualification claim: any checked-reader refusal is logged
/// before the test fails and must be adjudicated as a converter/reader defect.
#[test]
fn real_int8_generic_artifact_checked_reader_gate_when_supplied() {
    let Some(path) = std::env::var_os("FNLP_REAL_FNLPQ") else {
        eprintln!(
            "FNLPQ_REAL_READER RESULT=SKIPPED_NO_REAL_ARTIFACT reason=FNLP_REAL_FNLPQ-not-set"
        );
        return;
    };
    let path = Path::new(&path);
    let metadata = std::fs::metadata(path).expect("real artifact metadata must be readable");
    assert!(metadata.is_file(), "real artifact must be a regular file");
    assert_eq!(
        metadata.len(),
        REAL_INT8_GENERIC_BYTES,
        "real artifact byte count must bind this reader observation"
    );
    eprintln!(
        "FNLPQ_REAL_READER stage=preflight verdict=OBSERVED bytes={} path={}",
        metadata.len(),
        path.display()
    );

    let observed_raw_sha256 = raw_sha256_file(path);
    assert_eq!(
        observed_raw_sha256, REAL_INT8_GENERIC_RAW_SHA256,
        "real artifact raw SHA-256 must equal the supplied conversion output"
    );
    eprintln!(
        "FNLPQ_REAL_READER stage=raw-file-digest verdict=OBSERVED algorithm=sha256-unframed value={observed_raw_sha256}"
    );

    let artifact = match FnlpqArtifact::open_owned(path) {
        Ok(artifact) => artifact,
        Err(error) => {
            eprintln!(
                "FNLPQ_REAL_READER RESULT=REFUSED stage=open_checked_reader error={error} scope=reader-only"
            );
            panic!("checked reader refused real artifact: {error}");
        }
    };
    eprintln!(
        "FNLPQ_REAL_READER stage=open_checked_reader verdict=ACCEPTED model_id={} revision={} recipe_id={}",
        artifact.model_id(),
        artifact.revision(),
        artifact.recipe_id()
    );

    assert_eq!(artifact.prelude().file_len, REAL_INT8_GENERIC_BYTES);
    assert_eq!(artifact.prelude().section_count, 8);
    assert_eq!(artifact.prelude().tensor_count, 201);
    assert_eq!(artifact.model_id(), "Nanbeige4.2-3B");
    assert_eq!(artifact.revision(), PINNED_REVISION);
    assert_eq!(
        artifact.source_root_sha256(),
        REAL_INT8_GENERIC_SOURCE_ROOT_SHA256
    );
    assert_eq!(
        artifact.logical_model_sha256(),
        REAL_INT8_GENERIC_LOGICAL_MODEL_SHA256
    );
    assert_eq!(
        artifact.license_bundle_sha256(),
        REAL_INT8_GENERIC_LICENSE_BUNDLE_SHA256
    );

    let expected = expected_nanbeige42_census();
    assert_eq!(artifact.tensors().len(), expected.len());
    let expected_by_name = expected
        .iter()
        .map(|tensor| (tensor.name.as_str(), tensor))
        .collect::<BTreeMap<_, _>>();
    for (ordinal, tensor) in artifact.tensors().iter().enumerate() {
        let expected = expected_by_name
            .get(tensor.name.as_str())
            .expect("checked reader must refuse any tensor outside the frozen census");
        let expected_shape = expected
            .shape
            .iter()
            .map(|dimension| u32::try_from(*dimension).expect("frozen dimension fits v1"))
            .collect::<Vec<_>>();
        assert_eq!(tensor.canonical_dtype, "bf16", "tensor={}", tensor.name);
        assert_eq!(tensor.shape, expected_shape, "tensor={}", tensor.name);
        eprintln!(
            "FNLPQ_REAL_READER stage=tensor ordinal={ordinal} name={} dtype={} shape={:?} logical_bytes={} quantization={} data_bytes={} scale_bytes={} row_sum_bytes={} logical_sha256={}",
            tensor.name,
            tensor.canonical_dtype,
            tensor.shape,
            tensor.logical_bytes,
            tensor.quantization,
            tensor.data.len,
            tensor.scale.len,
            tensor.row_sum.len,
            tensor.canonical_logical_sha256,
        );
    }
    eprintln!(
        "FNLPQ_REAL_READER stage=census verdict=ACCEPTED tensors={} logical_model_sha256={}",
        artifact.tensors().len(),
        artifact.logical_model_sha256()
    );

    let mut tokenizer = None;
    let mut license = None;
    for section in artifact.sections() {
        let bytes = artifact
            .section_bytes(section.ordinal)
            .expect("checked section bytes remain available");
        assert_eq!(
            u64::try_from(bytes.len()).expect("section byte length fits u64"),
            section.stored_len,
            "section={} must retain its checked length",
            section.name
        );
        eprintln!(
            "FNLPQ_REAL_READER stage=section ordinal={} name={} kind={} bytes={} stored_sha256={} raw_sha256={}",
            section.ordinal,
            section.name,
            section.kind.header_name(),
            section.stored_len,
            hex(&section.stored_sha256),
            hex(&Sha256::digest(bytes)),
        );
        if section.kind == SectionKind::TokenizerModel {
            tokenizer = Some(section);
        }
        if section.kind == SectionKind::LicenseBundle {
            license = Some(section);
        }
    }

    let tokenizer = tokenizer.expect("v1 requires the embedded tokenizer section");
    assert_eq!(
        hex(&tokenizer.stored_sha256),
        REAL_INT8_GENERIC_TOKENIZER_STORED_SHA256
    );
    let license = license.expect("v1 requires the embedded license section");
    let license_bytes = artifact
        .section_bytes(license.ordinal)
        .expect("checked license bytes remain available");
    assert_eq!(
        framed_sha256_hex("fnlpq-license-bundle-v1", &[license_bytes])
            .expect("fixed license identity framing is valid"),
        artifact.license_bundle_sha256()
    );
    eprintln!(
        "FNLPQ_REAL_READER stage=embedded-identities verdict=ACCEPTED tokenizer_stored_sha256={} license_bundle_sha256={}",
        hex(&tokenizer.stored_sha256),
        artifact.license_bundle_sha256()
    );
    eprintln!(
        "FNLPQ_REAL_READER RESULT=CHECKED_ACCEPT scope=reader-only tensors={} sections={} raw_file_sha256={} conversion_qualified=false",
        artifact.tensors().len(),
        artifact.sections().len(),
        observed_raw_sha256
    );
}

#[test]
fn complete_nanbeige_census_loads_and_missing_shape_and_name_drift_refuse() {
    let complete = nanbeige_input();
    let complete_bytes = write(&complete)
        .expect("complete synthetic census artifact writes")
        .bytes;
    let artifact = FnlpqArtifact::from_bytes(complete_bytes)
        .expect("complete exact census must pass before any engine construction");
    assert_eq!(artifact.tensors().len(), 201);
    eprintln!(
        "FNLPQ_LOAD stage=census verdict=PASS tensors={} logical_model_sha256={}",
        artifact.tensors().len(),
        artifact.logical_model_sha256()
    );

    let mut missing = complete.clone();
    let missing_name = missing
        .tensors
        .last()
        .expect("complete census has a final tensor")
        .name
        .clone();
    missing.tensors.pop();
    refresh_logical_model_identity(&mut missing);
    assert_census_refusal(
        write(&missing).expect("missing fixture writes").bytes,
        "missing",
        &missing_name,
        "missing expected tensor",
    );

    let mut wrong_shape = complete.clone();
    let wrong_shape_name = wrong_shape.tensors[0].name.clone();
    wrong_shape.tensors[0].shape[0] += 1;
    refresh_logical_model_identity(&mut wrong_shape);
    assert_census_refusal(
        write(&wrong_shape)
            .expect("wrong-shape fixture writes")
            .bytes,
        "shape",
        &wrong_shape_name,
        "expected dtype=bf16 shape=",
    );

    let mut wrong_dtype = complete.clone();
    let wrong_dtype_name = wrong_dtype.tensors[0].name.clone();
    wrong_dtype.tensors[0].canonical_dtype = CanonicalDtype::F32;
    refresh_logical_model_identity(&mut wrong_dtype);
    assert_census_refusal(
        write(&wrong_dtype)
            .expect("wrong-dtype fixture writes")
            .bytes,
        "dtype",
        &wrong_dtype_name,
        "expected dtype=bf16 shape=",
    );

    let mut renamed = complete;
    let renamed_name = renamed
        .tensors
        .last()
        .expect("complete census has a final tensor")
        .name
        .clone();
    renamed
        .tensors
        .last_mut()
        .expect("complete census has tensors")
        .name = "z.renamed.tensor".to_owned();
    refresh_logical_model_identity(&mut renamed);
    assert_census_refusal(
        write(&renamed).expect("renamed fixture writes").bytes,
        "renamed",
        &renamed_name,
        "missing expected tensor",
    );

    let mut extra = nanbeige_input();
    extra.tensors.push(TensorInput {
        name: "z.extra.tensor".to_owned(),
        canonical_dtype: CanonicalDtype::Bf16,
        shape: vec![1],
        canonical_logical_sha256: logical_tensor_hex(
            "z.extra.tensor",
            &[1],
            STRUCTURAL_CENSUS_RECIPE,
        ),
        quantization: STRUCTURAL_CENSUS_RECIPE.to_owned(),
        data: SectionRange::new("generic-payload", 0, 0),
        scale: SectionRange::new("generic-scales", 0, 0),
        row_sum: SectionRange::new("generic-row-sums", 0, 0),
    });
    refresh_logical_model_identity(&mut extra);
    assert_census_refusal(
        write(&extra).expect("extra fixture writes").bytes,
        "extra",
        "z.extra.tensor",
        "unexpected tensor absent from the frozen 201-name census",
    );
}

#[test]
fn bounded_census_round_trip_preserves_every_mapped_record_bytes() {
    let mut input = nanbeige_input();
    let expected_payloads = install_structural_census_payloads(&mut input);
    let written = write(&input)
        .expect("non-empty structural census artifact writes")
        .bytes;
    let artifact = FnlpqArtifact::from_bytes(written.clone())
        .expect("checked loader accepts the non-empty structural census");
    let reserialized = artifact
        .reserialize()
        .expect("checked loader reserializes every structural census record");

    assert_eq!(
        reserialized, written,
        "checked load then canonical re-serialize must preserve every mapped record byte"
    );
    assert_eq!(artifact.tensors().len(), expected_payloads.len());

    for tensor in artifact.tensors() {
        assert_eq!(tensor.canonical_dtype, "bf16");
        assert_eq!(tensor.quantization, STRUCTURAL_CENSUS_RECIPE);
        let expected = expected_payloads
            .get(&tensor.name)
            .expect("every checked tensor has an expected mapped byte slice");
        let observed = checked_data_bytes(&artifact, tensor);
        assert_eq!(
            observed,
            expected.as_slice(),
            "mapped byte identity drifted for tensor={} bytes={}",
            tensor.name,
            expected.len()
        );
        eprintln!(
            "FNLPQ_ROUND_TRIP tensor={} dtype={} bytes={} sha256={}",
            tensor.name,
            tensor.canonical_dtype,
            observed.len(),
            hex(&Sha256::digest(observed))
        );
    }
    eprintln!(
        "FNLPQ_ROUND_TRIP RESULT=PASS tensors={} mapped_payload_bytes={}",
        expected_payloads.len(),
        expected_payloads.values().map(Vec::len).sum::<usize>()
    );
}

#[test]
fn bf16_verbatim_multi_tensor_round_trip_preserves_raw_words() {
    let cases: [(&str, &[u32], &[u8]); 3] = [
        (
            "model.embed_tokens.weight",
            &[4],
            &[0x00, 0x00, 0x80, 0x00, 0x80, 0x3f, 0x00, 0x80],
        ),
        (
            "model.layers.0.input_layernorm.weight",
            &[2, 2],
            &[0xff, 0x7f, 0x01, 0x80, 0x80, 0xff, 0x55, 0x3f],
        ),
        (
            "lm_head.weight",
            &[3],
            &[0x7f, 0x00, 0x34, 0x12, 0xab, 0xcd],
        ),
    ];
    let mut input = base_input("FnlpqBf16VerbatimRoundTrip");
    let mut payload = Vec::new();

    for (name, shape, bytes) in cases {
        let offset = u64::try_from(payload.len()).expect("synthetic offset fits u64");
        payload.extend_from_slice(bytes);
        input.tensors.push(TensorInput {
            name: (*name).to_owned(),
            canonical_dtype: CanonicalDtype::Bf16,
            shape: shape.to_vec(),
            canonical_logical_sha256: hex(
                &logical_tensor_sha256(name, "bf16", shape, BF16_RECIPE, bytes, &[], &[])
                    .expect("BF16 logical identity"),
            ),
            quantization: BF16_RECIPE.to_owned(),
            data: SectionRange::new(
                "generic-payload",
                offset,
                u64::try_from(bytes.len()).expect("synthetic byte length fits u64"),
            ),
            scale: SectionRange::new("generic-scales", 0, 0),
            row_sum: SectionRange::new("generic-row-sums", 0, 0),
        });
    }
    input
        .sections
        .iter_mut()
        .find(|section| section.name == "generic-payload")
        .expect("BF16 payload section")
        .bytes = payload;
    refresh_logical_model_identity(&mut input);

    let written = write(&input)
        .expect("every synthetic BF16 verbatim range has its declared byte extent")
        .bytes;
    let artifact = FnlpqArtifact::from_bytes(written.clone())
        .expect("checked loader accepts exact BF16 verbatim ranges");
    let expected_logical_model_sha256 = artifact.logical_model_sha256().to_owned();
    let reserialized = artifact
        .reserialize()
        .expect("checked BF16 artifact reserializes");
    assert_eq!(
        reserialized,
        written,
        "checked BF16 load then canonical re-serialize must preserve envelope bytes"
    );
    let round_tripped = FnlpqArtifact::from_bytes(reserialized)
        .expect("canonical re-serialization remains loadable");
    assert_eq!(
        round_tripped.logical_model_sha256(),
        expected_logical_model_sha256,
        "canonical re-serialization must preserve the writer-derived logical model identity"
    );

    for (name, _shape, expected) in cases {
        let tensor = artifact
            .tensors()
            .iter()
            .find(|tensor| tensor.name == name)
            .expect("every synthetic BF16 tensor remains declared");
        assert_eq!(tensor.canonical_dtype, "bf16");
        assert_eq!(tensor.quantization, BF16_RECIPE);
        assert_eq!(
            tensor.logical_bytes,
            u64::try_from(expected.len()).expect("synthetic byte length fits u64"),
            "logical byte declaration must retain the full BF16 extent for {name}"
        );
        assert_eq!(
            checked_data_bytes(&artifact, tensor),
            expected,
            "BF16 bytes must remain verbatim for {name}"
        );
        eprintln!(
            "FNLPQ_BF16_ROUND_TRIP tensor={name} bytes={} sha256={}",
            expected.len(),
            hex(&Sha256::digest(expected))
        );
    }
    eprintln!(
        "FNLPQ_BF16_ROUND_TRIP RESULT=PASS tensors={} logical_model_sha256={expected_logical_model_sha256}",
        cases.len()
    );
}

#[test]
fn bf16_verbatim_extent_must_match_declared_shape_on_write_and_load() {
    let mut underlength = tiny_input();
    underlength.tensors[0].shape = vec![2];
    refresh_logical_model_identity(&mut underlength);
    let error = write(&underlength).expect_err("underlength BF16 mapping must not serialize");
    assert!(matches!(
        error,
        FnlpqWriteError::Mapping {
            mapping: "data",
            ..
        }
    ));
    assert!(error.to_string().contains("bf16-verbatim-v1"));

    let mut valid = tiny_input();
    valid.tensors[0].shape = vec![2];
    valid.tensors[0].data = SectionRange::new("generic-payload", 0, 4);
    valid
        .sections
        .iter_mut()
        .find(|section| section.name == "generic-payload")
        .expect("generic payload section")
        .bytes = vec![0x80, 0x3f, 0x00, 0xc0];
    refresh_logical_model_identity(&mut valid);
    let mut corrupted = write(&valid)
        .expect("exactly sized BF16 mapping writes")
        .bytes;
    let header_len = usize::try_from(u64::from_le_bytes(
        corrupted[16..24].try_into().expect("fixed prelude"),
    ))
    .expect("header length fits host usize");
    let header_start = 80;
    let header_end = header_start + header_len;
    let shape_offset = corrupted[header_start..header_end]
        .windows(b"\"shape\":[2]".len())
        .position(|window| window == b"\"shape\":[2]")
        .expect("canonical tiny header shape");
    corrupted[header_start + shape_offset + b"\"shape\":[".len()] = b'3';
    let header_digest: [u8; 32] = Sha256::digest(&corrupted[header_start..header_end]).into();
    corrupted[48..80].copy_from_slice(&header_digest);

    let error = FnlpqArtifact::from_bytes(corrupted)
        .expect_err("underlength BF16 mapping must fail before logical identity use");
    assert!(matches!(error, FnlpqReadError::Header { .. }));
    assert!(error.to_string().contains("bf16-verbatim-v1"));
    eprintln!("FNLPQ_LOAD stage=bf16_extent verdict=PASS");
}

#[test]
fn logical_bytes_is_derived_from_declared_dtype_and_shape() {
    let valid = write(&tiny_input())
        .expect("tiny logical fixture writes")
        .bytes;
    let artifact = FnlpqArtifact::from_bytes(valid.clone())
        .expect("writer emits a logical byte count matching the tensor declaration");
    assert_eq!(artifact.tensors()[0].logical_bytes, 2);

    let mut corrupted = valid;
    replace_header_decimal(&mut corrupted, "logical_bytes", b'3');
    let error = FnlpqArtifact::from_bytes(corrupted)
        .expect_err("forged logical byte count must fail before any tensor consumer");
    assert!(matches!(error, FnlpqReadError::Header { .. }));
    assert!(error.to_string().contains("logical_bytes"));
    eprintln!("FNLPQ_LOAD stage=logical_bytes verdict=PASS");
}

#[test]
fn logical_identity_is_checked_at_write_and_load_boundaries() {
    let mut stale_tensor = tiny_input();
    stale_tensor.tensors[0].canonical_logical_sha256 = "0".repeat(64);
    assert!(matches!(
        write(&stale_tensor),
        Err(FnlpqWriteError::LogicalIdentity { .. })
    ));

    for source_name in [
        "model-config",
        "tokenizer-model",
        "tokenizer-config",
        "chat-template",
    ] {
        let mut stale_source = tiny_input();
        stale_source
            .sections
            .iter_mut()
            .find(|section| section.name == source_name)
            .expect("tiny input includes every materialized identity source")
            .bytes
            .push(0x7f);
        let error = write(&stale_source)
            .expect_err("changing a materialized source must invalidate the stale model digest");
        let detail = error.to_string();
        assert!(matches!(error, FnlpqWriteError::LogicalIdentity { .. }));
        assert!(detail.contains("logical_model_sha256"));
        eprintln!(
            "FNLPQ_LOGICAL_IDENTITY source={source_name} verdict=REFUSED reason=stale_model_digest"
        );
    }

    let valid = write(&tiny_input())
        .expect("tiny logical fixture writes")
        .bytes;
    let mut stale_model = valid.clone();
    replace_header_digest(&mut stale_model, "logical_model_sha256", &"0".repeat(64));
    let error = FnlpqArtifact::from_bytes(stale_model)
        .expect_err("stale model digest must fail before a consumer can load it");
    assert!(matches!(error, FnlpqReadError::Header { .. }));
    assert!(error.to_string().contains("logical_model_sha256"));

    let mut flipped_payload = valid;
    let first_payload = first_section_offset(&flipped_payload);
    flipped_payload[first_payload] ^= 1;
    let error = FnlpqArtifact::from_bytes(flipped_payload)
        .expect_err("single-byte payload corruption must not load silently");
    assert!(matches!(error, FnlpqReadError::Directory { .. }));
    eprintln!("FNLPQ_LOAD stage=identity verdict=PASS catching_layers=logical,section");
}

fn assert_census_refusal(
    bytes: Vec<u8>,
    fixture: &str,
    expected_tensor: &str,
    expected_reason_prefix: &str,
) {
    let error = FnlpqArtifact::from_bytes(bytes).expect_err(fixture);
    match error {
        FnlpqReadError::Census { tensor, reason } => {
            assert_eq!(tensor, expected_tensor, "census fixture={fixture}");
            assert!(
                reason.starts_with(expected_reason_prefix),
                "census fixture={fixture} expected reason prefix {expected_reason_prefix:?} actual={reason:?}"
            );
            eprintln!(
                "FNLPQ_LOAD stage=census fixture={fixture} verdict=REFUSED tensor={tensor} reason={reason}"
            );
        }
        other => panic!("census fixture={fixture} expected Census error, actual={other}"),
    }
}

fn nanbeige_input() -> FnlpqWriterInput {
    let mut input = base_input("Nanbeige4.2-3B");
    input.tensors = expected_nanbeige42_census()
        .into_iter()
        .map(|expected| {
            let shape = expected
                .shape
                .iter()
                .map(|dimension| u32::try_from(*dimension).expect("frozen shape fits v1"))
                .collect::<Vec<_>>();
            TensorInput {
                canonical_logical_sha256: logical_tensor_hex(
                    &expected.name,
                    &shape,
                    STRUCTURAL_CENSUS_RECIPE,
                ),
                name: expected.name,
                canonical_dtype: CanonicalDtype::Bf16,
                shape,
                quantization: STRUCTURAL_CENSUS_RECIPE.to_owned(),
                data: SectionRange::new("generic-payload", 0, 0),
                scale: SectionRange::new("generic-scales", 0, 0),
                row_sum: SectionRange::new("generic-row-sums", 0, 0),
            }
        })
        .collect();
    refresh_logical_model_identity(&mut input);
    input
}

/// Install a small unique byte slice for each 201-record structural census
/// member. This is deliberately bounded: the real 8.3 GiB BF16 closure is
/// exercised only by the model-gated converter path, while this always-on
/// fixture proves that every declared record survives the checked load path.
fn install_structural_census_payloads(input: &mut FnlpqWriterInput) -> BTreeMap<String, Vec<u8>> {
    const BYTES_PER_TENSOR: usize = 8;

    let mut expected_payloads = BTreeMap::new();
    let mut payload = Vec::with_capacity(input.tensors.len() * BYTES_PER_TENSOR);
    for (ordinal, tensor) in input.tensors.iter_mut().enumerate() {
        let offset = u64::try_from(payload.len()).expect("synthetic payload offset fits u64");
        let ordinal = u16::try_from(ordinal + 1).expect("201-record census fits u16");
        let [ordinal_low, ordinal_high] = ordinal.to_le_bytes();
        let bytes = [
            ordinal_low,
            ordinal_high,
            0x80,
            0x3f,
            0x00,
            0xc0,
            0x00,
            0x3f,
        ];
        payload.extend_from_slice(&bytes);
        tensor.data = SectionRange::new(
            "generic-payload",
            offset,
            u64::try_from(BYTES_PER_TENSOR).expect("fixed synthetic slice fits u64"),
        );
        expected_payloads.insert(tensor.name.clone(), bytes.to_vec());
    }
    input
        .sections
        .iter_mut()
        .find(|section| section.name == "generic-payload")
        .expect("synthetic generic payload section")
        .bytes = payload;
    refresh_logical_model_identity(input);
    expected_payloads
}

fn tiny_input() -> FnlpqWriterInput {
    let mut input = base_input("FnlpqTinyIdentity");
    input
        .sections
        .iter_mut()
        .find(|section| section.name == "generic-payload")
        .expect("payload section")
        .bytes = vec![0x80, 0x3f];
    input.tensors = vec![TensorInput {
        name: "tiny.weight".to_owned(),
        canonical_dtype: CanonicalDtype::Bf16,
        shape: vec![1],
        canonical_logical_sha256: logical_tensor_hex("tiny.weight", &[1], BF16_RECIPE),
        quantization: BF16_RECIPE.to_owned(),
        data: SectionRange::new("generic-payload", 0, 2),
        scale: SectionRange::new("generic-scales", 0, 0),
        row_sum: SectionRange::new("generic-row-sums", 0, 0),
    }];
    refresh_logical_model_identity(&mut input);
    input
}

fn base_input(model_id: &str) -> FnlpqWriterInput {
    FnlpqWriterInput {
        model_id: model_id.to_owned(),
        revision: PINNED_REVISION.to_owned(),
        recipe_id: "fnlpq-census-load-v1".to_owned(),
        source_root_sha256: framed_sha256_hex("fnlpq-source-root-v1", &[b"census fixture"])
            .expect("source identity"),
        logical_model_sha256: "0".repeat(64),
        sections: vec![
            SectionPayload::new("generic-payload", SectionKind::GenericTensorPayload, [], 8),
            SectionPayload::new("generic-scales", SectionKind::GenericTensorScales, [], 8),
            SectionPayload::new("generic-row-sums", SectionKind::GenericTensorRowSums, [], 8),
            SectionPayload::new("tokenizer-model", SectionKind::TokenizerModel, [0x0a], 8),
            SectionPayload::new("model-config", SectionKind::ModelConfig, b"{}", 8),
            SectionPayload::new("tokenizer-config", SectionKind::TokenizerConfig, b"{}", 8),
            SectionPayload::new("chat-template", SectionKind::ChatTemplate, b"template", 8),
            SectionPayload::new(
                "license-bundle",
                SectionKind::LicenseBundle,
                b"Apache-2.0\n",
                8,
            ),
        ],
        tensors: Vec::new(),
        packing_sets: vec![PackingSetInput {
            id: "generic".to_owned(),
            target: ArchTarget::Generic,
            section_names: vec![
                "generic-payload".to_owned(),
                "generic-scales".to_owned(),
                "generic-row-sums".to_owned(),
            ],
        }],
    }
}

fn refresh_logical_model_identity(input: &mut FnlpqWriterInput) {
    let sections = input.sections.clone();
    let mut tensor_digests = Vec::with_capacity(input.tensors.len());
    for tensor in &mut input.tensors {
        let data = mapped_bytes_in_sections(&sections, &tensor.data);
        let scale = mapped_bytes_in_sections(&sections, &tensor.scale);
        let row_sum = mapped_bytes_in_sections(&sections, &tensor.row_sum);
        let digest = logical_tensor_sha256(
            &tensor.name,
            tensor.canonical_dtype.header_name(),
            &tensor.shape,
            &tensor.quantization,
            data,
            scale,
            row_sum,
        )
        .expect("logical tensor identity");
        tensor.canonical_logical_sha256 = hex(&digest);
        tensor_digests.push((tensor.name.clone(), digest));
    }
    // `write` canonicalizes tensor records by their bytewise names before it
    // frames the model identity.  Keep synthetic inputs in that same order so
    // this helper prepares a valid writer input rather than a stale fixture
    // identity that happens to reflect insertion order.
    tensor_digests.sort_by(|(left, _), (right, _)| left.as_bytes().cmp(right.as_bytes()));
    let tensor_digests = tensor_digests
        .into_iter()
        .map(|(_, digest)| digest)
        .collect::<Vec<_>>();
    let sources = [
        ("model_config", section_bytes(input, "model-config")),
        ("tokenizer_model", section_bytes(input, "tokenizer-model")),
        ("tokenizer_config", section_bytes(input, "tokenizer-config")),
        ("chat_template", section_bytes(input, "chat-template")),
    ];
    input.logical_model_sha256 =
        hex(&logical_model_sha256(&tensor_digests, &sources).expect("logical model identity"));
}

fn logical_tensor_hex(name: &str, shape: &[u32], quantization: &str) -> String {
    hex(
        &logical_tensor_sha256(name, "bf16", shape, quantization, &[], &[], &[])
            .expect("logical tensor identity"),
    )
}

fn section_bytes<'a>(input: &'a FnlpqWriterInput, name: &str) -> &'a [u8] {
    &input
        .sections
        .iter()
        .find(|section| section.name == name)
        .expect("required fixture section")
        .bytes
}

fn mapped_bytes_in_sections<'a>(sections: &'a [SectionPayload], range: &SectionRange) -> &'a [u8] {
    let section = &sections
        .iter()
        .find(|section| section.name == range.section_name)
        .expect("fixture mapping section")
        .bytes;
    let start = usize::try_from(range.offset).expect("fixture offset");
    let end = start + usize::try_from(range.len).expect("fixture length");
    &section[start..end]
}

fn checked_data_bytes<'a>(
    artifact: &'a FnlpqArtifact,
    tensor: &franken_nlp::artifact::reader::CheckedTensor,
) -> &'a [u8] {
    let section = artifact
        .section_bytes(tensor.data.section_ordinal)
        .expect("checked tensor data section remains available");
    let start = usize::try_from(tensor.data.offset).expect("checked tensor data offset fits host");
    let len = usize::try_from(tensor.data.len).expect("checked tensor data length fits host");
    let end = start
        .checked_add(len)
        .expect("checked tensor data range cannot overflow host usize");
    section
        .get(start..end)
        .expect("checked tensor data range remains inside its checked section")
}

fn replace_header_digest(bytes: &mut [u8], field: &str, replacement: &str) {
    let header_len = usize::try_from(u64::from_le_bytes(
        bytes[16..24].try_into().expect("prelude"),
    ))
    .expect("fixture header fits host");
    let header_range = 80..80 + header_len;
    let header = std::str::from_utf8(&bytes[header_range.clone()]).expect("header UTF-8");
    let prefix = format!("\"{field}\":\"");
    let start = header.find(&prefix).expect("header field") + prefix.len();
    let end = start + 64;
    assert!(
        header[start..end]
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    );
    bytes[80 + start..80 + end].copy_from_slice(replacement.as_bytes());
    let digest: [u8; 32] = Sha256::digest(&bytes[header_range]).into();
    bytes[48..80].copy_from_slice(&digest);
}

fn replace_header_decimal(bytes: &mut [u8], field: &str, replacement: u8) {
    assert!(
        replacement.is_ascii_digit(),
        "replacement must be a decimal digit"
    );
    let header_len = usize::try_from(u64::from_le_bytes(
        bytes[16..24].try_into().expect("prelude"),
    ))
    .expect("fixture header fits host");
    let header_range = 80..80 + header_len;
    let header = std::str::from_utf8(&bytes[header_range.clone()]).expect("header UTF-8");
    let prefix = format!("\"{field}\":");
    let start = header.find(&prefix).expect("header field") + prefix.len();
    assert!(header.as_bytes()[start].is_ascii_digit());
    bytes[80 + start] = replacement;
    let digest: [u8; 32] = Sha256::digest(&bytes[header_range]).into();
    bytes[48..80].copy_from_slice(&digest);
}

fn first_section_offset(bytes: &[u8]) -> usize {
    let header_len = usize::try_from(u64::from_le_bytes(
        bytes[16..24].try_into().expect("prelude"),
    ))
    .expect("fixture header fits host");
    usize::try_from(u64::from_le_bytes(
        bytes[80 + header_len + 16..80 + header_len + 24]
            .try_into()
            .expect("first directory entry"),
    ))
    .expect("fixture section offset fits host")
}

fn raw_sha256_file(path: &Path) -> String {
    let mut file = File::open(path).expect("real artifact must open for raw SHA-256");
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .expect("real artifact raw SHA-256 read must succeed");
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    hex(&hasher.finalize())
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len().saturating_mul(2));
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}
