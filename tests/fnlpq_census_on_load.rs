use franken_nlp::artifact::converter::expected_nanbeige42_census;
use franken_nlp::artifact::format::{
    framed_sha256_hex, logical_model_sha256, logical_tensor_sha256, write, ArchTarget,
    CanonicalDtype, FnlpqWriteError, FnlpqWriterInput, PackingSetInput, SectionKind,
    SectionPayload, SectionRange, TensorInput,
};
use franken_nlp::artifact::reader::{FnlpqArtifact, FnlpqReadError};
use sha2::{Digest, Sha256};

const PINNED_REVISION: &str = "f56ec5a9650268aa098496734743c25ea778bd2d";
const BF16_RECIPE: &str = "bf16-verbatim-v1";

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
    missing.tensors.pop();
    refresh_logical_model_identity(&mut missing);
    assert_census_refusal(
        write(&missing).expect("missing fixture writes").bytes,
        "missing",
    );

    let mut wrong_shape = complete.clone();
    wrong_shape.tensors[0].shape[0] += 1;
    refresh_logical_model_identity(&mut wrong_shape);
    assert_census_refusal(
        write(&wrong_shape)
            .expect("wrong-shape fixture writes")
            .bytes,
        "shape",
    );

    let mut renamed = complete;
    renamed
        .tensors
        .last_mut()
        .expect("complete census has tensors")
        .name = "z.renamed.tensor".to_owned();
    refresh_logical_model_identity(&mut renamed);
    assert_census_refusal(
        write(&renamed).expect("renamed fixture writes").bytes,
        "renamed",
    );
}

#[test]
fn logical_identity_is_checked_at_write_and_load_boundaries() {
    let mut stale_tensor = tiny_input();
    stale_tensor.tensors[0].canonical_logical_sha256 = "0".repeat(64);
    assert!(matches!(
        write(&stale_tensor),
        Err(FnlpqWriteError::LogicalIdentity { .. })
    ));

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

fn assert_census_refusal(bytes: Vec<u8>, fixture: &str) {
    let error = FnlpqArtifact::from_bytes(bytes).expect_err(fixture);
    assert!(matches!(error, FnlpqReadError::Census { .. }));
    eprintln!("FNLPQ_LOAD stage=census fixture={fixture} verdict=REFUSED detail={error}");
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
                canonical_logical_sha256: logical_tensor_hex(&expected.name, &shape),
                name: expected.name,
                canonical_dtype: CanonicalDtype::Bf16,
                shape,
                quantization: BF16_RECIPE.to_owned(),
                data: SectionRange::new("generic-payload", 0, 0),
                scale: SectionRange::new("generic-scales", 0, 0),
                row_sum: SectionRange::new("generic-row-sums", 0, 0),
            }
        })
        .collect();
    refresh_logical_model_identity(&mut input);
    input
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
        canonical_logical_sha256: logical_tensor_hex("tiny.weight", &[1]),
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
        tensor_digests.push(digest);
    }
    let sources = [
        ("model_config", section_bytes(input, "model-config")),
        ("tokenizer_model", section_bytes(input, "tokenizer-model")),
        ("tokenizer_config", section_bytes(input, "tokenizer-config")),
        ("chat_template", section_bytes(input, "chat-template")),
    ];
    input.logical_model_sha256 =
        hex(&logical_model_sha256(&tensor_digests, &sources).expect("logical model identity"));
}

fn logical_tensor_hex(name: &str, shape: &[u32]) -> String {
    hex(
        &logical_tensor_sha256(name, "bf16", shape, BF16_RECIPE, &[], &[], &[])
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
    assert!(header[start..end]
        .bytes()
        .all(|byte| byte.is_ascii_hexdigit()));
    bytes[80 + start..80 + end].copy_from_slice(replacement.as_bytes());
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

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len().saturating_mul(2));
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}
