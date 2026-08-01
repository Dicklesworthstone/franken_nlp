//! Pre-write coverage for the `fnlp tokens` embedded-tokenizer boundary.
//!
//! CLI transcript coverage is added once the pinned tokenizer asset and the
//! shared CLI command slot are available.  These tests lock the lower-level
//! binary-versus-artifact check now, using only a synthetic valid ModelProto
//! and a synthetic checked `.fnlpq` envelope.

use franken_nlp::{
    artifact::{
        converter::expected_nanbeige42_census,
        format::{
            ArchTarget, CanonicalDtype, FnlpqWriterInput, PackingSetInput, SectionKind,
            SectionPayload, SectionRange, TensorInput, framed_sha256_hex, logical_model_sha256,
            logical_tensor_sha256, write,
        },
        reader::FnlpqArtifact,
    },
    tokenizer::{
        embedded::{EmbeddedTokenizer, TokenizerArtifactIntegrityError},
        sp_model::SpmErrorKind,
    },
};

const TOKENIZER_METADATA_RECIPE: &str = "tokenizer-metadata-fixture-v1";

fn varint(mut value: u64) -> Vec<u8> {
    let mut output = Vec::new();
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        output.push(byte);
        if value == 0 {
            return output;
        }
    }
}

fn add_varint(output: &mut Vec<u8>, field: u32, value: u64) {
    output.extend(varint((u64::from(field) << 3) | 0));
    output.extend(varint(value));
}

fn add_fixed32(output: &mut Vec<u8>, field: u32, value: u32) {
    output.extend(varint((u64::from(field) << 3) | 5));
    output.extend(value.to_le_bytes());
}

fn add_bytes(output: &mut Vec<u8>, field: u32, value: &[u8]) {
    output.extend(varint((u64::from(field) << 3) | 2));
    output.extend(varint(value.len() as u64));
    output.extend(value);
}

fn piece(surface: &str, piece_type: u64) -> Vec<u8> {
    let mut output = Vec::new();
    add_bytes(&mut output, 1, surface.as_bytes());
    add_fixed32(&mut output, 2, 0.0_f32.to_bits());
    add_varint(&mut output, 3, piece_type);
    output
}

fn test_model_bytes() -> &'static [u8] {
    let mut trainer = Vec::new();
    add_varint(&mut trainer, 3, 2); // BPE
    add_varint(&mut trainer, 40, 0); // unk
    add_varint(&mut trainer, 41, 1); // bos
    add_varint(&mut trainer, 42, 2); // eos
    add_varint(&mut trainer, 43, u64::MAX); // pad = -1

    let mut normalizer = Vec::new();
    add_bytes(&mut normalizer, 1, b"identity");

    let mut model = Vec::new();
    add_bytes(&mut model, 1, &piece("<unk>", 2));
    add_bytes(&mut model, 1, &piece("<s>", 3));
    add_bytes(&mut model, 1, &piece("</s>", 3));
    add_bytes(&mut model, 1, &piece("▁", 1));
    add_bytes(&mut model, 1, &piece("a", 1));
    add_bytes(&mut model, 2, &trainer);
    add_bytes(&mut model, 3, &normalizer);
    Box::leak(model.into_boxed_slice())
}

fn synthetic_artifact(tokenizer_model: Vec<u8>) -> FnlpqArtifact {
    // A Nanbeige-labelled metadata fixture is accepted only after the full
    // frozen census passes. Its payload mappings intentionally stay empty:
    // this boundary test authenticates only the tokenizer copy, and must not
    // claim the `bf16-verbatim-v1` representation it does not materialize.
    let (tensors, tensor_digests): (Vec<_>, Vec<_>) = expected_nanbeige42_census()
        .into_iter()
        .map(|expected| {
            let shape = expected
                .shape
                .iter()
                .map(|dimension| u32::try_from(*dimension).expect("frozen shape fits v1"))
                .collect::<Vec<_>>();
            let tensor_digest = logical_tensor_sha256(
                &expected.name,
                "bf16",
                &shape,
                TOKENIZER_METADATA_RECIPE,
                &[],
                &[],
                &[],
            )
            .expect("synthetic tensor identity");
            (
                TensorInput {
                    name: expected.name,
                    canonical_dtype: CanonicalDtype::Bf16,
                    shape,
                    canonical_logical_sha256: hex_lower(&tensor_digest),
                    quantization: TOKENIZER_METADATA_RECIPE.to_owned(),
                    data: SectionRange::new("generic-payload", 0, 0),
                    scale: SectionRange::new("generic-scales", 0, 0),
                    row_sum: SectionRange::new("generic-row-sums", 0, 0),
                },
                tensor_digest,
            )
        })
        .unzip();
    let logical_model_digest = logical_model_sha256(
        "Nanbeige4.2-3B",
        "f56ec5a9650268aa098496734743c25ea778bd2d",
        &tensor_digests,
        &[
            ("model_config", br#"{"hidden_size":2}"#.as_slice()),
            ("tokenizer_model", tokenizer_model.as_slice()),
            ("tokenizer_config", br#"{"bos_token":"<s>"}"#.as_slice()),
            ("chat_template", b"{{ message }}".as_slice()),
        ],
    )
    .expect("synthetic logical-model identity");
    let input = FnlpqWriterInput {
        model_id: "Nanbeige4.2-3B".to_owned(),
        revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
        recipe_id: "cli-tokens-prewrite-v1".to_owned(),
        source_root_sha256: framed_sha256_hex("fnlpq-source-root-v1", &[b"cli token test"])
            .expect("valid source identity"),
        logical_model_sha256: hex_lower(&logical_model_digest),
        sections: vec![
            SectionPayload::new("generic-payload", SectionKind::GenericTensorPayload, [], 64),
            SectionPayload::new("generic-scales", SectionKind::GenericTensorScales, [], 8),
            SectionPayload::new("generic-row-sums", SectionKind::GenericTensorRowSums, [], 8),
            SectionPayload::new(
                "tokenizer-model",
                SectionKind::TokenizerModel,
                tokenizer_model,
                32,
            ),
            SectionPayload::new(
                "model-config",
                SectionKind::ModelConfig,
                b"{\"hidden_size\":2}".to_vec(),
                8,
            ),
            SectionPayload::new(
                "tokenizer-config",
                SectionKind::TokenizerConfig,
                b"{\"bos_token\":\"<s>\"}".to_vec(),
                8,
            ),
            SectionPayload::new(
                "chat-template",
                SectionKind::ChatTemplate,
                b"{{ message }}".to_vec(),
                8,
            ),
            SectionPayload::new(
                "license-bundle",
                SectionKind::LicenseBundle,
                b"Apache-2.0\nModel origin: Nanbeige/Nanbeige4.2-3B\n".to_vec(),
                16,
            ),
        ],
        tensors,
        packing_sets: vec![PackingSetInput {
            id: "generic".to_owned(),
            target: ArchTarget::Generic,
            section_names: vec![
                "generic-payload".to_owned(),
                "generic-scales".to_owned(),
                "generic-row-sums".to_owned(),
            ],
        }],
    };
    let written = write(&input).expect("synthetic artifact writes");
    FnlpqArtifact::from_bytes(written.bytes).expect("synthetic artifact reads")
}

fn hex_lower(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[test]
fn embedded_tokenizer_encodes_and_matches_the_artifact_copy() {
    let embedded = EmbeddedTokenizer::from_bytes(test_model_bytes())
        .expect("synthetic embedded tokenizer is accepted");
    assert_eq!(embedded.tokenizer().encode_ids("a").unwrap(), vec![1, 3, 4]);
    assert_eq!(embedded.sha256_hex().len(), 64);
    embedded
        .verify_artifact(&synthetic_artifact(embedded.bytes().to_vec()))
        .expect("matching embedded and artifact tokenizer bytes pass");
    eprintln!(
        "CLI_TOKENS PREWRITE=PASS boundary=embedded-artifact digest={}",
        embedded.sha256_hex()
    );
}

#[test]
fn tokenizer_mismatch_names_both_digests_and_the_matching_remediation() {
    let embedded = EmbeddedTokenizer::from_bytes(test_model_bytes()).unwrap();
    let error = embedded
        .verify_artifact(&synthetic_artifact(b"different-tokenizer".to_vec()))
        .expect_err("a different checked artifact tokenizer must fail closed");
    let TokenizerArtifactIntegrityError::DigestMismatch {
        binary_sha256,
        artifact_sha256,
    } = error
    else {
        panic!("expected a digest mismatch");
    };
    let rendered = TokenizerArtifactIntegrityError::DigestMismatch {
        binary_sha256,
        artifact_sha256,
    }
    .to_string();
    assert!(rendered.contains("binary_sha256="));
    assert!(rendered.contains("artifact_sha256="));
    assert!(rendered.contains("fnlp pull"));
    eprintln!("CLI_TOKENS PREWRITE=PASS boundary=mismatch-fail-closed");
}

#[test]
fn invalid_embedded_tokenizer_bytes_are_rejected_without_a_fallback() {
    let error = EmbeddedTokenizer::from_bytes(b"\0")
        .expect_err("malformed embedded bytes must not select another tokenizer");
    assert!(matches!(
        error,
        franken_nlp::tokenizer::embedded::EmbeddedTokenizerError::Model(error)
            if matches!(error.kind, SpmErrorKind::InvalidFieldNumber)
    ));
    eprintln!("CLI_TOKENS PREWRITE=PASS boundary=embedded-parse-fail-closed");
}

#[test]
fn invalid_embedded_added_token_registry_is_rejected_without_a_fallback() {
    let error = EmbeddedTokenizer::from_bytes_with_added_tokens(test_model_bytes(), b"{")
        .expect_err("malformed added-token bytes must not select another tokenizer");
    assert!(matches!(
        error,
        franken_nlp::tokenizer::embedded::EmbeddedTokenizerError::AddedTokenRegistry(_)
    ));
}
