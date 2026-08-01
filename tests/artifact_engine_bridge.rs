use std::collections::BTreeMap;
use std::path::Path;

use franken_nlp::artifact::converter::StorageStage;
use franken_nlp::native_engine::{
    artifact_bridge::{
        ArtifactBridgeError, ArtifactIdentity, ArtifactLoadBudget, ArtifactTensorContract,
        ArtifactTensorDescriptor, CheckedArtifactSource, FORENSIC_METADATA_READ_CAP, TensorMapping,
        TensorMappingLengths, load_nanbeige42, load_synthetic_with_contract,
        scan_current_80_byte_forensic_census,
    },
    tensor::Bf16,
};

#[derive(Clone, Debug)]
struct StreamingFixture {
    identity: ArtifactIdentity,
    tensors: Vec<ArtifactTensorDescriptor>,
    mappings: BTreeMap<String, [Vec<u8>; 3]>,
    chunk_bytes: usize,
    resident_envelope_bytes: u64,
}

const REAL_ARTIFACT: &str = "/tmp/fnlp-conversion/nanbeige4.2-3b.fnlpq-v1.int8.generic.fnlpq";

impl StreamingFixture {
    fn expected() -> Self {
        let mut mappings = BTreeMap::new();
        mappings.insert(
            "model.embed_tokens.weight".to_owned(),
            [
                vec![0x80, 0x3f, 0x00, 0xc0, 0x00, 0x00, 0x80, 0x3f],
                Vec::new(),
                Vec::new(),
            ],
        );
        mappings.insert(
            "model.layers.0.self_attn.q_proj.weight".to_owned(),
            [
                vec![0xff, 0x02, 0x03, 0xfc],
                [0.25_f32.to_le_bytes(), 0.5_f32.to_le_bytes()].concat(),
                [1_i32.to_le_bytes(), (-1_i32).to_le_bytes()].concat(),
            ],
        );
        Self {
            identity: ArtifactIdentity {
                model_id: "SyntheticNanbeigeBridge".to_owned(),
                revision: "fixture-revision".to_owned(),
                recipe_id: "fixture-recipe".to_owned(),
                source_root_sha256: "1".repeat(64),
                logical_model_sha256: "2".repeat(64),
            },
            tensors: vec![
                ArtifactTensorDescriptor {
                    name: "model.embed_tokens.weight".to_owned(),
                    canonical_dtype: "bf16".to_owned(),
                    shape: vec![2, 2],
                    quantization: "bf16-verbatim-v1".to_owned(),
                    mapping_lengths: TensorMappingLengths {
                        data: 8,
                        scale: 0,
                        row_sum: 0,
                    },
                },
                ArtifactTensorDescriptor {
                    name: "model.layers.0.self_attn.q_proj.weight".to_owned(),
                    canonical_dtype: "i8".to_owned(),
                    shape: vec![2, 2],
                    quantization: "portable-quant-v1".to_owned(),
                    mapping_lengths: TensorMappingLengths {
                        data: 4,
                        scale: 8,
                        row_sum: 8,
                    },
                },
            ],
            mappings,
            chunk_bytes: 3,
            resident_envelope_bytes: 0,
        }
    }
}

impl CheckedArtifactSource for StreamingFixture {
    fn identity(&self) -> &ArtifactIdentity {
        &self.identity
    }

    fn tensors(&self) -> &[ArtifactTensorDescriptor] {
        &self.tensors
    }

    fn resident_envelope_bytes(&self) -> u64 {
        self.resident_envelope_bytes
    }

    fn stream_mapping(
        &self,
        tensor: &ArtifactTensorDescriptor,
        mapping: TensorMapping,
        visitor: &mut dyn FnMut(&[u8]) -> Result<(), ArtifactBridgeError>,
    ) -> Result<(), ArtifactBridgeError> {
        let mapping_index = match mapping {
            TensorMapping::Data => 0,
            TensorMapping::Scale => 1,
            TensorMapping::RowSum => 2,
        };
        let bytes = self
            .mappings
            .get(&tensor.name)
            .expect("fixture has all declared mappings");
        for chunk in bytes[mapping_index].chunks(self.chunk_bytes) {
            visitor(chunk)?;
        }
        Ok(())
    }
}

fn contracts() -> [ArtifactTensorContract; 2] {
    [
        ArtifactTensorContract {
            source_name: "model.embed_tokens.weight".to_owned(),
            internal_name: "embed".to_owned(),
            shape: vec![2, 2],
            stage: StorageStage::Bf16Verbatim,
        },
        ArtifactTensorContract {
            source_name: "model.layers.0.self_attn.q_proj.weight".to_owned(),
            internal_name: "layer.0.attn.q".to_owned(),
            shape: vec![2, 2],
            stage: StorageStage::Int8Stage2B,
        },
    ]
}

#[test]
fn streaming_bridge_materializes_bf16_and_portable_int8_without_envelope_residency() {
    let fixture = StreamingFixture::expected();
    let mut lines = Vec::new();
    let loaded = load_synthetic_with_contract(
        &fixture,
        ArtifactLoadBudget::streaming_only(28, 3),
        &contracts(),
        |line| lines.push(line.to_owned()),
    )
    .expect("bounded checked mappings materialize into typed native sets");

    let embed = loaded
        .weights
        .bf16()
        .get("embed")
        .expect("bf16 source route enters the bf16 set");
    let franken_nlp::native_engine::artifact_bridge::ArtifactBf16Tensor::Matrix(embed) = embed
    else {
        panic!("rank-two bf16 tensor must materialize as Bf16Matrix");
    };
    assert_eq!(
        embed.row(0).expect("first row"),
        &[Bf16::from_bits(0x3f80), Bf16::from_bits(0xc000)]
    );
    assert_eq!(
        embed.row(1).expect("second row"),
        &[Bf16::from_bits(0x0000), Bf16::from_bits(0x3f80)]
    );

    let quantized = loaded
        .weights
        .quantized()
        .get("layer.0.attn.q")
        .expect("int8 source route enters the quantized set");
    assert_eq!(quantized.values(), &[-1, 2, 3, -4]);
    assert_eq!(quantized.row_scales(), &[0.25, 0.5]);
    assert_eq!(quantized.row_sums(), &[1, -1]);
    assert_eq!(loaded.receipt.declared_mapping_bytes, 28);
    assert_eq!(loaded.receipt.largest_stream_chunk_bytes, 3);
    assert_eq!(loaded.receipt.declared_source_resident_bytes, 0);
    assert_eq!(
        loaded.receipt.modeled_declared_bytes_without_overhead(),
        Some(31)
    );
    assert!(lines.iter().any(|line| line
        == "LOAD STAGE=preflight scope=synthetic evidence=non_authoritative status=BEGIN"));
    assert!(lines.iter().any(|line| line.contains(
        "LOAD STAGE=census scope=synthetic evidence=non_authoritative status=PASS tensors=2"
    )));
    assert!(lines.iter().any(|line| line.contains("LOAD STAGE=tensor scope=synthetic evidence=non_authoritative status=PASS tensor=model.embed_tokens.weight route=embed storage=bf16-verbatim")));
    assert!(lines.iter().any(|line| line.contains("LOAD STAGE=tensor scope=synthetic evidence=non_authoritative status=PASS tensor=model.layers.0.self_attn.q_proj.weight route=layer.0.attn.q storage=int8-stage-2b")));
    assert!(lines.iter().any(|line| {
        line.contains("LOAD STAGE=complete scope=synthetic evidence=non_authoritative status=PASS bf16_tensors=1 quantized_tensors=1")
    }));
}

#[test]
fn streaming_bridge_refuses_tampered_row_sum_after_checked_sidecar_decode() {
    let mut fixture = StreamingFixture::expected();
    fixture
        .mappings
        .get_mut("model.layers.0.self_attn.q_proj.weight")
        .expect("quantized mapping")[2] = [0_i32.to_le_bytes(), (-1_i32).to_le_bytes()].concat();
    let error = load_synthetic_with_contract(
        &fixture,
        ArtifactLoadBudget::streaming_only(28, 3),
        &contracts(),
        |_| {},
    )
    .expect_err("row sums bind every canonical int8 row");
    assert!(matches!(
        error,
        ArtifactBridgeError::Tensor {
            ref tensor,
            stage: "row-sums",
            ..
        } if tensor == "model.layers.0.self_attn.q_proj.weight"
    ));
}

#[test]
fn streaming_bridge_refuses_nonfinite_quantization_scale_at_the_sidecar_stage() {
    let mut fixture = StreamingFixture::expected();
    fixture
        .mappings
        .get_mut("model.layers.0.self_attn.q_proj.weight")
        .expect("quantized mapping")[1] =
        [f32::INFINITY.to_le_bytes(), 0.5_f32.to_le_bytes()].concat();
    let error = load_synthetic_with_contract(
        &fixture,
        ArtifactLoadBudget::streaming_only(28, 3),
        &contracts(),
        |_| {},
    )
    .expect_err("nonfinite scales are never materialized into the quantized set");
    assert!(matches!(
        error,
        ArtifactBridgeError::Tensor {
            ref tensor,
            stage: "scales",
            ..
        } if tensor == "model.layers.0.self_attn.q_proj.weight"
    ));
}

#[test]
fn streaming_bridge_refuses_a_source_chunk_larger_than_its_admitted_scratch_bound() {
    let mut fixture = StreamingFixture::expected();
    fixture.chunk_bytes = 4;
    let error = load_synthetic_with_contract(
        &fixture,
        ArtifactLoadBudget::streaming_only(28, 3),
        &contracts(),
        |_| {},
    )
    .expect_err("a source chunk may not exceed the bridge scratch budget");
    assert_eq!(
        error,
        ArtifactBridgeError::Memory {
            subject: "stream-chunk-bytes",
            observed: 4,
            limit: 3,
        }
    );
}

#[test]
fn streaming_only_budget_refuses_a_reader_style_resident_envelope_before_allocation() {
    let mut fixture = StreamingFixture::expected();
    fixture.resident_envelope_bytes = 4_690_873_282;
    let error = load_synthetic_with_contract(
        &fixture,
        ArtifactLoadBudget::streaming_only(28, 3),
        &contracts(),
        |_| {},
    )
    .expect_err("the bridge may not build weights beside an owned envelope");
    assert_eq!(
        error,
        ArtifactBridgeError::Memory {
            subject: "resident-envelope-bytes",
            observed: 4_690_873_282,
            limit: 0,
        }
    );
}

#[test]
fn one_model_entry_refuses_unresolved_oq31_before_selecting_a_quantized_dtype() {
    let mut fixture = StreamingFixture::expected();
    fixture.identity.model_id = "Nanbeige4.2-3B".to_owned();
    fixture.tensors[1].canonical_dtype = "bf16".to_owned();
    let mut lines = Vec::new();
    let error = load_nanbeige42(
        &fixture,
        ArtifactLoadBudget::streaming_only(28, 3),
        |line| lines.push(line.to_owned()),
    )
    .expect_err("OQ-31 must be ratified before an artifact representation is selected");
    assert!(matches!(error, ArtifactBridgeError::SchemaAuthority { .. }));
    assert_eq!(
        lines,
        [
            "LOAD STAGE=activation scope=synthetic evidence=non_authoritative status=REFUSED reason=oq31-authority-unresolved"
                .to_owned()
        ]
    );
}

#[test]
fn real_artifact_forensic_census_reads_only_bounded_metadata_when_model_is_present() {
    let path = Path::new(REAL_ARTIFACT);
    if !path.is_file() {
        eprintln!(
            "LOAD STAGE=forensic-census scope=real-artifact-forensic evidence=non_authoritative status=SKIPPED reason=model-artifact-unavailable"
        );
        return;
    }
    let mut lines = Vec::new();
    let report = scan_current_80_byte_forensic_census(path, |line| lines.push(line.to_owned()))
        .expect("current-writer artifact metadata has a bounded forensic census");
    for line in &lines {
        eprintln!("{line}");
    }
    assert_eq!(report.observed_file_bytes, 4_690_873_282);
    assert_eq!(report.declared_file_bytes, 4_690_873_282);
    assert_eq!(report.header_bytes, 87_571);
    assert_eq!(report.section_count, 8);
    assert_eq!(report.tensor_count, 201);
    assert_eq!(report.model_id, "Nanbeige4.2-3B");
    assert_eq!(report.revision, "f56ec5a9650268aa098496734743c25ea778bd2d");
    assert_eq!(
        report.source_root_sha256,
        "71686a7075712c2b38e80fe387b9f23e3befac6e367bb4cb0d0ba5a90374c7fd"
    );
    assert_eq!(report.canonical_dtype_counts.get("bf16"), Some(&201));
    assert_eq!(report.bf16_verbatim_tensor_count, 46);
    assert_eq!(report.portable_quant_tensor_count, 155);
    assert!(
        report.metadata_bytes_read
            <= u64::try_from(FORENSIC_METADATA_READ_CAP).expect("test cap fits u64"),
        "bounded forensic metadata I/O must not become a whole-envelope read"
    );
    assert!(lines.iter().any(|line| {
        line.contains("LOAD STAGE=forensic-prelude scope=real-artifact-forensic evidence=non_authoritative status=OBSERVED")
    }));
    assert!(lines.iter().any(|line| {
        line.contains("LOAD STAGE=forensic-directory scope=real-artifact-forensic evidence=non_authoritative status=OBSERVED")
    }));
    assert!(lines.iter().any(|line| {
        line.contains("LOAD STAGE=forensic-census scope=real-artifact-forensic evidence=non_authoritative status=OBSERVED")
    }));
    assert!(lines.iter().any(|line| {
        line.contains("LOAD STAGE=l2-from-artifact scope=real-artifact-forensic evidence=non_authoritative status=BLOCKED comparison_profile=int8")
    }));
    eprintln!(
        "LOAD STAGE=forensic-metadata-bound scope=real-artifact-forensic evidence=non_authoritative status=OBSERVED metadata_bytes_read={} cap={}",
        report.metadata_bytes_read, FORENSIC_METADATA_READ_CAP
    );
}
