# ExecutionIdentity compatibility matrix

`ExecutionIdentity` v1 is the only authority for semantic cache, durable job,
receipt, golden, and calibration keys. The projection constructors in
`src/execution_identity.rs` are the only permitted key builders. A `YES` means
that changing the field invalidates that projection; a `NO` means its bytes are
deliberately absent from the named projection. The integration test parses this
table and mutates every field to prevent documentation/code drift.

`host_class` and `compiler_identity` are `YES` only for `fast-vN` profiles;
they are required there and forbidden for the other profiles.

| field | prefix_cache | semantic_job | receipt_identity | golden_fixture | calibration_artifact | rationale |
| --- | --- | --- | --- | --- | --- |
| schema_version | YES | YES | YES | YES | YES | Schema changes create a different canonical identity contract. |
| source_revision | YES | YES | YES | YES | YES | Upstream source behavior is semantic authority. |
| logical_model_digest | YES | YES | YES | YES | YES | Logical tensor bytes change model computation. |
| artifact_format | YES | YES | YES | YES | YES | Execution package format is part of deployed semantics and golden replay. |
| quant_recipe | YES | YES | YES | YES | YES | Quantization changes deployed numeric behavior. |
| packing_set_digest | YES | YES | YES | YES | YES | Packing rules can change effective execution semantics. |
| tokenizer_digest | YES | YES | YES | YES | YES | Token IDs and boundary behavior are authoritative. |
| template_digest | YES | YES | YES | YES | YES | Template bytes determine model input and task behavior. |
| task_spec | NO | YES | YES | YES | YES | Task policy changes job/output/calibration authority, not cached prefix states. |
| taskir_digest | NO | YES | YES | YES | YES | Compiled task semantics affect job and calibration policy. |
| prompt_digest | YES | YES | YES | YES | YES | Prompt bytes determine the prefix and comparison surface. |
| grammar_compiler_version | YES | YES | YES | YES | YES | Compiler behavior determines constrained-decode legality. |
| schema_digest | YES | YES | YES | YES | YES | Schema changes the constrained output language. |
| numerics_profile | YES | YES | YES | YES | YES | Every floating or quantized comparison is profile-scoped. |
| kv_dtype | YES | YES | YES | YES | YES | KV representation changes deployed state and golden replay. |
| sampler_version | NO | YES | YES | YES | YES | Sampling does not alter a cached prefix but changes generated streams. |
| thinking_mode | YES | YES | YES | YES | YES | Thinking markers change trusted template input. |
| tool_mode | YES | YES | YES | YES | YES | XML/JSON branches change trusted template input. |
| calibration_digest | NO | YES | YES | YES | YES | Selected calibration changes semantic job/receipt, golden, and calibration validity. |
| decision_policy_digest | NO | YES | YES | YES | YES | Decision policy changes task results but not the model prefix. |
| backend_semantic_version | YES | YES | YES | YES | YES | Backend behavior is part of named numerical semantics. |
| host_class | YES | YES | YES | YES | YES | Fast profiles are host-scoped; other profiles carry no host field. |
| compiler_identity | YES | YES | YES | YES | YES | Fast profiles are compiler-scoped; other profiles carry no compiler field. |

## Provenance boundary

`ProvenanceIdentity` is separate from `ExecutionIdentity`. It carries
`source_root_sha256`, `fnlpq_file_sha256`, `release_manifest_sha256`,
`license_bundle_sha256`, converter/build provenance, and publisher-attestation
status for artifact and release receipts. Correcting legal-notice bytes changes
the provenance receipt but never the logical-model digest or any semantic
projection.

| provenance field | artifact/release receipt | semantic caches/jobs/receipts/goldens/calibration | rationale |
| --- | --- | --- | --- |
| source_root_sha256 | YES | NO | Binds the upstream conversion closure rather than model computation. |
| fnlpq_file_sha256 | YES | NO | Binds a specific packaged artifact byte stream. |
| release_manifest_sha256 | YES | NO | Binds the release distribution metadata. |
| license_bundle_sha256 | YES | NO | Tracks legal bundle corrections outside logical-model authority. |
| converter_provenance | YES | NO | Binds conversion lineage for audit/replay. |
| build_provenance | YES | NO | Binds binary/build provenance for audit/replay. |
| publisher_attestation_status | YES | NO | Records publisher-verification status without changing computation. |
