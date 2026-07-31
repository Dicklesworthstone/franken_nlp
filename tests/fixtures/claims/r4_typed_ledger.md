## PERF-R4-TYPED-001

- Claim ID: r4-fixture-claim
- Evidence: r4-receipt=tests/fixtures/claims/r4_typed_measurement.json#sha256:3435af543ebf8aa97b7dd65b7e2d81b619b5e6e7dccd0d9960adfe1f35f06630; admission-receipt=tests/fixtures/claims/r4_typed_admission.json#sha256:46fc333b0097bd14a6fc8b07ed31c6d9e3537293835565103e1573151354c12d
- Fixture hashes: sha256:3435af543ebf8aa97b7dd65b7e2d81b619b5e6e7dccd0d9960adfe1f35f06630, sha256:46fc333b0097bd14a6fc8b07ed31c6d9e3537293835565103e1573151354c12d
- CPU feature string: fixture-cpu-v1
- Command + environment: fixture command
- Disposition: won
- Regime: R4-long-context
- Host fingerprint: fixture-host-v1
- Artifact recipe + packing + kernel table + load mode: recipe_id=r4-fixture-recipe; packing_sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa; kernel_table_sha256=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb; load_mode=resident
- Context point: tokens=32768; kv_dtype=bf16
- p50/p95/p99: prefill_ms(p50=4100,p95=4300,p99=4400); decode_tokens_per_s(p50=9.5,p95=10,p99=12.5)
- R4 measurement summary: kv_bytes=5905580032; peak_rss_bytes=9126805504
- Fairness controls: fixture controls
- Admission boundary outcomes: below_cap(tokens=32767,outcome=admitted,committed_bytes=9990000000,peak_bytes=9490000000); at_cap(tokens=32768,outcome=admitted,committed_bytes=10000000000,peak_bytes=9500000000); above_cap(tokens=32769,outcome=rejected,rejection=context_cap_exceeded)
- Bandwidth denominator: measured-dram-bytes
