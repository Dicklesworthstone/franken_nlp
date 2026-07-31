## PERF-R4-TYPED-001

- Claim ID: r4-fixture-claim
- Evidence: r4-receipt=docs/../tests/fixtures/claims/r4_typed_measurement.json#sha256:7d0bffb30199347f6c39fbcaedcdf28c60f9b333e3a48546b1ca0cf7da2e6f17; admission-receipt=docs/../tests/fixtures/claims/r4_typed_admission.json#sha256:c3edd0874eb75036d18e1d88f170dc6d10781fc86478c8d1a431efa2d595e041
- Fixture hashes: sha256:7d0bffb30199347f6c39fbcaedcdf28c60f9b333e3a48546b1ca0cf7da2e6f17, sha256:c3edd0874eb75036d18e1d88f170dc6d10781fc86478c8d1a431efa2d595e041
- CPU feature string: fixture-cpu-v1
- Command + environment: fixture command
- Disposition: won
- Regime: R4-long-context
- Host fingerprint: fixture-host-v1
- Artifact recipe + packing + kernel table + load mode: recipe_id=r4-fixture-recipe; packing_sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa; kernel_table_sha256=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb; load_mode=resident
- Context point: tokens=32768; kv_dtype=bf16
- p50/p95/p99: prefill_ms(p50=4100,p95=4300,p99=4400); decode_tokens_per_s(p50=12.5,p95=10,p99=9.5)
- R4 measurement summary: kv_bytes=5905580032; peak_rss_bytes=9126805504
- Fairness controls: fixture controls
- Admission boundary outcomes: outcome=admitted; committed_bytes=10000000000; peak_bytes=9500000000
- Bandwidth denominator: measured-dram-bytes
