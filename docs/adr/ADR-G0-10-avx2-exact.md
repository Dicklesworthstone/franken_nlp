# G0-10 Zen-3 AVX2 exactness

```adr-metadata
{
  "adr_id": "G0-10",
  "blocked_surface": ["AVX2 kernel dispatch"],
  "decision": "The central-green portable scalar model proves both required exact constructions—X3a low-7/high-bit decomposition and X3b widened signed-i16 multiplication—equal scalar/i64 across every i8 pair and adversarial K=10,752 vectors within the stated pair and aggregate bounds; raw saturating vpmaddubsw remains banned, while AVX2 dispatch is HOST-BLOCKED solely on retained throughput measurement from the required real Zen-3 Threadripper.",
  "evidence": [],
  "exact_commands": ["cargo test --locked --test g0 probe10_avx2_exact -- --nocapture", "python3 scripts/validate_adrs.py"],
  "fallback": null,
  "g0_item": {"name": "Zen-3 AVX2 exactness", "probe": 10},
  "host_pin": {"observed_host": "ARM64 macOS (orchestrator-declared); this host cannot supply Zen-3 AVX2 throughput evidence", "required_host": "real AMD Zen-3 Threadripper with retained CPU model, OS, rustc -Vv SHA-256, raw throughput transcript, and sustained-distribution methodology"},
  "killed_alternatives": [{"name": "raw saturating vpmaddubsw", "reason": "its i16 lane saturation is not repaired by accumulation cadence and is not an exact construction"}, {"name": "untested AVX2 dispatch selection", "reason": "portable exactness is proven, but real Zen-3 throughput has no retained host evidence"}],
  "source_pin": {"g0_harness": "tests/g0.rs#sha256:3727dc031d1459bad9116f1fc6a0fb64a728d836b72a1dcd6fbbd547819e79b8", "probe_source": "tests/g0/probe10_avx2_exact.rs#sha256:a554d2c14dd3d83aec2b78190e03ef0401dfcb860e2f269546f9a3149c83b97f"},
  "status": "BLOCKED",
  "x_raw_probe_evidence": {"digest_scope": "SHA-256 of the exact result-emitting probe source; no raw Zen-3 throughput transcript is retained under docs/adr/evidence/G0-10", "result_emitters": "tests/g0/probe10_avx2_exact.rs:54-63,66-119", "sha256": "a554d2c14dd3d83aec2b78190e03ef0401dfcb860e2f269546f9a3149c83b97f"},
  "x_unratified_rows": ["real Zen-3 Threadripper X3a versus X3b throughput comparison", "sustained p50/p95/p99 AVX2 dispatch evidence", "AVX2 default dispatch selection"]
}
```

## Executable evidence

`tests/g0/probe10_avx2_exact.rs:7-14` states K=10,752 and the exact pair and
aggregate bounds. X3a is modeled at `:23-33`, X3b at `:47-52`, every i8 pair
is checked at `:54-63`, and adversarial full-K/i64 plus pair-bound checks are
at `:66-119`. Those emit `G0_PROBE10 … RESULT=PASS` with
`authority=scalar-model-only`; they establish portable exactness, not
throughput. The sole remaining owner-hardware leg is the real Zen-3
Threadripper measurement named in `host_pin`.
