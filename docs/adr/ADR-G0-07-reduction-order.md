# G0-07 reduction-order invariance

```adr-metadata
{
  "adr_id": "G0-07",
  "blocked_surface": ["batch scheduler", "batched reduction"],
  "decision": "The central-green scalar probe fixes canonical per-row, increasing-column accumulation as the only tested bitwise batch-M equivalent to batch-1 for K={1024,3072,6144,10752} and M={1,8,64}; it also retains a concrete reassociation counterexample, while vectorized or otherwise reassociated reduction remains unratified.",
  "evidence": [],
  "exact_commands": ["cargo test --locked --test g0 probe7_reduction_order -- --nocapture", "python3 scripts/validate_adrs.py"],
  "fallback": null,
  "g0_item": {"name": "reduction-order invariance", "probe": 7},
  "host_pin": {"applicability": "scalar f32 operation-order contract; no host-performance conclusion"},
  "killed_alternatives": [{"name": "reverse or reassociated row reduction", "reason": "the probe compares bit patterns across all named K/M shapes and preserves a 1.0-versus-0.0 reassociation counterexample"}],
  "source_pin": {"g0_harness": "tests/g0.rs#sha256:3727dc031d1459bad9116f1fc6a0fb64a728d836b72a1dcd6fbbd547819e79b8", "probe_source": "tests/g0/probe7_reduction_order.rs#sha256:8f8f7bb59f2fa4f1e71844e5c447e9d91073ec7b9b33d296ccdac4776789b593"},
  "status": "BLOCKED",
  "x_raw_probe_evidence": {"digest_scope": "SHA-256 of the exact result-emitting probe source; no raw runtime transcript is retained under docs/adr/evidence/G0-07", "result_emitters": "tests/g0/probe7_reduction_order.rs:63-75,99-105", "sha256": "8f8f7bb59f2fa4f1e71844e5c447e9d91073ec7b9b33d296ccdac4776789b593"},
  "x_unratified_rows": ["vectorized reduction equivalence", "reassociated reduction equivalence", "production batch scheduler implementation"]
}
```

## Executable evidence

`tests/g0/probe7_reduction_order.rs:7-35` names the four K shapes, three
batch widths, and canonical scalar accumulator. `:37-76` asserts bitwise
batch equivalence and emits the `G0_PROBE7 … RESULT=PASS` matrix; `:79-105`
retains the named reassociation counterexample and its separate PASS line.
The hash binds the exact emitter source, not an unretained central-suite
transcript.
