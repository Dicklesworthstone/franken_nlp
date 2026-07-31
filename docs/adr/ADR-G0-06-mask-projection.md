# G0-06 mask and projection memory

```adr-metadata
{
  "adr_id": "G0-06",
  "blocked_surface": ["projection strategy", "grammar mask routing"],
  "decision": "The central-green accounting probe fixes the 166,144-row mask at 20,768 bytes and the full f32 logit vector at 664,576 bytes, while proving bitset-masked dense argmax equals row-sliced argmax for every preregistered legal-set size; its nanosecond emitters are explicitly local-measurement-only, so no dense/sparse crossover or production route is ratified.",
  "evidence": [],
  "exact_commands": ["cargo test --locked --test g0 probe6_mask_memory -- --nocapture", "python3 scripts/validate_adrs.py"],
  "fallback": null,
  "g0_item": {"name": "mask projection memory", "probe": 6},
  "host_pin": {"applicability": "local timing emitters only; no retained CPU/OS/rustc pin or raw timing transcript, so the host-sensitive crossover remains unratified"},
  "killed_alternatives": [{"name": "unmasked dense-prefix scan", "reason": "the probe scans the full 166,144-row bitset mask and asserts exact argmax equality against the row-sliced comparator"}, {"name": "dispatch selected from local nanoseconds", "reason": "the printed timing authority is explicitly local-measurement-only"}],
  "source_pin": {"g0_harness": "tests/g0.rs#sha256:3727dc031d1459bad9116f1fc6a0fb64a728d836b72a1dcd6fbbd547819e79b8", "probe_source": "tests/g0/probe6_mask_memory.rs#sha256:abe6bc832b16b9bc08e96c825f9098324ad4e55f120f8e2e9b8b90ada3043f17"},
  "status": "BLOCKED",
  "x_raw_probe_evidence": {"digest_scope": "SHA-256 of the exact result-emitting probe source; no raw timing transcript is retained under docs/adr/evidence/G0-06", "result_emitters": "tests/g0/probe6_mask_memory.rs:92-102", "sha256": "abe6bc832b16b9bc08e96c825f9098324ad4e55f120f8e2e9b8b90ada3043f17"},
  "x_unratified_rows": ["host-pinned dense-versus-sparse crossover", "production projection dispatch selection", "grammar-mask routing default"]
}
```

## Executable evidence

`tests/g0/probe6_mask_memory.rs:8-11` fixes the vocabulary and all seven legal
set sizes; `:60-91` asserts the two byte counts and exact dense/sparse argmax
agreement. The emitter at `:92-102` prints one local timing row per legal set
and `G0_PROBE6 … RESULT=PASS cases=7 authority=measurement-baseline-only`.
The source hash above is deliberately not represented as a retained timing
transcript.
