# G0-05 loop boundary fixture

```adr-metadata
{
  "adr_id": "G0-05",
  "blocked_surface": ["native forward", "KV layout"],
  "decision": "The central-green structural probe establishes the required 22-layer then final-RMSNorm then 22-layer then final-RMSNorm schedule, the 44 logical KV slots, and the rule that loop two receives loop one's normalized state; it does not ratify a numerical native forward or release the Phase -1 trace-provenance gate.",
  "evidence": [],
  "exact_commands": ["cargo test --locked --test g0 probe5_loop_boundary -- --nocapture", "python3 scripts/validate_adrs.py"],
  "fallback": null,
  "g0_item": {"name": "loop boundary scalar fixture", "probe": 5},
  "host_pin": {"applicability": "fixture-structure and scalar-schedule only; no host-sensitive numerical or performance conclusion"},
  "killed_alternatives": [{"name": "22-deep or single-final-norm schedule", "reason": "the scalar runner asserts both norms, slot 22 for loop-two layer zero, and all 44 logical visits"}],
  "source_pin": {"g0_harness": "tests/g0.rs#sha256:3727dc031d1459bad9116f1fc6a0fb64a728d836b72a1dcd6fbbd547819e79b8", "loop_runner": "src/native_engine/looprun.rs#sha256:b112f0a64bb6bf2843f241ccfabb581013ec10fd2395f3945af31f8255ebc17f", "probe_source": "tests/g0/probe5_loop_boundary.rs#sha256:f5a1ff1dcf035e4f0bbcbd3ec58e49df43bb0bbaa780d2cc88831617cbc44a57", "trace_fixture": "tests/fixtures/reference/hf-bf16-eager/prompt-000/trace.json#sha256:11f6ba511ea52a0725d4077d98f3a9ce3aa88b0d1b0f55ae62e728131e04212c"},
  "status": "BLOCKED",
  "x_raw_probe_evidence": {"digest_scope": "SHA-256 of the exact result-emitting probe source; no raw runtime transcript is retained under docs/adr/evidence/G0-05", "result_emitters": "tests/g0/probe5_loop_boundary.rs:107-109,134-137", "sha256": "f5a1ff1dcf035e4f0bbcbd3ec58e49df43bb0bbaa780d2cc88831617cbc44a57"},
  "x_unratified_rows": ["Phase -1 trace replay provenance", "native numerical forward parity", "product KV allocation and scheduling"]
}
```

## Executable evidence

`tests/g0/probe5_loop_boundary.rs:92-106` asserts the two norm inputs and
the explicit `(loop, layer, kv_slot, hidden)` boundary values, including
loop-two layer zero at slot 22. `tests/g0/probe5_loop_boundary.rs:113-137`
checks both prefill and append fixture phases for all 44 layer/K/V slots and
two norm taps, then emits the two `G0_PROBE5 … RESULT=PASS` lines. The source
digest binds those emitters rather than claiming an unretained raw transcript.
