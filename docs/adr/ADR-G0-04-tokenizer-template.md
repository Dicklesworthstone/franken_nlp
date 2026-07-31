# G0-04 tokenizer and template spike

```adr-metadata
{
  "adr_id": "G0-04",
  "blocked_surface": ["tokenizer", "template builder"],
  "decision": "The central-green probe establishes that the frozen auxiliary corpus contains the four required tokenizer cases and five required template cases, each with the required token-id and rendered-byte bindings; it is fixture-contract evidence only, so production tokenizer/template exactness remains blocked pending the implementation-level differential against the Phase -1 oracle.",
  "evidence": [],
  "exact_commands": ["cargo test --locked --test g0 probe4_tok_tpl -- --nocapture", "python3 scripts/validate_adrs.py"],
  "fallback": null,
  "g0_item": {"name": "tokenizer template spike", "probe": 4},
  "host_pin": {"applicability": "fixture-contract-only; no host-sensitive numerical or performance conclusion"},
  "killed_alternatives": [{"name": "unbound L0 fixture corpus", "reason": "the probe requires every named case and non-empty token ids plus rendered-byte and token-id digest fields"}],
  "source_pin": {"auxiliary_fixture": "tests/fixtures/reference/auxiliary.json#sha256:a11e5395d884f5a600887c2266545892c6bb37ec5a79e6dbdacdf5d554a1e0a9", "g0_harness": "tests/g0.rs#sha256:3727dc031d1459bad9116f1fc6a0fb64a728d836b72a1dcd6fbbd547819e79b8", "probe_source": "tests/g0/probe4_tok_tpl.rs#sha256:b4ac4ae15c6573c726936ede8b2b33d4bd9f20d6acbab5b8cb0da257a5c6184f"},
  "status": "BLOCKED",
  "x_raw_probe_evidence": {"digest_scope": "SHA-256 of the exact result-emitting probe source; no raw runtime transcript is retained under docs/adr/evidence/G0-04", "result_emitters": "tests/g0/probe4_tok_tpl.rs:101-106", "sha256": "b4ac4ae15c6573c726936ede8b2b33d4bd9f20d6acbab5b8cb0da257a5c6184f"},
  "x_unratified_rows": ["implementation-level SentencePiece-BPE token-id differential", "implementation-level chat-template rendered-byte differential", "Phase -1 oracle replay provenance"]
}
```

## Executable evidence

`tests/g0/probe4_tok_tpl.rs:14-26` names the four tokenizer and five template
cases. `tests/g0/probe4_tok_tpl.rs:70-98` requires the token-id and
rendered-byte digests. Its central-suite PASS emitters are
`tests/g0/probe4_tok_tpl.rs:101-106` (`G0_PROBE4 … RESULT=PASS`, authority
`fixture-contract-only`). No runtime transcript is represented as an evidence
artifact; the SHA-256 above binds the exact source that emitted the result.
