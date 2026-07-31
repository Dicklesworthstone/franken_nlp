# G0-08 converter range and RSS

```adr-metadata
{
  "adr_id": "G0-08",
  "blocked_surface": ["converter streaming"],
  "decision": "The central-green range model fixes a 64 MiB panel cap, rejects overflow, out-of-source, and over-cap ranges, and accounts for the six peak-RSS formula inputs at a 340 MiB synthetic estimate; it intentionally does not measure process RSS or a shard-scale source, so converter streaming remains blocked pending a retained host measurement.",
  "evidence": [],
  "exact_commands": ["cargo test --locked --test g0 probe8_converter_rss -- --nocapture", "python3 scripts/validate_adrs.py"],
  "fallback": null,
  "g0_item": {"name": "converter range RSS", "probe": 8},
  "host_pin": {"applicability": "range-model-only; process RSS, shard-scale source, CPU/OS/rustc pin, and raw measurement transcript are not retained"},
  "killed_alternatives": [{"name": "unbounded or whole-file range access", "reason": "checked ranges reject arithmetic overflow, source escape, and requests above the fixed 64 MiB panel cap"}, {"name": "RSS claim from the 340 MiB formula", "reason": "the probe explicitly states that it does not report process RSS"}],
  "source_pin": {"g0_harness": "tests/g0.rs#sha256:3727dc031d1459bad9116f1fc6a0fb64a728d836b72a1dcd6fbbd547819e79b8", "probe_source": "tests/g0/probe8_converter_rss.rs#sha256:036465d814a9fce5f75c2eaa4f2fa67b51add6e196cf80e2a720f442d55fa669"},
  "status": "BLOCKED",
  "x_raw_probe_evidence": {"digest_scope": "SHA-256 of the exact result-emitting probe source; no raw process-RSS transcript is retained under docs/adr/evidence/G0-08", "result_emitters": "tests/g0/probe8_converter_rss.rs:99-108", "sha256": "036465d814a9fce5f75c2eaa4f2fa67b51add6e196cf80e2a720f442d55fa669"},
  "x_unratified_rows": ["shard-scale converter RSS", "whole-copy baseline comparison", "converter streaming release gate"]
}
```

## Executable evidence

`tests/g0/probe8_converter_rss.rs:7-31` defines the cap and checked six-term
formula; `:65-98` proves the range refusals and the 340 MiB synthetic total.
The PASS emitters at `:99-108` print all six formula inputs and label their
authority `range-model-only`, so they cannot be used as a measured-RSS claim.
