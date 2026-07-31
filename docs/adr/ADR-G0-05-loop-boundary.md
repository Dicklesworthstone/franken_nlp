# G0-05 loop boundary fixture

```adr-metadata
{
  "adr_id": "G0-05",
  "blocked_surface": ["native forward", "KV layout"],
  "decision": "No loop-boundary execution contract is ratified; forward and KV layout remain blocked pending trace-tap evidence.",
  "evidence": [],
  "exact_commands": ["python3 scripts/validate_adrs.py"],
  "fallback": null,
  "g0_item": {"name": "loop boundary scalar fixture", "probe": 5},
  "host_pin": {"applicability": "not-host-sensitive bootstrap record"},
  "killed_alternatives": [{"name": "unratified forward loop", "reason": "probe evidence is absent"}],
  "source_pin": {"reference fixtures": "not-yet-probed"},
  "status": "BLOCKED"
}
```
