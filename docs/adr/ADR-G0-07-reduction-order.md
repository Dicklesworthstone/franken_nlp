# G0-07 reduction-order invariance

```adr-metadata
{
  "adr_id": "G0-07",
  "blocked_surface": ["batch scheduler", "batched reduction"],
  "decision": "No batch reduction order is ratified; batching remains blocked pending bitwise-invariance evidence.",
  "evidence": [],
  "exact_commands": ["python3 scripts/validate_adrs.py"],
  "fallback": null,
  "g0_item": {"name": "reduction-order invariance", "probe": 7},
  "host_pin": {"applicability": "host-sensitive arithmetic probe not yet run"},
  "killed_alternatives": [{"name": "unratified reduction order", "reason": "probe evidence is absent"}],
  "source_pin": {"model shape": "not-yet-probed"},
  "status": "BLOCKED"
}
```
