# G0-11 asupersync leverage census

```adr-metadata
{
  "adr_id": "G0-11",
  "blocked_surface": ["scheduler", "daemon", "jobs", "pull proofs"],
  "decision": "No asupersync leverage claim is ratified; every dependent surface remains blocked pending the ownership census evidence.",
  "evidence": [],
  "exact_commands": ["python3 scripts/validate_adrs.py"],
  "fallback": null,
  "g0_item": {"name": "asupersync leverage census", "probe": 11},
  "host_pin": {"applicability": "pin census not yet run"},
  "killed_alternatives": [{"name": "unratified foundation claim", "reason": "census evidence is absent"}],
  "source_pin": {"asupersync": "not-yet-probed"},
  "status": "BLOCKED"
}
```
