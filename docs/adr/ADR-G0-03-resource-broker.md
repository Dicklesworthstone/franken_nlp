# G0-03 EngineResources broker

```adr-metadata
{
  "adr_id": "G0-03",
  "blocked_surface": ["EngineResources admission"],
  "decision": "No EngineResources broker behavior is ratified; admission remains blocked pending the owning probe evidence.",
  "evidence": [],
  "exact_commands": ["python3 scripts/validate_adrs.py"],
  "fallback": null,
  "g0_item": {"name": "EngineResources broker", "probe": 3},
  "host_pin": {"applicability": "host-sensitive memory probe not yet run"},
  "killed_alternatives": [{"name": "unratified resource broker", "reason": "probe evidence is absent"}],
  "source_pin": {"asupersync": "not-yet-probed"},
  "status": "BLOCKED"
}
```
