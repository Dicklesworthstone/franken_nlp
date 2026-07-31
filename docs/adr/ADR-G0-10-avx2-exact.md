# G0-10 Zen-3 AVX2 exactness

```adr-metadata
{
  "adr_id": "G0-10",
  "blocked_surface": ["AVX2 kernel dispatch"],
  "decision": "No AVX2 construction is ratified; AVX2 dispatch remains blocked pending exactness and host-throughput evidence.",
  "evidence": [],
  "exact_commands": ["python3 scripts/validate_adrs.py"],
  "fallback": null,
  "g0_item": {"name": "Zen-3 AVX2 exactness", "probe": 10},
  "host_pin": {"applicability": "Zen-3 Threadripper probe not yet run"},
  "killed_alternatives": [{"name": "unratified AVX2 route", "reason": "probe evidence is absent"}],
  "source_pin": {"kernel candidate": "not-yet-probed"},
  "status": "BLOCKED"
}
```
