# G0-02 blocking and scoped CPU seam

```adr-metadata
{
  "adr_id": "G0-02",
  "blocked_surface": ["engine request scheduling", "batch scheduler"],
  "decision": "No blocking-pool or scoped-CPU seam is ratified; the named scheduling surfaces remain blocked pending the owning probe evidence.",
  "evidence": [],
  "exact_commands": ["python3 scripts/validate_adrs.py"],
  "fallback": null,
  "g0_item": {"name": "spawn_blocking scoped_cpu seam", "probe": 2},
  "host_pin": {"applicability": "host-sensitive probe not yet run"},
  "killed_alternatives": [{"name": "unratified scheduler seam", "reason": "probe evidence is absent"}],
  "source_pin": {"asupersync": "not-yet-probed"},
  "status": "BLOCKED"
}
```
