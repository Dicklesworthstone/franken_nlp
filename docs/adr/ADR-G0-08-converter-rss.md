# G0-08 converter range and RSS

```adr-metadata
{
  "adr_id": "G0-08",
  "blocked_surface": ["converter streaming"],
  "decision": "No converter range-access or peak-RSS contract is ratified; converter streaming remains blocked pending measurements.",
  "evidence": [],
  "exact_commands": ["python3 scripts/validate_adrs.py"],
  "fallback": null,
  "g0_item": {"name": "converter range RSS", "probe": 8},
  "host_pin": {"applicability": "host-sensitive measurement not yet run"},
  "killed_alternatives": [{"name": "unratified converter stream", "reason": "probe evidence is absent"}],
  "source_pin": {"safetensors fixture": "not-yet-probed"},
  "status": "BLOCKED"
}
```
