# G0 architecture decision record

Replace every placeholder before ratification.  A `BLOCKED` record is allowed
only while it names every downstream surface that cannot claim readiness.

```adr-metadata
{
  "adr_id": "G0-01",
  "blocked_surface": [
    "named downstream surface"
  ],
  "decision": "One evidence-scoped binding decision paragraph.",
  "evidence": [],
  "exact_commands": [
    "exact command line that produced or replayed the listed evidence"
  ],
  "fallback": null,
  "g0_item": {
    "name": "probe name",
    "probe": 1
  },
  "host_pin": {
    "applicability": "not-host-sensitive or the observed CPU/OS/rustc -Vv digest"
  },
  "killed_alternatives": [
    {
      "name": "named alternative",
      "reason": "why evidence rejected it"
    }
  ],
  "source_pin": {
    "component": "immutable revision or explicit not-applicable pin"
  },
  "status": "BLOCKED"
}
```

## Decision

Write the decision and its authority boundary here. Evidence remains under
`evidence/<adr_id>/` and is append-only.
