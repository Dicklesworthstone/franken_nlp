# G0 architecture-decision record schema

This document is the machine-readable contract for Phase-0 G0 architecture
decision records (ADRs). It is a pre-write: `scripts/check.sh` wiring, the
registry, the template, fixtures, and probe evidence are added only after the
Phase-0 scaffold is available. Until then, no dependent surface may treat a
G0 decision as ratified merely because this schema exists.

## Layout and identity

- G0 ADRs live directly in `docs/adr/`, one file per decision.
- A file is named `ADR-G0-<probe##>-<slug>.md`, where `<probe##>` is `01`
  through `11` and `<slug>` is lowercase ASCII words separated by single
  hyphens.
- The metadata `adr_id` is `G0-<probe##>` with an optional lowercase
  hyphenated sub-probe suffix. A sub-probe still names its parent probe in
  `g0_item.probe`.
- Raw evidence for one ADR lives under `docs/adr/evidence/<adr_id>/`. Evidence
  is append-only: a correction adds a new evidence file and new digest entry;
  it never overwrites archived bytes.
- `docs/adr/g0_registry.json` is canonical JSON. It has one registry row for
  every G0 item and at least one row for each primary probe 1 through 11.

## Metadata block

Every G0 ADR contains exactly one fenced block with the opening fence
` ```adr-metadata` and a JSON object as its body. The object is canonical JSON:
UTF-8, two-space indentation, sorted keys, no duplicate keys, and a trailing
newline. The validator rejects a second metadata block, malformed JSON, or a
non-canonical serialization.

```adr-metadata
{
  "adr_id": "G0-01",
  "blocked_surface": [],
  "decision": "One binding, evidence-scoped decision paragraph.",
  "evidence": [
    {
      "bytes": 123,
      "path": "evidence/G0-01/probe-transcript.txt",
      "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    }
  ],
  "exact_commands": [
    "exact command line that produced the evidence"
  ],
  "fallback": null,
  "g0_item": {
    "name": "probe name",
    "probe": 1
  },
  "host_pin": {
    "cpu_model": "observed CPU model or not-host-sensitive",
    "os": "observed operating system or not-host-sensitive",
    "rustc_vv_sha256": "SHA-256 of rustc -Vv output or not-host-sensitive"
  },
  "killed_alternatives": [
    {
      "name": "named alternative",
      "reason": "why the evidence killed it"
    }
  ],
  "source_pin": {
    "asupersync": "full revision or explicit not-applicable pin",
    "frankentorch": "full revision or explicit not-applicable pin"
  },
  "status": "RATIFIED"
}
```

The metadata object has these required fields.

| Field | Required shape and meaning |
| --- | --- |
| `adr_id` | `G0-<probe##>` with an optional lowercase hyphenated sub-probe suffix; unique across every ADR. |
| `g0_item` | Object with integer `probe` in 1..11 and non-empty `name`. |
| `status` | Exactly one of `RATIFIED`, `ABSENT-WITH-FALLBACK`, or `BLOCKED`. |
| `decision` | Non-empty, one-paragraph binding statement, scoped to the pins and evidence listed in this record. |
| `exact_commands` | Non-empty array of literal command-line strings that produced or replay the cited evidence. |
| `source_pin` | Non-empty object mapping each probed repository/suite component to its immutable revision or explicit not-applicable pin. |
| `host_pin` | Non-empty object that records CPU model, OS, and `rustc -Vv` digest for host-sensitive probes; a non-host-sensitive probe records that explicit applicability state. |
| `evidence` | Array of `{path, bytes, sha256}` records. Every path is relative to `docs/adr/`, remains under this ADR's `evidence/<adr_id>/` directory, and names a regular file with its observed byte length and SHA-256. |
| `killed_alternatives` | Non-empty array of `{name, reason}` records; no unnamed or unexplained alternative is accepted. |
| `fallback` | `null` or a non-empty replacement description. It is mandatory and non-empty for `ABSENT-WITH-FALLBACK`. |
| `blocked_surface` | Array of non-empty dependent-surface names. It is mandatory and non-empty for `BLOCKED`; each listed surface is forbidden from claiming readiness. |

Extensions are allowed only under `x_`-prefixed keys. A record may not use
`BLOCKED` as a silent omission: it must name the surfaces that remain blocked.
Likewise, `ABSENT-WITH-FALLBACK` must name the actual hand-rolled replacement,
not a future intention.

## Registry

`g0_registry.json` has this canonical shape:

```json
{
  "items": [
    {
      "adr_id": "G0-01",
      "adr_path": "ADR-G0-01-example.md",
      "g0_item": { "name": "probe name", "probe": 1 },
      "owner_bead": "franken_nlp-upv",
      "status": "RATIFIED"
    }
  ],
  "schema_version": 1
}
```

The registry and ADR metadata must agree byte-for-byte in semantic JSON terms
for `adr_id`, `g0_item`, and `status`. The validator rejects a registry row
without an ADR, an ADR without a row, duplicate IDs or paths, missing primary
probes, unknown statuses, and a path that escapes `docs/adr/`.

## Validation contract

`python3 scripts/validate_adrs.py` validates the completed tree. It prints the
item × status matrix to stdout. Detailed per-ADR diagnostics, including
missing fields and expected-versus-observed evidence digests, go to stderr,
ending with:

```text
ADR_REGISTRY RESULT=PASS|FAIL items=<n>/11 blocked=<comma-separated-adr-ids-or-none>
```

The validator is stdlib-only repository tooling. The scaffold wires this exact
command into `scripts/check.sh`; no release-binary dependency or model weight
is involved. A successful structural validation does not turn a `BLOCKED` ADR
into a ratified decision.

Before the scaffold supplies the committed registry and fixture tree,
`python3 scripts/validate_adrs.py --self-test` exercises the metadata-status
rules in memory and reports `mode=self-test`. It is a generator/validator smoke
check only; it is not evidence that a G0 registry exists or that any G0 probe
is ratified.
