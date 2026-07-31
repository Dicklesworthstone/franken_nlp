# OQ-35 supervision and Suite-substrate non-adoption record

Status: **no adoption**

Related G0 record: `ADR-G0-11-asupersync-census.md` remains `BLOCKED`.

## Decision

No supervisor, registry/name-lease surface, `franken_kernel`,
`franken_evidence`, `franken_decision`, or `frankenlab` crate is adopted by
this census. Presence in the neighbouring FrankenSuite workspace is not a
FrankenNLP dependency decision. A later adoption requires a named product
use, source and compatibility review, an exact `SUITE.lock` entry, and a
replayed executable receipt.

## SUITE.lock adoption note

`SUITE.lock` records the selected asupersync revision
`362dc5b174427f66cfa76ab2bdd68cce1a95c6cc` as an optional
`asupersync-runtime` dependency. It contains no dependency row for
`franken_kernel`, `franken_evidence`, `franken_decision`, or `frankenlab`.
This ADR intentionally makes **no generated-lock edit**: adding any one of
those rows would falsely imply an approved product adoption.

## Fallback

Until an adoption ADR is ratified, FrankenNLP uses an owned request-region
tree with an identified restart owner, explicit lifecycle acknowledgements,
and no implicit global registry. Evidence records remain local project data;
they do not acquire a suite-schema dependency by implication.
