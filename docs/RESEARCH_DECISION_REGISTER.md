# FrankenNLP research-decision register

This registry records only decisions promoted to a frozen implementation
contract. A resolved design question is not, by itself, implementation or
runtime proof.

| Id | Status | Decision | Authority |
| --- | --- | --- | --- |
| OQ-31 | RESOLVED | FNLPQ v1 uses a prelude-bounded canonical ASCII JSON header plus an 80-byte binary section directory for byte ranges; six domain-separated identities; duplicate-rejecting parsing; explicit packing-set multiplicity; and pre-allocation hostile-input limits. | [ADR OQ-31](adr/OQ-31-fnlpq-envelope-review.md), [FNLPQ envelope v1](specs/fnlpq-envelope-v1.md) |
