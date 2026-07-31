# FrankenNLP research-decision register

This registry records only decisions promoted to a frozen implementation or
scope contract. A resolved design question is not, by itself, implementation
or runtime proof.

| Id | Status | Decision | Authority |
| --- | --- | --- | --- |
| OQ-31 | RESOLVED | FNLPQ v1 uses a prelude-bounded canonical ASCII JSON header plus an 80-byte binary section directory for byte ranges; six domain-separated identities; duplicate-rejecting parsing; explicit packing-set multiplicity; and pre-allocation hostile-input limits. | [ADR OQ-31](adr/OQ-31-fnlpq-envelope-review.md), [FNLPQ envelope v1](specs/fnlpq-envelope-v1.md) |
| P7-METAL | DEFERRED | `ft-kernel-metal` may enter only after CPU parity certification, as opt-in prefill under the independent `metal-prefill-v1` profile; CPU remains the authority and portable floor. | [P7 Metal scope gate](adr/drafts/p7-metal-prefill-scope.md) |
| P7-SERVE | DEFERRED | A resident process or Metal track does not authorize `fnlp serve` or any routable listener. | [P7 Metal scope gate](adr/drafts/p7-metal-prefill-scope.md#adjacent-scope-decisions) |
| P7-TRANSLATION | DEFERRED | Translation remains closed until frozen multilingual evaluation and task-quality evidence exist. | [P7 Metal scope gate](adr/drafts/p7-metal-prefill-scope.md#adjacent-scope-decisions) |
