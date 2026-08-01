# FrankenNLP research-decision register

This registry records frozen implementation/scope contracts **and explicit
quarantines or reopenings that revoke a formerly claimed contract**. A resolved
design question is not, by itself, implementation or runtime proof; a reopened
row authorizes only the stated freeze/refusal until owner ratification and new
evidence restore a single authority.

| Id | Status | Decision | Authority |
| --- | --- | --- | --- |
| OQ-31 | AUTHORITY CONFLICT / REOPENED | Three incompatible v1 families simultaneously claimed frozen authority: an 80-byte-per-entry binary directory, a 256-byte-per-entry binary section table, and JSON-only ranges. None may authorize implementation or acceptance until the owner ratifies one choice and `franken_nlp-g6f` records it, recomputes candidate-bound field-inventory and hostile-corpus digests, freezes distinct raw-file versus domain-framed identities, and marks the rejected records historical. | [80-byte-entry candidate](adr/OQ-31-fnlpq-envelope-review.md), [256-byte-entry candidate](adr/0001-fnlpq-envelope-oq-31.md), [JSON-only candidate](adr/0031-fnlpq-envelope-review.md) |
| P7-METAL | DEFERRED | `ft-kernel-metal` may enter only after CPU parity certification, as opt-in prefill under the independent `metal-prefill-v1` profile; CPU remains the authority and portable floor. | [P7 Metal scope gate](adr/drafts/p7-metal-prefill-scope.md) |
| P7-SERVE | DEFERRED | A resident process or Metal track does not authorize `fnlp serve` or any routable listener. | [P7 Metal scope gate](adr/drafts/p7-metal-prefill-scope.md#adjacent-scope-decisions) |
| P7-TRANSLATION | DEFERRED | Translation remains closed until frozen multilingual evaluation and task-quality evidence exist. | [P7 Metal scope gate](adr/drafts/p7-metal-prefill-scope.md#adjacent-scope-decisions) |
