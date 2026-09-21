# Native INT8 model-backed redaction

`tasks::redact::quantized::Int8Redactor` connects the existing native INT8 NER
portfolio to the existing rules/NER union, independent source-occurrence checks,
transactional edits and residual verification. The eager redaction entry point
still requires BF16 evidence. A private closed profile selector connects the
shared pipeline; it is not inferred from a caller-provided result string.

## Execution contract

Construct the immutable redactor with a pinned `SourceTaskPlanner`, an explicit
strict-INT8 **NER** identity and `Int8RedactionConfig`. NER options, tokenizer,
trusted scaffold, model facts and task limits remain fixed. `redact` accepts the
original text, existing `RedactionRequest`, optional borrowed `Pseudonyms`, a
real `StrictInt8Engine`, reusable vocabulary and cancellation control. It cannot
accept a synthetic native driver or mutable task factory.

The first pass freshly compiles source-backed NER. The existing union independently
recovers every exact occurrence, including repeated/overlapping mentions, and
combines those with the selected rule detections. Existing masking, placeholders,
per-type actions, pseudonyms and opt-in byte/scalar edit maps are preserved. No
normalization changes the source; maps contain coordinates rather than raw
original matched values.

With `verify=true`, the edited text is freshly planned and run through the same
model, NER options and rules. Original-source offsets and plans are never reused.
Residual detections return a typed coordinate-only report and **no partial
redacted document**. Verification cannot silently downgrade to rules-only.
An oversized transformed document, exhausted second-pass budget, invalid evidence,
model failure or cancellation also returns no result. `verify=false` is explicit
and remains `NotRequested`, never a successful verification claim.

## Budgets and output

The two passes share checked, nonrefundable ceilings for forward positions,
projected logits, attention pairs, integer dot products, multiply-accumulates,
and grammar-mask visits. All required mask reservations must fit before the first
pass. Exact model work is reserved before each native attempt because the second
prompt does not exist until editing finishes. Early EOS does not restore that
reservation. No retry or third pass exists in the operation.

Each pass cold-prefills its own source on one reused engine. Native receipts are
checked against prompt/output geometry; independent occurrence validation runs
again before offsets can authorize edits. The complete outer `Int8RedactionRun`
size is checked, not only its text or nested result. It exports the native profile,
code-owned template digest, pass count and work accounting but no private NER
transcripts or prompt digests.
The public policy digest includes the native redaction version and NER options.

## Process-owned API

`NlpEngine::redact_int8` accepts owned source text, pinned planner/vocabulary Arcs,
a resident model, `hosted::RedactConfig`, optional `RedactionPseudonyms` and one
cancellation token. It runs planning, both native passes and transactional edits
inside one existing process-owned blocking invocation. It does not call the
single-request host twice or create another runtime, scheduler or model loader.

The real process ledger charges source capacity, explicitly priced preparation
and edit storage, one native KV/RoPE/scratch allocation, intermediate NER output,
and a separate complete-output reservation. The native allocation is reused,
not duplicated for verification. Temporary storage drains before physical
completion; `HostedOutput` retains the output charge for caller delivery.
Preparation/edit headroom must honestly cover the retained data structures and
allocator overhead. These commitments are not measured RSS or an allocator cap.
Hosted residual failures retain the count, not the potentially large coordinate
report: the report is drained while its temporary output reservation is still
owned. The borrowed native API exposes the full report to caller-owned admission.

The hosted pseudonym option explicitly uses full-256-bit HMAC and owns a shared
key handle plus namespace. It checks any saved key commitment, does not generate
keys, and never serializes raw key material. It does not invent a job-wide sealed
128-bit value set; callers needing that existing mode use the borrowed native
API with an already preflighted context. Caller-held key handles remain under
the caller's lifetime control. No secret is persisted by this API.

## Evidence and limitations

A clean declared detector union is **not proof that all PII is gone**. NER recall
and false positives require task evaluation; the same-model verification pass is
correlated, not independent proof or anonymity certification. Bounded rule scans,
tokenizer/grammar operations and edits are checked at call boundaries, not
preempted in the middle. No numerical, speed or quality improvement is claimed.

Regression source uses actual pinned planning and the existing edit pipeline
with private synthetic native receipts. Compilation, Rust tests, real-model
inference and production DSR were not run when this code was authored. Those
remain controller-owned under `WIRING.md`. This change adds library execution;
it does not activate public neural CLI, artifact publication or release gates.
