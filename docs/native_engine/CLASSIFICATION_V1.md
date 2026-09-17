# Raw-text classification and ordered batch execution

This is implemented **library code**, not production artifact activation or a
binary CLI command. Rust compilation, native model execution, classification
quality and calibration remain unverified by this increment. The model-free
fixtures use synthetic logits and a synthetic control census; neither is a
production authority.

## Supported workflow

`ClassificationPlanner::pinned` uses the pinned tokenizer and the host's archived
control registry. Set the task-owned template/tokenizer fields on the host's
real `classify-v1` identity, then plan a `ClassificationRequest` under an explicit
`PlanContext` and `ClassificationLimits`. The prepared bundle exposes its exact
private execution identity and aggregate work before admission.

`PreparedClassification::execute_eager_with_control` uses one already-admitted
`HfBf16EagerEngine`. All heads are preflighted against the complete resident KV
reservation and aggregate work budget before the first forward. Each prompt
uses the existing `EagerPrefixSession`, which clears logical KV before the next
head. Scored EOS contributes projection work but is not fed back into KV.

`execute_with_logits` supplies the same semantic pipeline to an embedding
backend through `ClassificationLogits`. The backend receives the exact typed
prompt segments, head index, continuation prefix and full-vocabulary projection
request. This route does not produce a native execution/work receipt.

## Label and decision semantics

`exclusive` scores the complete label set in one head. `multi_label` scores a
separate yes/no head for every label; multiple labels may be included, excluded,
or independently abstained. No softmax is taken across labels. Each result
retains all candidate scores and declares `uncalibrated`. Candidate-relative
weights are not calibrated correctness or membership probabilities.

Labels have exact unique UTF-8 IDs and descriptions. Their original spelling,
capitalization and whitespace are not normalized. Canonically ordered labels
map to internal TaskIR-safe identifiers, and those identifiers are restored to
the original IDs in the complete public ranking and score records. Response
codes are opaque equal-width A–Z strings, with exact byte-fallback tokenization;
large codebooks use explicit multi-token continuations plus scored EOS. Changing
a label definition changes execution identity. Reordering labels does not.

Caller text, IDs and descriptions never enter the trusted template renderer.
The renderer sees fixed instructions and internal slots; metadata and original
document bytes are spliced afterward through the control-excluding encoder.
This is token/control separation, not a claim of immunity to semantic prompt
injection. Metadata escaping, repeated prompt storage and output label expansion
all count against explicit bounds.

## NDJSON batches

`batch::classify::ClassificationBatchPlanner` accepts `BatchDocument.text` as the
original document. `task_args` contains `labels`, `mode`, `policy` and `budget`.
Omitted arguments use explicitly supplied host defaults; no labels or policy are
invented. Unknown fields, token sequences, EOS overrides and supplied identities
are rejected. Every budget may only shrink the frozen host ceiling.

Example record shape (not a binary invocation):

```json
{"id":"ticket-1","text":"Please refund the duplicate charge.","task_args":{"labels":[{"id":"Billing issue","description":"Questions about charges and invoices"},{"id":"Refund requested","description":"The customer requests a refund"}],"mode":"multi_label","policy":{"minimum_candidate_weight_ppm":600000,"minimum_margin_ppm":200000},"budget":{"max_input_tokens":4096,"max_output_tokens":8,"max_output_bytes":1048576,"max_grammar_states":1024,"max_kv_bytes":2147483648}}}
{"flush":true}
```

Construct `NativeClassificationBatch` from that planner, an existing admitted
engine, the host's genuine `ClassificationBatchAdmission` implementation and a
whole-adapter `BatchWork` allowance. Pass it to the existing `batch::run_ndjson`
with the reader, writer, transport limits and cancellation controller. It uses
the existing `fnlp-item-local-batch-v1` protocol, not a parallel transport or an
unreviewed extension of the frozen robot schema.

Each document yields one complete exclusive or multi-label bundle. A failed
binary head does not emit provisional label successes or become an abstention.
Malformed or bounded item refusals may continue; cancellation, broken model
state, identity substitution and invalid execution receipts stop admission.

The runner and native adapter independently constrain the same work. Both keep
charges across flush epochs. The adapter additionally retains them across
separate runner calls; rejected admission, failed execution and output refusal
do not refund an attempted request. An unwind or fatal error leaves the adapter
unusable. A prepared value from another host/limit configuration is refused
before admission, even when passed through the Rust API rather than JSON.

Successful results carry the actual host guard through serialization, write and
flush. Partial writes or failed flushes stop the runner without retrying or
appending an error to uncertain output. Local `Write::flush` is not proof of
remote processing, stable-storage durability or exactly-once delivery. Reusing
an output stream or retrying uncertain delivery remains a host-level decision.

## Evidence boundaries

The regression sources cover raw planning, complete scorer/transport wiring,
Unicode labels and documents, hostile marker text, original-label restoration,
independent decisions, aggregate admission, cross-epoch budgets, factory
substitution, cancellation and guard lifetime through sink failures. Native
capacity/work checks also have model-free unit tests. The Rust tests are written,
not executed by this code-first session. Independent Python specification checks
and source/whitespace inspection cannot substitute for the repository's required
clean-SHA DSR checkpoint, model-present execution or task-quality qualification.
