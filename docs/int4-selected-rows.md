# Grammar-first selected-row INT4 source tasks

The native grouped-INT4 decoder can explicitly project only the output rows
that the current JSON grammar permits. This uses the existing quantized model,
native session, selected-row kernels and source validator. Full-vocabulary
execution remains the default and the reference for qualification.

## Candidate CLI

One-off schema extraction exposes the strategy explicitly:

```sh
fnlp candidate source-int4 extract document.txt \
  --schema schema.json \
  --snapshot /models/nanbeige-g128.int4snap \
  --expected-model-sha256 "$MODEL_CONTENT_SHA256" \
  --group-width 128 --memory-mib "$MEMORY_MIB" \
  --max-new-tokens 128 --max-selected-rows 256 \
  --prefill-rows 4
```

The source-directory/manifest acquisition route also works. The optional
`--prefill-rows` changes prompt scheduling and composes with selected rows;
omitting it preserves sequential prefill. Add `--source-membership` for a
schema with annotated verbatim fields. The row cap must cover the complete
legal token set at every visited grammar state; this example cap is not a
promise that an arbitrary schema fits.

`--max-selected-rows` is available on one-off `extract`, `ner`, `keyphrases`,
`summarize` and `answer`. Shared batches, maps, durable jobs, redaction and the
semantic second-reader route reject it. Without the option, these tasks retain full-vocabulary execution and their
existing identity. Native work in the result reports actual projected rows;
grammar-mask, task, source and complete-result limits are unchanged.

For example, the same source-bound NER task can opt into the strategy:

```sh
fnlp candidate source-int4 ner document.txt \
  --snapshot /models/nanbeige-g128.int4snap \
  --expected-model-sha256 "$MODEL_CONTENT_SHA256" \
  --group-width 128 --memory-mib "$MEMORY_MIB" \
  --max-selected-rows 4096
```

The cap is a refusal boundary, not a prediction of how many rows a source
task will need. Summary prose and answer strings can admit wide token sets;
a small cap can refuse them. No task silently changes its schema or falls
back to the full head. NER and keyphrases retain exact source occurrences;
summaries and answers retain independently checked citations, passage
boundaries and their existing abstention semantics. A decoding failure does
not become an answer abstention.

## Library use

Seal the strategy before asking the host to admit the prepared extraction:

```rust
use franken_nlp::native_engine::constrained_int4::sparse::Int4JsonSparseLimits;

let prepared = prepared.with_selected_rows(Int4JsonSparseLimits {
    max_rows_per_step: 256,
})?;
```

The existing `NlpEngine::execute_int4_extract` and
`NlpEngine::execute_int4_extract_layer_major` entrypoints price the sealed
plan's work and dispatch its selected strategy. The layer-major variant also
requires its existing separate prompt-scratch reservation. Direct native
callers can use `constrained_int4::sparse::decode_json_int4_sparse` or
`sparse::prefill::decode_json_int4_sparse_layer_major` with the matching planned
work and native/KV budgets.

`PreparedInt4SourceTask::with_selected_rows` provides the same consuming
opt-in for built-in source tasks before hosted admission. The host's existing
serial and layer-major source entrypoints use the selected plan's work.
Its original typed finalizer stays sealed to the plan; callers cannot inject
a schema, substitute an extraction result, or relabel a full-head receipt.

`max_rows_per_step` must be in `1..=166144`. It is a complete legal-set ceiling:
if one selection has more legal rows, execution fails before that selection's
first projection. It never truncates the legal language or substitutes top-k
pruning. A wider schema can therefore require a larger ceiling. The grammar
mask still covers the complete vocabulary and retains its full traversal
charge.

## Execution and accounting

At every output step, the decoder materializes the complete grammar mask,
applies the sealed control exclusions and counts all legal rows. It then
projects every legal row in ascending-ID blocks of at most 32. Each logit must
be finite. Equal scores retain the lowest token ID, including signed-zero
ties. EOS is legal only at an accepting state and is explicitly projected.
Even a single legal non-EOS token is fed through the normal model before the
next selection.

The work ceiling uses `max_new_tokens * max_rows_per_step` output projections
and the unchanged worst-case forward-position and attention counts. The
completed receipt records actual projected rows and exact group-dependent
native work. Physical resident KV, grammar/source workspace and the existing
engine allocation remain separately admitted; this feature does not reduce
their memory reservation.

Both the execution identity and comparison policy bind the selected strategy
and row ceiling. Native results use
`grouped-int4-grammar-first-selected-rows-v1`; extraction results use
`grouped-int4-selected-row-schema-source-extraction-v1`; built-in source
results use `grouped-int4-selected-row-source-portfolio-v1`. A full-head result or
admission cannot be relabeled as selected-row execution. The current
multi-sequence extraction and source cohort APIs refuse selected-row plans
explicitly. Source batch and durable job admission also refuse injected
selected plans before native work or durable work journaling.

Independent JSON validation, exact-number preservation, source-membership
evidence, complete result-size checks and the final cancellation check remain
inside the owned native session. Errors abort that session and drain its KV
state. An accepting grammar prefix alone is not a completed result.

## Evidence status

The retained model-free regression definitions cover complete row coverage,
bounded block tails, full-argmax comparison on small masks, ties, nonfinite
selected logits, cancellation, work mismatches, group tails, identity
separation, source spans, exact numeric boundaries, layer-major composition
and cohort refusal. They have not been executed in this session.

Compilation and regression execution await the repository's controller-owned
DSR checkpoint on the named production graph. Actual-model equality against
the full-vocabulary reference and target-host performance remain separate
pending gates. Unrequested logits are not computed or inspected, so this path
does not reproduce full-head rejection of a nonfinite *unrequested* row. No
model-parity, speedup or release-qualification claim follows from the reduced
projection ceiling alone.
