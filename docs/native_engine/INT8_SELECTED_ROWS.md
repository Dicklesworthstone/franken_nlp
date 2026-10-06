# Explicit legal-row INT8 structured decoding

Status: source implementation, including native, task, hosted dispatch and five
candidate CLI commands. Rust compilation/tests, DSR, real-model dense/selected
parity, task quality, physical memory and throughput qualification are UNRUN.
This is not artifact activation, profile ratification or a performance award.

## Candidate CLI

With an asupersync-runtime build and an existing local current-candidate artifact:

```sh
fnlp candidate extract document.txt --schema schema.json \
  --model local.fnlpq --memory-mib 8192 --selected-rows 166144

fnlp candidate ner document.txt \
  --model local.fnlpq --memory-mib 8192 --selected-rows 166144
```

The same explicit switch is accepted by `candidate keyphrases`, `candidate
summarize`, and `candidate answer`. Answer still consumes the existing question
and passage JSON input. Extraction still requires `--source-membership` when
using verbatim source annotations; the head choice does not supply grounding.

`--selected-rows` admits 1..=166144 legal rows per selection. Every legal row is
scored, never just the first rows or a top-k subset. If the complete legal set
exceeds the cap, the request FAILS; it never prunes, silently grows a budget,
falls back to the dense path or retries. Zero, excessive, negative and malformed
caps are refused before schema/document/model IO.

166144 is the complete model vocabulary width, not a benchmark-selected setting.
With that cap the implementation still projects only the legal set at each step;
the worst-case head-work allowance remains as large as the full-head reference.
Smaller caps reduce worst-case admitted head rows but may refuse broad grammars.
No automatic threshold or host-class dispatch selection is performed.

Omitting the flag preserves the existing full-vocabulary implementation and its
identity. The switch is not placed in shared host arguments: generation, chat,
text-batch, map/corpus/job and other unsupported CLI routes cannot silently
ignore it. This change does not implement structured cross-document batching.

## Execution and semantics

The existing JSON/source grammar and pinned vocabulary oracle materialize a
complete dense legality mask BEFORE the lm-head. The driver removes excluded
control IDs and admits EOS precisely when the grammar state accepts. An accepting
numeric prefix can still compete with a longer legal number; accepting does not
force EOS. Empty legal sets fail without a head projection.

After counting and admitting the whole set, the driver scans legal IDs in
ascending order and projects bounded chunks of at most 32 rows through the real
StrictInt8Engine/Int8Session and LinearRows::Selected. There is no alternate
model evaluator or public fake-native receipt seam. The chunk argmax and global
argmax use the same strict greater-than comparison as full constrained decoding,
so ties, including signed zero, keep the lowest legal token ID across chunks.

Every requested logit must be finite. Unrequested logits are not evaluated or
validated. Thus sparse mode does NOT preserve the full reference's refusal of a
nonfinite ILLEGAL row. This explicit boundary is one reason the strategy has a
separate identity and execution label. For the same finite logits, the two
argmax procedures select the same legal token; native/model parity remains unrun.
No full-vocabulary probability, logprob, confidence or calibration is invented.

Singleton legal sets still score their token. Every selected non-EOS token is
fed back through the complete 44-slot causal decoder before another selection.
EOS is explicitly scored, never fabricated. Prefill still runs every prompt
token but does not project a head until the first grammar-constrained selection.
Tokens crossing JSON boundaries and multibyte Unicode use the original grammar.

## Work, memory and lifetime

For P prompt tokens, T maximum new tokens including EOS, and cap C, the admitted
ceiling is P + T - 1 native forward positions and T * C lm-head rows. Int8Work
also includes all decoder linear work and triangular causal attention. Native
integer projection budgets, JSON head/forward budgets and complete resident KV
capacity are checked before prefill. Mask traversal retains its full existing
per-step and aggregate limits; a small head cap does not discount grammar work.

Completed projected_logits counts the ACTUAL legal rows evaluated, including
EOS, not vocabulary width times tokens and not the number of physical chunks.
The same native projection ledger reconciles all head chunks and decoder work.
A late failing chunk cannot return a partial argmax or successful work receipt.

Fixed 32-ID stack storage and one at-most-32-logit allocation avoid an additional
vocabulary-sized heap row list. The existing dense mask and ordinary full-head
native workspace envelope remain sufficient; no extra uncharged memory rail or
separate runtime is introduced. Re-encoding the hidden activation between head
chunks is not claimed to improve throughput. Native counts are algorithmic work,
not measured instructions, traffic, latency or RSS.

Native, grammar, cancellation, finalization or output-budget failure is no-result
and no-retry. Source/schema/exact-decimal/typed-envelope finalization remains
inside the exclusive native session. Its existing poison-and-drain RAII clears
all 44 KV slots on exit. The process-hosted input/native/output reservations and
single coordinator are unchanged; output guards remain owned through delivery.

## Sealed plans and supported library routes

`Int8ExtractPlan::with_selected_rows(Int8JsonSparseLimits)` consumes an unadmitted
plan, and `PreparedInt8SourceTask::with_selected_rows` additionally seals the
source-task finalizer. The cap and strategy enter decision_policy_digest.
Exact prompt/schema/source bytes, tokenizer, template, control exclusions,
passage partitions, task options, model binding and numerical profile do not
change. Previously admitted full-head identities are refused. Selecting a mode
again on an already selected plan is also a refusal.

The existing NlpEngine::execute_int8_extract and execute_int8_source accept these
prepared plans and derive budgets from their sealed work. CLI sealing occurs
before model loading and native admission, without recompiling the source or
schema into a different prompt. All four source-task semantic finalizers remain
unchanged, including all ambiguous source occurrences, original Unicode spans,
keyphrase ordering/deduplication, required citations, and passage-join rejection.
Source membership does not prove semantic entailment or factual correctness.

Distinct execution labels are:

- Native: `portable-int8-grammar-first-selected-rows-v1`.
- Extraction: `strict-int8-selected-row-schema-source-extraction-v1`.
- Source tasks: `portable-int8-selected-row-source-portfolio-v1`.

Full-head labels remain unchanged. Finalizers check the label chosen by their
own sealed plan and reconcile completed work; relabeling a dense result does not
satisfy a selected plan. Closed finalizer checks are not public wire receipt
authentication or independent evidence of model execution.

## Validation boundary

Added 34 Rust regression definitions cover complete legal sets, EOS and numeric
prefixes, selected-only finite checks, source constraints, dense/sparse scripted
selection, signed-zero ties, all work axes, partial head chunks, global argmax,
cap refusal, cancellation, native errors, late finalization, sealed identities,
exact decimal/source output, all four semantic finalizers, five CLI routes,
pre-IO refusals and real pinned planning-to-host transfers. These are UNRUN.

Actually executed independent Python models:

- 196608 finite full-versus-selected argmax cases across all six-token grammar
  and exclusion masks, all EOS IDs, acceptance states and four score profiles;
  170784 oversized-set cap checks refused rather than pruning.
- 2040 chunked-argmax and exact requested-row coverage cases with 32-row chunks,
  including full model vocabulary width 166144; randomized seed 20261005.

Those checks validate abstract selection algorithms only. They are not Rust
execution, full-model numerical/task parity, DSR or performance qualification.
