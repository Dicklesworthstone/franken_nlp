# Evidence-only answer synthesis for long documents

`fnlp candidate map --task answer --question FILE --synthesize-answer` adds a
real native answer pass after complete independent passage QA. The final pass
receives the original question and **only independently verified verbatim source
quotes**. Generated passage answers are never promoted into evidence. All passage
answers, differing texts, abstentions and original coordinates remain visible
alongside the final answer.

```sh
fnlp candidate map document.txt \
  --task answer --question question.txt --synthesize-answer \
  --model ./model.fnlpq --memory-mib 12288 \
  --context-tokens 4096 --max-new-tokens 128 --max-chunks 32 \
  --max-synthesis-passages 32 --max-synthesis-evidence-bytes 16384
```

These are illustrative finite allowances, not measured requirements or assurance
that a particular document fits. The actual evidence set, question, passage
manifest and complete final prompt must fit the existing byte, token and context
limits. A compatible explicitly selected local candidate INT8 artifact and an
`asupersync-runtime` build are required. Nothing downloads or activates a model,
executes tools, or promotes candidate inference to release certification.

## Input and mode selection

Input remains one original UTF-8 document (or `-` for stdin) plus a separate local
UTF-8 question file. The question file cannot be `-`. This is not an NDJSON job or
a retrieval index. The existing `--options` file supplies complete `AnswerOptions`
for both native stages; absent options use the existing defaults. Original text,
question whitespace and Unicode are preserved.

Without `--synthesize-answer`, the original independent-passage QA path is
unchanged. Other map tasks reject synthesis arguments. The two new evidence
controls require the synthesis switch and cannot silently affect another mode:

| Option | Scope |
| --- | --- |
| `--synthesize-answer` | Explicit additional native answer pass over collected quotes |
| `--max-synthesis-passages` | Complete distinct `(chunk, quote)` passage set; default and CLI maximum 32 |
| `--max-synthesis-evidence-bytes` | Total verbatim quote bytes; default 16384, intersected with `--max-input-bytes` |

These are refusal thresholds, not top-k selectors. When all collected evidence
cannot fit, the operation fails rather than silently discarding later passages or
choosing a popular answer. Exact duplicate quotes in one chunk are deduplicated
only after every proposal's spans and occurrence metadata have been checked.
Equal quotes from different chunks remain distinct evidence passages.

## Native work and completion

Preflight first reserves the work upper bound for a maximum-admitted-context
final answer pass and one final grammar-mask allowance. It subtracts that
reservation from all five whole-invocation native-work axes and from the total
mask allowance before admitting discovery. Every actual nonblank map passage
then preflights against the remainder. The final evidence prompt is unknown until
discovery completes, so its exact task is compiled and admitted afterward. It
must fit the reserved upper bound and the actual resident model/KV capacity.

The five existing map work flags cover both stages together:
`--max-forward-positions`, `--max-projected-logits`, `--max-attention-pairs`,
`--max-dot-products`, and `--max-multiply-accumulates`. Enabling synthesis does not
multiply or reset them. `--max-mask-node-visits` applies to each map pass and the
final pass; `--max-total-mask-node-visits` includes all of them. Early EOS does not
refund reserved work into a later stage.

Only one existing process host, resident model, native engine and cancellation
controller are used. Map grammars drain before the final grammar is built. The
host additionally prices retained map values, copied evidence and origin spans,
final native result/token storage and result staging. These are modeled memory
commitments, not allocator interception or operating-system RSS enforcement.
The complete output guard survives serialization, writing and flushing.

The independent verification flags also remain nonrenewable across the whole
pipeline. `--max-qa-citations`, `--max-qa-evidence-spans` and `--max-qa-scan-steps`
cover discovery verification, evidence rechecks and final citation lifting.
Occurrence fanout is charged before expansion, including duplicates that later
collapse to one original source coordinate. These limits do not replace the
native source task's own bounded validation.

## Original-document citations, not synthetic-join citations

Every collected quote is rescanned inside its original chunk. Its occurrence
indicator and every original byte/scalar span must match. Only then can it become
a final-stage passage. The passage IDs are code-owned local labels, not model
instructions, document fingerprints or persistent entity identifiers.

Final citations must occur wholly inside admitted evidence passages. A quote
that crosses the artificial join between passages is rejected, as are foreign
passage IDs, missing occurrences and malformed local offsets. Every valid local
subquote occurrence is lifted through every corresponding original evidence
occurrence with checked arithmetic and exact source-byte comparison. Overlapping
evidence cannot manufacture multiple original occurrences: lifted coordinates
are sorted and deduplicated before choosing anchored versus ambiguous metadata.

Equal text elsewhere in the document is not newly asserted as support merely
because the final model cited a matching quote. Returned coordinates identify
only occurrences reachable through the collected evidence. UTF-8 byte offsets
and Unicode-scalar offsets remain separate and refer to the unchanged original
source.

## Output and meaning

Successful output is one ordinary candidate provenance envelope whose `output`
has execution `portable-int8-question-verbatim-evidence-synthesis-v1`. It contains
`synthesis`, the final status/answer/original-source citations, and `discovery`,
the complete independent passage-QA result. It also reports evidence counts,
reserved/planned/actual synthesis work, total work, mask accounting and total
independent verification work. Private prompt identities and generated token
transcripts are not published.

The synthesis status is one of:

- `answered`: the final model proposed an answer with checked source citations.
- `abstained`: the final model ran on collected evidence and explicitly abstained.
- `no_evidence_collected`: discovery completed without an answered passage; no
  final model pass ran and no synthetic model-abstention receipt was created.

The final reservation remains visible even in the no-evidence case. Empty or
whitespace-only original inputs retain the existing preflight refusal. A later
failure in discovery, evidence collection, final planning/inference, verification,
cancellation or complete-result sizing publishes no successful partial result.
Transport errors can still leave partial external bytes and are errors, not
successful delivery.

This pipeline compresses the document through model-selected quotes. It does not
prove that every relevant fact was discovered, that chunk boundaries preserved
all context, or that omitted passages contain no contradictory evidence. It is
not full-context-equivalent QA, exhaustive retrieval or a contradiction solver.
Preserving all candidate answers makes disagreement inspectable; it does not
calibrate confidence or decide which candidate is true. Source membership is
structural; semantic support and answer quality remain unassessed. Both model
abstention and no collected evidence are uncalibrated.

## Embedding and validation status

Library applications use `SourceQuestionSynthesis` and
`QuestionSynthesisLimits` in
`tasks::source_planning::quantized::long::question::synthesis`. The planner exposes
`preflight_int8_question_synthesis_with_control` and
`plan_int8_question_synthesis_with_control`. The consumed prepared value executes
both stages and does not accept caller-invented inference receipts. The hosted
entry point is `NlpEngine::synthesize_int8_document_answer` with
`SourceMapConfig<SourceQuestionSynthesis>`.

This extension adds 22 model-free regression definitions: 12 core, 4 CLI
(including one feature-disabled test), 4 pinned runtime-planning fixtures and 2
host-memory arithmetic fixtures. They cover quote-only final inputs, full-stage
reservations, Unicode lifting, overlap/duplicate handling, passage-join refusal,
late corruption, cancellation, no-evidence accounting and mode isolation.

Source/API, ownership and targeted GitHub diff review were performed. Compilation,
Cargo tests, scripts, Actions, DSR and full-model execution were not run under the
repository's controller-owned validation policy. These are added test definitions,
not passing-test evidence; no build, parity, quality, performance or release claim
is made.
