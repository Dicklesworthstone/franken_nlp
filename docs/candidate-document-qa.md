# Question answering across a complete document on native INT8

`fnlp candidate map --task answer --question FILE` applies one question to every
nonblank chunk of an original UTF-8 document. Each native passage answer retains
its own verified citations in original-document byte and Unicode-scalar
coordinates. All passage answers survive the ordered merge; different answer
texts are not hidden by majority voting or a final top-k.

```sh
printf '%s\n' 'Who approved the acquisition?' > question.txt
fnlp candidate map report.txt --task answer --question question.txt \
  --model ./model.fnlpq --memory-mib 8192
```

This uses an explicitly selected compatible local candidate INT8 artifact and a
build with `asupersync-runtime`. There is no model download or discovery. Omit
`report.txt`, or use `-`, to read the original document from stdin. Both files
are exact plain UTF-8, not JSON task envelopes. The question file is limited to
16 KiB, and its complete encoded question, passage manifest, trusted scaffold,
source and reserved output must still fit the actual task/context budget.
`--question -` is refused so stdin unambiguously belongs to the document.

The existing `candidate answer` command is unchanged: it still accepts explicit
JSON question/passages input. Existing NER, keyphrase, summary and reduced-summary
map modes are unchanged. A question file is mandatory with `map --task answer`
and forbidden with other map tasks. `--reduce-summary` cannot be combined with
question QA. Unknown or partial native options are refused.

## Meaning of the result

Every nonblank chunk is a separate native `answer-v1` execution with the SAME
question and one exact passage. Answers use the existing code-owned prompt,
source-backed constrained grammar and pinned tokenizer. The question and passage
manifest are untrusted context, not citation evidence. The coordinator owns
original source positions; model-supplied passage ids and offsets are not trusted.

The final `mapped.root.value` contains source-ordered chunk records. Each has
`chunk_id`, `source_span`, `status`, optional `answer`, `citations`, `model_work`
and `mask_node_visit_charge`. `status` is one of:

| Status | Meaning |
| --- | --- |
| `answered` | The native passage result proposed a nonempty answer with source-member citations. |
| `abstained` | The native passage result declined to answer; this is model-declared and uncalibrated. |
| `whitespace_only` | The original chunk was entirely Unicode whitespace; no model was called and no native abstention was invented. |

Whitespace ranges remain in source coverage and the chunk count, with zero
native work. An empty or wholly whitespace document is refused before inference.
Native calls and mask reservations cover only nonblank chunks. The output gives
separate native, whitespace, answered and abstained counts.

The outer `outcome` is a description of exact proposed answer strings:

| Outcome | Meaning |
| --- | --- |
| `no_answer_proposed` | Every native passage execution abstained. |
| `one_answer_text` | All answered passages used one exact answer string; some passages may still have abstained. |
| `multiple_answer_texts` | More than one exact string was proposed; ALL answers remain visible. |

`distinct_answer_texts` is an exact-text count, not a semantic agreement or
contradiction assessment. There is no invented global `answer` or global
`answerable=true` decision. The mode does not retrieve passages, reason jointly
across chunk boundaries, reconcile contradictions, rewrite answers or evaluate
whether a quote entails an answer. A fact requiring multiple passages may be
missed by every chunk. A quote's presence verifies structural source membership,
not semantic support. Calibration remains `uncalibrated`, semantic support stays
`not_assessed`, and candidate evidence stays `non_authoritative`.

## Complete planning and independent citation checks

A lossless partition preserves every original byte and the existing Unicode
scalar/CRLF boundary rules. It does not promise sentence boundaries or overlapping
windows. The question-aware planner tightens caller byte/token limits using the
actual trusted scaffold and encoded question, plus a conservative bound for the
versioned single-passage manifest. It then counts each ACTUAL question, manifest
and source encoding. An underestimated bound fails before native execution.

The CLI computes full source coverage, nonblank counts, native work and masks
before weight loading. The host independently compiles and admits every actual
nonblank passage plan before the first forward. The final output is checked
against the CLI's preflight: source extent, counts, outcome, complete planned work,
all five actual native counters, masks and independent verification ceilings.

Every citation is rescanned against its original chunk independently of native
source-grammar acceptance. The exact quote, complete occurrence list, local
passage id, anchored/ambiguous marker and every byte/scalar offset must agree.
Overlapping occurrences are retained. Verified spans are lifted to ORIGINAL
document coordinates and are exhaustive only within the corresponding chunk.
A corrupt last citation fails the whole invocation, even after earlier passages
have answered successfully. No successful partial document is published.

## Limits and resource ownership

`--options FILE` contains complete per-passage `AnswerOptions`:

```json
{"max_answer_scalars":2048,"max_citations":8,"max_quote_scalars":256}
```

`--max-new-tokens` and `--max-result-bytes` still bound each native passage.
Complete independent verification is additionally bounded by:

| Flag | Default | Scope |
| --- | ---: | --- |
| `--max-qa-citations` | 4096 | All independently checked citation fields across all native passages |
| `--max-qa-evidence-spans` | 16384 | All exact occurrence spans across all checked citations |
| `--max-qa-scan-steps` | 67108864 | Shared, nonrenewable independent occurrence-scan allowance |

These flags require `--question`. A passage cannot renew these allowances.
The existing `--max-input-bytes`, `--max-chunks`, `--max-chunk-bytes` and
`--max-tokenizer-calls` control the original source and partition. The existing
whole-run forward/logit/attention/projection/multiply-accumulate and mask ceilings
remain unchanged. Longer questions consume real prompt space and native work;
blank ranges do not manufacture model receipts. A final answer count does not
increase work authority or allow input truncation.

`--max-map-result-bytes` bounds reduction values and the complete QA envelope.
`--max-live-value-bytes` and `--max-total-value-bytes` bound live and cumulative
serialized values throughout the reduction tree. Overflow fails rather than
silently discarding answers. The candidate provenance wrapper retains the
existing 4096-byte additional allowance.

For embedding, `NlpEngine::answer_int8_document` accepts a resident `ResidentInt8`,
owned source, pinned planner/vocabulary Arcs, a `SourceMapConfig<SourceQuestion>`
and a cancellation token. The default `SourceMapConfig` specialization remains
`SourceMapTask` for existing users. These are concrete static entrypoints, not a
runtime task/plugin registry.

The host checks the resident model domain and identity and the FULL allocated
KV capacity. Both source and question string capacities are charged alongside
the explicit preparation reserve. All prepared prompts/grammars and the complete
native/evidence frontier remain reserved; it is not priced only for answered
passages. One engine and controller perform all native work. Native storage
is released before its guards; output ownership survives serialization, write
and flush. Ledger reservations are modeled commitments, not OS-enforced or
measured RSS guarantees. Blocking I/O and individual bounded tokenizer/scanner/
serializer calls are not preempted mid-operation.

The QA envelope has no raw question, prompt digest or generated-token transcript
field. Answers and quotes are still private, untrusted model/source data and may
echo input; they are not safe telemetry. `mapped.root.value` is marked untrusted
for downstream agents. Transport errors can leave partial external bytes and are
reported as failures without retry.

## Validation scope

The core adds 13 pinned-planning/private scripted regression definitions. Host
and CLI integration add 14 more definitions, including a feature-disabled no-I/O
case. Coverage includes exact preflight, changed-question commitments, Unicode
citations, skipped whitespace, multiple answers, abstention, malformed last
citations, all native counters, shared verification limits, reduction trees,
late cancellation, complete envelope bounds and real process-charge arithmetic.
Scripted core results are private corruption fixtures, not neural-success evidence.

This editing session performed source-only preimage/hash, targeted-diff,
whitespace and lexical delimiter checks. Rust compilation, Cargo tests, native
model execution and DSR were NOT run; repository instructions reserve executable
validation to the designated controller. No passing build, QA quality, calibrated
abstention or throughput result is claimed.
