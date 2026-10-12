# Durable document summaries and keyphrase rankings

Source-document jobs can now retain a single selected cited summary or a single
ranked keyphrase list per original document. Add the optional `reduction` object
to the existing `--source-document-config` JSON. Without that object, the exact
independent-map behavior and v1 recipe remain unchanged. An explicit null, an
invalid policy, or a mismatched task is an error, never a map-only fallback.

## Cited summaries

Use `--task summarize` and this document configuration:

```json
{
  "schema_version": 1,
  "max_chunks": 64,
  "max_chunk_bytes": 2048,
  "max_tokenizer_calls": 8192,
  "max_result_bytes": 4194304,
  "max_live_value_bytes": 16777216,
  "max_total_value_bytes": 67108864,
  "reduction_reserve_mib": 64,
  "reduction": {
    "schema_version": 1,
    "task": "summarize",
    "aggregation": {
      "max_unique_bullets": 4096,
      "max_citations": 16384,
      "max_evidence_spans": 65536,
      "max_value_bytes": 4194304,
      "max_scan_steps": 536870912
    },
    "max_bullets": 16,
    "max_result_bytes": 4194304,
    "max_lifetime_scan_work": 5368709120
  }
}
```

For example, with previously prepared protected job storage:

```sh
fnlp candidate job start documents.ndjson \
  --task summarize --source-document-config document.json \
  --model local.fnlpq --memory-mib 16384 --preparation-mib 1024 \
  --store-results --job-dir private/job --key-file private/key \
  --job-id "$JOB_ID" --limits job-limits.json --materialize
```

The storage, key, original-population and supported-build prerequisites are in
`owned-source-document-jobs.md`. These memory values are example admission
inputs, not measured requirements. All native and storage limits still need to
fit the user's actual artifact, input and job contract.

The existing INT8 summary reducer deduplicates only identical bullet text,
retains each bullet's verified original-source citations, and selects at the
complete root. Selection orders by distinct supporting chunks, then local-rank
sum, earliest source position, and byte-lexical text. It does not synthesize new
prose, reconcile contradictions, or equate frequency with importance/confidence.
The result retains structural-citation and semantic-support warnings. This is
not a new neural synthesis mode or a factual-support qualification.

## Keyphrase ranking

For `--task keyphrases`, replace only the `reduction` object with:

```json
{
  "schema_version": 1,
  "task": "keyphrases",
  "aggregation": {
    "max_unique_phrases": 4096,
    "max_evidence_spans": 65536,
    "max_value_bytes": 4194304,
    "max_scan_work": 536870912
  },
  "max_phrases": 16,
  "max_result_bytes": 4194304,
  "max_lifetime_scan_work": 5368709120
}
```

The existing native keyphrase reducer unions exact phrase text and independently
verifies its original-document occurrences. It ranks the complete union by
distinct supporting chunks, local-rank sum, earliest occurrence and byte order.
Repeated occurrences inside one chunk do not create extra support votes. No
case folding, fuzzy matching, neural reranking or recall guarantee is added.

## Resource and resume contract

All fields inside a supplied `reduction` object are required. Unknown, duplicate,
missing or invalid fields reject. A summary reducer cannot enter a keyphrase or
NER job, and these settings do not enable QA, global entity resolution, or
cross-document aggregation. The whole configuration remains bounded to 16 KiB.

`max_bullets`/`max_phrases` applies once at the final root. It never licenses
intermediate truncation or an oversized candidate/evidence union. Aggregation
value bytes must fit the shared map/reduce value ceiling. Reduced output bytes
must fit both the document result ceiling and `JobLimits.max_result_bytes`.
Per-chunk task options remain separate and fixed for the job.

Independent evidence verification has a conservative lifetime bound:

```text
per_attempt_scan_limit * JobLimits.max_attempts <= max_lifetime_scan_work
```

All arithmetic is checked. The examples above cover at most ten durable attempts;
a larger `max_attempts` requires a correspondingly funded immutable scan ceiling.
The existing journal already records attempts before execution. That attempt
count includes failures and uncertain acknowledgements, so this proof bounds
verification across resumes without adding or migrating a journal counter. It
is a worst-case allowance, not a new exact scan-work ledger. Successful native
results retain their observed verification-work fields.

Native model work and mask work are unchanged from the complete map plans:
there is no additional neural pass. The journal still debits the actual planned
whole-document work before execution and never refunds failed attempts. One
resident engine and one cooperative controller span maps and reduction. Every
chunk identity is checked before the first forward, and one output guard remains
held through durable spool and journal acknowledgement. No partial reduced result
is committed after a later chunk or verification failure.

Use `resume` with the same original inputs and semantic settings. Reduced recipes
are version 2 and bind the concrete reducer execution/policy versions, every
aggregation/selection limit, and lifetime scan ceiling. Changing any of those
settings changes the resume contract. Attaching or changing a policy also
invalidates already prepared objects, even when their native prompts match.
Committed documents are skipped; an uncommitted document retries as a whole.

The host continues to charge the maximum map/reduction frontier plus explicit
preparation, IO, journal and serializer reservations. A smaller final top-k does
not lower those worst-case input/frontier requirements. These are modeled
commitments, not measured or OS-enforced RSS bounds. Stdout remains metadata-only;
private reduced results stay in the explicitly protected spool/materialization.

## Library and validation

Construct the existing `Int8SourceDocumentJobPlanner`, then consume it with
`with_reduction(SourceDocumentReduction::Summary { .. })` or `Keyphrases { .. }`.
`NlpEngine::job_int8_source_document` selects the dedicated
`Int8ReducedSourceDocumentJobProcessor` from that frozen policy. Ordinary map
processors and their output type remain available and reject reduced recipes.

New regression definitions cover legacy mode, both pinned task planners,
prepared-object ownership, every reduction-limit axis, Unicode/source and work
accounting, cancellation, closed configuration, lifetime quota arithmetic,
repeatable recipe construction, and the CLI-to-native-reducer route.
Compilation, Rust tests, CI, DSR and native/model execution were not run by this
implementation worker under the controller-owned validation policy. These
source-reviewed definitions are not passing-test or inference-quality evidence.
