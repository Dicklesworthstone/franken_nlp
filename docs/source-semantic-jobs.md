# Durable summary and passage-answer review

`fnlp candidate source-int4 summarize-semantic-job start|resume` and
`answer-semantic-job start|resume` retain each existing source result together
with its complete correlated review. The summary route reviews every complete
bullet against the original document. The answer route reviews the exact
question/answer pair against all original passages. Generated text, tokens,
citations and `semantic_support: "not_assessed"` remain unchanged. A failed
generation or review commits no partial item; already committed items remain
available and are skipped on resume.

The [source review contract](source-semantic.md) and its complete
`Int4SourceSemanticOptions` file apply unchanged. Review remains an experimental,
uncalibrated assessment by the same model, with no repair or acceptance
authority. These commands require `metadata-store` and `asupersync-runtime` on
Linux x86-64 or AArch64.

## Fixed tasks and replayable input

Summary records contain globally unique IDs and exact source text:

```jsonl
{"id":"document-1","text":"Alice works in Shanghai."}
```

Answer records use `text` for the question and a closed passage-only argument:

```jsonl
{"id":"question-1","text":"Where does Alice work?","task_args":{"passages":[{"id":"p1","text":"Alice works in Shanghai."}]}}
```

Summaries refuse non-null `task_args`. Answers require the shown passage
argument; record-level task, generation options, budgets and review policies
are refused. `--options FILE`, when supplied, fixes the complete typed summary
or answer options for the whole job. Batch flush controls are not accepted.
The stored item cap bounds the complete record; source planning additionally
bounds the complete decoded input, including the QA question, joined passage
text and canonical passage manifest. Every original task and review prompt
must fit its context without chunking, truncation or retrieval.

Provision an existing protected job directory, a random 128-bit job ID and a
protected file containing exactly 32 raw secret bytes using the
[existing storage setup](candidate-owned-jobs-cli.md#explicit-retention-and-setup).
`--store-results` is required. The CLI does not create secrets or adopt a
different job identity. The protected spools are authenticated, not encrypted.

## One attempt and immutable lifetime limits

The semantic options' `max_model_work` covers one complete generation and
review attempt. `--max-mask-node-visits` bounds that attempt's generation masks.
The runner durably spends the full declared allowance before execution,
including for a failed attempt. The separate `--limits` file covers all items,
attempts and invocations; resume cannot enlarge it or refund prior work.

For the [example source review policy](source-semantic.md#complete-review-options),
this `source-review-job-limits.json` allows ten items and twelve complete
attempts with `--max-mask-node-visits 2000000000`:

```json
{
  "max_items": 10,
  "max_id_bytes": 128,
  "max_input_bytes_per_item": 65536,
  "max_snapshot_bytes": 1048576,
  "max_result_bytes": 4194304,
  "max_spool_bytes": 67108864,
  "max_materialized_bytes": 67108864,
  "max_journal_bytes": 67108864,
  "max_attempts": 12,
  "max_work": {
    "model": {
      "forward_positions": 1200000,
      "projected_logits": 2400000000,
      "attention_pairs": 12000000000000,
      "projections": {
        "dot_products": 120000000000000,
        "multiply_accumulates": 12000000000000000
      }
    },
    "mask_node_visits": 24000000000
  }
}
```

These are explicit ceilings, not measured costs. `max_id_bytes` may be at most
256. `max_input_bytes_per_item` must fit `--max-input-bytes` and also tightens
the frozen source-planning ceiling. `--max-result-bytes` bounds the original
source result; the semantic options' `limits.max_output_bytes` bounds the
complete reviewed result and must fit the job's `max_result_bytes`.

## Start and resume

Use the complete `source-semantic.json` policy from the linked guide. Set
`JOB_ID` to the original random job ID and `MODEL_SHA256` to the independently
obtained snapshot pin:

```sh
fnlp candidate source-int4 summarize-semantic-job start documents.ndjson \
  --snapshot ./nanbeige.int4.snapshot --expected-model-sha256 "$MODEL_SHA256" \
  --memory-mib 8192 --context-tokens 2048 --max-input-bytes 65536 \
  --semantic-options ./source-semantic.json --max-mask-node-visits 2000000000 \
  --input-encoding safe-bpe-v1 --compute-threads 2 \
  --store-results --job-dir /private/source-review-job --job-id "$JOB_ID" \
  --key-file /private/source-review-job.key --limits ./source-review-job-limits.json

fnlp candidate source-int4 summarize-semantic-job resume documents.ndjson \
  --snapshot ./nanbeige.int4.snapshot --expected-model-sha256 "$MODEL_SHA256" \
  --memory-mib 8192 --context-tokens 2048 --max-input-bytes 65536 \
  --semantic-options ./source-semantic.json --max-mask-node-visits 2000000000 \
  --input-encoding safe-bpe-v1 --compute-threads 2 \
  --store-results --job-dir /private/source-review-job --job-id "$JOB_ID" \
  --key-file /private/source-review-job.key --limits ./source-review-job-limits.json \
  --materialize
```

For passage QA, use `answer-semantic-job start` or `resume` with the answer
records above and its own protected job identity. The same options and limits
apply. Both commands also accept `--source-dir` plus `--source-manifest` in
place of the snapshot pair, with optional `--save-snapshot NEW_FILE`.

Resume authenticates the exact population, fixed source options, both
model-bound compilers, review policy, tokenizer policy and lifetime limits
before repair or pending execution. Repeat any supplied `--options` and every
semantic setting. Explicit `--discard-uncommitted-tail` is available on resume
after authentication; otherwise uncommitted spool tails are refused.

To retain inputs, add `--spool-input --max-retained-input-mib 8` at start. On
resume omit the input positional argument and use
`--use-retained-input --max-retained-input-mib 8`. The same complete original
records are replayed under the original commitments; this changes input
ownership, not the task policy or lifetime work. Without retention, supply the
complete original population on every invocation.

Compact input gives each complete attempt one tokenizer allowance shared by
source/question/passages and all review prompts. Its exact caps are frozen in
the recipe. One persistent CPU team may span all pending records and reviews;
physical thread width can change on authenticated resume without renewing work.
The full preparation, input, native, combined-output, journal, serialization
and worker peak is quoted before key, source-options, population or model IO,
after bounded semantic-options and limits reads. Record-specific semantic
validation still occurs during item preparation.

Stdout uses `fnlp-candidate-int4-semantic-source-job-v1` and contains progress
and public model provenance only. Private results remain in the protected
spool. `--materialize` publishes ordered verified `materialized.ndjson` after
all items commit; the [job management commands](job-management-cli.md) can
inspect or verify progress separately. A final output or cancellation failure
does not roll back acknowledged items. There is no automatic retry.

Source parsing and independent review cover this implementation and its
regression definitions. Compilation, executable tests, DSR and independently
pinned model qualification remain unrun.
