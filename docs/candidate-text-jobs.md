# Retained native generation and chat

`fnlp candidate text-job start/resume` adds explicit durable execution for
`--task generate` and `--task chat`. It reuses the existing pinned native INT8
chat compiler, addressed sampler, batch adapter, process host and authenticated
job journal. It does not introduce a second generation engine or an automatic
retry loop. The live `candidate text-batch` and token-stream commands remain
unchanged.

This is current-candidate execution, not release activation, model quality,
numerical qualification or a certified performance claim. It requires
`metadata-store` and `asupersync-runtime` on supported Linux x86-64/AArch64
builds plus a compatible explicitly selected local candidate artifact.

## Original input contract

The retained route uses the common job envelope `{id,text,task_args?}`,
NOT the live text-batch `prompt`/`messages` envelope. IDs must be unique.
For generation:

```jsonl
{"id":"item-1","text":"Write a short description of a quiet garden."}
{"id":"item-2","text":"Describe a rainy evening.","task_args":{"sample_index":2}}
```

For chat, `text` is the new final user message. Optional `history` supplies the
preceding typed messages, in their original order:

```jsonl
{"id":"conversation-1","text":"Make that explanation shorter.","task_args":{"sample_index":0,"history":[{"role":"system","content":"Be helpful."},{"role":"user","content":"Explain a compiler."},{"role":"assistant","content":"A compiler translates a program into another representation."}]}}
```

No role, text, whitespace or Unicode normalization is applied. The existing
compiler validates the complete history and refuses invalid ordering or context
exhaustion instead of dropping turns. Generate records reject nonempty history.
`task_args` can contain ONLY `history` and `sample_index`; omitted values mean
empty history and sample zero. It cannot supply a budget, model, generation
policy, seed, tools or task override. Duplicate JSON keys and changed original
envelopes fail closed. Flush commands are not durable population records.

Every resume receives the COMPLETE original ordered NDJSON population. Original
inputs are not persisted for you; preserve them separately. Input JSON framing
and escaping count toward `JobLimits.max_input_bytes_per_item`, which must fit
`--max-input-bytes`. This conservative ceiling is distinct from model tokens.

## Start and resume

Provision an existing protected owner-only directory, a random 128-bit job ID,
a protected regular file containing exactly 32 RAW random job-key bytes and
immutable JobLimits JSON. See `owned-jobs.md` for storage/platform details and
`candidate-scored-jobs.md` for the JobLimits fields. There is no automatic
storage or key creation and no promise of encryption at rest.

```sh
fnlp candidate text-job start original.ndjson --task generate \
  --model ./model.fnlpq --memory-mib 8192 \
  --store-results --job-dir ./protected-job --job-id "$JOB_ID" \
  --key-file ./job.key --limits ./job-limits.json

fnlp candidate text-job resume original.ndjson --task generate \
  --model ./model.fnlpq --memory-mib 8192 \
  --store-results --job-dir ./protected-job --job-id "$JOB_ID" \
  --key-file ./job.key --limits ./job-limits.json --materialize
```

Use `--task chat` on both invocations for conversation records. Resume verifies
the original key, input population, model identity and effective generation
recipe before storage repair or model forwards, then skips committed items.
`--discard-uncommitted-tail` is resume-only: it requests the common journal's
post-authentication repair, not deletion or rewriting of committed results.
`--materialize` publishes fixed protected `materialized.ndjson` after every item
commits. A failed final report can follow successful durable progress; do not
assume rollback or automatically resubmit under a new job ID.

## Sampling is stable by semantic address

No `--seed` means greedy decoding. Sampled generation requires the same explicit
64-lowercase-hex `--seed` on start and resume; no global RNG, hidden default seed
or attempt-derived reseeding occurs. The seed is a generation option, NOT the
protected job-authentication key. Stable item ID and sample index address the
existing sampler. Skipping committed items or restarting an incomplete attempt
does not create a different sampling address. This preserves the existing
algorithmic contract; real-model replay has not been measured here.

Temperature, top-k, top-p, minimum length, exact stop suffixes, bans, repetition,
presence/frequency penalties, logit biases, output limits and logprob capture
use the shared `candidate generate/chat` options. All are authenticated along
with exact planning limits and sampler/processor versions. Keep the same flags
on resume. A changed policy or seed is a different job, not a recovery option.

Each stored result is a complete validated native completion including finish
reason, token IDs, effective seed, work and optional raw full-vocabulary
logprobs. EOS and matching stop suffix behavior remain unchanged. Token/byte
limits are the existing explicit finish reasons; cancellation, malformed UTF-8,
native failure or failed result validation cannot commit partial text. In-flight
KV/tokens are not checkpointed; explicit resume restarts only the uncommitted
item and charges another complete attempt.

## Ownership, limits and output

One resident model, native allocation and process-owned invocation serves all
pending items. The process ledger reserves full KV capacity, sampler workspace,
native scratch, input population, private recipe, journal and output storage.
Output ownership survives spool synchronization and journal acknowledgement,
not merely model return. These are modeled reservations, not measured RSS caps.

All five native work counters and attempt count have nonrenewable lifetime
ceilings in JobLimits. Full worst-case work is charged before generation;
early EOS, stop completion, failure and resume do not refund prior attempts.
The same fixed native ceiling also bounds each physical invocation. Document
history, source and output admission cannot expand through per-item JSON.
`max_result_bytes` must cover the complete generated-result envelope, not just
`--max-output-bytes`; the CLI's derived bound includes escaping, tokens, scores
and metadata. Snapshot/transport and journal/serialization memory limits are
checked separately. `--timeout-seconds` covers the invocation; blocking IO and
bounded tokenization are checked around calls, not claimed preemptible.

Standard output is completion METADATA only. Private generated text, histories,
seeds and token scores remain in the explicit protected result spool or
materialization. Diagnostics do not print nested private content. Unavailable
feature graphs or invalid command options refuse before key/config/input IO.

This retained adapter executes serially. It rejects `--prefill-rows`,
`--cohort-rows` and `--active-rows`; it does not silently ignore or implicitly
change scheduling. Existing live cohort/refill execution is unchanged.

## Verification scope

Added regression definitions cover pinned generate/chat plans, seed and sample
identity reconstruction, keyed recipe/population mismatches, every work/limit
axis, history validation, explicit consent, metadata-only reporting and no-IO
failures. Rust compilation/test execution, DSR, filesystem crash/restart
execution with this producer, real-model replay and performance qualification
remain pending. Source inspection is not a substitute for those receipts.
