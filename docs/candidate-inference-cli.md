# Explicit current-candidate inference CLI

`fnlp candidate generate` and `fnlp candidate chat` connect the existing pinned
prompt compiler, strict-INT8 engine and process-hosted runtime. They require a
binary built with `asupersync-runtime` and an explicitly selected local
current-candidate INT8 `.fnlpq` file. They do not download, discover, install or
activate a model. The ordinary default build retains the help surface but
refuses execution before reading input or opening the model.

This is a **non-authoritative candidate execution route**, not a certified
artifact/release path. The unresolved format, model-root, numerical-fidelity,
platform, runtime-certification and quality gates remain independent. The
JSON response states `scope=real-artifact-current-candidate` and
`evidence=non_authoritative`; neither option flags nor successful decoding
upgrade that grade.

## Commands

With an existing compatible candidate artifact and runtime-enabled binary:

```sh
printf '%s' 'Explain why the sky is blue.' |
  fnlp candidate generate --model ./local-int8-candidate.fnlpq --memory-mib 8192

fnlp candidate generate ./prompt.txt \
  --model ./local-int8-candidate.fnlpq --memory-mib 8192 \
  --max-new-tokens 128 --context-tokens 2048

fnlp candidate chat ./messages.json \
  --model ./local-int8-candidate.fnlpq --memory-mib 8192
```

The chat input is a JSON array, not JSONL or a whole API request:

```json
[
  {"role":"system","content":"Answer clearly and accurately."},
  {"role":"user","content":"What is a parse tree?"},
  {"role":"assistant","content":"A tree representing a syntactic analysis."},
  {"role":"user","content":"Give a small example."}
]
```

Only `system`, `user` and `assistant` messages are accepted. System messages
are permitted only first, the final message must be a user message, and the
existing task planner applies its complete history contract. Duplicate JSON
keys (including escaped equivalents), unknown fields, tool messages, invalid
UTF-8, excess nesting, excessive message counts and over-limit input refuse.
The complete history is cold-prefilled; no history is silently truncated.
Caller text is encoded separately from trusted template controls, using all
six tokenizer-special IDs and four template-only thinking/tool IDs as the
untrusted exclusion set. The source bytes remain data; no tool is executed.

Greedy decoding is the default. Sampling requires exactly 64 lowercase hex
digits; no ambient seed or global random generator is consulted:

```sh
fnlp candidate generate ./prompt.txt \
  --model ./local-int8-candidate.fnlpq --memory-mib 8192 \
  --seed 000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f \
  --temperature-milli 800 --top-k 40 --top-p-ppm 950000 --logprobs
```

Temperature, top-k and top-p flags without a seed refuse rather than silently
changing greedy behavior. Log-probabilities are the existing raw
full-vocabulary scores, not processed sampling probabilities or confidence.
This one-shot surface uses item ID `cli`, sample index zero and request sequence
one. Reproducibility remains scoped to the same effective request, artifact,
backend and numerical profile; no cross-host parity claim is introduced.

## Admission and delivery

`--memory-mib` is mandatory. It configures the existing process ledger, not an
OS-level RSS limiter. The route selects the current-thread preset, one blocking
coordinator and no scoped children. It does not create a second scheduler or
run an unadmitted native forward. Weight loading and execution use the existing
hosted calls, cancellation/drain ownership and checked native work budgets.

The modeled preparation reserve defaults to 256 MiB and is acquired before
reading input or constructing metadata/tokenizers/plans. `--preparation-mib`
can increase it; a conservative input/staging/tokenizer floor is enforced.
`--max-weight-mib` defaults to 6144 MiB. Loading separately reserves 128 MiB of
tokenizer/metadata headroom, 64 MiB allocator headroom and a 64 KiB streaming
chunk. Native execution separately prices the context-derived KV/RoPE/scratch
payload, 32 MiB sampler capacity, 64 MiB allocator headroom and bounded output.
These are explicit modeled reservations, not measured allocation/RSS receipts.
An insufficient aggregate memory ceiling fails admission; allocations are not
made cheaper by changing a label.

Input reads stop after the requested cap plus one byte, without consuming the
rest of an oversized pipe. Metadata inspection precedes prompt compilation;
weight materialization begins only after a valid bounded plan exists. The
metadata reader is dropped before loading. All retained model identity facts
are compared after the independent load to detect identity changes between
opens; the host also compares resident-model identity to the sealed plan.
This is not a publisher-authenticity guarantee or a replacement for unresolved
secure activation authority.

`--timeout-seconds` is a cooperative wall-time allowance including input,
preparation and loading. Remaining time, rather than a fresh full timeout, is
passed into each hosted stage and checked before publication. Blocking input,
metadata I/O and portions of loading are not preempted. Native checkpoints have
the separately named `--max-checkpoints` allowance; the loader has its own finite
64-checkpoint stage allowance. No signal-handler or hard-latency claim is made.

Successful output is one newline-terminated JSON object containing explicit
candidate evidence metadata and the existing typed INT8 chat result, including
complete model-work counters. The whole object is staged through a bounded
writer before stdout receives any bytes. The hosted output guard and preparation
charge survive serialization, write and flush. Parser/model/runtime diagnostics
use closed messages and do not echo paths, prompts, token strings or nested
exception details. A broken pipe or failed flush returns failure; transport may
already have received a prefix, but that is never reported as successful task
completion.

## Validation status

The implementation includes model-free command, option, input, control-registry,
identity-planning and output-delivery regression sources. The planner tests use
the real embedded tokenizer and template; no test fabricates a successful native
inference result. Compilation, Rust test execution, model-present inference,
performance measurement and controller DSR qualification were not performed in
the implementation environment. The controller remains responsible for the
validation flow specified in `WIRING.md`.
