# Source INT4 fixed-batch CLI

`fnlp candidate source-int4 text-batch` connects NDJSON input to the existing
shared-model INT4 generation, pinned chat finalization and charged host path.
It runs one finite group of 1..64 requests. This is not the INT8 text-batch
corpus protocol, a daemon, dynamic refill, or a stream of provisional tokens.
It requires the existing `asupersync-runtime` build and an explicit local model.

## Input and invocation

For generate, each line is an object with a unique `id`, `prompt`, and optional
unsigned `sample_index` (default zero):

```json
{"id":"document-a","prompt":"Give a short greeting."}
{"id":"document-b","sample_index":7,"prompt":"Explain why leaves change color in one sentence."}
```

Run against an already-created local candidate snapshot with an independently
retained model-content pin. `MODEL_SHA256` below must be that 64-character
lowercase pin, not a value extracted from the untrusted snapshot being loaded.

```sh
fnlp candidate source-int4 text-batch requests.ndjson --task generate \
  --snapshot model.int4snap --expected-model-sha256 "$MODEL_SHA256" \
  --group-width 128 --memory-mib 8192 --context-tokens 512 \
  --max-new-tokens 32 --max-records 8
```

Omitting the input path, or using `-`, reads stdin. To convert the pinned local
source closure instead, replace the snapshot/pin flags with `--source-dir DIR
--source-manifest FILE`. `--save-snapshot NEW_FILE` retains the existing
create-only save behavior for source conversion. Saving happens only after the
whole batch is planned and admitted; a saved snapshot remains if execution
later fails. There is no fallback from a bad snapshot to another model source.

For chat, use `--task chat` and replace each `prompt` with `messages`:

```json
{"id":"chat-a","messages":[{"role":"system","content":"Reply briefly."},{"role":"user","content":"Hello."}]}
{"id":"chat-b","messages":[{"role":"user","content":"Name a primary color."}]}
```

The strict existing transcript parser is reused: at most one initial system
message, then alternating user/assistant messages ending in user. Duplicate
JSON keys, unknown record fields, duplicate IDs, invalid UTF-8, blank records,
empty input and records beyond the configured ceiling fail before model loading.
A final nonempty line without a newline is allowed. CR/LF bytes count toward
both input limits; oversized records are not drained without bound.

## Execution and output

Shared generation flags (`--seed`, sampling processors, stops, penalties,
logit biases, `--logprobs`, output/token limits) apply independently to each
record. Per-record policy overrides are not accepted in this input version.
The exact ID and sample index are passed to the existing compiler, rather than
replacing every item with `cli` or using a physical row index as the seed key.
The output's nonzero `request_seq` values are delivery addresses only.

The command performs one shared-model hosted batch execution. Prompt lengths
and stopping times may differ; the existing ragged driver omits finished rows,
projects only needed heads, keeps each sequence's 44 KV slots separate, and
uses the same sampler and finalizer as single-request INT4. No kernel, numeric
profile, task identity, automatic dispatch preference or worker implementation
is changed. Use `--kernel ID` for an explicitly checked kernel, or omit it for
scalar. `--prefill-rows` is refused in this command: fixed-cohort execution is a
different schedule from the single-request prompt-prefill option.

Stdout is one canonical completed JSON object, followed by a newline. Its
`protocol` is `fnlp-source-int4-text-batch-v1`, with `schema_version: 1` and
`evidence: non_authoritative`. It includes model/source identity, group width,
actual integer-kernel identifier, and optional snapshot operation metadata.
It does not expose source/snapshot paths or private prompt digests.

`items[i]` contains the original ID, sample index and assigned request sequence;
`output.results[i]` is the corresponding independently finalized INT4 chat/text
result in the same input order. `output` also retains aggregate planned/actual
work and group-step counts from the task layer. The complete object is staged
before its first write, with the hosted result guard and CLI preparation charge
held through serialization, write and flush. A later-row planning/native/task
failure emits no successful prefix. A failed write/flush still exits nonzero;
transport bytes already written cannot be retracted or treated as a receipt.

## Admission

`--max-records` defaults to 8 and accepts 1..64. It is the ceiling for the
entire invocation, not a window size. The CLI conservatively admits storage for
that declared ceiling before reading input, even if fewer records arrive.
`--context-tokens` is the allocated capacity PER row, not an aggregate context.
`--max-input-bytes` bounds one complete NDJSON line; `--max-total-input-bytes`
defaults to 1 MiB and bounds the whole transport (maximum 64 MiB).

Complete admission includes every declared KV capacity and sampler, shared
RoPE/native workspace, retained raw generation payload, final aggregate output,
publication staging, and explicit allocator/preparation reserves. The memory
floor uses the maximum of the load, optional save and execution phases, plus
always-live preparation/resident weights/cleanup. It neither sums mutually
exclusive scratch lifetimes nor prices the cohort as one request. Large groups
or byte limits can require larger `--preparation-mib` and `--memory-mib` values.
These are modeled reservations, not measured or OS-enforced RSS limits.

Every actual compiled request is independently budgeted and the host repeats
cohort admission before allocation. `--max-checkpoints` covers the ONE complete
native cohort; it is not renewed per row. Load and snapshot stage quotas retain
their separate meanings. `--timeout-seconds` covers input, loading, planning,
optional saving and execution through remaining-time checks. Blocking IO is
cooperative and cannot be preempted by these checkpoints.

## Validation status and remaining work

Fourteen Rust regression-source functions were added across the preparation,
root-route and runtime modules. They cover framing/duplicates/UTF-8, source and
snapshot argument exclusivity, 1/2/8/64-row memory/work bounds, prefill and missing-
runtime refusal, real pinned planning, row identity under permutation, actual
host quotes, late planning failure, complete serialization and failed flush.
The existing classification-prefill refusal fixture is retained with its new
command field initialized explicitly.

The independent Python envelope model checked 24,960 declared/actual cohort
shapes, including phase occupancy, native/sampler/raw/final output storage and
ID-frame allowances. New source was checked for UTF-8, whitespace, balanced
delimiters and uploaded blob identity. None of these checks compiles or executes
Rust. Rust compilation/tests, CLI model execution, batch-versus-single model
parity, native cancellation interleavings and benchmarks were not run. They
remain controller-owned qualification work; no speedup or parity is claimed.

Automatic corpus windows/refill, per-record generation overrides, SIGINT
integration for this new route and heterogeneous grammar scheduling are not
implemented by this command. Existing single INT4 commands and the separate
INT8 corpus routes remain unchanged.
