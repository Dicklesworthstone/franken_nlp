# Source INT4 corpus execution

`fnlp candidate source-int4 text-corpus` processes a finite NDJSON corpus in
bounded windows using one lazily loaded resident INT4 model. It connects the
existing source/snapshot loader, pinned text/chat planner, ordinary or packed
batch host, and independent task finalizers. It does not run a separate model
load per window and does not retain every document's KV or generated output.
The existing `text-batch` command remains a single aggregate-result operation.

```sh
fnlp candidate source-int4 text-corpus requests.ndjson --task generate \
  --snapshot model.int4snap --expected-model-sha256 "$MODEL_SHA256" \
  --memory-mib 8192 --context-tokens 512 --max-new-tokens 32 \
  --cohort-rows 8 --max-records 1000 --prefill-rows 32
```

Use the existing `asupersync-runtime` build. `MODEL_SHA256` is an independently
retained 64-character lowercase model-content pin, not a value read from the
untrusted snapshot. Source conversion instead takes `--source-dir DIR` and
`--source-manifest FILE`, optionally `--save-snapshot NEW_FILE`. These modes
remain exclusive, and a snapshot failure never falls back to source conversion.
Optional saving happens once, after the first window is completely planned and
admitted. A saved snapshot remains if later execution fails. Empty input neither
loads nor saves a model, although configuration and CPU eligibility are checked.

## Input and execution

Generate records contain a unique `id`, `prompt`, and optional unsigned
`sample_index`; chat uses `--task chat` and a `messages` array instead of prompt.
The same strict transcript parser is used by fixed INT4 batches. Unknown fields,
duplicate JSON keys, invalid UTF-8, blank lines, malformed role order, oversized
records and duplicate IDs stop the corpus. IDs are unique across the whole
invocation, not merely the current window. This is not snapshot-wide validation
before any work: a duplicate encountered later may follow already published
windows. A final nonempty record without LF is accepted; CR/LF count toward caps.

`--cohort-rows` defaults to 8 and accepts 1..64. The reader returns a full window
without waiting for the next line; a shorter final window runs at EOF. No new
window is read or planned while its predecessor is executing or being published.
The standard input buffer may prefetch up to its bounded capacity. Exact record
IDs and sample indexes reach the compiler; global nonzero `request_seq` values
are delivery addresses and do not enter the sampler's semantic key.

Each window executes through one existing supervised hosted batch call. Without
`--prefill-rows`, it advances one token per unfinished sequence per native pass.
With it, the existing fair packed driver admits at most that many total token
rows per pass. Windows reuse the same model/planner, not each other's KV state.
All per-request generation options, task ceilings, numerical profiles, kernel
checks and finalizers remain unchanged. `--kernel` is explicit; scalar stays the
default. There is no dynamic slot refill or per-record policy override.

## Completion protocol

Stdout uses `fnlp-source-int4-text-corpus-v1`, schema version 1. Each canonical
NDJSON `window_complete` frame contains `window`, `first_request_seq`, `records`,
and `batch`. The nested `batch` retains the existing fixed-batch response shape,
including public model metadata, `items`, finalized `output.results`, actual
kernel and execution schedule. `batch.items[i]` maps to `batch.output.results[i]`.
Snapshot operation metadata is present only on the first completed window.
No new filesystem path, private prompt digest, or private-content checksum is
emitted. This remains an explicitly non-authoritative candidate protocol.

Every task in a window must finish and validate before that window is serialized.
The entire canonical frame, including its newline, must fit before its first
external write. The hosted result guard and preparation charge remain held
through writing and flushing; then results drop before the next window is read.
Later input, planning, model, budget, execution or output failures stop without
retry and without a terminal success frame. Earlier complete windows remain
available. A partial write cannot be retracted and a failed flush is not success.

Whole-corpus success requires BOTH a final `corpus_complete` frame AND process
exit 0. The footer records complete-window/item counts, consumed input bytes,
`output_bytes_before_footer`, reserved and actual native work, and reserved
execution checkpoints. Its output-byte field deliberately excludes the footer
itself; the transport limit includes it. The footer is emitted only after actual
EOF and after every consumed record's window completed. Empty input emits only
this zero-count footer. This is not a durable job or an exactly-once pipe receipt.

## Whole-run limits and memory

`--max-records` defaults to 1000 and is capped at one million. The complete seen-ID
set is separately included in the preparation floor (a conservative 1024-byte
allowance per permitted ID). Native/KV/sampler/raw-result storage scales with
`--cohort-rows`, not the corpus size. Current-window byte staging and optional
packed rails retain their existing independent reservations. Declared capacities
are admitted even for a shorter final window. Preparation and allocator reserves
are modeled commitments, not measured or operating-system-enforced RSS limits.

`--max-total-input-bytes` and `--max-total-output-bytes` each default to 64 MiB
and are capped at 16 GiB. `--max-input-bytes` still bounds each complete line.
Before model preparation for a window, output admission reserves room for its
full declared frame and a 4096-byte footer. A small remaining transport allowance
can therefore refuse a window even when an unusually short result might fit.

The corpus ledger reserves each window's actual compiled worst-case work before
optional saving or inference, across all five axes: `--max-total-forward-positions`,
`--max-total-projected-logits`, `--max-total-attention-pairs`,
`--max-total-group-dots`, and `--max-total-multiply-accumulates`. Independent causal
attention work is summed; documents are never concatenated into one context.
Every axis is checked before any cumulative charge changes. Early stopping does
not refund reserved work to a later window; actual work is reported separately.

`--max-checkpoints` is the native allowance PER window and is charged in full to
`--max-total-execution-checkpoints` before that window executes. This bounds total
execution without pretending to measure checkpoint consumption. Source loading
and optional snapshot persistence each happen at most once and keep their own
stage quotas. `--timeout-seconds` covers the entire invocation via remaining-time
checks. Blocking IO remains cooperative, not preemptible. No window creates a
new resource domain or renews the whole-run budget.

## Validation and remaining scope

Eleven new Rust regression functions cover the window reader and cumulative
ledger: more-than-64-record traversal, delivery positions, byte/EOF boundaries,
cross-window duplicates, terminal reader failure, cancellation, independent work
axes, checkpoint reservations, early-stop accounting, completion eligibility and
footer transport space. Existing fixed-batch tests are retained and updated for
the shared preparation function's explicit starting delivery sequence.

These tests have not been compiled or run. The additional stream-loop regression
file was rejected by the connector and is not part of this change; no missing
module reference was retained. Real corpus/model execution, full-vs-windowed
model parity, hosted cancellation/IO failure campaigns and performance evidence
remain outstanding. No numerical qualification, speedup or release-gate closure
is asserted. Dynamic refill, durable resume, dedicated signal integration and
per-record generation overrides remain separate work.
