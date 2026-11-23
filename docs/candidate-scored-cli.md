# Candidate classification and dimensional sentiment

`fnlp candidate classify`, `fnlp candidate classify-map` and
`fnlp candidate sentiment` use full-vocabulary
finite-continuation scoring on the existing strict-INT8 model. They do not ask
free generation to produce labels or sentiment JSON. Every exact candidate,
including its terminal EOS, is scored before independent task finalization.
These commands require `asupersync-runtime`, an explicit local current-candidate
INT8 `.fnlpq` and a process memory ceiling. They do not download or activate a
model, authenticate a publisher or confer numerical/task-quality certification.

## Classification

```sh
fnlp candidate classify request.json --model ./model.fnlpq --memory-mib 8192
```

The input is one JSON object, not NDJSON or raw source text:

```json
{
  "document": "The delivery arrived damaged and I need a replacement.",
  "labels": [
    {"id": "complaint", "description": "A dissatisfied customer reporting a problem"},
    {"id": "question", "description": "A request for information"}
  ],
  "mode": "exclusive",
  "policy": {"minimum_candidate_weight_ppm": 0, "minimum_margin_ppm": 0}
}
```

`mode` defaults to `exclusive` and `policy` to the two zero thresholds. Exclusive
classification needs at least two labels. `multi_label` instead runs an
independent yes/no head per label; its weights are NOT normalized across labels.
A one-label multi-label request is allowed. The existing result preserves every
candidate, ranking, decision, policy and score-space disclosure. A threshold
abstention is a typed decision; an execution error is not an abstention.

At most 128 unique exact label IDs are accepted, with 256 bytes per ID, 4096
bytes per description and 65536 bytes across IDs and descriptions. Descriptions
may be omitted. IDs cannot be empty, whitespace-only or contain control
characters. IDs are not normalized: distinct Unicode spellings remain distinct.
The fixed planner uses opaque response codes and encodes labels/descriptions as
untrusted data, so user label strings are not executable templates or tokens.


## Complete-document classification

Use `classify-map` when the document requires more than one native context:

```sh
fnlp candidate classify-map request.json \
  --model ./model.fnlpq --memory-mib 16384 \
  --max-chunks 64
```

The request uses the same `document`, `labels`, `mode` and `policy` fields as
`classify`. Both exclusive and independent multi-label classification are
supported. The command partitions the complete original source into
nonoverlapping chunks that fit every label head, retaining original UTF-8 bytes,
CRLF boundaries and byte/scalar coordinates. It refuses a document that cannot
fit the declared limits; it never returns a successful prefix or truncates input.

Every chunk and label head is planned before weights are loaded. The hosted
invocation independently reconstructs and checks the complete plan before its
first native forward. All five native work ceilings in the table below apply
to the **whole document**. Each chunk receives an exact slice, and one native
context is reused serially after logical KV is cleared. A failure or cancellation
in any chunk prevents complete document success.

The result retains every complete chunk result and a separately named
source-scalar-weighted aggregate. Scalar weights refer to the exact original
Unicode source, including source assigned to abstained chunks. The aggregate
preserves the declared classification thresholds and makes abstention explicit.
Multi-label decisions remain independent. This is a descriptive mixture of
independent chunk classifications; it is not a full-context probability,
calibrated confidence, or a claim that separate chunks capture cross-document
relationships.

Additional bounds are explicit:

| Option | Default | Scope |
|---|---:|---|
| `--max-chunks` | 64 | Complete document; hard maximum 256 |
| `--max-chunk-bytes` | Input byte ceiling | Additional per-chunk bound; native context still decides fit |
| `--max-tokenizer-calls` | 8192 | Complete partition, including rejected sizing probes |
| `--max-result-bytes` | 1048576 | One complete native chunk result |
| `--max-retained-result-bytes` | 16777216 | Sum of complete retained chunk records |
| `--max-map-result-bytes` | 16777216 | Complete native map and aggregate, with up to 4096 additional envelope bytes |
| `--minimum-classified-coverage-ppm` | 1000000 | Required non-abstaining source coverage for an aggregate decision |

Retained-result and complete-map byte limits are independent, each with a 64 MiB
hard maximum. All final serialization and output staging remain admitted until
write and flush complete. The output includes original source spans, chunk and
head counts, planned/actual work, retained byte counts, and the explicit
`single_context_equivalence=false` disclosure.

Compact input is opt-in:

```sh
fnlp candidate classify-map request.json \
  --model ./model.fnlpq --memory-mib 16384 \
  --input-encoding safe-bpe-v1
```

This command uses the complete-document encoding options
`--max-tokenizer-input-bytes`, `--max-tokenizer-output-tokens`,
`--max-tokenizer-pair-checks` and `--max-tokenizer-pair-bytes`.
The byte and token ceilings are cumulative preparation allowances, separate
from the original JSON input ceiling and each native context. Capacity checks,
label metadata, accepted and rejected partition probes, and every final
chunk/head plan share one allowance per complete preparation. The CLI preflight
and owned hosted preparation each use that same fixed policy with their own
ledger. Neither renews work at chunk or label boundaries. A depleted allowance
refuses the request without switching encoding.

The default preparation reservation is 512 MiB for byte input and 1024 MiB for
compact input. A pure worst-case quote includes the declared label/chunk
frontier, retained plans and results, encoder workspace, native scratch, KV,
weights and output staging before input/model IO. The example's 16 GiB process
ceiling is a configured admission limit, not measured RSS or a claim that every
larger configuration fits. Increasing declared work or output limits may require
a larger preparation or process reservation.

Library callers can use
`ClassificationPlanner::plan_int8_document_with_control` and
`PreparedInt8ClassificationMap`, or the owned
`NlpEngine::map_int8_classify` /
`NlpEngine::map_int8_classify_with_encoding` entry points with
`Int8ClassificationMapHostConfig`. The host accepts one already charged
`ResidentInt8`. Its caller identity must already bind the exact classifier,
tokenizer and input policy. `requirements_with_encoding` exposes the pure
memory quote before constructing the request.

Completion verification binds the exact source, taxonomy, policies, identities,
original limits and complete public result. Those private bindings are omitted
from JSON and Debug output; caller-provided JSON cannot manufacture native
completion authority. The original cancellation/deadline follows preparation,
execution, aggregation, verification and publication. This command currently
uses serial native scoring; it accepts no cohort, prefill-schedule or retained-job
options.

## Dimensional sentiment

```sh
fnlp candidate sentiment request.json --model ./model.fnlpq --memory-mib 8192
```

```json
{
  "document": "I am excited that the team finished the project together.",
  "axes": ["valence", "arousal", "dominance", "approach"],
  "policy": {"minimum_peak_weight_ppm": 0, "maximum_normalized_entropy_ppm": 1000000}
}
```

Omitted axes select all four; omitted policy uses the values shown. Each axis
has its own prompt and complete five-bin language: -1, -0.5, 0, 0.5, 1. The
native implementation shares candidate prefixes within an axis, then clears
logical KV between axes while retaining one native allocation. It does not
normalize probabilities across axes or collapse affect to positive/negative.
Results retain distributions, means, spread, normalized entropy and explicit
estimated/abstained decisions. These are uncalibrated conditional candidate
weights, NOT correctness confidence, psychological measurements, diagnoses or
inferences about hidden mental states.

## Input, budgets and output

### Compact untrusted input

The single-context classification and sentiment commands accept explicit compact input:

```sh
fnlp candidate classify request.json --model ./model.fnlpq --memory-mib 8192 \
  --input-encoding safe-bpe-v1
fnlp candidate sentiment request.json --model ./model.fnlpq --memory-mib 8192 \
  --input-encoding safe-bpe-v1
```

The default remains `byte-fallback`. `safe-bpe-v1` preserves the original UTF-8
bytes, including whitespace, Unicode spelling and marker-shaped strings, while
excluding template-control token IDs from untrusted text. Ordinary language can
use fewer prompt tokens and fit a longer document into the same context. The
whole prompt must still fit: the command never truncates a source or falls back
to another encoding when a limit is exhausted. This policy changes prompt token
sequences and does not promise identical model output, quality or measured speed.

One finite tokenizer ledger covers the complete scored request. Classification
charges the document once and every canonical label-metadata record, including
all independent multi-label heads. Sentiment encodes the document once for all
requested axes. `--max-input-bytes` also bounds cumulative untrusted bytes and
the available prompt-token allowance bounds cumulative untrusted tokens.
`--max-tokenizer-pair-checks` defaults to 4194304 and
`--max-tokenizer-pair-bytes` to 67108864. These options require compact input;
zero or excessive limits reject before IO. Tokenizer work is separate from the
five native-scoring work ceilings below.

The selected limits determine an additional transient encoder workspace. The
command includes that workspace in preparation admission before reading input
or opening model metadata. A larger byte ceiling may require a larger
`--preparation-mib`; for example, compact input with a 1 MiB input ceiling needs
more than the default 512 MiB preparation reservation. The complete policy is
bound into the execution identity and appears as `input_encoding` on the result
envelope. Byte-default responses retain their existing shape. Marker containment
does not prevent a model from reacting to instruction-shaped document prose.

### Shared request bounds

All three commands accept a file or `-`/omitted path for stdin. Input is bounded while
reading, with a 65536-byte default and 1 MiB hard maximum. UTF-8 source bytes are
preserved. Duplicate keys (including escaped aliases), unknown fields, malformed
policy, repeated axes and invalid labels reject before model metadata access.
Request JSON cannot supply a TaskBudget, model identity, tokenizer, instruction,
scoring mode or work receipt. There are no sampling, thinking, tool or mask
options on this surface. Supported policy integers range from 0 to 1000000.

Default context is 2048 tokens; `--max-candidate-tokens` defaults to 16 and
includes EOS. The host reserves that depth before prompt admission. The live
context is the largest head, not summed forward work across independent heads.
For single-context classification and sentiment, `--max-result-bytes` defaults
to 1 MiB for the whole native result; bounded
candidate provenance adds at most 4096 bytes. `--preparation-mib` defaults to
512, with an additional checked floor for worst-case repeated prompt storage.
These are modeled memory commitments, not observed or OS-enforced RSS limits.

All whole-bundle work axes are checked against explicit CLI ceilings BEFORE
weight materialization. Defaults are:

| Option | Default |
|---|---:|
| `--max-forward-positions` | 131072 |
| `--max-projected-logits` | 100000000 |
| `--max-attention-pairs` | 100000000000 |
| `--max-dot-products` | 100000000000 |
| `--max-multiply-accumulates` | 1000000000000000 |

These count algorithmic work, not elapsed time or measured throughput. Native
schedules price continuation-branch attention and decoder/head projections,
not just the number of labels. Every head receives its own exact budget slice;
a failed head never replenishes it or yields partial bundle success. Increasing
forward ceilings may require a larger preparation reservation.

The shared Session starts one process host, reserves preparation before input
or tokenizer allocation, checks candidate metadata, compiles and admits the
complete plan, then loads weights. All retained model facts are compared after
the independent load. The single native invocation reuses that charged resident
model; the result's output guard and preparation reservation survive complete
JSON staging, write and flush. A failed write is an error, not task success.
The response carries `scope=real-artifact-current-candidate`,
`evidence=non_authoritative`, actual artifact provenance and the native result.
No private prompt-derived identity digest is exported as telemetry.

The default cooperative deadline is 3600 seconds and follows input, preparation,
loading and execution without a fresh duration per stage. Compact tokenization
uses the original preparation controller's cooperative checkpoints. Blocking
input/file calls cannot be forcibly interrupted.
Native checkpoint allowance is separate from the loader's finite allowance.
Without `asupersync-runtime`, commands expose help but refuse before reading
input or opening a model. Existing generate/chat/source/batch commands remain
on their previous dispatch paths. All three scored commands accept one request per invocation;
the source-task `candidate batch` family is unchanged.

## Validation scope

The implementation adds model-free parser/command/resource tests and real-pinned
planning tests for both classification modes and all sentiment axes. Native
sentiment separately has private synthetic scoring/finalization tests. These
are test sources, not evidence of a successful full-model run. Complete-document
classification adds synthetic finalization and pinned planning definitions for
source coverage, all work axes, independent multi-label semantics, abstention,
private binding mutation refusal, exact byte limits, cumulative compact encoding,
hosted ownership admission and all eleven typed cancellation causes. Rust compilation,
Rust tests, real-model inference, performance, task quality and controller DSR
qualification were not executed in this environment. The reference native
scoring backend and existing artifact candidate retain their prior evidence
status.
