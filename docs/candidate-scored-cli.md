# Candidate classification and dimensional sentiment

`fnlp candidate classify` and `fnlp candidate sentiment` use full-vocabulary
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

Both commands accept a file or `-`/omitted path for stdin. Input is bounded while
reading, with a 65536-byte default and 1 MiB hard maximum. UTF-8 source bytes are
preserved. Duplicate keys (including escaped aliases), unknown fields, malformed
policy, repeated axes and invalid labels reject before model metadata access.
Request JSON cannot supply a TaskBudget, model identity, tokenizer, instruction,
scoring mode or work receipt. There are no sampling, thinking, tool or mask
options on this surface. Supported policy integers range from 0 to 1000000.

Default context is 2048 tokens; `--max-candidate-tokens` defaults to 16 and
includes EOS. The host reserves that depth before prompt admission. The live
context is the largest head, not summed forward work across independent heads.
`--max-result-bytes` defaults to 1 MiB for the whole native result; bounded
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
loading and execution without a fresh duration per stage. Blocking input/file
calls and bounded tokenizer/compiler operations are not internally preempted.
Native checkpoint allowance is separate from the loader's finite allowance.
Without `asupersync-runtime`, commands expose help but refuse before reading
input or opening a model. Existing generate/chat/source/batch commands remain
on their previous dispatch paths. These two new commands are single-request;
the source-task `candidate batch` family is unchanged.

## Validation scope

The implementation adds model-free parser/command/resource tests and real-pinned
planning tests for both classification modes and all sentiment axes. Native
sentiment separately has private synthetic scoring/finalization tests. These
are test sources, not evidence of a successful full-model run. Rust compilation,
Rust tests, real-model inference, performance, task quality and controller DSR
qualification were not executed in this environment. The reference native
scoring backend and existing artifact candidate retain their prior evidence
status.
