# Native candidate redaction

`fnlp candidate redact` runs source-bound native INT8 NER plus the existing
rule detectors, applies transactional edits, and by default reruns BOTH detector
families on the transformed text. It requires `asupersync-runtime`, an explicit
compatible local model and a process memory ceiling. It does not activate a
catalog, fetch weights or certify model quality. The existing model-free
`fnlp redact --rules-only` command is unchanged.

```sh
fnlp candidate redact document.txt --model ./model.fnlpq --memory-mib 8192
fnlp candidate redact document.txt --action placeholder \
  --model ./model.fnlpq --memory-mib 8192
```

Input is exact UTF-8 from a file or omitted/`-` for stdin. One completed JSON
response contains candidate provenance and the native redaction result. Only
final transformed text is returned. Intermediate NER transcripts and rejected
partially redacted documents are not exported; maps do not copy detected original
values. Undetected content can remain in the edited text. Output and preparation
charges survive complete serialization, writing and flushing.

## Compact input for one document

The default byte-fallback encoder uses one token per source byte. Select
`--input-encoding safe-bpe-v1` when you need compact tokens while preserving
the exact original UTF-8 bytes and source coordinates:

```sh
fnlp candidate redact document.txt --model ./model.fnlpq --memory-mib 12288 \
  --input-encoding safe-bpe-v1
```

Compact input excludes template-control token IDs. Instruction-shaped text
remains untrusted document content; this is a byte-preserving token boundary,
not a promise that the model cannot follow document instructions. The changed
prompt tokens do not imply identical model output, detector recall or measured
speed. Both original and actual edited text must independently fit context.

Without `--chunked`, one nonrenewable tokenizer ledger spans original-source
preflight, execution planning and fresh edited-source verification. The
`--max-tokenizer-pair-checks` (4194304 default) and
`--max-tokenizer-pair-bytes` (67108864 default) ceilings apply to their total.
Consumed work stays consumed after a failed or cancelled planning call.
Input-byte and output-token work totals use the admitted per-call peak times
three preparations, or two with `--no-verify`. Exhaustion returns no successful
redaction and never resets the allowance or silently selects byte fallback.

The original reader still obeys `--max-input-bytes`. Tokenizer admission also
prices the larger of that limit and `--max-result-bytes`, since placeholders
and pseudonyms can expand the edited source. Compact preparation defaults to
1024 MiB; byte preparation retains 512 MiB. Before key, document or model IO,
compact admission checks both preparation owners, tokenizer workspace, model
and native state, the overlapping NER/edit buffers and final output against
the process ceiling. These are modeled reservations, not measured RSS.

Compact responses add an `input_encoding` object containing `policy`,
`complete_limits` and actual cumulative `work`. The native policy digest binds
the exact per-call encoding and complete limits. Byte responses retain their
existing shape and policy identity.

For a document larger than one NER context, combine compact input with
`--chunked`. The [long-document guide](candidate-document-redaction.md) describes
its per-call probe limits and separate `--max-total-tokenizer-*` allowance across
all chunks, preflight, reconstruction and fresh verification. Combine
`--chunked --ndjson` for compact populations, including explicitly retained jobs;
each original document attempt keeps its own cumulative operation. The
[corpus](candidate-redaction-corpus.md) and [job](candidate-redaction-jobs.md)
guides describe the memory and authenticated replay contracts. Compact
populations without `--chunked` remain refused before private IO.
Tokenizer work flags require `--input-encoding safe-bpe-v1`.

## Scope and verification

Default NER types are person, organization and location. `--ner-options FILE`
accepts a complete bounded `NerOptions` object (`types`, `max_entities`,
`max_mention_scalars`), capped at 16 KiB. Unknown fields, repeated types and
invalid limits fail before model metadata access. The default rules detect
ASCII contact shapes, HTTP(S) URLs, IP literals and Luhn-valid card shapes.
`--rules email,phone,url,ip-address,credit-card,date` explicitly replaces that
set; dates are opt-in. These rules do not normalize every Unicode obfuscation.

The default `--action mask` replaces each detected scalar with `*`, so masks
do not expand source byte length. `placeholder` emits typed replacements.
`pseudonymize` uses the full HMAC digest described below. Overlapping findings
are combined using the existing union policy; incompatible mixed regions are
masked instead of inventing a type. `--include-map` opts into sensitive
original/transformed coordinates inside the result, not telemetry.

Verification sees the actual edited text, not old offsets or original NER
results. Residual detections, a failed native pass, cancellation, or exceeded
bounds yield no successful output. `--no-verify` is an explicit opt-out and
retains the native `not_requested` status. A clean declared union is NOT a
claim that all PII was found. Model omissions and correlated verification errors
remain possible; source membership proves occurrence, not recognition accuracy.

## Full-length pseudonyms

```sh
fnlp candidate redact document.txt --action pseudonymize \
  --key-stdin --key-id rotation-1 --namespace dataset-1 \
  --model ./model.fnlpq --memory-mib 8192 < private.key
```

The caller supplies authorized high-entropy binary key material (32..4096 bytes).
It is read directly through the existing bounded secret reader, never argv or a
UTF-8 String. Key length checks do not establish entropy. With `--key-stdin`,
the document must use a separate file. No new key-file permission mechanism or
key-generation source is introduced. Keep shell traces and output access under
your own privacy controls. Secret buffers use the existing best-effort cleanup,
not a compiler-proof zeroization guarantee.

The same key, namespace, PII type and exact UTF-8 value produce the same
full-256-bit pseudonym. No per-document 128-bit collision preflight is claimed.
An optional `--expected-key-commitment` is checked before reading the document;
a different key fails rather than silently changing identities. Result metadata
retains the existing nonsecret key/scope commitments. Pseudonyms remain linkable
and are not anonymization. The native host owns its key share until physical
completion, including cancellation and failure.

## Bounded native execution

The shared source options retain 2048 context tokens, 512 maximum output tokens
per NER pass, 65536 original input bytes and a 1 MiB complete result by default.
The actual original NER plan is checked before weight materialization. A verified
request reserves its exact first-pass work plus one full admitted-context second
pass, summing all five independent work counters rather than concatenating
attention contexts. At most two passes run, on one engine and resident model.
`--max-mask-node-visits` is explicitly per pass; the total allowance is its
checked product with one or two passes. Failed passes do not renew it.

The second pass cannot be compiled until edits exist. Placeholder/pseudonym
expansion may exceed its context even when the original fits; this is a refusal,
never truncation, rules-only fallback or an implicit verification opt-out.
Transformed byte admission is widened only up to the already priced result cap;
exact native context checks remain independent. Complete-result size includes
policy metadata and any opted-in coordinate map, not just edited text.

Rule scans have a separately finite per-scan `--max-rule-work`. Detection and
edit counts share `--max-detections` (4096 default, at most 16384). The hosted
edit reserve (`--edit-reserve-mib`, 64 default) has a checked floor for region
metadata and replacement staging. Preparation defaults to 512 MiB for byte
input or 1024 MiB for compact input.
These are modeled ledger commitments, not OS-enforced or measured RSS bounds.
One cooperative elapsed limit follows key/input reads, planning, loading and
execution; blocking I/O and individual tokenizer operations are not preempted.
Without the runtime feature, refusal precedes key, document, options and model
I/O. Failures use closed diagnostics and no candidate result prefix is printed.


## Persistent INT8 workers and bounded prompt prefill

Both one-shot short redaction and `--chunked` complete-document redaction
accept `--compute-threads 1..=64` and `--prefill-rows 1..=64`. Omit the
prefill option for sequential prompt processing; width one retains the serial
projection path. The options can be selected independently or together.

```sh
fnlp candidate redact document.txt \
  --model ./model.fnlpq --memory-mib 16384 \
  --chunked --input-encoding safe-bpe-v1 \
  --compute-threads 4 --prefill-rows 8
```

The memory allowance is illustrative. Admission includes both preparation
owners, the original source and secret metadata, native KV and scratch,
intermediate NER/map results, editing storage, final output, coordinator
prompt scratch and fixed worker buffers before private key, document, options
or model IO. Prompt scratch is additional to worker storage. Worker width
does not multiply the number of engines or renew any allowance.

One team spans original detection, every original NER chunk, editing, and
fresh detection on the actual transformed text. Compact input retains the
same spent tokenizer operation across preflight, reconstruction and verification.
Native, mask, rule and grounding budgets keep their existing scopes. The host
retains its input, secret and memory ownership until the workers physically
join on success, cancellation, error or unwinding.

Threaded results add an optional top-level `compute` receipt with physical
worker counters. Semantic output, detector-policy identity and compact
`input_encoding` metadata retain their existing meanings. Serial responses
omit `compute`; all output remains guarded through serialization, write and
flush under the original cooperative deadline. These counters do not measure
throughput or establish numerical parity or detector quality.

This increment exposes scheduling for one-shot requests. Live and retained
population commands currently reject these schedule options before private
IO because their team must span the entire population invocation.

Embedding applications use `NlpEngine::redact_int8_layer_major`,
`redact_int8_with_threads` or `redact_int8_layer_major_with_threads`; the
complete-document equivalents start with `redact_int8_document` and use the
same suffixes. The native redactors also expose `redact_layer_major` for
callers that already own the engine and its prefill reservation.

## Validation status

Regression sources cover command routing, detector scope, verification opt-out,
secret-source conflicts, binary-key bounds and commitments, limits, actual pinned
NER preparation, independent work accounting and cancellation. They are not
neural-success fixtures. Rust compilation/tests, model inference, PII recall,
benchmarks and controller DSR were not executed for this implementation.
All successful outputs retain `evidence=non_authoritative`.

The compact-input increment adds regression definitions for shared work across
preflight/reconstruction/verification, exact source-byte preservation, all eleven
cancellation causes, policy binding, expanded-workspace admission, total live
memory, unsupported compositions and optional response metadata. These definitions
have received source review but remain unexecuted pending the required DSR
checkpoint; neither detector quality nor end-to-end compact-model behavior is
qualified by this change.
